from __future__ import annotations
import argparse
import gc
import json
import os
import pickle
import random
import resource
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader
from paths import WORKSPACE as WORKSPACE_ROOT

UCE = WORKSPACE_ROOT / "models/UCE"

MODEL_FILES = UCE / "model_files"

WEIGHT = MODEL_FILES / "4layer_model.torch"

TOKENS = MODEL_FILES / "all_tokens.torch"

HUMAN_PE = (
    MODEL_FILES
    / "protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"
)

SOURCE_H5AD = (
    WORKSPACE_ROOT
    / "data/patient_encoder_feasibility/uce_inputs"
    / "cima_417_natural1024_union5.h5ad"
)

SOURCE_FILTER = (
    WORKSPACE_ROOT
    / "data/patient_encoder_feasibility/uce_inputs"
    / "cima_417_natural1024_union5_global_filter.npz"
)


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values],
        dtype=object,
    )

def verify_assets() -> dict[str, dict[str, object]]:
    observed: dict[str, dict[str, object]] = {}
    for path in (WEIGHT, TOKENS, HUMAN_PE):
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[str(path.relative_to(MODEL_FILES))] = {"bytes": path.stat().st_size}
    return observed

def source_frozen_genes() -> np.ndarray:
    with h5py.File(SOURCE_H5AD, "r") as handle:
        genes = decode(handle["var/gene_symbol"][:]).astype(str)
    archive = np.load(SOURCE_FILTER, allow_pickle=False)
    keep = np.asarray(archive["keep_min_cells_10"], dtype=bool)
    if keep.shape != genes.shape:
        raise ValueError("CIMA frozen UCE gene mask has the wrong shape")
    selected = genes[keep]
    return selected

def h5_contract(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("raw_counts", False) not in (True, np.bool_(True)):
            raise ValueError("input store does not certify raw counts")
        shape = np.asarray(handle["matrix"].attrs["shape"], dtype=np.int64)
        genes = decode(handle["var/gene_symbol"][:]).astype(str)
        donors = decode(handle["obs/donor_id"][:]).astype(str)
        indptr = np.asarray(handle["matrix/indptr"][:], dtype=np.int64)
    if tuple(shape) != (len(donors), len(genes)) or len(indptr) != len(donors) + 1:
        raise ValueError("raw-count H5 CSR contract failed")
    donor_order = pd.unique(donors)
    if len(donor_order) != int(len(set(donors.tolist()))):
        raise ValueError("donor identity contract failed")
    return genes, donors, indptr, donor_order.astype(str)

def read_group(
    path: Path,
    genes: np.ndarray,
    donors: np.ndarray,
    indptr: np.ndarray,
    row_start: int,
    row_end: int,
    column_index: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    start, end = int(row_start), int(row_end)
    if not 0 <= start < end <= len(donors):
        raise ValueError("invalid contiguous cell shard")
    data_start, data_end = int(indptr[start]), int(indptr[end])
    with h5py.File(path, "r") as handle:
        data = np.asarray(handle["matrix/data"][data_start:data_end])
        indices = np.asarray(handle["matrix/indices"][data_start:data_end], dtype=np.int64)
    local_indptr = indptr[start : end + 1] - data_start
    matrix = sparse.csr_matrix(
        (data, indices, local_indptr), shape=(end - start, len(genes))
    )[:, column_index]
    matrix.sort_indices()
    values = matrix.data
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError("UCE input is not finite non-negative integer counts")
    detected = np.asarray(matrix.getnnz(axis=1)).ravel()
    if detected.min(initial=25) < 25:
        raise ValueError("a target cell has fewer than 25 CIMA-frozen UCE genes")
    return matrix, donors[start:end]

class OfficialUCEEngine:
    def __init__(self, batch_size: int, device: str):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; UCE CPU fallback is forbidden")
        self.device = torch.device(device)
        torch.zeros(1, device=self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        sys.path.insert(0, str(UCE))
        from eval_data import MultiDatasetSentenceCollator, MultiDatasetSentences
        from model import TransformerModel

        self.dataset_class = MultiDatasetSentences
        self.collator_class = MultiDatasetSentenceCollator
        self.batch_size = batch_size

        protein = torch.load(HUMAN_PE, map_location="cpu", weights_only=False)
        self.protein_keys = [str(key).upper() for key in protein.keys()]
        self.protein_index = {gene: index for index, gene in enumerate(self.protein_keys)}
        del protein
        with (MODEL_FILES / "species_offsets.pkl").open("rb") as handle:
            self.human_offset = int(pickle.load(handle)["human"])

        chrom = pd.read_csv(MODEL_FILES / "species_chrom.csv")
        chrom["spec_chrom"] = pd.Categorical(
            chrom["species"].astype(str) + "_" + chrom["chromosome"].astype(str)
        )
        self.chrom = chrom.loc[chrom["species"].eq("human")].set_index("gene_symbol")

        model = TransformerModel(
            token_dim=5120,
            d_model=1280,
            nhead=20,
            d_hid=5120,
            nlayers=4,
            dropout=0.05,
            output_dim=1280,
        )
        empty = torch.zeros(145469, 5120)
        empty.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(empty)
        state = torch.load(WEIGHT, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)
        del state, empty
        tokens = torch.load(TOKENS, map_location="cpu", weights_only=False)
        if tokens.shape[0] == 143574:
            torch.manual_seed(23)
            tokens = torch.vstack(
                (tokens, torch.normal(mean=0, std=1, size=(1895, 5120)))
            )
        tokens.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(tokens)
        del tokens
        self.model = model.eval().to(self.device)
        gc.collect()

    def supported_columns(
        self, input_genes: np.ndarray, frozen_source_genes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        frozen = {gene.upper() for gene in frozen_source_genes}
        chrom = set(self.chrom.index.astype(str))
        use = np.asarray(
            [
                gene.upper() in frozen
                and gene.upper() in self.protein_index
                and gene.upper() in chrom
                for gene in input_genes
            ],
            dtype=bool,
        )
        selected = input_genes[use].astype(str)
        if len(selected) < 1_000:
            raise ValueError(f"too few supported CIMA-frozen genes: {len(selected)}")
        if len({gene.upper() for gene in selected}) != len(selected):
            raise ValueError("case-insensitive duplicate genes are unsupported")
        return np.flatnonzero(use), selected

    def sidecars(self, genes: np.ndarray, work: Path, name: str) -> tuple[Path, Path, Path]:
        upper = [gene.upper() for gene in genes]
        pe = torch.tensor(
            [self.protein_index[gene] + self.human_offset for gene in upper],
            dtype=torch.long,
        )
        located = self.chrom.loc[upper]
        chroms = located["spec_chrom"].cat.codes.to_numpy()
        starts = located["start"].to_numpy()
        pe_path = work / f"{name}_pe_idx.torch"
        chrom_path = work / f"{name}_chroms.pkl"
        starts_path = work / f"{name}_starts.pkl"
        torch.save({name: pe}, pe_path)
        with chrom_path.open("wb") as handle:
            pickle.dump({name: chroms}, handle)
        with starts_path.open("wb") as handle:
            pickle.dump({name: starts}, handle)
        return pe_path, chrom_path, starts_path

    def embed(self, matrix: sparse.csr_matrix, genes: np.ndarray, work: Path, seed: int) -> np.ndarray:
        work.mkdir(parents=True, exist_ok=True)
        name = "donor_group"
        count_path = work / f"{name}_counts.npz"
        counts = np.memmap(
            count_path, dtype="int64", mode="w+", shape=matrix.shape
        )
        for start in range(0, matrix.shape[0], 128):
            end = min(start + 128, matrix.shape[0])
            counts[start:end] = matrix[start:end].toarray().astype(np.int64, copy=False)
        counts.flush()
        del counts
        pe_path, chrom_path, starts_path = self.sidecars(genes, work, name)
        args = SimpleNamespace(
            pad_length=1536,
            cls_token_idx=3,
            chrom_token_right_idx=2,
            pad_token_idx=0,
            CHROM_TOKEN_OFFSET=143574,
            sample_size=1024,
            dir=f"{work}{os.sep}",
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dataset = self.dataset_class(
            sorted_dataset_names=[name],
            shapes_dict={name: matrix.shape},
            args=args,
            dataset_to_protein_embeddings_path=pe_path,
            datasets_to_chroms_path=chrom_path,
            datasets_to_starts_path=starts_path,
            npzs_dir=f"{work}{os.sep}",
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator_class(args),
            num_workers=0,
        )
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for batch_sentences, mask, _, _ in loader:
                sentences = batch_sentences.permute(1, 0).to(self.device)
                mask = mask.to(self.device)
                embedded = self.model.pe_embedding(sentences.long())
                embedded = nn.functional.normalize(embedded, dim=2)
                _, output = self.model.forward(embedded, mask=mask)
                chunks.append(output.float().cpu().numpy())
        result = np.vstack(chunks).astype(np.float32, copy=False)
        if result.shape != (matrix.shape[0], 1280) or not np.isfinite(result).all():
            raise RuntimeError("UCE output contract failed")
        return result

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group-cells", type=int, default=10_000)
    parser.add_argument("--group-mod", type=int, default=1)
    parser.add_argument("--group-rem", type=int, default=0)
    parser.add_argument("--group-id", type=int)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.group_cells < 1 or args.group_mod < 1 or not 0 <= args.group_rem < args.group_mod:
        raise ValueError("invalid group partition")
    assets = verify_assets()
    frozen_genes = source_frozen_genes()
    genes, donors, indptr, donor_order = h5_contract(args.input)
    groups = [
        (start, min(start + args.group_cells, len(donors)))
        for start in range(0, len(donors), args.group_cells)
    ]
    group_ids = list(range(len(groups)))
    if args.group_id is not None:
        group_ids = [args.group_id]
    group_ids = [index for index in group_ids if index % args.group_mod == args.group_rem]
    if args.max_groups is not None:
        group_ids = group_ids[: args.max_groups]
    args.output.mkdir(parents=True, exist_ok=True)
    engine = OfficialUCEEngine(args.batch_size, args.device)
    column_index, selected_genes = engine.supported_columns(genes, frozen_genes)
    for group_id in group_ids:
        if group_id < 0 or group_id >= len(groups):
            raise ValueError(f"group id outside 0..{len(groups) - 1}: {group_id}")
        group_dir = args.output / f"group_{group_id:05d}"
        marker = group_dir / "COMPLETE.json"
        if marker.is_file():
            record = json.loads(marker.read_text())
            if record.get("status") == "PASS" and (group_dir / "cell_embeddings.npz").is_file():
                print(json.dumps({"group_id": group_id, "status": "cached"}), flush=True)
                continue
        group_dir.mkdir(parents=True, exist_ok=True)
        work = group_dir / "work"
        if work.exists():
            shutil.rmtree(work)
        started = time.perf_counter()
        matrix, cell_donors = read_group(
            args.input,
            genes,
            donors,
            indptr,
            groups[group_id][0],
            groups[group_id][1],
            column_index,
        )
        seed = 20260711 + group_id
        embeddings = engine.embed(matrix, selected_genes, work, seed)
        output_path = group_dir / "cell_embeddings.npz"
        temporary = group_dir / "cell_embeddings.tmp.npz"
        np.savez(
            temporary,
            embeddings=embeddings,
            donor_id=cell_donors.astype(str),
        )
        os.replace(temporary, output_path)
        partial_donors = pd.unique(cell_donors).astype(str)
        donor_sums = []
        donor_counts = []
        for donor in partial_donors:
            selected = embeddings[cell_donors == donor]
            donor_sums.append(selected.sum(axis=0, dtype=np.float64))
            donor_counts.append(len(selected))
        mean_path = group_dir / "donor_partial_uce.npz"
        np.savez(
            mean_path,
            embedding_sum=np.asarray(donor_sums, dtype=np.float64),
            n_cells=np.asarray(donor_counts, dtype=np.int64),
            donor_id=partial_donors,
        )
        record = {
            "status": "PASS",
            "method": "official UCE-4L",
            "group_id": group_id,
            "row_start": int(groups[group_id][0]),
            "row_end": int(groups[group_id][1]),
            "n_donors_represented": int(len(partial_donors)),
            "n_cells": int(len(cell_donors)),
            "n_genes_input": int(len(genes)),
            "n_genes_cima_frozen_and_uce_supported": int(len(selected_genes)),
            "target_gene_statistics_computed": False,
            "target_labels_accessed": False,
            "seed": seed,
            "uce_output_shape": list(embeddings.shape),
            "uce_output_std": float(embeddings.std()),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(engine.device) / 2**20),
            "process_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            "gpu": torch.cuda.get_device_name(engine.device),
            "assets": assets,
        }
        marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        shutil.rmtree(work)
        del matrix, embeddings
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(record, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    main()

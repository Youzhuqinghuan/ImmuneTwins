from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

SCHEMA = "immunetwins_external_raw_counts_v2"


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
            for value in values
        ],
        dtype=object,
    )

def h5ad_column(group: h5py.Group, key: str) -> np.ndarray:
    node = group[key]
    if isinstance(node, h5py.Dataset):
        return decode(node[:]) if node.dtype.kind in {"O", "S", "U"} else node[:]
    if isinstance(node, h5py.Group) and {"codes", "categories"}.issubset(node):
        codes = np.asarray(node["codes"][:], dtype=np.int64)
        categories = decode(node["categories"][:])
        if (codes < 0).any() or (codes >= len(categories)).any():
            raise ValueError(f"{group.name}/{key} contains missing categorical values")
        return categories[codes]
    raise ValueError(f"unsupported H5AD encoding: {group.name}/{key}")

def donor_seed(seed: int, donor_id: str) -> int:
    payload = f"{seed}\0{donor_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

def deterministic_selection(
    donors: np.ndarray, *, seed: int, max_cells: int
) -> tuple[np.ndarray, pd.DataFrame]:
    donor_ids = np.asarray(sorted(pd.unique(donors.astype(str)).tolist()), dtype=str)
    donor_index = {donor_id: index for index, donor_id in enumerate(donor_ids)}
    codes = np.fromiter(
        (donor_index[donor_id] for donor_id in donors),
        dtype=np.int32,
        count=len(donors),
    )
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(donor_ids))
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    selected = np.zeros(len(donors), dtype=bool)
    records: list[dict[str, object]] = []
    for donor_position, donor_id in enumerate(donor_ids):
        rows = order[boundaries[donor_position] : boundaries[donor_position + 1]]
        source_count = len(rows)
        retained = min(source_count, max_cells)
        if retained < source_count:
            rng = np.random.default_rng(donor_seed(seed, donor_id))
            rows = np.sort(rng.choice(rows, size=retained, replace=False))
        selected[rows] = True
        records.append(
            {
                "donor_id": donor_id,
                "source_cells": source_count,
                "selected_cells": retained,
            }
        )
    manifest = pd.DataFrame.from_records(records)
    if selected.sum() != manifest["selected_cells"].sum():
        raise RuntimeError("deterministic donor selection coverage failed")
    return selected, manifest

def unique_gene_map(feature_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    genes: list[str] = []
    position: dict[str, int] = {}
    old_to_new = np.empty(len(feature_names), dtype=np.int64)
    for index, raw_name in enumerate(feature_names):
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"blank gene symbol at source column {index}")
        target = position.get(name)
        if target is None:
            target = len(genes)
            position[name] = target
            genes.append(name)
        old_to_new[index] = target
    return np.asarray(genes, dtype=object), old_to_new

def append_dataset(dataset: h5py.Dataset, cursor: int, values: np.ndarray) -> int:
    stop = cursor + len(values)
    if stop > len(dataset):
        raise RuntimeError("preallocated CSR payload is too small")
    dataset[cursor:stop] = values
    return stop

def build(args: argparse.Namespace) -> dict[str, object]:
    destinations = [args.output, args.audit, args.donor_manifest]
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite OneK1K artifacts: {existing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.donor_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial output requires inspection: {temporary}")

    started = time.perf_counter()
    with h5py.File(args.input, "r") as source:
        if "X" not in source or not {"data", "indices", "indptr"}.issubset(source["X"]):
            raise ValueError("OneK1K X is not a CSR-encoded H5AD matrix")
        donors = h5ad_column(source["obs"], args.donor_column).astype(str)
        if any(not donor for donor in donors):
            raise ValueError("OneK1K donor identifiers contain blanks")
        source_genes = h5ad_column(source["var"], args.gene_column).astype(str)
        genes, old_to_new = unique_gene_map(source_genes)
        indptr = np.asarray(source["X/indptr"][:], dtype=np.int64)
        shape = tuple(int(value) for value in source["X"].attrs["shape"])
        if shape != (len(donors), len(source_genes)) or len(indptr) != len(donors) + 1:
            raise ValueError("OneK1K X/obs/var dimensions disagree")

        selected_mask, donor_manifest = deterministic_selection(
            donors, seed=args.seed, max_cells=args.max_cells
        )
        selected_rows = np.flatnonzero(selected_mask)
        selected_donors = donors[selected_rows]
        estimated_nnz = int(np.sum(indptr[selected_rows + 1] - indptr[selected_rows]))
        string_dtype = h5py.string_dtype("utf-8")
        output_indptr = np.empty(len(selected_rows) + 1, dtype=np.int64)
        output_indptr[0] = 0
        output_row = 0
        payload_cursor = 0

        with h5py.File(temporary, "w") as output:
            output.attrs.update(
                {
                    "schema_version": SCHEMA,
                    "cohort": args.cohort,
                    "view": f"deterministic_max{args.max_cells}",
                    "seed": args.seed,
                    "raw_counts": True,
                    "labels_present": False,
                    "target_age_opened": False,
                    "n_donors": int(len(donor_manifest)),
                    "max_cells_per_donor": args.max_cells,
                }
            )
            obs = output.create_group("obs")
            obs.create_dataset(
                "donor_id",
                data=selected_donors.astype(object),
                dtype=string_dtype,
                compression="lzf",
            )
            obs.create_dataset(
                "source_obs_index",
                data=selected_rows.astype(np.int64),
                compression="lzf",
            )
            var = output.create_group("var")
            var.create_dataset(
                "gene_symbol", data=genes.astype(object), dtype=string_dtype
            )
            matrix = output.create_group("matrix")
            matrix.attrs["encoding-type"] = "csr_matrix"
            data_out = matrix.create_dataset(
                "data",
                shape=(estimated_nnz,),
                maxshape=(estimated_nnz,),
                dtype=np.float32,
                chunks=(min(1_000_000, max(1, estimated_nnz)),),
                compression="lzf",
            )
            indices_out = matrix.create_dataset(
                "indices",
                shape=(estimated_nnz,),
                maxshape=(estimated_nnz,),
                dtype=np.int32,
                chunks=(min(1_000_000, max(1, estimated_nnz)),),
                compression="lzf",
            )

            for row_start in range(0, len(donors), args.chunk_cells):
                row_end = min(row_start + args.chunk_cells, len(donors))
                keep = np.flatnonzero(selected_mask[row_start:row_end])
                if not len(keep):
                    continue
                data_start, data_end = int(indptr[row_start]), int(indptr[row_end])
                block = sparse.csr_matrix(
                    (
                        np.asarray(source["X/data"][data_start:data_end], dtype=np.float32),
                        np.asarray(source["X/indices"][data_start:data_end], dtype=np.int64),
                        indptr[row_start : row_end + 1] - data_start,
                    ),
                    shape=(row_end - row_start, len(source_genes)),
                )[keep]
                if (
                    not np.isfinite(block.data).all()
                    or (block.data < 0).any()
                    or not np.equal(block.data, np.floor(block.data)).all()
                ):
                    raise ValueError("OneK1K X is not finite non-negative integer counts")
                block.indices = old_to_new[block.indices].astype(np.int32, copy=False)
                block._shape = (block.shape[0], len(genes))
                block.sum_duplicates()
                block.sort_indices()
                payload_cursor = append_dataset(data_out, payload_cursor, block.data)
                append_dataset(
                    indices_out,
                    payload_cursor - len(block.indices),
                    block.indices.astype(np.int32, copy=False),
                )
                local_counts = np.diff(block.indptr).astype(np.int64, copy=False)
                output_indptr[output_row + 1 : output_row + 1 + len(local_counts)] = (
                    payload_cursor - int(block.nnz) + np.cumsum(local_counts)
                )
                output_row += block.shape[0]
                if output_row % 100_000 < args.chunk_cells:
                    print(
                        json.dumps(
                            {
                                "selected_cells_written": output_row,
                                "selected_cells_total": int(len(selected_rows)),
                                "nnz_written": payload_cursor,
                            }
                        ),
                        flush=True,
                    )
            if output_row != len(selected_rows):
                raise RuntimeError("OneK1K selected-row write coverage failed")
            data_out.resize((payload_cursor,))
            indices_out.resize((payload_cursor,))
            matrix.create_dataset("indptr", data=output_indptr, compression="lzf")
            matrix.attrs["shape"] = np.asarray(
                [len(selected_rows), len(genes)], dtype=np.int64
            )
            output.attrs["n_cells"] = int(len(selected_rows))
            output.attrs["n_genes"] = int(len(genes))
            output.attrs["total_nnz"] = payload_cursor
            output.flush()
    os.replace(temporary, args.output)

    donor_manifest.to_csv(args.donor_manifest, sep="\t", index=False)
    record = {
        "status": "PASS",
        "schema_version": SCHEMA,
        "cohort": args.cohort,
        "source_h5ad": str(args.input.resolve()),
        "source_h5ad_bytes": args.input.stat().st_size,
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "donor_manifest": str(args.donor_manifest.resolve()),
        "n_donors": int(len(donor_manifest)),
        "n_cells": int(donor_manifest["selected_cells"].sum()),
        "n_source_genes": int(len(source_genes)),
        "n_unique_gene_symbols": int(len(genes)),
        "collapsed_duplicate_gene_columns": int(len(source_genes) - len(genes)),
        "cell_policy": f"up to {args.max_cells} cells per donor without replacement",
        "sampling_seed": args.seed,
        "sampling_key": "sha256(seed, donor_id)",
        "minimum_selected_cells_per_donor": int(donor_manifest["selected_cells"].min()),
        "maximum_selected_cells_per_donor": int(donor_manifest["selected_cells"].max()),
        "target_labels_accessed": False,
        "target_age_dataset_opened": False,
        "target_statistics_fitted": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.audit.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return record

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--donor-manifest", required=True, type=Path)
    parser.add_argument("--cohort", default="OneK1K")
    parser.add_argument("--donor-column", default="donor_id")
    parser.add_argument("--gene-column", default="feature_name")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--max-cells", type=int, default=1500)
    parser.add_argument("--chunk-cells", type=int, default=4096)
    args = parser.parse_args()
    if args.max_cells < 1 or args.chunk_cells < 1:
        raise ValueError("cell limits must be positive")
    build(args)
    return 0

if __name__ == "__main__":
    main()

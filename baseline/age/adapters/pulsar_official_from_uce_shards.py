#!/usr/bin/env python3
"""Extract official PULSAR donor embeddings from frozen UCE cell embeddings."""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from baseline.age.workspace import WORKSPACE_ROOT


PULSAR_REPO = WORKSPACE_ROOT / "models/PULSAR-code"
PULSAR_MODEL = WORKSPACE_ROOT / "models/PULSAR"
PULSAR_COMMIT = "7442e4e7f17e0d3bbf5e15737e127d66979fd08b"


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values],
        dtype=object,
    )


def h5ad_column(group: h5py.Group, key: str) -> np.ndarray:
    value = group[key]
    if isinstance(value, h5py.Dataset):
        return decode(value[:]) if value.dtype.kind in {"O", "S", "U"} else value[:]
    codes = np.asarray(value["codes"][:], dtype=np.int64)
    categories = decode(value["categories"][:])
    result = np.empty(len(codes), dtype=object)
    result[codes < 0] = ""
    result[codes >= 0] = categories[codes[codes >= 0]]
    return result


def assert_source() -> dict[str, str]:
    commit = subprocess.check_output(
        ["git", "-C", str(PULSAR_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    return {"commit": commit}


def load_model(device: torch.device):
    sys.path.insert(0, str(PULSAR_REPO / "src"))
    from pulsar.model import PULSAR, PULSARConfig

    config = PULSARConfig.from_pretrained(str(PULSAR_MODEL), local_files_only=True)
    model = PULSAR.from_pretrained(
        str(PULSAR_MODEL), config=config, local_files_only=True
    )
    return model.eval().to(device).to(torch.bfloat16)


def load_cima(path: Path, member_column: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r", rdcc_nbytes=512 * 1024**2) as handle:
        member = np.asarray(handle[f"obs/{member_column}"][:], dtype=bool)
        donors_all = h5ad_column(handle["obs"], "donor_id").astype(str)
        positions = np.flatnonzero(member)
        donors = donors_all[positions]
        # One sequential read is much faster than h5py point selection of
        # non-contiguous rows.
        all_embeddings = np.asarray(handle["obsm/X_uce"][:], dtype=np.float32)
        embeddings = all_embeddings[positions].copy()
        del all_embeddings
    counts = {donor: int((donors == donor).sum()) for donor in np.unique(donors)}
    wrong = {donor: count for donor, count in counts.items() if count != 1024}
    if len(counts) != 417 or wrong:
        raise ValueError(f"CIMA fixed-view contract failed: {len(counts)} donors, {wrong}")
    return embeddings, donors


def load_external(root: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    chunks: list[np.ndarray] = []
    donor_chunks: list[np.ndarray] = []
    provenance: list[dict[str, object]] = []
    directories = sorted(root.glob("group_*"))
    if not directories:
        raise FileNotFoundError(f"no completed UCE groups under {root}")
    for directory in directories:
        marker_path = directory / "COMPLETE.json"
        embedding_path = directory / "cell_embeddings.npz"
        if not marker_path.is_file() or not embedding_path.is_file():
            raise FileNotFoundError(f"incomplete UCE group: {directory}")
        marker = json.loads(marker_path.read_text())
        if marker.get("status") != "PASS" or marker.get("target_labels_accessed") is not False:
            raise ValueError(f"invalid UCE proof: {marker_path}")
        archive = np.load(embedding_path, allow_pickle=False)
        x = np.asarray(archive["embeddings"], dtype=np.float32)
        donors = archive["donor_id"].astype(str)
        if x.shape != (len(donors), 1280) or not np.isfinite(x).all():
            raise ValueError(f"invalid UCE cell matrix: {embedding_path}")
        chunks.append(x)
        donor_chunks.append(donors)
        provenance.append(
            {
                "path": str(embedding_path.resolve()),
                "n_cells": int(len(donors)),
            }
        )
    return np.vstack(chunks), np.concatenate(donor_chunks), provenance


def extract(
    model,
    device: torch.device,
    cells: np.ndarray,
    cell_donors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    donor_order = list(dict.fromkeys(cell_donors.astype(str).tolist()))
    np.random.seed(0)
    torch.manual_seed(0)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for donor in donor_order:
            donor_cells = cells[cell_donors == donor]
            if not len(donor_cells):
                raise ValueError(f"donor without cells: {donor}")
            indices = np.random.choice(len(donor_cells), size=1024, replace=True)
            tensor = torch.from_numpy(donor_cells[indices]).to(
                device=device, dtype=torch.bfloat16
            ).unsqueeze(0)
            output = model(tensor)[0][:, 0, :].float().cpu().numpy()[0]
            if output.shape != (512,) or not np.isfinite(output).all():
                raise RuntimeError(f"invalid PULSAR output for {donor}")
            outputs.append(output.astype(np.float32))
    return np.vstack(outputs), np.asarray(donor_order, dtype=str)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--uce", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--member-column", default="member_20260711")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; PULSAR CPU fallback is forbidden")
    device = torch.device(args.device)
    torch.zeros(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    source = assert_source()
    model = load_model(device)
    started = time.perf_counter()
    if args.cohort == "CIMA":
        cells, donors = load_cima(args.uce, args.member_column)
        upstream = [{"path": str(args.uce.resolve())}]
    else:
        cells, donors, upstream = load_external(args.uce)
    embeddings, donor_order = extract(model, device, cells, donors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        embeddings=embeddings,
        donor_id=donor_order,
    )
    os.replace(temporary, args.output)
    manifest = {
        "status": "PASS",
        "method": "PULSAR",
        "cohort": args.cohort,
        "n_donors": int(len(donor_order)),
        "n_input_cells": int(len(donors)),
        "embedding_shape": list(embeddings.shape),
        "official_sample_cell_num": 1024,
        "official_resample_num": 1,
        "official_sampling_seed": 0,
        "official_sampling_replace": True,
        "official_ridge_alpha_downstream": 0.1,
        "target_labels_accessed": False,
        "target_statistics_accessed": False,
        "source": source,
        "upstream_uce": upstream,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

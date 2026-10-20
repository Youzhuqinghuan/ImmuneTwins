from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from baseline.age.adapters.prepare_scphase_practical_inputs import CONTRACT, OFFICIAL_COMMIT, decode, load_store, write_master, write_shard
from baseline.age.adapters.scphase_lazy_dense_loader import STORAGE_CONTRACT


def donor_rows(values: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    donor_ids = sorted(set(values.astype(str).tolist()))
    donor_index = {donor_id: index for index, donor_id in enumerate(donor_ids)}
    codes = np.fromiter(
        (donor_index[str(value)] for value in values),
        dtype=np.int32,
        count=len(values),
    )
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(donor_ids))
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    rows = [
        order[boundaries[index] : boundaries[index + 1]]
        for index in range(len(donor_ids))
    ]
    return donor_ids, rows

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-store", required=True, type=Path)
    parser.add_argument("--external-audit", required=True, type=Path)
    parser.add_argument("--frozen-scaler", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_audit = json.loads(args.external_audit.read_text())
    if (
        source_audit.get("status") != "PASS"
        or source_audit.get("target_labels_accessed") is not False
        or Path(source_audit.get("output", "")).resolve() != args.external_store.resolve()
    ):
        raise ValueError("external raw-count audit is not a label-free PASS artifact")

    scaler = np.load(args.frozen_scaler, allow_pickle=False)
    hvg_genes = scaler["genes"].astype(str).tolist()
    means = np.asarray(scaler["mean"], dtype=np.float64)
    stds = np.asarray(scaler["std"], dtype=np.float64)
    if (
        len(hvg_genes) != 5000
        or len(set(hvg_genes)) != 5000
        or means.shape != (5000,)
        or stds.shape != (5000,)
        or not np.isfinite(means).all()
        or not np.isfinite(stds).all()
        or (stds <= 0).any()
    ):
        raise ValueError("invalid frozen CIMA scPhase scaler")

    matrix, obs, external_genes = load_store(args.external_store)
    if len(set(external_genes)) != len(external_genes):
        raise ValueError("external raw-count store has duplicated gene symbols")
    external_position = {gene: index for index, gene in enumerate(external_genes)}
    target_positions = np.asarray(
        [index for index, gene in enumerate(hvg_genes) if gene in external_position],
        dtype=np.int64,
    )
    source_positions = np.asarray(
        [external_position[hvg_genes[index]] for index in target_positions],
        dtype=np.int64,
    )
    subset = matrix[:, source_positions].tocsr()
    del matrix
    gc.collect()

    donor_values = decode(np.asarray(obs["donor_id"])).astype(str)
    donors, rows_by_donor = donor_rows(donor_values)
    shard_dir = args.output_dir / "external_shards"
    shard_dir.mkdir()
    refs: list[dict[str, object]] = []
    for position, (donor, rows) in enumerate(zip(donors, rows_by_donor, strict=True)):
        raw = np.zeros((len(rows), 5000), dtype=np.float32)
        raw[:, target_positions] = subset[rows].toarray()
        transformed = (raw - means) / stds
        np.minimum(transformed, 10.0, out=transformed)
        if not np.isfinite(transformed).all():
            raise ValueError(f"non-finite frozen-scaled expression for {donor}")
        refs.append(
            write_shard(shard_dir / f"donor_{position:04d}.npy", transformed)
        )
        if (position + 1) % 25 == 0 or position + 1 == len(donors):
            print(
                f"wrote {position + 1}/{len(donors)} {args.cohort} donor shards",
                flush=True,
            )
    del subset, obs
    gc.collect()

    metadata = {
        "contract": CONTRACT,
        "storage_contract": STORAGE_CONTRACT,
        "upstream_input_state": "deterministic_max1500_raw_count_input",
        "cache_policy": "none",
        "cell_policy": "up to 1500 cells per donor without replacement",
        "preprocessing_role": "external_apply",
        "cv_strategy": "external_inference_only",
        "donor_order": "lexicographic",
        "target_labels_accessed": False,
        "target_statistics_fitted": False,
        "external_cohort": args.cohort,
        "observed_hvg_count": int(len(source_positions)),
        "missing_hvg_count": int(5000 - len(source_positions)),
        "missing_gene_fill": "zero before applying frozen CIMA scaler",
        "scale_api": "frozen CIMA scanpy.pp.scale(max_value=10)",
    }
    master = args.output_dir / "external_master.pkl"
    master_manifest = write_master(
        path=master,
        refs=refs,
        donor_ids=donors,
        labels=np.full(len(donors), np.nan),
        metadata=metadata,
        input_store=args.external_store,
        labels_present=False,
    )
    record = {
        "status": "PASS",
        "method_consumers": ["scPhase", "IDEAL-Age"],
        "official_scphase_commit": OFFICIAL_COMMIT,
        "cohort": args.cohort,
        "external_master": str(master.resolve()),
        "external_master_manifest": str(master_manifest.resolve()),
        "frozen_scaler": str(args.frozen_scaler.resolve()),
        "n_donors": len(donors),
        "n_cells": int(sum(int(ref["n_cells"]) for ref in refs)),
        "minimum_cells_per_donor": int(min(int(ref["n_cells"]) for ref in refs)),
        "maximum_cells_per_donor": int(max(int(ref["n_cells"]) for ref in refs)),
        "observed_hvgs": int(len(source_positions)),
        "missing_hvgs": int(5000 - len(source_positions)),
        "target_labels_accessed": False,
        "target_statistics_fitted": False,
    }
    (args.output_dir / "preparation_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    main()

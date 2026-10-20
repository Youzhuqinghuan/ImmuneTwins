from __future__ import annotations
import argparse
import csv
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from baseline.age.workspace import WORKSPACE_ROOT, resolve_workspace_artifact

DEFAULT_REPO = WORKSPACE_ROOT / "external_models_0714/pascient"

EXPECTED_COMMIT = "e1c645fa4c43546fd4f03c948c170429034556d0"

PREDICTION_COLUMNS = ["method", "cohort", "donor_id", "predicted_age"]


def move_batch_to_device(batch, device: torch.device):
    batch.x = batch.x.to(device, non_blocking=True)
    batch.padded_mask = batch.padded_mask.to(device, non_blocking=True)
    batch.sample_metadata = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.sample_metadata.items()
    }
    batch.cell_metadata.cell_level_labels = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.cell_metadata.cell_level_labels.items()
    }
    if batch.cell_metadata.ancestry_matrix is not None:
        batch.cell_metadata.ancestry_matrix = batch.cell_metadata.ancestry_matrix.to(
            device, non_blocking=True
        )
    if batch.dropout_mask is not None:
        batch.dropout_mask = batch.dropout_mask.to(device, non_blocking=True)
    return batch

def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite predictions: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("oof", "external"))
    parser.add_argument("--fold-runs", required=True, nargs=5, type=Path)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--official-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    repo = args.official_repo.resolve()
    sys.path.insert(0, str(repo))

    import hydra
    import lightning as L
    from omegaconf import OmegaConf

    from baseline.age.adapters.pascient_cima_datamodule import (
        PaSCientPredictionDataModule,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("PaSCient prediction is GPU-only, but CUDA is unavailable")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(
            f"invalid CUDA device {args.device}; visible devices={torch.cuda.device_count()}"
        )
    device = torch.device(f"cuda:{args.device}")
    probe = torch.ones(1, device=device)
    if float((probe + 1).item()) != 2.0:
        raise RuntimeError("CUDA tensor-operation probe failed")
    del probe

    output_artifacts = [
        args.output,
        args.output.with_suffix(".fold_predictions.csv"),
        args.output.with_suffix(".audit.json"),
    ]
    existing_outputs = [str(path) for path in output_artifacts if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"refusing to overwrite PaSCient predictions: {existing_outputs}")

    fold_records: list[tuple[Path, Path, dict, object]] = []
    observed_folds: set[int] = set()
    expected_training_store: str | None = None
    for run_dir in args.fold_runs:
        run_dir = run_dir.resolve()
        completion_path = run_dir / "run_complete.json"
        contract_path = run_dir / "data_contract.json"
        config_path = run_dir / "resolved_config.yaml"
        for required in (completion_path, contract_path, config_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        completion = json.loads(completion_path.read_text())
        contract = json.loads(contract_path.read_text())
        if completion.get("status") != "PASS":
            raise ValueError(f"incomplete PaSCient fold run: {run_dir}")
        if contract.get("schema_version") != "official_pascient_cima_fold_v1":
            raise ValueError(f"wrong PaSCient data contract schema in {contract_path}")
        fold = int(contract["fold"])
        if fold in observed_folds:
            raise ValueError(f"duplicate fold {fold}")
        observed_folds.add(fold)
        training_store = str(resolve_workspace_artifact(contract["store_path"]))
        if expected_training_store is None:
            expected_training_store = training_store
        elif (
            training_store != expected_training_store
        ):
            raise ValueError("five PaSCient folds did not use one frozen CIMA data contract")
        checkpoint = resolve_workspace_artifact(completion["evaluation_checkpoint"])
        cfg = OmegaConf.load(config_path)
        fold_records.append((run_dir, checkpoint, contract, cfg))
    if observed_folds != set(range(5)):
        raise ValueError(f"exactly folds 0-4 are required; found {sorted(observed_folds)}")
    fold_records.sort(key=lambda item: int(item[2]["fold"]))
    if (
        args.mode == 'oof'
        and str(args.store.resolve()) != expected_training_store
    ):
        raise ValueError("OOF prediction must use the CIMA training store")

    per_fold_rows: list[dict[str, object]] = []
    for _, checkpoint, contract, cfg in fold_records:
        fold = int(contract["fold"])
        if args.mode == "oof":
            donor_ids = list(map(str, contract["validation_donors"]))
        else:
            donor_ids = None

        # Reset RNGs before each fold so every fold model sees the same
        # 1,500-cell target sample.
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)
        L.seed_everything(12345, workers=True)
        datamodule = PaSCientPredictionDataModule(
            store_path=str(args.store.resolve()),
            donor_ids=donor_ids,
            sample_size=1500,
            batch_size=32,
            num_workers=args.num_workers,
            target_sum=10000.0,
            persistent_workers=args.num_workers > 0,
            multiprocessing_context=None,
            prefetch_factor=1,
        )
        model = hydra.utils.instantiate(cfg.model, metrics=[])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" not in payload:
            raise ValueError(f"not a Lightning checkpoint: {checkpoint}")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device)
        model.eval()

        scaler = contract["age_scaler"]
        age_mean = float(scaler["mean"])
        age_std = float(scaler["std"])
        seen: set[str] = set()
        with torch.inference_mode():
            for batch in datamodule.predict_dataloader():
                batch = move_batch_to_device(batch, device)
                donor_indices = batch.sample_metadata["donor_id"].detach().cpu().tolist()
                output = model.compute_step(batch, prefix="predict", log=False)
                standardized = output["patient_preds"][:, 0, 0].detach().float().cpu().numpy()
                if len(standardized) != len(donor_indices):
                    raise RuntimeError("PaSCient donor/prediction batch length mismatch")
                for donor_index, standardized_age in zip(
                    donor_indices, standardized, strict=True
                ):
                    donor_id = datamodule.int2donor[int(donor_index)]
                    if donor_id in seen:
                        raise ValueError(f"duplicate prediction for fold {fold}, donor {donor_id}")
                    seen.add(donor_id)
                    per_fold_rows.append(
                        {
                            "fold": fold,
                            "donor_id": donor_id,
                            "standardized_prediction": float(standardized_age),
                            "predicted_age": float(standardized_age) * age_std + age_mean,
                            "checkpoint": str(checkpoint),
                        }
                    )
                del output, batch
        expected_ids = set(donor_ids or datamodule.store.donor_ids)
        if seen != expected_ids:
            raise ValueError(
                f"fold {fold} prediction donor mismatch: "
                f"missing={sorted(expected_ids - seen)[:5]}, extra={sorted(seen - expected_ids)[:5]}"
            )

    if args.mode == "oof":
        if len({row["donor_id"] for row in per_fold_rows}) != len(per_fold_rows):
            raise ValueError("OOF donor appears in more than one fold")
        final_values = {
            str(row["donor_id"]): float(row["predicted_age"])
            for row in per_fold_rows
        }
        if len(final_values) != 417:
            raise ValueError(f"CIMA OOF coverage must be exactly 417 donors; got {len(final_values)}")
        aggregation = "one held-out fold prediction per CIMA donor"
    else:
        grouped: dict[str, list[float]] = {}
        for row in per_fold_rows:
            grouped.setdefault(str(row["donor_id"]), []).append(float(row["predicted_age"]))
        invalid = {donor: len(values) for donor, values in grouped.items() if len(values) != 5}
        if invalid:
            raise ValueError(f"external five-fold ensemble is incomplete: {invalid}")
        final_values = {donor: float(np.mean(values)) for donor, values in grouped.items()}
        aggregation = "unweighted arithmetic mean of five frozen CIMA fold models"

    final_rows = [
        {
            "method": "PaSCient",
            "cohort": args.cohort,
            "donor_id": donor_id,
            "predicted_age": format(final_values[donor_id], ".17g"),
        }
        for donor_id in sorted(final_values)
    ]
    write_predictions(args.output, final_rows)
    fold_path = args.output.with_suffix(".fold_predictions.csv")
    with fold_path.open("w", newline="") as handle:
        fields = [
            "fold",
            "donor_id",
            "standardized_prediction",
            "predicted_age",
            "checkpoint",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(per_fold_rows, key=lambda row: (row["fold"], row["donor_id"])))
    audit = {
        "schema_version": "official_pascient_prediction_v1",
        "official_repo": str(repo),
        "official_commit": EXPECTED_COMMIT,
        "official_prediction_call": "SamplePredictor.compute_step(prefix='predict', log=False)",
        "mode": args.mode,
        "cohort": args.cohort,
        "aggregation": aggregation,
        "n_predictions": len(final_rows),
        "n_fold_predictions": len(per_fold_rows),
        "expression_store": str(args.store.resolve()),
        "expression_store_labels_present": False,
        "target_label_files_opened": [],
        "target_prediction_distribution_used_for_calibration": False,
        "predictions": str(args.output.resolve()),
        "fold_predictions": str(fold_path.resolve()),
        "cuda_device": args.device,
        "cuda_name": torch.cuda.get_device_name(args.device),
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    main()

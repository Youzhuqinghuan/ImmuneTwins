from __future__ import annotations
from paths import expand_paths
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.age.checkpoint import load_checkpoint, require_cuda
from pipeline.age.inference import predict_external_shards
from pipeline.age.train import paper_method_name
from data.age import ImmunobiologyUceShardStore

def atomic_table(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-run", required=True, type=Path)
    parser.add_argument("--external-shards", required=True, type=Path)
    parser.add_argument("--donor-manifest", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cell-batch-size", type=int, default=4096)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "immunetwin_predictions.sealed.tsv"
    manifest_path = args.output_dir / "immunetwin_predictions.seal.json"
    if predictions_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite named external ImmuneTwin results")
    device = require_cuda(args.device)

    frozen_path = args.frozen_run / "cima_frozen_models.json"
    frozen = expand_paths(json.loads(frozen_path.read_text()))
    if (
        frozen.get("status") != "FROZEN_BEFORE_EXTERNAL"
        or frozen.get("all_CIMA_models_frozen_before_external") is not True
        or set(frozen.get("models", {})) != {"attention", "mean"}
    ):
        raise ValueError("full-CIMA frozen-model manifest contract failed")

    expected = pd.read_csv(args.donor_manifest, sep="\t")
    if "donor_id" not in expected or expected["donor_id"].duplicated().any():
        raise ValueError("invalid label-free external donor manifest")
    expected_ids = set(expected["donor_id"].astype(str))
    store = ImmunobiologyUceShardStore(
        args.external_shards,
        expected_groups=None,
    )
    embedded = store.expected_donor_manifest(
        expected_donors=len(expected_ids), expected_cells_per_donor=None
    )
    if set(embedded["donor_id"].astype(str)) != expected_ids:
        raise ValueError("UCE shard donors differ from the raw-count donor manifest")

    outputs: list[pd.DataFrame] = []
    checkpoint_records: dict[str, object] = {}
    for pooling in ("attention", "mean"):
        checkpoint_path = args.frozen_run / pooling / "full_cima" / "checkpoint.pt"
        model, scaler, payload = load_checkpoint(checkpoint_path, device)
        if (
            payload.get("outer_fold") is not None
            or payload.get("model_config", {}).get("pooling") != pooling
            or len(payload.get("training_donor_ids", [])) != 417
        ):
            raise ValueError(f"invalid full-CIMA {pooling} checkpoint identity")
        prediction = predict_external_shards(
            model,
            scaler,
            store,
            device,
            cell_batch_size=args.cell_batch_size,
            expected_donors=len(expected_ids),
            expected_cells_per_donor=None,
        )
        if set(prediction["donor_id"].astype(str)) != expected_ids:
            raise ValueError(f"{pooling} prediction donor coverage failed")
        prediction.insert(0, "cohort", args.cohort)
        prediction.insert(0, "method", paper_method_name(pooling))
        outputs.append(prediction)
        checkpoint_records[pooling] = {
            "path": str(checkpoint_path.resolve()),
            "training_scope": "all 417 CIMA donors",
        }

    combined = pd.concat(outputs, ignore_index=True).sort_values(["method", "donor_id"])
    if (
        len(combined) != 2 * len(expected_ids)
        or combined.duplicated(["method", "donor_id"]).any()
        or not np.isfinite(combined["predicted_age"]).all()
    ):
        raise ValueError("combined ImmuneTwin external prediction coverage failed")
    atomic_table(combined, predictions_path)
    record = {
        "status": "PASS",
        "methods": ["ImmuneTwin", "ImmuneTwin-Mean"],
        "cohort": args.cohort,
        "n_donors": len(expected_ids),
        "n_predictions": len(combined),
        "predictions": str(predictions_path.resolve()),
        "frozen_cima_models_manifest": str(frozen_path.resolve()),
        "checkpoints": checkpoint_records,
        "external_uce": store.audit(),
        "external_donor_manifest": str(args.donor_manifest.resolve()),
        "target_labels_accessed": False,
        "target_statistics_accessed": False,
        "target_calibration": False,
        "donor_independent_streaming_aggregation": True,
    }
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    main()

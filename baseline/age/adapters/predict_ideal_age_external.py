from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from baseline.age.adapters.run_ideal_age_full_ensemble import  OFFICIAL_REPO, frame
from baseline.age.adapters.run_ideal_age_official_cima import  import_official_model
from baseline.age.adapters.scphase_lazy_dense_loader import validate_master

def donor_seed(seed: int, donor_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{donor_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-master", required=True, type=Path)
    parser.add_argument("--frozen-scaler", required=True, type=Path)
    parser.add_argument("--deepsets-run", required=True, type=Path)
    parser.add_argument("--ensemble-run", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cells", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    from autogluon.tabular import TabularPredictor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "ideal_age_predictions.sealed.tsv"
    component_path = args.output_dir / "ideal_age_components.sealed.tsv"
    manifest_path = args.output_dir / "ideal_age_predictions.seal.json"
    if any(path.exists() for path in (final_path, component_path, manifest_path)):
        raise FileExistsError("refusing to overwrite frozen IDEAL-Age external results")
    if not torch.cuda.is_available():
        raise RuntimeError("IDEAL-Age DeepSets inference requires CUDA")
    device = torch.device(args.device)
    if float((torch.ones(1, device=device) + 1).item()) != 2.0:
        raise RuntimeError("IDEAL-Age CUDA tensor probe failed")

    commit = subprocess.check_output(
        ["git", "-C", str(OFFICIAL_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    ensemble_audit_path = args.ensemble_run / "audit.json"
    ensemble_audit = json.loads(ensemble_audit_path.read_text())
    weights = np.asarray(
        ensemble_audit["ensemble"]["full_cima_weights_for_external"],
        dtype=np.float64,
    )
    if weights.shape != (2,) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("invalid CIMA-only IDEAL-Age ensemble weights")

    payload = validate_master(
        args.external_master.resolve(),
        expected_role="external_apply",
    )
    refs = payload["data_list_dense"]
    donors = np.asarray(payload["sample_ids"], dtype=str)
    if len(refs) != len(donors) or len(set(donors)) != len(donors):
        raise ValueError("invalid external IDEAL-Age donor master")
    scaler = np.load(args.frozen_scaler, allow_pickle=False)
    genes = scaler["genes"].astype(str)
    if len(genes) != 5000 or len(set(genes)) != 5000:
        raise ValueError("IDEAL-Age frozen scaler does not contain 5000 unique genes")

    official = import_official_model()
    checkpoint_path = args.deepsets_run / "full_cima.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = official.DeepSetsAgePredictorAttn(
        input_dim=5000, hidden_dim=2**11, dropout=0.2, num_classes=10
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    pseudobulk = np.empty((len(donors), 5000), dtype=np.float32)
    deep_prediction = np.empty(len(donors), dtype=np.float64)
    selected_counts = np.empty(len(donors), dtype=np.int64)
    source_counts = np.empty(len(donors), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(donors), args.batch_size):
            stop = min(start + args.batch_size, len(donors))
            blocks: list[np.ndarray] = []
            for index in range(start, stop):
                shard = np.load(refs[index]["path"], mmap_mode="r", allow_pickle=False)
                values = np.asarray(shard, dtype=np.float32)
                if values.ndim != 2 or values.shape[1] != 5000:
                    raise ValueError(f"invalid IDEAL-Age donor shard: {donors[index]}")
                source_counts[index] = len(values)
                pseudobulk[index] = values.mean(axis=0, dtype=np.float64)
                selected_n = min(len(values), args.max_cells)
                rng = np.random.default_rng(donor_seed(args.seed, donors[index]))
                choice = np.sort(rng.choice(len(values), selected_n, replace=False))
                blocks.append(np.asarray(values[choice], dtype=np.float32))
                selected_counts[index] = selected_n
                del shard
            maximum = max(map(len, blocks))
            cells = torch.zeros(
                (len(blocks), maximum, 5000), dtype=torch.float32, device=device
            )
            mask = torch.zeros((len(blocks), maximum), dtype=torch.float32, device=device)
            for row, block in enumerate(blocks):
                count = len(block)
                cells[row, :count] = torch.from_numpy(block).to(device)
                mask[row, :count] = 1.0
            prediction, _ = model(cells, mask)
            deep_prediction[start:stop] = prediction.float().cpu().numpy()
            if stop % 100 < args.batch_size or stop == len(donors):
                print(f"IDEAL-Age DeepSets {stop}/{len(donors)} donors", flush=True)

    predictor_dir = args.ensemble_run / "autogluon_bagged_models"
    predictor = TabularPredictor.load(str(predictor_dir))
    pseudo_prediction = predictor.predict(frame(pseudobulk, genes, donors)).to_numpy(
        dtype=np.float64
    )
    components = np.column_stack([pseudo_prediction, deep_prediction])
    final_prediction = components @ weights
    if not np.isfinite(components).all() or not np.isfinite(final_prediction).all():
        raise ValueError("IDEAL-Age produced non-finite external predictions")

    component_frame = pd.DataFrame(
        {
            "donor_id": donors,
            "autogluon_pseudobulk": pseudo_prediction,
            "deepsets_attention": deep_prediction,
            "source_cells": source_counts,
            "deepsets_cells": selected_counts,
            "ensemble_prediction": final_prediction,
        }
    )
    component_frame.to_csv(component_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "method": "IDEAL-Age",
            "cohort": args.cohort,
            "donor_id": donors,
            "predicted_age": final_prediction,
        }
    ).to_csv(final_path, sep="\t", index=False)
    record = {
        "status": "PASS",
        "method": "IDEAL-Age",
        "cohort": args.cohort,
        "n_donors": len(donors),
        "official_repo_commit": commit,
        "components": [
            "AutoGluon FASTAI/NN_TORCH pseudo-bulk weighted ensemble",
            "DeepSetsAgePredictorAttn",
        ],
        "component_weights": weights.tolist(),
        "component_weight_fit_scope": "CIMA OOF predictions only",
        "full_cima_deepsets_checkpoint": str(checkpoint_path.resolve()),
        "autogluon_predictor": str(predictor_dir.resolve()),
        "external_master": str(args.external_master.resolve()),
        "minimum_source_cells": int(source_counts.min()),
        "maximum_source_cells": int(source_counts.max()),
        "deepsets_cell_policy": "up to 1000 cells without replacement; no padding in the attention mask",
        "target_labels_accessed": False,
        "target_statistics_fitted": False,
        "target_calibration": False,
    }
    temporary = manifest_path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from baseline.age.workspace import  WORKSPACE_ROOT

SCPHASE = WORKSPACE_ROOT / "external_models_official/scPhase"

DOMAIN_WEIGHT_KEY = "domain_classifier.domain_classifier.6.weight"

from baseline.age.adapters.scphase_lazy_dense_loader import AttnMoE_Dataset, sparse_collate_fn, validate_master

def ordered_checkpoints(paths: list[Path], expected: int) -> list[Path]:
    by_fold: dict[int, Path] = {}
    for path in paths:
        match = re.search(r"Fold(\d+)\.pt$", path.name)
        if match is None:
            raise ValueError(f"checkpoint does not use the released FoldN filename: {path}")
        fold = int(match.group(1))
        if fold in by_fold:
            raise ValueError(f"duplicate checkpoint fold {fold}")
        by_fold[fold] = path
    if set(by_fold) != set(range(1, expected + 1)):
        raise ValueError(
            f"checkpoint folds must be exactly 1..{expected}, found {sorted(by_fold)}"
        )
    return [by_fold[fold] for fold in range(1, expected + 1)]

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--external-master", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, action="append", type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not args.external_master.is_file():
        raise FileNotFoundError(args.external_master)
    config = json.loads(args.config.read_text())
    contract = config.get("adapter_contract")
    if not isinstance(contract, dict):
        raise ValueError("config lacks the strict scPhase adapter contract")
    if contract.get("storage_contract") != "scphase_per_donor_lazy_dense_float32_v1":
        raise ValueError("external inference requires the frozen lazy-dense storage contract")
    expected_checkpoints = int(contract["checkpoint_count"])
    checkpoints = ordered_checkpoints(args.checkpoint, expected_checkpoints)
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; scPhase inference must not fall back to CPU")
    probe = torch.ones(1, device=args.device) + 1
    if probe.item() != 2:
        raise RuntimeError("CUDA tensor-operation probe failed")

    sys.path.insert(0, str(SCPHASE / "scphase"))
    from model import SCMIL_AttnMoE

    external = validate_master(
        args.external_master,  expected_role="external_apply"
    )
    data = external["data_list_dense"]
    donors = [str(item) for item in external["sample_ids"]]
    external_metadata = external.get("adapter_metadata", {})
    if len(data) != len(donors) or len(set(donors)) != len(donors):
        raise ValueError("invalid external master donor contract")
    if external_metadata.get("preprocessing_role") != "external_apply":
        raise ValueError("external master is not a label-free external preprocessing artifact")
    if external_metadata.get("cv_strategy") != "external_inference_only":
        raise ValueError("external master carries an internal CV strategy")
    if external_metadata.get("target_labels_accessed") is not False:
        raise ValueError("external master does not prove label-free construction")
    if external_metadata.get("target_statistics_fitted") is not False:
        raise ValueError("external master fitted target-cohort statistics")
    allowed_external_states = {
        str(contract.get("upstream_input_state")),
        "deterministic_max1500_raw_count_input",
    }
    if external_metadata.get("upstream_input_state") not in allowed_external_states:
        raise ValueError("external master has an unsupported raw-count cell-view contract")

    dummy_labels = np.zeros(len(donors), dtype=np.float32)
    one_group = np.zeros(len(donors), dtype=np.int64)
    dataset = AttnMoE_Dataset(data, dummy_labels, one_group, donors)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=sparse_collate_fn
    )
    fold_predictions: list[np.ndarray] = []
    expected_domains = int(contract["checkpoint_num_domains"])
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location=args.device)
        if DOMAIN_WEIGHT_KEY not in state:
            raise ValueError(f"checkpoint lacks released domain-classifier weight: {checkpoint}")
        checkpoint_domains = int(state[DOMAIN_WEIGHT_KEY].shape[0])
        if checkpoint_domains != expected_domains:
            raise ValueError(
                f"checkpoint {checkpoint.name} has {checkpoint_domains} domains, expected {expected_domains}"
            )
        network = SCMIL_AttnMoE(
            config["model_params"],
            config["ablation_params"],
            checkpoint_domains,
            args.device,
            args.device,
        )
        network.load_state_dict(state)
        network.eval()
        values: list[float] = []
        with torch.inference_mode():
            for expression, _, _, _ in loader:
                expression = expression[0].float().to(args.device)
                _, age, _, _ = network(expression, alpha=1.0)
                value = float(age.detach().cpu().numpy().squeeze())
                if not np.isfinite(value):
                    raise ValueError(f"non-finite external prediction from {checkpoint.name}")
                values.append(value)
        fold_predictions.append(np.asarray(values, dtype=np.float64))
        del state, network
        torch.cuda.empty_cache()

    matrix = np.vstack(fold_predictions)
    ensemble = matrix.mean(axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["method", "cohort", "donor_id", "predicted_age"])
        for donor, value in zip(donors, ensemble):
            writer.writerow(["scPhase", args.cohort, donor, format(float(value), ".17g")])
    fold_path = args.output.with_suffix(".fold_predictions.csv")
    with fold_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["donor_id"]
            + [f"fold_{fold}_prediction" for fold in range(1, expected_checkpoints + 1)]
            + ["fold_sd"]
        )
        for index, donor in enumerate(donors):
            values = matrix[:, index]
            writer.writerow(
                [donor]
                + [format(float(value), ".17g") for value in values]
                + [format(float(values.std()), ".17g")]
            )
    manifest = {
        "method": "scPhase",
        "cohort": args.cohort,
        "n_donors": len(donors),
        "checkpoint_count": expected_checkpoints,
        "training_cv_strategy": contract["cv_strategy"],
        "ensemble": f"unweighted arithmetic mean of {expected_checkpoints} raw age predictions",
        "checkpoint_sd_use": "descriptive only",
        "external_input_state": external_metadata["upstream_input_state"],
        "target_labels_accessed": False,
        "target_statistics_accessed": False,
        "target_calibration": False,
    }
    args.output.with_suffix(".inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    main()

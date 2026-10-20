from __future__ import annotations
import argparse
import json
from pathlib import Path
from baseline.age.workspace import WORKSPACE_ROOT

OFFICIAL = WORKSPACE_ROOT / "external_models_official/scPhase/configs/config_age.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", default="CIMA_Age")
    parser.add_argument("--device-model", default="cuda:0")
    parser.add_argument("--device-encoder", default="cuda:0")
    args = parser.parse_args()

    if not args.master.is_file():
        raise FileNotFoundError(
            f"CSR master is absent: {args.master}. Refusing official H5AD dense fallback."
        )
    master_manifest_path = args.master.with_suffix(".manifest.json")
    if not master_manifest_path.is_file():
        raise FileNotFoundError(f"scPhase master manifest is absent: {master_manifest_path}")
    master_manifest = json.loads(master_manifest_path.read_text())
    if master_manifest.get("input_contract") != "official_age_cell_ranger_hvg5000_scale10_v1":
        raise ValueError("master does not prove the released scPhase Age preprocessing contract")
    if not master_manifest.get("labels_present"):
        if master_manifest.get("schema") != "scphase_lazy_dense_master_v1":
            raise ValueError("training master must contain CIMA age labels")
    if master_manifest.get("storage_contract") != "scphase_per_donor_lazy_dense_float32_v1":
        raise ValueError("formal scPhase training requires the no-cache lazy-dense master")
    strategy = master_manifest.get("cv_strategy")
    n_groups = int(master_manifest.get("n_groups", 0))
    if strategy == "LeaveOneGroupOut_real_RNA_experiment":
        if n_groups <= 1:
            raise ValueError("official-native LOGO master must retain multiple RNA_experiment groups")
        checkpoint_count = n_groups
        checkpoint_num_domains = n_groups - 1
    elif strategy == "KFold5_single_group_order_adapter":
        if n_groups != 1:
            raise ValueError("common five-fold master must contain exactly one synthetic group")
        checkpoint_count = 5
        checkpoint_num_domains = 1
    else:
        raise ValueError(f"unsupported scPhase training CV strategy: {strategy!r}")
    config = json.loads(OFFICIAL.read_text())
    config["path_params"]["_master_pickle_file"] = str(args.master.resolve())
    config["path_params"]["master_pickle_file"] = str(args.master.resolve())
    config["path_params"]["data_h5ad_file"] = str(
        args.master.with_name("FORBIDDEN_UNADAPTED_DENSE_FALLBACK.h5ad").resolve()
    )
    config["path_params"]["RESULTS_DIR"] = str(args.results_dir.resolve())
    config["path_params"]["LOGNAME"] = "training_CIMA_Age.log"
    config["path_params"]["MODEL_NAME"] = args.model_name
    config["run_params"]["device_model"] = args.device_model
    config["run_params"]["device_encoder"] = args.device_encoder
    config["adapter_contract"] = {
        "master_manifest": str(master_manifest_path.resolve()),
        "input_contract": master_manifest["input_contract"],
        "cv_strategy": strategy,
        "checkpoint_count": checkpoint_count,
        "checkpoint_num_domains": checkpoint_num_domains,
        "upstream_input_state": master_manifest["upstream_input_state"],
        "storage_contract": master_manifest["storage_contract"],
        "loader_adapter": "runtime_symbol_patch_no_cache_lazy_dense",
        "n_cells": master_manifest["n_cells"],
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps(config, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    main()

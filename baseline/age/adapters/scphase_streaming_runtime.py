from __future__ import annotations
import argparse
import importlib
import json
import runpy
import sys
from pathlib import Path
from baseline.age.workspace import REPOSITORY_ROOT, WORKSPACE_ROOT

SCPHASE = WORKSPACE_ROOT / "external_models_official/scPhase"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    contract = config.get("adapter_contract")
    if not isinstance(contract, dict):
        raise ValueError("scPhase config lacks an adapter contract")
    if contract.get("loader_adapter") != "runtime_symbol_patch_no_cache_lazy_dense":
        raise ValueError("scPhase config does not authorize the frozen loader-only patch")
    if contract.get("storage_contract") != "scphase_per_donor_lazy_dense_float32_v1":
        raise ValueError("scPhase runtime requires the frozen lazy-dense storage contract")
    expected_training = {
        "epochs": 150, "batch_size": 1, "num_workers": 4,
        "early_stopping_patience": 20, "val_size": 0.15,
    }
    if config.get("training_params") != expected_training:
        raise ValueError("scPhase official Age training hyperparameters changed")
    master = Path(config["path_params"].get("_master_pickle_file", ""))
    manifest = Path(contract.get("master_manifest", ""))
    if not master.is_file() or not manifest.is_file():
        raise FileNotFoundError("scPhase lazy master/manifest is absent")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; scPhase must not use its CPU fallback")
    for device in sorted(
        {config["run_params"]["device_model"], config["run_params"]["device_encoder"]}
    ):
        if not str(device).startswith("cuda:"):
            raise ValueError("both frozen scPhase devices must be CUDA devices")
        if (torch.ones(1, device=device) + 1).item() != 2:
            raise RuntimeError(f"CUDA tensor-operation probe failed on {device}")

    sys.path.insert(0, str(REPOSITORY_ROOT))
    sys.path.insert(0, str(SCPHASE))
    sys.path.insert(0, str(SCPHASE / "scphase"))

    import data_loader as released_data_loader
    from baseline.age.adapters import scphase_lazy_dense_loader as lazy

    # run_cv.py imports these names directly, so they are patched before
    # run.py executes; the released training code is unchanged.
    released_data_loader.load_data = lazy.load_data
    released_data_loader.AttnMoE_Dataset = lazy.AttnMoE_Dataset
    released_data_loader.sparse_collate_fn = lazy.sparse_collate_fn
    sys.modules["scphase.data_loader"] = released_data_loader

    released_cv = importlib.import_module("scphase.run_cv")
    if not (
        released_cv.load_data is lazy.load_data
        and released_cv.AttnMoE_Dataset is lazy.AttnMoE_Dataset
        and released_cv.sparse_collate_fn is lazy.sparse_collate_fn
    ):
        raise RuntimeError("scPhase loader patch did not bind the exact run_cv import symbols")

    sys.argv = [str(SCPHASE / "run.py"), "--config", str(args.config.resolve()), "--cv"]
    runpy.run_path(str(SCPHASE / "run.py"), run_name="__main__")
    return 0

if __name__ == "__main__":
    main()

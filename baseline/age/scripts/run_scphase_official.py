from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from baseline.age.workspace import REPOSITORY_ROOT, WORKSPACE_ROOT

SCPHASE = WORKSPACE_ROOT / "external_models_official/scPhase"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    master = Path(config["path_params"].get("_master_pickle_file", ""))
    if not master.is_file():
        raise FileNotFoundError(
            f"lazy-dense master is absent: {master}. Refusing official H5AD dense fallback."
        )
    contract = config.get("adapter_contract")
    if not isinstance(contract, dict):
        raise ValueError("config lacks the strict scPhase preprocessing/CV contract")
    if contract.get("input_contract") != "official_age_cell_ranger_hvg5000_scale10_v1":
        raise ValueError("config does not prove the released scPhase Age preprocessing")
    manifest = Path(contract["master_manifest"])
    if not manifest.is_file():
        raise ValueError("scPhase master manifest is absent or has changed since config creation")
    if contract.get("storage_contract") != "scphase_per_donor_lazy_dense_float32_v1":
        raise ValueError("scPhase training requires the frozen lazy-dense storage contract")
    if contract.get("loader_adapter") != "runtime_symbol_patch_no_cache_lazy_dense":
        raise ValueError("scPhase training does not authorize the no-cache loader patch")
    if config["training_params"] != {
        "epochs": 150,
        "batch_size": 1,
        "num_workers": 4,
        "early_stopping_patience": 20,
        "val_size": 0.15,
    }:
        raise ValueError("scPhase official Age training hyperparameters changed")
    checkout_status = subprocess.check_output(
        ["git", "-C", str(SCPHASE), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    for line in checkout_status.splitlines():
        if line.startswith("?? "):
            if line[3:].endswith(".py"):
                raise ValueError(f"untracked Python file could shadow scPhase imports: {line[3:]}")
        else:
            raise ValueError(f"scPhase official checkout has tracked modifications: {line}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; scPhase must not use its CPU fallback")
    device = config["run_params"]["device_model"]
    probe = torch.ones(1, device=device) + 1
    if probe.item() != 2:
        raise RuntimeError("CUDA tensor-operation probe failed")

    runtime = REPOSITORY_ROOT / "baseline/age/adapters/scphase_streaming_runtime.py"
    command = [sys.executable, str(runtime), "--config", str(args.config.resolve())]
    environment = os.environ.copy()
    python_paths = [str(SCPHASE), str(SCPHASE / "scphase")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    python_paths.insert(0, str(REPOSITORY_ROOT))
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    os.chdir(SCPHASE)
    os.execve(sys.executable, command, environment)
    return 0


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from baseline.age.workspace import REPOSITORY_ROOT, WORKSPACE_ROOT, resolve_workspace_artifact
from baseline.age.adapters.pascient_loss import LOSS_TARGET, LOSS_CONTRACT, loss_contract

DEFAULT_REPO = WORKSPACE_ROOT / "external_models_0714/pascient"

EXPECTED_COMMIT = "e1c645fa4c43546fd4f03c948c170429034556d0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    training_scope = parser.add_mutually_exclusive_group(required=True)
    training_scope.add_argument("--fold", type=int, choices=range(5))
    training_scope.add_argument(
        "--full-cima",
        action="store_true",
        help="Fit the external model on all CIMA donors.",
    )
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--official-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--expected-donors", type=int, default=417)
    parser.add_argument(
        "--resource-pilot",
        action="store_true",
        help="Use Lightning fast_dev_run for one real train and validation batch.",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    repo = args.official_repo.resolve()
    if not args.store.is_file() or not args.split.is_file():
        raise FileNotFoundError("CIMA store or split file is missing")
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise FileExistsError(f"refusing to mix with an existing run artifact: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPOSITORY_ROOT))
    sys.path.insert(0, str(repo))
    os.environ.setdefault("PROJECT_ROOT", str(repo))

    import h5py
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    if not torch.cuda.is_available():
        raise RuntimeError("PaSCient formal training is GPU-only, but CUDA is unavailable")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(
            f"invalid CUDA device {args.device}; visible devices={torch.cuda.device_count()}"
        )
    probe = torch.ones(1, device=f"cuda:{args.device}")
    if float((probe + 1).item()) != 2.0:
        raise RuntimeError("CUDA tensor-operation probe failed")
    del probe

    with h5py.File(args.store, "r") as handle:
        store_format = handle.attrs.get("format")
        if isinstance(store_format, bytes):
            store_format = store_format.decode("utf-8")
        if store_format != "pascient_donor_csr_v1":
            raise ValueError("training input is not a PaSCient donor CSR store")
        if bool(handle.attrs.get("labels_present", True)):
            raise ValueError("training expression store unexpectedly contains labels")
        n_genes = len(handle["genes"])
        n_store_donors = len(handle["donor_ids"])
    store_manifest_path = args.store.with_suffix(".manifest.json")
    if not store_manifest_path.is_file():
        raise FileNotFoundError(store_manifest_path)
    store_manifest = json.loads(store_manifest_path.read_text())
    if resolve_workspace_artifact(store_manifest.get("output", "")) != args.store.resolve():
        raise ValueError("PaSCient store manifest points to a different output path")
    if store_manifest.get("format") != "pascient_donor_csr_v1":
        raise ValueError("PaSCient store manifest has the wrong format")
    if store_manifest.get("labels_present") is not False:
        raise ValueError("PaSCient store manifest does not certify label absence")
    if int(store_manifest.get("n_donors", -1)) != int(args.expected_donors):
        raise ValueError("PaSCient store manifest donor count mismatch")
    if n_store_donors != int(args.expected_donors):
        raise ValueError("PaSCient HDF5 donor count mismatch")
    if int(store_manifest.get("n_genes", -1)) != n_genes:
        raise ValueError("PaSCient store manifest gene count mismatch")
    if int(store_manifest.get("output_size_bytes", -1)) != args.store.stat().st_size:
        raise ValueError("PaSCient store size differs from its builder manifest")
    gene_list_path = resolve_workspace_artifact(store_manifest.get("gene_list", ""))
    if not gene_list_path.is_file():
        raise FileNotFoundError(gene_list_path)

    config_dir = repo / "pascient/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=pascient_multilabel",
                "model/losses/sample_prediction_loss=continuous",
            ],
        )

    # The released experiment is a categorical patient endpoint.  Only its
    # endpoint schema is changed to the release's own continuous loss/head.
    cfg.model.losses.sample_prediction_loss.labels = ["age"]
    cfg.model.patient_predictor.output_dim = 1
    # Hydra merges the disease experiment's nine-class loss options into the
    # selected continuous config.  ``weight`` is a CrossEntropy-only argument
    # and is not part of the released MSELossViews configuration.
    if "weight" in cfg.model.losses.sample_prediction_loss.loss_fn:
        with open_dict(cfg.model.losses.sample_prediction_loss.loss_fn):
            del cfg.model.losses.sample_prediction_loss.loss_fn["weight"]
    # The release broadcasts [N,1] against [N,1,1]. Keep its per-view MSE
    # objective, but compare each patient only with their own age target.
    cfg.model.losses.sample_prediction_loss.loss_fn._target_ = LOSS_TARGET
    cfg.model.num_genes = int(n_genes)
    cfg.metrics = None
    cfg.logger = None

    # Paths and devices only; model, optimizer, schedule, sampling count,
    # batch size and seed keep the official defaults.
    full_cima = bool(args.full_cima)
    data_config = {
        "_target_": (
            "baseline.age.adapters.pascient_cima_datamodule."
            + ("CIMAFullPaSCientDataModule" if full_cima else "CIMAPaSCientDataModule")
        ),
        "store_path": str(args.store.resolve()),
        "split_path": str(args.split.resolve()),
        "repeat": int(args.repeat),
        "contract_path": str((args.output / "data_contract.json").resolve()),
        "expected_donors": int(args.expected_donors),
        "sample_size": 1500,
        "batch_size": 32,
        "num_workers": int(args.num_workers),
        "target_sum": 10000.0,
        "persistent_workers": bool(args.num_workers > 0),
        "multiprocessing_context": None,
        "prefetch_factor": 1,
        "age_loss_contract": LOSS_CONTRACT,
    }
    if not full_cima:
        data_config["fold"] = int(args.fold)
    cfg.data = OmegaConf.create(data_config)
    cfg.seed = 12345
    cfg.train = True
    cfg.test = False
    cfg.trainer.accelerator = "gpu"
    cfg.trainer.devices = [int(args.device)]
    cfg.trainer.max_epochs = 4
    if args.resource_pilot:
        OmegaConf.update(cfg, "trainer.fast_dev_run", 1, force_add=True)
    cfg.paths.root_dir = str(repo)
    cfg.paths.data_dir = str((repo / "data").resolve()) + "/"
    cfg.paths.log_dir = str((args.output / "logs").resolve()) + "/"
    cfg.paths.output_dir = str(args.output.resolve())
    cfg.paths.work_dir = str(REPOSITORY_ROOT)
    cfg.callbacks.model_checkpoint.dirpath = str((args.output / "checkpoints").resolve())

    loss_target = str(cfg.model.losses.sample_prediction_loss.loss_fn._target_)
    if loss_target != LOSS_TARGET:
        raise RuntimeError(f"unexpected PaSCient continuous loss: {loss_target}")
    expected = {
        "epochs": int(cfg.trainer.max_epochs),
        "sample_size": int(cfg.data.sample_size),
        "batch_size": int(cfg.data.batch_size),
        "seed": int(cfg.seed),
        "lr": float(cfg.model.optimizer.lr),
        "weight_decay": float(cfg.model.optimizer.weight_decay),
        "scheduler_patience": int(cfg.model.scheduler.patience),
        "early_stopping_patience": int(cfg.callbacks.early_stopping.patience),
    }
    if expected != {
        "epochs": 4,
        "sample_size": 1500,
        "batch_size": 32,
        "seed": 12345,
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "scheduler_patience": 20,
        "early_stopping_patience": 100,
    }:
        raise RuntimeError(f"official PaSCient defaults drifted: {expected}")

    OmegaConf.resolve(cfg)
    config_path = args.output / "resolved_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    launch = {
        "schema_version": "official_pascient_launch_v1",
        "official_repo": str(repo),
        "official_commit": EXPECTED_COMMIT,
        "official_entrypoint": "pascient.scripts.train.train",
        "training_scope": "full_cima" if full_cima else "cv_fold",
        "fold": args.fold,
        "repeat": args.repeat,
        "resolved_config": str(config_path.resolve()),
        "cuda_device": args.device,
        "cuda_name": torch.cuda.get_device_name(args.device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "runtime_only_num_workers": args.num_workers,
        "resource_pilot": bool(args.resource_pilot),
        "expected_donors": int(args.expected_donors),
        "store_size_bytes": args.store.stat().st_size,
        "age_loss_contract": loss_contract(),
        "python_executable": sys.executable,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (args.output / "launch_manifest.json").write_text(
        json.dumps(launch, indent=2, sort_keys=True) + "\n"
    )

    # Released Lightning orchestration function.
    from pascient.scripts.train import train as official_train

    if args.resource_pilot:
        torch.cuda.reset_peak_memory_stats(args.device)
    official_train(cfg)

    if args.resource_pilot:
        completion = {
            **launch,
            "status": "PILOT_PASS",
            "official_train_function_called": True,
            "fast_dev_run_batches": 1,
            "peak_cuda_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(args.device)
            ),
            "peak_cuda_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(args.device)
            ),
            "formal_hyperparameters_unchanged": {
                "epochs": 4,
                "batch_size": 32,
                "sample_size": 1500,
                "optimizer": "Adam",
                "learning_rate": 0.0001,
                "weight_decay": 0.0001,
            },
        }
        (args.output / "pilot_complete.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(completion, indent=2, sort_keys=True))
        return 0

    last_checkpoint = args.output / "checkpoints/last.ckpt"
    if not last_checkpoint.is_file():
        raise FileNotFoundError("official PaSCient run did not produce checkpoints/last.ckpt")
    last_payload = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    best_paths = {
        str(state["best_model_path"])
        for state in last_payload.get("callbacks", {}).values()
        if isinstance(state, dict) and state.get("best_model_path")
    }
    if len(best_paths) != 1:
        raise RuntimeError(
            "could not recover the unique official ModelCheckpoint best_model_path "
            f"from last.ckpt: {sorted(best_paths)}"
        )
    evaluation_checkpoint = Path(best_paths.pop())
    if not evaluation_checkpoint.is_file():
        raise FileNotFoundError(evaluation_checkpoint)
    completion = {
        **launch,
        "status": "PASS",
        "last_checkpoint": str(last_checkpoint.resolve()),
        "official_best_checkpoint_for_audit_only": str(evaluation_checkpoint.resolve()),
        "evaluation_checkpoint": str(last_checkpoint.resolve()),
        "checkpoint_selection": (
            "fixed released epoch 4 last.ckpt; no validation or external label selects an epoch"
            if full_cima
            else "fixed released epoch 4 last.ckpt; outer held-out CIMA labels do not select an epoch"
        ),
        "literal_released_mse_broadcasting_preserved": False,
    }
    (args.output / "run_complete.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    main()

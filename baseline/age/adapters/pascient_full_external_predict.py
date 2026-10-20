from __future__ import annotations
import argparse
import json
import os
import random
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch
from baseline.age.workspace import  WORKSPACE_ROOT, resolve_workspace_artifact

DEFAULT_REPO = WORKSPACE_ROOT / "external_models_0714/pascient"

EXPECTED_COMMIT = "e1c645fa4c43546fd4f03c948c170429034556d0"

from baseline.age.adapters.pascient_external_predict import move_batch_to_device, write_predictions

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run", required=True, type=Path)
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
        raise RuntimeError("PaSCient external prediction is GPU-only, but CUDA is unavailable")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(
            f"invalid CUDA device {args.device}; visible devices={torch.cuda.device_count()}"
        )
    device = torch.device(f"cuda:{args.device}")
    probe = torch.ones(1, device=device)
    if float((probe + 1).item()) != 2.0:
        raise RuntimeError("CUDA tensor-operation probe failed")
    del probe

    output_artifacts = [args.output, args.output.with_suffix(".audit.json")]
    existing = [str(path) for path in output_artifacts if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite PaSCient external outputs: {existing}")

    run_dir = args.full_run.resolve()
    completion_path = run_dir / "run_complete.json"
    contract_path = run_dir / "data_contract.json"
    config_path = run_dir / "resolved_config.yaml"
    for required in (completion_path, contract_path, config_path, args.store):
        if not required.is_file():
            raise FileNotFoundError(required)
    completion = json.loads(completion_path.read_text())
    contract = json.loads(contract_path.read_text())
    if completion.get("status") != "PASS":
        raise ValueError("full-CIMA PaSCient run is incomplete")
    if completion.get("training_scope") != "full_cima":
        raise ValueError("external checkpoint was not trained with the full-CIMA route")
    if contract.get("schema_version") != "official_pascient_cima_full_v1":
        raise ValueError("wrong full-CIMA PaSCient data contract")
    checkpoint = resolve_workspace_artifact(completion["evaluation_checkpoint"])

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    L.seed_everything(12345, workers=True)
    datamodule = PaSCientPredictionDataModule(
        store_path=str(args.store.resolve()),
        donor_ids=None,
        sample_size=1500,
        batch_size=32,
        num_workers=args.num_workers,
        target_sum=10000.0,
        persistent_workers=args.num_workers > 0,
        multiprocessing_context=None,
        prefetch_factor=1,
    )
    cfg = OmegaConf.load(config_path)
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
    values: dict[str, float] = {}
    inference_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
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
                    raise ValueError(f"duplicate external prediction for donor {donor_id}")
                seen.add(donor_id)
                values[donor_id] = float(standardized_age) * age_std + age_mean
            del output, batch
            print(f"predicted {len(seen)}/{len(datamodule.store.donor_ids)} donors; "
                  f"elapsed_s={time.monotonic()-inference_started:.1f}; "
                  f"peak_cuda_gib={torch.cuda.max_memory_allocated(device)/1024**3:.2f}",flush=True)
    expected_ids = set(datamodule.store.donor_ids)
    if seen != expected_ids:
        raise ValueError(
            "external prediction donor mismatch: "
            f"missing={sorted(expected_ids - seen)[:5]}, extra={sorted(seen - expected_ids)[:5]}"
        )

    rows = [
        {
            "method": "PaSCient",
            "cohort": args.cohort,
            "donor_id": donor_id,
            "predicted_age": format(values[donor_id], ".17g"),
        }
        for donor_id in sorted(values)
    ]
    write_predictions(args.output, rows)
    audit = {
        "schema_version": "official_pascient_full_external_prediction_v1",
        "official_repo": str(repo),
        "official_commit": EXPECTED_COMMIT,
        "official_prediction_call": "SamplePredictor.compute_step(prefix='predict', log=False)",
        "training_scope": "all 417 CIMA donors",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_selection": "fixed released epoch 4 last.ckpt",
        "cohort": args.cohort,
        "n_predictions": len(rows),
        "expression_store": str(args.store.resolve()),
        "expression_store_labels_present": False,
        "target_label_files_opened": [],
        "target_prediction_distribution_used_for_calibration": False,
        "age_scaler_fit_scope": scaler["fit_scope"],
        "predictions": str(args.output.resolve()),
        "cuda_device": args.device,
        "cuda_name": torch.cuda.get_device_name(args.device),
        "runtime": {
            "python_executable":sys.executable,"python_version":sys.version.split()[0],
            "torch_version":torch.__version__,"torch_cuda_version":torch.version.cuda,
            "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
            "random_seed":12345,"cells_per_donor":1500,"batch_size":32,
            "num_workers":args.num_workers,
            "elapsed_inference_seconds":time.monotonic()-inference_started,
            "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device),
            "peak_process_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
        },
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    main()

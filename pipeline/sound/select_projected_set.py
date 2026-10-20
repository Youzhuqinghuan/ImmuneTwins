from __future__ import annotations
import argparse
from dataclasses import fields
import json
import math
from pathlib import Path
import time
from typing import Sequence
import pandas as pd
import torch
from model.encoder import VaccineImmuneTwinConfig
from model.projected_set import MomentFiberCurrentModel
from data.sound import FoldLatentStore, SealedBaseUCEStore, UnitMaterializer, build_task_units, canonical_task, load_fold_scale, split_inner_units, split_outer_units
from pipeline.sound.provenance import V2_TRAIN_SCHEMA
from pipeline.sound.common import atomic_table, atomic_torch_save, require_cuda, set_seed
from pipeline.sound.train import evaluate_donors_v2, load_configuration_v2, train_epoch_v2

SCHEMA = "sound_vaccine_moment_fiber_pilot_1"

def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-units", type=Path, required=True)
    parser.add_argument("--inner-splits", type=Path, required=True)
    parser.add_argument("--training-latent", type=Path, required=True)
    parser.add_argument("--scale", type=Path, required=True)
    parser.add_argument("--sealed-source-manifest", type=Path, required=True)
    parser.add_argument("--sealed-uce", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser

def _load_model(
    checkpoint_path: Path,
    requested: VaccineImmuneTwinConfig,
) -> tuple[MomentFiberCurrentModel, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path.resolve(strict=True), map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("format") != V2_TRAIN_SCHEMA
        or checkpoint.get("arm") != "current"
        or checkpoint.get("phase") != "best_validation"
    ):
        raise ValueError("pilot requires a current-arm best_validation v2 checkpoint")
    observed = checkpoint.get("model_config")
    if not isinstance(observed, dict) or observed.get("current_operator") != "set_affine":
        raise ValueError("pilot base must be ImmuneTwin-SetTransport")
    permitted = {field.name for field in fields(VaccineImmuneTwinConfig)}
    shared = VaccineImmuneTwinConfig(
        **{key: value for key, value in observed.items() if key in permitted}
    )
    for name in (
        "uce_dim",
        "visit_dim",
        "attention_dim",
        "latent_dim",
        "phase_count",
        "mean_program_rank",
        "covariance_program_rank",
        "source_geometry_dim",
        "operator_slot_count",
    ):
        if getattr(shared, name) != getattr(requested, name):
            raise ValueError(f"base/current configuration differs at {name}")
    model = MomentFiberCurrentModel(requested)
    loaded = model.load_frozen_set_transport_state(checkpoint["model_state_dict"])
    return model, {
        "path": str(checkpoint_path.resolve()),
        "paper_task": checkpoint.get("paper_task"),
        "outer_fold": checkpoint.get("outer_fold"),
        "phase": checkpoint.get("phase"),
        "loaded_backbone_tensors": len(loaded),
    }

def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    task = canonical_task(args.task)
    payload, runtime, model_config, loss_config = load_configuration_v2(
        args.config, arm="current", task=task
    )
    if model_config.current_operator != "moment_fiber":
        raise ValueError("pilot config must select moment_fiber")
    if args.base_checkpoint.resolve().parent.name != f"outer{args.fold}_seed42":
        raise ValueError("base checkpoint fold differs from requested fold")
    device = require_cuda()
    set_seed(runtime.seed)
    model, dependency = _load_model(args.base_checkpoint, model_config)
    if dependency["paper_task"] != task or dependency["outer_fold"] != args.fold:
        raise ValueError("base checkpoint task/fold differs from requested pilot")
    torch.set_num_threads(4)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=runtime.learning_rate,
        weight_decay=runtime.weight_decay,
    )
    response_scale = model.transport.response_scale.detach().cpu().numpy()
    units = build_task_units(args.task_units, task)
    outer_train, _ = split_outer_units(units, args.fold)
    fit_units, validation_units = split_inner_units(
        outer_train, args.inner_splits, outer_fold=args.fold, validation_fold=0
    )
    scale = load_fold_scale(args.scale, outer_fold=args.fold)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    uce = SealedBaseUCEStore(args.sealed_source_manifest, args.sealed_uce)
    latent = FoldLatentStore(args.training_latent, scale)
    try:
        materializer = UnitMaterializer(
            uce=uce,
            latent=latent,
            query_uce_cells=runtime.query_uce_cells,
            support_uce_cells=runtime.support_uce_cells,
            seed=runtime.seed,
        )
        baseline = evaluate_donors_v2(
            model,
            materializer,
            validation_units,
            arm="current",
            runtime=runtime,
            loss_config=loss_config,
            bandwidth=scale.mmd_bandwidth,
            response_scale=response_scale,
            clinical_mode="off",
            device=device,
        )
        history: list[dict[str, float]] = []
        best = float(baseline["distribution_mmd2"])
        best_epoch = 0
        stale = 0
        atomic_torch_save(
            output / "best.pt",
            {
                "schema_version": SCHEMA,
                "epoch": 0,
                "base_checkpoint": dependency,
                "model_config": model.config.to_dict(),
                "model_state_dict": {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                },
                "validation": baseline,
            },
        )
        for epoch in range(1, runtime.maximum_epochs + 1):
            training = train_epoch_v2(
                model,
                optimizer,
                materializer,
                fit_units,
                arm="current",
                epoch=epoch,
                runtime=runtime,
                loss_config=loss_config,
                bandwidth=scale.mmd_bandwidth,
                response_scale=response_scale,
                clinical_mode="off",
                device=device,
            )
            validation = evaluate_donors_v2(
                model,
                materializer,
                validation_units,
                arm="current",
                runtime=runtime,
                loss_config=loss_config,
                bandwidth=scale.mmd_bandwidth,
                response_scale=response_scale,
                clinical_mode="off",
                device=device,
            )
            row = {
                "epoch": float(epoch),
                **{f"train_{key}": float(value) for key, value in training.items()},
                **{
                    f"validation_{key}": float(value)
                    for key, value in validation.items()
                },
            }
            history.append(row)
            score = float(validation["distribution_mmd2"])
            if math.isfinite(score) and score < best - 1e-8:
                best = score
                best_epoch = epoch
                stale = 0
                atomic_torch_save(
                    output / "best.pt",
                    {
                        "schema_version": SCHEMA,
                        "epoch": epoch,
                        "base_checkpoint": dependency,
                        "model_config": model.config.to_dict(),
                        "model_state_dict": {
                            name: value.detach().cpu()
                            for name, value in model.state_dict().items()
                        },
                        "validation": validation,
                    },
                )
            else:
                stale += 1
            atomic_table(output / "history.tsv", pd.DataFrame(history))
            _atomic_json(
                output / "progress.json",
                {
                    "status": "running",
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "baseline_mmd2": float(baseline["distribution_mmd2"]),
                    "best_mmd2": best,
                    "stale_epochs": stale,
                },
            )
            if stale >= runtime.patience:
                break
        best_payload = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        summary = {
            "status": "pass",
            "schema_version": SCHEMA,
            "task": task,
            "outer_fold": args.fold,
            "seed": runtime.seed,
            "model_initialized_after_seed": True,
            "fit_donors": len({unit.donor_id for unit in fit_units}),
            "validation_donors": len({unit.donor_id for unit in validation_units}),
            "training_latent_cells": runtime.training_latent_cells,
            "base_checkpoint": dependency,
            "baseline_validation": baseline,
            "best_epoch": int(best_payload["epoch"]),
            "best_validation": best_payload["validation"],
            "mmd2_change": float(best_payload["validation"]["distribution_mmd2"])
            - float(baseline["distribution_mmd2"]),
            "elapsed_seconds": time.time() - started,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "history_path": str((output / "history.tsv").resolve()),
            "checkpoint_path": str((output / "best.pt").resolve()),
            "config_path": str(args.config.resolve()),
            "test_set_opened": False,
        }
        _atomic_json(output / "summary.json", summary)
        _atomic_json(output / "progress.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        latent.close()
        uce.close()

if __name__ == "__main__":
    main()

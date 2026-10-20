from __future__ import annotations
from paths import expand_paths
import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import time
from typing import Mapping, Sequence
import pandas as pd
import torch
from model.digital_twin import MetaAdapterConfig, MetaAdapterDigitalTwin
from model.encoder import VaccineImmuneTwinConfig
from model.projected_set import MomentFiberCurrentModel
from data.sound import FoldLatentStore, UnifiedUCEStore, UnitMaterializer, build_task_units, canonical_task, load_fold_scale, split_outer_units
from model.losses import PatientLossV2Config
from data.history import HISTORY_POLICIES, augment_with_complete_prior_rna
from pipeline.sound.provenance import atomic_json
from pipeline.sound.common import atomic_table, atomic_torch_save, require_cuda, set_seed
from pipeline.sound.train import load_configuration_v2, predict_heldout_v2, runtime_resource_snapshot, train_epoch_v2

SCHEMA = "sound_meta_adapter_digital_twin_formal_1"

PAPER_METHOD = "ImmuneTwins (full)"

IMPLEMENTATION = "history_generated_shared_basis_lora_1"

def _parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--task", required=True)
    value.add_argument("--fold", type=int, choices=range(5), required=True)
    value.add_argument("--selected-epoch", type=int, required=True)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--task-units", type=Path, required=True)
    value.add_argument("--training-latent", type=Path, required=True)
    value.add_argument("--test-latent", type=Path, required=True)
    value.add_argument("--scale", type=Path, required=True)
    value.add_argument("--visit-manifest", type=Path, required=True)
    value.add_argument(
        "--history-policy", choices=HISTORY_POLICIES, default="formal"
    )
    value.add_argument("--longitudinal-visit-manifest", type=Path)
    value.add_argument("--projected-set-checkpoint", type=Path, required=True)
    value.add_argument("--architecture-screen", type=Path, required=True)
    value.add_argument("--epoch-selection", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value

def _load_projected_set(path: Path, task: str, fold: int) -> MomentFiberCurrentModel:
    payload = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if (
        payload.get("phase") != "paper_model"
        or payload.get("paper_task") != task
        or int(payload.get("outer_fold", -1)) != fold
    ):
        raise ValueError("ProjectedSet paper checkpoint identity is invalid")
    observed = payload.get("model_config")
    if not isinstance(observed, Mapping) or observed.get("current_operator") != "moment_fiber":
        raise ValueError("MetaAdapter requires the formal ProjectedSet moment-fiber model")
    permitted = {field.name for field in fields(VaccineImmuneTwinConfig)}
    config = VaccineImmuneTwinConfig(
        **{key: item for key, item in observed.items() if key in permitted}
    )
    model = MomentFiberCurrentModel(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model

def _checkpoint(
    *,
    model: MetaAdapterDigitalTwin,
    task: str,
    fold: int,
    epoch: int,
    history: Sequence[Mapping[str, float]],
    parent: Path,
    architecture_screen: Path,
    epoch_selection: Path,
    history_policy: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA,
        "phase": "paper_model",
        "paper_method": PAPER_METHOD,
        "paper_task": task,
        "outer_fold": fold,
        "epoch": epoch,
        "seed": 42,
        "implementation": IMPLEMENTATION,
        "history_policy": str(history_policy),
        "model_config": model.config.to_dict(),
        "twin_config": model.twin_config.to_dict(),
        "identifier_state_dict": {
            name: value.detach().cpu()
            for name, value in model.identifier.state_dict().items()
        },
        "history": list(history),
        "parent_projected_set": {
            "path": str(parent.resolve(strict=True)),
        },
        "architecture_screen": {
            "path": str(architecture_screen.resolve(strict=True)),
        },
        "epoch_selection": {
            "path": str(epoch_selection.resolve(strict=True)),
        },
    }

def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    started = time.time()
    task = canonical_task(args.task)
    if task == "t1_y1_d0_to_y1_d7":
        raise ValueError("T1 has empty history and exactly aliases ProjectedSet")
    if args.selected_epoch < 1:
        raise ValueError("selected epoch must be positive")

    architecture = expand_paths(json.loads(
        args.architecture_screen.resolve(strict=True).read_text(encoding="utf-8")
    ))
    if (
        architecture.get("implementation") != IMPLEMENTATION
        or architecture.get("history_input", {}).get("variant") != "rna"
        or architecture.get("history_input", {}).get("token_mode") not in {"transition", "visit"}
    ):
        raise ValueError("formal MetaAdapter requires a frozen RNA history architecture")
    architecture_history_policy = architecture.get("history_input", {}).get(
        "support_policy", "formal"
    )
    if architecture_history_policy != args.history_policy:
        raise ValueError("runtime RNA history policy differs from architecture freeze")
    optimization = architecture.get("optimization", {})
    twin_payload = architecture.get("twin_config", {})
    loss_payload = architecture.get("loss", {})
    if not isinstance(optimization, Mapping) or not isinstance(twin_payload, Mapping):
        raise ValueError("architecture freeze lacks optimization or twin config")
    if not isinstance(loss_payload, Mapping):
        raise ValueError("architecture freeze lacks loss config")

    _, runtime_base, _, loss_base = load_configuration_v2(
        args.config, arm="current", task=task
    )
    runtime = replace(
        runtime_base,
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization.get("weight_decay", runtime_base.weight_decay)),
        batch_size=int(optimization["batch_size"]),
        maximum_epochs=int(args.selected_epoch),
        patience=int(args.selected_epoch),
        history_training_mode="residual_from_current",
        freeze_pretrained_encoder=True,
        selection_objective="raw_donor_delta_rmse",
        selection_direction="lower",
    )
    loss = PatientLossV2Config(
        distribution_weight=float(loss_payload["distribution_weight"]),
        response_mean_weight=float(loss_payload["response_mean_weight"]),
        response_direction_weight=float(loss_payload["response_direction_weight"]),
        response_norm_weight=float(loss_payload["response_norm_weight"]),
        centered_geometry_weight=float(loss_payload["centered_geometry_weight"]),
        geometry_gram_fraction=loss_base.geometry_gram_fraction,
        history_sensitivity_weight=float(loss_payload["history_sensitivity_weight"]),
        history_sensitivity_temperature=float(
            loss_payload["history_sensitivity_temperature"]
        ),
        history_current_anchor_weight=float(
            loss_payload["history_current_anchor_weight"]
        ),
        history_current_anchor_temperature=float(
            loss_payload["history_current_anchor_temperature"]
        ),
        geometry_scale_floor=loss_base.geometry_scale_floor,
        epsilon=loss_base.epsilon,
    )
    twin_fields = set(MetaAdapterConfig.__dataclass_fields__)
    twin_config = MetaAdapterConfig(
        **{key: value for key, value in twin_payload.items() if key in twin_fields}
    )
    if twin_config.use_gene_attention or twin_config.use_clinical_history:
        raise ValueError("this formal runner is restricted to the selected RNA-only input")
    if twin_config.use_visit_tokens != (architecture['history_input']['token_mode'] == 'visit'):
        raise ValueError('RNA visit-token flag differs from the frozen input specification')

    set_seed(42)
    device = require_cuda()
    torch.set_num_threads(4)
    units = build_task_units(args.task_units, task)
    if args.history_policy != "formal" and args.longitudinal_visit_manifest is None:
        raise ValueError("expanded RNA history requires --longitudinal-visit-manifest")
    history_audit = None
    if args.history_policy != "formal":
        units, history_audit = augment_with_complete_prior_rna(
            units,
            args.longitudinal_visit_manifest,
            policy=args.history_policy,
        )
    outer_train, outer_test = split_outer_units(units, args.fold)
    scale = load_fold_scale(args.scale, outer_fold=args.fold)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    uce = UnifiedUCEStore(args.visit_manifest)
    training_latent = FoldLatentStore(args.training_latent, scale)
    test_latent: FoldLatentStore | None = None
    try:
        backbone = _load_projected_set(args.projected_set_checkpoint, task, args.fold)
        model = MetaAdapterDigitalTwin(backbone, torch.empty(0, 32), twin_config).to(device)
        optimizer = torch.optim.AdamW(
            model.identifier.parameters(),
            lr=runtime.learning_rate,
            weight_decay=runtime.weight_decay,
        )
        response_scale = backbone.transport.response_scale.detach().cpu().numpy()
        train_materializer = UnitMaterializer(
            uce=uce,
            latent=training_latent,
            query_uce_cells=runtime.query_uce_cells,
            support_uce_cells=runtime.support_uce_cells,
            seed=42,
        )
        history: list[dict[str, float]] = []
        for epoch in range(1, args.selected_epoch + 1):
            metrics = train_epoch_v2(
                model,
                optimizer,
                train_materializer,
                outer_train,
                arm="full",
                epoch=epoch,
                runtime=runtime,
                loss_config=loss,
                bandwidth=scale.mmd_bandwidth,
                response_scale=response_scale,
                clinical_mode="off",
                device=device,
            )
            history.append({"epoch": float(epoch), **metrics})
            atomic_table(output / "final_history.tsv", pd.DataFrame(history))
            atomic_json(
                output / "progress.json",
                {
                    "status": "training",
                    "epoch": epoch,
                    "selected_epoch": args.selected_epoch,
                },
            )

        atomic_torch_save(
            output / "checkpoint.pt",
            _checkpoint(
                model=model,
                task=task,
                fold=args.fold,
                epoch=args.selected_epoch,
                history=history,
                parent=args.projected_set_checkpoint,
                architecture_screen=args.architecture_screen,
                epoch_selection=args.epoch_selection,
                history_policy=args.history_policy,
            ),
        )
        training_latent.close()
        test_latent = FoldLatentStore(args.test_latent, scale)
        test_materializer = UnitMaterializer(
            uce=uce,
            latent=test_latent,
            query_uce_cells=runtime.query_uce_cells,
            support_uce_cells=runtime.support_uce_cells,
            seed=42,
        )
        manifest, artifacts = predict_heldout_v2(
            model=model,
            arm="full",
            units=outer_test,
            materializer=test_materializer,
            runtime=runtime,
            clinical_mode="off",
            gene_selection=None,
            output=output,
            device=device,
            paper_method=PAPER_METHOD,
        )
        summary = {
            "status": "pass",
            "schema_version": SCHEMA,
            "paper_method": PAPER_METHOD,
            "paper_task": task,
            "outer_fold": args.fold,
            "seed": 42,
            "selected_epoch": args.selected_epoch,
            "architecture_and_loss_fixed_before_outer_test_predictions": True,
            "epoch_selected_with_outer_training_donors_only": True,
            "training_donors": len({unit.donor_id for unit in outer_train}),
            "training_units": len(outer_train),
            "test_donors": len({unit.donor_id for unit in outer_test}),
            "test_units": len(outer_test),
            "prediction_units": len(manifest),
            "heldout_target_used_for_training_or_adaptation": False,
            "history_information_used": True,
            "history_policy": str(args.history_policy),
            "history_policy_audit": (
                None if history_audit is None else history_audit.to_dict()
            ),
            "implementation": IMPLEMENTATION,
            "runtime_config": asdict(runtime),
            "loss_config": loss.to_dict(),
            "twin_config": twin_config.to_dict(),
            "parent_projected_set": {
                "path": str(args.projected_set_checkpoint.resolve(strict=True)),
            },
            "architecture_screen": {
                "path": str(args.architecture_screen.resolve(strict=True)),
            },
            "epoch_selection": {
                "path": str(args.epoch_selection.resolve(strict=True)),
            },
            "artifacts": {
                name: {"path": str(path.resolve(strict=True))}
                for name, path in {"checkpoint": output / "checkpoint.pt", **artifacts}.items()
            },
            "elapsed_seconds": time.time() - started,
            **runtime_resource_snapshot(device),
            "torch": torch.__version__,
        }
        atomic_json(output / "run.json", summary)
        atomic_json(output / "progress.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as error:
        atomic_json(
            output / "progress.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        try:
            training_latent.close()
        except Exception:
            pass
        if test_latent is not None:
            test_latent.close()
        uce.close()

if __name__ == "__main__":
    main()

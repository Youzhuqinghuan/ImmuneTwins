from __future__ import annotations
import argparse
from dataclasses import asdict, fields, replace
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence
import pandas as pd
import torch
from model.digital_twin import MetaAdapterConfig, MetaAdapterDigitalTwin
from model.encoder import VaccineImmuneTwinConfig
from model.projected_set import MomentFiberCurrentModel
from data.clinical import OuterTrainingVisitPreprocessor, build_sound_visit_features
from data.sound import FoldLatentStore, GeneSummaryStore, UnifiedUCEStore, UnitMaterializer, build_task_units, canonical_task, load_fold_scale, save_gene_selection, split_inner_units, split_outer_units
from data.twin_inputs import CLINICAL_HISTORY_SUBSETS, clinical_history_table, select_support_transition_genes
from model.losses import PatientLossV2Config
from data.history import HISTORY_POLICIES, augment_with_complete_prior_rna
from pipeline.sound.provenance import atomic_json
from data.gene_coordinates import frozen_gene_principal_coordinates
from pipeline.sound.common import atomic_table, atomic_torch_save, require_cuda, set_seed
from pipeline.sound.train import evaluate_donors_v2, load_configuration_v2, train_epoch_v2
from pipeline.sound.train import estimate_phase_support_response_reference, save_phase_support_response_reference, set_model_support_response_reference

SCHEMA = "sound_meta_adapter_inner_screen_1"

PAPER_METHOD = "ImmuneTwins-MetaAdapter"

VARIANTS = {
    "rna": {"gene": False, "clinical": False},
    "rna_gene": {"gene": True, "clinical": False},
    "rna_clinical": {"gene": False, "clinical": True},
    "full": {"gene": True, "clinical": True},
}

def _parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--task", required=True)
    value.add_argument("--fold", type=int, choices=range(5), default=0)
    value.add_argument(
        "--selection-scope",
        choices=("architecture_screen", "nested_epoch"),
        default="architecture_screen",
        help=(
            "architecture_screen is restricted to outer fold 0; nested_epoch "
            "may be used after the architecture is frozen to choose refit duration "
            "inside any outer-training partition"
        ),
    )
    value.add_argument("--inner-validation-fold", type=int, choices=range(4), required=True)
    value.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    value.add_argument(
        "--clinical-subset",
        choices=sorted(CLINICAL_HISTORY_SUBSETS),
        default="curated",
        help="Prespecified clinical input group used by clinical variants",
    )
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--task-units", type=Path, required=True)
    value.add_argument("--inner-splits", type=Path, required=True)
    value.add_argument("--training-latent", type=Path, required=True)
    value.add_argument("--scale", type=Path, required=True)
    value.add_argument("--visit-manifest", type=Path, required=True)
    value.add_argument(
        "--history-policy", choices=HISTORY_POLICIES, default="formal"
    )
    value.add_argument("--longitudinal-visit-manifest", type=Path)
    value.add_argument("--projected-set-checkpoint", type=Path, required=True)
    value.add_argument("--gene-embedding", type=Path, required=True)
    value.add_argument("--labs", type=Path, required=True)
    value.add_argument("--hai", type=Path, required=True)
    value.add_argument("--igg", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--maximum-epochs", type=int, default=80)
    value.add_argument("--patience", type=int, default=12)
    value.add_argument("--learning-rate", type=float, default=1.0e-3)
    value.add_argument("--weight-decay", type=float, default=None)
    value.add_argument("--batch-size", type=int, default=8)
    value.add_argument("--gene-count", type=int, default=128)
    value.add_argument("--context-dim", type=int, default=128)
    value.add_argument("--visit-slots", type=int, default=4)
    value.add_argument(
        "--history-token-mode", choices=("visit", "transition"), default="visit"
    )
    value.add_argument("--context-layers", type=int, default=2)
    value.add_argument("--adapter-rank", type=int, default=8)
    value.add_argument("--adapter-scale", type=float, default=0.5)
    value.add_argument("--gene-dropout", type=float, default=0.1)
    value.add_argument("--history-gate", action="store_true")
    value.add_argument("--recovery-basis", action="store_true")
    value.add_argument("--response-weight", type=float, default=2.0)
    value.add_argument("--direction-weight", type=float, default=0.5)
    value.add_argument("--norm-weight", type=float, default=0.5)
    value.add_argument("--geometry-weight", type=float, default=0.75)
    value.add_argument("--history-weight", type=float, default=0.5)
    value.add_argument("--anchor-weight", type=float, default=0.5)
    value.add_argument("--history-temperature", type=float, default=1.0)
    value.add_argument("--anchor-temperature", type=float, default=1.0)
    return value

def _load_projected_set(path: Path, task: str, fold: int) -> MomentFiberCurrentModel:
    payload = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    observed = payload.get("model_config")
    if not isinstance(observed, Mapping) or observed.get("current_operator") != "moment_fiber":
        raise ValueError("MetaAdapter requires a ProjectedSet moment-fiber checkpoint")
    identity = payload.get("base_checkpoint", payload)
    paper_task = identity.get("paper_task", payload.get("paper_task"))
    outer_fold = identity.get("outer_fold", payload.get("outer_fold", -1))
    if paper_task != task or int(outer_fold) != int(fold):
        raise ValueError("ProjectedSet checkpoint differs from the requested task/fold")
    permitted = {field.name for field in fields(VaccineImmuneTwinConfig)}
    config = VaccineImmuneTwinConfig(
        **{key: value for key, value in observed.items() if key in permitted}
    )
    model = MomentFiberCurrentModel(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model

def _checkpoint(
    *,
    model: MetaAdapterDigitalTwin,
    epoch: int,
    validation: Mapping[str, float],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA,
        "paper_method": PAPER_METHOD,
        "implementation": "history_generated_shared_basis_lora_1",
        "task": canonical_task(args.task),
        "outer_fold": int(args.fold),
        "inner_validation_fold": int(args.inner_validation_fold),
        "variant": str(args.variant),
        "clinical_subset": str(args.clinical_subset),
        "history_policy": str(args.history_policy),
        "epoch": int(epoch),
        "seed": 42,
        "parent_projected_set": {
            "path": str(args.projected_set_checkpoint.resolve(strict=True)),
        },
        "twin_config": model.twin_config.to_dict(),
        "identifier_state_dict": {
            name: value.detach().cpu()
            for name, value in model.identifier.state_dict().items()
        },
        "validation": dict(validation),
    }

def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    task = canonical_task(args.task)
    if task == "t1_y1_d0_to_y1_d7":
        raise ValueError("T1 has empty history and exactly aliases ProjectedSet")
    if args.selection_scope == "architecture_screen" and args.fold != 0:
        raise ValueError("architecture screening is frozen to outer fold 0")
    if args.maximum_epochs < 1 or args.patience < 1 or args.batch_size < 2:
        raise ValueError("epochs, patience and batch size are invalid")
    variant = VARIANTS[args.variant]

    _, runtime_base, _, loss_base = load_configuration_v2(
        args.config, arm="current", task=task
    )
    runtime = replace(
        runtime_base,
        learning_rate=float(args.learning_rate),
        weight_decay=(
            runtime_base.weight_decay
            if args.weight_decay is None
            else float(args.weight_decay)
        ),
        batch_size=int(args.batch_size),
        maximum_epochs=int(args.maximum_epochs),
        patience=int(args.patience),
        gene_candidates_per_fold=int(args.gene_count),
        history_training_mode="residual_from_current",
        freeze_pretrained_encoder=True,
        selection_objective="raw_donor_delta_rmse",
        selection_direction="lower",
    )
    loss = PatientLossV2Config(
        distribution_weight=1.0,
        response_mean_weight=float(args.response_weight),
        response_direction_weight=float(args.direction_weight),
        response_norm_weight=float(args.norm_weight),
        centered_geometry_weight=float(args.geometry_weight),
        geometry_gram_fraction=loss_base.geometry_gram_fraction,
        history_sensitivity_weight=float(args.history_weight),
        history_sensitivity_temperature=float(args.history_temperature),
        history_current_anchor_weight=float(args.anchor_weight),
        history_current_anchor_temperature=float(args.anchor_temperature),
        geometry_scale_floor=loss_base.geometry_scale_floor,
        epsilon=loss_base.epsilon,
    )

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
    outer_train, outer_test_metadata = split_outer_units(units, args.fold)
    fit_units, validation_units = split_inner_units(
        outer_train,
        args.inner_splits,
        outer_fold=args.fold,
        validation_fold=args.inner_validation_fold,
    )
    fit_donors = {unit.donor_id for unit in fit_units}
    scale = load_fold_scale(args.scale, outer_fold=args.fold)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()

    clinical = None
    clinical_preprocessor = None
    if bool(variant["clinical"]):
        clinical_raw = build_sound_visit_features(args.labs, args.hai, args.igg)
        clinical_preprocessor = OuterTrainingVisitPreprocessor(min_train_observations=1)
        clinical = clinical_preprocessor.fit_transform(
            clinical_history_table(clinical_raw, args.clinical_subset),
            outer_training_donors=fit_donors,
        )

    uce = UnifiedUCEStore(args.visit_manifest)
    latent = FoldLatentStore(args.training_latent, scale)
    genes = GeneSummaryStore(args.visit_manifest) if bool(variant["gene"]) else None
    try:
        gene_selection = None
        gene_coordinates = torch.empty(0, 32)
        if genes is not None:
            gene_selection, gene_audit = select_support_transition_genes(
                genes,
                units=fit_units,
                embedding_path=args.gene_embedding,
                n_genes=runtime.gene_candidates_per_fold,
            )
            gene_coordinate_dim = min(32, len(gene_selection.names) - 1)
            coordinates, coordinate_audit = frozen_gene_principal_coordinates(
                gene_selection.embeddings.numpy(), output_dim=gene_coordinate_dim
            )
            gene_coordinates = torch.as_tensor(coordinates)
            save_gene_selection(gene_selection, output / "selected_response_genes.tsv")
            atomic_json(output / "gene_selection_audit.json", asdict(gene_audit))
            atomic_json(output / "gene_coordinate_audit.json", coordinate_audit)

        materializer = UnitMaterializer(
            uce=uce,
            latent=latent,
            query_uce_cells=runtime.query_uce_cells,
            support_uce_cells=runtime.support_uce_cells,
            gene_store=genes,
            gene_selection=gene_selection,
            clinical_features=clinical,
            seed=42,
        )
        backbone = _load_projected_set(args.projected_set_checkpoint, task, args.fold)
        twin_config = MetaAdapterConfig(
            context_dim=int(args.context_dim),
            visit_slots=int(args.visit_slots),
            context_layers=int(args.context_layers),
            context_heads=4,
            adapter_rank=int(args.adapter_rank),
            adapter_scale=float(args.adapter_scale),
            gene_embedding_dim=int(gene_coordinates.shape[1]),
            clinical_value_dim=materializer.clinical_value_dim,
            use_gene_attention=bool(variant["gene"]),
            use_clinical_history=bool(variant["clinical"]),
            use_visit_tokens=args.history_token_mode == "visit",
            use_history_gate=bool(args.history_gate),
            use_recovery_basis=bool(args.recovery_basis),
            gene_dropout=float(args.gene_dropout),
        )
        model = MetaAdapterDigitalTwin(backbone, gene_coordinates, twin_config).to(device)
        support_reference_audit: dict[str, object] | None = None
        if args.recovery_basis:
            support_reference = estimate_phase_support_response_reference(
                latent,
                fit_units,
                cells=runtime.training_latent_cells,
                seed=42,
                stage="selection",
                phase_count=backbone.config.phase_count,
            )
            set_model_support_response_reference(model, support_reference)
            support_reference_audit = save_phase_support_response_reference(
                support_reference, output / "support_response_reference.json"
            )
        optimizer = torch.optim.AdamW(
            model.identifier.parameters(),
            lr=runtime.learning_rate,
            weight_decay=runtime.weight_decay,
        )
        response_scale = backbone.transport.response_scale.detach().cpu().numpy()
        baseline = evaluate_donors_v2(
            backbone,
            materializer,
            validation_units,
            arm="current",
            runtime=runtime,
            loss_config=replace(
                loss,
                history_sensitivity_weight=0.0,
                history_current_anchor_weight=0.0,
            ),
            bandwidth=scale.mmd_bandwidth,
            response_scale=response_scale,
            clinical_mode="off",
            device=device,
        )

        history: list[dict[str, float]] = []
        best_score = math.inf
        best_epoch = 0
        stale = 0
        for epoch in range(1, runtime.maximum_epochs + 1):
            training = train_epoch_v2(
                model,
                optimizer,
                materializer,
                fit_units,
                arm="full",
                epoch=epoch,
                runtime=runtime,
                loss_config=loss,
                bandwidth=scale.mmd_bandwidth,
                response_scale=response_scale,
                clinical_mode="delta" if bool(variant["clinical"]) else "off",
                device=device,
            )
            validation = evaluate_donors_v2(
                model,
                materializer,
                validation_units,
                arm="full",
                runtime=runtime,
                loss_config=loss,
                bandwidth=scale.mmd_bandwidth,
                response_scale=response_scale,
                clinical_mode="delta" if bool(variant["clinical"]) else "off",
                device=device,
            )
            row = {
                "epoch": float(epoch),
                **{f"train_{key}": float(value) for key, value in training.items()},
                **{f"validation_{key}": float(value) for key, value in validation.items()},
            }
            history.append(row)
            score = float(validation["raw_donor_delta_rmse"])
            if math.isfinite(score) and score < best_score - 1.0e-8:
                best_score = score
                best_epoch = epoch
                stale = 0
                atomic_torch_save(
                    output / "best.pt",
                    _checkpoint(model=model, epoch=epoch, validation=validation, args=args),
                )
            else:
                stale += 1
            atomic_table(output / "history.tsv", pd.DataFrame(history))
            atomic_json(
                output / "progress.json",
                {
                    "status": "running",
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_raw_donor_delta_rmse": best_score,
                    "stale_epochs": stale,
                },
            )
            if stale >= runtime.patience:
                break

        best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        summary = {
            "status": "pass",
            "schema_version": SCHEMA,
            "paper_method": PAPER_METHOD,
            "implementation": best["implementation"],
            "task": task,
            "outer_fold": int(args.fold),
            "inner_validation_fold": int(args.inner_validation_fold),
            "selection_scope": str(args.selection_scope),
            "variant": args.variant,
            "clinical_subset": (
                str(args.clinical_subset) if bool(variant["clinical"]) else None
            ),
            "history_policy": str(args.history_policy),
            "history_policy_audit": (
                None if history_audit is None else history_audit.to_dict()
            ),
            "seed": 42,
            "fit_donors": len(fit_donors),
            "validation_donors": len({unit.donor_id for unit in validation_units}),
            "outer_test_metadata_units": len(outer_test_metadata),
            "outer_test_target_opened": False,
            "baseline_validation": baseline,
            "best_epoch": int(best["epoch"]),
            "best_validation": best["validation"],
            "parent_projected_set": best["parent_projected_set"],
            "twin_config": twin_config.to_dict(),
            "runtime": asdict(runtime),
            "loss": loss.to_dict(),
            "elapsed_seconds": time.time() - started,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "checkpoint_path": str((output / "best.pt").resolve()),
            "selected_response_gene_count": 0 if gene_selection is None else len(gene_selection.names),
            "clinical_feature_names": (
                []
                if clinical_preprocessor is None
                else list(clinical_preprocessor.feature_names_ or ())
            ),
            "support_response_reference": support_reference_audit,
        }
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "progress.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as error:
        atomic_json(
            output / "progress.json",
            {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
        )
        raise
    finally:
        if genes is not None:
            genes.close()
        latent.close()
        uce.close()

if __name__ == "__main__":
    main()

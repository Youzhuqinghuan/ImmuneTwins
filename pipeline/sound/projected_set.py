from __future__ import annotations
from paths import expand_paths
import argparse
from dataclasses import asdict, fields
import fcntl
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import pandas as pd
import torch
from model.encoder import VaccineImmuneTwinConfig
from model.projected_set import MomentFiberCurrentModel
from data.sound import FoldLatentStore, SealedBaseUCEStore, UnitMaterializer, build_task_units, canonical_task, load_fold_scale, split_outer_units
from pipeline.sound.provenance import V2_COMPLETE_SCHEMA, V2_TRAIN_SCHEMA, atomic_json, load_v2_protocol, validate_completed_v2_run
from pipeline.sound.common import atomic_table, atomic_torch_save, require_cuda, set_seed
from pipeline.sound.train import MECHANISM_VARIANTS, build_formal_run_payload, load_configuration_v2, predict_heldout_v2, runtime_resource_snapshot, train_epoch_v2

SCHEMA = "sound_vaccine_projected_set_formal_1"

PILOT_SCHEMA = "sound_vaccine_moment_fiber_pilot_1"

PAPER_METHOD = "ImmuneTwin-ProjectedSet"

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-units", type=Path, required=True)
    parser.add_argument("--training-latent", type=Path, required=True)
    parser.add_argument("--test-latent", type=Path, required=True)
    parser.add_argument("--scale", type=Path, required=True)
    parser.add_argument("--sealed-source-manifest", type=Path, required=True)
    parser.add_argument("--sealed-uce", type=Path, required=True)
    parser.add_argument("--selection-pilot", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser

def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved)}

def _protocol_contains(protocol: Any, path: Path, *, label: str) -> None:
    resolved = path.resolve(strict=True)
    matches = []
    for section in ("configuration", "inputs"):
        records = protocol.payload.get(section, {}).get("files", [])
        for record in records:
            if Path(str(record.get("path", ""))).resolve() == resolved:
                matches.append(record)
    if not matches:
        raise RuntimeError(f"{label} is absent from the frozen ProjectedSet protocol")

def _parent_artifact(run: Mapping[str, Any], name: str) -> Path:
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get(name), Mapping):
        raise RuntimeError(f"SetTransport parent lacks artifact {name!r}")
    record = artifacts[name]
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    return path

def _load_parent_model(
    checkpoint_path: Path,
    requested: VaccineImmuneTwinConfig,
    *,
    required_phase: str,
    task: str,
    fold: int,
) -> tuple[MomentFiberCurrentModel, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path.resolve(strict=True), map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("format") != V2_TRAIN_SCHEMA
        or checkpoint.get("arm") != "current"
        or checkpoint.get("phase") != required_phase
        or checkpoint.get("paper_task") != task
        or int(checkpoint.get("outer_fold", -1)) != fold
    ):
        raise ValueError("SetTransport checkpoint identity or phase is invalid")
    observed = checkpoint.get("model_config")
    if not isinstance(observed, dict) or observed.get("current_operator") != "set_affine":
        raise ValueError("ProjectedSet parent must be ImmuneTwin-SetTransport")
    permitted = {field.name for field in fields(VaccineImmuneTwinConfig)}
    parent_config = VaccineImmuneTwinConfig(
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
        if getattr(parent_config, name) != getattr(requested, name):
            raise ValueError(f"ProjectedSet and SetTransport differ at {name}")
    model = MomentFiberCurrentModel(requested)
    loaded = model.load_frozen_set_transport_state(checkpoint["model_state_dict"])
    dependency = {
        **_record(checkpoint_path),
        "phase": required_phase,
        "paper_task": task,
        "outer_fold": fold,
        "loaded_backbone_tensors": len(loaded),
    }
    return model, dependency

def _validate_selection_pilot(
    pilot_root: Path,
    *,
    config_path: Path,
    parent_selection: Path,
    task: str,
    fold: int,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    summary_path = pilot_root.resolve(strict=True) / "summary.json"
    summary = expand_paths(json.loads(summary_path.read_text(encoding="utf-8")))
    expected = {
        "status": "pass",
        "schema_version": PILOT_SCHEMA,
        "task": task,
        "outer_fold": fold,
        "seed": 42,
        "model_initialized_after_seed": True,
        "test_set_opened": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"selection pilot field {key!r} is invalid")
    base = summary.get("base_checkpoint")
    if not isinstance(base, Mapping):
        raise RuntimeError("selection pilot lacks its SetTransport dependency")
    if (
        Path(str(base.get("path", ""))).resolve(strict=True)
        != parent_selection.resolve(strict=True)
    ):
        raise RuntimeError("selection pilot did not use the matching parent selection model")
    best_path = Path(str(summary.get("checkpoint_path", ""))).resolve(strict=True)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if (
        best.get("schema_version") != PILOT_SCHEMA
        or int(best.get("epoch", -1)) != int(summary.get("best_epoch", -2))
    ):
        raise RuntimeError("selection pilot summary and checkpoint disagree")
    expected_config = load_configuration_v2(
        config_path, arm="current", task=task
    )[2].to_dict()
    if best.get("model_config") != expected_config:
        raise RuntimeError("selection pilot checkpoint used a different model config")
    history_path = Path(str(summary.get("history_path", ""))).resolve(strict=True)
    if history_path != (pilot_root.resolve(strict=True) / "history.tsv"):
        raise RuntimeError("selection pilot history path is invalid")
    return summary, best, summary_path, history_path

def _formal_checkpoint(
    *,
    phase: str,
    model: MomentFiberCurrentModel,
    task: str,
    fold: int,
    epoch: int,
    history: Sequence[Mapping[str, float]],
    parent: Mapping[str, object],
    job: Any,
    selection_pilot: Mapping[str, object],
) -> dict[str, object]:
    return {
        "format": V2_TRAIN_SCHEMA,
        "projected_set_schema": SCHEMA,
        "phase": phase,
        "arm": "current",
        "paper_method": job.paper_method,
        "paper_task": task,
        "outer_fold": fold,
        "epoch": epoch,
        "selected_epoch": int(selection_pilot["best_epoch"]),
        "model_config": model.config.to_dict(),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": None,
        "history": list(history),
        "parent_set_transport": dict(parent),
        "selection_pilot": {
            key: selection_pilot[key]
            for key in (
                "best_epoch", "best_validation", "baseline_validation",
                "checkpoint_path", "history_path",
            )
        },
    }

def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    started = time.time()
    task = canonical_task(args.task)
    payload, runtime, model_config, loss_config = load_configuration_v2(
        args.config, arm="current", task=task
    )
    if (
        model_config.current_operator != "moment_fiber"
        or model_config.fiber_context != "adaptive"
        or model_config.fiber_retraction != "adaptive_final"
    ):
        raise ValueError("formal runner requires the adaptive ProjectedSet config")
    if runtime.seed != 42 or args.seed != 42:
        raise ValueError("ProjectedSet formal evaluation is frozen to seed 42")

    protocol = load_v2_protocol(args.protocol_manifest, verify_files=True)
    if protocol.scope != "current_only":
        raise RuntimeError("ProjectedSet protocol identity or scope is invalid")
    identifier = f"current__{task}__outer{args.fold}_seed42"
    job = protocol.jobs.get(identifier)
    if job is None or job.paper_method != PAPER_METHOD:
        raise RuntimeError("ProjectedSet job or paper-method identity is invalid")
    expected_output = (
        protocol.formal_output_root
        / "current"
        / task
        / f"outer{args.fold}_seed42"
    ).resolve()
    if args.output.resolve() != expected_output:
        raise RuntimeError("ProjectedSet output differs from the frozen job directory")

    parent_protocol = load_v2_protocol(args.parent_protocol, verify_files=False)
    parent_identifier = f"current__{task}__outer{args.fold}_seed42"
    parent_output = (
        parent_protocol.formal_output_root
        / "current"
        / task
        / f"outer{args.fold}_seed42"
    )
    parent_run = validate_completed_v2_run(
        parent_output,
        protocol=parent_protocol,
        job_identifier=parent_identifier,
    )
    parent_selection = _parent_artifact(parent_run.run, "best_validation_checkpoint")
    parent_final = _parent_artifact(parent_run.run, "checkpoint")
    pilot_summary, pilot_best, pilot_summary_path, pilot_history_path = (
        _validate_selection_pilot(
            args.selection_pilot,
            config_path=args.config,
            parent_selection=parent_selection,
            task=task,
            fold=args.fold,
        )
    )
    pilot_best_path = Path(str(pilot_summary["checkpoint_path"])).resolve(strict=True)

    for label, path in (
        ("config", args.config),
        ("task units", args.task_units),
        ("training latent", args.training_latent),
        ("test latent", args.test_latent),
        ("scale", args.scale),
        ("sealed source manifest", args.sealed_source_manifest),
        ("sealed UCE", args.sealed_uce),
        ("parent protocol", args.parent_protocol),
        ("parent selection checkpoint", parent_selection),
        ("parent final checkpoint", parent_final),
        ("selection pilot summary", pilot_summary_path),
        ("selection pilot checkpoint", pilot_best_path),
        ("selection pilot history", pilot_history_path),
    ):
        _protocol_contains(protocol, path, label=label)

    units = build_task_units(args.task_units, task)
    outer_train, outer_test = split_outer_units(units, args.fold)
    test_donors = len({unit.donor_id for unit in outer_test})
    if len(outer_test) != job.expected_units or test_donors != job.expected_donors:
        raise RuntimeError("held-out coverage differs from the ProjectedSet protocol")

    device = require_cuda()
    torch.set_num_threads(4)
    set_seed(runtime.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output / ".run.lock").open("w")
    fcntl.flock(lock_handle, fcntl.LOCK_EX)
    complete_path = args.output / "_COMPLETE.json"
    if complete_path.is_file():
        if not args.resume:
            raise FileExistsError("completed ProjectedSet output already exists")
        validated = validate_completed_v2_run(
            args.output, protocol=protocol, job_identifier=identifier
        )
        print(json.dumps({"status": "resume", "run": str(validated.run_path)}, indent=2))
        return
    partial = [path for path in args.output.iterdir() if path.name != ".run.lock"]
    if partial:
        raise RuntimeError(
            "partial ProjectedSet output requires explicit audit before restart: "
            + ", ".join(sorted(path.name for path in partial))
        )

    selection_model, selection_parent = _load_parent_model(
        parent_selection,
        model_config,
        required_phase="best_validation",
        task=task,
        fold=args.fold,
    )
    selection_model.load_state_dict(pilot_best["model_state_dict"], strict=True)
    selection_dependency = {
        "summary": _record(pilot_summary_path),
        "checkpoint": _record(pilot_best_path),
        "best_epoch": int(pilot_summary["best_epoch"]),
        "best_validation": dict(pilot_summary["best_validation"]),
        "baseline_validation": dict(pilot_summary["baseline_validation"]),
        "parent": selection_parent,
    }
    atomic_torch_save(
        args.output / "best_validation.pt",
        _formal_checkpoint(
            phase="best_validation",
            model=selection_model,
            task=task,
            fold=args.fold,
            epoch=int(pilot_summary["best_epoch"]),
            history=[],
            parent=selection_parent,
            job=job,
            selection_pilot=pilot_summary,
        ),
    )
    selection_history = pd.read_csv(pilot_history_path, sep="\t")
    atomic_table(args.output / "selection_history.tsv", selection_history)
    atomic_json(
        args.output / "selection.json",
        {
            "status": "pass",
            "schema_version": SCHEMA,
            "paper_method": job.paper_method,
            "paper_task": task,
            "outer_fold": args.fold,
            "seed": 42,
            "objective": "distribution_mmd2",
            "best_epoch": int(pilot_summary["best_epoch"]),
            "selection_dependency": selection_dependency,
            "outer_test_opened": False,
        },
    )

    set_seed(runtime.seed)
    model, final_parent = _load_parent_model(
        parent_final,
        model_config,
        required_phase="paper_model",
        task=task,
        fold=args.fold,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=runtime.learning_rate,
        weight_decay=runtime.weight_decay,
    )
    response_scale = model.transport.response_scale.detach().cpu().numpy()
    scale = load_fold_scale(args.scale, outer_fold=args.fold)
    uce = SealedBaseUCEStore(args.sealed_source_manifest, args.sealed_uce)
    training_latent = FoldLatentStore(args.training_latent, scale)
    test_latent: FoldLatentStore | None = None
    final_history: list[dict[str, float]] = []
    try:
        training_materializer = UnitMaterializer(
            uce=uce,
            latent=training_latent,
            query_uce_cells=runtime.query_uce_cells,
            support_uce_cells=runtime.support_uce_cells,
            seed=runtime.seed,
        )
        selected_epoch = int(pilot_summary["best_epoch"])
        for epoch in range(1, selected_epoch + 1):
            metrics = train_epoch_v2(
                model,
                optimizer,
                training_materializer,
                outer_train,
                arm="current",
                epoch=epoch,
                runtime=runtime,
                loss_config=loss_config,
                bandwidth=scale.mmd_bandwidth,
                response_scale=response_scale,
                clinical_mode="off",
                device=device,
            )
            final_history.append({"epoch": float(epoch), **metrics})
            atomic_torch_save(
                args.output / "final_last.pt",
                {
                    **_formal_checkpoint(
                        phase="final_fit",
                        model=model,
                        task=task,
                        fold=args.fold,
                        epoch=epoch,
                        history=final_history,
                        parent=final_parent,
                        job=job,
                        selection_pilot=pilot_summary,
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
            )
        atomic_table(
            args.output / "final_history.tsv",
            (
                pd.DataFrame(final_history)
                if final_history
                else pd.DataFrame(columns=["epoch"])
            ),
        )
        atomic_torch_save(
            args.output / "checkpoint.pt",
            _formal_checkpoint(
                phase="paper_model",
                model=model,
                task=task,
                fold=args.fold,
                epoch=selected_epoch,
                history=final_history,
                parent=final_parent,
                job=job,
                selection_pilot=pilot_summary,
            ),
        )
        training_latent.close()
        test_latent = FoldLatentStore(args.test_latent, scale)
        test_materializer = UnitMaterializer(
            uce=uce,
            latent=test_latent,
            query_uce_cells=runtime.query_uce_cells,
            support_uce_cells=runtime.support_uce_cells,
            seed=runtime.seed,
        )
        manifest, prediction_artifacts = predict_heldout_v2(
            model=model,
            arm="current",
            units=outer_test,
            materializer=test_materializer,
            runtime=runtime,
            clinical_mode="off",
            gene_selection=None,
            output=args.output,
            device=device,
            paper_method=job.paper_method,
        )
    finally:
        try:
            training_latent.close()
        except Exception:
            pass
        if test_latent is not None:
            test_latent.close()
        uce.close()

    formal_artifacts: dict[str, Path] = {
        "checkpoint": args.output / "checkpoint.pt",
        "best_validation_checkpoint": args.output / "best_validation.pt",
        "selection": args.output / "selection.json",
        "selection_history": args.output / "selection_history.tsv",
        "final_history": args.output / "final_history.tsv",
        **prediction_artifacts,
    }
    run_payload = build_formal_run_payload(
        job=job,
        test_units=len(outer_test),
        test_donors=test_donors,
        artifacts=formal_artifacts,
        extra={
            "projected_set_schema": SCHEMA,
            "clinical_mode": "off",
            "clinical_information_used": False,
            "history_information_used": False,
            "gene_prior_mode": "none",
            "history_training_mode": "none",
            "current_model_dependency": None,
            "lineage_gene_summary_dependency": None,
            "parent_set_transport_dependency": {
                "protocol": _record(args.parent_protocol),
                "selection_inner_fit_checkpoint": _record(parent_selection),
                "final_outer_training_checkpoint": _record(parent_final),
                "inner_selection_never_uses_outer_trained_parent": True,
            },
            "selection_pilot_dependency": selection_dependency,
            "training_donors": len({unit.donor_id for unit in outer_train}),
            "training_units": len(outer_train),
            "selected_epoch": int(pilot_summary["best_epoch"]),
            "final_history": final_history,
            "runtime_config": asdict(runtime),
            "model_config": model_config.to_dict(),
            "loss_config": loss_config.to_dict(),
            "prediction_units": len(manifest),
            "mechanism_inference_same_checkpoint": True,
            "mechanism_variants_generated_before_heldout_target_materialization": True,
            "mechanism_variants": list(MECHANISM_VARIANTS),
            "elapsed_seconds": time.time() - started,
            **runtime_resource_snapshot(device),
            "torch": torch.__version__,
            "runner_source": str(Path(__file__).resolve()),
        },
    )
    run_path = args.output / "run.json"
    atomic_json(run_path, run_payload)
    atomic_json(
        complete_path,
        {
            "status": "pass",
            "schema_version": V2_COMPLETE_SCHEMA,
            "job_identifier": job.identifier,
            "run_json": str(run_path.resolve()),
        },
    )
    validate_completed_v2_run(
        args.output, protocol=protocol, job_identifier=identifier
    )
    print(json.dumps(run_payload, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

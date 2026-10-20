from __future__ import annotations
from paths import ROOT, expand_paths
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields, replace
import fcntl
import inspect
import json
import math
from pathlib import Path
import random
import resource
import time
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
import torch
import yaml
from model.encoder import TwinArm, VaccineImmuneTwinConfig
from data.sound import TASK_PHASE, FoldLatentStore, GeneSelection, GeneSummaryStore, SealedBaseUCEStore, UnifiedUCEStore, UnitInputs, UnitMaterializer, VaccineUnit, build_task_units, canonical_task, iter_donor_batches, load_fold_scale, save_gene_selection, select_outer_training_genes, split_inner_units, split_outer_units, stable_seed
from model.losses import PatientLossV2Config, donor_macro_mean, patient_vaccine_objective_v2, response_vectors
from data.lineage import LineageGeneFeatureBatch, LineageGenePreprocessor, LineageGeneSummaryStore, fit_lineage_gene_preprocessor, materialize_lineage_gene_features, refit_outer_training_lineage_gene_preprocessor
from pipeline.sound.provenance import V2_COMPLETE_SCHEMA, V2_METHODS, V2_TRAIN_SCHEMA, V2JobRecord, V2Protocol, atomic_json, load_v2_protocol, validate_completed_v2_run
from pipeline.sound.common import _state_dict_cpu, _validate_manifest_audit, atomic_table, atomic_torch_save, build_model, prepare_clinical_history, require_cuda, save_npz, set_encoder_frozen, set_seed

SELECTION_SCHEMA = "sound_vaccine_immunetwin_selection_2"

CONFIG_SCHEMA = "sound_vaccine_immunetwin_2"

MECHANISM_SCHEMA = "sound_vaccine_immunetwin_mechanism_inference_2"

MECHANISM_VARIANTS = (
    "own_history",
    "empty_support",
    "donor_shuffled_support",
    "reversed_support",
)

FORMAL_MECHANISM_ARTIFACTS = frozenset(
    {
        "mechanism_predictions",
        "mechanism_inference",
        "donor_shuffle_assignments",
        "own_history_prediction",
        "same_checkpoint_empty_support_prediction",
        "shuffled_patient_support_prediction",
        "reversed_transition_prediction",
        "response_coordinates_w",
        "patient_response_coordinates_w",
    }
)

FORMAL_GENE_MECHANISM_ARTIFACTS = frozenset(
    {
        "unit_gene_weights",
        "patient_gene_weights",
        "patient_gene_weight_summary",
        "patient_gene_weight_variance",
    }
)

CLINICAL_MODES = ("off", "delta", "baseline_delta")

HISTORY_TRAINING_MODES = ("residual_from_current", "joint")

SELECTION_OBJECTIVES = {
    "distribution_mmd2",
    "raw_donor_delta_rmse",
    "raw_donor_delta_mae",
    "raw_donor_response_cosine",
    "response_dispersion_ratio",
    "response_norm_ratio",
    "patient_objective",
}

@dataclass(frozen=True)
class RuntimeV2Config:
    seed: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    maximum_epochs: int
    patience: int
    gradient_clip_norm: float
    warmup_frozen_epochs: int
    encoder_learning_rate_multiplier: float
    query_uce_cells: int
    support_uce_cells: int
    training_latent_cells: int
    prediction_latent_cells: int
    gene_candidates_per_fold: int
    selection_objective: str
    selection_direction: str
    response_scale_floor: float
    history_training_mode: str = "residual_from_current"
    freeze_pretrained_encoder: bool = False

@dataclass(frozen=True)
class ResponseTransform:
    center: np.ndarray
    scale: np.ndarray
    fit_donors: int
    fit_units: int
    cells_per_visit: int
    stage: str
    mean_program_basis: np.ndarray | None = None
    covariance_program_basis: np.ndarray | None = None
    mean_program_explained_fraction: float | None = None
    covariance_program_explained_fraction: float | None = None
    relative_covariance_eigenvalue_minimum: float | None = None
    relative_covariance_eigenvalue_maximum: float | None = None

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "stage": self.stage,
            "center": self.center.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
            "fit_donors": self.fit_donors,
            "fit_units": self.fit_units,
            "cells_per_visit": self.cells_per_visit,
            "aggregation": "season mean within donor, then equal donor mean",
            "sampling": "fixed seed, equal cells per source and target visit",
            "coordinate_system": "fold-standardized scVI 16-D",
        }
        if self.mean_program_basis is not None:
            payload["mean_program_basis"] = self.mean_program_basis.astype(
                float
            ).tolist()
            payload["mean_program_rank"] = int(self.mean_program_basis.shape[0])
            payload["mean_program_explained_fraction"] = float(
                self.mean_program_explained_fraction or 0.0
            )
        if self.covariance_program_basis is not None:
            payload["covariance_program_basis"] = (
                self.covariance_program_basis.astype(float).tolist()
            )
            payload["covariance_program_rank"] = int(
                self.covariance_program_basis.shape[0]
            )
            payload["covariance_program_explained_fraction"] = float(
                self.covariance_program_explained_fraction or 0.0
            )
            payload["relative_covariance_eigenvalue_minimum"] = float(
                self.relative_covariance_eigenvalue_minimum or 0.0
            )
            payload["relative_covariance_eigenvalue_maximum"] = float(
                self.relative_covariance_eigenvalue_maximum or 0.0
            )
        return payload

@dataclass(frozen=True)
class PhaseSupportResponseReference:
    """Donor-equal empirical reference for completed support responses only."""

    center: np.ndarray
    observed_mask: np.ndarray
    fit_donors: int
    fit_units: int
    support_transitions: int
    support_samples: int
    phase_donors: tuple[int, ...]
    phase_transitions: tuple[int, ...]
    phase_samples: tuple[int, ...]
    cells_per_visit: int
    stage: str

    def to_json(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "fit_scope": (
                "inner_fit_donors_only"
                if self.stage == "selection"
                else "outer_training_donors_only"
            ),
            "center": self.center.astype(float).tolist(),
            "observed_mask": self.observed_mask.astype(bool).tolist(),
            "phase_count": int(self.center.shape[0]),
            "response_dimension": int(self.center.shape[1]),
            "fit_donors": self.fit_donors,
            "fit_units": self.fit_units,
            "support_transitions": self.support_transitions,
            "support_samples": self.support_samples,
            "phase_donors": list(self.phase_donors),
            "phase_transitions": list(self.phase_transitions),
            "phase_samples": list(self.phase_samples),
            "cells_per_visit": self.cells_per_visit,
            "aggregation": (
                "completed transitions averaged within donor and phase, then "
                "equal-weighted across donors"
            ),
            "sampling": (
                "fixed deterministic sample-id draw; equal cells per completed "
                "support visit"
            ),
            "forecast_targets_read": False,
            "coordinate_system": "fold-standardized scVI 16-D",
        }


def _optional_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}

_TASK_OVERRIDE_SECTIONS = frozenset({"model", "loss"})

_MODEL_CONFIG_KEYS = frozenset(
    {field.name for field in fields(VaccineImmuneTwinConfig)}
    # Stored under ``model`` but read by RuntimeV2Config.
    | {"gene_candidates_per_fold"}
)

_TASK_OVERRIDE_MODEL_KEYS = frozenset(
    key for key in _MODEL_CONFIG_KEYS if key != "gene_candidates_per_fold"
)

_LOSS_CONFIG_KEYS = frozenset(
    {field.name for field in fields(PatientLossV2Config)}
    | {"mmd_weight", "response_mse_weight", "response_cosine_weight"}
)

def _validate_config_keys(
    section: str,
    value: object,
    *,
    allowed: frozenset[str],
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}.{section} must be a mapping")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(
            f"{context}.{section} contains unknown keys: {sorted(unknown)}"
        )
    return value

def _validated_task_overrides(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if "task_overrides" not in payload:
        return {}
    raw = payload["task_overrides"]
    if not isinstance(raw, Mapping):
        raise ValueError("task_overrides must be a mapping")
    for task_name, value in raw.items():
        if task_name not in TASK_PHASE:
            try:
                resolved = canonical_task(str(task_name))
            except ValueError:
                resolved = None
            if resolved is not None:
                raise ValueError(
                    "task_overrides keys must use canonical task identifiers; "
                    f"replace {task_name!r} with {resolved!r}"
                )
            raise ValueError(f"task_overrides contains unknown task: {task_name!r}")
        if not isinstance(value, Mapping):
            raise ValueError(f"task override {task_name} must be a mapping")
        forbidden = set(value) - set(_TASK_OVERRIDE_SECTIONS)
        if forbidden:
            raise ValueError(
                "task overrides may change only model and loss; "
                f"{task_name} changes {sorted(forbidden)}"
            )
        for section, allowed in (
            ("model", _TASK_OVERRIDE_MODEL_KEYS),
            ("loss", frozenset(field.name for field in fields(PatientLossV2Config))),
        ):
            if section in value:
                _validate_config_keys(
                    section,
                    value[section],
                    allowed=allowed,
                    context=f"task_overrides.{task_name}",
                )
    return raw

def load_configuration_v2(
    path: str | Path,
    *,
    arm: TwinArm | None = None,
    task: str | None = None,
) -> tuple[
    dict[str, Any], RuntimeV2Config, VaccineImmuneTwinConfig, PatientLossV2Config
]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = expand_paths(yaml.safe_load(handle))
    if not isinstance(payload, dict):
        raise ValueError("vaccine ImmuneTwin config must be a mapping")
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(
            f"vaccine v2 config schema must be {CONFIG_SCHEMA}"
        )
    effective_payload = dict(payload)
    overrides = _validated_task_overrides(payload)
    if task is not None:
        task_id = canonical_task(task)
        selected_override = _optional_mapping(overrides.get(task_id))
        for section in ("model", "loss"):
            merged = dict(_optional_mapping(payload.get(section)))
            merged.update(_optional_mapping(selected_override.get(section)))
            effective_payload[section] = merged

    _validate_config_keys(
        "model",
        effective_payload.get("model", {}),
        allowed=_MODEL_CONFIG_KEYS,
        context="config",
    )
    _validate_config_keys(
        "loss",
        effective_payload.get("loss", {}),
        allowed=_LOSS_CONFIG_KEYS,
        context="config",
    )

    optimization = _optional_mapping(effective_payload.get("optimization"))
    features_config = _optional_mapping(effective_payload.get("features"))
    pretraining = _optional_mapping(effective_payload.get("pretraining"))
    selection = _optional_mapping(effective_payload.get("selection"))
    response = _optional_mapping(effective_payload.get("response_transform"))
    history_training = _optional_mapping(effective_payload.get("history_training"))
    history_training_mode = str(
        history_training.get("mode", "residual_from_current")
    )
    if history_training_mode not in HISTORY_TRAINING_MODES:
        raise ValueError(
            "history_training.mode must be residual_from_current or joint"
        )
    objective = str(
        selection.get(
            "objective",
            optimization.get("early_stopping_objective", "raw_donor_delta_rmse"),
        )
    )
    direction = str(selection.get("direction", "lower"))
    if objective not in SELECTION_OBJECTIVES:
        raise ValueError(f"unknown v2 selection objective: {objective}")
    if direction not in {"lower", "higher", "target_one"}:
        raise ValueError("selection direction must be lower, higher, or target_one")
    runtime = RuntimeV2Config(
        seed=int(payload.get("seed", 42)),
        batch_size=int(optimization.get("batch_size_donors", 16)),
        learning_rate=float(optimization.get("learning_rate", 3e-4)),
        weight_decay=float(optimization.get("weight_decay", 1e-4)),
        maximum_epochs=int(optimization.get("maximum_epochs", 300)),
        patience=int(optimization.get("patience", 20)),
        gradient_clip_norm=float(optimization.get("gradient_clip_norm", 5.0)),
        warmup_frozen_epochs=int(pretraining.get("warmup_frozen_epochs", 5)),
        encoder_learning_rate_multiplier=float(
            pretraining.get("encoder_learning_rate_multiplier", 0.1)
        ),
        query_uce_cells=int(features_config.get("query_uce_cells", 512)),
        support_uce_cells=int(features_config.get("support_uce_cells", 512)),
        training_latent_cells=int(features_config.get("training_latent_cells", 512)),
        prediction_latent_cells=int(
            features_config.get("prediction_latent_cells", 2048)
        ),
        gene_candidates_per_fold=int(
            _optional_mapping(effective_payload.get("model")).get(
                "gene_candidates_per_fold", 512
            )
        ),
        selection_objective=objective,
        selection_direction=direction,
        response_scale_floor=float(response.get("scale_floor", 1e-3)),
        history_training_mode=history_training_mode,
        freeze_pretrained_encoder=bool(
            pretraining.get("freeze_pretrained_encoder", False)
        ),
    )
    if runtime.seed != 42:
        raise ValueError("formal vaccine v2 protocol requires seed 42")
    positive = (
        runtime.batch_size,
        runtime.maximum_epochs,
        runtime.patience,
        runtime.query_uce_cells,
        runtime.support_uce_cells,
        runtime.training_latent_cells,
        runtime.prediction_latent_cells,
        runtime.gene_candidates_per_fold,
    )
    if any(value < 1 for value in positive) or runtime.response_scale_floor <= 0:
        raise ValueError("v2 runtime counts and response scale floor must be positive")

    model_payload = _optional_mapping(effective_payload.get("model"))
    model_fields = {field.name for field in fields(VaccineImmuneTwinConfig)}
    model_config = VaccineImmuneTwinConfig(
        **{key: value for key, value in model_payload.items() if key in model_fields}
    )
    if arm is not None:
        if arm not in V2_METHODS:
            raise ValueError(f"unknown ImmuneTwin v2 arm: {arm}")
        if arm != "current" and model_config.history_response_dim != 16:
            raise ValueError(
                "v2 history arms require history_response_dim=16; "
                "implicit fallback to the v1 zero-width history input is forbidden"
            )
    loss = _optional_mapping(effective_payload.get("loss"))
    loss_config = PatientLossV2Config(
        distribution_weight=float(
            loss.get("distribution_weight", loss.get("mmd_weight", 1.0))
        ),
        response_mean_weight=float(
            loss.get("response_mean_weight", loss.get("response_mse_weight", 2.0))
        ),
        response_direction_weight=float(
            loss.get(
                "response_direction_weight", loss.get("response_cosine_weight", 0.5)
            )
        ),
        response_norm_weight=float(loss.get("response_norm_weight", 0.5)),
        centered_geometry_weight=float(
            loss.get("centered_geometry_weight", 0.5)
        ),
        geometry_gram_fraction=float(loss.get("geometry_gram_fraction", 0.25)),
        history_sensitivity_weight=float(
            loss.get("history_sensitivity_weight", 0.0)
        ),
        history_sensitivity_temperature=float(
            loss.get("history_sensitivity_temperature", 1.0)
        ),
        history_current_anchor_weight=float(
            loss.get("history_current_anchor_weight", 0.0)
        ),
        history_current_anchor_temperature=float(
            loss.get("history_current_anchor_temperature", 1.0)
        ),
        geometry_scale_floor=float(loss.get("geometry_scale_floor", 1e-3)),
        epsilon=float(loss.get("epsilon", 1e-8)),
    )
    if arm == "current" and (
        loss_config.history_sensitivity_weight != 0
        or loss_config.history_current_anchor_weight != 0
    ):
        # Defined only for history-conditioned fits, not for the current-state model.
        loss_config = replace(
            loss_config,
            history_sensitivity_weight=0.0,
            history_current_anchor_weight=0.0,
        )
    return effective_payload, runtime, model_config, loss_config

def validate_arm_task_v2(arm: str, task: str) -> tuple[TwinArm, str]:
    if arm not in V2_METHODS:
        raise ValueError(f"unknown ImmuneTwin arm: {arm}")
    task_id = canonical_task(task)
    if task_id == "t1_y1_d0_to_y1_d7" and arm != "current":
        raise ValueError("T1 history arms must reuse the current checkpoint/prediction")
    return arm, task_id  # type: ignore[return-value]

def resolve_clinical_history_v2(arm: TwinArm, requested: str | None) -> bool:
    if requested not in {None, "on", "off"}:
        raise ValueError("clinical history toggle must be on or off")
    enabled = arm in {"w", "full"} if requested is None else requested == "on"
    if enabled and arm not in {"w", "full"}:
        raise ValueError("clinical history is allowed only in w/full")
    return enabled

def resolve_clinical_mode(
    payload: Mapping[str, Any], *, arm: TwinArm, enabled: bool
) -> str:
    if not enabled:
        return "off"
    clinical = _optional_mapping(payload.get("clinical"))
    features_config = _optional_mapping(payload.get("features"))
    by_arm = _optional_mapping(
        clinical.get("mode_by_arm", features_config.get("clinical_mode_by_arm"))
    )
    value = by_arm.get(
        arm,
        clinical.get(
            "mode",
            features_config.get("clinical_history_mode", "baseline_delta"),
        ),
    )
    mode = str(value)
    if mode not in CLINICAL_MODES:
        raise ValueError("clinical config mode must be off, delta, or baseline_delta")
    return mode

def _clinical_model_dim(features: pd.DataFrame | None, mode: str) -> int:
    if mode == "off":
        return 0
    if features is None:
        raise ValueError("clinical mode is enabled but clinical features are absent")
    value_columns = [column for column in features if str(column).startswith("value::")]
    if not value_columns:
        raise ValueError("clinical feature table has no value:: columns")
    return len(value_columns) * (2 if mode == "delta" else 4)

def _sample_covariance(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float64) - values.mean(axis=0, keepdims=True)
    return centered.T @ centered / max(len(values) - 1, 1)

def _upper_triangle(values: np.ndarray) -> np.ndarray:
    return values[np.triu_indices(values.shape[0])]

def _scaled_uncentered_program_basis(
    values: np.ndarray, rank: int
) -> tuple[np.ndarray | None, float | None]:
    if rank == 0:
        return None, None
    if rank < 0:
        raise ValueError("response program rank cannot be negative")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("response program matrix must be finite and two-dimensional")
    _, singular, right = np.linalg.svd(values, full_matrices=False)
    available = min(rank, right.shape[0])
    directions = right[:available]
    scores = values @ directions.T
    score_scale = np.sqrt(np.mean(np.square(scores), axis=0))
    scaled = score_scale[:, None] * directions
    if available < rank:
        scaled = np.concatenate(
            (scaled, np.zeros((rank - available, values.shape[1]))), axis=0
        )
    total = float(np.square(singular).sum())
    explained = (
        float(np.square(singular[:available]).sum() / total) if total > 0 else 0.0
    )
    return scaled.astype(np.float32), explained

def _relative_covariance_eigenvalues(
    source_covariance: np.ndarray, target_covariance: np.ndarray
) -> np.ndarray:
    symmetric = 0.5 * (source_covariance + source_covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    floor = max(float(eigenvalues.max()) * 1e-5, 1e-6)
    inverse_sqrt = (
        eigenvectors * np.power(np.maximum(eigenvalues, floor), -0.5)[None, :]
    ) @ eigenvectors.T
    relative = inverse_sqrt @ target_covariance @ inverse_sqrt
    return np.linalg.eigvalsh(0.5 * (relative + relative.T))

def estimate_response_transform(
    latent: FoldLatentStore,
    units: Sequence[VaccineUnit],
    *,
    cells: int,
    seed: int,
    stage: str,
    scale_floor: float,
    mean_program_rank: int = 0,
    covariance_program_rank: int = 0,
) -> ResponseTransform:
    """Fit a 16-D response transform on only the supplied donor units."""

    if not units:
        raise ValueError("response transform fit units cannot be empty")
    by_donor: dict[str, list[np.ndarray]] = {}
    covariance_by_donor: dict[str, list[np.ndarray]] = {}
    relative_covariance_eigenvalues: list[np.ndarray] = []
    for unit in sorted(units, key=lambda item: (item.donor_id, item.season, item.unit_id)):
        source = latent.take(
            unit.source_sample,
            cells=cells,
            seed=stable_seed(
                "response_transform", stage, unit.unit_id, "source", base_seed=seed
            ),
        )[0]
        target = latent.take(
            unit.target_sample,
            cells=cells,
            seed=stable_seed(
                "response_transform", stage, unit.unit_id, "target", base_seed=seed
            ),
        )[0]
        if source.shape != (cells, 16) or target.shape != (cells, 16):
            raise ValueError("response transform expects sampled [cells,16] scVI states")
        response = (
            target.mean(axis=0, dtype=np.float64)
            - source.mean(axis=0, dtype=np.float64)
        )
        if not np.isfinite(response).all():
            raise ValueError("response transform encountered a non-finite response")
        by_donor.setdefault(unit.donor_id, []).append(response)
        source_covariance = _sample_covariance(source)
        target_covariance = _sample_covariance(target)
        covariance_response = _upper_triangle(target_covariance - source_covariance)
        covariance_by_donor.setdefault(unit.donor_id, []).append(covariance_response)
        relative_covariance_eigenvalues.append(
            _relative_covariance_eigenvalues(source_covariance, target_covariance)
        )
    donor_responses = np.stack(
        [np.stack(by_donor[donor]).mean(axis=0) for donor in sorted(by_donor)]
    )
    center = donor_responses.mean(axis=0)
    scale = np.sqrt(np.mean(np.square(donor_responses - center), axis=0))
    scale = np.maximum(scale, float(scale_floor))
    donor_covariance_responses = np.stack(
        [
            np.stack(covariance_by_donor[donor]).mean(axis=0)
            for donor in sorted(covariance_by_donor)
        ]
    )
    mean_program_basis, mean_program_explained = (
        _scaled_uncentered_program_basis(donor_responses, mean_program_rank)
    )
    covariance_program_basis, covariance_program_explained = (
        _scaled_uncentered_program_basis(
            donor_covariance_responses, covariance_program_rank
        )
    )
    relative_values = np.concatenate(relative_covariance_eigenvalues)
    relative_minimum = float(np.quantile(relative_values, 0.01))
    relative_maximum = float(np.quantile(relative_values, 0.99))
    if not 0.0 < relative_minimum < relative_maximum:
        raise ValueError("training relative-covariance bounds are invalid")
    return ResponseTransform(
        center=center.astype(np.float32),
        scale=scale.astype(np.float32),
        fit_donors=len(by_donor),
        fit_units=len(units),
        cells_per_visit=cells,
        stage=stage,
        mean_program_basis=mean_program_basis,
        covariance_program_basis=covariance_program_basis,
        mean_program_explained_fraction=mean_program_explained,
        covariance_program_explained_fraction=covariance_program_explained,
        relative_covariance_eigenvalue_minimum=relative_minimum,
        relative_covariance_eigenvalue_maximum=relative_maximum,
    )

def save_response_transform(transform: ResponseTransform, path: Path) -> dict[str, object]:
    payload = {"status": "pass", **transform.to_json()}
    atomic_json(path, payload)
    return {
        "path": str(path.resolve()),
        **payload,
    }

def set_model_response_transform(
    model: torch.nn.Module, transform: ResponseTransform
) -> None:
    center = torch.as_tensor(transform.center)
    scale = torch.as_tensor(transform.scale)
    setter = getattr(model, "set_response_transform", None)
    if callable(setter):
        setter(center, scale)
    else:
        transport = getattr(model, "transport", None)
        setter = getattr(transport, "set_response_transform", None)
        if not callable(setter):
            raise RuntimeError("model lacks a response-transform setter")
        setter(center, scale)

    program_setter = getattr(model, "set_response_programs", None)
    if not callable(program_setter):
        transport = getattr(model, "transport", None)
        program_setter = getattr(transport, "set_response_programs", None)
    has_programs = (
        transform.mean_program_basis is not None
        or transform.covariance_program_basis is not None
    )
    if not has_programs:
        if callable(program_setter):
            raise ValueError("program-affine model requires fitted response programs")
        return
    if not callable(program_setter):
        raise ValueError("response programs were fitted for a model without a program setter")
    if (
        transform.mean_program_basis is None
        or transform.covariance_program_basis is None
        or transform.relative_covariance_eigenvalue_minimum is None
        or transform.relative_covariance_eigenvalue_maximum is None
    ):
        raise ValueError("response program transform is incomplete")
    program_setter(
        mean_program_basis=torch.as_tensor(transform.mean_program_basis),
        covariance_program_basis=torch.as_tensor(
            transform.covariance_program_basis
        ),
        relative_covariance_eigenvalue_minimum=(
            transform.relative_covariance_eigenvalue_minimum
        ),
        relative_covariance_eigenvalue_maximum=(
            transform.relative_covariance_eigenvalue_maximum
        ),
    )

def estimate_phase_support_response_reference(
    latent: FoldLatentStore,
    units: Sequence[VaccineUnit],
    *,
    cells: int,
    seed: int,
    stage: str,
    phase_count: int,
) -> PhaseSupportResponseReference:
    """Estimate phase means from completed transitions.

    Each transition is reduced to a visit-mean response; repeated seasons are
    averaged within donor and phase, so every donor has equal weight.
    """

    if not units:
        raise ValueError("support-response reference fit units cannot be empty")
    if cells < 1 or phase_count < 1:
        raise ValueError("support-response reference cells and phases must be positive")
    if stage not in {"selection", "final"}:
        raise ValueError("support-response reference stage must be selection or final")
    unit_ids = [str(unit.unit_id) for unit in units]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("support-response reference fit units must be unique")

    # Exclude every forecast-target sample of the fitting task from this
    # preprocessing step.
    forecast_targets = {str(unit.target_sample) for unit in units}
    sample_cache: dict[str, np.ndarray] = {}

    def visit_mean(sample_id: str) -> np.ndarray:
        cached = sample_cache.get(sample_id)
        if cached is not None:
            return cached
        values = latent.take(
            sample_id,
            cells=cells,
            seed=stable_seed(
                "phase_support_response_reference", sample_id, base_seed=seed
            ),
        )[0]
        values = np.asarray(values)
        if values.shape != (cells, 16):
            raise ValueError(
                "support-response reference expects sampled [cells,16] scVI states"
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"support-response reference encountered non-finite sample {sample_id}"
            )
        mean = values.mean(axis=0, dtype=np.float64)
        sample_cache[sample_id] = mean
        return mean

    by_phase_donor: list[dict[str, list[np.ndarray]]] = [
        {} for _ in range(phase_count)
    ]
    phase_transition_samples: list[set[str]] = [set() for _ in range(phase_count)]
    transition_count = 0
    for unit in sorted(units, key=lambda item: (item.donor_id, item.season, item.unit_id)):
        for transition in unit.supports:
            source_sample = str(transition.source_sample)
            target_sample = str(transition.target_sample)
            if source_sample in forecast_targets or target_sample in forecast_targets:
                raise ValueError(
                    "completed support overlaps a forecast target; refusing leakage"
                )
            if source_sample == target_sample:
                raise ValueError("completed support source and target must differ")
            phase = int(transition.phase)
            if phase < 0 or phase >= phase_count:
                raise ValueError("completed support phase is outside model phase range")
            response = visit_mean(target_sample) - visit_mean(source_sample)
            by_phase_donor[phase].setdefault(str(unit.donor_id), []).append(response)
            phase_transition_samples[phase].update((source_sample, target_sample))
            transition_count += 1

    center = np.zeros((phase_count, 16), dtype=np.float32)
    observed = np.zeros(phase_count, dtype=bool)
    phase_donors: list[int] = []
    phase_transitions: list[int] = []
    phase_samples: list[int] = []
    for phase, donor_values in enumerate(by_phase_donor):
        donors = sorted(donor_values)
        samples = sorted(phase_transition_samples[phase])
        phase_donors.append(len(donors))
        phase_transitions.append(sum(len(donor_values[donor]) for donor in donors))
        phase_samples.append(len(samples))
        if donors:
            donor_means = np.stack(
                [np.stack(donor_values[donor]).mean(axis=0) for donor in donors]
            )
            center[phase] = donor_means.mean(axis=0).astype(np.float32)
            observed[phase] = True
    if not observed.any():
        raise ValueError("support-response reference requires completed supports")

    support_samples = sorted(sample_cache)
    fit_donors = sorted({str(unit.donor_id) for unit in units})
    return PhaseSupportResponseReference(
        center=center,
        observed_mask=observed,
        fit_donors=len(fit_donors),
        fit_units=len(units),
        support_transitions=transition_count,
        support_samples=len(support_samples),
        phase_donors=tuple(phase_donors),
        phase_transitions=tuple(phase_transitions),
        phase_samples=tuple(phase_samples),
        cells_per_visit=cells,
        stage=stage,
    )

def save_phase_support_response_reference(
    reference: PhaseSupportResponseReference, path: Path
) -> dict[str, object]:
    payload = {"status": "pass", **reference.to_json()}
    atomic_json(path, payload)
    return {
        "path": str(path.resolve()),
        **payload,
    }

def set_model_support_response_reference(
    model: torch.nn.Module, reference: PhaseSupportResponseReference
) -> None:
    """Duck-typed integration point for the residual-posterior model."""

    setter = getattr(model, "set_support_response_reference", None)
    if not callable(setter):
        raise RuntimeError(
            "residual-posterior model lacks set_support_response_reference"
        )
    setter(
        torch.as_tensor(reference.center),
        torch.as_tensor(reference.observed_mask, dtype=torch.bool),
    )

def prepare_clinical_stage(
    *,
    stage: str,
    enabled: bool,
    fit_units: Sequence[VaccineUnit],
    transform_units: Sequence[VaccineUnit],
    excluded_donors: set[str],
    labs_path: Path | None,
    hai_path: Path | None,
    igg_path: Path | None,
    contract_path: Path,
    output: Path,
) -> tuple[pd.DataFrame | None, dict[str, object]]:
    fit_donors = {unit.donor_id for unit in fit_units}
    audit_path = output / f"{stage}_clinical_preprocessor.json"
    if not enabled:
        artifact: dict[str, object] = {
            "status": "disabled",
            "stage": stage,
            "fit_donors": len(fit_donors),
        }
        atomic_json(audit_path, artifact)
        artifact.update({"path": str(audit_path.resolve())})
        return None, artifact
    if any(path is None for path in (labs_path, hai_path, igg_path)):
        raise ValueError("clinical history requires labs, HAI, and IgG paths")
    required_support_samples = {
        sample
        for unit in transform_units
        for transition in unit.supports
        for sample in (transition.source_sample, transition.target_sample)
    }
    stage_root = output / f"{stage}_preprocessing"
    features, fitted = prepare_clinical_history(
        labs_path=labs_path,  # type: ignore[arg-type]
        hai_path=hai_path,  # type: ignore[arg-type]
        igg_path=igg_path,  # type: ignore[arg-type]
        contract_path=contract_path,
        outer_training_donors=fit_donors,
        outer_test_donors=set(excluded_donors),
        required_support_samples=required_support_samples,
        output=stage_root,
    )
    raw_path = stage_root / "clinical_preprocessor.json"
    artifact = {
        "status": "pass",
        "stage": stage,
        "fit_scope": "inner_fit_donors_only"
        if stage == "selection"
        else "outer_training_donors_only",
        "fit_donors": len(fit_donors),
        "transform_units": len(transform_units),
        "raw_preprocessor_path": str(raw_path.resolve()),
        "preprocessor": fitted,
    }
    atomic_json(audit_path, artifact)
    artifact.update({"path": str(audit_path.resolve())})
    return features, artifact

def prepare_gene_stage(
    *,
    stage: str,
    enabled: bool,
    store: GeneSummaryStore | None,
    manifest_path: Path | None,
    fit_units: Sequence[VaccineUnit],
    embedding_path: Path | None,
    n_genes: int,
    output: Path,
) -> tuple[GeneSelection | None, dict[str, object]]:
    fit_donors = {unit.donor_id for unit in fit_units}
    audit_path = output / f"{stage}_gene_preprocessor.json"
    if not enabled:
        artifact: dict[str, object] = {
            "status": "disabled",
            "stage": stage,
            "fit_donors": len(fit_donors),
        }
        atomic_json(audit_path, artifact)
        artifact.update({"path": str(audit_path.resolve())})
        return None, artifact
    if store is None or manifest_path is None or embedding_path is None:
        raise ValueError("gene/full arm requires unified gene summaries and embeddings")
    selection = select_outer_training_genes(
        store,
        manifest_path=manifest_path,
        outer_training_donors=fit_donors,
        embedding_path=embedding_path,
        n_genes=n_genes,
    )
    table_path = output / f"{stage}_gene_selection.tsv"
    save_gene_selection(selection, table_path)
    artifact = {
        "status": "pass",
        "stage": stage,
        "fit_scope": "inner_fit_donors_only"
        if stage == "selection"
        else "outer_training_donors_only",
        "fit_donors": len(fit_donors),
        "selected_genes": len(selection.names),
        "selection_path": str(table_path.resolve()),
    }
    atomic_json(audit_path, artifact)
    artifact.update({"path": str(audit_path.resolve())})
    return selection, artifact

def prepare_lineage_gene_stage(
    *,
    stage: str,
    store: LineageGeneSummaryStore,
    fit_units: Sequence[VaccineUnit],
    embedding_path: Path,
    n_genes: int,
    output: Path,
) -> tuple[GeneSelection, LineageGenePreprocessor, dict[str, object]]:
    """Fit and save the five-lineage gene transform."""

    fit_donors = {unit.donor_id for unit in fit_units}
    if stage == "selection":
        preprocessor = fit_lineage_gene_preprocessor(
            store=store,
            units=fit_units,
            fit_donors=fit_donors,
            gene_embedding_path=embedding_path,
            stage="selection",
            max_genes=n_genes,
        )
    elif stage == "final":
        preprocessor = refit_outer_training_lineage_gene_preprocessor(
            store=store,
            units=fit_units,
            outer_training_donors=fit_donors,
            gene_embedding_path=embedding_path,
            max_genes=n_genes,
        )
    else:
        raise ValueError("lineage gene stage must be selection or final")
    parameters_path = output / f"{stage}_lineage_gene_preprocessor.npz"
    save_npz(
        parameters_path,
        schema_version=np.asarray("sound_lineage_gene_preprocessor_1"),
        stage=np.asarray(preprocessor.stage),
        gene_names=np.asarray(preprocessor.gene_names, dtype=str),
        raw_gene_indices=preprocessor.raw_gene_indices,
        gene_embeddings=preprocessor.gene_embeddings,
        delta_mean=preprocessor.delta_mean,
        delta_scale=preprocessor.delta_scale,
        delta_variance=preprocessor.delta_variance,
        mean_detection=preprocessor.mean_detection,
        selection_score=preprocessor.selection_score,
        selection_rank=preprocessor.selection_rank,
    )
    selection = GeneSelection(
        names=preprocessor.gene_names,
        raw_indices=preprocessor.raw_gene_indices.copy(),
        variances=preprocessor.delta_variance.max(axis=0).copy(),
        embeddings=torch.as_tensor(preprocessor.gene_embeddings).clone(),
    )
    selection_path = output / f"{stage}_gene_selection.tsv"
    save_gene_selection(selection, selection_path)
    audit_path = output / f"{stage}_gene_preprocessor.json"
    artifact: dict[str, object] = {
        **dict(preprocessor.provenance),
        "status": "pass",
        "stage": stage,
        "gene_prior_mode": "lineage",
        "parameters_path": str(parameters_path.resolve()),
        "selection_path": str(selection_path.resolve()),
    }
    atomic_json(audit_path, artifact)
    artifact.update(
        {
            "path": str(audit_path.resolve()),
        }
    )
    return selection, preprocessor, artifact

def _slice_lineage_features(
    values: LineageGeneFeatureBatch, indices: Sequence[int], *, unit_ids: Sequence[str]
) -> LineageGeneFeatureBatch:
    positions = np.asarray(indices, dtype=np.int64)
    return replace(
        values,
        unit_ids=tuple(str(value) for value in unit_ids),
        delta=values.delta[positions].copy(),
        raw_delta_logcpm=values.raw_delta_logcpm[positions].copy(),
        confidence=values.confidence[positions].copy(),
        feature_mask=values.feature_mask[positions].copy(),
        lineage_mask=values.lineage_mask[positions].copy(),
        support_mask=values.support_mask[positions].copy(),
        support_phase=values.support_phase[positions].copy(),
        support_dt=values.support_dt[positions].copy(),
    )

def _empty_lineage_features(
    values: LineageGeneFeatureBatch, *, unit_ids: Sequence[str]
) -> LineageGeneFeatureBatch:
    batch = len(unit_ids)
    return replace(
        values,
        unit_ids=tuple(str(value) for value in unit_ids),
        delta=np.empty((batch, 0, *values.delta.shape[2:]), dtype=values.delta.dtype),
        raw_delta_logcpm=np.empty(
            (batch, 0, *values.raw_delta_logcpm.shape[2:]),
            dtype=values.raw_delta_logcpm.dtype,
        ),
        confidence=np.empty(
            (batch, 0, *values.confidence.shape[2:]), dtype=values.confidence.dtype
        ),
        feature_mask=np.empty(
            (batch, 0, *values.feature_mask.shape[2:]), dtype=bool
        ),
        lineage_mask=np.empty(
            (batch, 0, *values.lineage_mask.shape[2:]), dtype=bool
        ),
        support_mask=np.empty((batch, 0), dtype=bool),
        support_phase=np.empty((batch, 0), dtype=np.int64),
        support_dt=np.empty((batch, 0), dtype=np.float32),
    )

def _reverse_lineage_features(
    values: LineageGeneFeatureBatch,
    preprocessor: LineageGenePreprocessor,
) -> LineageGeneFeatureBatch:
    expected = values.raw_delta_logcpm.shape[2:]
    if (
        expected != preprocessor.delta_mean.shape
        or expected != preprocessor.delta_scale.shape
    ):
        raise ValueError("lineage reverse transform differs from fitted preprocessor")
    raw_delta = -values.raw_delta_logcpm.copy()
    delta = (
        raw_delta
        - preprocessor.delta_mean.reshape(1, 1, *preprocessor.delta_mean.shape)
    ) / preprocessor.delta_scale.reshape(1, 1, *preprocessor.delta_scale.shape)
    raw_delta[~values.feature_mask] = 0.0
    delta[~values.feature_mask] = 0.0
    return replace(
        values,
        delta=delta.astype(values.delta.dtype, copy=False),
        raw_delta_logcpm=raw_delta.astype(values.raw_delta_logcpm.dtype, copy=False),
    )

def _stack_inputs_v2(
    values: Sequence[UnitInputs],
    device: torch.device,
    *,
    clinical_mode: str,
    donor_index: np.ndarray | None = None,
    lineage_features: LineageGeneFeatureBatch | None = None,
) -> dict[str, torch.Tensor]:
    if not values:
        raise ValueError("cannot stack an empty unit batch")
    if clinical_mode not in CLINICAL_MODES:
        raise ValueError(f"unknown clinical mode: {clinical_mode}")

    def stack(name: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        array = np.stack([getattr(item, name) for item in values])
        return torch.as_tensor(array, device=device, dtype=dtype)

    support_width = max(int(item.support_mask.shape[0]) for item in values)

    def stack_support(
        name: str,
        *,
        dtype: torch.dtype | None = None,
        optional_dtype: type[np.generic] | type[bool] | None = None,
    ) -> torch.Tensor:
        arrays: list[np.ndarray] = []
        trailing_shape: tuple[int, ...] | None = None
        for item in values:
            count = int(item.support_mask.shape[0])
            if optional_dtype is None:
                array = np.asarray(getattr(item, name))
            else:
                array = np.asarray(
                    getattr(
                        item,
                        name,
                        np.empty((count, 0), dtype=optional_dtype),
                    )
                )
                # UnitInputs keeps a schema-level (0, 0) default when optional
                # program features were not materialized.  Expand only its
                # support axis so the absent feature dimension remains empty.
                if count and array.shape == (0, 0):
                    array = np.empty((count, 0), dtype=optional_dtype)
            if array.ndim < 1 or array.shape[0] != count:
                raise ValueError(
                    f"support field {name} does not align with support_mask"
                )
            observed_shape = tuple(array.shape[1:])
            if trailing_shape is None:
                trailing_shape = observed_shape
            elif observed_shape != trailing_shape:
                raise ValueError(
                    f"support field {name} has inconsistent trailing shapes"
                )
            arrays.append(array)
        assert trailing_shape is not None
        padded = np.zeros(
            (len(values), support_width, *trailing_shape), dtype=arrays[0].dtype
        )
        for index, array in enumerate(arrays):
            padded[index, : array.shape[0]] = array
        array = padded
        return torch.as_tensor(array, device=device, dtype=dtype)

    baseline_value = stack_support("clinical_baseline_value")
    endpoint_value = stack_support("clinical_endpoint_value")
    delta_value = stack_support("clinical_delta_value")
    baseline_missing = stack_support(
        "clinical_baseline_missing", dtype=torch.bool
    )
    endpoint_missing = stack_support(
        "clinical_endpoint_missing", dtype=torch.bool
    )
    delta_valid = stack_support("clinical_delta_valid", dtype=torch.bool)
    if clinical_mode == "off":
        clinical_change = torch.empty(
            (len(values), support_width, 0), device=device, dtype=torch.float32
        )
    elif clinical_mode == "delta":
        clinical_change = torch.cat((delta_value, delta_valid.to(delta_value.dtype)), dim=-1)
    else:
        baseline_valid = (~baseline_missing).to(baseline_value.dtype)
        clinical_change = torch.cat(
            (baseline_value, baseline_valid, delta_value, delta_valid.to(delta_value.dtype)),
            dim=-1,
        )
    if donor_index is None:
        donor_index = np.arange(len(values), dtype=np.int64)
    result = {
        "query_uce": stack("query_uce"),
        "source_latent": stack("source_latent"),
        "phase": torch.as_tensor(
            [item.unit.phase for item in values], device=device, dtype=torch.long
        ),
        "support_source_uce": stack_support("support_source_uce"),
        "support_target_uce": stack_support("support_target_uce"),
        "support_phase": stack_support("support_phase", dtype=torch.long),
        "support_dt": stack_support("support_dt"),
        "support_mask": stack_support("support_mask", dtype=torch.bool),
        "support_source_latent_mean": stack_support("support_source_latent_mean"),
        "support_target_latent_mean": stack_support("support_target_latent_mean"),
        "support_response_latent": stack_support("support_response_latent"),
        "clinical_baseline_value": baseline_value,
        "clinical_endpoint_value": endpoint_value,
        "clinical_delta_value": delta_value,
        "clinical_baseline_missing": baseline_missing,
        "clinical_endpoint_missing": endpoint_missing,
        "clinical_delta_valid": delta_valid,
        "clinical_change": clinical_change,
        "gene_change": stack_support("gene_change"),
        "gene_confidence": stack_support("gene_confidence"),
        "program_delta": stack_support(
            "program_delta", optional_dtype=np.float32
        ),
        "program_valid": stack_support(
            "program_valid", dtype=torch.bool, optional_dtype=bool
        ),
        "donor_index": torch.as_tensor(donor_index, device=device, dtype=torch.long),
    }
    if lineage_features is not None:
        expected_ids = tuple(item.unit.unit_id for item in values)
        if lineage_features.unit_ids != expected_ids:
            raise ValueError("lineage gene feature unit order differs from batch")
        if lineage_features.delta.shape[:2] != result["support_mask"].shape:
            raise ValueError("lineage gene support shape differs from UCE history")
        if not np.array_equal(
            lineage_features.support_mask, result["support_mask"].cpu().numpy()
        ):
            raise ValueError("lineage gene support mask differs from UCE history")
        if not np.array_equal(
            lineage_features.support_phase, result["support_phase"].cpu().numpy()
        ):
            raise ValueError("lineage gene support phase differs from UCE history")
        result.update(
            {
                "lineage_gene_delta": torch.as_tensor(
                    lineage_features.delta, device=device
                ),
                "lineage_gene_confidence": torch.as_tensor(
                    lineage_features.confidence, device=device
                ),
                "lineage_gene_mask": torch.as_tensor(
                    lineage_features.feature_mask, device=device, dtype=torch.bool
                ),
            }
        )
    return result

def _forward_v2(
    model: torch.nn.Module,
    arm: TwinArm,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    parameters = inspect.signature(model.forward).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    arguments: dict[str, torch.Tensor | str] = {
        "arm": arm,
        "query_uce": batch["query_uce"],
        "source_latent": batch["source_latent"],
        "phase": batch["phase"],
        "support_source_uce": batch["support_source_uce"],
        "support_target_uce": batch["support_target_uce"],
        "support_phase": batch["support_phase"],
        "support_dt": batch["support_dt"],
        "support_mask": batch["support_mask"],
        "clinical_change": batch["clinical_change"],
        "gene_change": batch["gene_change"],
        "gene_confidence": batch["gene_confidence"],
    }
    optional = (
        "support_source_latent_mean",
        "support_target_latent_mean",
        "support_response_latent",
        "clinical_baseline_value",
        "clinical_endpoint_value",
        "clinical_delta_value",
        "clinical_baseline_missing",
        "clinical_endpoint_missing",
        "clinical_delta_valid",
        "lineage_gene_delta",
        "lineage_gene_confidence",
        "lineage_gene_mask",
        "program_delta",
        "program_valid",
    )
    for name in optional:
        if name in batch and (accepts_kwargs or name in parameters):
            arguments[name] = batch[name]
    return model(**arguments)  # type: ignore[arg-type]

def _materialize_inputs_v2(
    materializer: UnitMaterializer,
    units: Sequence[VaccineUnit],
    *,
    cells: int,
    draw_key: object,
    arm: TwinArm,
    lineage_store: LineageGeneSummaryStore | None = None,
    lineage_preprocessor: LineageGenePreprocessor | None = None,
) -> tuple[list[UnitInputs], LineageGeneFeatureBatch | None]:
    values = [
        materializer.inputs(
            unit,
            cells=cells,
            draw_key=draw_key,
            include_history=arm != "current",
        )
        for unit in units
    ]
    if (lineage_store is None) != (lineage_preprocessor is None):
        raise ValueError("lineage store and preprocessor must be supplied together")
    lineage_features = (
        materialize_lineage_gene_features(
            store=lineage_store,
            preprocessor=lineage_preprocessor,
            units=units,
        )
        if lineage_store is not None and lineage_preprocessor is not None
        else None
    )
    return values, lineage_features

def _materialize_batch_v2(
    materializer: UnitMaterializer,
    units: Sequence[VaccineUnit],
    *,
    cells: int,
    draw_key: object,
    device: torch.device,
    arm: TwinArm,
    clinical_mode: str,
    donor_index: np.ndarray,
    lineage_store: LineageGeneSummaryStore | None = None,
    lineage_preprocessor: LineageGenePreprocessor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    values, lineage_features = _materialize_inputs_v2(
        materializer,
        units,
        cells=cells,
        draw_key=draw_key,
        arm=arm,
        lineage_store=lineage_store,
        lineage_preprocessor=lineage_preprocessor,
    )
    targets = [
        materializer.target(unit, cells=cells, draw_key=draw_key)[0] for unit in units
    ]
    return (
        _stack_inputs_v2(
            values,
            device,
            clinical_mode=clinical_mode,
            donor_index=donor_index,
            lineage_features=lineage_features,
        ),
        torch.as_tensor(np.stack(targets), device=device),
    )

def patient_vaccine_objective_v2_scaled(
    predicted: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    donor_ids: Sequence[object],
    bandwidth: float,
    config: PatientLossV2Config,
    response_scale: torch.Tensor,
    shuffled_history_predicted: torch.Tensor | None = None,
    history_mask: torch.Tensor | None = None,
    current_response: torch.Tensor | None = None,
    history_current_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Use the native scaled loss when available."""

    supports_scale = "response_scale" in inspect.signature(
        patient_vaccine_objective_v2
    ).parameters
    arguments: dict[str, object] = {
        "donor_ids": donor_ids,
        "bandwidth": bandwidth,
        "config": config,
    }
    if supports_scale:
        arguments["response_scale"] = response_scale
    signature = inspect.signature(patient_vaccine_objective_v2).parameters
    if shuffled_history_predicted is not None:
        if "shuffled_history_predicted" not in signature:
            raise RuntimeError("v2 loss lacks shuffled-history support")
        arguments["shuffled_history_predicted"] = shuffled_history_predicted
        arguments["history_mask"] = history_mask
    if current_response is not None:
        if "current_response" not in signature:
            raise RuntimeError("v2 loss lacks current-anchor support")
        arguments["current_response"] = current_response
        arguments["history_current_mask"] = history_current_mask
    result = patient_vaccine_objective_v2(
        predicted,
        source,
        target,
        **arguments,  # type: ignore[arg-type]
    )
    predicted_response, observed_response = response_vectors(predicted, source, target)
    scale = torch.as_tensor(
        response_scale,
        device=predicted_response.device,
        dtype=predicted_response.dtype,
    )
    if scale.shape != (predicted_response.shape[-1],) or not torch.all(scale > 0):
        raise ValueError("response scale must be a positive latent-channel vector")
    raw_per_unit = (predicted_response - observed_response).square().mean(dim=-1)
    standardized_per_unit = ((predicted_response - observed_response) / scale).square().mean(
        dim=-1
    )
    standardized = donor_macro_mean(standardized_per_unit, donor_ids)
    raw = donor_macro_mean(raw_per_unit, donor_ids)
    if not supports_scale:
        result["loss"] = result["loss"] + config.response_mean_weight * (
            standardized - result["response_mean_loss"]
        )
        result["response_mean_loss"] = standardized
    result["raw_response_mean_loss"] = raw
    result["standardized_response_mean_loss"] = standardized
    if not torch.isfinite(result["loss"]):
        raise FloatingPointError("scaled v2 patient objective is non-finite")
    return result

def raw_donor_response_metrics(
    predicted_response: np.ndarray,
    observed_response: np.ndarray,
    donor_ids: Sequence[object],
    *,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    predicted = np.asarray(predicted_response, dtype=np.float64)
    observed = np.asarray(observed_response, dtype=np.float64)
    donors = np.asarray([str(value) for value in donor_ids], dtype=str)
    if predicted.shape != observed.shape or predicted.ndim != 2:
        raise ValueError("response arrays must share [units,latent]")
    if len(donors) != len(predicted) or len(donors) == 0:
        raise ValueError("donor ids must align to nonempty response arrays")
    order = sorted(set(donors.tolist()))
    predicted_donor = np.stack([predicted[donors == donor].mean(axis=0) for donor in order])
    observed_donor = np.stack([observed[donors == donor].mean(axis=0) for donor in order])
    residual = predicted_donor - observed_donor
    rmse = np.sqrt(np.mean(np.square(residual), axis=1)).mean()
    mae = np.mean(np.abs(residual), axis=1).mean()
    observed_norm = np.linalg.norm(observed_donor, axis=1)
    predicted_norm = np.linalg.norm(predicted_donor, axis=1)
    valid = observed_norm > epsilon
    cosine = np.zeros(len(order), dtype=np.float64)
    cosine[valid] = np.sum(
        predicted_donor[valid] * observed_donor[valid], axis=1
    ) / np.maximum(predicted_norm[valid] * observed_norm[valid], epsilon)
    predicted_centered = predicted_donor - predicted_donor.mean(axis=0, keepdims=True)
    observed_centered = observed_donor - observed_donor.mean(axis=0, keepdims=True)
    predicted_dispersion = float(np.sqrt(np.mean(np.square(predicted_centered))))
    observed_dispersion = float(np.sqrt(np.mean(np.square(observed_centered))))
    return {
        "raw_donor_delta_rmse": float(rmse),
        "raw_donor_delta_mae": float(mae),
        "raw_donor_response_cosine": float(cosine[valid].mean()) if valid.any() else 0.0,
        "response_dispersion_ratio": float(
            (predicted_dispersion + epsilon) / (observed_dispersion + epsilon)
        ),
        "response_norm_ratio": float(
            (predicted_norm.mean() + epsilon) / (observed_norm.mean() + epsilon)
        ),
        "donors": float(len(order)),
        "units": float(len(donors)),
    }

def _weighted_scalar_average(
    records: Sequence[tuple[int, Mapping[str, float]]]
) -> dict[str, float]:
    if not records:
        raise ValueError("epoch produced no donor batches")
    total = sum(weight for weight, _ in records)
    names = set.intersection(*(set(record) for _, record in records))
    return {
        name: float(sum(weight * record[name] for weight, record in records) / total)
        for name in sorted(names)
    }

_STACKED_HISTORY_FIELDS = (
    "support_source_uce",
    "support_target_uce",
    "support_phase",
    "support_dt",
    "support_mask",
    "support_source_latent_mean",
    "support_target_latent_mean",
    "support_response_latent",
    "clinical_baseline_value",
    "clinical_endpoint_value",
    "clinical_delta_value",
    "clinical_baseline_missing",
    "clinical_endpoint_missing",
    "clinical_delta_valid",
    "clinical_change",
    "gene_change",
    "gene_confidence",
    "program_delta",
    "program_valid",
    "lineage_gene_delta",
    "lineage_gene_confidence",
    "lineage_gene_mask",
)

def _donor_block_partner_indices(
    donor_ids: Sequence[str],
    seasons: Sequence[str],
    *,
    seed: int,
    support_counts: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic no-self shuffle preserving history availability."""

    donors = tuple(str(value) for value in donor_ids)
    season_values = tuple(str(value) for value in seasons)
    if len(donors) != len(season_values) or not donors:
        raise ValueError("donor ids and seasons must align to a nonempty batch")
    counts = (
        tuple(int(value) for value in support_counts)
        if support_counts is not None
        else tuple(0 for _ in donors)
    )
    if len(counts) != len(donors):
        raise ValueError("support counts must align to donor ids")
    if len(set(donors)) < 2:
        return np.arange(len(donors), dtype=np.int64), np.zeros(len(donors), dtype=bool)
    indices: list[int] = []
    valid: list[bool] = []
    for query_index, (donor, season, count) in enumerate(
        zip(donors, season_values, counts)
    ):
        order = sorted(
            {
                candidate_donor
                for candidate_donor, candidate_count in zip(donors, counts)
                if candidate_count == count
            }
        )
        if len(order) < 2:
            indices.append(query_index)
            valid.append(False)
            continue
        offset = 1 + stable_seed(
            "donor_block_history_offset", count, base_seed=seed
        ) % (len(order) - 1)
        partner_donor = order[(order.index(donor) + offset) % len(order)]
        candidates = [
            index
            for index, (value, candidate_count) in enumerate(zip(donors, counts))
            if value == partner_donor and candidate_count == count
        ]
        matched = [index for index in candidates if season_values[index] == season]
        pool = matched or candidates
        choice = stable_seed(
            "donor_block_history_partner",
            query_index,
            donor,
            season,
            partner_donor,
            base_seed=seed,
        ) % len(pool)
        selected = pool[choice]
        if donors[selected] == donor:
            raise RuntimeError("history shuffle assigned a donor to itself")
        indices.append(selected)
        valid.append(True)
    return np.asarray(indices, dtype=np.int64), np.asarray(valid, dtype=bool)

def _shuffle_stacked_history_batch(
    batch: Mapping[str, torch.Tensor],
    *,
    donor_ids: Sequence[str],
    seasons: Sequence[str],
    seed: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    counts = batch["support_mask"].sum(dim=1).detach().cpu().tolist()
    indices, mask = _donor_block_partner_indices(
        donor_ids, seasons, seed=seed, support_counts=counts
    )
    device = batch["support_mask"].device
    positions = torch.as_tensor(indices, device=device, dtype=torch.long)
    result = dict(batch)
    for name in _STACKED_HISTORY_FIELDS:
        if name in batch:
            result[name] = batch[name].index_select(0, positions)
    valid = torch.as_tensor(mask, device=device, dtype=torch.bool)
    valid = valid & batch["support_mask"].any(dim=1)
    valid = valid & result["support_mask"].any(dim=1)
    return result, valid

def _donor_block_history_shuffle(
    values: Sequence[UnitInputs],
    *,
    lineage_features: LineageGeneFeatureBatch | None,
    seed: int,
) -> tuple[list[UnitInputs], LineageGeneFeatureBatch | None, np.ndarray]:
    donors = [value.unit.donor_id for value in values]
    seasons = [value.unit.season for value in values]
    counts = [int(value.support_mask.sum()) for value in values]
    indices, mask = _donor_block_partner_indices(
        donors, seasons, seed=seed, support_counts=counts
    )
    shuffled: list[UnitInputs] = []
    for query, position in zip(values, indices):
        support = values[int(position)]
        if query.unit.donor_id == support.unit.donor_id:
            shuffled.append(query)
        else:
            shuffled.append(_copy_support_inputs(query, support))
    mask &= np.asarray(
        [
            bool(query.support_mask.any()) and bool(values[int(position)].support_mask.any())
            for query, position in zip(values, indices)
        ],
        dtype=bool,
    )
    shuffled_lineage = None
    if lineage_features is not None:
        shuffled_lineage = _slice_lineage_features(
            lineage_features,
            indices,
            unit_ids=[value.unit.unit_id for value in values],
        )
    return shuffled, shuffled_lineage, mask

def train_epoch_v2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    materializer: UnitMaterializer,
    units: Sequence[VaccineUnit],
    *,
    arm: TwinArm,
    epoch: int,
    runtime: RuntimeV2Config,
    loss_config: PatientLossV2Config,
    bandwidth: float,
    response_scale: np.ndarray,
    clinical_mode: str,
    device: torch.device,
    lineage_store: LineageGeneSummaryStore | None = None,
    lineage_preprocessor: LineageGenePreprocessor | None = None,
) -> dict[str, float]:
    model.train()
    residual_fit = (
        arm != "current" and runtime.history_training_mode == "residual_from_current"
    )
    if not residual_fit:
        set_encoder_frozen(
            model,
            runtime.freeze_pretrained_encoder
            or epoch <= runtime.warmup_frozen_epochs,
        )
    records: list[tuple[int, dict[str, float]]] = []
    batches = iter_donor_batches(
        units,
        batch_size_donors=runtime.batch_size,
        seed=stable_seed("epoch_order", epoch, base_seed=runtime.seed),
    )
    for batch_index, donor_batch in enumerate(batches):
        draw_key = ("train_v2", epoch, batch_index)
        values, lineage_features = _materialize_inputs_v2(
            materializer,
            donor_batch.units,
            cells=runtime.training_latent_cells,
            draw_key=draw_key,
            arm=arm,
            lineage_store=lineage_store,
            lineage_preprocessor=lineage_preprocessor,
        )
        batch = _stack_inputs_v2(
            values,
            device,
            clinical_mode=clinical_mode,
            donor_index=donor_batch.donor_index,
            lineage_features=lineage_features,
        )
        targets = [
            materializer.target(unit, cells=runtime.training_latent_cells, draw_key=draw_key)[0]
            for unit in donor_batch.units
        ]
        target = torch.as_tensor(np.stack(targets), device=device)
        donor_ids = [unit.donor_id for unit in donor_batch.units]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = _forward_v2(model, arm, batch)
            shuffled_prediction: torch.Tensor | None = None
            history_mask: torch.Tensor | None = None
            if loss_config.history_sensitivity_weight > 0:
                if not residual_fit:
                    raise ValueError(
                        "positive history_sensitivity_weight is restricted to "
                        "residual_from_current history training"
                    )
                shuffled_values, shuffled_lineage, shuffle_mask = (
                    _donor_block_history_shuffle(
                        values,
                        lineage_features=lineage_features,
                        seed=stable_seed(
                            "training_history_shuffle",
                            epoch,
                            batch_index,
                            base_seed=runtime.seed,
                        ),
                    )
                )
                shuffled_batch = _stack_inputs_v2(
                    shuffled_values,
                    device,
                    clinical_mode=clinical_mode,
                    donor_index=donor_batch.donor_index,
                    lineage_features=shuffled_lineage,
                )
                shuffled_prediction = _forward_v2(
                    model, arm, shuffled_batch
                )["predicted_cells"]
                history_mask = torch.as_tensor(
                    shuffle_mask, device=device, dtype=torch.bool
                )
        losses = patient_vaccine_objective_v2_scaled(
            output["predicted_cells"].float(),
            batch["source_latent"].float(),
            target.float(),
            donor_ids=donor_ids,
            bandwidth=bandwidth,
            config=loss_config,
            response_scale=torch.as_tensor(response_scale, device=device),
            shuffled_history_predicted=(
                shuffled_prediction.float()
                if shuffled_prediction is not None
                else None
            ),
            history_mask=history_mask,
            current_response=(
                output["current_response"].float()
                if loss_config.history_current_anchor_weight > 0
                else None
            ),
            history_current_mask=(
                batch["support_mask"].any(dim=1)
                if loss_config.history_current_anchor_weight > 0
                else None
            ),
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), runtime.gradient_clip_norm)
        optimizer.step()
        scalar = {
            key: float(value.detach().cpu())
            for key, value in losses.items()
            if value.ndim == 0
        }
        records.append((len(donor_batch.donor_ids), scalar))
    return _weighted_scalar_average(records)

def evaluate_donors_v2(
    model: torch.nn.Module,
    materializer: UnitMaterializer,
    units: Sequence[VaccineUnit],
    *,
    arm: TwinArm,
    runtime: RuntimeV2Config,
    loss_config: PatientLossV2Config,
    bandwidth: float,
    response_scale: np.ndarray,
    clinical_mode: str,
    device: torch.device,
    lineage_store: LineageGeneSummaryStore | None = None,
    lineage_preprocessor: LineageGenePreprocessor | None = None,
) -> dict[str, float]:
    model.eval()
    predicted_parts: list[torch.Tensor] = []
    source_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    shuffled_parts: list[torch.Tensor] = []
    history_mask_parts: list[torch.Tensor] = []
    current_response_parts: list[torch.Tensor] = []
    history_current_mask_parts: list[torch.Tensor] = []
    donor_ids: list[str] = []
    with torch.inference_mode():
        batches = iter_donor_batches(
            units, batch_size_donors=runtime.batch_size, seed=runtime.seed
        )
        for batch_index, donor_batch in enumerate(batches):
            batch, target = _materialize_batch_v2(
                materializer,
                donor_batch.units,
                cells=runtime.training_latent_cells,
                draw_key=("validation_v2", batch_index),
                device=device,
                arm=arm,
                clinical_mode=clinical_mode,
                donor_index=donor_batch.donor_index,
                lineage_store=lineage_store,
                lineage_preprocessor=lineage_preprocessor,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = _forward_v2(model, arm, batch)
                if loss_config.history_sensitivity_weight > 0:
                    if runtime.history_training_mode != "residual_from_current" or arm == "current":
                        raise ValueError(
                            "positive history_sensitivity_weight is restricted to "
                            "residual_from_current history training"
                        )
                    # Reuse the already materialized tensors.  Donor-level
                    # index remapping preserves every T2 season as one block.
                    shuffled_batch, shuffle_mask = _shuffle_stacked_history_batch(
                        batch,
                        donor_ids=[unit.donor_id for unit in donor_batch.units],
                        seasons=[unit.season for unit in donor_batch.units],
                        seed=stable_seed(
                            "validation_history_shuffle",
                            batch_index,
                            base_seed=runtime.seed,
                        ),
                    )
                    shuffled_parts.append(
                        _forward_v2(model, arm, shuffled_batch)[
                            "predicted_cells"
                        ].float()
                    )
                    history_mask_parts.append(shuffle_mask)
            predicted_parts.append(output["predicted_cells"].float())
            if loss_config.history_current_anchor_weight > 0:
                current_response_parts.append(output["current_response"].float())
                history_current_mask_parts.append(batch["support_mask"].any(dim=1))
            source_parts.append(batch["source_latent"].float())
            target_parts.append(target.float())
            donor_ids.extend(unit.donor_id for unit in donor_batch.units)
        predicted = torch.cat(predicted_parts)
        source = torch.cat(source_parts)
        target = torch.cat(target_parts)
        losses = patient_vaccine_objective_v2_scaled(
            predicted,
            source,
            target,
            donor_ids=donor_ids,
            bandwidth=bandwidth,
            config=loss_config,
            response_scale=torch.as_tensor(response_scale, device=device),
            shuffled_history_predicted=(
                torch.cat(shuffled_parts) if shuffled_parts else None
            ),
            history_mask=(
                torch.cat(history_mask_parts) if history_mask_parts else None
            ),
            current_response=(
                torch.cat(current_response_parts) if current_response_parts else None
            ),
            history_current_mask=(
                torch.cat(history_current_mask_parts)
                if history_current_mask_parts
                else None
            ),
        )
        predicted_response, observed_response = response_vectors(
            predicted, source, target
        )
    result = {
        key: float(value.detach().cpu())
        for key, value in losses.items()
        if value.ndim == 0
    }
    result.update(
        raw_donor_response_metrics(
            predicted_response.cpu().numpy(),
            observed_response.cpu().numpy(),
            donor_ids,
        )
    )
    result["patient_objective"] = result["loss"]
    return result

def _selection_score(metrics: Mapping[str, float], runtime: RuntimeV2Config) -> float:
    value = float(metrics[runtime.selection_objective])
    if not math.isfinite(value):
        return math.inf
    if runtime.selection_direction == "lower":
        return value
    if runtime.selection_direction == "higher":
        return -value
    return abs(math.log(max(value, 1e-12)))

def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }

def _restore_rng_state(payload: Mapping[str, object]) -> None:
    random.setstate(payload["python"])  # type: ignore[arg-type]
    np.random.set_state(payload["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(payload["torch_cpu"])  # type: ignore[arg-type]
    cuda = payload.get("torch_cuda", [])
    if torch.cuda.is_available() and cuda:
        torch.cuda.set_rng_state_all(cuda)  # type: ignore[arg-type]


def validate_current_checkpoint_v2(
    path: Path,
    *,
    task: str,
    outer_fold: int,
    required_phase: str | None = None,
) -> dict[str, Any]:
    """Require a current-state checkpoint for the same task and fold."""

    resolved = path.resolve(strict=True)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != V2_TRAIN_SCHEMA:
        raise ValueError("current checkpoint is not a vaccine v2 training checkpoint")
    expected = {
        "arm": "current",
        "paper_task": task,
        "outer_fold": int(outer_fold),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"current checkpoint {name} mismatch")
    phase = str(payload.get("phase", ""))
    if phase not in {"best_validation", "paper_model"}:
        raise ValueError("current checkpoint must be best_validation or paper_model")
    if required_phase is not None and phase != required_phase:
        raise ValueError(
            f"current checkpoint phase must be {required_phase}, got {phase}"
        )
    if not isinstance(payload.get("model_config"), Mapping):
        raise ValueError("current checkpoint lacks model_config")
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise ValueError("current checkpoint lacks model_state_dict")
    return payload

def initialize_history_from_current_v2(*args, **kwargs):
    raise ValueError("The retired latent/gene history trainer is not a paper entry point; use pipeline.sound.digital_twin")

def optimizer_for_v2(
    model: torch.nn.Module, runtime: RuntimeV2Config
) -> torch.optim.Optimizer:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("optimizer has no trainable model parameters")
    encoder_ids = {
        id(parameter)
        for parameter in model.visit_encoder.parameters()
        if parameter.requires_grad
    }
    encoder = [parameter for parameter in trainable if id(parameter) in encoder_ids]
    other = [parameter for parameter in trainable if id(parameter) not in encoder_ids]
    groups: list[dict[str, object]] = []
    if other:
        groups.append({"params": other, "lr": runtime.learning_rate})
    if encoder:
        groups.append(
            {
                "params": encoder,
                "lr": runtime.learning_rate
                * runtime.encoder_learning_rate_multiplier,
            }
        )
    return torch.optim.AdamW(groups, weight_decay=runtime.weight_decay)

def _checkpoint_payload_v2(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    arm: TwinArm,
    task: str,
    outer_fold: int,
    epoch: int,
    phase: str,
    history: Sequence[Mapping[str, object]],
    loaded_parameters: Sequence[str],
    preprocessing: Mapping[str, Any],
    response_transform: ResponseTransform,
    runtime: RuntimeV2Config,
    loss_config: PatientLossV2Config,
    best_epoch: int | None = None,
    best_score: float | None = None,
    initialization_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": V2_TRAIN_SCHEMA,
        "phase": phase,
        "arm": arm,
        "paper_method": V2_METHODS[arm],
        "paper_task": task,
        "outer_fold": int(outer_fold),
        "epoch": int(epoch),
        "model_config": model.config.to_dict(),
        "model_state_dict": _state_dict_cpu(model),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "cima_initialized_parameters": list(loaded_parameters),
        "history": list(history),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "preprocessing": dict(preprocessing),
        "response_transform": response_transform.to_json(),
        "runtime_config": asdict(runtime),
        "loss_config": loss_config.to_dict(),
        "residual_gene_embedding_pca": (
            dict(getattr(model, "_residual_gene_embedding_pca_audit"))
            if isinstance(
                getattr(model, "_residual_gene_embedding_pca_audit", None),
                Mapping,
            )
            else None
        ),
        "history_initialization": (
            dict(initialization_audit) if initialization_audit is not None else None
        ),
        "rng_state": _rng_state(),
    }

def select_epoch_v2(
    *,
    arm: TwinArm,
    task: str,
    outer_fold: int,
    fit_units: Sequence[VaccineUnit],
    validation_units: Sequence[VaccineUnit],
    materializer: UnitMaterializer,
    runtime: RuntimeV2Config,
    model_config: VaccineImmuneTwinConfig,
    loss_config: PatientLossV2Config,
    bandwidth: float,
    cima_checkpoint: Path,
    gene_selection: GeneSelection | None,
    response_transform: ResponseTransform,
    clinical_mode: str,
    output: Path,
    preprocessing: Mapping[str, Any],
    device: torch.device,
    resume: bool,
    config_path: Path | None = None,
    current_checkpoint: Path | None = None,
    lineage_store: LineageGeneSummaryStore | None = None,
    lineage_preprocessor: LineageGenePreprocessor | None = None,
    support_response_reference: PhaseSupportResponseReference | None = None,
) -> tuple[int, list[dict[str, object]], list[str]]:
    needs_support_reference = (
        arm != "current" and model_config.history_identifier == "residual_posterior"
    )
    if needs_support_reference != (support_response_reference is not None):
        raise ValueError(
            "residual_posterior history selection requires exactly one "
            "fit-scope support-response reference"
        )
    completed_path = output / "selection.json"
    best_path = output / "best_validation.pt"
    last_path = output / "selection_last.pt"
    if resume and completed_path.is_file() and best_path.is_file():
        completed = expand_paths(json.loads(completed_path.read_text(encoding="utf-8")))
        if completed.get("status") != "pass":
            raise RuntimeError("selection completion is not passing")
        audit = completed.get("history_initialization")
        if isinstance(audit, Mapping):
            audit_path = Path(str(audit.get("path", "")))
            if not audit_path.is_file():
                raise RuntimeError("selection current-initialization audit changed")
        return (
            int(completed["best_epoch"]),
            list(completed["history"]),
            list(completed["cima_initialized_parameters"]),
        )
    model, loaded = build_model(
        arm=arm,
        config=model_config,
        cima_checkpoint=cima_checkpoint,
        gene_selection=gene_selection,
        seed=runtime.seed,
    )
    set_model_response_transform(model, response_transform)
    initialization_audit: dict[str, Any] | None = None
    if arm != "current" and runtime.history_training_mode == "residual_from_current":
        if current_checkpoint is None or config_path is None:
            raise ValueError(
                "residual_from_current selection requires current checkpoint and config"
            )
        initialization_audit = initialize_history_from_current_v2(
            model,
            arm=arm,
            current_checkpoint=current_checkpoint,
            task=task,
            outer_fold=outer_fold,
            config_path=config_path,
            response_transform=response_transform,
            stage="selection",
            output=output,
        )
    model.to(device)
    if runtime.freeze_pretrained_encoder:
        set_encoder_frozen(model, True)
    optimizer = optimizer_for_v2(model, runtime)
    active_response_scale = model.transport.response_scale.detach().cpu().numpy()
    start_epoch = 1
    history: list[dict[str, object]] = []
    best_score = math.inf
    best_epoch = 0
    stale = 0
    if resume and last_path.is_file():
        prior = torch.load(last_path, map_location="cpu", weights_only=False)
        if prior.get("phase") != "selection":
            raise RuntimeError("selection resume checkpoint phase changed")
        model.load_state_dict(prior["model_state_dict"], strict=True)
        optimizer.load_state_dict(prior["optimizer_state_dict"])
        history = list(prior["history"])
        start_epoch = int(prior["epoch"]) + 1
        best_epoch = int(prior.get("best_epoch") or 0)
        observed_best_score = prior.get("best_score")
        best_score = (
            math.inf if observed_best_score is None else float(observed_best_score)
        )
        stale = int(prior.get("stale_epochs") or 0)
        _restore_rng_state(prior["rng_state"])
    if support_response_reference is not None:
        set_model_support_response_reference(model, support_response_reference)
    for epoch in range(start_epoch, runtime.maximum_epochs + 1):
        training = train_epoch_v2(
            model,
            optimizer,
            materializer,
            fit_units,
            arm=arm,
            epoch=epoch,
            runtime=runtime,
            loss_config=loss_config,
            bandwidth=bandwidth,
            response_scale=active_response_scale,
            clinical_mode=clinical_mode,
            device=device,
            lineage_store=lineage_store,
            lineage_preprocessor=lineage_preprocessor,
        )
        validation = evaluate_donors_v2(
            model,
            materializer,
            validation_units,
            arm=arm,
            runtime=runtime,
            loss_config=loss_config,
            bandwidth=bandwidth,
            response_scale=active_response_scale,
            clinical_mode=clinical_mode,
            device=device,
            lineage_store=lineage_store,
            lineage_preprocessor=lineage_preprocessor,
        )
        score = _selection_score(validation, runtime)
        row: dict[str, object] = {
            "epoch": epoch,
            "selection_objective": runtime.selection_objective,
            "selection_direction": runtime.selection_direction,
            "selection_score": score,
            **{f"train_{key}": value for key, value in training.items()},
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        if score < best_score - 1e-8:
            best_score = score
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                best_path,
                _checkpoint_payload_v2(
                    model=model,
                    optimizer=None,
                    arm=arm,
                    task=task,
                    outer_fold=outer_fold,
                    epoch=epoch,
                    phase="best_validation",
                    history=history,
                    loaded_parameters=loaded,
                    preprocessing=preprocessing,
                    response_transform=response_transform,
                    runtime=runtime,
                    loss_config=loss_config,
                    best_epoch=best_epoch,
                    best_score=best_score,
                    initialization_audit=initialization_audit,
                ),
            )
        else:
            stale += 1
        last_payload = _checkpoint_payload_v2(
            model=model,
            optimizer=optimizer,
            arm=arm,
            task=task,
            outer_fold=outer_fold,
            epoch=epoch,
            phase="selection",
            history=history,
            loaded_parameters=loaded,
            preprocessing=preprocessing,
            response_transform=response_transform,
            runtime=runtime,
            loss_config=loss_config,
            best_epoch=best_epoch,
            best_score=best_score,
            initialization_audit=initialization_audit,
        )
        last_payload["stale_epochs"] = stale
        atomic_torch_save(last_path, last_payload)
        atomic_table(output / "selection_history.tsv", pd.DataFrame(history))
        if stale >= runtime.patience:
            break
    if best_epoch < 1:
        raise RuntimeError("inner validation did not produce a finite selection epoch")
    completed = {
        "status": "pass",
        "schema_version": SELECTION_SCHEMA,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "objective": runtime.selection_objective,
        "direction": runtime.selection_direction,
        "history": history,
        "cima_initialized_parameters": loaded,
        "fit_donors": len({unit.donor_id for unit in fit_units}),
        "validation_donors": len({unit.donor_id for unit in validation_units}),
        "validation_units": len(validation_units),
        "validation_aggregation": "season within donor, then equal donor macro",
        "response_transform": response_transform.to_json(),
        "preprocessing": dict(preprocessing),
        "history_initialization": initialization_audit,
    }
    atomic_json(completed_path, completed)
    return best_epoch, history, loaded

def fit_final_v2(
    *,
    arm: TwinArm,
    task: str,
    outer_fold: int,
    train_units: Sequence[VaccineUnit],
    materializer: UnitMaterializer,
    epochs: int,
    runtime: RuntimeV2Config,
    model_config: VaccineImmuneTwinConfig,
    loss_config: PatientLossV2Config,
    bandwidth: float,
    cima_checkpoint: Path,
    gene_selection: GeneSelection | None,
    response_transform: ResponseTransform,
    clinical_mode: str,
    output: Path,
    preprocessing: Mapping[str, Any],
    device: torch.device,
    resume: bool,
    config_path: Path | None = None,
    current_checkpoint: Path | None = None,
    lineage_store: LineageGeneSummaryStore | None = None,
    lineage_preprocessor: LineageGenePreprocessor | None = None,
    support_response_reference: PhaseSupportResponseReference | None = None,
) -> tuple[torch.nn.Module, list[dict[str, float]], list[str]]:
    if epochs < 1:
        raise ValueError("final fit requires the selected positive epoch count")
    needs_support_reference = (
        arm != "current" and model_config.history_identifier == "residual_posterior"
    )
    if needs_support_reference != (support_response_reference is not None):
        raise ValueError(
            "residual_posterior history final fit requires exactly one "
            "fit-scope support-response reference"
        )
    model, loaded = build_model(
        arm=arm,
        config=model_config,
        cima_checkpoint=cima_checkpoint,
        gene_selection=gene_selection,
        seed=runtime.seed,
    )
    set_model_response_transform(model, response_transform)
    initialization_audit: dict[str, Any] | None = None
    if arm != "current" and runtime.history_training_mode == "residual_from_current":
        if current_checkpoint is None or config_path is None:
            raise ValueError(
                "residual_from_current final fit requires current checkpoint and config"
            )
        initialization_audit = initialize_history_from_current_v2(
            model,
            arm=arm,
            current_checkpoint=current_checkpoint,
            task=task,
            outer_fold=outer_fold,
            config_path=config_path,
            response_transform=response_transform,
            stage="final",
            output=output,
        )
    model.to(device)
    if runtime.freeze_pretrained_encoder:
        set_encoder_frozen(model, True)
    optimizer = optimizer_for_v2(model, runtime)
    active_response_scale = model.transport.response_scale.detach().cpu().numpy()
    last_path = output / "final_last.pt"
    start_epoch = 1
    history: list[dict[str, float]] = []
    if resume and last_path.is_file():
        prior = torch.load(last_path, map_location="cpu", weights_only=False)
        if prior.get("phase") != "final_fit":
            raise RuntimeError("final-fit resume checkpoint phase changed")
        model.load_state_dict(prior["model_state_dict"], strict=True)
        optimizer.load_state_dict(prior["optimizer_state_dict"])
        history = list(prior["history"])
        start_epoch = int(prior["epoch"]) + 1
        _restore_rng_state(prior["rng_state"])
    if support_response_reference is not None:
        set_model_support_response_reference(model, support_response_reference)
    for epoch in range(start_epoch, epochs + 1):
        values = train_epoch_v2(
            model,
            optimizer,
            materializer,
            train_units,
            arm=arm,
            epoch=epoch,
            runtime=runtime,
            loss_config=loss_config,
            bandwidth=bandwidth,
            response_scale=active_response_scale,
            clinical_mode=clinical_mode,
            device=device,
            lineage_store=lineage_store,
            lineage_preprocessor=lineage_preprocessor,
        )
        history.append({"epoch": float(epoch), **values})
        atomic_torch_save(
            last_path,
            _checkpoint_payload_v2(
                model=model,
                optimizer=optimizer,
                arm=arm,
                task=task,
                outer_fold=outer_fold,
                epoch=epoch,
                phase="final_fit",
                history=history,
                loaded_parameters=loaded,
                preprocessing=preprocessing,
                response_transform=response_transform,
                runtime=runtime,
                loss_config=loss_config,
                best_epoch=epochs,
                initialization_audit=initialization_audit,
            ),
        )
    checkpoint_path = output / "checkpoint.pt"
    atomic_torch_save(
        checkpoint_path,
        _checkpoint_payload_v2(
            model=model,
            optimizer=None,
            arm=arm,
            task=task,
            outer_fold=outer_fold,
            epoch=epochs,
            phase="paper_model",
            history=history,
            loaded_parameters=loaded,
            preprocessing=preprocessing,
            response_transform=response_transform,
            runtime=runtime,
            loss_config=loss_config,
            best_epoch=epochs,
            initialization_audit=initialization_audit,
        ),
    )
    return model, history, loaded


_SUPPORT_ARRAY_FIELDS = (
    "support_source_uce",
    "support_target_uce",
    "support_phase",
    "support_dt",
    "support_mask",
    "support_source_latent_mean",
    "support_target_latent_mean",
    "support_response_latent",
    "clinical_baseline_value",
    "clinical_endpoint_value",
    "clinical_delta_value",
    "clinical_baseline_missing",
    "clinical_endpoint_missing",
    "clinical_delta_valid",
    "clinical_change",
    "gene_change",
    "gene_confidence",
    "program_delta",
    "program_valid",
)

def _copy_support_inputs(query: UnitInputs, support: UnitInputs) -> UnitInputs:
    """Attach one donor's completed history to another query."""

    if query.unit.donor_id == support.unit.donor_id:
        raise ValueError("donor-shuffled support cannot come from the query donor")
    if any(
        query.unit.target_sample
        in (transition.source_sample, transition.target_sample)
        for transition in support.unit.supports
    ):
        raise ValueError("donor-shuffled support contains the query forecast target")
    updates = {
        name: np.asarray(getattr(support, name)).copy()
        for name in _SUPPORT_ARRAY_FIELDS
    }
    return replace(query, **updates)

def _empty_support_inputs(value: UnitInputs) -> UnitInputs:
    """Remove every support field while retaining the same query/source cells."""

    updates: dict[str, np.ndarray] = {}
    for name in _SUPPORT_ARRAY_FIELDS:
        array = np.asarray(getattr(value, name))
        if array.ndim < 1:
            raise ValueError(f"support field {name} has no support axis")
        updates[name] = np.empty((0, *array.shape[1:]), dtype=array.dtype)
    return replace(value, **updates)

def _reverse_support_inputs(value: UnitInputs) -> UnitInputs:
    """Reverse completed transitions.

    The two endpoints are swapped, so latent response, gene change and clinical
    delta change sign; the interval class is kept and the direction is recorded
    in the manifest.
    """

    return replace(
        value,
        support_source_uce=np.asarray(value.support_target_uce).copy(),
        support_target_uce=np.asarray(value.support_source_uce).copy(),
        support_phase=np.asarray(value.support_phase).copy(),
        support_dt=np.asarray(value.support_dt).copy(),
        support_mask=np.asarray(value.support_mask).copy(),
        support_source_latent_mean=np.asarray(
            value.support_target_latent_mean
        ).copy(),
        support_target_latent_mean=np.asarray(
            value.support_source_latent_mean
        ).copy(),
        support_response_latent=-np.asarray(value.support_response_latent).copy(),
        clinical_baseline_value=np.asarray(
            value.clinical_endpoint_value
        ).copy(),
        clinical_endpoint_value=np.asarray(
            value.clinical_baseline_value
        ).copy(),
        clinical_delta_value=-np.asarray(value.clinical_delta_value).copy(),
        clinical_baseline_missing=np.asarray(
            value.clinical_endpoint_missing
        ).copy(),
        clinical_endpoint_missing=np.asarray(
            value.clinical_baseline_missing
        ).copy(),
        clinical_delta_valid=np.asarray(value.clinical_delta_valid).copy(),
        # _stack_inputs_v2 builds the clinical tensor from the value and
        # missingness fields above; clinical_change keeps its validity bits.
        clinical_change=np.asarray(value.clinical_change).copy(),
        gene_change=-np.asarray(value.gene_change).copy(),
        gene_confidence=np.asarray(value.gene_confidence).copy(),
        program_delta=-np.asarray(value.program_delta).copy(),
        program_valid=np.asarray(value.program_valid).copy(),
    )

def _deterministic_donor_shuffle(
    values: Sequence[UnitInputs], *, seed: int
) -> tuple[dict[str, UnitInputs], pd.DataFrame]:
    """Create a deterministic donor-level derangement of histories."""

    if not values:
        raise ValueError("cannot shuffle an empty held-out unit set")
    identifiers = [value.unit.unit_id for value in values]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("held-out unit identifiers must be unique")
    tasks = {value.unit.task for value in values}
    folds = {value.unit.outer_fold for value in values}
    support_counts = {int(value.support_mask.shape[0]) for value in values}
    if len(tasks) != 1 or len(folds) != 1:
        raise ValueError("mechanism inference requires one task and one fold")
    if support_counts == {0}:
        raise ValueError("empty histories are exact aliases and cannot be shuffled")

    by_count_and_donor: dict[int, dict[str, list[UnitInputs]]] = {}
    for value in values:
        count = int(value.support_mask.shape[0])
        by_count_and_donor.setdefault(count, {}).setdefault(
            value.unit.donor_id, []
        ).append(value)
    for count, by_donor in sorted(by_count_and_donor.items()):
        if len(by_donor) < 2:
            raise ValueError(
                "donor-shuffled support requires at least two held-out donors "
                f"with support count {count}"
            )
    shuffle_seed = stable_seed(
        "v2_mechanism_donor_shuffle",
        next(iter(tasks)),
        next(iter(folds)),
        base_seed=seed,
    )
    assigned: dict[str, UnitInputs] = {}
    rows: list[dict[str, object]] = []
    for query in sorted(
        values, key=lambda item: (item.unit.donor_id, item.unit.season, item.unit.unit_id)
    ):
        support_count = int(query.support_mask.shape[0])
        by_donor = by_count_and_donor[support_count]
        donors = sorted(by_donor)
        offset = 1 + stable_seed(
            "v2_mechanism_donor_shuffle_offset",
            support_count,
            base_seed=shuffle_seed,
        ) % (len(donors) - 1)
        donor_partner = {
            donor: donors[(index + offset) % len(donors)]
            for index, donor in enumerate(donors)
        }
        partner_donor = donor_partner[query.unit.donor_id]
        candidates = sorted(
            by_donor[partner_donor],
            key=lambda item: (item.unit.season, item.unit.unit_id),
        )
        season_matched = [
            item for item in candidates if item.unit.season == query.unit.season
        ]
        pool = season_matched or candidates
        partner_index = stable_seed(
            "v2_mechanism_partner_unit",
            query.unit.unit_id,
            partner_donor,
            base_seed=seed,
        ) % len(pool)
        partner = pool[partner_index]
        assigned[query.unit.unit_id] = _copy_support_inputs(query, partner)
        rows.append(
            {
                "paper_task": query.unit.task,
                "outer_fold": query.unit.outer_fold,
                "query_unit_id": query.unit.unit_id,
                "query_donor_id": query.unit.donor_id,
                "query_season": query.unit.season,
                "support_unit_id": partner.unit.unit_id,
                "support_donor_id": partner.unit.donor_id,
                "support_season": partner.unit.season,
                "support_count": int(partner.support_mask.sum()),
                "shuffle_seed": int(shuffle_seed),
                "donor_rotation_offset": int(offset),
                "same_donor": False,
                "forecast_target_used": False,
            }
        )
    return assigned, pd.DataFrame(rows)


def _write_w_mechanism_artifacts(
    output: Path, rows: Sequence[Mapping[str, object]]
) -> tuple[Path, Path]:
    unit = pd.DataFrame(rows)
    unit_path = output / "response_coordinates_w.tsv"
    atomic_table(unit_path, unit)
    identity = ["paper_task", "outer_fold", "donor_id", "mechanism_variant"]
    w_columns = [column for column in unit if str(column).startswith("w_")]
    if unit.empty:
        patient = pd.DataFrame(columns=[*identity, "seasons", *w_columns])
    else:
        grouped = unit.groupby(identity, sort=True, dropna=False)
        patient = grouped[w_columns].mean().reset_index()
        seasons = grouped["season"].nunique().rename("seasons").reset_index()
        patient = patient.merge(seasons, on=identity, validate="one_to_one")
        patient = patient[[*identity, "seasons", *w_columns]]
    patient_path = output / "patient_response_coordinates_w.tsv"
    atomic_table(patient_path, patient)
    return unit_path, patient_path

def _write_gene_mechanism_artifacts(
    output: Path, rows: Sequence[Mapping[str, object]]
) -> dict[str, Path]:
    columns = [
        "unit_id",
        "paper_task",
        "donor_id",
        "season",
        "outer_fold",
        "mechanism_variant",
        "support_donor_id",
        "support_season",
        "gene_symbol",
        "gene_weight",
        "history_mean_signed_logcpm_change",
    ]
    unit = pd.DataFrame(rows, columns=columns)
    unit_path = output / "unit_gene_weights.tsv"
    atomic_table(unit_path, unit)
    patient_keys = [
        "paper_task",
        "outer_fold",
        "donor_id",
        "mechanism_variant",
        "gene_symbol",
    ]
    if unit.empty:
        patient = pd.DataFrame(
            columns=[
                *patient_keys,
                "gene_weight",
                "history_mean_signed_logcpm_change",
                "seasons",
            ]
        )
    else:
        grouped = unit.groupby(patient_keys, sort=True, dropna=False)
        patient = grouped[
            ["gene_weight", "history_mean_signed_logcpm_change"]
        ].mean().reset_index()
        seasons = grouped["season"].nunique().rename("seasons").reset_index()
        patient = patient.merge(seasons, on=patient_keys, validate="one_to_one")
    patient_path = output / "patient_gene_weights.tsv"
    atomic_table(patient_path, patient)

    summary_rows: list[dict[str, object]] = []
    summary_keys = ["paper_task", "outer_fold", "donor_id", "mechanism_variant"]
    if not patient.empty:
        own: dict[tuple[object, ...], np.ndarray] = {}
        for key, group in patient.groupby(summary_keys, sort=True, dropna=False):
            weights = group.sort_values("gene_symbol")["gene_weight"].to_numpy(
                dtype=np.float64
            )
            if str(key[-1]) == "own_history":
                own[key[:-1]] = weights
        for key, group in patient.groupby(summary_keys, sort=True, dropna=False):
            ordered = group.sort_values("gene_symbol")
            weights = ordered["gene_weight"].to_numpy(dtype=np.float64)
            positive = weights[weights > 0]
            entropy = float(-(positive * np.log(positive)).sum()) if len(positive) else 0.0
            reference = own.get(key[:-1])
            summary_rows.append(
                {
                    **dict(zip(summary_keys, key)),
                    "genes": int(len(weights)),
                    "gene_weight_sum": float(weights.sum()),
                    "maximum_gene_weight": float(weights.max()) if len(weights) else 0.0,
                    "gene_weight_entropy": entropy,
                    "effective_gene_count": float(np.exp(entropy)) if len(positive) else 0.0,
                    "within_patient_gene_weight_variance": float(weights.var())
                    if len(weights)
                    else 0.0,
                    "l1_distance_from_own_history": float(
                        np.abs(weights - reference).sum()
                    )
                    if reference is not None and reference.shape == weights.shape
                    else float("nan"),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary_path = output / "patient_gene_weight_summary.tsv"
    atomic_table(summary_path, summary)

    variance_rows: list[dict[str, object]] = []
    variance_keys = [
        "paper_task",
        "outer_fold",
        "mechanism_variant",
        "gene_symbol",
    ]
    if not patient.empty:
        for key, group in patient.groupby(variance_keys, sort=True, dropna=False):
            weights = group["gene_weight"].to_numpy(dtype=np.float64)
            variance_rows.append(
                {
                    **dict(zip(variance_keys, key)),
                    "donors": int(len(weights)),
                    "patient_mean_gene_weight": float(weights.mean()),
                    "patient_gene_weight_variance": float(weights.var()),
                    "patient_gene_weight_sd": float(weights.std()),
                    "patient_min_gene_weight": float(weights.min()),
                    "patient_max_gene_weight": float(weights.max()),
                }
            )
    variance_path = output / "patient_gene_weight_variance.tsv"
    atomic_table(variance_path, pd.DataFrame(variance_rows))
    return {
        "unit_gene_weights": unit_path,
        "patient_gene_weights": patient_path,
        "patient_gene_weight_summary": summary_path,
        "patient_gene_weight_variance": variance_path,
    }

def materialize_observed_targets_v2(
    *,
    output: Path,
    records: Sequence[Mapping[str, object]],
    units: Sequence[VaccineUnit],
    materializer: UnitMaterializer,
    runtime: RuntimeV2Config,
) -> pd.DataFrame:
    if not (output / "predictions.tsv").is_file():
        raise RuntimeError("held-out target materialization requires a prediction manifest")
    by_key = {(unit.donor_id, unit.season): unit for unit in units}
    observed_rows: list[dict[str, object]] = []
    for record in records:
        key = (str(record["donor_id"]), str(record["season"]))
        unit = by_key[key]
        target_seed = stable_seed(
            "bridge_test_cells", unit.task, unit.unit_id, "target", base_seed=runtime.seed
        )
        _, observed_raw, states, rows = materializer.target(
            unit,
            cells=runtime.prediction_latent_cells,
            draw_key="paper_observed_after_v2_seal",
            target_seed=target_seed,
        )
        observed_path = Path(str(record["observed_path"]))
        save_npz(
            observed_path,
            embedding=observed_raw.astype(np.float32),
            cell_state=states,
            source_row_index=rows,
            target_day=np.asarray(unit.target_day),
            role=np.asarray("observed_future"),
            donor_id=np.asarray(unit.donor_id),
            episode_id=np.asarray(unit.episode_id),
            paper_task=np.asarray(unit.task),
            cell_scope=np.asarray("all_cells"),
        )
        observed_rows.append(
            {
                "paper_task": unit.task,
                "donor_id": unit.donor_id,
                "season": unit.season,
                "outer_fold": unit.outer_fold,
                "observed_path": str(observed_path.resolve()),
                "observed_cells": int(len(observed_raw)),
            }
        )
    observed = pd.DataFrame(observed_rows)
    atomic_table(output / "observed_targets.tsv", observed)
    return observed

def predict_heldout_v2(
    *,
    model: torch.nn.Module,
    arm: TwinArm,
    units: Sequence[VaccineUnit],
    materializer: UnitMaterializer,
    runtime: RuntimeV2Config,
    clinical_mode: str,
    gene_selection: GeneSelection | None,
    output: Path,
    device: torch.device,
    lineage_store: LineageGeneSummaryStore | None = None,
    lineage_preprocessor: LineageGenePreprocessor | None = None,
    paper_method: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Write four predictions from the same checkpoint."""

    if not units:
        raise ValueError("held-out prediction units cannot be empty")
    if len({unit.task for unit in units}) != 1 or len(
        {unit.outer_fold for unit in units}
    ) != 1:
        raise ValueError("held-out mechanism inference requires one task and fold")
    checkpoint_path = output / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "same-checkpoint mechanism inference requires output/checkpoint.pt"
        )
    method_label = V2_METHODS[arm] if paper_method is None else str(paper_method).strip()
    if not method_label:
        raise ValueError("paper_method cannot be empty")
    model.eval()
    if (lineage_store is None) != (lineage_preprocessor is None):
        raise ValueError("lineage store and preprocessor must be supplied together")

    own_by_unit: dict[str, UnitInputs] = {}
    ordered_units = sorted(
        units, key=lambda item: (item.donor_id, item.season, item.unit_id)
    )
    for unit in ordered_units:
        source_seed = stable_seed(
            "bridge_test_cells",
            unit.task,
            unit.unit_id,
            "source",
            base_seed=runtime.seed,
        )
        own_by_unit[unit.unit_id] = materializer.inputs(
            unit,
            cells=runtime.prediction_latent_cells,
            draw_key="paper_prediction_v2",
            source_seed=source_seed,
            include_history=arm != "current",
        )
    all_lineage_features = (
        materialize_lineage_gene_features(
            store=lineage_store,
            preprocessor=lineage_preprocessor,
            units=ordered_units,
        )
        if lineage_store is not None and lineage_preprocessor is not None
        else None
    )
    lineage_position = {
        unit_id: index
        for index, unit_id in enumerate(all_lineage_features.unit_ids)
    } if all_lineage_features is not None else {}
    exact_aliases = arm == "current" or all(
        not bool(value.support_mask.any()) for value in own_by_unit.values()
    )
    if exact_aliases:
        shuffled_by_unit: dict[str, UnitInputs] = dict(own_by_unit)
        shuffle_rows = [
            {
                "paper_task": value.unit.task,
                "outer_fold": value.unit.outer_fold,
                "query_unit_id": value.unit.unit_id,
                "query_donor_id": value.unit.donor_id,
                "query_season": value.unit.season,
                "support_unit_id": value.unit.unit_id,
                "support_donor_id": value.unit.donor_id,
                "support_season": value.unit.season,
                "support_count": int(value.support_mask.sum()),
                "shuffle_seed": "",
                "donor_rotation_offset": 0,
                "same_donor": True,
                "forecast_target_used": False,
                "assignment_status": "not_applied_exact_current_alias",
            }
            for value in own_by_unit.values()
        ]
        shuffle_assignments = pd.DataFrame(shuffle_rows)
    else:
        shuffled_by_unit, shuffle_assignments = _deterministic_donor_shuffle(
            list(own_by_unit.values()), seed=runtime.seed
        )
        shuffle_assignments["assignment_status"] = "applied_donor_derangement"

    records: list[dict[str, object]] = []
    mechanism_rows: list[dict[str, object]] = []
    w_rows: list[dict[str, object]] = []
    gene_rows: list[dict[str, object]] = []
    shuffle_lookup = {
        str(row.query_unit_id): row
        for row in shuffle_assignments.itertuples(index=False)
    }

    with torch.inference_mode():
        batches = iter_donor_batches(
            units, batch_size_donors=runtime.batch_size, seed=runtime.seed
        )
        for donor_batch in batches:
            own_values = [own_by_unit[unit.unit_id] for unit in donor_batch.units]
            if exact_aliases:
                variant_values = {variant: own_values for variant in MECHANISM_VARIANTS}
            else:
                variant_values = {
                    "own_history": own_values,
                    "empty_support": [
                        _empty_support_inputs(value) for value in own_values
                    ],
                    "donor_shuffled_support": [
                        shuffled_by_unit[value.unit.unit_id] for value in own_values
                    ],
                    "reversed_support": [
                        _reverse_support_inputs(value) for value in own_values
                    ],
                }

            variant_lineage: dict[str, LineageGeneFeatureBatch | None] = {
                variant: None for variant in MECHANISM_VARIANTS
            }
            if all_lineage_features is not None:
                assert lineage_preprocessor is not None
                query_ids = [value.unit.unit_id for value in own_values]
                own_features = _slice_lineage_features(
                    all_lineage_features,
                    [lineage_position[value] for value in query_ids],
                    unit_ids=query_ids,
                )
                variant_lineage["own_history"] = own_features
                variant_lineage["empty_support"] = _empty_lineage_features(
                    own_features, unit_ids=query_ids
                )
                shuffled_positions = [
                    lineage_position[
                        str(shuffle_lookup[unit_id].support_unit_id)
                    ]
                    for unit_id in query_ids
                ]
                variant_lineage["donor_shuffled_support"] = (
                    _slice_lineage_features(
                        all_lineage_features,
                        shuffled_positions,
                        unit_ids=query_ids,
                    )
                )
                variant_lineage["reversed_support"] = _reverse_lineage_features(
                    own_features, lineage_preprocessor
                )

            variant_outputs: dict[str, dict[str, np.ndarray]] = {}
            variants_to_run = ("own_history",) if exact_aliases else MECHANISM_VARIANTS
            for variant in variants_to_run:
                values = variant_values[variant]
                batch = _stack_inputs_v2(
                    values,
                    device,
                    clinical_mode=clinical_mode,
                    donor_index=donor_batch.donor_index,
                    lineage_features=variant_lineage[variant],
                )
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with context:
                    result = _forward_v2(model, arm, batch)
                predicted_raw = materializer.latent.scale.inverse(
                    result["predicted_cells"].float().cpu().numpy()
                )
                w = result.get("w", torch.empty(len(values), 0, device=device))
                weights = result.get(
                    "gene_weights", torch.empty(len(values), 0, device=device)
                )
                variant_outputs[variant] = {
                    "predicted_raw": np.asarray(predicted_raw, dtype=np.float32),
                    "w": w.float().cpu().numpy(),
                    "gene_weights": weights.float().cpu().numpy(),
                    "mean_shift": result["mean_shift"].float().cpu().numpy(),
                }
            if exact_aliases:
                own_output = variant_outputs["own_history"]
                variant_outputs = {
                    variant: own_output for variant in MECHANISM_VARIANTS
                }

            for index, own_item in enumerate(own_values):
                unit = own_item.unit
                identity = f"{unit.donor_id}__{unit.season}"
                source_path = output / f"source_day{unit.source_day}" / f"{identity}.npz"
                own_path = output / "predicted" / f"{identity}.npz"
                observed_path = output / "observed" / f"{identity}.npz"
                common = {
                    "donor_id": np.asarray(unit.donor_id),
                    "episode_id": np.asarray(unit.episode_id),
                    "paper_task": np.asarray(unit.task),
                    "cell_scope": np.asarray("all_cells"),
                }
                save_npz(
                    source_path,
                    embedding=own_item.source_raw.astype(np.float32),
                    cell_state=own_item.source_states,
                    source_row_index=own_item.source_row_indices,
                    target_day=np.asarray(unit.source_day),
                    role=np.asarray(f"observed_day{unit.source_day}"),
                    **common,
                )
                source_record = {"path": str(source_path.resolve())}
                identity_row = {
                    "paper_task": unit.task,
                    "donor_id": unit.donor_id,
                    "season": unit.season,
                    "outer_fold": unit.outer_fold,
                }

                for variant in MECHANISM_VARIANTS:
                    variant_item = variant_values[variant][index]
                    output_values = variant_outputs[variant]
                    is_alias = exact_aliases and variant != "own_history"
                    predicted_path = (
                        own_path
                        if variant == "own_history" or is_alias
                        else output
                        / "mechanism_predictions"
                        / variant
                        / f"{identity}.npz"
                    )
                    if variant == "own_history" or not is_alias:
                        support_row = shuffle_lookup[unit.unit_id]
                        if variant == "donor_shuffled_support" and not exact_aliases:
                            support_donor = str(support_row.support_donor_id)
                            support_season = str(support_row.support_season)
                            support_unit_id = str(support_row.support_unit_id)
                        elif variant == "empty_support":
                            support_donor = ""
                            support_season = ""
                            support_unit_id = ""
                        else:
                            support_donor = unit.donor_id
                            support_season = unit.season
                            support_unit_id = unit.unit_id
                        save_npz(
                            predicted_path,
                            embedding=output_values["predicted_raw"][index],
                            cell_state=own_item.source_states,
                            source_row_index=own_item.source_row_indices,
                            source_day=np.asarray(unit.source_day),
                            target_day=np.asarray(unit.target_day),
                            standardized_mean_shift=output_values["mean_shift"][index],
                            method=np.asarray(method_label),
                            mechanism_variant=np.asarray(variant),
                            support_owner_donor_id=np.asarray(support_donor),
                            support_owner_season=np.asarray(support_season),
                            support_owner_unit_id=np.asarray(support_unit_id),
                            support_direction=np.asarray(
                                -1 if variant == "reversed_support" else 0
                                if variant == "empty_support"
                                else 1
                            ),
                            prediction_uses_test_future=np.asarray(False),
                            **common,
                        )
                    predicted_record = {"path": str(predicted_path.resolve())}
                    support_row = shuffle_lookup[unit.unit_id]
                    if variant == "donor_shuffled_support" and not exact_aliases:
                        support_donor = str(support_row.support_donor_id)
                        support_season = str(support_row.support_season)
                        support_unit_id = str(support_row.support_unit_id)
                    elif variant == "empty_support":
                        support_donor = ""
                        support_season = ""
                        support_unit_id = ""
                    else:
                        support_donor = unit.donor_id
                        support_season = unit.season
                        support_unit_id = unit.unit_id
                    support_direction = (
                        -1
                        if variant == "reversed_support"
                        else 0
                        if variant == "empty_support"
                        else 1
                    )
                    mechanism_rows.append(
                        {
                            "unit_id": unit.unit_id,
                            **identity_row,
                            "mechanism_variant": variant,
                            "source_path": source_record["path"],
                            "predicted_path": predicted_record["path"],
                            "checkpoint_path": str(checkpoint_path.resolve()),
                            "support_owner_unit_id": support_unit_id,
                            "support_owner_donor_id": support_donor,
                            "support_owner_season": support_season,
                            "support_count": int(variant_item.support_mask.sum()),
                            "support_direction": support_direction,
                            "support_phase_policy": "retained_interval_class",
                            "exact_alias_of_own_history": bool(is_alias),
                            "prediction_uses_test_future": False,
                        }
                    )
                    w_values = output_values["w"]
                    w_rows.append(
                        {
                            "unit_id": unit.unit_id,
                            **identity_row,
                            "mechanism_variant": variant,
                            "support_donor_id": support_donor,
                            "support_season": support_season,
                            **{
                                f"w_{column}": float(w_values[index, column])
                                for column in range(w_values.shape[1])
                            },
                        }
                    )
                    gene_weights = output_values["gene_weights"]
                    if gene_selection is not None and gene_weights.shape[1]:
                        if gene_weights.shape[1] != len(gene_selection.names):
                            raise RuntimeError(
                                "model gene-weight width differs from fold gene selection"
                            )
                        lineage_variant = variant_lineage[variant]
                        for gene_index, gene_symbol in enumerate(gene_selection.names):
                            if lineage_variant is not None:
                                raw = lineage_variant.raw_delta_logcpm[index, ..., gene_index]
                                valid = lineage_variant.feature_mask[index, ..., gene_index]
                                history_change = float(raw[valid].mean()) if valid.any() else 0.0
                            else:
                                history_change = (
                                    float(
                                        variant_item.gene_change[:, gene_index].mean()
                                    )
                                    if len(variant_item.gene_change)
                                    else 0.0
                                )
                            gene_rows.append(
                                {
                                    "unit_id": unit.unit_id,
                                    **identity_row,
                                    "mechanism_variant": variant,
                                    "support_donor_id": support_donor,
                                    "support_season": support_season,
                                    "gene_symbol": gene_symbol,
                                    "gene_weight": float(
                                        gene_weights[index, gene_index]
                                    ),
                                    "history_mean_signed_logcpm_change": history_change,
                                }
                            )

                own_record = next(
                    row
                    for row in reversed(mechanism_rows)
                    if row["unit_id"] == unit.unit_id
                    and row["mechanism_variant"] == "own_history"
                )
                records.append(
                    {
                        **identity_row,
                        "episode_id": unit.episode_id,
                        "cell_scope": "all_cells",
                        "source_day": unit.source_day,
                        "target_day": unit.target_day,
                        "method": method_label,
                        "source_path": source_record["path"],
                        "predicted_path": own_record["predicted_path"],
                        "observed_path": str(observed_path.resolve()),
                        "prediction_uses_test_future": False,
                        "task": "conditional",
                        "vaccinated": True,
                    }
                )

    mechanism = pd.DataFrame(mechanism_rows)
    expected_rows = len(units) * len(MECHANISM_VARIANTS)
    if len(mechanism) != expected_rows or mechanism.duplicated(
        ["unit_id", "mechanism_variant"]
    ).any():
        raise RuntimeError("mechanism prediction manifest coverage is incomplete")
    t1 = mechanism[mechanism["paper_task"] == "t1_y1_d0_to_y1_d7"]
    if not t1.empty:
        for _, rows in t1.groupby("unit_id", sort=False):
            if rows["predicted_path"].nunique() != 1:
                raise RuntimeError("T1 mechanism variants must be exact current aliases")

    mechanism_path = output / "mechanism_predictions_pre_target.tsv"
    atomic_table(mechanism_path, mechanism)
    variant_artifacts: dict[str, Path] = {}
    artifact_names = {
        "own_history": "own_history_prediction",
        "empty_support": "same_checkpoint_empty_support_prediction",
        "donor_shuffled_support": "shuffled_patient_support_prediction",
        "reversed_support": "reversed_transition_prediction",
    }
    for variant in MECHANISM_VARIANTS:
        path = output / "mechanism_manifests" / f"{variant}.tsv"
        atomic_table(
            path,
            mechanism.loc[mechanism["mechanism_variant"] == variant].reset_index(
                drop=True
            ),
        )
        variant_artifacts[artifact_names[variant]] = path

    shuffle_path = output / "donor_shuffle_assignments.tsv"
    atomic_table(shuffle_path, shuffle_assignments)
    w_path, patient_w_path = _write_w_mechanism_artifacts(output, w_rows)
    gene_artifacts: dict[str, Path] = {}
    if gene_selection is not None:
        gene_artifacts = _write_gene_mechanism_artifacts(output, gene_rows)

    mechanism_definition = {
        "status": "pass",
        "schema_version": MECHANISM_SCHEMA,
        "paper_method": method_label,
        "paper_task": str(units[0].task),
        "outer_fold": int(units[0].outer_fold),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "same_checkpoint_for_all_variants": True,
        "all_variants_generated_before_heldout_target_materialization": True,
        "heldout_target_values_materialized_before_mechanism_predictions": False,
        "variants": {
            "own_history": "the patient's own completed support transitions",
            "empty_support": "all support arrays removed; no history contribution",
            "donor_shuffled_support": (
                "deterministic donor-level cyclic derangement; a same-season unit "
                "is selected when available, otherwise a deterministic unit from "
                "the assigned donor; only completed support transitions are copied"
            ),
            "reversed_support": (
                "support source/target UCE and latent endpoints are swapped; latent "
                "response, gene change and clinical delta are negated; clinical "
                "endpoint values/missingness are swapped; confidence, validity, "
                "interval phase class and positive elapsed duration are retained; "
                "direction=-1 is explicit in the manifest because the model has no "
                "reverse-phase category"
            ),
        },
        "exact_alias_rule": (
            "current-arm and empty-history jobs reuse the exact own NPZ path "
            "for all four variants; T1 is therefore an exact current alias"
        ),
        "exact_aliases_used": bool(exact_aliases),
        "patient_artifact_aggregation": (
            "T2 seasons are averaged within donor before patient-level w and gene "
            "summaries; gene-weight variance is population variance (ddof=0) "
            "across held-out donors within task, fold, variant and gene"
        ),
        "donor_shuffle_assignments_path": str(shuffle_path.resolve()),
        "mechanism_manifest_path": str(mechanism_path.resolve()),
        "prediction_uses_test_future": False,
    }
    mechanism_definition_path = output / "mechanism_inference.json"
    atomic_json(mechanism_definition_path, mechanism_definition)

    atomic_table(output / "predictions.tsv", pd.DataFrame(records))
    materialize_observed_targets_v2(
        output=output,
        records=records,
        units=units,
        materializer=materializer,
        runtime=runtime,
    )
    manifest = pd.DataFrame(records)
    manifest_path = output / "predictions.tsv"
    atomic_table(manifest_path, manifest)
    artifacts = {
        "predictions": manifest_path,
        "observed_targets": output / "observed_targets.tsv",
        "response_coordinates_w": w_path,
        "patient_response_coordinates_w": patient_w_path,
        "mechanism_predictions": mechanism_path,
        "mechanism_inference": mechanism_definition_path,
        "donor_shuffle_assignments": shuffle_path,
        **variant_artifacts,
        **gene_artifacts,
    }
    return manifest, artifacts


def _resolve_protocol(
    args: argparse.Namespace, *, arm: str, task: str
) -> tuple[V2Protocol | None, V2JobRecord | None]:
    if args.selection_only:
        return None, None
    if args.protocol_manifest is None:
        raise ValueError("formal v2 final training requires a protocol manifest")
    protocol = load_v2_protocol(args.protocol_manifest, verify_files=True)
    identifier = f"{arm}__{task}__outer{args.fold}_seed{args.seed}"
    job = protocol.jobs.get(identifier)
    if job is None:
        raise RuntimeError(f"job is absent from v2 protocol: {identifier}")
    expected_output = (
        protocol.formal_output_root / arm / task / f"outer{args.fold}_seed42"
    ).resolve()
    if args.output.resolve() != expected_output:
        raise RuntimeError("requested output does not match the v2 job directory")
    return protocol, job

def _open_feature_stores(
    args: argparse.Namespace, arm: TwinArm, *, gene_prior_mode: str = "all_pbmc"
) -> tuple[Any, GeneSummaryStore | None, Path | None, dict[str, Any] | None, str]:
    if args.sealed_source_manifest is not None:
        if arm != "current" or args.sealed_uce is None:
            raise ValueError("sealed UCE is valid only for current and requires --sealed-uce")
        return (
            SealedBaseUCEStore(args.sealed_source_manifest, args.sealed_uce),
            None,
            None,
            None,
            "sealed_base321_current_queries",
        )
    if args.visit_manifest is None or args.visit_manifest_audit is None:
        raise ValueError("unified visit mode requires manifest and passing audit")
    audit = _validate_manifest_audit(args.visit_manifest, args.visit_manifest_audit)
    uce = UnifiedUCEStore(args.visit_manifest)
    gene_store = (
        GeneSummaryStore(args.visit_manifest)
        if arm in {"gene", "full"} and gene_prior_mode == "all_pbmc"
        else None
    )
    return uce, gene_store, args.visit_manifest, audit, "unified_base321_plus_delta168"

def _validate_lineage_summary_audit(summary: Path, audit_path: Path) -> dict[str, Any]:
    summary = summary.resolve(strict=True)
    audit_path = audit_path.resolve(strict=True)
    payload = expand_paths(json.loads(audit_path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict) or payload.get("status") != "pass":
        raise ValueError("lineage gene summary audit is not passing")
    output = _optional_mapping(payload.get("outputs")).get("lineage_gene_summary_h5")
    if not isinstance(output, Mapping):
        raise ValueError("lineage gene summary audit lacks output identity")
    if Path(str(output.get("path", ""))).resolve() != summary:
        raise ValueError("lineage gene summary audit path mismatch")
    return payload

def resolve_history_dependencies_v2(
    args: argparse.Namespace,
    *,
    arm: TwinArm,
    task: str,
    runtime: RuntimeV2Config,
    model_config: VaccineImmuneTwinConfig,
    loss_config: PatientLossV2Config = None,
) -> tuple[Path | None, Path | None]:
    """Resolve current-state model dependencies before loading CUDA or data."""

    current_run = getattr(args, "current_run", None)
    direct = getattr(args, "current_checkpoint", None)
    history = arm != "current"
    if not history:
        if current_run is not None or direct is not None:
            raise ValueError("current arm must not receive a current dependency")
        return None, None
    if runtime.history_training_mode == "joint":
        if current_run is not None or direct is not None:
            raise ValueError("joint ablation must not receive a current dependency")
    else:
        if (
            loss_config.history_sensitivity_weight < 0
            or loss_config.history_current_anchor_weight < 0
        ):
            raise ValueError("history ranking weights cannot be negative")
        if args.selection_only:
            if (current_run is None) == (direct is None):
                raise ValueError(
                    "residual_from_current selection requires exactly one of "
                    "--current-run or --current-checkpoint"
                )
            selection = (
                Path(current_run) / "best_validation.pt"
                if current_run is not None
                else Path(direct)
            )
            final = None
        else:
            if current_run is None or direct is not None:
                raise ValueError(
                    "formal residual_from_current training requires --current-run; "
                    "a single --current-checkpoint is selection-only"
                )
            selection = Path(current_run) / "best_validation.pt"
            final = Path(current_run) / "checkpoint.pt"
        validate_current_checkpoint_v2(
            selection,
            task=task,
            outer_fold=args.fold,
            required_phase="best_validation",
        )
        if final is not None:
            validate_current_checkpoint_v2(
                final,
                task=task,
                outer_fold=args.fold,
                required_phase="paper_model",
            )
        args.current_selection_checkpoint = selection.resolve(strict=True)
        args.current_final_checkpoint = (
            final.resolve(strict=True) if final is not None else None
        )

    if arm in {"gene", "full"}:
        if args.gene_embeddings is None:
            raise ValueError("gene/full arm requires --gene-embeddings")
        if model_config.gene_prior_mode == "lineage":
            if (
                args.lineage_gene_summary is None
                or args.lineage_gene_summary_audit is None
            ):
                raise ValueError(
                    "lineage gene mode requires --lineage-gene-summary and "
                    "--lineage-gene-summary-audit"
                )
            _validate_lineage_summary_audit(
                args.lineage_gene_summary, args.lineage_gene_summary_audit
            )
        elif (
            args.lineage_gene_summary is not None
            or args.lineage_gene_summary_audit is not None
        ):
            raise ValueError("all_pbmc gene mode must not receive lineage summary inputs")
    elif (
        args.lineage_gene_summary is not None
        or args.lineage_gene_summary_audit is not None
    ):
        raise ValueError("lineage gene summary is valid only for gene/full arms")
    if loss_config.history_sensitivity_weight > 0 and (
        runtime.history_training_mode != "residual_from_current" or not history
    ):
        raise ValueError(
            "positive history_sensitivity_weight requires a residual_from_current "
            "history arm"
        )
    if loss_config.history_current_anchor_weight > 0 and (
        runtime.history_training_mode != "residual_from_current" or not history
    ):
        raise ValueError(
            "positive history_current_anchor_weight requires a residual_from_current "
            "history arm"
        )
    return (
        getattr(args, "current_selection_checkpoint", None),
        getattr(args, "current_final_checkpoint", None),
    )

def _artifact_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve())}

def require_data1_output(path: Path) -> None:
    try:
        path.resolve().relative_to(ROOT)
    except ValueError as error:
        raise ValueError(
            "training outputs must remain below the repository"
        ) from error

def runtime_resource_snapshot(device: torch.device) -> dict[str, object]:
    gpu = device.type == "cuda"
    return {
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20)
        if gpu
        else 0.0,
        "gpu": torch.cuda.get_device_name(device) if gpu else "cpu_test_only",
        "process_peak_rss_mib": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
    }

def build_formal_run_payload(
    *,
    job: V2JobRecord,
    test_units: int,
    test_donors: int,
    artifacts: Mapping[str, Path],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required = set(FORMAL_MECHANISM_ARTIFACTS)
    if job.arm in {"gene", "full"}:
        required.update(FORMAL_GENE_MECHANISM_ARTIFACTS)
    missing = required - set(artifacts)
    if missing:
        raise ValueError(
            f"formal v2 run lacks mechanism artifacts: {sorted(missing)}"
        )
    payload: dict[str, Any] = {
        "status": "pass",
        "schema_version": V2_TRAIN_SCHEMA,
        "arm": job.arm,
        "paper_method": job.paper_method,
        "paper_task": job.task,
        "outer_fold": job.outer_fold,
        "seed": 42,
        "task_outer_fold_independent_model": True,
        "cells_treated_as_independent_n": False,
        "vectors_pooled_across_outer_folds": False,
        "heldout_target_used_for_training_or_adaptation": False,
        "test_units": int(test_units),
        "test_donors": int(test_donors),
        "artifacts": {name: _artifact_record(path) for name, path in artifacts.items()},
    }
    if extra:
        payload.update(extra)
    return payload

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(V2_METHODS), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("config/vaccine_immunetwin.yaml")
    )
    parser.add_argument("--task-units", type=Path, required=True)
    parser.add_argument("--inner-splits", type=Path, required=True)
    parser.add_argument("--training-latent", type=Path, required=True)
    parser.add_argument("--test-latent", type=Path)
    parser.add_argument("--scale", type=Path, required=True)
    parser.add_argument("--cima-checkpoint", type=Path)
    feature = parser.add_mutually_exclusive_group(required=True)
    feature.add_argument("--visit-manifest", type=Path)
    feature.add_argument("--sealed-source-manifest", type=Path)
    parser.add_argument("--visit-manifest-audit", type=Path)
    parser.add_argument("--sealed-uce", type=Path)
    parser.add_argument("--gene-embeddings", type=Path)
    parser.add_argument("--lineage-gene-summary", type=Path)
    parser.add_argument("--lineage-gene-summary-audit", type=Path)
    parser.add_argument(
        "--current-run",
        type=Path,
        help=(
            "current-arm output root; history selection uses best_validation.pt "
            "and formal refit uses checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--current-checkpoint",
        type=Path,
        help="direct current best_validation.pt for --selection-only history search",
    )
    parser.add_argument("--clinical-history", choices=("on", "off"), default=None)
    parser.add_argument("--labs-path", type=Path)
    parser.add_argument("--hai-path", type=Path)
    parser.add_argument("--igg-path", type=Path)
    parser.add_argument(
        "--history-contract",
        type=Path,
        default=Path("config/history_fields.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--protocol-manifest", type=Path)
    return parser

def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.max_epochs is not None or args.patience is not None:
        raise ValueError(
            "v2 entrypoints forbid --max-epochs/--patience overrides; "
            "use a content-addressed configuration file"
        )
    started = time.time()
    arm, task = validate_arm_task_v2(args.arm, args.task)
    payload, runtime, base_model_config, loss_config = load_configuration_v2(
        args.config, arm=arm, task=task
    )
    uses_phase_support_reference = (
        arm != "current"
        and base_model_config.history_identifier == "residual_posterior"
    )
    if args.seed != runtime.seed:
        raise ValueError("CLI seed must equal frozen config seed 42")
    clinical_requested = resolve_clinical_history_v2(arm, args.clinical_history)
    clinical_mode = resolve_clinical_mode(payload, arm=arm, enabled=clinical_requested)
    clinical_enabled = clinical_mode != "off"
    require_data1_output(args.output)
    if not args.selection_only and args.test_latent is None:
        raise ValueError("formal final training requires --test-latent")
    protocol, job = _resolve_protocol(args, arm=arm, task=task)
    current_selection_checkpoint, current_final_checkpoint = (
        resolve_history_dependencies_v2(
            args,
            arm=arm,
            task=task,
            runtime=runtime,
            model_config=base_model_config,
            loss_config=loss_config,
        )
    )
    device = require_cuda()
    torch.set_num_threads(4)
    set_seed(runtime.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output / ".run.lock").open("w")
    fcntl.flock(lock_handle, fcntl.LOCK_EX)
    complete_path = args.output / "_COMPLETE.json"
    if not args.selection_only and complete_path.exists() and not args.resume:
        lock_handle.close()
        raise FileExistsError(
            "completed formal v2 output already exists; use --resume for validation"
        )
    if args.resume and not args.selection_only and complete_path.exists():
        assert protocol is not None and job is not None
        validated = validate_completed_v2_run(
            args.output, protocol=protocol, job_identifier=job.identifier
        )
        print(json.dumps({"status": "resume", "run": str(validated.run_path)}, indent=2))
        lock_handle.close()
        return

    cima_checkpoint = (
        args.cima_checkpoint
        if args.cima_checkpoint is not None
        else Path(payload["pretraining"]["checkpoint"])
    ).resolve(strict=True)
    units = build_task_units(args.task_units, task)
    outer_train, outer_test = split_outer_units(units, args.fold)
    fit_units, validation_units = split_inner_units(
        outer_train, args.inner_splits, outer_fold=args.fold, validation_fold=0
    )
    if job is not None:
        test_donors = len({unit.donor_id for unit in outer_test})
        if len(outer_test) != job.expected_units or test_donors != job.expected_donors:
            raise RuntimeError("held-out coverage differs from frozen protocol job")
    scale = load_fold_scale(args.scale, outer_fold=args.fold)
    uce, gene_store, gene_manifest, manifest_audit, feature_mode = _open_feature_stores(
        args, arm, gene_prior_mode=base_model_config.gene_prior_mode
    )
    lineage_store = (
        LineageGeneSummaryStore(args.lineage_gene_summary)
        if arm in {"gene", "full"}
        and base_model_config.gene_prior_mode == "lineage"
        else None
    )
    training_latent = FoldLatentStore(args.training_latent, scale)
    test_latent: FoldLatentStore | None = None
    try:
        selection_clinical, selection_clinical_artifact = prepare_clinical_stage(
            stage="selection",
            enabled=clinical_enabled,
            fit_units=fit_units,
            transform_units=[*fit_units, *validation_units],
            excluded_donors={unit.donor_id for unit in validation_units},
            labs_path=args.labs_path,
            hai_path=args.hai_path,
            igg_path=args.igg_path,
            contract_path=args.history_contract,
            output=args.output,
        )
        selection_lineage_preprocessor: LineageGenePreprocessor | None = None
        if (
            arm in {"gene", "full"}
            and base_model_config.gene_prior_mode == "lineage"
        ):
            assert lineage_store is not None and args.gene_embeddings is not None
            (
                selection_genes,
                selection_lineage_preprocessor,
                selection_gene_artifact,
            ) = prepare_lineage_gene_stage(
                stage="selection",
                store=lineage_store,
                fit_units=fit_units,
                embedding_path=args.gene_embeddings,
                n_genes=runtime.gene_candidates_per_fold,
                output=args.output,
            )
        else:
            selection_genes, selection_gene_artifact = prepare_gene_stage(
                stage="selection",
                enabled=arm in {"gene", "full"},
                store=gene_store,
                manifest_path=gene_manifest,
                fit_units=fit_units,
                embedding_path=args.gene_embeddings,
                n_genes=runtime.gene_candidates_per_fold,
                output=args.output,
            )
        selection_response = estimate_response_transform(
            training_latent,
            fit_units,
            cells=runtime.training_latent_cells,
            seed=runtime.seed,
            stage="selection",
            scale_floor=runtime.response_scale_floor,
            mean_program_rank=(
                base_model_config.mean_program_rank
                if base_model_config.current_operator
                in {
                    "program_affine",
                    "program_query_affine",
                    "set_affine",
                    "moment_fiber",
                }
                else 0
            ),
            covariance_program_rank=(
                base_model_config.covariance_program_rank
                if base_model_config.current_operator
                in {
                    "program_affine",
                    "program_query_affine",
                    "set_affine",
                    "moment_fiber",
                }
                else 0
            ),
        )
        selection_response_artifact = save_response_transform(
            selection_response, args.output / "selection_response_transform.json"
        )
        selection_support_reference: PhaseSupportResponseReference | None = None
        selection_support_reference_artifact: dict[str, object] | None = None
        if uses_phase_support_reference:
            selection_support_reference = estimate_phase_support_response_reference(
                training_latent,
                fit_units,
                cells=runtime.training_latent_cells,
                seed=runtime.seed,
                stage="selection",
                phase_count=base_model_config.phase_count,
            )
            selection_support_reference_artifact = (
                save_phase_support_response_reference(
                    selection_support_reference,
                    args.output / "selection_support_response_reference.json",
                )
            )
        selection_preprocessing = {
            "clinical": selection_clinical_artifact,
            "gene": selection_gene_artifact,
            "response": selection_response_artifact,
        }
        if selection_support_reference_artifact is not None:
            selection_preprocessing["support_response_reference"] = (
                selection_support_reference_artifact
            )
        selection_config = replace(
            base_model_config,
            clinical_dim=_clinical_model_dim(selection_clinical, clinical_mode),
            history_response_dim=16 if arm != "current" else 0,
            # A current-state checkpoint holds the shared backbone without history
            # buffers.
            history_identifier=(
                base_model_config.history_identifier
                if arm != "current"
                else "legacy_transition"
            ),
        )
        selection_materializer = UnitMaterializer(
            uce=uce,
            latent=training_latent,
            query_uce_cells=runtime.query_uce_cells,
            support_uce_cells=runtime.support_uce_cells,
            gene_store=gene_store,
            gene_selection=(
                selection_genes
                if base_model_config.gene_prior_mode == "all_pbmc"
                else None
            ),
            clinical_features=selection_clinical,
            seed=runtime.seed,
        )
        selected_epoch, selection_history, loaded = select_epoch_v2(
            arm=arm,
            task=task,
            outer_fold=args.fold,
            fit_units=fit_units,
            validation_units=validation_units,
            materializer=selection_materializer,
            runtime=runtime,
            model_config=selection_config,
            loss_config=loss_config,
            bandwidth=scale.mmd_bandwidth,
            cima_checkpoint=cima_checkpoint,
            gene_selection=selection_genes,
            response_transform=selection_response,
            clinical_mode=clinical_mode,
            output=args.output,
            preprocessing=selection_preprocessing,
            device=device,
            resume=args.resume,
            config_path=args.config,
            current_checkpoint=current_selection_checkpoint,
            lineage_store=lineage_store,
            lineage_preprocessor=selection_lineage_preprocessor,
            support_response_reference=selection_support_reference,
        )
        if args.selection_only:
            selection_only_path = args.output / "selection_only.json"
            atomic_json(
                selection_only_path,
                {
                    "status": "pass",
                    "schema_version": SELECTION_SCHEMA,
                    "selection_only": True,
                    "arm": arm,
                    "paper_task": task,
                    "outer_fold": args.fold,
                    "seed": runtime.seed,
                    "best_epoch": selected_epoch,
                    "outer_test_target_opened": False,
                    "test_latent_opened": False,
                    "elapsed_seconds": time.time() - started,
                    **runtime_resource_snapshot(device),
                    "history_training_mode": runtime.history_training_mode,
                    "history_identifier": base_model_config.history_identifier,
                    "gene_prior_mode": base_model_config.gene_prior_mode,
                    "support_response_reference": (
                        selection_support_reference_artifact
                        if uses_phase_support_reference
                        else None
                    ),
                    "current_dependency": (
                        _artifact_record(current_selection_checkpoint)
                        if current_selection_checkpoint is not None
                        else None
                    ),
                },
            )
            print(json.dumps(expand_paths(json.loads(selection_only_path.read_text())), indent=2))
            lock_handle.close()
            return

        final_clinical, final_clinical_artifact = prepare_clinical_stage(
            stage="final",
            enabled=clinical_enabled,
            fit_units=outer_train,
            transform_units=[*outer_train, *outer_test],
            excluded_donors={unit.donor_id for unit in outer_test},
            labs_path=args.labs_path,
            hai_path=args.hai_path,
            igg_path=args.igg_path,
            contract_path=args.history_contract,
            output=args.output,
        )
        final_lineage_preprocessor: LineageGenePreprocessor | None = None
        if (
            arm in {"gene", "full"}
            and base_model_config.gene_prior_mode == "lineage"
        ):
            assert lineage_store is not None and args.gene_embeddings is not None
            (
                final_genes,
                final_lineage_preprocessor,
                final_gene_artifact,
            ) = prepare_lineage_gene_stage(
                stage="final",
                store=lineage_store,
                fit_units=outer_train,
                embedding_path=args.gene_embeddings,
                n_genes=runtime.gene_candidates_per_fold,
                output=args.output,
            )
        else:
            final_genes, final_gene_artifact = prepare_gene_stage(
                stage="final",
                enabled=arm in {"gene", "full"},
                store=gene_store,
                manifest_path=gene_manifest,
                fit_units=outer_train,
                embedding_path=args.gene_embeddings,
                n_genes=runtime.gene_candidates_per_fold,
                output=args.output,
            )
        final_response = estimate_response_transform(
            training_latent,
            outer_train,
            cells=runtime.training_latent_cells,
            seed=runtime.seed,
            stage="final",
            scale_floor=runtime.response_scale_floor,
            mean_program_rank=(
                base_model_config.mean_program_rank
                if base_model_config.current_operator
                in {
                    "program_affine",
                    "program_query_affine",
                    "set_affine",
                    "moment_fiber",
                }
                else 0
            ),
            covariance_program_rank=(
                base_model_config.covariance_program_rank
                if base_model_config.current_operator
                in {
                    "program_affine",
                    "program_query_affine",
                    "set_affine",
                    "moment_fiber",
                }
                else 0
            ),
        )
        final_response_artifact = save_response_transform(
            final_response, args.output / "final_response_transform.json"
        )
        final_support_reference: PhaseSupportResponseReference | None = None
        final_support_reference_artifact: dict[str, object] | None = None
        if uses_phase_support_reference:
            final_support_reference = estimate_phase_support_response_reference(
                training_latent,
                outer_train,
                cells=runtime.training_latent_cells,
                seed=runtime.seed,
                stage="final",
                phase_count=base_model_config.phase_count,
            )
            final_support_reference_artifact = save_phase_support_response_reference(
                final_support_reference,
                args.output / "final_support_response_reference.json",
            )
        final_preprocessing = {
            "clinical": final_clinical_artifact,
            "gene": final_gene_artifact,
            "response": final_response_artifact,
        }
        if final_support_reference_artifact is not None:
            final_preprocessing["support_response_reference"] = (
                final_support_reference_artifact
            )
        final_config = replace(
            base_model_config,
            clinical_dim=_clinical_model_dim(final_clinical, clinical_mode),
            history_response_dim=16 if arm != "current" else 0,
            history_identifier=(
                base_model_config.history_identifier
                if arm != "current"
                else "legacy_transition"
            ),
        )
        final_materializer = UnitMaterializer(
            uce=uce,
            latent=training_latent,
            query_uce_cells=runtime.query_uce_cells,
            support_uce_cells=runtime.support_uce_cells,
            gene_store=gene_store,
            gene_selection=(
                final_genes
                if base_model_config.gene_prior_mode == "all_pbmc"
                else None
            ),
            clinical_features=final_clinical,
            seed=runtime.seed,
        )
        model, final_history, loaded = fit_final_v2(
            arm=arm,
            task=task,
            outer_fold=args.fold,
            train_units=outer_train,
            materializer=final_materializer,
            epochs=selected_epoch,
            runtime=runtime,
            model_config=final_config,
            loss_config=loss_config,
            bandwidth=scale.mmd_bandwidth,
            cima_checkpoint=cima_checkpoint,
            gene_selection=final_genes,
            response_transform=final_response,
            clinical_mode=clinical_mode,
            output=args.output,
            preprocessing=final_preprocessing,
            device=device,
            resume=args.resume,
            config_path=args.config,
            current_checkpoint=current_final_checkpoint,
            lineage_store=lineage_store,
            lineage_preprocessor=final_lineage_preprocessor,
            support_response_reference=final_support_reference,
        )
        training_latent.close()
        test_latent = FoldLatentStore(args.test_latent, scale)
        test_materializer = UnitMaterializer(
            uce=uce,
            latent=test_latent,
            query_uce_cells=runtime.query_uce_cells,
            support_uce_cells=runtime.support_uce_cells,
            gene_store=gene_store,
            gene_selection=(
                final_genes
                if base_model_config.gene_prior_mode == "all_pbmc"
                else None
            ),
            clinical_features=final_clinical,
            seed=runtime.seed,
        )
        manifest, prediction_artifacts = predict_heldout_v2(
            model=model,
            arm=arm,
            units=outer_test,
            materializer=test_materializer,
            runtime=runtime,
            clinical_mode=clinical_mode,
            gene_selection=final_genes,
            output=args.output,
            device=device,
            lineage_store=lineage_store,
            lineage_preprocessor=final_lineage_preprocessor,
        )
    finally:
        try:
            training_latent.close()
        except Exception:
            pass
        if test_latent is not None:
            test_latent.close()
        uce.close()
        if gene_store is not None:
            gene_store.close()
        if lineage_store is not None:
            lineage_store.close()

    assert protocol is not None and job is not None
    checkpoint_path = args.output / "checkpoint.pt"
    formal_artifacts: dict[str, Path] = {
        "checkpoint": checkpoint_path,
        **prediction_artifacts,
        "selection": args.output / "selection.json",
        "best_validation_checkpoint": args.output / "best_validation.pt",
        "selection_history": args.output / "selection_history.tsv",
        "selection_clinical_preprocessor": args.output
        / "selection_clinical_preprocessor.json",
        "selection_gene_preprocessor": args.output / "selection_gene_preprocessor.json",
        "selection_response_transform": args.output
        / "selection_response_transform.json",
        "final_clinical_preprocessor": args.output / "final_clinical_preprocessor.json",
        "final_gene_preprocessor": args.output / "final_gene_preprocessor.json",
        "final_response_transform": args.output / "final_response_transform.json",
    }
    if clinical_enabled:
        formal_artifacts.update(
            {
                "selection_clinical_parameters": args.output
                / "selection_preprocessing"
                / "clinical_preprocessor.json",
                "final_clinical_parameters": args.output
                / "final_preprocessing"
                / "clinical_preprocessor.json",
            }
        )
    if final_genes is not None:
        formal_artifacts.update(
            {
                "selection_gene_selection": args.output
                / "selection_gene_selection.tsv",
                "final_gene_selection": args.output / "final_gene_selection.tsv",
            }
        )
    if runtime.history_training_mode == "residual_from_current" and arm != "current":
        formal_artifacts.update(
            {
                "selection_current_initialization": args.output
                / "selection_current_initialization.json",
                "final_current_initialization": args.output
                / "final_current_initialization.json",
            }
        )
    if uses_phase_support_reference:
        formal_artifacts.update(
            {
                "selection_support_response_reference": args.output
                / "selection_support_response_reference.json",
                "final_support_response_reference": args.output
                / "final_support_response_reference.json",
            }
        )
    if base_model_config.gene_prior_mode == "lineage" and final_genes is not None:
        formal_artifacts.update(
            {
                "selection_lineage_gene_parameters": args.output
                / "selection_lineage_gene_preprocessor.npz",
                "final_lineage_gene_parameters": args.output
                / "final_lineage_gene_preprocessor.npz",
            }
        )
    run_payload = build_formal_run_payload(
        job=job,
        test_units=len(outer_test),
        test_donors=len({unit.donor_id for unit in outer_test}),
        artifacts=formal_artifacts,
        extra={
            "feature_mode": feature_mode,
            "gene_prior_mode": base_model_config.gene_prior_mode,
            "history_training_mode": runtime.history_training_mode,
            "history_identifier": base_model_config.history_identifier,
            "current_model_dependency": (
                {
                    "selection_inner_fit_checkpoint": _artifact_record(
                        current_selection_checkpoint
                    ),
                    "final_outer_training_checkpoint": _artifact_record(
                        current_final_checkpoint
                    ),
                    "inner_selection_never_uses_outer_trained_current": True,
                }
                if current_selection_checkpoint is not None
                and current_final_checkpoint is not None
                else None
            ),
            "lineage_gene_summary_dependency": (
                {
                    "summary": _artifact_record(args.lineage_gene_summary),
                    "builder_audit": _artifact_record(
                        args.lineage_gene_summary_audit
                    ),
                }
                if args.lineage_gene_summary is not None
                else None
            ),
            "clinical_mode": clinical_mode,
            "clinical_information_used": clinical_enabled,
            "history_information_used": arm != "current",
            "donor_identifier_as_feature": False,
            "training_batch_unit": "distinct donors expanded to all task seasons",
            "inner_selection_preprocessing_fit_scope": "component_specific_with_outer_training_rna_coordinates",
            "inner_selection_preprocessing_components": {
                "rna_coordinates_center_scale": "outer_training_fitted_and_reused",
                "stage_specific_transforms": "see selection_preprocessing and selection_response_transform",
            },
            "final_preprocessing_fit_scope": "outer_training_donors_only",
            "response_residual_standardization": "channelwise outer-fit response scale",
            "training_donors": len({unit.donor_id for unit in outer_train}),
            "training_units": len(outer_train),
            "validation_donors": len({unit.donor_id for unit in validation_units}),
            "selected_epoch": selected_epoch,
            "selection_history": selection_history,
            "final_history": final_history,
            "runtime_config": asdict(runtime),
            "model_config": final_config.to_dict(),
            "loss_config": loss_config.to_dict(),
            "selection_preprocessing": selection_preprocessing,
            "final_preprocessing": final_preprocessing,
            "selection_response_transform": selection_response.to_json(),
            "final_response_transform": final_response.to_json(),
            "cima_initialized_parameters": loaded,
            "visit_manifest_audit": manifest_audit,
            "prediction_units": len(manifest),
            "own_predictions_only": False,
            "mechanism_inference_same_checkpoint": True,
            "mechanism_variants_generated_before_heldout_target_materialization": True,
            "mechanism_variants": list(MECHANISM_VARIANTS),
            "shuffle_reverse_status": (
                "pass: target-blind own, empty, donor-shuffled and "
                "reversed-support predictions"
            ),
            "elapsed_seconds": time.time() - started,
            **runtime_resource_snapshot(device),
            "torch": torch.__version__,
            "runner_source": str(Path(__file__).resolve()),
        },
    )
    run_path = args.output / "run.json"
    atomic_json(run_path, run_payload)
    atomic_json(
        args.output / "_COMPLETE.json",
        {
            "status": "pass",
            "schema_version": V2_COMPLETE_SCHEMA,
            "job_identifier": job.identifier,
            "run_json": str(run_path.resolve()),
        },
    )
    validate_completed_v2_run(
        args.output, protocol=protocol, job_identifier=job.identifier
    )
    print(json.dumps(run_payload, indent=2))
    lock_handle.close()

if __name__ == "__main__":
    main()

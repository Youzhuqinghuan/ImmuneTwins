from __future__ import annotations
from paths import expand_paths
import hashlib
import json
import math
import os
import random
import resource
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from data.treatment import UceVisitStore, validate_episodes, validate_transitions
from model.treatment import SharedVaccineLogicConfig, build_shared_meta_adapter, build_shared_projected_set, build_shared_set_transport, shared_model_parameter_audit
from model.digital_twin import MetaAdapterConfig, MetaAdapterDigitalTwin
from model.encoder import VaccineImmuneTwinConfig
from model.projected_set import MomentFiberCurrentModel
from model.patient import SetAffineCurrentModel
from model.losses import PatientLossV2Config, patient_vaccine_objective_v2, rbf_mmd2_per_unit

SCHEMA = "immunetwins_drug_discovery_shared_vaccine_logic_v2"

CHECKPOINT_SCHEMA = "immunetwins_drug_shared_checkpoint_1"

FORMAL_METHODS = (
    "scGen",
    "CellOT",
    "trVAE",
    "ImmuneTwins (no digital twin)",
    "ImmuneTwins (full)",
)

DIAGNOSTIC_METHODS = ("Persistence", "ImmuneTwins (shuffled history)")

def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value

def load_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    payload = expand_paths(yaml.safe_load(resolved.read_text(encoding="utf-8")))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA:
        raise ValueError("not a Drug v2 shared-architecture config")
    payload["_config_path"] = str(resolved)
    if payload.get("shared_architecture"):
        validate_shared_architecture(payload)
    return payload

def validate_shared_architecture(config: Mapping[str, Any]) -> dict[str, Any]:
    """Require the shared network and declare its task components."""
    contract = config["shared_architecture"]
    reference = Path(contract["reference"])
    if not reference.is_absolute():
        reference = Path(__file__).resolve().parents[2] / reference
    reference = reference.resolve(strict=True)
    payload = expand_paths(json.loads(reference.read_text()))
    expected = MetaAdapterConfig(**payload["twin_config"])
    shared = SharedVaccineLogicConfig(**dict(config["model"]))
    actual = shared.adapter_config()
    differences = {k: {"reference": v, "drug": asdict(actual)[k]}
                   for k, v in asdict(expected).items() if asdict(actual)[k] != v}
    components = contract.get("task_components", [])
    if set(components) - {"native_history_response_recovery", "native_history_visit_tokens"}:
        raise ValueError("Unknown shared-architecture task component")
    # These fields parameterize the training objective, not the network forward.
    loss_parameters = {"response_mean_loss_weight", "response_direction_loss_weight"}
    allowed = set(loss_parameters)
    if "native_history_visit_tokens" in components:
        allowed.add("use_visit_tokens")
        if not actual.use_visit_tokens:
            raise ValueError("Declared history visit component must be active")
    if "native_history_response_recovery" in components:
        allowed |= {"use_recovery_basis", "recovery_same_phase", "recovery_coefficient_initial"}
        if not (actual.use_recovery_basis and actual.recovery_same_phase):
            raise ValueError("Drug recovery component requires observed same-regimen history")
    forbidden = {k: v for k, v in differences.items() if k not in allowed}
    if forbidden:
        raise ValueError(f"Drug model differs from the frozen vaccine architecture: {forbidden}")
    backbone_reference = reference.parent / "moment_fiber_adaptive.yaml"
    backbone_payload = expand_paths(yaml.safe_load(backbone_reference.read_text()))
    expected_backbone = VaccineImmuneTwinConfig(**backbone_payload["model"])
    if shared.backbone_config() != expected_backbone:
        raise ValueError("Drug backbone differs from the frozen vaccine ProjectedSet architecture")
    if contract.get("output") != "predicted_cells" or contract.get("external_prediction_gate") != "forbidden":
        raise ValueError("ImmuneTwins must return the shared model's direct predicted_cells")
    if contract.get("model_class") != "model.digital_twin.MetaAdapterDigitalTwin":
        raise ValueError("Unexpected paper model class")
    result = {"status": "PASS", "reference": str(reference),
            "twin_config": asdict(actual), "external_prediction_gate": False,
            "prediction_output": "MetaAdapterDigitalTwin.forward.predicted_cells"}
    if components:
        result.update(task_components=list(components), task_component_configuration={
                          k: v for k, v in differences.items() if k not in loss_parameters},
                      shared_backbone_and_adapter_dimensions_unchanged=True)
    loss_differences = {k: v for k, v in differences.items() if k in loss_parameters}
    if loss_differences:
        result["task_loss_configuration"] = loss_differences
    return result

def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values],
        dtype=str,
    )


def _stable_seed(*parts: object, base: int = 42) -> int:
    return int(
        hashlib.sha256("\0".join(map(str, (base, *parts))).encode()).hexdigest()[:16],
        16,
    ) % (2**32)

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def require_cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU training is prohibited")
    device = torch.device(device_name)
    probe = torch.ones(8, device=device)
    if float((probe * probe).sum().item()) != 8.0:
        raise RuntimeError("CUDA tensor-operation probe failed")
    return device

def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)

def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)

def _normalized_log_counts(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    value = matrix.astype(np.float32, copy=True)
    totals = np.asarray(value.sum(axis=1)).ravel()
    factors = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    value = sparse.diags(factors) @ value
    value.data = np.log1p(value.data)
    return value.tocsr()

def _outer_training_visits(
    episodes: pd.DataFrame, transitions: pd.DataFrame, outer_fold: int, cohort: str
) -> tuple[list[str], list[str]]:
    train = episodes.loc[
        episodes["cohort_id"].astype(str).eq(cohort)
        & ~episodes["outer_fold"].astype(int).eq(int(outer_fold))
    ]
    patients = sorted(train["patient_id"].astype(str).unique())
    rows = transitions.loc[transitions["patient_id"].astype(str).isin(patients)]
    visits = sorted(
        set(rows["source_visit_id"].astype(str))
        | set(rows["target_visit_id"].astype(str))
    )
    return patients, visits

def prepare_fold_latents(config_path: str | Path, outer_fold: int) -> dict[str, Any]:
    """Fit a 16-D response coordinate using outer-training patients only."""

    config = load_config(config_path)
    data = _mapping(config["data"], "data")
    model = SharedVaccineLogicConfig(**dict(_mapping(config["model"], "model")))
    raw_path = Path(str(data["raw_count_store"])).resolve(strict=True)
    visits_path = Path(str(data["visits"])).resolve(strict=True)
    transitions = validate_transitions(pd.read_csv(data["transitions"], sep="\t"))
    episodes = pd.read_csv(data["episodes"], sep="\t")
    validate_episodes(episodes, transitions, folds=int(config["outer_folds"]))
    development = str(data["development_cohort"])
    patients, fit_visits = _outer_training_visits(
        episodes, transitions, outer_fold, development
    )
    visit_order = pd.read_csv(visits_path, sep="\t")["visit_id"].astype(str).tolist()
    fold_root = Path(str(data.get("latent_reference_root", data["output_root"]))) / f"fold_{outer_fold}"
    latent_path = fold_root / "latent_visits.h5"
    transform_path = fold_root / "latent_transform.npz"
    audit_path = fold_root / "latent_transform.json"
    if latent_path.exists() and transform_path.exists() and audit_path.exists():
        prior = expand_paths(json.loads(audit_path.read_text()))
        if (
            prior.get('status') == 'PASS'
            and prior.get('outer_fold') == int(outer_fold)
        ):
            return prior

    if data.get("latent_reference_root"):
        raise RuntimeError("A diagnostic initialization control must reuse a valid, immutable reference coordinate")

    with h5py.File(raw_path, "r") as handle:
        shape = tuple(int(value) for value in handle["matrix"].attrs["shape"])
        counts = sparse.csr_matrix(
            (
                handle["matrix/data"][:],
                handle["matrix/indices"][:],
                handle["matrix/indptr"][:],
            ),
            shape=shape,
        )
        cell_visits = _decode(handle["obs/donor_id"][:])
        genes = _decode(handle["var/gene_symbol"][:])
    normalized = _normalized_log_counts(counts)
    fit_mask = np.isin(cell_visits, np.asarray(fit_visits, dtype=str))
    if int(fit_mask.sum()) < model.latent_dim + 1:
        raise ValueError("outer-training cells are insufficient for the latent transform")
    seed = int(config["seed"]) + int(outer_fold)
    svd = TruncatedSVD(
        n_components=model.latent_dim,
        algorithm="randomized",
        n_iter=7,
        random_state=seed,
    )
    fit_coordinates = svd.fit_transform(normalized[fit_mask]).astype(np.float32)
    center = fit_coordinates.mean(axis=0)
    scale = fit_coordinates.std(axis=0)
    scale = np.maximum(scale, 1e-3).astype(np.float32)
    coordinates = np.empty((normalized.shape[0], model.latent_dim), dtype=np.float32)
    chunk = 4096
    for start in range(0, normalized.shape[0], chunk):
        stop = min(start + chunk, normalized.shape[0])
        coordinates[start:stop] = (
            svd.transform(normalized[start:stop]).astype(np.float32) - center
        ) / scale
    counts_per_visit = {visit: int(np.sum(cell_visits == visit)) for visit in visit_order}
    expected_cells = int(data["cells_per_visit"])
    if set(counts_per_visit.values()) != {expected_cells}:
        raise ValueError(f"latent visits do not all contain {expected_cells} cells")
    fold_root.mkdir(parents=True, exist_ok=True)
    temporary = latent_path.with_suffix(".tmp.h5")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(
            schema_version="immunetwins_drug_fold_latent_v2",
            status="complete",
            outer_fold=int(outer_fold),
            cells_per_visit=expected_cells,
            latent_dim=model.latent_dim,
        )
        target = handle.create_dataset(
            "embedding",
            (len(visit_order), expected_cells, model.latent_dim),
            dtype="f4",
            chunks=(1, expected_cells, model.latent_dim),
        )
        for index, visit in enumerate(visit_order):
            target[index] = coordinates[cell_visits == visit]
        handle.create_dataset(
            "visit_id",
            data=np.asarray(visit_order, dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        handle.flush()
    os.replace(temporary, latent_path)
    np.savez_compressed(
        transform_path,
        components=svd.components_.astype(np.float32),
        singular_values=svd.singular_values_.astype(np.float32),
        explained_variance_ratio=svd.explained_variance_ratio_.astype(np.float32),
        center=center,
        scale=scale,
        genes=genes,
        fit_patients=np.asarray(patients, dtype=str),
        fit_visits=np.asarray(fit_visits, dtype=str),
    )
    result = {
        "schema_version": "immunetwins_drug_fold_latent_audit_v2",
        "status": "PASS",
        "outer_fold": int(outer_fold),
        "latent_dim": model.latent_dim,
        "fit_patients": len(patients),
        "fit_visits": len(fit_visits),
        "fit_cells": int(fit_mask.sum()),
        "all_cells": int(len(cell_visits)),
        "genes": int(len(genes)),
        "explained_variance_fraction": float(svd.explained_variance_ratio_.sum()),
        "raw_count_store": str(raw_path),
        "latent_store": str(latent_path.resolve()),
        "transform": str(transform_path.resolve()),
        "test_patients_used_for_fit": False,
    }
    _atomic_json(audit_path, result)
    return result

class LatentVisitStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve(strict=True)
        self.handle = h5py.File(self.path, "r")
        if self.handle.attrs.get("status") != "complete":
            raise ValueError("fold latent store is incomplete")
        visits = _decode(self.handle["visit_id"][:])
        self.index = {visit: index for index, visit in enumerate(visits)}
        self.cells = int(self.handle["embedding"].shape[1])
        self.width = int(self.handle["embedding"].shape[2])

    def take(self, visit_id: str) -> np.ndarray:
        if visit_id not in self.index:
            raise KeyError(f"visit absent from fold latent store: {visit_id}")
        return np.asarray(self.handle["embedding"][self.index[visit_id]], dtype=np.float32)

    def close(self) -> None:
        if getattr(self, "handle", None) is not None:
            self.handle.close()
            self.handle = None

    def __del__(self) -> None:
        self.close()

@dataclass(frozen=True)
class EpisodeArrays:
    patient_id: str
    transition_id: str
    source_visit_id: str
    target_visit_id: str
    response: float
    phase: int
    source_uce: np.ndarray
    source_latent: np.ndarray
    target_latent: np.ndarray
    support_phase: np.ndarray
    support_dt: np.ndarray
    support_source_uce: np.ndarray
    support_target_uce: np.ndarray
    support_source_latent: np.ndarray
    support_target_latent: np.ndarray
    support_mask: np.ndarray

class DrugFoldData:
    """Materialize deterministic full-visit episodes for one outer fold."""

    def __init__(self, config_path: str | Path, outer_fold: int):
        self.config = load_config(config_path)
        self.outer_fold = int(outer_fold)
        data = _mapping(self.config["data"], "data")
        latent_audit = prepare_fold_latents(config_path, outer_fold)
        self.transitions = validate_transitions(pd.read_csv(data["transitions"], sep="\t"))
        self.transition_index = self.transitions.set_index("transition_id", drop=False)
        self.episodes = pd.read_csv(data["episodes"], sep="\t")
        validate_episodes(
            self.episodes, self.transitions, folds=int(self.config["outer_folds"])
        )
        self.uce = UceVisitStore(data["uce_store"])
        self.latent = LatentVisitStore(latent_audit["latent_store"])
        cells = int(data["cells_per_visit"])
        if self.uce.cells_available != cells or self.latent.cells != cells:
            raise ValueError("Drug v2 requires deterministic complete visits")
        self.cells = cells
        self.development = str(data["development_cohort"])
        self.external = str(data["external_cohort"])
        self._uce_cache: dict[str, np.ndarray] = {}
        self._latent_cache: dict[str, np.ndarray] = {}

    def close(self) -> None:
        self.uce.close()
        self.latent.close()

    def rows(self, split: str) -> pd.DataFrame:
        if split == "train":
            mask = self.episodes["cohort_id"].astype(str).eq(self.development)
            mask &= ~self.episodes["outer_fold"].astype(int).eq(self.outer_fold)
        elif split == "test":
            mask = self.episodes["cohort_id"].astype(str).eq(self.development)
            mask &= self.episodes["outer_fold"].astype(int).eq(self.outer_fold)
        elif split == "external":
            mask = self.episodes["cohort_id"].astype(str).eq(self.external)
        else:
            raise ValueError("split must be train, test or external")
        return self.episodes.loc[mask].sort_values("patient_id").reset_index(drop=True)

    def _uce(self, visit: str) -> np.ndarray:
        if visit not in self._uce_cache:
            self._uce_cache[visit] = self.uce.take(
                visit,
                cells=self.cells,
                seed=_stable_seed("visit", visit, base=int(self.config["seed"])),
            )
        return self._uce_cache[visit]

    def _latent(self, visit: str) -> np.ndarray:
        if visit not in self._latent_cache:
            self._latent_cache[visit] = self.latent.take(visit)
        return self._latent_cache[visit]

    def materialize(self, row: pd.Series, *, include_target: bool = True) -> EpisodeArrays:
        support_ids = expand_paths(json.loads(str(row["support_transition_ids"])))
        supports = [self.transition_index.loc[str(value)] for value in support_ids]
        phase_by_drug = {
            str(key): int(value)
            for key, value in dict(
                _mapping(self.config["data"], "data").get("phase_by_drug", {})
            ).items()
        }
        maximum = max(1, len(supports))
        source_uce = np.zeros((maximum, self.cells, 1280), dtype=np.float32)
        target_uce = np.zeros_like(source_uce)
        source_latent = np.zeros((maximum, self.cells, 16), dtype=np.float32)
        target_latent = np.zeros_like(source_latent)
        mask = np.zeros(maximum, dtype=bool)
        support_phase = np.zeros(maximum, dtype=np.int64)
        support_dt = np.zeros(maximum, dtype=np.float32)
        for index, support in enumerate(supports):
            source_visit = str(support["source_visit_id"])
            target_visit = str(support["target_visit_id"])
            source_uce[index] = self._uce(source_visit)
            target_uce[index] = self._uce(target_visit)
            source_latent[index] = self._latent(source_visit)
            target_latent[index] = self._latent(target_visit)
            mask[index] = True
            support_phase[index] = phase_by_drug.get(str(support["drug_id"]), 0)
            support_dt[index] = math.log1p(
                float(support["target_day"]) - float(support["source_day"])
            ) / math.log1p(365.0)
        response = float(row["response"]) if include_target and pd.notna(row["response"]) else float("nan")
        return EpisodeArrays(
            patient_id=str(row["patient_id"]),
            transition_id=str(row["transition_id"]),
            source_visit_id=str(row["source_visit_id"]),
            target_visit_id=str(row["target_visit_id"]),
            response=response,
            phase=phase_by_drug.get(str(row["drug_id"]), 1),
            source_uce=self._uce(str(row["source_visit_id"])),
            source_latent=self._latent(str(row["source_visit_id"])),
            target_latent=(self._latent(str(row["target_visit_id"])) if include_target
                           else np.full((self.cells, 16), np.nan, dtype=np.float32)),
            support_phase=support_phase,
            support_dt=support_dt,
            support_source_uce=source_uce,
            support_target_uce=target_uce,
            support_source_latent=source_latent,
            support_target_latent=target_latent,
            support_mask=mask,
        )

    def arrays(self, rows: pd.DataFrame, *, include_target: bool = True) -> list[EpisodeArrays]:
        return [self.materialize(row, include_target=include_target) for _, row in rows.iterrows()]

def patient_baseline_coordinates(items: Sequence[EpisodeArrays]) -> tuple[list[EpisodeArrays], np.ndarray]:
    """Use the earliest completed source as the patient origin.

    Predictions are translated back before scoring.
    """
    centered, origins = [], []
    for item in items:
        valid = np.flatnonzero(item.support_mask)
        if len(valid) != 1:
            raise ValueError("Baseline-coordinate IO requires exactly one completed transition")
        origin = item.support_source_latent[valid[0]].mean(0)
        if not np.isfinite(origin).all():
            raise ValueError("Patient baseline origin is not finite")
        origins.append(origin)
        centered.append(replace(item,
            source_latent=item.source_latent - origin,
            target_latent=item.target_latent - origin,
            support_source_latent=item.support_source_latent - origin,
            support_target_latent=item.support_target_latent - origin))
    return centered, np.asarray(origins, dtype=np.float32).reshape(len(items), 16)

def treatment_stage_inputs(items: Sequence[EpisodeArrays], *, same_regimen_history: bool = False) -> list[EpisodeArrays]:
    """Condition inputs: early/late stages and optional regimen-coded history.

    Recovery uses a separate reference for each treatment.
    """
    if any(np.count_nonzero(item.support_mask) != 1 for item in items):
        raise ValueError("Stage IO requires one completed treatment transition")
    if same_regimen_history:
        if any(not np.all(item.support_phase[item.support_mask] == item.phase) for item in items):
            raise ValueError("Same-regimen recovery requires matching historical treatment labels")
        return list(items)
    return [replace(item, support_phase=np.zeros_like(item.support_phase)) for item in items]

def completed_training_episodes(items: Sequence[EpisodeArrays]) -> list[EpisodeArrays]:
    """Use fit patients' observed transitions as auxiliary supervision."""
    result = []
    for item in items:
        indices = np.flatnonzero(item.support_mask)
        if len(indices) != 1:
            raise ValueError("Auxiliary training needs one completed transition")
        k = indices[0]
        result.append(replace(item, transition_id=item.transition_id+"__completed_history",
            source_visit_id=item.patient_id+"__completed_history_source",
            target_visit_id=item.source_visit_id, phase=0, response=float("nan"),
            source_uce=item.support_source_uce[k], source_latent=item.support_source_latent[k],
            target_latent=item.support_target_latent[k],
            support_phase=item.support_phase[:0], support_dt=item.support_dt[:0],
            support_source_uce=item.support_source_uce[:0], support_target_uce=item.support_target_uce[:0],
            support_source_latent=item.support_source_latent[:0], support_target_latent=item.support_target_latent[:0],
            support_mask=item.support_mask[:0]))
    return result

def _batch(items: Sequence[EpisodeArrays], device: torch.device) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot build an empty patient batch")
    support = max(len(item.support_mask) for item in items)
    batch = len(items)

    def zeros(*shape: int) -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

    support_source_uce = zeros(batch, support, items[0].source_uce.shape[0], 1280)
    support_target_uce = np.zeros_like(support_source_uce)
    support_source_latent = zeros(batch, support, items[0].source_latent.shape[0], 16)
    support_target_latent = np.zeros_like(support_source_latent)
    support_mask = np.zeros((batch, support), dtype=bool)
    support_phase = np.zeros((batch, support), dtype=np.int64)
    support_dt = np.zeros((batch, support), dtype=np.float32)
    for index, item in enumerate(items):
        count = len(item.support_mask)
        support_source_uce[index, :count] = item.support_source_uce
        support_target_uce[index, :count] = item.support_target_uce
        support_source_latent[index, :count] = item.support_source_latent
        support_target_latent[index, :count] = item.support_target_latent
        support_mask[index, :count] = item.support_mask
        support_phase[index, :count] = item.support_phase
        support_dt[index, :count] = item.support_dt
    source_latent = torch.as_tensor(np.stack([item.source_latent for item in items]), device=device)
    target_latent = torch.as_tensor(np.stack([item.target_latent for item in items]), device=device)
    support_source_latent_tensor = torch.as_tensor(support_source_latent, device=device)
    support_target_latent_tensor = torch.as_tensor(support_target_latent, device=device)
    support_mask_tensor = torch.as_tensor(support_mask, device=device)
    support_response = support_target_latent_tensor.mean(dim=2) - support_source_latent_tensor.mean(dim=2)
    return {
        "arm": "full",
        "query_uce": torch.as_tensor(np.stack([item.source_uce for item in items]), device=device),
        "source_latent": source_latent,
        "target_latent": target_latent,
        "phase": torch.as_tensor(
            [item.phase for item in items], dtype=torch.long, device=device
        ),
        "query_uce_mask": torch.ones(batch, source_latent.shape[1], dtype=torch.bool, device=device),
        "source_latent_mask": torch.ones(batch, source_latent.shape[1], dtype=torch.bool, device=device),
        "support_source_uce": torch.as_tensor(support_source_uce, device=device),
        "support_target_uce": torch.as_tensor(support_target_uce, device=device),
        "support_phase": torch.as_tensor(
            support_phase, dtype=torch.long, device=device
        ),
        "support_dt": torch.as_tensor(support_dt, device=device),
        "support_mask": support_mask_tensor,
        "support_source_latent_mean": support_source_latent_tensor.mean(dim=2),
        "support_target_latent_mean": support_target_latent_tensor.mean(dim=2),
        "support_response_latent": support_response,
        "gene_change": torch.empty(batch, support, 0, device=device),
        "gene_confidence": torch.empty(batch, support, 0, device=device),
        "clinical_delta_value": torch.empty(batch, support, 0, device=device),
        "clinical_delta_valid": torch.empty(batch, support, 0, dtype=torch.bool, device=device),
        "patient_id": [item.patient_id for item in items],
        "transition_id": [item.transition_id for item in items],
        "response": np.asarray([item.response for item in items], dtype=np.float32),
    }

def _forward(
    model: SetAffineCurrentModel | MomentFiberCurrentModel | MetaAdapterDigitalTwin,
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    tensor_arguments = {
        key: value
        for key, value in batch.items()
        if isinstance(value, torch.Tensor) and key not in {"target_latent"}
    }
    tensor_arguments["arm"] = "full" if isinstance(model, MetaAdapterDigitalTwin) else "current"
    return model(**tensor_arguments)

def _shuffle_history(batch: Mapping[str, Any], seed: int, *, same_regimen: bool = False) -> dict[str, Any]:
    result = dict(batch)
    count = len(batch["patient_id"])
    if count < 2 and not same_regimen:
        return result
    rng = np.random.default_rng(seed)
    permutation = np.arange(count)
    if same_regimen:
        phase = batch["phase"].detach().cpu().numpy()
        eligible = np.zeros(count, dtype=bool)
        for group in np.unique(phase):
            indices = np.flatnonzero(phase == group)
            if len(indices) > 1:
                order = rng.permutation(indices)
                permutation[order] = np.roll(order, 1)
                eligible[indices] = True
        result["history_contrast_mask"] = torch.as_tensor(eligible, device=batch["phase"].device)
    else:
        order = rng.permutation(count)
        permutation[order] = np.roll(order, 1)
    history_keys = (
        "support_source_uce",
        "support_target_uce",
        "support_phase",
        "support_dt",
        "support_mask",
        "support_source_latent_mean",
        "support_target_latent_mean",
        "support_response_latent",
        "gene_change",
        "gene_confidence",
        "clinical_delta_value",
        "clinical_delta_valid",
    )
    for key in history_keys:
        result[key] = batch[key][permutation]
    return result

def _empty_history(batch: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(batch)
    result["support_mask"] = torch.zeros_like(batch["support_mask"])
    return result

def _sample_covariance(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float64) - values.mean(axis=0, keepdims=True)
    return centered.T @ centered / max(len(values) - 1, 1)

def _scaled_basis(values: np.ndarray, rank: int) -> np.ndarray:
    _, _, right = np.linalg.svd(values, full_matrices=False)
    available = min(rank, len(right))
    directions = right[:available]
    scores = values @ directions.T
    result = np.sqrt(np.mean(np.square(scores), axis=0))[:, None] * directions
    if available < rank:
        result = np.concatenate(
            [result, np.zeros((rank - available, values.shape[1]))], axis=0
        )
    return result.astype(np.float32)

def fit_response_programs(
    items: Sequence[EpisodeArrays], config: SharedVaccineLogicConfig
) -> dict[str, Any]:
    if len(items) < 2:
        raise ValueError("response programs require at least two training patients")
    mean_responses, covariance_responses, relative_values = [], [], []
    upper = np.triu_indices(config.latent_dim)
    for item in items:
        source = item.source_latent
        target = item.target_latent
        mean_responses.append(target.mean(axis=0) - source.mean(axis=0))
        source_covariance = _sample_covariance(source)
        target_covariance = _sample_covariance(target)
        covariance_responses.append((target_covariance - source_covariance)[upper])
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (source_covariance + source_covariance.T)
        )
        floor = max(float(eigenvalues.max()) * 1e-5, 1e-6)
        inverse = (
            eigenvectors * np.maximum(eigenvalues, floor) ** -0.5
        ) @ eigenvectors.T
        relative = inverse @ target_covariance @ inverse
        relative_values.extend(np.linalg.eigvalsh(0.5 * (relative + relative.T)))
    mean_matrix = np.asarray(mean_responses, dtype=np.float32)
    covariance_matrix = np.asarray(covariance_responses, dtype=np.float32)
    scale = np.maximum(mean_matrix.std(axis=0), 0.05).astype(np.float32)
    minimum = max(float(np.quantile(relative_values, 0.01)), 1e-3)
    maximum = max(float(np.quantile(relative_values, 0.99)), minimum + 1e-3)
    phase_count = config.backbone_config().phase_count
    support_reference = np.zeros(
        (phase_count, config.latent_dim), dtype=np.float32
    )
    support_reference_observed = np.zeros(phase_count, dtype=bool)
    for phase in range(phase_count):
        patient_responses = []
        for item in items:
            selected = item.support_mask & (item.support_phase.astype(int) == phase)
            if np.any(selected):
                response = (
                    item.support_target_latent[selected].mean(axis=1)
                    - item.support_source_latent[selected].mean(axis=1)
                )
                patient_responses.append(response.mean(axis=0))
        if patient_responses:
            support_reference[phase] = np.mean(patient_responses, axis=0)
            support_reference_observed[phase] = True
    return {
        "response_center": mean_matrix.mean(axis=0).astype(np.float32),
        "response_scale": scale,
        "mean_program_basis": _scaled_basis(mean_matrix, config.mean_program_rank),
        "covariance_program_basis": _scaled_basis(
            covariance_matrix, config.covariance_program_rank
        ),
        "relative_covariance_eigenvalue_minimum": minimum,
        "relative_covariance_eigenvalue_maximum": maximum,
        "support_response_reference": support_reference,
        "support_response_reference_observed": support_reference_observed,
        "fit_patients": sorted(item.patient_id for item in items),
    }

def set_response_programs(
    model: SetAffineCurrentModel | MomentFiberCurrentModel | MetaAdapterDigitalTwin,
    programs: Mapping[str, Any],
) -> None:
    backbone = model.backbone if isinstance(model, MetaAdapterDigitalTwin) else model
    backbone.set_response_transform(
        torch.as_tensor(programs["response_center"]),
        torch.as_tensor(programs["response_scale"]),
    )
    backbone.set_response_programs(
        mean_program_basis=torch.as_tensor(programs["mean_program_basis"]),
        covariance_program_basis=torch.as_tensor(programs["covariance_program_basis"]),
        relative_covariance_eigenvalue_minimum=float(
            programs["relative_covariance_eigenvalue_minimum"]
        ),
        relative_covariance_eigenvalue_maximum=float(
            programs["relative_covariance_eigenvalue_maximum"]
        ),
    )
    if isinstance(model, MetaAdapterDigitalTwin):
        model.set_support_response_reference(
            torch.as_tensor(programs["support_response_reference"]),
            torch.as_tensor(programs["support_response_reference_observed"]),
        )

def _training_response_scale(
    programs: Mapping[str, Any], config: SharedVaccineLogicConfig
) -> np.ndarray:
    if config.standardize_response_loss:
        return np.asarray(programs["response_scale"], dtype=np.float32)
    return np.ones(config.latent_dim, dtype=np.float32)

def estimate_bandwidth(items: Sequence[EpisodeArrays], seed: int) -> float:
    cells = np.concatenate([item.target_latent for item in items], axis=0)
    rng = np.random.default_rng(seed)
    count = min(4096, len(cells) * 2)
    left = cells[rng.integers(0, len(cells), size=count)]
    right = cells[rng.integers(0, len(cells), size=count)]
    distances = np.sqrt(np.sum(np.square(left - right), axis=1))
    positive = distances[distances > 0]
    return float(np.median(positive)) if len(positive) else 1.0

def _loss_config(
    model: SetAffineCurrentModel | MomentFiberCurrentModel | MetaAdapterDigitalTwin,
) -> PatientLossV2Config:
    """Reuse the selected vaccine-stage objective for each shared stage."""

    if isinstance(model, MetaAdapterDigitalTwin):
        return PatientLossV2Config(
            distribution_weight=1.0,
            response_mean_weight=model.twin_config.response_mean_loss_weight,
            response_direction_weight=model.twin_config.response_direction_loss_weight,
            response_norm_weight=0.5,
            centered_geometry_weight=0.75,
            history_sensitivity_weight=0.5,
            history_sensitivity_temperature=0.05,
            history_current_anchor_weight=0.25,
            history_current_anchor_temperature=0.1,
        )
    if isinstance(model, MomentFiberCurrentModel):
        return PatientLossV2Config(
            distribution_weight=1.0,
            response_mean_weight=1.0,
            response_direction_weight=0.25,
            response_norm_weight=0.25,
            centered_geometry_weight=0.5,
        )
    return PatientLossV2Config(
        distribution_weight=0.25,
        response_mean_weight=1.0,
        response_direction_weight=0.25,
        response_norm_weight=0.25,
        centered_geometry_weight=0.5,
    )

def _validation_metric(
    model: SetAffineCurrentModel | MomentFiberCurrentModel | MetaAdapterDigitalTwin,
    items: Sequence[EpisodeArrays],
    device: torch.device,
    *,
    metric: str,
    bandwidth: float,
    details: dict[str, float] | None = None,
) -> float:
    model.eval()
    with torch.inference_mode():
        batch = _batch(items, device)
        output = _forward(model, batch)
        if metric == "distribution_mmd2":
            values = rbf_mmd2_per_unit(
                output["predicted_cells"],
                batch["target_latent"],
                bandwidth=bandwidth,
            )
            return float(values.mean().cpu())
        if metric != "raw_patient_delta_rmse":
            raise ValueError(f"unknown validation metric: {metric}")
        predicted = output["predicted_cells"].mean(dim=1)
        observed = batch["target_latent"].mean(dim=1)
        source = batch["source_latent"].mean(dim=1)
        residual = (predicted - source) - (observed - source)
        if details is not None:
            left, right = predicted - source, observed - source
            denominator = left.norm(dim=-1) * right.norm(dim=-1)
            cosine = torch.where(denominator > 1e-12,
                (left * right).sum(dim=-1) / denominator.clamp_min(1e-12), 0.)
            details.update(validation_mae=float(residual.abs().mean().cpu()),
                           validation_cosine=float(cosine.mean().cpu()))
        return float(torch.sqrt(residual.square().mean(dim=1)).mean().cpu())

@torch.no_grad()
def _update_parameter_ema(model: torch.nn.Module, averaged: dict[str, torch.Tensor], decay: float) -> None:
    """Average trainable parameters only."""
    for name, parameter in model.named_parameters():
        if name in averaged:
            if not parameter.requires_grad:
                raise ValueError("Frozen parameters must not enter the EMA")
            averaged[name].lerp_(parameter.detach(), 1.0 - decay)

def _fit_stage(
    model: SetAffineCurrentModel | MomentFiberCurrentModel | MetaAdapterDigitalTwin,
    train_items: Sequence[EpisodeArrays],
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    bandwidth: float,
    response_scale: np.ndarray,
    seed: int,
    selection_metric: str = "raw_patient_delta_rmse",
    validation_items: Sequence[EpisodeArrays] | None = None,
    patience: int | None = None,
    batch_size: int | None = None,
    output_heads_only: bool = False,
    parameter_l2_weight: float = 0.0,
    history_negative_sampling: str = "random_patient",
    raw_mean_mae_weight: float = 0.0,
    parameter_ema_decay: float = 0.0,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, float]]]:
    model.to(device)
    if output_heads_only:
        if not isinstance(model, MetaAdapterDigitalTwin):
            raise ValueError("Restricted fine-tuning is only defined for the native MetaAdapter")
        prefixes = tuple("identifier." + name + "." for name in (
            "slot_scale", "patient_scale", "mean_head_scale", "covariance_head_scale",
            "magnitude_head_scale", "slot_prompt", "patient_prompt", "recovery_coefficient"))
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
    if parameter_l2_weight < 0:
        raise ValueError("Parameter regularization cannot be negative")
    if history_negative_sampling not in ("random_patient", "same_regimen"):
        raise ValueError("Unknown history-negative sampling rule")
    if raw_mean_mae_weight < 0 or not math.isfinite(raw_mean_mae_weight):
        raise ValueError("Raw mean MAE weight must be finite and nonnegative")
    if not 0 <= parameter_ema_decay < 1:
        raise ValueError("Parameter EMA decay must be in [0, 1)")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("stage has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    history_config = isinstance(model, MetaAdapterDigitalTwin)
    if parameter_ema_decay and not history_config:
        raise ValueError("This task's parameter averaging is restricted to MetaAdapter")
    averaged = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                if parameter.requires_grad} if parameter_ema_decay else {}
    objective = _loss_config(model)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_metric = math.inf
    stale = 0
    history: list[dict[str, float]] = []
    train_batch = _batch(train_items, device)
    scale = torch.as_tensor(response_scale, device=device)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = np.arange(len(train_items))
        if batch_size is not None:
            np.random.default_rng(seed + epoch).shuffle(order)
        step_size = batch_size or len(train_items)
        epoch_losses: dict[str, float] = {}
        for start in range(0, len(order), step_size):
            indices = order[start:start + step_size].tolist()
            batch = {
                k: v[indices] if isinstance(v, (torch.Tensor, np.ndarray))
                else [v[i] for i in indices] if isinstance(v, list) else v
                for k, v in train_batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch)
            shuffled_prediction = current_response = history_mask = current_history_mask = None
            if history_config:
                shuffled = _shuffle_history(batch, seed + epoch + start,
                    same_regimen=history_negative_sampling == "same_regimen")
                shuffled_prediction = _forward(model, shuffled)["predicted_cells"]
                current_response = output["current_response"]
                current_history_mask = batch["support_mask"].any(dim=1)
                history_mask = current_history_mask & shuffled.get("history_contrast_mask", current_history_mask)
            losses = patient_vaccine_objective_v2(
                output["predicted_cells"], batch["source_latent"], batch["target_latent"],
                donor_ids=batch["patient_id"], bandwidth=bandwidth, config=objective,
                response_scale=scale, shuffled_history_predicted=shuffled_prediction,
                history_mask=history_mask, current_response=current_response,
                history_current_mask=current_history_mask,
            )
            if raw_mean_mae_weight:
                # Patient-averaged absolute error in the scoring coordinate.
                losses["raw_mean_mae_loss"] = (output["predicted_cells"].mean(dim=1)
                    - batch["target_latent"].mean(dim=1)).abs().mean(dim=-1).mean()
                losses["loss"] = losses["loss"] + raw_mean_mae_weight * losses["raw_mean_mae_loss"]
            if parameter_l2_weight:
                losses["parameter_l2"] = sum(parameter.square().sum() for parameter in trainable)
                losses["loss"] = losses["loss"] + parameter_l2_weight * losses["parameter_l2"]
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    epoch_losses[key] = epoch_losses.get(key, 0.) + float(value.detach().cpu()) * len(indices) / len(order)
        losses = {key: torch.tensor(value) for key, value in epoch_losses.items()}
        online = {}
        if averaged:
            _update_parameter_ema(model, averaged, parameter_ema_decay)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in averaged:
                        online[name] = parameter.detach().clone()
                        parameter.copy_(averaged[name])
        validation_details: dict[str, float] = {}
        metric = (
            _validation_metric(
                model,
                validation_items,
                device,
                metric=selection_metric,
                bandwidth=bandwidth,
                details=validation_details,
            )
            if validation_items is not None
            else float(losses["raw_response_mean_loss"].sqrt().detach().cpu())
            if "raw_response_mean_loss" in losses
            else float(losses["response_mean_loss"].sqrt().detach().cpu())
        )
        record = {
            "epoch": float(epoch),
            "loss": float(losses["loss"].detach().cpu()),
            "selection_value": metric,
            "selection_metric": selection_metric,
            **validation_details,
            "distribution_mmd2": float(losses["distribution_mmd2"].detach().cpu()),
            "response_direction_loss": float(
                losses["response_direction_loss"].detach().cpu()
            ),
        }
        if averaged:
            record["parameter_ema_decay"] = parameter_ema_decay
        for key in ("response_mean_loss", "raw_response_mean_loss", "history_gain",
                    "history_effect_size", "history_vs_current_gain", "raw_mean_mae_loss"):
            if key in losses:
                record[key] = float(losses[key])
        history.append(record)
        if validation_items is None:
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_epoch = epoch
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in online:
                        parameter.copy_(online[name])
            continue
        if metric < best_metric - 1e-6:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if patience is not None and stale >= patience:
                break
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in online:
                    parameter.copy_(online[name])
    if best_state is None:
        raise RuntimeError("training produced no finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    return best_state, best_epoch, history

def _inner_splits(rows: pd.DataFrame, folds: int, seed: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    assignment: dict[str, int] = {}
    patient_rows = rows.drop_duplicates("patient_id").copy()
    action_column = "drug_id" if "drug_id" in patient_rows else "cohort_id"
    response_column = (
        "ordinal_response"
        if "ordinal_response" in patient_rows
        else "response"
    )
    # The state-forecasting endpoint has no scalar transition response; its
    # stratifier is the downstream ordinal outcome.  Cycle through response
    # strata within each action arm so small strata do not all land in fold 0.
    for action, action_rows in patient_rows.groupby(action_column, sort=True):
        cursor = 0
        for label, group in action_rows.groupby(response_column, sort=True, dropna=False):
            patients = sorted(
                group["patient_id"].astype(str),
                key=lambda patient: _stable_seed(
                    "inner", patient, action, label, base=seed
                ),
            )
            for patient in patients:
                assignment[patient] = cursor % folds
                cursor += 1
    splits = []
    patient_values = rows["patient_id"].astype(str)
    for fold in range(folds):
        validation = {patient for patient, value in assignment.items() if value == fold}
        if not validation:
            continue
        mask = patient_values.isin(validation)
        splits.append((rows.loc[~mask].reset_index(drop=True), rows.loc[mask].reset_index(drop=True)))
    return splits

def _configure_set_trainable(model: SetAffineCurrentModel) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("visit_encoder."))
    model.visit_encoder.eval()

def _build_set(
    shared: SharedVaccineLogicConfig,
    pretraining: Mapping[str, Any],
    programs: Mapping[str, Any],
    *,
    head_initialization: str = "native",
    initialization_device: torch.device | None = None,
) -> tuple[SetAffineCurrentModel, dict[str, Any]]:
    model, audit = build_shared_set_transport(
        shared,
        pretraining["checkpoint"],
    )
    initialization = pretraining.get("initialization", "cima_pretrained")
    if initialization == "random_frozen_diagnostic":
        # Control with the same architecture and selection.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(pretraining["initialization_seed"]))
            for part in (model.visit_encoder.cell_projection, model.visit_encoder.pool):
                for module in part.modules():
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()
        audit.update(initialization=initialization, cima_learned_parameters_used=False,
                     initialization_seed=int(pretraining["initialization_seed"]),
                     loaded_then_reset_parameters=["cell_projection", "pool"])
    elif initialization != "cima_pretrained":
        raise ValueError(f"Unknown initialization {initialization}")
    else:
        audit.update(initialization=initialization, cima_learned_parameters_used=True)
    _configure_set_trainable(model)
    set_response_programs(model, programs)
    if head_initialization == "training_mean":
        if initialization_device is None or initialization_device.type != "cuda":
            raise RuntimeError("Data-derived head initialization requires CUDA")
        # Initialize direction and magnitude from the fit-patient response centroid.
        basis = torch.as_tensor(programs["mean_program_basis"], device=initialization_device)
        center = torch.as_tensor(programs["response_center"], device=initialization_device)
        coefficients = center @ torch.linalg.pinv(basis)
        magnitude = coefficients.norm().clamp_min(1e-4)
        with torch.no_grad():
            model.mean_direction_head.weight.mul_(0.01)
            model.mean_direction_head.bias.copy_((coefficients / magnitude).cpu())
            model.mean_magnitude_head.weight.zero_()
            model.mean_magnitude_head.bias.copy_((magnitude + torch.log(-torch.expm1(-magnitude))).cpu().reshape(1))
        audit["response_head_initialization"] = {"kind": head_initialization,
            "fit_patients": list(programs["fit_patients"]), "coefficient_norm": float(magnitude),
            "additional_prediction_path": False, "CIMA_parameters_changed": False}
    elif head_initialization != "native":
        raise ValueError(f"Unknown response head initialization {head_initialization}")
    return model, audit

def _common_validation_choice(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose one learning rate/epoch from patient-weighted inner curves only."""
    curves: dict[float, list[tuple[int, np.ndarray]]] = {}
    for record in records:
        count = len(record["validation_patients"])
        for candidate in record["meta_training_candidates"]:
            values = np.asarray([row["selection_value"] for row in candidate["curve"]])
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite inner validation curve")
            curves.setdefault(float(candidate["learning_rate"]), []).append((count, values))
    candidates = []
    for lr, entries in curves.items():
        if len(entries) != len(records) or len({len(v) for _, v in entries}) != 1:
            raise ValueError("Common-curve selection requires every inner fold at every epoch")
        mean = sum(n * v for n, v in entries) / sum(n for n, _ in entries)
        epoch = int(np.argmin(mean)) + 1
        candidates.append({"learning_rate": lr, "epoch": epoch,
                           "validation_rmse": float(mean[epoch - 1]),
                           "patient_weighted_curve": mean.tolist()})
    winner = min(candidates, key=lambda row: (row["validation_rmse"], row["epoch"], row["learning_rate"]))
    result = {"rule": "minimum_patient_weighted_common_inner_validation_curve",
            "learning_rate": winner["learning_rate"], "epoch": winner["epoch"],
            "validation_rmse": winner["validation_rmse"], "candidates": candidates,
            "test_patients_used": False}
    selected_rows = [(len(record["validation_patients"]), next(candidate for candidate
        in record["meta_training_candidates"] if float(candidate["learning_rate"]) == winner["learning_rate"])
        ["curve"][winner["epoch"] - 1]) for record in records]
    for key in ("validation_mae", "validation_cosine"):
        if all(key in row for _, row in selected_rows):
            result[key] = sum(n * row[key] for n, row in selected_rows) / sum(n for n, _ in selected_rows)
    return result

def _select_epochs(
    config: Mapping[str, Any], data: DrugFoldData, outer_fold: int, device: torch.device,
    *, only_inner_fold: int | None = None,
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    optimization = _mapping(config["optimization"], "optimization")
    shared = SharedVaccineLogicConfig(**dict(_mapping(config["model"], "model")))
    pretraining = _mapping(config["pretraining"], "pretraining")
    rows = data.rows("train")
    records: list[dict[str, Any]] = []
    recovery_gate_features: list[np.ndarray] = []
    recovery_gate_targets: list[np.ndarray] = []
    recovery_gate_patients: list[str] = []
    selected: dict[str, list[int]] = {"set_transport": [], "projected_set": [], "meta_adapter": []}
    splits = _inner_splits(
        rows, int(optimization["inner_folds"]), int(config["seed"]) + outer_fold
    )
    if only_inner_fold is not None and only_inner_fold not in range(len(splits)):
        raise ValueError("Requested inner fold is out of range")
    for inner_fold, (train_rows, validation_rows) in enumerate(splits):
        if only_inner_fold is not None and inner_fold != only_inner_fold:
            continue
        print(f"outer={outer_fold} inner={inner_fold} stage=parent start", flush=True)
        train_items = data.arrays(train_rows)
        validation_items = data.arrays(validation_rows)
        if optimization.get("patient_baseline_coordinates", False):
            train_items, _ = patient_baseline_coordinates(train_items)
            validation_items, _ = patient_baseline_coordinates(validation_items)
        parent_items = train_items
        if optimization.get("completed_transition_supervision", False):
            train_items = treatment_stage_inputs(train_items, same_regimen_history=shared.recovery_same_phase)
            validation_items = treatment_stage_inputs(validation_items, same_regimen_history=shared.recovery_same_phase)
            parent_items = train_items + completed_training_episodes(train_items)
        programs = fit_response_programs(parent_items, shared)
        training_response_scale = _training_response_scale(programs, shared)
        bandwidth = estimate_bandwidth(parent_items, int(config["seed"]) + inner_fold)
        seed = int(config["seed"]) + 1000 * outer_fold + 100 * inner_fold
        _set_seed(seed)
        set_model, _ = _build_set(shared, pretraining, programs,
            head_initialization=optimization.get("response_head_initialization", "native"),
            initialization_device=device)
        set_state, set_epoch, set_history = _fit_stage(
            set_model,
            parent_items,
            device,
            epochs=int(optimization["set_transport_max_epochs"]),
            learning_rate=float(optimization["set_transport_learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
            bandwidth=bandwidth,
            response_scale=training_response_scale,
            seed=seed,
            validation_items=validation_items,
            patience=int(optimization["set_transport_patience"]),
        )
        projected = build_shared_projected_set(shared, set_state)
        projected_state, projected_epoch, projected_history = _fit_stage(
            projected,
            parent_items,
            device,
            epochs=int(optimization["projected_set_max_epochs"]),
            learning_rate=float(optimization["projected_set_learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
            bandwidth=bandwidth,
            response_scale=training_response_scale,
            seed=seed + 1,
            selection_metric="distribution_mmd2",
            validation_items=validation_items,
            patience=int(optimization["projected_set_patience"]),
        )
        projected.load_state_dict(projected_state, strict=True)
        print(f"outer={outer_fold} inner={inner_fold} parent_epochs={set_epoch}/{projected_epoch}", flush=True)
        common_curve = optimization.get("meta_adapter_epoch_selection") == "patient_weighted_common_curve"
        learning_rates = optimization.get("meta_adapter_learning_rates", [optimization["meta_adapter_learning_rate"]])
        if len(learning_rates) > 1 and not common_curve:
            raise ValueError("Multiple learning rates require joint common-curve inner selection")
        meta_candidates = []
        for learning_rate in learning_rates:
            if common_curve:
                _set_seed(seed + 2)
            meta = build_shared_meta_adapter(shared, projected)
            set_response_programs(meta, programs)
            meta_state, meta_epoch, meta_history = _fit_stage(
                meta, train_items, device,
                epochs=int(optimization["meta_adapter_max_epochs"]),
                learning_rate=float(learning_rate),
                weight_decay=float(optimization["weight_decay"]), bandwidth=bandwidth,
                response_scale=(np.ones_like(training_response_scale)
                    if optimization.get("meta_adapter_raw_response_loss", False)
                    else training_response_scale), seed=seed + 2,
                validation_items=validation_items,
                patience=None if common_curve else int(optimization["meta_adapter_patience"]),
                batch_size=optimization.get("meta_adapter_batch_size"),
                output_heads_only=optimization.get("meta_adapter_output_heads_only", False),
                parameter_l2_weight=float(optimization.get("meta_adapter_parameter_l2", 0.0)),
                history_negative_sampling=optimization.get("history_negative_sampling", "random_patient"),
                raw_mean_mae_weight=float(optimization.get("meta_adapter_raw_mean_mae_weight", 0.0)),
                parameter_ema_decay=float(optimization.get("meta_adapter_parameter_ema_decay", 0.0)),
            )
            meta_candidates.append({"learning_rate": float(learning_rate), "curve": meta_history})
            print(f"outer={outer_fold} inner={inner_fold} meta_lr={learning_rate} done", flush=True)
        if shared.use_recovery_basis and shared.recovery_same_phase and not config.get("shared_architecture"):
            features, targets = _recovery_gate_training_values(
                meta, validation_items, device
            )
            recovery_gate_features.append(features)
            recovery_gate_targets.append(targets)
            recovery_gate_patients.extend(item.patient_id for item in validation_items)
        selected["set_transport"].append(set_epoch)
        selected["projected_set"].append(projected_epoch)
        selected["meta_adapter"].append(meta_epoch)
        records.append(
            {
                "inner_fold": inner_fold,
                "train_patients": sorted(train_rows["patient_id"].astype(str).unique()),
                "validation_patients": sorted(
                    validation_rows["patient_id"].astype(str).unique()
                ),
                "bandwidth": bandwidth,
                "best_epochs": {
                    "set_transport": set_epoch,
                    "projected_set": projected_epoch,
                    "meta_adapter": meta_epoch,
                },
                "best_validation_metric": {
                    "set_transport": min(row["selection_value"] for row in set_history),
                    "projected_set": min(
                        row["selection_value"] for row in projected_history
                    ),
                    "meta_adapter": min(
                        row["selection_value"] for row in meta_history
                    ),
                },
                "selection_metrics": {
                    "set_transport": "raw_patient_delta_rmse",
                    "projected_set": "distribution_mmd2",
                    "meta_adapter": "raw_patient_delta_rmse",
                },
                "meta_training_candidates": meta_candidates,
            }
        )
        del set_model, projected, meta, set_state, projected_state, meta_state
        torch.cuda.empty_cache()
    epochs = {
        name: max(1, int(math.ceil(float(np.median(values)))))
        for name, values in selected.items()
    }
    recovery_gate = None
    if recovery_gate_features:
        recovery_gate = _fit_recovery_gate(
            np.concatenate(recovery_gate_features),
            np.concatenate(recovery_gate_targets),
            recovery_gate_patients,
        )
    optimization_choice = {"learning_rate": float(optimization["meta_adapter_learning_rate"]),
                           "rule": "median_inner_best_epoch", "test_patients_used": False}
    if optimization.get("meta_adapter_epoch_selection") == "patient_weighted_common_curve":
        optimization_choice = _common_validation_choice(records)
        epochs["meta_adapter"] = optimization_choice["epoch"]
    return epochs, records, recovery_gate, optimization_choice

def _recovery_gate_features(
    output: Mapping[str, torch.Tensor], batch: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predicted = output["predicted_cells"].mean(dim=1)
    source = batch["source_latent"].mean(dim=1)
    predicted_delta = predicted - source
    basis = output["recovery_basis"]
    coefficient = output["recovery_coefficient"].squeeze(1)
    base_delta = predicted_delta - coefficient.unsqueeze(1) * basis
    basis_norm = torch.linalg.vector_norm(basis, dim=1)
    base_norm = torch.linalg.vector_norm(base_delta, dim=1)
    dot = torch.sum(base_delta * basis, dim=1)
    cosine = dot / (basis_norm * base_norm).clamp_min(1e-12)
    features = torch.stack(
        (
            coefficient,
            batch["phase"].float(),
            basis_norm,
            base_norm,
            dot,
            cosine,
        ),
        dim=1,
    )
    return (
        features.detach().cpu().numpy().astype(np.float64),
        base_delta.detach().cpu().numpy().astype(np.float64),
        basis.detach().cpu().numpy().astype(np.float64),
        coefficient.detach().cpu().numpy().astype(np.float64),
    )

def _recovery_gate_training_values(
    model: MetaAdapterDigitalTwin,
    items: Sequence[EpisodeArrays],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.inference_mode():
        batch = _batch(items, device)
        output = _forward(model, batch)
        features, base_delta, basis, _ = _recovery_gate_features(output, batch)
        observed_delta = (
            batch["target_latent"].mean(dim=1)
            - batch["source_latent"].mean(dim=1)
        ).cpu().numpy()
    numerator = np.sum(basis * (observed_delta - base_delta), axis=1)
    denominator = np.maximum(np.sum(np.square(basis), axis=1), 1e-12)
    target = np.clip(numerator / denominator, -2.0, 0.0)
    return features, target

def _ridge_gate_fit(
    features: np.ndarray, target: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-8)
    standardized = (features - center) / scale
    intercept = float(target.mean())
    coefficient = np.linalg.solve(
        standardized.T @ standardized + float(alpha) * np.eye(features.shape[1]),
        standardized.T @ (target - intercept),
    )
    return center, scale, intercept, coefficient

def _fit_recovery_gate(
    features: np.ndarray, target: np.ndarray, patients: Sequence[str]
) -> dict[str, Any]:
    if len(features) != len(target) or len(features) != len(patients):
        raise ValueError("recovery gate OOF arrays do not align")
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        prediction = np.empty(len(target), dtype=np.float64)
        for held_out in range(len(target)):
            keep = np.arange(len(target)) != held_out
            center, scale, intercept, coefficient = _ridge_gate_fit(
                features[keep], target[keep], alpha
            )
            prediction[held_out] = intercept + (
                (features[held_out] - center) / scale
            ) @ coefficient
        prediction = np.clip(prediction, -2.0, 0.0)
        candidates.append(
            {
                "alpha": float(alpha),
                "leave_one_patient_out_mse": float(
                    np.mean(np.square(prediction - target))
                ),
            }
        )
    selected = min(candidates, key=lambda value: value["leave_one_patient_out_mse"])
    center, scale, intercept, coefficient = _ridge_gate_fit(
        features, target, float(selected["alpha"])
    )
    return {
        "feature_names": [
            "neural_recovery_coefficient",
            "treatment_phase",
            "recovery_basis_norm",
            "base_response_norm",
            "base_response_basis_dot",
            "base_response_basis_cosine",
        ],
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
        "intercept": intercept,
        "coefficient": coefficient.tolist(),
        "selected_alpha": float(selected["alpha"]),
        "selection_candidates": candidates,
        "fit_patients": sorted(map(str, patients)),
        "fit_values_are_inner_oof": True,
    }

def _apply_recovery_gate(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    recovery_gate: Mapping[str, Any],
) -> torch.Tensor:
    features, _, _, _ = _recovery_gate_features(output, batch)
    center = np.asarray(recovery_gate["feature_center"], dtype=np.float64)
    scale = np.asarray(recovery_gate["feature_scale"], dtype=np.float64)
    coefficient = np.asarray(recovery_gate["coefficient"], dtype=np.float64)
    calibrated = float(recovery_gate["intercept"]) + (
        (features - center) / scale
    ) @ coefficient
    calibrated = np.clip(calibrated, -2.0, 0.0)
    calibrated_tensor = torch.as_tensor(
        calibrated,
        dtype=output["recovery_coefficient"].dtype,
        device=output["recovery_coefficient"].device,
    )
    learned = output["recovery_coefficient"].squeeze(1)
    shift = (calibrated_tensor - learned).unsqueeze(1) * output["recovery_basis"]
    return output["predicted_cells"] + shift.unsqueeze(1)

def _predict(
    model: SetAffineCurrentModel | MomentFiberCurrentModel | MetaAdapterDigitalTwin,
    items: Sequence[EpisodeArrays],
    device: torch.device,
    *,
    history_mode: str = "correct",
    seed: int = 42,
    recovery_gate: Mapping[str, Any] | None = None,
) -> np.ndarray:
    if recovery_gate is not None and getattr(model, "_strict_shared_forward", False):
        raise ValueError("External recovery gate is forbidden for the unified paper model")
    model.eval()
    with torch.inference_mode():
        batch = _batch(items, device)
        if history_mode == "empty":
            batch = _empty_history(batch)
        elif history_mode == "shuffled":
            batch = _shuffle_history(batch, seed)
        elif history_mode != "correct":
            raise ValueError("history_mode must be correct, empty or shuffled")
        output = _forward(model, batch)
        predicted = (
            _apply_recovery_gate(output, batch, recovery_gate)
            if recovery_gate is not None
            else output["predicted_cells"]
        )
        return predicted.float().cpu().numpy()

def save_prediction_bundle(
    path: Path,
    method: str,
    split: str,
    items: Sequence[EpisodeArrays],
    predicted: np.ndarray,
    outer_fold: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray("immunetwins_drug_prediction_bundle_v2"),
        method=np.asarray(method),
        split=np.asarray(split),
        outer_fold=np.asarray(int(outer_fold)),
        patient_id=np.asarray([item.patient_id for item in items], dtype=str),
        transition_id=np.asarray([item.transition_id for item in items], dtype=str),
        response=np.asarray([item.response for item in items], dtype=np.float32),
        source=np.stack([item.source_latent for item in items]).astype(np.float32),
        observed=np.stack([item.target_latent for item in items]).astype(np.float32),
        predicted=np.asarray(predicted, dtype=np.float32),
        prediction_uses_test_future=np.asarray(False),
    )
    os.replace(temporary, path)

def train_fold(
    config_path: str | Path, outer_fold: int, *, device_name: str = "cuda:0",
    selection_reference: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    if not config.get("shared_architecture"):
        raise ValueError("New ImmuneTwins training requires the frozen vaccine architecture contract; legacy runs are read-only")
    if (Path(config["data"]["output_root"]) / f"fold_{outer_fold}/model_run.json").exists():
        raise FileExistsError("Completed shared-model results are immutable; select a new output root")
    device = require_cuda(device_name)
    torch.set_num_threads(4)
    data = DrugFoldData(config_path, outer_fold)
    try:
        if selection_reference is None:
            selected_epochs, inner_records, recovery_gate, optimization_choice = _select_epochs(
                config, data, outer_fold, device
            )
        else:
            cache = expand_paths(json.loads(Path(selection_reference).read_text()))
            if (
                cache['outer_fold'] != outer_fold
                or cache['test_targets_opened'] is not False
                or cache['recovery_gate'] is not None
            ):
                raise ValueError("Inner-only selection cache does not match this experiment")
            inner_records = cache["records"]
            expected = set(data.rows("train").patient_id.astype(str))
            held = []
            for record in inner_records:
                fit, val = set(record["train_patients"]), set(record["validation_patients"])
                if fit & val or fit | val != expected:
                    raise ValueError("Cached inner selection leaks or changes outer patients")
                held.extend(val)
            if set(held) != expected or len(held) != len(expected):
                raise ValueError("Cached inner validation must cover each training patient once")
            optimization_choice = _common_validation_choice(inner_records)
            if optimization_choice != cache["optimization_choice"]:
                raise ValueError("Cached selection was not the common-curve minimum")
            selected_epochs = {name: max(1, int(math.ceil(np.median([
                row["best_epochs"][name] for row in inner_records]))))
                for name in ("set_transport", "projected_set")}
            selected_epochs["meta_adapter"] = optimization_choice["epoch"]
            recovery_gate = None
            print(f"outer={outer_fold} verified existing inner selection; refitting only", flush=True)
        train_items = data.arrays(data.rows("train"))
        test_items = data.arrays(data.rows("test"))
        external_items = data.arrays(data.rows("external"))
        shared = SharedVaccineLogicConfig(**dict(_mapping(config["model"], "model")))
        optimization = _mapping(config["optimization"], "optimization")
        original_test_items, original_external_items = test_items, external_items
        test_origins = external_origins = None
        if optimization.get("patient_baseline_coordinates", False):
            train_items, _ = patient_baseline_coordinates(train_items)
            test_items, test_origins = patient_baseline_coordinates(test_items)
            if external_items:
                external_items, external_origins = patient_baseline_coordinates(external_items)
        pretraining = _mapping(config["pretraining"], "pretraining")
        parent_items = train_items
        if optimization.get("completed_transition_supervision", False):
            train_items = treatment_stage_inputs(train_items, same_regimen_history=shared.recovery_same_phase)
            test_items = treatment_stage_inputs(test_items, same_regimen_history=shared.recovery_same_phase)
            if external_items:
                external_items = treatment_stage_inputs(external_items, same_regimen_history=shared.recovery_same_phase)
            parent_items = train_items + completed_training_episodes(train_items)
        programs = fit_response_programs(parent_items, shared)
        training_response_scale = _training_response_scale(programs, shared)
        bandwidth = estimate_bandwidth(parent_items, int(config["seed"]) + outer_fold)
        seed = int(config["seed"]) + 10_000 + outer_fold
        _set_seed(seed)
        set_model, pretraining_audit = _build_set(shared, pretraining, programs,
            head_initialization=optimization.get("response_head_initialization", "native"),
            initialization_device=device)
        set_state, _, set_history = _fit_stage(
            set_model,
            parent_items,
            device,
            epochs=selected_epochs["set_transport"],
            learning_rate=float(optimization["set_transport_learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
            bandwidth=bandwidth,
            response_scale=training_response_scale,
            seed=seed,
        )
        set_parameter_audit = shared_model_parameter_audit(set_model)
        projected = build_shared_projected_set(shared, set_state)
        projected_state, _, projected_history = _fit_stage(
            projected,
            parent_items,
            device,
            epochs=selected_epochs["projected_set"],
            learning_rate=float(optimization["projected_set_learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
            bandwidth=bandwidth,
            response_scale=training_response_scale,
            seed=seed + 1,
        )
        projected.load_state_dict(projected_state, strict=True)
        projected_parameter_audit = shared_model_parameter_audit(projected)
        meta = build_shared_meta_adapter(shared, projected)
        set_response_programs(meta, programs)
        architecture_audit = validate_shared_architecture(config) if config.get("shared_architecture") else None
        if architecture_audit is not None:
            if recovery_gate is not None:
                raise ValueError("Unified architecture cannot use a fitted external recovery gate")
            meta._strict_shared_forward = True
        meta_parameter_audit = shared_model_parameter_audit(meta)
        meta_state, _, meta_history = _fit_stage(
            meta,
            train_items,
            device,
            epochs=selected_epochs["meta_adapter"],
            learning_rate=float(optimization_choice["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
            bandwidth=bandwidth,
            response_scale=(np.ones_like(training_response_scale)
                if optimization.get("meta_adapter_raw_response_loss", False)
                else training_response_scale),
            seed=seed + 2,
            batch_size=optimization.get("meta_adapter_batch_size"),
            output_heads_only=optimization.get("meta_adapter_output_heads_only", False),
            parameter_l2_weight=float(optimization.get("meta_adapter_parameter_l2", 0.0)),
            history_negative_sampling=optimization.get("history_negative_sampling", "random_patient"),
            raw_mean_mae_weight=float(optimization.get("meta_adapter_raw_mean_mae_weight", 0.0)),
            parameter_ema_decay=float(optimization.get("meta_adapter_parameter_ema_decay", 0.0)),
        )
        meta_parameter_audit = shared_model_parameter_audit(meta)
        meta.load_state_dict(meta_state, strict=True)
        fold_root = Path(str(_mapping(config["data"], "data")["output_root"])) / f"fold_{outer_fold}"
        bundles = {
            "ImmuneTwins (no digital twin)": (projected, "correct", None),
            "ImmuneTwins (full)": (meta, "correct", recovery_gate),
            "ImmuneTwins (shuffled history)": (meta, "shuffled", recovery_gate),
        }
        if test_origins is not None:
            bundles = {"ImmuneTwins (full)": (meta, "correct", None)}
        prediction_splits = [("test", test_items)]
        if external_items:
            prediction_splits.append(("external", external_items))
        for split, items in prediction_splits:
            for method, (candidate, mode, candidate_gate) in bundles.items():
                predicted = _predict(
                    candidate,
                    items,
                    device,
                    history_mode=mode,
                    seed=seed + (1 if split == "test" else 2),
                    recovery_gate=candidate_gate,
                )
                origins = test_origins if split == "test" else external_origins
                scoring_items = items
                if origins is not None:
                    predicted = predicted + origins[:, None, :]
                    scoring_items = original_test_items if split == "test" else original_external_items
                save_prediction_bundle(
                    fold_root / "predictions" / method.replace(" ", "_").replace("/", "_") / f"{split}.npz",
                    method,
                    split,
                    scoring_items,
                    predicted,
                    outer_fold,
                )
        checkpoint_path = fold_root / "shared_model_checkpoint.pt"
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA,
            "outer_fold": int(outer_fold),
            "seed": seed,
            "shared_architecture": "CIMA->SetTransport->ProjectedSet->MetaAdapter",
            "patient_baseline_coordinate_IO": bool(optimization.get("patient_baseline_coordinates", False)),
            "completed_transition_supervision": bool(optimization.get("completed_transition_supervision", False)),
            "model_config": shared.to_dict(),
            "selected_epochs": selected_epochs,
            "optimization_choice": optimization_choice,
            "set_transport_state_dict": set_state,
            "projected_set_state_dict": projected_state,
            "meta_adapter_state_dict": meta_state,
            "response_programs": {
                key: value
                for key, value in programs.items()
                if key != "fit_patients"
            },
            "response_program_fit_patients": programs["fit_patients"],
            "recovery_gate": recovery_gate,
            "shared_architecture_audit": architecture_audit,
            "bandwidth": bandwidth,
            "pretraining": pretraining_audit,
            "train_patients": sorted(item.patient_id for item in train_items),
            "test_patients": sorted(item.patient_id for item in test_items),
            "parameter_audit": {
                "set_transport": set_parameter_audit,
                "projected_set": projected_parameter_audit,
                "meta_adapter": meta_parameter_audit,
            },
            "history": {
                "set_transport": set_history,
                "projected_set": projected_history,
                "meta_adapter": meta_history,
            },
        }
        _atomic_torch(checkpoint_path, checkpoint)
        result = {
            "schema_version": "immunetwins_drug_shared_fold_run_v2",
            "status": "PASS",
            "inner_selection_reference": str(selection_reference) if selection_reference is not None else None,
            "outer_fold": int(outer_fold),
            "train_patients": len(train_items),
            "test_patients": len(test_items),
            "external_patients": len(external_items),
            "patient_disjoint": not bool(
                {item.patient_id for item in train_items}
                & {item.patient_id for item in test_items}
            ),
            "selected_epochs": selected_epochs,
            "optimization_choice": optimization_choice,
            "inner_selection": inner_records,
            "recovery_gate": recovery_gate,
            "shared_architecture_audit": architecture_audit,
            "bandwidth": bandwidth,
            "checkpoint": str(checkpoint_path.resolve()),
            "pretraining": pretraining_audit,
            "parameter_audit": checkpoint["parameter_audit"],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "process_peak_rss_mib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            ),
        }
        _atomic_json(fold_root / "model_run.json", result)
        return result
    finally:
        data.close()

def audit_pipeline(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    data = _mapping(config["data"], "data")
    pretraining = _mapping(config["pretraining"], "pretraining")
    paths = {
        name: Path(str(data[name])).resolve(strict=True)
        for name in ("visits", "transitions", "episodes", "uce_store", "raw_count_store")
    }
    checkpoint = Path(str(pretraining["checkpoint"])).resolve(strict=True)
    transitions = validate_transitions(pd.read_csv(paths["transitions"], sep="\t"))
    episodes = pd.read_csv(paths["episodes"], sep="\t")
    episode_audit = validate_episodes(
        episodes, transitions, folds=int(config["outer_folds"])
    )
    shared = SharedVaccineLogicConfig(**dict(_mapping(config["model"], "model")))
    set_model, cima = build_shared_set_transport(
        shared,
        checkpoint,
    )
    _configure_set_trainable(set_model)
    result = {
        "schema_version": "immunetwins_drug_shared_audit_v2",
        "status": "PASS",
        "config": str(Path(config_path).resolve()),
        "data": {name: {"path": str(path)} for name, path in paths.items()},
        "episodes": episode_audit,
        "cima": cima,
        "shared_architecture": "CIMA->SetTransport->ProjectedSet->MetaAdapter",
        "set_transport_parameter_audit": shared_model_parameter_audit(set_model),
        "cells_per_visit": int(data["cells_per_visit"]),
        "independent_unit": "patient",
        "sound_or_vaccine_assets_referenced": any(
            "sound" in str(data[name]).lower() or "vaccine" in str(data[name]).lower()
            for name in ("visits", "transitions", "episodes", "uce_store", "raw_count_store")
        ),
        "formal_paper_baselines": list(FORMAL_METHODS[:3]),
    }
    output = Path(str(data["output_root"]))
    _atomic_json(output / "pipeline_audit.json", result)
    return result

def _method_directory(method: str) -> str:
    if method in {"scGen", "CellOT", "trVAE"}:
        return method
    return method.replace(" ", "_").replace("/", "_")

def _load_bundle(root: Path, fold: int, method: str, split: str) -> dict[str, np.ndarray]:
    path = (
        root
        / f"fold_{fold}"
        / "predictions"
        / _method_directory(method)
        / f"{split}.npz"
    )
    with np.load(path.resolve(strict=True), allow_pickle=False) as payload:
        if str(payload["schema_version"]) != "immunetwins_drug_prediction_bundle_v2":
            raise ValueError(f"prediction bundle schema mismatch: {path}")
        if str(payload["method"]) != method or str(payload["split"]) != split:
            raise ValueError(f"prediction bundle identity mismatch: {path}")
        return {name: payload[name].copy() for name in payload.files}

def _mmd2(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    denominator = 2.0 * float(bandwidth) ** 2

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distance = (
            np.sum(np.square(left), axis=1)[:, None]
            + np.sum(np.square(right), axis=1)[None, :]
            - 2.0 * left @ right.T
        )
        return np.exp(-np.maximum(distance, 0.0) / denominator)

    return float(max(kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean(), 0.0))

def _response_reference(items: Sequence[EpisodeArrays]) -> dict[str, np.ndarray | float]:
    means = np.stack([item.target_latent.mean(axis=0) for item in items])
    labels = np.asarray([item.response for item in items], dtype=np.float64)
    if set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("outer-training response readout requires both response classes")
    scale = np.maximum(means.std(axis=0), 0.05)
    center0 = means[labels == 0].mean(axis=0)
    center1 = means[labels == 1].mean(axis=0)

    def raw(values: np.ndarray) -> np.ndarray:
        distance0 = np.mean(np.square((values - center0) / scale), axis=1)
        distance1 = np.mean(np.square((values - center1) / scale), axis=1)
        return distance0 - distance1

    training_scores = raw(means)
    temperature = max(float(np.std(training_scores)), 0.1)
    return {
        "nonresponder_center": center0,
        "responder_center": center1,
        "scale": scale,
        "temperature": temperature,
    }

def _response_probability(predicted_mean: np.ndarray, reference: Mapping[str, Any]) -> np.ndarray:
    center0 = np.asarray(reference["nonresponder_center"])
    center1 = np.asarray(reference["responder_center"])
    scale = np.asarray(reference["scale"])
    distance0 = np.mean(np.square((predicted_mean - center0) / scale), axis=1)
    distance1 = np.mean(np.square((predicted_mean - center1) / scale), axis=1)
    logits = (distance0 - distance1) / float(reference["temperature"])
    logits = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))

def _score_bundle(
    payload: Mapping[str, np.ndarray],
    method: str,
    fold: int,
    bandwidth: float,
    response_reference: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    source = np.asarray(payload["source"], dtype=np.float32)
    observed = np.asarray(payload["observed"], dtype=np.float32)
    predicted = np.asarray(payload["predicted"], dtype=np.float32)
    if source.shape != observed.shape or source.shape != predicted.shape:
        raise ValueError(f"{method} prediction arrays do not share shape")
    observed_delta = observed.mean(axis=1) - source.mean(axis=1)
    predicted_delta = predicted.mean(axis=1) - source.mean(axis=1)
    residual = predicted_delta - observed_delta
    denominator = np.linalg.norm(predicted_delta, axis=1) * np.linalg.norm(
        observed_delta, axis=1
    )
    cosine = np.divide(
        np.sum(predicted_delta * observed_delta, axis=1),
        denominator,
        out=np.zeros(len(denominator), dtype=np.float64),
        where=denominator > 1e-8,
    )
    probabilities = (
        _response_probability(predicted.mean(axis=1), response_reference)
        if response_reference is not None
        else np.full(len(predicted), np.nan)
    )
    return [
        {
            "patient_id": str(payload["patient_id"][index]),
            "transition_id": str(payload["transition_id"][index]),
            "outer_fold": int(fold),
            "method": method,
            "response": float(payload["response"][index]),
            "response_probability": float(probabilities[index]),
            "nonresponse_probability": float(1.0 - probabilities[index]),
            "MMD2": _mmd2(predicted[index], observed[index], bandwidth),
            "Delta_RMSE": float(np.sqrt(np.mean(np.square(residual[index])))),
            "Delta_MAE": float(np.mean(np.abs(residual[index]))),
            "Delta_cosine": float(cosine[index]),
        }
        for index in range(len(predicted))
    ]

def _metric_estimate(frame: pd.DataFrame, metric: str) -> float | None:
    if metric in {"MMD2", "Delta_RMSE", "Delta_MAE", "Delta_cosine"}:
        return float(frame[metric].mean())
    labeled = frame["response"].notna()
    labels = frame.loc[labeled, "response"].to_numpy(float)
    if len(np.unique(labels)) < 2:
        return None
    probability = frame.loc[labeled, "response_probability"].to_numpy(float)
    if metric == "Response_AUROC":
        return float(roc_auc_score(labels, probability))
    if metric == "Response_AUPRC":
        return float(average_precision_score(labels, probability))
    if metric == "Response_Brier":
        return float(brier_score_loss(labels, probability))
    if metric == "Nonresponse_AUROC":
        return float(roc_auc_score(1.0 - labels, 1.0 - probability))
    if metric == "Nonresponse_AUPRC":
        return float(average_precision_score(1.0 - labels, 1.0 - probability))
    raise ValueError(f"unknown metric {metric}")

def _bootstrap_metrics(
    per_patient: pd.DataFrame,
    methods: Sequence[str],
    metrics: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    records = []
    for method_index, method in enumerate(methods):
        frame = per_patient.loc[per_patient["method"].eq(method)].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"no patient predictions for {method}")
        estimate = {metric: _metric_estimate(frame, metric) for metric in metrics}
        rng = np.random.default_rng(seed + method_index)
        draws: dict[str, list[float]] = {metric: [] for metric in metrics}
        for _ in range(resamples):
            sampled = frame.iloc[rng.integers(0, len(frame), size=len(frame))]
            for metric in metrics:
                value = _metric_estimate(sampled, metric)
                if value is not None and math.isfinite(value):
                    draws[metric].append(value)
        for metric in metrics:
            values = np.asarray(draws[metric], dtype=float)
            records.append(
                {
                    "method": method,
                    "metric": metric,
                    "estimate": estimate[metric],
                    "ci_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                    "ci_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                    "patients": int(frame["patient_id"].nunique()),
                }
            )
    return pd.DataFrame(records)

def _wide_table(long: pd.DataFrame, method_order: Sequence[str]) -> pd.DataFrame:
    value = long.copy()
    value["display"] = value.apply(
        lambda row: (
            "NA"
            if pd.isna(row["estimate"])
            else f"{row['estimate']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        ),
        axis=1,
    )
    result = value.pivot(index="method", columns="metric", values="display")
    result = result.reindex(method_order).reset_index()
    result.columns.name = None
    return result

def _paired_history_table(frame: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    metrics = ("MMD2", "Delta_RMSE", "Delta_MAE", "Delta_cosine")
    comparisons = (
        ("ImmuneTwins (full)", "ImmuneTwins (no digital twin)"),
        ("ImmuneTwins (full)", "ImmuneTwins (shuffled history)"),
    )
    rows = []
    for left, right in comparisons:
        joined = frame.loc[frame["method"].isin([left, right])].pivot(
            index="patient_id", columns="method", values=list(metrics)
        )
        rng = np.random.default_rng(seed + len(rows))
        for metric in metrics:
            difference = joined[(metric, left)].to_numpy() - joined[(metric, right)].to_numpy()
            draws = np.asarray(
                [
                    difference[rng.integers(0, len(difference), size=len(difference))].mean()
                    for _ in range(resamples)
                ]
            )
            rows.append(
                {
                    "comparison": f"{left} minus {right}",
                    "metric": metric,
                    "estimate": float(difference.mean()),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "favorable_direction": "negative" if metric != "Delta_cosine" else "positive",
                }
            )
    return pd.DataFrame(rows)

def _paired_method_table(
    frame: pd.DataFrame,
    *,
    left: str,
    rights: Sequence[str],
    metrics: Sequence[str],
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Patient-paired differences, including non-decomposable classification metrics."""

    rows = []
    for right_index, right in enumerate(rights):
        left_frame = frame.loc[frame["method"].eq(left)].sort_values("patient_id")
        right_frame = frame.loc[frame["method"].eq(right)].sort_values("patient_id")
        if left_frame["patient_id"].tolist() != right_frame["patient_id"].tolist():
            raise RuntimeError(f"patient order differs for paired comparison with {right}")
        left_frame = left_frame.reset_index(drop=True)
        right_frame = right_frame.reset_index(drop=True)
        rng = np.random.default_rng(seed + right_index)
        for metric in metrics:
            left_estimate = _metric_estimate(left_frame, metric)
            right_estimate = _metric_estimate(right_frame, metric)
            if left_estimate is None or right_estimate is None:
                estimate = np.nan
            else:
                estimate = left_estimate - right_estimate
            draws = []
            for _ in range(resamples):
                indices = rng.integers(0, len(left_frame), size=len(left_frame))
                left_value = _metric_estimate(left_frame.iloc[indices], metric)
                right_value = _metric_estimate(right_frame.iloc[indices], metric)
                if left_value is not None and right_value is not None:
                    draws.append(left_value - right_value)
            values = np.asarray(draws, dtype=float)
            rows.append(
                {
                    "comparison": f"{left} minus {right}",
                    "metric": metric,
                    "estimate": float(estimate),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "favorable_direction": (
                        "negative"
                        if metric in {"MMD2", "Delta_RMSE", "Response_Brier"}
                        else "positive"
                    ),
                }
            )
    return pd.DataFrame(rows)

def _episode_representation(
    items: Sequence[EpisodeArrays], kind: str
) -> np.ndarray:
    values = []
    for item in items:
        history_delta = (
            item.support_target_latent[0].mean(axis=0)
            - item.support_source_latent[0].mean(axis=0)
        )
        future_delta = item.target_latent.mean(axis=0) - item.source_latent.mean(axis=0)
        choices = {
            "current state": item.source_latent.mean(axis=0),
            "completed history delta": history_delta,
            "observed future state (oracle)": item.target_latent.mean(axis=0),
            "observed future delta (oracle)": future_delta,
        }
        if kind not in choices:
            raise ValueError(f"unknown episode representation: {kind}")
        values.append(choices[kind])
    return np.stack(values).astype(np.float32)

def _centroid_readout_reference(
    values: np.ndarray, labels: np.ndarray
) -> dict[str, np.ndarray | float]:
    if set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("centroid readout requires both response classes")
    scale = np.maximum(values.std(axis=0), 0.05)
    center0 = values[labels == 0].mean(axis=0)
    center1 = values[labels == 1].mean(axis=0)
    distance0 = np.mean(np.square((values - center0) / scale), axis=1)
    distance1 = np.mean(np.square((values - center1) / scale), axis=1)
    return {
        "nonresponder_center": center0,
        "responder_center": center1,
        "scale": scale,
        "temperature": max(float(np.std(distance0 - distance1)), 0.1),
    }

def _history_signal_summary(
    alignment: pd.DataFrame, *, resamples: int, seed: int
) -> dict[str, Any]:
    cosine = alignment["history_future_cosine"].to_numpy(float)
    history_norm = alignment["history_delta_norm"].to_numpy(float)
    future_norm = alignment["future_delta_norm"].to_numpy(float)
    rng = np.random.default_rng(seed)
    cosine_draws, norm_draws = [], []
    for _ in range(resamples):
        indices = rng.integers(0, len(alignment), size=len(alignment))
        cosine_draws.append(float(cosine[indices].mean()))
        if np.std(history_norm[indices]) > 0 and np.std(future_norm[indices]) > 0:
            norm_draws.append(
                float(np.corrcoef(history_norm[indices], future_norm[indices])[0, 1])
            )
    return {
        "patients": int(alignment["patient_id"].nunique()),
        "mean_history_future_cosine": float(cosine.mean()),
        "mean_history_future_cosine_ci_low": float(np.quantile(cosine_draws, 0.025)),
        "mean_history_future_cosine_ci_high": float(np.quantile(cosine_draws, 0.975)),
        "median_history_future_cosine": float(np.median(cosine)),
        "patients_with_positive_history_future_cosine": int((cosine > 0).sum()),
        "history_future_norm_pearson": float(
            np.corrcoef(history_norm, future_norm)[0, 1]
        ),
        "history_future_norm_pearson_ci_low": float(np.quantile(norm_draws, 0.025)),
        "history_future_norm_pearson_ci_high": float(np.quantile(norm_draws, 0.975)),
        "interpretation_boundary": (
            "C1-to-C3 mFOLFOX6 history versus C3-to-C5 "
            "mFOLFOX6-plus-pembrolizumab future response"
        ),
    }

def summarize_results(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = Path(str(_mapping(config["data"], "data")["output_root"]))
    folds = int(config["outer_folds"])
    methods = FORMAL_METHODS + DIAGNOSTIC_METHODS
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    readout_kinds = (
        "current state",
        "completed history delta",
        "observed future state (oracle)",
        "observed future delta (oracle)",
    )
    source_audits: dict[str, list[dict[str, Any]]] = {name: [] for name in FORMAL_METHODS[:3]}
    for fold in range(folds):
        run = expand_paths(json.loads((root / f"fold_{fold}" / "model_run.json").read_text()))
        if run.get("status") != "PASS":
            raise RuntimeError(f"model fold {fold} is incomplete")
        data = DrugFoldData(config_path, fold)
        try:
            train_items = data.arrays(data.rows("train"))
            test_items = data.arrays(data.rows("test"))
            reference = _response_reference(train_items)
            train_labels = np.asarray([item.response for item in train_items])
            for kind in readout_kinds:
                readout_reference = _centroid_readout_reference(
                    _episode_representation(train_items, kind), train_labels
                )
                probabilities = _response_probability(
                    _episode_representation(test_items, kind), readout_reference
                )
                readout_rows.extend(
                    {
                        "patient_id": item.patient_id,
                        "outer_fold": int(fold),
                        "method": kind,
                        "response": float(item.response),
                        "response_probability": float(probability),
                    }
                    for item, probability in zip(test_items, probabilities)
                )
            for item in test_items:
                history_delta = _episode_representation(
                    [item], "completed history delta"
                )[0]
                future_delta = _episode_representation(
                    [item], "observed future delta (oracle)"
                )[0]
                denominator = float(
                    np.linalg.norm(history_delta) * np.linalg.norm(future_delta)
                )
                alignment_rows.append(
                    {
                        "patient_id": item.patient_id,
                        "outer_fold": int(fold),
                        "response": float(item.response),
                        "history_future_cosine": float(
                            np.dot(history_delta, future_delta) / max(denominator, 1e-12)
                        ),
                        "history_delta_norm": float(np.linalg.norm(history_delta)),
                        "future_delta_norm": float(np.linalg.norm(future_delta)),
                    }
                )
        finally:
            data.close()
        bandwidth = float(run["bandwidth"])
        for method in methods:
            if method == "Persistence":
                for split, target in (("test", internal_rows), ("external", external_rows)):
                    template = _load_bundle(root, fold, "ImmuneTwins (no digital twin)", split)
                    template["predicted"] = template["source"].copy()
                    target.extend(
                        _score_bundle(
                            template,
                            method,
                            fold,
                            bandwidth,
                            reference if split == "test" else None,
                        )
                    )
                continue
            internal_rows.extend(
                _score_bundle(
                    _load_bundle(root, fold, method, "test"),
                    method,
                    fold,
                    bandwidth,
                    reference,
                )
            )
            external_rows.extend(
                _score_bundle(
                    _load_bundle(root, fold, method, "external"),
                    method,
                    fold,
                    bandwidth,
                    None,
                )
            )
            if method in source_audits:
                source_audits[method].append(
                    expand_paths(json.loads(
                        (root / f"fold_{fold}" / "baselines" / method / "run.json").read_text()
                    ))
                )
    internal = pd.DataFrame(internal_rows)
    if internal.groupby(["method", "patient_id"]).size().ne(1).any():
        raise RuntimeError("internal OOF table is not one row per method and patient")
    external_fold = pd.DataFrame(external_rows)
    external = (
        external_fold.groupby(["method", "patient_id"], as_index=False)[
            ["MMD2", "Delta_RMSE", "Delta_MAE", "Delta_cosine"]
        ]
        .mean()
    )
    resamples = int(_mapping(config["evaluation"], "evaluation")["bootstrap_resamples"])
    seed = int(config["seed"])
    internal_metrics = (
        "MMD2",
        "Delta_RMSE",
        "Delta_MAE",
        "Delta_cosine",
        "Response_AUROC",
        "Response_AUPRC",
        "Response_Brier",
        "Nonresponse_AUROC",
        "Nonresponse_AUPRC",
    )
    external_metrics = ("MMD2", "Delta_RMSE", "Delta_MAE", "Delta_cosine")
    internal_long = _bootstrap_metrics(
        internal, methods, internal_metrics, resamples=resamples, seed=seed
    )
    external_long = _bootstrap_metrics(
        external, methods, external_metrics, resamples=resamples, seed=seed + 100
    )
    formal_internal = internal_long.loc[internal_long["method"].isin(FORMAL_METHODS)]
    formal_external = external_long.loc[external_long["method"].isin(FORMAL_METHODS)]
    history = _paired_history_table(internal, resamples, seed + 200)
    paired_comparators = _paired_method_table(
        internal,
        left="ImmuneTwins (full)",
        rights=(
            "scGen",
            "CellOT",
            "trVAE",
            "ImmuneTwins (no digital twin)",
        ),
        metrics=(
            "MMD2",
            "Delta_RMSE",
            "Delta_cosine",
            "Response_AUROC",
            "Response_AUPRC",
            "Response_Brier",
        ),
        resamples=resamples,
        seed=seed + 250,
    )
    readout = pd.DataFrame(readout_rows)
    readout_long = _bootstrap_metrics(
        readout,
        readout_kinds,
        ("Response_AUROC", "Response_AUPRC", "Response_Brier"),
        resamples=resamples,
        seed=seed + 300,
    )
    alignment = pd.DataFrame(alignment_rows)
    signal = _history_signal_summary(
        alignment, resamples=resamples, seed=seed + 400
    )
    root.mkdir(parents=True, exist_ok=True)
    internal.to_csv(root / "patient_oof_metrics.tsv", sep="\t", index=False)
    external_fold.to_csv(root / "external_fold_metrics.tsv", sep="\t", index=False)
    external.to_csv(root / "external_patient_metrics.tsv", sep="\t", index=False)
    internal_long.to_csv(root / "all_methods_metric_long.tsv", sep="\t", index=False)
    external_long.to_csv(root / "external_all_methods_metric_long.tsv", sep="\t", index=False)
    formal_internal.to_csv(root / "formal_method_metric_long.tsv", sep="\t", index=False)
    formal_external.to_csv(root / "external_formal_method_metric_long.tsv", sep="\t", index=False)
    _wide_table(formal_internal, FORMAL_METHODS).to_csv(
        root / "formal_method_metric_table.tsv", sep="\t", index=False
    )
    _wide_table(formal_external, FORMAL_METHODS).to_csv(
        root / "external_formal_method_metric_table.tsv", sep="\t", index=False
    )
    _wide_table(internal_long, methods).to_csv(
        root / "all_methods_metric_table.tsv", sep="\t", index=False
    )
    history.to_csv(root / "history_mechanism_paired.tsv", sep="\t", index=False)
    paired_comparators.to_csv(
        root / "paired_full_vs_comparators.tsv", sep="\t", index=False
    )
    _wide_table(readout_long, readout_kinds).to_csv(
        root / "clinical_readout_diagnostic_table.tsv", sep="\t", index=False
    )
    alignment.to_csv(root / "history_future_alignment.tsv", sep="\t", index=False)
    _atomic_json(root / "history_signal_summary.json", signal)
    source_summary = {
        method: {
            "paper": values[0]["paper"],
            "official_source": values[0]["official_source"],
            "official_commit": values[0]["official_commit"],
            "all_folds_complete": len(values) == folds
            and all(value.get("status") == "PASS" for value in values),
        }
        for method, values in source_audits.items()
    }
    result = {
        "schema_version": "immunetwins_drug_shared_summary_v2",
        "status": "PASS",
        "development_patients": int(internal["patient_id"].nunique()),
        "external_patients": int(external["patient_id"].nunique()),
        "outer_folds": folds,
        "formal_methods": list(FORMAL_METHODS),
        "diagnostic_methods": list(DIAGNOSTIC_METHODS),
        "official_baseline_sources": source_summary,
        "common_response_readout": "outer-training target-centroid distance in the fold-local 16-D coordinate",
        "acquired_resistance_claim": False,
        "nonresponse_is_response_complement": True,
        "counterfactual_drug_ranking": False,
        "tables": {
            "formal_internal": str((root / "formal_method_metric_table.tsv").resolve()),
            "formal_external": str((root / "external_formal_method_metric_table.tsv").resolve()),
            "all_methods": str((root / "all_methods_metric_table.tsv").resolve()),
            "history_mechanism": str((root / "history_mechanism_paired.tsv").resolve()),
            "paired_full_vs_comparators": str(
                (root / "paired_full_vs_comparators.tsv").resolve()
            ),
            "clinical_readout_diagnostic": str(
                (root / "clinical_readout_diagnostic_table.tsv").resolve()
            ),
            "history_future_alignment": str(
                (root / "history_future_alignment.tsv").resolve()
            ),
            "history_signal_summary": str(
                (root / "history_signal_summary.json").resolve()
            ),
        },
    }
    _atomic_json(root / "summary.json", result)
    return result

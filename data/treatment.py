from __future__ import annotations
from paths import expand_paths
import hashlib
import json
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import pandas as pd

TRANSITION_COLUMNS = (
    "transition_id",
    "cohort_id",
    "patient_id",
    "source_visit_id",
    "target_visit_id",
    "source_day",
    "target_day",
    "drug_id",
    "dose",
    "response",
)

EPISODE_COLUMNS = TRANSITION_COLUMNS + ("outer_fold", "support_transition_ids")

def validate_transitions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(TRANSITION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"transition table lacks columns: {sorted(missing)}")
    value = frame.copy()
    for column in (
        "transition_id",
        "cohort_id",
        "patient_id",
        "source_visit_id",
        "target_visit_id",
        "drug_id",
    ):
        value[column] = value[column].astype(str)
        if value[column].str.strip().eq("").any():
            raise ValueError(f"transition column {column} contains blank values")
    if value["transition_id"].duplicated().any():
        raise ValueError("transition_id must be unique")
    if value.groupby("patient_id")["cohort_id"].nunique().gt(1).any():
        raise ValueError("patient_id must be globally cohort-qualified")
    for column in ("source_day", "target_day", "dose", "response"):
        value[column] = pd.to_numeric(value[column], errors="coerce")
    numeric = value[["source_day", "target_day", "dose"]]
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("source_day, target_day and dose must be finite")
    if (value["target_day"] <= value["source_day"]).any():
        raise ValueError("every treatment transition must point forward in time")
    if (value["dose"] < 0).any():
        raise ValueError("dose must be non-negative")
    observed = value["response"].dropna()
    if not observed.isin([0.0, 1.0]).all():
        raise ValueError("observed response labels must be 0/1")
    return value.sort_values(
        ["patient_id", "source_day", "target_day", "transition_id"]
    ).reset_index(drop=True)

def validate_episodes(
    episodes: pd.DataFrame, transitions: pd.DataFrame, *, folds: int | None = None
) -> dict[str, Any]:
    missing = set(EPISODE_COLUMNS) - set(episodes.columns)
    if missing:
        raise ValueError(f"episode table lacks columns: {sorted(missing)}")
    source = validate_transitions(transitions).set_index("transition_id", drop=False)
    if episodes["transition_id"].astype(str).duplicated().any():
        raise ValueError("episode query transitions must be unique")
    patient_to_fold: dict[str, int] = {}
    support_counts: list[int] = []
    for row in episodes.itertuples(index=False):
        query = source.loc[str(row.transition_id)]
        patient, fold = str(row.patient_id), int(row.outer_fold)
        if patient_to_fold.setdefault(patient, fold) != fold:
            raise ValueError("one patient appears in multiple outer folds")
        support_ids = expand_paths(json.loads(str(row.support_transition_ids)))
        if not isinstance(support_ids, list) or len(support_ids) != len(set(support_ids)):
            raise ValueError("support_transition_ids must be a unique JSON list")
        support_counts.append(len(support_ids))
        for transition_id in support_ids:
            support = source.loc[str(transition_id)]
            if str(support["patient_id"]) != patient:
                raise ValueError("support transition belongs to another patient")
            if float(support["target_day"]) > float(query["source_day"]):
                raise ValueError("support transition was not completed before the query")
    observed_folds = sorted(set(patient_to_fold.values()))
    if folds is not None and observed_folds != list(range(folds)):
        raise ValueError(f"outer folds must be 0..{folds - 1}, got {observed_folds}")
    return {
        "schema_version": "immunetwins_drug_episodes_v1",
        "status": "PASS",
        "patients": len(patient_to_fold),
        "episodes": len(episodes),
        "folds": observed_folds,
        "episodes_with_history": int(sum(value > 0 for value in support_counts)),
        "maximum_support_transitions": int(max(support_counts, default=0)),
        "patient_disjoint": True,
        "support_is_same_patient_completed_history": True,
        "future_query_target_in_support": False,
    }

def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]

def _seed(seed: int, *values: str) -> int:
    payload = "\0".join([str(seed), *map(str, values)]).encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**32)

class UceVisitStore:
    """Read a [visit, cell, 1280] UCE feature store."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve(strict=True)
        self.handle = h5py.File(self.path, "r")
        if self.handle.attrs.get("status") != "complete":
            self.close()
            raise ValueError("UCE visit store is not marked complete")
        if "embedding" not in self.handle or "visit_id" not in self.handle:
            self.close()
            raise ValueError("UCE store requires embedding and visit_id datasets")
        shape = self.handle["embedding"].shape
        if len(shape) != 3 or shape[2] != 1280 or shape[1] < 1:
            self.close()
            raise ValueError("UCE embedding must be [visit,cell,1280]")
        visit_ids = _decode(self.handle["visit_id"][:])
        if len(visit_ids) != shape[0] or len(visit_ids) != len(set(visit_ids)):
            self.close()
            raise ValueError("UCE visit identifiers are not unique/aligned")
        self.index = {value: position for position, value in enumerate(visit_ids)}
        self.input_dim = shape[2]
        self.cells_available = shape[1]

    def take(self, visit_id: str, *, cells: int, seed: int) -> np.ndarray:
        if visit_id not in self.index:
            raise KeyError(f"visit absent from UCE store: {visit_id}")
        values = np.asarray(self.handle["embedding"][self.index[visit_id]], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite UCE values for visit {visit_id}")
        rng = np.random.default_rng(_seed(seed, visit_id))
        positions = rng.choice(len(values), size=cells, replace=len(values) < cells)
        return values[positions]

    def close(self) -> None:
        if getattr(self, "handle", None) is not None:
            self.handle.close()
            self.handle = None

    def __del__(self) -> None:
        self.close()

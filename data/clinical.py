from __future__ import annotations
from paths import expand_paths
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Collection, Iterable, Mapping
import numpy as np
import pandas as pd
import yaml

DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "config" / "history_fields.yaml"

class ClinicalContractError(ValueError):
    """Raised when raw inputs or requested supports are invalid."""

class ClinicalLeakageError(ClinicalContractError):
    """Raised when a future-derived field is presented as a candidate input."""

@dataclass(frozen=True)
class VisitFeatureTable:
    """Visit-level metadata, numeric values and aligned missingness masks."""

    metadata: pd.DataFrame
    values: pd.DataFrame
    missing: pd.DataFrame

    def __post_init__(self) -> None:
        if not self.metadata.index.is_unique:
            raise ClinicalContractError("sample_id must be unique after visit aggregation")
        if not self.metadata.index.equals(self.values.index):
            raise ClinicalContractError("metadata and values must have identical sample_id order")
        if not self.values.index.equals(self.missing.index) or not self.values.columns.equals(
            self.missing.columns
        ):
            raise ClinicalContractError("values and missing masks must have identical shape")
        expected = self.values.isna()
        if not expected.equals(self.missing.astype(bool)):
            raise ClinicalContractError("missing mask must describe pre-imputation values")
        for required in ("donor_id", "visit_name"):
            if required not in self.metadata:
                raise ClinicalContractError(f"metadata is missing {required!r}")

def load_history_field_contract(path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        contract = expand_paths(yaml.safe_load(handle))
    if not isinstance(contract, dict) or "sources" not in contract or "tasks" not in contract:
        raise ClinicalContractError("invalid vaccine history field contract")
    scope = contract.get("leakage_guard", {}).get("preprocessing_fit_scope")
    if scope != "outer_training_donors_only":
        raise ClinicalContractError("preprocessing fit scope must be outer_training_donors_only")
    return contract

def _reject_future_derived_columns(columns: Iterable[str], contract: Mapping[str, Any]) -> None:
    guard = contract.get("leakage_guard", {})
    exact = {str(value).casefold() for value in guard.get("reject_exact_columns", [])}
    patterns = [re.compile(str(value)) for value in guard.get("reject_column_name_patterns", [])]
    rejected = []
    for raw_name in columns:
        name = str(raw_name)
        if name.casefold() in exact or any(pattern.search(name) for pattern in patterns):
            rejected.append(name)
    if rejected:
        raise ClinicalLeakageError(
            "future-derived columns are forbidden clinical inputs: " + ", ".join(sorted(rejected))
        )

def _required_headers(source: Mapping[str, Any], contract: Mapping[str, Any]) -> tuple[str, ...]:
    ids = contract["identifiers"]
    required = {
        str(ids["sample_id"]),
        str(ids["donor_id"]),
        str(ids["visit_name"]),
        *(str(value) for value in source.get("required_headers", [])),
    }
    if source.get("layout") == "wide":
        required.update(str(value) for value in source.get("numeric_fields", []))
    elif source.get("layout") == "antigen_long":
        required.add(str(source["category_field"]))
        required.update(str(value) for value in source.get("category_fallback_fields", []))
        required.update(str(value) for value in source.get("measurement_fields", {}))
    return tuple(sorted(required))

def audit_source_headers(
    source_name: str,
    columns: Iterable[str],
    contract: Mapping[str, Any] | None = None,
) -> None:
    """Validate an actual raw-file header before reading candidate features."""

    contract = dict(contract or load_history_field_contract())
    if source_name not in contract["sources"]:
        raise ClinicalContractError(f"unknown source {source_name!r}")
    columns = tuple(str(value) for value in columns)
    _reject_future_derived_columns(columns, contract)
    missing = sorted(set(_required_headers(contract["sources"][source_name], contract)) - set(columns))
    if missing:
        raise ClinicalContractError(f"{source_name} is missing required raw headers: {missing}")

def _metadata_for_source(frame: pd.DataFrame, contract: Mapping[str, Any]) -> pd.DataFrame:
    ids = contract["identifiers"]
    sample_col, donor_col, visit_col = ids["sample_id"], ids["donor_id"], ids["visit_name"]
    work = frame[[sample_col, donor_col, visit_col]].dropna(subset=[sample_col]).copy()
    work[sample_col] = work[sample_col].astype(str)
    for column in (donor_col, visit_col):
        counts = work.groupby(sample_col, sort=False)[column].nunique(dropna=True)
        if (counts > 1).any():
            bad = counts[counts > 1].index.tolist()[:5]
            raise ClinicalContractError(f"sample_id maps to multiple {column} values: {bad}")
    metadata = work.groupby(sample_col, sort=False)[[donor_col, visit_col]].first()
    metadata.index.name = "sample_id"
    return metadata.rename(columns={donor_col: "donor_id", visit_col: "visit_name"})

def _transform(values: pd.Series, method: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if method == "identity":
        return numeric
    if method == "log1p":
        if (numeric.dropna() < 0).any():
            raise ClinicalContractError("log1p was requested for a feature containing negative values")
        return np.log1p(numeric)
    if method == "signed_log1p":
        return np.sign(numeric) * np.log1p(np.abs(numeric))
    raise ClinicalContractError(f"unsupported numeric transform {method!r}")

def _aggregate_wide(
    frame: pd.DataFrame,
    source_name: str,
    source: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    sample_col = contract["identifiers"]["sample_id"]
    work = frame.dropna(subset=[sample_col]).copy()
    work[sample_col] = work[sample_col].astype(str)
    transform = str(source.get("transform", "identity"))
    outputs: dict[str, pd.Series] = {}
    for raw_field in source.get("numeric_fields", []):
        feature = f"{source_name}::{raw_field}"
        numeric = _transform(work[str(raw_field)], transform)
        outputs[feature] = numeric.groupby(work[sample_col], sort=False).mean()
    result = pd.DataFrame(outputs)
    result.index.name = "sample_id"
    return result

def _infer_antigen_category(
    frame: pd.DataFrame,
    source: Mapping[str, Any],
) -> pd.Series:
    categories = [str(value) for value in source.get("categories", [])]
    category_lookup = {value.casefold(): value for value in categories}
    raw = frame[str(source["category_field"])].astype("string")
    normalized = raw.str.strip().str.casefold().map(category_lookup)
    fallback = pd.Series("", index=frame.index, dtype="string")
    for field in source.get("category_fallback_fields", []):
        fallback = fallback.str.cat(frame[str(field)].fillna("").astype(str), sep=" ")
    for category in categories:
        unresolved = normalized.isna() & fallback.str.contains(re.escape(category), case=False, na=False)
        normalized.loc[unresolved] = category
    return normalized

def _aggregate_antigen_long(
    frame: pd.DataFrame,
    source_name: str,
    source: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    sample_col = contract["identifiers"]["sample_id"]
    work = frame.dropna(subset=[sample_col]).copy()
    work[sample_col] = work[sample_col].astype(str)
    excluded = {str(value).casefold() for value in source.get("exclude_antigen_names", [])}
    antigen_name = work.get("msd.antigenName", pd.Series("", index=work.index)).astype(str).str.casefold()
    work = work.loc[~antigen_name.isin(excluded)].copy()
    work["__category"] = _infer_antigen_category(work, source)
    work = work.dropna(subset=["__category"])

    categories = [str(value) for value in source.get("categories", [])]
    transform = str(source.get("transform", "identity"))
    outputs: dict[str, pd.Series] = {}
    for raw_field, short_name in source.get("measurement_fields", {}).items():
        work["__value"] = _transform(work[str(raw_field)], transform)
        grouped = work.groupby([sample_col, "__category"], sort=False)["__value"].mean().unstack()
        for category in categories:
            feature = f"{source_name}::{category}::{short_name}"
            outputs[feature] = (
                grouped[category] if category in grouped else pd.Series(dtype=float, name=feature)
            )
        if source.get("include_overall_mean", False):
            feature = f"{source_name}::overall::{short_name}"
            outputs[feature] = work.groupby(sample_col, sort=False)["__value"].mean()
    result = pd.DataFrame(outputs)
    result.index.name = "sample_id"
    return result

def build_visit_features_from_frames(
    labs: pd.DataFrame,
    hai: pd.DataFrame,
    igg: pd.DataFrame,
    *,
    contract: Mapping[str, Any] | None = None,
) -> VisitFeatureTable:
    """Aggregate raw Sound frames to one row per sample_id."""

    contract = dict(contract or load_history_field_contract())
    frames = {"labs": labs, "hai": hai, "igg": igg}
    metadata_parts = []
    value_parts = []
    for source_name, frame in frames.items():
        audit_source_headers(source_name, frame.columns, contract)
        metadata_parts.append(_metadata_for_source(frame, contract))
        source = contract["sources"][source_name]
        if source["layout"] == "wide":
            value_parts.append(_aggregate_wide(frame, source_name, source, contract))
        elif source["layout"] == "antigen_long":
            value_parts.append(_aggregate_antigen_long(frame, source_name, source, contract))
        else:
            raise ClinicalContractError(f"unsupported layout {source['layout']!r}")

    stacked_metadata = pd.concat(metadata_parts)
    for column in ("donor_id", "visit_name"):
        counts = stacked_metadata.groupby(level=0)[column].nunique(dropna=True)
        if (counts > 1).any():
            bad = counts[counts > 1].index.tolist()[:5]
            raise ClinicalContractError(f"raw sources disagree on {column} for samples {bad}")
    metadata = stacked_metadata.groupby(level=0, sort=True).first()
    metadata.index.name = "sample_id"
    values = pd.concat(value_parts, axis=1).reindex(metadata.index)
    values = values.astype(float)
    return VisitFeatureTable(metadata=metadata, values=values, missing=values.isna())

def build_sound_visit_features(
    labs_path: str | Path,
    hai_path: str | Path,
    igg_path: str | Path,
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    nrows: int | None = None,
) -> VisitFeatureTable:
    """Read the three raw Sound files and build aligned visit-level features."""

    contract = load_history_field_contract(contract_path)
    frames = {}
    for name, path in {"labs": labs_path, "hai": hai_path, "igg": igg_path}.items():
        header = pd.read_csv(path, nrows=0).columns
        audit_source_headers(name, header, contract)
        frames[name] = pd.read_csv(path, nrows=nrows, low_memory=False)
    return build_visit_features_from_frames(**frames, contract=contract)

class OuterTrainingVisitPreprocessor:
    """Median imputation and z-scoring fitted only on outer-training donors."""

    def __init__(self, *, min_train_observations: int = 1) -> None:
        if min_train_observations < 1:
            raise ValueError("min_train_observations must be positive")
        self.min_train_observations = int(min_train_observations)
        self.feature_names_: tuple[str, ...] | None = None
        self.medians_: pd.Series | None = None
        self.means_: pd.Series | None = None
        self.scales_: pd.Series | None = None
        self.outer_training_donors_: frozenset[str] | None = None
        self.n_training_visits_: int | None = None

    def fit(
        self,
        table: VisitFeatureTable,
        *,
        outer_training_donors: Collection[str],
    ) -> "OuterTrainingVisitPreprocessor":
        _reject_future_derived_columns(table.values.columns, load_history_field_contract())
        donors = frozenset(str(value) for value in outer_training_donors)
        if not donors:
            raise ClinicalContractError("outer_training_donors cannot be empty")
        train_mask = table.metadata["donor_id"].astype(str).isin(donors)
        train = table.values.loc[train_mask]
        if train.empty:
            raise ClinicalContractError("no visit rows belong to the requested outer-training donors")
        counts = train.notna().sum(axis=0)
        active = counts[counts >= self.min_train_observations].index
        if active.empty:
            raise ClinicalContractError("no clinical feature is observed in outer-training donors")
        train = train.loc[:, active]
        medians = train.median(axis=0, skipna=True)
        imputed = train.fillna(medians)
        means = imputed.mean(axis=0)
        scales = imputed.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        self.feature_names_ = tuple(str(value) for value in active)
        self.medians_, self.means_, self.scales_ = medians, means, scales
        self.outer_training_donors_ = donors
        self.n_training_visits_ = int(train.shape[0])
        return self

    def transform(self, table: VisitFeatureTable) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise ClinicalContractError("preprocessor must be fitted on outer-training donors first")
        _reject_future_derived_columns(table.values.columns, load_history_field_contract())
        assert self.medians_ is not None and self.means_ is not None and self.scales_ is not None
        absent = sorted(set(self.feature_names_) - set(table.values.columns))
        if absent:
            raise ClinicalContractError(f"transform table is missing fitted features: {absent}")
        raw = table.values.loc[:, self.feature_names_]
        missing = raw.isna().astype(np.float32)
        scaled = ((raw.fillna(self.medians_) - self.means_) / self.scales_).astype(np.float32)
        scaled.columns = [f"value::{name}" for name in self.feature_names_]
        missing.columns = [f"missing::{name}" for name in self.feature_names_]
        result = pd.concat([scaled, missing], axis=1)
        result.index.name = "sample_id"
        return result

    def fit_transform(
        self,
        table: VisitFeatureTable,
        *,
        outer_training_donors: Collection[str],
    ) -> pd.DataFrame:
        return self.fit(table, outer_training_donors=outer_training_donors).transform(table)

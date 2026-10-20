from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import h5py
import numpy as np
import torch
from data.lineage_summary import DATA1_ROOT, LINEAGES, SCHEMA as LINEAGE_SUMMARY_SCHEMA
from data.sound import VaccineUnit

PREPROCESSOR_SCHEMA = "sound_lineage_gene_preprocessor_1"

FEATURE_BATCH_SCHEMA = "sound_lineage_gene_feature_batch_1"

FIT_STAGES = {"selection", "outer_train_refit"}

MAX_GENES = 128

def _decode(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)

def _decode_array(values: np.ndarray) -> np.ndarray:
    return np.asarray([_decode(value) for value in values], dtype=str)


def _require_data1(path: str | Path, *, role: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        resolved.relative_to(DATA1_ROOT)
    except ValueError as error:
        raise ValueError(f"{role} must remain under {DATA1_ROOT}: {resolved}") from error
    return resolved

def _embedding_lookup(path: Path) -> tuple[dict[str, np.ndarray], int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("UCE gene embedding artifact must be a dictionary")
    lookup: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        array = np.asarray(torch.as_tensor(value, dtype=torch.float32).reshape(-1))
        if not len(array):
            continue
        lookup.setdefault(str(key).upper(), array.astype(np.float32, copy=False))
    if not lookup:
        raise ValueError("UCE gene embedding artifact is empty")
    widths = {len(value) for value in lookup.values()}
    if len(widths) != 1:
        raise ValueError("UCE gene embeddings do not share one width")
    return lookup, int(next(iter(widths)))


def _validate_units_before_reads(
    units: Sequence[VaccineUnit],
    store: "LineageGeneSummaryStore",
    *,
    require_support: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not units:
        raise ValueError("at least one vaccine unit is required")
    tasks = {str(unit.task) for unit in units}
    if len(tasks) != 1:
        raise ValueError("lineage gene preprocessing must be run separately per task")
    if len({unit.unit_id for unit in units}) != len(units):
        raise ValueError("vaccine unit ids are not unique")

    target_samples = {str(unit.target_sample) for unit in units}
    support_samples: set[str] = set()
    support_count = 0
    for unit in units:
        if require_support and not unit.supports:
            raise ValueError(f"unit has no completed history support: {unit.unit_id}")
        for transition in unit.supports:
            support_count += 1
            source = str(transition.source_sample)
            endpoint = str(transition.target_sample)
            if source == endpoint:
                raise ValueError(f"support transition has identical endpoints: {unit.unit_id}")
            if str(unit.target_sample) in {source, endpoint}:
                raise ValueError(
                    f"completed support contains the forecast target: {unit.unit_id}"
                )
            support_samples.update((source, endpoint))
            for sample in (source, endpoint):
                if sample not in store.index:
                    raise KeyError(f"support sample is absent from lineage store: {sample}")
                if store.donor_by_sample[sample] != str(unit.donor_id):
                    raise ValueError(
                        f"support sample {sample} does not belong to donor {unit.donor_id}"
                    )
    if require_support and support_count == 0:
        raise ValueError("no completed support transitions were supplied")
    overlap = target_samples & support_samples
    if overlap:
        raise ValueError(
            "a forecast target appears in the support endpoint set: "
            f"{sorted(overlap)[:5]}"
        )
    return tuple(sorted(support_samples)), tuple(sorted(target_samples))

class LineageGeneSummaryStore:
    """Reader for the all-cell lineage summary."""

    def __init__(self, path: str | Path, *, allow_pilot: bool = False):
        self.path = _require_data1(path, role="lineage gene summary")
        self.handle = h5py.File(self.path, "r")
        try:
            if self.handle.attrs.get("schema_version") != LINEAGE_SUMMARY_SCHEMA:
                raise ValueError("lineage gene summary schema changed")
            if self.handle.attrs.get("status") != "complete":
                raise ValueError("lineage gene summary is incomplete")
            if self.handle.attrs.get("normalization") != "none_raw_sum_counts":
                raise ValueError("lineage gene summary is not raw sum counts")
            if self.handle.attrs.get("all_available_cells", False) not in (
                True,
                np.bool_(True),
            ):
                raise ValueError("lineage gene summary does not use all available cells")
            if bool(self.handle.attrs.get("cross_visit_gene_selection", True)):
                raise ValueError("lineage gene summary performed cross-visit selection")
            if bool(self.handle.attrs.get("development_only", False)) and not allow_pilot:
                raise ValueError("pilot lineage summary requires allow_pilot=True")
            observed_lineages = tuple(_decode_array(self.handle["lineage/name"][:]))
            if observed_lineages != LINEAGES:
                raise ValueError("lineage order differs from the frozen five-lineage contract")
            self.genes = _decode_array(self.handle["var/gene_symbol"][:])
            self.samples = _decode_array(self.handle["sample_id"][:])
            self.donors = _decode_array(self.handle["donor_id"][:])
            if len(self.genes) != len(set(self.genes.tolist())):
                raise ValueError("lineage gene summary duplicates gene symbols")
            if len(self.samples) != len(set(self.samples.tolist())):
                raise ValueError("lineage gene summary duplicates sample ids")
            if len(self.samples) != len(self.donors):
                raise ValueError("lineage sample and donor metadata lengths differ")
            self.index = {sample: index for index, sample in enumerate(self.samples)}
            self.donor_by_sample = dict(zip(self.samples.tolist(), self.donors.tolist()))
            expected_gene_shape = (len(self.samples), len(LINEAGES), len(self.genes))
            expected_visit_shape = (len(self.samples), len(LINEAGES))
            if self.handle["sum_counts"].shape != expected_gene_shape:
                raise ValueError("lineage sum_counts shape changed")
            if self.handle["detection_fraction"].shape != expected_gene_shape:
                raise ValueError("lineage detection_fraction shape changed")
            if self.handle["n_cells"].shape != expected_visit_shape:
                raise ValueError("lineage n_cells shape changed")
            if self.handle["total_umi"].shape != expected_visit_shape:
                raise ValueError("lineage total_umi shape changed")
            if self.handle["sum_counts"].dtype != np.dtype("uint64"):
                raise ValueError("lineage sum_counts must be uint64")
            if self.handle["detection_fraction"].dtype != np.dtype("float32"):
                raise ValueError("lineage detection_fraction must be float32")
        except Exception:
            self.handle.close()
            raise
        self._read_samples: list[str] = []


    @property
    def read_samples(self) -> tuple[str, ...]:
        return tuple(self._read_samples)

    def clear_read_log(self) -> None:
        self._read_samples.clear()

    def read_logcpm(
        self, sample_ids: Sequence[str], gene_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read only requested samples and return logCPM, detection, n_cells."""

        gene_indices = np.asarray(gene_indices, dtype=np.int64)
        if not len(gene_indices):
            raise ValueError("at least one gene index is required")
        if np.any(gene_indices < 0) or np.any(gene_indices >= len(self.genes)):
            raise ValueError("gene index is outside the lineage summary")
        if not np.all(gene_indices[1:] > gene_indices[:-1]):
            raise ValueError("gene indices must be strictly increasing")
        rows: list[int] = []
        for sample in sample_ids:
            value = str(sample)
            if value not in self.index:
                raise KeyError(f"sample is absent from lineage summary: {value}")
            rows.append(self.index[value])

        counts = np.empty(
            (len(rows), len(LINEAGES), len(gene_indices)), dtype=np.float32
        )
        detection = np.empty_like(counts)
        n_cells = np.empty((len(rows), len(LINEAGES)), dtype=np.int64)
        total_umi = np.empty((len(rows), len(LINEAGES)), dtype=np.float64)
        for output_index, (sample, row) in enumerate(zip(sample_ids, rows)):
            for lineage_index in range(len(LINEAGES)):
                counts[output_index, lineage_index] = np.asarray(
                    self.handle["sum_counts"][row, lineage_index, gene_indices],
                    dtype=np.float32,
                )
                detection[output_index, lineage_index] = np.asarray(
                    self.handle["detection_fraction"][
                        row, lineage_index, gene_indices
                    ],
                    dtype=np.float32,
                )
            n_cells[output_index] = np.asarray(
                self.handle["n_cells"][row], dtype=np.int64
            )
            total_umi[output_index] = np.asarray(
                self.handle["total_umi"][row], dtype=np.float64
            )
            self._read_samples.append(str(sample))
        if (
            not np.isfinite(counts).all()
            or not np.isfinite(detection).all()
            or not np.isfinite(total_umi).all()
            or (counts < 0).any()
            or (detection < 0).any()
            or (detection > 1).any()
            or (n_cells < 0).any()
            or (total_umi < 0).any()
        ):
            raise ValueError("lineage summary slice contains invalid values")
        logcpm = np.log1p(
            counts * (1_000_000.0 / np.maximum(total_umi[:, :, None], 1.0))
        ).astype(np.float32)
        return logcpm, detection, n_cells

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "LineageGeneSummaryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

@dataclass(frozen=True)
class LineageGenePreprocessor:
    stage: str
    gene_names: tuple[str, ...]
    raw_gene_indices: np.ndarray
    gene_embeddings: np.ndarray
    delta_mean: np.ndarray
    delta_scale: np.ndarray
    delta_variance: np.ndarray
    mean_detection: np.ndarray
    selection_score: np.ndarray
    selection_rank: np.ndarray
    min_mean_detection: float
    scale_floor: float
    source_h5_path: str
    gene_embedding_path: str
    provenance: Mapping[str, object]

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)


@dataclass(frozen=True)
class LineageGeneFeatureBatch:
    unit_ids: tuple[str, ...]
    lineages: tuple[str, ...]
    gene_names: tuple[str, ...]
    delta: np.ndarray
    raw_delta_logcpm: np.ndarray
    confidence: np.ndarray
    feature_mask: np.ndarray
    lineage_mask: np.ndarray
    support_mask: np.ndarray
    support_phase: np.ndarray
    support_dt: np.ndarray
    provenance: Mapping[str, object]

def fit_lineage_gene_preprocessor(
    *,
    store: LineageGeneSummaryStore,
    units: Sequence[VaccineUnit],
    fit_donors: Iterable[str],
    gene_embedding_path: str | Path,
    stage: str,
    max_genes: int = MAX_GENES,
    min_mean_detection: float = 0.01,
    scale_floor: float = 1.0e-3,
) -> LineageGenePreprocessor:
    """Fit selection and delta scaling from the supplied donors' supports only."""

    if stage not in FIT_STAGES:
        raise ValueError(f"stage must be one of {sorted(FIT_STAGES)}")
    if not 1 <= max_genes <= MAX_GENES:
        raise ValueError(f"max_genes must be in [1, {MAX_GENES}]")
    if not 0 <= min_mean_detection <= 1:
        raise ValueError("min_mean_detection must be in [0, 1]")
    if not np.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor must be finite and positive")

    donor_set = {str(value) for value in fit_donors}
    if not donor_set:
        raise ValueError("fit donor set is empty")
    fit_units = sorted(
        (unit for unit in units if str(unit.donor_id) in donor_set),
        key=lambda unit: unit.unit_id,
    )
    observed_donors = {str(unit.donor_id) for unit in fit_units}
    if observed_donors != donor_set:
        raise ValueError(
            "fit donors and supplied units differ; "
            f"missing units for {sorted(donor_set - observed_donors)[:5]}"
        )
    support_samples, target_samples = _validate_units_before_reads(
        fit_units, store, require_support=True
    )
    embedding_file = _require_data1(gene_embedding_path, role="UCE gene embeddings")
    lookup, embedding_dim = _embedding_lookup(embedding_file)
    supported_indices = np.asarray(
        [
            index
            for index, gene in enumerate(store.genes)
            if str(gene).upper() in lookup
        ],
        dtype=np.int64,
    )
    if not len(supported_indices):
        raise ValueError("no lineage-summary genes have UCE embeddings")

    width = len(supported_indices)
    # The paper-level unit is a donor, not a transition.  T2 contains one or
    # two seasons per donor, so accumulating transitions directly would give a
    # two-season donor twice the influence on gene selection and scaling.  We
    # first average all legal support transitions within each donor and only
    # then estimate the across-donor moments.
    delta_sum = np.zeros((len(LINEAGES), width), dtype=np.float64)
    delta_square_sum = np.zeros_like(delta_sum)
    confidence_sum = np.zeros_like(delta_sum)
    lineage_observations = np.zeros(len(LINEAGES), dtype=np.int64)
    transition_count = 0
    donor_transition_counts: dict[str, int] = {}
    store.clear_read_log()
    for donor in sorted(donor_set):
        donor_delta_sum = np.zeros_like(delta_sum)
        donor_confidence_sum = np.zeros_like(delta_sum)
        donor_lineage_observations = np.zeros(len(LINEAGES), dtype=np.int64)
        donor_transitions = 0
        for unit in (value for value in fit_units if str(value.donor_id) == donor):
            for transition in unit.supports:
                logcpm, detection, n_cells = store.read_logcpm(
                    [transition.source_sample, transition.target_sample],
                    supported_indices,
                )
                delta = logcpm[1].astype(np.float64) - logcpm[0].astype(np.float64)
                confidence = np.maximum(detection[0], detection[1]).astype(np.float64)
                valid_lineages = (n_cells[0] > 0) & (n_cells[1] > 0)
                for lineage_index in np.flatnonzero(valid_lineages):
                    donor_delta_sum[lineage_index] += delta[lineage_index]
                    donor_confidence_sum[lineage_index] += confidence[lineage_index]
                    donor_lineage_observations[lineage_index] += 1
                donor_transitions += 1
                transition_count += 1
        if donor_transitions < 1:
            raise ValueError(f"fit donor has no completed support transition: {donor}")
        donor_transition_counts[donor] = donor_transitions
        valid_donor_lineages = donor_lineage_observations > 0
        donor_denominator = donor_lineage_observations[:, None].clip(min=1)
        donor_delta = donor_delta_sum / donor_denominator
        donor_confidence = donor_confidence_sum / donor_denominator
        for lineage_index in np.flatnonzero(valid_donor_lineages):
            delta_sum[lineage_index] += donor_delta[lineage_index]
            delta_square_sum[lineage_index] += donor_delta[lineage_index] ** 2
            confidence_sum[lineage_index] += donor_confidence[lineage_index]
            lineage_observations[lineage_index] += 1
    if transition_count == 0 or not np.all(lineage_observations >= 2):
        raise ValueError(
            "gene preprocessing needs at least two valid fit donors per lineage"
        )
    accessed_samples = tuple(sorted(set(store.read_samples)))
    if set(accessed_samples) != set(support_samples):
        raise RuntimeError("feature reads differ from the declared support endpoint set")
    if set(accessed_samples) & set(target_samples):
        raise RuntimeError("a forecast target was read during gene preprocessing")

    denominators = lineage_observations[:, None].astype(np.float64)
    means = delta_sum / denominators
    variances = np.maximum(delta_square_sum / denominators - means**2, 0.0)
    mean_detection = confidence_sum / denominators
    eligible_by_lineage = mean_detection >= float(min_mean_detection)
    score_by_lineage = np.where(
        eligible_by_lineage,
        variances * np.sqrt(np.maximum(mean_detection, 0.0)),
        -np.inf,
    )
    scores = score_by_lineage.max(axis=0)
    eligible = np.isfinite(scores) & (scores > 0)
    eligible_indices = np.flatnonzero(eligible)
    if not len(eligible_indices):
        raise ValueError("no UCE-supported gene passed detection and variance selection")
    symbols = store.genes[supported_indices]
    ranked_local = sorted(
        eligible_indices.tolist(), key=lambda index: (-float(scores[index]), str(symbols[index]))
    )[:max_genes]
    rank_by_local = {local: rank for rank, local in enumerate(ranked_local)}
    selected_supported_local = np.asarray(ranked_local, dtype=np.int64)
    selected_raw = supported_indices[selected_supported_local]
    raw_order = np.argsort(selected_raw)
    selected_supported_local = selected_supported_local[raw_order]
    selected_raw = selected_raw[raw_order]
    selected_names = tuple(store.genes[selected_raw].astype(str).tolist())
    embeddings = np.stack(
        [lookup[name.upper()] for name in selected_names]
    ).astype(np.float32)
    if embeddings.shape != (len(selected_names), embedding_dim):
        raise RuntimeError("selected UCE gene embedding shape changed")
    selected_mean = means[:, selected_supported_local].astype(np.float32)
    selected_variance = variances[:, selected_supported_local].astype(np.float32)
    selected_scale = np.sqrt(
        np.maximum(selected_variance.astype(np.float64), float(scale_floor) ** 2)
    ).astype(np.float32)
    selected_detection = mean_detection[:, selected_supported_local].astype(np.float32)
    selected_score = scores[selected_supported_local].astype(np.float32)
    selected_rank = np.asarray(
        [rank_by_local[int(value)] for value in selected_supported_local], dtype=np.int64
    )

    donor_values = tuple(sorted(donor_set))
    fit_scope = (
        "inner_fit_donors_only"
        if stage == "selection"
        else "outer_training_donors_only"
    )
    provenance: dict[str, object] = {
        "status": "pass",
        "schema_version": PREPROCESSOR_SCHEMA,
        "stage": stage,
        "fit_scope": fit_scope,
        "fit_donors": len(donor_values),
        "fit_units": len(fit_units),
        "fit_transitions": transition_count,
        "fit_transition_aggregation": (
            "mean_support_transitions_within_donor_then_equal_donor"
        ),
        "fit_donor_transition_counts": donor_transition_counts,
        "fit_support_samples": len(support_samples),
        "forecast_targets_excluded_from_feature_reads": True,
        "feature_read_samples": list(accessed_samples),
        "lineage_summary_path": str(store.path),
        "gene_embedding_path": str(embedding_file),
        "UCE_supported_genes": int(len(supported_indices)),
        "eligible_genes": int(len(eligible_indices)),
        "selected_genes": len(selected_names),
        "maximum_genes": max_genes,
        "min_mean_detection": float(min_mean_detection),
        "scale_floor": float(scale_floor),
        "lineages": list(LINEAGES),
        "lineage_observations": {
            lineage: int(lineage_observations[index])
            for index, lineage in enumerate(LINEAGES)
        },
        "lineage_observation_unit": "donor",
        "selection": [
            {
                "gene_symbol": selected_names[index],
                "raw_gene_index": int(selected_raw[index]),
                "selection_rank": int(selected_rank[index]),
                "selection_score": float(selected_score[index]),
                "maximum_lineage_variance": float(selected_variance[:, index].max()),
                "maximum_mean_detection": float(selected_detection[:, index].max()),
            }
            for index in range(len(selected_names))
        ],
    }
    result = LineageGenePreprocessor(
        stage=stage,
        gene_names=selected_names,
        raw_gene_indices=selected_raw,
        gene_embeddings=embeddings,
        delta_mean=selected_mean,
        delta_scale=selected_scale,
        delta_variance=selected_variance,
        mean_detection=selected_detection,
        selection_score=selected_score,
        selection_rank=selected_rank,
        min_mean_detection=float(min_mean_detection),
        scale_floor=float(scale_floor),
        source_h5_path=str(store.path),
        gene_embedding_path=str(embedding_file),
        provenance=provenance,
    )
    return result

def refit_outer_training_lineage_gene_preprocessor(
    *,
    store: LineageGeneSummaryStore,
    units: Sequence[VaccineUnit],
    outer_training_donors: Iterable[str],
    gene_embedding_path: str | Path,
    max_genes: int = MAX_GENES,
    min_mean_detection: float = 0.01,
    scale_floor: float = 1.0e-3,
) -> LineageGenePreprocessor:
    """Refit the preprocessing on all outer-training donors."""

    return fit_lineage_gene_preprocessor(
        store=store,
        units=units,
        fit_donors=outer_training_donors,
        gene_embedding_path=gene_embedding_path,
        stage="outer_train_refit",
        max_genes=max_genes,
        min_mean_detection=min_mean_detection,
        scale_floor=scale_floor,
    )

def materialize_lineage_gene_features(
    *,
    store: LineageGeneSummaryStore,
    preprocessor: LineageGenePreprocessor,
    units: Sequence[VaccineUnit],
) -> LineageGeneFeatureBatch:
    """Materialize padded support tensors."""

    if str(store.path) != preprocessor.source_h5_path:
        raise ValueError("preprocessor was fitted from a different lineage summary")
    if tuple(store.genes[preprocessor.raw_gene_indices].astype(str)) != (
        preprocessor.gene_names
    ):
        raise ValueError("selected gene order differs from the lineage summary")
    support_samples, target_samples = _validate_units_before_reads(
        units, store, require_support=True
    )
    maximum_support = max(len(unit.supports) for unit in units)
    shape = (len(units), maximum_support, len(LINEAGES), preprocessor.n_genes)
    raw_delta = np.zeros(shape, dtype=np.float32)
    delta = np.zeros(shape, dtype=np.float32)
    confidence = np.zeros(shape, dtype=np.float32)
    feature_mask = np.zeros(shape, dtype=bool)
    lineage_mask = np.zeros(shape[:-1], dtype=bool)
    support_mask = np.zeros(shape[:2], dtype=bool)
    support_phase = np.full(shape[:2], -1, dtype=np.int64)
    support_dt = np.zeros(shape[:2], dtype=np.float32)

    store.clear_read_log()
    for unit_index, unit in enumerate(units):
        for support_index, transition in enumerate(unit.supports):
            logcpm, detected, n_cells = store.read_logcpm(
                [transition.source_sample, transition.target_sample],
                preprocessor.raw_gene_indices,
            )
            value = logcpm[1] - logcpm[0]
            valid_lineages = (n_cells[0] > 0) & (n_cells[1] > 0)
            standardized = (
                value - preprocessor.delta_mean
            ) / preprocessor.delta_scale
            evidence = np.maximum(detected[0], detected[1])
            # A feature is evidence only when the lineage exists at both
            # endpoints and the gene is detected at least once.  Otherwise a
            # raw zero would become a non-zero standardized value after
            # subtracting the fit-donor mean and would consume softmax mass.
            valid_features = valid_lineages[:, None] & (evidence > 0.0)
            standardized[~valid_features] = 0.0
            value[~valid_features] = 0.0
            evidence[~valid_features] = 0.0
            raw_delta[unit_index, support_index] = value
            delta[unit_index, support_index] = standardized
            confidence[unit_index, support_index] = evidence
            feature_mask[unit_index, support_index] = valid_features
            lineage_mask[unit_index, support_index] = valid_lineages
            support_mask[unit_index, support_index] = True
            support_phase[unit_index, support_index] = int(transition.phase)
            support_dt[unit_index, support_index] = float(transition.elapsed_days)
    accessed_samples = tuple(sorted(set(store.read_samples)))
    if set(accessed_samples) != set(support_samples):
        raise RuntimeError("materialized reads differ from declared support endpoints")
    if set(accessed_samples) & set(target_samples):
        raise RuntimeError("feature materialization read a forecast target")
    if not (
        np.isfinite(raw_delta).all()
        and np.isfinite(delta).all()
        and np.isfinite(confidence).all()
    ):
        raise ValueError("materialized lineage gene tensors contain non-finite values")
    provenance: dict[str, object] = {
        "status": "pass",
        "schema_version": FEATURE_BATCH_SCHEMA,
        "preprocessor_stage": preprocessor.stage,
        "units": len(units),
        "maximum_support": maximum_support,
        "lineages": list(LINEAGES),
        "genes": preprocessor.n_genes,
        "tensor_shape": list(shape),
        "support_samples": len(support_samples),
        "feature_read_samples": list(accessed_samples),
        "forecast_target_samples": len(target_samples),
        "forecast_targets_excluded_from_feature_reads": True,
        "delta_transform": "fit_donor_logCPM_delta_center_and_scale",
        "confidence": "maximum_endpoint_detection_fraction",
        "feature_mask": (
            "both_endpoints_have_lineage_cells_and_gene_detected_at_either_endpoint"
        ),
    }
    return LineageGeneFeatureBatch(
        unit_ids=tuple(unit.unit_id for unit in units),
        lineages=LINEAGES,
        gene_names=preprocessor.gene_names,
        delta=delta,
        raw_delta_logcpm=raw_delta,
        confidence=confidence,
        feature_mask=feature_mask,
        lineage_mask=lineage_mask,
        support_mask=support_mask,
        support_phase=support_phase,
        support_dt=support_dt,
        provenance=provenance,
    )

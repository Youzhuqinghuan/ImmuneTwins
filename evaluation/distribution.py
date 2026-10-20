from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from evaluation.units import ScoreConfig, _bootstrap_mean, _draw_indices, _resolve_artifact
from evaluation.controls import _stable_seed

TASKS = (
    "t1_y1_d0_to_y1_d7",
    "t2_d7_to_d90",
    "t3_y2_d0_to_y2_d90",
)

LINEAGES = ("B", "CD4_T", "CD8_T", "NK", "Myeloid")

RESPONSE_DIMENSIONS = 16

@dataclass(frozen=True)
class PaperScoreConfig:
    evaluation: ScoreConfig
    retrieval_permutations: int = 500
    lineage_draws: int = 5
    lineage_cells: int = 64
    lineage_min_cells: int = 32

def rbf_mmd2_biased(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return non-negative biased RBF MMD squared for batched cell sets.

    Inputs have shape ``draws x cells x features`` and are already divided by
    the outer-training-only bandwidth.  The biased estimator is used because it
    is non-negative and exactly zero for identical empirical distributions.
    """

    if left.ndim != 3 or left.shape != right.shape:
        raise ValueError("MMD inputs must share draws x cells x features")
    kernel_xx = torch.exp(-0.5 * torch.cdist(left, left).square()).mean((1, 2))
    kernel_yy = torch.exp(-0.5 * torch.cdist(right, right).square()).mean((1, 2))
    kernel_xy = torch.exp(-0.5 * torch.cdist(left, right).square()).mean((1, 2))
    return torch.clamp(kernel_xx + kernel_yy - 2.0 * kernel_xy, min=0.0)

def _disjoint_draw_indices(
    length: int,
    config: ScoreConfig,
    *identity: object,
    role: str,
) -> tuple[np.ndarray, np.ndarray]:
    required = 2 * config.cells_per_draw
    if length < required:
        raise RuntimeError(
            f"observed distribution has {length} cells; {required} are required "
            "for a disjoint observed-vs-observed noise floor"
        )
    rng = np.random.default_rng(
        _stable_seed("evaluation", *identity, role, base_seed=config.technical_seed)
    )
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for _ in range(config.evaluation_draws):
        selected = rng.choice(length, size=required, replace=False)
        left.append(selected[: config.cells_per_draw])
        right.append(selected[config.cells_per_draw :])
    return np.stack(left), np.stack(right)

def _state_to_lineage(state: str) -> str:
    value = str(state)
    if "B cell" in value or "Plasma" in value:
        return "B"
    if "CD4" in value or value == "Treg":
        return "CD4_T"
    if "CD8" in value or "MAIT" in value:
        return "CD8_T"
    if "NK" in value:
        return "NK"
    if (
        "monocyte" in value
        or value.startswith(("cDC", "pDC", "ASDC"))
        or "dendritic" in value.casefold()
    ):
        return "Myeloid"
    raise RuntimeError(f"cell state is not covered by the five-lineage contract: {value}")

def _lineages(states: np.ndarray) -> np.ndarray:
    return np.asarray([_state_to_lineage(value) for value in states], dtype="U8")

def _load_distribution(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        embedding = np.asarray(payload["embedding"], dtype=np.float32)
        states = np.asarray(payload["cell_state"]).astype(str)
    if embedding.ndim != 2 or embedding.shape[1] != RESPONSE_DIMENSIONS:
        raise RuntimeError(f"invalid state embedding: {path}")
    if len(states) != len(embedding):
        raise RuntimeError(f"cell-state annotation length differs from embeddings: {path}")
    if not np.isfinite(embedding).all():
        raise RuntimeError(f"non-finite state embedding: {path}")
    return embedding, _lineages(states)

def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return np.nan
    return float(np.dot(left, right) / denominator)

def _composition_l1(predicted: np.ndarray, observed: np.ndarray) -> float:
    pred = np.asarray([(predicted == lineage).mean() for lineage in LINEAGES])
    obs = np.asarray([(observed == lineage).mean() for lineage in LINEAGES])
    if not math.isclose(float(pred.sum()), 1.0, abs_tol=1e-8):
        raise RuntimeError("predicted broad-lineage composition is incomplete")
    if not math.isclose(float(obs.sum()), 1.0, abs_tol=1e-8):
        raise RuntimeError("observed broad-lineage composition is incomplete")
    return float(np.abs(pred - obs).sum())

def _lineage_scores(
    *,
    source: np.ndarray,
    predicted: np.ndarray,
    observed: np.ndarray,
    source_lineage: np.ndarray,
    predicted_lineage: np.ndarray,
    observed_lineage: np.ndarray,
    bandwidth: float,
    row: pd.Series,
    config: PaperScoreConfig,
    device: torch.device,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for lineage in LINEAGES:
        source_values = source[source_lineage == lineage]
        predicted_values = predicted[predicted_lineage == lineage]
        observed_values = observed[observed_lineage == lineage]
        available = min(len(source_values), len(predicted_values), len(observed_values))
        record: dict[str, object] = {
            "paper_method": str(row["paper_method"]),
            "paper_task": str(row["paper_task"]),
            "donor_id": str(row["donor_id"]),
            "season": str(row["season"]),
            "outer_fold": int(row["outer_fold"]),
            "age_source": float(row["age_source"]),
            "age_stratum": str(row["age_stratum"]),
            "lineage": lineage,
            "source_cells": int(len(source_values)),
            "predicted_cells": int(len(predicted_values)),
            "observed_cells": int(len(observed_values)),
        }
        if available < config.lineage_min_cells:
            record.update({"lineage_mmd2": np.nan, "lineage_delta_rmse": np.nan})
            records.append(record)
            continue
        cells = min(config.lineage_cells, available)
        identity = (
            row["paper_method"],
            row["paper_task"],
            row["donor_id"],
            row["season"],
            lineage,
        )
        sampled: list[np.ndarray] = []
        for role, values in (("lineage_pred", predicted_values), ("lineage_obs", observed_values)):
            rng = np.random.default_rng(
                _stable_seed("paper_lineage", *identity, role, base_seed=config.evaluation.technical_seed)
            )
            sampled.append(
                np.stack(
                    [rng.choice(len(values), size=cells, replace=False) for _ in range(config.lineage_draws)]
                )
            )
        pred_index, obs_index = sampled
        with torch.inference_mode():
            mmd2 = rbf_mmd2_biased(
                torch.as_tensor(predicted_values[pred_index] / bandwidth, device=device),
                torch.as_tensor(observed_values[obs_index] / bandwidth, device=device),
            )
        pred_delta = predicted_values.mean(axis=0) - source_values.mean(axis=0)
        obs_delta = observed_values.mean(axis=0) - source_values.mean(axis=0)
        record.update(
            {
                "lineage_mmd2": float(mmd2.mean().item()),
                "lineage_delta_rmse": float(np.sqrt(np.mean((pred_delta - obs_delta) ** 2))),
            }
        )
        records.append(record)
    return records

def _score_unit(
    *,
    row: pd.Series,
    plan_root: Path | tuple[Path, ...],
    center: np.ndarray,
    scale: np.ndarray,
    bandwidth: float,
    config: PaperScoreConfig,
    device: torch.device,
    observed_lower_cache: dict[tuple[str, str, str], np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    manifest_path = Path(str(row["manifest_path"]))
    paths = {
        field: _resolve_artifact(
            manifest_path=manifest_path, row=row, field=field, allowed_root=plan_root
        )
        for field in ("source_path", "predicted_path", "observed_path")
    }
    source_raw, source_lineage = _load_distribution(paths["source_path"])
    predicted_raw, predicted_lineage = _load_distribution(paths["predicted_path"])
    observed_raw, observed_lineage = _load_distribution(paths["observed_path"])
    source = (source_raw - center) / scale
    predicted = (predicted_raw - center) / scale
    observed = (observed_raw - center) / scale
    identity = (row["paper_task"], row["donor_id"], row["season"])
    source_index = _draw_indices(len(source), config.evaluation, *identity, role="source")
    predicted_index = (
        source_index.copy()
        if source.shape == predicted.shape and np.array_equal(source, predicted)
        else _draw_indices(len(predicted), config.evaluation, *identity, role="predicted")
    )
    observed_index = _draw_indices(len(observed), config.evaluation, *identity, role="observed")
    source_draws = source[source_index]
    predicted_draws = predicted[predicted_index]
    observed_draws = observed[observed_index]
    with torch.inference_mode():
        mmd2 = rbf_mmd2_biased(
            torch.as_tensor(predicted_draws / bandwidth, device=device),
            torch.as_tensor(observed_draws / bandwidth, device=device),
        ).cpu().numpy()
    lower_key = tuple(str(value) for value in identity)
    lower = observed_lower_cache.get(lower_key)
    if lower is None:
        lower_left, lower_right = _disjoint_draw_indices(
            len(observed), config.evaluation, *identity, role="observed_lower"
        )
        with torch.inference_mode():
            lower = rbf_mmd2_biased(
                torch.as_tensor(observed[lower_left] / bandwidth, device=device),
                torch.as_tensor(observed[lower_right] / bandwidth, device=device),
            ).cpu().numpy()
        observed_lower_cache[lower_key] = lower
    source_mean = source_draws.mean(axis=1)
    predicted_mean = predicted_draws.mean(axis=1)
    observed_mean = observed_draws.mean(axis=1)
    pred_response = predicted_mean - source_mean
    obs_response = observed_mean - source_mean
    composition_l1 = _composition_l1(predicted_lineage, observed_lineage)
    draw_records: list[dict[str, object]] = []
    for draw in range(config.evaluation.evaluation_draws):
        pred_norm = float(np.linalg.norm(pred_response[draw]))
        obs_norm = float(np.linalg.norm(obs_response[draw]))
        record: dict[str, object] = {
            "paper_method": str(row["paper_method"]),
            "paper_task": str(row["paper_task"]),
            "donor_id": str(row["donor_id"]),
            "season": str(row["season"]),
            "outer_fold": int(row["outer_fold"]),
            "age_source": float(row["age_source"]),
            "age_stratum": str(row["age_stratum"]),
            "draw": draw,
            "mmd2": float(mmd2[draw]),
            "observed_lower_mmd2": float(lower[draw]),
            "delta_rmse": float(np.sqrt(np.mean((pred_response[draw] - obs_response[draw]) ** 2))),
            "delta_mae": float(np.mean(np.abs(pred_response[draw] - obs_response[draw]))),
            "response_cosine": _cosine(pred_response[draw], obs_response[draw]),
            "predicted_response_norm": pred_norm,
            "observed_response_norm": obs_norm,
            "response_norm_ratio": pred_norm / obs_norm if obs_norm > 1e-12 else np.nan,
            "broad_composition_l1": composition_l1,
        }
        for index in range(RESPONSE_DIMENSIONS):
            record[f"source_mean_{index}"] = float(source_mean[draw, index])
            record[f"predicted_mean_{index}"] = float(predicted_mean[draw, index])
            record[f"observed_response_{index}"] = float(obs_response[draw, index])
            record[f"predicted_response_{index}"] = float(pred_response[draw, index])
        draw_records.append(record)
    lineage_records = _lineage_scores(
        source=source,
        predicted=predicted,
        observed=observed,
        source_lineage=source_lineage,
        predicted_lineage=predicted_lineage,
        observed_lineage=observed_lineage,
        bandwidth=bandwidth,
        row=row,
        config=config,
        device=device,
    )
    return draw_records, lineage_records

def _reciprocal_ranks(similarities: np.ndarray) -> np.ndarray:
    n = similarities.shape[0]
    result = np.empty(n, dtype=np.float64)
    for index in range(n):
        value = similarities[index, index]
        greater = int(np.sum(similarities[index] > value + 1e-12))
        tied = int(np.sum(np.isclose(similarities[index], value, atol=1e-12, rtol=0.0)))
        result[index] = 1.0 / (greater + (tied + 1) / 2.0)
    return result

def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=1, keepdims=True) * np.linalg.norm(
        right, axis=1, keepdims=True
    ).T
    return np.divide(
        left @ right.T,
        denominator,
        out=np.zeros((len(left), len(right)), dtype=np.float64),
        where=denominator > 1e-12,
    )

def add_permutation_retrieval(
    draws: pd.DataFrame, *, permutations: int, seed: int
) -> pd.DataFrame:
    result = draws.copy()
    result["response_mrr"] = np.nan
    result["response_mrr_permutation_null"] = np.nan
    result["response_mrr_excess"] = np.nan
    result["retrieval_candidate_count"] = 0
    pred_columns = [f"predicted_response_{index}" for index in range(RESPONSE_DIMENSIONS)]
    obs_columns = [f"observed_response_{index}" for index in range(RESPONSE_DIMENSIONS)]
    group_columns = ["paper_method", "outer_fold", "paper_task", "season", "age_stratum", "draw"]
    for key, group in result.groupby(group_columns, sort=False):
        predicted = group[pred_columns].to_numpy(dtype=np.float64)
        observed = group[obs_columns].to_numpy(dtype=np.float64)
        similarities = _cosine_matrix(predicted, observed)
        actual = _reciprocal_ranks(similarities)
        # Use the same candidate permutation schedule for every method in the
        # same biological evaluation stratum.  Including ``paper_method`` in
        # the random seed makes exact prediction aliases receive different
        # Monte Carlo nulls and therefore different MRR-excess scores.
        rng = np.random.default_rng(
            _stable_seed("retrieval_null", *key[1:], base_seed=seed)
        )
        null_sum = np.zeros(len(group), dtype=np.float64)
        for _ in range(permutations):
            null_sum += _reciprocal_ranks(similarities[:, rng.permutation(len(group))])
        null = null_sum / permutations
        result.loc[group.index, "response_mrr"] = actual
        result.loc[group.index, "response_mrr_permutation_null"] = null
        result.loc[group.index, "response_mrr_excess"] = actual - null
        result.loc[group.index, "retrieval_candidate_count"] = len(group)
    return result

def _aggregate_metrics(draws: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit_keys = [
        "paper_method", "paper_task", "donor_id", "season", "outer_fold", "age_source", "age_stratum"
    ]
    scalar = [
        "mmd2", "observed_lower_mmd2", "delta_rmse", "delta_mae", "response_cosine",
        "predicted_response_norm", "observed_response_norm", "response_norm_ratio",
        "broad_composition_l1", "response_mrr", "response_mrr_permutation_null",
        "response_mrr_excess", "retrieval_candidate_count",
    ]
    response_columns = [
        *(f"predicted_response_{index}" for index in range(RESPONSE_DIMENSIONS)),
        *(f"observed_response_{index}" for index in range(RESPONSE_DIMENSIONS)),
    ]
    unit = draws.groupby(unit_keys, as_index=False)[scalar + response_columns].mean()
    donor_keys = ["paper_method", "paper_task", "donor_id", "age_stratum"]
    donor = unit.groupby(donor_keys, as_index=False)[scalar].mean()
    response_unit = unit[
        ["paper_method", "paper_task", "donor_id", "season", "outer_fold", "age_stratum", *response_columns]
    ].copy()
    return unit, donor, response_unit

def _bootstrap_record(
    values: np.ndarray, *, draws: int, seed: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    point = float(values.mean())
    low, high = _bootstrap_mean(values, draws, seed)
    return point, low, high

from __future__ import annotations
from paths import expand_paths
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
import torch
import yaml
from evaluation.scales import _scales
from evaluation.distribution import PaperScoreConfig, TASKS, _aggregate_metrics, _bootstrap_record, _score_unit, add_permutation_retrieval
from evaluation.units import ScoreConfig, _atomic_json, _episode_metadata
from evaluation.controls import  _stable_seed
from pipeline.sound.common import PAPER_METHODS

PRIMARY_METRICS: dict[str, str] = {
    "mmd2": "lower",
    "delta_rmse": "lower",
    "delta_mae": "lower",
    "response_cosine": "higher",
    "response_mrr_excess": "higher",
}

EMPTY_HISTORY_REFERENCE = PAPER_METHODS["current"]

EMPTY_HISTORY_ALIASES = tuple(
    PAPER_METHODS[arm] for arm in ("w", "gene", "full")
)

def _v2_paper_score_config(path: Path) -> PaperScoreConfig:
    """Parse the parameters used by the RBF-MMD scorer."""

    payload = expand_paths(yaml.safe_load(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict) or payload.get("schema_version") != "sound_vaccine_donor_evaluation_v2":
        raise ValueError("strict v2 scoring requires the v2 donor evaluation contract")
    evaluation = payload.get("evaluation", {})
    terminal = evaluation.get("terminal_distribution", {})
    required_terminal = {
        "name": "biased_rbf_mmd_squared",
        "estimator": "biased_v_statistic",
        "nonnegative_clamp": True,
        "implementation": "evaluation.distribution.rbf_mmd2_biased",
    }
    for field, expected in required_terminal.items():
        if terminal.get(field) != expected:
            raise ValueError(f"v2 RBF-MMD contract field {field!r} changed")
    if evaluation.get("sinkhorn_reported_or_claimed") is not False:
        raise ValueError("strict v2 contract must explicitly exclude Sinkhorn claims")
    uncertainty = payload.get("uncertainty", {})
    publication = payload.get("publication_rescore", {})
    score = ScoreConfig(
        evaluation_draws=int(evaluation["evaluation_draws"]),
        cells_per_draw=int(evaluation["evaluation_cells_per_distribution_draw"]),
        technical_seed=int(evaluation["technical_draw_seed"]),
        # These ScoreConfig fields are not used by the RBF scorer.
        epsilon=ScoreConfig.epsilon,
        max_iterations=ScoreConfig.max_iterations,
        tolerance=ScoreConfig.tolerance,
        bootstrap_draws=int(uncertainty["draws"]),
        bootstrap_seed=int(uncertainty["bootstrap_seed"]),
    )
    return PaperScoreConfig(
        evaluation=score,
        retrieval_permutations=int(publication["retrieval_permutations"]),
        lineage_draws=int(publication["lineage_draws"]),
        lineage_cells=int(publication["lineage_cells"]),
        lineage_min_cells=int(publication["lineage_min_cells"]),
    )

def add_empty_history_aliases(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Reuse current-arm T1 predictions for history arms by exact construction."""

    result = predictions.copy()
    task = "t1_y1_d0_to_y1_d7"
    current = result.loc[
        result["paper_method"].eq(EMPTY_HISTORY_REFERENCE)
        & result["paper_task"].eq(task)
    ]
    methods = set(result["paper_method"].astype(str))
    aliases: dict[str, str] = {}
    for method in EMPTY_HISTORY_ALIASES:
        if method not in methods:
            continue
        existing = result.loc[
            result["paper_method"].eq(method) & result["paper_task"].eq(task)
        ]
        if len(existing):
            raise RuntimeError(
                f"{method} must reuse current T1; separately trained empty-history outputs are invalid"
            )
        if current.empty:
            raise RuntimeError(f"{method} requires {EMPTY_HISTORY_REFERENCE} T1 predictions")
        copied = current.copy()
        copied["paper_method"] = method
        result = pd.concat([result, copied], ignore_index=True)
        aliases[method] = EMPTY_HISTORY_REFERENCE
    return result, aliases

def score_predictions(
    *,
    predictions: pd.DataFrame,
    artifact_root: Path | tuple[Path, ...],
    plan_root: Path,
    assets_path: Path,
    config: PaperScoreConfig,
    device: torch.device,
    progress_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = _episode_metadata(plan_root)
    merged = predictions.merge(
        metadata,
        on=["paper_task", "donor_id", "season"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_contract"),
    )
    if merged["age_source"].isna().any():
        raise RuntimeError("an ImmuneTwin prediction is absent from task metadata")
    if not pd.to_numeric(merged["outer_fold"], errors="raise").astype(int).eq(
        pd.to_numeric(merged["outer_fold_contract"], errors="raise").astype(int)
    ).all():
        raise RuntimeError("ImmuneTwin prediction outer fold differs from task metadata")
    scales = _scales(assets_path)
    ordered = merged.sort_values(
        ["paper_method", "paper_task", "outer_fold", "donor_id", "season"],
        kind="stable",
    )
    draw_records: list[dict[str, object]] = []
    lineage_records: list[dict[str, object]] = []
    lower_cache: dict[tuple[str, str, str], np.ndarray] = {}
    for completed, (_, row) in enumerate(ordered.iterrows(), start=1):
        center, scale, bandwidth, _ = scales[int(row["outer_fold"])]
        draws, lineages = _score_unit(
            row=row,
            plan_root=artifact_root,
            center=center,
            scale=scale,
            bandwidth=bandwidth,
            config=config,
            device=device,
            observed_lower_cache=lower_cache,
        )
        draw_records.extend(draws)
        lineage_records.extend(lineages)
        if progress_path is not None and (
            completed == 1 or completed % 10 == 0 or completed == len(ordered)
        ):
            _atomic_json(
                progress_path,
                {
                    "status": "scoring" if completed < len(ordered) else "metrics_complete",
                    "completed_prediction_units": completed,
                    "total_prediction_units": len(ordered),
                    "paper_method": str(row["paper_method"]),
                    "paper_task": str(row["paper_task"]),
                },
            )
    draws = add_permutation_retrieval(
        pd.DataFrame(draw_records),
        permutations=config.retrieval_permutations,
        seed=config.evaluation.technical_seed,
    )
    return draws, pd.DataFrame(lineage_records)

def strongest_independent_paired_effects(
    donor: pd.DataFrame,
    *,
    new_methods: Iterable[str],
    independent_baselines: Sequence[str],
    config: PaperScoreConfig,
) -> pd.DataFrame:
    """Participant-paired gains versus the strongest independent baseline.

    The reference is chosen separately for each task and metric. Positive
    effects favour ImmuneTwins.
    """

    baselines = donor.loc[donor["paper_method"].isin(independent_baselines)]
    missing = set(independent_baselines) - set(baselines["paper_method"].astype(str))
    if missing:
        raise RuntimeError(f"result lacks independent baselines: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for task in TASKS:
        baseline_task = baselines.loc[baselines["paper_task"].eq(task)]
        for metric, direction in PRIMARY_METRICS.items():
            means = baseline_task.groupby("paper_method")[metric].mean()
            reference_method = (
                str(means.idxmin()) if direction == "lower" else str(means.idxmax())
            )
            reference = baseline_task.loc[
                baseline_task["paper_method"].eq(reference_method)
            ].set_index("donor_id")
            for method in new_methods:
                candidate = donor.loc[
                    donor["paper_method"].eq(method) & donor["paper_task"].eq(task)
                ].set_index("donor_id")
                if set(candidate.index.astype(str)) != set(reference.index.astype(str)):
                    raise RuntimeError(
                        f"paired donor support differs for {method}, {reference_method}, {task}"
                    )
                reference = reference.loc[candidate.index]
                left = candidate[metric].to_numpy(dtype=np.float64)
                right = reference[metric].to_numpy(dtype=np.float64)
                effect = right - left if direction == "lower" else left - right
                point, low, high = _bootstrap_record(
                    effect,
                    draws=config.evaluation.bootstrap_draws,
                    seed=_stable_seed(
                        "paired_strongest_independent",
                        method,
                        reference_method,
                        task,
                        metric,
                        base_seed=config.evaluation.bootstrap_seed,
                    ),
                )
                records.append(
                    {
                        "paper_method": method,
                        "paper_task": task,
                        "metric": metric,
                        "effect_direction": "positive_favours_ImmuneTwin",
                        "strongest_independent_baseline": reference_method,
                        "baseline_point_estimate": float(means[reference_method]),
                        "n_donors": len(candidate),
                        "paired_effect": point,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(records)


def main():
    """Score a prediction manifest using the fixed paper evaluation protocol."""
    import argparse
    from paths import ROOT, WORKSPACE
    from data.assets import default_assets_path
    from evaluation.distribution import _aggregate_metrics
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--plan-root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=default_assets_path())
    parser.add_argument('--config', type=Path, default=ROOT/'config/sound_evaluation.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from pipeline.sound.common import require_cuda
    device = require_cuda()
    args.output.resolve().relative_to(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    predictions = pd.read_csv(args.predictions, sep='\t')
    config = _v2_paper_score_config(args.config)
    draws, lineage = score_predictions(predictions=predictions, artifact_root=(WORKSPACE, ROOT),
        plan_root=args.plan_root, assets_path=args.assets, config=config, device=device,
        progress_path=args.output/'progress.json')
    units, donors, response_units = _aggregate_metrics(draws)
    for name, frame in [('draws',draws),('units',units),('donors',donors),('response_units',response_units),('lineage',lineage)]:
        frame.to_csv(args.output/(name+'.tsv'), sep='\t', index=False)

if __name__ == '__main__':
    main()

from __future__ import annotations
from paths import expand_paths
import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Sequence
import pandas as pd
from pipeline.sound.provenance import atomic_json

SCHEMA = "sound_meta_adapter_epoch_selection_1"

METRICS = (
    "raw_donor_delta_rmse",
    "raw_donor_delta_mae",
    "raw_donor_response_cosine",
    "distribution_mmd2",
    "response_dispersion_ratio",
    "response_norm_ratio",
    "history_shuffle_gain",
    "history_effect_size",
    "history_vs_current_gain",
)

def _parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--summary", type=Path, action="append", required=True)
    value.add_argument(
        "--selection-rule",
        choices=("median_fold_best", "pooled_common_curve"),
        default="median_fold_best",
    )
    value.add_argument("--output", type=Path, required=True)
    return value

def _weighted(
    summaries: Sequence[dict[str, object]], section: str, metric: str
) -> float:
    weights = [int(item["validation_donors"]) for item in summaries]
    values = [float(item[section][metric]) for item in summaries]  # type: ignore[index]
    return sum(weight * value for weight, value in zip(weights, values)) / sum(weights)

def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if len(args.summary) != 4:
        raise ValueError("exactly four inner-fold summaries are required")
    paths = [path.resolve(strict=True) for path in args.summary]
    summaries = [expand_paths(json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    tasks = {str(item["task"]) for item in summaries}
    outer_folds = {int(item["outer_fold"]) for item in summaries}
    inner_folds = {int(item["inner_validation_fold"]) for item in summaries}
    seeds = {int(item["seed"]) for item in summaries}
    implementations = {str(item["implementation"]) for item in summaries}
    variants = {str(item["variant"]) for item in summaries}
    twin_configs = {
        json.dumps(item["twin_config"], sort_keys=True) for item in summaries
    }
    losses = {json.dumps(item["loss"], sort_keys=True) for item in summaries}
    for item in summaries:
        if item.get("status") != "pass" or bool(item.get("outer_test_target_opened")):
            raise ValueError("selection input is failed or opened an outer-test target")
    if not (
        len(tasks)
        == len(outer_folds)
        == len(seeds)
        == len(implementations)
        == len(variants)
        == len(twin_configs)
        == len(losses)
        == 1
    ):
        raise ValueError("inner-fold summaries do not share one frozen configuration")
    if inner_folds != {0, 1, 2, 3} or seeds != {42} or variants != {"rna"}:
        raise ValueError("inner folds, seed or selected input variant are invalid")

    epochs = [int(item["best_epoch"]) for item in summaries]
    median_epoch = statistics.median(epochs)
    pooled_curve: list[dict[str, float]] = []
    if args.selection_rule == "median_fold_best":
        selected_epoch = int(math.floor(median_epoch + 0.5))
        selection_rule = "rounded_median_of_four_inner_fold_best_epochs"
    else:
        histories = [
            pd.read_csv(path.parent / "history.tsv", sep="\t") for path in paths
        ]
        common_epochs = min(len(history) for history in histories)
        donor_count = sum(int(item["validation_donors"]) for item in summaries)
        for epoch in range(1, common_epochs + 1):
            rmse = sum(
                float(
                    history.loc[
                        history["epoch"].astype(int).eq(epoch),
                        "validation_raw_donor_delta_rmse",
                    ].iloc[0]
                )
                * int(item["validation_donors"])
                for history, item in zip(histories, summaries)
            ) / donor_count
            pooled_curve.append(
                {"epoch": float(epoch), "donor_weighted_validation_rmse": rmse}
            )
        selected_epoch = int(
            min(
                pooled_curve,
                key=lambda row: (row["donor_weighted_validation_rmse"], row["epoch"]),
            )["epoch"]
        )
        selection_rule = "minimum_donor_weighted_four_fold_common_validation_curve"
    baseline = {metric: _weighted(summaries, "baseline_validation", metric) for metric in METRICS}
    adapted = {metric: _weighted(summaries, "best_validation", metric) for metric in METRICS}
    payload = {
        "status": "pass",
        "schema_version": SCHEMA,
        "paper_method": "ImmuneTwins (full)",
        "implementation": next(iter(implementations)),
        "task": next(iter(tasks)),
        "outer_fold": next(iter(outer_folds)),
        "seed": 42,
        "selection_metric": "raw_donor_delta_rmse",
        "selection_rule": selection_rule,
        "inner_best_epochs": epochs,
        "inner_best_epoch_median": float(median_epoch),
        "selected_epoch": selected_epoch,
        "pooled_common_validation_curve": pooled_curve,
        "validation_donors": sum(int(item["validation_donors"]) for item in summaries),
        "donor_weighted_parent_metrics": baseline,
        "donor_weighted_meta_adapter_metrics": adapted,
        "outer_test_target_opened": False,
        "summaries": [
            {"path": str(path)} for path in paths
        ],
        "twin_config": summaries[0]["twin_config"],
        "loss": summaries[0]["loss"],
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

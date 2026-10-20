#!/usr/bin/env python3
"""Fit CIMA-only age heads and score external transfer."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from baseline.age.workspace import WORKSPACE_ROOT


SPLIT = Path(
    os.environ.get(
        "IMMUNETWINS_CIMA_FOLDS",
        WORKSPACE_ROOT / "config/splits/cima_random_donor.tsv",
    )
)
CIMA_AGES = WORKSPACE_ROOT / "data/patient_encoder_feasibility/manifests/cima_417_donors.tsv"
EXTERNAL_AGES = Path(
    os.environ.get(
        "IMMUNETWINS_EXTERNAL_AGE_LABELS",
        WORKSPACE_ROOT / "data/manifests/immunobiology_aging_234_donors.tsv",
    )
)
CIMA_UCE_MEAN = (
    WORKSPACE_ROOT
    / "results/patient_encoder_feasibility_2026-07-11/pulsar/mean_uce"
    / "cima_mean_uce_embeddings_views.npz"
)
EXTERNAL_UCE_ROOT = (
    WORKSPACE_ROOT
    / "results/official_age_baselines/UCE_mean/strict_zero_shot_v1/external_uce_shards"
)
CIMA_PULSAR = (
    WORKSPACE_ROOT
    / "results/official_age_baselines/PULSAR/strict_zero_shot_v1/cima_embeddings.npz"
)
EXTERNAL_PULSAR = (
    WORKSPACE_ROOT
    / "results/official_age_baselines/PULSAR/strict_zero_shot_v1/"
    "immunobiology_embeddings.npz"
)
SCPHASE_ROOT = (
    WORKSPACE_ROOT
    / "results/official_age_baselines/scPhase/practical_fixed1500_seed20260711"
)
IDEAL_ROOT = (
    WORKSPACE_ROOT
    / "results/official_age_baselines/IDEAL-Age/full_ensemble_strict_v1"
)
IDEAL_LABEL = "IDEAL-Age"
METHODS = ("UCE", "PULSAR", IDEAL_LABEL, "scPhase")
OUTPUT = WORKSPACE_ROOT / "results/official_age_baselines/summary/strict_zero_shot_v1"


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    matrix = np.asarray(archive["embeddings"], dtype=np.float64)
    donors = archive["donor_id"].astype(str)
    if matrix.shape[0] != len(donors) or len(set(donors)) != len(donors):
        raise ValueError(f"invalid donor embedding archive: {path}")
    return matrix, donors


def load_cima_uce(path: Path, seed: int = 20260711) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    selected = np.asarray(archive["seed"], dtype=np.int64) == seed
    matrix = np.asarray(archive["embeddings"][selected], dtype=np.float64)
    donors = archive["donor_id"][selected].astype(str)
    if matrix.shape != (417, 1280) or len(set(donors)) != 417:
        raise ValueError("CIMA UCE fixed-view archive contract failed")
    return matrix, donors


def load_external_uce(
    root: Path,
    *,
    expected_groups: int | None = 36,
    expected_donors: int | None = 234,
    expected_cells_per_donor: int | None = 1500,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    groups = sorted(root.glob("group_*"))
    if expected_groups is not None and len(groups) != expected_groups:
        raise ValueError(
            f"expected {expected_groups} UCE target shards, found {len(groups)}"
        )
    total_cells = 0
    for group in groups:
        marker_path = group / "COMPLETE.json"
        partial_path = group / "donor_partial_uce.npz"
        marker = json.loads(marker_path.read_text())
        if marker.get("status") != "PASS" or marker.get("target_labels_accessed") is not False:
            raise ValueError(f"UCE shard proof failed: {marker_path}")
        archive = np.load(partial_path, allow_pickle=False)
        donors = archive["donor_id"].astype(str)
        partial_sums = np.asarray(archive["embedding_sum"], dtype=np.float64)
        partial_counts = np.asarray(archive["n_cells"], dtype=np.int64)
        for donor, vector, count in zip(donors, partial_sums, partial_counts):
            sums[donor] = sums.get(donor, np.zeros(1280, dtype=np.float64)) + vector
            counts[donor] = counts.get(donor, 0) + int(count)
        total_cells += int(partial_counts.sum())
    donor_order = np.asarray(sorted(sums), dtype=str)
    if expected_donors is not None and len(donor_order) != expected_donors:
        raise ValueError("external UCE donor coverage failed")
    wrong = {
        donor: counts[donor]
        for donor in donor_order
        if expected_cells_per_donor is not None
        and counts[donor] != expected_cells_per_donor
    }
    if wrong:
        raise ValueError(f"external UCE cell counts failed: {wrong}")
    matrix = np.vstack([sums[donor] / counts[donor] for donor in donor_order])
    return matrix, donor_order, {
        "n_groups": len(groups),
        "n_cells": total_cells,
    }


def align_labels(donors: np.ndarray, ages: pd.DataFrame) -> np.ndarray:
    mapping = ages.set_index("donor_id")["age"]
    missing = sorted(set(donors) - set(mapping.index))
    if missing:
        raise ValueError(f"age labels missing for donors: {missing[:10]}")
    return mapping.loc[donors].to_numpy(dtype=np.float64)


def fit_uce_ols(x: np.ndarray, y: np.ndarray, device: str) -> tuple[np.ndarray, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the UCE age readout")
    xt = torch.as_tensor(x, dtype=torch.float64, device=device)
    yt = torch.as_tensor(y, dtype=torch.float64, device=device)
    x_mean = xt.mean(dim=0)
    y_mean = yt.mean()
    centered = xt - x_mean
    gram = centered @ centered.T
    dual = torch.linalg.pinv(gram, hermitian=True) @ (yt - y_mean)
    weight = centered.T @ dual
    intercept = y_mean - x_mean @ weight
    return weight.cpu().numpy(), float(intercept.cpu())


def uce_predict(x: np.ndarray, weight: np.ndarray, intercept: float) -> np.ndarray:
    # Algebraically identical to predicting every cell with the same linear
    # readout and taking the arithmetic mean within donor.
    return np.asarray(x @ weight + intercept, dtype=np.float64)


def common_oof(
    method: str,
    x: np.ndarray,
    donors: np.ndarray,
    y: np.ndarray,
    split: pd.DataFrame,
    device: str,
) -> pd.DataFrame:
    fold_map = split.loc[split["repeat"].eq(0)].set_index("donor_id")["fold"]
    if set(donors) != set(fold_map.index):
        raise ValueError(f"{method}: CIMA donor split mismatch")
    folds = fold_map.loc[donors].to_numpy(dtype=int)
    prediction = np.full(len(donors), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        if method == "UCE":
            weight, intercept = fit_uce_ols(x[~test], y[~test], device)
            prediction[test] = uce_predict(x[test], weight, intercept)
        elif method == "PULSAR":
            model = Ridge(alpha=0.1)
            model.fit(x[~test], y[~test])
            prediction[test] = model.predict(x[test])
        else:
            raise ValueError(method)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"{method}: incomplete OOF predictions")
    return pd.DataFrame(
        {"method": method, "cohort": "CIMA", "donor_id": donors, "age": y,
         "predicted_age": prediction, "fold": folds}
    )


def scphase_cima() -> pd.DataFrame:
    from baseline.age.adapters.scphase_lazy_dense_loader import validate_master

    master_path = SCPHASE_ROOT / "input_v2/cima_common5_master_compat.pkl"
    master = validate_master(master_path,  expected_role="cima_fit")
    donors = np.asarray(master["sample_ids"], dtype=str)
    labels = np.asarray(master["labels"], dtype=np.float64)
    config = json.loads((SCPHASE_ROOT / "config_common5_compat.json").read_text())
    seed = int(config["run_params"]["seed"])
    split = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(donors))
    raw = pd.read_csv(
        SCPHASE_ROOT / "training_common5/predictions/CIMA_Age_scPhase_predictions.csv"
    )
    parts: list[pd.DataFrame] = []
    for fold, (_, test_index) in enumerate(split, start=1):
        frame = raw.loc[raw["fold"].eq(fold)].sort_values("sample_idx")
        if len(frame) != len(test_index):
            raise ValueError(f"scPhase fold {fold} prediction count mismatch")
        expected = np.arange(len(test_index))
        if not np.array_equal(frame["sample_idx"].to_numpy(), expected):
            raise ValueError(f"scPhase fold {fold} sample order mismatch")
        if not np.allclose(frame["y_true"].to_numpy(), labels[test_index]):
            raise ValueError(f"scPhase fold {fold} label reconstruction mismatch")
        parts.append(
            pd.DataFrame(
                {
                    "method": "scPhase",
                    "cohort": "CIMA",
                    "donor_id": donors[test_index],
                    "age": labels[test_index],
                    "predicted_age": frame["y_pred"].to_numpy(dtype=float),
                    "fold": fold - 1,
                }
            )
        )
    result = pd.concat(parts, ignore_index=True)
    if len(result) != 417 or result["donor_id"].duplicated().any():
        raise ValueError("scPhase CIMA OOF coverage failed")
    return result


def ideal_age_predictions(ages: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the IDEAL-Age ensemble outputs."""
    cima_path = IDEAL_ROOT / "cima_oof_predictions.tsv"
    external_path = IDEAL_ROOT / "immunobiology_zero_shot_predictions.sealed.tsv"
    audit_path = IDEAL_ROOT / "audit.json"
    audit = json.loads(audit_path.read_text())
    required_audit = {
        "status": "complete",
        "official_commit": "d419455549907f70da55ac3fd39021045a061819",
        "external_fit": "CIMA only",
        "external_target_labels_accessed": False,
        "external_target_statistics_fitted": False,
    }
    for key, expected in required_audit.items():
        if audit.get(key) != expected:
            raise ValueError(f"IDEAL-Age audit changed: {key}={audit.get(key)!r}")

    raw_cima = pd.read_csv(cima_path, sep="\t")
    required_cima = {
        "method", "cohort", "fold", "donor_id", "observed_age", "predicted_age"
    }
    if set(raw_cima.columns) != required_cima:
        raise ValueError("IDEAL-Age CIMA OOF columns changed")
    if (
        len(raw_cima) != 417
        or raw_cima["donor_id"].duplicated().any()
        or not raw_cima["method"].eq("IDEAL-Age").all()
        or not raw_cima["cohort"].eq("CIMA").all()
        or not np.isfinite(raw_cima["predicted_age"]).all()
    ):
        raise ValueError("IDEAL-Age CIMA OOF coverage failed")
    expected_age = align_labels(raw_cima["donor_id"].to_numpy(dtype=str), ages)
    if not np.allclose(raw_cima["observed_age"].to_numpy(dtype=float), expected_age):
        raise ValueError("IDEAL-Age CIMA OOF age alignment failed")
    cima = raw_cima.rename(columns={"observed_age": "age"}).copy()
    cima["method"] = IDEAL_LABEL
    cima = cima[["method", "cohort", "donor_id", "age", "predicted_age", "fold"]]

    external = pd.read_csv(external_path, sep="\t")
    required_external = {"method", "cohort", "donor_id", "predicted_age"}
    if set(external.columns) != required_external:
        raise ValueError("IDEAL-Age external prediction columns changed")
    if (
        len(external) != 234
        or external["donor_id"].duplicated().any()
        or not external["method"].eq("IDEAL-Age").all()
        or not external["cohort"].eq("Immunobiology").all()
        or not np.isfinite(external["predicted_age"]).all()
    ):
        raise ValueError("IDEAL-Age external prediction coverage failed")
    external = external.copy()
    external["method"] = IDEAL_LABEL
    external["cohort"] = "Immunobiology_of_Aging"
    return cima, external


def regression_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame["age"].to_numpy(dtype=float)
    p = frame["predicted_age"].to_numpy(dtype=float)
    residual = p - y
    slope, intercept = np.polyfit(y, p, 1)
    true_sd = float(np.std(y, ddof=1))
    pred_sd = float(np.std(p, ddof=1))
    return {
        "n": int(len(frame)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "median_ae": float(np.median(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r2": float(1 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)),
        "pearson": float(stats.pearsonr(y, p).statistic),
        "spearman": float(stats.spearmanr(y, p).statistic),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "true_sd": true_sd,
        "prediction_sd": pred_sd,
        "prediction_sd_ratio": float(pred_sd / true_sd),
        "prediction_min": float(p.min()),
        "prediction_max": float(p.max()),
        "within_10_years_pct": float(100 * np.mean(np.abs(residual) <= 10)),
    }


def main() -> int:
    global SCPHASE_ROOT, IDEAL_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--cima-ages", type=Path, default=CIMA_AGES)
    parser.add_argument("--external-ages", type=Path, default=EXTERNAL_AGES)
    parser.add_argument("--cima-uce-mean", type=Path, default=CIMA_UCE_MEAN)
    parser.add_argument("--external-uce-root", type=Path, default=EXTERNAL_UCE_ROOT)
    parser.add_argument("--cima-pulsar", type=Path, default=CIMA_PULSAR)
    parser.add_argument("--external-pulsar", type=Path, default=EXTERNAL_PULSAR)
    parser.add_argument("--scphase-root", type=Path, default=SCPHASE_ROOT)
    parser.add_argument("--ideal-root", type=Path, default=IDEAL_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    SCPHASE_ROOT = args.scphase_root
    IDEAL_ROOT = args.ideal_root
    ages = pd.read_csv(args.cima_ages, sep="\t")[["donor_id", "age"]]
    split = pd.read_csv(args.split, sep="\t")

    cima_uce, cima_uce_donors = load_cima_uce(args.cima_uce_mean)
    y_uce = align_labels(cima_uce_donors, ages)
    external_uce, external_uce_donors, _ = load_external_uce(
        args.external_uce_root
    )
    uce_oof = common_oof(
        "UCE", cima_uce, cima_uce_donors, y_uce, split, args.device
    )
    uce_weight, uce_intercept = fit_uce_ols(cima_uce, y_uce, args.device)
    np.savez(
        args.output / "uce_cell_linear_readout.npz",
        weight=uce_weight,
        intercept=np.asarray([uce_intercept]),
    )
    uce_external = pd.DataFrame(
        {
            "method": "UCE",
            "cohort": "Immunobiology_of_Aging",
            "donor_id": external_uce_donors,
            "predicted_age": uce_predict(external_uce, uce_weight, uce_intercept),
        }
    )

    cima_pulsar, cima_pulsar_donors = load_npz(args.cima_pulsar)
    external_pulsar, external_pulsar_donors = load_npz(args.external_pulsar)
    y_pulsar = align_labels(cima_pulsar_donors, ages)
    pulsar_oof = common_oof(
        "PULSAR", cima_pulsar, cima_pulsar_donors, y_pulsar, split, args.device
    )
    pulsar_model = Ridge(alpha=0.1)
    pulsar_model.fit(cima_pulsar, y_pulsar)
    pulsar_head_path = args.output / "pulsar_official_ridge_alpha0.1.npz"
    np.savez(
        pulsar_head_path,
        coef=np.asarray(pulsar_model.coef_, dtype=np.float64),
        intercept=np.asarray([pulsar_model.intercept_], dtype=np.float64),
        alpha=np.asarray([0.1], dtype=np.float64),
    )
    pulsar_external = pd.DataFrame(
        {
            "method": "PULSAR",
            "cohort": "Immunobiology_of_Aging",
            "donor_id": external_pulsar_donors,
            "predicted_age": pulsar_model.predict(external_pulsar),
        }
    )

    scphase_oof = scphase_cima()
    scphase_external = pd.read_csv(
        SCPHASE_ROOT / "external_strict/immunobiology_frozen_predictions.csv"
    )[["method", "cohort", "donor_id", "predicted_age"]]

    ideal_oof, ideal_external = ideal_age_predictions(ages)

    cima_long = pd.concat(
        [uce_oof, pulsar_oof, ideal_oof, scphase_oof], ignore_index=True
    )
    external_frozen = pd.concat(
        [uce_external, pulsar_external, ideal_external, scphase_external],
        ignore_index=True,
    )
    expected_cima_rows = len(METHODS) * 417
    expected_external_rows = len(METHODS) * 234
    if (
        len(cima_long) != expected_cima_rows
        or cima_long.duplicated(["method", "donor_id"]).any()
        or len(external_frozen) != expected_external_rows
        or external_frozen.duplicated(["method", "donor_id"]).any()
    ):
        raise ValueError("unified four-method prediction coverage failed")
    cima_path = args.output / "cima_oof_predictions.tsv"
    external_path = args.output / "immunobiology_frozen_predictions.tsv"
    cima_long.to_csv(cima_path, sep="\t", index=False)
    external_frozen.to_csv(external_path, sep="\t", index=False)

    target_age = pd.read_csv(args.external_ages, sep="\t").rename(
        columns={"subject.subjectGuid": "donor_id", "exact_age": "age"}
    )[["donor_id", "age"]]
    scored = external_frozen.merge(
        target_age, on="donor_id", how="left", validate="many_to_one"
    )
    if scored["age"].isna().any() or len(scored) != len(METHODS) * 234:
        raise ValueError("external scoring label coverage failed")
    scored.to_csv(args.output / "immunobiology_scored_predictions.tsv", sep="\t", index=False)
    records: list[dict[str, object]] = []
    for method in METHODS:
        cima_metrics = regression_metrics(cima_long.loc[cima_long["method"].eq(method)])
        external_metrics = regression_metrics(scored.loc[scored["method"].eq(method)])
        records.append(
            {
                "method": method,
                **{f"cima_{key}": value for key, value in cima_metrics.items()},
                **{f"external_{key}": value for key, value in external_metrics.items()},
                "mae_increase": float(external_metrics["mae"] - cima_metrics["mae"]),
            }
        )
    table = pd.DataFrame(records)
    table.to_csv(args.output / "table_pretraining_transfer.tsv", sep="\t", index=False)
    display_columns = [
        "method",
        "cima_mae",
        "external_mae",
        "mae_increase",
        "external_rmse",
        "external_bias",
        "external_pearson",
        "external_spearman",
        "external_calibration_slope",
        "external_prediction_sd_ratio",
    ]
    display = table[display_columns].copy()
    display.columns = [
        "Method",
        "CIMA MAE",
        "External MAE",
        "MAE increase",
        "External RMSE",
        "External bias",
        "Pearson r",
        "Spearman rho",
        "Calibration slope",
        "Prediction SD ratio",
    ]
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda value: f"{float(value):.3f}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    (args.output / "table_pretraining_transfer.md").write_text(
        "\n".join([header, separator, *rows]) + "\n"
    )
    (args.output / "metrics.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    ideal_audit = json.loads((IDEAL_ROOT / "audit.json").read_text())
    reproduction_audit = {
        "status": "PASS",
        "paper_ready_method_count": 4,
        "shared_evaluation_contract": {
            "source_fit": "CIMA only",
            "cima_evaluation": "donor-level out-of-fold predictions",
            "external_evaluation": (
                "234-donor Immunobiology zero-shot predictions written before "
                "chronological-age labels were opened"
            ),
            "target_calibration": False,
            "target_statistics_accessed_before_prediction": False,
        },
        "methods": [
            {
                "paper_label": "UCE",
                "reproduction_identity": "official encoder plus age-task adapter",
                "paper_ready": True,
                "official_source_commit": "9c416007be15ad6753dc84af4468c1dc10421ab9",
                "age_adapter": (
                    "CIMA-fitted unregularized linear cell-age readout; donor "
                    "prediction is the arithmetic mean of cell predictions"
                ),
                "claim_boundary": (
                    "the released encoder has no donor-age head; the declared "
                    "patient rule is a CIMA-fitted cell-age readout followed by mean"
                ),
            },
            {
                "paper_label": "PULSAR",
                "reproduction_identity": "official downstream age recipe",
                "paper_ready": True,
                "official_source_commit": "7442e4e7f17e0d3bbf5e15737e127d66979fd08b",
                "age_adapter": (
                    "official 1,024-cell replacement sampling with seed 0, one "
                    "PULSAR donor embedding and Ridge(alpha=0.1)"
                ),
                "claim_boundary": "may be reported directly as PULSAR",
            },
            {
                "paper_label": IDEAL_LABEL,
                "reproduction_identity": "complete released ensemble design",
                "paper_ready": True,
                "official_source_commit": (
                    "d419455549907f70da55ac3fd39021045a061819"
                ),
                "model_components": ideal_audit["components"],
                "source_adaptation": (
                    "CIMA-selected HVG5000 input and CIMA-fitted scaler; released "
                    "105-epoch DeepSets-Attention branch, pseudo-bulk FASTAI and "
                    "NN_TORCH branches, and CIMA-only weighted integration"
                ),
                "claim_boundary": "may be reported directly as IDEAL-Age",
            },
            {
                "paper_label": "scPhase",
                "reproduction_identity": "official age-task implementation",
                "paper_ready": True,
                "official_source_commit": "cb21b1a5175f5b7630af183ac8ac1b58d5204426",
                "source_adaptation": (
                    "storage-only lazy donor loader; CIMA-fitted HVG5000 and "
                    "scaler; external prediction is the frozen arithmetic mean "
                    "of five CIMA-fold checkpoints"
                ),
                "claim_boundary": "may be reported directly as scPhase",
            },
        ],
    }
    (args.output / "reproduction_audit.json").write_text(
        json.dumps(reproduction_audit, indent=2, sort_keys=True) + "\n"
    )
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()

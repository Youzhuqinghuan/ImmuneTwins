"""Donor-level immune-age scores and paired bootstrap intervals."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from paths import ROOT

SEED = 20260903

METRICS = (
    "mae",
    "rmse",
    "bias",
    "pearson",
    "calibration_slope",
    "prediction_sd_ratio",
    "within_10_years_pct",
)

def _metric_arrays(age: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - age
    centered_age = age - age.mean()
    centered_predicted = predicted - predicted.mean()
    covariance = float(np.sum(centered_age * centered_predicted))
    age_ss = float(np.sum(centered_age**2))
    predicted_ss = float(np.sum(centered_predicted**2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "pearson": float(covariance / np.sqrt(age_ss * predicted_ss))
        if age_ss > 0 and predicted_ss > 0
        else float("nan"),
        "calibration_slope": float(covariance / age_ss) if age_ss > 0 else float("nan"),
        "prediction_sd_ratio": float(np.sqrt(predicted_ss / age_ss))
        if age_ss > 0
        else float("nan"),
        "within_10_years_pct": float(100.0 * np.mean(np.abs(residual) <= 10.0)),
    }

def _bootstrap_metrics(
    frame: pd.DataFrame,
    indices: np.ndarray,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    age = frame["age"].to_numpy(dtype=np.float64)
    predicted = frame["predicted_age"].to_numpy(dtype=np.float64)
    point = _metric_arrays(age, predicted)
    sampled_age = age[indices]
    sampled_predicted = predicted[indices]
    residual = sampled_predicted - sampled_age
    centered_age = sampled_age - sampled_age.mean(axis=1, keepdims=True)
    centered_predicted = sampled_predicted - sampled_predicted.mean(axis=1, keepdims=True)
    covariance = np.sum(centered_age * centered_predicted, axis=1)
    age_ss = np.sum(centered_age**2, axis=1)
    predicted_ss = np.sum(centered_predicted**2, axis=1)
    bootstrap = {
        "mae": np.mean(np.abs(residual), axis=1),
        "rmse": np.sqrt(np.mean(residual**2, axis=1)),
        "bias": np.mean(residual, axis=1),
        "pearson": covariance / np.sqrt(age_ss * predicted_ss),
        "calibration_slope": covariance / age_ss,
        "prediction_sd_ratio": np.sqrt(predicted_ss / age_ss),
        "within_10_years_pct": 100.0 * np.mean(np.abs(residual) <= 10.0, axis=1),
    }
    intervals = {
        metric: tuple(float(value) for value in np.quantile(bootstrap[metric], [0.025, 0.975]))
        for metric in METRICS
    }
    return point, intervals

def _concordance(age: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Lin's CCC, using empirical moments (1/n) along the donor axis."""
    age_mean = age.mean(axis=-1, keepdims=True)
    predicted_mean = predicted.mean(axis=-1, keepdims=True)
    a, p = age - age_mean, predicted - predicted_mean
    denominator = (np.mean(a*a, axis=-1) + np.mean(p*p, axis=-1)
                   + np.squeeze(predicted_mean-age_mean, axis=-1)**2)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise ValueError("CCC requires a positive finite denominator")
    return 2*np.mean(a*p, axis=-1)/denominator


def score(predictions: pd.DataFrame, draws: int = 10000) -> pd.DataFrame:
    keys = ['cohort', 'method', 'donor_id']
    if predictions.duplicated(keys).any():
        raise ValueError('Each method must predict each donor once.')
    if not np.isfinite(predictions[['age', 'predicted_age']].to_numpy()).all():
        raise ValueError('Age predictions must be finite.')
    rows = []
    for cohort, frame in predictions.groupby('cohort', sort=True):
        order = sorted(frame.donor_id.unique())
        labels = frame[['donor_id', 'age']].drop_duplicates().set_index('donor_id')
        if not labels.index.is_unique:
            raise ValueError('Methods disagree on donor ages.')
        rng = np.random.default_rng(SEED + sum(ord(char) for char in cohort))
        indices = rng.integers(0, len(order), size=(draws, len(order)))
        for method, group in frame.groupby('method', sort=True):
            if set(group.donor_id) != set(order):
                raise ValueError(f'{cohort}/{method}: incomplete donor support')
            group = group.set_index('donor_id').loc[order].reset_index()
            point, intervals = _bootstrap_metrics(group, indices)
            for metric in METRICS:
                low, high = intervals[metric]
                rows.append(dict(cohort=cohort, method=method, n_donors=len(order),
                    metric=metric, estimate=point[metric], ci_low=low, ci_high=high,
                    preferred_direction='lower' if metric in ('mae','rmse') else
                    ('zero' if metric=='bias' else 'one' if metric in
                     ('calibration_slope','prediction_sd_ratio') else 'higher')))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True,
                        help='TSV: cohort, method, donor_id, age, predicted_age')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=10000)
    args = parser.parse_args()
    if args.draws < 1000: parser.error('Use at least 1000 bootstrap draws.')
    args.output.resolve().relative_to(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    predictions = pd.read_csv(args.predictions, sep='\t', dtype={'donor_id':str})
    result = score(predictions, args.draws)
    result.to_csv(args.output/'age_method_metric_bootstrap.tsv', sep='\t', index=False)
    record = dict(input=str(args.predictions.resolve()),
                  draws=args.draws, seed=SEED, bootstrap_unit='donor', rows=len(result))
    (args.output/'inputs.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()

"""Fit and apply the final Sound model: neural response plus personal reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.reference import (DEFAULT_CONFIG, TASKS, load_config, load_prepared, load_means,
                            neural_run, read_table, resolve_path, save_json, save_table,
                            scale_arrays)
from model.reference import CompositionReference, StateReference, apply_reference, scale_response
from evaluation.distribution import _lineages, rbf_mmd2_biased
from evaluation.units import _draw_indices
from evaluation.sound import _v2_paper_score_config
from paths import ROOT


def model_path(cfg, task, fold):
    return Path(cfg["output_root"]) / "models" / f"{task}_outer{fold}.json"


def load_model(cfg, task, fold):
    payload = json.loads(model_path(cfg, task, fold).read_text())
    if (payload.get("schema_version") != "sound_personal_reference_model_1"
            or payload["task"] != task or payload["outer_fold"] != fold):
        raise ValueError("Reference model task/fold does not match the prediction")
    payload["reference_file"] = str(model_path(cfg, task, fold))
    payload["composition"] = {k: np.asarray(v, float) if k in ("m", "mu") else v
                              for k, v in payload["composition"].items()}
    return payload


def _least_squares(displacement, observed, donors, upper, draws=10000):
    """Donor-equal clipped least squares and donor bootstrap."""
    tab = pd.DataFrame(dict(donor=donors, num=np.sum(displacement * observed, axis=1),
                            den=np.sum(displacement * displacement, axis=1))).groupby("donor").mean()
    num, den = tab.num.to_numpy(), tab.den.to_numpy()
    raw = num.sum() / max(den.sum(), 1e-20)
    rng = np.random.default_rng(42)
    bootstrap = []
    for start in range(0, draws, 1000):
        ix = rng.integers(len(tab), size=(min(1000, draws-start), len(tab)))
        bootstrap.extend(np.clip(num[ix].sum(1) / np.maximum(den[ix].sum(1), 1e-20), 0, upper))
    return float(np.clip(raw, 0, upper)), dict(
        raw=float(raw), low=float(np.quantile(bootstrap, .025)),
        high=float(np.quantile(bootstrap, .975)), n=len(tab))


def _gain_interval(values, draws=10000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(42)
    bootstrap = np.concatenate([
        values[rng.integers(len(values), size=(min(1000, draws-start), len(values)))].mean(1)
        for start in range(0, draws, 1000)])
    return dict(mean=float(values.mean()), low=float(np.quantile(bootstrap, .025)),
                high=float(np.quantile(bootstrap, .975)), n=len(values))


def _mmd(source, predicted, observed, task, donor, season, center, scale, bandwidth, score_cfg, device):
    """Use the same draw identities as the public evaluation pipeline."""
    s, p, t = [(np.asarray(x, np.float32) - center) / scale for x in (source, predicted, observed)]
    si = _draw_indices(len(s), score_cfg.evaluation, task, donor, season, role="source")
    pi = si if np.array_equal(s, p) else _draw_indices(
        len(p), score_cfg.evaluation, task, donor, season, role="predicted")
    ti = _draw_indices(len(t), score_cfg.evaluation, task, donor, season, role="observed")
    with torch.inference_mode():
        values = rbf_mmd2_biased(torch.as_tensor(p[pi] / bandwidth, device=device),
                                 torch.as_tensor(t[ti] / bandwidth, device=device)).cpu().numpy()
    return float(values.mean(dtype=np.float64))


def load_oof(cfg, task, fold, units):
    """Reject calibration inputs with wrong donors, coordinates or cell-row alignment."""
    stage = "projected" if task == TASKS[0] else "full"
    rows, arrays = [], []
    expected = units.loc[units.task.eq(task) & units.fold.ne(fold)]
    splits = read_table(Path(cfg["prepared_root"]) / "inner_splits.tsv")
    inner_map = splits.loc[splits.outer_fold.eq(fold)].set_index("donor_id").inner_fold
    expected_test = set(units.loc[units.task.eq(task) & units.fold.eq(fold), "donor"])
    for inner in range(4):
        base = Path(cfg["inner_root"]) / "runs" / f"outer{fold}" / f"inner{inner}" / task / stage / "oof"
        table = read_table(base / "predictions.tsv")
        audit = json.loads((base / "split_audit.json").read_text())
        allowed = set(audit["validation_donors"])
        fit = set(audit["fit_donors"])
        test = set(audit["outer_test_donors"])
        expected_validation = set(expected.loc[expected.donor.map(inner_map).eq(inner), "donor"])
        expected_fit = set(expected.donor) - expected_validation
        if allowed != expected_validation or fit != expected_fit or test != expected_test:
            raise ValueError("Calibration donor lists differ from the configured split")
        if fit & allowed or fit & test or allowed & test:
            raise ValueError("Overlapping fitting, calibration and outer-test donors")
        if audit["outer"] != fold or audit["inner"] != inner or audit["task"] != task:
            raise ValueError("Calibration split identity differs from its folder")
        for row in table.itertuples():
            if row.donor_id not in allowed or row.donor_outer_fold == fold or row.outer_fold != fold:
                raise ValueError("Invalid calibration donor or coordinate fold")
            unit = expected.loc[expected.donor.eq(row.donor_id) & expected.season.eq(row.season)]
            if len(unit) != 1 or row.source_sample != unit.iloc[0].source or row.target_sample != unit.iloc[0].target:
                raise ValueError("Calibration unit does not match the task manifest")
            loaded = {}
            for key in ("source", "predicted", "observed"):
                with np.load(resolve_path(getattr(row, key + "_path"), cfg), allow_pickle=False) as data:
                    loaded[key] = {name: data[name] for name in ("embedding", "source_row_index", "cell_state") if name in data}
            source, pred, target = [loaded[key]["embedding"] for key in ("source", "predicted", "observed")]
            if source.shape != (2048, 16) or pred.shape != source.shape or target.shape != source.shape:
                raise ValueError("Calibration cell-set shape differs from the paper contract")
            if not all(np.isfinite(x).all() for x in (source, pred, target)):
                raise ValueError("Non-finite calibration cell set")
            if not np.array_equal(loaded["source"]["source_row_index"], loaded["predicted"]["source_row_index"]):
                raise ValueError("Calibration prediction changed source-cell order")
            rows.append(dict(task=task, outer=fold, inner=inner, donor=row.donor_id,
                             season=row.season, source=row.source_sample, target=row.target_sample,
                             unit=row.episode_id))
            arrays.append((source, pred, target, loaded["predicted"]["cell_state"]))
    frame = pd.DataFrame(rows)
    if frame.duplicated(["donor", "season"]).any() or set(frame.unit) != set(expected.unit):
        raise ValueError("Calibration predictions do not cover outer-training units exactly once")
    return frame, arrays


def fit_fold(cfg, fold, tasks, device):
    counts, visits, units = load_prepared(cfg)
    splits = read_table(Path(cfg["prepared_root"]) / "inner_splits.tsv")
    comp = CompositionReference(counts, visits, splits, cfg["lambda_grid"])
    center, scale, bandwidth = scale_arrays(cfg, fold)
    means = load_means(cfg, fold)
    score_cfg = _v2_paper_score_config(ROOT / "config/sound_evaluation.yaml")
    for task in tasks:
        train = units.loc[units.task.eq(task) & units.fold.ne(fold)].reset_index(drop=True)
        destination = model_path(cfg, task, fold)
        if destination.exists():
            raise FileExistsError(f"Reference fit already exists: {destination}")
        parameters, cv, _ = comp.select(train, fold)
        frame, arrays = load_oof(cfg, task, fold, units)
        predicted = np.stack([(p.mean(0, dtype=np.float64)-s.mean(0, dtype=np.float64))/scale
                              for s, p, t, labels in arrays])
        observed = np.stack([(t.mean(0, dtype=np.float64)-s.mean(0, dtype=np.float64))/scale
                             for s, p, t, labels in arrays])
        gamma, gamma_ci = _least_squares(predicted, observed, frame.donor,
                                          cfg["response_scale_bounds"][1], cfg["bootstrap_draws"])
        ks, state_info = 0., {}
        if task != TASKS[0]:
            dm = splits.loc[splits.outer_fold.eq(fold)].set_index("donor_id").inner_fold
            references = {}
            for inner in range(4):
                fitting = train.loc[train.donor.map(dm).ne(inner)].reset_index(drop=True)
                inner_parameters, _, _ = comp.select(fitting, fold)
                references[inner] = StateReference(
                    visits, set(fitting.donor), inner_parameters["lambda"], means, center, scale)
            displacements = [references[row.inner].displacement(
                row.source, row.donor, _lineages(arrays[i][3]))[0]
                for i, row in enumerate(frame.itertuples())]
            delta = np.stack([value.mean(0) for value in displacements])
            candidate, candidate_ci = _least_squares(
                delta, observed - gamma * predicted, frame.donor,
                cfg["lineage_return_bounds"][1], cfg["bootstrap_draws"])
            gains = []
            for i, row in enumerate(frame.itertuples()):
                source, pred, target, labels = arrays[i]
                scaled = scale_response(source, pred, gamma)
                corrected = (scaled + candidate * displacements[i] * scale).astype(np.float32)
                arguments = (target, task, row.donor, row.season, center, scale, bandwidth, score_cfg, device)
                gains.append(_mmd(source, scaled, *arguments) - _mmd(source, corrected, *arguments))
            donor_gains = pd.Series(gains, index=frame.donor).groupby(level=0).mean()
            interval = _gain_interval(donor_gains, cfg["bootstrap_draws"])
            kept = interval["mean"] > 0 and interval["low"] > 0
            ks = candidate if kept else 0.
            state_info = dict(raw_coefficient=candidate, coefficient_interval=candidate_ci,
                              inner_gain=interval, retained=kept)
        save_json(destination, dict(
            schema_version="sound_personal_reference_model_1", task=task, outer_fold=fold,
            cell_types=list(counts.columns), composition=parameters, gamma=gamma,
            gamma_interval=gamma_ci, state_coefficient=ks, state_selection=state_info,
            lambda_cv=cv, fit_donors=sorted(set(train.donor)), seed=cfg["seed"],
            inner_epoch_selection_uses_validation_labels=True,
            outer_test_targets_used_for_fitting=False))
        print(f"Fitted {task} outer{fold}: gamma={gamma:.8g}, K={parameters['K']:.8g}, K_state={ks:.8g}", flush=True)


class ReferencePredictor:
    """Apply fitted parameters to current cells and strictly earlier personal observations."""

    def __init__(self, cfg, task, fold):
        self.cfg, self.task, self.fold = cfg, task, fold
        counts, self.visits, self.units = load_prepared(cfg)
        self.comp = CompositionReference(counts, self.visits)
        self.parameters = load_model(cfg, task, fold)
        if list(counts.columns) != self.parameters["cell_types"]:
            raise ValueError("Cell-type order differs from the fitted reference model")
        expected = set(self.units.loc[self.units.task.eq(task) & self.units.fold.ne(fold), "donor"])
        if set(self.parameters["fit_donors"]) != expected:
            raise ValueError("Reference parameters were fitted on a different donor partition")
        self.center, self.scale, _ = scale_arrays(cfg, fold)
        self.state = None
        if self.parameters["state_coefficient"]:
            self.state = StateReference(self.visits, expected, self.parameters["composition"]["lambda"],
                                        load_means(cfg, fold), self.center, self.scale)

    def apply(self, unit, source, predicted, labels):
        if unit.donor in self.parameters["fit_donors"]:
            raise ValueError("The held-out predictor received a training donor")
        query = pd.DataFrame([dict(source=unit.source)])
        forecast = self.comp.predict(query, self.parameters["composition"])[0]
        displacement = self.state.displacement(unit.source, unit.donor, _lineages(labels))[0] if self.state else None
        cells, indices, missing = apply_reference(
            source, predicted, labels, self.parameters["cell_types"], forecast,
            self.parameters["gamma"], unit.unit, displacement=displacement,
            state_coefficient=self.parameters["state_coefficient"], coordinate_scale=self.scale,
            seed=self.parameters["seed"])
        n_history = self.comp.refs[unit.source][1]
        sd = float(self.comp.sd(n_history, self.parameters["composition"]))
        return cells, indices, dict(missing_mass=missing, forecast_clr=forecast,
                                    n_history=n_history, composition_sd=sd)


def predict(cfg, folds, tasks, output=None):
    output = Path(output) if output is not None else Path(cfg["output_root"]) / "predictions"
    output.mkdir(parents=True, exist_ok=False)
    rows, compositions = [], []
    for fold in folds:
        for task in tasks:
            predictor = ReferencePredictor(cfg, task, fold)
            units = predictor.units.loc[predictor.units.task.eq(task) & predictor.units.fold.eq(fold)]
            base = neural_run(cfg, task, fold)
            for unit in units.itertuples():
                day = 7 if task == TASKS[1] else 0
                source_path = base / f"source_day{day}" / f"{unit.unit}.npz"
                base_path = base / "predicted" / f"{unit.unit}.npz"
                with np.load(source_path, allow_pickle=False) as archive:
                    source = archive["embedding"].copy()
                    source_rows = archive["source_row_index"].copy()
                with np.load(base_path, allow_pickle=False) as archive:
                    data = {key: archive[key] for key in archive.files}
                if not np.array_equal(source_rows, data["source_row_index"]):
                    raise ValueError("Neural prediction is not aligned with source cells")
                cells, indices, info = predictor.apply(unit, source, data["embedding"], data["cell_state"])
                n = len(source)
                data = {key: value[indices] if value.ndim and len(value) == n else value for key, value in data.items()}
                data.update(embedding=cells, method=np.asarray("ImmuneTwins"),
                            base_prediction_file=np.asarray(str(base_path)),
                            personal_reference_applied=np.asarray(True),
                            standardized_mean_shift=np.asarray(
                                (cells.astype(np.float64).mean(0) - source.astype(np.float64).mean(0)) / predictor.scale,
                                dtype=np.float32))
                destination = output / task / f"outer{fold}" / f"{unit.unit}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, **data)
                rows.append(dict(paper_method="ImmuneTwins", paper_task=task, donor_id=unit.donor,
                                 season=unit.season, outer_fold=fold, episode_id=unit.unit,
                                 source_day=day, target_day=7 if task == TASKS[0] else 90,
                                 source_path=str(source_path), predicted_path=str(destination.resolve()),
                                 manifest_path=str((output / "predictions.tsv").resolve()),
                                 observed_path=str(base / "observed" / f"{unit.unit}.npz"),
                                 reference_model_path=str(model_path(cfg, task, fold)),
                                 missing_mass=info["missing_mass"]))
                for cell_type, value in zip(predictor.parameters["cell_types"], info["forecast_clr"]):
                    compositions.append(dict(paper_task=task, outer_fold=fold, donor_id=unit.donor,
                                             season=unit.season, cell_type=cell_type, forecast_clr=value,
                                             n_history=info["n_history"], composition_sd=info["composition_sd"]))
    save_table(output / "predictions.tsv", pd.DataFrame(rows))
    save_table(output / "composition.tsv", pd.DataFrame(compositions))
    save_json(output / "complete.json", dict(schema_version="sound_reference_predictions_1",
              units=len(rows), folds=folds, tasks=tasks, future_observations_opened=False))
    return pd.DataFrame(rows)


def _parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--folds", type=int, nargs="+", choices=range(5), default=list(range(5)))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    return parser


def train_main():
    parser = _parser("Fit the personal-reference layer after neural training.")
    parser.add_argument("--step", choices=["all", "inner", "fit"], default="all")
    args = parser.parse_args()
    cfg = load_config(args.config)
    from pipeline.sound.common import require_cuda
    device = require_cuda()
    torch.set_num_threads(4)
    for fold in args.folds:
        if args.step in ("all", "inner"):
            from pipeline.sound.reference_inner import InnerTrainer
            trainer = InnerTrainer(cfg)
            for task in args.tasks:
                trainer.run(fold, task, device)
        if args.step in ("all", "fit"):
            fit_fold(cfg, fold, args.tasks, device)


def predict_main():
    parser = _parser("Apply the final Sound model to held-out prediction units.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    predict(load_config(args.config), args.folds, args.tasks, args.output)


if __name__ == "__main__":
    train_main()

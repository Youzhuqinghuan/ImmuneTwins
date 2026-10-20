"""Prepare and load the inputs for Sound personal-reference fitting and prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from paths import ROOT, expand_paths
from data.sound import TASK_PHASE, canonical_task, decode_array

TASKS = tuple(TASK_PHASE)
DEFAULT_CONFIG = ROOT / "config/sound_reference.json"


def load_config(path=DEFAULT_CONFIG):
    cfg = expand_paths(json.loads(Path(path).read_text()))
    if cfg.get("schema_version") != "sound_personal_reference_1" or cfg["seed"] != 42:
        raise ValueError("Expected the Sound personal-reference configuration with seed 42")
    return cfg


def resolve_path(value, cfg):
    """Relocate an explicit manifest prefix before opening any input file."""
    value = str(expand_paths(value))
    for old, new in cfg.get("path_prefix_replacements", {}).items():
        if value == old or value.startswith(old.rstrip("/") + "/"):
            value = new.rstrip("/") + value[len(old.rstrip("/")):]
            break
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve(strict=True)


def read_table(path, **kwargs):
    return pd.read_csv(path, sep="\t", **kwargs)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=encode) + "\n")
    temporary.replace(path)


def save_table(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def neural_run(cfg, task, fold, *, current=False, affine=False):
    task = canonical_task(task)
    key = "affine" if affine else "current" if current or task == TASKS[0] else (
        "history_t2" if task == TASKS[1] else "history_t3")
    return resolve_path(cfg["neural_runs"][key].format(task=task, fold=fold), cfg)


def scale_arrays(cfg, fold):
    path = resolve_path(cfg["inputs"]["scales"].format(fold=fold), cfg)
    with np.load(path, allow_pickle=False) as data:
        return data["center"].copy(), data["scale"].copy(), float(data["mmd_bandwidth"])


def load_prepared(cfg):
    base = Path(cfg["prepared_root"])
    counts = read_table(base / "counts.tsv", index_col=0).astype(float)
    visits = read_table(base / "visits.tsv", index_col=0)
    visits["rest"] = visits["rest"].astype(str).str.lower().eq("true")
    units = read_table(base / "units.tsv")
    if not counts.index.equals(visits.index):
        raise ValueError("Counts and visits are not aligned")
    if not np.isfinite(counts.values).all() or (counts.values < 0).any():
        raise ValueError("Reference counts must be finite and nonnegative")
    if (counts.sum(axis=1) <= 0).any() or visits[["day", "donor", "batch", "fold"]].isna().any().any():
        raise ValueError("Reference visits have empty counts or missing metadata")
    if units.duplicated(["task", "donor", "season"]).any():
        raise ValueError("Duplicate prediction units")
    for row in units.itertuples():
        if visits.loc[row.source, "donor"] != row.donor or visits.loc[row.target, "donor"] != row.donor:
            raise ValueError("Prediction source and target must belong to the same donor")
        if not visits.loc[row.source, "day"] < visits.loc[row.target, "day"]:
            raise ValueError("Prediction targets must be later than sources")
        if int(visits.loc[row.source, "fold"]) != int(row.fold):
            raise ValueError("Prediction fold differs from donor fold")
    return counts, visits, units


def load_means(cfg, fold):
    return read_table(Path(cfg["prepared_root"]) / f"means_outer{fold}.tsv")


def inner_inputs(cfg):
    return json.loads((Path(cfg["prepared_root"]) / "inner_inputs.json").read_text())


def _summarize_store(path, fold, role, chunk_size=32768):
    """Bounded-memory counts and means, preserving the original summation order."""
    with h5py.File(path, "r") as h:
        if h.attrs.get("status") != "complete" or h["embedding"].shape[1] != 16:
            raise ValueError(f"Invalid frozen RNA-coordinate store: {path}")
        names = ("sample_id", "broad_lineage", "fine_cell_type_L2", "batch")
        cats = {k: decode_array(h[f"obs/{k}/categories"][:]) for k in names}
        ns, nl, nt = (len(cats[k]) for k in names[:3])
        counts = np.zeros((ns, nt), dtype=np.int64)
        lineage_counts = np.zeros(ns * nl, dtype=np.int64)
        sums = np.zeros((ns * nl, 16), dtype=np.float64)
        batches = {}
        n = len(h["embedding"])
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            s, l, t, b = [h[f"obs/{k}/codes"][start:stop] for k in names]
            if min(s.min(), l.min(), t.min(), b.min()) < 0:
                raise ValueError("Unannotated cells are outside the reference input contract")
            counts += np.bincount(s * nt + t, minlength=ns * nt).reshape(ns, nt)
            joint = s * nl + l
            lineage_counts += np.bincount(joint, minlength=ns * nl)
            values = h["embedding"][start:stop].astype(np.float64)
            for j in range(16):
                sums[:, j] += np.bincount(joint, weights=values[:, j], minlength=ns * nl)
            for sample_code, batch_code in np.unique(np.stack([s, b], axis=1), axis=0):
                sample = str(cats["sample_id"][sample_code])
                batch = str(cats["batch"][batch_code])
                if sample in batches and batches[sample] != batch:
                    raise ValueError(f"Multiple batches for sample {sample}")
                batches[sample] = batch
        frame = pd.DataFrame(counts, index=cats["sample_id"], columns=cats["fine_cell_type_L2"])
        frame = frame.loc[frame.sum(axis=1) > 0]
        rows = []
        for i, sample in enumerate(cats["sample_id"]):
            for j, lineage in enumerate(cats["broad_lineage"]):
                k = i * nl + j
                if lineage_counts[k]:
                    rows.append(dict(outer_fold=fold, role=role, sample_id=sample,
                                     lineage=lineage, n_cells=int(lineage_counts[k]),
                                     **{f"mean_{d}": sums[k, d] / lineage_counts[k] for d in range(16)}))
    return frame, batches, pd.DataFrame(rows)


def prepare(cfg, folds):
    """Prepare reference inputs from cell stores and visit metadata."""
    destination = Path(cfg["prepared_root"])
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "preparation.json").exists():
        raise FileExistsError(f"Prepared inputs already exist: {destination}; choose a new prepared_root")
    counts0 = None
    batches0 = None
    records = []
    for fold in folds:
        parts, batches, means = [], {}, []
        for role in ("training", "test"):
            path = resolve_path(cfg["inputs"]["latents"].format(fold=fold, role=role), cfg)
            count, batch, mean = _summarize_store(path, fold, role)
            if batches.keys() & batch.keys():
                raise ValueError("Training and test stores share sample IDs")
            parts.append(count); batches.update(batch); means.append(mean)
            records.append(dict(path=str(path), bytes=path.stat().st_size,
                                fold=fold, role=role, cells=int(count.values.sum())))
        counts = pd.concat(parts).fillna(0).sort_index().sort_index(axis=1)
        counts = counts.loc[:, counts.sum(axis=0) > 0]
        if counts0 is None:
            counts0, batches0 = counts, batches
        elif not counts.equals(counts0) or batches != batches0:
            raise ValueError("Cell labels or sample batches differ across outer coordinate stores")
        save_table(destination / f"means_outer{fold}.tsv", pd.concat(means, ignore_index=True))
        print(f"Prepared outer {fold}: {len(counts)} visits, {len(counts.columns)} cell types", flush=True)
    if counts0 is None:
        raise ValueError("At least one fold is required")
    meta = read_table(resolve_path(cfg["inputs"]["visit_metadata"], cfg)).set_index("sample.sampleKitGuid")
    visits = meta.loc[counts0.index].copy()
    visits["day"] = pd.to_numeric(visits["sample.daysSinceFirstVisit"], errors="raise")
    visits["donor"] = visits["subject.subjectGuid"].astype(str)
    visits["rest"] = visits["sample.visitName"].isin(cfg["resting_visits"])
    outer = read_table(resolve_path(cfg["inputs"]["outer_splits"], cfg))
    visits["fold"] = visits.donor.map(outer.set_index("donor_id").outer_fold)
    visits["batch"] = pd.Series(batches0)
    counts0.index.name = visits.index.name = "sample"
    counts0.to_csv(destination / "counts.tsv", sep="\t")
    visits.to_csv(destination / "visits.tsv", sep="\t")

    episodes = read_table(resolve_path(cfg["inputs"]["task_units"], cfg))
    rows = []
    flags = ("first_vaccination_test", "same_episode_update_test", "cross_year_history_test")
    for task, flag in zip(TASKS, flags):
        for u in episodes.loc[episodes[flag].astype(str).str.lower().eq("true")].itertuples():
            src = u.sample_D7 if task == TASKS[1] else u.sample_D0
            tgt = u.sample_D7 if task == TASKS[0] else u.sample_D90
            rows.append(dict(task=task, donor=u.donor_id, fold=int(u.outer_fold), season=u.season,
                             source=src, target=tgt, unit=f"{u.donor_id}__{u.season}"))
    save_table(destination / "units.tsv", pd.DataFrame(rows).drop_duplicates(["task", "donor", "season"]))
    split_path = resolve_path(cfg["inputs"]["inner_splits"], cfg)
    save_table(destination / "inner_splits.tsv", read_table(split_path))

    features = read_table(resolve_path(cfg["inputs"]["visit_manifest"], cfg))
    for key in ("uce_features_h5", "gene_summary_h5", "raw_counts_h5"):
        if key in features:
            features[key] = features[key].map(lambda x: str(resolve_path(x, cfg)))
    manifest_path = destination / "visit_features.tsv"
    save_table(manifest_path, features)
    inputs = {k: str(resolve_path(cfg["inputs"][k], cfg)) for k in
              ("task_units", "inner_splits", "sealed_source", "sealed_uce", "cima")}
    inputs["visit_manifest"] = str(manifest_path)
    inputs["folds"] = {str(fold): {
        "training_latent": str(resolve_path(cfg["inputs"]["latents"].format(fold=fold, role="training"), cfg)),
        "scale": str(resolve_path(cfg["inputs"]["scales"].format(fold=fold), cfg))
    } for fold in folds}
    save_json(destination / "inner_inputs.json", inputs)
    load_prepared(cfg)
    save_json(destination / "preparation.json", dict(
        schema_version="sound_reference_inputs_1", folds=folds, visits=len(visits),
        cell_types=list(counts0.columns), stores=records, chunk_size=32768))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--folds", type=int, nargs="+", choices=range(5), default=list(range(5)))
    args = parser.parse_args()
    prepare(load_config(args.config), args.folds)


if __name__ == "__main__":
    main()

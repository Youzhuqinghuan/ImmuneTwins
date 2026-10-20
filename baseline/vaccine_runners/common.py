from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

# Method settings shared by the vaccine and treatment adapters.
REFERENCE_PROFILES = {
    "scGen": {"max_epochs": 100, "batch_size": 1024, "patience": 10,
              "cell_type_policy": "measured_native_labels_no_custom_clustering"},
    "trVAE": {"max_epochs": 400, "batch_size": 1024, "patience": 20,
              "lr_patience": 13},
    "CellOT": {"max_iterations": 100000, "batch_size": 256,
               "n_inner_iterations": 10, "eval_every": 500,
               "patience": 20, "inner_patient_folds": 4},
}


def reference_profile(method: str) -> dict:
    """Return an independent copy of the method settings."""
    return copy.deepcopy(REFERENCE_PROFILES[method])


def reference_audit(method: str) -> dict:
    profile = reference_profile(method)
    return {"reference_profile": profile,
            "reference_source": str(Path(__file__).resolve())}


def validate_drug_reference(baseline: dict, method: str) -> dict:
    keys = {
        "scGen": {"max_epochs": "scgen_max_epochs", "batch_size": "batch_size_cells",
                  "patience": "scgen_early_stopping_patience", "cell_type_policy": "scgen_cell_type_policy"},
        "trVAE": {"max_epochs": "trvae_max_epochs", "batch_size": "batch_size_cells",
                  "patience": "trvae_early_stopping_patience", "lr_patience": "trvae_lr_patience"},
        "CellOT": {"max_iterations": "cellot_max_iterations", "batch_size": "cellot_batch_size",
                   "n_inner_iterations": "cellot_inner_iterations", "eval_every": "cellot_eval_every",
                   "patience": "cellot_early_stopping_patience", "inner_patient_folds": "cellot_inner_folds"},
    }
    for name, key in keys[method].items():
        if baseline.get(key) != reference_profile(method)[name]:
            raise ValueError(f"{method} protocol drift: {key}={baseline.get(key)!r}; "
                             f"required {reference_profile(method)[name]!r}")
    return reference_audit(method)


def validate_reference_arguments(method: str, args: argparse.Namespace) -> dict:
    for name, expected in reference_profile(method).items():
        argument = "early_stopping_patience" if method == "CellOT" and name == "patience" else name
        if hasattr(args, argument) and getattr(args, argument) != expected:
            raise ValueError(f"{method} primary protocol drift: --{argument.replace('_', '-')} "
                             f"must remain {expected}")
    return reference_audit(method)


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(ROOT / "config/frozen_contract.yaml"))
    p.add_argument("--fold", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-day", type=int, choices=[1, 7, 28, 90], default=7)
    p.add_argument("--source-day", type=int, choices=[0, 7], default=0)
    p.add_argument("--lineage", default="all")
    p.add_argument("--model", default="audit")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def decode(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def obs_values(obs: h5py.Group, key: str, start: int | None = None, stop: int | None = None):
    obj = obs[key]
    sl = slice(start, stop)
    if isinstance(obj, h5py.Group) and obj.attrs.get("encoding-type") == "categorical":
        categories = np.asarray([decode(x) for x in obj["categories"][:]], dtype=object)
        codes = obj["codes"][sl]
        out = np.empty(len(codes), dtype=object)
        valid = codes >= 0
        out[valid] = categories[codes[valid]]
        out[~valid] = None
        return out
    values = obj[sl]
    if values.dtype.kind in "SO":
        return np.asarray([decode(x) for x in values], dtype=object)
    return values


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

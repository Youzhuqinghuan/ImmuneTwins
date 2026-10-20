"""Published perturbation baselines for the treatment patient folds.

Each method runs in its own process with its own source snapshot and Python
environment. Only outer-training patients are fitted.
"""

from __future__ import annotations
from paths import WORKSPACE
import argparse
import importlib.util
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any, Sequence
import h5py
import numpy as np
import torch
from baseline.vaccine_runners.common import validate_drug_reference
from pipeline.treatment.state import (
    DrugFoldData,
    EpisodeArrays,
    _atomic_json,
    _mmd2,
    _mapping,
    _set_seed,
    _stable_seed,
    estimate_bandwidth,
    load_config,
    require_cuda,
    save_prediction_bundle,
)


RUNTIME = WORKSPACE / "vaccine_baseline_runtime"
SOURCES = {
    "scGen": {
        "paper": "Lotfollahi et al., Nature Methods 2019",
        "source": RUNTIME / "external/scgen",
        "site": RUNTIME / "environments/scgen_site",
        "commit": "d79e1f04233c30f9a4eb5b8d57718127909807d7",
    },
    "CellOT": {
        "paper": "Bunne et al., Nature Methods 2023",
        "source": RUNTIME / "external/cellot",
        "site": RUNTIME / "environments/cellot_site",
        "commit": "522d2b953da8ad244fcf36f64521487fdf763788",
    },
    "trVAE": {
        "paper": "Lotfollahi et al., Bioinformatics 2020",
        "source": RUNTIME / "external/scarches_v051",
        "site": RUNTIME / "environments/scarches_v051_site",
        "commit": "09c23f9489cd8a99cf6356548f471dfdbbe4bd5b",
    },
}

SCGEN_PATIENT_CV = "scGen (native labels + patient CV)"
METHODS = (*SOURCES, "scGenPatientCV")


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values],
        dtype=str,
    )


class RawCellTypeStore:
    """Exact per-visit labels for the cells retained in the GSE309595 stores."""

    def __init__(self, path: str | Path, cells_per_visit: int):
        self.path = Path(path).resolve(strict=True)
        with h5py.File(self.path, "r") as handle:
            visits = _decode(handle["obs/donor_id"][:])
            patients = _decode(handle["obs/patient_id"][:])
            labels = _decode(handle["obs/cell_type"][:])
        if not (len(visits) == len(patients) == len(labels)):
            raise RuntimeError("raw cell metadata columns have different lengths")
        self._labels: dict[str, np.ndarray] = {}
        self._patient: dict[str, str] = {}
        for visit in np.unique(visits):
            positions = np.flatnonzero(visits == visit)
            if len(positions) != cells_per_visit:
                raise RuntimeError(
                    f"raw visit {visit} has {len(positions)} cells, expected {cells_per_visit}"
                )
            visit_patients = np.unique(patients[positions])
            if len(visit_patients) != 1:
                raise RuntimeError(f"raw visit {visit} maps to multiple patients")
            self._labels[str(visit)] = labels[positions]
            self._patient[str(visit)] = str(visit_patients[0])

    def labels(self, visit_id: str, patient_id: str) -> np.ndarray:
        if visit_id not in self._labels:
            raise KeyError(f"raw cell labels absent for visit {visit_id}")
        if self._patient[visit_id] != patient_id:
            raise RuntimeError(f"raw cell labels for {visit_id} have a patient mismatch")
        return self._labels[visit_id].copy()


def _source_audit(method: str) -> dict[str, str]:
    record = SOURCES[method]
    source = Path(record["source"]).resolve(strict=True)
    result = {
        "paper": str(record["paper"]),
        "official_source": str(source),
        "official_commit": str(record["commit"]),
        "site": str(Path(record["site"]).resolve(strict=True)),
    }
    return result


def _stack_train(items: Sequence[EpisodeArrays]) -> tuple[np.ndarray, np.ndarray]:
    source = np.concatenate([item.source_latent for item in items], axis=0).astype(np.float32)
    target = np.concatenate([item.target_latent for item in items], axis=0).astype(np.float32)
    return source, target


def _save_splits(
    fold_root: Path,
    method: str,
    fold: int,
    test_items: Sequence[EpisodeArrays],
    test_prediction: np.ndarray,
    external_items: Sequence[EpisodeArrays],
    external_prediction: np.ndarray,
) -> None:
    directory = fold_root / "predictions" / method
    save_prediction_bundle(
        directory / "test.npz", method, "test", test_items, test_prediction, fold
    )
    if external_items:
        save_prediction_bundle(
            directory / "external.npz",
            method,
            "external",
            external_items,
            external_prediction,
            fold,
        )


def _run_scgen(
    items: Sequence[EpisodeArrays],
    test: Sequence[EpisodeArrays],
    external: Sequence[EpisodeArrays],
    output: Path,
    config: dict[str, Any],
    fold: int,
) -> dict[str, Any]:
    source_audit = _source_audit("scGen")
    sys.path[:0] = [source_audit["site"], source_audit["official_source"]]
    import anndata as ad
    import scgen

    source, target = _stack_train(items)
    source_condition = np.concatenate(
        [np.asarray([f"source_phase{item.phase}"] * len(item.source_latent)) for item in items]
    )
    target_condition = np.concatenate(
        [np.asarray([f"target_phase{item.phase}"] * len(item.target_latent)) for item in items]
    )
    baseline = _mapping(config["baselines"], "baselines")
    label_store = RawCellTypeStore(
        config["data"]["raw_count_store"], int(config["data"]["cells_per_visit"])
    )
    matrix = np.concatenate([source, target], axis=0)
    conditions = np.concatenate([source_condition, target_condition])
    states = np.concatenate([
        _scgen_labels(items, label_store, target=False),
        _scgen_labels(items, label_store, target=True),
    ])
    adata = ad.AnnData(
        X=matrix,
        obs={"condition": conditions, "cell_type": states},
    )
    scgen.SCGEN.setup_anndata(adata, batch_key="condition", labels_key="cell_type")
    model = scgen.SCGEN(adata)
    model.train(
        max_epochs=int(baseline["scgen_max_epochs"]),
        batch_size=int(baseline["batch_size_cells"]),
        early_stopping=True,
        early_stopping_patience=int(baseline["scgen_early_stopping_patience"]),
        check_val_every_n_epoch=1,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
    )
    if str(model.device).split(":")[0] != "cuda":
        raise RuntimeError("official scGen did not train on CUDA")

    categories = sorted(np.unique(states).tolist())

    def predict(values: Sequence[EpisodeArrays]) -> np.ndarray:
        result = []
        for item in values:
            source_key = f"source_phase{item.phase}"
            target_key = f"target_phase{item.phase}"
            labels = label_store.labels(item.source_visit_id, item.patient_id)
            if set(labels) - set(categories):
                raise ValueError("Held-out source cell types are absent from outer training")
            query = ad.AnnData(
                X=item.source_latent,
                obs={
                    "condition": np.asarray([source_key] * len(item.source_latent)),
                    "cell_type": labels,
                },
            )
            query.obs["condition"] = query.obs["condition"].astype("category")
            query.obs["condition"] = query.obs["condition"].cat.set_categories(
                sorted(np.unique(conditions).tolist())
            )
            query.obs["cell_type"] = query.obs["cell_type"].astype("category")
            query.obs["cell_type"] = query.obs["cell_type"].cat.set_categories(categories)
            predicted, _ = model.predict(
                ctrl_key=source_key, stim_key=target_key, adata_to_predict=query
            )
            result.append(np.asarray(predicted.X, dtype=np.float32))
        return np.stack(result)

    test_prediction = predict(test)
    external_prediction = (
        predict(external)
        if external
        else np.empty((0, test_prediction.shape[1], test_prediction.shape[2]), dtype=np.float32)
    )
    checkpoint = output / "checkpoint"
    model.save(checkpoint, overwrite=True, save_anndata=True)
    _save_splits(
        output.parent.parent,
        "scGen",
        fold,
        test,
        test_prediction,
        external,
        external_prediction,
    )
    return {
        **source_audit,
        "implementation": "official scGen repository",
        "training_patients": len(items),
        "training_cells": int(len(matrix)),
        "cell_state_labels": "measured_native_cell_type_as_in_vaccine_scGen",
        "cell_state_categories": categories,
        "reference_protocol": "baseline/vaccine_runners/11_run_scgen_baseline.py",
        "max_epochs": int(baseline["scgen_max_epochs"]),
        "actual_epochs": int(model.trainer.current_epoch),
        "actual_epochs_source": "Lightning_trainer_not_logger_dependent_history",
        "batch_size": int(baseline["batch_size_cells"]),
        "early_stopping_patience": int(baseline["scgen_early_stopping_patience"]),
        "validation_scope": "official_random_cells_from_outer_training_patients_only_same_as_vaccine",
        "hyperparameter_search": False,
        "treatment_action_conditioned": True,
        "checkpoint": str(checkpoint.resolve()),
        "model_parameters": int(sum(parameter.numel() for parameter in model.module.parameters())),
        "scgen_version": getattr(scgen, "__version__", "unknown"),
    }


def _scgen_labels(
    items: Sequence[EpisodeArrays],
    store: RawCellTypeStore,
    *,
    target: bool,
) -> np.ndarray:
    values = []
    for item in items:
        visit_id = item.target_visit_id if target else item.source_visit_id
        labels = store.labels(visit_id, item.patient_id)
        expected = len(item.target_latent) if target else len(item.source_latent)
        if len(labels) != expected:
            raise RuntimeError(f"cell labels and latent rows differ for {visit_id}")
        values.append(labels)
    return np.concatenate(values)


def _fit_scgen_native(
    scgen: Any,
    ad: Any,
    items: Sequence[EpisodeArrays],
    store: RawCellTypeStore,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    model_config: dict[str, Any],
) -> tuple[Any, Any, int]:
    _set_seed(seed)
    source, target = _stack_train(items)
    source_condition = np.concatenate(
        [np.asarray([f"source_phase{item.phase}"] * len(item.source_latent)) for item in items]
    )
    target_condition = np.concatenate(
        [np.asarray([f"target_phase{item.phase}"] * len(item.target_latent)) for item in items]
    )
    states = np.concatenate(
        [_scgen_labels(items, store, target=False), _scgen_labels(items, store, target=True)]
    )
    matrix = np.concatenate([source, target], axis=0)
    conditions = np.concatenate([source_condition, target_condition])
    adata = ad.AnnData(X=matrix, obs={"condition": conditions, "cell_type": states})
    adata.obs_names_make_unique()
    scgen.SCGEN.setup_anndata(adata, batch_key="condition", labels_key="cell_type")
    model = scgen.SCGEN(
        adata,
        n_hidden=int(model_config["n_hidden"]),
        n_latent=int(model_config["n_latent"]),
        n_layers=int(model_config["n_layers"]),
        dropout_rate=float(model_config["dropout_rate"]),
    )
    model.train(
        max_epochs=int(epochs),
        train_size=1.0,
        validation_size=None,
        batch_size=int(batch_size),
        early_stopping=False,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
    )
    if str(model.device).split(":")[0] != "cuda":
        raise RuntimeError("official scGen did not train on CUDA")
    history = model.history or {}
    history_lengths = [len(value) for value in history.values() if hasattr(value, "__len__")]
    actual_epochs = max(history_lengths, default=int(epochs))
    return model, adata, int(actual_epochs)


def _predict_scgen_native(
    model: Any,
    ad: Any,
    training_adata: Any,
    items: Sequence[EpisodeArrays],
    store: RawCellTypeStore,
) -> np.ndarray:
    condition_categories = sorted(
        training_adata.obs["condition"].astype(str).unique().tolist()
    )
    label_categories = sorted(
        training_adata.obs["cell_type"].astype(str).unique().tolist()
    )
    results = []
    for item in items:
        labels = store.labels(item.source_visit_id, item.patient_id)
        unseen = sorted(set(labels) - set(label_categories))
        if unseen:
            raise RuntimeError(
                f"inner-fit scGen lacks native labels required by {item.patient_id}: {unseen}"
            )
        source_key = f"source_phase{item.phase}"
        target_key = f"target_phase{item.phase}"
        if source_key not in condition_categories or target_key not in condition_categories:
            raise RuntimeError(f"inner-fit scGen lacks treatment phase {item.phase}")
        query = ad.AnnData(
            X=item.source_latent,
            obs={
                "condition": np.asarray([source_key] * len(item.source_latent)),
                "cell_type": labels,
            },
        )
        query.obs_names_make_unique()
        query.obs["condition"] = query.obs["condition"].astype("category")
        query.obs["condition"] = query.obs["condition"].cat.set_categories(
            condition_categories
        )
        query.obs["cell_type"] = query.obs["cell_type"].astype("category")
        query.obs["cell_type"] = query.obs["cell_type"].cat.set_categories(label_categories)
        predicted, _ = model.predict(
            ctrl_key=source_key,
            stim_key=target_key,
            adata_to_predict=query,
        )
        results.append(np.asarray(predicted.X, dtype=np.float32))
    return np.stack(results)


def _scgen_patient_splits(
    items: Sequence[EpisodeArrays], folds: int, seed: int
) -> list[tuple[list[EpisodeArrays], list[EpisodeArrays]]]:
    if folds < 2:
        raise ValueError("scGen patient selection requires at least two inner folds")
    assignment: dict[str, int] = {}
    phases = sorted({item.phase for item in items})
    for phase in phases:
        phase_items = [item for item in items if item.phase == phase]
        patients = sorted(
            {item.patient_id for item in phase_items},
            key=lambda patient: _stable_seed(
                "scgen_patient_inner", patient, phase, base=seed
            ),
        )
        if len(patients) < folds:
            raise RuntimeError(
                f"treatment phase {phase} has only {len(patients)} inner-training patients"
            )
        for index, patient in enumerate(patients):
            assignment[patient] = index % folds
    splits = []
    for inner_fold in range(folds):
        validation = [item for item in items if assignment[item.patient_id] == inner_fold]
        fit = [item for item in items if assignment[item.patient_id] != inner_fold]
        if not fit or not validation:
            raise RuntimeError(f"scGen inner fold {inner_fold} is empty")
        if {item.phase for item in validation} - {item.phase for item in fit}:
            raise RuntimeError(f"scGen inner fold {inner_fold} loses a treatment phase")
        splits.append((fit, validation))
    return splits


def _run_scgen_patient_cv(
    items: Sequence[EpisodeArrays],
    test: Sequence[EpisodeArrays],
    external: Sequence[EpisodeArrays],
    output: Path,
    config: dict[str, Any],
    fold: int,
) -> dict[str, Any]:
    source_audit = _source_audit("scGen")
    sys.path[:0] = [source_audit["site"], source_audit["official_source"]]
    import anndata as ad
    import scgen

    baseline = _mapping(config["baselines"], "baselines")
    protocol = dict(baseline.get("scgen_patient_cv", {}))
    required = {
        "inner_folds",
        "max_epoch_grid",
        "selection_metric",
        "n_hidden",
        "n_latent",
        "n_layers",
        "dropout_rate",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"baselines.scgen_patient_cv lacks {missing}")
    if protocol["selection_metric"] != "patient_macro_MMD2":
        raise ValueError("scGen patient CV currently requires patient_macro_MMD2")
    candidates = sorted({int(value) for value in protocol["max_epoch_grid"]})
    if not candidates or candidates[0] <= 0:
        raise ValueError("scGen max_epoch_grid must contain positive epochs")
    data_config = _mapping(config["data"], "data")
    label_store = RawCellTypeStore(
        data_config["raw_count_store"], int(data_config["cells_per_visit"])
    )
    splits = _scgen_patient_splits(
        items,
        folds=int(protocol["inner_folds"]),
        seed=int(config["seed"]) + fold,
    )
    model_config = {
        key: protocol[key]
        for key in ("n_hidden", "n_latent", "n_layers", "dropout_rate")
    }
    selection_records = []
    for epochs in candidates:
        fold_scores = []
        for inner_fold, (fit, validation) in enumerate(splits):
            inner_seed = _stable_seed(
                "scgen_patient_cv", fold, inner_fold, base=int(config["seed"])
            )
            model, adata, actual_epochs = _fit_scgen_native(
                scgen,
                ad,
                fit,
                label_store,
                epochs=epochs,
                batch_size=int(baseline["batch_size_cells"]),
                seed=inner_seed,
                model_config=model_config,
            )
            prediction = _predict_scgen_native(
                model, ad, adata, validation, label_store
            )
            bandwidth = estimate_bandwidth(fit, inner_seed)
            patient_scores = [
                _mmd2(prediction[index], item.target_latent, bandwidth)
                for index, item in enumerate(validation)
            ]
            score = float(np.mean(patient_scores))
            fold_scores.append(score)
            selection_records.append(
                {
                    "candidate_max_epochs": epochs,
                    "inner_fold": inner_fold,
                    "seed": int(inner_seed),
                    "fit_patients": sorted(item.patient_id for item in fit),
                    "validation_patients": sorted(item.patient_id for item in validation),
                    "actual_epochs": actual_epochs,
                    "bandwidth": bandwidth,
                    "patient_macro_MMD2": score,
                }
            )
            del prediction, adata, model
            torch.cuda.empty_cache()
        for record in selection_records:
            if record["candidate_max_epochs"] == epochs:
                record["candidate_mean_inner_MMD2"] = float(np.mean(fold_scores))
    selected_epochs = min(
        candidates,
        key=lambda value: (
            np.mean(
                [
                    record["patient_macro_MMD2"]
                    for record in selection_records
                    if record["candidate_max_epochs"] == value
                ]
            ),
            value,
        ),
    )
    final_seed = int(config["seed"]) + fold
    model, adata, actual_epochs = _fit_scgen_native(
        scgen,
        ad,
        items,
        label_store,
        epochs=selected_epochs,
        batch_size=int(baseline["batch_size_cells"]),
        seed=final_seed,
        model_config=model_config,
    )
    test_prediction = _predict_scgen_native(model, ad, adata, test, label_store)
    external_prediction = (
        _predict_scgen_native(model, ad, adata, external, label_store)
        if external
        else np.empty((0, test_prediction.shape[1], test_prediction.shape[2]), dtype=np.float32)
    )
    checkpoint = output / "checkpoint"
    model.save(checkpoint, overwrite=True, save_anndata=True)
    _save_splits(
        output.parent.parent,
        SCGEN_PATIENT_CV,
        fold,
        test,
        test_prediction,
        external,
        external_prediction,
    )
    categories = sorted(adata.obs["cell_type"].astype(str).unique().tolist())
    return {
        **source_audit,
        "implementation": "official scGen repository with native labels and patient-level selection",
        "comparison_role": "independent_paper_method_protocol_corrected",
        "training_patients": len(items),
        "training_cells": int(adata.n_obs),
        "cell_state_labels": "GSE309595 native cell_type",
        "cell_state_categories": categories,
        "inner_patient_folds": int(protocol["inner_folds"]),
        "selection_metric": str(protocol["selection_metric"]),
        "max_epoch_grid": candidates,
        "selected_max_epochs": int(selected_epochs),
        "actual_final_epochs": int(actual_epochs),
        "selection_records": selection_records,
        "model_config": model_config,
        "optimizer_policy": "official scvi-tools TrainingPlan defaults",
        "training_fraction": 1.0,
        "internal_cell_validation": False,
        "treatment_action_conditioned": True,
        "checkpoint": str(checkpoint.resolve()),
        "model_parameters": int(sum(parameter.numel() for parameter in model.module.parameters())),
        "scgen_version": getattr(scgen, "__version__", "unknown"),
    }


def _decode_trvae(model: Any, latent: np.ndarray, condition: str) -> np.ndarray:
    model.model.eval()
    code = model.model.condition_encoder[condition]
    result = []
    for start in range(0, len(latent), 512):
        value = torch.as_tensor(
            latent[start : start + 512],
            dtype=torch.float32,
            device=next(model.model.parameters()).device,
        )
        condition_code = torch.full(
            (len(value),), code, dtype=torch.long, device=value.device
        )
        with torch.no_grad():
            result.append(model.model.decoder(value, condition_code)[0].cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def _run_trvae(
    items: Sequence[EpisodeArrays],
    test: Sequence[EpisodeArrays],
    external: Sequence[EpisodeArrays],
    output: Path,
    config: dict[str, Any],
    fold: int,
) -> dict[str, Any]:
    source_audit = _source_audit("trVAE")
    sys.path[:0] = [source_audit["site"], source_audit["official_source"]]
    import anndata as ad
    import scarches
    from scarches.models.trvae.trvae_model import TRVAE

    source, target = _stack_train(items)
    combined = np.concatenate([source, target], axis=0)
    center = combined.mean(axis=0)
    scale = np.maximum(combined.std(axis=0), 1e-3)
    matrix = ((combined - center) / scale).astype(np.float32)
    source_condition = np.concatenate(
        [np.asarray([f"source_phase{item.phase}"] * len(item.source_latent)) for item in items]
    )
    target_condition = np.concatenate(
        [np.asarray([f"target_phase{item.phase}"] * len(item.target_latent)) for item in items]
    )
    conditions = np.concatenate([source_condition, target_condition])
    condition_levels = sorted(np.unique(conditions).tolist())
    adata = ad.AnnData(X=matrix, obs={"condition": conditions})
    baseline = _mapping(config["baselines"], "baselines")
    model = TRVAE(
        adata,
        condition_key="condition",
        conditions=condition_levels,
        hidden_layer_sizes=[256, 64],
        latent_dim=10,
        dr_rate=0.05,
        use_mmd=True,
        mmd_on="y",
        recon_loss="mse",
        beta=1,
        use_bn=False,
        use_ln=True,
    )
    model.train(
        n_epochs=int(baseline["trvae_max_epochs"]),
        lr=1e-3,
        eps=0.01,
        batch_size=int(baseline["batch_size_cells"]),
        train_frac=0.9,
        use_stratified_sampling=True,
        n_workers=0,
        seed=int(config["seed"]) + fold,
        monitor=False,
        use_early_stopping=True,
        reload_best=True,
        early_stopping_kwargs={
            "early_stopping_metric": "val_unweighted_loss",
            "patience": int(baseline["trvae_early_stopping_patience"]),
            "threshold": 0,
            "reduce_lr": True,
            "lr_patience": int(baseline["trvae_lr_patience"]),
            "lr_factor": 0.1,
        },
    )
    if next(model.model.parameters()).device.type != "cuda":
        raise RuntimeError("official trVAE did not train on CUDA")

    def predict(values: Sequence[EpisodeArrays]) -> np.ndarray:
        result = []
        for item in values:
            source_key = f"source_phase{item.phase}"
            target_key = f"target_phase{item.phase}"
            query = ((item.source_latent - center) / scale).astype(np.float32)
            latent = model.get_latent(
                query, np.asarray([source_key] * len(query)), mean=True
            )
            decoded = _decode_trvae(model, latent, target_key)
            result.append(decoded * scale + center)
        return np.stack(result).astype(np.float32)

    test_prediction = predict(test)
    external_prediction = (
        predict(external)
        if external
        else np.empty((0, test_prediction.shape[1], test_prediction.shape[2]), dtype=np.float32)
    )
    checkpoint = output / "checkpoint"
    model.save(checkpoint, overwrite=True, save_anndata=True)
    _save_splits(
        output.parent.parent,
        "trVAE",
        fold,
        test,
        test_prediction,
        external,
        external_prediction,
    )
    return {
        **source_audit,
        "implementation": "official scArches trVAE v0.5.1 source snapshot",
        "training_patients": len(items),
        "training_cells": int(len(matrix)),
        "checkpoint": str(checkpoint.resolve()),
        "treatment_action_conditioned": True,
        "model_parameters": int(sum(parameter.numel() for parameter in model.model.parameters())),
        "scarches_version": getattr(scarches, "__version__", "unknown"),
        "actual_epochs": len(model.trainer.logs["epoch_loss"]),
        "best_epoch_zero_based": int(model.trainer.best_epoch),
    }


def _load_cellot_runner() -> Any:
    path = (
        Path(__file__).resolve().parents[1]
        / "baseline/vaccine_runners/12_run_cellot_baseline.py"
    )
    runner_dir = str(path.parent)
    if runner_dir not in sys.path:
        sys.path.insert(0, runner_dir)
    spec = importlib.util.spec_from_file_location("immunetwins_official_cellot_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the source-locked CellOT runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_cellot(
    items: Sequence[EpisodeArrays],
    test: Sequence[EpisodeArrays],
    external: Sequence[EpisodeArrays],
    output: Path,
    config: dict[str, Any],
    fold: int,
) -> dict[str, Any]:
    source_audit = _source_audit("CellOT")
    sys.path[:0] = [source_audit["site"], source_audit["official_source"]]
    from absl import flags

    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["immunetwins_drug_cellot"])
    runner = _load_cellot_runner()
    baseline = _mapping(config["baselines"], "baselines")
    inner_folds = int(baseline["cellot_inner_folds"])
    phase_models: dict[int, tuple[Any, np.ndarray, np.ndarray]] = {}
    phase_checkpoints: dict[str, Any] = {}
    inner_records: list[dict[str, Any]] = []
    training_cells = 0
    for phase in sorted({item.phase for item in items}):
        phase_items = [item for item in items if item.phase == phase]
        source, target = _stack_train(phase_items)
        training_cells += int(2 * len(source))
        combined = np.concatenate([source, target], axis=0)
        center = combined.mean(axis=0).astype(np.float32)
        scale = np.maximum(combined.std(axis=0), 1e-3).astype(np.float32)
        bandwidth = estimate_bandwidth(
            phase_items, int(config["seed"]) + fold * 10 + phase
        )
        # Stratify the tiny phase-specific training cohort across inner folds.
        # A raw hash modulo can leave an action arm without a validation patient,
        # especially for the six-person VRd+Dara arm.
        ordered_patients = sorted(
            {item.patient_id for item in phase_items},
            key=lambda patient_id: _stable_seed(
                "cellot", phase, patient_id, base=int(config["seed"]) + fold
            ),
        )
        effective_inner_folds = min(inner_folds, len(ordered_patients))
        assignments = {
            patient_id: index % effective_inner_folds
            for index, patient_id in enumerate(ordered_patients)
        }
        best_steps: list[int] = []
        model_config = runner.model_config(source.shape[1])
        for inner_fold in range(effective_inner_folds):
            fit = [
                item for item in phase_items if assignments[item.patient_id] != inner_fold
            ]
            validation = [
                item for item in phase_items if assignments[item.patient_id] == inner_fold
            ]
            if not fit or not validation:
                continue
            fit_source, fit_target = _stack_train(fit)
            val_source, val_target = _stack_train(validation)
            result = runner.train_steps(
                (fit_source - center) / scale,
                (fit_target - center) / scale,
                model_config,
                int(config["seed"]) + fold * 100 + phase * 10 + inner_fold,
                int(baseline["cellot_max_iterations"]),
                int(baseline["cellot_batch_size"]),
                int(baseline["cellot_inner_iterations"]),
                "cuda",
                validation=(
                    (val_source - center) / scale,
                    (val_target - center) / scale,
                ),
                eval_every=int(baseline["cellot_eval_every"]),
                patience=int(baseline["cellot_early_stopping_patience"]),
                bandwidth=bandwidth,
            )
            best_steps.append(int(result[3]))
            inner_records.append(
                {
                    "phase": phase,
                    "inner_fold": inner_fold,
                    "validation_patients": sorted(
                        item.patient_id for item in validation
                    ),
                    "best_step": int(result[3]),
                    "best_validation_mmd": float(result[4]),
                    "history": result[5],
                    "actual_steps": int(result[5][-1]["step"]),
                }
            )
            _atomic_json(output / "progress.json", {"status": "RUNNING", "outer_fold": fold,
                         "completed_inner_fits": inner_records})
        if not best_steps:
            raise RuntimeError(f"CellOT phase {phase} produced no inner-fold selection")
        final_steps = max(1, int(math.ceil(float(np.median(best_steps)))))
        f, g, opts, _, _, _, _ = runner.train_steps(
            (source - center) / scale,
            (target - center) / scale,
            model_config,
            int(config["seed"]) + 10_000 + fold * 10 + phase,
            final_steps,
            int(baseline["cellot_batch_size"]),
            int(baseline["cellot_inner_iterations"]),
            "cuda",
        )
        phase_models[phase] = (g, center, scale)
        phase_checkpoints[str(phase)] = {
            "f_state": f.state_dict(),
            "g_state": g.state_dict(),
            "opt_f_state": opts.f.state_dict(),
            "opt_g_state": opts.g.state_dict(),
            "config": model_config,
            "center": center,
            "scale": scale,
            "final_steps": final_steps,
        }

    def predict(values: Sequence[EpisodeArrays]) -> np.ndarray:
        result = []
        for item in values:
            if item.phase not in phase_models:
                raise ValueError(f"CellOT has no outer-training action phase {item.phase}")
            g, center, scale = phase_models[item.phase]
            source_tensor = torch.as_tensor(
                (item.source_latent - center) / scale,
                dtype=torch.float32,
                device="cuda",
            ).requires_grad_(True)
            transported = g.transport(source_tensor).detach().cpu().numpy()
            result.append(transported * scale + center)
        return np.stack(result).astype(np.float32)

    test_prediction = predict(test)
    external_prediction = (
        predict(external)
        if external
        else np.empty((0, test_prediction.shape[1], test_prediction.shape[2]), dtype=np.float32)
    )
    checkpoint = output / "checkpoint.pt"
    torch.save(
        {
            "treatment_action_conditioned": True,
            "phases": phase_checkpoints,
        },
        checkpoint,
    )
    _save_splits(
        output.parent.parent,
        "CellOT",
        fold,
        test,
        test_prediction,
        external,
        external_prediction,
    )
    return {
        **source_audit,
        "implementation": "official CellOT ICNN and losses",
        "training_patients": len(items),
        "training_cells": training_cells,
        "treatment_action_conditioned": True,
        "inner_patient_folds": inner_records,
        "final_steps_by_phase": {
            phase: int(record["final_steps"])
            for phase, record in phase_checkpoints.items()
        },
        "checkpoint": str(checkpoint.resolve()),
        "model_parameters": int(
            sum(
                tensor.numel()
                for record in phase_checkpoints.values()
                for name in ("f_state", "g_state")
                for tensor in record[name].values()
            )
        ),
    }


def run_baseline_fold(
    config_path: str | Path,
    outer_fold: int,
    method: str,
    *,
    device_name: str = "cuda:0",
    output_root: Path | None = None,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown official baseline: {method}")
    started = time.perf_counter()
    config = load_config(config_path)
    protocol_audit = validate_drug_reference(config["baselines"], method) if method in SOURCES else {}
    device = require_cuda(device_name)
    if device.index not in {None, 0}:
        raise ValueError("baseline process must expose its assigned GPU as cuda:0")
    torch.set_num_threads(4)
    seed = int(config["seed"]) + int(outer_fold)
    _set_seed(seed)
    data = DrugFoldData(config_path, outer_fold)
    try:
        train = data.arrays(data.rows("train"))
        test = data.arrays(data.rows("test"))
        external = data.arrays(data.rows("external"))
        root = (output_root or Path(str(_mapping(config["data"], "data")["output_root"]))) / f"fold_{outer_fold}"
        method_label = SCGEN_PATIENT_CV if method == "scGenPatientCV" else method
        output = root / "baselines" / method_label
        if (output / "run.json").exists():
            raise FileExistsError("Completed baseline results are immutable; pass a fresh --output-root")
        output.mkdir(parents=True, exist_ok=True)
        if method == "scGenPatientCV":
            if config["baselines"]["scgen_patient_cv"].get("status", "").startswith("archived"):
                raise ValueError("scGen epoch-grid path is archived; use the fixed vaccine-reference scGen protocol")
            payload = _run_scgen_patient_cv(
                train, test, external, output, config, outer_fold
            )
        elif method == "scGen":
            payload = _run_scgen(train, test, external, output, config, outer_fold)
        elif method == "trVAE":
            payload = _run_trvae(train, test, external, output, config, outer_fold)
        else:
            payload = _run_cellot(train, test, external, output, config, outer_fold)
        result = {
            "schema_version": "immunetwins_drug_official_baseline_fold_v2",
            "status": "PASS",
            "method": method_label,
            "runner_method_key": method,
            "outer_fold": int(outer_fold),
            "seed": seed,
            "patient_disjoint": True,
            "training_patients": len(train),
            "test_patients": len(test),
            "external_patients": len(external),
            "test_future_used_for_training_or_adaptation": False,
            "fold_latent_store": str(data.latent.path),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            **protocol_audit,
            **payload,
        }
        _atomic_json(output / "run.json", result)
        return result
    finally:
        data.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.method.lower() == 'trvae':
        parser.error('Use cli.py baseline trvae --task treatment --fold N for signed trVAE.')
    print(
        json.dumps(
            run_baseline_fold(
                args.config, args.outer_fold, args.method, device_name=args.device,
                output_root=args.output_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

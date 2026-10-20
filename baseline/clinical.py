"""KYDAR response baseline using the official PaSCient implementation.

The upstream ``SamplePredictor``, components, loss and Lightning training
steps are used unchanged; a callback resets metric state between training
and validation. This file supplies the KYDAR reader, patient folds and the
released binary configuration.
"""

from __future__ import annotations
from paths import WORKSPACE
import argparse
import importlib.machinery
import json
import random
import subprocess
import sys
import time
import types
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset


OFFICIAL_REPO = WORKSPACE / "external_models_0714/pascient"
SCHEMA = "immunetwins_pascient_kydar_metricfix_20260905"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values],
        dtype=str,
    )


def _require_cuda(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PaSCient KYDAR training requires a visible CUDA device")
    probe = torch.ones((32, 32), device=device) @ torch.ones((32, 32), device=device)
    if float(probe.sum().item()) <= 0:
        raise RuntimeError("CUDA tensor-operation probe failed")
    return device


def _source_audit(repo: Path) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "paper": "Wang et al., Cell Systems 2026",
        "official_source": str(repo),
        "official_commit": commit,
        "comparison_role": "independent_paper_official_core_with_dataset_adapter",
    }


def _read_gene_order(path: Path) -> np.ndarray:
    genes = np.asarray(
        pd.read_csv(path, sep="\t", header=None).iloc[:, 0].astype(str), dtype=str
    )
    upper = np.char.upper(genes)
    if len(upper) != len(np.unique(upper)):
        raise ValueError("PaSCient gene order contains case-insensitive duplicates")
    return genes


def load_kydar_baseline(
    raw_counts: Path, visits_path: Path, gene_order_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the 28 baseline patient sets in PaSCient's released gene order."""

    raw_counts = raw_counts.resolve(strict=True)
    visits_path = visits_path.resolve(strict=True)
    gene_order_path = gene_order_path.resolve(strict=True)
    visits = pd.read_csv(visits_path, sep="\t")
    baseline = visits.loc[visits["time_point"].eq("Baseline")].copy()
    baseline = baseline.sort_values("patient_id").reset_index(drop=True)
    if len(baseline) != 28 or baseline["patient_id"].nunique() != 28:
        raise RuntimeError("KYDAR baseline must contain 28 unique patients")
    patients = baseline["patient_id"].astype(str).to_numpy()
    labels = baseline["response"].to_numpy(dtype=np.int64)
    if np.bincount(labels, minlength=2).tolist() != [7, 21]:
        raise RuntimeError("KYDAR response class counts changed")

    reference = _read_gene_order(gene_order_path)
    with h5py.File(raw_counts, "r") as handle:
        donor_ids = _decode(handle["obs/donor_id"][:])
        raw_genes = _decode(handle["var/gene_symbol"][:])
        shape = tuple(int(value) for value in handle["matrix"].attrs["shape"])
        matrix = sparse.csr_matrix(
            (
                handle["matrix/data"][:].astype(np.float32, copy=False),
                handle["matrix/indices"][:].astype(np.int32, copy=False),
                handle["matrix/indptr"][:].astype(np.int32, copy=False),
            ),
            shape=shape,
        )
    raw_lookup = {gene.upper(): index for index, gene in enumerate(raw_genes)}
    reference_index, raw_index = [], []
    for index, gene in enumerate(reference):
        matched = raw_lookup.get(gene.upper())
        if matched is not None:
            reference_index.append(index)
            raw_index.append(matched)
    reference_index = np.asarray(reference_index, dtype=np.int64)
    raw_index = np.asarray(raw_index, dtype=np.int64)
    if len(reference_index) < 0.95 * len(reference):
        raise RuntimeError("KYDAR has insufficient overlap with PaSCient gene order")

    cells_per_patient = []
    patient_arrays = []
    for row in baseline.itertuples(index=False):
        positions = np.flatnonzero(donor_ids == str(row.visit_id))
        if not len(positions):
            raise RuntimeError(f"KYDAR raw store lacks {row.visit_id}")
        dense = np.zeros((len(positions), len(reference)), dtype=np.float32)
        dense[:, reference_index] = matrix[positions][:, raw_index].toarray()
        totals = dense.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise RuntimeError(f"KYDAR baseline {row.visit_id} contains an empty cell")
        dense = np.log1p(dense * (10_000.0 / totals)).astype(np.float32)
        patient_arrays.append(dense)
        cells_per_patient.append(len(dense))
    maximum_cells = max(cells_per_patient)
    values = np.zeros(
        (len(patient_arrays), maximum_cells, len(reference)), dtype=np.float32
    )
    mask = np.zeros((len(patient_arrays), maximum_cells), dtype=bool)
    for index, dense in enumerate(patient_arrays):
        values[index, : len(dense)] = dense
        mask[index, : len(dense)] = True
    audit = {
        "raw_counts": str(raw_counts),
        "visits": str(visits_path),
        "gene_order": str(gene_order_path),
        "patients": len(patients),
        "responders": int(labels.sum()),
        "nonresponders": int(len(labels) - labels.sum()),
        "genes": len(reference),
        "overlap_genes": len(reference_index),
        "cells_per_patient_min": min(cells_per_patient),
        "cells_per_patient_max": maximum_cells,
        "normalization": "per_cell_CP10000_log1p_official_PaSCient_rule",
    }
    return values, mask, patients, labels, audit


class _IndexDataset(Dataset):
    def __init__(self, indices: Sequence[int]):
        self.indices = [int(value) for value in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> int:
        return self.indices[index]


def _components(repo: Path) -> dict[str, Any]:
    repo_text = str(repo.resolve())
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    # The pinned source imports s3fs at import time; a stub avoids an unused
    # dependency.
    if "s3fs" not in sys.modules:
        s3fs_stub = types.ModuleType("s3fs")
        s3fs_stub.__spec__ = importlib.machinery.ModuleSpec("s3fs", loader=None)
        s3fs_stub.S3FileSystem = type("UnavailableS3FileSystem", (), {})
        sys.modules["s3fs"] = s3fs_stub

    return locals()


def _collate(
    rows: Sequence[int], values: np.ndarray, mask: np.ndarray, labels: np.ndarray, classes: dict[str, Any]
) -> Any:
    index = np.asarray(rows, dtype=np.int64)
    return classes["SampleBatch"](
        x=torch.from_numpy(values[index]).unsqueeze(1),
        padded_mask=torch.from_numpy(mask[index]).unsqueeze(1),
        sample_metadata={"response": torch.from_numpy(labels[index])},
        cell_metadata=classes["CellMetaData"](cell_level_labels={}),
        view_names=["view_0"],
    )


def _loader(
    indices: Sequence[int], values: np.ndarray, mask: np.ndarray, labels: np.ndarray,
    classes: dict[str, Any], *, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        _IndexDataset(indices),
        batch_size=32,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        collate_fn=partial(
            _collate, values=values, mask=mask, labels=labels, classes=classes
        ),
    )


def _build_model(
    n_genes: int, class_counts: np.ndarray, classes: dict[str, Any]
) -> Any:
    """Construct the released ``pascient_binary`` architecture explicitly."""

    gene_encoder = classes["BasicMLP"](
        input_dim=n_genes,
        hidden_dim=1024,
        output_dim=512,
        n_hidden_layers=-1,
    )
    patient_encoder = classes["BasicMLP"](
        input_dim=1024,
        hidden_dim=512,
        output_dim=512,
        n_hidden_layers=0,
        activation_cls=nn.PReLU,
        activation_out_cls=nn.PReLU,
    )
    patient_predictor = classes["BasicMLP"](
        input_dim=512,
        hidden_dim=2,
        output_dim=1,
        n_hidden_layers=-1,
    )
    losses = SimpleNamespace(
        sample_prediction_loss=SimpleNamespace(
            weight=1.0,
            loss_fn=partial(
                classes["CrossEntropyLossViews"],
                weight=class_counts.astype(float).tolist(),
                label_smoothing=0.0,
            ),
            labels=["response"],
        )
    )
    model = classes["SamplePredictor"](
        num_genes=n_genes,
        optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=1e-4),
        scheduler=partial(
            torch.optim.lr_scheduler.ReduceLROnPlateau,
            mode="min",
            factor=0.1,
            patience=20,
        ),
        masking_strategy=classes["DummyMasking"](),
        gene2cell_encoder=gene_encoder,
        cell2cell_encoder=classes["CellToCellIdentity"](),
        cell2patient_aggregation=classes["LinearAttnAggregator"](),
        cell2output=classes["CellToOutputNone"](),
        patient_encoder=patient_encoder,
        cell_decoder=None,
        patient_predictor=patient_predictor,
        dropout=0.0,
        metrics=[
            classes["AccuracyMetric"](
                task="multiclass",
                num_classes=2,
                labels=["response"],
                name="sample_accuracy",
            )
        ],
        losses=losses,
        compile=False,
        cell_contrastive_strategy="classic",
    )
    signature = {
        "gene_to_cell": [
            (layer.in_features, layer.out_features)
            for layer in model.gene2cell_encoder.layers
            if isinstance(layer, nn.Linear)
        ],
        "patient_encoder": [
            (layer.in_features, layer.out_features)
            for layer in model.patient_encoder.layers
            if isinstance(layer, nn.Linear)
        ],
        "patient_predictor": [
            (layer.in_features, layer.out_features)
            for layer in model.patient_predictor.layers
            if isinstance(layer, nn.Linear)
        ],
    }
    expected = {
        "gene_to_cell": [(n_genes, 1024)],
        "patient_encoder": [(1024, 512), (512, 512)],
        "patient_predictor": [(512, 2)],
    }
    if signature != expected:
        raise RuntimeError(
            f"PaSCient resolved architecture differs from pinned binary config: {signature}"
        )
    return model


def _move_batch(batch: Any, device: torch.device, classes: dict[str, Any]) -> Any:
    return classes["SampleBatch"](
        x=batch.x.to(device),
        padded_mask=batch.padded_mask.to(device),
        sample_metadata={key: value.to(device) for key, value in batch.sample_metadata.items()},
        cell_metadata=classes["CellMetaData"](cell_level_labels={}),
        view_names=list(batch.view_names),
    )


def _predict(model: Any, loader: DataLoader, device: torch.device, classes: dict[str, Any]) -> np.ndarray:
    model.eval().to(device)
    probabilities = []
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, device, classes)
            cells = model.gene2cell_encoder(batch.x)
            cells = model.cell2cell_encoder(cells, padding_mask=batch.padded_mask)
            patient = model.cell2patient_aggregation.aggregate(
                data=cells, mask=batch.padded_mask
            )
            patient = model.patient_encoder(patient)
            logits = model.patient_predictor(patient)[:, 0]
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32)


def _fit(
    *, values: np.ndarray, mask: np.ndarray, labels: np.ndarray,
    fit: np.ndarray, validation: np.ndarray, output: Path, device: torch.device,
    classes: dict[str, Any], seed: int, max_epochs: int
) -> tuple[Any, dict[str, Any]]:
    import lightning as lightning
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    lightning.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = _build_model(
        values.shape[2], np.bincount(labels[fit], minlength=2), classes
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=output,
        filename="best",
        monitor="val/sample_accuracy",
        mode="max",
        save_top_k=1,
    )
    stopping = EarlyStopping(
        monitor="val/sample_accuracy", mode="max", patience=100
    )
    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=[device.index or 0],
        max_epochs=max_epochs,
        min_epochs=1,
        deterministic=False,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        logger=False,
        enable_progress_bar=False,
        callbacks=[_metric_lifecycle_callback(), checkpoint, stopping],
        num_sanity_val_steps=0,
    )
    started = time.perf_counter()
    trainer.fit(
        model,
        train_dataloaders=_loader(
            fit, values, mask, labels, classes, shuffle=True, seed=seed
        ),
        val_dataloaders=_loader(
            validation, values, mask, labels, classes, shuffle=False, seed=seed
        ),
    )
    if not checkpoint.best_model_path:
        raise RuntimeError("PaSCient did not produce a validation checkpoint")
    saved = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"])
    validation_probability = _predict(
        model, _loader(validation, values, mask, labels, classes, shuffle=False, seed=seed),
        device, classes,
    )
    direct_accuracy = float(np.mean((validation_probability > 0.5) == labels[validation]))
    if not np.isclose(direct_accuracy, float(checkpoint.best_model_score), atol=1e-6):
        raise RuntimeError("Saved validation accuracy does not match restored-model predictions")
    return model, {
        "seed": seed,
        "fit_patients": fit.tolist(),
        "validation_patients": validation.tolist(),
        "class_counts": np.bincount(labels[fit], minlength=2).tolist(),
        "best_epoch": int(saved["epoch"]) + 1,
        "best_validation_accuracy": float(checkpoint.best_model_score),
        "restored_validation_accuracy": direct_accuracy,
        "metric_scope": "validation_patients_only_current_epoch_explicit_recursive_reset",
        "checkpoint": str(Path(checkpoint.best_model_path).resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _reset_metrics(model: Any) -> None:
    """Reset the wrapper AND its child: TorchMetrics.reset is not recursive."""
    from torchmetrics import Metric

    for metric in model.metrics:
        for component in metric.modules():
            if isinstance(component, Metric):
                component.reset()


def _metric_lifecycle_callback() -> Any:
    from lightning.pytorch.callbacks import Callback

    class PhaseLocalMetrics(Callback):
        def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
            _reset_metrics(pl_module)

        def on_validation_epoch_start(self, trainer: Any, pl_module: Any) -> None:
            _reset_metrics(pl_module)

    return PhaseLocalMetrics()


def _refit(
    *, values: np.ndarray, mask: np.ndarray, labels: np.ndarray,
    fit: np.ndarray, device: torch.device, classes: dict[str, Any], seed: int,
    epochs: int
) -> tuple[Any, dict[str, Any]]:
    """Refit the selected official architecture on every outer-training patient."""

    import lightning as lightning

    lightning.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = _build_model(
        values.shape[2], np.bincount(labels[fit], minlength=2), classes
    )
    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=[device.index or 0],
        max_epochs=epochs,
        min_epochs=epochs,
        deterministic=False,
        log_every_n_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[_metric_lifecycle_callback()],
    )
    started = time.perf_counter()
    trainer.fit(
        model,
        train_dataloaders=_loader(
            fit, values, mask, labels, classes, shuffle=True, seed=seed
        ),
        val_dataloaders=_loader(
            fit, values, mask, labels, classes, shuffle=False, seed=seed
        ),
    )
    return model, {
        "seed": seed,
        "fit_patients": fit.tolist(),
        "class_counts": np.bincount(labels[fit], minlength=2).tolist(),
        "epochs": epochs,
        "scheduler_monitor": "outer_training_patients_val_loss_no_model_selection",
        "elapsed_seconds": time.perf_counter() - started,
    }


def run(
    *, raw_counts: Path, visits: Path, gene_order: Path, output: Path,
    device_name: str, repo: Path = OFFICIAL_REPO, max_epochs: int = 100,
    outer_fold_only: int | None = None
) -> dict[str, Any]:
    source = _source_audit(repo)
    device = _require_cuda(device_name)
    values, mask, patients, labels, data_audit = load_kydar_baseline(
        raw_counts, visits, gene_order
    )
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists() or any(output.glob("fold_*/run.json")):
        raise FileExistsError("Use a fresh output directory; completed results are immutable")
    classes = _components(repo)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(len(labels), np.nan, dtype=np.float32)
    folds = []
    torch.cuda.reset_peak_memory_stats(device)
    for outer_fold, (fit, test) in enumerate(splitter.split(values, labels)):
        if outer_fold_only is not None and outer_fold != outer_fold_only:
            continue
        fold_root = output / f"fold_{outer_fold}"
        inner = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=142 + outer_fold
        )
        member_records = []
        for inner_fold, (inner_fit, inner_validation) in enumerate(
            inner.split(values[fit], labels[fit])
        ):
            model, record = _fit(
                values=values,
                mask=mask,
                labels=labels,
                fit=fit[inner_fit],
                validation=fit[inner_validation],
                output=fold_root / f"inner_{inner_fold}",
                device=device,
                classes=classes,
                seed=12_345 + outer_fold * 10 + inner_fold,
                max_epochs=max_epochs,
            )
            record.update(
                {
                    "inner_fold": inner_fold,
                    "fit_patient_ids": patients[fit[inner_fit]].tolist(),
                    "validation_patient_ids": patients[fit[inner_validation]].tolist(),
                }
            )
            member_records.append(record)
            del model
            torch.cuda.empty_cache()
        selected_epoch = int(
            np.median([record["best_epoch"] for record in member_records])
        )
        model, refit_record = _refit(
            values=values,
            mask=mask,
            labels=labels,
            fit=fit,
            device=device,
            classes=classes,
            seed=22_345 + outer_fold,
            epochs=selected_epoch,
        )
        refit_record["fit_patient_ids"] = patients[fit].tolist()
        test_loader = _loader(
            test, values, mask, labels, classes, shuffle=False, seed=42
        )
        probabilities = _predict(model, test_loader, device, classes)
        oof[test] = probabilities
        prediction_frame = pd.DataFrame(
            {
                "patient_id": patients[test],
                "label": labels[test],
                "response_probability": probabilities,
                "outer_fold": outer_fold,
            }
        )
        prediction_frame.to_csv(
            fold_root / "predictions.tsv", sep="\t", index=False
        )
        fold_record = {
            "outer_fold": outer_fold,
            "train_patient_ids": patients[fit].tolist(),
            "test_patient_ids": patients[test].tolist(),
            "patient_disjoint": not bool(set(patients[fit]) & set(patients[test])),
            "inner_members": member_records,
            "selected_epoch": selected_epoch,
            "outer_training_refit": refit_record,
            "selection_and_refit": "median_inner_selected_epoch_then_refit_all_outer_training_patients",
        }
        _atomic_json(fold_root / "run.json", fold_record)
        folds.append(fold_record)
        del model
        torch.cuda.empty_cache()

    selected = np.flatnonzero(np.isfinite(oof))
    prediction_frame = pd.DataFrame(
        {
            "patient_id": patients[selected],
            "label": labels[selected],
            "response_probability": oof[selected],
        }
    )
    prediction_frame.to_csv(output / "predictions.tsv", sep="\t", index=False)
    metrics = None
    if len(selected) == len(labels):
        metrics = {
            "Method": "PaSCient (official core, nested patient CV)",
            "Patients": len(labels),
            "AUROC": float(roc_auc_score(labels, oof)),
            "AUPRC": float(average_precision_score(labels, oof)),
            "Brier": float(brier_score_loss(labels, oof)),
        }
        pd.DataFrame([metrics]).to_csv(
            output / "method_by_metric.tsv", sep="\t", index=False
        )
    result = {
        "schema_version": SCHEMA,
        "status": "PASS" if len(folds) == (1 if outer_fold_only is not None else 5) else "FAIL",
        **source,
        "data": data_audit,
        "outer_fold_only": outer_fold_only,
        "outer_folds_complete": len(folds),
        "max_epochs": max_epochs,
        "official_binary_defaults": {
            "released_sample_size": 100,
            "batch_size": 32,
            "gene_to_cell": "Linear(num_genes,1024)",
            "cell_to_cell": "Identity",
            "cell_to_patient": "LinearAttnAggregator",
            "patient_encoder": "1024-512-512_PReLU",
            "patient_predictor": "Linear(512,2)",
            "loss": "CrossEntropyLossViews_reciprocal_training_class_counts",
            "optimizer": "Adam(lr=1e-4,weight_decay=1e-4)",
            "scheduler": "ReduceLROnPlateau(factor=0.1,patience=20)",
            "validation_monitor": "sample_accuracy",
            "metric_lifecycle_repair": "explicit_recursive_reset_at_train_and_validation_epoch_start",
            "outer_refit": "median_inner_selected_epoch_then_all_outer_training_patients",
            "compatibility": "unused_s3fs_import_stub_only",
        },
        "dataset_adapter": {
            "effective_cells_per_patient": int(mask.sum(axis=1).min()),
            "cell_sampling": "fixed_common_64_cell_patient_store_no_epoch_resampling",
            "target_label": "KYDAR_treatment_response_instead_of_released_disease_label",
            "class_weights": "reciprocal_outer_training_class_counts",
            "outer_split_and_selection": "project_common_patient_disjoint_nested_CV",
        },
        "folds": folds,
        "metrics": metrics,
        "patient_disjoint": all(record["patient_disjoint"] for record in folds),
        "test_labels_or_cells_used_for_training_or_selection": False,
        "gpu": torch.cuda.get_device_name(device),
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
    }
    _atomic_json(output / "run.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-counts", required=True, type=Path)
    parser.add_argument("--visits", required=True, type=Path)
    parser.add_argument("--gene-order", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--outer-fold-only", type=int, choices=range(5))
    args = parser.parse_args()
    result = run(
        raw_counts=args.raw_counts,
        visits=args.visits,
        gene_order=args.gene_order,
        output=args.output,
        device_name=args.device,
        max_epochs=args.max_epochs,
        outer_fold_only=args.outer_fold_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def perception_main():
    """Apply retained PERCEPTION probability calibration or refit its original nested protocol."""
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import StratifiedKFold
    from paths import ROOT
    parser = argparse.ArgumentParser(description=perception_main.__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--calibrators', type=Path)
    parser.add_argument('--refit', action='store_true', help='Original sklearn calibration: 65 small CPU fits')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.calibrators) == args.refit:
        parser.error('Choose --calibrators for inference or --refit for nested calibration.')
    args.output.resolve().relative_to(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    p = pd.read_csv(args.predictions, sep='\t', float_precision='round_trip').sort_values('patient_id')
    if not p.patient_id.is_unique: raise ValueError('Duplicate patient scores.')
    scores = p.response_score.to_numpy(float)
    probabilities = np.full(len(p), np.nan)
    records = []
    if args.calibrators:
        records = json.loads(args.calibrators.read_text())['folds']
        covered = np.zeros(len(p), dtype=int)
        for fold in records:
            if fold['classes'] != [0,1]: raise ValueError('Unexpected calibration classes.')
            idx = p.patient_id.isin(fold['test_patients']).to_numpy()
            covered[idx] += 1
            probabilities[idx] = expit(scores[idx]*fold['coefficient'][0][0]+fold['intercept'][0])
        if not (covered == 1).all(): raise ValueError('Each patient needs one held-out calibrator.')
    else:
        y = p.label.to_numpy(int)
        outer = StratifiedKFold(5, shuffle=True, random_state=42)
        for fold, (tr,te) in enumerate(outer.split(scores[:,None],y)):
            inner = StratifiedKFold(3,shuffle=True,random_state=142+fold)
            candidates = []
            for C in (.01,.1,1.,10.):
                ip = np.full(len(tr),np.nan)
                for itr,ite in inner.split(scores[tr,None],y[tr]):
                    ip[ite] = LogisticRegression(C=C,max_iter=5000).fit(scores[tr][itr,None],y[tr][itr]).predict_proba(scores[tr][ite,None])[:,1]
                candidates.append((brier_score_loss(y[tr],ip),C))
            C = min(candidates)[1]
            model = LogisticRegression(C=C,max_iter=5000).fit(scores[tr,None],y[tr])
            probabilities[te] = model.predict_proba(scores[te,None])[:,1]
            records.append(dict(fold=fold,selected_C=C,coefficient=model.coef_.tolist(),
                intercept=model.intercept_.tolist(),classes=model.classes_.tolist(),
                train_patients=p.patient_id.iloc[tr].tolist(),test_patients=p.patient_id.iloc[te].tolist(),
                candidates=[dict(C=c,Brier=float(b)) for b,c in candidates]))
    if not np.isfinite(probabilities).all(): raise ValueError('Incomplete calibrated probabilities.')
    p['response_probability'] = probabilities
    p.to_csv(args.output/'patient_predictions.tsv',sep='\t',index=False)
    (args.output/'calibrators.json').write_text(json.dumps({'folds':records},indent=2)+'\n')

if __name__ == "__main__":
    main()

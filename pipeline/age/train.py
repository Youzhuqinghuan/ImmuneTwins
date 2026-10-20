from __future__ import annotations
from paths import expand_paths
import argparse
import contextlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader
from pipeline.age.checkpoint import TargetScaler, atomic_json, load_checkpoint, require_cuda, save_checkpoint, set_global_seed
from pipeline.age.inference import predict_external_shards
from data.age import DEFAULT_CIMA_FOLDS, DEFAULT_CIMA_UCE, DEFAULT_EXTERNAL_UCE_SHARDS, CimaDonorDataset, CimaUceStore, ImmunobiologyUceShardStore, collate_donors, load_common_cima_folds
from model.age import AgeModelConfig, ImmuneTwinAgeModel

@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 8
    inference_batch_size: int = 16
    cell_batch_size: int = 4096
    max_epochs: int = 120
    min_epochs: int = 15
    patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    amp: str = "bf16"

    def __post_init__(self) -> None:
        if self.seed != 42:
            raise ValueError("the frozen ImmuneTwin algorithm seed is 42")
        if self.amp not in {"none", "bf16"}:
            raise ValueError("amp must be none or bf16")
        for name in (
            "batch_size",
            "inference_batch_size",
            "cell_batch_size",
            "max_epochs",
            "min_epochs",
            "patience",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.min_epochs > self.max_epochs:
            raise ValueError("min_epochs cannot exceed max_epochs")
        if self.num_workers != 0:
            raise ValueError(
                "the 2.04-GiB CIMA member cache is process-local; formal training requires num_workers=0"
            )

def paper_method_name(pooling: str) -> str:
    if pooling == "attention":
        return "ImmuneTwin"
    if pooling == "mean":
        return "ImmuneTwin-Mean"
    raise ValueError(pooling)

def require_inner_selection_converged(
    *, epochs_run: int, max_epochs: int, stale_epochs: int, patience: int, outer_fold: int
) -> None:
    if epochs_run >= max_epochs and stale_epochs < patience:
        raise RuntimeError(
            f"outer fold {outer_fold} inner selection reached max_epochs={max_epochs} "
            f"with stale={stale_epochs} < patience={patience}; increase --max-epochs"
        )


def _loader(
    store: CimaUceStore,
    donor_ids: Sequence[str],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        CimaDonorDataset(store, donor_ids),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_donors,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

def _amp_context(device: torch.device, amp: str):
    if amp == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()

def _new_model(model_config: AgeModelConfig, device: torch.device) -> ImmuneTwinAgeModel:
    return ImmuneTwinAgeModel(model_config).to(device)

def _fit_one_epoch(
    model: ImmuneTwinAgeModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: TargetScaler,
    device: torch.device,
    config: TrainingConfig,
) -> float:
    model.train()
    total_loss = 0.0
    total_donors = 0
    for batch in loader:
        cells = batch["cells"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        age = batch["age"].to(device, non_blocking=True)
        target = scaler.transform_tensor(age)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, config.amp):
            prediction = model(cells, mask)
            loss = nn.functional.mse_loss(prediction.float(), target.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        count = len(age)
        total_loss += float(loss.detach()) * count
        total_donors += count
    return total_loss / total_donors

@torch.inference_mode()
def _scaled_mse(
    model: ImmuneTwinAgeModel,
    loader: DataLoader,
    scaler: TargetScaler,
    device: torch.device,
    amp: str,
) -> float:
    model.eval()
    squared_error = 0.0
    total_donors = 0
    for batch in loader:
        cells = batch["cells"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        age = batch["age"].to(device, non_blocking=True)
        with _amp_context(device, amp):
            prediction = model(cells, mask)
        target = scaler.transform_tensor(age)
        squared_error += float(torch.sum((prediction.float() - target.float()) ** 2))
        total_donors += len(age)
    return squared_error / total_donors

@torch.inference_mode()
def _predict_cima(
    model: ImmuneTwinAgeModel,
    scaler: TargetScaler,
    loader: DataLoader,
    device: torch.device,
    amp: str,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, float | str]] = []
    for batch in loader:
        cells = batch["cells"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with _amp_context(device, amp):
            scaled = model(cells, mask)
        predicted = scaler.inverse_tensor(scaled.float()).cpu().numpy()
        observed = batch["age"].numpy()
        rows.extend(
            {
                "donor_id": donor_id,
                "age": float(age),
                "predicted_age": float(prediction),
            }
            for donor_id, age, prediction in zip(batch["donor_id"], observed, predicted)
        )
    return pd.DataFrame(rows)

def _ages(store: CimaUceStore, donor_ids: Sequence[str]) -> np.ndarray:
    return np.asarray([store.age(donor_id) for donor_id in donor_ids], dtype=np.float64)

def select_epoch_from_inner_cima_fold(
    store: CimaUceStore,
    folds: pd.DataFrame,
    outer_fold: int,
    model_config: AgeModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
) -> tuple[int, list[dict[str, float | int]], int]:
    """Select the epoch using one CIMA fold nested inside the outer training set."""
    inner_validation_fold = (outer_fold + 1) % 5
    inner_train = folds.loc[
        ~folds["fold"].isin([outer_fold, inner_validation_fold]), "donor_id"
    ].tolist()
    inner_validation = folds.loc[
        folds["fold"].eq(inner_validation_fold), "donor_id"
    ].tolist()
    if set(inner_train) & set(inner_validation):
        raise RuntimeError("inner CIMA donor leakage")
    scaler = TargetScaler.fit(_ages(store, inner_train))
    set_global_seed(training_config.seed + outer_fold)
    model = _new_model(model_config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    train_loader = _loader(
        store,
        inner_train,
        batch_size=training_config.batch_size,
        shuffle=True,
        seed=training_config.seed + outer_fold,
        num_workers=training_config.num_workers,
    )
    validation_loader = _loader(
        store,
        inner_validation,
        batch_size=training_config.inference_batch_size,
        shuffle=False,
        seed=training_config.seed,
        num_workers=training_config.num_workers,
    )
    history: list[dict[str, float | int]] = []
    best_epoch = 1
    best_loss = math.inf
    stale = 0
    stopped_by_patience = False
    for epoch in range(1, training_config.max_epochs + 1):
        train_loss = _fit_one_epoch(
            model, train_loader, optimizer, scaler, device, training_config
        )
        validation_loss = _scaled_mse(
            model, validation_loader, scaler, device, training_config.amp
        )
        history.append(
            {
                "epoch": epoch,
                "inner_train_scaled_mse": train_loss,
                "inner_validation_scaled_mse": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= training_config.min_epochs and stale >= training_config.patience:
            stopped_by_patience = True
            break
    if not stopped_by_patience:
        require_inner_selection_converged(
            epochs_run=len(history),
            max_epochs=training_config.max_epochs,
            stale_epochs=stale,
            patience=training_config.patience,
            outer_fold=outer_fold,
        )
    return best_epoch, history, inner_validation_fold

def train_exact_epochs(
    store: CimaUceStore,
    donor_ids: Sequence[str],
    model_config: AgeModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
    epochs: int,
    seed_offset: int,
) -> tuple[ImmuneTwinAgeModel, TargetScaler, list[float]]:
    scaler = TargetScaler.fit(_ages(store, donor_ids))
    set_global_seed(training_config.seed + seed_offset)
    model = _new_model(model_config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    loader = _loader(
        store,
        donor_ids,
        batch_size=training_config.batch_size,
        shuffle=True,
        seed=training_config.seed + seed_offset,
        num_workers=training_config.num_workers,
    )
    losses = [
        _fit_one_epoch(model, loader, optimizer, scaler, device, training_config)
        for _ in range(epochs)
    ]
    return model, scaler, losses

def regression_metrics(observed: Sequence[float], predicted: Sequence[float]) -> dict:
    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    residual = p - y
    y_sd = float(y.std(ddof=0))
    return {
        "n": len(y),
        "mae_years": float(np.mean(np.abs(residual))),
        "rmse_years": float(np.sqrt(np.mean(residual**2))),
        "mean_signed_bias_years": float(residual.mean()),
        "pearson": float(stats.pearsonr(y, p).statistic),
        "spearman": float(stats.spearmanr(y, p).statistic),
        "calibration_slope": float(np.polyfit(y, p, 1)[0]),
        "predicted_age_SD_over_observed_age_SD": float(p.std(ddof=0) / y_sd),
    }

def _atomic_csv(frame: pd.DataFrame, path: Path, *, sep: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, sep=sep, lineterminator="\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def run_outer_fold(
    store: CimaUceStore,
    folds: pd.DataFrame,
    outer_fold: int,
    model_config: AgeModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
    output: Path,
    resume: bool,
) -> pd.DataFrame:
    fold_dir = output / f"fold_{outer_fold}"
    if fold_dir.exists() and any(fold_dir.iterdir()):
        if resume:
            return validate_completed_fold(
                fold_dir,
                outer_fold=outer_fold,
                pooling=model_config.pooling,
                expected_donors=folds.loc[
                    folds["fold"].eq(outer_fold), "donor_id"
                ].tolist(),
            )
        raise FileExistsError(f"non-empty fold output already exists: {fold_dir}")
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_epoch, history, inner_validation_fold = select_epoch_from_inner_cima_fold(
        store, folds, outer_fold, model_config, training_config, device
    )
    outer_train = folds.loc[~folds["fold"].eq(outer_fold), "donor_id"].tolist()
    held_out = folds.loc[folds["fold"].eq(outer_fold), "donor_id"].tolist()
    model, scaler, refit_losses = train_exact_epochs(
        store,
        outer_train,
        model_config,
        training_config,
        device,
        best_epoch,
        seed_offset=outer_fold,
    )
    checkpoint = fold_dir / "checkpoint.pt"
    save_checkpoint(
            checkpoint,
            model,
            scaler,
            seed=training_config.seed + outer_fold,
            epochs=best_epoch,
            outer_fold=outer_fold,
            training_donor_ids=outer_train,
        )
    held_out_loader = _loader(
        store,
        held_out,
        batch_size=training_config.inference_batch_size,
        shuffle=False,
        seed=training_config.seed,
        num_workers=training_config.num_workers,
    )
    predictions = _predict_cima(
        model, scaler, held_out_loader, device, training_config.amp
    )
    predictions["fold"] = outer_fold
    predictions["method"] = paper_method_name(model_config.pooling)
    predictions["cohort"] = "CIMA"
    columns = ["method", "cohort", "donor_id", "age", "predicted_age", "fold"]
    prediction_path = fold_dir / "held_out_predictions.csv"
    _atomic_csv(predictions[columns].sort_values("donor_id"), prediction_path)
    atomic_json(
        fold_dir / "selection.json",
        {
            "status": "PASS",
            "outer_fold": outer_fold,
            "inner_validation_fold": inner_validation_fold,
            "held_out_outer_fold_used_for_early_stopping": False,
            "best_epoch": best_epoch,
            "selection_max_epochs": training_config.max_epochs,
            "inner_selection_stopped_by_patience": True,
            "n_outer_train_donors": len(outer_train),
            "n_held_out_donors": len(held_out),
            "inner_selection_history": history,
            "outer_train_refit_scaled_mse": refit_losses,
            "target_scaler": scaler.to_dict(),
        },
    )
    return predictions[columns]

def validate_completed_fold(
    fold_dir: Path,
    *,
    outer_fold: int,
    pooling: str,
    expected_donors: Sequence[str],
) -> pd.DataFrame:
    selection_path = fold_dir / "selection.json"
    prediction_path = fold_dir / "held_out_predictions.csv"
    checkpoint_path = fold_dir / "checkpoint.pt"
    for path in (selection_path, prediction_path, checkpoint_path):
        if not path.is_file():
            raise ValueError(f"cannot resume incomplete fold; missing {path}")
    selection = expand_paths(json.loads(selection_path.read_text()))
    if (
        selection.get('status') != 'PASS'
        or int(selection.get('outer_fold', -1)) != outer_fold
        or selection.get('held_out_outer_fold_used_for_early_stopping') is not False
        or selection.get('inner_selection_stopped_by_patience') is not True
    ):
        raise ValueError(f"cannot resume fold with a failed artifact audit: {fold_dir}")
    _, _, checkpoint = load_checkpoint(checkpoint_path, "cpu")
    if (
        checkpoint.get("outer_fold") != outer_fold
        or checkpoint["model_config"].get("pooling") != pooling
    ):
        raise ValueError(f"fold checkpoint identity failed: {checkpoint_path}")
    predictions = pd.read_csv(prediction_path)
    expected_columns = ["method", "cohort", "donor_id", "age", "predicted_age", "fold"]
    if predictions.columns.tolist() != expected_columns:
        raise ValueError(f"fold prediction schema failed: {prediction_path}")
    if (
        set(predictions["donor_id"].astype(str)) != set(map(str, expected_donors))
        or set(predictions["method"]) != {paper_method_name(pooling)}
        or set(predictions["cohort"]) != {"CIMA"}
        or set(predictions["fold"].astype(int)) != {outer_fold}
        or not np.isfinite(predictions[["age", "predicted_age"]]).all().all()
    ):
        raise ValueError(f"fold prediction content failed: {prediction_path}")
    print(f"Resume: verified and reused {fold_dir}", flush=True)
    return predictions

def assemble_oof(output: Path, pooling: str) -> tuple[pd.DataFrame, list[int]]:
    parts: list[pd.DataFrame] = []
    epochs: list[int] = []
    for fold in range(5):
        fold_dir = output / f"fold_{fold}"
        selection = expand_paths(json.loads((fold_dir / "selection.json").read_text()))
        if (
            selection.get("status") != "PASS"
            or selection.get("held_out_outer_fold_used_for_early_stopping") is not False
            or selection.get("inner_selection_stopped_by_patience") is not True
        ):
            raise ValueError(f"fold {fold} selection contract failed")
        checkpoint_path = fold_dir / "checkpoint.pt"
        _, _, checkpoint = load_checkpoint(checkpoint_path, "cpu")
        if (
            checkpoint.get("outer_fold") != fold
            or checkpoint["model_config"].get("pooling") != pooling
        ):
            raise ValueError(f"fold {fold} checkpoint identity mismatch")
        prediction_path = fold_dir / "held_out_predictions.csv"
        parts.append(pd.read_csv(prediction_path))
        epochs.append(int(selection["best_epoch"]))
    oof = pd.concat(parts, ignore_index=True).sort_values("donor_id").reset_index(drop=True)
    if len(oof) != 417 or oof["donor_id"].nunique() != 417:
        raise ValueError("OOF predictions do not cover 417 unique CIMA donors")
    if (
        oof.columns.tolist()
        != ["method", "cohort", "donor_id", "age", "predicted_age", "fold"]
        or set(oof["method"]) != {paper_method_name(pooling)}
        or set(oof["cohort"]) != {"CIMA"}
        or set(oof["fold"].astype(int)) != set(range(5))
        or not np.isfinite(oof[["age", "predicted_age"]]).all().all()
    ):
        raise ValueError("OOF prediction content contract failed")
    oof_path = output / "cima_common5fold_oof.csv"
    _atomic_csv(oof, oof_path)
    atomic_json(
        output / "cima_common5fold_oof.manifest.json",
        {
            "status": "PASS",
            "method": paper_method_name(pooling),
            "cohort": "CIMA",
            "n_donors": 417,
            "split_repeat": 0,
            "outer_folds": 5,
            "metrics": regression_metrics(oof["age"], oof["predicted_age"]),
            "selected_epochs": epochs,
            "all_inner_selectors_stopped_by_patience_before_boundary": all(
                expand_paths(json.loads((output / f"fold_{fold}/selection.json").read_text())).get(
                    "inner_selection_stopped_by_patience"
                )
                is True
                for fold in range(5)
            ),
        },
    )
    return oof, epochs

def run_full_training(
    store: CimaUceStore,
    folds: pd.DataFrame,
    selected_epochs: Sequence[int],
    model_config: AgeModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
    output: Path,
    resume: bool,
) -> None:
    if len(selected_epochs) != 5:
        raise ValueError("full-CIMA epoch count requires all five CIMA outer folds")
    full_epochs = int(statistics.median(map(int, selected_epochs)))
    full_dir = output / "full_cima"
    checkpoint = full_dir / "checkpoint.pt"
    training_path = full_dir / "training.json"
    if full_dir.exists() and any(full_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"non-empty full-CIMA output already exists: {full_dir}")
        validate_completed_full_training(
            full_dir,
            pooling=model_config.pooling,
            selected_epochs=selected_epochs,
            device=device,
        )
        print(f"Resume: verified and reused full-CIMA model {full_dir}", flush=True)
        return
    full_dir.mkdir(parents=True, exist_ok=True)
    donor_ids = folds["donor_id"].tolist()
    model, scaler, losses = train_exact_epochs(
        store,
        donor_ids,
        model_config,
        training_config,
        device,
        full_epochs,
        seed_offset=100,
    )
    save_checkpoint(
            checkpoint,
            model,
            scaler,
            seed=training_config.seed + 100,
            epochs=full_epochs,
            outer_fold=None,
            training_donor_ids=donor_ids,
        )
    atomic_json(
        training_path,
        {
            "status": "PASS",
            "n_training_donors": len(donor_ids),
            "selected_epochs_from_CIMA_outer_folds": list(map(int, selected_epochs)),
            "full_epochs_rule": "integer median of five CIMA outer-fold selected epochs",
            "full_epochs": full_epochs,
            "full_train_scaled_mse": losses,
            "target_scaler": scaler.to_dict(),
            "external_information_used_for_training_or_epoch_selection": False,
        },
    )

def run_external(
    *,
    selected_epochs: Sequence[int],
    model_config: AgeModelConfig,
    device: torch.device,
    external_root: Path,
    output: Path,
    cell_batch_size: int,
    resume: bool,
) -> None:
    full_dir = output / "full_cima"
    external_artifacts = (
        full_dir / "immunobiology_raw_predictions.csv",
        full_dir / "immunobiology_expected_donors.tsv",
        full_dir / "immunobiology_raw_predictions.manifest.json",
    )
    if all(path.is_file() for path in external_artifacts):
        if not resume:
            raise FileExistsError(f"external outputs already exist: {full_dir}")
        validate_completed_full(
            full_dir, model_config.pooling
        )
        print(f"Resume: verified and reused external predictions {full_dir}", flush=True)
        return
    if any(path.exists() for path in external_artifacts) and not resume:
        raise FileExistsError(f"partial external outputs already exist: {full_dir}")
    model, scaler = validate_completed_full_training(
        full_dir,
        pooling=model_config.pooling,
        selected_epochs=selected_epochs,
        device=device,
    )
    external_store = ImmunobiologyUceShardStore(
        external_root
    )
    expected_donors = external_store.expected_donor_manifest()
    expected_path = full_dir / "immunobiology_expected_donors.tsv"
    _atomic_csv(expected_donors, expected_path, sep="\t")
    external = predict_external_shards(
        model,
        scaler,
        external_store,
        device,
        cell_batch_size=cell_batch_size,
    )
    if set(external["donor_id"]) != set(expected_donors["donor_id"]):
        raise ValueError("predictions do not cover the independently frozen donor manifest")
    raw = pd.DataFrame(
        {
            "method": paper_method_name(model_config.pooling),
            "cohort": "Immunobiology",
            "donor_id": external["donor_id"],
            "predicted_age": external["predicted_age"],
        }
    ).sort_values("donor_id")
    prediction_path = full_dir / "immunobiology_raw_predictions.csv"
    _atomic_csv(raw, prediction_path)
    atomic_json(
        full_dir / "immunobiology_raw_predictions.manifest.json",
        {
            "status": "PASS",
            "formal_protocol_eligible": True,
            "schema": ["method", "cohort", "donor_id", "predicted_age"],
            "method": paper_method_name(model_config.pooling),
            "cohort": "Immunobiology",
            "n_predictions": len(raw),
            "labels_present": False,
            "external_label_path_opened": False,
            "target_cohort_statistics_used": False,
            "prediction_is_per_donor_function_only": True,
            "expected_donors_frozen_before_model_inference": True,
            "all_CIMA_models_frozen_before_external": True,
            "external_uce_audit": external_store.audit(),
        },
    )

def validate_completed_full_training(
    full_dir: Path,
    *,
    pooling: str,
    selected_epochs: Sequence[int],
    device: torch.device,
) -> tuple[ImmuneTwinAgeModel, TargetScaler]:
    training_path = full_dir / "training.json"
    checkpoint_path = full_dir / "checkpoint.pt"
    if not training_path.is_file() or not checkpoint_path.is_file():
        raise ValueError(
            f"cannot resume: full-CIMA checkpoint/training manifest is incomplete in {full_dir}"
        )
    training = expand_paths(json.loads(training_path.read_text()))
    expected_epochs = list(map(int, selected_epochs))
    if (
        training.get('status') != 'PASS'
        or training.get('n_training_donors') != 417
        or training.get('selected_epochs_from_CIMA_outer_folds') != expected_epochs
        or training.get('full_epochs') != int(statistics.median(expected_epochs))
        or training.get('external_information_used_for_training_or_epoch_selection') is not False
    ):
        raise ValueError(f"full-CIMA training artifact audit failed: {full_dir}")
    model, scaler, checkpoint = load_checkpoint(checkpoint_path, device)
    if (
        checkpoint.get("outer_fold") is not None
        or checkpoint["model_config"].get("pooling") != pooling
        or len(checkpoint.get("training_donor_ids", [])) != 417
    ):
        raise ValueError(f"full-CIMA checkpoint identity failed: {checkpoint_path}")
    return model, scaler

def validate_completed_full(
    full_dir: Path, pooling: str
) -> None:
    training_path = full_dir / "training.json"
    checkpoint_path = full_dir / "checkpoint.pt"
    prediction_path = full_dir / "immunobiology_raw_predictions.csv"
    expected_path = full_dir / "immunobiology_expected_donors.tsv"
    manifest_path = full_dir / "immunobiology_raw_predictions.manifest.json"
    for path in (
        training_path,
        checkpoint_path,
        prediction_path,
        expected_path,
        manifest_path,
    ):
        if not path.is_file():
            raise ValueError(f"cannot resume incomplete full run; missing {path}")
    training = expand_paths(json.loads(training_path.read_text()))
    manifest = expand_paths(json.loads(manifest_path.read_text()))
    if (
        training.get('status') != 'PASS'
        or manifest.get('status') not in {'PASS', 'READY_FOR_INDEPENDENT_SEAL'}
        or manifest.get('labels_present') is not False
        or manifest.get('external_label_path_opened') is not False
        or manifest.get('target_cohort_statistics_used') is not False
        or manifest.get('all_CIMA_models_frozen_before_external') is not True
    ):
        raise ValueError(f"cannot resume full run with a failed artifact audit: {full_dir}")
    _, _, checkpoint = load_checkpoint(checkpoint_path, "cpu")
    if checkpoint["model_config"].get("pooling") != pooling:
        raise ValueError(f"full checkpoint pooling failed: {checkpoint_path}")
    predictions = pd.read_csv(prediction_path)
    expected_columns = ["method", "cohort", "donor_id", "predicted_age"]
    expected = pd.read_csv(expected_path, sep="\t")
    if (
        predictions.columns.tolist() != expected_columns
        or expected.columns.tolist() != ["donor_id"]
        or len(predictions) != 234
        or predictions["donor_id"].nunique() != 234
        or set(predictions["donor_id"].astype(str)) != set(expected["donor_id"].astype(str))
        or set(predictions["method"]) != {paper_method_name(pooling)}
        or set(predictions["cohort"]) != {"Immunobiology"}
        or not np.isfinite(predictions["predicted_age"]).all()
    ):
        raise ValueError(f"full external prediction content failed: {prediction_path}")


def _cima_uce_input_identity(path: Path) -> dict[str, object]:
    qc_path = path.with_suffix(".qc.json")
    if not qc_path.is_file():
        raise FileNotFoundError(
            f"formal CIMA model freeze requires the UCE QC manifest: {qc_path}"
        )
    qc = expand_paths(json.loads(qc_path.read_text()))
    if (
        qc.get('pass') is not True
        or Path(str(qc.get('output', ''))).resolve() != path.resolve()
    ):
        raise ValueError(f"CIMA UCE QC identity failed: {qc_path}")
    return {
        "path": str(path.resolve()),
        "qc_manifest": str(qc_path.resolve()),
    }

def write_cima_frozen_models_manifest(
    *,
    run_output: Path,
    poolings: Sequence[str],
    selected_epochs: dict[str, list[int]],
    model_configs: dict[str, AgeModelConfig],
    training_config: TrainingConfig,
    cima_h5ad: Path,
    cima_folds: Path,
) -> dict:
    if set(poolings) != {"attention", "mean"}:
        raise ValueError("external freeze requires both attention and mean CIMA models")
    models: dict[str, dict] = {}
    for pooling in ("attention", "mean"):
        pooling_output = run_output / pooling
        checkpoint_path = pooling_output / "full_cima/checkpoint.pt"
        training_path = pooling_output / "full_cima/training.json"
        oof_manifest_path = pooling_output / "cima_common5fold_oof.manifest.json"
        training = expand_paths(json.loads(training_path.read_text()))
        oof_manifest = expand_paths(json.loads(oof_manifest_path.read_text()))
        if (
            training.get('status') != 'PASS'
            or training.get('selected_epochs_from_CIMA_outer_folds') != list(map(int, selected_epochs[pooling]))
            or oof_manifest.get('status') != 'PASS'
            or oof_manifest.get('all_inner_selectors_stopped_by_patience_before_boundary') is not True
        ):
            raise ValueError(f"cannot freeze invalid CIMA artifacts for {pooling}")
        models[pooling] = {
            "paper_method": paper_method_name(pooling),
            "model_config": model_configs[pooling].to_dict(),
            "checkpoint": str(checkpoint_path.resolve()),
            "selected_epochs": list(map(int, selected_epochs[pooling])),
            "full_epochs": int(training["full_epochs"]),
        }
    training_payload = asdict(training_config)
    payload = {
        "format": "immunetwin_cima_frozen_models_v1",
        "status": "FROZEN_BEFORE_EXTERNAL",
        "all_CIMA_models_frozen_before_external": True,
        "external_cells_opened": False,
        "poolings": ["attention", "mean"],
        "training_config": training_payload,
        "cima_uce_input": _cima_uce_input_identity(cima_h5ad),
        "cima_split": {
            "path": str(cima_folds.resolve()),
            "repeat": 0,
        },
        "models": models,
    }
    path = run_output / "cima_frozen_models.json"
    atomic_json(path, payload)
    return payload

def load_cima_frozen_models_manifest(run_output: Path) -> dict:
    path = run_output / "cima_frozen_models.json"
    if not path.is_file():
        raise FileNotFoundError(
            "external inference is blocked until cima_frozen_models.json exists"
        )
    payload = expand_paths(json.loads(path.read_text()))
    if (
        payload.get("status") != "FROZEN_BEFORE_EXTERNAL"
        or payload.get("all_CIMA_models_frozen_before_external") is not True
        or payload.get("external_cells_opened") is not False
        or set(payload.get("models", {})) != {"attention", "mean"}
    ):
        raise ValueError("CIMA frozen-model manifest contract failed")
    return payload

def validate_cima_contract(store: CimaUceStore, folds: pd.DataFrame) -> None:
    if store.member_column != "member_20260711":
        raise ValueError("formal age training requires exactly member_20260711")
    if len(folds) != 417 or set(folds["donor_id"]) != set(store.donor_ids):
        raise ValueError("CIMA store/common fold donor coverage failed")
    observed = np.asarray([store.age(item) for item in folds["donor_id"]])
    if not np.array_equal(observed, folds["age"].to_numpy(dtype=float)):
        raise ValueError("CIMA H5AD ages differ from common fold ages")

def parse_fold_list(value: str) -> list[int]:
    if value == "all":
        return list(range(5))
    folds = sorted({int(item) for item in value.split(",") if item.strip()})
    if not folds or any(fold not in range(5) for fold in folds):
        raise argparse.ArgumentTypeError("folds must be all or comma-separated values 0..4")
    return folds

def main() -> int:
    parser = argparse.ArgumentParser(
        description="CIMA-only ImmuneTwin attention and mean age pretraining"
    )
    parser.add_argument(
        "--stage", choices=("oof", "full", "external", "all"), default="all"
    )
    parser.add_argument("--pooling", choices=("attention", "mean", "both"), default="both")
    parser.add_argument("--folds", type=parse_fold_list, default=parse_fold_list("all"))
    parser.add_argument("--cima-h5ad", type=Path, default=DEFAULT_CIMA_UCE)
    parser.add_argument("--cima-folds", type=Path, default=DEFAULT_CIMA_FOLDS)
    parser.add_argument("--external-shards", type=Path, default=DEFAULT_EXTERNAL_UCE_SHARDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--cell-batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed fold/full units with existing checkpoints and predictions",
    )
    args = parser.parse_args()
    if args.stage in {"full", "external", "all"} and (
        args.pooling != "both" or set(args.folds) != set(range(5))
    ):
        parser.error("full/external/all stages require --pooling both --folds all")
    device = require_cuda(args.device)
    config = TrainingConfig(
        batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
        cell_batch_size=args.cell_batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        amp=args.amp,
    )
    store = CimaUceStore(args.cima_h5ad, member_column="member_20260711")
    folds = load_common_cima_folds(args.cima_folds, expected_donors=store.donor_ids)
    validate_cima_contract(store, folds)
    if args.stage != "external":
        print("Caching CIMA member_20260711 UCE cells once in process RAM...", flush=True)
        cima_cache_audit = store.cache_member_view()
        print(json.dumps(cima_cache_audit, sort_keys=True), flush=True)
    else:
        cima_cache_audit = store.cache_audit()
    poolings = ["attention", "mean"] if args.pooling == "both" else [args.pooling]
    model_configs = {pooling: AgeModelConfig(pooling=pooling) for pooling in poolings}
    args.output.mkdir(parents=True, exist_ok=True)
    run_record = {
        "format": "immunetwin_age_run_v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "poolings": poolings,
        "folds": args.folds,
        "training_config": asdict(config),
        "model_configs": {
            pooling: model_configs[pooling].to_dict() for pooling in poolings
        },
        "cima_h5ad": str(args.cima_h5ad.resolve()),
        "cima_h5ad_member": "member_20260711",
        "cima_member_cache": cima_cache_audit,
        "cima_folds": str(args.cima_folds.resolve()),
        "external_shards": str(args.external_shards.resolve()),
        "formal_external_run": True,
        "external_labels_accessed": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
    }
    atomic_json(args.output / "run_manifest.started.json", run_record)
    if args.stage in {"oof", "all"}:
        for pooling in poolings:
            pooling_output = args.output / pooling
            pooling_output.mkdir(parents=True, exist_ok=True)
            for outer_fold in args.folds:
                run_outer_fold(
                    store,
                    folds,
                    outer_fold,
                    model_configs[pooling],
                    config,
                    device,
                    pooling_output,
                    args.resume,
                )

    selected_by_pooling: dict[str, list[int]] = {}
    for pooling in poolings:
        pooling_output = args.output / pooling
        pooling_output.mkdir(parents=True, exist_ok=True)
        if set(args.folds) == set(range(5)) and all(
            (pooling_output / f"fold_{fold}/selection.json").is_file()
            for fold in range(5)
        ):
            _, selected_epochs = assemble_oof(pooling_output, pooling)
        else:
            selected_epochs = []
        selected_by_pooling[pooling] = selected_epochs

    if args.stage in {"full", "all"}:
        for pooling in poolings:
            selected_epochs = selected_by_pooling[pooling]
            if len(selected_epochs) != 5:
                raise ValueError("full stage requires completed CIMA folds 0..4")
            run_full_training(
                store,
                folds,
                selected_epochs,
                model_configs[pooling],
                config,
                device,
                args.output / pooling,
                resume=args.resume,
            )

    if args.stage in {"full", "all"}:
        frozen_manifest = write_cima_frozen_models_manifest(
            run_output=args.output,
            poolings=poolings,
            selected_epochs=selected_by_pooling,
            model_configs=model_configs,
            training_config=config,
            cima_h5ad=args.cima_h5ad,
            cima_folds=args.cima_folds,
        )
    elif args.stage == "external":
        frozen_manifest = load_cima_frozen_models_manifest(
            args.output
        )
    else:
        frozen_manifest = None

    if frozen_manifest is not None:
        run_record["all_CIMA_models_frozen_before_external"] = True

    if args.stage in {"external", "all"}:
        if frozen_manifest is None:
            raise RuntimeError("external phase lacks the pre-external CIMA freeze")
        for pooling in poolings:
            run_external(
                selected_epochs=selected_by_pooling[pooling],
                model_config=model_configs[pooling],
                device=device,
                external_root=args.external_shards,
                output=args.output / pooling,
                cell_batch_size=config.cell_batch_size,
                resume=args.resume,
            )
    run_record["status"] = "PASS"
    run_record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(args.output / "run_manifest.json", run_record)
    return 0

if __name__ == "__main__":
    main()

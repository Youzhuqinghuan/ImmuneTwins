from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import random
import time
import h5py
import numpy as np
import pandas as pd
import torch
from evaluation.units import _atomic_json
from evaluation.controls import  _stable_seed
from baseline.bridge_model import VaccineBridgeConfig, VaccineBridgeModel, load_cima_mean_projection
from data.latent import Reader

SCHEMA = "sound_vaccine_pretraining_bridge_1"

METHOD_LABELS = {
    "uce": "UCE",
    "uce_cima": "CIMA-pretrained UCE",
    "pulsar": "PULSAR",
}

EXPECTED_TEST_UNITS = {
    "t1_y1_d0_to_y1_d7": 83,
    "t2_d7_to_d90": 163,
    "t3_y2_d0_to_y2_d90": 75,
}

@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 300
    patience: int = 20
    cells_per_training_condition: int = 512
    generated_cells: int = 2048

class FeatureStore:
    def __init__(self, path: Path, method: str):
        self.path = path.resolve(strict=True)
        self.handle = h5py.File(self.path, "r")
        if self.handle.attrs.get("status") != "complete":
            raise RuntimeError(f"incomplete bridge feature store: {self.path}")
        self.visit_ids = np.asarray(
            [value.decode() if isinstance(value, bytes) else str(value)
             for value in self.handle["visit_id"][:]],
            dtype=str,
        )
        if len(self.visit_ids) != 321 or len(set(self.visit_ids)) != 321:
            raise RuntimeError("bridge feature visit index is invalid")
        self.index = {visit: index for index, visit in enumerate(self.visit_ids)}
        expected = (321, 512) if method == "pulsar" else (321, 512, 1280)
        observed = self.handle["embedding"].shape
        if observed != expected:
            raise RuntimeError(f"feature shape is {observed}, expected {expected}")

    def take(self, visit_ids: list[str]) -> np.ndarray:
        positions = np.asarray([self.index[visit] for visit in visit_ids], dtype=int)
        order = np.argsort(positions)
        sorted_values = np.asarray(self.handle["embedding"][positions[order]], dtype=np.float32)
        inverse = np.argsort(order)
        return sorted_values[inverse]

    def close(self) -> None:
        self.handle.close()

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _model(method: str, seed: int, cima_checkpoint: Path | None) -> tuple[VaccineBridgeModel, list[str]]:
    _set_seed(seed)
    if method == "pulsar":
        config = VaccineBridgeConfig(input_kind="pulsar", input_dim=512)
    else:
        config = VaccineBridgeConfig(input_kind="uce_cells", input_dim=1280)
    model = VaccineBridgeModel(config)
    loaded: list[str] = []
    if method == "uce_cima":
        if cima_checkpoint is None:
            raise ValueError("uce_cima requires a CIMA checkpoint")
        checkpoint = torch.load(cima_checkpoint, map_location="cpu", weights_only=False)
        loaded = load_cima_mean_projection(model, checkpoint)
    return model, loaded

def _delta_targets(
    reader: Reader,
    visits: pd.DataFrame,
    *,
    cells: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    deltas: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for _, visit in visits.reset_index(drop=True).iterrows():
        source_rows = reader.rows(str(visit["model_source_sample"]))
        target_rows = reader.rows(str(visit["model_target_sample"]))
        if not len(source_rows) or not len(target_rows):
            raise RuntimeError(f"training visit is absent from fold latent store: {visit['visit_id']}")
        source_seed = _stable_seed(
            "bridge_train_cells", visit["paper_task"], visit["visit_id"], "source",
            base_seed=seed,
        )
        target_seed = _stable_seed(
            "bridge_train_cells", visit["paper_task"], visit["visit_id"], "target",
            base_seed=seed,
        )
        source = reader.take(source_rows, cells, np.random.default_rng(source_seed), exact=True)[0]
        target = reader.take(target_rows, cells, np.random.default_rng(target_seed), exact=True)[0]
        delta = target.mean(axis=0) - source.mean(axis=0)
        if delta.shape != (16,) or not np.isfinite(delta).all():
            raise RuntimeError("invalid 16-D training response")
        deltas.append(delta.astype(np.float32))
        records.append(
            {
                "visit_id": str(visit["visit_id"]),
                "source_seed": int(source_seed),
                "target_seed": int(target_seed),
                "source_cells": cells,
                "target_cells": cells,
            }
        )
    return np.vstack(deltas), records

def _scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale = np.maximum(scale, 1e-3)
    return center, scale

def _batches(length: int, batch_size: int, seed: int) -> list[np.ndarray]:
    generator = np.random.default_rng(seed)
    order = generator.permutation(length)
    return [order[start:start + batch_size] for start in range(0, length, batch_size)]

def _epoch(
    model: VaccineBridgeModel,
    optimizer: torch.optim.Optimizer,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for positions in _batches(len(features), batch_size, seed):
        x = torch.as_tensor(features[positions], device=device)
        y = torch.as_tensor(targets[positions], device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(x)
            loss = torch.nn.functional.mse_loss(prediction.float(), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))

def _loss(
    model: VaccineBridgeModel,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            x = torch.as_tensor(features[start:stop], device=device)
            y = torch.as_tensor(targets[start:stop], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(x)
            losses.append(float(torch.nn.functional.mse_loss(prediction.float(), y).cpu()))
    return float(np.mean(losses))

def _select_epoch(
    method: str,
    features: np.ndarray,
    deltas: np.ndarray,
    fit: np.ndarray,
    validation: np.ndarray,
    *,
    config: TrainConfig,
    device: torch.device,
    cima_checkpoint: Path | None,
) -> tuple[int, list[dict[str, float]]]:
    center, scale = _scaler(deltas[fit])
    targets = (deltas - center) / scale
    model, _ = _model(method, config.seed, cima_checkpoint)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, float]] = []
    best_epoch = 1
    best = np.inf
    stale = 0
    for epoch in range(1, config.max_epochs + 1):
        train_loss = _epoch(
            model, optimizer, features[fit], targets[fit],
            batch_size=config.batch_size,
            seed=config.seed + epoch,
            device=device,
        )
        validation_loss = _loss(
            model, features[validation], targets[validation],
            batch_size=config.batch_size, device=device,
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best - 1e-6:
            best = validation_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    return best_epoch, history

def _select_epoch_all_folds(
    method: str,
    features: np.ndarray,
    deltas: np.ndarray,
    masks: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    config: TrainConfig,
    device: torch.device,
    cima_checkpoint: Path | None,
) -> tuple[int, dict[str, object]]:
    """Select one epoch from the common donor-blocked four-fold curve."""

    fold_records: list[dict[str, object]] = []
    histories: list[list[dict[str, float]]] = []
    for inner_fold, fit, validation in masks:
        best_epoch, history = _select_epoch(
            method,
            features,
            deltas,
            fit,
            validation,
            config=config,
            device=device,
            cima_checkpoint=cima_checkpoint,
        )
        histories.append(history)
        fold_records.append(
            {
                "inner_fold": inner_fold,
                "fit_units": int(fit.sum()),
                "validation_units": int(validation.sum()),
                "fold_best_epoch": best_epoch,
                "history": history,
            }
        )
        torch.cuda.empty_cache()
    common_epochs = min(len(history) for history in histories)
    if common_epochs < 1:
        raise RuntimeError("four-fold PULSAR selection produced an empty common curve")
    common_curve = []
    for position in range(common_epochs):
        validation = [history[position]["validation_loss"] for history in histories]
        training = [history[position]["train_loss"] for history in histories]
        common_curve.append(
            {
                "epoch": position + 1,
                "mean_train_loss": float(np.mean(training)),
                "mean_validation_loss": float(np.mean(validation)),
                "validation_loss_by_inner_fold": validation,
            }
        )
    selected = min(
        common_curve,
        key=lambda record: (float(record["mean_validation_loss"]), int(record["epoch"])),
    )
    return int(selected["epoch"]), {
        "selection_rule": "minimum_mean_loss_on_common_four_fold_validation_curve",
        "common_epochs": common_epochs,
        "per_inner_fold": fold_records,
        "common_validation_curve": common_curve,
    }

def _fit_final(
    method: str,
    features: np.ndarray,
    deltas: np.ndarray,
    epochs: int,
    *,
    config: TrainConfig,
    device: torch.device,
    cima_checkpoint: Path | None,
) -> tuple[VaccineBridgeModel, np.ndarray, np.ndarray, list[str], list[float]]:
    center, scale = _scaler(deltas)
    targets = (deltas - center) / scale
    model, loaded = _model(method, config.seed, cima_checkpoint)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    losses = []
    for epoch in range(1, epochs + 1):
        losses.append(_epoch(
            model, optimizer, features, targets,
            batch_size=config.batch_size,
            seed=config.seed + epoch,
            device=device,
        ))
    return model, center, scale, loaded, losses

def _predict(
    model: VaccineBridgeModel,
    features: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            x = torch.as_tensor(features[start:start + batch_size], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                values = model(x)
            chunks.append(values.float().cpu().numpy())
    standardized = np.vstack(chunks)
    result = standardized * scale + center
    if result.shape != (len(features), 16) or not np.isfinite(result).all():
        raise RuntimeError("invalid bridge response prediction")
    return result.astype(np.float32)

def _save_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHOD_LABELS), required=True)
    parser.add_argument("--task", choices=tuple(EXPECTED_TEST_UNITS), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--feature-audit", required=True, type=Path)
    parser.add_argument("--source-visits", required=True, type=Path)
    parser.add_argument("--inner-splits", required=True, type=Path)
    parser.add_argument("--training-latent", required=True, type=Path)
    parser.add_argument("--test-latent", required=True, type=Path)
    parser.add_argument("--cima-checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--inner-selection-mode",
        choices=("fold0", "all_four"),
        default="fold0",
        help="Use the fold-0 rule or the donor-blocked four-fold curve.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seed != 42:
        parser.error("the frozen bridge protocol uses seed 42")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        parser.error("explicit CUDA_VISIBLE_DEVICES is required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("exactly one CUDA GPU must be visible; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    probe = torch.ones((64, 64), device=device) @ torch.ones((64, 64), device=device)
    if float(probe.sum()) <= 0:
        parser.error("CUDA probe failed")
    torch.set_num_threads(4)
    config = TrainConfig(seed=args.seed, max_epochs=args.max_epochs, patience=args.patience)
    args.output.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output / ".run.lock").open("w")
    fcntl.flock(lock_handle, fcntl.LOCK_EX)
    run_path = args.output / "run.json"
    manifest_path = args.output / "predictions.tsv"
    checkpoint_path = args.output / "checkpoint.pt"
    if args.resume and run_path.is_file() and manifest_path.is_file() and checkpoint_path.is_file():
        prior = json.loads(run_path.read_text())
        if (
            prior.get("status") == "pass"
        ):
            print(json.dumps({"status": "resume", "run": str(run_path)}, indent=2))
            return
    feature_audit = json.loads(args.feature_audit.read_text())
    expected_feature_schema = (
        "sound_vaccine_pulsar_features_1" if args.method == "pulsar"
        else "sound_vaccine_uce_features_1"
    )
    if (
        feature_audit.get('status') != 'pass'
        or feature_audit.get('schema_version') != expected_feature_schema
    ):
        raise RuntimeError("bridge feature audit is not passing")
    visits = pd.read_csv(args.source_visits, sep="\t")
    visits = visits.loc[visits["paper_task"].eq(args.task)].copy()
    train_visits = visits.loc[~visits["outer_fold"].astype(int).eq(args.fold)].copy()
    test_visits = visits.loc[visits["outer_fold"].astype(int).eq(args.fold)].copy()
    if len(train_visits) + len(test_visits) != EXPECTED_TEST_UNITS[args.task]:
        raise RuntimeError("task visit support changed")
    inner = pd.read_csv(args.inner_splits, sep="\t")
    inner = inner.loc[inner["outer_fold"].astype(int).eq(args.fold)]
    train_donors = set(train_visits["donor_id"].astype(str))
    inner_masks: list[tuple[int, np.ndarray, np.ndarray]] = []
    selected_inner_folds = [0] if args.inner_selection_mode == "fold0" else [0, 1, 2, 3]
    validation_coverage: set[str] = set()
    for inner_fold in selected_inner_folds:
        validation_donors = set(
            inner.loc[inner["inner_fold"].astype(int).eq(inner_fold), "donor_id"].astype(str)
        ) & train_donors
        fit_mask = ~train_visits["donor_id"].astype(str).isin(validation_donors).to_numpy()
        validation_mask = ~fit_mask
        if not fit_mask.any() or not validation_mask.any():
            raise RuntimeError(f"inner donor validation split {inner_fold} is empty")
        inner_masks.append((inner_fold, fit_mask, validation_mask))
        validation_coverage.update(validation_donors)
    if args.inner_selection_mode == "all_four" and validation_coverage != train_donors:
        missing = sorted(train_donors - validation_coverage)
        extra = sorted(validation_coverage - train_donors)
        raise RuntimeError(
            f"four-fold donor validation does not cover outer training donors; missing={missing}, extra={extra}"
        )
    started = time.time()
    store = FeatureStore(args.features, "pulsar" if args.method == "pulsar" else "uce")
    train_reader = Reader(args.training_latent)
    try:
        train_features = store.take(train_visits["visit_id"].astype(str).tolist())
        train_deltas, sampling_records = _delta_targets(
            train_reader,
            train_visits,
            cells=config.cells_per_training_condition,
            seed=config.seed,
        )
        if args.inner_selection_mode == "all_four":
            selected_epoch, selection_history = _select_epoch_all_folds(
                args.method,
                train_features,
                train_deltas,
                inner_masks,
                config=config,
                device=device,
                cima_checkpoint=args.cima_checkpoint,
            )
        else:
            _, fit_mask, validation_mask = inner_masks[0]
            selected_epoch, selection_history = _select_epoch(
                args.method,
                train_features,
                train_deltas,
                fit_mask,
                validation_mask,
                config=config,
                device=device,
                cima_checkpoint=args.cima_checkpoint,
            )
        model, center, scale, loaded, final_losses = _fit_final(
            args.method,
            train_features,
            train_deltas,
            selected_epoch,
            config=config,
            device=device,
            cima_checkpoint=args.cima_checkpoint,
        )
        test_features = store.take(test_visits["visit_id"].astype(str).tolist())
        predicted_deltas = _predict(
            model, test_features, center, scale,
            batch_size=config.batch_size, device=device,
        )
        checkpoint = {
            "format": SCHEMA,
            "method": args.method,
            "paper_method": METHOD_LABELS[args.method],
            "paper_task": args.task,
            "outer_fold": args.fold,
            "seed": config.seed,
            "model_config": model.config.to_dict(),
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "response_center": center,
            "response_scale": scale,
            "selected_epoch": selected_epoch,
            "inner_selection_mode": args.inner_selection_mode,
            "training_donor_ids": sorted(train_visits["donor_id"].astype(str).unique()),
            "cima_initialized_parameters": loaded,
        }
        temporary_checkpoint = checkpoint_path.with_suffix(".tmp.pt")
        torch.save(checkpoint, temporary_checkpoint)
        os.replace(temporary_checkpoint, checkpoint_path)
        # Compare single-item calls; BF16 results depend on batch shape.
        before_reload = _predict(
            model,
            test_features[:1],
            center,
            scale,
            batch_size=1,
            device=device,
        )
        reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        reload_config = VaccineBridgeConfig(**reloaded["model_config"])
        reload_model = VaccineBridgeModel(reload_config)
        reload_model.load_state_dict(reloaded["model_state_dict"], strict=True)
        reload_model.to(device)
        after_reload = _predict(
            reload_model,
            test_features[:1],
            np.asarray(reloaded["response_center"]),
            np.asarray(reloaded["response_scale"]),
            batch_size=1,
            device=device,
        )
        reload_max_abs = float(np.max(np.abs(before_reload - after_reload)))
        if reload_max_abs > 1e-6:
            raise RuntimeError(f"bridge checkpoint reload mismatch: {reload_max_abs}")
        del reload_model
        test_reader = Reader(args.test_latent)
        records: list[dict[str, object]] = []
        try:
            for index, (_, visit) in enumerate(test_visits.reset_index(drop=True).iterrows()):
                source_rows = test_reader.rows(str(visit["model_source_sample"]))
                target_rows = test_reader.rows(str(visit["model_target_sample"]))
                if not len(source_rows) or not len(target_rows):
                    raise RuntimeError(f"held-out visit is absent: {visit['visit_id']}")
                source_seed = _stable_seed(
                    "bridge_test_cells", args.task, visit["visit_id"], "source",
                    base_seed=config.seed,
                )
                target_seed = _stable_seed(
                    "bridge_test_cells", args.task, visit["visit_id"], "target",
                    base_seed=config.seed,
                )
                source, source_state, source_rows_saved = test_reader.take(
                    source_rows,
                    config.generated_cells,
                    np.random.default_rng(source_seed),
                    exact=True,
                )
                observed, observed_state, observed_rows_saved = test_reader.take(
                    target_rows,
                    config.generated_cells,
                    np.random.default_rng(target_seed),
                    exact=True,
                )
                predicted = source + predicted_deltas[index][None, :]
                identity = f"{visit['donor_id']}__{visit['season']}"
                source_path = args.output / f"source_day{int(visit['model_source_day'])}" / f"{identity}.npz"
                predicted_path = args.output / "predicted" / f"{identity}.npz"
                observed_path = args.output / "observed" / f"{identity}.npz"
                common = {
                    "donor_id": np.asarray(str(visit["donor_id"])),
                    "episode_id": np.asarray(str(visit["episode_id"])),
                    "paper_task": np.asarray(args.task),
                    "cell_scope": np.asarray("all_cells"),
                }
                _save_npz(
                    source_path,
                    embedding=source.astype(np.float32),
                    cell_state=source_state,
                    source_row_index=source_rows_saved,
                    target_day=np.asarray(int(visit["model_source_day"])),
                    role=np.asarray(f"observed_day{int(visit['model_source_day'])}"),
                    **common,
                )
                _save_npz(
                    predicted_path,
                    embedding=predicted.astype(np.float32),
                    cell_state=source_state,
                    source_row_index=source_rows_saved,
                    source_day=np.asarray(int(visit["model_source_day"])),
                    target_day=np.asarray(int(visit["model_target_day"])),
                    delta=predicted_deltas[index],
                    method=np.asarray(METHOD_LABELS[args.method]),
                    **common,
                )
                _save_npz(
                    observed_path,
                    embedding=observed.astype(np.float32),
                    cell_state=observed_state,
                    source_row_index=observed_rows_saved,
                    target_day=np.asarray(int(visit["model_target_day"])),
                    role=np.asarray("observed_future"),
                    **common,
                )
                records.append(
                    {
                        "donor_id": str(visit["donor_id"]),
                        "episode_id": str(visit["episode_id"]),
                        "season": str(visit["season"]),
                        "outer_fold": args.fold,
                        "cell_scope": "all_cells",
                        "target_day": int(visit["model_target_day"]),
                        "method": METHOD_LABELS[args.method],
                        "source_path": str(source_path),
                        "observed_path": str(observed_path),
                        "predicted_path": str(predicted_path),
                        "prediction_uses_test_future": False,
                        "task": "conditional",
                        "vaccinated": True,
                        "source_day": int(visit["model_source_day"]),
                        "paper_task": args.task,
                    }
                )
        finally:
            test_reader.close()
        manifest = pd.DataFrame(records)
        expected_fold_units = int((visits["outer_fold"].astype(int) == args.fold).sum())
        if len(manifest) != expected_fold_units:
            raise RuntimeError("held-out bridge prediction coverage is incomplete")
        temporary_manifest = manifest_path.with_suffix(".tmp.tsv")
        manifest.to_csv(temporary_manifest, sep="\t", index=False)
        os.replace(temporary_manifest, manifest_path)
    finally:
        train_reader.close()
        store.close()
    payload: dict[str, object] = {
        "status": "pass",
        "schema_version": SCHEMA,
        "comparison_role": "pretraining_encoder_transfer_control",
        "method": args.method,
        "paper_method": METHOD_LABELS[args.method],
        "paper_task": args.task,
        "outer_fold": args.fold,
        "seed": config.seed,
        "training_config": asdict(config),
        "training_donors": int(train_visits["donor_id"].nunique()),
        "training_units": len(train_visits),
        "test_donors": int(test_visits["donor_id"].nunique()),
        "test_units": len(test_visits),
        "selected_epoch": selected_epoch,
        "selection_history": selection_history,
        "final_training_losses": final_losses,
        "inner_selection_mode": args.inner_selection_mode,
        "inner_validation_fold": 0 if args.inner_selection_mode == "fold0" else None,
        "inner_validation_folds": selected_inner_folds,
        "donor_blocked": True,
        "one_all_pbmc_model": True,
        "donor_identifier_as_feature": False,
        "clinical_information_used": False,
        "history_information_used": False,
        "test_source_only_for_prediction": True,
        "test_future_used_for_training_or_adaptation": False,
        "training_target": "outer-fold 16-D target-minus-source mean state response",
        "prediction_distribution": "held-out source cells shifted by the predicted patient response",
        "feature_audit": str(args.feature_audit.resolve()),
        "source_visit_manifest": str(args.source_visits.resolve()),
        "cima_checkpoint": str(args.cima_checkpoint.resolve()) if args.cima_checkpoint else None,
        "cima_initialized_parameters": loaded,
        "training_sampling": sampling_records,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_reload_max_abs_difference": reload_max_abs,
        "manifest": str(manifest_path.resolve()),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "runner_source": str(Path(__file__).resolve()),
    }
    _atomic_json(run_path, payload)
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()

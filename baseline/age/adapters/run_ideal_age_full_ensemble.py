from __future__ import annotations
import importlib.metadata as package_metadata
import inspect
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from baseline.age.workspace import WORKSPACE_ROOT

OFFICIAL_REPO = WORKSPACE_ROOT / "models/ImmuneTwins/baselines/IDEAL-Age"

MODEL_TYPES = ("FASTAI", "NN_TORCH")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def frame(features: np.ndarray, genes: np.ndarray, donors: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        features,
        index=pd.Index(donors, name="donor_id"),
        columns=genes,
        copy=False,
    )

def assert_cuda() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal IDEAL-Age training requires a visible CUDA GPU")
    device = torch.device("cuda:0")
    left = torch.randn(256, 256, device=device, requires_grad=True)
    value = (left @ left.T).mean()
    value.backward()
    capability = torch.cuda.get_device_capability(device)
    proof = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "autogluon_tabular_version": package_metadata.version("autogluon.tabular"),
        "fastai_version": package_metadata.version("fastai"),
        "device": torch.cuda.get_device_name(device),
        "capability": list(capability),
        "forward_backward_finite": bool(torch.isfinite(value).item()),
    }
    del left, value
    torch.cuda.empty_cache()
    if not proof["forward_backward_finite"]:
        raise RuntimeError("CUDA forward/backward proof was non-finite")
    return proof

def fit_autogluon_bagged(
    train: pd.DataFrame,
    train_age: np.ndarray,
    folds: np.ndarray,
    external: pd.DataFrame,
    model_dir: Path,
    time_limit: int,
    extra_time_limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit the released neural branches on the common five folds."""

    from autogluon.tabular import TabularPredictor

    seed_all(seed)
    training = train.copy()
    training["age"] = train_age
    training["__common_fold"] = folds
    if model_dir.exists():
        predictor = TabularPredictor.load(str(model_dir))
        print("resume saved AutoGluon predictor", flush=True)
    else:
        predictor = TabularPredictor(
            label="age",
            problem_type="regression",
            path=str(model_dir),
            verbosity=2,
            eval_metric="mean_absolute_error",
            groups="__common_fold",
        )
        kwargs: dict[str, object] = {
            "time_limit": time_limit,
            "included_model_types": list(MODEL_TYPES),
            "fit_weighted_ensemble": True,
            "hyperparameters": {
                "FASTAI": {"ag_args_fit": {"num_gpus": 1}},
                "NN_TORCH": {"ag_args_fit": {"num_gpus": 1}},
            },
        }
        # AutoGluon 1.4 exposes a global resource argument.  Keep this explicit
        # so both neural pseudo-bulk models are scheduled on CUDA.
        if "num_gpus" not in inspect.signature(predictor.fit).parameters:
            raise RuntimeError("AutoGluon build lacks the required num_gpus fit contract")
        kwargs["num_gpus"] = 1
        kwargs["num_cpus"] = 8
        predictor.fit(training, **kwargs)
    model_names = predictor.model_names()
    fastai = [name for name in model_names if "FastAI" in name]
    nn_torch = [name for name in model_names if "NeuralNetTorch" in name]
    if fastai and not nn_torch:
        print("adding the timed-out NN_TORCH branch with fit_extra", flush=True)
        predictor.fit_extra(
            hyperparameters={
                "NN_TORCH": {"ag_args_fit": {"num_gpus": 1}}
            },
            time_limit=extra_time_limit,
            fit_weighted_ensemble=True,
            num_cpus=8,
            num_gpus=1,
            fit_strategy="sequential",
        )
        model_names = predictor.model_names()
        fastai = [name for name in model_names if "FastAI" in name]
        nn_torch = [name for name in model_names if "NeuralNetTorch" in name]
    if not fastai or not nn_torch:
        raise RuntimeError(
            "official AutoGluon fit did not produce both FASTAI and NN_TORCH"
        )
    oof_prediction = predictor.predict_oof().to_numpy(dtype=np.float64)
    external_prediction = predictor.predict(external).to_numpy(dtype=np.float64)
    leaderboard = predictor.leaderboard(silent=True)
    leaderboard.to_csv(model_dir.parent / f"{model_dir.name}_leaderboard.tsv", sep="\t", index=False)
    details = {
        "model_names": model_names,
        "best_model": predictor.model_best,
        "fastai_models": fastai,
        "nn_torch_models": nn_torch,
        "time_limit_seconds": time_limit,
        "fit_extra_time_limit_seconds": extra_time_limit,
        "num_gpus": 1,
        "num_cpus": 8,
        "bagging_folds": 5,
        "bagging_fold_source": "cima_random_donor.tsv repeat=0",
        "predictor_path": str(model_dir.resolve()),
    }
    return oof_prediction, external_prediction, details

def greedy_ensemble_weights(
    predictions: np.ndarray, ages: np.ndarray, ensemble_size: int
) -> np.ndarray:
    """Caruana greedy ensemble selection used by AutoGluon weighted ensembles."""

    if predictions.ndim != 2 or predictions.shape[0] != len(ages):
        raise ValueError("ensemble prediction matrix has the wrong shape")
    if ensemble_size <= 0 or not np.isfinite(predictions).all():
        raise ValueError("invalid ensemble inputs")
    counts = np.zeros(predictions.shape[1], dtype=np.int64)
    accumulated = np.zeros(len(ages), dtype=np.float64)
    for iteration in range(ensemble_size):
        candidates = (accumulated[:, None] + predictions) / (iteration + 1)
        losses = np.mean(np.abs(candidates - ages[:, None]), axis=0)
        selected = int(np.argmin(losses))
        counts[selected] += 1
        accumulated += predictions[:, selected]
    weights = counts.astype(np.float64) / ensemble_size
    if not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("greedy ensemble weights do not sum to one")
    return weights

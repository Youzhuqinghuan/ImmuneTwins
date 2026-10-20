from __future__ import annotations
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from model.age import AgeModelConfig, ImmuneTwinAgeModel

CHECKPOINT_FORMAT = "immunetwin_age_v1"

@dataclass(frozen=True)
class TargetScaler:
    mean: float
    scale: float
    n_fit_donors: int

    @classmethod
    def fit(cls, ages: np.ndarray) -> "TargetScaler":
        values = np.asarray(ages, dtype=np.float64)
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("target scaler needs at least two finite donor ages")
        scale = float(values.std(ddof=0))
        if scale <= 0:
            raise ValueError("target ages have zero variance")
        return cls(float(values.mean()), scale, len(values))

    def transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale

    def inverse_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale + self.mean

    def inverse_numpy(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def require_cuda(device: str) -> torch.device:
    requested = torch.device(device)
    if requested.type != "cuda":
        raise RuntimeError("ImmuneTwin age training requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU training fallback is forbidden")
    probe = torch.ones((8, 8), device=requested)
    observed = probe @ probe
    if observed.device.type != "cuda" or not torch.isfinite(observed).all():
        raise RuntimeError(f"CUDA tensor-operation check failed on {requested}")
    return requested

def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

def save_checkpoint(
    path: Path,
    model: ImmuneTwinAgeModel,
    scaler: TargetScaler,
    *,
    seed: int,
    epochs: int,
    outer_fold: int | None,
    training_donor_ids: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "target_scaler": scaler.to_dict(),
        "seed": int(seed),
        "epochs": int(epochs),
        "outer_fold": outer_fold,
        "training_donor_ids": sorted(map(str, training_donor_ids)),
        "uce_encoder_frozen_upstream": True,
        "target_scaler_fit_scope": "training_donors_only",
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)

def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[ImmuneTwinAgeModel, TargetScaler, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported age checkpoint format: {payload.get('format')}")
    model = ImmuneTwinAgeModel(AgeModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    scaler = TargetScaler(**payload["target_scaler"])
    return model, scaler, payload

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

RESPONSE = {
    "Kydar02": 1,
    "Kydar07": 1,
    "Kydar08": 1,
    "Kydar10": 1,
    "Kydar11": 1,
    "Kydar12": 1,
    "Kydar15": 1,
    "Kydar19": 1,
    "Kydar20": 1,
    "Kydar22": 1,
    "Kydar23": 1,
    "Kydar25": 1,
    "Kydar26": 1,
    "Kydar30": 1,
    "Kydar32": 1,
    "Kydar33": 1,
    "Kydar34": 1,
    "Kydar36": 1,
    "Kydar38": 1,
    "Kydar39": 1,
    "Kydar40": 1,
    "Kydar04": 0,
    "Kydar14": 0,
    "Kydar17": 0,
    "Kydar21": 0,
    "Kydar24": 0,
    "Kydar27": 0,
    "Kydar31": 0,
}

def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)

def _require_cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training is required; CPU fallback is prohibited")
    device = torch.device(device_name)
    probe = torch.arange(8, device=device, dtype=torch.float32)
    if float(probe.sum()) != 28.0:
        raise RuntimeError("CUDA tensor-operation probe failed")
    return device

def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values],
        dtype=str,
    )

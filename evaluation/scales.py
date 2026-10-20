from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml
from paths import expand_paths

def _scales(
    assets_path: Path,
) -> dict[int, tuple[np.ndarray, np.ndarray, float, Path]]:
    assets = expand_paths(yaml.safe_load(assets_path.read_text()))
    result: dict[int, tuple[np.ndarray, np.ndarray, float, Path]] = {}
    for fold in range(5):
        mapping_path = Path(str(assets["sound"]["metric_maps"]).format(fold=fold))
        mapping = json.loads(mapping_path.read_text())
        path = Path(mapping["all"])
        with np.load(path, allow_pickle=False) as payload:
            if "training" not in str(payload["fit_role"]).casefold():
                raise RuntimeError(f"metric scale is not outer-training only: {path}")
            result[fold] = (
                payload["center"].astype(np.float32),
                payload["scale"].astype(np.float32),
                float(payload["mmd_bandwidth"]),
                path,
            )
    return result

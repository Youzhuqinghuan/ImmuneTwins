from __future__ import annotations
from dataclasses import dataclass
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation.controls import TASKS, _stable_seed

@dataclass(frozen=True)
class ScoreConfig:
    evaluation_draws: int = 20
    cells_per_draw: int = 256
    technical_seed: int = 42
    epsilon: float = 0.05
    max_iterations: int = 200
    tolerance: float = 1e-6
    bootstrap_draws: int = 10_000
    bootstrap_seed: int = 20260829

def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)

def _resolve_artifact(
    *, manifest_path: Path, row: pd.Series, field: str, allowed_root: Path | tuple[Path, ...]
) -> Path:
    role_directory = {
        "source_path": f"source_day{int(row['source_day'])}",
        "predicted_path": "predicted",
        "observed_path": "observed",
    }[field]
    raw = Path(str(row[field]))
    local = manifest_path.parent / role_directory / raw.name
    candidates = [local]
    if raw.is_absolute():
        candidates.append(raw)
    roots = (allowed_root,) if isinstance(allowed_root, Path) else allowed_root
    roots = tuple(path.resolve() for path in roots)
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            if any(resolved.is_relative_to(root) for root in roots):
                return resolved
        except (FileNotFoundError, ValueError):
            continue
    raise RuntimeError(
        f"cannot resolve {field} under the configured asset root for {manifest_path}: {raw}"
    )

def _episode_metadata(plan_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for task_id, task in TASKS.items():
        path = plan_root / f"{task_id}_episodes.tsv"
        frame = pd.read_csv(path, sep="\t")
        frame = frame.loc[
            pd.to_numeric(frame["target_day"], errors="raise")
            .astype(int)
            .eq(int(task["target_day"])),
            ["donor_id", "season", "outer_fold", "age_source"],
        ].copy()
        frame["paper_task"] = task_id
        frames.append(frame)
    metadata = pd.concat(frames, ignore_index=True)
    if metadata.duplicated(["paper_task", "donor_id", "season"]).any():
        raise RuntimeError("episode metadata duplicates a prediction unit")
    age = pd.to_numeric(metadata["age_source"], errors="raise")
    metadata["age_stratum"] = np.where(age < 40, "young", np.where(age >= 50, "older", "excluded"))
    if metadata["age_stratum"].eq("excluded").any():
        raise RuntimeError("Sound evaluation contains age 40-49 outside the frozen strata")
    return metadata

def _draw_indices(length: int, config: ScoreConfig, *identity: object, role: str) -> np.ndarray:
    if length < config.cells_per_draw:
        raise RuntimeError(
            f"saved distribution has {length} cells, fewer than draw size {config.cells_per_draw}"
        )
    rng = np.random.default_rng(
        _stable_seed("evaluation", *identity, role, base_seed=config.technical_seed)
    )
    return np.stack(
        [
            rng.choice(length, size=config.cells_per_draw, replace=False)
            for _ in range(config.evaluation_draws)
        ]
    )

def _bootstrap_mean(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    estimates = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1000):
        stop = min(start + 1000, draws)
        index = rng.integers(0, n, size=(stop - start, n))
        estimates[start:stop] = values[index].mean(axis=1)
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))




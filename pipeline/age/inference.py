from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from pipeline.age.checkpoint import TargetScaler
from data.age import ImmunobiologyUceShardStore
from model.age import ImmuneTwinAgeModel, NonlinearAttentionPool

@dataclass
class _MeanAccumulator:
    total: np.ndarray
    count: int = 0

    def update(self, values: np.ndarray) -> None:
        self.total += values.sum(axis=0, dtype=np.float64)
        self.count += len(values)

    def state(self) -> np.ndarray:
        return self.total / self.count

@dataclass
class _AttentionAccumulator:
    weighted_total: np.ndarray
    denominator: float = 0.0
    maximum: float = -np.inf
    count: int = 0

    def update(self, values: np.ndarray, logits: np.ndarray) -> None:
        local_maximum = float(np.max(logits))
        new_maximum = max(self.maximum, local_maximum)
        old_factor = 0.0 if self.denominator == 0 else np.exp(self.maximum - new_maximum)
        local_weights = np.exp(logits.astype(np.float64) - new_maximum)
        self.weighted_total *= old_factor
        self.weighted_total += np.sum(
            local_weights[:, None] * values.astype(np.float64), axis=0
        )
        self.denominator = self.denominator * old_factor + float(local_weights.sum())
        self.maximum = new_maximum
        self.count += len(values)

    def state(self) -> np.ndarray:
        return self.weighted_total / self.denominator

def _project_shard(
    model: ImmuneTwinAgeModel,
    embeddings: np.ndarray,
    device: str | torch.device,
    cell_batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    projected_chunks: list[np.ndarray] = []
    logit_chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(embeddings), cell_batch_size):
            cells = torch.as_tensor(
                embeddings[start : start + cell_batch_size],
                dtype=torch.float32,
                device=device,
            )
            projected = model.project_cells(cells)
            projected_chunks.append(projected.float().cpu().numpy())
            if model.config.pooling == "attention":
                if not isinstance(model.pool, NonlinearAttentionPool):
                    raise TypeError("attention model has the wrong pooling module")
                logit_chunks.append(model.pool.logits(projected).float().cpu().numpy())
    projected_array = np.concatenate(projected_chunks, axis=0)
    logits_array = np.concatenate(logit_chunks) if logit_chunks else None
    return projected_array, logits_array

@torch.inference_mode()
def predict_external_shards(
    model: ImmuneTwinAgeModel,
    scaler: TargetScaler,
    store: ImmunobiologyUceShardStore,
    device: str | torch.device,
    *,
    cell_batch_size: int = 4096,
    expected_donors: int | None = 234,
    expected_cells_per_donor: int | None = 1500,
) -> pd.DataFrame:
    """Aggregate each donor independently while reading every shard once."""
    accumulators: dict[str, _MeanAccumulator | _AttentionAccumulator] = {}
    for _, embeddings, donor_ids in store.iter_shards():
        projected, logits = _project_shard(
            model, embeddings, device=device, cell_batch_size=cell_batch_size
        )
        for donor_id in pd.unique(donor_ids).astype(str):
            selected = donor_ids == donor_id
            if donor_id not in accumulators:
                zeros = np.zeros(model.config.state_dim, dtype=np.float64)
                if model.config.pooling == "attention":
                    accumulators[donor_id] = _AttentionAccumulator(zeros)
                else:
                    accumulators[donor_id] = _MeanAccumulator(zeros)
            accumulator = accumulators[donor_id]
            if isinstance(accumulator, _AttentionAccumulator):
                if logits is None:
                    raise RuntimeError("attention logits are absent")
                accumulator.update(projected[selected], logits[selected])
            else:
                accumulator.update(projected[selected])
    donor_ids = sorted(accumulators)
    if expected_donors is not None and len(donor_ids) != expected_donors:
        raise ValueError(f"external set has {len(donor_ids)} donors; expected {expected_donors}")
    wrong_counts = {
        donor_id: accumulators[donor_id].count
        for donor_id in donor_ids
        if expected_cells_per_donor is not None
        and accumulators[donor_id].count != expected_cells_per_donor
    }
    if wrong_counts:
        raise ValueError(f"external donor cell-count contract failed: {wrong_counts}")
    states = np.stack([accumulators[donor_id].state() for donor_id in donor_ids]).astype(
        np.float32
    )
    predictions: list[np.ndarray] = []
    for start in range(0, len(states), 256):
        state = torch.as_tensor(states[start : start + 256], device=device)
        scaled = model.predict_from_state(state)
        predictions.append(scaler.inverse_tensor(scaled).float().cpu().numpy())
    return pd.DataFrame(
        {
            "donor_id": donor_ids,
            "predicted_age": np.concatenate(predictions).astype(np.float64),
        }
    )

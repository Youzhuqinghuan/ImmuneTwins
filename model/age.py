from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class AgeModelConfig:
    input_dim: int = 1280
    state_dim: int = 128
    attention_dim: int = 64
    age_hidden_dim: int = 32
    pooling: str = "attention"

    def __post_init__(self) -> None:
        if self.pooling not in {"attention", "mean"}:
            raise ValueError("pooling must be 'attention' or 'mean'")
        for name in ("input_dim", "state_dim", "attention_dim", "age_hidden_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

def _validate_cells_and_mask(
    cells: torch.Tensor, mask: torch.Tensor | None, input_dim: int
) -> torch.Tensor:
    if cells.ndim != 3 or cells.shape[-1] != input_dim:
        raise ValueError(
            f"cells must have shape [donor, cell, {input_dim}], got {tuple(cells.shape)}"
        )
    if cells.shape[1] < 1:
        raise ValueError("every batch needs at least one cell position")
    if mask is None:
        mask = torch.ones(cells.shape[:2], dtype=torch.bool, device=cells.device)
    elif mask.shape != cells.shape[:2]:
        raise ValueError(
            f"mask must have shape {tuple(cells.shape[:2])}, got {tuple(mask.shape)}"
        )
    else:
        mask = mask.to(device=cells.device, dtype=torch.bool)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every donor must contain at least one unmasked cell")
    return mask

class NonlinearAttentionPool(nn.Module):
    """A single tanh-score softmax attention pooling operation."""

    def __init__(self, state_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(state_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def logits(self, projected_cells: torch.Tensor) -> torch.Tensor:
        return self.score(projected_cells).squeeze(-1)

    def forward(
        self, projected_cells: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(projected_cells)
        logits = logits.masked_fill(~mask, -torch.inf)
        weights = torch.softmax(logits, dim=1)
        state = torch.sum(weights.unsqueeze(-1) * projected_cells, dim=1)
        return state, weights

class MaskedMeanPool(nn.Module):
    """Mean-pooling control with the same projection and age head."""

    def forward(
        self, projected_cells: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask.to(projected_cells.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True)
        state = torch.sum(weights.unsqueeze(-1) * projected_cells, dim=1)
        return state, weights

class ImmuneTwinAgeModel(nn.Module):
    """Patient encoder over frozen UCE embeddings with attention or mean pooling.

    Predictions are in the fold-scaled target coordinate; the inverse scaling
    is stored in each fold checkpoint.
    """

    def __init__(self, config: AgeModelConfig | None = None, **kwargs: object) -> None:
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("pass either config or keyword fields, not both")
        self.config = config if config is not None else AgeModelConfig(**kwargs)
        self.cell_projection = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.state_dim),
            nn.GELU(),
        )
        # Construct shared modules before the arm-specific pool so attention and
        # mean runs initialized with the same seed receive byte-identical shared
        # projection/head parameters.
        self.age_head = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.age_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.age_hidden_dim, 1),
        )
        if self.config.pooling == "attention":
            self.pool: nn.Module = NonlinearAttentionPool(
                self.config.state_dim, self.config.attention_dim
            )
        else:
            self.pool = MaskedMeanPool()

    def project_cells(self, cells: torch.Tensor) -> torch.Tensor:
        if cells.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"last cell dimension must be {self.config.input_dim}, got {cells.shape[-1]}"
            )
        return self.cell_projection(cells)

    def encode(
        self, cells: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = _validate_cells_and_mask(cells, mask, self.config.input_dim)
        projected = self.project_cells(cells)
        return self.pool(projected, mask)

    def predict_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"state must have shape [donor, {self.config.state_dim}]"
            )
        return self.age_head(state).squeeze(-1)

    def forward(
        self, cells: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        state, _ = self.encode(cells, mask)
        return self.predict_from_state(state)

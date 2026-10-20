from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class VaccineBridgeConfig:
    input_kind: str = "uce_cells"
    input_dim: int = 1280
    state_dim: int = 128
    response_hidden_dim: int = 64
    response_dim: int = 16

    def __post_init__(self) -> None:
        if self.input_kind not in {"uce_cells", "pulsar"}:
            raise ValueError("input_kind must be uce_cells or pulsar")
        for name in ("input_dim", "state_dim", "response_hidden_dim", "response_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

class VaccineBridgeModel(nn.Module):
    """Encode a current patient state and predict its 16-D response vector.

    UCE inputs retain the age-stage cell projection followed by arithmetic mean
    patient pooling. PULSAR inputs are already patient-level and therefore use
    one matched projection into the same 128-D state. Both variants share the
    same response-head shape.
    """

    def __init__(self, config: VaccineBridgeConfig) -> None:
        super().__init__()
        self.config = config
        self.state_projection = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.state_dim),
            nn.GELU(),
        )
        self.response_head = nn.Sequential(
            nn.Linear(config.state_dim, config.response_hidden_dim),
            nn.GELU(),
            nn.Linear(config.response_hidden_dim, config.response_dim),
        )

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if self.config.input_kind == "uce_cells":
            if values.ndim != 3 or values.shape[-1] != self.config.input_dim:
                raise ValueError("UCE values must be batch x cells x input_dim")
            return self.state_projection(values).mean(dim=1)
        if values.ndim != 2 or values.shape[-1] != self.config.input_dim:
            raise ValueError("PULSAR values must be batch x input_dim")
        return self.state_projection(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.response_head(self.encode(values))

def load_cima_mean_projection(
    model: VaccineBridgeModel, checkpoint: dict[str, object]
) -> list[str]:
    """Initialize the cell projection from the CIMA mean-pooling model."""

    if model.config.input_kind != "uce_cells":
        raise ValueError("CIMA cell projection can initialize only a UCE-cell model")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict) or config.get("pooling") != "mean":
        raise ValueError("checkpoint is not the CIMA arithmetic-mean model")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("CIMA checkpoint lacks model_state_dict")
    mapping = {
        "state_projection.0.weight": "cell_projection.0.weight",
        "state_projection.0.bias": "cell_projection.0.bias",
        "state_projection.1.weight": "cell_projection.1.weight",
        "state_projection.1.bias": "cell_projection.1.bias",
    }
    current = model.state_dict()
    loaded: list[str] = []
    for destination, source in mapping.items():
        if source not in state or current[destination].shape != state[source].shape:
            raise ValueError(f"CIMA projection parameter mismatch: {source}")
        current[destination] = state[source].detach().clone()
        loaded.append(destination)
    model.load_state_dict(current, strict=True)
    return loaded

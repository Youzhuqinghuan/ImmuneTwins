from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from typing import Literal
import torch
from torch import nn
from model.age import NonlinearAttentionPool
SUPPORTED_RANDOM_EFFECT_DIMS = (1, 2, 4, 8)

TwinArm = Literal["current", "w", "gene", "full"]

VisitPooling = Literal["attention_only", "gated_attention_mean"]

GenePriorMode = Literal["all_pbmc", "lineage"]

HistoryIdentifier = Literal["legacy_transition", "residual_posterior"]

FiberContext = Literal["local", "pooled", "adaptive"]

FiberRetraction = Literal["every_block", "final_only", "adaptive_final", "none"]

CurrentOperator = Literal[
    "cell_film",
    "program_affine",
    "program_query_affine",
    "neural_operator",
    "set_affine",
    "moment_fiber",
]

@dataclass(frozen=True)
class VaccineImmuneTwinConfig:
    uce_dim: int = 1280
    visit_dim: int = 128
    attention_dim: int = 64
    latent_dim: int = 16
    phase_count: int = 3
    phase_dim: int = 16
    cell_hidden_dim: int = 128
    transition_hidden_dim: int = 128
    clinical_dim: int = 0
    w_dim: int = 8
    gene_embedding_dim: int = 5120
    gene_state_dim: int = 64
    gene_top_k: int = 64
    gene_temperature: float = 1.0
    operator_scale: float = 0.10
    visit_pooling: VisitPooling = "attention_only"
    history_response_dim: int = 0
    visit_gate_initial_logit: float = 0.0
    gene_prior_mode: GenePriorMode = "all_pbmc"
    history_adapter_init_std: float = 0.01
    history_identifier: HistoryIdentifier = "legacy_transition"
    residual_random_effect_dim: int = 4
    residual_noise_floor: float = 1e-5
    residual_initial_noise_variance: float = 1.0
    residual_loading_init_std: float = 0.05
    residual_gene_embedding_dim: int = 16
    residual_lineage_count: int = 5
    residual_gene_top_k: int = 16
    residual_gene_temperature: float = 1.0
    residual_gene_max_precision_per_expert: float = 1.0
    residual_gene_initial_precision_logit: float = -3.0
    current_operator: CurrentOperator = "cell_film"
    mean_program_rank: int = 7
    covariance_program_rank: int = 11
    source_geometry_dim: int = 64
    # Patient-conditioned residual operator with eight shared state slots.
    operator_slot_count: int = 8
    operator_depth: int = 4
    operator_rank: int = 8
    # Shape-refinement options: the adaptive model learns local/global context
    # and update gates; the fixed variants are ablations.
    fiber_context: FiberContext = "local"
    fiber_retraction: FiberRetraction = "every_block"

    def __post_init__(self) -> None:
        if self.visit_pooling not in {"attention_only", "gated_attention_mean"}:
            raise ValueError(
                "visit_pooling must be attention_only or gated_attention_mean"
            )
        if self.gene_prior_mode not in {"all_pbmc", "lineage"}:
            raise ValueError("gene_prior_mode must be all_pbmc or lineage")
        if self.history_identifier not in {
            "legacy_transition",
            "residual_posterior",
        }:
            raise ValueError(
                "history_identifier must be legacy_transition or residual_posterior"
            )
        if self.current_operator not in {
            "cell_film",
            "program_affine",
            "program_query_affine",
            "neural_operator",
            "set_affine",
            "moment_fiber",
        }:
            raise ValueError(
                "current_operator must be cell_film, program_affine, or "
                "program_query_affine, neural_operator, set_affine, or moment_fiber"
            )
        if self.current_operator in {
            "program_affine",
            "program_query_affine",
            "set_affine",
            "moment_fiber",
        }:
            if self.visit_pooling != "attention_only":
                raise ValueError(
                    "program transport requires the exact CIMA attention-only encoder"
                )
            if self.latent_dim != 16:
                raise ValueError("program transport requires the audited scVI-16 space")
        if self.history_identifier == "residual_posterior":
            if self.latent_dim != 16 or self.history_response_dim != 16:
                raise ValueError(
                    "residual_posterior requires latent_dim=16 and "
                    "history_response_dim=16"
                )
            if self.residual_random_effect_dim not in SUPPORTED_RANDOM_EFFECT_DIMS:
                raise ValueError(
                    "residual_random_effect_dim must be one of "
                    f"{SUPPORTED_RANDOM_EFFECT_DIMS}"
                )
        if (
            self.history_identifier == "legacy_transition"
            and self.gene_prior_mode == "lineage"
            and (
                self.gene_embedding_dim != 5120 or self.latent_dim != 16
            )
        ):
            raise ValueError(
                "lineage gene prior requires 5120-D UCE gene embeddings and "
                "the 16-D response space"
            )
        integer_fields = (
            "uce_dim",
            "visit_dim",
            "attention_dim",
            "latent_dim",
            "phase_count",
            "phase_dim",
            "cell_hidden_dim",
            "transition_hidden_dim",
            "w_dim",
            "gene_embedding_dim",
            "gene_state_dim",
            "gene_top_k",
            "residual_random_effect_dim",
            "residual_gene_embedding_dim",
            "residual_lineage_count",
            "residual_gene_top_k",
            "mean_program_rank",
            "covariance_program_rank",
            "source_geometry_dim",
            "operator_slot_count",
            "operator_depth",
            "operator_rank",
        )
        for name in integer_fields:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.clinical_dim < 0:
            raise ValueError("clinical_dim cannot be negative")
        if self.current_operator == "neural_operator":
            if self.operator_slot_count != 8:
                raise ValueError(
                    "neural_operator requires exactly eight shared patient slots"
                )
            if not 4 <= self.operator_depth <= 6:
                raise ValueError("neural_operator depth must be between four and six")
            if self.operator_rank > self.visit_dim:
                raise ValueError(
                    "neural_operator low-rank width cannot exceed visit_dim"
                )
        if self.current_operator in {"set_affine", "moment_fiber"}:
            if self.operator_slot_count != 8:
                raise ValueError(
                    f"{self.current_operator} requires exactly eight patient slots"
                )
        if self.current_operator == "moment_fiber" and not 2 <= self.operator_depth <= 6:
            raise ValueError("moment_fiber depth must be between two and six")
        if self.fiber_context not in {"local", "pooled", "adaptive"}:
            raise ValueError("fiber_context must be local, pooled, or adaptive")
        if self.fiber_retraction not in {
            "every_block",
            "final_only",
            "adaptive_final",
            "none",
        }:
            raise ValueError(
                "fiber_retraction must be every_block, final_only, "
                "adaptive_final, or none"
            )
        if self.history_response_dim < 0:
            raise ValueError("history_response_dim cannot be negative")
        if not math.isfinite(self.visit_gate_initial_logit):
            raise ValueError("visit_gate_initial_logit must be finite")
        if (
            not math.isfinite(self.history_adapter_init_std)
            or self.history_adapter_init_std <= 0
        ):
            raise ValueError("history_adapter_init_std must be finite and positive")
        if self.gene_temperature <= 0:
            raise ValueError("gene_temperature must be positive")
        for name in (
            "residual_noise_floor",
            "residual_initial_noise_variance",
            "residual_loading_init_std",
            "residual_gene_temperature",
            "residual_gene_max_precision_per_expert",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.residual_gene_initial_precision_logit):
            raise ValueError(
                "residual_gene_initial_precision_logit must be finite"
            )
        if self.residual_initial_noise_variance <= self.residual_noise_floor:
            raise ValueError(
                "residual_initial_noise_variance must exceed residual_noise_floor"
            )
        if not 0 < self.operator_scale <= 1:
            raise ValueError("operator_scale must be in (0, 1]")

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)

def _cell_mask(
    cells: torch.Tensor, mask: torch.Tensor | None, expected_dim: int
) -> torch.Tensor:
    if cells.ndim != 3 or cells.shape[-1] != expected_dim:
        raise ValueError(
            f"cells must have shape [batch, cells, {expected_dim}], "
            f"got {tuple(cells.shape)}"
        )
    if cells.shape[1] < 1:
        raise ValueError("each visit needs at least one cell")
    if mask is None:
        return torch.ones(cells.shape[:2], dtype=torch.bool, device=cells.device)
    if mask.shape != cells.shape[:2]:
        raise ValueError(
            f"cell mask must have shape {tuple(cells.shape[:2])}, "
            f"got {tuple(mask.shape)}"
        )
    result = mask.to(device=cells.device, dtype=torch.bool)
    if not torch.all(result.any(dim=1)):
        raise ValueError("each visit needs at least one unmasked cell")
    return result

class PretrainedVisitEncoder(nn.Module):
    """CIMA projection with optional attention/mean mixing.

    ``attention_only`` is the CIMA pooling path. ``gated_attention_mean`` mixes
    attention and masked mean states of the same projected cells with one
    learned scalar per channel.
    """

    def __init__(self, config: VaccineImmuneTwinConfig) -> None:
        super().__init__()
        self.config = config
        self.cell_projection = nn.Sequential(
            nn.LayerNorm(config.uce_dim),
            nn.Linear(config.uce_dim, config.visit_dim),
            nn.GELU(),
        )
        self.pool = NonlinearAttentionPool(config.visit_dim, config.attention_dim)
        # Initial gate logit; the gate is not part of the CIMA checkpoint.
        self.channel_gate_logits = nn.Parameter(
            torch.full((config.visit_dim,), float(config.visit_gate_initial_logit))
        )

    def forward(
        self, cells: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = _cell_mask(cells, mask, self.config.uce_dim)
        projected = self.cell_projection(cells)
        attention_state, attention_weights = self.pool(projected, mask)
        if self.config.visit_pooling == "attention_only":
            return attention_state, attention_weights
        mean_weights = mask.to(projected.dtype)
        mean_weights = mean_weights / mean_weights.sum(dim=1, keepdim=True)
        mean_state = torch.sum(mean_weights.unsqueeze(-1) * projected, dim=1)
        gate = torch.sigmoid(self.channel_gate_logits).to(projected.dtype)
        state = gate * attention_state + (1.0 - gate) * mean_state
        return state, attention_weights

def load_pretrained_visit_encoder(
    encoder: PretrainedVisitEncoder, checkpoint: dict[str, object]
) -> list[str]:
    """Load only the projection and attention pool from a CIMA age checkpoint."""

    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict) or model_config.get("pooling") != "attention":
        raise ValueError("checkpoint is not the CIMA attention-pooling model")
    expected = {
        "input_dim": encoder.config.uce_dim,
        "state_dim": encoder.config.visit_dim,
        "attention_dim": encoder.config.attention_dim,
    }
    for name, value in expected.items():
        if int(model_config.get(name, -1)) != value:
            raise ValueError(f"CIMA checkpoint {name} differs from vaccine model")
    source = checkpoint.get("model_state_dict")
    if not isinstance(source, dict):
        raise ValueError("CIMA checkpoint lacks model_state_dict")
    destination = encoder.state_dict()
    loaded: list[str] = []
    # Load only the CIMA projection and pooling parameters; the channel gate
    # keeps its own initialization.
    cima_names = [
        name
        for name in destination
        if name.startswith("cell_projection.") or name.startswith("pool.")
    ]
    for name in cima_names:
        if name not in source or destination[name].shape != source[name].shape:
            raise ValueError(f"CIMA checkpoint parameter mismatch: {name}")
        destination[name] = source[name].detach().clone()
        loaded.append(name)
    encoder.load_state_dict(destination, strict=True)
    return loaded

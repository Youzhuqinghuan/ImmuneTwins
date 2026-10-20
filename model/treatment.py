from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import torch
from torch import nn
from pipeline.age.checkpoint import CHECKPOINT_FORMAT
from model.digital_twin import MetaAdapterConfig, MetaAdapterDigitalTwin
from model.encoder import VaccineImmuneTwinConfig, load_pretrained_visit_encoder
from model.projected_set import MomentFiberCurrentModel
from model.patient import SetAffineCurrentModel


@dataclass(frozen=True)
class SharedVaccineLogicConfig:
    """Treatment-task dimensions for the vaccine model classes.

    The treatment model reuses the CIMA -> SetTransport -> ProjectedSet ->
    MetaAdapter classes and the vaccine model dimensions.
    """

    latent_dim: int = 16
    transition_hidden_dim: int = 128
    source_geometry_dim: int = 64
    mean_program_rank: int = 7
    covariance_program_rank: int = 11
    operator_depth: int = 4
    adapter_context_dim: int = 64
    adapter_visit_slots: int = 2
    adapter_context_layers: int = 1
    adapter_context_heads: int = 4
    adapter_rank: int = 8
    adapter_scale: float = 0.5
    use_recovery_basis: bool = False
    recovery_same_phase: bool = False
    recovery_coefficient_initial: float = 0.54
    use_visit_tokens: bool = False
    standardize_response_loss: bool = True
    meta_response_mean_loss_weight: float = 2.0
    meta_response_direction_loss_weight: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "latent_dim",
            "transition_hidden_dim",
            "source_geometry_dim",
            "mean_program_rank",
            "covariance_program_rank",
            "operator_depth",
            "adapter_context_dim",
            "adapter_visit_slots",
            "adapter_context_layers",
            "adapter_context_heads",
            "adapter_rank",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.latent_dim != 16:
            raise ValueError("the shared ProjectedSet contract requires latent_dim=16")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    def backbone_config(self) -> VaccineImmuneTwinConfig:
        return VaccineImmuneTwinConfig(
            latent_dim=self.latent_dim,
            transition_hidden_dim=self.transition_hidden_dim,
            source_geometry_dim=self.source_geometry_dim,
            mean_program_rank=self.mean_program_rank,
            covariance_program_rank=self.covariance_program_rank,
            operator_depth=self.operator_depth,
            current_operator="moment_fiber",
            fiber_context="adaptive",
            fiber_retraction="adaptive_final",
        )

    def adapter_config(self) -> MetaAdapterConfig:
        return MetaAdapterConfig(
            context_dim=self.adapter_context_dim,
            visit_slots=self.adapter_visit_slots,
            context_layers=self.adapter_context_layers,
            context_heads=self.adapter_context_heads,
            adapter_rank=self.adapter_rank,
            adapter_scale=self.adapter_scale,
            clinical_value_dim=0,
            use_gene_attention=False,
            use_clinical_history=False,
            use_visit_tokens=self.use_visit_tokens,
            use_history_gate=False,
            use_recovery_basis=self.use_recovery_basis,
            recovery_same_phase=self.recovery_same_phase,
            recovery_coefficient_initial=self.recovery_coefficient_initial,
            response_mean_loss_weight=self.meta_response_mean_loss_weight,
            response_direction_loss_weight=self.meta_response_direction_loss_weight,
            gene_dropout=0.0,
        )

def load_cima_into_shared_backbone(
    model: SetAffineCurrentModel | MomentFiberCurrentModel,
    checkpoint_path: str | Path,

) -> dict[str, Any]:
    """Load and freeze the same CIMA visit encoder used by the vaccine stage."""

    path = Path(checkpoint_path).resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not an ImmuneTwin age-v1 checkpoint")
    loaded = load_pretrained_visit_encoder(model.visit_encoder, payload)
    for parameter in model.visit_encoder.parameters():
        parameter.requires_grad_(False)
    model.visit_encoder.eval()
    return {
        "status": "PASS",
        "checkpoint": str(path),
        "loaded_parameters": loaded,
        "excluded_age_head": True,
        "frozen": True,
    }

def build_shared_set_transport(
    config: SharedVaccineLogicConfig,
    checkpoint_path: str | Path,

) -> tuple[SetAffineCurrentModel, dict[str, Any]]:
    """Build the task current-state parent with the vaccine SetTransport class."""

    requested = config.backbone_config()
    payload = requested.to_dict()
    payload["current_operator"] = "set_affine"
    model = SetAffineCurrentModel(VaccineImmuneTwinConfig(**payload))
    audit = load_cima_into_shared_backbone(
        model, checkpoint_path
    )
    return model, audit

def build_shared_projected_set(
    config: SharedVaccineLogicConfig,
    set_transport_state: dict[str, torch.Tensor],
) -> MomentFiberCurrentModel:
    """Freeze SetTransport and train only the shape-refinement operator."""

    model = MomentFiberCurrentModel(config.backbone_config())
    model.load_frozen_set_transport_state(set_transport_state)
    return model

def build_shared_meta_adapter(
    config: SharedVaccineLogicConfig,
    projected_set: MomentFiberCurrentModel,
) -> MetaAdapterDigitalTwin:
    """Freeze ProjectedSet and attach the vaccine history adapter."""

    model = MetaAdapterDigitalTwin(
        projected_set,
        torch.empty(0, config.adapter_config().gene_embedding_dim),
        config.adapter_config(),
    )
    # The visit tokenizer is unused when use_visit_tokens=False, so it is frozen.
    if not config.use_visit_tokens:
        for parameter in model.identifier.history_encoder.visit_tokenizer.parameters():
            parameter.requires_grad_(False)
    return model

def shared_model_parameter_audit(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }

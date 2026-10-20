from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from model.encoder import PretrainedVisitEncoder, VaccineImmuneTwinConfig
from model.response import ProgramAffineTransport

SET_AFFINE_NAME = "set_affine"

def _attention_heads(width: int) -> int:
    for heads in (4, 2, 1):
        if width % heads == 0:
            return heads
    raise RuntimeError("unreachable attention-head selection")

def _cell_mask(
    cells: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    expected_dim: int,
    name: str,
) -> torch.Tensor:
    if cells.ndim != 3 or cells.shape[-1] != expected_dim:
        raise ValueError(
            f"{name} must be [batch,cells,{expected_dim}], got {tuple(cells.shape)}"
        )
    if cells.shape[1] < 1:
        raise ValueError(f"{name} requires at least one cell")
    if mask is None:
        return torch.ones(cells.shape[:2], dtype=torch.bool, device=cells.device)
    if mask.shape != cells.shape[:2]:
        raise ValueError(f"{name} mask shape changed")
    result = mask.to(device=cells.device, dtype=torch.bool)
    if not torch.all(result.any(dim=1)):
        raise ValueError(f"every {name} set requires at least one valid cell")
    return result

class CurrentPatientSetEncoder(nn.Module):
    """Pool two unpaired current-visit cell sets into shared patient slots."""

    def __init__(self, config: VaccineImmuneTwinConfig) -> None:
        super().__init__()
        if config.current_operator != SET_AFFINE_NAME:
            raise ValueError("CurrentPatientSetEncoder requires set_affine")
        if config.operator_slot_count != 8:
            raise ValueError("set_affine requires exactly eight patient slots")
        self.config = config
        width = config.visit_dim
        heads = _attention_heads(width)

        self.slots = nn.Parameter(torch.empty(config.operator_slot_count, width))
        nn.init.normal_(self.slots, mean=0.0, std=0.02)
        self.phase_embedding = nn.Embedding(config.phase_count, width)
        self.source_projection = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.slot_norm = nn.LayerNorm(width)
        self.uce_norm = nn.LayerNorm(width)
        self.source_norm = nn.LayerNorm(width)
        self.uce_cross_attention = nn.MultiheadAttention(
            width, heads, batch_first=True, dropout=0.0
        )
        self.source_cross_attention = nn.MultiheadAttention(
            width, heads, batch_first=True, dropout=0.0
        )
        self.mixer_norm = nn.LayerNorm(width)
        self.slot_mixer = nn.MultiheadAttention(
            width, heads, batch_first=True, dropout=0.0
        )
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )

    def forward(
        self,
        projected_uce: torch.Tensor,
        source_cells: torch.Tensor,
        phase: torch.Tensor,
        *,
        uce_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = projected_uce.shape[0]
        width = self.config.visit_dim
        if projected_uce.ndim != 3 or projected_uce.shape[-1] != width:
            raise ValueError("projected_uce must be [batch,cells,visit_dim]")
        if source_cells.shape[0] != batch:
            raise ValueError("UCE and scVI sets must share only the batch dimension")
        if phase.shape != (batch,):
            raise ValueError("phase must be [batch]")
        phase_index = phase.to(device=projected_uce.device, dtype=torch.long)
        if phase_index.numel() and (
            torch.any(phase_index < 0)
            or torch.any(phase_index >= self.config.phase_count)
        ):
            raise ValueError("phase index is outside the configured range")

        slots = self.slots.to(projected_uce.dtype).unsqueeze(0).expand(batch, -1, -1)
        slots = slots + self.phase_embedding(phase_index).to(slots.dtype).unsqueeze(1)
        query = self.slot_norm(slots)
        uce_values = self.uce_norm(projected_uce)
        source_values = self.source_norm(self.source_projection(source_cells))
        uce_update, uce_attention = self.uce_cross_attention(
            query,
            uce_values,
            uce_values,
            key_padding_mask=~uce_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        source_update, source_attention = self.source_cross_attention(
            query,
            source_values,
            source_values,
            key_padding_mask=~source_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        slots = slots + uce_update + source_update
        mixed, _ = self.slot_mixer(
            self.mixer_norm(slots),
            self.mixer_norm(slots),
            self.mixer_norm(slots),
            need_weights=False,
        )
        slots = slots + mixed
        slots = slots + self.output(self.output_norm(slots))
        return slots, uce_attention, source_attention

class SetAffineCurrentModel(nn.Module):
    """Set Transformer patient encoder with SPD affine endpoint decoder."""

    def __init__(self, config: VaccineImmuneTwinConfig) -> None:
        super().__init__()
        if config.current_operator != SET_AFFINE_NAME:
            raise ValueError("SetAffineCurrentModel requires set_affine")
        self.config = config
        self.visit_encoder = PretrainedVisitEncoder(config)
        self.patient_set_encoder = CurrentPatientSetEncoder(config)
        self.transport = ProgramAffineTransport(config)

        width = config.visit_dim
        context_dim = 2 * width + config.source_geometry_dim
        self.slot_score_norm = nn.LayerNorm(width)
        self.slot_score = nn.Linear(width, 1, bias=False)
        self.patient_encoder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, config.transition_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.transition_hidden_dim),
        )
        self.mean_direction_head = nn.Linear(
            config.transition_hidden_dim, config.mean_program_rank
        )
        self.mean_magnitude_head = nn.Linear(config.transition_hidden_dim, 1)
        self.covariance_head = nn.Linear(
            config.transition_hidden_dim, config.covariance_program_rank
        )

        # The fitted program rows have training-response RMS scale, so an
        # initial coefficient norm of one is a stable, interpretable start.
        inverse_softplus_one = math.log(math.expm1(1.0))
        nn.init.zeros_(self.mean_magnitude_head.weight)
        nn.init.constant_(self.mean_magnitude_head.bias, inverse_softplus_one)
        # An exactly identity relative covariance has a repeated eigenspectrum;
        # its eigenvectors do not have a finite backward derivative.  A small
        # tangent perturbation keeps the initial transport near identity while
        # making the SPD square-root gradient well defined.
        nn.init.normal_(self.covariance_head.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.covariance_head.bias)

    @torch.no_grad()
    def set_response_transform(
        self, response_center: torch.Tensor, response_scale: torch.Tensor
    ) -> None:
        self.transport.set_response_transform(response_center, response_scale)

    @torch.no_grad()
    def set_response_programs(self, **programs: torch.Tensor | float) -> None:
        self.transport.set_response_programs(**programs)

    def forward(
        self,
        *,
        arm: str,
        query_uce: torch.Tensor,
        source_latent: torch.Tensor,
        phase: torch.Tensor,
        query_uce_mask: torch.Tensor | None = None,
        source_latent_mask: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if arm != "current":
            raise ValueError("set_affine backbone is current-only")
        query_mask = _cell_mask(
            query_uce,
            query_uce_mask,
            expected_dim=self.config.uce_dim,
            name="query_uce",
        )
        source_mask = _cell_mask(
            source_latent,
            source_latent_mask,
            expected_dim=self.config.latent_dim,
            name="source_latent",
        )
        if query_uce.shape[0] != source_latent.shape[0]:
            raise ValueError("UCE and scVI sets must share the batch dimension")

        projected = self.visit_encoder.cell_projection(query_uce)
        global_state, global_attention = self.visit_encoder.pool(projected, query_mask)
        slots, slot_uce_attention, slot_source_attention = self.patient_set_encoder(
            projected,
            source_latent,
            phase,
            uce_mask=query_mask,
            source_mask=source_mask,
        )
        slot_attention = torch.softmax(
            self.slot_score(self.slot_score_norm(slots)).squeeze(-1), dim=1
        )
        slot_state = torch.sum(slot_attention.unsqueeze(-1) * slots, dim=1)
        source_geometry = self.transport.encode_source_geometry(source_latent)
        patient_state = self.patient_encoder(
            torch.cat(
                (
                    global_state.float(),
                    slot_state.float(),
                    source_geometry.float(),
                ),
                dim=-1,
            )
        )

        raw_direction = self.mean_direction_head(patient_state.float())
        mean_direction = F.normalize(raw_direction, p=2.0, dim=-1, eps=1e-6)
        mean_magnitude = F.softplus(self.mean_magnitude_head(patient_state.float()))
        mean_coefficients = mean_direction * mean_magnitude
        covariance_coefficients = self.covariance_head(patient_state.float())
        predicted, operator = self.transport.apply_coefficients(
            source_latent,
            mean_coefficients,
            covariance_coefficients,
        )

        batch = query_uce.shape[0]
        empty = torch.empty(batch, 0, device=query_uce.device, dtype=query_uce.dtype)
        return {
            "predicted_cells": predicted,
            "query_state": global_state,
            "query_attention": global_attention,
            "patient_slots": slots,
            "patient_slot_attention": slot_attention,
            "slot_uce_attention": slot_uce_attention,
            "slot_source_attention": slot_source_attention,
            "source_geometry_state": source_geometry.to(predicted.dtype),
            "patient_state": patient_state.to(predicted.dtype),
            "mean_program_direction": mean_direction.to(predicted.dtype),
            "mean_program_magnitude": mean_magnitude.to(predicted.dtype),
            "w": empty,
            "gene_weights": empty,
            **operator,
        }

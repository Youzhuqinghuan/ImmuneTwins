from __future__ import annotations
from dataclasses import replace
import math
from typing import Any, Mapping
import torch
from torch import nn
from torch.nn import functional as F
from model.encoder import VaccineImmuneTwinConfig
from model.patient import SetAffineCurrentModel, _cell_mask

MOMENT_FIBER_NAME = "moment_fiber"

def _attention_heads(width: int) -> int:
    for heads in (4, 2, 1):
        if width % heads == 0:
            return heads
    raise RuntimeError("unreachable attention-head selection")

def _masked_mean_covariance(
    cells: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean, centered values and unbiased covariance in float32."""

    values = cells.float()
    valid = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    count = valid.sum(dim=1, keepdim=True)
    if torch.any(count <= values.shape[-1]):
        raise ValueError("moment fiber requires more valid cells than dimensions")
    mean = torch.sum(values * valid, dim=1, keepdim=True) / count
    centered = (values - mean) * valid
    covariance = torch.einsum("bni,bnj->bij", centered, centered)
    covariance = covariance / (count.squeeze(-1) - 1.0).unsqueeze(-1)
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    return mean, centered, covariance

def _cholesky_factor(covariance: torch.Tensor) -> torch.Tensor:
    """Return a stable lower factor without repeated-eigenvalue ambiguity."""

    factor, info = torch.linalg.cholesky_ex(covariance.float(), check_errors=False)
    if torch.any(info != 0):
        raise ValueError("moment fiber covariance is not numerically full rank")
    return factor

def canonical_moment_fiber(
    cells: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map a finite set to zero mean and identity unbiased covariance."""

    mean, centered, covariance = _masked_mean_covariance(cells, mask)
    lower = _cholesky_factor(covariance)
    canonical = torch.linalg.solve_triangular(
        lower,
        centered.transpose(-1, -2),
        upper=False,
    ).transpose(-1, -2)
    canonical = canonical * mask.unsqueeze(-1).to(canonical.dtype)
    # If Z = X L^{-T}, then X = Z L^T.  The returned right factor therefore
    # reconstructs the base set exactly in the fixed coordinate order.
    return canonical, mean, lower.transpose(-1, -2)

def retract_to_moment_fiber(
    coordinates: torch.Tensor,
    proposal: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a proposal to the tangent and retract to the finite-set fiber."""

    if proposal.shape != coordinates.shape:
        raise ValueError("fiber proposal shape changed")
    _, centered_proposal, _ = _masked_mean_covariance(proposal, mask)
    valid = mask.unsqueeze(-1).to(coordinates.dtype)
    count = valid.sum(dim=1, keepdim=True)
    cross = torch.einsum("bni,bnj->bij", coordinates, centered_proposal)
    cross = cross / (count.squeeze(-1) - 1.0).unsqueeze(-1)
    symmetric_cross = 0.5 * (cross + cross.transpose(-1, -2))
    tangent = (centered_proposal - coordinates @ symmetric_cross) * valid
    candidate = (coordinates + tangent) * valid
    retracted, _, _ = canonical_moment_fiber(candidate, mask)
    return retracted, tangent

class SoftLocalFiberBlock(nn.Module):
    """Permutation-equivariant local/global attention with selectable geometry."""

    def __init__(self, config: VaccineImmuneTwinConfig) -> None:
        super().__init__()
        self.config = config
        latent = config.latent_dim
        width = config.source_geometry_dim
        heads = _attention_heads(width)
        self.cell_input = nn.Sequential(
            nn.LayerNorm(latent),
            nn.Linear(latent, width),
            nn.GELU(),
        )
        self.cell_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.slot_norm = nn.LayerNorm(width)
        self.slot_attention = nn.MultiheadAttention(
            width, heads, batch_first=True, dropout=0.0
        )
        self.update = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, latent),
        )
        # Exact identity at initialization, while the output layer receives a
        # non-zero gradient on the first optimization step.
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        self.context_gate: nn.Module | None = None
        if config.fiber_context == "adaptive":
            self.context_gate = nn.Sequential(
                nn.LayerNorm(4 * width),
                nn.Linear(4 * width, width),
                nn.GELU(),
                nn.Linear(width, 1),
            )
            nn.init.zeros_(self.context_gate[-1].weight)
            nn.init.zeros_(self.context_gate[-1].bias)
        self.constraint_gate: nn.Module | None = None
        if config.fiber_retraction == "adaptive_final":
            self.constraint_gate = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, 1),
            )
            nn.init.zeros_(self.constraint_gate[-1].weight)
            nn.init.zeros_(self.constraint_gate[-1].bias)
        initial_bandwidth = math.sqrt(float(latent))
        self.raw_bandwidth = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_bandwidth)))
        )

    def _soft_local_attention(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        mask: torch.Tensor,
        *,
        query_chunk: int = 256,
    ) -> torch.Tensor:
        normalized = self.cell_norm(features)
        query = self.query(normalized)
        key = self.key(normalized)
        value = self.value(normalized)
        scale = float(query.shape[-1]) ** -0.5
        bandwidth = F.softplus(self.raw_bandwidth).clamp_min(1e-3)
        key_mask = mask.unsqueeze(1)
        outputs: list[torch.Tensor] = []
        for start in range(0, query.shape[1], query_chunk):
            stop = min(start + query_chunk, query.shape[1])
            q = query[:, start:stop]
            q_coordinates = coordinates[:, start:stop]
            logits = torch.einsum("bqh,bkh->bqk", q, key) * scale
            distance2 = torch.sum(
                (q_coordinates.unsqueeze(2) - coordinates.unsqueeze(1)).square(),
                dim=-1,
            )
            logits = logits - distance2 / (2.0 * bandwidth.square())
            logits = logits.masked_fill(~key_mask, -torch.inf)
            weights = torch.softmax(logits, dim=-1)
            result = torch.einsum("bqk,bkh->bqh", weights, value)
            result = result * mask[:, start:stop].unsqueeze(-1).to(result.dtype)
            outputs.append(result)
        return torch.cat(outputs, dim=1)

    def forward(
        self,
        coordinates: torch.Tensor,
        patient_slots: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self.cell_input(coordinates)
        valid = mask.unsqueeze(-1).to(features.dtype)
        normalized = self.cell_norm(features)
        pooled = torch.sum(self.value(normalized) * valid, dim=1)
        pooled = pooled / valid.sum(dim=1).clamp_min(1.0)
        pooled = pooled.unsqueeze(1).expand(-1, features.shape[1], -1) * valid
        slot_context, slot_attention = self.slot_attention(
            normalized,
            self.slot_norm(patient_slots),
            self.slot_norm(patient_slots),
            need_weights=True,
            average_attn_weights=True,
        )
        if self.config.fiber_context == "pooled":
            context = pooled
            context_gate = torch.zeros_like(mask, dtype=features.dtype)
        else:
            local = self._soft_local_attention(features, coordinates, mask)
            if self.config.fiber_context == "local":
                context = local
                context_gate = torch.ones_like(mask, dtype=features.dtype)
            else:
                assert self.context_gate is not None
                context_gate = torch.sigmoid(
                    self.context_gate(
                        torch.cat((features, local, pooled, slot_context), dim=-1)
                    ).squeeze(-1)
                )
                context_gate = context_gate * mask.to(context_gate.dtype)
                context = (
                    context_gate.unsqueeze(-1) * local
                    + (1.0 - context_gate.unsqueeze(-1)) * pooled
                )
        proposal = self.config.operator_scale * self.update(
            torch.cat((features, context, slot_context), dim=-1)
        )
        if self.config.fiber_retraction == "every_block":
            updated, realized_update = retract_to_moment_fiber(
                coordinates, proposal, mask
            )
            constraint_gate = torch.ones(
                coordinates.shape[0], device=coordinates.device, dtype=coordinates.dtype
            )
        elif self.config.fiber_retraction == "adaptive_final":
            assert self.constraint_gate is not None
            constrained, _ = retract_to_moment_fiber(coordinates, proposal, mask)
            ambient = (coordinates + proposal * valid) * valid
            patient_state = patient_slots.mean(dim=1)
            constraint_gate = torch.sigmoid(
                self.constraint_gate(patient_state).squeeze(-1)
            )
            mix = constraint_gate.view(-1, 1, 1)
            updated = (mix * constrained + (1.0 - mix) * ambient) * valid
            realized_update = updated - coordinates
        else:
            constraint_gate = torch.zeros(
                coordinates.shape[0], device=coordinates.device, dtype=coordinates.dtype
            )
            realized_update = proposal * valid
            updated = (coordinates + realized_update) * valid
        return updated, {
            "fiber_slot_attention": slot_attention,
            "fiber_context_gate": context_gate,
            "fiber_constraint_gate": constraint_gate,
            "fiber_update_norm": torch.sqrt(
                torch.mean(realized_update.square(), dim=(1, 2)).clamp_min(0.0)
            ),
            "fiber_bandwidth": F.softplus(self.raw_bandwidth).expand(
                coordinates.shape[0]
            ),
        }

class MomentFiberOperator(nn.Module):
    """Refine standardized higher-order shape while preserving base moments."""

    def __init__(self, config: VaccineImmuneTwinConfig) -> None:
        super().__init__()
        self.config = config
        width = config.source_geometry_dim
        self.slot_projection = nn.Sequential(
            nn.LayerNorm(config.visit_dim),
            nn.Linear(config.visit_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList(
            [SoftLocalFiberBlock(config) for _ in range(config.operator_depth)]
        )

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        self,
        base_cells: torch.Tensor,
        patient_slots: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coordinates, base_mean, base_root = canonical_moment_fiber(base_cells, mask)
        slots = self.slot_projection(patient_slots.float())
        attentions: list[torch.Tensor] = []
        tangent_norms: list[torch.Tensor] = []
        bandwidths: list[torch.Tensor] = []
        context_gates: list[torch.Tensor] = []
        constraint_gates: list[torch.Tensor] = []
        for block in self.blocks:
            coordinates, state = block(coordinates, slots, mask)
            attentions.append(state["fiber_slot_attention"])
            tangent_norms.append(state["fiber_update_norm"])
            bandwidths.append(state["fiber_bandwidth"])
            context_gates.append(state["fiber_context_gate"])
            constraint_gates.append(state["fiber_constraint_gate"])
        if self.config.fiber_retraction in {"final_only", "adaptive_final"}:
            coordinates, _, _ = canonical_moment_fiber(coordinates, mask)
        refined = base_mean + coordinates @ base_root
        refined = torch.where(mask.unsqueeze(-1), refined, base_cells.float())
        refined_mean, _, refined_covariance = _masked_mean_covariance(refined, mask)
        _, _, base_covariance = _masked_mean_covariance(base_cells, mask)
        return refined, {
            "base_predicted_cells": base_cells.float(),
            "moment_fiber_coordinates": coordinates,
            "fiber_slot_attention": torch.stack(attentions, dim=1),
            "fiber_context_gate": torch.stack(context_gates, dim=1),
            "fiber_constraint_gate": torch.stack(constraint_gates, dim=1),
            "fiber_update_norm": torch.stack(tangent_norms, dim=1),
            "fiber_bandwidth": torch.stack(bandwidths, dim=1),
            "fiber_mean_preservation_error": (
                refined_mean - base_mean
            ).abs().amax(dim=(1, 2)),
            "fiber_covariance_preservation_error": torch.linalg.matrix_norm(
                refined_covariance - base_covariance, ord="fro", dim=(-2, -1)
            ),
        }

class MomentFiberCurrentModel(SetAffineCurrentModel):
    """Frozen SetTransport backbone plus a trainable geometric shape operator."""

    def __init__(self, config: VaccineImmuneTwinConfig) -> None:
        if config.current_operator != MOMENT_FIBER_NAME:
            raise ValueError("MomentFiberCurrentModel requires moment_fiber")
        base_config = replace(config, current_operator="set_affine")
        super().__init__(base_config)
        self.config = config
        self.fiber_operator = MomentFiberOperator(config)

    def load_frozen_set_transport_state(
        self, state: Mapping[str, torch.Tensor]
    ) -> tuple[str, ...]:
        incompatible = self.load_state_dict(dict(state), strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                "set-transport checkpoint contains unexpected parameters: "
                f"{incompatible.unexpected_keys}"
            )
        expected_missing = {
            name for name in self.state_dict() if name.startswith("fiber_operator.")
        }
        if set(incompatible.missing_keys) != expected_missing:
            raise ValueError("set-transport checkpoint does not match the frozen backbone")
        self.configure_fiber_training()
        return tuple(
            name for name in self.state_dict() if not name.startswith("fiber_operator.")
        )

    def configure_fiber_training(self) -> tuple[str, ...]:
        trainable: list[str] = []
        for name, parameter in self.named_parameters():
            enabled = name.startswith("fiber_operator.")
            parameter.requires_grad_(enabled)
            if enabled:
                trainable.append(name)
        return tuple(trainable)

    def forward(
        self,
        *,
        arm: str,
        query_uce: torch.Tensor,
        source_latent: torch.Tensor,
        phase: torch.Tensor,
        query_uce_mask: torch.Tensor | None = None,
        source_latent_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        base = SetAffineCurrentModel.forward(
            self,
            arm=arm,
            query_uce=query_uce,
            source_latent=source_latent,
            phase=phase,
            query_uce_mask=query_uce_mask,
            source_latent_mask=source_latent_mask,
            **kwargs,
        )
        mask = _cell_mask(
            source_latent,
            source_latent_mask,
            expected_dim=self.config.latent_dim,
            name="source_latent",
        )
        refined, fiber = self.fiber_operator(
            base["predicted_cells"], base["patient_slots"], mask
        )
        mean_shift = base["mean_shift"].float().unsqueeze(1)
        return {
            **base,
            **fiber,
            "predicted_cells": refined,
            "shape_delta": refined - source_latent.float() - mean_shift,
        }

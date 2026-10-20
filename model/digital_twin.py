from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from model.projected_set import MomentFiberCurrentModel, _cell_mask

@dataclass(frozen=True)
class MetaAdapterConfig:
    context_dim: int = 128
    visit_slots: int = 4
    context_layers: int = 2
    context_heads: int = 4
    adapter_rank: int = 8
    adapter_scale: float = 0.5
    gene_embedding_dim: int = 32
    clinical_value_dim: int = 13
    use_gene_attention: bool = True
    use_clinical_history: bool = False
    use_visit_tokens: bool = True
    use_history_gate: bool = False
    use_recovery_basis: bool = False
    recovery_same_phase: bool = False
    recovery_coefficient_initial: float = 0.54
    response_mean_loss_weight: float = 2.0
    response_direction_loss_weight: float = 0.5
    gene_dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "context_dim",
            "visit_slots",
            "context_layers",
            "context_heads",
            "adapter_rank",
            "gene_embedding_dim",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.context_dim % self.context_heads:
            raise ValueError("context_dim must be divisible by context_heads")
        if not math.isfinite(self.adapter_scale) or self.adapter_scale <= 0:
            raise ValueError("adapter_scale must be finite and positive")
        if self.clinical_value_dim < 0:
            raise ValueError("clinical_value_dim cannot be negative")
        if not 0 < self.recovery_coefficient_initial < 2:
            raise ValueError("recovery_coefficient_initial must be in (0, 2)")
        if self.response_mean_loss_weight < 0:
            raise ValueError("response_mean_loss_weight cannot be negative")
        if self.response_direction_loss_weight < 0:
            raise ValueError("response_direction_loss_weight cannot be negative")
        if not 0 <= self.gene_dropout < 1:
            raise ValueError("gene_dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparse probability mapping from Martins and Astudillo (2016)."""

    shifted = logits - logits.amax(dim=dim, keepdim=True)
    sorted_logits, _ = torch.sort(shifted, descending=True, dim=dim)
    count = logits.shape[dim]
    ranks = torch.arange(
        1, count + 1, device=logits.device, dtype=logits.dtype
    )
    view = [1] * logits.ndim
    view[dim] = count
    ranks = ranks.view(view)
    cumulative = sorted_logits.cumsum(dim)
    support = 1 + ranks * sorted_logits > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (cumulative.gather(dim, support_size - 1) - 1) / support_size.to(
        logits.dtype
    )
    return torch.clamp(shifted - tau, min=0)

class HistoryVisitTokenizer(nn.Module):
    """Convert each historical cell set into several permutation-invariant slots."""

    def __init__(self, backbone: MomentFiberCurrentModel, config: MetaAdapterConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        visit_dim = backbone.config.visit_dim
        self.slots = nn.Parameter(torch.empty(config.visit_slots, visit_dim))
        nn.init.normal_(self.slots, mean=0.0, std=0.02)
        self.slot_norm = nn.LayerNorm(visit_dim)
        self.cell_norm = nn.LayerNorm(visit_dim)
        self.cross_attention = nn.MultiheadAttention(
            visit_dim,
            max(1, min(4, config.context_heads)),
            batch_first=True,
            dropout=0.0,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(visit_dim),
            nn.Linear(visit_dim, config.context_dim),
            nn.GELU(),
            nn.Linear(config.context_dim, config.context_dim),
        )

    def forward(self, cells: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
        batch, support, cell_count, width = cells.shape
        if width != self.backbone.config.uce_dim:
            raise ValueError("historical UCE width differs from the backbone")
        if support_mask.shape != (batch, support) or support_mask.dtype != torch.bool:
            raise ValueError("support_mask must be boolean [batch,support]")
        if support == 0:
            return cells.new_empty(batch, 0, self.config.visit_slots, self.config.context_dim)
        flat = cells.reshape(batch * support, cell_count, width)
        projected = self.backbone.visit_encoder.cell_projection(flat)
        queries = self.slots.to(projected.dtype).unsqueeze(0).expand(
            batch * support, -1, -1
        )
        tokens, _ = self.cross_attention(
            self.slot_norm(queries),
            self.cell_norm(projected),
            self.cell_norm(projected),
            need_weights=False,
        )
        tokens = self.output(tokens).reshape(
            batch, support, self.config.visit_slots, self.config.context_dim
        )
        return tokens * support_mask[:, :, None, None].to(tokens.dtype)

class PatientSpecificLowRankAdapter(nn.Module):
    """Shared low-rank bases with support-generated patient-specific scales."""

    def __init__(self, width: int, rank: int, scale: float) -> None:
        super().__init__()
        self.scale = float(scale) / math.sqrt(float(rank))
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        # Orthogonal shared bases; patient scales start at zero, so the adapter
        # initially leaves the base model unchanged.
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight)

    def forward(self, values: torch.Tensor, patient_scale: torch.Tensor) -> torch.Tensor:
        if patient_scale.shape != (values.shape[0], self.down.out_features):
            raise ValueError("patient adapter scale has the wrong shape")
        shape = [values.shape[0]] + [1] * (values.ndim - 2) + [patient_scale.shape[-1]]
        adapted = self.down(values) * patient_scale.reshape(shape)
        return values + self.scale * self.up(adapted)

class PatientSpecificLowRankMap(nn.Module):
    """A history-generated LoRA update for a frozen prediction head."""

    def __init__(self, input_width: int, output_width: int, rank: int, scale: float) -> None:
        super().__init__()
        self.scale = float(scale) / math.sqrt(float(rank))
        self.down = nn.Linear(input_width, rank, bias=False)
        self.up = nn.Linear(rank, output_width, bias=False)
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight)

    def forward(self, values: torch.Tensor, patient_scale: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or patient_scale.shape != (
            values.shape[0],
            self.down.out_features,
        ):
            raise ValueError("history-generated prediction-head scale has the wrong shape")
        return self.scale * self.up(self.down(values) * patient_scale)

class EpisodicHistoryEncoder(nn.Module):
    """Encode RNA visits, response genes and selected clinical history."""

    def __init__(
        self,
        backbone: MomentFiberCurrentModel,
        gene_coordinates: torch.Tensor,
        config: MetaAdapterConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.visit_tokenizer = HistoryVisitTokenizer(backbone, config)
        width = config.context_dim
        self.phase_embedding = nn.Embedding(3, width)
        self.role_embedding = nn.Embedding(3, width)
        self.order_embedding = nn.Embedding(4, width)
        self.time_embedding = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        self.rna_transition_projection = nn.Sequential(
            nn.LayerNorm(3 * backbone.config.latent_dim),
            nn.Linear(3 * backbone.config.latent_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

        coordinates = torch.as_tensor(gene_coordinates, dtype=torch.float32)
        if config.use_gene_attention:
            if coordinates.ndim != 2 or coordinates.shape[0] < 2:
                raise ValueError("gene attention requires [genes,dimensions] coordinates")
            if coordinates.shape[1] != config.gene_embedding_dim:
                raise ValueError("gene coordinate width differs from config")
        elif coordinates.numel() and coordinates.ndim != 2:
            raise ValueError("gene coordinates must be a matrix")
        self.register_buffer("gene_coordinates", coordinates, persistent=False)
        self.gene_projection: nn.Module | None = None
        self.gene_query: nn.Module | None = None
        self.gene_effect_score: nn.Module | None = None
        self.gene_output: nn.Module | None = None
        if config.use_gene_attention:
            self.gene_projection = nn.Sequential(
                nn.LayerNorm(config.gene_embedding_dim),
                nn.Linear(config.gene_embedding_dim, width),
                nn.GELU(),
            )
            self.gene_query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
            self.gene_effect_score = nn.Sequential(
                nn.Linear(2, width // 2), nn.GELU(), nn.Linear(width // 2, 1)
            )
            self.gene_output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))

        self.clinical_projection: nn.Module | None = None
        if config.use_clinical_history:
            if config.clinical_value_dim < 1:
                raise ValueError("clinical history requires positive clinical_value_dim")
            self.clinical_projection = nn.Sequential(
                nn.LayerNorm(2 * config.clinical_value_dim),
                nn.Linear(2 * config.clinical_value_dim, width),
                nn.GELU(),
                nn.Linear(width, width),
            )

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.context_heads,
            dim_feedforward=2 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_transformer = nn.TransformerEncoder(
            layer, num_layers=config.context_layers, enable_nested_tensor=False
        )
        self.query_projection = nn.Linear(backbone.config.transition_hidden_dim, width)
        self.query_attention = nn.MultiheadAttention(
            width, config.context_heads, batch_first=True, dropout=0.0
        )
        self.output_norm = nn.LayerNorm(width)

    @property
    def n_genes(self) -> int:
        return int(self.gene_coordinates.shape[0]) if self.gene_coordinates.ndim == 2 else 0

    def _gene_token(
        self,
        transition_query: torch.Tensor,
        gene_change: torch.Tensor,
        gene_confidence: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, support, _ = transition_query.shape
        if not self.config.use_gene_attention:
            return (
                transition_query.new_zeros(batch, support, self.config.context_dim),
                transition_query.new_empty(batch, support, 0),
            )
        if gene_change.shape != (batch, support, self.n_genes):
            raise ValueError("gene_change differs from selected response genes")
        if gene_confidence.shape != gene_change.shape:
            raise ValueError("gene_confidence shape changed")
        assert self.gene_projection is not None
        assert self.gene_query is not None
        assert self.gene_effect_score is not None
        assert self.gene_output is not None
        genes = self.gene_projection(self.gene_coordinates).to(transition_query.dtype)
        query = self.gene_query(transition_query)
        semantic = torch.einsum("bsd,gd->bsg", query, genes) / math.sqrt(
            float(self.config.context_dim)
        )
        effect = torch.stack(
            (torch.tanh(gene_change), gene_confidence.clamp(0, 1)), dim=-1
        )
        logits = semantic + self.gene_effect_score(effect).squeeze(-1)
        if self.training and self.config.gene_dropout > 0:
            keep = torch.rand_like(logits).ge(self.config.gene_dropout)
            logits = logits.masked_fill(~keep, torch.finfo(logits.dtype).min)
        weights = sparsemax(logits, dim=-1)
        weights = weights * support_mask.unsqueeze(-1).to(weights.dtype)
        signed = torch.tanh(gene_change) * gene_confidence.clamp(0, 1)
        token = torch.einsum("bsg,bsg,gd->bsd", weights, signed, genes)
        return self.gene_output(token), weights

    def forward(
        self,
        *,
        current_patient_state: torch.Tensor,
        support_source_uce: torch.Tensor,
        support_target_uce: torch.Tensor,
        support_phase: torch.Tensor,
        support_dt: torch.Tensor,
        support_mask: torch.Tensor,
        support_source_latent_mean: torch.Tensor,
        support_target_latent_mean: torch.Tensor,
        support_response_latent: torch.Tensor,
        gene_change: torch.Tensor,
        gene_confidence: torch.Tensor,
        clinical_delta_value: torch.Tensor,
        clinical_delta_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, support = support_mask.shape
        if support == 0:
            return {
                "history_context": current_patient_state.new_zeros(
                    batch, self.config.context_dim
                ),
                "history_tokens": current_patient_state.new_empty(
                    batch, 0, self.config.context_dim
                ),
                "history_token_mask": support_mask.new_empty(batch, 0),
                "gene_weights": current_patient_state.new_empty(batch, 0),
            }
        if support_source_uce.shape[:2] != (batch, support):
            raise ValueError("support source visits do not align with support_mask")
        if support_target_uce.shape != support_source_uce.shape:
            raise ValueError("support endpoint UCE tensors must share a shape")
        if support_response_latent.shape != (
            batch,
            support,
            self.backbone.config.latent_dim,
        ):
            raise ValueError("support RNA response shape changed")

        latent_shape = (batch, support, self.backbone.config.latent_dim)
        if (
            support_source_latent_mean.shape != latent_shape
            or support_target_latent_mean.shape != latent_shape
        ):
            raise ValueError("support latent visit means must align with transitions")

        phase = self.phase_embedding(support_phase.long())
        elapsed = self.time_embedding(
            support_dt.unsqueeze(-1).to(current_patient_state.dtype)
        )
        order = self.order_embedding(
            torch.arange(support, device=current_patient_state.device).clamp_max(3)
        ).reshape(1, support, 1, -1)
        rna_transition = self.rna_transition_projection(
            torch.cat(
                (
                    support_source_latent_mean,
                    support_target_latent_mean,
                    support_response_latent,
                ),
                dim=-1,
            )
        )
        source: torch.Tensor | None = None
        target: torch.Tensor | None = None
        if self.config.use_visit_tokens:
            source = self.visit_tokenizer(support_source_uce, support_mask)
            target = self.visit_tokenizer(support_target_uce, support_mask)
            source = source + phase[:, :, None, :] + elapsed[:, :, None, :] + order
            target = target + phase[:, :, None, :] + elapsed[:, :, None, :] + order
            source = source + self.role_embedding.weight[0].reshape(1, 1, 1, -1)
            target = target + self.role_embedding.weight[1].reshape(1, 1, 1, -1)
            transition_query = source.mean(dim=2) + target.mean(dim=2) + rna_transition
        else:
            transition_query = (
                rna_transition + phase + elapsed + order.squeeze(2)
            )
        gene_token, gene_weights = self._gene_token(
            transition_query,
            gene_change,
            gene_confidence,
            support_mask,
        )
        transition = transition_query + gene_token
        if self.config.use_clinical_history:
            assert self.clinical_projection is not None
            if clinical_delta_value.shape != (
                batch,
                support,
                self.config.clinical_value_dim,
            ):
                raise ValueError("clinical delta width differs from config")
            if clinical_delta_valid.shape != clinical_delta_value.shape:
                raise ValueError("clinical validity shape changed")
            transition = transition + self.clinical_projection(
                torch.cat(
                    (
                        clinical_delta_value
                        * clinical_delta_valid.to(clinical_delta_value.dtype),
                        clinical_delta_valid.to(clinical_delta_value.dtype),
                    ),
                    dim=-1,
                )
            )
        transition = transition + self.role_embedding.weight[2].reshape(1, 1, -1)
        transition = transition * support_mask.unsqueeze(-1).to(transition.dtype)

        chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        for index in range(support):
            valid = support_mask[:, index : index + 1]
            if self.config.use_visit_tokens:
                assert source is not None and target is not None
                chunks.extend(
                    (source[:, index], target[:, index], transition[:, index : index + 1])
                )
                mask_chunks.extend(
                    (
                        valid.expand(-1, self.config.visit_slots),
                        valid.expand(-1, self.config.visit_slots),
                        valid,
                    )
                )
            else:
                chunks.append(transition[:, index : index + 1])
                mask_chunks.append(valid)
        tokens = torch.cat(chunks, dim=1)
        token_mask = torch.cat(mask_chunks, dim=1)
        encoded = self.context_transformer(tokens, src_key_padding_mask=~token_mask)
        query = self.query_projection(current_patient_state).unsqueeze(1)
        context, attention = self.query_attention(
            query,
            encoded,
            encoded,
            key_padding_mask=~token_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        context = self.output_norm(context.squeeze(1))
        has_history = support_mask.any(dim=1)
        context = context * has_history.unsqueeze(-1).to(context.dtype)
        patient_gene_weights = gene_weights.sum(dim=1) / support_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1).to(gene_weights.dtype)
        return {
            "history_context": context,
            "history_tokens": encoded,
            "history_token_mask": token_mask,
            "history_attention": attention.squeeze(1),
            "gene_weights": patient_gene_weights,
        }

class HistoryGeneratedAdapters(nn.Module):
    """Generate and apply patient-specific low-rank weights from history."""

    def __init__(
        self,
        backbone: MomentFiberCurrentModel,
        gene_coordinates: torch.Tensor,
        config: MetaAdapterConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.history_encoder = EpisodicHistoryEncoder(backbone, gene_coordinates, config)
        self.slot_adapter = PatientSpecificLowRankAdapter(
            backbone.config.visit_dim, config.adapter_rank, config.adapter_scale
        )
        self.patient_adapter = PatientSpecificLowRankAdapter(
            backbone.config.transition_hidden_dim,
            config.adapter_rank,
            config.adapter_scale,
        )
        self.mean_head_adapter = PatientSpecificLowRankMap(
            backbone.config.transition_hidden_dim,
            backbone.config.mean_program_rank,
            config.adapter_rank,
            config.adapter_scale,
        )
        self.covariance_head_adapter = PatientSpecificLowRankMap(
            backbone.config.transition_hidden_dim,
            backbone.config.covariance_program_rank,
            config.adapter_rank,
            config.adapter_scale,
        )
        self.magnitude_head_adapter = PatientSpecificLowRankMap(
            backbone.config.transition_hidden_dim,
            1,
            config.adapter_rank,
            config.adapter_scale,
        )
        self.slot_scale = nn.Linear(config.context_dim, config.adapter_rank)
        self.patient_scale = nn.Linear(config.context_dim, config.adapter_rank)
        self.mean_head_scale = nn.Linear(config.context_dim, config.adapter_rank)
        self.covariance_head_scale = nn.Linear(config.context_dim, config.adapter_rank)
        self.magnitude_head_scale = nn.Linear(config.context_dim, config.adapter_rank)
        # Context prompts act on the patient representation and low-rank maps on
        # the weights; both start at zero.
        self.slot_prompt = nn.Linear(
            config.context_dim, backbone.config.visit_dim, bias=False
        )
        self.patient_prompt = nn.Linear(
            config.context_dim,
            backbone.config.transition_hidden_dim,
            bias=False,
        )
        self.history_gate: nn.Module | None = None
        if config.use_history_gate:
            self.history_gate = nn.Sequential(
                nn.LayerNorm(config.context_dim),
                nn.Linear(config.context_dim, config.context_dim // 2),
                nn.GELU(),
                nn.Linear(config.context_dim // 2, 1),
            )
        self.recovery_coefficient: nn.Linear | None = None
        if config.use_recovery_basis:
            self.recovery_coefficient = nn.Linear(config.context_dim, 1)
            nn.init.zeros_(self.recovery_coefficient.weight)
            initial_probability = config.recovery_coefficient_initial / 2.0
            nn.init.constant_(
                self.recovery_coefficient.bias,
                math.log(initial_probability / (1.0 - initial_probability)),
            )
        nn.init.zeros_(self.slot_scale.weight)
        nn.init.zeros_(self.slot_scale.bias)
        nn.init.zeros_(self.patient_scale.weight)
        nn.init.zeros_(self.patient_scale.bias)
        for head in (
            self.mean_head_scale,
            self.covariance_head_scale,
            self.magnitude_head_scale,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.slot_prompt.weight)
        nn.init.zeros_(self.patient_prompt.weight)

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.history_encoder(**kwargs)
        context = encoded["history_context"]
        gate = (
            torch.sigmoid(self.history_gate(context))
            if self.history_gate is not None
            else context.new_ones(context.shape[0], 1)
        )
        recovery_coefficient = (
            -2.0 * torch.sigmoid(self.recovery_coefficient(context))
            if self.recovery_coefficient is not None
            else context.new_zeros(context.shape[0], 1)
        )
        return {
            **encoded,
            "history_gate": gate,
            "recovery_coefficient": recovery_coefficient,
            "slot_adapter_scale": gate * torch.tanh(self.slot_scale(context)),
            "patient_adapter_scale": gate * torch.tanh(self.patient_scale(context)),
            "mean_head_adapter_scale": gate * torch.tanh(self.mean_head_scale(context)),
            "covariance_head_adapter_scale": gate
            * torch.tanh(self.covariance_head_scale(context)),
            "magnitude_head_adapter_scale": gate
            * torch.tanh(self.magnitude_head_scale(context)),
            "slot_history_prompt": gate * self.slot_prompt(context),
            "patient_history_prompt": gate * self.patient_prompt(context),
        }

class MetaAdapterDigitalTwin(nn.Module):
    """Frozen ProjectedSet personalized by history-generated low-rank adapters."""

    def __init__(
        self,
        backbone: MomentFiberCurrentModel,
        gene_coordinates: torch.Tensor,
        config: MetaAdapterConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.twin_config = config
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.identifier = HistoryGeneratedAdapters(backbone, gene_coordinates, config)
        self.register_buffer(
            "support_response_reference",
            torch.zeros(backbone.config.phase_count, backbone.config.latent_dim),
        )
        self.register_buffer(
            "support_response_reference_observed",
            torch.zeros(backbone.config.phase_count, dtype=torch.bool),
        )

    def set_support_response_reference(
        self, reference: torch.Tensor, observed: torch.Tensor
    ) -> None:
        reference = torch.as_tensor(
            reference,
            device=self.support_response_reference.device,
            dtype=self.support_response_reference.dtype,
        )
        observed = torch.as_tensor(
            observed,
            device=self.support_response_reference_observed.device,
            dtype=torch.bool,
        )
        if reference.shape != self.support_response_reference.shape:
            raise ValueError("support-response reference shape changed")
        if observed.shape != self.support_response_reference_observed.shape:
            raise ValueError("support-response observed mask shape changed")
        if not torch.isfinite(reference).all():
            raise ValueError("support-response reference must be finite")
        self.support_response_reference.copy_(reference)
        self.support_response_reference_observed.copy_(observed)

    def train(self, mode: bool = True) -> "MetaAdapterDigitalTwin":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        *,
        arm: str,
        query_uce: torch.Tensor,
        source_latent: torch.Tensor,
        phase: torch.Tensor,
        support_source_uce: torch.Tensor,
        support_target_uce: torch.Tensor,
        support_phase: torch.Tensor,
        support_dt: torch.Tensor,
        support_mask: torch.Tensor,
        support_source_latent_mean: torch.Tensor,
        support_target_latent_mean: torch.Tensor,
        support_response_latent: torch.Tensor,
        gene_change: torch.Tensor,
        gene_confidence: torch.Tensor,
        clinical_delta_value: torch.Tensor,
        clinical_delta_valid: torch.Tensor,
        query_uce_mask: torch.Tensor | None = None,
        source_latent_mask: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if arm != "full":
            raise ValueError("MetaAdapter digital twin uses the full arm")
        current = self.backbone(
            arm="current",
            query_uce=query_uce,
            source_latent=source_latent,
            phase=phase,
            query_uce_mask=query_uce_mask,
            source_latent_mask=source_latent_mask,
        )
        current_response = current["predicted_cells"].mean(dim=1) - source_latent.mean(
            dim=1
        )
        if support_mask.shape[1] == 0 or not bool(support_mask.any()):
            batch = query_uce.shape[0]
            return {
                **current,
                "current_response": current_response,
                "history_context": query_uce.new_zeros(
                    batch, self.twin_config.context_dim
                ),
                "history_gate": query_uce.new_zeros(batch, 1),
                "recovery_coefficient": query_uce.new_zeros(batch, 1),
                "recovery_basis": query_uce.new_zeros(batch, self.config.latent_dim),
                "recovery_correction": query_uce.new_zeros(
                    batch, self.config.latent_dim
                ),
                "slot_adapter_scale": query_uce.new_zeros(
                    batch, self.twin_config.adapter_rank
                ),
                "patient_adapter_scale": query_uce.new_zeros(
                    batch, self.twin_config.adapter_rank
                ),
                "mean_head_adapter_scale": query_uce.new_zeros(
                    batch, self.twin_config.adapter_rank
                ),
                "covariance_head_adapter_scale": query_uce.new_zeros(
                    batch, self.twin_config.adapter_rank
                ),
                "magnitude_head_adapter_scale": query_uce.new_zeros(
                    batch, self.twin_config.adapter_rank
                ),
                "slot_history_prompt": query_uce.new_zeros(
                    batch, self.config.visit_dim
                ),
                "patient_history_prompt": query_uce.new_zeros(
                    batch, self.config.transition_hidden_dim
                ),
                "gene_weights": query_uce.new_empty(batch, 0),
                "has_history": torch.zeros(
                    batch, dtype=torch.bool, device=query_uce.device
                ),
            }

        history = self.identifier(
            current_patient_state=current["patient_state"].float(),
            support_source_uce=support_source_uce,
            support_target_uce=support_target_uce,
            support_phase=support_phase,
            support_dt=support_dt,
            support_mask=support_mask,
            support_source_latent_mean=support_source_latent_mean,
            support_target_latent_mean=support_target_latent_mean,
            support_response_latent=support_response_latent,
            gene_change=gene_change,
            gene_confidence=gene_confidence,
            clinical_delta_value=clinical_delta_value,
            clinical_delta_valid=clinical_delta_valid,
        )
        prompted_slots = (
            current["patient_slots"].float()
            + self.twin_config.adapter_scale
            * history["slot_history_prompt"].unsqueeze(1)
        )
        adapted_slots = self.identifier.slot_adapter(
            prompted_slots, history["slot_adapter_scale"]
        )
        slot_attention = torch.softmax(
            self.backbone.slot_score(
                self.backbone.slot_score_norm(adapted_slots)
            ).squeeze(-1),
            dim=1,
        )
        slot_state = torch.sum(slot_attention.unsqueeze(-1) * adapted_slots, dim=1)
        patient_state = self.backbone.patient_encoder(
            torch.cat(
                (
                    current["query_state"].float(),
                    slot_state,
                    current["source_geometry_state"].float(),
                ),
                dim=-1,
            )
        )
        patient_state = (
            patient_state
            + self.twin_config.adapter_scale * history["patient_history_prompt"]
        )
        patient_state = self.identifier.patient_adapter(
            patient_state, history["patient_adapter_scale"]
        )

        raw_direction = self.backbone.mean_direction_head(patient_state.float())
        raw_direction = raw_direction + self.identifier.mean_head_adapter(
            patient_state.float(), history["mean_head_adapter_scale"]
        )
        mean_direction = F.normalize(raw_direction, p=2.0, dim=-1, eps=1e-6)
        raw_magnitude = self.backbone.mean_magnitude_head(patient_state.float())
        raw_magnitude = raw_magnitude + self.identifier.magnitude_head_adapter(
            patient_state.float(), history["magnitude_head_adapter_scale"]
        )
        mean_magnitude = F.softplus(raw_magnitude)
        mean_coefficients = mean_direction * mean_magnitude
        covariance_coefficients = self.backbone.covariance_head(patient_state.float())
        covariance_coefficients = (
            covariance_coefficients
            + self.identifier.covariance_head_adapter(
                patient_state.float(), history["covariance_head_adapter_scale"]
            )
        )
        adjusted, transport = self.backbone.transport.apply_coefficients(
            source_latent, mean_coefficients, covariance_coefficients
        )
        source_mask = _cell_mask(
            source_latent,
            source_latent_mask,
            expected_dim=self.config.latent_dim,
            name="source_latent",
        )
        refined, fiber = self.backbone.fiber_operator(
            adjusted, adapted_slots, source_mask
        )
        recovery_basis = refined.new_zeros(refined.shape[0], self.config.latent_dim)
        recovery_correction = recovery_basis.clone()
        if self.twin_config.use_recovery_basis:
            support_phase_index = support_phase.long().clamp(
                min=0, max=self.config.phase_count - 1
            )
            observed_reference = self.support_response_reference_observed[
                support_phase_index
            ]
            if self.twin_config.recovery_same_phase:
                recovery_mask = (
                    support_mask
                    & support_phase_index.eq(phase.long().unsqueeze(1))
                    & observed_reference
                )
                requested = support_mask & support_phase_index.eq(
                    phase.long().unsqueeze(1)
                )
                if bool((requested & ~observed_reference).any()):
                    raise RuntimeError(
                        "same-phase recovery basis requires a fitted response center"
                    )
            else:
                recovery_mask = (
                    support_mask & support_phase_index.eq(0) & observed_reference
                )
                if bool((phase.eq(1) & support_mask.any(dim=1)).any()) and not bool(
                    self.support_response_reference_observed[0]
                ):
                    raise RuntimeError(
                        "T2 recovery basis requires a fitted acute-response center"
                    )
            centered_support = support_response_latent - self.support_response_reference[
                support_phase_index
            ].to(support_response_latent.dtype)
            weights = recovery_mask.to(centered_support.dtype)
            recovery_basis = (centered_support * weights.unsqueeze(-1)).sum(dim=1)
            recovery_basis = recovery_basis / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            eligible = (
                recovery_mask.any(dim=1)
                if self.twin_config.recovery_same_phase
                else phase.eq(1) & recovery_mask.any(dim=1)
            )
            recovery_correction = (
                history["recovery_coefficient"]
                * recovery_basis
                * eligible.unsqueeze(-1).to(recovery_basis.dtype)
            )
            refined = refined + recovery_correction.unsqueeze(1)
        has_history = support_mask.any(dim=1)
        return {
            **current,
            **transport,
            **fiber,
            **history,
            "recovery_basis": recovery_basis,
            "recovery_correction": recovery_correction,
            "predicted_cells": refined,
            "shape_delta": refined
            - source_latent.float()
            - transport["mean_shift"].float().unsqueeze(1),
            "patient_slots": adapted_slots,
            "patient_slot_attention": slot_attention,
            "patient_state": patient_state,
            "mean_program_direction": mean_direction,
            "mean_program_magnitude": mean_magnitude,
            "mean_program_coefficients": mean_coefficients,
            "covariance_program_coefficients": covariance_coefficients,
            "current_response": current_response,
            "w": history["history_context"],
            "has_history": has_history,
        }

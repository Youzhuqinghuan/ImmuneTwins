from __future__ import annotations
import torch
from torch import nn
from model.encoder import VaccineImmuneTwinConfig

PROGRAM_TRANSPORT_OPERATORS = {
    "program_affine",
    "program_query_affine",
    "set_affine",
}

def _batch_covariance(centered: torch.Tensor) -> torch.Tensor:
    denominator = max(centered.shape[1] - 1, 1)
    return torch.einsum("bni,bnj->bij", centered, centered) / float(denominator)

def _symmetric_power(
    matrix: torch.Tensor,
    power: float,
    *,
    eigenvalue_minimum: torch.Tensor | float,
    eigenvalue_maximum: torch.Tensor | float | None = None,
) -> torch.Tensor:
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    minimum = torch.as_tensor(
        eigenvalue_minimum, device=matrix.device, dtype=matrix.dtype
    )
    eigenvalues = torch.maximum(eigenvalues, minimum)
    if eigenvalue_maximum is not None:
        maximum = torch.as_tensor(
            eigenvalue_maximum, device=matrix.device, dtype=matrix.dtype
        )
        eigenvalues = torch.minimum(eigenvalues, maximum)
    powered = torch.pow(eigenvalues, power)
    return (eigenvectors * powered.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)

class ProgramAffineTransport(nn.Module):
    """Predict donor mean and covariance changes in fitted response programs."""

    def __init__(self, config: VaccineImmuneTwinConfig) -> None:
        super().__init__()
        if config.current_operator not in PROGRAM_TRANSPORT_OPERATORS:
            raise ValueError("ProgramAffineTransport requires a program transport operator")
        self.config = config
        latent = config.latent_dim
        triangle = latent * (latent + 1) // 2
        self.register_buffer(
            "upper_indices", torch.triu_indices(latent, latent), persistent=False
        )
        self.register_buffer(
            "mean_program_basis",
            torch.zeros(config.mean_program_rank, latent),
            persistent=True,
        )
        self.register_buffer(
            "covariance_program_basis",
            torch.zeros(config.covariance_program_rank, triangle),
            persistent=True,
        )
        self.register_buffer("response_scale", torch.ones(latent), persistent=True)
        self.register_buffer(
            "relative_covariance_eigenvalue_minimum", torch.tensor(0.1), persistent=True
        )
        self.register_buffer(
            "relative_covariance_eigenvalue_maximum", torch.tensor(5.0), persistent=True
        )
        self.register_buffer("programs_loaded", torch.tensor(False), persistent=True)

        source_summary_dim = 2 * latent + triangle
        self.source_geometry_encoder = nn.Sequential(
            nn.LayerNorm(source_summary_dim),
            nn.Linear(source_summary_dim, config.source_geometry_dim),
            nn.GELU(),
            nn.Linear(config.source_geometry_dim, config.source_geometry_dim),
            nn.GELU(),
        )
        patient_dim = config.visit_dim + config.source_geometry_dim
        if config.current_operator == "program_affine":
            self.patient_response_encoder: nn.Module | None = nn.Sequential(
                nn.LayerNorm(patient_dim),
                nn.Linear(patient_dim, config.transition_hidden_dim),
                nn.GELU(),
            )
            self.mean_coefficient_head: nn.Module | None = nn.Sequential(
                nn.LayerNorm(config.transition_hidden_dim),
                nn.Linear(
                    config.transition_hidden_dim,
                    config.transition_hidden_dim // 2,
                ),
                nn.GELU(),
                nn.Linear(
                    config.transition_hidden_dim // 2,
                    config.mean_program_rank,
                ),
            )
            self.covariance_coefficient_head: nn.Module | None = nn.Sequential(
                nn.LayerNorm(config.transition_hidden_dim),
                nn.Linear(
                    config.transition_hidden_dim,
                    config.transition_hidden_dim // 2,
                ),
                nn.GELU(),
                nn.Linear(
                    config.transition_hidden_dim // 2,
                    config.covariance_program_rank,
                ),
            )
        else:
            self.patient_response_encoder = None
            self.mean_coefficient_head = None
            self.covariance_coefficient_head = None

    @torch.no_grad()
    def set_response_transform(
        self, response_center: torch.Tensor, response_scale: torch.Tensor
    ) -> None:
        del response_center
        scale = torch.as_tensor(
            response_scale, device=self.response_scale.device, dtype=self.response_scale.dtype
        )
        if scale.shape != self.response_scale.shape or not torch.all(scale > 0):
            raise ValueError("response_scale is invalid for program-affine transport")
        self.response_scale.copy_(scale)

    @torch.no_grad()
    def set_response_programs(
        self,
        *,
        mean_program_basis: torch.Tensor,
        covariance_program_basis: torch.Tensor,
        relative_covariance_eigenvalue_minimum: float,
        relative_covariance_eigenvalue_maximum: float,
    ) -> None:
        mean_basis = torch.as_tensor(
            mean_program_basis,
            device=self.mean_program_basis.device,
            dtype=self.mean_program_basis.dtype,
        )
        covariance_basis = torch.as_tensor(
            covariance_program_basis,
            device=self.covariance_program_basis.device,
            dtype=self.covariance_program_basis.dtype,
        )
        if mean_basis.shape != self.mean_program_basis.shape:
            raise ValueError(
                "mean program basis shape differs: "
                f"{tuple(mean_basis.shape)} != {tuple(self.mean_program_basis.shape)}"
            )
        if covariance_basis.shape != self.covariance_program_basis.shape:
            raise ValueError(
                "covariance program basis shape differs: "
                f"{tuple(covariance_basis.shape)} != "
                f"{tuple(self.covariance_program_basis.shape)}"
            )
        minimum = float(relative_covariance_eigenvalue_minimum)
        maximum = float(relative_covariance_eigenvalue_maximum)
        if not 0.0 < minimum < maximum:
            raise ValueError("relative covariance eigenvalue bounds are invalid")
        if not torch.isfinite(mean_basis).all() or not torch.isfinite(
            covariance_basis
        ).all():
            raise ValueError("response program bases must be finite")
        self.mean_program_basis.copy_(mean_basis)
        self.covariance_program_basis.copy_(covariance_basis)
        self.relative_covariance_eigenvalue_minimum.fill_(minimum)
        self.relative_covariance_eigenvalue_maximum.fill_(maximum)
        self.programs_loaded.fill_(True)

    def _symmetric_from_upper(self, upper: torch.Tensor) -> torch.Tensor:
        batch = upper.shape[0]
        latent = self.config.latent_dim
        result = torch.zeros(
            batch, latent, latent, device=upper.device, dtype=upper.dtype
        )
        row, column = self.upper_indices
        result[:, row, column] = upper
        result[:, column, row] = upper
        return result

    def _source_summary(self, source: torch.Tensor) -> torch.Tensor:
        source_mean = source.mean(dim=1)
        centered = source - source_mean.unsqueeze(1)
        source_covariance = _batch_covariance(centered)
        diagonal = torch.diagonal(source_covariance, dim1=-2, dim2=-1)
        log_standard_deviation = 0.5 * torch.log(diagonal.clamp_min(1e-6))
        row, column = self.upper_indices
        covariance_upper = source_covariance[:, row, column]
        return torch.cat(
            (source_mean, log_standard_deviation, covariance_upper), dim=-1
        )

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def encode_source_geometry(self, source_cells: torch.Tensor) -> torch.Tensor:
        if source_cells.ndim != 3 or source_cells.shape[-1] != self.config.latent_dim:
            raise ValueError("source_cells must be [batch,cells,latent_dim]")
        return self.source_geometry_encoder(self._source_summary(source_cells.float()))

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def apply_coefficients(
        self,
        source_cells: torch.Tensor,
        mean_coefficients: torch.Tensor,
        covariance_coefficients: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not bool(self.programs_loaded.item()):
            raise RuntimeError("fold-training response programs were not loaded")
        if source_cells.ndim != 3 or source_cells.shape[-1] != self.config.latent_dim:
            raise ValueError("source_cells must be [batch,cells,latent_dim]")
        batch = source_cells.shape[0]
        if mean_coefficients.shape != (batch, self.config.mean_program_rank):
            raise ValueError("mean response-program coefficient shape changed")
        if covariance_coefficients.shape != (
            batch,
            self.config.covariance_program_rank,
        ):
            raise ValueError("covariance response-program coefficient shape changed")

        output_dtype = source_cells.dtype
        source = source_cells.float()
        source_mean = source.mean(dim=1)
        centered = source - source_mean.unsqueeze(1)
        source_covariance = _batch_covariance(centered)
        mean_shift = mean_coefficients @ self.mean_program_basis.float()
        covariance_delta_upper = (
            covariance_coefficients @ self.covariance_program_basis.float()
        )
        covariance_delta = self._symmetric_from_upper(covariance_delta_upper)

        latent = self.config.latent_dim
        identity = torch.eye(latent, device=source.device, dtype=source.dtype).unsqueeze(0)
        source_floor = (
            torch.diagonal(source_covariance, dim1=-2, dim2=-1).mean(dim=-1)
            .clamp_min(1e-4)
            .view(-1, 1, 1)
            * 1e-4
        )
        regularized_source = source_covariance + source_floor * identity
        source_sqrt = _symmetric_power(
            regularized_source,
            0.5,
            eigenvalue_minimum=source_floor.squeeze(-1),
        )
        source_inverse_sqrt = _symmetric_power(
            regularized_source,
            -0.5,
            eigenvalue_minimum=source_floor.squeeze(-1),
        )
        proposed_target = regularized_source + covariance_delta
        relative_target = (
            source_inverse_sqrt @ proposed_target @ source_inverse_sqrt
        )
        relative_sqrt = _symmetric_power(
            relative_target,
            0.5,
            eigenvalue_minimum=self.relative_covariance_eigenvalue_minimum.float(),
            eigenvalue_maximum=self.relative_covariance_eigenvalue_maximum.float(),
        )
        affine = source_sqrt @ relative_sqrt @ source_inverse_sqrt
        transformed_centered = centered @ affine.transpose(-1, -2)
        transformed_centered = transformed_centered - transformed_centered.mean(
            dim=1, keepdim=True
        )
        predicted = (
            source_mean.unsqueeze(1)
            + mean_shift.unsqueeze(1)
            + transformed_centered
        )
        shape_delta = transformed_centered - centered
        standardized_mean_shift = mean_shift / self.response_scale.float()
        return predicted.to(output_dtype), {
            "mean_shift": mean_shift.to(output_dtype),
            "standardized_mean_shift": standardized_mean_shift.to(output_dtype),
            "current_response": mean_shift.to(output_dtype),
            "current_standardized_mean_shift": standardized_mean_shift.to(output_dtype),
            "shape_delta": shape_delta.to(output_dtype),
            "mean_program_coefficients": mean_coefficients.to(output_dtype),
            "covariance_program_coefficients": covariance_coefficients.to(output_dtype),
        }

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        self,
        source_cells: torch.Tensor,
        query_state: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.config.current_operator != "program_affine":
            raise ValueError("direct program-affine forward is disabled for query pooling")
        if query_state.shape != (source_cells.shape[0], self.config.visit_dim):
            raise ValueError("query_state shape differs from the configured patient encoder")
        if (
            self.patient_response_encoder is None
            or self.mean_coefficient_head is None
            or self.covariance_coefficient_head is None
        ):
            raise RuntimeError("direct patient coefficient heads were not constructed")
        geometry_state = self.encode_source_geometry(source_cells)
        patient_state = self.patient_response_encoder(
            torch.cat((query_state.float(), geometry_state), dim=-1)
        )
        mean_coefficients = self.mean_coefficient_head(patient_state)
        covariance_coefficients = self.covariance_coefficient_head(patient_state)
        predicted, state = self.apply_coefficients(
            source_cells,
            mean_coefficients,
            covariance_coefficients,
        )
        state["source_geometry_state"] = geometry_state.to(predicted.dtype)
        return predicted, state

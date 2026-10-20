from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Sequence
import torch
from torch.nn import functional as F

@dataclass(frozen=True)
class PatientLossV2Config:
    distribution_weight: float = 1.0
    response_mean_weight: float = 2.0
    response_direction_weight: float = 0.5
    response_norm_weight: float = 0.5
    centered_geometry_weight: float = 0.5
    geometry_gram_fraction: float = 0.25
    history_sensitivity_weight: float = 0.0
    history_sensitivity_temperature: float = 1.0
    history_current_anchor_weight: float = 0.0
    history_current_anchor_temperature: float = 1.0
    geometry_scale_floor: float = 1e-3
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "distribution_weight",
            "response_mean_weight",
            "response_direction_weight",
            "response_norm_weight",
            "centered_geometry_weight",
            "history_sensitivity_weight",
            "history_current_anchor_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0 <= self.geometry_gram_fraction <= 1:
            raise ValueError("geometry_gram_fraction must be in [0, 1]")
        if (
            not math.isfinite(self.history_sensitivity_temperature)
            or self.history_sensitivity_temperature <= 0
        ):
            raise ValueError(
                "history_sensitivity_temperature must be finite and positive"
            )
        if (
            not math.isfinite(self.history_current_anchor_temperature)
            or self.history_current_anchor_temperature <= 0
        ):
            raise ValueError(
                "history_current_anchor_temperature must be finite and positive"
            )
        if self.geometry_scale_floor <= 0 or self.epsilon <= 0:
            raise ValueError("numeric stability constants must be positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

def _donors(donor_ids: Sequence[object], units: int) -> tuple[str, ...]:
    donors = tuple(str(value) for value in donor_ids)
    if len(donors) != units:
        raise ValueError(f"donor_ids has {len(donors)} entries for {units} units")
    if not donors:
        raise ValueError("at least one donor unit is required")
    return donors

def _donor_order(donor_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(donor_ids)))

def donor_macro_mean(values: torch.Tensor, donor_ids: Sequence[object]) -> torch.Tensor:
    """Mean seasons within donor, then give every donor equal weight."""

    if values.ndim < 1:
        raise ValueError("values must have a leading unit dimension")
    donors = _donors(donor_ids, values.shape[0])
    donor_values = []
    for donor in _donor_order(donors):
        positions = [index for index, value in enumerate(donors) if value == donor]
        donor_values.append(values[positions].mean(dim=0))
    return torch.stack(donor_values, dim=0).mean(dim=0)

def donor_mean_vectors(
    values: torch.Tensor, donor_ids: Sequence[object]
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Return one season-averaged vector per donor in deterministic order."""

    if values.ndim != 2:
        raise ValueError("response values must be [units, latent]")
    donors = _donors(donor_ids, values.shape[0])
    order = _donor_order(donors)
    result = []
    for donor in order:
        positions = [index for index, value in enumerate(donors) if value == donor]
        result.append(values[positions].mean(dim=0))
    return torch.stack(result, dim=0), order

def _sets(
    predicted: torch.Tensor, source: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if predicted.ndim != 3 or source.ndim != 3 or target.ndim != 3:
        raise ValueError("cell distributions must be [units, cells, latent]")
    if predicted.shape != source.shape:
        raise ValueError("predicted and source distributions must share shape")
    if predicted.shape[0] != target.shape[0] or predicted.shape[-1] != target.shape[-1]:
        raise ValueError("target unit/latent dimensions changed")
    if predicted.shape[1] < 1 or target.shape[1] < 1:
        raise ValueError("every episode distribution needs at least one cell")
    # Losses are computed in float32 under BF16 autocast.
    return predicted.float(), source.float(), target.float()

def rbf_mmd2_per_unit(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    bandwidth: float,
) -> torch.Tensor:
    """Biased non-negative RBF MMD squared for each donor-season unit."""

    if predicted.ndim != 3 or target.ndim != 3:
        raise ValueError("MMD inputs must be [units, cells, latent]")
    if predicted.shape[0] != target.shape[0] or predicted.shape[-1] != target.shape[-1]:
        raise ValueError("MMD unit/latent dimensions changed")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    predicted, target = predicted.float(), target.float()
    denominator = 2.0 * float(bandwidth) ** 2
    pp = torch.exp(-torch.cdist(predicted, predicted).square() / denominator).mean(
        dim=(1, 2)
    )
    tt = torch.exp(-torch.cdist(target, target).square() / denominator).mean(
        dim=(1, 2)
    )
    pt = torch.exp(-torch.cdist(predicted, target).square() / denominator).mean(
        dim=(1, 2)
    )
    return (pp + tt - 2.0 * pt).clamp_min(0.0)

def response_vectors(
    predicted: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted, source, target = _sets(predicted, source, target)
    source_mean = source.mean(dim=1)
    return predicted.mean(dim=1) - source_mean, target.mean(dim=1) - source_mean

def response_direction_per_unit(
    predicted_response: torch.Tensor,
    observed_response: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """One-minus-cosine, with zero loss where direction is undefined."""

    cosine = F.cosine_similarity(
        predicted_response, observed_response, dim=-1, eps=epsilon
    )
    valid = observed_response.norm(dim=-1) > epsilon
    return torch.where(valid, 1.0 - cosine, torch.zeros_like(cosine))

def response_norm_calibration_per_unit(
    predicted_response: torch.Tensor,
    observed_response: torch.Tensor,
) -> torch.Tensor:
    """Huber calibration of response amplitude on a stable log1p scale."""

    predicted_norm = predicted_response.norm(dim=-1)
    observed_norm = observed_response.norm(dim=-1)
    return F.smooth_l1_loss(
        torch.log1p(predicted_norm),
        torch.log1p(observed_norm),
        reduction="none",
    )

def centered_response_geometry_loss(
    predicted_response: torch.Tensor,
    observed_response: torch.Tensor,
    donor_ids: Sequence[object],
    *,
    gram_fraction: float = 0.25,
    scale_floor: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Match centred donor coordinates and their pairwise inner products.

    Responses from repeated seasons are averaged before centring.  Simultaneous
    permutation of donors leaves every returned scalar unchanged.
    """

    if not 0 <= gram_fraction <= 1:
        raise ValueError("gram_fraction must be in [0, 1]")
    predicted_donor, predicted_order = donor_mean_vectors(
        predicted_response, donor_ids
    )
    observed_donor, observed_order = donor_mean_vectors(observed_response, donor_ids)
    if predicted_order != observed_order:
        raise RuntimeError("predicted and observed donor orders differ")
    if predicted_donor.shape[0] < 2:
        zero = predicted_donor.sum() * 0.0
        return {
            "loss": zero,
            "centered_alignment": zero,
            "gram": zero,
            "predicted_donor_response": predicted_donor,
            "observed_donor_response": observed_donor,
        }
    predicted_centered = predicted_donor - predicted_donor.mean(dim=0, keepdim=True)
    observed_centered = observed_donor - observed_donor.mean(dim=0, keepdim=True)
    scale = observed_centered.square().mean().sqrt().detach().clamp_min(scale_floor)
    predicted_scaled = predicted_centered / scale
    observed_scaled = observed_centered / scale
    alignment = F.smooth_l1_loss(predicted_scaled, observed_scaled)
    latent = predicted_scaled.shape[-1]
    predicted_gram = predicted_scaled @ predicted_scaled.T / float(latent)
    observed_gram = observed_scaled @ observed_scaled.T / float(latent)
    gram = F.smooth_l1_loss(predicted_gram, observed_gram)
    loss = (1.0 - gram_fraction) * alignment + gram_fraction * gram
    return {
        "loss": loss,
        "centered_alignment": alignment,
        "gram": gram,
        "predicted_donor_response": predicted_donor,
        "observed_donor_response": observed_donor,
    }

def history_shuffle_sensitivity_loss(
    predicted_response: torch.Tensor,
    shuffled_history_response: torch.Tensor,
    observed_response: torch.Tensor,
    donor_ids: Sequence[object],
    *,
    history_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Smooth pairwise ranking against mismatched patient history.

    The softplus comparison has a non-zero derivative at equal errors, so a
    zero-initialized adapter receives a learning signal.
    """

    if predicted_response.shape != shuffled_history_response.shape:
        raise ValueError("true and shuffled history responses must share shape")
    if predicted_response.shape != observed_response.shape or predicted_response.ndim != 2:
        raise ValueError("history responses must share [units, latent]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("history sensitivity temperature must be finite and positive")
    donors = _donors(donor_ids, predicted_response.shape[0])
    if history_mask is None:
        mask = torch.ones(
            predicted_response.shape[0],
            dtype=torch.bool,
            device=predicted_response.device,
        )
    else:
        if history_mask.shape != (predicted_response.shape[0],):
            raise ValueError("history_mask must be [units]")
        mask = history_mask.to(device=predicted_response.device, dtype=torch.bool)
    if not mask.any():
        zero = predicted_response.sum() * 0.0
        return {"loss": zero, "history_gain": zero, "history_effect_size": zero}
    true_error = (predicted_response - observed_response).square().mean(dim=-1)
    shuffled_error = (shuffled_history_response - observed_response).square().mean(
        dim=-1
    )
    ranking = F.softplus((true_error - shuffled_error) / float(temperature))
    gain = shuffled_error - true_error
    effect = (predicted_response - shuffled_history_response).norm(dim=-1)
    selected_donors = [donor for donor, keep in zip(donors, mask.tolist()) if keep]
    return {
        "loss": donor_macro_mean(ranking[mask], selected_donors),
        "history_gain": donor_macro_mean(gain[mask], selected_donors),
        "history_effect_size": donor_macro_mean(effect[mask], selected_donors),
    }

def history_current_anchor_loss(
    predicted_response: torch.Tensor,
    current_response: torch.Tensor,
    observed_response: torch.Tensor,
    donor_ids: Sequence[object],
    *,
    response_scale: torch.Tensor,
    history_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Rank own-history response error against the current-state response.

    Errors are divided by outer-training response scales; ``current_response``
    is detached.
    """

    if predicted_response.shape != current_response.shape:
        raise ValueError("history and current responses must share shape")
    if predicted_response.shape != observed_response.shape or predicted_response.ndim != 2:
        raise ValueError("anchor responses must share [units, latent]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("history current-anchor temperature must be finite and positive")
    donors = _donors(donor_ids, predicted_response.shape[0])
    scale = torch.as_tensor(
        response_scale,
        device=predicted_response.device,
        dtype=predicted_response.dtype,
    )
    if scale.shape != (predicted_response.shape[-1],):
        raise ValueError("history current-anchor response_scale must be [latent]")
    if not torch.isfinite(scale).all() or not torch.all(scale > 0):
        raise ValueError(
            "history current-anchor response_scale must be finite and positive"
        )
    if history_mask is None:
        mask = torch.ones(
            predicted_response.shape[0],
            dtype=torch.bool,
            device=predicted_response.device,
        )
    else:
        if history_mask.shape != (predicted_response.shape[0],):
            raise ValueError("history current-anchor mask must be [units]")
        mask = history_mask.to(device=predicted_response.device, dtype=torch.bool)
    if not mask.any():
        zero = predicted_response.sum() * 0.0
        return {"loss": zero, "history_vs_current_gain": zero}

    observed = observed_response.to(predicted_response.dtype)
    current = current_response.detach().to(predicted_response.dtype)
    own_error = ((predicted_response - observed) / scale).square().mean(dim=-1)
    current_error = ((current - observed) / scale).square().mean(dim=-1)
    ranking = F.softplus((own_error - current_error) / float(temperature))
    gain = current_error - own_error
    selected_donors = [donor for donor, keep in zip(donors, mask.tolist()) if keep]
    return {
        "loss": donor_macro_mean(ranking[mask], selected_donors),
        "history_vs_current_gain": donor_macro_mean(gain[mask], selected_donors),
    }

def patient_vaccine_objective_v2(
    predicted: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    donor_ids: Sequence[object],
    bandwidth: float,
    config: PatientLossV2Config | None = None,
    response_scale: torch.Tensor | None = None,
    shuffled_history_predicted: torch.Tensor | None = None,
    history_mask: torch.Tensor | None = None,
    current_response: torch.Tensor | None = None,
    history_current_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return the participant-averaged objective and its components.

    When ``response_scale`` is supplied, only the response-residual MSE uses
    the standardized coordinate.
    """

    config = config or PatientLossV2Config()
    predicted, source, target = _sets(predicted, source, target)
    donors = _donors(donor_ids, predicted.shape[0])
    predicted_response, observed_response = response_vectors(predicted, source, target)

    distribution_per_unit = rbf_mmd2_per_unit(
        predicted, target, bandwidth=bandwidth
    )
    response_residual = predicted_response - observed_response
    if response_scale is not None:
        scale = torch.as_tensor(
            response_scale,
            device=response_residual.device,
            dtype=response_residual.dtype,
        )
        expected = (response_residual.shape[-1],)
        if scale.shape != expected:
            raise ValueError(
                f"response_scale must have shape {expected}, got {tuple(scale.shape)}"
            )
        if not torch.isfinite(scale).all():
            raise ValueError("response_scale must contain only finite values")
        if not torch.all(scale > 0):
            raise ValueError("response_scale must be strictly positive")
        response_residual = response_residual / scale
    response_mean_per_unit = response_residual.square().mean(dim=-1)
    direction_per_unit = response_direction_per_unit(
        predicted_response, observed_response, epsilon=config.epsilon
    )
    norm_per_unit = response_norm_calibration_per_unit(
        predicted_response, observed_response
    )
    distribution = donor_macro_mean(distribution_per_unit, donors)
    response_mean = donor_macro_mean(response_mean_per_unit, donors)
    direction = donor_macro_mean(direction_per_unit, donors)
    norm = donor_macro_mean(norm_per_unit, donors)
    predicted_norm = donor_macro_mean(predicted_response.norm(dim=-1), donors)
    observed_norm = donor_macro_mean(observed_response.norm(dim=-1), donors)
    norm_ratio = (predicted_norm + config.epsilon) / (
        observed_norm + config.epsilon
    )
    geometry = centered_response_geometry_loss(
        predicted_response,
        observed_response,
        donors,
        gram_fraction=config.geometry_gram_fraction,
        scale_floor=config.geometry_scale_floor,
    )

    zero = predicted_response.sum() * 0.0
    history = {"loss": zero, "history_gain": zero, "history_effect_size": zero}
    if shuffled_history_predicted is not None:
        shuffled, _, _ = _sets(shuffled_history_predicted, source, target)
        shuffled_response = shuffled.mean(dim=1) - source.mean(dim=1)
        history = history_shuffle_sensitivity_loss(
            predicted_response,
            shuffled_response,
            observed_response,
            donors,
            history_mask=history_mask,
            temperature=config.history_sensitivity_temperature,
        )
    elif config.history_sensitivity_weight > 0:
        raise ValueError(
            "history_sensitivity_weight is positive but shuffled history predictions are absent"
        )
    elif history_mask is not None and history_mask.any():
        raise ValueError("history_mask was supplied without shuffled history predictions")

    current_anchor = {"loss": zero, "history_vs_current_gain": zero}
    if current_response is not None:
        if response_scale is None:
            raise ValueError(
                "history current-anchor ranking requires outer-training response_scale"
            )
        current_anchor = history_current_anchor_loss(
            predicted_response,
            current_response,
            observed_response,
            donors,
            response_scale=response_scale,
            history_mask=history_current_mask,
            temperature=config.history_current_anchor_temperature,
        )
    elif config.history_current_anchor_weight > 0:
        raise ValueError(
            "history_current_anchor_weight is positive but current response is absent"
        )
    elif history_current_mask is not None and history_current_mask.any():
        raise ValueError("history_current_mask was supplied without current response")

    total = (
        config.distribution_weight * distribution
        + config.response_mean_weight * response_mean
        + config.response_direction_weight * direction
        + config.response_norm_weight * norm
        + config.centered_geometry_weight * geometry["loss"]
        + config.history_sensitivity_weight * history["loss"]
        + config.history_current_anchor_weight * current_anchor["loss"]
    )
    components = (
        distribution,
        response_mean,
        direction,
        norm,
        geometry["loss"],
        history["loss"],
        current_anchor["loss"],
        total,
    )
    if not all(torch.isfinite(value) for value in components):
        raise FloatingPointError("v2 patient objective produced a non-finite scalar")
    return {
        "loss": total,
        "distribution_mmd2": distribution,
        "response_mean_loss": response_mean,
        "response_direction_loss": direction,
        "response_norm_calibration_loss": norm,
        "predicted_response_norm": predicted_norm,
        "observed_response_norm": observed_norm,
        "response_norm_ratio": norm_ratio,
        "centered_geometry_loss": geometry["loss"],
        "centered_alignment_loss": geometry["centered_alignment"],
        "centered_gram_loss": geometry["gram"],
        "history_shuffle_sensitivity_loss": history["loss"],
        "history_shuffle_gain": history["history_gain"],
        "history_effect_size": history["history_effect_size"],
        "history_current_anchor_loss": current_anchor["loss"],
        "history_vs_current_gain": current_anchor["history_vs_current_gain"],
        "predicted_response": predicted_response,
        "observed_response": observed_response,
        "predicted_donor_response": geometry["predicted_donor_response"],
        "observed_donor_response": geometry["observed_donor_response"],
    }

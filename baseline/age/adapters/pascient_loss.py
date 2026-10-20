from __future__ import annotations
import torch
from torch import nn

LOSS_TARGET = "baseline.age.adapters.pascient_loss.PatientMSELossViews"

LOSS_CONTRACT = "paired_patient_mse_views_v1"

class PatientMSELossViews(nn.MSELoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3 or target.ndim != 2:
            raise ValueError("PaSCient regression requires prediction [N,V,D], target [N,D]")
        if input.shape[0] != target.shape[0] or input.shape[2] != target.shape[1]:
            raise ValueError("PaSCient prediction and target patients/endpoints do not match")
        if input.shape[1] < 1:
            raise ValueError("PaSCient requires at least one prediction view")
        return torch.stack([
            super(PatientMSELossViews, self).forward(input[:, view, :], target)
            for view in range(input.shape[1])
        ]).mean(dim=0)

def loss_contract() -> dict[str, object]:
    return {"contract": LOSS_CONTRACT, "loss_target": LOSS_TARGET,
            "preserved": False, "true_before_loss": "[N,D]",
            "true_inside_loss": "[N,D]", "prediction_per_view": "[N,D]",
            "broadcast_result": "none; identical shapes required",
            "view_aggregation": "arithmetic mean of per-view MSE"}

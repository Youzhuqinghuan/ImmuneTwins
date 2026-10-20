from __future__ import annotations
from paths import expand_paths
import json
import os
from pathlib import Path
import random
from typing import Any
import numpy as np
import pandas as pd
import torch
from model.encoder import load_pretrained_visit_encoder
from model.projected_set import MomentFiberCurrentModel
from model.patient import SetAffineCurrentModel
from data.clinical import OuterTrainingVisitPreprocessor, VisitFeatureTable, build_sound_visit_features

PAPER_METHODS: dict[str, str] = {
    "current": "ImmuneTwin-current",
    "w": "ImmuneTwin-w",
    "gene": "ImmuneTwin-gene",
    "full": "ImmuneTwin-full",
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)

def atomic_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)

def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)

def save_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def require_cuda() -> torch.device:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("explicit CUDA_VISIBLE_DEVICES is required")
    visible = [
        value.strip()
        for value in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if value.strip()
    ]
    if visible == ["0"]:
        raise RuntimeError("physical GPU0 is reserved and cannot run vaccine training")
    if len(visible) != 1:
        raise RuntimeError("exactly one nonzero physical GPU must be visible per job")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one CUDA device is required; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    probe = torch.ones((64, 64), device=device) @ torch.ones((64, 64), device=device)
    if not torch.isfinite(probe).all() or float(probe.sum()) <= 0:
        raise RuntimeError("real CUDA tensor-operation probe failed")
    return device

def _state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}

def build_model(*, arm, config, cima_checkpoint,  gene_selection, seed):
    if config.current_operator != "set_affine" or arm != "current":
        raise ValueError("Use the ProjectedSet and digital-twin stage entry points for those stages")
    if gene_selection is not None:
        raise ValueError("set_affine current model cannot consume history genes")
    set_seed(seed)
    model = SetAffineCurrentModel(config)
    checkpoint = torch.load(cima_checkpoint, map_location="cpu", weights_only=False)
    loaded = load_pretrained_visit_encoder(model.visit_encoder, checkpoint)
    if any("age_head" in name for name in loaded):
        raise RuntimeError("age head was incorrectly loaded into vaccine model")
    return model, loaded

def set_encoder_frozen(
    model: torch.nn.Module
    | torch.nn.Module
    | torch.nn.Module
    | torch.nn.Module
    | SetAffineCurrentModel
    | MomentFiberCurrentModel,
    frozen: bool,
) -> None:
    for parameter in model.visit_encoder.parameters():
        parameter.requires_grad_(not frozen)

def _validate_manifest_audit(manifest: Path, audit_path: Path | None) -> dict[str, Any] | None:
    if audit_path is None:
        return None
    audit = expand_paths(json.loads(audit_path.read_text()))
    entry = audit.get("outputs", {}).get("unified_manifest", {})
    if (
        audit.get('status') != 'pass'
        or audit.get('schema_version') != 'sound_immunetwin_visit_manifest_1'
        or entry.get('path') != str(manifest.resolve())
    ):
        raise ValueError("unified visit manifest audit is not passing")
    return audit

def prepare_clinical_history(
    *,
    labs_path: Path,
    hai_path: Path,
    igg_path: Path,
    contract_path: Path,
    outer_training_donors: set[str],
    outer_test_donors: set[str],
    required_support_samples: set[str],
    output: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit visit preprocessing on outer-training donors and transform visits."""

    if outer_training_donors & outer_test_donors:
        raise ValueError("clinical preprocessor donor sets overlap")
    table = build_sound_visit_features(
        labs_path,
        hai_path,
        igg_path,
        contract_path=contract_path,
    )
    missing_raw = sorted(required_support_samples - set(table.metadata.index.astype(str)))
    if missing_raw:
        raise ValueError(f"clinical table misses required history samples: {missing_raw[:5]}")
    support_order = sorted(required_support_samples)
    table = VisitFeatureTable(
        metadata=table.metadata.loc[support_order].copy(),
        values=table.values.loc[support_order].copy(),
        missing=table.missing.loc[support_order].copy(),
    )
    processor = OuterTrainingVisitPreprocessor().fit(
        table, outer_training_donors=outer_training_donors
    )
    if processor.outer_training_donors_ != frozenset(outer_training_donors):
        raise RuntimeError("clinical preprocessor fit donor set changed")
    transformed = processor.transform(table)
    assert processor.feature_names_ is not None
    assert processor.medians_ is not None
    assert processor.means_ is not None
    assert processor.scales_ is not None
    artifact: dict[str, Any] = {
        "status": "pass",
        "fit_scope": "outer_training_donors_only",
        "outer_training_donors": sorted(outer_training_donors),
        "outer_test_donors_excluded": sorted(outer_test_donors),
        "n_fit_visits": processor.n_training_visits_,
        "raw_feature_names": list(processor.feature_names_),
        "transformed_feature_names": list(transformed.columns),
        "clinical_dim": int(transformed.shape[1]),
        "median": {key: float(value) for key, value in processor.medians_.items()},
        "mean": {key: float(value) for key, value in processor.means_.items()},
        "scale": {key: float(value) for key, value in processor.scales_.items()},
        "test_future_rows_used_for_fit": False,
        "forecast_target_rows_used_for_fit": False,
        "preprocessing_rows": "completed support-transition endpoints only",
        "current_query_clinical_used": False,
        "input_role": "target-minus-source change on completed support transitions only",
        "missingness_role": "missing-mask target-minus-source change included",
    }
    atomic_json(output / "clinical_preprocessor.json", artifact)
    return transformed, artifact

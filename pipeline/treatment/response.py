from __future__ import annotations
from paths import expand_paths
import json
import resource
import time
from pathlib import Path
from typing import Any, Sequence
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from data.myeloma import RESPONSE, _atomic_json, _decode, _require_cuda
from model.treatment import SharedVaccineLogicConfig, build_shared_meta_adapter, build_shared_projected_set
from pipeline.treatment.state import EpisodeArrays, _batch, _build_set, _fit_stage, _forward, _training_response_scale, estimate_bandwidth, fit_response_programs, load_config, set_response_programs, validate_shared_architecture

PROGRAMS = {
    "proteasome": ["PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7", "PSMB1", "PSMB2", "PSMB3", "PSMB4", "PSMB5", "PSMB6", "PSMB7"],
    "ER_protein_folding": ["XBP1", "HSPA5", "HSP90B1", "CALR", "CANX", "PDIA3", "PDIA4", "PDIA6", "DNAJB9", "HERPUD1", "DDIT3", "ATF4"],
    "plasma_identity": ["SDC1", "TNFRSF17", "SLAMF7", "MZB1", "JCHAIN", "DERL3", "IRF4", "PRDM1"],
    "cell_cycle": ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "TYMS", "STMN1"],
    "interferon_response": ["IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "OAS1", "OAS2", "IRF7"],
    "antigen_presentation": ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "HLA-DRA", "HLA-DRB1", "CD74"],
    "NFkB_survival": ["NFKB1", "NFKB2", "RELA", "RELB", "NFKBIA", "TNFAIP3", "TRAF2", "BIRC3"],
    "anti_apoptosis": ["BCL2", "MCL1", "BCL2L1", "BCL2A1", "BIRC2", "XIAP"],
}

def normalize_counts(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    totals = np.asarray(counts.sum(axis=1)).ravel()
    x = counts.multiply((1e4 / np.maximum(totals, 1))[:, None]).tocsr()
    x.data = np.log1p(x.data).astype(np.float32)
    return x

class ResponseCohort:
    """KYDAR input adapter."""

    def __init__(self, root: Path):
        self.root = root
        self.visits = pd.read_csv(root / "kydar_visits.tsv", sep="\t")
        self.baseline = self.visits.query("time_point == 'Baseline'").sort_values("patient_id")
        self.patients = self.baseline.patient_id.astype(str).to_numpy()
        self.labels = np.asarray([RESPONSE[p] for p in self.patients], dtype=np.int64)
        with h5py.File(root / "kydar_raw_counts.h5") as h:
            self.genes = _decode(h["var/gene_symbol"][:])
            self.donors = _decode(h["obs/donor_id"][:])
            m = h["matrix"]
            counts = sparse.csr_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m.attrs["shape"]))
        self.expression = normalize_counts(counts)
        with h5py.File(root / "kydar_uce_visits.h5") as h:
            self.uce = dict(zip(_decode(h["visit_id"][:]), h["embedding"][:]))
        self.rows = {v: np.flatnonzero(self.donors == v) for v in self.visits.visit_id}
        self.visit_map = {(r.patient_id, r.time_point): r.visit_id for r in self.visits.itertuples()}
        if any(len(self.rows[v]) != 64 or self.uce[v].shape != (64, 1280) for v in self.rows):
            raise ValueError("Clinical baselines and ImmuneTwins must use the same fixed 64 cells")

    def fit_latent(self, patients: Sequence[str], n_genes: int, device: torch.device, seed: int) -> dict[str, Any]:
        # Fit only baseline/Cycle-4 cells of fit patients; Cycle 10 is excluded.
        visits = self.visits.loc[self.visits.patient_id.isin(patients) & self.visits.time_point.isin(["Baseline", "Cycle 4"])]
        rows = np.concatenate([self.rows[v] for v in visits.visit_id])
        x = self.expression[rows]
        mean = np.asarray(x.mean(axis=0)).ravel()
        variance = np.asarray(x.power(2).mean(axis=0)).ravel() - mean ** 2
        top = np.argsort(-variance, kind="stable")[:n_genes]
        panel = np.flatnonzero(np.isin(self.genes, sorted({g for v in PROGRAMS.values() for g in v})))
        columns = np.unique(np.r_[top, panel])
        dense = torch.as_tensor(x[:, columns].toarray(), device=device)
        center = dense.mean(0)
        torch.manual_seed(seed)
        _, _, vectors = torch.pca_lowrank(dense - center, q=32, center=False, niter=4)
        components = vectors[:, :16].T
        scores = (dense - center) @ components.T
        scale = scores.std(0, unbiased=False).clamp_min(.05)
        return {"genes": self.genes[columns], "columns": columns, "center": center.cpu().numpy(),
                "components": components.cpu().numpy(), "scale": scale.cpu().numpy(),
                "fit_patients": sorted(patients), "fit_visits": visits.visit_id.tolist()}

    def latent(self, visit: str, transform: dict[str, Any]) -> np.ndarray:
        x = self.expression[self.rows[visit]][:, transform["columns"]].toarray()
        return (((x - transform["center"]) @ transform["components"].T) / transform["scale"]).astype(np.float32)

    def episodes(self, patients: Sequence[str], transform: dict[str, Any], *, supervised: bool) -> list[EpisodeArrays]:
        items = []
        for patient in patients:
            baseline = self.visit_map[patient, "Baseline"]
            future = self.visit_map.get((patient, "Cycle 4"))
            if supervised and future is None:
                continue
            source = self.latent(baseline, transform)
            target = self.latent(future, transform) if supervised else np.zeros_like(source)
            items.append(EpisodeArrays(
                patient, f"{patient}__Baseline_response", baseline,
                future if supervised else "UNOBSERVED_AT_QUERY", float("nan"), 1,
                self.uce[baseline], source, target,
                np.empty(0, np.int64), np.empty(0, np.float32),
                np.empty((0, 64, 1280), np.float32), np.empty((0, 64, 1280), np.float32),
                np.empty((0, 64, 16), np.float32), np.empty((0, 64, 16), np.float32), np.empty(0, bool),
            ))
        return items

class ClinicalReadout(nn.Module):
    """Task-specific linear output head."""

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor, state_weight: float):
        super().__init__()
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone())
        multiplier = torch.ones_like(mean)
        multiplier[128:] = state_weight
        self.register_buffer("multiplier", multiplier)
        self.linear = nn.Linear(len(mean), 1, device=mean.device)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear((values - self.mean) / self.scale * self.multiplier).squeeze(-1)

def response_features(output: dict[str, torch.Tensor], source: torch.Tensor,
                      pooling: dict[str, Any] | None = None) -> torch.Tensor:
    """Read out the cell set, optionally with fixed RBF pooling of higher moments.

    RBF anchors and bandwidth are fitted on baseline cells of the fit patients.
    """
    current = source.mean(1)
    delta = output["predicted_cells"].mean(1) - current
    parts = [output["query_state"], current, delta]
    if pooling is not None:
        if pooling["kind"] != "native_distribution_rbf32":
            raise ValueError("Unknown clinical set readout")
        anchors = torch.as_tensor(pooling["anchors"], device=source.device, dtype=source.dtype)
        bandwidth = float(pooling["bandwidth"])
        if anchors.shape != (32, source.shape[-1]) or not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("Invalid fit-patient RBF readout parameters")
        def embed(cells: torch.Tensor) -> torch.Tensor:
            return torch.exp(-torch.cdist(cells, anchors.expand(len(cells), -1, -1)).square()
                             / (2 * bandwidth ** 2)).mean(1)
        current_set = embed(source)
        parts.extend([current_set, embed(output["predicted_cells"]) - current_set])
    return torch.cat(parts, dim=-1)

def fit_response_pooling(items: Sequence[EpisodeArrays], seed: int) -> dict[str, Any]:
    """Deterministic distribution readout fitted within each split."""
    cells = np.concatenate([item.source_latent for item in items])
    if len(cells) < 32:
        raise ValueError("RBF clinical readout needs at least 32 fit cells")
    rng = np.random.default_rng(seed)
    anchors = cells[rng.choice(len(cells), 32, replace=False)]
    left, right = rng.integers(0, len(cells), size=(2, 4096))
    distance = np.linalg.norm(cells[left] - cells[right], axis=1)
    positive = distance[distance > 0]
    if not len(positive):
        raise ValueError("Degenerate fit-cell geometry")
    return {"kind": "native_distribution_rbf32", "anchors": anchors,
            "bandwidth": float(np.median(positive)),
            "fit_patients": sorted(item.patient_id for item in items),
            "fit_landmark": "Baseline", "outcomes_used": False}

def fit_readout(x: np.ndarray, y: np.ndarray, C: float, state_weight: float, device: torch.device,
                *, prior_correction: bool = False) -> ClinicalReadout:
    if device.type != "cuda":
        raise RuntimeError("Clinical model training requires CUDA")
    values = torch.as_tensor(x, device=device)
    labels = torch.as_tensor(y, dtype=torch.float32, device=device)
    model = ClinicalReadout(values.mean(0), values.std(0, unbiased=False).clamp_min(1e-6), state_weight)
    count = torch.bincount(labels.long(), minlength=2)
    if torch.any(count == 0):
        raise ValueError("Clinical fitting requires both response classes")
    weights = len(labels) / (2 * count[labels.long()])
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=250, tolerance_grad=1e-7, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = (nn.functional.binary_cross_entropy_with_logits(model(values), labels, reduction="none") * weights).sum()
        loss = loss + .5 / C * sum(p.square().sum() for p in model.linear.parameters())
        loss.backward()
        return loss

    optimizer.step(closure)
    if prior_correction:
        # Undo the class-balanced loss weights using fit labels, restoring the
        # fit-population prior.
        with torch.no_grad():
            model.linear.bias.add_(torch.log(count[1].float() / count[0]))
    return model.eval()

@torch.no_grad()
def extract(model: nn.Module, items: list[EpisodeArrays], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    batch = _batch(items, device)
    output = _forward(model, batch)
    return (response_features(output, batch["source_latent"], getattr(model, "_clinical_pooling", None)).cpu().numpy(),
            output["predicted_cells"].mean(1).cpu().numpy(),
            output["query_attention"].cpu().numpy())

def metrics(y: np.ndarray, probability: np.ndarray, score: np.ndarray | None = None) -> dict[str, float]:
    score = probability if score is None else score
    return {"AUROC": float(roc_auc_score(y, score)), "AUPRC": float(average_precision_score(y, score)),
            "Brier": float(brier_score_loss(y, probability))}

def clinical_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Keep baseline-only clinical components independent of longitudinal flags."""
    overrides = config["clinical_application"].get("shared_model_overrides", {})
    if not overrides:
        return config
    allowed = {"use_recovery_basis", "recovery_same_phase", "recovery_coefficient_initial",
               "use_visit_tokens", "meta_response_mean_loss_weight", "meta_response_direction_loss_weight"}
    if set(overrides) - allowed:
        raise ValueError("Clinical task overrides cannot change the shared backbone")
    model = {**config["model"], **overrides}
    if model["use_recovery_basis"] or model["recovery_same_phase"] or model.get("use_visit_tokens", False):
        raise ValueError("Baseline-only clinical queries cannot activate a history-recovery component")
    architecture = {**config["shared_architecture"]}
    architecture["task_components"] = [x for x in architecture.get("task_components", [])
                                        if x not in {"native_history_response_recovery", "native_history_visit_tokens"}]
    resolved = {**config, "model": model, "shared_architecture": architecture}
    validate_shared_architecture(resolved)
    return resolved

def fit_operator(cohort: ResponseCohort, fit: Sequence[str], validation: Sequence[str], config: dict[str, Any], device: torch.device,
                 seed: int, epochs: tuple[int, int] | None = None):
    if set(fit) & set(validation):
        raise ValueError("Patient leakage in operator fitting")
    app = config["clinical_application"]
    shared = SharedVaccineLogicConfig(**clinical_training_config(config)["model"])
    transform = cohort.fit_latent(fit, app["latent_genes"], device, seed)
    train = cohort.episodes(fit, transform, supervised=True)
    val = cohort.episodes(validation, transform, supervised=True) if len(validation) else None
    programs = fit_response_programs(train, shared)
    bandwidth = estimate_bandwidth(train, seed)
    scale = _training_response_scale(programs, shared)
    torch.manual_seed(seed)
    parent, audit = _build_set(shared, config["pretraining"], programs)
    state, set_epoch, set_curve = _fit_stage(parent, train, device,
        epochs=epochs[0] if epochs else app["set_max_epochs"], learning_rate=.001, weight_decay=.0001,
        bandwidth=bandwidth, response_scale=scale, seed=seed, validation_items=val,
        patience=app["set_patience"] if val else None, batch_size=8)
    projected = build_shared_projected_set(shared, state)
    set_response_programs(projected, programs)
    _, projected_epoch, projected_curve = _fit_stage(projected, train, device,
        epochs=epochs[1] if epochs else app["projected_max_epochs"], learning_rate=.0003, weight_decay=.0001,
        bandwidth=bandwidth, response_scale=scale, seed=seed + 1, validation_items=val,
        selection_metric="distribution_mmd2",
        patience=app["projected_patience"] if val else None, batch_size=8)
    model = build_shared_meta_adapter(shared, projected).to(device)
    set_response_programs(model, programs)
    model.requires_grad_(False).eval()
    if app.get("readout_features", "native_mean") == "native_distribution_rbf32":
        model._clinical_pooling = fit_response_pooling(cohort.episodes(fit, transform, supervised=False), seed)
    elif app.get("readout_features", "native_mean") != "native_mean":
        raise ValueError("Unsupported clinical readout features")
    record = {"fit_patients": sorted(fit), "validation_patients": sorted(validation),
              "state_supervision_patients": [x.patient_id for x in train], "set_epoch": set_epoch,
              "projected_epoch": projected_epoch, "set_curve": set_curve, "projected_curve": projected_curve,
              "pretraining": audit, "history_at_query": "empty", "post_treatment_query_input": False,
              "model_class": f"{type(model).__module__}.{type(model).__name__}",
              "projected_selection_metric": "distribution_mmd2",
              "readout_features": app.get("readout_features", "native_mean"),
              "pooling_fit_patients": getattr(model, "_clinical_pooling", {}).get("fit_patients", [])}
    return model, transform, programs, record

def train(config_path: Path, outer_fold: int | None = None, final: bool = False) -> dict[str, Any]:
    if final and outer_fold is not None:
        raise ValueError("A source-final fit cannot also select an outer fold")
    config = clinical_training_config(load_config(config_path))
    validate_shared_architecture(config)
    app = config["clinical_application"]
    output = Path(app["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    device = _require_cuda("cuda:0")
    cohort = ResponseCohort(Path(app["features_root"]))
    contract = {"schema": "native_ImmuneTwins_clinical_application_v1",
                "clinical_application": app, "architecture": config["model"], "pretraining": config["pretraining"],
                "inputs": {n: str(cohort.root / n) for n in ("kydar_raw_counts.h5", "kydar_uce_visits.h5", "kydar_visits.tsv")},
                "programs": PROGRAMS, "external_outcomes_used_for_selection": False}
    contract_path = output / "contract.json"
    if contract_path.exists():
        prior = expand_paths(json.loads(contract_path.read_text()))
        keys = ("schema", "clinical_application", "architecture", "programs", "external_outcomes_used_for_selection")
        if any(prior.get(key) != contract[key] for key in keys):
            raise ValueError("Clinical contract changed; do not overwrite completed runs")
        if any(prior.get("pretraining", {}).get(key) != value for key, value in config["pretraining"].items()):
            raise ValueError("Clinical pretraining configuration changed")
    _atomic_json(contract_path, contract)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(cohort.patients, cohort.labels))
    if final:
        splits = [(np.arange(len(cohort.patients)), np.empty(0, int))]
    for fold, (fit, test) in enumerate(splits):
        if outer_fold is not None and fold != outer_fold:
            continue
        root = output / ("source_final" if final else f"fold_{fold}")
        if (root / "run.json").exists():
            print(f"skip immutable completed {root}", flush=True)
            continue
        root.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        records, cached = [], []
        inner = StratifiedKFold(app["inner_folds"], shuffle=True, random_state=142 + fold)
        for k, (a, b) in enumerate(inner.split(fit, cohort.labels[fit])):
            print(f"fold={fold} final={final} inner={k} native operator start", flush=True)
            model, transform, _, record = fit_operator(cohort, cohort.patients[fit[a]], cohort.patients[fit[b]], config, device, 12345 + fold * 10 + k)
            x, _, _ = extract(model, cohort.episodes(cohort.patients[fit[a]], transform, supervised=False), device)
            z, _, _ = extract(model, cohort.episodes(cohort.patients[fit[b]], transform, supervised=False), device)
            records.append(record)
            cached.append((a, b, x, z))
            del model
            torch.cuda.empty_cache()
        candidates = []
        for weight in app["head_state_weight"]:
            for C in app["head_C"]:
                probabilities = np.full(len(fit), np.nan)
                for a, b, x, z in cached:
                    head = fit_readout(x, cohort.labels[fit[a]], C, weight, device,
                                       prior_correction=app.get("head_prior_correction", False))
                    with torch.no_grad():
                        probabilities[b] = head(torch.as_tensor(z, device=device)).sigmoid().cpu().numpy()
                score = metrics(cohort.labels[fit], probabilities)
                candidates.append({"C": C, "state_weight": weight, **score,
                                   "selection": score["AUROC"] - .2 * score["Brier"]})
        chosen = max(candidates, key=lambda r: r["selection"])
        epochs = (int(np.median([r["set_epoch"] for r in records])), int(np.median([r["projected_epoch"] for r in records])))
        print(f"fold={fold} chosen solely in inner patients: {chosen}; refit epochs={epochs}", flush=True)
        model, transform, programs, refit = fit_operator(cohort, cohort.patients[fit], [], config, device, 22345 + fold, epochs)
        x, _, _ = extract(model, cohort.episodes(cohort.patients[fit], transform, supervised=False), device)
        head = fit_readout(x, cohort.labels[fit], chosen["C"], chosen["state_weight"], device,
                           prior_correction=app.get("head_prior_correction", False))
        torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "head_state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                    "transform": transform, "programs": programs, "shared_config": config["model"],
                    "head_state_weight": chosen["state_weight"], "fit_patients": cohort.patients[fit].tolist(),
                    "clinical_pooling": getattr(model, "_clinical_pooling", None),
                    "pretraining": config["pretraining"]}, root / "checkpoint.pt")
        if len(test):
            query = cohort.episodes(cohort.patients[test], transform, supervised=False)
            z, future, attention = extract(model, query, device)
            with torch.no_grad():
                probability = head(torch.as_tensor(z, device=device)).sigmoid().cpu().numpy()
            pd.DataFrame({"patient_id": cohort.patients[test], "label": cohort.labels[test], "response_probability": probability,
                          "outer_fold": fold}).to_csv(root / "predictions.tsv", sep="\t", index=False)
            np.savez_compressed(root / "native_outputs.npz", patient_id=cohort.patients[test].astype(str), features=z,
                                predicted_latent_mean=future, attention=attention)
        _atomic_json(root / "run.json", {"status": "PASS", "fit_patients": cohort.patients[fit].tolist(),
            "test_patients": cohort.patients[test].tolist(), "inner": records, "head_candidates": candidates,
            "selected": chosen, "refit": refit, "elapsed_seconds": time.monotonic() - start,
            "peak_GPU_MiB": torch.cuda.max_memory_allocated(device) / 2**20,
            "process_peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
        print(f"completed {root}, elapsed={time.monotonic()-start:.1f}s", flush=True)
    return {"status": "PASS", "output": str(output)}

def score(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    app = config["clinical_application"]
    root = Path(app["output_root"])
    predictions = pd.concat([pd.read_csv(root / f"fold_{k}" / "predictions.tsv", sep="\t") for k in range(5)]).sort_values("patient_id")
    if predictions.patient_id.duplicated().any() or set(predictions.patient_id) != set(RESPONSE):
        raise ValueError("Incomplete or duplicated patient OOF predictions")
    predictions.to_csv(root / "predictions.tsv", sep="\t", index=False)
    frames = {"ImmuneTwins (shared model + response head)": predictions}
    frames.update({name: pd.read_csv(path, sep="\t").sort_values("patient_id") for name, path in app["baseline_predictions"].items()})
    rows = []
    for method, frame in frames.items():
        if not np.array_equal(frame.patient_id, predictions.patient_id) or not np.array_equal(frame.label, predictions.label):
            raise ValueError(f"Patient/endpoint mismatch: {method}")
        discrimination = frame.response_score.to_numpy() if "response_score" in frame else frame.response_probability.to_numpy()
        rows.append({"Method": method, "Patients": len(frame), **metrics(frame.label.to_numpy(), frame.response_probability.to_numpy(), discrimination)})
    response_table = pd.DataFrame(rows)
    response_table.to_csv(root / "method_by_metric.tsv", sep="\t", index=False)
    state_path = Path(app.get("state_results", root.parent / "closure_20260905_complete/state_method_by_metric.tsv"))
    state_table = pd.read_csv(state_path, sep="\t").assign(Task="future_immune_state", Dataset="GSE309595", Patients=20)
    state_table = state_table.loc[state_table.Method.isin(["ImmuneTwins (full)", "scGen", "CellOT", "trVAE"])]
    if state_path.parent.name == "closure_20260905_complete":
        state_table.loc[state_table.Method.eq("ImmuneTwins (full)"), "Method"] = "ImmuneTwins (shared backbone + historical recovery calibration)"
    combined = pd.concat([state_table, response_table.assign(Task="treatment_response", Dataset="GSE161195")], ignore_index=True)
    columns = ["Task", "Dataset", "Method", "Patients"] + [c for c in combined if c not in {"Task", "Dataset", "Method", "Patients"}]
    combined[columns].to_csv(root / "paper_method_by_metric.tsv", sep="\t", index=False)
    return {"output": str(root), "results": rows}

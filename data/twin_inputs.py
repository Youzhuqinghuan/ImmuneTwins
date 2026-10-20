from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from data.clinical import VisitFeatureTable
from data.sound import GeneSelection, GeneSummaryStore, VaccineUnit, _embedding_lookup

CURATED_CLINICAL_FEATURES = (
    "hai::H1N1::percent_inhibition",
    "hai::H3N2::percent_inhibition",
    "hai::Victoria::percent_inhibition",
    "hai::Yamagata::percent_inhibition",
    "igg::H1N1::concentration",
    "igg::H3N2::concentration",
    "igg::Victoria::concentration",
    "igg::Yamagata::concentration",
    "labs::bc.lymphocyte_count",
    "labs::bc.monocyte_count",
    "labs::bc.neutrophil_count",
    "labs::infl.hs_crp",
    "labs::infl.esr",
)

HAI_HISTORY_FEATURES = CURATED_CLINICAL_FEATURES[:4]

IGG_HISTORY_FEATURES = CURATED_CLINICAL_FEATURES[4:8]

CLINICAL_HISTORY_SUBSETS = {
    "curated": CURATED_CLINICAL_FEATURES,
    "hai": HAI_HISTORY_FEATURES,
    "igg": IGG_HISTORY_FEATURES,
    "hai_igg": HAI_HISTORY_FEATURES + IGG_HISTORY_FEATURES,
}

def clinical_history_table(
    table: VisitFeatureTable,
    subset: str = "curated",
) -> VisitFeatureTable:
    """Select a prespecified clinical-history input without changing visit rows."""

    if subset not in CLINICAL_HISTORY_SUBSETS:
        raise ValueError(
            f"unknown clinical history subset {subset!r}; "
            f"expected one of {sorted(CLINICAL_HISTORY_SUBSETS)}"
        )
    features = CLINICAL_HISTORY_SUBSETS[subset]
    missing = sorted(set(features) - set(table.values.columns))
    if missing:
        raise ValueError(f"Sound clinical table lacks {subset} features: {missing}")
    values = table.values.loc[:, features].copy()
    return VisitFeatureTable(
        metadata=table.metadata.copy(),
        values=values,
        missing=values.isna(),
    )

@dataclass(frozen=True)
class SupportGeneSelectionAudit:
    transitions: int
    donors: int
    supported_genes: int
    eligible_genes: int
    selected_genes: int
    minimum_mean_detection: float

def select_support_transition_genes(
    store: GeneSummaryStore,
    *,
    units: Sequence[VaccineUnit],
    embedding_path: str | Path,
    n_genes: int = 128,
    minimum_mean_detection: float = 0.05,
) -> tuple[GeneSelection, SupportGeneSelectionAudit]:
    """Select variable completed-response genes using inner-fit supports only."""

    if not units or n_genes < 2:
        raise ValueError("support gene selection needs units and at least two genes")
    transitions = {
        (
            unit.donor_id,
            transition.source_sample,
            transition.target_sample,
            int(transition.phase),
        )
        for unit in units
        for transition in unit.supports
    }
    if not transitions:
        raise ValueError("support gene selection cannot run on empty history")
    embedding_path = Path(embedding_path).resolve(strict=True)
    payload = torch.load(embedding_path, map_location="cpu", weights_only=False)
    lookup = _embedding_lookup(payload)
    supported = np.asarray(
        [
            index
            for index, gene in enumerate(store.genes)
            if str(gene).upper() in lookup
        ],
        dtype=np.int64,
    )
    sources = [value[1] for value in sorted(transitions)]
    targets = [value[2] for value in sorted(transitions)]
    source = store.logcpm(sources, supported)
    target = store.logcpm(targets, supported)
    delta = target.astype(np.float64) - source.astype(np.float64)
    detection = np.maximum(
        store.detection_fraction(sources, supported),
        store.detection_fraction(targets, supported),
    ).astype(np.float64)
    variance = delta.var(axis=0, ddof=0)
    mean_detection = detection.mean(axis=0)
    eligible = mean_detection >= float(minimum_mean_detection)
    if int(eligible.sum()) < n_genes:
        raise ValueError(
            f"only {int(eligible.sum())} support-response genes pass detection; "
            f"{n_genes} requested"
        )
    score = variance * np.sqrt(mean_detection)
    symbols = store.genes[supported]
    candidates = np.flatnonzero(eligible)
    rank = sorted(candidates.tolist(), key=lambda i: (-float(score[i]), str(symbols[i])))
    chosen_local = np.asarray(rank[:n_genes], dtype=np.int64)
    chosen_raw = supported[chosen_local]
    chosen_score = score[chosen_local].astype(np.float32)
    chosen_names = symbols[chosen_local].astype(str)
    raw_order = np.argsort(chosen_raw)
    chosen_raw = chosen_raw[raw_order]
    chosen_score = chosen_score[raw_order]
    chosen_names = chosen_names[raw_order]
    embeddings = torch.stack([lookup[name.upper()] for name in chosen_names]).float()
    selection = GeneSelection(
        names=tuple(chosen_names.tolist()),
        raw_indices=chosen_raw,
        variances=chosen_score,
        embeddings=embeddings,
    )
    audit = SupportGeneSelectionAudit(
        transitions=len(transitions),
        donors=len({value[0] for value in transitions}),
        supported_genes=len(supported),
        eligible_genes=int(eligible.sum()),
        selected_genes=n_genes,
        minimum_mean_detection=float(minimum_mean_detection),
    )
    return selection, audit

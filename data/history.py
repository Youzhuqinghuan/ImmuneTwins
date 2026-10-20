from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence
import pandas as pd
from data.sound import SupportTransition, VaccineUnit

NATURAL_VARIATION_PHASE = 2

HISTORY_POLICIES = (
    "formal",
    "complete_prior_rna",
    "complete_prior_flu",
    "complete_prior_natural",
)

@dataclass(frozen=True)
class PriorRNAHistoryAudit:
    policy: str
    units: int
    units_with_added_history: int
    added_transitions: int
    added_flu_transitions: int
    added_natural_variation_transitions: int

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "units": self.units,
            "units_with_added_history": self.units_with_added_history,
            "added_transitions": self.added_transitions,
            "added_flu_transitions": self.added_flu_transitions,
            "added_natural_variation_transitions": (
                self.added_natural_variation_transitions
            ),
        }

def _timeline(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {
        "subject.subjectGuid",
        "sample.sampleKitGuid",
        "sample.visitName",
        "sample.daysSinceFirstVisit",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Sound visit timeline lacks columns: {sorted(missing)}")
    frame = frame.loc[:, sorted(required)].copy()
    frame.columns = [
        {
            "subject.subjectGuid": "donor_id",
            "sample.sampleKitGuid": "sample_id",
            "sample.visitName": "visit_name",
            "sample.daysSinceFirstVisit": "absolute_day",
        }[column]
        for column in frame.columns
    ]
    frame["donor_id"] = frame["donor_id"].astype(str)
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["visit_name"] = frame["visit_name"].astype(str)
    frame["absolute_day"] = pd.to_numeric(frame["absolute_day"], errors="raise")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Sound visit timeline duplicates a physical sample")
    return frame

def _completed_transitions(frame: pd.DataFrame) -> list[tuple[SupportTransition, float, str]]:
    by_name = {
        str(row.visit_name): row
        for row in frame.sort_values(["absolute_day", "visit_name"], kind="stable").itertuples(
            index=False
        )
    }
    candidates: list[tuple[SupportTransition, float, str]] = []

    def append(source_name: str, target_name: str, phase: int, kind: str) -> None:
        source = by_name.get(source_name)
        target = by_name.get(target_name)
        if source is None or target is None:
            return
        elapsed_days = float(target.absolute_day) - float(source.absolute_day)
        if elapsed_days <= 0:
            raise ValueError(
                f"non-positive Sound transition interval: {source_name} -> {target_name}"
            )
        candidates.append(
            (
                SupportTransition(
                    source_sample=str(source.sample_id),
                    target_sample=str(target.sample_id),
                    phase=int(phase),
                    elapsed_days=elapsed_days / 90.0,
                ),
                float(target.absolute_day),
                kind,
            )
        )

    for year in (1, 2):
        prefix = f"Flu Year {year}"
        flu_names = [
            f"{prefix} Day 0",
            f"{prefix} Day 7",
            f"{prefix} Day 90",
        ]
        if all(name in by_name for name in flu_names):
            append(flu_names[0], flu_names[1], 0, "flu")
            append(flu_names[1], flu_names[2], 1, "flu")
    # Keep the natural control as one complete episode.  Five donors lack D90;
    # admitting only their acute half would make history availability depend on
    # an endpoint-specific missingness pattern.
    if all(
        name in by_name
        for name in (
            "Immune Variation Day 0",
            "Immune Variation Day 7",
            "Immune Variation Day 90",
        )
    ):
        append(
            "Immune Variation Day 0",
            "Immune Variation Day 7",
            NATURAL_VARIATION_PHASE,
            "natural_variation",
        )
        append(
            "Immune Variation Day 7",
            "Immune Variation Day 90",
            NATURAL_VARIATION_PHASE,
            "natural_variation",
        )
    return candidates

def augment_with_complete_prior_rna(
    units: Sequence[VaccineUnit],
    visit_timeline: str | Path,
    *,
    policy: str = "complete_prior_rna",
) -> tuple[list[VaccineUnit], PriorRNAHistoryAudit]:
    """Add completed same-donor RNA transitions observed by the query visit.

    T1 receives no added transitions. For T2 and T3, a transition is added when
    its endpoint precedes the query in study time; duplicates are removed by
    sample ID.
    """

    if policy not in HISTORY_POLICIES:
        raise ValueError(f"unknown RNA history policy: {policy}")
    if policy == "formal":
        return list(units), PriorRNAHistoryAudit(policy, len(units), 0, 0, 0, 0)
    include_flu = policy in {"complete_prior_rna", "complete_prior_flu"}
    include_natural = policy in {"complete_prior_rna", "complete_prior_natural"}

    timeline = _timeline(visit_timeline)
    by_sample = timeline.set_index("sample_id", drop=False)
    by_donor = {
        donor: frame.copy()
        for donor, frame in timeline.groupby("donor_id", sort=False)
    }
    result: list[VaccineUnit] = []
    units_with_added = 0
    added_flu = 0
    added_natural = 0
    frozen_flu_samples = {
        sample
        for unit in units
        for sample in (
            unit.source_sample,
            unit.target_sample,
            *(
                endpoint
                for transition in unit.supports
                for endpoint in (transition.source_sample, transition.target_sample)
            ),
        )
    }

    for unit in units:
        if unit.source_sample not in by_sample.index:
            raise KeyError(f"query sample absent from Sound timeline: {unit.source_sample}")
        if str(by_sample.loc[unit.source_sample, "donor_id"]) != unit.donor_id:
            raise ValueError("query sample donor differs from vaccine task unit")
        # T1 is forecast without history.
        if unit.task == "t1_y1_d0_to_y1_d7":
            result.append(unit)
            continue

        query_day = float(by_sample.loc[unit.source_sample, "absolute_day"])
        existing = {
            (transition.source_sample, transition.target_sample): transition
            for transition in unit.supports
        }
        additions: list[tuple[SupportTransition, float, str]] = []
        for transition, endpoint_day, kind in _completed_transitions(
            by_donor.get(unit.donor_id, timeline.iloc[0:0])
        ):
            key = (transition.source_sample, transition.target_sample)
            if key in existing:
                continue
            if kind == "flu" and not include_flu:
                continue
            if kind == "natural_variation" and not include_natural:
                continue
            if kind == "flu" and not set(key).issubset(frozen_flu_samples):
                continue
            if endpoint_day >= query_day:
                continue
            if unit.target_sample in key:
                raise ValueError("forecast target entered completed RNA history")
            additions.append((transition, endpoint_day, kind))

        if additions:
            units_with_added += 1
        for _, _, kind in additions:
            if kind == "flu":
                added_flu += 1
            else:
                added_natural += 1

        combined = [
            (
                transition,
                float(by_sample.loc[transition.target_sample, "absolute_day"]),
            )
            for transition in unit.supports
        ]
        combined.extend((transition, endpoint_day) for transition, endpoint_day, _ in additions)
        combined.sort(key=lambda item: (item[1], item[0].source_sample, item[0].target_sample))
        result.append(replace(unit, supports=tuple(item[0] for item in combined)))

    audit = PriorRNAHistoryAudit(
        policy=policy,
        units=len(units),
        units_with_added_history=units_with_added,
        added_transitions=added_flu + added_natural,
        added_flu_transitions=added_flu,
        added_natural_variation_transitions=added_natural,
    )
    return result, audit

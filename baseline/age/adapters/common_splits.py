from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        delimiter = "\t" if path.suffix in {".tsv", ".txt"} else ","
        return list(csv.DictReader(handle, delimiter=delimiter))

def common_fold_order(
    donors: list[str], split: Path, repeat: int, seed: int
) -> list[str]:
    """Order donors so official shuffled KFold reproduces the frozen split."""

    rows = [row for row in read_table(split) if int(row["repeat"]) == repeat]
    fold_by_donor: dict[str, int] = {}
    for row in rows:
        donor = row["donor_id"]
        if donor in fold_by_donor:
            raise ValueError(f"duplicate donor in frozen split repeat {repeat}: {donor}")
        fold_by_donor[donor] = int(row["fold"])
    if set(donors) != set(fold_by_donor):
        raise ValueError("input donors and frozen split donors do not match exactly")
    if set(fold_by_donor.values()) != set(range(5)):
        raise ValueError("frozen common split must contain exactly folds 0..4")

    positions_by_fold = {
        fold: test_idx
        for fold, (_, test_idx) in enumerate(
            KFold(n_splits=5, shuffle=True, random_state=seed).split(
                np.arange(len(donors))
            )
        )
    }
    output: list[str | None] = [None] * len(donors)
    for fold, positions in positions_by_fold.items():
        fold_donors = sorted(
            donor for donor in donors if fold_by_donor[donor] == fold
        )
        if len(fold_donors) != len(positions):
            raise ValueError(
                f"fold {fold} has {len(fold_donors)} donors but official KFold "
                f"creates {len(positions)} positions"
            )
        for position, donor in zip(positions, fold_donors):
            output[int(position)] = donor
    if any(donor is None for donor in output):
        raise RuntimeError("failed to construct common-fold donor order")

    ordered = [str(donor) for donor in output]
    for fold, (_, test_idx) in enumerate(
        KFold(n_splits=5, shuffle=True, random_state=seed).split(
            np.arange(len(ordered))
        )
    ):
        observed = {ordered[index] for index in test_idx}
        expected = {
            donor
            for donor, donor_fold in fold_by_donor.items()
            if donor_fold == fold
        }
        if observed != expected:
            raise RuntimeError(f"KFold order proof failed for fold {fold}")
    return ordered



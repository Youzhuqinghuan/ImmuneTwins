from __future__ import annotations
from paths import ROOT
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence
import h5py
import numpy as np
import pandas as pd
import torch

TaskId = Literal[
    "t1_y1_d0_to_y1_d7",
    "t2_d7_to_d90",
    "t3_y2_d0_to_y2_d90",
]

TASK_ALIASES: dict[str, TaskId] = {
    "T1": "t1_y1_d0_to_y1_d7",
    "T1_first_response": "t1_y1_d0_to_y1_d7",
    "t1_y1_d0_to_y1_d7": "t1_y1_d0_to_y1_d7",
    "T2": "t2_d7_to_d90",
    "T2_same_episode_recovery": "t2_d7_to_d90",
    "t2_d7_to_d90": "t2_d7_to_d90",
    "T3": "t3_y2_d0_to_y2_d90",
    "T3_cross_year_recovery": "t3_y2_d0_to_y2_d90",
    "t3_y2_d0_to_y2_d90": "t3_y2_d0_to_y2_d90",
}

TASK_PHASE: dict[TaskId, int] = {
    "t1_y1_d0_to_y1_d7": 0,
    "t2_d7_to_d90": 1,
    "t3_y2_d0_to_y2_d90": 2,
}

EXPECTED_UNITS: dict[TaskId, int] = {
    "t1_y1_d0_to_y1_d7": 83,
    "t2_d7_to_d90": 163,
    "t3_y2_d0_to_y2_d90": 75,
}

def canonical_task(task: str) -> TaskId:
    try:
        return TASK_ALIASES[str(task)]
    except KeyError as error:
        raise ValueError(f"unknown vaccine task: {task}") from error

def stable_seed(*values: object, base_seed: int = 42) -> int:
    payload = "|".join([str(base_seed), *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (
        2**32
    )

def decode_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=str,
    )

@dataclass(frozen=True)
class SupportTransition:
    source_sample: str
    target_sample: str
    phase: int
    elapsed_days: float

@dataclass(frozen=True)
class VaccineUnit:
    unit_id: str
    task: TaskId
    donor_id: str
    season: str
    outer_fold: int
    source_sample: str
    target_sample: str
    source_day: int
    target_day: int
    phase: int
    supports: tuple[SupportTransition, ...]

    @property
    def episode_id(self) -> str:
        return f"{self.donor_id}__{self.season}"

def _required_columns(frame: pd.DataFrame) -> None:
    required = {
        "donor_id",
        "season",
        "outer_fold",
        "sample_D0",
        "sample_D7",
        "sample_D90",
        "first_vaccination_test",
        "same_episode_update_test",
        "cross_year_history_test",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"task unit table lacks columns: {sorted(missing)}")
    if frame.duplicated(["donor_id", "season"]).any():
        raise ValueError("task unit table duplicates donor-season")

def build_task_units(
    task_units: str | Path | pd.DataFrame,
    task: str,
    *,
    enforce_formal_counts: bool = True,
) -> list[VaccineUnit]:
    """Build the query, target and support inputs for one task.

    Support elapsed time is a fraction of 90 days; the discrete phase
    identifies D0->D7 versus D7->D90.
    """

    frame = (
        task_units.copy()
        if isinstance(task_units, pd.DataFrame)
        else pd.read_csv(task_units, sep="\t")
    )
    _required_columns(frame)
    task_id = canonical_task(task)
    rows: list[VaccineUnit] = []
    ordered = frame.sort_values(["donor_id", "season"], kind="stable")

    if task_id == "t1_y1_d0_to_y1_d7":
        selected = ordered.loc[ordered["first_vaccination_test"].astype(bool)]
        for row in selected.itertuples(index=False):
            rows.append(
                VaccineUnit(
                    unit_id=f"{task_id}__{row.donor_id}__{row.season}",
                    task=task_id,
                    donor_id=str(row.donor_id),
                    season=str(row.season),
                    outer_fold=int(row.outer_fold),
                    source_sample=str(row.sample_D0),
                    target_sample=str(row.sample_D7),
                    source_day=0,
                    target_day=7,
                    phase=TASK_PHASE[task_id],
                    supports=(),
                )
            )
    elif task_id == "t2_d7_to_d90":
        selected = ordered.loc[ordered["same_episode_update_test"].astype(bool)]
        for row in selected.itertuples(index=False):
            support = SupportTransition(
                source_sample=str(row.sample_D0),
                target_sample=str(row.sample_D7),
                phase=0,
                elapsed_days=7.0 / 90.0,
            )
            rows.append(
                VaccineUnit(
                    unit_id=f"{task_id}__{row.donor_id}__{row.season}",
                    task=task_id,
                    donor_id=str(row.donor_id),
                    season=str(row.season),
                    outer_fold=int(row.outer_fold),
                    source_sample=str(row.sample_D7),
                    target_sample=str(row.sample_D90),
                    source_day=7,
                    target_day=90,
                    phase=TASK_PHASE[task_id],
                    supports=(support,),
                )
            )
    else:
        selected = ordered.loc[ordered["cross_year_history_test"].astype(bool)]
        for row in selected.itertuples(index=False):
            donor_rows = ordered.loc[ordered["donor_id"].astype(str).eq(str(row.donor_id))]
            prior = donor_rows.loc[
                donor_rows["first_vaccination_test"].astype(bool)
                & ~donor_rows["season"].astype(str).eq(str(row.season))
            ]
            if len(prior) != 1:
                raise ValueError(
                    f"T3 donor {row.donor_id} needs exactly one completed year-1 episode"
                )
            year1 = prior.iloc[0]
            if int(year1["outer_fold"]) != int(row.outer_fold):
                raise ValueError("all seasons for a donor must share one outer fold")
            supports = (
                SupportTransition(
                    source_sample=str(year1["sample_D0"]),
                    target_sample=str(year1["sample_D7"]),
                    phase=0,
                    elapsed_days=7.0 / 90.0,
                ),
                SupportTransition(
                    source_sample=str(year1["sample_D7"]),
                    target_sample=str(year1["sample_D90"]),
                    phase=1,
                    elapsed_days=83.0 / 90.0,
                ),
            )
            rows.append(
                VaccineUnit(
                    unit_id=f"{task_id}__{row.donor_id}__{row.season}",
                    task=task_id,
                    donor_id=str(row.donor_id),
                    season=str(row.season),
                    outer_fold=int(row.outer_fold),
                    source_sample=str(row.sample_D0),
                    target_sample=str(row.sample_D90),
                    source_day=0,
                    target_day=90,
                    phase=TASK_PHASE[task_id],
                    supports=supports,
                )
            )

    if len({unit.unit_id for unit in rows}) != len(rows):
        raise RuntimeError("constructed task units are not unique")
    if enforce_formal_counts and len(rows) != EXPECTED_UNITS[task_id]:
        raise ValueError(
            f"{task_id} unit count changed: {len(rows)} != {EXPECTED_UNITS[task_id]}"
        )
    return rows

def split_outer_units(
    units: Sequence[VaccineUnit], outer_fold: int
) -> tuple[list[VaccineUnit], list[VaccineUnit]]:
    train = [unit for unit in units if unit.outer_fold != int(outer_fold)]
    test = [unit for unit in units if unit.outer_fold == int(outer_fold)]
    train_donors = {unit.donor_id for unit in train}
    test_donors = {unit.donor_id for unit in test}
    if not train or not test or train_donors & test_donors:
        raise ValueError("outer split must be nonempty and donor-disjoint")
    return train, test

def split_inner_units(
    train_units: Sequence[VaccineUnit],
    inner_splits_path: str | Path,
    *,
    outer_fold: int,
    validation_fold: int = 0,
) -> tuple[list[VaccineUnit], list[VaccineUnit]]:
    table = pd.read_csv(inner_splits_path, sep="\t")
    required = {"outer_fold", "donor_id", "inner_fold"}
    if not required.issubset(table.columns):
        raise ValueError(f"inner split table lacks {sorted(required - set(table.columns))}")
    table = table.loc[pd.to_numeric(table["outer_fold"]).astype(int).eq(int(outer_fold))]
    if table["donor_id"].astype(str).duplicated().any():
        raise ValueError("inner split duplicates a donor within outer fold")
    validation_donors = set(
        table.loc[
            pd.to_numeric(table["inner_fold"]).astype(int).eq(int(validation_fold)),
            "donor_id",
        ].astype(str)
    )
    known = set(table["donor_id"].astype(str))
    requested = {unit.donor_id for unit in train_units}
    if not requested.issubset(known):
        raise ValueError("some outer-training donors are missing from inner splits")
    fit = [unit for unit in train_units if unit.donor_id not in validation_donors]
    validation = [unit for unit in train_units if unit.donor_id in validation_donors]
    if not fit or not validation:
        raise ValueError("inner donor fit/validation split is empty")
    if {x.donor_id for x in fit} & {x.donor_id for x in validation}:
        raise RuntimeError("inner validation leaked a donor")
    return fit, validation

@dataclass(frozen=True)
class FoldScale:
    center: np.ndarray
    scale: np.ndarray
    mmd_bandwidth: float
    outer_fold: int
    path: Path

    def standardize(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.center) / self.scale).astype(
            np.float32
        )

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) * self.scale + self.center).astype(
            np.float32
        )

def load_fold_scale(path: str | Path, *, outer_fold: int) -> FoldScale:
    path = Path(path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as payload:
        required = {"center", "scale", "mmd_bandwidth", "outer_fold", "fit_role"}
        if not required.issubset(payload.files):
            raise ValueError(f"fold scale lacks {sorted(required - set(payload.files))}")
        center = np.asarray(payload["center"], dtype=np.float32)
        scale = np.asarray(payload["scale"], dtype=np.float32)
        bandwidth = float(payload["mmd_bandwidth"])
        recorded_fold = int(payload["outer_fold"])
        fit_role = str(payload["fit_role"])
    if center.shape != (16,) or scale.shape != (16,) or not np.all(scale > 0):
        raise ValueError("fold scale must contain positive 16-D center/scale")
    if recorded_fold != int(outer_fold) or "training" not in fit_role:
        raise ValueError("fold scale is not fitted on the requested outer training fold")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("MMD bandwidth must be finite and positive")
    return FoldScale(center, scale, bandwidth, recorded_fold, path)

class UnifiedUCEStore:
    """Read UCE sets through the base321/delta168 unified manifest."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        frame = pd.read_csv(self.manifest_path, sep="\t")
        required = {
            "sample_id",
            "visit_id",
            "feature_store",
            "store_index",
            "store_visit_id",
            "uce_features_h5",
        }
        if not required.issubset(frame.columns):
            raise ValueError(f"unified visit manifest lacks {sorted(required - set(frame.columns))}")
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError("unified manifest duplicates sample_id")
        self.frame = frame.set_index(frame["sample_id"].astype(str), drop=False)
        self._handles: dict[Path, h5py.File] = {}

    def _handle(self, path: str | Path) -> h5py.File:
        resolved = Path(path).resolve(strict=True)
        if resolved not in self._handles:
            handle = h5py.File(resolved, "r")
            if handle.attrs.get("status") != "complete":
                handle.close()
                raise ValueError(f"incomplete UCE store: {resolved}")
            self._handles[resolved] = handle
        return self._handles[resolved]

    def take(
        self,
        sample_id: str,
        *,
        cells: int,
        seed: int,
    ) -> np.ndarray:
        if cells < 1:
            raise ValueError("UCE cell count must be positive")
        try:
            row = self.frame.loc[str(sample_id)]
        except KeyError as error:
            raise KeyError(f"sample absent from unified UCE manifest: {sample_id}") from error
        handle = self._handle(row["uce_features_h5"])
        index = int(row["store_index"])
        observed_id = handle["visit_id"][index]
        observed_id = observed_id.decode() if isinstance(observed_id, bytes) else str(observed_id)
        if observed_id != str(row["store_visit_id"]):
            raise ValueError("UCE store index no longer matches unified manifest")
        values = np.asarray(handle["embedding"][index], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 1280 or not np.isfinite(values).all():
            raise ValueError("UCE visit embedding shape/content changed")
        rng = np.random.default_rng(seed)
        positions = rng.choice(len(values), size=cells, replace=len(values) < cells)
        return values[positions].astype(np.float32, copy=False)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "UnifiedUCEStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

class SealedBaseUCEStore:
    """Reader for the 321 current-query UCE visits, used by current-state models."""

    def __init__(self, source_manifest_path: str | Path, uce_path: str | Path):
        self.source_manifest_path = Path(source_manifest_path).resolve(strict=True)
        self.path = Path(uce_path).resolve(strict=True)
        frame = pd.read_csv(self.source_manifest_path, sep="\t").reset_index(drop=True)
        required = {"visit_id", "model_source_sample", "paper_task"}
        if not required.issubset(frame.columns):
            raise ValueError(f"sealed source manifest lacks {sorted(required - set(frame.columns))}")
        if len(frame) != 321 or frame["model_source_sample"].astype(str).duplicated().any():
            raise ValueError("sealed current-query manifest must contain 321 unique samples")
        self.handle = h5py.File(self.path, "r")
        if (
            self.handle.attrs.get("status") != "complete"
            or self.handle["embedding"].shape != (321, 512, 1280)
        ):
            self.handle.close()
            raise ValueError("sealed base UCE artifact shape/status changed")
        observed = decode_array(self.handle["visit_id"][:])
        if not np.array_equal(observed, frame["visit_id"].astype(str).to_numpy()):
            self.handle.close()
            raise ValueError("sealed UCE visit order differs from source manifest")
        self.index = {
            str(sample): index
            for index, sample in enumerate(frame["model_source_sample"].astype(str))
        }

    def take(self, sample_id: str, *, cells: int, seed: int) -> np.ndarray:
        try:
            index = self.index[str(sample_id)]
        except KeyError as error:
            raise KeyError(f"sample absent from sealed current-query UCE store: {sample_id}") from error
        values = np.asarray(self.handle["embedding"][index], dtype=np.float32)
        rng = np.random.default_rng(seed)
        positions = rng.choice(len(values), size=cells, replace=len(values) < cells)
        return values[positions].astype(np.float32, copy=False)

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "SealedBaseUCEStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

class GeneSummaryStore:
    """Read visit pseudobulk and fit candidate genes on outer-training donors."""

    def __init__(self, manifest_path: str | Path):
        manifest = pd.read_csv(manifest_path, sep="\t")
        paths = manifest["gene_summary_h5"].astype(str).unique()
        if len(paths) != 1:
            raise ValueError("unified manifest must point to one gene summary store")
        self.path = Path(paths[0]).resolve(strict=True)
        self.handle = h5py.File(self.path, "r")
        if self.handle.attrs.get("status") != "complete":
            self.handle.close()
            raise ValueError("gene summary store is incomplete")
        self.genes = decode_array(self.handle["var/gene_symbol"][:])
        self.samples = decode_array(self.handle["sample_id"][:])
        self.index = {sample: i for i, sample in enumerate(self.samples)}
        if len(self.index) != len(self.samples):
            raise ValueError("gene summary duplicates sample_id")
        if "detection_fraction" not in self.handle:
            raise ValueError(
                "gene prior requires the all-cell gene summary with detection_fraction"
            )
        if self.handle.attrs.get("all_available_cells", False) not in (
            True,
            np.bool_(True),
        ):
            raise ValueError("gene prior cannot use a 512-cell sampled pseudobulk summary")

    def logcpm(self, sample_ids: Sequence[str], gene_indices: np.ndarray) -> np.ndarray:
        rows = np.asarray([self.index[str(sample)] for sample in sample_ids], dtype=np.int64)
        gene_indices = np.asarray(gene_indices, dtype=np.int64)
        if len(gene_indices) == 0 or np.any(np.diff(gene_indices) < 0):
            raise ValueError("gene indices must be nonempty and sorted")
        order = np.argsort(rows)
        counts = np.asarray(self.handle["sum_counts"][rows[order]][:, gene_indices], dtype=np.float32)
        totals = np.asarray(self.handle["total_umi"][rows[order]], dtype=np.float32)
        inverse = np.argsort(order)
        counts, totals = counts[inverse], totals[inverse]
        return np.log1p(counts * (1_000_000.0 / np.maximum(totals[:, None], 1.0))).astype(
            np.float32
        )

    def raw_counts(self, sample_ids: Sequence[str], gene_indices: np.ndarray) -> np.ndarray:
        rows = np.asarray([self.index[str(sample)] for sample in sample_ids], dtype=np.int64)
        gene_indices = np.asarray(gene_indices, dtype=np.int64)
        order = np.argsort(rows)
        values = np.asarray(
            self.handle["sum_counts"][rows[order]][:, gene_indices], dtype=np.float32
        )
        return values[np.argsort(order)]

    def detection_fraction(
        self, sample_ids: Sequence[str], gene_indices: np.ndarray
    ) -> np.ndarray:
        rows = np.asarray([self.index[str(sample)] for sample in sample_ids], dtype=np.int64)
        gene_indices = np.asarray(gene_indices, dtype=np.int64)
        order = np.argsort(rows)
        values = np.asarray(
            self.handle["detection_fraction"][rows[order]][:, gene_indices],
            dtype=np.float32,
        )[np.argsort(order)]
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("gene detection fractions must remain in [0, 1]")
        return values

    def close(self) -> None:
        self.handle.close()

@dataclass(frozen=True)
class GeneSelection:
    names: tuple[str, ...]
    raw_indices: np.ndarray
    variances: np.ndarray
    embeddings: torch.Tensor

def _embedding_lookup(payload: object) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("UCE gene embedding artifact must be a dictionary")
    result: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.numel() < 1:
            continue
        result.setdefault(str(key).upper(), tensor)
    if not result:
        raise ValueError("UCE gene embedding dictionary is empty")
    widths = {int(value.numel()) for value in result.values()}
    if len(widths) != 1:
        raise ValueError("UCE gene embeddings do not share one width")
    return result

def select_outer_training_genes(
    store: GeneSummaryStore,
    *,
    manifest_path: str | Path,
    outer_training_donors: Iterable[str],
    embedding_path: str | Path,
    n_genes: int,
) -> GeneSelection:
    """Select variable UCE-supported genes using outer-training visits only."""

    if n_genes < 1:
        raise ValueError("n_genes must be positive")
    manifest = pd.read_csv(manifest_path, sep="\t")
    donors = {str(value) for value in outer_training_donors}
    train = manifest.loc[manifest["donor_id"].astype(str).isin(donors)].copy()
    if train.empty or not set(train["donor_id"].astype(str)).issubset(donors):
        raise ValueError("gene selection has no outer-training visits")
    embedding_path = Path(embedding_path).resolve(strict=True)
    payload = torch.load(embedding_path, map_location="cpu", weights_only=False)
    lookup = _embedding_lookup(payload)
    supported = np.asarray(
        [index for index, gene in enumerate(store.genes) if str(gene).upper() in lookup],
        dtype=np.int64,
    )
    if len(supported) < n_genes:
        raise ValueError(f"only {len(supported)} genes have UCE embeddings; {n_genes} requested")
    # h5py benefits from sorted columns; the selected output is sorted again by
    # raw index after ranking, and the rank is retained in the variance field.
    sample_ids = train["sample_id"].astype(str).tolist()
    values = store.logcpm(sample_ids, supported)
    variances = values.var(axis=0, dtype=np.float64)
    symbols = store.genes[supported]
    rank = sorted(range(len(supported)), key=lambda i: (-float(variances[i]), str(symbols[i])))
    chosen_local = np.asarray(rank[:n_genes], dtype=np.int64)
    chosen_raw = supported[chosen_local]
    chosen_variance = variances[chosen_local].astype(np.float32)
    chosen_names = symbols[chosen_local].astype(str)
    raw_order = np.argsort(chosen_raw)
    chosen_raw = chosen_raw[raw_order]
    chosen_variance = chosen_variance[raw_order]
    chosen_names = chosen_names[raw_order]
    embeddings = torch.stack([lookup[name.upper()] for name in chosen_names]).to(torch.float32)
    return GeneSelection(
        names=tuple(chosen_names.tolist()),
        raw_indices=chosen_raw,
        variances=chosen_variance,
        embeddings=embeddings,
    )

def save_gene_selection(selection: GeneSelection, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "selection_index": np.arange(len(selection.names), dtype=int),
            "gene_symbol": selection.names,
            "raw_gene_index": selection.raw_indices,
            "outer_training_logcpm_variance": selection.variances,
        }
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)

def _formal_reader_class():
    from data.latent import Reader
    return Reader

class FoldLatentStore:
    """Wrapper around the fold-specific latent reader."""

    def __init__(self, path: str | Path, scale: FoldScale):
        cache_root = os.environ.get("IMMUNETWIN_LATENT_INDEX_CACHE_ROOT")
        if not cache_root:
            raise RuntimeError(
                "IMMUNETWIN_LATENT_INDEX_CACHE_ROOT must be explicitly set below the repository"
            )
        try:
            Path(cache_root).resolve().relative_to(ROOT)
        except ValueError as error:
            raise RuntimeError("latent index cache must remain below the repository") from error
        Reader = _formal_reader_class()
        self.reader = Reader(Path(path))
        self.scale = scale


    def take(
        self,
        sample_id: str,
        *,
        cells: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows = self.reader.rows(str(sample_id))
        if not len(rows):
            raise KeyError(f"sample absent from fold latent store: {sample_id}")
        raw, states, row_indices = self.reader.take(
            rows, cells, np.random.default_rng(seed), exact=True
        )
        standardized = self.scale.standardize(raw)
        return standardized, raw.astype(np.float32), states, row_indices

    def close(self) -> None:
        self.reader.close()

@dataclass
class UnitInputs:
    unit: VaccineUnit
    query_uce: np.ndarray
    source_latent: np.ndarray
    source_raw: np.ndarray
    source_states: np.ndarray
    source_row_indices: np.ndarray
    support_source_uce: np.ndarray
    support_target_uce: np.ndarray
    support_phase: np.ndarray
    support_dt: np.ndarray
    support_mask: np.ndarray
    support_source_latent_mean: np.ndarray
    support_target_latent_mean: np.ndarray
    support_response_latent: np.ndarray
    clinical_baseline_value: np.ndarray
    clinical_endpoint_value: np.ndarray
    clinical_delta_value: np.ndarray
    clinical_baseline_missing: np.ndarray
    clinical_endpoint_missing: np.ndarray
    clinical_delta_valid: np.ndarray
    clinical_change: np.ndarray
    gene_change: np.ndarray
    gene_confidence: np.ndarray
    program_delta: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    program_valid: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=bool)
    )

@dataclass(frozen=True)
class DonorBatch:
    """A donor-balanced batch containing every season for selected donors."""

    units: tuple[VaccineUnit, ...]
    donor_index: np.ndarray
    donor_ids: tuple[str, ...]

class UnitMaterializer:
    """Materialize model inputs."""

    def __init__(
        self,
        *,
        uce: UnifiedUCEStore,
        latent: FoldLatentStore,
        query_uce_cells: int,
        support_uce_cells: int,
        gene_store: GeneSummaryStore | None = None,
        gene_selection: GeneSelection | None = None,
        clinical_features: pd.DataFrame | None = None,
        program_delta_by_unit: dict[str, np.ndarray] | None = None,
        program_valid_by_unit: dict[str, np.ndarray] | None = None,
        seed: int = 42,
    ) -> None:
        self.uce = uce
        self.latent = latent
        self.query_uce_cells = int(query_uce_cells)
        self.support_uce_cells = int(support_uce_cells)
        self.gene_store = gene_store
        self.gene_selection = gene_selection
        self.clinical_features = None
        self.program_delta_by_unit: dict[str, np.ndarray] = {}
        self.program_valid_by_unit: dict[str, np.ndarray] = {}
        self._program_dim = 0
        self._clinical_value_columns: tuple[str, ...] = ()
        self._clinical_missing_columns: tuple[str, ...] = ()
        self._clinical_legacy_layout: tuple[tuple[str, int], ...] = ()
        self.seed = int(seed)
        if (gene_store is None) != (gene_selection is None):
            raise ValueError("gene store and selection must be supplied together")
        if (program_delta_by_unit is None) != (program_valid_by_unit is None):
            raise ValueError("program values and validity masks must be supplied together")
        if program_delta_by_unit is not None and program_valid_by_unit is not None:
            if set(program_delta_by_unit) != set(program_valid_by_unit):
                raise ValueError("program value and mask unit identifiers differ")
            widths: set[int] = set()
            for unit_id in sorted(program_delta_by_unit):
                delta = np.asarray(program_delta_by_unit[unit_id], dtype=np.float32)
                valid = np.asarray(program_valid_by_unit[unit_id], dtype=bool)
                if delta.ndim != 2 or delta.shape != valid.shape:
                    raise ValueError("program evidence must share [support,program] shape")
                if not np.isfinite(delta).all() or np.any(delta[~valid] != 0.0):
                    raise ValueError("program evidence must be finite and masked to zero")
                widths.add(int(delta.shape[1]))
                self.program_delta_by_unit[str(unit_id)] = delta.copy()
                self.program_valid_by_unit[str(unit_id)] = valid.copy()
            if len(widths) != 1:
                raise ValueError("program evidence width differs between units")
            self._program_dim = widths.pop()
        if clinical_features is not None:
            frame = clinical_features.copy()
            frame.index = frame.index.astype(str)
            frame.columns = frame.columns.astype(str)
            if not frame.index.is_unique:
                raise ValueError("clinical history sample index must be unique")
            if not frame.columns.is_unique:
                raise ValueError("clinical history columns must be unique")
            invalid = [
                column
                for column in frame.columns
                if not (column.startswith("value::") or column.startswith("missing::"))
            ]
            if invalid:
                raise ValueError(
                    "clinical history columns must use value:: or missing:: prefixes: "
                    f"{invalid[:5]}"
                )
            value_columns = tuple(
                column for column in frame.columns if column.startswith("value::")
            )
            missing_by_name = {
                column.removeprefix("missing::"): column
                for column in frame.columns
                if column.startswith("missing::")
            }
            value_names = tuple(column.removeprefix("value::") for column in value_columns)
            if set(value_names) != set(missing_by_name):
                missing_masks = sorted(set(value_names) - set(missing_by_name))
                missing_values = sorted(set(missing_by_name) - set(value_names))
                raise ValueError(
                    "clinical value/missing columns must be paired; "
                    f"missing masks={missing_masks[:5]}, missing values={missing_values[:5]}"
                )
            missing_columns = tuple(missing_by_name[name] for name in value_names)
            numeric = frame.to_numpy(dtype=np.float64)
            if not np.isfinite(numeric).all():
                raise ValueError("clinical history contains non-finite values")
            if missing_columns:
                missing_values = frame.loc[:, missing_columns].to_numpy(dtype=np.float64)
                if not np.isin(missing_values, (0.0, 1.0)).all():
                    raise ValueError("clinical missing masks must be binary 0/1")
            value_index = {name: index for index, name in enumerate(value_names)}
            legacy_layout: list[tuple[str, int]] = []
            for column in frame.columns:
                prefix, name = column.split("::", maxsplit=1)
                legacy_layout.append(
                    ("value" if prefix == "value" else "valid", value_index[name])
                )
            self.clinical_features = frame
            self._clinical_value_columns = value_columns
            self._clinical_missing_columns = missing_columns
            self._clinical_legacy_layout = tuple(legacy_layout)

    @property
    def n_genes(self) -> int:
        return 0 if self.gene_selection is None else len(self.gene_selection.names)

    @property
    def clinical_dim(self) -> int:
        return 0 if self.clinical_features is None else int(self.clinical_features.shape[1])

    @property
    def clinical_value_dim(self) -> int:
        return len(self._clinical_value_columns)

    @property
    def program_dim(self) -> int:
        return self._program_dim

    def inputs(
        self,
        unit: VaccineUnit,
        *,
        cells: int,
        draw_key: object,
        source_seed: int | None = None,
        include_history: bool = True,
    ) -> UnitInputs:
        transitions = unit.supports if include_history else ()
        if any(
            unit.target_sample in (transition.source_sample, transition.target_sample)
            for transition in transitions
        ):
            raise ValueError("completed support cannot contain the forecast target")
        query = self.uce.take(
            unit.source_sample,
            cells=self.query_uce_cells,
            seed=stable_seed("query_uce", unit.unit_id, draw_key, base_seed=self.seed),
        )
        source, source_raw, states, rows = self.latent.take(
            unit.source_sample,
            cells=cells,
            seed=(
                stable_seed("source_latent", unit.unit_id, draw_key, base_seed=self.seed)
                if source_seed is None
                else int(source_seed)
            ),
        )
        support_source: list[np.ndarray] = []
        support_target: list[np.ndarray] = []
        support_source_latent_mean: list[np.ndarray] = []
        support_target_latent_mean: list[np.ndarray] = []
        support_response_latent: list[np.ndarray] = []
        phases: list[int] = []
        elapsed: list[float] = []
        gene_change: list[np.ndarray] = []
        confidence: list[np.ndarray] = []
        clinical_change: list[np.ndarray] = []
        clinical_baseline_value: list[np.ndarray] = []
        clinical_endpoint_value: list[np.ndarray] = []
        clinical_delta_value: list[np.ndarray] = []
        clinical_baseline_missing: list[np.ndarray] = []
        clinical_endpoint_missing: list[np.ndarray] = []
        clinical_delta_valid: list[np.ndarray] = []
        latent_mean_cache: dict[str, np.ndarray] = {}

        def support_mean(sample_id: str) -> np.ndarray:
            cached = latent_mean_cache.get(sample_id)
            if cached is not None:
                return cached
            standardized, _, _, _ = self.latent.take(
                sample_id,
                cells=cells,
                seed=stable_seed(
                    "support_latent",
                    unit.unit_id,
                    draw_key,
                    sample_id,
                    base_seed=self.seed,
                ),
            )
            standardized = np.asarray(standardized)
            if standardized.shape != (cells, 16):
                raise ValueError(
                    "support latent sample must have shape "
                    f"({cells}, 16), got {standardized.shape} for {sample_id}"
                )
            if not np.isfinite(standardized).all():
                raise ValueError(f"support latent sample contains non-finite values: {sample_id}")
            value = standardized.mean(axis=0, dtype=np.float64).astype(np.float32)
            latent_mean_cache[sample_id] = value
            return value

        for index, transition in enumerate(transitions):
            support_source.append(
                self.uce.take(
                    transition.source_sample,
                    cells=self.support_uce_cells,
                    seed=stable_seed(
                        "support_source_uce", unit.unit_id, draw_key, index, base_seed=self.seed
                    ),
                )
            )
            support_target.append(
                self.uce.take(
                    transition.target_sample,
                    cells=self.support_uce_cells,
                    seed=stable_seed(
                        "support_target_uce", unit.unit_id, draw_key, index, base_seed=self.seed
                    ),
                )
            )
            source_mean = support_mean(transition.source_sample)
            target_mean = support_mean(transition.target_sample)
            support_source_latent_mean.append(source_mean)
            support_target_latent_mean.append(target_mean)
            support_response_latent.append((target_mean - source_mean).astype(np.float32))
            phases.append(int(transition.phase))
            elapsed.append(float(transition.elapsed_days))
            if self.clinical_features is not None:
                missing = {
                    sample
                    for sample in (transition.source_sample, transition.target_sample)
                    if sample not in self.clinical_features.index
                }
                if missing:
                    raise KeyError(f"clinical history is missing samples: {sorted(missing)}")
                source_row = self.clinical_features.loc[transition.source_sample]
                target_row = self.clinical_features.loc[transition.target_sample]
                baseline_value = source_row.loc[
                    list(self._clinical_value_columns)
                ].to_numpy(dtype=np.float32)
                endpoint_value = target_row.loc[
                    list(self._clinical_value_columns)
                ].to_numpy(dtype=np.float32)
                baseline_missing = source_row.loc[
                    list(self._clinical_missing_columns)
                ].to_numpy(dtype=np.float32).astype(bool)
                endpoint_missing = target_row.loc[
                    list(self._clinical_missing_columns)
                ].to_numpy(dtype=np.float32).astype(bool)
                delta_valid = ~(baseline_missing | endpoint_missing)
                delta_value = np.where(
                    delta_valid, endpoint_value - baseline_value, 0.0
                ).astype(np.float32)
                clinical_baseline_value.append(baseline_value)
                clinical_endpoint_value.append(endpoint_value)
                clinical_delta_value.append(delta_value)
                clinical_baseline_missing.append(baseline_missing)
                clinical_endpoint_missing.append(endpoint_missing)
                clinical_delta_valid.append(delta_valid)
                clinical_change.append(
                    np.asarray(
                        [
                            delta_value[value_index]
                            if kind == "value"
                            else float(delta_valid[value_index])
                            for kind, value_index in self._clinical_legacy_layout
                        ],
                        dtype=np.float32,
                    )
                )
            if self.gene_store is not None and self.gene_selection is not None:
                indices = self.gene_selection.raw_indices
                pair = self.gene_store.logcpm(
                    [transition.source_sample, transition.target_sample], indices
                )
                detected = self.gene_store.detection_fraction(
                    [transition.source_sample, transition.target_sample], indices
                )
                gene_change.append((pair[1] - pair[0]).astype(np.float32))
                # A gene changing from off to on (or vice versa) remains valid
                # evidence.  Endpoint maximum detection therefore measures
                # reliability without erasing switch-like vaccine responses.
                confidence.append(detected.max(axis=0).astype(np.float32))
        support_count = len(transitions)
        if support_count:
            support_source_array = np.stack(support_source)
            support_target_array = np.stack(support_target)
            support_source_latent_array = np.stack(support_source_latent_mean)
            support_target_latent_array = np.stack(support_target_latent_mean)
            support_response_latent_array = np.stack(support_response_latent)
        else:
            support_source_array = np.empty(
                (0, self.support_uce_cells, 1280), dtype=np.float32
            )
            support_target_array = support_source_array.copy()
            support_source_latent_array = np.empty((0, 16), dtype=np.float32)
            support_target_latent_array = np.empty((0, 16), dtype=np.float32)
            support_response_latent_array = np.empty((0, 16), dtype=np.float32)
        if gene_change:
            gene_array = np.stack(gene_change)
            confidence_array = np.stack(confidence)
        else:
            gene_array = np.empty((support_count, self.n_genes), dtype=np.float32)
            confidence_array = np.empty_like(gene_array)
        if clinical_change:
            clinical_array = np.stack(clinical_change)
            clinical_baseline_value_array = np.stack(clinical_baseline_value)
            clinical_endpoint_value_array = np.stack(clinical_endpoint_value)
            clinical_delta_value_array = np.stack(clinical_delta_value)
            clinical_baseline_missing_array = np.stack(clinical_baseline_missing)
            clinical_endpoint_missing_array = np.stack(clinical_endpoint_missing)
            clinical_delta_valid_array = np.stack(clinical_delta_valid)
        else:
            clinical_array = np.empty(
                (support_count, self.clinical_dim), dtype=np.float32
            )
            clinical_baseline_value_array = np.empty(
                (support_count, self.clinical_value_dim), dtype=np.float32
            )
            clinical_endpoint_value_array = np.empty_like(clinical_baseline_value_array)
            clinical_delta_value_array = np.empty_like(clinical_baseline_value_array)
            clinical_baseline_missing_array = np.empty(
                (support_count, self.clinical_value_dim), dtype=bool
            )
            clinical_endpoint_missing_array = np.empty_like(
                clinical_baseline_missing_array
            )
            clinical_delta_valid_array = np.empty_like(clinical_baseline_missing_array)
        if include_history and self.program_delta_by_unit:
            if unit.unit_id not in self.program_delta_by_unit:
                raise KeyError(f"program evidence is missing unit: {unit.unit_id}")
            program_delta_array = self.program_delta_by_unit[unit.unit_id].copy()
            program_valid_array = self.program_valid_by_unit[unit.unit_id].copy()
            if program_delta_array.shape[0] != support_count:
                raise ValueError("program evidence support count differs from vaccine unit")
        else:
            program_delta_array = np.empty(
                (support_count, self.program_dim), dtype=np.float32
            )
            program_valid_array = np.empty_like(program_delta_array, dtype=bool)
        return UnitInputs(
            unit=unit,
            query_uce=query,
            source_latent=source,
            source_raw=source_raw,
            source_states=states,
            source_row_indices=rows,
            support_source_uce=support_source_array,
            support_target_uce=support_target_array,
            support_phase=np.asarray(phases, dtype=np.int64),
            support_dt=np.asarray(elapsed, dtype=np.float32),
            support_mask=np.ones(support_count, dtype=bool),
            support_source_latent_mean=support_source_latent_array,
            support_target_latent_mean=support_target_latent_array,
            support_response_latent=support_response_latent_array,
            clinical_baseline_value=clinical_baseline_value_array,
            clinical_endpoint_value=clinical_endpoint_value_array,
            clinical_delta_value=clinical_delta_value_array,
            clinical_baseline_missing=clinical_baseline_missing_array,
            clinical_endpoint_missing=clinical_endpoint_missing_array,
            clinical_delta_valid=clinical_delta_valid_array,
            clinical_change=clinical_array,
            gene_change=gene_array,
            gene_confidence=confidence_array,
            program_delta=program_delta_array,
            program_valid=program_valid_array,
        )

    def target(
        self,
        unit: VaccineUnit,
        *,
        cells: int,
        draw_key: object,
        target_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.latent.take(
            unit.target_sample,
            cells=cells,
            seed=(
                stable_seed("target_latent", unit.unit_id, draw_key, base_seed=self.seed)
                if target_seed is None
                else int(target_seed)
            ),
        )

def iter_donor_batches(
    units: Sequence[VaccineUnit],
    *,
    batch_size_donors: int,
    seed: int,
) -> Iterator[DonorBatch]:
    """Shuffle donors, then expand every selected donor to all of its units."""

    if batch_size_donors < 1:
        raise ValueError("batch_size_donors must be positive")
    grouped: dict[str, list[VaccineUnit]] = {}
    for unit in units:
        grouped.setdefault(unit.donor_id, []).append(unit)
    donor_ids = np.asarray(sorted(grouped), dtype=str)
    order = np.random.default_rng(seed).permutation(len(donor_ids))
    for start in range(0, len(order), batch_size_donors):
        selected = tuple(
            str(donor_ids[int(index)])
            for index in order[start : start + batch_size_donors]
        )
        flattened: list[VaccineUnit] = []
        donor_index: list[int] = []
        for local_index, donor_id in enumerate(selected):
            donor_units = sorted(
                grouped[donor_id], key=lambda unit: (unit.season, unit.unit_id)
            )
            flattened.extend(donor_units)
            donor_index.extend([local_index] * len(donor_units))
        yield DonorBatch(
            units=tuple(flattened),
            donor_index=np.asarray(donor_index, dtype=np.int64),
            donor_ids=selected,
        )

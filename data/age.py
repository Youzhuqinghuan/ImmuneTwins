from __future__ import annotations
from paths import expand_paths
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

WORKSPACE_ROOT = Path(
    os.environ.get("IMMUNETWINS_WORKSPACE_ROOT", REPOSITORY_ROOT.parent)
).expanduser()

DEFAULT_CIMA_UCE = Path(
    os.environ.get(
        "IMMUNETWINS_CIMA_UCE",
        WORKSPACE_ROOT
        / "results/patient_encoder_feasibility_2026-07-11/pulsar/uce_cima/"
        "cima_417_natural1024_union5_uce_embeddings.h5ad",
    )
).expanduser()

DEFAULT_CIMA_FOLDS = Path(
    os.environ.get(
        "IMMUNETWINS_CIMA_FOLDS",
        WORKSPACE_ROOT / "data/cima_random_donor.tsv",
    )
).expanduser()

DEFAULT_EXTERNAL_UCE_SHARDS = Path(
    os.environ.get(
        "IMMUNETWINS_EXTERNAL_UCE_SHARDS",
        WORKSPACE_ROOT
        / "results/official_age_baselines/UCE_mean/strict_zero_shot_v1/"
        "external_uce_shards",
    )
).expanduser()


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values],
        dtype=str,
    )

def _read_obs_column(obs: h5py.Group, name: str) -> np.ndarray:
    node = obs[name]
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[:])
    if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        codes = np.asarray(node["codes"][:], dtype=np.int64)
        categories = _decode(np.asarray(node["categories"][:]))
        if (codes < 0).any() or (codes >= len(categories)).any():
            raise ValueError(f"obs/{name} contains missing or invalid categorical codes")
        return categories[codes]
    raise ValueError(f"unsupported H5AD obs encoding for {name}")

@dataclass(frozen=True)
class CimaDonorRecord:
    donor_id: str
    age: float
    row_indices: np.ndarray

class CimaUceStore:
    """Index CIMA metadata once and lazily read only one donor's X_uce rows."""

    def __init__(
        self,
        path: str | Path = DEFAULT_CIMA_UCE,
        member_column: str = "member_20260711",
        expected_donors: int | None = 417,
        expected_cells_per_donor: int | None = 1024,
        input_dim: int = 1280,
    ) -> None:
        self.path = Path(path)
        self.member_column = member_column
        self.input_dim = input_dim
        self._handle: h5py.File | None = None
        self._member_cache: np.ndarray | None = None
        self._cache_slices: dict[str, slice] = {}
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with h5py.File(self.path, "r") as handle:
            if "X_uce" not in handle["obsm"]:
                raise ValueError("CIMA H5AD lacks obsm/X_uce")
            embeddings = handle["obsm/X_uce"]
            if embeddings.ndim != 2 or embeddings.shape[1] != input_dim:
                raise ValueError(f"CIMA X_uce must have width {input_dim}")
            n_rows = int(embeddings.shape[0])
            obs = handle["obs"]
            for required in ("donor_id", "age", member_column):
                if required not in obs:
                    raise ValueError(f"CIMA H5AD lacks obs/{required}")
            donors = _read_obs_column(obs, "donor_id").astype(str)
            ages = np.asarray(_read_obs_column(obs, "age"), dtype=np.float64)
            member = np.asarray(_read_obs_column(obs, member_column), dtype=bool)
        if not (len(donors) == len(ages) == len(member) == n_rows):
            raise ValueError("CIMA X_uce/obs row counts disagree")
        selected_rows = np.flatnonzero(member)
        selected_donors = donors[selected_rows]
        records: list[CimaDonorRecord] = []
        for donor in sorted(pd.unique(selected_donors).astype(str)):
            rows = selected_rows[selected_donors == donor]
            donor_ages = np.unique(ages[rows])
            if len(donor_ages) != 1 or not np.isfinite(donor_ages[0]):
                raise ValueError(f"age is not a finite donor constant for {donor}")
            if expected_cells_per_donor is not None and len(rows) != expected_cells_per_donor:
                raise ValueError(
                    f"{donor} has {len(rows)} {member_column} cells; "
                    f"expected {expected_cells_per_donor}"
                )
            records.append(CimaDonorRecord(donor, float(donor_ages[0]), rows))
        if expected_donors is not None and len(records) != expected_donors:
            raise ValueError(f"CIMA has {len(records)} selected donors; expected {expected_donors}")
        self.records = tuple(records)
        self._record_by_id = {record.donor_id: record for record in records}

    @property
    def donor_ids(self) -> tuple[str, ...]:
        return tuple(record.donor_id for record in self.records)

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def load_donor(self, donor_id: str) -> np.ndarray:
        try:
            record = self._record_by_id[str(donor_id)]
        except KeyError as error:
            raise KeyError(f"unknown CIMA donor: {donor_id}") from error
        if self._member_cache is not None:
            matrix = self._member_cache[self._cache_slices[record.donor_id]]
        else:
            matrix = np.asarray(
                self._file()["obsm/X_uce"][record.row_indices], dtype=np.float32
            )
        if matrix.shape != (len(record.row_indices), self.input_dim):
            raise ValueError(f"invalid X_uce slice for {donor_id}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"non-finite X_uce values for {donor_id}")
        return matrix

    def cache_member_view(self) -> dict[str, object]:
        """Read the shard once into a contiguous array."""
        if self._member_cache is not None:
            return self.cache_audit()
        total_cells = sum(len(record.row_indices) for record in self.records)
        cache = np.empty((total_cells, self.input_dim), dtype=np.float32)
        cache_slices: dict[str, slice] = {}
        cursor = 0
        source = self._file()["obsm/X_uce"]
        for record in self.records:
            stop = cursor + len(record.row_indices)
            cache[cursor:stop] = source[record.row_indices]
            cache_slices[record.donor_id] = slice(cursor, stop)
            cursor = stop
        if cursor != total_cells or not np.isfinite(cache).all():
            raise ValueError("CIMA member cache construction failed")
        self._member_cache = cache
        self._cache_slices = cache_slices
        self.close()
        return self.cache_audit()

    def cache_audit(self) -> dict[str, object]:
        return {
            "member_column": self.member_column,
            "cached": self._member_cache is not None,
            "n_donors": len(self.records),
            "n_cells": (
                int(self._member_cache.shape[0])
                if self._member_cache is not None
                else sum(len(record.row_indices) for record in self.records)
            ),
            "input_dim": self.input_dim,
            "bytes": int(self._member_cache.nbytes) if self._member_cache is not None else None,
        }

    def age(self, donor_id: str) -> float:
        return self._record_by_id[str(donor_id)].age

def load_common_cima_folds(
    path: str | Path = DEFAULT_CIMA_FOLDS,
    expected_donors: Sequence[str] | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"repeat", "fold", "donor_id", "age"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"CIMA split lacks columns: {sorted(missing)}")
    frame = frame.loc[frame["repeat"].eq(0), ["donor_id", "fold", "age"]].copy()
    frame["donor_id"] = frame["donor_id"].astype(str)
    if frame["donor_id"].duplicated().any():
        raise ValueError("repeat-0 CIMA split repeats donors")
    folds = sorted(frame["fold"].astype(int).unique().tolist())
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"common CIMA folds must be 0..4, got {folds}")
    frame["fold"] = frame["fold"].astype(int)
    frame["age"] = frame["age"].astype(float)
    if expected_donors is not None:
        expected = set(map(str, expected_donors))
        observed = set(frame["donor_id"])
        if observed != expected:
            raise ValueError(
                f"CIMA split donor mismatch: missing={sorted(expected-observed)[:5]}, "
                f"extra={sorted(observed-expected)[:5]}"
            )
    return frame.sort_values("donor_id").reset_index(drop=True)

class CimaDonorDataset(Dataset[dict[str, object]]):
    def __init__(self, store: CimaUceStore, donor_ids: Sequence[str]) -> None:
        self.store = store
        self.donor_ids = tuple(map(str, donor_ids))
        missing = sorted(set(self.donor_ids) - set(store.donor_ids))
        if missing:
            raise ValueError(f"unknown CIMA dataset donors: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.donor_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        donor_id = self.donor_ids[index]
        return {
            "donor_id": donor_id,
            "age": self.store.age(donor_id),
            "cells": self.store.load_donor(donor_id),
        }

def collate_donors(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    if not samples:
        raise ValueError("cannot collate an empty donor batch")
    arrays = [np.asarray(sample["cells"], dtype=np.float32) for sample in samples]
    width = arrays[0].shape[1]
    if any(array.ndim != 2 or array.shape[1] != width or len(array) < 1 for array in arrays):
        raise ValueError("donor cell matrices must be non-empty and have one common width")
    max_cells = max(len(array) for array in arrays)
    cells = torch.zeros((len(arrays), max_cells, width), dtype=torch.float32)
    mask = torch.zeros((len(arrays), max_cells), dtype=torch.bool)
    for row, array in enumerate(arrays):
        count = len(array)
        cells[row, :count] = torch.from_numpy(array)
        mask[row, :count] = True
    return {
        "donor_id": [str(sample["donor_id"]) for sample in samples],
        "age": torch.tensor([float(sample["age"]) for sample in samples]),
        "cells": cells,
        "mask": mask,
    }

@dataclass(frozen=True)
class ExternalShard:
    group_id: int
    embeddings_path: Path
    marker_path: Path
    n_cells: int

class ImmunobiologyUceShardStore:
    """Iterator over Immunobiology UCE cell-embedding shards."""

    def __init__(
        self,
        root: str | Path = DEFAULT_EXTERNAL_UCE_SHARDS,
        expected_groups: int | None = 36,
        input_dim: int = 1280,
    ) -> None:
        self.root = Path(root)
        self.input_dim = input_dim
        groups = sorted(self.root.glob("group_*"))
        if expected_groups is not None and len(groups) != expected_groups:
            raise ValueError(f"found {len(groups)} external groups; expected {expected_groups}")
        shards: list[ExternalShard] = []
        for group in groups:
            try:
                group_id = int(group.name.split("_")[-1])
            except ValueError as error:
                raise ValueError(f"invalid external shard directory: {group}") from error
            marker_path = group / "COMPLETE.json"
            embeddings_path = group / "cell_embeddings.npz"
            if not marker_path.is_file() or not embeddings_path.is_file():
                raise FileNotFoundError(f"incomplete external shard: {group}")
            marker = expand_paths(json.loads(marker_path.read_text()))
            if marker.get("status") != "PASS":
                raise ValueError(f"external shard is not PASS: {marker_path}")
            if marker.get("target_labels_accessed") is not False:
                raise ValueError(f"external shard is not label-free: {marker_path}")
            shards.append(
                ExternalShard(
                    group_id=group_id,
                    embeddings_path=embeddings_path,
                    marker_path=marker_path,
                    n_cells=int(marker["n_cells"]),
                )
            )
        if not shards:
            raise ValueError(f"no external UCE shards under {self.root}")
        if [shard.group_id for shard in shards] != list(range(len(shards))):
            raise ValueError("external UCE group IDs are not contiguous from zero")
        self.shards = tuple(shards)


    def expected_donor_manifest(
        self,
        *,
        expected_donors: int | None = 234,
        expected_cells_per_donor: int | None = 1500,
    ) -> pd.DataFrame:
        """Record donor coverage from shard IDs before inference."""
        counts: dict[str, int] = {}
        for shard in self.shards:
            with np.load(shard.embeddings_path, allow_pickle=False) as archive:
                donor_ids = archive["donor_id"].astype(str)
            if len(donor_ids) != shard.n_cells:
                raise ValueError(f"external donor-ID rows failed: {shard.embeddings_path}")
            donors, donor_counts = np.unique(donor_ids, return_counts=True)
            for donor_id, count in zip(donors.astype(str), donor_counts.astype(int)):
                counts[donor_id] = counts.get(donor_id, 0) + int(count)
        donor_ids = sorted(counts)
        if expected_donors is not None and len(donor_ids) != expected_donors:
            raise ValueError(
                f"label-free external manifest has {len(donor_ids)} donors; expected {expected_donors}"
            )
        wrong = {
            donor_id: counts[donor_id]
            for donor_id in donor_ids
            if expected_cells_per_donor is not None
            and counts[donor_id] != expected_cells_per_donor
        }
        if wrong:
            raise ValueError(f"label-free external donor cell counts failed: {wrong}")
        return pd.DataFrame({"donor_id": donor_ids})

    def iter_shards(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        for shard in self.shards:
            with np.load(shard.embeddings_path, allow_pickle=False) as archive:
                if set(archive.files) != {"embeddings", "donor_id"}:
                    raise ValueError(f"unexpected external shard members: {archive.files}")
                embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
                donor_ids = archive["donor_id"].astype(str)
            if embeddings.shape != (len(donor_ids), self.input_dim):
                raise ValueError(f"external shard shape failed: {shard.embeddings_path}")
            if len(donor_ids) != shard.n_cells or not np.isfinite(embeddings).all():
                raise ValueError(f"external shard values failed: {shard.embeddings_path}")
            yield shard.group_id, embeddings, donor_ids

    def audit(self) -> dict[str, object]:
        return {
            "root": str(self.root.resolve()),
            "n_groups": len(self.shards),
            "n_cells_from_markers": sum(shard.n_cells for shard in self.shards),
            "input_dim": self.input_dim,
            "target_labels_accessed": False,
        }

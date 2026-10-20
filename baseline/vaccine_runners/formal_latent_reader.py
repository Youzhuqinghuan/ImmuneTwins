#!/usr/bin/env python3
"""Read-only row index for latent stores."""

from __future__ import annotations

import fcntl
import zlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

from common import ROOT, decode


SCHEMA_VERSION = 1
CACHE_ROOT = Path(os.environ.get(
    "IMMUNETWIN_LATENT_INDEX_CACHE_ROOT",
    str(Path(__file__).resolve().parents[2] / ".runtime/latent_index"),
))
AUDIT_ROOT = Path(os.environ.get(
    "IMMUNETWIN_LATENT_INDEX_AUDIT_ROOT",
    str(ROOT / "results/runtime/formal_latent_reader_index"),
))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _save_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cache_valid(directory: Path, source: Path) -> bool:
    meta_path = directory / "meta.json"
    required = ["valid_order.npy", "lineage_codes.npy", "unique.npy", "starts.npy", "counts.npy"]
    if not meta_path.is_file() or any(not (directory / name).is_file() for name in required):
        return False
    try:
        meta = json.loads(meta_path.read_text())
        stat = source.stat()
        return (meta.get("status") == "pass"
                and meta.get("schema_version") == SCHEMA_VERSION
                and meta.get("source") == str(source.resolve())
                and meta.get("source_size") == stat.st_size
                and meta.get("source_mtime_ns") == stat.st_mtime_ns)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _ensure_cache(source: Path, handle: h5py.File) -> tuple[Path, bool]:
    stat = source.stat()
    identity = f"{source.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    key = f"{source.stem}-{zlib.crc32(identity):08x}"
    directory = CACHE_ROOT / key
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = CACHE_ROOT / f"{key}.lock"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if _cache_valid(directory, source):
            return directory, True
        started = time.time()
        directory.mkdir(parents=True, exist_ok=True)
        sample_codes = handle["obs/sample_id/codes"][:]
        lineage_codes = handle["obs/broad_lineage/codes"][:]
        order = np.argsort(sample_codes, kind="stable")
        sorted_codes = sample_codes[order]
        mask = sorted_codes >= 0
        unique, starts, counts = np.unique(
            sorted_codes[mask], return_index=True, return_counts=True
        )
        valid_order = order[mask]
        arrays = {
            "valid_order.npy": valid_order,
            "lineage_codes.npy": lineage_codes,
            "unique.npy": unique,
            "starts.npy": starts,
            "counts.npy": counts,
        }
        for name, values in arrays.items():
            _save_array(directory / name, np.asarray(values))
        stat = source.stat()
        payload = {
            "status": "pass",
            "schema_version": SCHEMA_VERSION,
            "source": str(source.resolve()),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "n_rows": int(len(sample_codes)),
            "n_valid_rows": int(len(valid_order)),
            "n_sample_codes": int(len(unique)),
            "cache_directory": str(directory),
            "cache_role": "read-only row lookup; embeddings and labels remain in formal H5",
            "scientific_values_changed": False,
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(directory / "meta.json", payload)
        _atomic_json(AUDIT_ROOT / f"{key}.json", payload)
        return directory, False


class Reader:
    """Replacement for the latent readers of the three baselines."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.h = h5py.File(self.path, "r")
        if self.h.attrs.get("status") != "complete":
            self.h.close()
            raise RuntimeError(f"incomplete latent store: {self.path}")
        self.samples = np.asarray([decode(x) for x in self.h["obs/sample_id/categories"][:]])
        self.lineages = np.asarray([decode(x) for x in self.h["obs/broad_lineage/categories"][:]])
        self.states = np.asarray([decode(x) for x in self.h["obs/fine_cell_type_L2/categories"][:]])
        directory, self.index_cache_hit = _ensure_cache(self.path, self.h)
        self.valid_order = np.load(directory / "valid_order.npy", mmap_mode="r", allow_pickle=False)
        self.lineage_codes = np.load(directory / "lineage_codes.npy", mmap_mode="r", allow_pickle=False)
        unique = np.load(directory / "unique.npy", allow_pickle=False)
        starts = np.load(directory / "starts.npy", allow_pickle=False)
        counts = np.load(directory / "counts.npy", allow_pickle=False)
        self.by_sample = {
            str(self.samples[int(code)]): (int(start), int(count))
            for code, start, count in zip(unique, starts, counts)
        }

    def rows(self, sample: str, lineage: str = "all") -> np.ndarray:
        location = self.by_sample.get(str(sample))
        if location is None:
            return np.empty(0, dtype=np.int64)
        start, count = location
        rows = np.asarray(
            self.valid_order[start:start + count], dtype=np.int64
        )
        if lineage == "all":
            return rows
        matches = np.flatnonzero(self.lineages == lineage)
        if len(matches) != 1:
            raise RuntimeError(f"lineage category not unique: {lineage}")
        code = int(matches[0])
        return np.asarray(rows[self.lineage_codes[rows] == code], dtype=np.int64)

    def take(self, rows: np.ndarray, size: int, rng: np.random.Generator,
             exact: bool = False):
        if not len(rows):
            raise RuntimeError("empty cell distribution")
        n = size if exact else min(size, len(rows))
        chosen = rng.choice(rows, size=n, replace=len(rows) < n)
        unique_rows, inverse = np.unique(chosen, return_inverse=True)
        x = self.h["embedding"][unique_rows].astype(np.float32)[inverse]
        state_codes = self.h["obs/fine_cell_type_L2/codes"][unique_rows][inverse]
        states = np.asarray([
            self.states[int(code)] if code >= 0 else "unknown" for code in state_codes
        ], dtype=str)
        source_rows = self.h["row_index"][unique_rows][inverse].astype(np.int64)
        return x, states, source_rows

    def close(self):
        self.h.close()

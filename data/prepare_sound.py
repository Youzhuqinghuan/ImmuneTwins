from __future__ import annotations
from paths import expand_paths
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

DELTA_RAW_SCHEMA = "sound_immunetwin_visit_delta_raw_1"

DELTA_UCE_SCHEMA = "sound_immunetwin_visit_delta_uce_1"

GENE_SUMMARY_SCHEMA = "sound_immunetwin_visit_gene_summary_1"

FULL_GENE_SUMMARY_SCHEMA = "sound_immunetwin_full_visit_gene_summary_1"

UNIFIED_MANIFEST_SCHEMA = "sound_immunetwin_visit_manifest_1"

DAY_COLUMNS = (("D0", 0, "sample_D0"), ("D7", 7, "sample_D7"), ("D90", 90, "sample_D90"))

REQUIRED_TASK_COLUMNS = {
    "donor_id",
    "season",
    "outer_fold",
    *(column for _, _, column in DAY_COLUMNS),
}

def _decode(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)

def _decode_array(values: np.ndarray) -> np.ndarray:
    return np.asarray([_decode(value) for value in values], dtype=str)


def _stable_seed(*values: object, base_seed: int) -> int:
    payload = "|".join([str(base_seed), *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (
        2**32
    )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)

def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)

def _artifact(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
    }

def _resume_valid(
    audit_path: Path,
    *,
    schema: str,
    output_paths: Mapping[str, Path],
) -> dict[str, object] | None:
    if not audit_path.is_file() or any(not path.is_file() for path in output_paths.values()):
        return None
    try:
        audit = expand_paths(json.loads(audit_path.read_text()))
    except (OSError, json.JSONDecodeError):
        return None
    if audit.get("status") != "pass" or audit.get("schema_version") != schema:
        return None
    return {**audit, "resumed": True}

def _base_sample_column(frame: pd.DataFrame) -> str:
    for candidate in ("model_source_sample", "sample_id"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("base visit manifest needs model_source_sample or sample_id")

def derive_full_visits(
    task_units_path: Path,
    *,
    expected_units: int | None = 163,
    expected_full_visits: int | None = 489,
) -> pd.DataFrame:
    """Derive one canonical D0/D7/D90 row per physical visit."""

    task_units = pd.read_csv(task_units_path, sep="\t")
    missing_columns = REQUIRED_TASK_COLUMNS - set(task_units.columns)
    if missing_columns:
        raise ValueError(f"task units missing columns: {sorted(missing_columns)}")
    if expected_units is not None and len(task_units) != expected_units:
        raise ValueError(f"task unit count changed: {len(task_units)} != {expected_units}")
    if task_units.duplicated(["donor_id", "season"]).any():
        raise ValueError("task units duplicate donor-season")

    records: list[dict[str, object]] = []
    ordered_units = task_units.sort_values(["donor_id", "season"], kind="stable")
    for row in ordered_units.itertuples(index=False):
        donor = str(row.donor_id)
        season = str(row.season)
        outer_fold = int(row.outer_fold)
        for day_name, day_value, sample_column in DAY_COLUMNS:
            sample = str(getattr(row, sample_column))
            if not sample or sample.casefold() in {"nan", "none"}:
                raise ValueError(f"{donor}/{season}/{day_name} lacks a sample")
            records.append(
                {
                    "visit_id": f"sound__{donor}__{season}__{day_name}",
                    "donor_id": donor,
                    "season": season,
                    "outer_fold": outer_fold,
                    "nominal_day": day_name,
                    "nominal_day_value": day_value,
                    "sample_id": sample,
                }
            )
    full = pd.DataFrame(records)
    if expected_full_visits is not None and len(full) != expected_full_visits:
        raise ValueError(f"full visit count changed: {len(full)} != {expected_full_visits}")
    if full["visit_id"].duplicated().any() or full["sample_id"].duplicated().any():
        raise ValueError("full visit contract requires unique visit and sample IDs")
    full = full.reset_index(drop=True)
    full["full_index"] = np.arange(len(full), dtype=np.int64)
    return full

def derive_visit_contract(
    task_units_path: Path,
    base_visit_manifest_path: Path,
    *,
    expected_units: int | None = 163,
    expected_full_visits: int | None = 489,
    expected_base_visits: int | None = 321,
    expected_delta_visits: int | None = 168,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return full, base and delta visit tables using sample ID as the join key."""

    full = derive_full_visits(
        task_units_path,
        expected_units=expected_units,
        expected_full_visits=expected_full_visits,
    )

    base_source = pd.read_csv(base_visit_manifest_path, sep="\t").reset_index(drop=True)
    sample_column = _base_sample_column(base_source)
    if expected_base_visits is not None and len(base_source) != expected_base_visits:
        raise ValueError(
            f"base visit count changed: {len(base_source)} != {expected_base_visits}"
        )
    if base_source[sample_column].astype(str).duplicated().any():
        raise ValueError("base visit manifest contains duplicate samples")
    base_samples = set(base_source[sample_column].astype(str))
    full_samples = set(full["sample_id"])
    outside = sorted(base_samples - full_samples)
    if outside:
        raise ValueError(f"base visit manifest is not a subset of task units: {outside[:5]}")

    base_lookup = full.set_index("sample_id", drop=False).loc[
        base_source[sample_column].astype(str)
    ].reset_index(drop=True)
    base_lookup["store_index"] = np.arange(len(base_lookup), dtype=np.int64)
    base_lookup["store_visit_id"] = (
        base_source["visit_id"].astype(str).to_numpy()
        if "visit_id" in base_source
        else base_lookup["visit_id"].to_numpy()
    )
    for column in (
        "raw_h5ad",
        "cells",
        "sampling_seed",
        "selected_nnz",
    ):
        if column in base_source:
            base_lookup[column] = base_source[column].to_numpy()

    delta = full.loc[~full["sample_id"].isin(base_samples)].copy().reset_index(drop=True)
    delta["store_index"] = np.arange(len(delta), dtype=np.int64)
    delta["store_visit_id"] = delta["visit_id"]
    if expected_delta_visits is not None and len(delta) != expected_delta_visits:
        raise ValueError(f"delta visit count changed: {len(delta)} != {expected_delta_visits}")
    if set(base_lookup["sample_id"]) & set(delta["sample_id"]):
        raise RuntimeError("base and delta sample sets overlap")
    if set(base_lookup["sample_id"]) | set(delta["sample_id"]) != full_samples:
        raise RuntimeError("base and delta do not reconstruct the full visit contract")
    return full, base_lookup, delta

def _categorical_categories(group: h5py.Group) -> np.ndarray:
    if not isinstance(group, h5py.Group) or "categories" not in group or "codes" not in group:
        raise ValueError("expected an AnnData categorical group")
    return _decode_array(group["categories"][:])

def _raw_inventory(
    raw_root: Path,
    *,
    expected_raw_files: int | None,
) -> tuple[list[Path], dict[str, Path], np.ndarray]:
    files = sorted(raw_root.glob("*.h5ad"))
    if expected_raw_files is not None and len(files) != expected_raw_files:
        raise ValueError(f"expected {expected_raw_files} raw H5ADs, found {len(files)}")
    if not files:
        raise ValueError(f"no raw H5ADs found under {raw_root}")
    sample_to_file: dict[str, Path] = {}
    gene_reference: np.ndarray | None = None
    for path in files:
        with h5py.File(path, "r") as handle:
            sample_categories = _categorical_categories(handle["obs/sample.sampleKitGuid"])
            gene_key = "var/gene_symbols" if "var/gene_symbols" in handle else "var/gene_symbol"
            genes = _decode_array(handle[gene_key][:])
        for sample in sample_categories:
            if sample in sample_to_file:
                raise ValueError(f"sample occurs in multiple raw files: {sample}")
            sample_to_file[str(sample)] = path
        if gene_reference is None:
            gene_reference = genes
        elif not np.array_equal(gene_reference, genes):
            raise ValueError("raw Sound H5AD gene orders differ")
    if gene_reference is None:
        raise RuntimeError("raw H5AD inventory did not contain genes")
    return files, sample_to_file, gene_reference

def _sample_rows(
    sample_codes: np.ndarray,
    sample_code: int,
    *,
    cells: int,
    seed: int,
) -> np.ndarray:
    rows = np.flatnonzero(sample_codes == sample_code)
    if len(rows) < cells:
        raise ValueError(f"sample has {len(rows)} cells; {cells} required")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=cells, replace=False)).astype(np.int64)

def _read_csr_rows(
    handle: h5py.File, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    local_indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    source_indptr = handle["X/indptr"]
    source_data = handle["X/data"]
    source_indices = handle["X/indices"]
    for offset, row in enumerate(rows):
        start = int(source_indptr[int(row)])
        stop = int(source_indptr[int(row) + 1])
        values = np.asarray(source_data[start:stop], dtype=np.float32)
        columns = np.asarray(source_indices[start:stop], dtype=np.int64)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError("raw Sound expression is not finite non-negative counts")
        data_chunks.append(values)
        index_chunks.append(columns)
        local_indptr[offset + 1] = local_indptr[offset] + len(values)
    data = np.concatenate(data_chunks) if data_chunks else np.empty(0, dtype=np.float32)
    indices = (
        np.concatenate(index_chunks) if index_chunks else np.empty(0, dtype=np.int64)
    )
    return data, indices, local_indptr

def _append(dataset: h5py.Dataset, values: np.ndarray) -> tuple[int, int]:
    start = len(dataset)
    stop = start + len(values)
    dataset.resize((stop,))
    dataset[start:stop] = values
    return start, stop

def build_delta_raw_counts(
    *,
    task_units_path: Path,
    base_visit_manifest_path: Path,
    raw_root: Path,
    output_h5: Path,
    delta_manifest_path: Path,
    audit_path: Path,
    cells_per_visit: int = 512,
    seed: int = 42,
    expected_units: int | None = 163,
    expected_full_visits: int | None = 489,
    expected_base_visits: int | None = 321,
    expected_delta_visits: int | None = 168,
    expected_raw_files: int | None = 8,
    resume: bool = False,
) -> dict[str, object]:
    """Materialize raw counts and pseudobulk for visits absent from base321."""

    if cells_per_visit < 1:
        raise ValueError("cells_per_visit must be positive")
    full, base, delta = derive_visit_contract(
        task_units_path,
        base_visit_manifest_path,
        expected_units=expected_units,
        expected_full_visits=expected_full_visits,
        expected_base_visits=expected_base_visits,
        expected_delta_visits=expected_delta_visits,
    )
    _, sample_to_file, genes = _raw_inventory(
        raw_root, expected_raw_files=expected_raw_files
    )
    missing = sorted(set(delta["sample_id"]) - set(sample_to_file))
    if missing:
        raise ValueError(f"delta samples missing from raw H5ADs: {missing[:5]}")
    outputs = {"delta_raw_h5": output_h5, "delta_manifest": delta_manifest_path}
    if resume:
        cached = _resume_valid(
            audit_path,
            schema=DELTA_RAW_SCHEMA,
            output_paths=outputs,
        )
        if cached is not None:
            return cached

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_h5.with_suffix(output_h5.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    manifest_records: list[dict[str, object]] = []
    n_genes = len(genes)
    total_cells = len(delta) * cells_per_visit
    with h5py.File(temporary, "w") as destination:
        destination.attrs.update(
            {
                "schema_version": DELTA_RAW_SCHEMA,
                "status": "building",
                "seed": seed,
                "cells_per_visit": cells_per_visit,
                "n_visits": len(delta),
                "n_cells": total_cells,
                "n_genes": n_genes,
                "raw_counts": True,
                "base_artifacts_modified": False,
            }
        )
        matrix = destination.create_group("matrix")
        data_out = matrix.create_dataset(
            "data", shape=(0,), maxshape=(None,), chunks=(1_048_576,), dtype="f4"
        )
        indices_out = matrix.create_dataset(
            "indices", shape=(0,), maxshape=(None,), chunks=(1_048_576,), dtype="i8"
        )
        indptr_values = [0]
        obs = destination.create_group("obs")
        obs_values: dict[str, list[object]] = {
            "donor_id": [],
            "person_id": [],
            "sample_id": [],
            "season": [],
            "nominal_day": [],
            "raw_h5ad": [],
            "raw_row_index": [],
        }
        visit_group = destination.create_group("visit")
        pseudobulk = destination.create_group("pseudobulk")
        sum_counts = pseudobulk.create_dataset(
            "sum_counts",
            shape=(len(delta), n_genes),
            chunks=(1, min(n_genes, 4096)),
            dtype="u4",
        )
        total_umi = pseudobulk.create_dataset("total_umi", shape=(len(delta),), dtype="u8")
        n_cells_out = pseudobulk.create_dataset("n_cells", shape=(len(delta),), dtype="i4")

        for raw_path_string, frame in delta.assign(
            raw_h5ad=delta["sample_id"].map(sample_to_file).astype(str)
        ).groupby("raw_h5ad", sort=True):
            raw_path = Path(raw_path_string)
            with h5py.File(raw_path, "r", rdcc_nbytes=256 * 1024**2) as source:
                sample_group = source["obs/sample.sampleKitGuid"]
                categories = _categorical_categories(sample_group)
                category_lookup = {sample: index for index, sample in enumerate(categories)}
                sample_codes = np.asarray(sample_group["codes"][:], dtype=np.int64)
                for row in frame.sort_values("store_index", kind="stable").itertuples(index=False):
                    sample = str(row.sample_id)
                    if sample not in category_lookup:
                        raise ValueError(f"raw sample category missing: {sample}")
                    sampling_seed = _stable_seed(
                        "immunetwin_full_visit_delta",
                        sample,
                        row.nominal_day,
                        base_seed=seed,
                    )
                    selected_rows = _sample_rows(
                        sample_codes,
                        category_lookup[sample],
                        cells=cells_per_visit,
                        seed=sampling_seed,
                    )
                    values, columns, local_indptr = _read_csr_rows(source, selected_rows)
                    data_start, data_stop = _append(data_out, values)
                    index_start, index_stop = _append(indices_out, columns)
                    if (data_start, data_stop) != (index_start, index_stop):
                        raise RuntimeError("CSR data and indices diverged")
                    indptr_values.extend((local_indptr[1:] + data_start).tolist())
                    counts = np.bincount(columns, weights=values, minlength=n_genes)
                    if counts.max(initial=0) > np.iinfo(np.uint32).max:
                        raise OverflowError("visit pseudobulk exceeds uint32")
                    sum_counts[int(row.store_index)] = counts.astype(np.uint32)
                    total_umi[int(row.store_index)] = np.uint64(round(float(values.sum())))
                    n_cells_out[int(row.store_index)] = cells_per_visit
                    repeat = cells_per_visit
                    obs_values["donor_id"].extend([str(row.visit_id)] * repeat)
                    obs_values["person_id"].extend([str(row.donor_id)] * repeat)
                    obs_values["sample_id"].extend([sample] * repeat)
                    obs_values["season"].extend([str(row.season)] * repeat)
                    obs_values["nominal_day"].extend([str(row.nominal_day)] * repeat)
                    obs_values["raw_h5ad"].extend([str(raw_path.resolve())] * repeat)
                    obs_values["raw_row_index"].extend(selected_rows.tolist())
                    record = row._asdict()
                    record.update(
                        {
                            "raw_h5ad": str(raw_path.resolve()),
                            "cells": repeat,
                            "sampling_seed": int(sampling_seed),
                            "selected_nnz": int(len(values)),
                        }
                    )
                    manifest_records.append(record)

        final_indptr = np.asarray(indptr_values, dtype=np.int64)
        if len(final_indptr) != total_cells + 1 or final_indptr[-1] != len(data_out):
            raise RuntimeError("delta CSR final shape is inconsistent")
        matrix.create_dataset("indptr", data=final_indptr, dtype="i8")
        matrix.attrs["shape"] = np.asarray([total_cells, n_genes], dtype=np.int64)
        var = destination.create_group("var")
        var.create_dataset("gene_symbol", data=genes.astype(object), dtype=string_dtype)
        for key, values in obs_values.items():
            if key == "raw_row_index":
                obs.create_dataset(key, data=np.asarray(values, dtype=np.int64), dtype="i8")
            else:
                obs.create_dataset(key, data=np.asarray(values, dtype=object), dtype=string_dtype)
        manifest = pd.DataFrame(manifest_records).sort_values("store_index", kind="stable")
        for column in (
            "visit_id",
            "sample_id",
            "donor_id",
            "season",
            "nominal_day",
            "raw_h5ad",
        ):
            visit_group.create_dataset(
                column,
                data=manifest[column].astype(str).to_numpy(dtype=object),
                dtype=string_dtype,
            )
        for column in (
            "store_index",
            "outer_fold",
            "nominal_day_value",
            "cells",
            "sampling_seed",
            "selected_nnz",
        ):
            visit_group.create_dataset(
                column, data=pd.to_numeric(manifest[column]).to_numpy(dtype=np.int64)
            )
        destination.attrs["status"] = "complete"
        destination.flush()
    os.replace(temporary, output_h5)
    manifest = pd.DataFrame(manifest_records).sort_values("store_index", kind="stable")
    _atomic_table(manifest, delta_manifest_path)
    audit: dict[str, object] = {
        "status": "pass",
        "schema_version": DELTA_RAW_SCHEMA,
        "resumed": False,
        "outputs": {name: _artifact(path) for name, path in outputs.items()},
        "n_task_units": len(full) // 3,
        "n_full_visits": len(full),
        "n_base_visits": len(base),
        "n_delta_visits": len(delta),
        "n_delta_cells": total_cells,
        "n_genes": n_genes,
        "cells_per_visit": cells_per_visit,
        "sampling_replacement": False,
        "sampling_uses_other_visits": False,
        "gene_statistics_across_visits_computed": False,
        "base_artifacts_modified": False,
    }
    _atomic_json(audit_path, audit)
    return audit

def _uce_inputs(uce_root: Path) -> None:
    directories = sorted(path for path in uce_root.glob("group_*") if path.is_dir())
    if not directories:
        raise ValueError(f"no UCE groups under {uce_root}")
    for directory in directories:
        marker = directory / "COMPLETE.json"
        embedding = directory / "cell_embeddings.npz"
        if not marker.is_file() or not embedding.is_file():
            raise ValueError(f"incomplete UCE group: {directory}")
        record = expand_paths(json.loads(marker.read_text()))
        if record.get("status") != "PASS":
            raise ValueError(f"UCE marker contract failed: {directory}")

def prepare_delta_uce(
    *,
    delta_manifest_path: Path,
    uce_root: Path,
    output_h5: Path,
    audit_path: Path,
    cells_per_visit: int = 512,
    feature_dim: int = 1280,
    group_cells: int = 10_240,
    expected_delta_visits: int | None = 168,
    resume: bool = False,
) -> dict[str, object]:
    """Consolidate UCE group outputs."""

    manifest = pd.read_csv(delta_manifest_path, sep="\t").sort_values(
        "store_index", kind="stable"
    )
    required = {"visit_id", "store_index", "cells"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"delta manifest missing columns: {sorted(required - set(manifest))}")
    if expected_delta_visits is not None and len(manifest) != expected_delta_visits:
        raise ValueError("delta manifest visit count changed")
    if manifest["visit_id"].astype(str).duplicated().any():
        raise ValueError("delta manifest contains duplicate visit IDs")
    if not np.array_equal(
        pd.to_numeric(manifest["store_index"]).to_numpy(dtype=np.int64),
        np.arange(len(manifest), dtype=np.int64),
    ):
        raise ValueError("delta store_index must be contiguous from zero")
    if not pd.to_numeric(manifest["cells"]).astype(int).eq(cells_per_visit).all():
        raise ValueError("delta manifest cell count differs from UCE contract")
    directories = sorted(path for path in uce_root.glob("group_*") if path.is_dir())
    _uce_inputs(uce_root)
    expected_groups = math.ceil(len(manifest) * cells_per_visit / group_cells)
    if len(directories) != expected_groups:
        raise ValueError(f"expected {expected_groups} UCE groups, found {len(directories)}")
    outputs = {"delta_uce_h5": output_h5}
    if resume:
        cached = _resume_valid(
            audit_path,
            schema=DELTA_UCE_SCHEMA,
            output_paths=outputs,
        )
        if cached is not None:
            return cached

    chunks: list[np.ndarray] = []
    identifiers: list[np.ndarray] = []
    for directory in directories:
        with np.load(directory / "cell_embeddings.npz", allow_pickle=False) as payload:
            if set(payload.files) != {"embeddings", "donor_id"}:
                raise ValueError(f"unexpected UCE archive members: {directory}")
            values = np.asarray(payload["embeddings"], dtype=np.float32)
            visit_ids = payload["donor_id"].astype(str)
        if values.shape != (len(visit_ids), feature_dim) or not np.isfinite(values).all():
            raise ValueError(f"invalid UCE matrix: {directory}")
        chunks.append(values)
        identifiers.append(visit_ids)
    cells = np.vstack(chunks)
    cell_visits = np.concatenate(identifiers)
    expected_cells = len(manifest) * cells_per_visit
    if cells.shape != (expected_cells, feature_dim):
        raise ValueError(f"delta UCE shape changed: {cells.shape}")
    expected_visit_ids = manifest["visit_id"].astype(str).tolist()
    positions = {visit: np.flatnonzero(cell_visits == visit) for visit in expected_visit_ids}
    wrong = {visit: len(rows) for visit, rows in positions.items() if len(rows) != cells_per_visit}
    if wrong or set(np.unique(cell_visits)) != set(expected_visit_ids):
        raise ValueError(f"UCE visit membership changed: {wrong}")

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_h5.with_suffix(output_h5.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as destination:
        destination.attrs.update(
            {
                "schema_version": DELTA_UCE_SCHEMA,
                "status": "building",
                "n_visits": len(manifest),
                "cells_per_visit": cells_per_visit,
                "feature_dim": feature_dim,
                "base_artifacts_modified": False,
            }
        )
        embedding = destination.create_dataset(
            "embedding",
            shape=(len(manifest), cells_per_visit, feature_dim),
            chunks=(1, min(cells_per_visit, 64), feature_dim),
            dtype="f4",
        )
        means = destination.create_dataset(
            "mean_embedding", shape=(len(manifest), feature_dim), dtype="f4"
        )
        for index, visit in enumerate(expected_visit_ids):
            selected = cells[positions[visit]]
            embedding[index] = selected
            means[index] = selected.mean(axis=0)
        string_dtype = h5py.string_dtype("utf-8")
        destination.create_dataset(
            "visit_id", data=np.asarray(expected_visit_ids, dtype=object), dtype=string_dtype
        )
        destination.attrs["status"] = "complete"
        destination.flush()
    os.replace(temporary, output_h5)
    audit: dict[str, object] = {
        "status": "pass",
        "schema_version": DELTA_UCE_SCHEMA,
        "resumed": False,
        "outputs": {name: _artifact(path) for name, path in outputs.items()},
        "n_visits": len(manifest),
        "n_cells": expected_cells,
        "cells_per_visit": cells_per_visit,
        "feature_dim": feature_dim,
        "base_artifacts_modified": False,
    }
    _atomic_json(audit_path, audit)
    return audit

def _visit_sums_from_csr_store(
    path: Path,
    visit_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream visit sums from a raw-count store without materializing all cells."""

    with h5py.File(path, "r") as handle:
        if "var/gene_symbol" in handle:
            genes = _decode_array(handle["var/gene_symbol"][:])
        elif "var/gene_symbols" in handle:
            genes = _decode_array(handle["var/gene_symbols"][:])
        else:
            raise ValueError(f"raw-count store lacks gene symbols: {path}")
        observed_visits = _decode_array(handle["obs/donor_id"][:])
        indptr = handle["matrix/indptr"]
        data = handle["matrix/data"]
        indices = handle["matrix/indices"]
        sums = np.zeros((len(visit_ids), len(genes)), dtype=np.uint32)
        n_cells = np.zeros(len(visit_ids), dtype=np.int32)
        total_umi = np.zeros(len(visit_ids), dtype=np.uint64)
        for output_index, visit in enumerate(visit_ids):
            rows = np.flatnonzero(observed_visits == str(visit))
            if not len(rows):
                raise ValueError(f"visit missing from raw-count store: {visit}")
            n_cells[output_index] = len(rows)
            chunks: list[np.ndarray] = []
            column_chunks: list[np.ndarray] = []
            # Visits are stored contiguously, so each visit is read in one HDF5 slice.
            boundaries = np.flatnonzero(np.diff(rows) != 1) + 1
            for run in np.split(rows, boundaries):
                start = int(indptr[int(run[0])])
                stop = int(indptr[int(run[-1]) + 1])
                chunks.append(np.asarray(data[start:stop], dtype=np.float32))
                column_chunks.append(np.asarray(indices[start:stop], dtype=np.int64))
            values = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
            columns = (
                np.concatenate(column_chunks)
                if column_chunks
                else np.empty(0, dtype=np.int64)
            )
            counts = np.bincount(columns, weights=values, minlength=len(genes))
            if counts.max(initial=0) > np.iinfo(np.uint32).max:
                raise OverflowError("visit pseudobulk exceeds uint32")
            sums[output_index] = counts.astype(np.uint32)
            total_umi[output_index] = np.uint64(round(float(values.sum())))
    return genes, sums, n_cells, total_umi

def build_gene_summary(
    *,
    task_units_path: Path,
    base_visit_manifest_path: Path,
    delta_manifest_path: Path,
    base_raw_h5: Path,
    delta_raw_h5: Path,
    output_h5: Path,
    audit_path: Path,
    expected_units: int | None = 163,
    expected_full_visits: int | None = 489,
    expected_base_visits: int | None = 321,
    expected_delta_visits: int | None = 168,
    resume: bool = False,
) -> dict[str, object]:
    """Create an all-visit gene summary."""

    full, base, expected_delta = derive_visit_contract(
        task_units_path,
        base_visit_manifest_path,
        expected_units=expected_units,
        expected_full_visits=expected_full_visits,
        expected_base_visits=expected_base_visits,
        expected_delta_visits=expected_delta_visits,
    )
    delta = pd.read_csv(delta_manifest_path, sep="\t").sort_values(
        "store_index", kind="stable"
    )
    if set(delta["sample_id"].astype(str)) != set(expected_delta["sample_id"]):
        raise ValueError("materialized delta samples differ from derived delta")
    outputs = {"gene_summary_h5": output_h5}
    if resume:
        cached = _resume_valid(
            audit_path,
            schema=GENE_SUMMARY_SCHEMA,
            output_paths=outputs,
        )
        if cached is not None:
            return cached

    base_ids = base["store_visit_id"].astype(str).tolist()
    base_genes, base_sums, base_cells, base_umi = _visit_sums_from_csr_store(
        base_raw_h5, base_ids
    )
    with h5py.File(delta_raw_h5, "r") as handle:
        if handle.attrs.get("status") != "complete":
            raise ValueError("delta raw-count store is incomplete")
        delta_genes = _decode_array(handle["var/gene_symbol"][:])
        delta_ids = _decode_array(handle["visit/visit_id"][:])
        delta_sums = np.asarray(handle["pseudobulk/sum_counts"][:], dtype=np.uint32)
        delta_cells = np.asarray(handle["pseudobulk/n_cells"][:], dtype=np.int32)
        delta_umi = np.asarray(handle["pseudobulk/total_umi"][:], dtype=np.uint64)
    if not np.array_equal(base_genes, delta_genes):
        raise ValueError("base and delta gene orders differ")
    if delta_ids.tolist() != delta["visit_id"].astype(str).tolist():
        raise ValueError("delta pseudobulk visit order differs from manifest")

    base_by_sample = {
        str(row.sample_id): int(row.store_index) for row in base.itertuples(index=False)
    }
    delta_by_sample = {
        str(row.sample_id): int(row.store_index) for row in delta.itertuples(index=False)
    }
    all_sums = np.zeros((len(full), len(base_genes)), dtype=np.uint32)
    all_cells = np.zeros(len(full), dtype=np.int32)
    all_umi = np.zeros(len(full), dtype=np.uint64)
    stores: list[str] = []
    for row in full.itertuples(index=False):
        sample = str(row.sample_id)
        target_index = int(row.full_index)
        if sample in base_by_sample:
            source_index = base_by_sample[sample]
            all_sums[target_index] = base_sums[source_index]
            all_cells[target_index] = base_cells[source_index]
            all_umi[target_index] = base_umi[source_index]
            stores.append("base321")
        else:
            source_index = delta_by_sample[sample]
            all_sums[target_index] = delta_sums[source_index]
            all_cells[target_index] = delta_cells[source_index]
            all_umi[target_index] = delta_umi[source_index]
            stores.append("delta")

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_h5.with_suffix(output_h5.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(temporary, "w") as destination:
        destination.attrs.update(
            {
                "schema_version": GENE_SUMMARY_SCHEMA,
                "status": "complete",
                "n_visits": len(full),
                "n_genes": len(base_genes),
                "normalization": "none_raw_sum_counts",
                "all_available_cells": False,
                "base_artifacts_modified": False,
            }
        )
        destination.create_dataset(
            "sum_counts",
            data=all_sums,
            chunks=(1, min(len(base_genes), 4096)),
            dtype="u4",
        )
        destination.create_dataset("total_umi", data=all_umi, dtype="u8")
        destination.create_dataset("n_cells", data=all_cells, dtype="i4")
        destination.create_dataset(
            "visit_id", data=full["visit_id"].to_numpy(dtype=object), dtype=string_dtype
        )
        destination.create_dataset(
            "sample_id", data=full["sample_id"].to_numpy(dtype=object), dtype=string_dtype
        )
        destination.create_dataset(
            "source_store", data=np.asarray(stores, dtype=object), dtype=string_dtype
        )
        destination.create_group("var").create_dataset(
            "gene_symbol", data=base_genes.astype(object), dtype=string_dtype
        )
        destination.flush()
    os.replace(temporary, output_h5)
    audit: dict[str, object] = {
        "status": "pass",
        "schema_version": GENE_SUMMARY_SCHEMA,
        "resumed": False,
        "outputs": {name: _artifact(path) for name, path in outputs.items()},
        "n_visits": len(full),
        "n_base_visits": len(base),
        "n_delta_visits": len(delta),
        "n_genes": len(base_genes),
        "normalization": "none_raw_sum_counts",
        "all_available_cells": False,
        "cross_visit_gene_selection": False,
        "base_artifacts_modified": False,
    }
    _atomic_json(audit_path, audit)
    return audit

def _all_cell_sample_summary(
    handle: h5py.File,
    rows: np.ndarray,
    *,
    n_genes: int,
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Aggregate one sample in bounded row chunks from an on-disk CSR matrix."""

    if not len(rows):
        raise ValueError("cannot summarize an empty physical visit")
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be positive")
    rows = np.asarray(rows, dtype=np.int64)
    if not np.all(rows[1:] > rows[:-1]):
        raise ValueError("physical visit rows must be strictly increasing")
    indptr = handle["X/indptr"]
    data_source = handle["X/data"]
    index_source = handle["X/indices"]
    sums = np.zeros(n_genes, dtype=np.uint64)
    detected = np.zeros(n_genes, dtype=np.uint64)
    boundaries = np.flatnonzero(np.diff(rows) != 1) + 1
    for run in np.split(rows, boundaries):
        for offset in range(0, len(run), row_chunk_size):
            block_rows = run[offset : offset + row_chunk_size]
            first = int(block_rows[0])
            last = int(block_rows[-1])
            data_start = int(indptr[first])
            data_stop = int(indptr[last + 1])
            values = np.asarray(data_source[data_start:data_stop], dtype=np.float64)
            columns = np.asarray(index_source[data_start:data_stop], dtype=np.int64)
            local_indptr = np.asarray(indptr[first : last + 2], dtype=np.int64) - data_start
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or not np.equal(values, np.floor(values)).all()
            ):
                raise ValueError("raw Sound expression is not finite non-negative counts")
            block = sparse.csr_matrix(
                (values, columns, local_indptr),
                shape=(len(block_rows), n_genes),
            )
            block.sum_duplicates()
            block.eliminate_zeros()
            block_sum = np.asarray(block.sum(axis=0), dtype=np.float64).ravel()
            if not np.equal(block_sum, np.floor(block_sum)).all():
                raise ValueError("all-cell pseudobulk lost integer count semantics")
            sums += block_sum.astype(np.uint64)
            detected += np.asarray(block.getnnz(axis=0), dtype=np.uint64)
    if np.any(detected > len(rows)):
        raise RuntimeError("gene detection count exceeds visit cell count")
    return sums, detected, int(sums.sum(dtype=np.uint64))

def build_full_gene_summary(
    *,
    task_units_path: Path,
    raw_root: Path,
    output_h5: Path,
    audit_path: Path,
    expected_units: int | None = 163,
    expected_full_visits: int | None = 489,
    expected_raw_files: int | None = 8,
    row_chunk_size: int = 1024,
    resume: bool = False,
) -> dict[str, object]:
    """Sum every available cell in each of the 489 Sound visits, one bounded CSR
    row block at a time.
    """

    full = derive_full_visits(
        task_units_path,
        expected_units=expected_units,
        expected_full_visits=expected_full_visits,
    )
    _, sample_to_file, genes = _raw_inventory(
        raw_root, expected_raw_files=expected_raw_files
    )
    missing = sorted(set(full["sample_id"]) - set(sample_to_file))
    if missing:
        raise ValueError(f"full visits missing from raw H5ADs: {missing[:5]}")
    visits_with_paths = full.assign(
        raw_h5ad=full["sample_id"].map(sample_to_file).map(
            lambda path: str(Path(path).resolve())
        )
    )
    outputs = {"full_gene_summary_h5": output_h5}
    if resume:
        cached = _resume_valid(
            audit_path,
            schema=FULL_GENE_SUMMARY_SCHEMA,
            output_paths=outputs,
        )
        if cached is not None:
            return cached

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_h5.with_suffix(output_h5.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    n_genes = len(genes)
    cell_counts = np.zeros(len(full), dtype=np.int64)
    total_umis = np.zeros(len(full), dtype=np.uint64)
    written = np.zeros(len(full), dtype=bool)
    with h5py.File(temporary, "w") as destination:
        destination.attrs.update(
            {
                "schema_version": FULL_GENE_SUMMARY_SCHEMA,
                "status": "building",
                "n_visits": len(full),
                "n_genes": n_genes,
                "normalization": "none_raw_sum_counts",
                "all_available_cells": True,
                "row_chunk_size": row_chunk_size,
                "cross_visit_gene_selection": False,
            }
        )
        sum_counts = destination.create_dataset(
            "sum_counts",
            shape=(len(full), n_genes),
            chunks=(1, min(n_genes, 4096)),
            dtype="u8",
        )
        detection_fraction = destination.create_dataset(
            "detection_fraction",
            shape=(len(full), n_genes),
            chunks=(1, min(n_genes, 4096)),
            dtype="f4",
        )
        n_cells_out = destination.create_dataset("n_cells", shape=(len(full),), dtype="i8")
        total_umi_out = destination.create_dataset("total_umi", shape=(len(full),), dtype="u8")
        destination.create_dataset(
            "visit_id", data=full["visit_id"].to_numpy(dtype=object), dtype=string_dtype
        )
        destination.create_dataset(
            "sample_id", data=full["sample_id"].to_numpy(dtype=object), dtype=string_dtype
        )
        destination.create_dataset(
            "donor_id", data=full["donor_id"].to_numpy(dtype=object), dtype=string_dtype
        )
        destination.create_dataset(
            "season", data=full["season"].to_numpy(dtype=object), dtype=string_dtype
        )
        destination.create_dataset(
            "nominal_day", data=full["nominal_day"].to_numpy(dtype=object), dtype=string_dtype
        )
        destination.create_dataset(
            "nominal_day_value",
            data=full["nominal_day_value"].to_numpy(dtype=np.int16),
            dtype="i2",
        )
        destination.create_dataset(
            "outer_fold", data=full["outer_fold"].to_numpy(dtype=np.int8), dtype="i1"
        )
        destination.create_dataset(
            "raw_h5ad",
            data=visits_with_paths["raw_h5ad"].to_numpy(dtype=object),
            dtype=string_dtype,
        )
        destination.create_group("var").create_dataset(
            "gene_symbol", data=genes.astype(object), dtype=string_dtype
        )

        for raw_path_string, frame in visits_with_paths.groupby("raw_h5ad", sort=True):
            raw_path = Path(raw_path_string)
            with h5py.File(raw_path, "r", rdcc_nbytes=256 * 1024**2) as source:
                sample_group = source["obs/sample.sampleKitGuid"]
                categories = _categorical_categories(sample_group)
                category_lookup = {sample: index for index, sample in enumerate(categories)}
                sample_codes = np.asarray(sample_group["codes"][:], dtype=np.int64)
                pending: list[tuple[int, object, np.ndarray]] = []
                for row in frame.itertuples(index=False):
                    sample = str(row.sample_id)
                    if sample not in category_lookup:
                        raise ValueError(f"raw sample category missing: {sample}")
                    rows = np.flatnonzero(sample_codes == category_lookup[sample]).astype(np.int64)
                    if not len(rows):
                        raise ValueError(f"raw physical visit is empty: {sample}")
                    pending.append((int(rows[0]), row, rows))
                # Preserve mostly sequential raw-HDF5 access even though the
                # canonical output is donor/season/day ordered.
                for _, row, rows in sorted(pending, key=lambda value: value[0]):
                    counts, detected, total_umi = _all_cell_sample_summary(
                        source,
                        rows,
                        n_genes=n_genes,
                        row_chunk_size=row_chunk_size,
                    )
                    index = int(row.full_index)
                    sum_counts[index] = counts
                    detection_fraction[index] = (
                        detected.astype(np.float64) / float(len(rows))
                    ).astype(np.float32)
                    n_cells_out[index] = len(rows)
                    total_umi_out[index] = total_umi
                    cell_counts[index] = len(rows)
                    total_umis[index] = total_umi
                    written[index] = True
        if not written.all():
            raise RuntimeError("not every physical visit received an all-cell summary")
        destination.attrs.update(
            {
                "status": "complete",
                "total_cells": int(cell_counts.sum()),
                "min_cells_per_visit": int(cell_counts.min()),
                "max_cells_per_visit": int(cell_counts.max()),
            }
        )
        destination.flush()
    os.replace(temporary, output_h5)
    audit: dict[str, object] = {
        "status": "pass",
        "schema_version": FULL_GENE_SUMMARY_SCHEMA,
        "resumed": False,
        "outputs": {name: _artifact(path) for name, path in outputs.items()},
        "n_visits": len(full),
        "n_genes": n_genes,
        "all_available_cells": True,
        "total_cells": int(cell_counts.sum()),
        "min_cells_per_visit": int(cell_counts.min()),
        "median_cells_per_visit": float(np.median(cell_counts)),
        "max_cells_per_visit": int(cell_counts.max()),
        "min_total_umi_per_visit": int(total_umis.min()),
        "max_total_umi_per_visit": int(total_umis.max()),
        "row_chunk_size": row_chunk_size,
        "normalization": "none_raw_sum_counts",
        "detection_value": "fraction_of_all_visit_cells_with_nonzero_raw_count",
        "cross_visit_gene_selection": False,
        "all_cell_matrix_materialized": False,
    }
    _atomic_json(audit_path, audit)
    return audit

def _h5_visit_ids(path: Path, key: str) -> list[str]:
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("status") != "complete":
            raise ValueError(f"incomplete feature store: {path}")
        return _decode_array(handle[key][:]).tolist()

def build_unified_manifest(
    *,
    task_units_path: Path,
    base_visit_manifest_path: Path,
    delta_manifest_path: Path,
    base_raw_h5: Path,
    base_uce_h5: Path,
    delta_raw_h5: Path,
    delta_uce_h5: Path,
    gene_summary_h5: Path,
    output_manifest_path: Path,
    audit_path: Path,
    expected_units: int | None = 163,
    expected_full_visits: int | None = 489,
    expected_base_visits: int | None = 321,
    expected_delta_visits: int | None = 168,
    resume: bool = False,
) -> dict[str, object]:
    """Combine the base321 and delta168 stores through a visit manifest."""

    full, base, expected_delta = derive_visit_contract(
        task_units_path,
        base_visit_manifest_path,
        expected_units=expected_units,
        expected_full_visits=expected_full_visits,
        expected_base_visits=expected_base_visits,
        expected_delta_visits=expected_delta_visits,
    )
    delta = pd.read_csv(delta_manifest_path, sep="\t").sort_values(
        "store_index", kind="stable"
    )
    if set(delta["sample_id"].astype(str)) != set(expected_delta["sample_id"]):
        raise ValueError("delta manifest differs from the derived missing visits")
    outputs = {"unified_manifest": output_manifest_path}
    if resume:
        cached = _resume_valid(
            audit_path,
            schema=UNIFIED_MANIFEST_SCHEMA,
            output_paths=outputs,
        )
        if cached is not None:
            return cached

    base_store_ids = base["store_visit_id"].astype(str).tolist()
    if _h5_visit_ids(base_uce_h5, "visit_id") != base_store_ids:
        raise ValueError("base UCE visit order differs from its sealed manifest")
    delta_store_ids = delta["visit_id"].astype(str).tolist()
    if _h5_visit_ids(delta_raw_h5, "visit/visit_id") != delta_store_ids:
        raise ValueError("delta raw visit order differs from delta manifest")
    if _h5_visit_ids(delta_uce_h5, "visit_id") != delta_store_ids:
        raise ValueError("delta UCE visit order differs from delta manifest")
    if _h5_visit_ids(gene_summary_h5, "visit_id") != full["visit_id"].astype(str).tolist():
        raise ValueError("gene summary visit order differs from full contract")
    with h5py.File(gene_summary_h5, "r") as handle:
        if handle.attrs.get("all_available_cells", False) not in (True, np.bool_(True)):
            raise ValueError("unified manifest requires an all-available-cell gene summary")
    with h5py.File(base_raw_h5, "r") as handle:
        if int(handle.attrs.get("n_visits", -1)) != len(base):
            raise ValueError("base raw store visit count differs from sealed manifest")

    base_by_sample = {
        str(row.sample_id): row for row in base.itertuples(index=False)
    }
    delta_by_sample = {
        str(row.sample_id): row for row in delta.itertuples(index=False)
    }
    records: list[dict[str, object]] = []
    for row in full.itertuples(index=False):
        sample = str(row.sample_id)
        if sample in base_by_sample:
            source = base_by_sample[sample]
            store = "base321"
            raw_path = base_raw_h5
            uce_path = base_uce_h5
        else:
            source = delta_by_sample[sample]
            store = "delta"
            raw_path = delta_raw_h5
            uce_path = delta_uce_h5
        records.append(
            {
                "full_index": int(row.full_index),
                "visit_id": str(row.visit_id),
                "donor_id": str(row.donor_id),
                "season": str(row.season),
                "outer_fold": int(row.outer_fold),
                "nominal_day": str(row.nominal_day),
                "nominal_day_value": int(row.nominal_day_value),
                "sample_id": sample,
                "feature_store": store,
                "store_index": int(source.store_index),
                "store_visit_id": str(source.store_visit_id),
                "raw_counts_h5": str(raw_path.resolve()),
                "uce_features_h5": str(uce_path.resolve()),
                "gene_summary_h5": str(gene_summary_h5.resolve()),
                "gene_summary_index": int(row.full_index),
                "cells_per_visit": int(getattr(source, "cells", 0)),
            }
        )
    unified = pd.DataFrame(records).sort_values("full_index", kind="stable")
    if unified["visit_id"].duplicated().any() or unified["sample_id"].duplicated().any():
        raise RuntimeError("unified manifest lost visit/sample uniqueness")
    if set(unified["feature_store"]) != {"base321", "delta"}:
        raise RuntimeError("unified manifest is missing a feature store")
    _atomic_table(unified, output_manifest_path)
    audit: dict[str, object] = {
        "status": "pass",
        "schema_version": UNIFIED_MANIFEST_SCHEMA,
        "resumed": False,
        "outputs": {name: _artifact(path) for name, path in outputs.items()},
        "n_visits": len(unified),
        "n_base_visits": int(unified["feature_store"].eq("base321").sum()),
        "n_delta_visits": int(unified["feature_store"].eq("delta").sum()),
        "n_unique_samples": int(unified["sample_id"].nunique()),
        "n_unique_donors": int(unified["donor_id"].nunique()),
        "base_artifacts_modified": False,
        "physical_concatenation_performed": False,
    }
    _atomic_json(audit_path, audit)
    return audit

def _expected_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-units", type=int, default=163)
    parser.add_argument("--expected-full-visits", type=int, default=489)
    parser.add_argument("--expected-base-visits", type=int, default=321)
    parser.add_argument("--expected-delta-visits", type=int, default=168)

def _common_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-units", required=True, type=Path)
    parser.add_argument("--base-visit-manifest", required=True, type=Path)
    _expected_arguments(parser)
    parser.add_argument("--resume", action="store_true")

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    raw = commands.add_parser("build-delta")
    _common_contract_arguments(raw)
    raw.add_argument("--raw-root", required=True, type=Path)
    raw.add_argument("--output-h5", required=True, type=Path)
    raw.add_argument("--delta-manifest", required=True, type=Path)
    raw.add_argument("--audit", required=True, type=Path)
    raw.add_argument("--cells-per-visit", type=int, default=512)
    raw.add_argument("--seed", type=int, default=42)
    raw.add_argument("--expected-raw-files", type=int, default=8)

    uce = commands.add_parser("prepare-delta-uce")
    uce.add_argument("--delta-manifest", required=True, type=Path)
    uce.add_argument("--uce-root", required=True, type=Path)
    uce.add_argument("--output-h5", required=True, type=Path)
    uce.add_argument("--audit", required=True, type=Path)
    uce.add_argument("--cells-per-visit", type=int, default=512)
    uce.add_argument("--feature-dim", type=int, default=1280)
    uce.add_argument("--group-cells", type=int, default=10_240)
    uce.add_argument("--expected-delta-visits", type=int, default=168)
    uce.add_argument("--resume", action="store_true")

    genes = commands.add_parser("build-gene-summary")
    _common_contract_arguments(genes)
    genes.add_argument("--delta-manifest", required=True, type=Path)
    genes.add_argument("--base-raw-h5", required=True, type=Path)
    genes.add_argument("--delta-raw-h5", required=True, type=Path)
    genes.add_argument("--output-h5", required=True, type=Path)
    genes.add_argument("--audit", required=True, type=Path)

    full_genes = commands.add_parser("build-full-gene-summary")
    full_genes.add_argument("--task-units", required=True, type=Path)
    full_genes.add_argument("--raw-root", required=True, type=Path)
    full_genes.add_argument("--output-h5", required=True, type=Path)
    full_genes.add_argument("--audit", required=True, type=Path)
    full_genes.add_argument("--expected-units", type=int, default=163)
    full_genes.add_argument("--expected-full-visits", type=int, default=489)
    full_genes.add_argument("--expected-raw-files", type=int, default=8)
    full_genes.add_argument("--row-chunk-size", type=int, default=1024)
    full_genes.add_argument("--resume", action="store_true")

    manifest = commands.add_parser("build-manifest")
    _common_contract_arguments(manifest)
    manifest.add_argument("--delta-manifest", required=True, type=Path)
    manifest.add_argument("--base-raw-h5", required=True, type=Path)
    manifest.add_argument("--base-uce-h5", required=True, type=Path)
    manifest.add_argument("--delta-raw-h5", required=True, type=Path)
    manifest.add_argument("--delta-uce-h5", required=True, type=Path)
    manifest.add_argument("--gene-summary-h5", required=True, type=Path)
    manifest.add_argument("--output-manifest", required=True, type=Path)
    manifest.add_argument("--audit", required=True, type=Path)
    return parser

def _expected_kwargs(args: argparse.Namespace) -> dict[str, int]:
    return {
        "expected_units": args.expected_units,
        "expected_full_visits": args.expected_full_visits,
        "expected_base_visits": args.expected_base_visits,
        "expected_delta_visits": args.expected_delta_visits,
    }

def main() -> None:
    args = _parser().parse_args()
    if args.command == "build-delta":
        result = build_delta_raw_counts(
            task_units_path=args.task_units,
            base_visit_manifest_path=args.base_visit_manifest,
            raw_root=args.raw_root,
            output_h5=args.output_h5,
            delta_manifest_path=args.delta_manifest,
            audit_path=args.audit,
            cells_per_visit=args.cells_per_visit,
            seed=args.seed,
            expected_raw_files=args.expected_raw_files,
            resume=args.resume,
            **_expected_kwargs(args),
        )
    elif args.command == "prepare-delta-uce":
        result = prepare_delta_uce(
            delta_manifest_path=args.delta_manifest,
            uce_root=args.uce_root,
            output_h5=args.output_h5,
            audit_path=args.audit,
            cells_per_visit=args.cells_per_visit,
            feature_dim=args.feature_dim,
            group_cells=args.group_cells,
            expected_delta_visits=args.expected_delta_visits,
            resume=args.resume,
        )
    elif args.command == "build-gene-summary":
        result = build_gene_summary(
            task_units_path=args.task_units,
            base_visit_manifest_path=args.base_visit_manifest,
            delta_manifest_path=args.delta_manifest,
            base_raw_h5=args.base_raw_h5,
            delta_raw_h5=args.delta_raw_h5,
            output_h5=args.output_h5,
            audit_path=args.audit,
            resume=args.resume,
            **_expected_kwargs(args),
        )
    elif args.command == "build-full-gene-summary":
        result = build_full_gene_summary(
            task_units_path=args.task_units,
            raw_root=args.raw_root,
            output_h5=args.output_h5,
            audit_path=args.audit,
            expected_units=args.expected_units,
            expected_full_visits=args.expected_full_visits,
            expected_raw_files=args.expected_raw_files,
            row_chunk_size=args.row_chunk_size,
            resume=args.resume,
        )
    else:
        result = build_unified_manifest(
            task_units_path=args.task_units,
            base_visit_manifest_path=args.base_visit_manifest,
            delta_manifest_path=args.delta_manifest,
            base_raw_h5=args.base_raw_h5,
            base_uce_h5=args.base_uce_h5,
            delta_raw_h5=args.delta_raw_h5,
            delta_uce_h5=args.delta_uce_h5,
            gene_summary_h5=args.gene_summary_h5,
            output_manifest_path=args.output_manifest,
            audit_path=args.audit,
            resume=args.resume,
            **_expected_kwargs(args),
        )
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

from __future__ import annotations
import gc
import json
import os
import pickle
from pathlib import Path
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from baseline.age.adapters.scphase_lazy_dense_loader import MASTER_SCHEMA, STORAGE_CONTRACT

CONTRACT = "official_age_cell_ranger_hvg5000_scale10_v1"

OFFICIAL_COMMIT = "cb21b1a5175f5b7630af183ac8ac1b58d5204426"


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )

def load_store(path: Path) -> tuple[sparse.csr_matrix, dict[str, np.ndarray], list[str]]:
    print(f"loading CSR store: {path}", flush=True)
    with h5py.File(path, "r") as handle:
        shape = tuple(int(value) for value in handle["matrix"].attrs["shape"])
        data = handle["matrix/data"][:]
        indices = handle["matrix/indices"][:]
        indptr = handle["matrix/indptr"][:]
        obs = {
            key: (decode(value[:]) if value.dtype.kind in {"O", "S", "U"} else value[:])
            for key, value in handle["obs"].items()
        }
        genes = decode(handle["var/gene_symbol"][:]).tolist()
    matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)
    if matrix.shape[0] != len(obs["donor_id"]) or matrix.shape[1] != len(genes):
        raise ValueError(f"invalid fixed-cell store dimensions: {path}")
    print(
        f"loaded {matrix.shape[0]} cells x {matrix.shape[1]} genes, nnz={matrix.nnz}",
        flush=True,
    )
    return matrix, obs, genes


def write_shard(path: Path, matrix: np.ndarray) -> dict[str, object]:
    matrix = np.asarray(matrix, dtype=np.float32, order="C")
    temporary = path.with_suffix(".npy.partial")
    with temporary.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "n_cells": int(matrix.shape[0]),
        "n_genes": int(matrix.shape[1]),
        "dtype": "float32",
        "bytes": int(path.stat().st_size),
    }

def write_master(
    *,
    path: Path,
    refs: list[dict[str, object]],
    donor_ids: list[str],
    labels: np.ndarray,
    metadata: dict[str, object],
    input_store: Path,
    labels_present: bool,
) -> Path:
    for ref, donor in zip(refs, donor_ids):
        ref["donor_id"] = donor
    payload = {
        "schema": MASTER_SCHEMA,
        "storage_contract": STORAGE_CONTRACT,
        "data_list_dense": refs,
        # Plain Python lists keep the descriptor pickle readable across the
        # materializer's NumPy 2 environment and scPhase's NumPy 1 environment.
        "labels": np.asarray(labels, dtype=np.float32).tolist(),
        "groups": [0] * len(donor_ids),
        "sample_ids": donor_ids,
        "adapter_metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "schema": MASTER_SCHEMA,
        "storage_contract": STORAGE_CONTRACT,
        "method": "scPhase",
        "official_repo_commit": OFFICIAL_COMMIT,
        "input_store": str(input_store.resolve()),
        "input_contract": CONTRACT,
        "output": str(path.resolve()),
        "n_donors": len(donor_ids),
        "n_cells": int(sum(int(ref["n_cells"]) for ref in refs)),
        "n_genes": 5000,
        "n_groups": 1,
        "estimated_dense_gib": float(
            sum(int(ref["n_cells"]) for ref in refs) * 5000 * 4 / 1024**3
        ),
        "total_shard_bytes": int(sum(int(ref["bytes"]) for ref in refs)),
        "loader_adapter": "runtime_symbol_patch_no_cache_lazy_dense",
        "labels_present": labels_present,
        **metadata,
    }
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path

def exact_hvg_ties(indices, dispersion, count=5000):
    """Retain the stated feature count; ties follow the fixed source gene order."""
    indices = np.asarray(indices)
    values = np.asarray(dispersion)[indices]
    if len(indices) < count or not np.isfinite(values).all():
        raise ValueError('Cannot resolve HVG ties with missing or nonfinite selected dispersions')
    selected = np.argsort(-values, kind='stable')[:count]
    return np.sort(indices[selected])

def fit_age_preprocessing(matrix, obs, genes, *, exact_hvg=False):
    """Fit the HVG and scaling recipe on the supplied cells."""
    if len(set(genes)) != len(genes):
        raise ValueError("CIMA gene identifiers are duplicated")
    batches = decode(np.asarray(obs["RNA_experiment"]))
    hvg_adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame({"RNA_experiment": pd.Categorical(batches)}),
        var=pd.DataFrame(index=pd.Index(genes, name="gene_symbol")),
    )
    print("running released Age cell_ranger HVG selection on CIMA", flush=True)
    hvg_api = (
        "scanpy.pp.highly_variable_genes(flavor=cell_ranger,n_top_genes=5000,"
        "batch_key=RNA_experiment)"
    )
    try:
        sc.pp.highly_variable_genes(
            hvg_adata,
            n_top_genes=5000,
            flavor="cell_ranger",
            min_mean=0.0125,
            max_mean=3,
            min_disp=0.5,
            batch_key="RNA_experiment",
        )
        hvg_indices = np.flatnonzero(hvg_adata.var["highly_variable"].to_numpy())
        if exact_hvg and len(hvg_indices) > 5000:
            hvg_indices = exact_hvg_ties(hvg_indices, hvg_adata.var['dispersions_norm'].to_numpy())
            hvg_api += '; exact Top5000, dispersion ties resolved by source gene order'
    except ValueError as error:
        # Many CIMA genes are all-zero within single RNA experiments, which makes
        # recent pandas reject per-batch quantile edges. HVGs are therefore selected
        # globally after removing genes absent from all cells.
        print(f"batch-aware cell_ranger fallback: {error}", flush=True)
        expressed = np.asarray(matrix.getnnz(axis=0)).ravel() > 0
        expressed_indices = np.flatnonzero(expressed)
        global_adata = ad.AnnData(
            X=matrix[:, expressed_indices],
            var=pd.DataFrame(index=pd.Index([genes[i] for i in expressed_indices])),
        )
        try:
            sc.pp.highly_variable_genes(
                global_adata,
                n_top_genes=5000,
                flavor="cell_ranger",
                min_mean=0.0125,
                max_mean=3,
                min_disp=0.5,
            )
            local = np.flatnonzero(global_adata.var["highly_variable"].to_numpy())
            hvg_api = (
                "scanpy.pp.highly_variable_genes(flavor=cell_ranger,n_top_genes=5000); "
                "global CIMA fallback after all-zero gene removal"
            )
            if exact_hvg and len(local) > 5000:
                local = exact_hvg_ties(local, global_adata.var['dispersions_norm'].to_numpy())
                hvg_api += '; exact Top5000, dispersion ties resolved by source gene order'
            hvg_indices = expressed_indices[local]
        except ValueError as global_error:
            print(f"global cell_ranger fallback to CIMA variance: {global_error}", flush=True)
            means = np.asarray(matrix.mean(axis=0)).ravel()
            means_sq = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
            variances = np.maximum(means_sq - means * means, 0.0)
            hvg_indices = np.sort(
                np.argsort(-variances, kind="stable")[:5000]
            )
            hvg_api = "CIMA-only variance Top5000 fallback after public cell_ranger failure"
        del global_adata
    if len(hvg_indices) != 5000:
        raise RuntimeError(f"scPhase preprocessing selected {len(hvg_indices)} genes, not 5000")
    hvg_genes = [genes[index] for index in hvg_indices]
    hvg_matrix = matrix[:, hvg_indices].copy()
    del hvg_adata, matrix
    gc.collect()

    scaled = ad.AnnData(
        X=hvg_matrix,
        var=pd.DataFrame(index=pd.Index(hvg_genes, name="gene_symbol")),
    )
    print("running released Age scale(max_value=10) on CIMA", flush=True)
    sc.pp.scale(scaled, max_value=10)
    means = scaled.var["mean"].to_numpy(dtype=np.float64)
    stds = scaled.var["std"].to_numpy(dtype=np.float64)
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
        raise RuntimeError("invalid CIMA-fitted scPhase scaler")
    scaled_matrix = np.asarray(scaled.X, dtype=np.float32)
    del scaled, hvg_matrix
    gc.collect()
    return hvg_indices, hvg_genes, means, stds, scaled_matrix, hvg_api

def apply_age_preprocessing(raw, means, stds):
    """Apply Scanpy's zero-centred scale(max_value=10) with frozen moments."""
    values = raw.toarray() if sparse.issparse(raw) else np.array(raw, copy=True)
    # Sparse subtraction by Scanpy's float64 means densifies to float64.
    # Cast only after the frozen centring, scaling and symmetric clipping.
    values = np.asarray(values, dtype=np.float64)
    values -= means
    values /= stds
    np.clip(values, -10.0, 10.0, out=values)
    return values.astype(np.float32)

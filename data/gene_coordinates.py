from __future__ import annotations
from typing import Any
import numpy as np

PCA_SCHEMA = "sound_residual_gene_embedding_pca_1"


def frozen_gene_principal_coordinates(
    embeddings: np.ndarray,
    *,
    output_dim: int,
    eigenvalue_floor: float = 1.0e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return deterministic PCA coordinates and their metadata.

    The eigendecomposition uses the gene-by-gene Gram matrix. Component signs
    are fixed by making the largest-magnitude gene coordinate in each component
    positive.
    """

    raw = np.asarray(embeddings)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 1:
        raise ValueError("embeddings must have shape [at least two genes, width]")
    if not np.issubdtype(raw.dtype, np.floating):
        raise ValueError("embeddings must be floating point")
    if not np.isfinite(raw).all():
        raise ValueError("embeddings must contain only finite values")
    if not isinstance(output_dim, int) or output_dim < 1:
        raise ValueError("output_dim must be a positive integer")
    if output_dim >= raw.shape[0]:
        raise ValueError("output_dim must be smaller than the number of genes")
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0:
        raise ValueError("eigenvalue_floor must be finite and positive")

    values = raw.astype(np.float64, copy=False)
    centered = values - values.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    threshold = max(float(eigenvalues[0]) * eigenvalue_floor, eigenvalue_floor)
    positive = int(np.count_nonzero(eigenvalues > threshold))
    if positive < output_dim:
        raise ValueError(
            "frozen gene embeddings lack enough non-degenerate PCA components: "
            f"{positive} < {output_dim}"
        )

    selected_values = eigenvalues[:output_dim]
    coordinates = eigenvectors[:, :output_dim] * np.sqrt(selected_values)[None, :]
    signs: list[int] = []
    for component in range(output_dim):
        column = coordinates[:, component]
        anchor = int(np.argmax(np.abs(column)))
        sign = 1 if column[anchor] >= 0 else -1
        coordinates[:, component] *= sign
        signs.append(sign)
    coordinates = coordinates.astype(np.float32)
    if np.any(np.linalg.norm(coordinates, axis=1) <= eigenvalue_floor):
        raise ValueError("PCA produced a zero-norm gene coordinate")

    total = float(eigenvalues.sum())
    explained = (
        selected_values / total if total > 0 else np.zeros_like(selected_values)
    )
    audit: dict[str, Any] = {
        "status": "pass",
        "schema_version": PCA_SCHEMA,
        "method": "gene_as_observation_centered_gram_pca",
        "outcome_labels_used": False,
        "donor_or_sample_metadata_used": False,
        "input_shape": [int(value) for value in raw.shape],
        "output_shape": [int(value) for value in coordinates.shape],
        "output_dim": int(output_dim),
        "positive_components": positive,
        "eigenvalue_floor": float(eigenvalue_floor),
        "component_signs": signs,
        "explained_variance_ratio": explained.astype(float).tolist(),
    }
    return coordinates, audit

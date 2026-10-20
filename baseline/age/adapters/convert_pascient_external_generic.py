from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import h5py
import numpy as np
from scipy import sparse
from baseline.age.adapters.convert_pascient_fixed_view import EXPECTED_COMMIT, decode

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-audit", required=True, type=Path)
    parser.add_argument("--frozen-gene-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    args = parser.parse_args()

    destinations = [
        args.output,
        args.output.with_suffix(".genes.txt"),
        args.output.with_suffix(".manifest.json"),
    ]
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite external PaSCient artifacts: {existing}")
    audit = json.loads(args.input_audit.read_text())
    if (
        audit.get("status") != "PASS"
        or audit.get("target_labels_accessed") is not False
        or Path(audit.get("output", "")).resolve() != args.input.resolve()
    ):
        raise ValueError("external source is not a label-free audited raw-count view")

    target_genes = [
        line.strip()
        for line in args.frozen_gene_list.read_text().splitlines()
        if line.strip()
    ]
    if not target_genes or len(target_genes) != len(set(target_genes)):
        raise ValueError("frozen CIMA PaSCient gene list is empty or duplicated")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    total_nnz = 0
    zero_after_mapping = 0
    cell_counts: list[int] = []
    with h5py.File(args.input, "r") as source:
        if bool(source.attrs.get("labels_present", True)):
            raise ValueError("external raw-count store exposes labels")
        source_genes = decode(source["var/gene_symbol"][:])
        if len(source_genes) != len(set(source_genes)):
            raise ValueError("external source gene symbols are duplicated")
        source_index = {gene: index for index, gene in enumerate(source_genes)}
        target_positions = np.asarray(
            [index for index, gene in enumerate(target_genes) if gene in source_index],
            dtype=np.int64,
        )
        source_positions = np.asarray(
            [source_index[target_genes[index]] for index in target_positions],
            dtype=np.int64,
        )
        if len(source_positions) < 1000:
            raise ValueError("too few frozen PaSCient genes are observed in the external cohort")

        cell_donors = np.asarray(decode(source["obs/donor_id"][:]), dtype=object)
        donor_ids = sorted(set(map(str, cell_donors.tolist())))
        rows_by_donor: dict[str, list[int]] = {donor: [] for donor in donor_ids}
        for row_index, donor in enumerate(cell_donors):
            rows_by_donor[str(donor)].append(row_index)
        matrix = source["matrix"]
        shape = tuple(int(value) for value in matrix.attrs["shape"])
        if shape != (len(cell_donors), len(source_genes)):
            raise ValueError("external fixed-view dimensions are inconsistent")
        indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
        # Donors can occupy many disjoint HDF5 runs, so the CSR matrix is read
        # once into memory instead of issuing many small compressed reads.
        full_matrix = sparse.csr_matrix(
            (
                np.asarray(matrix["data"][:], dtype=np.float32),
                np.asarray(matrix["indices"][:], dtype=np.int32),
                indptr,
            ),
            shape=shape,
        )

        with h5py.File(args.output, "w") as output:
            output.attrs["format"] = "pascient_donor_csr_v1"
            output.attrs["labels_present"] = False
            output.attrs["input_protocol"] = "audited_external_deterministic_max1500"
            output.attrs["source_h5"] = str(args.input.resolve())
            output.create_dataset(
                "donor_ids", data=np.asarray(donor_ids, dtype=object), dtype=string_dtype
            )
            output.create_dataset(
                "genes", data=np.asarray(target_genes, dtype=object), dtype=string_dtype
            )
            donor_group = output.create_group("donors")
            started = time.monotonic()
            for donor_index, donor in enumerate(donor_ids):
                rows = np.asarray(rows_by_donor[donor], dtype=np.int64)
                source_matrix = full_matrix[rows]
                if np.any(np.asarray(source_matrix.sum(axis=1)).ravel() <= 0):
                    raise ValueError(f"{donor} contains a zero-total source cell")
                observed = sparse.csr_matrix(
                    source_matrix[:, source_positions], dtype=np.float32
                )
                observed.indices = target_positions[observed.indices].astype(
                    np.int32, copy=False
                )
                observed._shape = (len(rows), len(target_genes))
                observed.sum_duplicates()
                observed.sort_indices()
                donor_zero = int(
                    np.sum(np.asarray(observed.sum(axis=1)).ravel() <= 0)
                )
                zero_after_mapping += donor_zero
                group = donor_group.create_group(str(donor_index))
                group.attrs["donor_id"] = donor
                group.attrs["n_cells"] = len(rows)
                group.attrs["source_n_cells"] = len(rows)
                group.attrs["zero_count_cells_dropped"] = 0
                group.attrs["zero_after_gene_mapping_retained"] = donor_zero
                group.create_dataset("data", data=observed.data, compression="lzf")
                group.create_dataset(
                    "indices",
                    data=observed.indices.astype(np.int32, copy=False),
                    compression="lzf",
                )
                group.create_dataset(
                    "indptr",
                    data=observed.indptr.astype(np.int32, copy=False),
                    compression="lzf",
                )
                total_nnz += int(observed.nnz)
                cell_counts.append(len(rows))
                output.flush()
                if (donor_index + 1) % 25 == 0 or donor_index + 1 == len(donor_ids):
                    elapsed = time.monotonic() - started
                    rate = (donor_index + 1) / elapsed
                    eta = (len(donor_ids) - donor_index - 1) / rate
                    print(
                        f"converted {donor_index + 1}/{len(donor_ids)} {args.cohort} donors; "
                        f"elapsed_s={elapsed:.1f}; eta_s={eta:.1f}",
                        flush=True,
                    )
            output.attrs["n_genes"] = len(target_genes)
            output.attrs["observed_frozen_genes"] = len(source_positions)
            output.attrs["missing_frozen_genes"] = len(target_genes) - len(source_positions)
            output.attrs["total_nnz"] = total_nnz
            output.attrs["zero_count_cells_dropped"] = 0
            output.attrs["zero_after_gene_mapping_retained"] = zero_after_mapping

    gene_path = args.output.with_suffix(".genes.txt")
    gene_path.write_text("\n".join(target_genes) + "\n")
    record = {
        "format": "pascient_donor_csr_v1",
        "official_repo_commit": EXPECTED_COMMIT,
        "input_protocol": "audited_external_deterministic_max1500",
        "cohort": args.cohort,
        "input_h5": str(args.input.resolve()),
        "input_audit": str(args.input_audit.resolve()),
        "output": str(args.output.resolve()),
        "output_size_bytes": args.output.stat().st_size,
        "gene_list": str(gene_path.resolve()),
        "frozen_cima_gene_list": str(args.frozen_gene_list.resolve()),
        "n_donors": len(donor_ids),
        "n_cells": int(sum(cell_counts)),
        "minimum_cells_per_donor": int(min(cell_counts)),
        "maximum_cells_per_donor": int(max(cell_counts)),
        "n_genes": len(target_genes),
        "observed_frozen_genes": int(len(source_positions)),
        "missing_frozen_genes": int(len(target_genes) - len(source_positions)),
        "missing_gene_fill": "implicit sparse zero in frozen CIMA gene order",
        "total_nnz": total_nnz,
        "zero_after_gene_mapping_retained": zero_after_mapping,
        "labels_present": False,
        "target_age_datasets_accessed": [],
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    main()

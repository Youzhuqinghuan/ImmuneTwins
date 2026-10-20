from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from baseline.age.workspace import WORKSPACE_ROOT, resolve_workspace_artifact

OFFICIAL_GENE_ORDER = WORKSPACE_ROOT / "external_models_0714/pascient/data/gene_order.tsv"

EXPECTED_COMMIT = "e1c645fa4c43546fd4f03c948c170429034556d0"


def decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]

def contiguous_blocks(rows: np.ndarray) -> list[tuple[int, int]]:
    if rows.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(rows) != 1) + 1
    groups = np.split(rows, cuts)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]

def read_csr_blocks(
    matrix: h5py.Group,
    indptr: np.ndarray,
    rows: np.ndarray,
    n_source_genes: int,
) -> sparse.csr_matrix:
    blocks: list[sparse.csr_matrix] = []
    for lo, hi in contiguous_blocks(rows):
        data_lo = int(indptr[lo])
        data_hi = int(indptr[hi])
        local_indptr = indptr[lo : hi + 1] - data_lo
        blocks.append(
            sparse.csr_matrix(
                (
                    matrix["data"][data_lo:data_hi].astype(np.float32, copy=False),
                    matrix["indices"][data_lo:data_hi].astype(np.int32, copy=False),
                    local_indptr.astype(np.int64, copy=False),
                ),
                shape=(hi - lo, n_source_genes),
            )
        )
    return sparse.vstack(blocks, format="csr")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-audit", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--official-gene-order", type=Path, default=OFFICIAL_GENE_ORDER)
    parser.add_argument(
        "--pilot-donors-per-fold",
        type=int,
        default=0,
        help="Nonzero creates a balanced pilot subset and a sibling .split.tsv file.",
    )
    args = parser.parse_args()

    destinations = [
        args.output,
        args.output.with_suffix(".genes.txt"),
        args.output.with_suffix(".manifest.json"),
    ]
    if args.pilot_donors_per_fold:
        destinations.append(args.output.with_suffix(".split.tsv"))
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite PaSCient fixed-view artifacts: {existing}")

    audit = json.loads(args.input_audit.read_text())
    if audit.get("status") != "PASS":
        raise ValueError("fixed-view source audit is not PASS")
    if resolve_workspace_artifact(audit.get("output", "")) != args.input.resolve():
        raise ValueError("fixed-view audit points to a different input")
    if audit.get("view") != "official_1500" or int(audit.get("cells_per_donor", -1)) != 1500:
        raise ValueError("source is not an audited PaSCient official_1500 view")

    split = pd.read_csv(args.split, sep="\t")
    required = {"donor_id", "age", "fold", "repeat"}
    if not required.issubset(split.columns):
        raise ValueError(f"split is missing {sorted(required - set(split.columns))}")
    split = split.loc[split["repeat"].eq(0)].copy()
    split["donor_id"] = split["donor_id"].astype(str)
    if len(split) != 417 or split["donor_id"].duplicated().any():
        raise ValueError("repeat 0 must contain 417 unique CIMA donors")
    if args.pilot_donors_per_fold:
        if args.pilot_donors_per_fold < 1:
            raise ValueError("pilot donors per fold must be positive")
        split = pd.concat(
            [
                group.sort_values("donor_id").head(args.pilot_donors_per_fold)
                for _, group in split.groupby("fold", sort=True)
            ],
            ignore_index=True,
        )
        split = split.sort_values(["fold", "donor_id"]).reset_index(drop=True)
        split.to_csv(args.output.with_suffix(".split.tsv"), sep="\t", index=False)
    donor_ids = sorted(split["donor_id"].tolist())

    official_genes = [
        line.strip() for line in args.official_gene_order.read_text().splitlines() if line.strip()
    ]
    if len(official_genes) != len(set(official_genes)):
        raise ValueError("released PaSCient gene order contains duplicates")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    total_nnz = 0
    mapped_zero_cells = 0
    with h5py.File(args.input, "r") as source:
        if source.attrs.get("schema_version") != "iadt_pascient_cima_raw_v1":
            raise ValueError("unsupported fixed-view HDF5 schema")
        source_genes = decode(source["var/gene_symbol"][:])
        if len(source_genes) != len(set(source_genes)):
            raise ValueError("source gene symbols are not unique")
        source_index = {gene: index for index, gene in enumerate(source_genes)}
        target_genes = [gene for gene in official_genes if gene in source_index]
        source_positions = np.asarray([source_index[gene] for gene in target_genes], dtype=np.int64)
        if not target_genes:
            raise ValueError("released PaSCient/CIMA gene intersection is empty")

        cell_donors = np.asarray(decode(source["obs/donor_id"][:]), dtype=object)
        if set(donor_ids) - set(cell_donors):
            raise ValueError("pilot/formal split donors are absent from fixed-view input")
        # Build the row lookup once instead of scanning the donor vector per donor.
        requested_donors = set(donor_ids)
        donor_rows_lists: dict[str, list[int]] = {donor: [] for donor in donor_ids}
        for row_index, donor in enumerate(cell_donors):
            if donor in requested_donors:
                donor_rows_lists[donor].append(row_index)
        donor_rows = {
            donor: np.asarray(rows, dtype=np.int64)
            for donor, rows in donor_rows_lists.items()
        }
        matrix = source["matrix"]
        shape = tuple(int(value) for value in matrix.attrs["shape"])
        if shape != (len(cell_donors), len(source_genes)):
            raise ValueError("fixed-view matrix shape metadata is inconsistent")
        indptr = matrix["indptr"][:].astype(np.int64, copy=False)

        with h5py.File(args.output, "w") as output:
            output.attrs["format"] = "pascient_donor_csr_v1"
            output.attrs["labels_present"] = False
            output.attrs["input_protocol"] = "audited_fixed_official_1500_view"
            output.attrs["source_h5"] = str(args.input.resolve())
            output.create_dataset(
                "donor_ids", data=np.asarray(donor_ids, dtype=object), dtype=string_dtype
            )
            output.create_dataset(
                "genes", data=np.asarray(target_genes, dtype=object), dtype=string_dtype
            )
            donor_group = output.create_group("donors")
            conversion_started = time.monotonic()
            for donor_index, donor in enumerate(donor_ids):
                rows = donor_rows[donor]
                if len(rows) != 1500:
                    raise ValueError(f"{donor} has {len(rows)} cells, expected exactly 1500")
                source_matrix = read_csr_blocks(matrix, indptr, rows, len(source_genes))
                if np.any(np.asarray(source_matrix.sum(axis=1)).ravel() <= 0):
                    raise ValueError(f"{donor} contains a zero-total source cell")
                aligned = sparse.csr_matrix(source_matrix[:, source_positions], dtype=np.float32)
                aligned.sum_duplicates()
                aligned.sort_indices()
                zero_after_mapping = int(
                    np.sum(np.asarray(aligned.sum(axis=1)).ravel() <= 0)
                )
                mapped_zero_cells += zero_after_mapping
                group = donor_group.create_group(str(donor_index))
                group.attrs["donor_id"] = donor
                group.attrs["n_cells"] = len(rows)
                group.attrs["source_n_cells"] = len(rows)
                group.attrs["zero_count_cells_dropped"] = 0
                group.attrs["zero_after_gene_mapping_retained"] = zero_after_mapping
                group.create_dataset("data", data=aligned.data, compression="lzf")
                group.create_dataset(
                    "indices", data=aligned.indices.astype(np.int32, copy=False), compression="lzf"
                )
                group.create_dataset(
                    "indptr", data=aligned.indptr.astype(np.int32, copy=False), compression="lzf"
                )
                total_nnz += int(aligned.nnz)
                output.flush()
                elapsed = time.monotonic() - conversion_started
                rate = (donor_index + 1) / elapsed if elapsed > 0 else 0.0
                remaining = (len(donor_ids) - donor_index - 1) / rate if rate > 0 else float("inf")
                print(
                    "converted "
                    f"{donor_index + 1}/{len(donor_ids)} donors; "
                    f"donor={donor}; elapsed_s={elapsed:.1f}; eta_s={remaining:.1f}; "
                    f"total_nnz={total_nnz}",
                    flush=True,
                )
            output.attrs["n_genes"] = len(target_genes)
            output.attrs["total_nnz"] = total_nnz
            output.attrs["missing_frozen_genes"] = len(official_genes) - len(target_genes)
            output.attrs["zero_count_cells_dropped"] = 0
            output.attrs["zero_after_gene_mapping_retained"] = mapped_zero_cells

    gene_path = args.output.with_suffix(".genes.txt")
    gene_path.write_text("\n".join(target_genes) + "\n")
    record = {
        "format": "pascient_donor_csr_v1",
        "official_repo_commit": EXPECTED_COMMIT,
        "input_protocol": "audited_fixed_official_1500_view",
        "input_h5": str(args.input.resolve()),
        "input_audit": str(args.input_audit.resolve()),
        "output": str(args.output.resolve()),
        "output_size_bytes": args.output.stat().st_size,
        "gene_list": str(gene_path.resolve()),
        "n_donors": len(donor_ids),
        "n_cells_per_donor": 1500,
        "n_genes": len(target_genes),
        "missing_official_genes": len(official_genes) - len(target_genes),
        "gene_matching": "released ordered exact case-sensitive intersection",
        "total_nnz": total_nnz,
        "zero_after_gene_mapping_retained": mapped_zero_cells,
        "labels_present": False,
        "pilot": bool(args.pilot_donors_per_fold),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()

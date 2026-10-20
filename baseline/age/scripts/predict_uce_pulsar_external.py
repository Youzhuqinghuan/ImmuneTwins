from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from baseline.age.scripts.evaluate_pretraining_baselines import align_labels, fit_uce_ols, load_cima_uce, load_external_uce, load_npz, uce_predict

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cima-ages", required=True, type=Path)
    parser.add_argument("--cima-uce", required=True, type=Path)
    parser.add_argument("--external-uce-root", required=True, type=Path)
    parser.add_argument("--cima-pulsar", required=True, type=Path)
    parser.add_argument("--external-pulsar", required=True, type=Path)
    parser.add_argument("--donor-manifest", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "uce_pulsar_predictions.sealed.tsv"
    manifest_path = args.output_dir / "uce_pulsar_predictions.seal.json"
    if prediction_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite UCE/PULSAR external predictions")
    ages = pd.read_csv(args.cima_ages, sep="\t")[["donor_id", "age"]]
    expected = pd.read_csv(args.donor_manifest, sep="\t")
    if "donor_id" not in expected or expected["donor_id"].duplicated().any():
        raise ValueError("invalid label-free external donor manifest")
    expected_ids = set(expected["donor_id"].astype(str))

    cima_uce, cima_uce_donors = load_cima_uce(args.cima_uce)
    cima_age = align_labels(cima_uce_donors, ages)
    external_uce, uce_donors, uce_proof = load_external_uce(
        args.external_uce_root,
        expected_groups=None,
        expected_donors=len(expected_ids),
        expected_cells_per_donor=None,
    )
    if set(uce_donors) != expected_ids:
        raise ValueError("UCE donor coverage differs from raw-count manifest")
    if "selected_cells" in expected and int(expected["selected_cells"].sum()) != int(
        uce_proof["n_cells"]
    ):
        raise ValueError("UCE cell coverage differs from raw-count manifest")
    uce_weight, uce_intercept = fit_uce_ols(cima_uce, cima_age, args.device)
    uce_head = args.output_dir / "uce_cima_linear_readout.npz"
    np.savez(
        uce_head,
        weight=np.asarray(uce_weight, dtype=np.float64),
        intercept=np.asarray([uce_intercept], dtype=np.float64),
    )
    uce = pd.DataFrame(
        {
            "method": "UCE",
            "cohort": args.cohort,
            "donor_id": uce_donors,
            "predicted_age": uce_predict(external_uce, uce_weight, uce_intercept),
        }
    )

    cima_pulsar, cima_pulsar_donors = load_npz(args.cima_pulsar)
    external_pulsar, pulsar_donors = load_npz(args.external_pulsar)
    if set(pulsar_donors) != expected_ids:
        raise ValueError("PULSAR donor coverage differs from raw-count manifest")
    pulsar_age = align_labels(cima_pulsar_donors, ages)
    pulsar_model = Ridge(alpha=0.1)
    pulsar_model.fit(cima_pulsar, pulsar_age)
    pulsar_head = args.output_dir / "pulsar_cima_ridge_alpha0.1.npz"
    np.savez(
        pulsar_head,
        coef=np.asarray(pulsar_model.coef_, dtype=np.float64),
        intercept=np.asarray([pulsar_model.intercept_], dtype=np.float64),
        alpha=np.asarray([0.1], dtype=np.float64),
    )
    pulsar = pd.DataFrame(
        {
            "method": "PULSAR",
            "cohort": args.cohort,
            "donor_id": pulsar_donors,
            "predicted_age": pulsar_model.predict(external_pulsar),
        }
    )
    combined = pd.concat([uce, pulsar], ignore_index=True).sort_values(
        ["method", "donor_id"]
    )
    if (
        len(combined) != 2 * len(expected_ids)
        or combined.duplicated(["method", "donor_id"]).any()
        or not np.isfinite(combined["predicted_age"]).all()
    ):
        raise ValueError("combined UCE/PULSAR prediction coverage failed")
    temporary = prediction_path.with_suffix(".partial.tsv")
    combined.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, prediction_path)
    record = {
        "status": "PASS",
        "methods": ["UCE", "PULSAR"],
        "cohort": args.cohort,
        "n_donors": len(expected_ids),
        "n_predictions": len(combined),
        "fit_cohort": "CIMA_only",
        "uce_patient_prediction": "mean UCE cell embedding plus unregularized CIMA linear readout",
        "uce_external_proof": uce_proof,
        "pulsar_age_recipe": "official Ridge(alpha=0.1)",
        "target_labels_accessed": False,
        "target_statistics_accessed": False,
        "target_calibration": False,
    }
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    main()

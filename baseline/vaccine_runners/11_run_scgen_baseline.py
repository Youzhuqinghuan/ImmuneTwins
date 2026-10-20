#!/usr/bin/env python3
"""Official scGen donor-blocked baseline on single-cell state latents."""

from __future__ import annotations

import fcntl
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, parser, write_json, reference_profile, validate_reference_arguments
from formal_latent_reader import Reader
from trusted_torch_checkpoint import trusted_local_full_object_load


SCGEN_SOURCE = ROOT / "external/scgen"
SCGEN_SITE = ROOT / "environments/scgen_site"


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> None:
    p = parser("Official scGen future single-cell distribution baseline")
    p.add_argument("--training-latent-h5", required=True)
    p.add_argument("--test-latent-h5", required=True)
    p.add_argument("--cells-per-episode-condition", type=int, default=512)
    p.add_argument("--generated-cells", type=int, default=2048)
    p.add_argument("--max-epochs", type=int, default=reference_profile("scGen")["max_epochs"])
    p.add_argument("--batch-size", type=int, default=reference_profile("scGen")["batch_size"])
    p.add_argument("--scope", choices=["cold_start", "year2_locked"], default="cold_start")
    args = p.parse_args()
    reference = validate_reference_arguments("scGen", args)
    if args.fold not in range(5):
        p.error("--fold must be 0..4")
    if args.lineage != "all":
        p.error("--lineage is deprecated; the vaccine runner always uses all cells")
    run_name = (
        f"scgen_{args.scope}_outer{args.fold}_day{args.source_day}"
        f"_to_day{args.target_day}_seed{args.seed}"
    )
    output_dir = Path(args.output_dir or ROOT / "results/baseline_single_cell/scgen" / run_name)
    result_path = output_dir / "run.json"
    manifest_path = output_dir / "predictions.tsv"
    checkpoint = output_dir / "checkpoint"
    if args.dry_run:
        print(json.dumps({
            "run_name": run_name, "official_source": str(SCGEN_SOURCE),
            "training_latent": args.training_latent_h5, "test_latent": args.test_latent_h5,
            "outer_fold": args.fold, "target_day": args.target_day,
            "cell_scope": "all_cells",
            "gpu_training_required": True, "output": str(output_dir),
        }, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    run_lock_handle = (output_dir / ".run.lock").open("w")
    fcntl.flock(run_lock_handle, fcntl.LOCK_EX)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("explicit CUDA_VISIBLE_DEVICES is required")
    if args.resume and result_path.is_file() and manifest_path.is_file() and (checkpoint / "model.pt").is_file():
        prior = json.loads(result_path.read_text())
        if prior.get("status") == "pass":
            print(json.dumps({"status": "resume", "result": str(result_path)}, indent=2))
            return
    sys.path.insert(0, str(SCGEN_SITE))
    sys.path.insert(0, str(SCGEN_SOURCE))
    import anndata as ad
    import scgen
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU must be visible; CPU fallback prohibited")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    splits = pd.read_csv(ROOT / "config/splits/sound_outer_groupkfold5_donors.tsv", sep="\t")
    episodes = pd.read_csv(ROOT / "data/episode_manifests/sound_flu_episodes.tsv", sep="\t")
    test_donors = set(splits.loc[splits.outer_fold == args.fold, "donor_id"].astype(str))
    complete = f"complete_D0_D{args.target_day}"
    target_sample = f"sample_D{args.target_day}"
    train = episodes.loc[(~episodes.donor_id.astype(str).isin(test_donors)) &
                         episodes[complete].astype(bool)].copy()
    test = episodes.loc[(episodes.donor_id.astype(str).isin(test_donors)) &
                        episodes[complete].astype(bool)].copy()
    if args.scope == "year2_locked":
        train = train.loc[train.season.astype(int) == 1].copy()
        test = test.loc[test.season.astype(int) == 2].copy()
    train_reader, test_reader = Reader(Path(args.training_latent_h5)), Reader(Path(args.test_latent_h5))
    started = time.time()
    try:
        rng = np.random.default_rng(args.seed)
        source_condition = f"D{args.source_day}"
        matrices, conditions, labels = [], [], []
        used_episodes = []
        for index, row in train.reset_index(drop=True).iterrows():
            d0_rows = train_reader.rows(str(row.sample_D0))
            future_rows = train_reader.rows(str(row[target_sample]))
            if not len(d0_rows) or not len(future_rows):
                continue
            d0, d0_state, _ = train_reader.take(
                d0_rows, args.cells_per_episode_condition, rng
            )
            future, future_state, _ = train_reader.take(
                future_rows, args.cells_per_episode_condition, rng
            )
            matrices.extend([d0, future])
            conditions.extend(
                [
                    [source_condition] * len(d0),
                    [f"D{args.target_day}"] * len(future),
                ]
            )
            # Official scGen uses labels_key for cell identity, so the all-cell
            # model receives the measured L2 cell type of every sampled cell.
            labels.extend([d0_state, future_state])
            used_episodes.append(f"{row.donor_id}__season{int(row.season)}")
        if not matrices:
            raise RuntimeError("no complete training trajectories for official scGen")
        x_train = np.vstack(matrices).astype(np.float32)
        obs = pd.DataFrame(
            {
                "condition": np.concatenate(conditions),
                "cell_type": np.concatenate(labels),
            }
        )
        adata = ad.AnnData(X=x_train, obs=obs)
        scgen.SCGEN.setup_anndata(
            adata, batch_key="condition", labels_key="cell_type"
        )
        model = scgen.SCGEN(adata)
        model.train(max_epochs=args.max_epochs, batch_size=args.batch_size,
                    early_stopping=True, early_stopping_patience=reference_profile("scGen")["patience"],
                    check_val_every_n_epoch=1, accelerator="gpu", devices=1)
        if str(model.device).split(":")[0] != "cuda":
            raise RuntimeError(f"official scGen did not train on CUDA: {model.device}")
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint, overwrite=True, save_anndata=True)
        probe = adata[:min(128, len(adata))].copy()
        latent_before = model.get_latent_representation(probe)
        with trusted_local_full_object_load(torch, checkpoint / "model.pt"):
            reloaded = scgen.SCGEN.load(checkpoint)
        reloaded.to_device("cuda")
        latent_after = reloaded.get_latent_representation(probe)
        reload_max_abs = float(np.max(np.abs(latent_before - latent_after)))
        if reload_max_abs > 1e-6:
            raise RuntimeError(f"official scGen reload mismatch: {reload_max_abs}")
        records = []
        for test_index, row in test.reset_index(drop=True).iterrows():
            d0_rows = test_reader.rows(str(row.sample_D0))
            future_rows = test_reader.rows(str(row[target_sample]))
            if not len(d0_rows) or not len(future_rows):
                continue
            query_x, query_state, query_rows = test_reader.take(
                d0_rows, args.generated_cells,
                np.random.default_rng(args.seed + 10_000 + test_index), exact=True,
            )
            unseen_states = sorted(
                set(query_state) - set(adata.obs.cell_type.cat.categories)
            )
            if unseen_states:
                raise RuntimeError(
                    "held-out scGen source contains cell types absent from "
                    f"outer training: {unseen_states}"
                )
            query = ad.AnnData(X=query_x, obs=pd.DataFrame({
                "condition": pd.Categorical(
                    [source_condition] * len(query_x),
                    categories=[source_condition, f"D{args.target_day}"],
                ),
                "cell_type": pd.Categorical(
                    query_state,
                    categories=adata.obs.cell_type.cat.categories,
                ),
            }))
            np.random.seed(args.seed + 20_000 + test_index)
            predicted, delta = reloaded.predict(
                ctrl_key=source_condition,
                stim_key=f"D{args.target_day}",
                adata_to_predict=query,
            )
            observed_x, observed_state, observed_rows = test_reader.take(
                future_rows, args.generated_cells,
                np.random.default_rng(args.seed + 30_000 + test_index), exact=True,
            )
            identity = f"{row.donor_id}__season{int(row.season)}"
            source_path = output_dir / f"source_day{args.source_day}" / f"{identity}.npz"
            predicted_path = output_dir / "predicted" / f"{identity}.npz"
            observed_path = output_dir / "observed" / f"{identity}.npz"
            save_npz(predicted_path, embedding=np.asarray(predicted.X, dtype=np.float32),
                     cell_state=query_state, source_row_index=query_rows,
                     source_day=np.asarray(args.source_day),
                     delta=np.asarray(delta, dtype=np.float32), donor_id=np.asarray(str(row.donor_id)),
                     episode_id=np.asarray(identity), target_day=np.asarray(args.target_day),
                     cell_scope=np.asarray("all_cells"), method=np.asarray("scGen_official"))
            save_npz(source_path, embedding=query_x, cell_state=query_state,
                     source_row_index=query_rows, donor_id=np.asarray(str(row.donor_id)),
                     episode_id=np.asarray(identity), target_day=np.asarray(args.source_day),
                     cell_scope=np.asarray("all_cells"),
                     role=np.asarray(f"observed_day{args.source_day}"))
            save_npz(observed_path, embedding=observed_x, cell_state=observed_state,
                     source_row_index=observed_rows, donor_id=np.asarray(str(row.donor_id)),
                     episode_id=np.asarray(identity), target_day=np.asarray(args.target_day),
                     cell_scope=np.asarray("all_cells"), role=np.asarray("observed_future"))
            records.append({"donor_id": str(row.donor_id), "episode_id": identity,
                            "season": int(row.season), "outer_fold": args.fold,
                            "cell_scope": "all_cells", "target_day": args.target_day,
                            "method": "scGen_official", "source_path": str(source_path),
                            "observed_path": str(observed_path),
                            "predicted_path": str(predicted_path),
                            "prediction_uses_test_future": False})
        manifest = pd.DataFrame(records)
        if manifest.empty:
            raise RuntimeError("official scGen produced no held-out distributions")
        manifest.to_csv(manifest_path, sep="\t", index=False)
    finally:
        train_reader.close(); test_reader.close()
    commit = subprocess.check_output(["git", "-C", str(SCGEN_SOURCE), "rev-parse", "HEAD"],
                                     text=True).strip()
    payload = {
        **reference,
        "actual_epochs": int(model.trainer.current_epoch),
        "status": "pass", "run_name": run_name, "scope": args.scope,
        "year2_used_for_training_or_selection": False if args.scope == "year2_locked" else None,
        "implementation": "official scGen repository",
        "official_commit": commit, "official_source": str(SCGEN_SOURCE),
        "outer_fold": args.fold, "target_day": args.target_day,
        "cell_scope": "all_cells",
        "seed": args.seed, "training_donors": int(train.donor_id.nunique()),
        "training_episodes_used": len(used_episodes), "training_cells": len(x_train),
        "input_role": f"frozen per-cell scVI state embeddings at Day {args.source_day}",
        "output_role": "future per-cell state embedding distribution",
        "donor_blocked": True, "source_day": args.source_day,
        "test_source_only_for_prediction": True,
        "scgen_labels_key": "cell_type",
        "scgen_labels_level": "fine_cell_type_L2",
        "scgen_labels_categories": sorted(
            adata.obs.cell_type.astype(str).unique().tolist()
        ),
        "donor_id_used_as_scgen_label": False,
        "test_future_used_for_training_or_adaptation": False,
        "checkpoint": str(checkpoint),
        "checkpoint_load_compatibility": (
            "PyTorch>=2.6 scoped weights_only=False for the exact trusted locally generated "
            "model.pt path; torch.load restored immediately after official scGen reload"
        ),
        "reload_max_abs_difference": reload_max_abs,
        "manifest": str(manifest_path),
        "n_test_distributions": len(manifest), "elapsed_seconds": time.time() - started,
        "cuda_device_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "scgen_version": getattr(scgen, "__version__", "unknown"),
    }
    write_json(result_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

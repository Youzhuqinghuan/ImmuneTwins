#!/usr/bin/env python3
"""Official scArches trVAE conditional-generation baseline on cell-state latents."""

from __future__ import annotations

import fcntl
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, parser, write_json, reference_profile, validate_reference_arguments
from formal_latent_reader import Reader
from trusted_torch_checkpoint import trusted_local_full_object_load


SCARCHES_SOURCE = ROOT / "external/scarches_v051"
SCARCHES_SITE = ROOT / "environments/scarches_v051_site"
SCARCHES_TAG_COMMIT = "09c23f9489cd8a99cf6356548f471dfdbbe4bd5b"


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def decode_condition(model, latent: np.ndarray, condition: str) -> np.ndarray:
    import torch
    model.model.eval()
    condition_code = model.model.condition_encoder[condition]
    outputs = []
    for start in range(0, len(latent), 512):
        z = torch.as_tensor(latent[start:start + 512], dtype=torch.float32,
                            device=next(model.model.parameters()).device)
        c = torch.full((len(z),), condition_code, dtype=torch.long, device=z.device)
        with torch.no_grad():
            reconstruction = model.model.decoder(z, c)[0]
        outputs.append(reconstruction.detach().cpu().numpy())
    return np.vstack(outputs).astype(np.float32)


def main() -> None:
    p = parser("Official scArches trVAE conditional-generation baseline")
    p.add_argument("--training-latent-h5", required=True)
    p.add_argument("--test-latent-h5", required=True)
    p.add_argument("--scale-npz", required=True)
    p.add_argument("--cells-per-episode-condition", type=int, default=512)
    p.add_argument("--generated-cells", type=int, default=2048)
    p.add_argument("--max-epochs", type=int, default=reference_profile("trVAE")["max_epochs"])
    p.add_argument("--batch-size", type=int, default=reference_profile("trVAE")["batch_size"])
    p.add_argument("--scope", choices=["cold_start", "year2_locked"], default="cold_start")
    args = p.parse_args()
    reference = validate_reference_arguments("trVAE", args)
    if args.fold not in range(5): p.error("--fold must be 0..4")
    if args.lineage != "all":
        p.error("--lineage is deprecated; the vaccine runner always uses all cells")
    run_name = (
        f"trvae_{args.scope}_outer{args.fold}_day{args.source_day}"
        f"_to_day{args.target_day}_seed{args.seed}"
    )
    output_dir = Path(args.output_dir or ROOT / "results/baseline_single_cell/trvae" / run_name)
    result_path, manifest_path = output_dir / "run.json", output_dir / "predictions.tsv"
    checkpoint = output_dir / "checkpoint"
    if args.dry_run:
        print(json.dumps({"run_name": run_name, "official_tag": "v0.5.1",
                          "official_tag_commit": SCARCHES_TAG_COMMIT,
                          "training_latent": args.training_latent_h5,
                          "test_latent": args.test_latent_h5,
                          "cell_scope": "all_cells", "gpu_training_required": True,
                          "output": str(output_dir)}, indent=2)); return
    output_dir.mkdir(parents=True, exist_ok=True)
    run_lock_handle = (output_dir / ".run.lock").open("w")
    fcntl.flock(run_lock_handle, fcntl.LOCK_EX)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("explicit CUDA_VISIBLE_DEVICES is required")
    if args.resume and result_path.is_file() and manifest_path.is_file() and (checkpoint / "model_params.pt").is_file():
        prior = json.loads(result_path.read_text())
        if prior.get("status") == "pass":
            print(json.dumps({"status": "resume", "result": str(result_path)}, indent=2)); return
    sys.path.insert(0, str(SCARCHES_SITE)); sys.path.insert(0, str(SCARCHES_SOURCE))
    import anndata as ad
    import scarches
    import torch
    from scarches.models.trvae.trvae_model import TRVAE
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU must be visible; CPU fallback prohibited")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed); torch.set_num_threads(4)
    with np.load(args.scale_npz, allow_pickle=False) as frozen:
        center, scale = frozen["center"].astype(np.float32), frozen["scale"].astype(np.float32)
        fit_role = str(frozen["fit_role"])
    if "training" not in fit_role.lower():
        raise RuntimeError("trVAE scale must be fitted on outer-training cells only")
    splits = pd.read_csv(ROOT / "config/splits/sound_outer_groupkfold5_donors.tsv", sep="\t")
    episodes = pd.read_csv(ROOT / "data/episode_manifests/sound_flu_episodes.tsv", sep="\t")
    test_donors = set(splits.loc[splits.outer_fold == args.fold, "donor_id"].astype(str))
    complete, target_column = f"complete_D0_D{args.target_day}", f"sample_D{args.target_day}"
    train = episodes.loc[(~episodes.donor_id.astype(str).isin(test_donors)) &
                         episodes[complete].astype(bool)].copy()
    test = episodes.loc[(episodes.donor_id.astype(str).isin(test_donors)) &
                        episodes[complete].astype(bool)].copy()
    if args.scope == "year2_locked":
        train = train.loc[train.season.astype(int) == 1].copy()
        test = test.loc[test.season.astype(int) == 2].copy()
    reader, test_reader = Reader(Path(args.training_latent_h5)), Reader(Path(args.test_latent_h5))
    started = time.time()
    try:
        rng = np.random.default_rng(args.seed)
        source_condition = f"D{args.source_day}"
        matrices, conditions, used = [], [], []
        for _, row in train.reset_index(drop=True).iterrows():
            d0_rows, future_rows = (reader.rows(str(row.sample_D0)),
                                    reader.rows(str(row[target_column])))
            if not len(d0_rows) or not len(future_rows): continue
            d0 = reader.take(d0_rows, args.cells_per_episode_condition, rng)[0]
            future = reader.take(future_rows, args.cells_per_episode_condition, rng)[0]
            matrices.extend([(d0 - center) / scale, (future - center) / scale])
            conditions.extend(
                [
                    [source_condition] * len(d0),
                    [f"D{args.target_day}"] * len(future),
                ]
            )
            used.append(f"{row.donor_id}__season{int(row.season)}")
        if not matrices: raise RuntimeError("no complete training trajectories for trVAE")
        x_train = np.vstack(matrices).astype(np.float32)
        adata = ad.AnnData(X=x_train, obs=pd.DataFrame({"condition": np.concatenate(conditions)}))
        model = TRVAE(adata, condition_key="condition",
                      conditions=[source_condition, f"D{args.target_day}"],
                      hidden_layer_sizes=[256, 64], latent_dim=10, dr_rate=0.05,
                      use_mmd=True, mmd_on="y", recon_loss="mse", beta=1,
                      use_bn=False, use_ln=True)
        model.train(n_epochs=args.max_epochs, lr=1e-3, eps=0.01,
                    batch_size=args.batch_size, train_frac=0.9,
                    use_stratified_sampling=True, n_workers=0, seed=args.seed,
                    monitor=False, use_early_stopping=True, reload_best=True,
                    early_stopping_kwargs={"early_stopping_metric": "val_unweighted_loss",
                                           "patience": reference_profile("trVAE")["patience"], "threshold": 0,
                                           "reduce_lr": True, "lr_patience": reference_profile("trVAE")["lr_patience"],
                                           "lr_factor": 0.1})
        if next(model.model.parameters()).device.type != "cuda":
            raise RuntimeError("official trVAE did not train on CUDA")
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint, overwrite=True, save_anndata=True)
        probe = x_train[:min(128, len(x_train))]
        probe_condition = np.asarray([source_condition] * len(probe))
        z_before = model.get_latent(probe, probe_condition, mean=True)
        decoded_before = decode_condition(model, z_before, f"D{args.target_day}")
        with trusted_local_full_object_load(torch, checkpoint / "model_params.pt"):
            reloaded = TRVAE.load(checkpoint, map_location="cuda")
        reloaded.model.to("cuda"); reloaded.is_trained_ = True
        z_after = reloaded.get_latent(probe, probe_condition, mean=True)
        decoded_after = decode_condition(reloaded, z_after, f"D{args.target_day}")
        reload_max_abs = float(np.max(np.abs(decoded_before - decoded_after)))
        if reload_max_abs > 1e-6:
            raise RuntimeError(f"trVAE reload mismatch: {reload_max_abs}")
        records = []
        for index, row in test.reset_index(drop=True).iterrows():
            d0_rows, future_rows = (test_reader.rows(str(row.sample_D0)),
                                    test_reader.rows(str(row[target_column])))
            if not len(d0_rows) or not len(future_rows): continue
            query, query_state, query_rows = test_reader.take(
                d0_rows, args.generated_cells, np.random.default_rng(args.seed + 20_000 + index), True
            )
            query_scaled = (query - center) / scale
            latent = reloaded.get_latent(
                query_scaled,
                np.asarray([source_condition] * len(query_scaled)),
                mean=True,
            )
            predicted = decode_condition(reloaded, latent, f"D{args.target_day}") * scale + center
            observed, observed_state, observed_rows = test_reader.take(
                future_rows, args.generated_cells,
                np.random.default_rng(args.seed + 30_000 + index), True,
            )
            identity = f"{row.donor_id}__season{int(row.season)}"
            source_path = output_dir / f"source_day{args.source_day}" / f"{identity}.npz"
            predicted_path, observed_path = (output_dir / "predicted" / f"{identity}.npz",
                                              output_dir / "observed" / f"{identity}.npz")
            save_npz(source_path, embedding=query, cell_state=query_state,
                     source_row_index=query_rows, donor_id=np.asarray(str(row.donor_id)),
                     episode_id=np.asarray(identity), target_day=np.asarray(args.source_day),
                     cell_scope=np.asarray("all_cells"),
                     role=np.asarray(f"observed_day{args.source_day}"))
            save_npz(predicted_path, embedding=predicted.astype(np.float32), cell_state=query_state,
                     source_row_index=query_rows, source_day=np.asarray(args.source_day),
                     donor_id=np.asarray(str(row.donor_id)),
                     episode_id=np.asarray(identity), target_day=np.asarray(args.target_day),
                     cell_scope=np.asarray("all_cells"), method=np.asarray("trVAE_scArches_official"))
            save_npz(observed_path, embedding=observed, cell_state=observed_state,
                     source_row_index=observed_rows, donor_id=np.asarray(str(row.donor_id)),
                     episode_id=np.asarray(identity), target_day=np.asarray(args.target_day),
                     cell_scope=np.asarray("all_cells"), role=np.asarray("observed_future"))
            records.append({"donor_id": str(row.donor_id), "episode_id": identity,
                            "season": int(row.season), "outer_fold": args.fold,
                            "cell_scope": "all_cells", "target_day": args.target_day,
                            "method": "trVAE_scArches_official",
                            "source_path": str(source_path), "observed_path": str(observed_path),
                            "predicted_path": str(predicted_path),
                            "prediction_uses_test_future": False})
        manifest = pd.DataFrame(records)
        if manifest.empty: raise RuntimeError("trVAE produced no held-out distributions")
        manifest.to_csv(manifest_path, sep="\t", index=False)
    finally:
        reader.close(); test_reader.close()
    model_params = sum(parameter.numel() for parameter in reloaded.model.parameters())
    payload = {
        **reference,
        "status": "pass", "run_name": run_name, "scope": args.scope,
        "year2_used_for_training_or_selection": False if args.scope == "year2_locked" else None,
        "implementation": "official scArches TRVAE source snapshot compatible with v0.5.1",
        "official_tag": "v0.5.1", "official_tag_commit": SCARCHES_TAG_COMMIT,
        "source_internal_version": scarches.__version__, "official_source": str(SCARCHES_SOURCE),
        "outer_fold": args.fold, "target_day": args.target_day,
        "cell_scope": "all_cells",
        "seed": args.seed, "training_donors": int(train.donor_id.nunique()),
        "training_episodes_used": len(used), "training_cells": len(x_train),
        "model_parameters": model_params, "trained_epochs": int(model.trainer.epoch + 1),
        "best_epoch": None if model.trainer.best_epoch is None else int(model.trainer.best_epoch + 1),
        "input_role": f"frozen per-cell scVI state embeddings at Day {args.source_day}",
        "output_role": "condition-switched future per-cell state embedding distribution",
        "cell_type_role": "retained output metadata only; not a trVAE input",
        "donor_blocked": True, "source_day": args.source_day,
        "test_source_only_for_prediction": True,
        "mmd_location": "decoder_first_layer_y",
        "test_future_used_for_training_or_adaptation": False,
        "training_only_scale": str(args.scale_npz),
        "checkpoint": str(checkpoint),
        "checkpoint_load_compatibility": (
            "PyTorch>=2.6 scoped weights_only=False for the exact trusted locally generated "
            "model_params.pt path; torch.load restored immediately after official trVAE reload"
        ),
        "reload_max_abs_difference": reload_max_abs,
        "manifest": str(manifest_path),
        "n_test_distributions": len(manifest), "elapsed_seconds": time.time() - started,
        "cuda_device_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
    }
    write_json(result_path, payload); print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

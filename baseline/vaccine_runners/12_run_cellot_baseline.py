#!/usr/bin/env python3
"""Official CellOT baseline with four inner donor folds and final retraining."""

from __future__ import annotations

import copy
import fcntl
import json
import math
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


CELLOT_SOURCE = ROOT / "external/cellot"
CELLOT_SITE = ROOT / "environments/cellot_site"


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def episode_arrays(reader: Reader, episodes: pd.DataFrame, target_column: str,
                   cells: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    source, target, used = [], [], []
    for _, row in episodes.reset_index(drop=True).iterrows():
        source_rows = reader.rows(str(row.sample_D0))
        target_rows = reader.rows(str(row[target_column]))
        if not len(source_rows) or not len(target_rows):
            continue
        source.append(reader.take(source_rows, cells, rng)[0])
        target.append(reader.take(target_rows, cells, rng)[0])
        used.append(f"{row.donor_id}__season{int(row.season)}")
    if not source:
        raise RuntimeError("no complete trajectories for CellOT")
    return np.vstack(source).astype(np.float32), np.vstack(target).astype(np.float32), used


def torch_mmd(x, y, bandwidth: float):
    import torch
    gamma = 0.5 / bandwidth**2
    kxx = torch.exp(-gamma * torch.cdist(x, x).square())
    kyy = torch.exp(-gamma * torch.cdist(y, y).square())
    kxy = torch.exp(-gamma * torch.cdist(x, y).square())
    nx, ny = len(x), len(y)
    return ((kxx.sum() - torch.diagonal(kxx).sum()) / max(1, nx * (nx - 1)) +
            (kyy.sum() - torch.diagonal(kyy).sum()) / max(1, ny * (ny - 1)) -
            2 * kxy.mean())


def model_config(input_dim: int) -> dict:
    return {
        "model": {"name": "cellot", "input_dim": input_dim,
                  "hidden_units": [64, 64, 64, 64], "softplus_W_kernels": False,
                  "g": {"fnorm_penalty": 1},
                  "kernel_init_fxn": {"name": "uniform", "b": 0.1}},
        "optim": {"optimizer": "Adam", "lr": 0.0001, "beta1": 0.5,
                  "beta2": 0.9, "weight_decay": 0},
    }


def new_model(config: dict, seed: int, device: str):
    import torch
    from cellot.models.cellot import load_networks, load_opts
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    f, g = load_networks(config)
    f, g = f.to(device), g.to(device)
    opts = load_opts(config, f, g)
    return f, g, opts


def train_steps(source: np.ndarray, target: np.ndarray, config: dict, seed: int,
                max_steps: int, batch_size: int, n_inner: int, device: str,
                validation: tuple[np.ndarray, np.ndarray] | None = None,
                eval_every: int = 500, patience: int = 20, bandwidth: float = 1.0):
    import torch
    from cellot.models.cellot import compute_loss_f, compute_loss_g

    f, g, opts = new_model(config, seed, device)
    rng = np.random.default_rng(seed)
    best_metric, best_step, best_state, stale = float("inf"), max_steps, None, 0
    history = []
    for step in range(1, max_steps + 1):
        target_batch = torch.as_tensor(
            target[rng.integers(0, len(target), size=batch_size)], device=device
        )
        for _ in range(n_inner):
            source_batch = torch.as_tensor(
                source[rng.integers(0, len(source), size=batch_size)], device=device
            ).requires_grad_(True)
            opts.g.zero_grad(set_to_none=True)
            transport = g.transport(source_batch)
            loss_g = compute_loss_g(f, g, source_batch, transport).mean()
            if not g.softplus_W_kernels and g.fnorm_penalty > 0:
                loss_g = loss_g + g.penalize_w()
            loss_g.backward(); opts.g.step()
        source_batch = torch.as_tensor(
            source[rng.integers(0, len(source), size=batch_size)], device=device
        ).requires_grad_(True)
        opts.f.zero_grad(set_to_none=True)
        transport = g.transport(source_batch)
        loss_f = compute_loss_f(f, g, source_batch, target_batch, transport).mean()
        loss_f.backward(); opts.f.step(); f.clamp_w()
        if not torch.isfinite(loss_f) or not torch.isfinite(loss_g):
            raise RuntimeError(f"nonfinite CellOT loss at step {step}")
        if validation is not None and (step == 1 or step % eval_every == 0):
            val_source, val_target = validation
            eval_rng = np.random.default_rng(seed + step)
            source_eval = torch.as_tensor(
                val_source[eval_rng.choice(len(val_source), size=min(1024, len(val_source)), replace=False)],
                device=device,
            ).requires_grad_(True)
            target_eval = torch.as_tensor(
                val_target[eval_rng.choice(len(val_target), size=min(1024, len(val_target)), replace=False)],
                device=device,
            )
            transported = g.transport(source_eval).detach()
            metric = float(torch_mmd(transported, target_eval, bandwidth).detach().cpu())
            history.append({"step": step, "validation_mmd": metric,
                            "loss_f": float(loss_f.detach().cpu()),
                            "loss_g": float(loss_g.detach().cpu())})
            print(json.dumps({"phase": "cellot_inner_validation", "seed": seed,
                              **history[-1]}), flush=True)
            if metric < best_metric - 1e-7:
                best_metric, best_step, stale = metric, step, 0
                best_state = {"f": copy.deepcopy(f.state_dict()),
                              "g": copy.deepcopy(g.state_dict())}
            else:
                stale += 1
                if stale >= patience:
                    break
        elif validation is None and (step == 1 or step % 5000 == 0 or step == max_steps):
            print(json.dumps({"phase": "cellot_final_retraining", "seed": seed,
                              "step": step, "max_steps": max_steps,
                              "loss_f": float(loss_f.detach().cpu()),
                              "loss_g": float(loss_g.detach().cpu())}), flush=True)
    if validation is not None and best_state is None:
        raise RuntimeError("CellOT inner validation produced no checkpoint")
    return f, g, opts, best_step, best_metric, history, best_state


def main() -> None:
    p = parser("Official CellOT future single-cell distribution baseline")
    p.add_argument("--training-latent-h5", required=True)
    p.add_argument("--test-latent-h5", required=True)
    p.add_argument("--scale-npz", required=True)
    p.add_argument("--cells-per-episode-condition", type=int, default=512)
    p.add_argument("--generated-cells", type=int, default=2048)
    p.add_argument("--max-iterations", type=int, default=reference_profile("CellOT")["max_iterations"])
    p.add_argument("--batch-size", type=int, default=reference_profile("CellOT")["batch_size"])
    p.add_argument("--n-inner-iterations", type=int, default=reference_profile("CellOT")["n_inner_iterations"])
    p.add_argument("--eval-every", type=int, default=reference_profile("CellOT")["eval_every"])
    p.add_argument("--early-stopping-patience", type=int, default=reference_profile("CellOT")["patience"])
    p.add_argument("--scope", choices=["cold_start", "year2_locked"], default="cold_start")
    args = p.parse_args()
    reference = validate_reference_arguments("CellOT", args)
    if args.fold not in range(5): p.error("--fold must be 0..4")
    if args.lineage != "all":
        p.error("--lineage is deprecated; the vaccine runner always uses all cells")
    run_name = (
        f"cellot_{args.scope}_outer{args.fold}_day{args.source_day}"
        f"_to_day{args.target_day}_seed{args.seed}"
    )
    output_dir = Path(args.output_dir or ROOT / "results/baseline_single_cell/cellot" / run_name)
    result_path, manifest_path = output_dir / "run.json", output_dir / "predictions.tsv"
    checkpoint = output_dir / "checkpoint.pt"
    if args.dry_run:
        print(json.dumps({"run_name": run_name, "official_source": str(CELLOT_SOURCE),
                          "four_inner_donor_folds": True, "final_retraining": True,
                          "max_iterations": args.max_iterations,
                          "cell_scope": "all_cells", "output": str(output_dir)}, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    run_lock_handle = (output_dir / ".run.lock").open("w")
    fcntl.flock(run_lock_handle, fcntl.LOCK_EX)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("explicit CUDA_VISIBLE_DEVICES is required")
    if args.resume and result_path.is_file() and manifest_path.is_file() and checkpoint.is_file():
        prior = json.loads(result_path.read_text())
        if prior.get("status") == "pass":
            print(json.dumps({"status": "resume", "result": str(result_path)}, indent=2)); return
    sys.path.insert(0, str(CELLOT_SITE)); sys.path.insert(0, str(CELLOT_SOURCE))
    from absl import flags
    if not flags.FLAGS.is_parsed(): flags.FLAGS(["cellot_v4"])
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU must be visible; CPU fallback prohibited")
    device = "cuda"
    with np.load(args.scale_npz, allow_pickle=False) as frozen:
        center, scale = frozen["center"].astype(np.float32), frozen["scale"].astype(np.float32)
        bandwidth = float(frozen["mmd_bandwidth"])
        fit_role = str(frozen["fit_role"])
    if "training" not in fit_role.lower():
        raise RuntimeError("CellOT scale must be fitted on outer-training cells only")
    splits = pd.read_csv(ROOT / "config/splits/sound_outer_groupkfold5_donors.tsv", sep="\t")
    inner = pd.read_csv(ROOT / "config/splits/sound_inner_groupkfold4_donors.tsv", sep="\t")
    episodes = pd.read_csv(ROOT / "data/episode_manifests/sound_flu_episodes.tsv", sep="\t")
    test_donors = set(splits.loc[splits.outer_fold == args.fold, "donor_id"].astype(str))
    complete, target_column = f"complete_D0_D{args.target_day}", f"sample_D{args.target_day}"
    training_episodes = episodes.loc[(~episodes.donor_id.astype(str).isin(test_donors)) &
                                     episodes[complete].astype(bool)].copy()
    test_episodes = episodes.loc[(episodes.donor_id.astype(str).isin(test_donors)) &
                                 episodes[complete].astype(bool)].copy()
    if args.scope == "year2_locked":
        training_episodes = training_episodes.loc[training_episodes.season.astype(int) == 1].copy()
        test_episodes = test_episodes.loc[test_episodes.season.astype(int) == 2].copy()
    fold_assignments = inner.loc[inner.outer_fold == args.fold].set_index("donor_id").inner_fold
    reader, test_reader = Reader(Path(args.training_latent_h5)), Reader(Path(args.test_latent_h5))
    started = time.time(); inner_records = []
    try:
        config = model_config(len(center))
        best_steps = []
        for inner_fold in range(reference_profile("CellOT")["inner_patient_folds"]):
            validation_donors = set(fold_assignments.loc[fold_assignments == inner_fold].index.astype(str))
            fit_episodes = training_episodes.loc[
                ~training_episodes.donor_id.astype(str).isin(validation_donors)
            ]
            validation_episodes = training_episodes.loc[
                training_episodes.donor_id.astype(str).isin(validation_donors)
            ]
            fit_source, fit_target, _ = episode_arrays(
                reader, fit_episodes, target_column,
                args.cells_per_episode_condition, args.seed + inner_fold * 1000,
            )
            val_source, val_target, _ = episode_arrays(
                reader, validation_episodes, target_column,
                args.cells_per_episode_condition, args.seed + inner_fold * 1000 + 1,
            )
            fit_source = (fit_source - center) / scale; fit_target = (fit_target - center) / scale
            val_source = (val_source - center) / scale; val_target = (val_target - center) / scale
            _, _, _, best_step, best_metric, history, _ = train_steps(
                fit_source, fit_target, config, args.seed + inner_fold,
                args.max_iterations, args.batch_size, args.n_inner_iterations, device,
                validation=(val_source, val_target), eval_every=args.eval_every,
                patience=args.early_stopping_patience, bandwidth=bandwidth,
            )
            best_steps.append(best_step)
            inner_records.append({"inner_fold": inner_fold, "best_step": best_step,
                                  "best_validation_mmd": best_metric, "history": history,
                                  "validation_donors": sorted(validation_donors)})
        final_steps = int(math.ceil(float(np.median(best_steps))))
        all_source, all_target, used = episode_arrays(
            reader, training_episodes, target_column,
            args.cells_per_episode_condition, args.seed + 9000,
        )
        all_source = (all_source - center) / scale; all_target = (all_target - center) / scale
        f, g, opts, _, _, _, _ = train_steps(
            all_source, all_target, config, args.seed + 10_000, final_steps,
            args.batch_size, args.n_inner_iterations, device,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"f_state": f.state_dict(), "g_state": g.state_dict(),
                    "opt_f_state": opts.f.state_dict(), "opt_g_state": opts.g.state_dict(),
                    "config": config, "final_steps": final_steps, "seed": args.seed}, checkpoint)
        probe = torch.as_tensor(all_source[:min(128, len(all_source))], device=device).requires_grad_(True)
        before = g.transport(probe).detach()
        f2, g2, _ = new_model(config, args.seed + 10_000, device)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        f2.load_state_dict(saved["f_state"]); g2.load_state_dict(saved["g_state"])
        probe2 = probe.detach().clone().requires_grad_(True)
        after = g2.transport(probe2).detach()
        reload_max_abs = float(torch.max(torch.abs(before - after)).cpu())
        if reload_max_abs > 1e-6:
            raise RuntimeError(f"CellOT reload mismatch: {reload_max_abs}")
        records = []
        for index, row in test_episodes.reset_index(drop=True).iterrows():
            d0_rows = test_reader.rows(str(row.sample_D0))
            future_rows = test_reader.rows(str(row[target_column]))
            if not len(d0_rows) or not len(future_rows): continue
            query, query_state, query_rows = test_reader.take(
                d0_rows, args.generated_cells, np.random.default_rng(args.seed + 20_000 + index), True
            )
            standardized = torch.as_tensor((query - center) / scale, device=device).requires_grad_(True)
            predicted = g2.transport(standardized).detach().cpu().numpy() * scale + center
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
                     cell_scope=np.asarray("all_cells"), method=np.asarray("CellOT_official"))
            save_npz(observed_path, embedding=observed, cell_state=observed_state,
                     source_row_index=observed_rows, donor_id=np.asarray(str(row.donor_id)),
                     episode_id=np.asarray(identity), target_day=np.asarray(args.target_day),
                     cell_scope=np.asarray("all_cells"), role=np.asarray("observed_future"))
            records.append({"donor_id": str(row.donor_id), "episode_id": identity,
                            "season": int(row.season), "outer_fold": args.fold,
                            "cell_scope": "all_cells", "target_day": args.target_day,
                            "method": "CellOT_official", "source_path": str(source_path),
                            "observed_path": str(observed_path),
                            "predicted_path": str(predicted_path),
                            "prediction_uses_test_future": False})
        manifest = pd.DataFrame(records)
        if manifest.empty: raise RuntimeError("CellOT produced no held-out distributions")
        manifest.to_csv(manifest_path, sep="\t", index=False)
    finally:
        reader.close(); test_reader.close()
    commit = subprocess.check_output(["git", "-C", str(CELLOT_SOURCE), "rev-parse", "HEAD"],
                                     text=True).strip()
    parameter_count = sum(p.numel() for p in f2.parameters()) + sum(p.numel() for p in g2.parameters())
    payload = {
        **reference,
        "status": "pass", "run_name": run_name, "scope": args.scope,
        "year2_used_for_training_or_selection": False if args.scope == "year2_locked" else None,
        "implementation": "official CellOT ICNN and losses", "official_commit": commit,
        "official_source": str(CELLOT_SOURCE), "outer_fold": args.fold,
        "target_day": args.target_day, "cell_scope": "all_cells", "seed": args.seed,
        "four_inner_donor_folds": inner_records, "final_steps": final_steps,
        "final_training_donors": int(training_episodes.donor_id.nunique()),
        "final_training_episodes": len(used), "model_parameters": parameter_count,
        "input_role": f"frozen per-cell scVI state embeddings at Day {args.source_day}",
        "output_role": "future per-cell state embedding distribution",
        "cell_type_role": "retained output metadata only; not a CellOT input",
        "donor_blocked": True, "source_day": args.source_day,
        "test_source_only_for_prediction": True,
        "test_future_used_for_training_or_adaptation": False,
        "training_only_scale": str(args.scale_npz),
        "checkpoint": str(checkpoint),
        "reload_max_abs_difference": reload_max_abs,
        "manifest": str(manifest_path),
        "n_test_distributions": len(manifest), "elapsed_seconds": time.time() - started,
        "cuda_device_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
    }
    write_json(result_path, payload); print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

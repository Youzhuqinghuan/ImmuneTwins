from __future__ import annotations
from paths import FROZEN_RESULTS

TASKS = ("treatment", "t1_y1_d0_to_y1_d7", "t2_d7_to_d90", "t3_y2_d0_to_y2_d90")

def origin(task, fold):
    if task == "treatment":
        return FROZEN_RESULTS / f"drug_discovery_mm/official_fixed_reference_20260905/fold_{fold}/baselines/trVAE"
    return FROZEN_RESULTS / f"baselines/vaccine_v2/official/trvae/{task}/trvae_{task}_outer{fold}_seed42"


import argparse
import fcntl
import json
import os
import pickle
import random
import resource
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from paths import ROOT, WORKSPACE
from baseline.vaccine_runners.common import reference_profile
from baseline.vaccine_runners.trusted_torch_checkpoint import trusted_local_full_object_load

RUNTIME = WORKSPACE / "vaccine_baseline_runtime"

DEFAULT_OUTPUT = ROOT / "results/baseline_correction/trvae_signed"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def signed_observation_head(model):
    """Keep parameter keys/initialization; remove only the nonnegative support."""
    decoder = model.model.decoder.recon_decoder
    if not (isinstance(decoder, torch.nn.Sequential) and len(decoder) == 2
            and isinstance(decoder[0], torch.nn.Linear)
            and isinstance(decoder[1], (torch.nn.ReLU, torch.nn.Identity))):
        raise RuntimeError("Unexpected upstream MSE observation head")
    decoder[1] = torch.nn.Identity()

def decode(model, query, source_condition, target_condition):
    model.model.eval()
    latent = model.get_latent(query, np.asarray([source_condition] * len(query)), mean=True)
    code = model.model.condition_encoder[target_condition]
    values = []
    for start in range(0, len(latent), 512):
        z = torch.as_tensor(latent[start:start + 512], dtype=torch.float32, device="cuda")
        c = torch.full((len(z),), code, dtype=torch.long, device=z.device)
        with torch.no_grad():
            values.append(model.model.decoder(z, c)[0].cpu().numpy())
    return np.concatenate(values).astype(np.float32)

def load_checkpoint(path, model_class):
    """Always restore the declared observation-domain adapter after native load."""
    contract = json.loads((Path(path).parent / "adapter.json").read_text())
    if contract["output_activation"] != "identity":
        raise RuntimeError("Not a signed-latent trVAE checkpoint")
    with trusted_local_full_object_load(torch, Path(path) / "model_params.pt"):
        model = model_class.load(path, map_location="cuda")
    signed_observation_head(model)
    model.model.to("cuda")
    model.is_trained_ = True
    return model

def treatment_inputs(fold, adata):
    manifest = WORKSPACE / "data/ImmuneTwins/drug_discovery_mm/manifests/gse309595/episodes.tsv"
    latent = FROZEN_RESULTS / f"drug_discovery_mm/gse309595_state/fold_{fold}/latent_visits.h5"
    episodes = pd.read_csv(manifest, sep="\t")
    episodes = episodes[episodes.cohort_id.eq("GSE309595")].sort_values("patient_id")
    train = episodes[episodes.outer_fold.ne(fold)]
    test = episodes[episodes.outer_fold.eq(fold)]
    with h5py.File(latent, "r") as h:
        visits = [v.decode() if isinstance(v, bytes) else str(v) for v in h["visit_id"][:]]
        index = {v: i for i, v in enumerate(visits)}
        arrays = [np.concatenate([h["embedding"][index[v]] for v in train[col]])
                  for col in ["source_visit_id", "target_visit_id"]]
    combined = np.concatenate(arrays).astype(np.float32)
    center = combined.mean(0)
    scale = np.maximum(combined.std(0), 1e-3)
    np.testing.assert_array_equal(adata.X, (combined - center) / scale)
    phases = train.drug_id.map({"VRd": 1, "VRd_Dara": 2})
    assert phases.notna().all()
    conditions = np.concatenate([np.repeat([f"{role}_phase{p}" for p in phases], 512)
                                 for role in ["source", "target"]])
    np.testing.assert_array_equal(adata.obs.condition.astype(str), conditions)
    assert set(train.patient_id).isdisjoint(test.patient_id)
    return center, scale, test.set_index("patient_id")

def train(task, fold, output_root):
    out = Path(output_root).resolve() / task / f"fold_{fold}"
    out.relative_to(ROOT / "results")
    if (out / "run.json").exists():
        raise FileExistsError(f"Completed results are immutable: {out}")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "checkpoint").exists():
        raise FileExistsError("Partial checkpoint exists; retain it and use a fresh output root")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 1 or visible[0] in {"", "0"} or not torch.cuda.is_available():
        raise RuntimeError("One explicit non-reserved CUDA GPU is required")
    assert float(torch.ones(1, device="cuda").sum()) == 1
    reserved = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid",
                                        "--format=csv,noheader"], text=True).strip()
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert actual_uuid.removeprefix("GPU-") != reserved.removeprefix("GPU-")
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    previous = origin(task, fold)
    run = json.loads((previous / "run.json").read_text())
    upstream = RUNTIME / "external/scarches_v051"
    sys.path[:0] = [str(RUNTIME / "environments/scarches_v051_site"), str(upstream)]
    import anndata as ad
    from scarches.models.trvae.trvae_model import TRVAE

    adata = ad.read_h5ad(previous / "checkpoint/adata.h5ad")
    with (previous / "checkpoint/attr.pkl").open("rb") as handle:
        attributes = pickle.load(handle)
    keys = ("condition_key", "conditions", "hidden_layer_sizes", "latent_dim", "dr_rate",
            "use_mmd", "mmd_on", "recon_loss", "beta", "use_bn", "use_ln", "mmd_boundary")
    kwargs = {k: attributes[k + "_"] for k in keys}
    assert kwargs["recon_loss"] == "mse" and np.isfinite(adata.X).all()
    if task == "treatment":
        center, scale, test = treatment_inputs(fold, adata)
    else:
        scale_path = Path(run["training_only_scale"])
        with np.load(scale_path, allow_pickle=False) as z:
            center, scale = z["center"].astype(np.float32), z["scale"].astype(np.float32)
            assert "training" in str(z["fit_role"]).lower()
        test = pd.read_csv(previous / "predictions.tsv", sep="\t")
        assert test.outer_fold.eq(fold).all()
    seed = int(run["seed"])
    profile = reference_profile("trVAE")
    if "reference_profile" in run:
        assert profile == run["reference_profile"]
    contract = {"output_activation": "identity", "reason": "Gaussian MSE targets are signed continuous latents",
                "only_architecture_change": "terminal parameter-free ReLU replaced with Identity",
                 "task": task, "fold": fold,
                "seed": seed, "reference_profile": profile, "model_kwargs": kwargs,
                 "test_outcome_selection": False,
                "baseline_effect_search": False, "training_matrix_exactly_reused": True,
                "training_negative_fraction": float((adata.X < 0).mean()),
                "cuda_device_uuid": actual_uuid,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    write_json(out / "adapter.json", contract)
    np.savez_compressed(out / "normalization.npz", center=center, scale=scale)
    write_json(out / "progress.json", {"status": "training", "pid": os.getpid(), "gpu": visible[0]})
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = TRVAE(adata, **kwargs)
    signed_observation_head(model)
    assert sum(p.numel() for p in model.model.parameters()) == int(run["model_parameters"])
    started = time.time()
    model.train(n_epochs=profile["max_epochs"], lr=1e-3, eps=0.01,
                batch_size=profile["batch_size"], train_frac=0.9,
                use_stratified_sampling=True, n_workers=0, seed=seed, monitor=False,
                use_early_stopping=True, reload_best=True,
                early_stopping_kwargs={"early_stopping_metric": "val_unweighted_loss",
                    "patience": profile["patience"], "threshold": 0, "reduce_lr": True,
                    "lr_patience": profile["lr_patience"], "lr_factor": 0.1})
    assert next(model.model.parameters()).device.type == "cuda"
    checkpoint = out / "checkpoint"
    model.save(checkpoint, overwrite=False, save_anndata=True)
    np.testing.assert_array_equal(adata.X, ad.read_h5ad(previous / "checkpoint/adata.h5ad").X)
    probe = np.asarray(adata.X[:128])
    src, dst = kwargs["conditions"][0], kwargs["conditions"][-1]
    before = decode(model, probe, src, dst)
    reloaded = load_checkpoint(checkpoint, TRVAE)
    after = decode(reloaded, probe, src, dst)
    np.testing.assert_array_equal(before, after)
    if task == "treatment":
        path = previous.parents[1] / "predictions/trVAE/test.npz"
        with np.load(path, allow_pickle=False) as z:
            payload = {k: z[k] for k in z.files}
        assert set(payload["patient_id"]) == set(test.index)
        predicted = []
        for i, patient in enumerate(payload["patient_id"]):
            phase = {"VRd": 1, "VRd_Dara": 2}[test.loc[patient, "drug_id"]]
            query = (payload["source"][i] - center) / scale
            predicted.append(decode(reloaded, query, f"source_phase{phase}", f"target_phase{phase}") * scale + center)
        payload["predicted"] = np.stack(predicted).astype(np.float32)
        assert np.isfinite(payload["predicted"]).all()
        np.savez_compressed(out / "test.npz", **payload)
        prediction_files = [out / "test.npz"]
        pred_values = payload["predicted"]
    else:
        rows = []
        prediction_files = []
        all_predicted = []
        for row in test.to_dict("records"):
            source = previous / f"source_day{run['source_day']}" / Path(row["source_path"]).name
            observed = previous / "observed" / Path(row["observed_path"]).name
            old_pred = previous / "predicted" / Path(row["predicted_path"]).name
            with np.load(source, allow_pickle=False) as z:
                query = (z["embedding"] - center) / scale
            values = decode(reloaded, query, f"D{run['source_day']}", f"D{run['target_day']}") * scale + center
            with np.load(old_pred, allow_pickle=False) as z:
                payload = {k: z[k] for k in z.files}
            payload["embedding"] = values.astype(np.float32)
            assert np.isfinite(values).all()
            destination = out / "predicted" / old_pred.name
            destination.parent.mkdir(exist_ok=True)
            np.savez_compressed(destination, **payload)
            row.update(source_path=str(source), observed_path=str(observed), predicted_path=str(destination))
            rows.append(row); prediction_files.append(destination); all_predicted.append(values)
        pd.DataFrame(rows).to_csv(out / "predictions.tsv", sep="\t", index=False)
        pred_values = np.concatenate(all_predicted)
    result = {**contract, "status": "PASS",
              "actual_epochs": len(model.trainer.logs["epoch_loss"]),
              "best_epoch_zero_based": int(model.trainer.best_epoch),
              "reload_max_abs_difference": float(np.max(np.abs(before - after))),
              "elapsed_seconds": time.time() - started,
              "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "prediction_below_training_center_fraction": float((pred_values < center).mean()),
              "predictions": [str(p) for p in prediction_files],
              "test_future_used_for_training_or_adaptation": False,
              "frozen_inputs_preserved": True}
    write_json(out / "run.json", result)
    write_json(out / "progress.json", {"status": "complete", "pid": os.getpid()})
    print(json.dumps(result, indent=2), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--all", action="store_true", help="Run all fixed folds; no model selection")
    parser.add_argument("--gpus", default="2,3,5", help="Physical nvidia-smi indices; resolved to UUIDs")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.all:
        run_all(args.output_root, [int(v) for v in args.gpus.split(",")])
    else:
        if args.task is None or args.fold is None:
            parser.error("--task and --fold are required for a single job")
        train(args.task, args.fold, args.output_root)

def run_all(output_root, gpus):
    """Bounded background queue."""
    root = Path(output_root).resolve()
    root.relative_to(ROOT / "results")
    assert gpus and 0 not in gpus and len(set(gpus)) == len(gpus)
    devices = {}
    for gpu in gpus:
        uuid = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid",
                                        "--format=csv,noheader"], text=True).strip()
        probe = subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(); "
                                "assert torch.ones(1,device='cuda').sum().item()==1"],
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": uuid}, capture_output=True, text=True)
        if probe.returncode:
            raise RuntimeError(f"Physical GPU {gpu} failed CUDA preflight: {probe.stderr}")
        devices[gpu] = uuid
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = queue.Queue()
    records = []
    for task in TASKS:
        for fold in range(5):
            out = root / task / f"fold_{fold}"
            row = {"task": task, "fold": fold, "status": "pending"}
            if (out / "run.json").exists():
                prior = json.loads((out / "run.json").read_text())
                assert prior["status"] == "PASS" and prior["output_activation"] == "identity"
                assert all(Path(p).is_file() for p in prior["predictions"])
                assert (out / "checkpoint/model_params.pt").is_file()
                row["status"] = "complete"
            else:
                jobs.put(row)
            records.append(row)
    mutex = threading.Lock()
    def status():
        write_json(root / "queue_status.json", {"pid": os.getpid(), "gpus": devices,
            "jobs": records, "complete": sum(r["status"] == "complete" for r in records),
            "failed": sum(r["status"] == "failed" for r in records)})
    status()
    def worker(gpu):
        while True:
            try:
                row = jobs.get_nowait()
            except queue.Empty:
                return
            out = root / row["task"] / f"fold_{row['fold']}"
            out.mkdir(parents=True, exist_ok=True)
            command = ["bash", str(ROOT / "run.sh"), "-m", "baseline.trvae",
                       "--task", row["task"], "--fold", str(row["fold"]),
                       "--output-root", str(root)]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": devices[gpu], "PYTHONUNBUFFERED": "1"}
            with (out / "training.log").open("ab") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                with mutex:
                    row.update(status="running", pid=process.pid, gpu=gpu)
                    status()
                code = process.wait()
            with mutex:
                row.update(status="complete" if code == 0 else "failed", exit_code=code)
                status()
            jobs.task_done()
            if code:
                return  # Do not drain the queue into a failed device or runtime.
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))
    if any(r["status"] != "complete" for r in records):
        raise RuntimeError("One or more fixed jobs failed; inspect queue_status.json")

if __name__ == "__main__":
    main()

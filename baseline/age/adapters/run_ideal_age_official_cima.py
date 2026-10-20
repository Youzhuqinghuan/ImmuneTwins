from __future__ import annotations
import json
import os
import pickle
import random
import sys
import types
from pathlib import Path
import numpy as np
import torch
from baseline.age.workspace import WORKSPACE_ROOT

OFFICIAL_REPO = WORKSPACE_ROOT / "models/ImmuneTwins/baselines/IDEAL-Age"

OFFICIAL_MODEL_FILE = OFFICIAL_REPO / "ag_extra_models.py"


def import_official_model():
    """Import the released class, stubbing optional imports."""

    autogluon = types.ModuleType("autogluon")
    core = types.ModuleType("autogluon.core")
    models = types.ModuleType("autogluon.core.models")

    class AbstractModel:
        pass

    models.AbstractModel = AbstractModel
    autogluon.core = core
    core.models = models
    sys.modules.setdefault("autogluon", autogluon)
    sys.modules.setdefault("autogluon.core", core)
    sys.modules.setdefault("autogluon.core.models", models)

    iage = types.ModuleType("iage_lr_train_by_dataset_debug")

    class ImportOnlyPlaceholder:
        pass

    iage.scNETAgePredictor = ImportOnlyPlaceholder
    iage.scNETAgePredictorFast = ImportOnlyPlaceholder
    sys.modules.setdefault("iage_lr_train_by_dataset_debug", iage)
    sys.path.insert(0, str(OFFICIAL_REPO))
    import ag_extra_models  # type: ignore

    return ag_extra_models

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def selected_cells(values: np.ndarray, donor_indices: np.ndarray, choices: np.ndarray) -> torch.Tensor:
    block = np.asarray(values[donor_indices], dtype=np.float32).reshape(len(donor_indices), -1, values.shape[-1])
    rows = np.arange(len(donor_indices))[:, None]
    return torch.from_numpy(np.ascontiguousarray(block[rows, choices[donor_indices]]))

def materialize_master(master_path: Path, output_path: Path, max_cells: int, seed: int):
    """Pack released/scPhase donor shards into one deterministic IDEAL input mmap."""
    with master_path.open("rb") as handle:
        master = pickle.load(handle)
    refs = master["data_list_dense"]
    donor_ids = np.asarray(master["sample_ids"], dtype=str)
    labels = np.asarray(master["labels"], dtype=np.float32)
    if output_path.exists():
        return np.load(output_path, mmap_mode="r"), donor_ids, labels
    first = np.load(refs[0]["path"], mmap_mode="r")
    if first.shape[0] < max_cells:
        raise ValueError(f"IDEAL-Age needs {max_cells} cells, shard has {first.shape[0]}")
    temporary = output_path.with_suffix(".partial.npy")
    packed = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(refs), max_cells, first.shape[1])
    )
    rng = np.random.default_rng(seed)
    for i, ref in enumerate(refs):
        shard = np.load(ref["path"], mmap_mode="r")
        # Read the small donor shard sequentially before selecting rows.  Direct
        # advanced indexing of a memmap can degrade into many page-sized reads.
        # Sorting changes only storage traversal; the released attention model
        # is permutation invariant.
        choice = np.sort(rng.choice(shard.shape[0], max_cells, replace=False))
        donor_block = np.asarray(shard, dtype=np.float32)
        packed[i] = donor_block[choice]
        if (i + 1) % 25 == 0 or i + 1 == len(refs):
            print(f"packed {i + 1}/{len(refs)} donors from {master_path.name}", flush=True)
    # One final flush avoids repeatedly synchronizing the full multi-gigabyte
    # mmap at every progress report.
    packed.flush()
    del packed
    os.replace(temporary, output_path)
    return np.load(output_path, mmap_mode="r"), donor_ids, labels

def predict(model, values, indices, choices, batch_size, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            donor_idx = indices[start : start + batch_size]
            cells = selected_cells(values, donor_idx, choices).to(device)
            mask = torch.ones(cells.shape[:2], dtype=torch.float32, device=device)
            pred, _ = model(cells, mask)
            outputs.append(pred.float().cpu().numpy())
    return np.concatenate(outputs)

def train_track(module, values, ages, train_idx, choices, args, run_dir, name, device):
    checkpoint = run_dir / f"{name}.pt"
    prediction_marker = run_dir / f"{name}.complete.json"
    if checkpoint.exists() and prediction_marker.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = module.DeepSetsAgePredictorAttn(
            input_dim=values.shape[-1], hidden_dim=2**11, dropout=0.2, num_classes=10
        ).to(device)
        model.load_state_dict(payload["model_state_dict"])
        return model

    seed_all(args.seed)
    model = module.DeepSetsAgePredictorAttn(
        input_dim=values.shape[-1], hidden_dim=2**11, dropout=0.2, num_classes=10
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rng = np.random.default_rng(args.seed)
    start_epoch = 0
    resume = run_dir / f"{name}.resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        if getattr(args, 'exact_resume', False) and 'random_state' not in state:
            raise RuntimeError('Exact resume requires saved RNG state; retain this partial run and restart separately')
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state["epoch"])
        if getattr(args, 'exact_resume', False):
            randomness = state['random_state']
            rng.bit_generator.state = randomness['batch_order']
            random.setstate(randomness['python'])
            np.random.set_state(randomness['numpy'])
            torch.set_rng_state(randomness['torch'].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in randomness['cuda']])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        order = rng.permutation(train_idx)
        for start in range(0, len(order), args.batch_size):
            donor_idx = order[start : start + args.batch_size]
            cells = selected_cells(values, donor_idx, choices).to(device)
            targets = torch.as_tensor(ages[donor_idx], dtype=torch.float32, device=device)
            mask = torch.ones(cells.shape[:2], dtype=torch.float32, device=device)
            bins = (targets.clamp(0, 100) / 10).long().clamp(0, 9)
            optimizer.zero_grad()
            pred, cache = model(cells, mask, return_logits=True)
            regression = torch.nn.functional.smooth_l1_loss(pred, targets)
            classification = torch.nn.functional.cross_entropy(cache["logits"], bins)
            with torch.no_grad():
                probs = cache["logits"].softmax(dim=-1)
                ks = torch.arange(10, device=device).float()
                classified_age = (probs * ((ks + 0.5) * 10.0)).sum(dim=-1)
            consistency = torch.nn.functional.smooth_l1_loss(pred, classified_age)
            loss = regression + 0.5 * classification + 0.1 * consistency
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"{name} epoch={epoch + 1:03d}/{args.epochs} loss={np.mean(losses):.6f}", flush=True)
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            random_state = ({'random_state': dict(batch_order=rng.bit_generator.state,
                python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())} if getattr(args, 'exact_resume', False) else {})
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    **random_state,
                },
                resume,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": args.epochs,
            "official_class_module": model.__class__.__module__,
            "official_class_name": model.__class__.__name__,
            "official_source": str(OFFICIAL_MODEL_FILE),
            "hidden_dim": 2**11,
            "max_cells": args.max_cells,
            "batch_size": args.batch_size,
            "learning_rate": 0.001,
            "loss": "SmoothL1 + 0.5*CrossEntropy + 0.1*SmoothL1-consistency",
        },
        checkpoint,
    )
    prediction_marker.write_text(json.dumps({"complete": True, "epochs": args.epochs}) + "\n")
    return model

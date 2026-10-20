"""Inner neural fits used to calibrate the final Sound personal-reference layer."""
from __future__ import annotations
import argparse
import json
import math
import os
import resource
import shutil
import sys
import time
from pathlib import Path
from dataclasses import asdict, replace
import pandas as pd
import torch
from paths import ROOT
from data.reference import TASKS, inner_inputs, neural_run, resolve_path
from data.sound import (build_task_units, split_outer_units, split_inner_units, load_fold_scale,
                        FoldLatentStore, SealedBaseUCEStore, UnifiedUCEStore, UnitMaterializer)
from pipeline.sound import train as tr
from pipeline.sound.common import  set_seed, atomic_torch_save, atomic_table
from pipeline.sound.provenance import atomic_json
from pipeline.sound.select_projected_set import _load_model as load_projected, SCHEMA as PS_SCHEMA
from pipeline.sound.select_digital_twin import _load_projected_set, _checkpoint as twin_checkpoint
from pipeline.sound.select_epoch import main as select_epoch
from model.digital_twin import MetaAdapterConfig, MetaAdapterDigitalTwin
from model.losses import PatientLossV2Config

class InnerTrainer:
    """Fit one outer-training calibration partition at a time on the selected GPU."""

    def __init__(self, config):
        self.cfg = config
        self.inputs = inner_inputs(config)
        self.output = Path(config["inner_root"])

    def path(self, value):
        return resolve_path(value, self.cfg)

    def run(self, outer, task, device):
        for inner in range(4):
            parent = self.parent(task, outer, inner, device)
            projected = self.projected(task, outer, inner, device, parent)
            if task != TASKS[0]:
                self.full(task, outer, inner, device, projected)
        if task != TASKS[0]:
            selected = self.output / "epoch_selection" / f"outer{outer}_{task}.json"
            if not selected.exists():
                arguments = []
                for inner in range(4):
                    arguments += ["--summary", str(self.run_dir("full", task, outer, inner) / "summary.json")]
                rule = "median_fold_best" if task == TASKS[1] else "pooled_common_curve"
                select_epoch(arguments + ["--selection-rule", rule, "--output", str(selected)])
            for inner in range(4):
                self.finalize_full(task, outer, inner, device)

    def run_dir(self, stage, task, outer, inner):
        return self.output / 'runs' / f'outer{outer}' / f'inner{inner}' / task / stage

    def split(self, task, outer, inner):
        units = build_task_units(self.path(self.inputs['task_units']), task)
        (train, test) = split_outer_units(units, outer)
        (fit, val) = split_inner_units(train, self.path(self.inputs['inner_splits']), outer_fold=outer, validation_fold=inner)
        fs = {u.donor_id for u in fit}
        vs = {u.donor_id for u in val}
        ts = {u.donor_id for u in test}
        assert not (fs & vs or fs & ts or vs & ts)
        return (fit, val, dict(fit_donors=sorted(fs), validation_donors=sorted(vs), outer_test_donors=sorted(ts),   outer=outer, inner=inner, task=task, weight_disjoint=True, inner_labels_used_for_epoch_selection=True, outer_test_targets_opened=False))

    def stores(self, outer, runtime, full=False):
        scale = load_fold_scale(self.path(self.inputs['folds'][str(outer)]['scale']), outer_fold=outer)
        latent = FoldLatentStore(self.path(self.inputs['folds'][str(outer)]['training_latent']), scale)
        if full:
            frame = pd.read_csv(self.path(self.inputs['visit_manifest']), sep='\t')
            for col in ['uce_features_h5', 'gene_summary_h5', 'raw_counts_h5']:
                for v in frame[col].astype(str).unique():
                    self.path(v)
            uce = UnifiedUCEStore(self.path(self.inputs['visit_manifest']))
        else:
            uce = SealedBaseUCEStore(self.path(self.inputs['sealed_source']), self.path(self.inputs['sealed_uce']))
        mat = UnitMaterializer(uce=uce, latent=latent, query_uce_cells=runtime.query_uce_cells, support_uce_cells=runtime.support_uce_cells, seed=42)
        return (mat, latent, uce, scale)

    def resource_info(self, device, started):
        torch.cuda.synchronize(device)
        return dict(elapsed_seconds=time.time() - started, peak_gpu_memory_mib=torch.cuda.max_memory_allocated(device) / 2 ** 20, process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, gpu=torch.cuda.get_device_name(device), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES', ''), torch=torch.__version__, python=sys.executable)

    def parent(self, task, outer, inner, device):
        (fit, val, audit) = self.split(task, outer, inner)
        if inner == 0:
            p = neural_run(self.cfg, task, outer, affine=True) / 'best_validation.pt'
            selection = json.loads(self.path(p.parent / 'selection.json').read_text())
            if selection.get('status') != 'pass' or not p.is_file():
                raise ValueError('Matched parent selection is incomplete')
            return p
        out = self.run_dir('parent', task, outer, inner)
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / 'split_audit.json', audit)
        config = self.path(ROOT / 'config/sound_backbone.yaml')
        (_, runtime, mc, loss) = tr.load_configuration_v2(config, arm='current', task=task)
        (mat, latent, uce, scale) = self.stores(outer, runtime)
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            transform = tr.estimate_response_transform(latent, fit, cells=runtime.training_latent_cells, seed=42, stage='selection', scale_floor=runtime.response_scale_floor, mean_program_rank=mc.mean_program_rank, covariance_program_rank=mc.covariance_program_rank)
            tr.save_response_transform(transform, out / 'response_transform.json')
            (epoch, hist, loaded) = tr.select_epoch_v2(arm='current', task=task, outer_fold=outer, fit_units=fit, validation_units=val, materializer=mat, runtime=runtime, model_config=mc, loss_config=loss, bandwidth=scale.mmd_bandwidth, cima_checkpoint=self.path(self.inputs['cima']),  gene_selection=None, response_transform=transform, clinical_mode='off', output=out,  preprocessing=audit, device=device, resume=True, config_path=config)
            atomic_json(out / 'summary.json', dict(status='pass', stage='parent', task=task, outer_fold=outer, inner_validation_fold=inner, best_epoch=epoch, epochs=len(hist),  **self.resource_info(device, started)))
            return out / 'best_validation.pt'
        finally:
            latent.close()
            uce.close()

    def evaluate(self, model, mat, val, arm, runtime, loss, scale, device):
        return tr.evaluate_donors_v2(model, mat, val, arm=arm, runtime=runtime, loss_config=loss, bandwidth=scale.mmd_bandwidth, response_scale=(model.backbone if hasattr(model, 'backbone') else model).transport.response_scale.detach().cpu().numpy(), clinical_mode='off', device=device)

    def infer(self, model, mat, val, runtime, arm, out, checkpoint, device, outer, audit):
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, out / 'checkpoint.pt')
        val_coordinate = [replace(u, outer_fold=outer) for u in val]
        (manifest_df, art) = tr.predict_heldout_v2(model=model, arm=arm, units=val_coordinate, materializer=mat, runtime=runtime, clinical_mode='off', gene_selection=None, output=out, device=device, paper_method='inner_full' if arm == 'full' else 'inner_no_twin')
        lookup = {u.episode_id: u for u in val}
        manifest_df['inner_validation_fold'] = audit['inner']
        manifest_df['donor_outer_fold'] = [lookup[str(x)].outer_fold for x in manifest_df.episode_id]
        manifest_df['source_sample'] = [lookup[str(x)].source_sample for x in manifest_df.episode_id]
        manifest_df['target_sample'] = [lookup[str(x)].target_sample for x in manifest_df.episode_id]
        atomic_table(out / 'predictions.tsv', manifest_df)
        atomic_json(out / 'split_audit.json', audit)
        return len(manifest_df)

    def save_last(self, out, model, opt, epoch, history, best, best_epoch, stale):
        atomic_torch_save(out / 'last.pt', dict(epoch=epoch, model_state_dict={k: v.detach().cpu() for (k, v) in model.state_dict().items()}, optimizer_state_dict=opt.state_dict(), rng_state=tr._rng_state(), history=history, best=best, best_epoch=best_epoch, stale=stale))

    def projected(self, task, outer, inner, device, parent_ck):
        out = self.run_dir('projected', task, outer, inner)
        out.mkdir(parents=True, exist_ok=True)
        (fit, val, audit) = self.split(task, outer, inner)
        audit['parent_checkpoint'] = str(parent_ck)
        if (out / 'summary.json').exists() and (out / 'oof/predictions.tsv').exists():
            old = json.loads((out / 'split_audit.json').read_text())
            assert old == audit
            return out / 'best.pt'
        atomic_json(out / 'split_audit.json', audit)
        cfg = self.path(ROOT / 'config/sound_projected_set.yaml')
        (_, runtime, mc, loss) = tr.load_configuration_v2(cfg, arm='current', task=task)
        set_seed(42)
        (model, dep) = load_projected(parent_ck, mc)
        model.to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=runtime.learning_rate, weight_decay=runtime.weight_decay)
        (mat, latent, uce, scale) = self.stores(outer, runtime)
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)

        def ck(epoch, ev):
            return dict(schema_version=PS_SCHEMA, epoch=epoch, base_checkpoint=dep, model_config=model.config.to_dict(), model_state_dict={k: v.detach().cpu() for (k, v) in model.state_dict().items()}, validation=ev, split_audit=audit)
        try:
            baseline = self.evaluate(model, mat, val, 'current', runtime, loss, scale, device)
            best = float(baseline['distribution_mmd2'])
            best_epoch = 0
            stale = 0
            history = []
            start = 1
            if (out / 'last.pt').exists():
                last = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
                model.load_state_dict(last['model_state_dict'])
                opt.load_state_dict(last['optimizer_state_dict'])
                tr._restore_rng_state(last['rng_state'])
                start = last['epoch'] + 1
                history = last['history']
                best = last['best']
                best_epoch = last['best_epoch']
                stale = last['stale']
            else:
                atomic_torch_save(out / 'best.pt', ck(0, baseline))
            for epoch in range(start, runtime.maximum_epochs + 1):
                if stale >= runtime.patience:
                    break
                training = tr.train_epoch_v2(model, opt, mat, fit, arm='current', epoch=epoch, runtime=runtime, loss_config=loss, bandwidth=scale.mmd_bandwidth, response_scale=model.transport.response_scale.detach().cpu().numpy(), clinical_mode='off', device=device)
                ev = self.evaluate(model, mat, val, 'current', runtime, loss, scale, device)
                history.append(dict(epoch=epoch, **{f'train_{k}': float(v) for (k, v) in training.items()}, **{f'validation_{k}': float(v) for (k, v) in ev.items()}))
                score = float(ev['distribution_mmd2'])
                if math.isfinite(score) and score < best - 1e-08:
                    best = score
                    best_epoch = epoch
                    stale = 0
                    atomic_torch_save(out / 'best.pt', ck(epoch, ev))
                else:
                    stale += 1
                atomic_table(out / 'history.tsv', pd.DataFrame(history))
                self.save_last(out, model, opt, epoch, history, best, best_epoch, stale)
                atomic_json(out / 'progress.json', dict(status='training', epoch=epoch, best_epoch=best_epoch, best_mmd2=best, stale_epochs=stale))
            chosen = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(chosen['model_state_dict'])
            n = self.infer(model, mat, val, runtime, 'current', out / 'oof', out / 'best.pt', device, outer, audit)
            summary = dict(status='pass', stage='projected', task=task, outer_fold=outer, inner_validation_fold=inner, seed=42, fit_donors=len(audit['fit_donors']), validation_donors=len(audit['validation_donors']), baseline_validation=baseline, best_epoch=best_epoch, epochs=len(history), best_validation=chosen['validation'], prediction_units=n,  runtime=asdict(runtime), loss=loss.to_dict(), **self.resource_info(device, started))
            atomic_json(out / 'summary.json', summary)
            atomic_json(out / 'progress.json', summary)
            print(json.dumps(summary), flush=True)
            return out / 'best.pt'
        finally:
            latent.close()
            uce.close()

    def full_setup(self, task, projected_ck, outer):
        arch = json.loads(self.path(ROOT / 'config' / ('sound_twin_t2.json' if task == TASKS[1] else 'sound_twin_t3.json')).read_text())
        (_, rt, _, loss0) = tr.load_configuration_v2(ROOT / 'config/sound_projected_set.yaml', arm='current', task=task)
        op = arch['optimization']
        runtime = replace(rt, learning_rate=op['learning_rate'], weight_decay=op['weight_decay'], batch_size=op['batch_size'], maximum_epochs=80, patience=12, history_training_mode='residual_from_current', freeze_pretrained_encoder=True, selection_objective='raw_donor_delta_rmse', selection_direction='lower')
        loss = PatientLossV2Config(**{**loss0.to_dict(), **arch['loss']})
        set_seed(42)
        backbone = _load_projected_set(projected_ck, task, outer)
        model = MetaAdapterDigitalTwin(backbone, torch.empty(0, 32), MetaAdapterConfig(**arch['twin_config']))
        return (runtime, loss, model, backbone, arch)

    def full(self, task, outer, inner, device, projected_ck, extend_to=None):
        out = self.run_dir('full', task, outer, inner)
        out.mkdir(parents=True, exist_ok=True)
        (fit, val, audit) = self.split(task, outer, inner)
        audit['parent_checkpoint'] = str(projected_ck)
        if (out / 'summary.json').exists() and extend_to is None:
            assert json.loads((out / 'split_audit.json').read_text()) == audit
            return
        atomic_json(out / 'split_audit.json', audit)
        extension_archive = None
        if extend_to is not None:
            extension_archive = out / 'selection_before_extension'
            extension_archive.mkdir(exist_ok=True)
            for name in ['summary.json', 'history.tsv', 'progress.json', 'best.pt']:
                if (out / name).exists() and (not (extension_archive / name).exists()):
                    shutil.copy2(out / name, extension_archive / name)
        (runtime, loss, model, backbone, arch) = self.full_setup(task, projected_ck, outer)
        model.to(device)
        opt = torch.optim.AdamW(model.identifier.parameters(), lr=runtime.learning_rate, weight_decay=runtime.weight_decay)
        (mat, latent, uce, scale) = self.stores(outer, runtime, True)
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        ckargs = argparse.Namespace(task=task, fold=outer, inner_validation_fold=inner, variant='rna', clinical_subset='curated', history_policy='formal', projected_set_checkpoint=projected_ck)
        try:
            baseline = self.evaluate(backbone, mat, val, 'current', runtime, replace(loss, history_sensitivity_weight=0.0, history_current_anchor_weight=0.0), scale, device)
            best = math.inf
            best_epoch = 0
            stale = 0
            history = []
            start = 1
            if (out / 'last.pt').exists():
                last = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
                model.load_state_dict(last['model_state_dict'])
                opt.load_state_dict(last['optimizer_state_dict'])
                tr._restore_rng_state(last['rng_state'])
                start = last['epoch'] + 1
                history = last['history']
                best = last['best']
                best_epoch = last['best_epoch']
                stale = last['stale']
            for epoch in range(start, (extend_to or runtime.maximum_epochs) + 1):
                if extend_to is None and stale >= runtime.patience:
                    break
                training = tr.train_epoch_v2(model, opt, mat, fit, arm='full', epoch=epoch, runtime=runtime, loss_config=loss, bandwidth=scale.mmd_bandwidth, response_scale=backbone.transport.response_scale.detach().cpu().numpy(), clinical_mode='off', device=device)
                ev = self.evaluate(model, mat, val, 'full', runtime, loss, scale, device)
                history.append(dict(epoch=epoch, **{f'train_{k}': float(v) for (k, v) in training.items()}, **{f'validation_{k}': float(v) for (k, v) in ev.items()}))
                ck = twin_checkpoint(model=model, epoch=epoch, validation=ev, args=ckargs)
                ck['split_audit'] = audit
                atomic_torch_save(out / 'epochs' / f'epoch{epoch:03d}.pt', ck)
                score = float(ev['raw_donor_delta_rmse'])
                if math.isfinite(score) and score < best - 1e-08:
                    best = score
                    best_epoch = epoch
                    stale = 0
                    atomic_torch_save(out / 'best.pt', ck)
                else:
                    stale += 1
                atomic_table(out / 'history.tsv', pd.DataFrame(history))
                self.save_last(out, model, opt, epoch, history, best, best_epoch, stale)
                atomic_json(out / 'progress.json', dict(status='training', epoch=epoch, best_epoch=best_epoch, best_rmse=best, stale_epochs=stale))
            chosen = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
            summary = dict(status='pass', stage='full', task=task, outer_fold=outer, inner_validation_fold=inner, seed=42, fit_donors=len(audit['fit_donors']), validation_donors=len(audit['validation_donors']), baseline_validation=baseline, best_epoch=best_epoch, epochs=len(history), best_validation=chosen['validation'],  runtime=asdict(runtime), loss=loss.to_dict(), twin_config=model.twin_config.to_dict(), variant='rna', implementation=chosen['implementation'], outer_test_target_opened=False, **self.resource_info(device, started))
            atomic_json(out / 'summary.json', summary)
            atomic_json(out / 'progress.json', summary)
            print(json.dumps(summary), flush=True)
            if extension_archive is not None:
                atomic_json(out / 'extension_summary.json', summary)
                shutil.copy2(out / 'history.tsv', out / 'extension_history.tsv')
                for name in ['summary.json', 'history.tsv', 'progress.json', 'best.pt']:
                    if (extension_archive / name).exists():
                        shutil.copy2(extension_archive / name, out / name)
        finally:
            latent.close()
            uce.close()

    def finalize_full(self, task, outer, inner, device):
        out = self.run_dir('full', task, outer, inner)
        selection = self.output / 'epoch_selection' / f'outer{outer}_{task}.json'
        if (out / 'oof/predictions.tsv').exists():
            return
        chosen_epoch = int(json.loads(selection.read_text())['selected_epoch'])
        ck = out / 'epochs' / f'epoch{chosen_epoch:03d}.pt'
        projected_ck = self.run_dir('projected', task, outer, inner) / 'best.pt'
        if not ck.exists():
            self.full(task, outer, inner, device, projected_ck, extend_to=chosen_epoch)
        (runtime, loss, model, backbone, arch) = self.full_setup(task, projected_ck, outer)
        model.identifier.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['identifier_state_dict'])
        model.to(device)
        (fit, val, audit) = self.split(task, outer, inner)
        audit['epoch_selection'] = str(selection)
        (mat, latent, uce, scale) = self.stores(outer, runtime, True)
        try:
            n = self.infer(model, mat, val, runtime, 'full', out / 'oof', ck, device, outer, audit)
            atomic_json(out / 'oof/completion.json', dict(status='pass', task=task, outer=outer, inner=inner, selected_epoch=chosen_epoch, prediction_units=n))
        finally:
            latent.close()
            uce.close()

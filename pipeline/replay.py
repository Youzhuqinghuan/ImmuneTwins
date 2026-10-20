from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import yaml
from paths import expand_paths

ROOT = Path(__file__).resolve().parents[1]

WORKSPACE = Path(os.environ.get('IMMUNETWINS_WORKSPACE_ROOT', ROOT.parent))

ARCHIVE = Path(os.environ.get('IMMUNETWINS_FROZEN_RESULTS', WORKSPACE/'results/ImmuneTwins'))

CIMA = ARCHIVE/'age/simple_v1/attention/full_cima/checkpoint.pt'

TASKS = ('t1_y1_d0_to_y1_d7', 't2_d7_to_d90', 't3_y2_d0_to_y2_d90')

def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def sound_checkpoint(task, fold, *, twin=False, config=None):
    from model.encoder import VaccineImmuneTwinConfig
    from model.projected_set import MomentFiberCurrentModel
    from model.digital_twin import MetaAdapterConfig, MetaAdapterDigitalTwin
    from data.reference import load_config, neural_run
    config = config or load_config()
    parent = neural_run(config, task, fold, current=True)
    saved = load(parent/'checkpoint.pt')
    model = MomentFiberCurrentModel(VaccineImmuneTwinConfig(**saved['model_config']))
    model.load_state_dict(saved['model_state_dict'], strict=True)
    if not twin or task == TASKS[0]:
        return model.eval(), parent
    folder = neural_run(config, task, fold)
    saved = load(folder/'checkpoint.pt')
    model = MetaAdapterDigitalTwin(model, torch.empty(0, 32), MetaAdapterConfig(**saved['twin_config']))
    model.identifier.load_state_dict(saved['identifier_state_dict'], strict=True)
    return model.eval(), folder

def comparison(name, predicted, reference, *, ids, output, checkpoint):
    predicted, reference = np.asarray(predicted), np.asarray(reference)
    if predicted.shape != reference.shape or not np.isfinite(predicted).all():
        raise ValueError(f'{name}: shape/finite-value mismatch')
    difference = np.abs(predicted.astype(np.float64)-reference.astype(np.float64))
    record = {'name':name, 'n_patients_or_units':len(ids), 'shape':list(predicted.shape),
              'array_equal':bool(np.array_equal(predicted,reference)),
              'max_abs_error':float(difference.max(initial=0)),
              'mean_abs_error':float(difference.mean()), 'checkpoint':str(checkpoint)}
    np.savez_compressed(output/(name+'.npz'), ids=np.asarray(ids,dtype=str), predicted=predicted)
    print(json.dumps(record), flush=True)
    return record

def replay_age(folds, device, output):
    from data.age import CimaUceStore
    from pipeline.age.checkpoint import load_checkpoint
    from pipeline.age.train import TrainingConfig, _loader, _predict_cima
    store=CimaUceStore(); config=TrainingConfig(); rows=[]
    try:
        for fold in folds:
            folder=ARCHIVE/f'age/simple_v1/attention/fold_{fold}'
            old=pd.read_csv(folder/'held_out_predictions.csv',float_precision='round_trip').sort_values('donor_id')
            model,scaler,_=load_checkpoint(folder/'checkpoint.pt',device)
            loader=_loader(store,old.donor_id.astype(str).tolist(),batch_size=config.inference_batch_size,
                           shuffle=False,seed=42,num_workers=0)
            with torch.inference_mode():new=_predict_cima(model,scaler,loader,device,config.amp)
            new=new.set_index('donor_id').loc[old.donor_id]
            rows.append(comparison(f'age_cima_fold{fold}',new.predicted_age,old.predicted_age,
                        ids=old.donor_id,output=output,checkpoint=folder/'checkpoint.pt'))
    finally:store.close()
    return rows

def replay_sound(folds, device, output, tasks=TASKS, reference_config=None):
    """Replay neural controls and the current model including the personal-reference layer."""
    from types import SimpleNamespace
    from data.reference import (DEFAULT_CONFIG, load_config, inner_inputs,
                                resolve_path)
    from data.sound import (UnifiedUCEStore, FoldLatentStore, UnitMaterializer, load_fold_scale,
                            build_task_units, split_outer_units, iter_donor_batches, stable_seed)
    from pipeline.sound.train import _stack_inputs_v2, _forward_v2
    from pipeline.sound.reference import ReferencePredictor
    cfg = load_config(reference_config or DEFAULT_CONFIG)
    inputs = inner_inputs(cfg)
    uce = UnifiedUCEStore(resolve_path(inputs["visit_manifest"], cfg))
    rows = []
    try:
        for fold in folds:
            scale = load_fold_scale(resolve_path(cfg["inputs"]["scales"].format(fold=fold), cfg), outer_fold=fold)
            latent = FoldLatentStore(resolve_path(cfg["inputs"]["latents"].format(fold=fold, role="test"), cfg), scale)
            try:
                mat = UnitMaterializer(uce=uce, latent=latent, query_uce_cells=512, support_uce_cells=512, seed=42)
                for task in tasks:
                    units = build_task_units(resolve_path(cfg["inputs"]["task_units"], cfg), task)
                    _, test = split_outer_units(units, fold)
                    reference = ReferencePredictor(cfg, task, fold)
                    for twin in ([False] if task == TASKS[0] else [False, True]):
                        model, folder = sound_checkpoint(task, fold, twin=twin, config=cfg)
                        model.to(device)
                        got, expected, final, expected_final, ids = [], [], [], [], []
                        for group in iter_donor_batches(test, batch_size_donors=8, seed=42):
                            values = [mat.inputs(u, cells=2048, draw_key="paper_prediction_v2",
                                source_seed=stable_seed("bridge_test_cells", u.task, u.unit_id, "source", base_seed=42),
                                include_history=twin) for u in group.units]
                            batch = _stack_inputs_v2(values, device, clinical_mode="off", donor_index=group.donor_index)
                            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                                predicted = _forward_v2(model, "full" if twin else "current", batch)["predicted_cells"].float().cpu().numpy()
                            predicted = scale.inverse(predicted)
                            for i, u in enumerate(group.units):
                                with np.load(folder / "predicted" / f"{u.episode_id}.npz", allow_pickle=False) as saved:
                                    expected.append(saved["embedding"])
                                    labels = saved["cell_state"].copy()
                                    source_rows = saved["source_row_index"].copy()
                                got.append(predicted[i]); ids.append(u.unit_id)
                                if twin or task == TASKS[0]:
                                    with np.load(folder / f"source_day{u.source_day}" / f"{u.episode_id}.npz", allow_pickle=False) as source:
                                        if not np.array_equal(source["source_row_index"], source_rows):
                                            raise ValueError("Neural replay changed the source-row contract")
                                        query = SimpleNamespace(source=u.source_sample, donor=u.donor_id, unit=u.episode_id)
                                        cells, indices, _ = reference.apply(query, source["embedding"], predicted[i], labels)
                                    saved_path = Path(cfg["reference_predictions"]) / task / f"outer{fold}" / f"{u.episode_id}.npz"
                                    with np.load(saved_path, allow_pickle=False) as saved:
                                        expected_final.append(saved["embedding"])
                                        if not np.array_equal(labels[indices], saved["cell_state"]):
                                            raise ValueError("Reference replay changed cell-type resampling")
                                        if not np.array_equal(source_rows[indices], saved["source_row_index"]):
                                            raise ValueError("Reference replay changed source-cell resampling")
                                    final.append(cells)
                        rows.append(comparison(f"sound_{task}_" + ("history" if twin else "current") + f"_fold{fold}",
                            got, expected, ids=ids, output=output, checkpoint=folder / "checkpoint.pt"))
                        if final:
                            rows.append(comparison(f"sound_{task}_full_fold{fold}", final, expected_final,
                                ids=ids, output=output, checkpoint=reference.parameters["reference_file"]))
                        del model
                        torch.cuda.empty_cache()
            finally:
                latent.close()
    finally:
        uce.close()
    return rows


def replay_age_external(folds,device,output):
    from data.age import ImmunobiologyUceShardStore, DEFAULT_EXTERNAL_UCE_SHARDS
    from pipeline.age.checkpoint import load_checkpoint
    from pipeline.age.inference import predict_external_shards
    model,scaler,_=load_checkpoint(CIMA,device);rows=[]
    one=ARCHIVE/'age/simple_v1/external_onek1k'
    manifest=json.loads((one/'immunetwin_predictions.seal.json').read_text())
    cohorts=[('immunobiology',DEFAULT_EXTERNAL_UCE_SHARDS,234,1500,
              ARCHIVE/'age/simple_v1/attention/full_cima/immunobiology_raw_predictions.csv'),
             ('onek1k',Path(manifest['external_uce']['root']),981,None,one/'immunetwin_predictions.sealed.tsv')]
    for name,shards,count,cells,reference in cohorts:
        store=ImmunobiologyUceShardStore(shards,expected_groups=None)
        got=predict_external_shards(model,scaler,store,device,expected_donors=count,expected_cells_per_donor=cells)
        old=pd.read_csv(reference,sep='\t' if reference.suffix=='.tsv' else ',',float_precision='round_trip')
        if 'method' in old:old=old[old.method.eq('ImmuneTwin')]
        old=old.sort_values('donor_id')
        got=got.set_index('donor_id').loc[old.donor_id]
        rows.append(comparison(f'age_{name}',got.predicted_age,old.predicted_age,ids=old.donor_id,
                               output=output,checkpoint=CIMA))
    return rows

def replay_state(folds,device,output):
    from data.treatment import UceVisitStore,validate_transitions
    from model.treatment import SharedVaccineLogicConfig,build_shared_projected_set,build_shared_meta_adapter
    from pipeline.treatment.state import DrugFoldData,LatentVisitStore,_predict
    # Read the saved latent coordinates directly.
    config=expand_paths(yaml.safe_load((ROOT/'config/treatment.yaml').read_text()))
    rows=[]
    for fold in folds:
        folder=ARCHIVE/f'drug_discovery_mm/task_adaptation_20260908/state/fold_{fold}'
        saved=load(folder/'shared_model_checkpoint.pt')
        data=DrugFoldData.__new__(DrugFoldData)
        data.config=config;data.outer_fold=fold
        data.transitions=validate_transitions(pd.read_csv(config['data']['transitions'],sep='\t'))
        data.transition_index=data.transitions.set_index('transition_id',drop=False)
        data.episodes=pd.read_csv(config['data']['episodes'],sep='\t')
        data.uce=UceVisitStore(config['data']['uce_store'])
        data.latent=LatentVisitStore(Path(config['data']['latent_reference_root'])/f'fold_{fold}/latent_visits.h5')
        data.cells=512;data.development='GSE309595';data.external='NONE';data._uce_cache={};data._latent_cache={}
        try:items=data.arrays(data.rows('test'),include_target=False)
        finally:data.close()
        shared=SharedVaccineLogicConfig(**saved['model_config'])
        current=build_shared_projected_set(shared,saved['set_transport_state_dict'])
        current.load_state_dict(saved['projected_set_state_dict'],strict=True)
        model=build_shared_meta_adapter(shared,current)
        model.load_state_dict(saved['meta_adapter_state_dict'],strict=True)
        model.to(device).eval()
        for name,candidate,mode in [('full',model,'correct'),('no_digital_twin',current,'correct'),('shuffled_history',model,'shuffled')]:
            got=_predict(candidate,items,device,history_mode=mode,seed=saved['seed']+1)
            with np.load(folder/f'predictions/ImmuneTwins_({name})/test.npz',allow_pickle=False) as reference:
                ids=[x.patient_id for x in items]
                if ids!=reference['patient_id'].tolist():raise ValueError('Patient ordering changed')
                rows.append(comparison(f'treatment_state_{name}_fold{fold}',got,reference['predicted'],ids=ids,
                                       output=output,checkpoint=folder/'shared_model_checkpoint.pt'))
        del model,current;torch.cuda.empty_cache()
    return rows

def replay_response(folds,device,output):
    from model.treatment import (SharedVaccineLogicConfig,build_shared_set_transport,
                                 build_shared_projected_set,build_shared_meta_adapter)
    from pipeline.treatment.response import ResponseCohort,ClinicalReadout,extract
    rows=[]
    cohort=ResponseCohort(WORKSPACE/'data/ImmuneTwins/drug_discovery_mm/features')
    for fold in folds:
        folder=ARCHIVE/f'drug_discovery_mm/response_distribution_20260907/fold_{fold}'
        saved=load(folder/'checkpoint.pt');shared=SharedVaccineLogicConfig(**saved['shared_config'])
        parent,_=build_shared_set_transport(shared,CIMA)
        current=build_shared_projected_set(shared,parent.state_dict())
        model=build_shared_meta_adapter(shared,current)
        model.load_state_dict(saved['model_state_dict'],strict=True)
        model._clinical_pooling=saved.get('clinical_pooling');model.to(device).eval()
        reference=pd.read_csv(folder/'predictions.tsv',sep='\t',float_precision='round_trip')
        ids=reference.patient_id.tolist()
        features,future,attention=extract(model,cohort.episodes(ids,saved['transform'],supervised=False),device)
        state=saved['head_state_dict']
        head=ClinicalReadout(state['mean'],state['scale'],saved['head_state_weight'])
        head.load_state_dict(state,strict=True);head.to(device).eval()
        with torch.no_grad():prob=head(torch.as_tensor(features,device=device)).sigmoid().cpu().numpy()
        # The original TSV was written directly from a float32 DataFrame.
        # Restore that dtype, rather than comparing its shortest decimal text
        # as a new float64 measurement.
        rows.append(comparison(f'treatment_response_probability_fold{fold}',prob,reference.response_probability.to_numpy(np.float32),
                               ids=ids,output=output,checkpoint=folder/'checkpoint.pt'))
        with np.load(folder/'native_outputs.npz',allow_pickle=False) as old:
            rows.append(comparison(f'treatment_response_state_fold{fold}',future,old['predicted_latent_mean'],
                                   ids=ids,output=output,checkpoint=folder/'checkpoint.pt'))
        del model,current,parent;torch.cuda.empty_cache()
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','age','age-external','sound','state','response'],default='all')
    parser.add_argument('--folds',type=int,nargs='+',default=list(range(5)))
    parser.add_argument('--output',type=Path,default=ROOT/'results/replay')
    parser.add_argument('--reference-config', type=Path, default=ROOT/'config/sound_reference.json')
    args=parser.parse_args()
    output=args.output.resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Select exactly one available GPU with CUDA_VISIBLE_DEVICES; GPU 0 is reserved')
    device=torch.device('cuda:0');_=torch.ones(4,4,device=device)@torch.ones(4,4,device=device)
    torch.set_num_threads(4)
    started=time.time();records=[]
    functions={'age':replay_age,'sound':replay_sound,'state':replay_state,'response':replay_response,
               'age-external':replay_age_external}
    try:
        for stage,fn in functions.items():
            if args.stage in ('all',stage):
                options = {'reference_config': args.reference_config} if stage == 'sound' else {}
                records.extend(fn(args.folds,device,output,**options))
    finally:
        payload={'stage':args.stage,'folds':args.folds,'records':records,'elapsed_seconds':time.time()-started,
                 'torch':torch.__version__,'device':torch.cuda.get_device_name(),
                 'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20,'training':False,'baseline_execution':False}
        (output/f'{args.stage}_comparison.json').write_text(json.dumps(payload,indent=2))
    if any(not r['array_equal'] for r in records):
        raise RuntimeError('Predictions differ from the frozen reference; inspect the comparison JSON.')

if __name__ == "__main__":
    main()

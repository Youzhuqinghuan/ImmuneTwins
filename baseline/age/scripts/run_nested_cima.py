from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from baseline.age.workspace import WORKSPACE_ROOT as W
from baseline.age.adapters.common_splits import common_fold_order

SPLIT = W / 'data/cima_random_donor.tsv'


def split_table():
    table = pd.read_csv(SPLIT, sep='\t', dtype={'donor_id': str})
    table = table.loc[table.repeat.eq(0), ['donor_id', 'age', 'fold']]
    if len(table) != 417 or not table.donor_id.is_unique or set(table.fold) != set(range(5)):
        raise ValueError('The frozen 417-donor common five folds have changed')
    return table.set_index('donor_id')

def ideal_scopes():
    return [(i,) for i in range(5)] + list(itertools.combinations(range(5), 2))

def scope_name(excluded):
    return 'exclude_' + '_'.join(map(str, sorted(excluded)))

def assert_isolated(fit, heldout, *, outer_test=()):
    fit, heldout, outer_test = set(fit), set(heldout), set(outer_test)
    if not fit or not heldout or fit & heldout or fit & outer_test:
        raise ValueError('Patient overlap or empty fit/held-out scope')

def scope_spec(method, excluded):
    table = split_table()
    donors = common_fold_order(table.index.tolist(), SPLIT, 0, 3407)
    folds = table.loc[donors, 'fold'].to_numpy()
    train = np.flatnonzero(~np.isin(folds, excluded))
    held = np.flatnonzero(np.isin(folds, excluded))
    if method == 'scphase':
        if len(excluded) != 1:
            raise ValueError('scPhase requires one outer fold')
        fit_local, validation_local = train_test_split(
            np.arange(len(train)), test_size=.15, random_state=3407)
        preprocessing_fit = train[fit_local]
        validation = train[validation_local]
    else:
        preprocessing_fit, validation = train, np.array([], dtype=int)
    fit_ids = [donors[i] for i in preprocessing_fit]
    held_ids = [donors[i] for i in held]
    assert_isolated(fit_ids, held_ids)
    return dict(method=method, excluded_folds=list(excluded), donor_order=donors,
                training_donors=[donors[i] for i in train],
                preprocessing_fit_donors=fit_ids,
                validation_donors=[donors[i] for i in validation], heldout_donors=held_ids)

def validate_component(folder, expected):
    record = json.loads((folder / 'complete.json').read_text())
    prepared = json.loads((folder / 'input/complete.json').read_text())
    if (
        record['status'] != 'PASS'
        or record['scope'] != expected
        or prepared['scope'] != expected
    ):
        raise ValueError(f'Changed or incorrectly scoped component: {folder}')
    return record


import argparse
import fcntl
import os
import sys
from baseline.age.workspace import REPOSITORY_ROOT as ROOT

DEFAULT_OUTPUT = ROOT / 'results/baseline_correction/cima_nested'

IDEAL_PYTHON = W / '.conda/ideal-age/bin/python'

SCPHASE_PYTHON = W / 'official_age_baselines/envs/scphase/bin/python'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'ideal', 'scphase', 'assemble', 'replay'], required=True)
    parser.add_argument('--method', choices=['ideal', 'scphase'], default='ideal')
    parser.add_argument('--exclude', nargs='+', type=int, choices=range(5), default=[0])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.exclude = sorted(set(args.exclude))
    if len(args.exclude) not in (1, 2):
        parser.error('Exclude one outer fold, or one outer plus one inner fold')
    args.output = args.output.resolve()
    if ROOT not in args.output.parents:
        parser.error('Choose an output directory inside this repository')
    if args.stage in {'ideal', 'scphase', 'replay'}:
        method = args.method if args.stage == 'replay' else args.stage
        interpreter = IDEAL_PYTHON if method == 'ideal' else SCPHASE_PYTHON
        if Path(sys.executable) != interpreter:
            environment = os.environ.copy()
            library = str(W / '.conda/immunetwins/lib')
            environment['LD_LIBRARY_PATH'] = library + (':' + environment['LD_LIBRARY_PATH']
                if environment.get('LD_LIBRARY_PATH') else '')
            os.execve(str(interpreter), [str(interpreter), '-u', '-m',
                      'baseline.age.scripts.run_nested_cima', *sys.argv[1:]], environment)
    if args.stage in {'prepare', 'ideal', 'scphase'}:
        method = args.method if args.stage == 'prepare' else args.stage
        folder = args.output / method / scope_name(args.exclude)
        folder.mkdir(parents=True, exist_ok=True)
        # A scope is fitted at most once in either queue mode.
        with (folder / '.stage.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            globals()[args.stage](args)
    else:
        globals()[args.stage](args)

import gc
import subprocess
import time
from types import SimpleNamespace
from sklearn.model_selection import KFold

RAW = W / 'data/0714/pascient/cima_official_1500_seed20260711_raw_counts.h5'

def official_ideal_source():
    from baseline.age.adapters.run_ideal_age_full_ensemble import OFFICIAL_REPO
    commit = subprocess.check_output(['git', '-C', str(OFFICIAL_REPO), 'rev-parse', 'HEAD'], text=True).strip()
    return dict(commit=commit)

def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)

def check_resources(gpu=False):
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(values['MemAvailable'].split()[0]) * 1024
    if available < 70 * 1024**3:
        raise RuntimeError('Less than 70 GiB RAM available before a nested-CIMA stage')
    if gpu:
        import torch
        selected = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not selected.startswith('GPU-') or ',' in selected:
            raise ValueError('Use one nonzero physical GPU UUID')
        index = subprocess.check_output(['nvidia-smi', '-i', selected,
            '--query-gpu=index', '--format=csv,noheader,nounits'], text=True).strip()
        if index == '0' or not torch.cuda.is_available():
            raise RuntimeError('A real nonreserved CUDA GPU is required')
        if (torch.ones(1, device='cuda') + 1).item() != 2:
            raise RuntimeError('CUDA operation failed')

def prepare(args):
    """Fit on the given cells, then apply the fitted moments donor by donor."""
    from baseline.age.adapters.prepare_scphase_practical_inputs import (
        CONTRACT, STORAGE_CONTRACT, load_store, fit_age_preprocessing,
        apply_age_preprocessing, write_shard, write_master)
    spec = scope_spec(args.method, args.exclude)
    output = args.output / args.method / scope_name(args.exclude) / 'input'
    marker = output / 'complete.json'
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior['scope'] != spec:
            raise ValueError('Prepared scope differs from this request')
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Incomplete input preparation: {output}')
    check_resources()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    matrix, obs, genes = load_store(RAW)
    donor_values = np.asarray(obs['donor_id'], dtype=str)
    fit_mask = np.isin(donor_values, spec['preprocessing_fit_donors'])
    if int(fit_mask.sum()) != 1500 * len(spec['preprocessing_fit_donors']):
        raise ValueError('Unexpected fixed-cell support in fit scope')
    fit_obs = {k: v[fit_mask] for k, v in obs.items()}
    indices, selected_genes, means, stds, fitted, api = fit_age_preprocessing(
        matrix[fit_mask], fit_obs, genes, exact_hvg=True)
    del fit_obs
    # Direct equivalence check against the actual public Scanpy fit output.
    probe_rows = np.flatnonzero(fit_mask)[:1500]
    applied = apply_age_preprocessing(matrix[probe_rows][:, indices], means, stds)
    np.testing.assert_array_equal(applied, fitted[:len(probe_rows)])
    del fitted, applied
    selected = matrix[:, indices].tocsr()
    del matrix
    gc.collect()
    genes_path = output / 'genes.txt'
    genes_path.write_text(''.join(f'{g}\n' for g in selected_genes))
    np.savez_compressed(output / 'scaler.npz', genes=np.asarray(selected_genes), mean=means, std=stds)
    shard_dir = output / 'shards'
    shard_dir.mkdir()
    refs, pseudobulk = [], []
    for i, donor in enumerate(spec['donor_order']):
        rows = np.flatnonzero(donor_values == donor)
        if len(rows) != 1500:
            raise ValueError(f'Expected 1500 cells for {donor}')
        values = apply_age_preprocessing(selected[rows], means, stds)
        refs.append(write_shard(shard_dir / f'donor_{i:04d}.npy', values))
        pseudobulk.append(values.mean(axis=0))
        if (i + 1) % 50 == 0:
            print(f'{args.method}/{scope_name(args.exclude)} prepared {i+1}/417 donors', flush=True)
    table = split_table()
    metadata = dict(contract=CONTRACT, storage_contract=STORAGE_CONTRACT,
        preprocessing_role='cima_fit', cv_strategy='KFold5_single_group_order_adapter',
        donor_order='common-fold',
         cache_policy='none',
        upstream_input_state='fixed_1500_raw_count_input', hvg_api=api,
        scale_api='scanpy.pp.scale(max_value=10)', preprocessing_fit_donors=spec['preprocessing_fit_donors'],
        target_statistics_fitted=False, target_labels_accessed=None,
        heldout_labels_used_for_fitting=False)
    master = output / 'cima_master.pkl'
    write_master(path=master, refs=refs, donor_ids=spec['donor_order'],
        labels=table.loc[spec['donor_order'], 'age'].to_numpy(), metadata=metadata,
        input_store=RAW, labels_present=True)
    np.savez_compressed(output / 'pseudobulk.npz', donor_id=np.asarray(spec['donor_order']),
        features=np.stack(pseudobulk), genes=np.asarray(selected_genes))
    import resource, scanpy
    write_json(marker, dict(status='PASS', scope=spec,
         source=str(RAW), source_size=RAW.stat().st_size,
        scanpy_version=scanpy.__version__, fit_apply_exact=True,
        seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))

def ideal(args):
    from baseline.age.adapters.run_ideal_age_full_ensemble import (
        assert_cuda, fit_autogluon_bagged, frame)
    from baseline.age.adapters.run_ideal_age_official_cima import (
        materialize_master, import_official_model, train_track, predict)
    import torch
    check_resources(gpu=True)
    proof = assert_cuda()
    source = official_ideal_source()
    # Fail before fitting if the existing environment cannot load either branch.
    spec = scope_spec('ideal', args.exclude)
    folder = args.output / 'ideal' / scope_name(args.exclude)
    prepared = json.loads((folder / 'input/complete.json').read_text())
    if prepared['scope'] != spec:
        raise ValueError('Mismatched prepared patient scope')
    marker = folder / 'complete.json'
    if marker.exists():
        validate_component(folder, spec)
        return
    data = np.load(folder / 'input/pseudobulk.npz', allow_pickle=False)
    donors, features, genes = data['donor_id'], data['features'], data['genes']
    table = split_table()
    ages = table.loc[donors, 'age'].to_numpy(dtype=np.float32)
    train = np.flatnonzero(np.isin(donors, spec['training_donors']))
    held = np.flatnonzero(np.isin(donors, spec['heldout_donors']))
    assert_isolated(donors[train], donors[held])
    bags = np.empty(len(train), dtype=int)
    for f, (_, idx) in enumerate(KFold(5, shuffle=True, random_state=42).split(train)):
        bags[idx] = f
    ag_output = folder / 'autogluon_predictions.npz'
    if ag_output.exists():
        pseudo = np.load(ag_output, allow_pickle=False)['prediction']
        details = json.loads((folder / 'autogluon.json').read_text())
    else:
        _, pseudo, details = fit_autogluon_bagged(
            frame(features[train], genes, donors[train]), ages[train], bags,
            frame(features[held], genes, donors[held]), folder / 'autogluon_models', 600, 1800, 42)
        np.savez_compressed(ag_output, donor_id=donors[held], prediction=pseudo)
        details['bagging_fold_source'] = 'KFold5 seed42 within this component training scope only'
        write_json(folder / 'autogluon.json', details)
    gc.collect(); torch.cuda.empty_cache()
    values, packed_donors, packed_age = materialize_master(
        folder / 'input/cima_master.pkl', folder / 'deepsets_input.npy', 1000, 42)
    np.testing.assert_array_equal(packed_donors, donors)
    np.testing.assert_array_equal(packed_age, ages)
    rng = np.random.default_rng(42)
    choices = np.stack([rng.choice(1000, 1000, replace=False) for _ in donors])
    cfg = SimpleNamespace(epochs=105, max_cells=1000, batch_size=8, seed=42, exact_resume=True)
    network = train_track(import_official_model(), values, ages, train, choices,
                          cfg, folder, 'deepsets', torch.device('cuda:0'))
    deep = predict(network, values, held, choices, 8, torch.device('cuda:0'))
    predictions = folder / 'predictions.tsv'
    pd.DataFrame(dict(donor_id=donors[held], autogluon=pseudo, deepsets=deep)).to_csv(
        predictions, sep='\t', index=False)
    write_json(marker, dict(status='PASS', scope=spec, cuda=proof,
        official_source=source,

        autogluon=details, seed=42, epochs=105))

def scphase(args):
    check_resources(gpu=True)
    folder = args.output / 'scphase' / scope_name(args.exclude)
    spec = scope_spec('scphase', args.exclude)
    if json.loads((folder / 'input/complete.json').read_text())['scope'] != spec:
        raise ValueError('scPhase input scope differs from its official validation split')
    if (folder / 'complete.json').exists():
        record = json.loads((folder / 'complete.json').read_text())
        if (record.get('status') == 'PASS' and (folder / 'predictions.tsv').is_file()
                and Path(record['checkpoint']).is_file()):
            return
    cfg_path = folder / 'config.json'
    subprocess.run([sys.executable, '-m', 'baseline.age.adapters.make_scphase_config',
        '--master', str(folder / 'input/cima_master.pkl'), '--results-dir', str(folder / 'training'),
        '--output', str(cfg_path), '--model-name', 'CIMA_Age_scPhase'], check=True)
    config = json.loads(cfg_path.read_text())
    config['run_params']['skip_groups'] = [f+1 for f in range(5) if f not in args.exclude]
    write_json(cfg_path, config)
    subprocess.run([sys.executable, '-m', 'baseline.age.scripts.run_scphase_official',
                    '--config', str(cfg_path)], check=True)
    raw = pd.read_csv(folder / 'training/predictions/CIMA_Age_scPhase_predictions.csv')
    donors = np.asarray(spec['donor_order'])
    splits = list(KFold(5, shuffle=True, random_state=3407).split(donors))
    held = donors[splits[args.exclude[0]][1]]
    if set(held) != set(spec['heldout_donors']):
        raise ValueError('Official KFold test patients differ from the declared preprocessing exclusion')
    if len(raw) != len(held) or set(raw.fold) != {args.exclude[0]+1}:
        raise ValueError('Official scPhase prediction order/coverage changed')
    np.testing.assert_allclose(raw.y_true, split_table().loc[held, 'age'], atol=0, rtol=0)
    pd.DataFrame(dict(donor_id=held, predicted_age=raw.y_pred)).to_csv(
        folder / 'predictions.tsv', sep='\t', index=False)
    checkpoint = folder / f'training/BestModel_CIMA_Age_scPhase_Fold{args.exclude[0]+1}.pt'
    write_json(folder / 'complete.json', dict(status='PASS', scope=spec,
        checkpoint=str(checkpoint),  seed=3407))

def assemble(args):
    from baseline.age.adapters.run_ideal_age_full_ensemble import greedy_ensemble_weights
    table = split_table()
    records, weights = [], []
    for outer in range(5):
        train_ids = table.index[table.fold.ne(outer)].tolist()
        held_ids = table.index[table.fold.eq(outer)].tolist()
        inner_rows = []
        for other in range(5):
            if other == outer:
                continue
            excluded = tuple(sorted((outer, other)))
            folder = args.output / 'ideal' / scope_name(excluded)
            spec = scope_spec('ideal', excluded)
            validate_component(folder, spec)
            ids = table.index[table.fold.eq(other)].tolist()
            assert_isolated(spec['training_donors'], ids, outer_test=held_ids)
            values = pd.read_csv(folder / 'predictions.tsv', sep='\t').set_index('donor_id').loc[ids]
            inner_rows.append(values)
        inner = pd.concat(inner_rows).loc[train_ids]
        if not inner.index.is_unique or set(inner.index) != set(train_ids):
            raise ValueError('Nested component OOF coverage is not exact')
        weight = greedy_ensemble_weights(inner[['autogluon', 'deepsets']].to_numpy(),
                                        table.loc[train_ids, 'age'].to_numpy(), 100)
        folder = args.output / 'ideal' / scope_name((outer,))
        validate_component(folder, scope_spec('ideal', (outer,)))
        held = pd.read_csv(folder / 'predictions.tsv', sep='\t').set_index('donor_id').loc[held_ids]
        predicted = held[['autogluon', 'deepsets']].to_numpy() @ weight
        records.extend(dict(method='IDEAL-Age', cohort='CIMA', fold=outer,
                            donor_id=d, predicted_age=p) for d, p in zip(held_ids, predicted))
        weights.append(dict(outer_fold=outer, weights=weight.tolist(),
                            fit_donors=train_ids, heldout_donors=held_ids))
        folder = args.output / 'scphase' / scope_name((outer,))
        record = json.loads((folder / 'complete.json').read_text())
        if (record['scope'] != scope_spec('scphase', (outer,))):
            raise ValueError('Changed scPhase completion chain')
        values = pd.read_csv(folder / 'predictions.tsv', sep='\t').set_index('donor_id').loc[held_ids]
        records.extend(dict(method='scPhase', cohort='CIMA', fold=outer,
                            donor_id=d, predicted_age=p) for d, p in values.predicted_age.items())
    output = pd.DataFrame(records)
    if output.duplicated(['method', 'donor_id']).any() or len(output) != 834:
        raise ValueError('Nested OOF coverage differs from two methods x 417 patients')
    output.to_csv(args.output / 'cima_predictions.tsv', sep='\t', index=False)
    scored = output.assign(age=output.donor_id.map(table.age))
    scored.to_csv(args.output / 'cima_scored_predictions.tsv', sep='\t', index=False)
    write_json(args.output / 'ensemble_weights.json', weights)
    write_json(args.output / 'complete.json', dict(status='PASS',
        external_predictions_changed=False, outer_folds=5, inner_folds=4,
        component_fit_scopes=len(ideal_scopes()), patient_isolation='all learned ancestors exclude outer test'))

def replay(args):
    """Reload saved weights and reproduce their held-out predictions."""
    import pickle
    import torch
    check_resources(gpu=True)
    folder = args.output / args.method / scope_name(args.exclude)
    record = json.loads((folder / 'complete.json').read_text())
    expected = pd.read_csv(folder / 'predictions.tsv', sep='\t', dtype={'donor_id': str})
    if record['scope'] != scope_spec(args.method, args.exclude):
        raise ValueError('Replay patient scope changed')
    if args.method == 'ideal':
        from autogluon.tabular import TabularPredictor
        from baseline.age.adapters.run_ideal_age_full_ensemble import frame
        from baseline.age.adapters.run_ideal_age_official_cima import import_official_model, predict
        validate_component(folder, record['scope'])
        official_ideal_source()
        bulk = np.load(folder / 'input/pseudobulk.npz', allow_pickle=False)
        donor = bulk['donor_id']
        held = np.flatnonzero(np.isin(donor, expected.donor_id))
        predictor = TabularPredictor.load(str(folder / 'autogluon_models'))
        pseudo = predictor.predict(frame(bulk['features'][held], bulk['genes'], donor[held])).to_numpy()
        checkpoint = torch.load(folder / 'deepsets.pt', map_location='cpu', weights_only=False)
        network = import_official_model().DeepSetsAgePredictorAttn(
            input_dim=5000, hidden_dim=2**11, dropout=.2, num_classes=10).to('cuda')
        network.load_state_dict(checkpoint['model_state_dict'], strict=True)
        values = np.load(folder / 'deepsets_input.npy', mmap_mode='r', allow_pickle=False)
        rng = np.random.default_rng(42)
        choices = np.stack([rng.choice(1000, 1000, replace=False) for _ in donor])
        predicted = predict(network, values, held, choices, 8, torch.device('cuda'))
        observed = expected.set_index('donor_id').loc[donor[held]]
        # CSV carries float32 shortest-decimal output; compare in that original dtype.
        errors = dict(autogluon=float(np.max(np.abs(pseudo-observed.autogluon.to_numpy()))),
                      deepsets=float(np.max(np.abs(predicted-observed.deepsets.to_numpy(dtype=np.float32)))))
    else:
        from baseline.age.scripts.run_scphase_official import SCPHASE
        sys.path.insert(0, str(SCPHASE / 'scphase'))
        from model import SCMIL_AttnMoE
        from train_utils import set_seed
        set_seed(3407)
        config = json.loads((folder / 'config.json').read_text())
        state = torch.load(record['checkpoint'], map_location='cuda', weights_only=False)
        network = SCMIL_AttnMoE(config['model_params'], config['ablation_params'], 1, 'cuda:0', 'cuda:0')
        network.load_state_dict(state, strict=True)
        network.eval()
        with (folder / 'input/cima_master.pkl').open('rb') as handle:
            master = pickle.load(handle)
        refs = dict(zip(master['sample_ids'], master['data_list_dense']))
        values = []
        with torch.no_grad():
            for donor in expected.donor_id:
                matrix = np.load(refs[donor]['path'], allow_pickle=False)
                tensor = torch.as_tensor(matrix, device='cuda', dtype=torch.float32)
                _, age, _, _ = network(tensor, alpha=1.)
                values.append(float(age.item()))
        errors = dict(scphase=float(np.max(np.abs(np.asarray(values, dtype=np.float32)
                                                 -expected.predicted_age.to_numpy(dtype=np.float32)))))
    if max(errors.values()) > 1e-6:
        raise ValueError(f'Corrected checkpoint replay differs: {errors}')
    result = dict(status='PASS', method=args.method, excluded_folds=args.exclude,
                  patients=len(expected), max_absolute_error=errors,
                   tolerance=1e-6,
                  device=os.environ['CUDA_VISIBLE_DEVICES'])
    write_json(folder / 'replay.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()

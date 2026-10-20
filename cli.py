"""ImmuneTwins data preparation, training, inference and evaluation."""
from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import sys

STAGES = {
    'prepare': {
        'uce': ('data.uce', 'main'),
        'vaccine': ('data.prepare_sound', 'main'),
        'vaccine-reference': ('data.reference', 'main'),
        'vaccine-protocol': ('pipeline.sound.protocol', 'main'),
        'vaccine-plan': ('pipeline.sound.protocol', 'prepare_main'),
    },
    'train': {
        'age': ('pipeline.age.train', 'main'),
        'vaccine-affine': ('pipeline.sound.train', 'main'),
        'vaccine-shape-select': ('pipeline.sound.select_projected_set', 'main'),
        'vaccine-shape': ('pipeline.sound.projected_set', 'main'),
        'vaccine-history-select': ('pipeline.sound.select_digital_twin', 'main'),
        'vaccine-history-epochs': ('pipeline.sound.select_epoch', 'main'),
        'vaccine-history': ('pipeline.sound.digital_twin', 'main'),
        'vaccine-reference': ('pipeline.sound.reference', 'train_main'),
    },
    'predict': {'age-external': ('pipeline.age.external', 'main'),
                'vaccine': ('pipeline.sound.reference', 'predict_main')},
    'evaluate': {'age': ('evaluation.age', 'main'), 'vaccine': ('evaluation.sound', 'main')},
    'baseline': {
        'vaccine': ('baseline.run_official_adapter','main'),
        'vaccine-embedding': ('baseline.train_vaccine_bridge','main'),
        'trvae': ('baseline.trvae','main'),
        'cima-nested': ('baseline.age.scripts.run_nested_cima','main'),
        'pascient-age': ('baseline.age.scripts.run_pascient_official','main'),
        'treatment': ('baseline.treatment','main'),
        'clinical': ('baseline.clinical','main'),
        'perception': ('baseline.clinical','perception_main'),
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for stage, tasks in STAGES.items():
        group = commands.add_parser(stage, help=f'{stage.capitalize()} a task')
        group.add_argument('task', choices=tasks)
        group.add_argument('arguments', nargs=argparse.REMAINDER)
    commands.add_parser('reproduce', help='Replay saved paper weights and compare all predictions', add_help=False)
    treatment = commands.add_parser('treatment', help='Treatment-state and clinical-response pipelines')
    treatment.add_argument('task', choices=['state', 'response'])
    treatment.add_argument('step', choices=['prepare', 'train', 'evaluate'])
    treatment.add_argument('--config', type=Path, default=Path('config/treatment.yaml'))
    treatment.add_argument('--fold', type=int, choices=range(5), default=0)
    treatment.add_argument('--final', action='store_true')
    args, extra = parser.parse_known_args()
    if args.action == 'reproduce':
        from pipeline.replay import main as replay
        sys.argv = ['cli.py reproduce', *extra]
        return replay()
    if extra: parser.error('Unrecognized arguments: '+' '.join(extra))
    if args.action == 'treatment':
        if args.task == 'state':
            from pipeline.treatment.state import prepare_fold_latents, train_fold, summarize_results
            if args.step == 'prepare': return prepare_fold_latents(args.config, args.fold)
            if args.step == 'train': return train_fold(args.config, args.fold)
            return summarize_results(args.config)
        from pipeline.treatment.response import train, score
        if args.step == 'prepare': parser.error('The response task prepares training-patient coordinates inside train.')
        return train(args.config, args.fold, args.final) if args.step == 'train' else score(args.config)
    module, function = STAGES[args.action][args.task]
    sys.argv = [f'cli.py {args.action} {args.task}', *args.arguments]
    return getattr(importlib.import_module(module), function)()


if __name__ == '__main__':
    main()

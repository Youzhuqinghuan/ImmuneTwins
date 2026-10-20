# ImmuneTwins

ImmuneTwins models individual immune states and their responses to vaccination and treatment from single-cell RNA-seq. The pipeline prepares cell and visit features, learns a patient encoder through immune-age pretraining, and fits task-specific response models. Vaccine forecasting combines a response decoder, personal-history adaptation and a personal-reference layer; treatment models predict immune states and clinical response. Evaluation uses held-out donors.

| Stage | CLI entry points |
|---|---|
| Data preparation | `prepare uce`, `prepare vaccine`, `prepare vaccine-reference` |
| Immune-age modelling | `train age`, `predict age-external`, `evaluate age` |
| Vaccine forecasting | `train vaccine-*`, `predict vaccine`, `evaluate vaccine` |
| Treatment forecasting | `treatment state`, `treatment response` |
| Baselines and saved-model replay | `baseline`, `reproduce` |

Model definitions are in `model/`, input preparation in `data/`, task workflows in `pipeline/`, and scoring in `evaluation/`. Settings are in `config/`.

Use Python 3.10 with the dependencies in [config/requirements.txt](config/requirements.txt). Neural training requires a CUDA GPU. Set the following variables in your shell or in `.local/env.sh`, which `run.sh` loads automatically:

```bash
export IMMUNETWINS_PYTHON=/path/to/dedicated/environment/bin/python
export IMMUNETWINS_WORKSPACE_ROOT=/path/to/immune-assets
export IMMUNETWINS_EXTERNAL_ROOT=/path/to/external-cohort-assets
export IMMUNETWINS_FROZEN_RESULTS=/path/to/immune-assets/results/ImmuneTwins
```

Copy `config/assets.example.yaml` to `config/assets.local.yaml` and configure your data paths. Datasets and weights are separate assets.

Run commands through `bash run.sh cli.py`. Use `--help` to list tasks and their arguments; `vaccine-*` above denotes the vaccine training subcommands.

```bash
bash run.sh cli.py --help
bash run.sh cli.py train --help
bash run.sh cli.py train vaccine-history --help
```

For Sound, first fit the affine, shape and history stages and set their output paths in `config/sound_reference.json`. Then prepare and fit the personal reference and generate final predictions:

```bash
bash run.sh cli.py prepare vaccine-reference --config config/sound_reference.json
CUDA_VISIBLE_DEVICES=1 bash run.sh cli.py train vaccine-reference --config config/sound_reference.json
bash run.sh cli.py predict vaccine --config config/sound_reference.json
```

The output `predictions.tsv` is the input to `evaluate vaccine`. To regenerate predictions from saved models and compare them with retained results:

```bash
CUDA_VISIBLE_DEVICES=1 bash run.sh cli.py reproduce --stage all --output results/replay
```

Select an available GPU for your machine. See `evaluate vaccine --help`, `treatment --help` and `reproduce --help` through the same CLI for scoring, treatment tasks and replay options.

Licensed under the [MIT License](LICENSE).

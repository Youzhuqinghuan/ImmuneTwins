"""Asset locations; model outputs are kept separate from inputs."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get('IMMUNETWINS_WORKSPACE_ROOT', ROOT/'assets')).expanduser().resolve()
EXTERNAL_ROOT = Path(os.environ.get('IMMUNETWINS_EXTERNAL_ROOT', WORKSPACE/'external')).expanduser().resolve()
FROZEN_RESULTS = Path(os.environ.get('IMMUNETWINS_FROZEN_RESULTS', WORKSPACE/'results/ImmuneTwins')).expanduser().resolve()
RESULTS = ROOT/'results'
CIMA_CHECKPOINT = Path(os.environ.get('IMMUNETWINS_CIMA_CHECKPOINT', FROZEN_RESULTS/'age/simple_v1/attention/full_cima/checkpoint.pt'))

def expand_paths(value):
    """Expand asset placeholders without altering numerical configuration values."""
    if isinstance(value, str):
        for key, path in {'REPO':ROOT, 'WORKSPACE':WORKSPACE, 'EXTERNAL':EXTERNAL_ROOT,
                          'FROZEN_RESULTS':FROZEN_RESULTS, 'CIMA_CHECKPOINT':CIMA_CHECKPOINT}.items():
            value = value.replace('${'+key+'}', str(path))
        return value
    if isinstance(value, dict): return {k:expand_paths(v) for k,v in value.items()}
    if isinstance(value, list): return [expand_paths(v) for v in value]
    if isinstance(value, tuple): return tuple(expand_paths(v) for v in value)
    return value

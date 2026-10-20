#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f .local/env.sh ]]; then source .local/env.sh; fi
: "${IMMUNETWINS_PYTHON:?Set IMMUNETWINS_PYTHON to the dedicated Python executable}"
export IMMUNETWINS_ASSETS="${IMMUNETWINS_ASSETS:-$PWD/config/assets.local.yaml}"
export TMPDIR="$PWD/.runtime/tmp" TMP="$PWD/.runtime/tmp" TEMP="$PWD/.runtime/tmp"
export XDG_CACHE_HOME="$PWD/.runtime/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface" HF_HUB_CACHE="$XDG_CACHE_HOME/huggingface/hub"
export HF_DATASETS_CACHE="$XDG_CACHE_HOME/huggingface/datasets" PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export IMMUNETWIN_LATENT_INDEX_CACHE_ROOT="$PWD/.runtime/latent_index"
export IMMUNETWIN_LATENT_INDEX_AUDIT_ROOT="$PWD/results/runtime/latent_index"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
unset PYTHONPATH
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
exec "$IMMUNETWINS_PYTHON" "$@"

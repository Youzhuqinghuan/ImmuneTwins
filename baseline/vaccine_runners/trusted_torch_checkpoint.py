#!/usr/bin/env python3
"""Load scvi-tools, scGen and scArches checkpoints under PyTorch 2.6 or later."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def trusted_local_full_object_load(torch_module, checkpoint_file: Path):
    """Allow weights_only=False for one existing checkpoint.

    PyTorch 2.6 changed the torch.load default to weights_only=True, whereas
    scvi-tools, scGen and scArches serializers store Python objects. The
    previous torch.load is restored on exit.
    """
    trusted = Path(checkpoint_file).resolve(strict=True)
    original = torch_module.load

    def compatible_load(file, *args, **kwargs):
        try:
            candidate = Path(file).resolve(strict=True)
        except (TypeError, OSError):
            candidate = None
        if candidate == trusted and "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original(file, *args, **kwargs)

    torch_module.load = compatible_load
    try:
        yield
    finally:
        torch_module.load = original

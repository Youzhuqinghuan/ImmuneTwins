from __future__ import annotations
import os
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]

def default_assets_path() -> Path:
    """Return an explicit override, a local config, or the public template."""

    override = os.environ.get("IMMUNETWINS_ASSETS")
    if override:
        return Path(override).expanduser()
    local = SOURCE_ROOT / "config" / "assets.local.yaml"
    if local.is_file():
        return local
    return SOURCE_ROOT / "config" / "assets.example.yaml"

from __future__ import annotations
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]

from paths import WORKSPACE as DEFAULT_WORKSPACE_ROOT

WORKSPACE_ROOT = Path(
    os.environ.get("IMMUNETWINS_WORKSPACE_ROOT", DEFAULT_WORKSPACE_ROOT)
).expanduser().resolve()


def resolve_workspace_artifact(value: str | Path) -> Path:
    """Resolve a path recorded in a saved manifest within the active workspace.

    Relative paths are read from the workspace root; an absolute path keeps
    only its part below the workspace directory.
    """

    path = Path(value).expanduser()
    if not path.is_absolute():
        return (WORKSPACE_ROOT / path).resolve()
    if path.is_relative_to(WORKSPACE_ROOT) or WORKSPACE_ROOT.name not in path.parts:
        return path.resolve()
    below = path.parts[path.parts.index(WORKSPACE_ROOT.name) + 1:]
    return WORKSPACE_ROOT.joinpath(*below).resolve()



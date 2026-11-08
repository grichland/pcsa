"""Every filesystem location the package uses. The only module that may locate the repository.

Everything the package writes goes under `output/` (or `PCSA_OUTPUT`). Tools installed by `pcsa setup`
live under `.tools/`. Both are ignored by git.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]  # src/pcsa/common/paths.py -> repo

OUTPUT_ENV = "PCSA_OUTPUT"
OUTPUT_SUBDIRS = ("runs", "scout")


def repo_root() -> Path:
    return _REPO


def assets_dir() -> Path:
    return _REPO / "assets"


def config_dir() -> Path:
    return _REPO / "config"


def tools_dir() -> Path:
    return _REPO / ".tools"


def tools_lock() -> Path:
    return _REPO / "tools.lock.json"


def output_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    raw = env.get(OUTPUT_ENV)
    return Path(raw).expanduser().resolve() if raw else _REPO / "output"


def runs_dir(environ: Mapping[str, str] | None = None) -> Path:
    return output_dir(environ) / "runs"


def ensure_output_tree(environ: Mapping[str, str] | None = None) -> Path:
    """Create `output/` and its subfolders; returns the output root."""
    root = output_dir(environ)
    for name in OUTPUT_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def tool(name: str) -> Path:
    """Install location of one tool from tools.lock.json (`into` key)."""
    return tools_dir() / name

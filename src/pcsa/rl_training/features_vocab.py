# From research/pcsa-rl/rl/features_vocab.py (sha256 8000d1eec5eb, 2026-11-08 20:13); keep diffable.
# Inference only: build() and main() (which rebuild data/vocab.json from pcsa's assets) stay in research.
"""Integer ids for species, items, abilities and moves (0 = unknown/pad), from data/vocab.json (a copy of
research's rl/data/vocab.json; the ids are the network's embedding rows, so the file must match the checkpoint)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from poke_env.data.normalize import to_id_str

VOCAB_PATH = Path(__file__).parent / "data" / "vocab.json"
KINDS = ("species", "item", "ability", "move")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def tables() -> dict[str, dict[str, int]]:
    return _read(VOCAB_PATH)


def sizes() -> dict[str, int]:
    """Embedding rows per kind (largest id + 1)."""
    return {kind: max(table.values()) + 1 for kind, table in tables().items()}


@lru_cache(maxsize=8192)
def index(kind: str, name: str | None) -> int:
    """The id of `name` (any spelling poke-env's to_id_str normalises), 0 when unknown."""
    if not name:
        return 0
    return tables()[kind].get(to_id_str(name), 0)

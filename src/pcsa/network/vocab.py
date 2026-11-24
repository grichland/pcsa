"""CG's numeric embedding ids -> CG string ids (assets/game/cg/vocab.json, made by tools/data/cg_vocab.py).

Every list is indexed by the id: species and items are table index + 1, moves and abilities the table index, 0 is
unknown (None) everywhere, and item id 1 is "holds nothing" ("").
"""

from __future__ import annotations

import json
from functools import cache

from pcsa.common.paths import assets_dir

KINDS = ("species", "moves", "items", "abilities")


@cache
def _data() -> dict:
    return json.loads((assets_dir() / "game" / "cg" / "vocab.json").read_text(encoding="utf-8"))


@cache
def tables() -> dict[str, list[str | None]]:
    return {k: _data()[k] for k in KINDS}


def mega_stones() -> list[str]:
    """The items (CG ids) a species' Mega form needs."""
    return list(_data()["mega_stones"])


def name(kind: str, number: int) -> str | None:
    """The CG id of `number` in vocabulary `kind`; None for 0 or a number outside the table."""
    table = tables()[kind]
    return table[number] if 0 <= number < len(table) else None

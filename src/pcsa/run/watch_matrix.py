"""The decision matrix of `pcsa watch`: the neural player's `matrix` note, checked and given display names.

The note holds our 11 actions (4 moves, the same 4 with Mega Evolution, 3 switches) against the opponent's 9
kept actions (4 moves, 5 switches): a value per pair, a probability per column, our final distribution, the
temperature `tau` that turned the weighted rows into it and the temperature `opp_temperature` the column
probabilities were taken at (the page works the row and column totals out itself, and what they would be at
another opponent temperature). Its names are ids (`dragondance`, `garchomp`); the page shows the written name:
from the record when it spells the name out, else from the game data, else the id with a capital. A note of
another shape gives no matrix.
"""

from __future__ import annotations

from functools import lru_cache
from math import isfinite

from pcsa.run.watch_names import shown, to_id

ROWS, COLS = 11, 9
MEGA_AT, SWITCH_AT = 4, 8
MEGA_SUFFIX = " + mega"
ORDER_PARTS = 2


def _number(value) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _positive(value) -> float | None:
    """A finite number above zero (a temperature: `tau`, `opp_temperature`); None for anything else."""
    number = _number(value)
    return number if number is not None and isfinite(number) and number > 0 else None


def _numbers(value, size: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != size:
        return None
    out = [_number(x) for x in value]
    return None if any(x is None for x in out) else out  # type: ignore[return-value]


def _strings(value, size: int) -> list[str] | None:
    if not isinstance(value, list) or len(value) != size or not all(isinstance(x, str) for x in value):
        return None
    return list(value)


def _flags(value, size: int, default: bool) -> list[bool]:
    return [bool(x) for x in value] if isinstance(value, list) and len(value) == size else [default] * size


@lru_cache(maxsize=1)
def _game_names() -> dict[str, str]:
    """id -> written name of every move and species the game data knows; {} when the data cannot be loaded."""
    try:
        from poke_env.data import GenData

        data = GenData.from_gen(9)
        tables = (data.pokedex, data.moves)
        return {key: entry["name"] for table in tables for key, entry in table.items() if entry.get("name")}
    except Exception:  # the page still works: ids are shown with a capital
        return {}


def pretty(ident: str, names: dict[str, str] | None = None) -> str:
    """The written name of a move or species id; '' for an empty slot."""
    if not ident:
        return ""
    key = to_id(ident)
    found = shown(ident, names or {})
    if found != ident:
        return found
    return _game_names().get(key) or ident[:1].upper() + ident[1:]


def row_label(ident: str, names: dict[str, str] | None = None) -> str:
    if ident.endswith(MEGA_SUFFIX):
        return f"{pretty(ident[: -len(MEGA_SUFFIX)], names)} + Mega"
    return pretty(ident, names)


def row_of_order(order, rows: list[str]) -> int | None:
    """The matrix row an order string plays ('move closecombat mega', '/choose switch Garchomp'), or None."""
    parts = str(order or "").removeprefix("/choose ").split()
    if len(parts) < ORDER_PARTS or parts[0] not in ("move", "switch"):
        return None
    if parts[0] == "move":
        moves = [to_id(r) for r in rows[:MEGA_AT]]
        target = to_id(parts[1])
        if target not in moves:
            return None
        return moves.index(target) + (MEGA_AT if "mega" in parts[2:] else 0)
    target = to_id("".join(parts[1:]))
    bench = [to_id(r) for r in rows[SWITCH_AT:]]
    exact = [i for i, name in enumerate(bench) if name and name == target]
    close = [i for i, name in enumerate(bench) if name and (name.startswith(target) or target.startswith(name))]
    found = exact or close
    return SWITCH_AT + found[0] if found else None


def matrix_view(raw, names: dict[str, str] | None = None, order=None, network=None) -> dict | None:
    """The matrix note as the page draws it: the note's own lists, the written names of its rows and columns, the
    row the network chose (`chosen`) and the row that was played (`played`: another one after an override)."""
    if not isinstance(raw, dict):
        return None
    rows, cols = _strings(raw.get("rows"), ROWS), _strings(raw.get("cols"), COLS)
    probs, p_cols = _numbers(raw.get("probs"), ROWS), _numbers(raw.get("p_cols"), COLS)
    cells = raw.get("cells")
    if rows is None or cols is None or probs is None or p_cols is None:
        return None
    grid = [_numbers(line, COLS) for line in cells] if isinstance(cells, list) and len(cells) == ROWS else [None]
    if any(line is None for line in grid):
        return None
    chosen = raw.get("chosen")
    chosen = chosen if isinstance(chosen, int) and not isinstance(chosen, bool) and 0 <= chosen < ROWS else None
    wanted = row_of_order(network, rows) if network else None
    played = row_of_order(order, rows)
    kinds = raw.get("col_kind")
    default_kinds = ["move"] * 4 + ["switch"] * (COLS - 4)
    return {
        "rows": rows,
        "row_labels": [row_label(r, names) for r in rows],
        "legal": _flags(raw.get("legal"), ROWS, True),
        "cols": cols,
        "col_labels": [pretty(c, names) for c in cols],
        "col_kind": [str(k) for k in kinds] if isinstance(kinds, list) and len(kinds) == COLS else default_kinds,
        "shown": _flags(raw.get("shown"), COLS, False),
        # which of their columns they can take at all (a fainted switch target is False); None in older records
        "col_legal": _flags(raw.get("col_legal"), COLS, True) if isinstance(raw.get("col_legal"), list) else None,
        "p_cols": p_cols,
        "cells": grid,
        "probs": probs,
        "chosen": wanted if wanted is not None else chosen,
        "played": played if played is not None else chosen,
        "mega_p": _number(raw.get("mega_p")),
        "tau": _positive(raw.get("tau")),
        "opp_top": raw.get("opp_top") if type(raw.get("opp_top")) is int and raw["opp_top"] >= 0 else None,
        "opp_temperature": _positive(raw.get("opp_temperature")),
    }

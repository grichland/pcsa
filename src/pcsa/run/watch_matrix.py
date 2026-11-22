"""The decision matrix of `pcsa watch`, built from the network's outputs (`notes.cg.net`).

The network scores our 11 actions (4 moves, the same 4 with Mega, 3 switches) against their 13 (4 moves, the same 4
with Mega, 5 switches). The page draws 9 columns: each of their moves is folded with its Mega variant, weighted by
the two probabilities, so every row's score (the sum over columns of probability times cell) stays the network's
`score`; the Mega share is `mega_p`. A net of another shape gives no matrix.

Their move columns: a checkpoint without belief_slots has one column for every move not yet seen ("Unrevealed
move"); with belief_slots (`opp_seen` in the net) each column is a move, shown or named by the belief head.

With decision-time search (`search` in the net) the view also carries what the search found, folded the same way:
its root strategy for our rows, its Q table and their strategy by column, and how it ran. `chosen` is then the
network's own pick and `played` the search's. With a choice rule (`choice` in the net) that changed the pick,
`chosen` is the network's own pick too and `choice` says which rule played instead.
"""

from __future__ import annotations

from math import isfinite

from pcsa.run.watch_names import pretty, species_label, to_id

ROWS, NET_COLS, COLS = 11, 13, 9
MEGA_AT, SWITCH_AT = 4, 8
MOVES = 4
MEGA_SUFFIX = " + mega"
UNSEEN = "unseen"  # the column of the move they have not shown yet ("" in the net)
UNSEEN_LABEL = "Unrevealed move"
ORDER_PARTS = 2


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if isfinite(value) else None


def _positive(value) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _numbers(value, size: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != size:
        return None
    out = [_number(x) for x in value]
    return None if any(x is None for x in out) else out  # type: ignore[return-value]


def _ids(value, size: int) -> list[str | None] | None:
    if not isinstance(value, list) or len(value) != size:
        return None
    if not all(x is None or isinstance(x, str) for x in value):
        return None
    return list(value)


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
        if not target or target not in moves:
            return None
        return moves.index(target) + (MEGA_AT if "mega" in parts[2:] else 0)
    target = to_id("".join(parts[1:]))
    bench = [to_id(r) for r in rows[SWITCH_AT:]]
    exact = [i for i, name in enumerate(bench) if name and name == target]
    close = [i for i, name in enumerate(bench) if name and (name.startswith(target) or target.startswith(name))]
    found = exact or close
    return SWITCH_AT + found[0] if found else None


def fold(p_opp: list[float], matrix: list[list[float]]) -> tuple[list[float], list[list[float]], float]:
    """Their 13 columns as the page's 9: (p_cols, cells, mega_p). A move and its Mega variant become one column
    weighted by their probabilities (the plain move's cell when both are 0); the switches are kept."""
    p_cols = [p_opp[m] + p_opp[m + MOVES] for m in range(MOVES)] + p_opp[SWITCH_AT:]
    cells = []
    for line in matrix:
        moves = [
            (p_opp[m] * line[m] + p_opp[m + MOVES] * line[m + MOVES]) / p_cols[m] if p_cols[m] > 0 else line[m]
            for m in range(MOVES)
        ]
        cells.append(moves + line[SWITCH_AT:])
    return p_cols, cells, sum(p_opp[MEGA_AT:SWITCH_AT])


def _rows(ids: list[str | None]) -> list[str]:
    out = []
    for i, ident in enumerate(ids):
        name = ident or ""
        out.append(f"{name}{MEGA_SUFFIX}" if name and MEGA_AT <= i < SWITCH_AT else name)
    return out


def _cols(ids: list[str | None]) -> list[str]:
    """The 9 column ids: their moves (the unseen slot named), then the switch targets; '' where none."""
    moves = [UNSEEN if ident == "" else ident or "" for ident in ids[:MOVES]]
    return moves + [ident or "" for ident in ids[SWITCH_AT:]]


def _col_label(ident: str, kind: str, names: dict[str, str] | None) -> str:
    if ident == UNSEEN:
        return UNSEEN_LABEL
    return species_label(ident, names) if kind == "switch" and ident else pretty(ident, names)


def _legal(value, closed=None) -> list[bool]:
    """Our legal rows, without those a rule mask closed."""
    indices = {i for i in value if type(i) is int} if isinstance(value, list) else set(range(ROWS))
    shut = {i for i in closed if type(i) is int} if isinstance(closed, list) else set()
    return [i in indices and i not in shut for i in range(ROWS)]


def _closed(value) -> list[bool]:
    shut = {i for i in value if type(i) is int} if isinstance(value, list) else set()
    return [i in shut for i in range(ROWS)]


def _seen(net: dict, cols: list[str]) -> list[bool]:
    """Which of their move columns were shown: the net's `opp_seen`, else every named one."""
    named = [bool(c) and c != UNSEEN for c in cols[:MOVES]]
    seen = net.get("opp_seen")
    if isinstance(seen, list) and len(seen) == MOVES and all(isinstance(x, bool) for x in seen):
        return [n and s for n, s in zip(named, seen, strict=True)]
    return named


def _grid(net: dict) -> tuple[list[float], list[list[float]], list[float]] | None:
    """The net's (pi, matrix, p_opp) when each has the contract's shape, else None."""
    pi, p_opp = _numbers(net.get("pi"), ROWS), _numbers(net.get("p_opp"), NET_COLS)
    raw = net.get("matrix")
    if pi is None or p_opp is None or not isinstance(raw, list) or len(raw) != ROWS:
        return None
    lines = [_numbers(line, NET_COLS) for line in raw]
    if any(line is None for line in lines):
        return None
    return pi, lines, p_opp  # type: ignore[return-value]


def _table(value) -> list[list[float | None]] | None:
    """An [11][13] table of numbers or None."""
    if not isinstance(value, list) or len(value) != ROWS:
        return None
    out = []
    for line in value:
        if not isinstance(line, list) or len(line) != NET_COLS:
            return None
        out.append([_number(x) for x in line])
    return out


def _row_index(value) -> int | None:
    return value if type(value) is int and 0 <= value < ROWS else None


def search_view(net: dict, matrix: list[list[float]]) -> dict | None:
    """The page's view of the search note: how it ran and, when it has them, its strategy per row and its Q table
    folded to the page's columns (a cell the search never held keeps the network's)."""
    found = net.get("search")
    if not isinstance(found, dict):
        return None
    out: dict = {
        "used": bool(found.get("used")),
        "why": str(found["why"])[:200] if found.get("why") else None,
        "changed": bool(found.get("changed")),
        "model_pick": _row_index(found.get("model_pick")),
        "pick": _row_index(found.get("pick")),
        "probs": None,
        "cells": None,
        "p_cols": None,
    }
    for key in ("sims", "worlds", "dropped", "depth", "nodes", "ms", "budget_ms", "p_other"):
        out[key] = _number(found.get(key))
    out["stopped"] = found.get("stopped") if found.get("stopped") in ("time", "sims", "clear", "even") else None
    probs, p_them = _numbers(found.get("strategy"), ROWS), _numbers(found.get("p_them"), NET_COLS)
    q = _table(found.get("q"))
    if probs is None or p_them is None or q is None:
        return out
    filled = [[x if x is not None else matrix[a][c] for c, x in enumerate(line)] for a, line in enumerate(q)]
    p_cols, cells, _ = fold(p_them, filled)
    out.update(probs=probs, cells=cells, p_cols=p_cols, visits=_numbers(found.get("visits"), ROWS))
    return out


def matrix_view(net, names: dict[str, str] | None = None, order=None, action=None) -> dict | None:
    """The page's matrix from the net: rows and columns with written names, the folded cells, our policy, the row
    CG played (`chosen`, from `action`) and the row the record's order played (`played`)."""
    if not isinstance(net, dict):
        return None
    grid = _grid(net)
    row_ids, col_ids = _ids(net.get("rows"), ROWS), _ids(net.get("cols"), NET_COLS)
    if grid is None or row_ids is None or col_ids is None:
        return None
    pi, matrix, p_opp = grid
    p_cols, cells, mega_p = fold(p_opp, matrix)
    rows, cols = _rows(row_ids), _cols(col_ids)
    seen = _seen(net, cols)
    chosen = action if type(action) is int and 0 <= action < ROWS else None
    played = row_of_order(order, rows)
    search = search_view(net, matrix)
    rule = choice_view(net)
    own = search["model_pick"] if search is not None and search["used"] else None
    own = rule["model_pick"] if rule is not None and rule["changed"] else own
    if own is not None:
        played = played if played is not None else chosen
        chosen = own  # the network's own row; the row played is the search's or the rule's
    kinds = ["move"] * MOVES + ["switch"] * (COLS - MOVES)
    return {
        "rows": rows,
        "row_labels": [
            species_label(r, names) if i >= SWITCH_AT else row_label(r, names) for i, r in enumerate(rows)
        ],
        "legal": _legal(net.get("legal"), net.get("closed")),
        "rule_closed": _closed(net.get("closed")),  # legal in the game, closed by a rule mask
        "cols": cols,
        "col_labels": [_col_label(c, k, names) for c, k in zip(cols, kinds, strict=True)],
        "col_kind": kinds,
        "shown": seen + [False] * (COLS - MOVES),
        # a move column named by the belief head, not shown yet
        "believed": [bool(c) and c != UNSEEN and not s for c, s in zip(cols[:MOVES], seen, strict=True)]
        + [False] * (COLS - MOVES),
        # a column they cannot take: an empty slot, or a switch the network gives no chance at all
        "col_legal": [bool(c) and (k == "move" or p > 0) for c, k, p in zip(cols, kinds, p_cols, strict=True)],
        "p_cols": p_cols,
        "cells": cells,
        "probs": pi,
        "chosen": chosen if chosen is not None else played,
        "played": played if played is not None else chosen,
        "mega_p": mega_p,
        "tau": _positive(net.get("tau")),
        "opp_top": None,
        "opp_temperature": None,
        "blend": None,
        "masked": None,
        "search": search,
        "choice": rule,
    }


def choice_view(net: dict) -> dict | None:
    """The choice rule's note: the rule, its k, and the network's pick against the rule's."""
    found = net.get("choice")
    if not isinstance(found, dict):
        return None
    k = {key: found.get(key) if type(found.get(key)) is int else None for key in ("top_ours", "top_theirs")}
    return {"rule": str(found.get("rule") or "")[:20], **k, "changed": bool(found.get("changed")),
            "model_pick": _row_index(found.get("model_pick")), "pick": _row_index(found.get("pick"))}  # fmt: skip

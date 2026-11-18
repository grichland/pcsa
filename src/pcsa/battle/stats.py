"""Level-50 stat arithmetic under Champions rules, and natures."""

from __future__ import annotations

import itertools
from functools import lru_cache

from pcsa.battle import game
from pcsa.battle.game import STATS

IV = 31
NATURE_DOWN, NATURE_NEUTRAL, NATURE_UP = 90, 100, 110
POINTS_MAX = 32
POINTS_TOTAL = 66


def stat_value(base: int, points: int, nature: int = NATURE_NEUTRAL, level: int = game.LEVEL) -> int:
    """Non-HP stat; one Champions stat point equals eight EVs."""
    raw = ((2 * base + IV + 2 * points) * level) // 100 + 5
    return raw * nature // 100


def hp_value(base: int, points: int, level: int = game.LEVEL) -> int:
    return ((2 * base + IV + 2 * points) * level) // 100 + level + 10


def solve_points(base: dict, stats: dict) -> tuple[dict, dict] | None:
    """Stat points and nature multipliers that reproduce exact level-50 stats, or None."""
    return _solve_points(tuple(sorted(base.items())), tuple(sorted(stats.items())))


@lru_cache(maxsize=4096)
def _solve_points(base_items: tuple, stats_items: tuple) -> tuple[dict, dict] | None:
    base, stats = dict(base_items), dict(stats_items)
    best = None
    for plus, minus in [(None, None), *itertools.permutations(STATS[1:], 2)]:
        nature = {s: NATURE_UP if s == plus else NATURE_DOWN if s == minus else NATURE_NEUTRAL for s in STATS[1:]}
        options = []
        for s in STATS:
            if stats.get(s) is None:
                options.append([None])
            elif s == "hp":
                options.append([p for p in range(POINTS_MAX + 1) if hp_value(base["hp"], p) == stats["hp"]])
            else:
                options.append([p for p in range(POINTS_MAX + 1) if stat_value(base[s], p, nature[s]) == stats[s]])
        for combo in itertools.product(*options):
            total = sum(p for p in combo if p is not None)
            rank = (total != POINTS_TOTAL, total)
            if total <= POINTS_TOTAL and (best is None or rank < best[0]):
                best = (rank, dict(zip(STATS, combo, strict=True)), nature)
    return (best[1], best[2]) if best else None


def nature_name(nature: dict[str, int]) -> str | None:
    """The nature whose raised and lowered stats match the multipliers (None when it is not a real nature)."""
    plus = next((s for s, v in nature.items() if v == NATURE_UP), None)
    minus = next((s for s, v in nature.items() if v == NATURE_DOWN), None)
    for entry in game.cg_table("natures"):  # in table order: a neutral nature reads as the first neutral one
        if entry["plus"] == plus and entry["minus"] == minus:
            return entry["name"]
    return None

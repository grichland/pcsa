"""Moves: base move data with Champions PP, and move names."""

from __future__ import annotations

from functools import lru_cache

from pcsa.battle import game

UNKNOWN_MOVE = {"type": "", "category": "", "power": 1, "accuracy": 101, "pp": 0}  # for a move with no data


@lru_cache(maxsize=1)
def champions_moves() -> dict[str, dict]:
    """Legal move id -> {type, category, power, accuracy, pp}."""
    return {
        key: {
            "type": e["type"],
            "category": e["category"],
            "power": e["basePower"],
            "accuracy": e["accuracy"],
            "pp": e["pp"],
        }
        for key, e in game.cg_table("moves").items()
    }


def base_data(move_id: str) -> dict:
    """Base move data with Champions PP; a move outside the legal table gets UNKNOWN_MOVE."""
    return dict(champions_moves().get(move_id) or UNKNOWN_MOVE)


def max_pp(move_id: str) -> int:
    return base_data(move_id)["pp"]


@lru_cache(maxsize=1)
def move_names() -> frozenset[str]:
    """Display names of the legal moves."""
    return frozenset(e["name"] for e in game.cg_table("moves").values() if e.get("name"))


def complete_move_name(name: str) -> bool:
    """A real move name that is not the start of a longer one."""
    names = move_names()
    return name in names and not any(other != name and other.startswith(name) for other in names)

"""Moves: base move data with Champions PP, move names and observed forms."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache

from pcsa.battle import game
from pcsa.battle.game import mega_form_for_stone, species_entry
from pcsa.battle.ids import to_id

FORM_LINES = ("switch", "drag", "replace", "detailschange", "-formechange")
UNKNOWN_MOVE = {"type": "", "category": "", "power": 1, "accuracy": 101, "pp": 0}  # pcsa's answer for no data


def reconcile_form(current: str, candidate: str) -> str:
    """A generic Mimikyu icon cannot restore a Disguise that was already busted."""
    current, candidate = to_id(current), to_id(candidate)
    intact = {"mimikyubusted": "mimikyu"}
    return current if intact.get(current) == candidate else candidate


def observed_form(battle, mon) -> str:
    """The latest form in the Tracker's lines (the species field keeps the base species after a Mega)."""
    if not hasattr(battle, "observed_form"):
        raise NotImplementedError("moves.observed_form needs a Tracker; use observed_form_from_lines")
    return battle.observed_form(mon)


def observed_form_from_lines(species: str, ident: str | None, lines: Iterable[Sequence[str]]) -> str:
    """The latest form of the member `ident` ("p1: Name") from split protocol lines, starting at `species`."""
    form = species
    if ident is None or ": " not in ident:  # team preview keys the opponent's members by bare species
        return form
    side, name = ident.split(": ", 1)
    for parts in lines:
        if len(parts) > 4 and parts[1] == "-mega" and parts[2].split(": ", 1) == [side + "a", name]:
            resolved = mega_form_for_stone(parts[3], parts[4])
            if resolved:
                form = to_id(resolved)
        if len(parts) > 3 and parts[1] in FORM_LINES:
            actor = parts[2].split(": ", 1)
            if len(actor) != 2 or actor[0][:2] != side or actor[1] != name:
                continue
            candidate = to_id(parts[3].split(",", 1)[0])
            current = species_entry(form)
            persistent = "Mega" in current.get("forme", "") or form == "mimikyubusted"
            same_base = to_id(current.get("baseSpecies") or form) == to_id(
                species_entry(candidate).get("baseSpecies") or candidate
            )
            if parts[1] in ("switch", "drag") and persistent and same_base:
                continue  # send-out rows name the base species after a persistent form change
            form = reconcile_form(form, candidate)
    return form


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
    """Base move data with Champions PP; a move outside the legal table gets pcsa's no-data answer."""
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

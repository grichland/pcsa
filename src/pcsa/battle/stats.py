"""Level-50 stat arithmetic under Champions rules, natures, and the current form of a tracked Pokémon."""

from __future__ import annotations

import itertools
from functools import lru_cache

from pcsa.battle import game
from pcsa.battle.game import STATS, species_entry
from pcsa.battle.ids import to_id

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


def form_of(mon, battle=None) -> tuple[str, bool]:
    """(dex key of the current form, is a Mega); with the Tracker, the form its lines show wins."""
    if battle is not None:
        from pcsa.battle.moves import observed_form

        observed = observed_form(battle, mon)
        if observed != mon.species:
            return observed, game.is_mega(observed)
    busted = any(getattr(e, "name", e) == "DISGUISE" for e in mon.effects)
    return form_from_species(mon.species, mon.base_stats, busted)


def form_from_species(species: str, base_stats: dict, disguise_busted: bool = False) -> tuple[str, bool]:
    """The pure part of form_of: a Mega entry is found by its base stats (a tracker may keep the base species)."""
    if game.base_name(species) == "mimikyu" and disguise_busted:
        return "mimikyubusted", False
    if game.is_mega(species):
        return species, True
    for key in game.megas()[0].get(game.base_name(species), []):
        if species_entry(key)["baseStats"] == base_stats:
            return key, True
    return species, False


def our_stats(mon, form: str, is_mega: bool) -> tuple[dict, dict]:
    """Exact stats and stat points; a Mega's stats from the solved points."""
    stats = dict(mon.stats or {})
    if not stats.get("hp") and mon.max_hp:
        stats["hp"] = mon.max_hp
    solved = solve_points(species_entry(mon.species).get("baseStats") or mon.base_stats, stats)
    if is_mega and solved:
        points, nature = solved
        base = species_entry(form)["baseStats"]
        stats = {
            s: (hp_value(base[s], points[s]) if s == "hp" else stat_value(base[s], points[s], nature[s]))
            if points[s] is not None
            else None
            for s in STATS
        }
    exact = {s: (int(v) if v is not None else None) for s, v in stats.items()}
    return exact, (solved[0] if solved else {})


def known_ability(battle, mon) -> str:
    """The ability a log row showed (or a Mega's single ability); never a guess from the species."""
    if not hasattr(battle, "replay_data"):
        raise NotImplementedError("stats.known_ability needs a Tracker; use known_ability_from")
    if any(member is mon for member in battle.team.values()):
        return to_id(mon.ability or "")
    form, mega = form_of(mon)
    return known_ability_from(form, mega, mon.name, battle.replay_data)


def known_ability_from(form: str, is_mega: bool, name: str, lines) -> str:
    """The pure part of known_ability for an opponent: a Mega's single ability, else the last -ability line."""
    if is_mega:
        return to_id(next(iter(species_entry(form).get("abilities", {}).values()), ""))
    for parts in reversed(list(lines)):
        if len(parts) > 3 and parts[1] == "-ability" and parts[2].endswith(": " + name):
            return to_id(parts[3])
    return ""

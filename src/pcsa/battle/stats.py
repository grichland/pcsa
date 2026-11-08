"""Level-50 stat arithmetic under Champions rules, and the current form of a tracked Pokémon."""

from __future__ import annotations

import itertools
from functools import lru_cache

from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.game import STATS, species_entry

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
    """Stat points and nature multipliers that reproduce exact level-50 stats, or None.

    A -10% stat can fit two point values; ties go to a total of exactly 66 points, then the smaller
    total. Cached: callers must not modify the result."""
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
    from poke_env.data import GenData

    plus = next((s for s, v in nature.items() if v == NATURE_UP), None)
    minus = next((s for s, v in nature.items() if v == NATURE_DOWN), None)
    for name, entry in GenData.from_gen(9).natures.items():
        raised = next((s for s in STATS[1:] if float(entry.get(s, 1)) > 1), None)
        lowered = next((s for s in STATS[1:] if float(entry.get(s, 1)) < 1), None)
        if raised == plus and lowered == minus:
            return str(name).title()
    return None


def multipliers(item: str | None, ability: str | None, species: str) -> dict[str, float]:
    """Known stat multipliers from a held item (Choice items, Assault Vest, Eviolite) and ability."""
    mult = dict(game.rule("known_stat_multipliers.items").get(item or "", {}))
    if item == "eviolite" and species_entry(species).get("evos"):
        mult.update(game.rule("known_stat_multipliers.eviolite"))
    for s, m in game.rule("known_stat_multipliers.abilities").get(ability or "", {}).items():
        mult[s] = mult.get(s, 1.0) * m
    return mult


def stat_ranges(base: dict) -> dict[str, tuple[int, int]]:
    """Opponent min/max per stat under Champions rules."""
    out = {"hp": (hp_value(base["hp"], 0), hp_value(base["hp"], POINTS_MAX))}
    for s in STATS[1:]:
        out[s] = (stat_value(base[s], 0, NATURE_DOWN), stat_value(base[s], POINTS_MAX, NATURE_UP))
    return out


def form_of(mon, battle=None) -> tuple[str, bool]:
    """(dex key of the current form, is a Mega). poke-env keeps the base species after a Mega
    Evolution and swaps base stats, so a Mega entry is found by its base stats."""
    if battle is not None:
        from pcsa.battle.moves import observed_form

        observed = observed_form(battle, mon)
        if observed != mon.species:
            return observed, game.is_mega(observed)
    if game.base_name(mon.species) == "mimikyu" and any(e.name == "DISGUISE" for e in mon.effects):
        return "mimikyubusted", False
    if game.is_mega(mon.species):
        return mon.species, True
    for key in game.megas()[0].get(game.base_name(mon.species), []):
        if species_entry(key)["baseStats"] == mon.base_stats:
            return key, True
    return mon.species, False


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


def current_stats(mon, exact: bool = True, form: str | None = None) -> dict[str, float]:
    """Current-form stats (or those of `form`, e.g. the Mega the mon becomes this turn); bounded midpoints
    where exact stats are unknown (the opponent)."""
    form, mega = (form, game.is_mega(form)) if form else form_of(mon)
    values = our_stats(mon, form, mega)[0] if exact and getattr(mon, "stats", None) else {}
    ranges = stat_ranges(species_entry(form).get("baseStats") or mon.base_stats)
    return {stat: values.get(stat) or sum(bounds) / 2 for stat, bounds in ranges.items()}


PARALYSIS_SPEED = 0.5
STAGE_BASE = 2


def stage(value: int, boost: int) -> int:
    """A stat after a -6..+6 stage."""
    if boost >= 0:
        return value * (STAGE_BASE + boost) // STAGE_BASE
    return value * STAGE_BASE // (STAGE_BASE - boost)


def effective_speed(mon, exact: bool = True) -> float:
    """Speed as the engine would use it now: the current-form stat (ours exact, theirs a bounded midpoint),
    a known Choice Scarf, the speed stage, and paralysis."""
    speed = current_stats(mon, exact)["spe"]
    item = to_id_str(mon.item or "")
    if item and item != "unknown_item":
        speed *= multipliers(item, None, mon.species).get("spe", 1.0)
    speed = float(stage(int(speed), int((mon.boosts or {}).get("spe", 0))))
    if mon.status is not None and mon.status.name.lower() == "par":
        speed *= PARALYSIS_SPEED
    return speed


def known_ability(battle, mon) -> str:
    """The ability a log row showed (or a Mega's single ability); never poke-env's default guess."""
    if any(member is mon for member in battle.team.values()):
        return to_id_str(mon.ability or "")
    form, mega = form_of(mon)
    if mega:
        return to_id_str(next(iter(species_entry(form).get("abilities", {}).values()), ""))
    for parts in reversed(getattr(battle, "_replay_data", [])):
        if len(parts) > 3 and parts[1] == "-ability" and parts[2].endswith(": " + mon.name):
            return to_id_str(parts[3])
    return ""

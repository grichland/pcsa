"""Regenerator on the opponent's switch-out (13 Nov 2026 fix).

The phone's battle log prints no HP when the opponent sends a Pokemon out, and the screen's action HUD reads only
the opponent's name (pcsa.phone.fields.action_hud), so the translator carries the last HP it saw. A Regenerator
user that switched out (a voluntary switch, a pivot or a drag; not a faint) is back with 1/3 of its HP more:
the HP the next `|switch|` line carries is raised here, silently (Showdown prints nothing for it either). The next
`[HP Lost]` row states the real HP again.

"Surely Regenerator": revealed, every possible ability of the species, or every seen build of the species
(assets/sets/seen_builds.json) runs it.
"""

from __future__ import annotations

from poke_env.data.normalize import to_id_str

from pcsa.battle import game

REGENERATOR = "regenerator"
FULL = 100  # their HP is tracked in percent
HEAL_PERCENT = FULL / 3.0


def seen_abilities(species: str) -> frozenset[str]:
    """The abilities the seen builds of `species` run (empty when none is seen or the file cannot be read)."""
    from pcsa.battle.seen_builds import shared

    try:
        store = shared()
    except Exception:  # noqa: BLE001 - a broken file never stops tracking
        return frozenset()
    key = to_id_str(species)
    ranked = store.ranked(key) or store.ranked(to_id_str(game.base_name(key)))
    return frozenset(to_id_str(s.build.ability) for s in ranked if s.build.ability)


def surely_regenerator(mon) -> bool:
    if mon.ability:
        return to_id_str(mon.ability) == REGENERATOR
    possible = frozenset(to_id_str(a) for a in (mon.possible_abilities or []))
    if possible == {REGENERATOR}:
        return True
    if REGENERATOR not in possible:
        return False
    seen = seen_abilities(mon.species)
    return seen == {REGENERATOR}


def on_switch_out(mon) -> bool:
    """Heal an opposing `mon` (HP in percent) leaving the field alive; True when it healed."""
    if mon is None or mon.fainted or not mon.active or mon.current_hp is None or mon.current_hp <= 0:
        return False
    if mon.current_hp >= FULL or not surely_regenerator(mon):
        return False
    mon._current_hp = min(FULL, int(round(mon.current_hp + HEAL_PERCENT)))
    return True

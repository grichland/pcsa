"""Whether our active cannot switch out, judged from the tracked battle (Showdown's `trapped` request flag)."""

from __future__ import annotations

from poke_env.battle import Battle
from poke_env.battle.effect import Effect
from poke_env.battle.pokemon_type import PokemonType
from poke_env.data.normalize import to_id_str

from pcsa.battle import game

# Trapped while the foe that applied it stays in.
FOE_TRAPS = frozenset(
    {
        Effect.BIND,
        Effect.CLAMP,
        Effect.FIRE_SPIN,
        Effect.INFESTATION,
        Effect.MAGMA_STORM,
        Effect.PARTIALLY_TRAPPED,
        Effect.SAND_TOMB,
        Effect.SNAP_TRAP,
        Effect.THUNDER_CAGE,
        Effect.TRAPPED,
        Effect.WHIRLPOOL,
        Effect.WRAP,
        Effect.OCTOLOCK,
    }
)


def foe_abilities(mon) -> set[str]:
    """The foe's ability id when known, else every ability its species can have."""
    known = to_id_str(mon.ability or "")
    if known:
        return {known}
    entry = game.species_entry(mon.species)
    return {to_id_str(a) for a in (entry.get("abilities") or {}).values()}


def _applied_by_foe_still_here(battle: Battle, mon) -> bool:
    """No opposing switch since the trap on `mon` began (the trap ends silently when its user leaves)."""
    name = next((k.split(": ", 1)[-1] for k, v in battle.team.items() if v is mon), "")
    last_trap = last_foe = -1
    for i, parts in enumerate(getattr(battle, "_replay_data", [])):
        if len(parts) < 3:
            continue
        if parts[1] in ("switch", "drag") and parts[2].startswith("p2"):
            last_foe = i
        elif (
            parts[1] == "-start"
            and len(parts) > 3
            and parts[2].startswith("p1")
            and parts[2].split(": ", 1)[-1] == name
            and Effect.from_showdown_message(parts[3]) in FOE_TRAPS
        ):
            last_trap = i
    return last_trap >= last_foe


def is_trapped(battle: Battle, mine: dict | None = None) -> bool:
    """Our active cannot be switched out. `mine` is its request entry (item, ability) when it is not yet applied."""
    me, foe = battle.active_pokemon, battle.opponent_active_pokemon
    if me is None:
        return False
    mine = mine or {}
    item = to_id_str(mine.get("item") or (me.item if isinstance(me.item, str) else ""))
    ability = to_id_str(me.ability or mine.get("ability") or "")
    types = set(me.types)
    if PokemonType.GHOST in types or item == "shedshell":
        return False
    if Effect.NO_RETREAT in me.effects:
        return True
    if any(e in FOE_TRAPS for e in me.effects) and _applied_by_foe_still_here(battle, me):
        return True
    if foe is None or foe.fainted:
        return False
    theirs = foe_abilities(foe)
    if theirs == {"shadowtag"} and ability != "shadowtag":
        return True
    if theirs == {"arenatrap"}:
        airborne = PokemonType.FLYING in types or ability == "levitate" or item == "airballoon"
        return not (airborne or Effect.MAGNET_RISE in me.effects)
    return theirs == {"magnetpull"} and PokemonType.STEEL in types

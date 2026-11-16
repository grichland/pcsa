"""Two guard rules for a 4x weakness (e.g. Mega Golisopod, Bug/Steel, is one-shot by Fire; base Golisopod is not).

  mega_weakness   a Mega Evolving move whose Mega form is 4x weak to a likely or shown attack (the base form is not)
                  and does not KO first: the same move without Mega Evolution
  weak4x_switch   our active is 4x weak to such an attack that lands first and KOs it, and no move of ours KOs first:
                  the network's best switch that takes it at most neutrally"""

from __future__ import annotations

from collections.abc import Iterable

from poke_env.battle import Move, PokemonType
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.players.opponent import move_object
from pcsa.players.tactics import mega_form_of

LIKELY = 0.15  # the opponent head's probability that makes an attack "likely"
FOURFOLD = 4.0
SUPER = 1.0  # a switch-in must take the attack at most neutrally


def multiplier(move_type: str, types: Iterable[str]) -> float:
    """Type chart multiplier of `move_type` on `types` (1.0 when unknown)."""
    chart = GenData.from_gen(9).type_chart
    try:
        attack = PokemonType.from_name(move_type)
        kinds = [PokemonType.from_name(t) for t in types if t]
    except Exception:  # a type the chart does not know
        return 1.0
    if not kinds:
        return 1.0
    return attack.damage_multiplier(*kinds, type_chart=chart) if len(kinds) <= 2 else 1.0  # noqa: PLR2004


def types_of(species: str) -> list[str]:
    return [t.lower() for t in (game.species_entry(to_id_str(species)) or {}).get("types", [])]


def mon_types(mon) -> list[str]:
    return [t.name.lower() for t in mon.types if t is not None]


def likely_attacks(theirs, their) -> list[tuple[Move, float | None]]:
    """Their damaging moves the opponent head gives at least LIKELY, then the ones they have shown (p None)."""
    out: dict[str, tuple[Move, float | None]] = {}
    for kind, move_id, p in their or ():
        move = move_object(to_id_str(move_id)) if kind == "move" and move_id else None
        if move is not None and move.category.name != "STATUS" and p >= LIKELY:
            out[move.id] = (move, p)
    for move_id in getattr(theirs, "moves", {}) or {}:
        move = move_object(to_id_str(move_id))
        if move is not None and move.category.name != "STATUS" and move.id not in out:
            out[move.id] = (move, None)
    return list(out.values())


def kos_first(spot, move: Move, against: Move) -> bool:
    """Our `move` KOs their active before `against` lands."""
    if spot.hit(move) < spot.theirs.current_hp_fraction:
        return False
    if move.priority != against.priority:
        return move.priority > against.priority
    return spot.outspeeds()


def _said(move: Move, p: float | None) -> str:
    return f"{move.id} ({p:.0%} likely)" if p is not None else f"{move.id} (shown)"


def mega_reason(spot, chosen_move: Move) -> str | None:
    """Why Mega Evolving with `chosen_move` walks into a new 4x weakness, else None."""
    form = mega_form_of(spot.mine)
    if form is None:
        return None
    mega, base = types_of(form), mon_types(spot.mine)
    for attack, p in likely_attacks(spot.theirs, spot.their):
        kind = attack.type.name.lower()
        if multiplier(kind, mega) >= FOURFOLD > multiplier(kind, base) and not kos_first(spot, chosen_move, attack):
            return f"Mega {form} ({'/'.join(mega)}) takes {_said(attack, p)} 4x; {spot.mine.species} does not"
    return None


def doomed_by(spot) -> tuple[Move, float | None] | None:
    """Their likely attack that hits our active 4x, lands first and KOs it, when none of our moves KOs first."""
    ours = mon_types(spot.mine)
    for attack, p in likely_attacks(spot.theirs, spot.their):
        if multiplier(attack.type.name.lower(), ours) < FOURFOLD:
            continue
        first = attack.priority > 0 or not spot.outspeeds()
        if not first or spot.taken(attack) < spot.mine.current_hp_fraction:
            continue
        moves = [a.order.order for a in spot.moves()]
        if not any(isinstance(m, Move) and kos_first(spot, m, attack) for m in moves):
            return attack, p
    return None


def safe_against(target, attack: Move) -> bool:
    return multiplier(attack.type.name.lower(), mon_types(target)) <= SUPER

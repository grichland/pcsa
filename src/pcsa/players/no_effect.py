"""Moves that surely do nothing to the target: the neural player's `no_effect` guard rule (pcsa.players.neural_rules).

`why(battle, attacker, defender, move)` names the reason, or None when the move may work. Only what is certain
counts: the type chart, a revealed ability or item, or an ability every possible ability of the species shares (an
unrevealed Levitate that the species may not have is not certain). Attacker abilities that break these immunities
(Mold Breaker and its kin, Scrappy, Corrosion) turn the check off.

  priority   a priority move into Psychic Terrain on a grounded target, or into Queenly Majesty / Dazzling /
             Armor Tail
  first turn Fake Out / First Impression after the user's first turn out
  type       an immune type (Ground into Flying, Electric into Ground, Normal and Fighting into Ghost ...);
             Ground into Levitate, Air Balloon or Magnet Rise; Thunder Wave into Ground
  ability    an absorbing ability (Volt Absorb, Lightning Rod, Motor Drive, Water Absorb, Storm Drain, Dry Skin, Flash
             Fire, Well-Baked Body, Sap Sipper, Earth Eater), Bulletproof, Soundproof; Good as Gold against status
             moves, Magic Bounce against reflectable ones
  status     a status the target cannot get: its type (Electric: paralysis, Fire: burn, Poison and Steel: poison,
             Grass: powder moves), its ability (Insomnia, Limber, Water Veil, Immunity, Purifying Salt ...), or the
             terrain (Misty Terrain: every status; Electric Terrain: sleep and Yawn) on a grounded target
"""

from __future__ import annotations

from poke_env.battle import Effect, Field, Move, MoveCategory, PokemonType, Status
from poke_env.data.normalize import to_id_str

FOE_TARGETS = frozenset({"normal", "any", "adjacentfoe", "alladjacentfoes", "randomnormal", "alladjacent"})
PRIORITY_BLOCKERS = frozenset({"queenlymajesty", "dazzling", "armortail"})
FIRST_TURN = frozenset({"fakeout", "firstimpression"})
MOLD_BREAKERS = frozenset({"moldbreaker", "teravolt", "turboblaze", "myceliummight"})
SCRAPPY = frozenset({"scrappy", "mindseye"})
VARIABLE_TYPE = frozenset(
    "weatherball terrainpulse judgment multiattack revelationdance naturalgift technoblast ragingbull ivycudgel "
    "terablast hiddenpower aurawheel".split()
)
TYPE_ABSORB = {
    PokemonType.ELECTRIC: frozenset({"voltabsorb", "lightningrod", "motordrive"}),
    PokemonType.WATER: frozenset({"waterabsorb", "stormdrain", "dryskin"}),
    PokemonType.FIRE: frozenset({"flashfire", "wellbakedbody"}),
    PokemonType.GRASS: frozenset({"sapsipper"}),
    PokemonType.GROUND: frozenset({"levitate", "eartheater"}),
}
FLAG_BLOCK = {"bullet": frozenset({"bulletproof"}), "sound": frozenset({"soundproof"})}
ANY_STATUS = frozenset({"comatose", "purifyingsalt"})
STATUS_ABILITY = {
    Status.SLP: frozenset({"insomnia", "vitalspirit", "sweetveil"}),
    Status.PAR: frozenset({"limber"}),
    Status.BRN: frozenset({"waterveil", "waterbubble", "thermalexchange"}),
    Status.PSN: frozenset({"immunity", "pastelveil"}),
    Status.TOX: frozenset({"immunity", "pastelveil"}),
}
STATUS_TYPES = {
    Status.PAR: (PokemonType.ELECTRIC,),
    Status.BRN: (PokemonType.FIRE,),
    Status.PSN: (PokemonType.POISON, PokemonType.STEEL),
    Status.TOX: (PokemonType.POISON, PokemonType.STEEL),
}


def _target(move: Move) -> str:
    return to_id_str(getattr(move.target, "name", "") or str(move.target or ""))


def abilities(mon) -> frozenset[str]:
    """The abilities `mon` may have: the revealed one, else every possible one."""
    if mon.ability:
        return frozenset({to_id_str(mon.ability)})
    return frozenset(to_id_str(a) for a in (mon.possible_abilities or []))


def surely(mon, names: frozenset[str]) -> bool:
    """Whether `mon` surely has one of `names` (revealed, or every possible ability is one of them)."""
    have = abilities(mon)
    return bool(have) and have <= names


def _types(mon) -> tuple:
    return tuple(t for t in (mon.type_1, mon.type_2) if t is not None)


def airborne(battle, mon) -> bool:
    """Surely out of reach of Ground moves (Gravity and Iron Ball ground everything)."""
    if Field.GRAVITY in battle.fields or mon.item == "ironball":
        return False
    if PokemonType.FLYING in _types(mon) or mon.item == "airballoon" or Effect.MAGNET_RISE in mon.effects:
        return True
    return surely(mon, frozenset({"levitate"}))


def priority_blocked(battle, defender, move: Move) -> bool:
    """A priority move aimed at the target that cannot land: Psychic Terrain and a grounded target, or a target whose
    known ability (Queenly Majesty, Dazzling, Armor Tail) blocks priority."""
    if (move.priority or 0) <= 0 or _target(move) not in FOE_TARGETS:
        return False
    if Field.PSYCHIC_TERRAIN in battle.fields and battle.is_grounded(defender):
        return True
    return surely(defender, PRIORITY_BLOCKERS)


def _type_reason(battle, attacker, defender, move: Move) -> str | None:
    if move.id in VARIABLE_TYPE:
        return None
    if move.type == PokemonType.GROUND and airborne(battle, defender):
        return "cannot reach an airborne target"
    if move.category == MoveCategory.STATUS and move.id != "thunderwave":
        return None
    if Field.GRAVITY in battle.fields and move.type == PokemonType.GROUND:
        return None
    scrappy = surely(attacker, SCRAPPY) and move.type in (PokemonType.NORMAL, PokemonType.FIGHTING)
    if not scrappy and defender.damage_multiplier(move) == 0:
        return f"{defender.species} is immune to {move.type.name.title()}"
    return None


def _ability_reason(attacker, defender, move: Move) -> str | None:
    if surely(attacker, MOLD_BREAKERS):
        return None
    absorb = TYPE_ABSORB.get(move.type) if move.id not in VARIABLE_TYPE else None
    if absorb and move.category != MoveCategory.STATUS and surely(defender, absorb):
        return f"{defender.species}'s ability absorbs it"
    for flag, names in FLAG_BLOCK.items():
        if flag in (move.flags or ()) and surely(defender, names):
            return f"{defender.species}'s ability blocks it"
    if move.category == MoveCategory.STATUS:
        if surely(defender, frozenset({"goodasgold"})):
            return "Good as Gold blocks status moves"
        if "reflectable" in (move.flags or ()) and surely(defender, frozenset({"magicbounce"})):
            return "Magic Bounce reflects it"
    return None


def _status_reason(battle, attacker, defender, move: Move) -> str | None:
    status = Status.SLP if move.volatile_status == Effect.YAWN else move.status
    if move.category != MoveCategory.STATUS or status is None:
        return None
    if status in (Status.PSN, Status.TOX) and surely(attacker, frozenset({"corrosion"})):
        return None
    if any(t in _types(defender) for t in STATUS_TYPES.get(status, ())):
        return f"{defender.species}'s type cannot be {status.name.lower()}"
    if "powder" in (move.flags or ()) and (PokemonType.GRASS in _types(defender) or defender.item == "safetygoggles"):
        return "powder moves do not affect it"
    if not surely(attacker, MOLD_BREAKERS) and surely(defender, ANY_STATUS | STATUS_ABILITY.get(status, frozenset())):
        return f"{defender.species}'s ability prevents it"
    if battle.is_grounded(defender):
        if Field.MISTY_TERRAIN in battle.fields:
            return "Misty Terrain prevents status"
        if status == Status.SLP and Field.ELECTRIC_TERRAIN in battle.fields:
            return "Electric Terrain prevents sleep"
    return None


def why(battle, attacker, defender, move: Move) -> str | None:
    """Why `move` from `attacker` surely does nothing to `defender`, or None."""
    if attacker is None or defender is None or _target(move) not in FOE_TARGETS:
        return None
    if move.id in FIRST_TURN and not attacker.first_turn:
        return "only works on the user's first turn out"
    if priority_blocked(battle, defender, move):
        return "priority is blocked here (Psychic Terrain or the target's ability)"
    return (
        _type_reason(battle, attacker, defender, move)
        or _ability_reason(attacker, defender, move)
        or _status_reason(battle, attacker, defender, move)
    )

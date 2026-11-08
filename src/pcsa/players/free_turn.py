"""The free-turn player: disciplined, plus a setup / Stealth Rock / status move on a turn the foe cannot punish.

Disciplined's sweeper denier and switching rules come first. Before its recovery and plain attack, the free-turn
gate (research R3, R6, R7, R11, web A1; thresholds in config/players/free_turn.yaml) opens when all hold:
their best hit on us < `free_hit_max` of our max HP and they cannot KO us; no SE STAB of theirs on us; no
phaze/Haze/Clear Smog/Encore/Taunt/Perish Song on the foe (revealed or assumed); we cannot KO them now; we are
not ahead in members (Stealth Rock on turn 1 excepted). Then the first that passes its own guard:
1. setup: a self-boosting move, raised stats below `setup_max_stage`, HP above `setup_min_hp`, a matching
   attack to use the boost (Agility-like only when slower; never vs Unaware);
2. Stealth Rock: once per battle, not up, 2+ foe members left, no Magic Bounce, no hazard removal known;
3. status: once per foe, un-statused and not immune (types, abilities, terrain, Safeguard, Substitute, sleep
   clause); Will-O-Wisp only at a physical attacker.
Else disciplined's recovery and attack. Notes: {"rule": "free:setup|free:rocks|free:status"}; when a free
move was legal but the gate failed, {"free": "not free: <why>"}.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml
from poke_env.battle import Effect, Field, Move, SideCondition, Status, Target

from pcsa.battle.builds import Catalog
from pcsa.battle.stats import known_ability
from pcsa.common.errors import ConfigError
from pcsa.players import opponent, threat
from pcsa.players.base import LegalAction
from pcsa.players.disciplined import PHAZE_TARGETS, DisciplinedParams, DisciplinedPlayer, _by_id, _left, _move
from pcsa.players.heuristic import HeuristicParams

ANTI_FREE = frozenset("haze clearsmog encore taunt roar whirlwind dragontail circlethrow perishsong".split())
HAZARD_REMOVAL = frozenset({"defog", "rapidspin", "tidyup", "mortalspin"})
ROCKS = "stealthrock"
STATUS_MOVES = {  # preference order
    "spore": "slp",
    "sleeppowder": "slp",
    "yawn": "slp",
    "willowisp": "brn",
    "toxic": "tox",
    "thunderwave": "par",
    "stunspore": "par",
    "glare": "par",
}
TYPE_IMMUNE = {"brn": {"FIRE"}, "tox": {"POISON", "STEEL"}, "par": {"ELECTRIC"}, "slp": set()}
MOVE_TYPE_IMMUNE = {"thunderwave": {"GROUND"}}
# abilities that block the condition or turn it to their advantage
ABILITY_IMMUNE = {
    "brn": {"waterveil", "waterbubble", "thermalexchange", "flareboost", "magicguard"},
    "tox": {"immunity", "pastelveil", "poisonheal", "toxicboost", "magicguard"},
    "par": {"limber"},
    "slp": {"insomnia", "vitalspirit", "sweetveil", "earlybird"},
}
STATUS_PROOF = frozenset(
    {"magicbounce", "goodasgold", "comatose", "purifyingsalt", "guts", "marvelscale", "quickfeet"}
)
POWDER_PROOF = frozenset({"overcoat"})
OFFENSE = {"atk": "PHYSICAL", "spa": "SPECIAL"}
UNAWARE = "unaware"
MAGIC_BOUNCE = "magicbounce"
FIRST_TURN = 1


@dataclass
class FreeTurnParams:
    free_hit_max: float = 0.4  # the turn is free only when their best hit is below this share of our max HP
    setup_max_stage: int = 2  # no setup once a raised stat is at this stage
    setup_min_hp: float = 0.5  # setup only above this HP fraction

    @classmethod
    def load(cls, path: Path) -> FreeTurnParams:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(f"unknown free_turn parameters in {path}: {sorted(unknown)}")
        return cls(**data)


def raises(move: Move | None) -> list[str]:
    """The stats a self-targeting boost move raises (empty for anything else)."""
    if move is None or move.target != Target.SELF or not move.boosts or sum(move.boosts.values()) <= 0:
        return []
    return [stat for stat, value in move.boosts.items() if value > 0]


def free_kind(move: Move | None) -> bool:
    """Whether `move` is one the free-turn gate could pick."""
    return move is not None and (bool(raises(move)) or move.id == ROCKS or move.id in STATUS_MOVES)


def _types(mon) -> set[str]:
    return {t.name for t in mon.types if t is not None}


class FreeTurnPlayer(DisciplinedPlayer):
    name = "free_turn"

    def __init__(
        self,
        rules: DisciplinedParams | None = None,
        free: FreeTurnParams | None = None,
        catalog: Catalog | None = None,
        heuristic: HeuristicParams | None = None,
    ) -> None:
        self.free = free or FreeTurnParams()
        super().__init__(rules, catalog, heuristic)

    def params(self) -> dict:
        return {**super().params(), **asdict(self.free)}

    def _reset(self, tag: str | None) -> None:
        super()._reset(tag)
        self._rocks_used = False
        self._statused: set[str] = set()

    # Foe knowledge --------------------------------------------------------------------------------------
    def _foe_moves(self, battle, mon) -> set[str]:
        """Revealed plus assumed move ids of an opposing member."""
        return set(mon.moves) | set(opponent.assumed(battle, mon, *self._known).moves)

    def _foe_ability(self, battle, mon) -> str:
        return opponent.assumed(battle, mon, *self._known).ability or ""

    def _foe_attacks(self, battle, mon) -> list[Move]:
        found = (opponent.move_object(i) for i in self._foe_moves(battle, mon))
        return [m for m in found if m is not None and m.category.name != "STATUS"]

    # The gate -------------------------------------------------------------------------------------------
    def _recover(self, battle, moves: list[LegalAction]) -> LegalAction | None:
        """Disciplined calls this after its switching rules: the free turn first, then its recovery."""
        return self._free_turn(battle, moves) or super()._recover(battle, moves)

    def _free_turn(self, battle, moves: list[LegalAction]) -> LegalAction | None:
        if not any(free_kind(_move(a)) for a in moves):
            return None
        why = self._not_free(battle)
        steps = (("setup", self._setup), ("rocks", self._rocks), ("status", self._status))
        if why is None and self._ahead(battle):
            steps = (("rocks", self._rocks),) if int(battle.turn or 0) == FIRST_TURN else ()
            why = None if steps else "ahead in members"
        if why is not None:
            self._notes["free"] = f"not free: {why}"
            return None
        for rule, step in steps:
            chosen = step(battle, moves)
            if chosen is not None:
                self._notes["rule"] = f"free:{rule}"
                self._notes["free"] = _move(chosen).id  # type: ignore[union-attr]
                return chosen
        return None

    def _not_free(self, battle) -> str | None:
        """Why the foe could punish a non-attacking move this turn, or None."""
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        hit = self._hit(battle, theirs, mine)[0]
        if hit >= self.free.free_hit_max:
            return f"hit {hit:.2f}"
        if self._ko(battle, theirs, mine):
            return "they KO"
        stab = _types(theirs)
        se = next(
            (
                m.id
                for m in self._foe_attacks(battle, theirs)
                if m.type.name in stab and mine.damage_multiplier(m) > 1
            ),
            None,
        )
        if se:
            return f"SE STAB {se}"
        anti = sorted(self._foe_moves(battle, theirs) & ANTI_FREE)
        if anti:
            return f"foe has {anti[0]}"
        if self._ko(battle, mine, theirs):
            return "we KO"
        return None

    # Free moves -----------------------------------------------------------------------------------------
    def _setup(self, battle, moves: list[LegalAction]) -> LegalAction | None:
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        if mine.current_hp_fraction <= self.free.setup_min_hp or self._foe_ability(battle, theirs) == UNAWARE:
            return None
        ours = {m.category.name for m in mine.moves.values()}
        for action in moves:
            stats = raises(_move(action))
            if not stats or max(mine.boosts.get(s, 0) for s in stats) >= self.free.setup_max_stage:
                continue
            offense = {OFFENSE[s] for s in stats if s in OFFENSE}
            if offense and not offense & ours:
                continue  # nothing to use the boost with
            if stats == ["spe"] and threat.outspeeds(battle, mine, theirs, *self._known):
                continue
            return action
        return None

    def _rocks(self, battle, moves: list[LegalAction]) -> LegalAction | None:
        rock = _by_id(moves, ROCKS)
        theirs = battle.opponent_active_pokemon
        if rock is None or self._rocks_used or SideCondition.STEALTH_ROCK in battle.opponent_side_conditions:
            return None
        if _left(battle, battle.opponent_team) < PHAZE_TARGETS or self._bounces(battle, theirs):
            return None
        alive = [m for m in battle.opponent_team.values() if not m.fainted]
        if any(self._foe_moves(battle, m) & HAZARD_REMOVAL for m in alive):
            return None
        self._rocks_used = True
        return rock

    def _bounces(self, battle, mon) -> bool:
        """Magic Bounce assumed, or possible while the ability is unrevealed."""
        if self._foe_ability(battle, mon) == MAGIC_BOUNCE:
            return True
        return not known_ability(battle, mon) and MAGIC_BOUNCE in (mon.possible_abilities or [])

    def _status(self, battle, moves: list[LegalAction]) -> LegalAction | None:
        theirs = battle.opponent_active_pokemon
        key = theirs.species
        if key in self._statused or not self._statusable(battle, theirs):
            return None
        legal = {_move(a).id: a for a in moves}  # type: ignore[union-attr]
        for move_id, condition in STATUS_MOVES.items():
            if move_id in legal and self._lands(battle, theirs, _move(legal[move_id]), condition):  # type: ignore[arg-type]
                self._statused.add(key)
                return legal[move_id]
        return None

    def _statusable(self, battle, theirs) -> bool:
        """No status, Substitute, Safeguard or status-proof ability."""
        if theirs.status is not None or Effect.SUBSTITUTE in theirs.effects:
            return False
        if SideCondition.SAFEGUARD in battle.opponent_side_conditions:
            return False
        ability = self._foe_ability(battle, theirs)
        return ability not in STATUS_PROOF and not (
            _grounded(theirs, ability) and Field.MISTY_TERRAIN in battle.fields
        )

    def _lands(self, battle, theirs, move: Move, condition: str) -> bool:
        """Whether `move` inflicts `condition` on the foe and it is worth inflicting."""
        types, ability = _types(theirs), self._foe_ability(battle, theirs)
        if types & (TYPE_IMMUNE[condition] | MOVE_TYPE_IMMUNE.get(move.id, set())):
            return False
        if ability in ABILITY_IMMUNE[condition] or (
            "powder" in (move.flags or {}) and _powder_proof(theirs, ability)
        ):
            return False
        if condition == "slp":
            return not self._sleep_blocked(battle, theirs, ability) and not (
                move.id == "yawn" and Effect.YAWN in theirs.effects
            )
        if condition == "brn":
            attacks = self._foe_attacks(battle, theirs)
            physical = sum(1 for m in attacks if m.category.name == "PHYSICAL")
            return physical > len(attacks) - physical
        return True

    def _sleep_blocked(self, battle, theirs, ability: str) -> bool:
        """Sleep clause (a foe member already asleep) or Electric/Misty Terrain under a grounded foe."""
        if any(m.status == Status.SLP for m in battle.opponent_team.values()):
            return True
        return _grounded(theirs, ability) and Field.ELECTRIC_TERRAIN in battle.fields


def _grounded(mon, ability: str) -> bool:
    return "FLYING" not in _types(mon) and ability != "levitate"


def _powder_proof(mon, ability: str) -> bool:
    return "GRASS" in _types(mon) or ability in POWDER_PROOF or (mon.item or "") == "safetygoggles"


def build(config, catalog: Catalog, heuristic: HeuristicParams) -> FreeTurnPlayer:
    """The player with disciplined's and the free-turn thresholds from the agent config's paths."""
    policy = config.agent.policy
    rules_path, free_path = config.directory / policy.disciplined, config.directory / policy.free_turn
    rules = DisciplinedParams.load(rules_path) if rules_path.is_file() else DisciplinedParams()
    free = FreeTurnParams.load(free_path) if free_path.is_file() else FreeTurnParams()
    return FreeTurnPlayer(rules, free, catalog, heuristic)

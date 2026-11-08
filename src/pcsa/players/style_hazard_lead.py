"""The hazard-lead style player: free_turn, with a Stealth Rock lead on a six that fits the hazard_lead style.

Preview: when our six fits `hazard_lead` (config/players/styles.yaml) the setter leads and the two attackers with
the best summed heuristic matchup against their six come with it (a Choice item holder never is the setter);
no fit, and the player is free_turn exactly (note {"style": "no fit..."}). In battle, while the plan holds
(thresholds in config/players/style_hazard_lead.yaml):
1. Rocks lead (research R6, web A5): on turn 1 the active setter uses Stealth Rock even when the turn is not free,
   if they cannot KO it (max roll) and their best hit is below `rocks_max_hit` of its current HP (both waived for
   a Focus Sash / Sturdy setter at full HP); never with a Choice item (it would lock into Rocks); free_turn's
   Stealth Rock guards apply (once, not up, 2+ foes left, no Magic Bounce, no known hazard removal).
2. Setter after Rocks: threatened (disciplined's `threat_threshold` or a KO) and unable to KO, it switches to a
   brought attacker taking < `pivot_max_hit` of its HP (back-to-back rule applies). The setter never sets up;
   attackers keep free_turn's gate.
3. Forced pick with Rocks up: the member that KOs their active, then the one that KOs most of their incoming
   members once the Rocks chip (`rocks_chip`) is counted, then max_damage's best attack.
The sweeper denier comes first; everything else is free_turn's. Notes: {"rule": "hazard:rocks|hazard:to_attacker|
hazard:forced"}; {"hazard": "no rocks: <why>"} when the turn-1 Rocks was legal but not used.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml
from poke_env.battle import PokemonType, SideCondition
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.builds import Catalog
from pcsa.common.clock import Deadline
from pcsa.common.errors import ConfigError
from pcsa.players import styles
from pcsa.players.base import DecisionView, LegalAction, TeamPreviewView
from pcsa.players.disciplined import PHAZE_TARGETS, DisciplinedParams, _by_id, _left, _move, boosted, is_setup
from pcsa.players.free_turn import FIRST_TURN, HAZARD_REMOVAL, ROCKS, FreeTurnParams, FreeTurnPlayer
from pcsa.players.heuristic import HeuristicParams

STYLE = "hazard_lead"
ROCKS_SHARE = 1 / 8
CHIP_PROOF_ITEMS = frozenset({"heavydutyboots"})
CHIP_PROOF_ABILITIES = frozenset({"magicguard"})
CHOICE_ITEMS = frozenset({"choicescarf", "choiceband", "choicespecs"})  # Rocks would lock the setter into Rocks
SASH_ITEMS, SASH_ABILITIES = frozenset({"focussash"}), frozenset({"sturdy"})  # survive one hit from full HP
TEAM_SIZE = 3


@dataclass
class HazardLeadParams:
    rocks_max_hit: float = 0.75  # turn-1 Rocks only while their best hit is below this share of our current HP
    pivot_max_hit: float = 0.5  # after Rocks, a switch-in attacker takes less than this share of its HP

    @classmethod
    def load(cls, path: Path) -> HazardLeadParams:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(f"unknown style_hazard_lead parameters in {path}: {sorted(unknown)}")
        return cls(**data)


@dataclass(frozen=True)
class Plan:
    """The preview's plan for one battle: base species ids of the setter and the attackers brought."""

    tag: str
    setter: str
    attackers: frozenset[str]


def rocks_chip(types: Iterable[PokemonType | None]) -> float:
    """Stealth Rock's damage on switch-in, as a share of max HP, for a member of these types."""
    known = [t for t in types if t is not None]
    if not known:
        return ROCKS_SHARE
    second, chart = (known[1] if len(known) > 1 else None), GenData.from_gen(9).type_chart
    return ROCKS_SHARE * PokemonType.ROCK.damage_multiplier(known[0], second, type_chart=chart)


def _base(mon) -> str:
    return game.base_name(mon.species)


def _choice(mon) -> bool:
    """A Choice item: Rocks would lock the holder into Rocks."""
    return to_id_str(mon.item or "") in CHOICE_ITEMS


class StyleHazardLeadPlayer(FreeTurnPlayer):
    name = "style_hazard_lead"

    def __init__(
        self,
        rules: DisciplinedParams | None = None,
        free: FreeTurnParams | None = None,
        hazard: HazardLeadParams | None = None,
        catalog: Catalog | None = None,
        heuristic: HeuristicParams | None = None,
    ) -> None:
        self.hazard = hazard or HazardLeadParams()
        self.styles = styles.load()
        self._plan: Plan | None = None
        super().__init__(rules, free, catalog, heuristic)

    def params(self) -> dict:
        return {**super().params(), **asdict(self.hazard)}

    # Preview --------------------------------------------------------------------------------------------
    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        battle = view.battle
        six = list(battle.team.values()) if battle is not None else []
        fit = self.styles.fit_mons(STYLE, six) if six else None
        setter = next((i for i in fit.candidates["setter"] if not _choice(six[i])), None) if fit else None
        if fit is None or setter is None:
            self._plan = None
            picks = super().choose_team(view, deadline)
            self._notes["style"] = "no fit" if fit is None else "no fit: every setter holds a Choice item"
            return picks
        self.tracker.reset()
        self._reset(None)
        size = battle.max_team_size or TEAM_SIZE  # type: ignore[union-attr]
        opponents = battle.teampreview_opponent_team or list(battle.opponent_team.values())  # type: ignore[union-attr]
        score = {i: sum(self.heuristic.matchup(six[i], opp) for opp in opponents) for i in range(len(six))}
        ranked = sorted((i for i in range(len(six)) if i != setter), key=lambda i: -score[i])
        attackers = [i for i in ranked if i in fit.candidates["attacker"]] + [
            i for i in ranked if i not in fit.candidates["attacker"]
        ]  # members outside the role only fill a gap
        picks = [setter, *attackers[: size - 1]]
        names = [_base(six[i]) for i in picks]
        self._plan = Plan(battle.battle_tag, names[0], frozenset(names[1:]))  # type: ignore[union-attr]
        self._notes.update({"style": STYLE, "lead": names[0], "attackers": names[1:]})
        return picks

    def _plan_for(self, battle) -> Plan | None:
        plan = self._plan
        return plan if plan is not None and battle is not None and plan.tag == battle.battle_tag else None

    def _setter_active(self, battle) -> bool:
        plan, mine = self._plan_for(battle), battle.active_pokemon
        return plan is not None and mine is not None and _base(mine) == plan.setter

    # Rules ----------------------------------------------------------------------------------------------
    def _decide(self, view: DecisionView) -> LegalAction | None:
        plan = self._plan_for(view.battle)
        chosen = self._hazard_rules(view, plan) if plan is not None else None
        return chosen if chosen is not None else super()._decide(view)

    def _hazard_rules(self, view: DecisionView, plan: Plan) -> LegalAction | None:
        battle = view.battle
        theirs = battle.opponent_active_pokemon  # type: ignore[union-attr]
        moves = [a for a in view.legal if _move(a) is not None]
        switches = [a for a in view.legal if a.order is not None and _move(a) is None]
        if theirs is None:
            return None
        if not moves:
            return self._forced_pick(battle, switches)
        if not self._setter_active(battle) or boosted(theirs) or any(is_setup(m) for m in theirs.moves.values()):
            return None  # an attacker, or the sweeper denier's turn
        return self._rocks_lead(battle, moves) or self._to_attacker(battle, switches, plan)

    def _rocks_lead(self, battle, moves: list[LegalAction]) -> LegalAction | None:
        if int(battle.turn or 0) != FIRST_TURN or _by_id(moves, ROCKS) is None:
            return None
        why = self._unsafe(battle) or self._rocks_blocked(battle)
        chosen = None if why else self._rocks(battle, moves)
        if chosen is None:
            self._notes["hazard"] = f"no rocks: {why or 'blocked'}"
            return None
        self._notes["rule"] = "hazard:rocks"
        return chosen

    def _unsafe(self, battle) -> str | None:
        """Why the setter might not survive the turn (or would be locked into Rocks), or None."""
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        if _choice(mine):
            return f"choice item {mine.item}"
        sash = to_id_str(mine.item or "") in SASH_ITEMS or to_id_str(mine.ability or "") in SASH_ABILITIES
        if sash and mine.current_hp_fraction >= 1:
            return None  # one hit leaves it at 1 HP at worst
        if self._ko(battle, theirs, mine):
            return "they KO"
        hit = self._hit(battle, theirs, mine)[0]
        if hit >= self.hazard.rocks_max_hit * mine.current_hp_fraction:
            return f"hit {hit:.2f}"
        return None

    def _rocks_blocked(self, battle) -> str | None:
        """free_turn's Stealth Rock guards, as a reason."""
        if self._rocks_used or self._rocks_up(battle):
            return "already used"
        if _left(battle, battle.opponent_team) < PHAZE_TARGETS:
            return "last foe"
        if self._bounces(battle, battle.opponent_active_pokemon):
            return "Magic Bounce"
        alive = [m for m in battle.opponent_team.values() if not m.fainted]
        removal = sorted(set().union(*(self._foe_moves(battle, m) & HAZARD_REMOVAL for m in alive)))
        return f"foe has {removal[0]}" if removal else None

    def _to_attacker(self, battle, switches: list[LegalAction], plan: Plan) -> LegalAction | None:
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        if not switches or not self._rocks_up(battle) or self._ko(battle, mine, theirs):
            return None
        if self._hit(battle, theirs, mine)[0] < self.rules.threat_threshold and not self._ko(battle, theirs, mine):
            return None
        bench = [a for a in switches if _base(a.order.order) in plan.attackers]  # type: ignore[union-attr]
        taken = {id(a): self._taken(battle, theirs, a.order.order) for a in bench}  # type: ignore[union-attr]
        fits = [a for a in bench if taken[id(a)] < self.hazard.pivot_max_hit]
        if not fits:
            return None
        best = min(fits, key=lambda a: taken[id(a)])
        reason = f"rocks up: {mine.species} -> {best.order.order.species} takes {taken[id(best)]:.2f}"  # type: ignore[union-attr]
        chosen = self._take_switch(battle, best, reason)
        if chosen is None:
            self._notes["hazard"] = f"no pivot: {self._blocked(battle)}"
        else:
            self._notes["rule"] = "hazard:to_attacker"
        return chosen

    def _rocks(self, battle, moves: list[LegalAction]) -> LegalAction | None:
        """free_turn's Stealth Rock, never from a Choice item holder."""
        return None if _choice(battle.active_pokemon) else super()._rocks(battle, moves)

    def _setup(self, battle, moves: list[LegalAction]) -> LegalAction | None:
        """free_turn's setup, never on the setter."""
        return None if self._setter_active(battle) else super()._setup(battle, moves)

    # Rocks chip -----------------------------------------------------------------------------------------
    def _rocks_up(self, battle) -> bool:
        return SideCondition.STEALTH_ROCK in battle.opponent_side_conditions

    def _chip(self, battle, foe) -> float:
        """What Stealth Rock takes from `foe` when it comes in (0 without Rocks, with Boots or Magic Guard)."""
        if not self._rocks_up(battle) or (foe.item or "") in CHIP_PROOF_ITEMS:
            return 0.0
        return 0.0 if self._foe_ability(battle, foe) in CHIP_PROOF_ABILITIES else rocks_chip(foe.types)

    def _ko_after_rocks(self, battle, attacker, foe) -> bool:
        """Whether `attacker`'s best hit KOs `foe` as it comes in over the Rocks (unrevealed: full HP)."""
        hp = foe.current_hp_fraction or 1.0
        return self._hit(battle, attacker, foe)[0] + self._chip(battle, foe) >= hp

    def _incoming(self, battle) -> list:
        """Their members that may still come in: revealed on the bench, or preview members while some are unseen."""
        revealed = list(battle.opponent_team.values())
        bench = [m for m in revealed if not m.fainted and not m.active]
        if len(revealed) >= (battle.max_team_size or TEAM_SIZE):
            return bench
        seen = {_base(m) for m in revealed}
        return bench + [m for m in battle.teampreview_opponent_team if _base(m) not in seen]

    def _forced_pick(self, battle, switches: list[LegalAction]) -> LegalAction | None:
        theirs = battle.opponent_active_pokemon
        if not switches or not self._rocks_up(battle):
            return None
        incoming = self._incoming(battle)
        after = {
            id(a): [_base(f) for f in incoming if self._ko_after_rocks(battle, a.order.order, f)]  # type: ignore[union-attr]
            for a in switches
        }

        def score(a: LegalAction) -> tuple:
            mon = a.order.order  # type: ignore[union-attr]
            return self._ko(battle, mon, theirs), len(after[id(a)]), self._best_attack(battle, mon, theirs)

        best = max(switches, key=score)
        self._notes.update({"rule": "hazard:forced", "ko_after_rocks": after[id(best)]})
        return best


def build(config, catalog: Catalog, heuristic: HeuristicParams) -> StyleHazardLeadPlayer:
    """The player with the disciplined, free-turn and hazard-lead thresholds from the agent config's paths."""
    policy, directory = config.agent.policy, config.directory
    rules_path, free_path = directory / policy.disciplined, directory / policy.free_turn
    hazard_path = directory / policy.style_hazard_lead
    rules = DisciplinedParams.load(rules_path) if rules_path.is_file() else DisciplinedParams()
    free = FreeTurnParams.load(free_path) if free_path.is_file() else FreeTurnParams()
    hazard = HazardLeadParams.load(hazard_path) if hazard_path.is_file() else HazardLeadParams()
    return StyleHazardLeadPlayer(rules, free, hazard, catalog, heuristic)

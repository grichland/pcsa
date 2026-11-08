"""The setup-sweeper style player: free_turn with one designated sweeper kept back until its checks are gone.

Preview (config/players/styles.yaml `setup_sweeper`): no fit -> exactly free_turn (note {"style": "no fit"}).
With a fit the sweeper comes with the two supporters whose best hits on its checks sum highest (on their whole
six when none checks it); the supporter with the best average heuristic matchup against their six leads.
A check is an opposing member that (a) outspeeds the sweeper and has a super-effective attack on it, (b) resists
every attack of the sweeper, or (c) has Unaware or one of free_turn's ANTI_FREE moves (phaze, Haze, Clear Smog,
Encore, Taunt, Perish Song), revealed or assumed. In battle revealed members are judged on the spot; a preview
check not yet seen counts as alive while they have unrevealed members left.
Before disciplined's rules (research R3, R8, web A2/B3; thresholds in config/players/style_setup_sweeper.yaml):
1. hold: while a check lives the sweeper is never chosen as a switch-in, voluntary or forced (unless it is the
   only one left); shield: the sweeper in front, threatened and without a KO first, switches to the member
   taking least of its HP, below `shield_max_hit` (back-to-back rule respected).
2. enter: no check alive, a forced switch sends the sweeper; on a voluntary turn a supporter that cannot KO
   switches to it when it takes < `entry_max_hit` of its HP.
3. sweep: for the sweeper the free-turn gate opens below `sweeper_free_hit_max` and setup stops at
   `setup_max_stage`; supporters never set up while a check lives.
Notes: {"rule": "sweeper:shield|sweeper:enter|sweeper:hold"}; at preview {"style", "sweeper", "checks"}.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

from pcsa.battle import game
from pcsa.battle.builds import Catalog
from pcsa.common.clock import Deadline
from pcsa.common.errors import ConfigError
from pcsa.players import styles, threat
from pcsa.players.base import DecisionView, LegalAction, TeamPreviewView
from pcsa.players.disciplined import DisciplinedParams, _move
from pcsa.players.free_turn import ANTI_FREE, UNAWARE, FreeTurnParams, FreeTurnPlayer, raises
from pcsa.players.heuristic import HeuristicParams

STYLE = "setup_sweeper"
TEAM_SIZE = 3


@dataclass
class SweeperParams:
    shield_max_hit: float = 0.5  # a shield switch-in takes less than this share of its HP
    entry_max_hit: float = 0.4  # the sweeper comes in by choice only taking less than this share of its HP
    sweeper_free_hit_max: float = 0.55  # free_turn's free_hit_max, for the sweeper
    setup_max_stage: int = 2  # free_turn's setup_max_stage, for the sweeper

    @classmethod
    def load(cls, path: Path) -> SweeperParams:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(f"unknown style_setup_sweeper parameters in {path}: {sorted(unknown)}")
        return cls(**data)


@dataclass(frozen=True)
class Plan:
    sweeper: str  # base species id of our sweeper
    checks: tuple[str, ...]  # base species ids of the members of their six that checked it at preview


def base(mon) -> str:
    return game.base_name(mon.species)


def _target(action: LegalAction) -> Any:
    return action.order.order if action.order is not None else None


class StyleSetupSweeperPlayer(FreeTurnPlayer):
    name = "style_setup_sweeper"

    def __init__(
        self,
        rules: DisciplinedParams | None = None,
        free: FreeTurnParams | None = None,
        sweep: SweeperParams | None = None,
        catalog: Catalog | None = None,
        heuristic: HeuristicParams | None = None,
    ) -> None:
        self.sweep = sweep or SweeperParams()
        self.styles: styles.Styles | None = None  # loaded at the first preview
        self._plan: Plan | None = None
        self._alive: list[str] | None = None  # this decision's live checks; None without a plan
        super().__init__(rules, free, catalog, heuristic)

    def params(self) -> dict:
        return {**super().params(), "sweeper": asdict(self.sweep)}

    # Preview --------------------------------------------------------------------------------------------
    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        picks = super().choose_team(view, deadline)
        self._plan = None
        battle = view.battle
        if battle is None:
            return picks
        if self.styles is None:
            self.styles = styles.load()
        team = list(battle.team.values())
        fit = self.styles.fit_mons(STYLE, team)
        if fit is None:
            self._notes["style"] = "no fit"
            return picks
        sweeper = fit.roles["sweeper"][0]
        # detached copies: at preview poke-env keys opponent_team by bare species, which form lookups cannot split
        foes = [copy.copy(f) for f in battle.teampreview_opponent_team]
        checks = {base(f): why for f in foes if (why := self._check(battle, team[sweeper], f))}
        targets = [f for f in foes if base(f) in checks] or foes
        score = {i: sum(self._hit(battle, team[i], t)[0] for t in targets) for i in range(len(team)) if i != sweeper}
        support = sorted(score, key=lambda i: -score[i])[: (battle.max_team_size or TEAM_SIZE) - 1]
        lead = max(support, key=lambda i: sum(self.heuristic.matchup(team[i], f) for f in foes))
        self._plan = Plan(base(team[sweeper]), tuple(checks))
        self._notes.update(
            style=STYLE,
            sweeper=team[sweeper].species,
            checks=", ".join(f"{k} ({v})" for k, v in checks.items()) or "none",
        )
        return [lead, *(i for i in support if i != lead), sweeper]

    def _check(self, battle, sweeper, foe) -> str | None:
        """Why `foe` checks our sweeper, or None."""
        anti = sorted(self._foe_moves(battle, foe) & ANTI_FREE)
        if anti:
            return anti[0]
        if self._foe_ability(battle, foe) == UNAWARE:
            return UNAWARE
        attacks = [m for m in sweeper.moves.values() if m.category.name != "STATUS"]
        if attacks and all(foe.damage_multiplier(m) < 1 for m in attacks):
            return "resists"
        if threat.outspeeds(battle, foe, sweeper, *self._known) and any(
            sweeper.damage_multiplier(m) > 1 for m in self._foe_attacks(battle, foe)
        ):
            return "faster, super effective"
        return None

    # Battle state ---------------------------------------------------------------------------------------
    def _sweeper(self, battle):
        """Our sweeper in this battle, or None (no plan, or not on our team)."""
        if self._plan is None or battle is None:
            return None
        return next((m for m in battle.team.values() if base(m) == self._plan.sweeper), None)

    def _checks(self, battle, sweeper) -> list[str]:
        """Their checks still alive: revealed ones judged now, unseen preview ones while slots are unrevealed."""
        seen = {base(m): m for m in battle.opponent_team.values()}
        alive = [s for s, m in seen.items() if not m.fainted and self._check(battle, sweeper, m)]
        if (battle.max_team_size or TEAM_SIZE) > len(seen):
            alive += [s for s in self._plan.checks if s not in seen]  # type: ignore[union-attr]
        return alive

    def _on_field(self, battle) -> bool:
        mine = battle.active_pokemon
        return self._plan is not None and mine is not None and base(mine) == self._plan.sweeper

    # Rules ----------------------------------------------------------------------------------------------
    def _decide(self, view: DecisionView) -> LegalAction | None:
        battle = view.battle
        sweeper = self._sweeper(battle)
        if battle is None or sweeper is None or sweeper.fainted:
            self._alive = None
            return super()._decide(view)
        self._alive = self._checks(battle, sweeper)
        moves = [a for a in view.legal if _move(a) is not None]
        switches = [a for a in view.legal if a.order is not None and _move(a) is None]
        sends = next((a for a in switches if _target(a) is sweeper), None)
        if not moves or battle.active_pokemon is None or battle.opponent_active_pokemon is None:
            return self._forced(battle, switches, sends)
        chosen = self._shield(battle, moves, switches) if self._on_field(battle) else self._enter(battle, sends)
        if chosen is not None:
            return chosen
        if self._alive and sends is not None:
            view = replace(view, legal=tuple(a for a in view.legal if a is not sends))
        return super()._decide(view)

    def _forced(self, battle, switches: list[LegalAction], sends: LegalAction | None) -> LegalAction | None:
        """A forced pick: the sweeper once no check lives, any other member while one does."""
        others = [a for a in switches if a is not sends]
        if sends is None:
            self._notes["rule"] = "forced"
            return None  # max_damage's pick
        if not self._alive or not others:
            self._notes["rule"] = "forced" if self._alive else "sweeper:enter"
            return sends
        self._notes["rule"] = "sweeper:hold"
        self._notes["sweeper"] = f"held back: {', '.join(self._alive)} alive"
        theirs = battle.opponent_active_pokemon
        if theirs is None:
            return others[0]
        return max(others, key=lambda a: self._best_attack(battle, _target(a), theirs))

    def _shield(self, battle, moves: list[LegalAction], switches: list[LegalAction]) -> LegalAction | None:
        """The sweeper in front, threatened while a check lives: out to the member that takes least."""
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        if not self._alive or not switches:
            return None
        known = self._known
        if not threat.threatened(battle, mine, theirs, *known, threshold=self.rules.threat_threshold):
            return None
        if self._ko_now(battle, moves) is not None or threat.priority_ko(battle, mine, theirs, *known):
            return None
        taken = {id(a): self._taken(battle, theirs, _target(a)) for a in switches}
        best = min(switches, key=lambda a: taken[id(a)])
        if taken[id(best)] >= self.sweep.shield_max_hit:
            self._notes["sweeper"] = f"threatened, no shield: best takes {taken[id(best)]:.2f}"
            return None
        reason = f"shield {mine.species}: {_target(best).species} takes {taken[id(best)]:.2f}"
        chosen = self._take_switch(battle, best, reason)
        if chosen is not None:
            self._notes["rule"] = "sweeper:shield"
        return chosen

    def _enter(self, battle, sends: LegalAction | None) -> LegalAction | None:
        """No check alive: a supporter that cannot KO hands over to the sweeper when it comes in safely."""
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        if self._alive or sends is None or self._ko(battle, mine, theirs):
            return None
        taken = self._taken(battle, theirs, _target(sends))
        if taken >= self.sweep.entry_max_hit:
            self._notes["sweeper"] = f"no entry: takes {taken:.2f}"
            return None
        chosen = self._take_switch(battle, sends, f"no check alive: {_target(sends).species} takes {taken:.2f}")
        if chosen is not None:
            self._notes["rule"] = "sweeper:enter"
        return chosen

    # Free turn ------------------------------------------------------------------------------------------
    def _free_turn(self, battle, moves: list[LegalAction]) -> LegalAction | None:
        """The sweeper's gate is the relaxed one."""
        if self._alive is None or not self._on_field(battle):
            return super()._free_turn(battle, moves)
        usual = self.free
        self.free = replace(
            usual, free_hit_max=self.sweep.sweeper_free_hit_max, setup_max_stage=self.sweep.setup_max_stage
        )
        try:
            return super()._free_turn(battle, moves)
        finally:
            self.free = usual

    def _setup(self, battle, moves: list[LegalAction]) -> LegalAction | None:
        if self._alive and not self._on_field(battle):
            if any(raises(_move(a)) for a in moves):
                self._notes["sweeper"] = "supporter: no setup while a check lives"
            return None
        return super()._setup(battle, moves)


def build(config, catalog: Catalog, heuristic: HeuristicParams) -> StyleSetupSweeperPlayer:
    """The player with disciplined's, free_turn's and its own thresholds from the agent config's paths."""
    policy = config.agent.policy
    paths = (config.directory / p for p in (policy.disciplined, policy.free_turn, policy.style_setup_sweeper))
    rules_path, free_path, sweep_path = paths
    rules = DisciplinedParams.load(rules_path) if rules_path.is_file() else DisciplinedParams()
    free = FreeTurnParams.load(free_path) if free_path.is_file() else FreeTurnParams()
    sweep = SweeperParams.load(sweep_path) if sweep_path.is_file() else SweeperParams()
    return StyleSetupSweeperPlayer(rules, free, sweep, catalog, heuristic)

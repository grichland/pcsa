"""The best-matchup max-damage player: max_damage, plus a voluntary switch out of a really bad matchup.

Matchups are the heuristic's score (`HeuristicPlayer.matchup`, weights from config/players/heuristic.yaml).
When the active's score is below `switch_below` and the best legal bench member's is above `switch_in_above`
(config/players/best_matchup.yaml), that member comes in. Everything else is MaxDamagePlayer's.

Back-to-back voluntary switches: the second is allowed once, and only when the opponent also switched on the
turn of the first (a switch of theirs before any faint of theirs that turn); a third in a row never is.
Forced switches neither count nor are restricted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml
from poke_env.battle import Move

from pcsa.common.clock import Deadline
from pcsa.common.errors import ConfigError
from pcsa.players.base import DecisionView, LegalAction, TeamPreviewView
from pcsa.players.heuristic import HeuristicParams, HeuristicPlayer
from pcsa.players.max_damage import MaxDamagePlayer

MAX_STREAK = 2  # voluntary switches in a row


@dataclass
class SwitchParams:
    switch_below: float = -2.0  # switch out when the active's matchup is below this...
    switch_in_above: float = 1.0  # ...and the best bench member's is above this

    @classmethod
    def load(cls, path: Path) -> SwitchParams:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(f"unknown best_matchup parameters in {path}: {sorted(unknown)}")
        return cls(**data)


def opponent_switched(battle, turn: int) -> bool:
    """Whether the opponent switched on `turn` by choice: a switch of theirs before any faint of theirs."""
    them = "p1" if battle.player_role == "p2" else "p2"
    inside = False
    for parts in battle._replay_data:
        kind, subject = (*parts[1:3], "", "")[:2]
        if kind == "turn":
            if inside:
                return False
            inside = subject == str(turn)
        elif inside and subject.startswith(them):
            if kind == "faint":
                return False
            if kind == "switch":
                return True
    return False


class BestMatchupMaxDamagePlayer(MaxDamagePlayer):
    name = "best_matchup_max_damage"

    def __init__(self, heuristic: HeuristicParams | None = None, switch: SwitchParams | None = None) -> None:
        self.heuristic = HeuristicPlayer(heuristic or HeuristicParams())
        self.switch = switch or SwitchParams()
        super().__init__()
        self._reset(None)

    def params(self) -> dict:
        return {**super().params(), **asdict(self.switch), "heuristic": asdict(self.heuristic.p)}

    def _reset(self, tag: str | None) -> None:
        self._tag, self._turn, self._last_switch, self._streak = tag, -1, None, 0

    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        self._reset(None)
        return super().choose_team(view, deadline)

    def choose_action(self, view: DecisionView, deadline: Deadline) -> LegalAction:
        battle = view.battle
        if battle is None or not view.legal:
            return super().choose_action(view, deadline)
        turn = int(battle.turn or 0)
        if battle.battle_tag != self._tag or turn < self._turn:
            self._reset(battle.battle_tag)
        self._turn = turn
        chosen = self._voluntary_switch(view)
        return chosen if chosen is not None else super().choose_action(view, deadline)

    def _voluntary_switch(self, view: DecisionView) -> LegalAction | None:
        """The bench member to bring in out of a really bad matchup, when the back-to-back rule allows it."""
        battle = view.battle
        active, opponent = battle.active_pokemon, battle.opponent_active_pokemon  # type: ignore[union-attr]
        moves = [a for a in view.legal if a.order is not None and isinstance(a.order.order, Move)]
        switches = [a for a in view.legal if a.order is not None and not isinstance(a.order.order, Move)]
        if not moves or not switches or active is None or opponent is None:
            return None
        here = self.heuristic.matchup(active, opponent, battle)
        if here >= self.switch.switch_below:
            return None
        score = {id(a): self.heuristic.matchup(a.order.order, opponent, battle) for a in switches}  # type: ignore[union-attr]
        best = max(switches, key=lambda a: score[id(a)])
        if score[id(best)] <= self.switch.switch_in_above:
            return None
        reason = f"bad matchup: {active.species} {here:.2f} -> {best.order.order.species} {score[id(best)]:.2f}"  # type: ignore[union-attr]
        blocked = self._blocked(battle)
        if blocked:
            self._notes["switch_blocked"] = f"{reason}; {blocked}"
            return None
        back_to_back = self._last_switch == self._turn - 1
        self._streak = self._streak + 1 if back_to_back else 1
        self._last_switch = self._turn
        self._notes["switch"] = reason + (" (back-to-back: they switched too)" if back_to_back else "")
        return best

    def _blocked(self, battle) -> str | None:
        """Why the back-to-back rule forbids a voluntary switch this turn, or None."""
        if self._last_switch != self._turn - 1:
            return None
        if self._streak >= MAX_STREAK:
            return "already switched twice in a row"
        if not opponent_switched(battle, self._turn - 1):
            return "switched last turn and the opponent did not"
        return None

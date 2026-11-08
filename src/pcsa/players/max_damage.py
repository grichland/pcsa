"""The max-damage player: the attack the heuristic's damage estimate scores highest, nothing else.

No voluntary switch, no status move, no setup. Mega Evolution when it is offered. A forced switch brings
in the bench member whose best attack scores highest against the opposing active. At team preview the
three with the highest summed best-attack scores against their six are brought.
"""

from __future__ import annotations

from poke_env.battle import Move

from pcsa.common.clock import Deadline
from pcsa.players.base import DecisionView, LegalAction, TeamPreviewView, random_team
from pcsa.players.tactics import Scoring, attack_score, mega_form_of

STAB = 1.5


class MaxDamagePlayer:
    name = "max_damage"

    def __init__(self, stab: float = STAB) -> None:
        self.stab = stab
        self._notes: dict = {}

    def params(self) -> dict:
        return {"stab": self.stab}

    def notes(self) -> dict:
        notes, self._notes = self._notes, {}
        return notes

    def _best_attack(self, battle, attacker, defender) -> float:
        attacks = [m for m in attacker.moves.values() if m.category.name != "STATUS"]
        if not attacks or defender is None:
            return 0.0
        return max(attack_score(battle, attacker, defender, m, Scoring(self.stab)) for m in attacks)

    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        battle = view.battle
        if battle is None:
            return random_team(view.rng, len(view.roster) or 6)
        team = list(battle.team.values())
        opponents = battle.teampreview_opponent_team or list(battle.opponent_team.values())
        score = [sum(self._best_attack(battle, mon, opp) for opp in opponents) for mon in team]
        return sorted(range(len(team)), key=lambda i: -score[i])[: battle.max_team_size or 3]

    def choose_action(self, view: DecisionView, deadline: Deadline) -> LegalAction:
        if not view.legal:
            raise ValueError("no legal action")
        battle = view.battle
        if battle is None:
            return self._from_screen(view)
        active, opponent = battle.active_pokemon, battle.opponent_active_pokemon
        moves = [a for a in view.legal if a.order is not None and isinstance(a.order.order, Move)]
        if moves and active is not None and opponent is not None:

            def score(a: LegalAction) -> float:
                form = mega_form_of(active) if a.order is not None and a.order.mega else None
                return attack_score(battle, active, opponent, a.order.order, Scoring(self.stab, True, form))  # type: ignore[union-attr]

            return max(moves, key=score)
        switches = [a for a in view.legal if a.order is not None and not isinstance(a.order.order, Move)]
        if switches and opponent is not None:
            self._notes["switch"] = "forced: the bench member with the strongest attack"
            return max(switches, key=lambda a: self._best_attack(battle, a.order.order, opponent))  # type: ignore[union-attr]
        return view.legal[0]

    def _from_screen(self, view: DecisionView) -> LegalAction:
        """No Battle: a move with PP over a switch, the most PP first; forced: the first switch."""
        if view.screen.forced:
            switches = [a for a in view.legal if a.action.kind == "switch"]
            return switches[0] if switches else view.legal[0]
        moves = [a for a in view.legal if a.action.kind == "move" and not a.action.mega]
        if moves:
            pp = {m.slot: (m.pp[0] if m.pp else 0) for m in view.screen.moves}
            self._notes["fallback"] = "screen-level decision: move with the most PP"
            return max(moves, key=lambda a: pp.get(a.action.move or 0, 0))
        return view.legal[0]

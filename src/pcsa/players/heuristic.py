"""The heuristic player: matchup scores, switch rules, hazards, setup, then the best-scoring attack.

Every number is a parameter (config/players/heuristic.yaml). The player works on a poke-env Battle
and the legal actions the loop hands it; with no Battle it decides from the screen alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml
from poke_env.battle import Target
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.player.battle_order import SingleBattleOrder

from pcsa.battle.stats import effective_speed
from pcsa.common.clock import Deadline
from pcsa.common.errors import ConfigError
from pcsa.players.base import DecisionView, LegalAction, TeamPreviewView, random_team
from pcsa.players.tactics import Scoring, attack_score, entry_hazard_fraction, mega_form_of

MAX_BOOST = 6


@dataclass
class HeuristicParams:
    speed_tier: float = 0.1
    hp_fraction: float = 0.4
    switch_threshold: float = -2.0
    switch_in_min: float = 0.0
    drop_threshold: int = -3
    opponent_team_size: int = 3
    hazard_min_opponents: int = 2
    setup_min_boost: int = 2
    setup_requires_full_hp: bool = True
    setup_max_boost: int = MAX_BOOST  # stop boosting a stat once it reaches this stage
    stab: float = 1.5
    exact_own_stats: bool = True
    mega: str = "first"  # first | matchup | last | never

    @classmethod
    def load(cls, path: Path) -> HeuristicParams:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(f"unknown heuristic parameters in {path}: {sorted(unknown)}")
        return cls(**data)


class HeuristicPlayer:
    name = "heuristic"

    def __init__(self, params: HeuristicParams | None = None, name: str | None = None) -> None:
        self.p = params or HeuristicParams()
        self.name = name or "heuristic"  # simple_heuristic (code defaults) | custom_heuristic (the tuned yaml)
        self._notes: dict = {}

    def params(self) -> dict:
        return asdict(self.p)

    def notes(self) -> dict:
        notes, self._notes = self._notes, {}
        return notes

    # Scoring ----------------------------------------------------------------------------------------
    def matchup(self, mon, opponent, battle=None) -> float:
        p = self.p
        score = max(opponent.damage_multiplier(t) for t in mon.types if t is not None)
        score -= max(mon.damage_multiplier(t) for t in opponent.types if t is not None)
        if mon.base_stats["spe"] != opponent.base_stats["spe"]:
            score += p.speed_tier if mon.base_stats["spe"] > opponent.base_stats["spe"] else -p.speed_tier
        score += p.hp_fraction * (mon.current_hp_fraction - opponent.current_hp_fraction)
        if battle is None:
            return score
        ours, theirs = effective_speed(mon), effective_speed(opponent, False)
        if (ours > theirs) != (mon.base_stats["spe"] > opponent.base_stats["spe"]):
            score += p.speed_tier * (1 if ours > theirs else -1)
        attacks = [m for m in mon.moves.values() if m.category.name != "STATUS"]
        if attacks:
            score += max(attack_score(battle, mon, opponent, m, Scoring(p.stab)) for m in attacks) / 200
        incoming = [m for m in opponent.moves.values() if m.category.name != "STATUS"]
        if incoming:
            score -= max(attack_score(battle, opponent, mon, m, Scoring(p.stab, exact=False)) for m in incoming) / 200
        hazards = entry_hazard_fraction(battle, mon)
        score -= hazards * 4
        if hazards >= mon.current_hp_fraction:
            score -= 100
        return score

    def should_switch(self, battle) -> bool:
        p, active, opponent = self.p, battle.active_pokemon, battle.opponent_active_pokemon
        if not any(self.matchup(m, opponent, battle) > p.switch_in_min for m in battle.available_switches):
            return False
        b = active.boosts
        if b["def"] <= p.drop_threshold or b["spd"] <= p.drop_threshold:
            return True
        stats = active.stats or {}
        attack = "atk" if (stats.get("atk") or 0) >= (stats.get("spa") or 0) else "spa"  # an unread stat counts as 0
        return b[attack] <= p.drop_threshold or self.matchup(active, opponent, battle) < p.switch_threshold

    def want_mega(self, battle, own_left: int) -> bool:
        mode = self.p.mega
        if not battle.can_mega_evolve or mode == "never":
            return False
        if mode == "matchup":
            return self.matchup(battle.active_pokemon, battle.opponent_active_pokemon) > 0 or own_left == 1
        return mode == "first" or own_left == 1

    def best_switch(self, battle):
        opponent = battle.opponent_active_pokemon
        switches = battle.available_switches
        return SingleBattleOrder(max(switches, key=lambda s: self.matchup(s, opponent, battle))) if switches else None

    def preferred(self, battle) -> SingleBattleOrder | None:
        """The order the heuristic would give with no phone constraints."""
        p, active, opponent = self.p, battle.active_pokemon, battle.opponent_active_pokemon
        if opponent is None:
            return None
        if active is None or not battle.available_moves:
            return self.best_switch(battle)
        if battle.available_switches and self.should_switch(battle):
            return self.best_switch(battle)
        own_left = sum(1 for m in battle.team.values() if not m.fainted)
        their_left = p.opponent_team_size - sum(1 for m in battle.opponent_team.values() if m.fainted)
        for move in battle.available_moves:
            hazard = SimpleHeuristicsPlayer.ENTRY_HAZARDS.get(move.id)
            if (
                their_left >= p.hazard_min_opponents
                and hazard is not None
                and hazard not in battle.opponent_side_conditions
            ):
                return SingleBattleOrder(move)
            if battle.side_conditions and move.id in SimpleHeuristicsPlayer.ANTI_HAZARDS_MOVES and own_left >= 2:
                return SingleBattleOrder(move)
        if (active.current_hp_fraction == 1 or not p.setup_requires_full_hp) and self.matchup(active, opponent) > 0:
            for move in battle.available_moves:
                if (
                    move.boosts
                    and sum(move.boosts.values()) >= p.setup_min_boost
                    and move.target == Target.SELF  # poke-env gives an enum; the string compare never matched
                    and min(active.boosts[s] for s, v in move.boosts.items() if v > 0) < p.setup_max_boost
                ):
                    return SingleBattleOrder(move)
        mega = self.want_mega(battle, own_left)
        scoring = Scoring(p.stab, p.exact_own_stats, mega_form_of(active) if mega else None)
        move = max(battle.available_moves, key=lambda m: attack_score(battle, active, opponent, m, scoring))
        return SingleBattleOrder(move, mega=mega)

    # Player interface ----------------------------------------------------------------------------------
    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        battle = view.battle
        if battle is None:
            return random_team(view.rng, len(view.roster) or 6)
        team = list(battle.team.values())
        opponents = battle.teampreview_opponent_team or list(battle.opponent_team.values())
        score = [sum(self.matchup(mon, opp) for opp in opponents) for mon in team]
        return sorted(range(len(team)), key=lambda i: -score[i])[: battle.max_team_size or 3]

    def choose_action(self, view: DecisionView, deadline: Deadline) -> LegalAction:
        if not view.legal:
            raise ValueError("no legal action")
        battle = view.battle
        if battle is None:
            return self.from_screen(view)
        by_message = {a.message: a for a in view.legal}
        if getattr(battle, "force_switch", False):
            order = self.best_switch(battle)
            return by_message.get(order.message, view.legal[0]) if order else view.legal[0]
        order = self.preferred(battle)
        if order is not None and order.message in by_message:
            return by_message[order.message]
        if order is not None and order.mega:  # Mega not offered: the same move without it
            plain = SingleBattleOrder(order.order).message
            if plain in by_message:
                return by_message[plain]
        active, opponent = battle.active_pokemon, battle.opponent_active_pokemon
        moves = [a for a in view.legal if a.order is not None and hasattr(a.order.order, "base_power")]
        if moves and active is not None and opponent is not None:
            self._notes["fallback"] = (
                f"preferred order {order.message if order else None!r} unavailable; best legal attack"
            )
            scoring = Scoring(self.p.stab, self.p.exact_own_stats)
            scored = [
                (attack_score(battle, active, opponent, a.order.order, scoring), i)
                for i, a in enumerate(moves)
                if a.order is not None
            ]
            return moves[max(scored)[1]]
        self._notes["fallback"] = "preferred order unavailable; first legal action"
        return view.legal[0]

    def from_screen(self, view: DecisionView) -> LegalAction:
        """No Battle: prefer a move with PP over a switch, highest remaining PP first; forced: first switch."""
        if view.screen.forced:
            switches = [a for a in view.legal if a.action.kind == "switch"]
            return switches[0] if switches else view.legal[0]
        moves = [a for a in view.legal if a.action.kind == "move" and not a.action.mega]
        if moves:
            pp = {m.slot: (m.pp[0] if m.pp else 0) for m in view.screen.moves}
            self._notes["fallback"] = "screen-level decision: move with the most PP"
            return max(moves, key=lambda a: pp.get(a.action.move or 0, 0))
        return view.legal[0]

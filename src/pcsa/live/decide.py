"""The degrade ladder: build the legal actions at the best rung the readings support, then ask the player.

full          tracked Battle, complete log
tracked       tracked Battle with gaps, unmapped rows or warnings
screen        no Battle: the screen's moves and the bookkeeping's switches
legal_random  nothing readable: every move row and every live switch
emergency     the deadline is nearly gone: the cheapest plan
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from pcsa.common import log
from pcsa.common.clock import Deadline
from pcsa.common.config import DegradeConfig
from pcsa.live import orders
from pcsa.players.base import DecisionView, Degrade, LegalAction, Player, ScreenFacts
from pcsa.players.heuristic import HeuristicPlayer
from pcsa.run.battle_state import BattleRecord


@dataclass(frozen=True)
class Decision:
    chosen: LegalAction
    legal: tuple[LegalAction, ...]
    degrade: Degrade
    reason: str
    dropped: list[dict]
    notes: dict


class Decider:
    def __init__(self, player: Player, config: DegradeConfig, rng: random.Random) -> None:
        self.player = player
        self.config = config
        self.rng = rng
        self.fallback = HeuristicPlayer()

    def decide(
        self,
        b: BattleRecord,
        screen: ScreenFacts,
        battle,
        out,
        deadline: Deadline,
        *,
        slot_moves: list[str | None],
        active: int | None,
        leaving: int | None = None,
    ) -> Decision:
        legal, degrade, reason, dropped = self.legal_actions(
            b, screen, battle, out, active=active, leaving=leaving, slot_moves=slot_moves
        )
        if deadline.remaining() < self.config.emergency_reserve_s:
            degrade, reason = Degrade.EMERGENCY, f"{deadline.remaining():.1f}s left"
            return Decision(legal[0], tuple(legal), degrade, reason, dropped, {})
        view = DecisionView(
            tuple(legal),
            screen,
            degrade,
            self.rng,
            battle if degrade in (Degrade.FULL, Degrade.TRACKED) else None,
            b.names,
        )
        try:
            chosen = self.player.choose_action(view, deadline)
            notes = self.player.notes()
            if chosen not in legal:
                raise ValueError(f"player returned an action outside the legal list: {chosen.message}")
        except Exception as error:
            log.exception("player_failed", battle=b.index, degrade=str(degrade))
            reason = f"{reason}; player failed: {type(error).__name__}"
            degrade = Degrade.SCREEN if degrade in (Degrade.FULL, Degrade.TRACKED) else degrade
            fallback_view = DecisionView(tuple(legal), screen, degrade, self.rng, None, b.names)
            chosen = self.fallback.choose_action(fallback_view, deadline)
            notes = {"fallback": "heuristic after player failure"}
        b.degrade_counts[str(degrade)] = b.degrade_counts.get(str(degrade), 0) + 1
        return Decision(chosen, tuple(legal), degrade, reason, dropped, notes)

    def legal_actions(
        self,
        b: BattleRecord,
        screen: ScreenFacts,
        battle,
        out,
        *,
        active: int | None,
        leaving: int | None,
        slot_moves: list[str | None],
    ) -> tuple[list[LegalAction], Degrade, str, list[dict]]:
        fainted = set(b.fainted)
        names = [n or "" for n in b.names]
        if battle is not None:
            kept, dropped = orders.battle_actions(
                battle, screen, slot_moves=slot_moves, names=names, fainted=fainted, active=active, leaving=leaving
            )
            if kept:
                shaky = bool(out is not None and (out.unmapped or any("rejected" in w for w in out.warnings)))
                degrade = Degrade.TRACKED if shaky or b.log.failures else Degrade.FULL
                return kept, degrade, "tracked battle", dropped
            reason = "every tracked order failed the screen check"
        else:
            dropped, reason = [], "no tracked battle"
        from_screen = orders.screen_actions(
            screen,
            fainted=fainted,
            active=active,
            leaving=leaving,
            mega_used=b.mega_used,
            team_size=len(b.requested or [0, 1, 2]),
        )
        if from_screen:
            return from_screen, Degrade.SCREEN, reason, dropped
        blind = orders.blind_actions(screen, fainted=fainted, active=active, team_size=len(b.requested or [0, 1, 2]))
        return (
            blind or [LegalAction(orders.SemanticAction("move", move=0), None)],
            Degrade.LEGAL_RANDOM,
            reason + "; nothing readable",
            dropped,
        )

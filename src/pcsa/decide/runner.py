"""Ask the Player, check its answer against the Offer, and fall back so that there is always an Order.

Fallback ladder: automatic turn -> nothing asked; the Player's answer if valid and in time (a Mega Evolution the
screen does not offer is dropped, the move kept); else the built-in "first"; else the first of `screen_actions`;
else the first of `blind_actions`. The Player is called on one worker thread until the request's deadline minus
the tap reserve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pcsa.common import log
from pcsa.common.clock import Clock, Deadline
from pcsa.common.config import DegradeConfig
from pcsa.decide.api import (
    Answer,
    Offer,
    Order,
    OwnTeam,
    Player,
    Request,
    TeamPreviewRequest,
    valid_picks,
)
from pcsa.decide.builtin import FIRST_PICKS, first_order
from pcsa.decide.offer_check import check
from pcsa.decide.order_text import order_text
from pcsa.decide.worker import Result, Worker
from pcsa.live.orders import blind_actions, screen_actions

Source = Literal["player", "builtin", "screen", "blind", "automatic"]


@dataclass(frozen=True)
class Decision:
    request_id: int
    order: Order
    text: str  # the record's "/choose ..." string
    source: Source
    reason: str
    rejected: tuple[dict, ...] = ()  # {order, reason}: the record's 'dropped'
    notes: dict = field(default_factory=dict)
    player_ms: float | None = None
    late: tuple[dict, ...] = ()  # answers that came back after their wait ended, dropped


@dataclass(frozen=True)
class PreviewDecision:
    picks: tuple[int, int, int]
    source: Literal["player", "builtin"]
    reason: str
    notes: dict = field(default_factory=dict)
    player_ms: float | None = None
    late: tuple[dict, ...] = ()


class Runner:
    def __init__(self, player: Player, clock: Clock, config: DegradeConfig) -> None:
        self.player = player
        self.clock = clock
        self.config = config
        self.worker = Worker(clock)

    def battle_start(self, battle_tag: str, team: OwnTeam) -> None:
        self.worker.submit("battle_start", lambda: self.player.battle_start(battle_tag, team), None)

    def battle_end(self, battle_tag: str, outcome: str, events: tuple = (), history: tuple = ()) -> None:
        end = self.player.battle_end
        self.worker.submit("battle_end", lambda: end(battle_tag, outcome, events=events, history=history), None)

    def close(self) -> None:
        self.worker.close()

    def decide(self, request: Request) -> Decision:
        late = self._late()
        offer = request.offer
        if offer.automatic:
            order = Order.move(0)
            return Decision(
                request.request_id, order, order_text(order, offer), "automatic", "automatic turn", late=late
            )
        rejected: list[dict] = []
        until = Deadline(self.clock, request.deadline).minus(self.config.tap_reserve_s)
        result, reason = self._ask("choose_move", lambda: self.player.choose_move(request), until, request.request_id)
        if reason is None and result is not None:
            answer: Answer = _without_mega(result.value, offer, request.request_id)
            reason = self._answer_problem(answer, offer, rejected)
            if reason is None:
                return Decision(
                    request.request_id,
                    answer.order,
                    order_text(answer.order, offer),
                    "player",
                    "player answer",
                    tuple(rejected),
                    dict(answer.notes),
                    result.ms,
                    late,
                )
        else:
            rejected.append({"order": _answer_text(result, offer), "reason": reason})
        player_ms = result.ms if result is not None else None
        return self._fallback(request, reason or "", rejected, player_ms, late)

    def teampreview(self, request: TeamPreviewRequest) -> PreviewDecision:
        late = self._late()
        until = Deadline(self.clock, request.deadline).minus(self.config.tap_reserve_s)
        result, reason = self._ask("teampreview", lambda: self.player.teampreview(request), until, None)
        if reason is None and result is not None:
            picks, notes, reason = _preview_answer(result.value)
            if reason is None:
                return PreviewDecision(picks, "player", "player answer", notes, result.ms, late)
        log.warning("player_overruled", battle=request.battle_tag, call="teampreview", reason=reason)
        ms = result.ms if result is not None else None
        return PreviewDecision(FIRST_PICKS, "builtin", reason or "", {"fallback": reason}, ms, late)

    def _ask(self, what: str, call, until: Deadline, request_id: int | None) -> tuple[Result | None, str | None]:
        """The Player's result and, when it cannot be used as it is, why not (late results are returned too)."""
        if until.expired():
            return None, f"no time left for the Player ({until.remaining():.1f}s)"
        budget = until.remaining()
        result = self.worker.wait(self.worker.submit(what, call, until, request_id), until)
        if result is None:
            return None, f"no answer within {budget:.1f}s"
        if result.skipped:
            return None, "the Player was still busy with an earlier call"
        if result.error is not None:
            log.warning("player_failed", call=what, request=request_id, error=repr(result.error))
            return None, f"Player raised {type(result.error).__name__}: {result.error}"
        if result.done_at is not None and result.done_at > until.at:
            return result, f"answer late by {result.done_at - until.at:.2f}s"
        return result, None

    def _answer_problem(self, answer, offer: Offer, rejected: list[dict]) -> str | None:
        if (
            not isinstance(answer, Answer)
            or not isinstance(answer.order, Order)
            or not isinstance(answer.notes, dict)
        ):
            reason = f"wrong answer type {type(answer).__name__}"
            rejected.append({"order": repr(answer)[:120], "reason": reason})
            return reason
        reason = check(answer.order, offer)
        if reason is not None:
            rejected.append({"order": order_text(answer.order, offer), "reason": reason})
        return reason

    def _fallback(self, request: Request, reason: str, rejected: list[dict], player_ms, late) -> Decision:
        offer = request.offer
        log.warning("player_overruled", battle=request.battle_tag, request=request.request_id, reason=reason)
        notes: dict = {"fallback": reason}
        try:
            order, notes["builtin"] = first_order(request)
            problem = check(order, offer)
        except ValueError as error:
            order, problem = None, str(error)
        if order is not None and problem is None:
            return Decision(
                request.request_id,
                order,
                order_text(order, offer),
                "builtin",
                reason,
                tuple(rejected),
                notes,
                player_ms,
                late,
            )
        rejected.append({"order": order_text(order, offer) if order else None, "reason": f"builtin: {problem}"})
        screen = screen_actions(offer, mega_used=request.hints.mega_used)
        source: Source = "screen" if screen else "blind"
        order = (screen or blind_actions(offer) or [Order.move(0)])[0]
        return Decision(
            request.request_id,
            order,
            order_text(order, offer),
            source,
            reason,
            tuple(rejected),
            notes,
            player_ms,
            late,
        )

    def _late(self) -> tuple[dict, ...]:
        late = []
        for result in self.worker.drain_late():
            entry = {"call": result.what, "request_id": result.request_id, "player_ms": result.ms}
            value = result.value
            if isinstance(value, Answer) and isinstance(value.order, Order):
                entry["order"] = value.order.label()
            entry["outcome"] = "skipped" if result.skipped else "error" if result.error else "dropped"
            log.event("player_late", **entry)
            late.append(entry)
        return tuple(late)


def _without_mega(answer, offer: Offer, request_id: int):
    """A move asked with Mega Evolution the screen does not offer is played without it: the move is the choice."""
    order = getattr(answer, "order", None)
    if not (isinstance(answer, Answer) and isinstance(order, Order) and order.kind == "move" and order.mega):
        return answer
    if offer.can_mega_evolve is True or not isinstance(answer.notes, dict):
        return answer
    log.event("player_repaired", request=request_id, repaired="mega dropped: not offered on screen")
    notes = {**answer.notes, "repaired": "mega dropped: not offered on screen"}
    return Answer(Order.move(order.row, mega=False), notes)  # type: ignore[arg-type]


def _answer_text(result: Result | None, offer: Offer) -> str | None:
    value = result.value if result is not None else None
    if isinstance(value, Answer) and isinstance(value.order, Order):
        return order_text(value.order, offer)
    return None


def _preview_answer(value) -> tuple[tuple[int, int, int], dict, str | None]:
    try:
        picks, notes = value
        picks = tuple(int(i) for i in picks)
    except (TypeError, ValueError):
        return FIRST_PICKS, {}, f"wrong answer type {type(value).__name__}"
    if not valid_picks(picks):
        return FIRST_PICKS, {}, f"invalid picks {picks}"
    return picks, dict(notes) if isinstance(notes, dict) else {"notes": repr(notes)}, None  # type: ignore[return-value]

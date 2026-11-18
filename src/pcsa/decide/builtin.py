"""The built-in Player "first": needs no game knowledge, always answers the same way for the same Offer."""

from __future__ import annotations

from pcsa.decide.api import Answer, Order, OwnTeam, Request, TeamPreviewRequest
from pcsa.decide.order_text import usable

FIRST_PICKS = (0, 1, 2)


class FirstPlayer:
    name = "first"

    def battle_start(self, battle_tag: str, team: OwnTeam) -> None:
        return None

    def teampreview(self, request: TeamPreviewRequest) -> tuple[tuple[int, int, int], dict]:
        return FIRST_PICKS, {"reason": "first three"}

    def choose_move(self, request: Request) -> Answer:
        return Answer(*first_order(request))

    def battle_end(self, battle_tag: str, outcome: str, events: tuple = (), history: tuple = ()) -> None:
        return None


def first_order(request: Request) -> tuple[Order, dict]:
    """The first move row not shown unavailable (row 0 when unread), else the first switch."""
    offer = request.offer
    if not offer.forced:
        if not offer.moves_read:
            return Order.move(0), {"reason": "move list not read: row 0"}
        row = next((m.row for m in offer.moves if usable(m)), None)
        if row is not None:
            return Order.move(row), {"reason": f"first usable row {row}"}
    name = next((s.name for s in offer.switches if s.name != offer.leaving), None)
    if name is None:
        raise ValueError("nothing on offer")
    return Order.switch(name), {"reason": "first switch" + (" (forced)" if offer.forced else "")}

"""The '/choose' strings of the decision record, written exactly as pcsa writes them (poke-env's order message).

A move is "/choose move <id>", with " mega" appended for Mega Evolution; recharge is "/choose move 1". A switch is
"/choose switch <Name>". A move row with no id at all is written as its label ("move2"), as pcsa writes an order
that has no Battle behind it. The legal list is switches first (Offer order), then usable moves (row order), then
the same moves with Mega when the button is shown.
"""

from __future__ import annotations

from pcsa.decide.api import MoveOffer, Offer, Order


def usable(move: MoveOffer) -> bool:
    """Not shown disabled and not shown at 0 PP (unread fields count as usable)."""
    return not move.disabled and (move.pp is None or move.pp[0] > 0)


def move_id(order: Order, offer: Offer) -> str | None:
    return next((m.id for m in offer.moves if m.row == order.row), None)


def order_text(order: Order, offer: Offer) -> str:
    if order.kind == "switch":
        return f"/choose switch {order.name}"
    move = move_id(order, offer)
    if move is None:
        return order.label()
    if move == "recharge":
        return "/choose move 1"
    return f"/choose move {move}{' mega' if order.mega else ''}"


def legal_orders(offer: Offer) -> list[Order]:
    """Everything the screen offers, in pcsa's legal-list order."""
    orders = [Order.switch(s.name) for s in offer.switches if s.name != offer.leaving]
    if offer.forced:
        return orders
    rows = [m for m in offer.moves if usable(m)]
    orders += [Order.move(m.row) for m in rows]
    if offer.can_mega_evolve:
        orders += [Order.move(m.row, mega=True) for m in rows]
    return orders


def legal_text(offer: Offer) -> list[str]:
    return [order_text(order, offer) for order in legal_orders(offer)]

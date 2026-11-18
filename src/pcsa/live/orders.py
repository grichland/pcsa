"""The screen check before tapping, and the orders the screen alone allows (the last rungs of the fallback ladder)."""

from __future__ import annotations

from pcsa.decide.api import MOVE_ROWS, Offer, Order
from pcsa.decide.offer_check import check
from pcsa.decide.order_text import usable


def screen_check(order: Order, offer: Offer) -> str | None:
    """None when the current screen shows the order, else the reason it does not."""
    return check(order, offer)


def _switches(offer: Offer) -> list[Order]:
    return [Order.switch(s.name) for s in offer.switches if s.name != offer.leaving]


def screen_actions(offer: Offer, *, mega_used: bool = False) -> list[Order]:
    """What the screen allows: usable move rows (with Mega when the button shows), then the switches."""
    orders = []
    if not offer.forced:
        for m in offer.moves:
            if 0 <= m.row < MOVE_ROWS and usable(m) and (m.pp is not None or m.id):
                orders.append(Order.move(m.row))
                if offer.can_mega_evolve and not mega_used:
                    orders.append(Order.move(m.row, mega=True))
    return orders + _switches(offer)


def blind_actions(offer: Offer) -> list[Order]:
    """Nothing readable: every move row, then every switch on offer."""
    rows = [] if offer.forced else [Order.move(k) for k in range(MOVE_ROWS)]
    return rows + _switches(offer)

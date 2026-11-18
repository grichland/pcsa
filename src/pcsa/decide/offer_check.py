"""Is an Order inside the Offer? Only what the screen shows decides; Hints never narrow it."""

from __future__ import annotations

from pcsa.decide.api import Offer, Order


def check(order: Order, offer: Offer) -> str | None:
    """None when the screen offers the order, else the reason it does not."""
    if order.kind == "switch":
        return _check_switch(order, offer)
    if offer.forced:
        return f"move on a forced ({offer.forced}) screen"
    row = next((m for m in offer.moves if m.row == order.row), None)
    if row is None and offer.moves_read:
        return f"move row {order.row} not on the move screen"
    if row is not None and row.disabled:
        return f"move row {order.row} shown disabled"
    if row is not None and row.pp is not None and row.pp[0] <= 0:
        return f"move row {order.row} shown with no PP"
    if order.mega and offer.can_mega_evolve is not True:
        return "Mega Evolution not offered on screen"
    return None


def _check_switch(order: Order, offer: Offer) -> str | None:
    if order.name == offer.leaving:
        return f"switch target {order.name} is leaving (pivot)"
    if all(s.name != order.name for s in offer.switches):
        return f"switch target {order.name} not on the party screen"
    return None

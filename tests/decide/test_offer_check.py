"""offer_check and the ported live/orders rungs."""

from __future__ import annotations

import pytest

from pcsa.decide.api import Order
from pcsa.decide.offer_check import check
from pcsa.live.orders import blind_actions, screen_actions, screen_check
from tests.decide.examples import forced_offer, read_offer, unread_offer


@pytest.mark.parametrize("row", range(4))
def test_any_row_when_moves_unread(row):
    assert check(Order.move(row), unread_offer()) is None


def test_read_rows_disabled_and_no_pp_rejected():
    offer = read_offer(disabled=(1,), pp_zero=(2,))
    assert check(Order.move(0), offer) is None
    assert "disabled" in check(Order.move(1), offer)
    assert "no PP" in check(Order.move(2), offer)


def test_mega_needs_the_button():
    assert check(Order.move(0, mega=True), read_offer(mega=True)) is None
    assert "Mega" in check(Order.move(0, mega=True), read_offer(mega=False))
    assert "Mega" in check(Order.move(0, mega=True), unread_offer())  # None: not read


def test_switch_rules():
    assert check(Order.switch("Garchomp"), unread_offer()) is None
    assert "not on the party screen" in check(Order.switch("Primarina"), unread_offer())
    pivot = forced_offer(leaving="Mamoswine", kind="pivot")
    assert "leaving" in check(Order.switch("Mamoswine"), pivot)
    assert "forced" in check(Order.move(0), forced_offer())


def test_mega_on_a_switch_is_rejected_at_construction():
    with pytest.raises(ValueError):
        Order("switch", name="Garchomp", mega=True)


def test_screen_check_is_offer_check():
    assert screen_check(Order.move(1), read_offer(disabled=(1,))) == check(Order.move(1), read_offer(disabled=(1,)))


def test_screen_and_blind_actions():
    offer = read_offer(disabled=(1,), mega=True)
    assert screen_actions(offer) == [
        Order.move(0),
        Order.move(0, mega=True),
        Order.move(2),
        Order.move(2, mega=True),
        Order.move(3),
        Order.move(3, mega=True),
        Order.switch("Garchomp"),
        Order.switch("Golisopod"),
    ]
    assert Order.move(0, mega=True) not in screen_actions(offer, mega_used=True)
    assert screen_actions(forced_offer()) == [Order.switch("Garchomp"), Order.switch("Golisopod")]
    assert blind_actions(unread_offer())[:4] == [Order.move(k) for k in range(4)]
    assert blind_actions(forced_offer(leaving="Mamoswine", kind="pivot")) == [
        Order.switch("Garchomp"),
        Order.switch("Golisopod"),
    ]

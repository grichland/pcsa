"""The record's '/choose' strings and legal list equal the recorded ones on every reference decision."""

from __future__ import annotations

import re

from pcsa.decide.api import MoveOffer, Offer, Order, SwitchOffer
from pcsa.decide.order_text import legal_orders, legal_text, order_text
from tests.decide.examples import forced_offer, read_offer, unread_offer
from tests.decide.reference import cases

CHOICE = re.compile(r"^(mega\+)?move([0-3])$")


def test_worked_example_strings():
    offer = unread_offer()
    assert order_text(Order.move(1), offer) == "/choose move earthquake"
    assert order_text(Order.move(1, mega=True), offer) == "/choose move earthquake mega"
    assert order_text(Order.switch("Garchomp"), offer) == "/choose switch Garchomp"
    assert legal_text(offer) == [
        "/choose switch Garchomp",
        "/choose switch Golisopod",
        "/choose move iceshard",
        "/choose move earthquake",
        "/choose move iciclespear",
        "/choose move endeavor",
    ]


def test_recharge_unknown_row_mega_and_forced():
    recharge = Offer((MoveOffer(0, "recharge"),), True, (), False, automatic=True)
    assert order_text(Order.move(0), recharge) == "/choose move 1"
    assert order_text(Order.move(2, mega=True), Offer((), False, (), None)) == "mega+move2"
    assert legal_text(read_offer(disabled=(1,), pp_zero=(3,), mega=True)) == [
        "/choose switch Garchomp",
        "/choose switch Golisopod",
        "/choose move iceshard",
        "/choose move iciclespear",
        "/choose move iceshard mega",
        "/choose move iciclespear mega",
    ]
    assert legal_text(forced_offer(leaving="Mamoswine", kind="pivot")) == [
        "/choose switch Garchomp",
        "/choose switch Golisopod",
    ]
    assert Order.switch("Mamoswine") not in legal_orders(forced_offer(leaving="Mamoswine"))
    assert legal_text(Offer((), False, (SwitchOffer("A"),), None, "faint")) == ["/choose switch A"]


def test_legal_list_equals_every_reference_decision():
    all_cases = cases()
    assert len(all_cases) >= 6000
    wrong = [c.key for c in all_cases if legal_text(c.request.offer) != c.raw["legal_kept"]]
    assert wrong == []


def test_recorded_move_orders_are_written_the_same():
    checked, wrong = 0, []
    for c in cases():
        recorded = c.raw["recorded"]
        match = CHOICE.match(str(recorded.get("choice")))
        if not match or not c.raw["compare"]["request_equal"] or not c.request.offer.moves:
            continue
        order = Order.move(int(match[2]), mega=bool(match[1]))
        checked += 1
        if order_text(order, c.request.offer) != recorded["order"]:
            wrong.append((c.key, recorded["choice"], recorded["order"]))
    assert checked > 3000
    assert wrong == []


def test_recorded_switch_orders_are_in_the_legal_list():
    checked, wrong = 0, []
    for c in cases():
        recorded = c.raw["recorded"]
        if not str(recorded.get("order")).startswith("/choose switch ") or not c.raw["compare"]["legal_equal"]:
            continue
        checked += 1
        name = recorded["order"].removeprefix("/choose switch ")
        if order_text(Order.switch(name), c.request.offer) not in legal_text(c.request.offer):
            wrong.append(c.key)
    assert checked > 1000
    assert wrong == []

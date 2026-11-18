"""T9: the 2b mapping both ways, on the worked example and every reference decision; CG's answers to Orders."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pcsa.battle.ids import to_id
from pcsa.cg import slots
from pcsa.cg.transport import CGError
from pcsa.decide.api import Order
from pcsa.decide.offer_check import check
from tests.decide.examples import ROW_IDS, forced_offer, gold_team, read_offer, request, unread_offer
from tests.decide.reference import cases

MAMOSWINE, EARTHQUAKE_ROW = 4, 1
MIN_ROW_CASES = 5000
GOLD = ("Gholdengo", "Garchomp", "Golisopod", "Primarina", "Mamoswine", "Cinderace")
START = {"type": "battle_start", "own_team": [{"species": to_id(n)} for n in GOLD]}


def move(name: str, mega: bool = False) -> dict:
    return {"seq": 2, "order": {"kind": "move", "move": name, "mega": mega}}


def switch(mon: str) -> dict:
    return {"seq": 2, "order": {"kind": "switch", "mon": mon}}


def test_worked_example():
    team = gold_team()
    assert team.names == GOLD and team.brought() == ("Mamoswine", "Garchomp", "Golisopod")
    assert slots.set_moves(team.export)[MAMOSWINE] == ROW_IDS
    assert slots.active_member(team, unread_offer()) == MAMOSWINE
    assert [slots.slot_of_name(team, n) for n in team.brought()] == [0, 1, 2]
    assert [slots.name_of_slot(team, s) for s in range(3)] == list(team.brought())
    assert slots.position_of_row(1) == 1 == slots.row_of_position(1)
    assert slots.preview_from_cg(slots.preview_to_cg((4, 1, 2))) == (4, 1, 2)
    assert slots.order_of(move("earthquake"), request()) == Order.move(EARTHQUAKE_ROW)  # "For CG: position 1"
    assert slots.order_of(move("iceshard", True), request(read_offer())) == Order.move(0, mega=True)


def test_a_switch_by_species_id():
    req = replace(request(forced_offer()), events=(START,))
    assert slots.order_of(switch("garchomp"), req) == Order.switch("Garchomp")
    assert slots.order_of(switch("garchompmegaz"), req) == Order.switch("Garchomp")  # the same base species
    assert check(slots.order_of(switch("golisopod"), req), req.offer) is None
    with pytest.raises(CGError):
        slots.order_of(switch("gholdengo"), req)  # not brought


def test_an_unread_row_takes_the_set_position():
    offer = unread_offer()
    blank = replace(offer, moves=tuple(replace(m, id=None) if m.row == 2 else m for m in offer.moves))
    assert slots.order_of(move("iciclespear"), request(blank)) == Order.move(2)


@pytest.mark.parametrize(
    "reply",
    [
        move("flamethrower"),
        move("earthquake", mega="yes"),  # type: ignore[arg-type]
        switch(""),
        {"seq": 2},
        {"seq": 2, "order": {"kind": "pass"}},
        {"seq": 2, "order": "earthquake"},
    ],
)
def test_what_cannot_be_placed_raises(reply):
    with pytest.raises(CGError):
        slots.order_of(reply, request())


def test_a_move_on_a_replacement_screen_raises():
    with pytest.raises(CGError):
        slots.order_of(move("earthquake"), request(forced_offer()))


@pytest.mark.parametrize("picks", [[0, 0, 1], [0, 1], [0, 1, 6], "012", [0, 1, True], None])
def test_bad_picks_raise(picks):
    with pytest.raises(CGError):
        slots.picks_of({"picks": picks})


def test_picks():
    assert slots.picks_of({"picks": [4, 1, 2]}) == (4, 1, 2)


def test_export_order_equals_names_order_on_every_run():
    wrong = []
    for c in cases():
        team = c.request.team
        sets = slots.set_moves(team.export)
        heads = [block.splitlines()[0] for block in team.export.strip().split("\n\n")]
        if len(sets) != len(team.names) or any(not h.startswith(n) for h, n in zip(heads, team.names, strict=True)):
            wrong.append(c.key)
    assert wrong == []


def test_mapping_on_every_reference_decision():
    """Rows read from the request equal the export's set order; the side's members are the brought names; every
    move id and brought species CG could name maps back to that row or name, inside the Offer."""
    checked, wrong = 0, []
    for c in cases():
        team, offer = c.request.team, c.request.offer
        side = sorted(m["ident"].split(": ", 1)[1] for m in c.raw["request"]["side"]["pokemon"])
        if side != sorted(team.brought()):
            wrong.append((c.key, "brought"))
        for name in team.brought():
            if slots.name_of_slot(team, slots.slot_of_name(team, name)) != name:
                wrong.append((c.key, "slot"))
            if slots.switch_name(team, [], to_id(name)) != name:
                wrong.append((c.key, "switch"))
        if offer.moves:
            checked += 1
            member = team.names.index(c.active)
            if slots.row_mismatches(team, member, offer) or slots.active_member(team, offer) != member:
                wrong.append((c.key, "rows"))
            for m in offer.moves:
                if m.id and slots.move_row(team, offer, m.id) != m.row:
                    wrong.append((c.key, "move"))
    assert checked > MIN_ROW_CASES
    assert wrong == []

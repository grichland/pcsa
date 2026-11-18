import pytest

from pcsa.decide.api import Order, OwnTeam, valid_picks

EXPORT_NAMES = ("Gholdengo", "Garchomp", "Golisopod", "Primarina", "Mamoswine", "Cinderace")


def test_move_and_switch_orders():
    assert Order.move(1).label() == "move1"
    assert Order.move(0, mega=True).label() == "mega+move0"
    assert Order.switch("Garchomp").label() == "switch:Garchomp"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "move"},
        {"kind": "move", "row": 4},
        {"kind": "move", "row": 0, "name": "Garchomp"},
        {"kind": "switch"},
        {"kind": "switch", "name": "Garchomp", "mega": True},
        {"kind": "switch", "name": "Garchomp", "row": 1},
        {"kind": "tera", "row": 0},
    ],
)
def test_malformed_orders_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Order(**kwargs)


def test_brought_is_lead_first():
    team = OwnTeam(export="", names=EXPORT_NAMES, picks=(4, 1, 2))
    assert team.brought() == ("Mamoswine", "Garchomp", "Golisopod")


def test_valid_picks():
    assert valid_picks((4, 1, 2))
    assert not valid_picks((4, 4, 2))
    assert not valid_picks((0, 1))
    assert not valid_picks((0, 1, 6))

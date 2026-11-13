"""Their Regenerator user heals 1/3 on switch-out; the next send-out line carries it (pcsa.battle.regenerator)."""

from __future__ import annotations

from types import SimpleNamespace as N

import pytest

from pcsa.battle import regenerator
from pcsa.battle.state import build
from pcsa.battle.teams import OwnMember, Team


def _team(*names: str) -> Team:
    return Team(
        None, tuple(OwnMember(i, n, n, None, None, (None,) * 4, None, None, None) for i, n in enumerate(names))
    )


def _switch_hp(seen: frozenset, monkeypatch, *, faint: bool = False, enabled: bool = True) -> list[str]:
    monkeypatch.setattr(regenerator, "seen_abilities", lambda species: seen)
    preview = [N(top=[(n, 0.9)], species=[(n, 0.9)]) for n in ("Slowking-Galar", "Lucario", "Rillaboom")]
    rows = {
        1: ["The Trainer sent out Slowking!", "Go! Garchomp!", "HP Lost: Slowking’s HP: 100% → 23%"],
        2: ["The Trainer withdrew Slowking!", "The Trainer sent out Lucario!"],
        3: ["The Trainer withdrew Lucario!", "The Trainer sent out Slowking!"],
    }
    if faint:
        rows[2] = [
            "HP Lost: Slowking’s HP: 23% → 0%",
            "The opposing Slowking fainted!",
            "The Trainer sent out Lucario!",
        ]
        del rows[3]
    _, out = build(
        "t", _team("Garchomp", "Gholdengo", "Golisopod"), preview, picks=[0, 1, 2], rows_by_turn=rows,
        decision_turn=4, regenerator=enabled,
    )  # fmt: skip
    return [line.rsplit("|", 1)[1] for line in out.lines if line.startswith("|switch|p2a: Slowking")]


def test_regenerator_from_every_seen_build_heals_a_third(monkeypatch) -> None:
    assert _switch_hp(frozenset({"regenerator"}), monkeypatch) == ["100/100", "56/100"]


def test_not_sure_of_regenerator_keeps_the_last_hp(monkeypatch) -> None:
    assert _switch_hp(frozenset({"regenerator", "owntempo"}), monkeypatch) == ["100/100", "23/100"]
    assert _switch_hp(frozenset(), monkeypatch) == ["100/100", "23/100"]


def test_setting_off_keeps_the_last_hp(monkeypatch) -> None:
    assert _switch_hp(frozenset({"regenerator"}), monkeypatch, enabled=False) == ["100/100", "23/100"]


def test_a_fainted_pokemon_does_not_heal(monkeypatch) -> None:
    assert _switch_hp(frozenset({"regenerator"}), monkeypatch, faint=True) == ["100/100"]


@pytest.mark.parametrize(
    ("ability", "possible", "seen", "sure"),
    [
        ("regenerator", [], frozenset(), True),
        ("owntempo", ["Regenerator"], frozenset({"regenerator"}), False),
        (None, ["Regenerator"], frozenset(), True),
        (None, ["Natural Cure", "Regenerator"], frozenset({"regenerator"}), True),
        (None, ["Natural Cure", "Regenerator"], frozenset(), False),
        (None, ["Natural Cure"], frozenset({"regenerator"}), False),
    ],
)
def test_surely_regenerator(monkeypatch, ability, possible, seen, sure) -> None:
    monkeypatch.setattr(regenerator, "seen_abilities", lambda species: seen)
    mon = N(ability=ability, possible_abilities=possible, species="x")
    assert regenerator.surely_regenerator(mon) is sure

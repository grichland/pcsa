"""The yawn_switch guard rule (14 Nov 2026, from the labelled Yawn loops: Garchomp stayed in drowsy against
Hippowdon, could not KO it, fell asleep and Hippowdon used Slack Off)."""

from __future__ import annotations

import json

import pytest
from poke_env.battle import Field, Status

from pcsa.battle.state import replay
from pcsa.players import mirror, neural_rules
from pcsa.players.neural_rules import Spot
from tests.conftest import REPO
from tests.players.test_disciplined import HITS, FakeThreat

FIXTURE = REPO / "tests" / "fixtures" / "yawn_rule" / "garchomp-drowsy-vs-hippowdon.json"
POWER_GEM = "/choose move powergem"


@pytest.fixture
def threat(monkeypatch):
    def install(hits: dict | None = None, **kwargs) -> FakeThreat:
        estimates = FakeThreat({**HITS, **(hits or {})}, **kwargs)
        monkeypatch.setattr(neural_rules, "threat", estimates)
        return estimates

    return install


def position():
    rec = json.loads(FIXTURE.read_text(encoding="utf-8"))
    b = replay(rec["protocol"], rec["request"])
    return b, mirror.legal_actions(b)


def spot(b, legal, chosen: str = POWER_GEM) -> Spot:
    by = {a.message: a for a in legal}
    return Spot(b, tuple(legal), by[chosen], [by[chosen], *(a for a in legal if a.message != chosen)], None, None)


def decide(s: Spot):
    out = neural_rules.apply(["yawn_switch"], s)
    return (out[0].message, out[1], out[2]) if out else None


def test_drowsy_and_no_ko_switches(threat) -> None:
    threat()  # every hit 10%: nothing KOs Hippowdon
    b, legal = position()
    assert b.active_pokemon.species.startswith("garchomp") and not b.active_pokemon.first_turn
    message, rule, why = decide(spot(b, legal))  # type: ignore[misc]
    assert rule == "yawn_switch" and message.startswith("/choose switch") and "drowsy" in why


def test_no_switch_when_a_move_kos(threat) -> None:
    b, legal = position()
    threat(move_hits={(b.active_pokemon.species, b.opponent_active_pokemon.species, "dracometeor"): 1.5})
    assert decide(spot(b, legal)) is None


def test_no_switch_when_it_cannot_fall_asleep_or_just_came_in(threat) -> None:
    threat()
    b, legal = position()
    mon = b.active_pokemon
    mon._active_turns = 1  # Yawned on the way in
    assert decide(spot(b, legal)) is None
    mon._active_turns = 3
    mon._status = Status.PAR  # already statused: Yawn's sleep fails
    assert decide(spot(b, legal)) is None
    mon._status = None
    b._fields = {Field.MISTY_TERRAIN: 1}  # grounded under Misty Terrain
    if b.is_grounded(mon):
        assert decide(spot(b, legal)) is None


def test_the_rule_is_known_to_the_settings() -> None:
    neural_rules.check(["yawn_switch"], "test")

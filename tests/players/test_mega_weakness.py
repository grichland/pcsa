"""The mega_weakness and weak4x_switch guard rules (pcsa.players.mega_weakness; 14 Nov 2026, from the hand-labelled
live losses: Mega Golisopod is Bug/Steel and takes Fire 4x, base Golisopod is Bug/Water)."""

from __future__ import annotations

import json

import pytest

from pcsa.battle.state import replay
from pcsa.players import mega_weakness as weak
from pcsa.players import mirror, neural_rules
from pcsa.players.neural_rules import Spot
from tests.conftest import REPO
from tests.players.test_disciplined import HITS, FakeThreat

FIXTURES = REPO / "tests" / "fixtures" / "mega_rule"


@pytest.fixture
def threat(monkeypatch):
    def install(hits: dict | None = None, **kwargs) -> FakeThreat:
        estimates = FakeThreat({**HITS, **(hits or {})}, **kwargs)
        monkeypatch.setattr(neural_rules, "threat", estimates)
        return estimates

    return install


def position(name: str):
    rec = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    m = rec["notes"]["matrix"]
    their = tuple(
        ("switch", "", p) if k == "switch" else ("move", c, p)
        for c, p, k in zip(m["cols"], m["p_cols"], m["col_kind"], strict=True)
        if c
    )
    b = replay(rec["protocol"], rec["request"])
    return b, mirror.legal_actions(b), their


def spot(b, legal, their, chosen: str, first: tuple[str, ...] = ()) -> Spot:
    by = {a.message: a for a in legal}
    order = [by[m] for m in first if m in by] + [a for a in legal if a.message not in first]
    return Spot(b, tuple(legal), by[chosen], order, None, None, their, 0.5)


def test_type_multipliers() -> None:
    assert weak.multiplier("fire", ["bug", "steel"]) == 4.0
    assert weak.multiplier("fire", ["bug", "water"]) == 1.0
    assert weak.multiplier("ice", ["dragon", "ground"]) == 4.0
    assert weak.multiplier("fire", []) == 1.0 and weak.multiplier("nonsense", ["bug"]) == 1.0
    assert weak.types_of("golisopodmega") == ["bug", "steel"]


def test_no_mega_into_a_likely_fire_move(threat) -> None:
    # 13 Nov battle 2, turn 3: Golisopod vs Galarian Slowking, Flamethrower predicted 63%; it Mega Evolved for First
    # Impression (Slowking to 17%) and was one-shot. The same First Impression without Mega Evolution is the pick.
    threat()
    b, legal, their = position("case29-mega-into-flamethrower.json")
    mega = "/choose move firstimpression mega"
    message, rule, why = neural_rules.apply(["mega_weakness"], spot(b, legal, their, mega, (mega,)))  # type: ignore[misc]
    assert (message.message, rule) == ("/choose move firstimpression", "mega_weakness")
    assert "flamethrower (63% likely)" in why and "bug/steel" in why


def test_mega_stays_when_our_move_kos_first_or_no_fire_is_likely(threat) -> None:
    b, legal, their = position("case29-mega-into-flamethrower.json")
    mega = "/choose move firstimpression mega"
    threat(move_hits={("golisopod", "slowkinggalar", "firstimpression"): 1.2})  # it KOs first (priority +2)
    assert neural_rules.apply(["mega_weakness"], spot(b, legal, their, mega)) is None
    threat()
    calm = tuple(x for x in their if x[1] != "flamethrower")  # no Fire move predicted, none shown
    assert neural_rules.apply(["mega_weakness"], spot(b, legal, calm, mega)) is None
    assert neural_rules.apply(["mega_weakness"], spot(b, legal, their, "/choose move firstimpression")) is None


def test_switch_out_of_a_4x_hit_that_lands_first(threat) -> None:
    # 12 Nov battle 11, turn 7: Mega Golisopod in front of a faster Fire user (Flamethrower 63%); stays in and dies.
    b, legal, their = position("stay-in-vs-flamethrower.json")
    me, foe = b.active_pokemon.species, b.opponent_active_pokemon.species
    threat(hits={(foe, me): (1.5, "flamethrower")})
    chosen = next(a.message for a in legal if "move" in a.message)
    switches = [a.message for a in legal if "switch" in a.message]
    out = neural_rules.apply(["weak4x_switch"], spot(b, legal, their, chosen))
    assert out is not None and out[1] == "weak4x_switch" and out[0].message in switches
    target = out[0].order.order
    assert weak.multiplier("fire", weak.mon_types(target)) <= 1.0 and "4x first and KOs" in out[2]


def test_no_switch_when_we_are_faster_and_ko_or_it_does_not_ko(threat) -> None:
    b, legal, their = position("stay-in-vs-flamethrower.json")
    me, foe = b.active_pokemon.species, b.opponent_active_pokemon.species
    chosen = next(a.message for a in legal if "move" in a.message)
    threat(hits={(foe, me): (0.3, "flamethrower")})  # the hit does not KO
    assert neural_rules.apply(["weak4x_switch"], spot(b, legal, their, chosen)) is None
    threat(hits={(foe, me): (1.5, "flamethrower"), (me, foe): (1.5, "x")}, faster={(me, foe)})  # we KO first
    assert neural_rules.apply(["weak4x_switch"], spot(b, legal, their, chosen)) is None


def test_the_rules_are_known_to_the_settings() -> None:
    neural_rules.check(["mega_weakness", "weak4x_switch"], "test")

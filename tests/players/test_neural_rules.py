"""The neural player's guard rules on hand-made positions.

The threat estimates are test_disciplined's table (FakeThreat); the network's ranking is the legal order given."""

from __future__ import annotations

import pytest

from pcsa.common.errors import ConfigError
from pcsa.players import mirror, neural_rules
from pcsa.players.neural import NeuralParams
from pcsa.players.neural_rules import RULES, Spot
from tests.conftest import REPO
from tests.players.test_disciplined import HITS, FakeThreat, battle

ALL = tuple(RULES)


@pytest.fixture
def threat(monkeypatch):
    def install(hits: dict | None = None, **kwargs) -> FakeThreat:
        estimates = FakeThreat({**HITS, **(hits or {})}, **kwargs)
        monkeypatch.setattr(neural_rules, "threat", estimates)
        return estimates

    return install


def spot(b, chosen: str, ranked: list[str] | None = None) -> Spot:
    legal = mirror.legal_actions(b)
    by_message = {a.message: a for a in legal}
    order = [by_message[m] for m in ranked] if ranked else list(legal)
    order += [a for a in legal if a not in order]
    return Spot(b, legal, by_message[chosen], order, None, None)


def decide(rules, s: Spot) -> tuple[str, str] | None:
    out = neural_rules.apply(rules, s)
    return (out[0].message, out[1]) if out else None


def test_the_repo_settings_turn_on_every_guard_and_bad_names_are_rejected() -> None:
    params = NeuralParams.load(REPO / "config" / "players" / "neural.yaml")
    assert sorted(params.rules) == sorted(ALL)
    with pytest.raises(ConfigError):
        neural_rules.check(["no_full_heal", "roost_spam"], "test")


def test_no_heal_at_full_hp(threat) -> None:
    threat()
    ranked = ["/choose move slackoff", "/choose move surf"]
    full = spot(battle("Slowbro"), "/choose move slackoff", ranked)
    assert decide(ALL, full) == ("/choose move surf", "no_full_heal")
    hurt = spot(battle("Slowbro", hp=0.5), "/choose move slackoff", ranked)
    assert decide(ALL, hurt) is None


def test_a_priority_ko_replaces_an_attack_that_does_not_ko_first(threat) -> None:
    threat(priority={("azumarill", "dragonite"): "aquajet"})
    assert decide(ALL, spot(battle("Azumarill"), "/choose move playrough")) == ("/choose move aquajet", "priority_ko")


def test_no_priority_override_when_the_networks_attack_kos_first(threat) -> None:
    threat(
        priority={("azumarill", "dragonite"): "aquajet"},
        faster={("azumarill", "dragonite")},
        move_hits={("azumarill", "dragonite", "playrough"): 1.2},
    )
    assert decide(ALL, spot(battle("Azumarill"), "/choose move playrough")) is None


def test_the_rules_leave_the_network_alone_otherwise(threat) -> None:
    threat()
    assert decide(ALL, spot(battle("Slowbro"), "/choose move surf")) is None
    assert decide((), spot(battle("Slowbro"), "/choose move slackoff")) is None


@pytest.mark.parametrize(
    ("event", "attacked"),
    [
        ("|move|p2a: Dragonite|Outrage|p1a: Azumarill", True),
        ("|move|p2a: Dragonite|Roost|p2a: Dragonite", False),
        ("|switch|p2a: Dragonite|Dragonite, L50|100/100", False),
    ],
)
def test_sucker_punch_ko_needs_the_foe_to_have_attacked(event: str, attacked: bool) -> None:
    assert neural_rules.foe_attacked(battle("Azumarill", [[event]])) is attacked

"""The neural player's override rules: which rules a team gets, and the rules on hand-made positions.

The threat estimates are test_disciplined's table (FakeThreat); the network's ranking is the legal order given."""

from __future__ import annotations

import pytest

from pcsa.common.errors import ConfigError
from pcsa.players import mirror, neural_rules
from pcsa.players.neural_rules import GENERAL, RuleParams, RulesConfig, Spot
from tests.conftest import REPO
from tests.players.test_disciplined import HITS, FakeThreat, battle

RULES_FILE = REPO / "config" / "players" / "neural_rules.yaml"
TEAM3 = ["Charizard-Mega-X", "Hippowdon", "Metagross", "Mimikyu", "Primarina", "Greninja"]


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
    return Spot(b, legal, by_message[chosen], order, None, None, RuleParams())


def decide(rules, s: Spot) -> tuple[str, str] | None:
    out = neural_rules.apply(rules, s)
    return (out[0].message, out[1]) if out else None


# Which rules ------------------------------------------------------------------------------------------------------
def test_the_repo_file_loads_and_matches_teams_by_roster() -> None:
    config = RulesConfig.load(RULES_FILE)
    assert config.general == GENERAL
    rules, profile = config.select("auto", TEAM3)
    assert profile is not None and profile.name == "team3" and rules == (*GENERAL, "rocks_first")
    rules, profile = config.select("auto", ["Dragonite", *TEAM3[1:]])  # not the whole roster: general only
    assert profile is None and rules == GENERAL
    assert config.select("none", TEAM3) == ((), None)
    assert config.select("team2", TEAM3)[0] == (*GENERAL, "scarf_lock")
    with pytest.raises(ConfigError):
        config.select("team9", TEAM3)


def test_unknown_rule_names_are_rejected(tmp_path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text("general: [priority_ko, roost_spam]\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        RulesConfig.load(path)


# Rules --------------------------------------------------------------------------------------------------------
def test_a_priority_ko_replaces_an_attack_that_does_not_ko_first(threat) -> None:
    threat(priority={("azumarill", "dragonite"): "aquajet"})
    b = battle("Azumarill")
    assert decide(GENERAL, spot(b, "/choose move playrough")) == ("/choose move aquajet", "priority_ko")


def test_no_priority_override_when_the_networks_attack_kos_first(threat) -> None:
    threat(
        priority={("azumarill", "dragonite"): "aquajet"},
        faster={("azumarill", "dragonite")},
        move_hits={("azumarill", "dragonite", "playrough"): 1.2},
    )
    assert decide(GENERAL, spot(battle("Azumarill"), "/choose move playrough")) is None


def test_no_heal_at_full_hp(threat) -> None:
    threat()
    ranked = ["/choose move slackoff", "/choose move surf"]
    full = spot(battle("Slowbro"), "/choose move slackoff", ranked)
    assert decide(GENERAL, full) == ("/choose move surf", "no_full_heal")
    hurt = spot(battle("Slowbro", hp=0.5), "/choose move slackoff", ranked)
    assert decide(["no_full_heal"], hurt) is None


def test_the_rules_leave_the_network_alone_otherwise(threat) -> None:
    threat()
    assert decide(GENERAL, spot(battle("Slowbro"), "/choose move surf")) is None
    assert decide((), spot(battle("Slowbro"), "/choose move slackoff")) is None


@pytest.mark.parametrize(
    ("event", "attacked"),
    [
        ("|move|p2a: Dragonite|Outrage|p1a: Azumarill", True),
        ("|move|p2a: Dragonite|Roost|p2a: Dragonite", False),
        ("|switch|p2a: Dragonite|Dragonite, L50|100/100", False),
    ],
)
def test_sucker_punch_reads_the_foes_last_action(event: str, attacked: bool) -> None:
    b = battle("Azumarill", [[event]])
    assert neural_rules.foe_attacked(b) is attacked


def test_mega_ids_count_as_their_base_species() -> None:
    assert neural_rules.base("charizardmegax") == "charizard" == neural_rules.base("Charizard-Mega-X")
    assert neural_rules.base("Yanmega") == "yanmega"
    config = RulesConfig.load(RULES_FILE)
    assert config.match(["charizardmegax", "hippowdon", "metagrossmega", "mimikyu", "primarina", "greninja"])

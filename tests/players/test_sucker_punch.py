"""The sucker_punch guard rule: Sucker Punch (and Thunderclap) only when the foe may well attack.

Blocked when it already failed this battle against the same foe, or when the opponent head's status moves and
switches exceed `passive_max`; never when it is the only legal action; priority_ko does not pick a blocked one."""

from __future__ import annotations

import pytest

from pcsa.battle.state import replay
from pcsa.players import mirror, neural_rules
from pcsa.players.neural import NeuralParams
from pcsa.players.neural_rules import Spot
from pcsa.players.sucker_punch import failed_against, passive_share
from tests.conftest import REPO
from tests.players.test_disciplined import HITS, PRELUDE, FakeThreat

TEAM = {
    "Golisopod": (182, ["leechlife", "suckerpunch", "liquidation", "swordsdance"]),
    "Gholdengo": (163, ["makeitrain", "shadowball", "nastyplot", "recover"]),
    "Garchomp": (183, ["earthquake", "dragonclaw", "swordsdance", "stoneedge"]),
}
RULES = ("no_effect", "no_full_heal", "useless_status", "locked_futile", "sucker_punch", "priority_ko")
FAILED = ["|move|p1a: Golisopod|Sucker Punch|p2a: Dragonite", "|-fail|p1a: Golisopod"]
LANDED = ["|move|p1a: Golisopod|Sucker Punch|p2a: Dragonite", "|-damage|p2a: Dragonite|60/100"]
PUNCH, LEECH = "/choose move suckerpunch", "/choose move leechlife"
ATTACKS = (("move", "outrage", 0.7), ("move", "roost", 0.1), ("switch", "", 0.2))  # passive 0.3
STALLS = (("move", "outrage", 0.3), ("move", "dragondance", 0.4), ("switch", "", 0.3))  # passive 0.7


@pytest.fixture
def threat(monkeypatch):
    def install(**kwargs) -> FakeThreat:
        estimates = FakeThreat(dict(HITS), **kwargs)
        monkeypatch.setattr(neural_rules, "threat", estimates)
        return estimates

    return install


def battle(turns: list[list[str]] | None = None, foe: str = "Dragonite"):
    lines = [*PRELUDE, "|switch|p1a: Golisopod|Golisopod, L50|182/182", f"|switch|p2a: {foe}|{foe}, L50|100/100"]
    lines.append("|turn|1")
    for turn, events in enumerate(turns or [], start=2):
        lines += [*events, f"|turn|{turn}"]
    side = [
        {
            "ident": f"p1: {name}",
            "details": f"{name}, L50",
            "condition": f"{hp}/{hp}",
            "active": name == "Golisopod",
            "stats": {"atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 100},
            "moves": moves,
            "baseAbility": "",
            "item": "",
        }
        for name, (hp, moves) in TEAM.items()
    ]
    moves = [{"move": m, "id": m, "pp": 10, "maxpp": 10, "target": "normal"} for m in TEAM["Golisopod"][1]]
    return replay(
        lines, {"side": {"name": "me", "id": "p1", "pokemon": side}, "rqid": 1, "active": [{"moves": moves}]}
    )


def spot(b, chosen: str = PUNCH, ranked: tuple[str, ...] = (PUNCH, LEECH), their=None, only=None) -> Spot:
    legal = [a for a in mirror.legal_actions(b) if only is None or a.message in only]
    by = {a.message: a for a in legal}
    order = [by[m] for m in ranked if m in by] + [a for a in legal if a.message not in ranked]
    return Spot(b, tuple(legal), by[chosen], order, None, None, their, 0.5)


def decide(s: Spot, rules=RULES) -> tuple[str, str, str] | None:
    out = neural_rules.apply(rules, s)
    return (out[0].message, out[1], out[2]) if out else None


def test_the_repo_settings_turn_the_rule_on() -> None:
    params = NeuralParams.load(REPO / "config" / "players" / "neural.yaml")
    assert params.rules.index("sucker_punch") < params.rules.index("priority_ko")
    assert params.sucker_punch_passive == 0.5
    gold940 = NeuralParams.load(REPO / "config" / "players" / "neural-gold940.yaml")
    assert "sucker_punch" not in gold940.rules  # the rollback plays as before


def test_blocked_after_it_failed_against_the_same_foe(threat) -> None:
    threat()
    b = battle([FAILED])
    assert failed_against(b) == {"p2: Dragonite"}
    message, rule, why = decide(spot(b))
    assert (message, rule) == (LEECH, "sucker_punch") and "already failed against dragonite" in why


def test_not_blocked_when_it_landed_or_failed_against_another_foe(threat) -> None:
    threat()
    assert decide(spot(battle([LANDED]))) is None
    switched = battle([FAILED, ["|switch|p2a: Garchomp|Garchomp, L50|100/100"]])
    assert decide(spot(switched)) is None  # it failed against Dragonite, not Garchomp
    back = battle(
        [FAILED, ["|switch|p2a: Garchomp|Garchomp, L50|100/100"], ["|switch|p2a: Dragonite|Dragonite, L50|100/100"]]
    )
    assert decide(spot(back))[1] == "sucker_punch"  # Dragonite again
    into_switch = [
        "|switch|p2a: Garchomp|Garchomp, L50|100/100",
        *(e.replace("Dragonite", "Garchomp") for e in FAILED),
    ]
    assert failed_against(battle([into_switch])) == set()  # the switch made it fail, not Garchomp's choice


def test_blocked_when_their_status_moves_and_switches_are_likely(threat) -> None:
    threat()
    assert passive_share(STALLS) == 0.7 and round(passive_share(ATTACKS), 6) == 0.3
    message, rule, why = decide(spot(battle(), their=STALLS))
    assert (message, rule) == (LEECH, "sucker_punch") and "0.70 > 0.5" in why
    assert decide(spot(battle(), their=ATTACKS)) is None
    assert decide(spot(battle(), their=None)) is None  # no opponent head: only the failed-before check


def test_not_blocked_when_it_is_the_only_legal_action(threat) -> None:
    threat()
    assert decide(spot(battle([FAILED]), their=STALLS, only={PUNCH})) is None


def test_priority_ko_does_not_pick_a_blocked_sucker_punch(threat) -> None:
    threat(priority={("golisopod", "dragonite"): "suckerpunch"})
    attacked = ["|move|p2a: Dragonite|Outrage|p1a: Golisopod", "|-damage|p1a: Golisopod|100/182"]
    b = battle([attacked])
    ranked = (LEECH, PUNCH)
    assert decide(spot(b, LEECH, ranked, their=ATTACKS))[:2] == (PUNCH, "priority_ko")
    assert decide(spot(b, LEECH, ranked, their=STALLS)) is None
    without = tuple(r for r in RULES if r != "sucker_punch")  # neural-gold940.yaml: as before
    assert decide(spot(b, LEECH, ranked, their=STALLS), without)[:2] == (PUNCH, "priority_ko")

"""The sucker_punch guard rule: Sucker Punch (and Thunderclap) only when the foe may well attack."""

from __future__ import annotations

import pytest

from pcsa.battle.state import replay
from pcsa.players import mirror, neural_rules
from pcsa.players import sucker_punch as sucker
from pcsa.players.neural import NeuralParams
from pcsa.players.neural_rules import Spot
from pcsa.players.sucker_punch import passive_share
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
ATTACKS = (("move", "outrage", 0.7), ("move", "roost", 0.1), ("switch", "", 0.2))
STALLS = (("move", "outrage", 0.3), ("move", "dragondance", 0.4), ("switch", "", 0.3))


@pytest.fixture
def threat(monkeypatch):
    def install(**kwargs) -> FakeThreat:
        estimates = FakeThreat(dict(HITS), **kwargs)
        monkeypatch.setattr(neural_rules, "threat", estimates)
        return estimates

    return install


def battle(turns: list[list[str]] | None = None, foe: str = "Dragonite", preview: tuple[str, ...] = ()):
    lines = [*PRELUDE[:3], "|clearpoke", *(f"|poke|p2|{name}, L50|" for name in preview), "|teampreview|3"]
    lines = [
        *(lines if preview else PRELUDE[:3]),
        *PRELUDE[3:],
        "|switch|p1a: Golisopod|Golisopod, L50|182/182",
        f"|switch|p2a: {foe}|{foe}, L50|100/100",
    ]
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
    assert "sucker_punch" not in NeuralParams().rules


def test_a_fail_earlier_in_the_battle_no_longer_blocks_it(threat) -> None:
    # the "already failed against this foe" clause was wrong in 4 of 5 labelled live cases
    threat()
    assert decide(spot(battle([FAILED]))) is None
    assert decide(spot(battle([FAILED]), their=ATTACKS)) is None
    assert decide(spot(battle([LANDED]))) is None


SIX = ("Dragonite", "Garchomp", "Snorlax", "Gengar", "Kingambit", "Corviknight")
SEEN_ALL = [
    ["|switch|p2a: Garchomp|Garchomp, L50|100/100"],
    ["|switch|p2a: Snorlax|Snorlax, L50|100/100"],
    ["|switch|p2a: Dragonite|Dragonite, L50|100/100"],
]
# Dragonite in: 0.35 attacking, 0.05 status, switches 0.1 to Garchomp (seen) and 0.5 to preview-only Pokemon
TO_PREVIEW = (
    ("move", "outrage", 0.35),
    ("move", "roost", 0.05),
    ("switch", "garchomp", 0.1),
    ("switch", "gengar", 0.3),
    ("switch", "kingambit", 0.2),
)


def test_switches_to_preview_only_pokemon_do_not_count_once_their_team_is_seen(threat) -> None:
    threat()
    b = battle(SEEN_ALL, preview=SIX)
    assert sucker.team_closed(b)
    assert sucker.impossible_switch(b, "gengar") and not sucker.impossible_switch(b, "garchomp")
    assert not sucker.impossible_switch(b, "")
    # Gengar and Kingambit masked; the rest renormalised: (0.05 + 0.1) / 0.5 = 0.3
    assert round(passive_share(TO_PREVIEW, b), 6) == 0.3
    assert round(passive_share(TO_PREVIEW), 6) == 0.65
    assert decide(spot(b, their=TO_PREVIEW)) is None


def test_switches_to_preview_pokemon_still_count_while_their_team_is_open(threat) -> None:
    threat()
    b = battle(SEEN_ALL[:1], preview=SIX)  # Dragonite and Garchomp seen: the third is any of the other four
    assert not sucker.team_closed(b)
    assert round(passive_share(TO_PREVIEW, b), 6) == 0.65
    message, rule, why = decide(spot(b, their=TO_PREVIEW))
    assert (message, rule) == (LEECH, "sucker_punch") and "0.65 > 0.5" in why


def test_mega_forms_count_as_their_base_species(threat) -> None:
    threat()
    mega = [["|switch|p2a: Garchomp|Garchomp-Mega, L50|100/100"], *SEEN_ALL[1:]]
    b = battle(mega, preview=SIX)
    assert sucker.team_closed(b) and not sucker.impossible_switch(b, "garchomp")
    assert sucker.impossible_switch(b, "gengar")


def test_blocked_when_their_status_moves_and_switches_are_likely(threat) -> None:
    threat()
    assert passive_share(STALLS) == 0.7 and round(passive_share(ATTACKS), 6) == 0.3
    message, rule, why = decide(spot(battle(), their=STALLS))
    assert (message, rule) == (LEECH, "sucker_punch") and "0.70 > 0.5" in why
    assert decide(spot(battle(), their=ATTACKS)) is None
    assert decide(spot(battle(), their=None)) is None


def test_never_blocked_one_against_one(threat) -> None:
    # the user's rule: with one Pokemon left on each side, the chance they do not attack is 0
    threat()
    gone = [
        "|switch|p2a: Snorlax|Snorlax, L50|100/100",
        "|faint|p2a: Snorlax",
        "|switch|p2a: Gengar|Gengar, L50|100/100",
        "|faint|p2a: Gengar",
        "|switch|p2a: Dragonite|Dragonite, L50|100/100",
    ]
    b = battle([FAILED, gone])
    for name in ("Gholdengo", "Garchomp"):
        b.team[f"p1: {name}"].faint()
    assert sucker.last_one_each(b)
    assert decide(spot(b, their=STALLS)) is None
    alive = battle([FAILED, gone])
    assert not sucker.last_one_each(alive) and decide(spot(alive, their=STALLS))[1] == "sucker_punch"


def test_not_blocked_when_it_is_the_only_legal_action(threat) -> None:
    threat()
    assert decide(spot(battle([FAILED]), their=STALLS, only={PUNCH})) is None


def test_priority_ko_does_not_pick_a_blocked_sucker_punch(threat) -> None:
    threat(priority={("golisopod", "dragonite"): "suckerpunch"})
    attacked = ["|move|p2a: Dragonite|Outrage|p1a: Golisopod", "|-damage|p1a: Golisopod|100/182"]
    b = battle([attacked])
    ranked = (LEECH, PUNCH)
    # no switch in their top 3 (priority_ko stands aside otherwise)
    attacks = (("move", "outrage", 0.7), ("move", "roost", 0.12), ("move", "extremespeed", 0.1), ("switch", "", 0.08))
    stalls = (("move", "dragondance", 0.45), ("move", "roost", 0.2), ("move", "outrage", 0.2), ("switch", "", 0.15))
    assert decide(spot(b, LEECH, ranked, their=attacks))[:2] == (PUNCH, "priority_ko")
    assert decide(spot(b, LEECH, ranked, their=stalls)) is None
    without = tuple(r for r in RULES if r != "sucker_punch")
    assert decide(spot(b, LEECH, ranked, their=stalls), without)[:2] == (PUNCH, "priority_ko")
    assert decide(spot(b, LEECH, ranked, their=ATTACKS)) is None  # a switch in their top 3


def test_the_opponent_heads_switch_columns_name_their_targets() -> None:
    torch = pytest.importorskip("torch")
    from pcsa.rl_training import features, features_log
    from pcsa.rl_training.former_notes import their_actions
    from pcsa.rl_training.spec import VOCAB

    b = battle(SEEN_ALL, preview=SIX)
    v = VOCAB["move"]
    out = {"kept_idx": torch.tensor([v, v + 2, v + 4]), "kept_p": torch.tensor([0.2, 0.3, 0.5])}
    bench = [m.species for _, m in features._their_members(b, features_log.read(b))[1:]]
    acts = their_actions(out, b)
    assert [a[1] for a in acts] == [bench[0], bench[2], bench[4]]
    assert {a[1] for a in acts} & {"gengar", "kingambit", "corviknight"}
    assert [a[1] for a in their_actions(out)] == ["", "", ""]

"""The encore_switch guard rule (the user's rule, 14 Nov 2026): when their active Encored us since it came in and
earlier turns show for sure that it moves before our active, we switch (the network picks which switch).

Live battles 14 and 35 of run 20261113-215202: a faster opposing Primarina Encored our Primarina's Calm Mind every
three turns; we never switched."""

from __future__ import annotations

import pytest

from pcsa.battle.state import replay
from pcsa.players import encore_switch, mirror, neural_rules
from pcsa.players.neural import NeuralParams
from pcsa.players.neural_rules import Spot
from tests.conftest import REPO
from tests.players.test_disciplined import HITS, PRELUDE, FakeThreat

TEAM = {
    "Primarina": (187, ["moonblast", "sparklingaria", "calmmind", "aquajet"]),
    "Golisopod": (182, ["leechlife", "firstimpression", "liquidation", "swordsdance"]),
    "Mamoswine": (187, ["earthquake", "iceshard", "iciclecrash", "stealthrock"]),
}
FOE = "Primarina"
MOONBLAST, CALM_MIND = "/choose move moonblast", "/choose move calmmind"
MAMOSWINE, GOLISOPOD = "/choose switch Mamoswine", "/choose switch Golisopod"
RANKED = (MOONBLAST, CALM_MIND, MAMOSWINE, GOLISOPOD)  # the network: Moonblast first, then Mamoswine of the switches


def foe(move: str, target: str = "p1a: Primarina", who: str = FOE) -> str:
    return f"|move|p2a: {who}|{move}|{target}"


def ours(move: str, target: str = FOE) -> str:
    return f"|move|p1a: Primarina|{move}|p2a: {target}"


# Battle 35, turns 2-4 (here 1-3) as the live log's protocol has them: Calm Mind into Calm Mind (theirs first), then their
# Encore before our Calm Mind, then Sparkling Aria before our encored Calm Mind; Leftovers rows every turn.
CM_CM = [foe("Calm Mind"), "|-boost|p2a: Primarina|spa|1", ours("Calm Mind"), "|-boost|p1a: Primarina|spa|1"]
ENCORE = [foe("Encore"), "|-start|p1a: Primarina|Encore", ours("Calm Mind"), "|-item|p1a: Primarina|Leftovers"]
ARIA = [
    foe("Sparkling Aria"),
    "|-damage|p1a: Primarina|164/187",
    ours("Calm Mind"),
    "|-item|p1a: Primarina|Leftovers",
]
ENDED = [foe("Calm Mind"), ours("Calm Mind"), "|-end|p1a: Primarina|Encore"]


def battle(turns: list[list[str]], foe_name: str = FOE, spe: int = 100, encored: bool = False, item: str = ""):
    lines = [*PRELUDE, "|switch|p1a: Primarina|Primarina, L50|187/187"]
    lines += [f"|switch|p2a: {foe_name}|{foe_name}, L50|100/100", "|turn|1"]
    for turn, events in enumerate(turns, start=2):
        lines += [*events, f"|turn|{turn}"]
    side = [
        {
            "ident": f"p1: {name}",
            "details": f"{name}, L50",
            "condition": f"{hp}/{hp}",
            "active": name == "Primarina",
            "stats": {"atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": spe},
            "moves": moves,
            "baseAbility": "",
            "item": item if name == "Primarina" else "",
        }
        for name, (hp, moves) in TEAM.items()
    ]
    moves = [
        {"move": m, "id": m, "pp": 10, "maxpp": 10, "target": "normal", "disabled": encored and m != "calmmind"}
        for m in TEAM["Primarina"][1]
    ]
    request = {"side": {"name": "me", "id": "p1", "pokemon": side}, "rqid": 1, "active": [{"moves": moves}]}
    return replay(lines, request)


def spot(b, chosen: str = MOONBLAST, only=None) -> Spot:
    legal = [a for a in mirror.legal_actions(b) if only is None or a.message in only]
    by = {a.message: a for a in legal}
    order = [by[m] for m in RANKED if m in by] + [a for a in legal if a.message not in RANKED]
    return Spot(b, tuple(legal), by.get(chosen, order[0]), order, None, None, None, 0.5)


def decide(s: Spot, rules=("encore_switch",)) -> tuple[str, str, str] | None:
    out = neural_rules.apply(rules, s)
    return (out[0].message, out[1], out[2]) if out else None


def test_the_repo_settings_turn_the_rule_on_first() -> None:
    params = NeuralParams.load(REPO / "config" / "players" / "neural.yaml")
    assert params.rules[0] == "encore_switch"
    assert "encore_switch" not in NeuralParams().rules  # off by default (the simulator's own players)


def test_battle_35_switches_once_the_encorer_has_moved_first_twice() -> None:
    b = battle([CM_CM, ENCORE], encored=True)
    f = encore_switch.facts(b)
    assert (f.encored_us, f.theirs_first, f.ours_first, f.turns) == (True, 2, 0, [1, 2])
    message, rule, why = decide(spot(b, CALM_MIND))
    assert (message, rule) == (MAMOSWINE, "encore_switch")  # the network's best switch
    assert "primarina used Encore and moved first (turns 1, 2)" in why


def test_it_fires_on_the_free_turn_after_the_encore_ended() -> None:
    message, rule, _ = decide(spot(battle([CM_CM, ENCORE, ARIA, ENDED])))  # Moonblast would come after an Encore
    assert (message, rule) == (MAMOSWINE, "encore_switch")


def test_one_observation_is_not_enough_when_a_tie_or_quick_claw_is_possible() -> None:
    assert decide(spot(battle([ENCORE], encored=True), CALM_MIND)) is None  # a mirror: a speed tie is possible
    assert encore_switch.facts(battle([ENCORE])).theirs_first == 1


def test_one_observation_is_enough_when_tie_and_quick_claw_are_ruled_out() -> None:
    reveal = [*ENCORE, "|-item|p2a: Primarina|Leftovers"]  # their item is known: no Quick Claw
    assert decide(spot(battle([reveal], spe=40)))[1] == "encore_switch"  # 40 is below any level-50 Primarina
    assert decide(spot(battle([reveal], spe=100))) is None  # 100 is in Primarina's range: a tie is possible


def test_never_when_we_moved_first_once() -> None:
    ours_first = [ours("Moonblast"), "|-damage|p2a: Primarina|60/100", foe("Calm Mind")]
    assert decide(spot(battle([CM_CM, ours_first, ENCORE, ARIA]))) is None


def test_no_encore_no_rule() -> None:
    assert decide(spot(battle([CM_CM, CM_CM, ARIA]))) is None


def test_the_encorer_must_still_be_on_the_field_since_its_encore() -> None:
    out_and_back = [
        ["|switch|p2a: Hydreigon|Hydreigon, L50|100/100", ours("Calm Mind", "Hydreigon")],
        ["|switch|p2a: Primarina|Primarina, L50|100/100", ours("Calm Mind")],
    ]
    assert decide(spot(battle([CM_CM, ENCORE, *out_and_back, CM_CM, CM_CM]))) is None  # a new stay, no Encore yet


@pytest.mark.parametrize(
    "change",
    [
        ["|-boost|p1a: Primarina|spe|1"],
        ["|-unboost|p2a: Primarina|spe|1"],
        ["|-status|p2a: Primarina|par"],
        ["|-sidestart|p1: me|move: Tailwind"],
        ["|-fieldstart|move: Trick Room"],
        ["|-item|p2a: Primarina|Choice Scarf"],
        ["|-enditem|p2a: Primarina|Sitrus Berry|[eat]"],
        ["|detailschange|p2a: Primarina|Primarina-Mega, L50", "|-mega|p2a: Primarina|Primarina|Primarinite"],
    ],
)
def test_a_speed_change_after_the_observations_drops_them(change: list[str]) -> None:
    turns = [CM_CM, CM_CM, ENCORE]
    turns[-1] = [*turns[-1], *change]
    assert decide(spot(battle(turns))) is None


def test_a_spe_boost_before_the_observations_is_part_of_their_state() -> None:
    boosted = [*CM_CM, "|-boost|p1a: Primarina|spe|1"]
    assert decide(spot(battle([boosted, CM_CM, ENCORE])))[1] == "encore_switch"  # turns 3 and 4 at +1


def test_a_leftovers_row_does_not_drop_them() -> None:
    assert decide(spot(battle([CM_CM, ENCORE, ["|-item|p2a: Primarina|Leftovers"]])))[1] == "encore_switch"


def test_quick_claw_and_priority_turns_do_not_count() -> None:
    claw = ["|-activate|p2a: Primarina|item: Quick Claw", *CM_CM]
    assert encore_switch.facts(battle([claw, ENCORE])).theirs_first == 1
    jet = [ours("Aqua Jet"), foe("Calm Mind")]  # +1 against 0: no order information
    assert encore_switch.facts(battle([jet, ENCORE])).ours_first == 0
    flinched = [foe("Moonblast"), "|cant|p1a: Primarina|flinch"]
    assert encore_switch.facts(battle([flinched, ENCORE])).theirs_first == 1


def test_a_possible_prankster_status_move_does_not_count() -> None:
    def whim(move: str) -> str:
        return foe(move, who="Whimsicott")

    status = [whim("Encore"), ours("Calm Mind", "Whimsicott")]
    attack = [whim("Moonblast"), ours("Calm Mind", "Whimsicott")]
    b = battle([status, status], foe_name="Whimsicott")
    assert encore_switch.facts(b).theirs_first == 0  # Prankster may have moved it first
    assert decide(spot(battle([attack, status, attack], foe_name="Whimsicott")))[1] == "encore_switch"


def test_nothing_without_a_legal_switch_or_when_the_network_switches() -> None:
    b = battle([CM_CM, ENCORE])
    assert decide(spot(b, only={MOONBLAST, CALM_MIND})) is None
    assert decide(spot(b, GOLISOPOD)) is None  # already a switch: the network's own


def test_it_comes_before_the_other_rules(monkeypatch) -> None:
    monkeypatch.setattr(neural_rules, "threat", FakeThreat(dict(HITS)))
    rules = NeuralParams.load(REPO / "config" / "players" / "neural.yaml").rules
    assert decide(spot(battle([CM_CM, ENCORE])), rules)[:2] == (MAMOSWINE, "encore_switch")

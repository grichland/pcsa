"""15 Nov 2026 (run 20261115-034548, battle 7, turns 12-14): "switch to Gholdengo" was chosen three times and Cinderace's
locked Pyro Ball was played each time. The switch row came from the tracked party order ([2, 0, 1]: Cinderace,
Golisopod, Gholdengo) while the game listed Cinderace, Gholdengo, Golisopod (frame 00383), so the taps opened the
fainted Golisopod's summary, the party screen stayed up, it was backed out of and the move timer ran out.
Fixture: tests/fixtures/live_20261115/battle7_switch.json (the log rows, the decisions, the party screen read)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pcsa.common import paths
from pcsa.common.clock import FakeClock
from pcsa.common.config import Config
from pcsa.common.types import PartyMember, Screen, SemanticAction
from pcsa.live import party_switch
from pcsa.live.action_check import verify_last_action
from pcsa.live.fake import ScriptedDevice, Step
from pcsa.phone.log_merge import BattleLog
from pcsa.run.battle_state import BattleRecord
from tests.live.harness import build_phone

FIX = json.loads(
    (paths.repo_root() / "tests" / "fixtures" / "live_20261115" / "battle7_switch.json").read_text(encoding="utf-8")
)
PARTY = tuple(
    PartyMember(row, name, tuple(hp), fainted) for row, (name, hp, fainted) in enumerate(FIX["party_screen_00383"])
)


def _record() -> BattleRecord:
    b = BattleRecord(7, "id", tuple(FIX["roster"]), BattleLog("id"), requested=list(FIX["requested"]))
    b.visible_order = list(FIX["visible_order_before_turn_12"])
    for turn, rows in FIX["log_rows"].items():
        b.log.turns[int(turn)] = rows
    return b


def _last(turn: int, decision: dict) -> dict:
    """b.last_action as turn.py records it, from the saved decision of that turn."""
    switch = decision["order"].removeprefix("/choose switch ") if decision["choice"].startswith("switch") else None
    move = {"pyroball": "Pyro Ball"}.get(decision["order"].removeprefix("/choose move ")) if not switch else None
    return {
        "label": decision["choice"],
        "kind": "switch" if switch else "move",
        "turn": turn,
        "move_name": move,
        "switch_name": switch,
        "active_name": "Cinderace",
    }


@pytest.mark.parametrize("turn", [12, 13, 14])
def test_a_switch_the_game_did_not_play_is_reported_once(turn: int) -> None:
    b = _record()
    b.last_turn = turn
    b.last_action = _last(turn, FIX["decisions"][str(turn)][0])
    assert b.last_action["switch_name"] == "Gholdengo"
    reason = verify_last_action(b)
    assert reason == "chose switch to Gholdengo; the log shows Cinderace used Pyro Ball"
    assert b.failures == [f"turn {turn}: {reason}"]
    assert verify_last_action(b) is None  # each action is checked once


@pytest.mark.parametrize("turn", [10, 11])
def test_the_move_played_as_chosen_is_not_reported(turn: int) -> None:
    b = _record()
    b.last_turn = turn
    b.last_action = _last(turn, FIX["decisions"][str(turn)][0])
    assert verify_last_action(b) is None and b.failures == []


def test_a_switch_that_went_in_and_a_different_move_are_told_apart() -> None:
    b = _record()
    b.last_turn = 9
    b.last_action = {"kind": "switch", "turn": 9, "switch_name": "Golisopod", "active_name": "Gholdengo"}
    assert verify_last_action(b) is None  # turn 9: "Go! Golisopod!"
    b.last_turn = 11
    b.last_action = {"kind": "move", "turn": 11, "move_name": "U-turn", "active_name": "Cinderace", "label": "move3"}
    assert verify_last_action(b) == "chose U-turn; the log shows Cinderace used Pyro Ball"


def test_nothing_is_checked_before_the_turn_is_read() -> None:
    b = _record()
    b.last_turn = 11
    b.last_action = _last(12, FIX["decisions"]["12"][0])
    assert verify_last_action(b) is None and "checked" not in b.last_action


def _pipeline(config: Config, steps: list[Step]):
    clock = FakeClock(1000.0)
    device = ScriptedDevice(steps, config.screen, clock)
    return SimpleNamespace(phone=build_phone(config, device, clock), clock=clock, cfg=config.agent), device


def test_the_tracked_order_pointed_at_the_fainted_member(config: Config) -> None:
    pipeline, _ = _pipeline(config, [Step(Screen.ACTION)])
    plan = pipeline.phone.executor.plan_action(SemanticAction("switch", slot=1), _record().visible_order, None)
    assert plan[1].name == "party_rows2"  # Golisopod's row on frame 00383: what the live run tapped


def test_the_switch_row_is_read_from_the_party_screen(config: Config) -> None:
    steps = [
        Step(Screen.ACTION, accept=("pokemon",)),
        Step(
            Screen.PARTY,
            {"party": PARTY, "focused_row": 0},
            accept=("party_rows", "switch_in"),
            advance_on=("switch_in",),
        ),
        Step(Screen.WAIT),
    ]
    pipeline, device = _pipeline(config, steps)
    b = _record()
    result, taken = party_switch.switch(pipeline, b, 1)  # slot 1: Gholdengo
    assert taken and result.error is None and device.rejected == []
    assert [name for _, name in device.taps] == ["pokemon", "party_rows1", "party_rows1", "switch_in"]
    assert b.visible_order == [2, 1, 0]  # Cinderace, Gholdengo, Golisopod, as the screen shows them


def test_a_party_screen_still_up_is_tried_again_then_backed_out_of(config: Config) -> None:
    steps = [
        Step(Screen.ACTION, accept=("pokemon",)),
        Step(
            Screen.PARTY,
            {"party": PARTY, "focused_row": 1},
            accept=("party_rows", "switch_in", "party_back"),
            advance_on=("party_back",),
        ),
        Step(Screen.ACTION),
    ]
    pipeline, device = _pipeline(config, steps)
    _, taken = party_switch.switch(pipeline, _record(), 1)
    taps = [name for _, name in device.taps]
    assert not taken
    assert taps.count("switch_in") == 2 and taps[-1] == "party_back" and "party_rows2" not in taps


def test_a_target_not_standing_on_the_party_screen_is_not_tapped(config: Config) -> None:
    steps = [
        Step(Screen.ACTION, accept=("pokemon",)),
        Step(Screen.PARTY, {"party": PARTY, "focused_row": 0}, accept=("party_back",)),
        Step(Screen.ACTION),
    ]
    pipeline, device = _pipeline(config, steps)
    _, taken = party_switch.switch(pipeline, _record(), 0)  # slot 0: Golisopod, fainted
    assert not taken and [name for _, name in device.taps] == ["pokemon", "party_back"]

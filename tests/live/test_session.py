"""The session shell on the scripted phone: a battle, outcomes, restarts that carry the battle, SV-7."""

from __future__ import annotations

import json
from pathlib import Path

from pcsa.common.config import Config
from pcsa.common.types import PartyMember, Screen
from pcsa.live.fake import Step
from tests.live.session_harness import TEAM_FIELDS, run_script

BOX = {"name": "x", "center": (1400, 300), "box": (1300, 280, 1500, 320)}
PARTY = (
    PartyMember(0, "Garchomp", (0, 180), fainted=True),
    PartyMember(1, "Primarina", (170, 170)),
    PartyMember(2, "Corviknight", (200, 200)),
)


def lobby_to_team_select() -> list[Step]:
    return [
        Step(Screen.LOBBY, {"team_label": "Team 1"}, accept=("begin_matchmaking",)),
        Step(Screen.MATCHMAKING, after_s=3),
        Step(Screen.TEAM_SELECT, TEAM_FIELDS, accept=("team_rows", "team_done"), advance_on=("team_done",)),
        Step(Screen.TRANSITION, after_s=2),
    ]


def turn() -> list[Step]:
    return [Step(Screen.ACTION, {"active_name": "Garchomp"}, accept=("fight",)), Step(Screen.WAIT, after_s=4)]


def forced() -> list[Step]:
    return [
        Step(Screen.FORCED, {"party": PARTY, "no_back": True}, accept=("party_rows",)),
        Step(Screen.WAIT, after_s=3),
    ]


def end_of_battle(outcome: str = "win") -> list[Step]:
    return [
        Step(Screen.RESULT, {"outcome": outcome}, after_s=2),
        Step(Screen.REWARDS, {"own_rank": {"tier": "Master Ball", "rank": 4, "rating": 1210.0}}, after_s=2),
        Step(Screen.OPTIONS, accept=("continue_battling",)),
        Step(Screen.LOBBY, {"team_label": "Team 1"}),
    ]


def test_one_battle_goes_through_the_runner(config: Config, tmp_path: Path, monkeypatch) -> None:
    steps = lobby_to_team_select() + turn() + forced() + turn() + end_of_battle("win")
    played = run_script(config, steps, tmp_path, monkeypatch)
    assert played.ending == "completed" and played.state.outcomes == ["win"] and played.state.restarts == 0
    assert played.device.rejected == [], played.device.rejected
    battle_id = played.battle(1)["battle_id"]
    assert played.player.calls == [("start", battle_id), ("preview", battle_id), ("end", battle_id, "win")]
    assert [c[0] for c in played.turns.calls] == ["action", "forced", "action"]
    preview = json.loads((played.recorder.dir / "battle-1" / "team_preview.json").read_text(encoding="utf-8"))
    assert preview["decision"]["picks"] == [0, 1, 2] and preview["decision"]["source"] == "player"
    assert preview["request"]["opponent_preview"][0] == "Snorlax"
    assert (
        preview["request"]["team"]["names"][0] == "Garchomp"
        and "Garchomp.item" in preview["request"]["team"]["unread"]
    )
    team_taps = [name for i, name in played.device.taps if i == 2]
    assert team_taps == ["team_rows0", "team_rows1", "team_rows2", "team_done"]


def test_forfeit_and_unknown_outcomes_are_recorded_as_such(config: Config, tmp_path: Path, monkeypatch) -> None:
    steps = (
        lobby_to_team_select()
        + turn()
        + [Step(Screen.FORFEIT_MSG, after_s=2), Step(Screen.OPTIONS, accept=("continue_battling",))]
        + lobby_to_team_select()
        + turn()
        + [
            Step(Screen.COMM_ERROR, {"kind": "comm_error", "dismiss": BOX}, accept=("@dismiss",)),
        ]
        + lobby_to_team_select()
        + turn()
        + end_of_battle("loss")
    )
    played = run_script(config, steps, tmp_path, monkeypatch)
    assert played.state.outcomes == ["opponent_forfeit", "unknown", "loss"]
    assert played.state.completed == 1 and played.state.played == 3
    assert played.battle(1)["outcome_source"] == "forfeit message"
    assert played.battle(2)["outcome"] == "unknown"
    ends = [c[2] for c in played.player.calls if c[0] == "end"]
    assert ends == ["opponent_forfeit", "unknown", "loss"]


def test_restart_mid_battle_carries_the_battle(config: Config, tmp_path: Path, monkeypatch) -> None:
    steps = lobby_to_team_select() + turn() + turn() + end_of_battle("win")
    played = run_script(config, steps, tmp_path, monkeypatch, fail_on={("action", 2)})
    assert played.ending == "completed" and played.state.restarts == 1
    assert played.state.outcomes == ["win"] and played.state.played == 1
    ids = {c[1] for c in played.turns.calls}
    assert len(ids) == 1 and [c[3] for c in played.turns.calls] == [1, 2, 3]
    assert played.turns.calls[2][2] is False  # the first look of a new session is not a repeat
    assert [c[0] for c in played.player.calls] == ["start", "preview", "end"]  # one battle_start only


def test_restart_at_a_forced_switch(config: Config, tmp_path: Path, monkeypatch) -> None:
    steps = lobby_to_team_select() + turn() + forced() + end_of_battle("loss")
    played = run_script(config, steps, tmp_path, monkeypatch, fail_on={("forced", 1)})
    assert played.state.restarts == 1 and played.state.outcomes == ["loss"]
    assert [c[0] for c in played.turns.calls] == ["action", "forced", "forced"]
    assert len({c[1] for c in played.turns.calls}) == 1


def test_restart_at_preview_repairs_the_same_match(config: Config, tmp_path: Path, monkeypatch) -> None:
    stuck = Step(Screen.TEAM_SELECT, TEAM_FIELDS, after_s=103)  # taps do not go through; the watchdog restarts
    steps = lobby_to_team_select()[:2] + [stuck] + lobby_to_team_select()[2:] + turn() + end_of_battle("win")
    played = run_script(config, steps, tmp_path, monkeypatch)
    assert played.state.restarts == 1 and played.state.outcomes == ["win"] and played.state.played == 1
    assert [c[0] for c in played.player.calls] == ["start", "preview", "end"]
    assert [name for i, name in played.device.taps if i == 3] == [
        "team_rows0",
        "team_rows1",
        "team_rows2",
        "team_done",
    ]
    assert any(e["event"] == "team_select_after_restart" and e["same_match"] for e in played.events())


def test_a_battle_already_on_at_start_is_not_adopted(config: Config, tmp_path: Path, monkeypatch) -> None:
    steps = (
        [
            Step(Screen.ACTION, {"active_name": "Garchomp"}, after_s=30),
            Step(Screen.WAIT, after_s=3),
            Step(Screen.FORCED, {"party": PARTY, "no_back": True}, after_s=20),
        ]
        + end_of_battle("loss")[:-1]
        + lobby_to_team_select()
        + turn()
        + end_of_battle("win")
    )
    played = run_script(config, steps, tmp_path, monkeypatch)
    assert played.state.outcomes == ["win"] and played.state.played == 1
    assert [i for i, _ in played.device.taps if i < 6] == [5]  # only Continue Battling on the options screen
    assert [c[0] for c in played.turns.calls] == ["action"]
    assert any(e["event"] == "battle_not_adopted" for e in played.events())


def test_a_pending_battle_from_a_resumed_run_is_dropped(config: Config, tmp_path: Path, monkeypatch) -> None:
    steps = lobby_to_team_select() + turn() + end_of_battle("win")
    played = run_script(config, steps, tmp_path, monkeypatch, pending=1)
    assert played.state.outcomes == ["unknown", "win"] and played.state.played == 2
    assert (played.recorder.dir / "battle-1" / "not_adopted.json").is_file()
    assert played.battle(2)["outcome"] == "win"

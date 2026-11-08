"""End-to-end scenarios on the scripted phone: a battle, a disconnect, an unknown screen, a restart."""

from __future__ import annotations

import json
from pathlib import Path

from pcsa.common import paths
from pcsa.common.config import Config, load_config
from pcsa.common.types import MoveInfo, PartyMember, Screen
from pcsa.live.fake import Step
from pcsa.phone.ocr import TextBox
from pcsa.run.watch import RunReader
from tests.live.harness import ROSTER, run_script

MOVES = tuple(
    MoveInfo(i, n, (10, 10)) for i, n in enumerate(("Earthquake", "Dragon Claw", "Swords Dance", "Protect"))
)
BOX = {"name": "x", "center": (1400, 300), "box": (1300, 280, 1500, 320)}
TEAM_FIELDS = {"roster": ROSTER, "markers": {}, "timer": "0:45", "selected_count": 0}


def lobby_to_team_select() -> list[Step]:
    return [
        Step(Screen.LOBBY, {"team_label": "Team 1"}, accept=("begin_matchmaking",)),
        Step(Screen.MATCHMAKING, after_s=3),
        Step(Screen.TEAM_SELECT, TEAM_FIELDS, accept=("team_rows", "team_done"), advance_on=("team_done",)),
        Step(Screen.TRANSITION, after_s=2),
    ]


def one_turn(turn: int, log_rows: tuple[str, ...]) -> list[Step]:
    """Action screen -> Fight peek -> log read (scripted) -> plan: Fight, move row."""
    hud = {
        "active_name": "Garchomp",
        "active_hp": (180, 180),
        "opponent_name": "Snorlax",
        "move_time": 40,
        "view_log": True,
    }
    return [
        Step(Screen.ACTION, hud, accept=("fight",), log_rows=log_rows, log_turn=turn),
        Step(
            Screen.MOVES,
            {"moves": MOVES, "focused_row": 0, "mega_available": False},
            accept=("move_back",),
            log_rows=log_rows,
            log_turn=turn,
        ),
        Step(Screen.ACTION, hud, accept=("fight",), log_rows=log_rows, log_turn=turn),
        Step(Screen.MOVES, {"moves": MOVES, "focused_row": 0, "mega_available": False}, accept=("move_rows",)),
        Step(Screen.WAIT, {}, accept=("move_rows",), advance_on=(), after_s=4),
    ]


def end_of_battle(outcome: str = "win") -> list[Step]:
    return [
        Step(Screen.RESULT, {"outcome": outcome}, after_s=2),
        Step(
            Screen.REWARDS,
            {"own_rank": {"tier": "Master Ball", "rank": 4, "placement": 900, "rating": 1210.0}},
            after_s=2,
        ),
        Step(Screen.OPTIONS, accept=("continue_battling",)),
        Step(Screen.LOBBY, {"team_label": "Team 1"}),
    ]


TURN1 = ("Go! Garchomp!", "Opponent sent out Snorlax!")
TURN2 = (
    "Move: Garchomp used Earthquake!",
    "HP Lost: Snorlax’s HP: 100% → 55%",
    "Move: The opposing Snorlax used Body Slam!",
    "HP Lost: Garchomp’s HP: 180 → 120",
)


def test_one_battle_start_to_finish(config: Config, tmp_path: Path) -> None:
    steps = lobby_to_team_select() + one_turn(1, TURN1) + one_turn(2, TURN2) + end_of_battle("win")
    state, device, recorder = run_script(config, steps, tmp_path)
    assert state.ending == "completed"  # type: ignore[attr-defined]
    assert state.outcomes == ["win"] and state.completed == 1 and state.restarts == 0
    assert device.rejected == [], device.rejected
    battle = json.loads((recorder.dir / "battle-1" / "battle.json").read_text(encoding="utf-8"))
    assert battle["outcome"] == "win" and len(battle["turns"]) == 2
    decisions = sorted((recorder.dir / "battle-1" / "decisions").glob("*.json"))
    assert [p.name for p in decisions] == ["0000-action.json", "0001-action.json"]
    first = json.loads(decisions[0].read_text(encoding="utf-8"))
    assert first["degrade"] in ("full", "tracked", "screen")
    assert (recorder.dir / "battle-1" / "battle_log.txt").is_file()
    assert state.rank_history[-1]["rating"] == 1210.0


def test_forced_replacement_after_a_faint(config: Config, tmp_path: Path) -> None:
    party = (
        PartyMember(0, "Garchomp", (0, 180), fainted=True),
        PartyMember(1, "Primarina", (170, 170)),
        PartyMember(2, "Corviknight", (200, 200)),
    )
    faint_rows = TURN2 + ("Fainted: Garchomp fainted!",)
    steps = (
        lobby_to_team_select()
        + one_turn(1, TURN1)
        + [
            Step(
                Screen.FORCED,
                {"party": party, "focused_row": None, "no_back": True},
                accept=("party_rows", "switch_in"),
                advance_on=("switch_in",),
                log_rows=faint_rows,
                log_turn=2,
            ),
            Step(Screen.WAIT, {}, after_s=3),
        ]
        + end_of_battle("loss")
    )
    state, device, recorder = run_script(config, steps, tmp_path)
    assert state.outcomes == ["loss"] and device.rejected == []
    forced = json.loads((recorder.dir / "battle-1" / "decisions" / "0001-forced.json").read_text(encoding="utf-8"))
    assert forced["choice"] in ("switch1", "switch2")


def test_forfeit_seen_only_by_the_move_peek_is_recorded(config: Config, tmp_path: Path) -> None:
    """The opponent forfeits while the turn peeks the moves: the message is gone by the next poll."""
    hud = {"active_name": "Garchomp", "active_hp": (180, 180), "opponent_name": "Snorlax", "move_time": 40}
    steps = (
        lobby_to_team_select()
        + one_turn(1, TURN1)
        + [
            Step(Screen.ACTION, hud, accept=("fight",), log_rows=TURN2, log_turn=2),
            # The message outlasts the peek and the blind action taps; the next poll lands on the options screen.
            Step(Screen.FORFEIT_MSG, after_s=5),
            Step(Screen.OPTIONS, accept=("continue_battling",)),
        ]
        + lobby_to_team_select()
        + one_turn(1, TURN1)
        + end_of_battle("win")
    )
    forfeit_step = len(lobby_to_team_select()) + len(one_turn(1, TURN1)) + 1
    state, device, recorder = run_script(config, steps, tmp_path)
    assert {index for index, _, _ in device.rejected} <= {forfeit_step}  # only the blind taps on the message miss
    assert state.outcomes == ["opponent_forfeit", "win"] and state.completed == 1 and state.played == 2
    battle = json.loads((recorder.dir / "battle-1" / "battle.json").read_text(encoding="utf-8"))
    assert battle["outcome"] == "opponent_forfeit" and battle["outcome_source"] == "forfeit message"
    assert RunReader(recorder.dir).state()["session_record"] == {"wins": 1, "losses": 0, "draws": 0, "forfeits": 1}


def test_comm_error_mid_battle_returns_to_lobby_and_plays_on(config: Config, tmp_path: Path) -> None:
    steps = (
        lobby_to_team_select()
        + one_turn(1, TURN1)
        + [
            Step(Screen.COMM_ERROR, {"kind": "comm_error", "dismiss": BOX}, accept=("@dismiss",)),
            Step(Screen.TITLE, {"target": BOX}, accept=("@target",)),
            Step(Screen.HOME, {"target": {**BOX, "box": (1330, 300, 1590, 520)}}, accept=("@target",)),
            Step(Screen.BATTLE_MODES, {"target": {**BOX, "box": (1280, 215, 1800, 285)}}, accept=("@target",)),
            Step(Screen.RANKED_FORMATS, {"target": {**BOX, "box": (1250, 455, 1800, 610)}}, accept=("@target",)),
        ]
        + lobby_to_team_select()
        + one_turn(1, TURN1)
        + end_of_battle("win")
    )
    state, device, recorder = run_script(config, steps, tmp_path)
    assert device.rejected == []
    assert state.outcomes == ["unknown", "win"]  # the interrupted battle is closed, the next one counted
    assert state.completed == 1 and state.played == 2


def test_unknown_screen_is_dismissed_and_the_run_continues(config: Config, tmp_path: Path) -> None:
    steps = (
        [Step(Screen.UNKNOWN, {"dismiss": BOX}, accept=("@dismiss",))]
        + lobby_to_team_select()
        + one_turn(1, TURN1)
        + end_of_battle("win")
    )
    state, device, recorder = run_script(config, steps, tmp_path)
    assert state.outcomes == ["win"] and device.rejected == []
    assert (recorder.dir / "unknown").is_dir() and any((recorder.dir / "unknown").iterdir())


def test_screencap_failure_mid_turn_is_retried(config: Config, tmp_path: Path) -> None:
    steps = lobby_to_team_select() + one_turn(1, TURN1) + end_of_battle("win")
    steps[4] = Step(
        steps[4].screen,
        steps[4].fields,
        accept=steps[4].accept,
        log_rows=steps[4].log_rows,
        log_turn=steps[4].log_turn,
        inject="screencap_fail",
    )
    state, device, recorder = run_script(config, steps, tmp_path)
    assert state.outcomes == ["win"] and device.failed_captures == 1


def test_a_stuck_screen_restarts_the_session_and_the_run_finishes(config: Config, tmp_path: Path) -> None:
    steps = [
        Step(Screen.LOBBY, {"team_label": "Team 1"}, accept=("begin_matchmaking",)),
        Step(Screen.MATCHMAKING, after_s=400),
    ]
    steps += lobby_to_team_select()[2:] + one_turn(1, TURN1) + end_of_battle("win")
    state, device, recorder = run_script(config, steps, tmp_path)
    assert state.restarts >= 1 and state.outcomes == ["win"]
    assert state.ending == "completed"  # type: ignore[attr-defined]


def test_starting_on_the_action_screen_adopts_the_battle(config: Config, tmp_path: Path) -> None:
    steps = one_turn(1, TURN1) + end_of_battle("loss")
    state, device, recorder = run_script(config, steps, tmp_path)
    assert state.outcomes == ["loss"] and device.rejected == []
    battle = json.loads((recorder.dir / "battle-1" / "battle.json").read_text(encoding="utf-8"))
    assert battle["adopted"] is True


def test_every_decision_precedes_its_deadline(config: Config, tmp_path: Path) -> None:
    steps = lobby_to_team_select() + one_turn(1, TURN1) + one_turn(2, TURN2) + end_of_battle("win")
    state, device, recorder = run_script(config, steps, tmp_path)
    battle = json.loads((recorder.dir / "battle-1" / "battle.json").read_text(encoding="utf-8"))
    for turn in battle["turns"]:
        assert turn["first_capture_to_last_tap_s"] <= config.agent.timing.max_turn_s


def events(recorder) -> list[dict]:
    lines = (recorder.dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_disconnect_mid_battle_is_tapped_through_and_the_battle_continues(config: Config, tmp_path: Path) -> None:
    forfeit = TextBox("Forfeit", 0.6, (1080, 470, 1260, 610))  # doubtful OCR, still enough to refuse a blind tap
    steps = (
        lobby_to_team_select()
        + one_turn(1, TURN1)
        + [
            Step(Screen.UNKNOWN, accept=("screen_centre",), note="frozen battle frame"),
            Step(Screen.UNKNOWN, texts=(forfeit,), after_s=5, note="forfeit under the centre"),
            Step(Screen.COMM_ERROR, {"kind": "comm_error"}, accept=("screen_centre",)),
            Step(Screen.DIALOG, {"dismiss": BOX}, accept=("@dismiss",)),
            Step(Screen.TRANSITION, accept=("screen_centre",), note="reconnecting"),
        ]
        + one_turn(2, TURN2)
        + end_of_battle("win")
    )
    state, device, recorder = run_script(config, steps, tmp_path)
    assert state.ending == "completed" and state.restarts == 0  # type: ignore[attr-defined]
    assert state.outcomes == ["win"] and device.rejected == [], device.rejected
    forfeit_step = next(i for i, step in enumerate(steps) if step.note == "forfeit under the centre")
    assert not [name for index, name in device.taps if index == forfeit_step]
    battle = json.loads((recorder.dir / "battle-1" / "battle.json").read_text(encoding="utf-8"))
    assert len(battle["turns"]) == 2  # the second turn was played after the disconnect
    log = events(recorder)
    taps = [e for e in log if e["event"] == "disconnect_tap"]
    assert [e["tap"] for e in taps] == ["centre", "centre", "control", "centre"]
    assert [e["event"] for e in log].count("disconnect_tap_skipped") == 1
    recovered = [e for e in log if e["event"] == "disconnect_recovered"]
    assert len(recovered) == 1 and recovered[0]["outcome"] == "battle_resumed" and recovered[0]["screen"] == "action"
    assert (recorder.dir / "unknown").is_dir() and len(list((recorder.dir / "unknown").glob("*.png"))) >= 2


def test_disconnect_past_the_bound_restarts_the_session(tmp_path: Path) -> None:
    overrides = ["navigator.disconnect_max_taps=4", "navigator.disconnect_settle_s=2"]
    config = load_config(paths.repo_root() / "config" / "agent.yaml", overrides=overrides, environ={})
    steps = (
        lobby_to_team_select()
        + one_turn(1, TURN1)
        + [Step(Screen.UNKNOWN, accept=("screen_centre",), advance_on=(), after_s=25, note="taps change nothing")]
        + lobby_to_team_select()  # the game drops back to the lobby while the restarted session waits
        + one_turn(1, TURN1)
        + end_of_battle("win")
    )
    state, device, recorder = run_script(config, steps, tmp_path)
    assert state.ending == "completed" and state.restarts == 1  # type: ignore[attr-defined]
    assert device.rejected == [], device.rejected
    log = events(recorder)
    assert [e["event"] for e in log].count("disconnect_tap") == 4
    exhausted = [e for e in log if e["event"] == "disconnect_exhausted"]
    assert len(exhausted) == 1 and exhausted[0]["rounds"] == 4
    restart = next(e for e in log if e["event"] == "session_restart")
    assert "disconnect not recovered" in restart["reason"]
    assert state.outcomes == ["win"]  # the restarted session found the lobby and played the next battle

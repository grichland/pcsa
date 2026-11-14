"""End-to-end scenarios on the scripted phone: a battle, a disconnect, an unknown screen, a restart."""

from __future__ import annotations

import json
from pathlib import Path

from pcsa.common import paths
from pcsa.common.config import Config, load_config
from pcsa.common.types import MoveInfo, PartyMember, Screen
from pcsa.live.fake import Step
from pcsa.phone.ocr import TextBox
from tests.live.harness import ROSTER, run_script

MOVES = tuple(
    MoveInfo(i, n, (10, 10)) for i, n in enumerate(("Earthquake", "Dragon Claw", "Swords Dance", "Protect"))
)
BOX = {"name": "x", "center": (1400, 300), "box": (1300, 280, 1500, 320)}
PARTY = (
    PartyMember(0, "Garchomp", (0, 180), fainted=True),
    PartyMember(1, "Primarina", (170, 170)),
    PartyMember(2, "Corviknight", (200, 200)),
)
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


def events(recorder) -> list[dict]:
    lines = (recorder.dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def assert_played_cleanly(recorder) -> None:
    """The player never raised, and every action was chosen on the full tracked battle."""
    failures = [e for e in events(recorder) if e["event"] == "player_failed"]
    assert failures == [], failures[0].get("traceback")
    actions = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(recorder.dir.glob("battle-*/decisions/*-action.json"))
    ]
    assert actions and {a["degrade"] for a in actions} == {"full"}, [a["degrade"] for a in actions]


def forced_steps(party: tuple[PartyMember, ...]) -> list[Step]:
    """One turn, a faint, the forced replacement screen, then a loss."""
    return (
        lobby_to_team_select()
        + one_turn(1, TURN1)
        + [
            Step(
                Screen.FORCED,
                {"party": party, "focused_row": None, "no_back": True},
                accept=("party_rows", "switch_in"),
                advance_on=("switch_in",),
                log_rows=TURN2 + ("Fainted: Garchomp fainted!",),
                log_turn=2,
            ),
            Step(Screen.WAIT, {}, after_s=3),
        ]
        + end_of_battle("loss")
    )


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
    assert_played_cleanly(recorder)
    assert (recorder.dir / "battle-1" / "battle_log.txt").is_file()
    assert state.rank_history[-1]["rating"] == 1210.0


def test_forced_replacement_after_a_faint(config: Config, tmp_path: Path) -> None:
    state, device, recorder = run_script(config, forced_steps(PARTY), tmp_path)
    assert state.outcomes == ["loss"] and device.rejected == []
    forced = json.loads((recorder.dir / "battle-1" / "decisions" / "0001-forced.json").read_text(encoding="utf-8"))
    assert forced["choice"] in ("switch1", "switch2")
    assert_played_cleanly(recorder)


def test_forced_replacement_into_the_first_bench_row(config: Config, tmp_path: Path) -> None:
    """Row 1's Switch In button covers row 2: the tap must count as Switch In, not as a row."""
    party = (PARTY[0], PARTY[1], PartyMember(2, "Corviknight", (0, 200), fainted=True))
    steps = forced_steps(party)
    state, device, recorder = run_script(config, steps, tmp_path)
    assert state.outcomes == ["loss"] and device.rejected == [], device.rejected
    forced = json.loads((recorder.dir / "battle-1" / "decisions" / "0001-forced.json").read_text(encoding="utf-8"))
    assert forced["choice"] == "switch1"
    forced_step = next(i for i, step in enumerate(steps) if step.screen is Screen.FORCED)
    assert [name for i, name in device.taps if i == forced_step] == ["party_rows1", "party_rows1", "switch_in"]


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
    ends = [e["outcome"] for e in events(recorder) if e["event"] == "battle_end"]
    assert ends == ["opponent_forfeit", "win"]


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
    assert_played_cleanly(recorder)


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


def test_party_without_back_is_polled_not_backed_out(config: Config, tmp_path: Path) -> None:
    """14 Nov: a U-turn pivot shows party rows, no Back, nothing fainted. The first poll tapped party_back (a stray
    surface); now it waits one poll and the second sees it as the forced replacement."""
    alive = (PartyMember(0, "Garchomp", (90, 180)), PARTY[1], PARTY[2])
    steps = (
        lobby_to_team_select()
        + one_turn(1, TURN1)
        + [
            Step(
                Screen.PARTY,
                {"party": alive, "focused_row": None, "no_back": True},
                accept=("party_rows", "switch_in"),
                advance_on=("switch_in",),
                log_rows=TURN2,
                log_turn=2,
            ),
            Step(Screen.WAIT, {}, after_s=3),
        ]
        + end_of_battle("win")
    )
    state, device, recorder = run_script(config, steps, tmp_path)
    assert state.outcomes == ["win"] and device.rejected == [], device.rejected
    log = events(recorder)
    assert not [e for e in log if e["event"] == "stray_surface"]
    assert [e["event"] for e in log].count("party_without_back") == 1
    assert "party_back" not in [name for _, name in device.taps]
    assert (recorder.dir / "battle-1" / "decisions" / "0001-forced.json").is_file()


def rated(rating: float) -> dict:
    return {"own_rank": {"tier": "Master Ball", "rank": 3, "placement": 900, "rating": rating}}


def test_unknown_outcome_is_read_from_the_rating_change(config: Config, tmp_path: Path) -> None:
    """14 Nov battles 15 and 36: no result banner was read. The rewards screen first shows the old rating, then
    the new one; a rise is a win."""
    steps = (
        [Step(Screen.LOBBY, {"team_label": "Team 1", **rated(1652.626)}, accept=("begin_matchmaking",))]
        + lobby_to_team_select()[1:]
        + one_turn(1, TURN1)
        + [
            Step(Screen.REWARDS, rated(1652.626), after_s=2),
            Step(Screen.REWARDS, rated(1669.362), after_s=2),
            Step(Screen.OPTIONS, rated(1669.362), accept=("continue_battling",)),
            Step(Screen.LOBBY, {"team_label": "Team 1", **rated(1669.362)}),
        ]
    )
    state, device, recorder = run_script(config, steps, tmp_path)
    assert state.outcomes == ["win"] and state.completed == 1 and device.rejected == []
    battle = json.loads((recorder.dir / "battle-1" / "battle.json").read_text(encoding="utf-8"))
    assert battle["outcome"] == "win"
    assert battle["outcome_source"] == "rating change 1652.626 -> 1669.362"


def test_no_rating_before_the_battle_leaves_the_outcome_unknown(config: Config, tmp_path: Path) -> None:
    steps = (
        lobby_to_team_select()
        + one_turn(1, TURN1)
        + [
            Step(Screen.REWARDS, rated(1640.0), after_s=2),
            Step(Screen.OPTIONS, rated(1640.0), accept=("continue_battling",)),
        ]
        + lobby_to_team_select()
        + one_turn(1, TURN1)
        + end_of_battle("win")
    )
    state, device, recorder = run_script(config, steps, tmp_path)
    assert state.outcomes == ["unknown", "win"] and device.rejected == []


def test_a_result_banner_ending_a_disconnect_is_read(monkeypatch) -> None:
    """14 Nov battle 15: the disconnect taps ended on the WON banner and the next poll was already past it, so the
    banner the recovery saw is the outcome."""
    from types import SimpleNamespace

    from pcsa.common.types import Observation
    from pcsa.live import handlers, reconnect
    from pcsa.phone.ocr import REFERENCE

    seen = []
    monkeypatch.setattr(handlers, "finalize", lambda s, outcome: seen.append(outcome))
    session = SimpleNamespace(battle=SimpleNamespace(index=15))
    banner = Observation(Screen.RESULT, REFERENCE, frozenset(), {"outcome": "win"}, "WON/LOST banner")
    reconnect.resolved(session, banner, 3, 13.7)
    assert seen == ["win"]
    reconnect.resolved(session, Observation(Screen.ACTION, REFERENCE, frozenset(), {}, "Fight and Pokemon"), 1, 4.0)
    assert seen == ["win"]  # a resumed battle is not finalized

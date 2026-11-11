"""A battle picked up again (the player's memory rebuilt from the decision records), the decision event without
nested notes, and the battle log files written while the battle is on. All on the scripted phone."""

from __future__ import annotations

import json
from pathlib import Path

from pcsa.battle import state as tracking
from pcsa.common import paths
from pcsa.common.config import Config, load_config
from pcsa.common.types import Screen
from pcsa.live import restore
from pcsa.live.decide import event_notes
from pcsa.live.fake import Step
from pcsa.players.heuristic import HeuristicPlayer
from tests.live.harness import run_script
from tests.live.test_harness import (
    PARTY,
    TURN1,
    TURN2,
    end_of_battle,
    events,
    forced_steps,
    lobby_to_team_select,
    one_turn,
)

TURN3 = ("Move: Garchomp used Dragon Claw!", "HP Lost: Snorlax’s HP: 55% → 20%")
MATRIX = {"rows": ["a", "b"], "cells": [[0.1, 0.9], [0.5, 0.5]], "probs": [0.25] * 11}


class FirstMove(HeuristicPlayer):
    """Always the first move row: the scripted turns accept a move tap, whatever the tracked battle knows."""

    def choose_action(self, view, deadline):
        return next(a for a in view.legal if a.action.kind == "move" and not a.action.mega)


class Remembering(FirstMove):
    """A player with a memory to restore: keeps what each `restore` call was given."""

    def __init__(self) -> None:
        super().__init__()
        self.restored: list[list] = []

    def restore(self, views) -> int:
        self.restored.append(list(views))
        return len(self.restored[-1])


def three_turns() -> list[Step]:
    return lobby_to_team_select() + one_turn(1, TURN1) + one_turn(2, TURN2) + one_turn(3, TURN3) + end_of_battle()


def restart_mid_battle() -> tuple[Config, list[Step]]:
    """One turn, a disconnect the session cannot tap through (the supervisor restarts it), the battle still on."""
    overrides = ["navigator.disconnect_max_taps=4", "navigator.disconnect_settle_s=2"]
    config = load_config(paths.repo_root() / "config" / "agent.yaml", overrides=overrides, environ={})
    stuck = Step(Screen.UNKNOWN, accept=("screen_centre",), advance_on=(), after_s=25, note="taps change nothing")
    return config, lobby_to_team_select() + one_turn(1, TURN1) + [stuck] + one_turn(2, TURN2) + end_of_battle("win")


def decision_files(recorder) -> list[Path]:
    return sorted((recorder.dir / "battle-1" / "decisions").glob("*.json"))


# Views from records ---------------------------------------------------------------------------------------
def test_views_rebuilt_from_the_records_are_the_original_decisions(config: Config, tmp_path: Path) -> None:
    state, device, recorder = run_script(config, forced_steps(PARTY), tmp_path)
    directory = recorder.dir / "battle-1" / "decisions"
    owner = json.loads((recorder.dir / "battle-1" / "battle_id.json").read_text(encoding="utf-8"))["battle_id"]
    records, kinds = restore.load_records(directory, owner)
    assert kinds == ["action", "forced"] and [r["sequence"] for r in records] == [0, 1]
    assert {r["schema"] for r in records} == {"pcsa-decision/2"} and {r["battle_id"] for r in records} == {owner}
    views = restore.rebuild_views(records, owner)
    assert len(views) == 2
    for record, view in zip(records, views, strict=True):
        assert [a.message for a in view.legal] == record["legal"]
        assert view.battle.turn == record["turn"] and str(view.degrade) == record["degrade"]
        assert view.screen.forced == (record["kind"] == "forced")
        text = (directory / f"{record['sequence']:04d}-{record['kind']}.txt").read_text(encoding="utf-8")
        assert tracking.battle_text(view.battle, record["revealed_abilities"]) == text  # the same Battle, in full
    assert [v.battle.turn for v in views] == [1, 1]  # the forced replacement does not visit the log
    assert [v.battle.battle_tag for v in views] == [f"{owner}-0", f"{owner}-1"]


def test_a_broken_record_ends_the_list(config: Config, tmp_path: Path) -> None:
    state, device, recorder = run_script(config, three_turns(), tmp_path)
    files = decision_files(recorder)
    assert [p.name for p in files] == ["0000-action.json", "0001-action.json", "0002-action.json"]
    records, _ = restore.load_records(files[0].parent)
    assert len(restore.rebuild_views(records, "b")) == 3
    damaged = [dict(r) for r in records]
    damaged[1]["legal"] = ["/choose move splash"]  # readable, but not what the rebuilt battle offers
    assert [v.battle.turn for v in restore.rebuild_views(damaged, "b")] == [1]
    screen_only = [dict(r) for r in records]
    screen_only[1].update(degrade="screen", protocol=None, request=None)  # the player was shown no Battle: skipped
    assert [v.battle.turn for v in restore.rebuild_views(screen_only, "b")] == [
        records[0]["turn"],
        records[2]["turn"],
    ]
    files[1].write_text('{"schema": "pcsa-decision/2", "sequence": 1, "protocol": [', encoding="utf-8")
    records, kinds = restore.load_records(files[0].parent)
    assert [r["sequence"] for r in records] == [0] and kinds == ["action", "action", "action"]


def test_records_of_another_battle_in_the_folder_do_not_count(config: Config, tmp_path: Path) -> None:
    state, device, recorder = run_script(config, three_turns(), tmp_path)
    files = decision_files(recorder)
    stale = json.loads(files[2].read_text(encoding="utf-8"))
    stale["battle_id"] = "an-earlier-battle"
    files[2].write_text(json.dumps(stale), encoding="utf-8")
    owner = json.loads(files[0].read_text(encoding="utf-8"))["battle_id"]
    records, kinds = restore.load_records(files[0].parent, owner)
    assert [r["sequence"] for r in records] == [0, 1] and kinds == ["action", "action"]


# Resuming ---------------------------------------------------------------------------------------------------
def test_a_resumed_battle_restores_the_player_once_and_keeps_its_records(tmp_path: Path) -> None:
    config, steps = restart_mid_battle()
    player = Remembering()
    state, device, recorder = run_script(config, steps, tmp_path, player=player)
    assert state.ending == "completed" and state.restarts == 1 and state.outcomes == ["win"]  # type: ignore[attr-defined]
    assert len(player.restored) == 1 and [v.battle.turn for v in player.restored[0]] == [1]
    first, second = (json.loads(p.read_text(encoding="utf-8")) for p in decision_files(recorder))
    assert [first["sequence"], second["sequence"]] == [0, 1]  # the numbering went on: record 0 was not overwritten
    assert first["battle_id"] == second["battle_id"]
    assert [a.message for a in player.restored[0][0].legal] == first["legal"]
    log = events(recorder)
    resumed = [e for e in log if e["event"] == "battle_resumed"]
    assert len(resumed) == 1
    assert {k: resumed[0][k] for k in ("records", "readable", "views", "restored")} == dict.fromkeys(
        ("records", "readable", "views", "restored"), 1
    )
    starts = [e for e in log if e["event"] == "battle_start"]
    assert [e["adopted"] for e in starts] == [False, True] and starts[0]["battle_id"] == starts[1]["battle_id"]
    assert log.index(resumed[0]) < max(i for i, e in enumerate(log) if e["event"] == "decision")


def test_a_player_without_restore_is_left_alone(tmp_path: Path) -> None:
    config, steps = restart_mid_battle()
    state, device, recorder = run_script(config, steps, tmp_path, player=FirstMove())
    assert state.restarts == 1 and state.outcomes == ["win"]
    assert [p.name for p in decision_files(recorder)] == ["0000-action.json", "0001-action.json"]
    resumed = [e for e in events(recorder) if e["event"] == "battle_resumed"]
    assert len(resumed) == 1 and resumed[0]["player_restores"] is False and resumed[0]["restored"] == 0
    assert not [e for e in events(recorder) if e["event"] in ("memory_restore_failed", "player_failed")]


def test_a_failing_restore_does_not_stop_the_battle(tmp_path: Path) -> None:
    class Failing(FirstMove):
        def restore(self, views) -> int:
            raise RuntimeError("no memory")

    config, steps = restart_mid_battle()
    state, device, recorder = run_script(config, steps, tmp_path, player=Failing())
    assert state.outcomes == ["win"] and len(decision_files(recorder)) == 2
    assert [e["event"] for e in events(recorder)].count("memory_restore_failed") == 1


def test_no_restore_on_a_normal_battle_start_or_after_a_disconnect(config: Config, tmp_path: Path) -> None:
    player = Remembering()
    rejoin = [Step(Screen.UNKNOWN, accept=("screen_centre",), note="frozen battle frame")]
    steps = lobby_to_team_select() + one_turn(1, TURN1) + rejoin + one_turn(2, TURN2) + end_of_battle("win")
    state, device, recorder = run_script(config, steps, tmp_path, player=player)
    assert state.outcomes == ["win"] and state.restarts == 0 and len(decision_files(recorder)) == 2
    assert [e["outcome"] for e in events(recorder) if e["event"] == "disconnect_recovered"] == ["battle_resumed"]
    assert player.restored == []  # the same session and player went on: nothing to give back


def test_a_battle_lost_before_the_lobby_is_not_resumed_by_the_next_one(tmp_path: Path) -> None:
    """The restarted session finds the lobby: the next battle reuses the folder and must not inherit its records."""
    config, _ = restart_mid_battle()
    stuck = Step(Screen.UNKNOWN, accept=("screen_centre",), advance_on=(), after_s=25)
    lobby = Step(Screen.LOBBY, {"team_label": "Team 1"}, after_s=1)  # seen, then a battle is already on
    steps = lobby_to_team_select() + one_turn(1, TURN1) + [stuck, lobby] + one_turn(1, TURN1) + end_of_battle("win")
    player = Remembering()
    state, device, recorder = run_script(config, steps, tmp_path, player=player)
    assert state.restarts == 1 and state.outcomes == ["win"]
    assert player.restored == []
    assert [e["battle"] for e in events(recorder) if e["event"] == "pending_battle_dropped"] == [1]


# The decision event ---------------------------------------------------------------------------------------
def test_event_notes_keep_scalars_and_short_lists() -> None:
    notes = {"rule": "rl", "p": 0.5, "matrix": MATRIX, "probs": [0.1] * 17, "top": [1, 2, 3], "why": None}
    assert event_notes(notes) == {"rule": "rl", "p": 0.5, "top": [1, 2, 3], "why": None}


def test_a_matrix_note_is_in_the_record_and_not_in_the_event(config: Config, tmp_path: Path) -> None:
    class WithMatrix(HeuristicPlayer):
        def notes(self) -> dict:
            return {**super().notes(), "matrix": MATRIX, "decisions": 3, "long": list(range(40))}

    steps = lobby_to_team_select() + one_turn(1, TURN1) + end_of_battle("win")
    state, device, recorder = run_script(config, steps, tmp_path, player=WithMatrix())
    record = json.loads(decision_files(recorder)[0].read_text(encoding="utf-8"))
    assert record["notes"]["matrix"] == MATRIX and record["notes"]["long"] == list(range(40))
    decisions = [e for e in events(recorder) if e["event"] == "decision"]
    assert [e["kind"] for e in decisions] == ["team", "action"]
    for event in decisions:
        assert "matrix" not in event and "long" not in event and event["decisions"] == 3


# The log files during the battle --------------------------------------------------------------------------
LOG_FILES = ("log.txt", "log.jsonl", "coverage.json", "battle_log.txt")


def test_the_log_files_are_written_while_the_battle_is_on(config: Config, tmp_path: Path) -> None:
    folder = tmp_path / "run" / "battle-1"
    seen: list[dict[str, int]] = []

    class Watching(HeuristicPlayer):
        def choose_action(self, view, deadline):
            seen.append({name: (folder / name).stat().st_size for name in LOG_FILES if (folder / name).is_file()})
            assert not (folder / "battle.json").exists()  # the battle has not ended
            return super().choose_action(view, deadline)

    state, device, recorder = run_script(config, three_turns(), tmp_path, player=Watching())
    assert state.outcomes == ["win"] and len(seen) == 3
    assert all(set(sizes) == set(LOG_FILES) for sizes in seen), seen
    for name in ("log.txt", "log.jsonl", "battle_log.txt"):
        assert seen[0][name] < seen[1][name] < seen[2][name], (name, seen)
    assert "Dragon Claw" in (folder / "log.txt").read_text(encoding="utf-8")
    final = (folder / "battle_log.txt").read_text(encoding="utf-8")
    assert "outcome: win" in final and seen[2]["battle_log.txt"] <= len(final.encode("utf-8"))


def test_an_unchanged_log_is_not_saved_again_and_a_failed_save_is_only_logged(config: Config, tmp_path: Path) -> None:
    from pcsa.live.log_save import LogSaver
    from pcsa.phone.log_merge import BattleLog
    from pcsa.run.battle_state import BattleRecord
    from pcsa.run.recorder import Recorder

    recorder = Recorder(tmp_path / "run")
    b = BattleRecord(1, "id", (), BattleLog("id"))
    saver = LogSaver(recorder, seed=1)
    b.log.turns[1] = list(TURN1)
    assert saver.after_visit(b, None) is True and saver.after_visit(b, None) is False
    b.log.turns[2] = list(TURN2)
    assert saver.after_visit(b, None) is True
    assert "Earthquake" in (recorder.dir / "battle-1" / "log.txt").read_text(encoding="utf-8")
    b.log.turns[3] = list(TURN3)
    b.log.save = None  # type: ignore[method-assign, assignment]  # the write fails
    assert saver.after_visit(b, None) is False
    recorder.close()

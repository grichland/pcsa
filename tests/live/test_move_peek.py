"""`timing.move_peek: when_needed` on the scripted phone: turns without the look at the move list, and every
reason that brings the look back for one turn. The default (`always`) is covered by test_harness.py."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pcsa.battle.stats import hp_value, stat_value
from pcsa.common import paths
from pcsa.common.config import Config, TimingConfig, load_config
from pcsa.common.errors import ConfigError
from pcsa.common.types import MoveInfo, Screen
from pcsa.live import move_peek
from pcsa.live.fake import Step
from pcsa.phone.log_merge import BattleLog
from pcsa.run.battle_state import BattleRecord
from tests.live import harness
from tests.live.test_harness import end_of_battle, events, lobby_to_team_select, one_turn

SHEET = ("Earthquake", "Draco Meteor", "Swords Dance", "Protect")  # Champions PP 12, 8, 20, 8
BASE = {"hp": 108, "atk": 130, "def": 95, "spa": 80, "spd": 85, "spe": 102}  # Garchomp
STATS = {s: hp_value(v, 0) if s == "hp" else stat_value(v, 0) for s, v in BASE.items()}
HUD = {
    "active_name": "Garchomp",
    "active_hp": (STATS["hp"], STATS["hp"]),
    "opponent_name": "Snorlax",
    "move_time": 40,
}
START = ("Go! Garchomp!", "Opponent sent out Snorlax!")
QUAKE = ("Move: Garchomp used Earthquake!", "HP Lost: Snorlax’s HP: 100% → 70%")
MEGA = (
    "Garchomp’s Garchompite is reacting to pcsa’s Omni Ring!",
    "Mega Evolution: Garchomp has Mega Evolved into Mega Garchomp!",
)
EARTHQUAKE, EARTHQUAKE_MEGA = "/choose move earthquake", "/choose move earthquake mega"
METEOR = "/choose move dracometeor"
LOOKED = ["fight", "move_back", "fight", "move_rows0", "move_rows0"]
NOT_LOOKED = ["fight", "move_rows0", "move_rows0"]
OVERRIDES = ["policy.neural_checkpoint=assets/models/__absent_in_tests__.pt", "timing.move_peek=when_needed"]


class Scripted:
    """Chooses the listed orders in turn (the first move when its order is not legal) and keeps what it was offered."""

    name = "scripted"

    def __init__(self, *wanted: str) -> None:
        self.wanted = list(wanted)
        self.offered: list[list[str]] = []

    def params(self) -> dict:
        return {}

    def notes(self) -> dict:
        return {}

    def choose_team(self, view, deadline) -> list[int]:
        return [0, 1, 2]

    def choose_action(self, view, deadline):
        self.offered.append([a.message for a in view.legal])
        want = self.wanted.pop(0) if self.wanted else EARTHQUAKE
        chosen = [a for a in view.legal if a.message == want]
        return chosen[0] if chosen else next(a for a in view.legal if a.action.kind == "move")


class StopLogReader(harness.ScriptedLogReader):
    """The harness log reader, reporting how the step's log read ended (Step.log_stop)."""

    def collect_log(self, b, visit: int, deadline):
        step = self.device.step
        info = super().collect_log(b, visit, deadline)
        if step.log_turn is not None:
            info.stop_reason = step.log_stop
        return info


@pytest.fixture
def when_needed(monkeypatch) -> Config:
    """The setting switched on, and a log reader that reports how its read ended."""
    monkeypatch.setattr(harness, "ScriptedLogReader", StopLogReader)
    return load_config(paths.repo_root() / "config" / "agent.yaml", OVERRIDES, environ={})


def sheet(**garchomp) -> dict:
    team = copy.deepcopy(harness.TEAM)
    team["members"][0].update(moves=SHEET, item="Garchompite", **garchomp)
    return team


@pytest.fixture
def fast(when_needed: Config, monkeypatch) -> Config:
    """when_needed with a team sheet that has Garchomp's moves, its Mega stone, and stats its Mega form can be
    worked out from (no stat points, neutral nature)."""
    monkeypatch.setattr(harness, "TEAM", sheet(stats=STATS))
    return when_needed


def shown(*pp: int, mega: bool = False) -> dict:
    """The move list as a look reads it, with the given PP left per slot; no row is focused."""
    maximum = (12, 8, 20, 8)
    moves = tuple(MoveInfo(i, n, (pp[i], maximum[i])) for i, n in enumerate(SHEET))
    return {"moves": moves, "focused_row": None, "mega_available": mega}


def no_look(turn: int, rows: tuple[str, ...], *, focused: bool = False) -> list[Step]:
    """Action screen -> log -> Fight, (Mega), the move row twice. `focused`: the row submits on its first tap and
    the second lands on the waiting screen; otherwise the first tap focuses the row and the second submits."""
    action = Step(Screen.ACTION, HUD, accept=("fight",), log_rows=rows, log_turn=turn)
    first = Step(Screen.MOVES, accept=("mega", "move_rows"), advance_on=("move_rows",))
    if focused:
        return [action, first, Step(Screen.WAIT, accept=("move_rows",), advance_on=(), after_s=12)]
    return [action, first, Step(Screen.MOVES, accept=("move_rows",)), Step(Screen.WAIT, after_s=12)]


def look(turn: int, rows: tuple[str, ...], seen: dict, *, stop: str = "top_confirmed") -> list[Step]:
    """Today's turn: Fight, read the list, Back, then Fight, (Mega), the row twice (no row is focused)."""
    log = {"log_rows": rows, "log_turn": turn, "log_stop": stop}
    return [
        Step(Screen.ACTION, HUD, accept=("fight",), **log),
        Step(Screen.MOVES, seen, accept=("move_back",), **log),
        Step(Screen.ACTION, HUD, accept=("fight",), **log),
        Step(Screen.MOVES, seen, accept=("mega", "move_rows"), advance_on=("move_rows",)),
        Step(Screen.MOVES, seen, accept=("move_rows",)),
        Step(Screen.WAIT, after_s=12),
    ]


def decisions_of(recorder) -> list[dict]:
    files = sorted(recorder.dir.glob("battle-*/decisions/*-action.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def play(config: Config, tmp_path: Path, turns: list[list[Step]], player: Scripted):
    """Run the turns as one battle: (the taps of each turn, the action decisions, the recorder)."""
    head = lobby_to_team_select()
    steps = head + [step for turn in turns for step in turn] + end_of_battle("win")
    state, device, recorder = harness.run_script(config, steps, tmp_path, player=player)
    assert device.rejected == [], device.rejected  # every tap landed on a control its screen has
    assert state.outcomes == ["win"] and state.completed == 1 and state.restarts == 0
    taps, start = [], len(head)
    for turn in turns:
        taps.append([name for index, name in device.taps if start <= index < start + len(turn)])
        start += len(turn)
    return taps, decisions_of(recorder), recorder


def pp(decision: dict) -> dict[str, int]:
    return {m["id"]: m["pp"] for m in decision["request"]["active"][0]["moves"]}


def mega_offered(player: Scripted) -> list[bool]:
    return [any(" mega" in message for message in offered) for offered in player.offered]


def test_the_setting_is_when_needed_in_agent_yaml_and_rejects_other_values(config: Config) -> None:
    assert config.agent.timing.move_peek == "when_needed"  # agent.yaml since 11 Nov; the code's default stays always
    assert TimingConfig().move_peek == "always"
    with pytest.raises(ConfigError, match="timing.move_peek"):
        load_config(paths.repo_root() / "config" / "agent.yaml", ["timing.move_peek=sometimes"], environ={})


def test_always_records_the_look_on_every_turn(tmp_path: Path) -> None:
    config = load_config(
        paths.repo_root() / "config" / "agent.yaml", [*OVERRIDES[:1], "timing.move_peek=always"], environ={}
    )
    steps = lobby_to_team_select() + one_turn(1, START) + one_turn(2, QUAKE) + end_of_battle("win")
    state, device, recorder = harness.run_script(config, steps, tmp_path)
    assert state.outcomes == ["win"] and device.rejected == []
    assert [d["peek"] for d in decisions_of(recorder)] == ["always", "always"]
    assert [e["peek"] for e in events(recorder) if e["event"] == "turn"] == ["always", "always"]
    assert [name for _, name in device.taps].count("move_back") == 2


def test_a_battle_is_played_to_the_end_without_a_look(fast: Config, tmp_path: Path) -> None:
    failed = (QUAKE[0], "But it failed!")
    turns = [no_look(1, START), no_look(2, QUAKE, focused=True), no_look(3, failed), no_look(4, QUAKE)]
    taps, decisions, recorder = play(fast, tmp_path, turns, Scripted())
    assert taps == [NOT_LOOKED] * 4  # Fight, the row twice, and never Back: the move list was not read
    assert [d["peek"] for d in decisions] == ["skipped"] * 4
    assert [e["peek"] for e in events(recorder) if e["event"] == "turn"] == ["skipped"] * 4
    assert {d["degrade"] for d in decisions} == {"full"} and all(d["order"] == EARTHQUAKE for d in decisions)
    # The request has today's shape: four rows named by the sheet, PP from the log count (a failed move spends PP).
    assert [pp(d)["earthquake"] for d in decisions] == [12, 11, 10, 9]
    assert pp(decisions[0]) == {"earthquake": 12, "dracometeor": 8, "swordsdance": 20, "protect": 8}
    rows = decisions[0]["request"]["active"][0]["moves"]
    assert [r["move"] for r in rows] == list(SHEET) and all(r["maxpp"] and not r["disabled"] for r in rows)
    assert "canMegaEvo" in decisions[0]["request"]["active"][0]  # the stone is held and no Mega Evolution yet


def test_mega_is_tapped_once_and_not_offered_after_the_log_shows_it(fast: Config, tmp_path: Path) -> None:
    turns = [no_look(1, START), no_look(2, MEGA + QUAKE), no_look(3, QUAKE)]
    player = Scripted(EARTHQUAKE_MEGA)
    taps, decisions, _ = play(fast, tmp_path, turns, player)
    assert taps == [["fight", "mega", "move_rows0", "move_rows0"], NOT_LOOKED, NOT_LOOKED]
    assert [d["peek"] for d in decisions] == ["skipped"] * 3  # the log's Mega Evolution row confirms it: no look
    assert [d["order"] for d in decisions] == [EARTHQUAKE_MEGA, EARTHQUAKE, EARTHQUAKE]
    assert ["canMegaEvo" in d["request"]["active"][0] for d in decisions] == [True, False, False]
    assert mega_offered(player) == [True, False, False]
    assert {d["degrade"] for d in decisions} == {"full"}


@pytest.mark.parametrize("button", [False, True])
def test_a_mega_the_log_does_not_show_is_settled_by_a_look(fast: Config, tmp_path: Path, button: bool) -> None:
    """Mega was chosen and the next log has no Mega Evolution row: the Mega button decides, as it does today."""
    turns = [no_look(1, START), look(2, QUAKE, shown(11, 8, 20, 8, mega=button)), no_look(3, QUAKE)]
    player = Scripted(EARTHQUAKE_MEGA)
    taps, decisions, _ = play(fast, tmp_path, turns, player)
    assert [d["peek"] for d in decisions] == ["skipped", "mega_unconfirmed", "skipped"]
    assert taps[1:] == [LOOKED, NOT_LOOKED]
    # Button gone: it happened, and Mega is not offered again. Button still there: it did not, and it still is.
    assert mega_offered(player) == [True, button, button]


def test_a_doubted_log_read_looks_for_that_turn_and_keeps_what_the_list_showed(fast: Config, tmp_path: Path) -> None:
    """The log read of turn 2 ran out of time and lost a `used Earthquake` row: the list shows one PP fewer."""
    turns = [
        no_look(1, START),
        look(2, QUAKE, shown(10, 8, 20, 8), stop="deadline"),
        no_look(3, QUAKE),
        no_look(4, QUAKE),
    ]
    taps, decisions, _ = play(fast, tmp_path, turns, Scripted())
    assert [d["peek"] for d in decisions] == ["skipped", "log_stop_deadline", "skipped", "skipped"]
    assert taps == [NOT_LOOKED, LOOKED, NOT_LOOKED, NOT_LOOKED]
    # The count says 11, 10, 9; the list said 10, and the turns after it stay one lower than the count.
    assert [pp(d)["earthquake"] for d in decisions] == [12, 10, 9, 8]


def test_taps_that_did_not_go_through_look_on_the_retry(fast: Config, tmp_path: Path) -> None:
    """The action screen is back on the next poll (known before the log is read), and once more after a wait
    (known from the log: no new turn). Both retries look first; the turn after them does not."""
    lost = [
        Step(Screen.ACTION, HUD, accept=("fight",), log_rows=START, log_turn=1),
        Step(Screen.MOVES, accept=("move_rows",), advance_on=(), after_s=5, note="the row taps change nothing"),
    ]
    waiting = Step(Screen.WAIT, accept=("move_rows",), advance_on=(), after_s=12, note="then the action screen again")
    lost_again = [*look(1, START, shown(12, 8, 20, 8))[:4], waiting]
    turns = [lost, lost_again, look(1, START, shown(12, 8, 20, 8)), no_look(2, QUAKE)]
    taps, decisions, _ = play(fast, tmp_path, turns, Scripted())
    assert [d["peek"] for d in decisions] == ["skipped", "retry", "retry", "skipped"]
    assert [d["same_turn"] for d in decisions] == [False, True, True, False]
    assert [d["refused"] for d in decisions] == [[], [0], [0, 1], []]  # a refused move is not tried again that turn
    retry = [["fight", "move_back", "fight", f"move_rows{row}", f"move_rows{row}"] for row in (1, 2)]
    assert taps == [NOT_LOOKED, *retry, NOT_LOOKED]


def test_pressure_is_counted_and_a_low_count_looks_so_an_empty_move_is_never_offered(
    fast: Config, tmp_path: Path
) -> None:
    start = ("Go! Garchomp!", "Opponent sent out Corviknight!", "The opposing Corviknight is exerting its pressure!")
    meteor = ("Move: Garchomp used Draco Meteor!", "HP Lost: Corviknight’s HP: 100% → 80%")
    quake = ("Move: Garchomp used Earthquake!", "HP Lost: Corviknight’s HP: 80% → 60%")
    turns = [
        no_look(1, start),
        no_look(2, meteor),
        look(3, meteor, shown(12, 0, 20, 8)),  # counted 4 of 8; the list shows it empty (uses the log never had)
        look(4, quake, shown(10, 0, 20, 8)),
    ]
    player = Scripted(METEOR, METEOR, METEOR, METEOR)
    taps, decisions, _ = play(fast, tmp_path, turns, player)
    assert [d["peek"] for d in decisions] == ["skipped", "skipped", "low_pp", "low_pp"]
    assert [pp(d)["dracometeor"] for d in decisions] == [8, 6, 0, 0]  # Pressure: two PP a use
    assert [METEOR in offered for offered in player.offered] == [True, True, False, False]
    assert [d["order"] for d in decisions] == [METEOR, METEOR, EARTHQUAKE, EARTHQUAKE]
    assert taps == [["fight", "move_rows1", "move_rows1"]] * 2 + [LOOKED] * 2


def test_a_team_sheet_without_move_names_looks_every_turn(when_needed: Config, tmp_path: Path) -> None:
    turns = [look(1, START, shown(12, 8, 20, 8)), look(2, QUAKE, shown(11, 8, 20, 8))]
    taps, decisions, _ = play(when_needed, tmp_path, turns, Scripted())  # the harness sheet names no moves
    assert [d["peek"] for d in decisions] == ["sheet", "sheet"] and {d["degrade"] for d in decisions} == {"full"}


def test_a_battle_picked_up_midway_looks_every_turn(fast: Config, tmp_path: Path) -> None:
    turns = [look(1, START, shown(12, 8, 20, 8)), look(2, QUAKE, shown(11, 8, 20, 8))]
    steps = [step for turn in turns for step in turn] + end_of_battle("loss")  # starts on an action screen
    state, device, recorder = harness.run_script(fast, steps, tmp_path, player=Scripted())
    assert state.outcomes == ["loss"] and device.rejected == []
    assert [d["peek"] for d in decisions_of(recorder)] == ["adopted", "adopted"]


def test_a_battle_without_a_team_sheet_looks_before_the_log() -> None:
    b = BattleRecord(1, "id", harness.ROSTER, BattleLog("id"), requested=[0, 1, 2])
    assert move_peek.why_before_log("when_needed", b, None, False) == "untracked"  # decided from the screen alone
    assert move_peek.why_before_log("always", b, None, False) == "always"


def test_a_request_the_battle_rejects_looks_after_all(when_needed: Config, tmp_path: Path, monkeypatch) -> None:
    """The harness stats fit no Garchomp, so after the Mega Evolution the Battle rejects the request: the decision
    is made from the screen alone, and that needs the move list."""
    monkeypatch.setattr(harness, "TEAM", sheet())
    turns = [no_look(1, START), look(2, MEGA + QUAKE, shown(11, 8, 20, 8))]
    taps, decisions, _ = play(when_needed, tmp_path, turns, Scripted(EARTHQUAKE_MEGA))
    assert [d["peek"] for d in decisions] == ["skipped", "untracked"]
    assert [d["degrade"] for d in decisions] == ["full", "screen"]
    assert taps[1] == LOOKED


def test_a_move_refused_by_the_game_is_not_offered_again_in_that_turn() -> None:
    """The action screen back in the same turn after a move was tapped: whatever blocks the move, it is marked
    unavailable for the rest of this turn, so the player picks something else instead of the turn timer."""
    from pcsa.common.types import MoveInfo
    from pcsa.live import move_peek
    from pcsa.players.base import ScreenFacts

    last = {"label": "move3", "turn": 10, "log_turn": 9}
    assert move_peek.refused_moves(last, 10, True, False) == (3,)  # the log shows no new turn
    assert move_peek.refused_moves(last, 10, False, True) == (3,)  # the log read failed, the screen never left
    assert move_peek.refused_moves(last, 11, False, True) == ()  # a new turn
    assert move_peek.refused_moves(last, 10, False, False) == ()
    assert move_peek.refused_moves({"label": "mega+move1", "turn": 4}, 4, True, False) == (1,)
    assert move_peek.refused_moves({"label": "move2", "turn": 4, "refused": (3,)}, 4, True, False) == (3, 2)
    assert move_peek.refused_moves({"label": "switch2", "turn": 4, "refused": (3,)}, 4, True, True) == ()
    assert move_peek.refused_moves({}, 1, True, True) == ()
    screen = ScreenFacts(tuple(MoveInfo(i, f"m{i}", (8, 8)) for i in range(4)), True, False, (10, 20))
    marked = move_peek.without_moves(screen, (3, 2))
    assert [m.disabled for m in marked.moves] == [False, False, True, True] and marked.mega_available
    assert move_peek.without_moves(screen, ()) is screen

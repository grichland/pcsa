"""The CG Player over the fake CG, in process and as a subprocess: answers, caching, restarts, timeouts, the log."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from pcsa.cg.adapter import CGPlayer
from pcsa.cg.fake_cg import FakeCG
from pcsa.cg.transport import CGError, CGTimeout, Subprocess, resolve
from pcsa.common.clock import FakeClock
from pcsa.common.config import load_config
from pcsa.common.errors import ConfigError
from pcsa.common.paths import repo_root
from pcsa.decide.api import MoveOffer, Offer, Order, SwitchOffer
from pcsa.decide.load import load_player
from tests.cg.helpers import (
    ENTRY,
    FAKE_CG,
    START,
    TAG,
    TURN1,
    TURN2,
    entry,
    fresh_fake,
    preview,
    problems,
    request,
    team,
)  # fmt: skip
from tests.configs import cg_config

TIMEOUT_S = 2.0


@pytest.fixture(autouse=True)
def _fake():
    fresh_fake()
    yield
    fresh_fake()


def inprocess(**kw) -> CGPlayer:
    return CGPlayer(cg_config(transport="inprocess", entry=ENTRY, timeout_s=TIMEOUT_S), **kw)


def subprocess_player(tmp_path: Path, *args: str) -> CGPlayer:
    command = [*FAKE_CG, "--log", str(tmp_path / "fake.jsonl"), *args]
    return CGPlayer(cg_config(command=command, timeout_s=TIMEOUT_S), run_dir=tmp_path / "run")


def sent(player: CGPlayer) -> list[str]:
    return [line["message"]["type"] for line in player.log.read(TAG) if line["dir"] == "sent"]


def test_a_battle_in_process() -> None:
    player = inprocess()
    player.battle_start(TAG, team(()))
    picks, notes = player.teampreview(preview())
    assert picks == (0, 1, 2) and notes["sent"] == 2
    answer = player.choose_move(request(0, TURN1))
    assert answer.order == Order.move(0) and answer.notes["cg"] == {"fake": True}
    again = player.choose_move(request(0, TURN1))
    assert again.order == answer.order and again.notes["cached"] is True
    r0 = request(0, TURN1)
    r1 = request(1, TURN1 + TURN2, history=(entry(r0),))
    assert player.choose_move(r1).order == Order.move(0)
    fallback = {"seq": 1, "reason": "builtin: late"}  # no order: the row had no id
    final = (START, *TURN1, *TURN2, {"k": "end", "result": "forfeit"})
    player.battle_end(TAG, "opponent_forfeit", final, (entry(r0), entry(r1, fallback)))
    assert sent(player) == ["battle_start", "preview_request", "events", "decision_request", "events",
                            "decision_request", "events", "battle_end"]  # fmt: skip
    last = FakeCG.instances[0].received[-2:]
    assert last[0]["events"] == [{"k": "end", "result": "forfeit"}] and last[0]["fallback"] == fallback
    assert last[1]["result"] == "forfeit"
    assert problems() == [] and len(FakeCG.instances) == 1


def test_cg_reply_shapes() -> None:
    """An error ends the wait and falls back; a stale seq's order is used; other errors are noted."""
    player = inprocess()
    player.teampreview(preview())
    cg = FakeCG.instances[0]
    cg.script = {0: {"error": "'seq' is not an integer", "type": "decision_request"}}
    with pytest.raises(CGError, match="CG error: 'seq' is not an integer"):
        player.choose_move(request(0, TURN1))
    order = {"kind": "move", "move": "shadowball", "mega": False}
    cg.script = {1: {"seq": 1, "error": "stale seq 1 (latest 2)", "order": order}}
    answer = player.choose_move(request(1, TURN1 + TURN2))
    assert answer.order == Order.move(1) and answer.notes["cg_error"] == "stale seq 1 (latest 2)"
    cg.script = {2: {"seq": 2, "error": "stale seq 2 (latest 3)"}}
    with pytest.raises(CGError, match="stale seq 2"):
        player.choose_move(request(2, TURN1 + TURN2, retry=True))
    cg.script = {3: {"seq": 3, "order": {"kind": "move", "move": "struggle", "mega": False},
                     "notes": {"last_resort": True, "why": "nothing known: struggle"}}}  # fmt: skip
    with pytest.raises(CGError, match="last resort .nothing known: struggle. cannot be placed"):
        player.choose_move(request(3, TURN1 + TURN2, retry=True))
    cg.script = {"events": {"error": "1 event(s) could not be read; the others were applied", "type": "events"}}
    answer = player.choose_move(request(4, TURN1 + TURN2 + ({"k": "turn", "n": 3},)))
    assert answer.notes["cg_errors"][0]["type"] == "events" and answer.notes["cg"] == {"fake": True}


def test_a_new_player_sends_the_whole_stream(tmp_path: Path) -> None:
    r0 = request(0, TURN1)
    fallback = {"seq": 0, "order": {"kind": "switch", "mon": "garchomp"}, "reason": "builtin: late"}
    answer = inprocess().choose_move(request(1, TURN1 + TURN2, history=(entry(r0, fallback),)))
    assert answer.order == Order.move(0)
    cg = FakeCG.instances[0]
    assert [m["type"] for m in cg.received] == ["battle_start", "preview_request", "events", "decision_request",
                                                "events", "decision_request"]  # fmt: skip
    assert cg.received[4]["fallback"] == fallback and problems() == []
    assert [r.get("seq") for r in cg.replies] == [None, 0, 1]  # the replayed answers are dropped


def test_overruled_picks_reach_cg_as_a_preview_fallback() -> None:
    """The runner brought Golisopod, Garchomp, Gholdengo, not CG's 0, 1, 2: CG is told before the leads, and its
    replacement is one that was brought."""
    player = inprocess()
    assert player.teampreview(preview())[0] == (0, 1, 2)
    brought = (2, 1, 0)
    r0 = request(0, TURN1, picks=brought)
    answer = player.choose_move(r0)
    assert answer.notes["picks_overruled"] == {"cg": [0, 1, 2], "brought": [2, 1, 0]}
    cg = FakeCG.instances[0]
    assert cg.received[2]["fallback"] == {"picks": [2, 1, 0], "reason": "overruled"}
    forced = Offer((), False, (SwitchOffer("Garchomp"), SwitchOffer("Golisopod")), None, "faint")
    r1 = request(1, TURN1 + TURN2, history=(entry(r0),), offer=forced, picks=brought)
    assert player.choose_move(r1).order == Order.switch("Golisopod")  # CG's first brought member, not active
    assert cg.battles[TAG].picks == [2, 1, 0] and problems() == []


def test_battle_end_after_a_cg_restart_resends_first() -> None:
    player = inprocess()
    player.teampreview(preview())
    r0 = request(0, TURN1)
    player.choose_move(r0)
    player.transport.restart("lost")
    final = (START, *TURN1, *TURN2, {"k": "end", "result": "win"})
    player.battle_end(TAG, "win", final, (entry(r0, None, {"kind": "move", "move": "makeitrain", "mega": False}),))
    fresh = FakeCG.instances[-1]
    assert len(FakeCG.instances) == 2
    assert [m["type"] for m in fresh.received] == ["battle_start", "preview_request", "events", "decision_request",
                                                   "events", "battle_end"]  # fmt: skip
    assert fresh.received[0]["picks"] == [0, 1, 2] and fresh.received[3]["played"]["move"] == "makeitrain"
    assert fresh.received[-2]["events"][-1] == {"k": "end", "result": "win"} and fresh.asked == []
    assert problems() == [] and player.streams == {}


def test_a_crash_in_process_restarts_and_resends() -> None:
    player = inprocess()
    player.teampreview(preview())
    FakeCG.exit_at, FakeCG.exits = 0, 1
    answer = player.choose_move(request(0, TURN1))
    assert answer.order == Order.move(0) and answer.notes["attempt"] == 2 and player.restarts == 1
    assert len(FakeCG.instances) == 2 and FakeCG.instances[1].received[0]["type"] == "battle_start"
    assert [line["message"] for line in player.log.read(TAG) if line["dir"] == "note"][0]["restart"]
    assert problems() == []


def test_a_timeout_raises_and_the_next_call_starts_a_new_cg() -> None:
    player = inprocess()
    player.teampreview(preview())
    FakeCG.slow = {0: TIMEOUT_S + 1.5}
    with pytest.raises(CGTimeout):
        player.choose_move(request(0, TURN1))
    r0 = request(0, TURN1)
    fallback = {"seq": 0, "order": {"kind": "move", "move": "shadowball", "mega": False}, "reason": "builtin: late"}
    player.choose_move(request(1, TURN1 + TURN2, history=(entry(r0, fallback),)))
    fresh = FakeCG.instances[-1]
    assert len(FakeCG.instances) == 2 and fresh.received[0]["type"] == "battle_start"
    assert [m.get("fallback") for m in fresh.received if m["type"] == "events"] == [None, fallback]
    assert problems() == []


def test_no_time_left_asks_nothing() -> None:
    clock = FakeClock(100.0)
    player = inprocess(clock=clock)
    with pytest.raises(CGTimeout):
        player.choose_move(request(0, TURN1, deadline=102.0))  # less than the tap reserve
    assert FakeCG.instances == [] and player.log.lines == []
    player.teampreview(preview(deadline=104.5))
    assert player.log.read(TAG)[1]["message"]["deadline_ms"] == 1500  # 4.5 s less the tap reserve and margin


def test_an_unplaceable_answer_raises() -> None:
    player = inprocess()
    other = Offer(tuple(MoveOffer(k, "tackle") for k in range(4)), False, (), None)
    with pytest.raises(CGError):
        player.choose_move(request(0, TURN1, offer=other))  # the fake names makeitrain: on no row


def test_subprocess_battle_crash_and_late_answer(tmp_path: Path) -> None:
    marker = tmp_path / "stopped"
    player = subprocess_player(tmp_path, "--exit-at", "1", "--once", str(marker), "--slow", "2=3.0")
    player.battle_start(TAG, team(()))
    assert player.teampreview(preview())[0] == (0, 1, 2)
    r0, r1 = request(0, TURN1), request(1, TURN1 + TURN2)
    assert player.choose_move(r0).order == Order.move(0)
    answer = player.choose_move(request(1, TURN1 + TURN2, history=(entry(r0),)))  # the process stops at seq 1
    assert answer.notes["attempt"] == 2 and player.restarts == 1 and marker.is_file()
    with pytest.raises(CGTimeout):
        player.choose_move(request(2, TURN1 + TURN2, history=(entry(r0), entry(r1)), retry=True))
    r2 = request(2, TURN1 + TURN2, retry=True)
    answer = player.choose_move(request(3, TURN1 + TURN2, history=(entry(r0), entry(r1), entry(r2)), retry=True))
    assert answer.notes["dropped_replies"] == 1  # seq 2's answer came late
    player.close()
    lines = [json.loads(x) for x in (tmp_path / "fake.jsonl").read_text().splitlines()]
    pids = list(dict.fromkeys(line["pid"] for line in lines))
    assert len(pids) == 2 and [line["problems"] for line in lines if line["problems"]] == []
    second = [line["message"]["type"] for line in lines if line["pid"] == pids[1]]
    assert second[:3] == ["battle_start", "preview_request", "events"]
    log = player.log.read(TAG)
    assert player.log.path(TAG) == tmp_path / "run" / "cg-messages" / f"{TAG}.jsonl"
    assert {line["cg"] for line in log} == {1, 2} and any(line["dir"] == "received" for line in log)
    assert all(line["message"] is not None for line in log)  # the `null` lines (no answer) are not replies


def test_config_section_is_checked(tmp_path: Path) -> None:
    agent = repo_root() / "config" / "agent.yaml"
    with pytest.raises(ConfigError, match="cg.command"):
        load_config(agent, ["decide.player=cg", "cg.command=null"], environ={})
    with pytest.raises(ConfigError, match="entry"):
        load_config(agent, ["cg.entry=not an entry"], environ={})
    with pytest.raises(ConfigError, match="timeout_s"):
        load_config(agent, ["decide.player=cg", "cg.command=[x]", "cg.timeout_s=39"], environ={})
    inprocess = ["decide.player=cg", "cg.transport=inprocess", f"cg.entry={ENTRY}", "cg.env={}"]
    config = load_config(agent, inprocess, environ={})
    player = load_player("cg", config, clock=FakeClock(0.0), run_dir=tmp_path)
    assert isinstance(player, CGPlayer) and player.params()["entry"] == ENTRY
    with pytest.raises(ConfigError, match="cg.args is for the inprocess"):
        load_config(agent, ["decide.player=cg", "cg.command=[x]", "cg.args={x: 1}"], environ={})
    with pytest.raises(ConfigError, match="python_path"):
        load_config(agent, ["cg.python_path=['']"], environ={})


def test_args_and_python_path_reach_cg(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    log = tmp_path / "fake.jsonl"
    cfg = cg_config(transport="inprocess", entry=ENTRY, args={"log_path": str(log)}, python_path=["tools"],
                    timeout_s=TIMEOUT_S)  # fmt: skip
    player = CGPlayer(cfg)
    assert player.teampreview(preview())[0] == (0, 1, 2) and log.is_file()
    assert str(repo_root() / "tools") in sys.path
    with pytest.raises(CGError, match="needs an entry that is a class"):
        resolve("pcsa.cg.fake_cg:main", {"x": 1})
    sub = Subprocess(["x"], python_path=["tools"])
    assert sub.env is not None and sub.env["PYTHONPATH"].split(os.pathsep)[0] == str(repo_root() / "tools")

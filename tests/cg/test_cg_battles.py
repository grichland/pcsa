"""The CG Player in whole battles on the fake device (tests/live/battle_rig.py) answered by the fake CG, in process
and as a subprocess. A full battle; restarts mid-battle with CG kept and with CG restarted; a CG timeout whose
fallback is reported on the next events message; a refused move."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pcsa.cg.adapter import CGPlayer
from pcsa.cg.fake_cg import FakeCG
from pcsa.common.clock import FakeClock
from pcsa.common.config import Config, load_config
from pcsa.common.errors import Recoverable
from pcsa.common.paths import repo_root
from pcsa.common.types import MoveInfo
from pcsa.live.turn import TurnPipeline
from tests.cg.helpers import ENTRY, FAKE_CG, fresh_fake
from tests.configs import cg_config
from tests.live.battle_rig import action, end_of_battle, lobby_to_team_select, peeked_action, run_battles
from tests.live.test_full_battles import START, VISITS, one_battle

TRANSPORTS = ["inprocess", "subprocess"]
TIMEOUT_S = 2.0
GHOLD_MOVES = tuple(MoveInfo(k, n, (8, 8)) for k, n in enumerate(("Make It Rain", "Shadow Ball", "Trick",
                                                                   "Nasty Plot")))  # fmt: skip


@pytest.fixture
def config() -> Config:
    return load_config(repo_root() / "config" / "agent.yaml", environ={})


@pytest.fixture(autouse=True)
def _fake():
    fresh_fake()
    yield
    fresh_fake()


class Rig:
    def __init__(self, transport: str, config: Config, tmp_path: Path, slow: dict | None = None, exit_at=None):
        self.transport, self.tmp = transport, tmp_path
        self.fake_log, self.marker = tmp_path / "fake.jsonl", tmp_path / "stopped"
        self.clock = FakeClock(1000.0)
        command = [*FAKE_CG, "--log", str(self.fake_log)]
        command += [f"--slow={seq}={s}" for seq, s in (slow or {}).items()]
        command += ["--exit-at", str(exit_at), "--once", str(self.marker)] if exit_at is not None else []
        if transport == "inprocess":
            FakeCG.slow, FakeCG.exit_at, FakeCG.exits = dict(slow or {}), exit_at, 1
        cg = cg_config(transport=transport, entry=ENTRY, command=command, timeout_s=TIMEOUT_S)
        self.config = config.model_copy(update={"agent": config.agent.model_copy(update={"cg": cg})})
        self.player = CGPlayer(self.config, clock=self.clock, run_dir=tmp_path / "run")

    def play(self, steps, monkeypatch, **kw):
        try:
            return run_battles(self.config, steps, self.tmp, monkeypatch, player=self.player, clock=self.clock, **kw)  # type: ignore[arg-type]
        finally:
            self.player.close()

    def starts(self) -> list[list[dict]]:
        """What each CG start received, in order."""
        if self.transport == "inprocess":
            return [cg.received for cg in FakeCG.instances]
        lines = [json.loads(x) for x in self.fake_log.read_text(encoding="utf-8").splitlines()]
        pids = list(dict.fromkeys(line["pid"] for line in lines))
        return [[line["message"] for line in lines if line["pid"] == pid] for pid in pids]

    def problems(self) -> list[str]:
        if self.transport == "inprocess":
            return [p for cg in FakeCG.instances for p in cg.problems]
        lines = [json.loads(x) for x in self.fake_log.read_text(encoding="utf-8").splitlines()]
        return [p for line in lines for p in line["problems"]]

    def messages(self, played, index: int = 1) -> list[dict]:
        path = played.recorder.dir / f"battle-{index}" / "cg-messages.jsonl"
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def replies(rig: Rig, start: int) -> list[dict]:
    """What one CG start answered, in order."""
    if rig.transport == "inprocess":
        return FakeCG.instances[start].replies
    lines = [json.loads(x) for x in rig.fake_log.read_text(encoding="utf-8").splitlines()]
    pid = list(dict.fromkeys(line["pid"] for line in lines))[start]
    return [line["reply"] for line in lines if line["pid"] == pid and line["reply"] is not None]


def types(messages: list[dict]) -> list[str]:
    return [m["type"] + (f":{m['seq']}" if "seq" in m else "") for m in messages]


def sent_events(messages: list[dict]) -> list[dict]:
    return [e for m in messages if m.get("type") == "events" for e in m["events"]]


def ended(played) -> tuple:
    """The battle's whole event list at its end (`end` last)."""
    return played.ends[-1][2]


def assert_each_start_whole(rig: Rig, final_events: tuple, brought=(0, 1, 2)) -> None:
    """Every CG start got battle_start first (a start after the preview with the picks brought) and every event at
    most once, in the controller's order; the last start got them all, the `end` event just before battle_end."""
    for k, received in enumerate(rig.starts()):
        start = dict(received[0])
        assert start.pop("picks", None) == (list(brought) if k else None)
        assert received[0]["type"] == "battle_start" and start == final_events[0]
        got = sent_events(received)
        assert got == list(final_events[1 : 1 + len(got)])
    last = rig.starts()[-1]
    assert final_events[-1]["k"] == "end" and sent_events(last)[-1] == final_events[-1]
    assert types(last)[-2:] == ["events", "battle_end"] and last[-1]["result"] == final_events[-1]["result"]
    assert rig.problems() == []


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_cg_a_full_battle(transport, config: Config, tmp_path: Path, monkeypatch) -> None:
    rig = Rig(transport, config, tmp_path)
    played = rig.play(one_battle(), monkeypatch, visits=VISITS)
    assert played.ending == "completed" and played.state.outcomes == ["win"]
    records = played.records()
    assert [r["order"] for r in records] == [
        "/choose move makeitrain", "/choose move makeitrain", "/choose switch Garchomp", "/choose move earthquake"
    ]  # fmt: skip
    assert [r["degrade"] for r in records] == ["player"] * 4 and {r["player"] for r in records} == {"cg"}
    assert records[0]["notes"]["cg"] == {"fake": True}
    preview = played.json("battle-1/team_preview.json")["decision"]
    assert preview["picks"] == [0, 1, 2] and preview["source"] == "player"
    sent = [line["message"] for line in rig.messages(played) if line["dir"] == "sent"]
    assert types(sent) == [
        "battle_start", "preview_request", "events", "decision_request:0", "events", "decision_request:1",
        "decision_request:2", "events", "decision_request:3", "events", "battle_end",
    ]  # fmt: skip
    assert sent[-1]["result"] == "win" and sent[8]["screen_offer"] == {"moves": [
        "earthquake", "dragonclaw", "swordsdance", "protect"]}  # fmt: skip
    final = ended(played)
    assert sent_events(sent) == list(final[1:]) and sent[-2]["events"] == [{"k": "end", "result": "win"}]
    assert_each_start_whole(rig, final)
    received = [line["message"] for line in rig.messages(played) if line["dir"] == "received"]
    assert [r.get("seq") for r in received] == [None, 0, 1, 2, 3]


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_cg_restart_mid_battle_with_cg_kept(transport, config: Config, tmp_path: Path, monkeypatch) -> None:
    """The session restarts after CG answered turn 2 (before the record) and before the forced switch is asked;
    each time it asks again with the whole history: the same answer, nothing sent twice."""
    real_save, saves = TurnPipeline._save, []

    def save(self, *args, **kw):
        saves.append(1)
        if len(saves) == 2:
            raise Recoverable("restart after the answer")
        return real_save(self, *args, **kw)

    monkeypatch.setattr(TurnPipeline, "_save", save)
    rig = Rig(transport, config, tmp_path)
    visits = [VISITS[0], VISITS[1], VISITS[1], VISITS[2]]  # the visit after the restart reads the same log
    played = rig.play(one_battle(), monkeypatch, visits=visits, faults={("decide", 4)})
    assert played.ending == "completed" and played.state.restarts == 2 and played.state.outcomes == ["win"]
    asked = played.requests()
    assert [r.request_id for r in asked] == [0, 1, 1, 2, 2, 3]
    assert asked[2].events == asked[1].events and asked[2].history == asked[1].history
    records = played.records()
    assert [r["degrade"] for r in records] == ["player"] * 4 and records[1]["notes"]["cached"] is True
    assert len(rig.starts()) == 1
    sent = [line["message"] for line in rig.messages(played) if line["dir"] == "sent"]
    assert types(sent) == [
        "battle_start", "preview_request", "events", "decision_request:0", "events", "decision_request:1",
        "decision_request:2", "events", "decision_request:3", "events", "battle_end",
    ]  # fmt: skip
    assert_each_start_whole(rig, ended(played))


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_cg_restart_mid_battle_with_cg_restarted(transport, config: Config, tmp_path: Path, monkeypatch) -> None:
    """CG stops at the forced switch without answering: it is started again and gets the whole stream."""
    rig = Rig(transport, config, tmp_path, exit_at=2)
    played = rig.play(one_battle(), monkeypatch, visits=VISITS)
    assert played.ending == "completed" and played.state.outcomes == ["win"]
    records = played.records()
    assert [r["degrade"] for r in records] == ["player"] * 4 and records[2]["notes"]["attempt"] == 2
    first, second = rig.starts()
    assert types(first)[-1] == "decision_request:2"
    assert types(second)[:8] == [
        "battle_start", "preview_request", "events", "decision_request:0", "events", "decision_request:1",
        "decision_request:2", "events",
    ]  # fmt: skip
    assert second[0]["picks"] == [0, 1, 2] and second[1]["replay"] is True
    answers = {r["seq"]: r["order"] for r in replies(rig, 0) if "seq" in r}
    for request in second[3], second[5]:  # replayed with what the first CG answered and the game took
        assert request["replay"] is True and request["played"] == answers[request["seq"]]
    assert "replay" not in second[6] and "played" not in second[6]
    assert [r.get("notes") for r in replies(rig, 1)[:3]] == [{"given": True}, {"replayed": True}, {"replayed": True}]
    assert [r.get("seq") for r in replies(rig, 1)[1:]] == [0, 1, 2, 3]
    assert [r["order"] for r in replies(rig, 1)[1:3]] == [answers[0], answers[1]]
    if transport == "inprocess":
        assert [m.get("seq") for m in FakeCG.instances[1].asked] == [2, 3]  # the replays asked nobody
    assert_each_start_whole(rig, ended(played))
    lines = rig.messages(played)
    assert [line["message"]["restart"] for line in lines if line["dir"] == "note" and "restart" in line["message"]][0]
    assert {line["cg"] for line in lines} == {1, 2}


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_cg_timeout_then_fallback_reported(transport, config: Config, tmp_path: Path, monkeypatch) -> None:
    """CG is too slow on turn 2: the runner falls back and the next events message tells CG what was tapped."""
    rig = Rig(transport, config, tmp_path, slow={1: TIMEOUT_S + 1.0})
    played = rig.play(one_battle(), monkeypatch, visits=VISITS)
    assert played.ending == "completed" and played.state.outcomes == ["win"]
    records = played.records()
    assert [r["degrade"] for r in records] == ["player", "builtin", "player", "player"]
    assert "CGTimeout" in records[1]["reason"]
    fallback = records[1]["live"]["fallback"]
    assert fallback["seq"] == 1 and fallback["order"] == {"kind": "move", "move": "makeitrain", "mega": False}
    assert played.requests()[2].history[-1]["fallback"] == fallback
    sent = [line["message"] for line in rig.messages(played) if line["dir"] == "sent"]
    carried = [m for m in sent if m.get("fallback")]
    assert [m["fallback"] for m in carried] == [fallback]
    assert sent.index(carried[0]) < next(i for i, m in enumerate(sent) if m.get("seq") == 2)
    starts = rig.starts()
    assert len(starts) == (2 if transport == "inprocess" else 1)  # in process a stuck handle is never called again
    assert [m["fallback"] for m in starts[-1] if m.get("fallback")] == [fallback]
    assert_each_start_whole(rig, ended(played))


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_cg_a_refused_move(transport, config: Config, tmp_path: Path, monkeypatch) -> None:
    """The action screen comes back in the same turn: the request says refused and CG picks another move."""
    steps = (lobby_to_team_select() + action("Gholdengo") + peeked_action("Gholdengo", GHOLD_MOVES, mega=False)
             + end_of_battle("loss"))  # fmt: skip
    rig = Rig(transport, config, tmp_path)
    played = rig.play(steps, monkeypatch, visits=[{1: START}, {1: START}])
    assert played.ending == "completed" and played.state.outcomes == ["loss"]
    records = played.records()
    assert [(r["order"], r["degrade"]) for r in records] == [
        ("/choose move makeitrain", "player"), ("/choose move shadowball", "player")
    ]  # fmt: skip
    request = [line["message"] for line in rig.messages(played) if line["dir"] == "sent"
               and line["message"].get("seq") == 1][0]  # fmt: skip
    refused = {"kind": "move", "move": "makeitrain", "mega": False}
    assert request["refused"] == {"order": refused} and "retry" not in request
    assert request["screen_offer"]["moves"][0] == "makeitrain"
    assert played.requests()[-1].events[-1] == {"k": "refused", "order": refused}
    assert_each_start_whole(rig, ended(played))


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_cg_a_tap_not_confirmed(transport, config: Config, tmp_path: Path, monkeypatch) -> None:
    """The move tap fails to send: CG is told the tap is not confirmed (no order) before the next request, and a
    replay would carry no `played` for it."""
    first = action("Gholdengo")
    first[0] = replace(first[0], inject="tap_fail")
    steps = lobby_to_team_select() + first + end_of_battle("loss")
    rig = Rig(transport, config, tmp_path)
    played = rig.play(steps, monkeypatch, visits=[{1: START}, {1: START}])
    sent = [line["message"] for line in rig.messages(played) if line["dir"] == "sent"]
    assert played.ending == "completed" and rig.problems() == []
    carried = [(i, m["fallback"]) for i, m in enumerate(sent) if m.get("fallback")]
    assert [f for _, f in carried] == [{"seq": 0, "reason": "tap not confirmed"}]
    assert carried[0][0] < next(i for i, m in enumerate(sent) if m.get("seq") == 1)
    assert played.requests()[-1].history[0]["fallback"] == {"seq": 0, "reason": "tap not confirmed"}
    assert "played" not in played.requests()[-1].history[0]

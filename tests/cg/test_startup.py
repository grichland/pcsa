"""Making CG for a run, over the fake CG: cg.env and cg.run_log_dir, the start-up status check (LIVE 0.9), the
model's counts per battle and the anomaly warning, a second preview fallback (S10-3) and an unconfirmed tap
(S10-4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pcsa.cg import startup
from pcsa.cg.adapter import CGPlayer
from pcsa.cg.fake_cg import FakeCG
from pcsa.cg.stream import Stream
from pcsa.cg.transport import CGError, Subprocess, make_transport
from pcsa.common.config import CGConfig, load_config
from pcsa.common.errors import ConfigError
from pcsa.common.paths import repo_root
from pcsa.live import decision_record
from pcsa.live.play import start_player
from tests.cg.helpers import ENTRY, START, TAG, TURN1, TURN2, entry, fresh_fake, preview, problems, request

NEURAL_ARGS = {"policy": "p:P", "checkpoint": "C:/m/iter.pt", "device": "cpu"}
GOOD = {"layout_ok": True, "mismatch": [], "allow_mismatch": False, "handed_to_fallback": False,
        "checkpoint": "C:/m/iter.pt"}  # fmt: skip
MOVE = {"kind": "move", "move": "makeitrain", "mega": False}


@pytest.fixture(autouse=True)
def _fake():
    fresh_fake()
    yield
    fresh_fake()


def neural_inprocess(**args) -> CGConfig:
    return CGConfig(transport="inprocess", entry=ENTRY, args={"player": startup.NEURAL, "player_args": NEURAL_ARGS,
                                                              **args})  # fmt: skip


def neural_command(*extra: str) -> CGConfig:
    command = ["py", "-m", "champions_gym.live", "--player", startup.NEURAL, "--player-args", json.dumps(NEURAL_ARGS)]
    return CGConfig(command=[*command, *extra], env={"CUDA_VISIBLE_DEVICES": "-1"})


def test_env_and_run_log_dir_are_checked() -> None:
    agent = repo_root() / "config" / "agent.yaml"
    config = load_config(agent, ["cg.env={CUDA_VISIBLE_DEVICES: '-1'}"], environ={})
    assert config.agent.cg.env == {"CUDA_VISIBLE_DEVICES": "-1"} and config.agent.cg.run_log_dir == "cg"
    with pytest.raises(ConfigError, match="env"):
        load_config(agent, ["cg.env={'not a name': x}"], environ={})
    with pytest.raises(ConfigError, match="cg.env is for the subprocess"):
        load_config(agent, ["cg.transport=inprocess", f"cg.entry={ENTRY}", "cg.env={A: b}"], environ={})
    for bad in ("../out", "C:/out", "''"):
        with pytest.raises(ConfigError, match="run_log_dir"):
            load_config(agent, [f"cg.run_log_dir={bad}"], environ={})
    with pytest.raises(ConfigError, match="start_timeout_s"):
        load_config(agent, ["cg.start_timeout_s=0"], environ={})


def test_env_reaches_the_subprocess() -> None:
    echo = "import os, json, sys\nfor _ in sys.stdin: print(json.dumps({'cuda': os.environ.get('CUDA_VISIBLE_DEVICES')}), flush=True)"
    sub = Subprocess([sys.executable, "-c", echo], python_path=["tools"], env={"CUDA_VISIBLE_DEVICES": "-1"})
    sub.ensure()
    try:
        reply, _ = sub.exchange([{"type": "status"}], lambda r: "cuda" in r, 30.0, lambda *_: None)
    finally:
        sub.close()
    assert reply == {"cuda": "-1"} and sub.env is not None and "PYTHONPATH" in sub.env


def test_log_dir_goes_to_cg_and_to_the_model_player(tmp_path: Path) -> None:
    folder = tmp_path / "cg"
    args = startup.with_log_dir(neural_inprocess(log_dir="x"), folder).args
    assert args["log_dir"] == str(folder) and args["player_args"] == {**NEURAL_ARGS, "log_dir": str(folder)}
    tower = startup.with_log_dir(CGConfig(transport="inprocess", entry=ENTRY, args={"player": "tower"}), folder)
    assert tower.args == {"player": "tower", "log_dir": str(folder)}
    command = startup.with_log_dir(neural_command("--log-dir", "old"), folder).command or []
    assert command.count("--log-dir") == 1 and command[command.index("--log-dir") + 1] == str(folder)
    assert json.loads(command[command.index("--player-args") + 1])["log_dir"] == str(folder)
    plain = startup.with_log_dir(CGConfig(command=["py", "-m", "champions_gym.live"]), folder).command
    assert plain == ["py", "-m", "champions_gym.live", "--log-dir", str(folder)]
    made = make_transport(neural_command().model_copy(update={"run_log_dir": "cg"}), tmp_path)
    assert isinstance(made, Subprocess) and str(folder) in made.command
    assert made.env is not None and made.env["CUDA_VISIBLE_DEVICES"] == "-1"
    assert startup.checkpoint(neural_command()) == "C:/m/iter.pt" and startup.expects_model(neural_inprocess())


def test_status_problems() -> None:
    tower = CGConfig(transport="inprocess", entry=ENTRY, args={"player": "tower"})
    assert startup.status_problems({"status": {"player_status": None}}, tower) == []
    assert startup.status_problems({"status": {"player_status": None}}, neural_command()) != []
    assert startup.status_problems({"error": "x"}, tower) == ["no status answer: {'error': 'x'}"]
    assert startup.status_problems({"status": {"player_status": GOOD}}, neural_command()) == []
    assert startup.status_problems({"status": {"player_status": {**GOOD, "checkpoint": "C:\\m\\iter.pt"}}},
                                   neural_command()) == []  # fmt: skip
    for key, value in (("layout_ok", False), ("mismatch", ["contract"]), ("allow_mismatch", True),
                       ("handed_to_fallback", "slow"), ("checkpoint", "C:/m/other.pt")):  # fmt: skip
        found = startup.status_problems({"status": {"player_status": {**GOOD, key: value}}}, neural_command())
        assert len(found) == 1 and key in found[0]


def test_start_checks_the_status_and_saves_it(tmp_path: Path) -> None:
    FakeCG.player_status = GOOD
    player = CGPlayer(neural_inprocess(), run_dir=tmp_path)
    assert player.start()["player_status"] == GOOD
    assert json.loads((tmp_path / "cg-start.json").read_text(encoding="utf-8"))["problems"] == []
    assert player.params()["start_status"]["status"]["player_status"] == GOOD
    FakeCG.player_status = {**GOOD, "mismatch": ["layout_hash"]}
    refused = CGPlayer(neural_inprocess(), run_dir=tmp_path / "b")
    with pytest.raises(ConfigError, match="mismatch"):
        start_player(refused)
    FakeCG.player_status = None
    with pytest.raises(CGError, match="model player is not in place"):
        CGPlayer(neural_inprocess()).start()
    with pytest.raises(CGError, match="CG did not start"):
        CGPlayer(CGConfig(transport="inprocess", entry=ENTRY, args={"unknown": 1})).start()
    crash = [sys.executable, "-c", "import sys; sys.stderr.write('no checkpoint here\\n'); sys.exit(3)"]
    with pytest.raises(CGError, match="no checkpoint here"):
        CGPlayer(CGConfig(command=crash, start_timeout_s=30.0), run_dir=tmp_path / "c").start()
    assert FakeCG.instances[0].statuses == [{"type": "status"}] and FakeCG.instances[0].received == []


def test_start_player_without_start_is_a_no_op() -> None:
    start_player(SimpleNamespace(name="first"))


def test_model_fallbacks_and_anomalies_are_counted() -> None:
    player = CGPlayer(CGConfig(transport="inprocess", entry=ENTRY, timeout_s=2.0))
    player.teampreview(preview())
    model = {"fallback": "policy raised", "ms": 3.5, "reused": False, "slow": False, "value": 0.1}
    FakeCG.instances[0].script[0] = {"seq": 0, "order": MOVE, "notes": {"model": model, "anomalies": 2}}
    FakeCG.instances[0].script[1] = {"seq": 1, "order": MOVE, "notes": {"model": {**model, "fallback": False}}}
    r0 = request(0, TURN1)
    assert player.choose_move(r0).notes["cg"]["model"] == model
    player.choose_move(request(0, TURN1))  # asked again: cached, not counted twice
    player.choose_move(request(1, TURN1 + TURN2, history=(entry(r0),)))
    player.battle_end(TAG, "win", (START, *TURN1, *TURN2, {"k": "end", "result": "win"}), (entry(r0),))
    counts = {"decisions": 2, "fallbacks": 1, "reasons": {"policy raised": 1}, "ms_median": 3.5, "ms_max": 3.5}
    assert player.written[-1]["model"] == counts and player.written[-1]["status"]["battle"] == TAG
    assert player.run_summary()["cg"]["model_fallbacks"] == 1
    notes = [line["message"] for line in player.log.read(TAG) if line["dir"] == "note"]
    assert {"anomalies": 2} in notes and any("status_query" in n for n in notes)
    assert "status" not in [line["message"].get("type") for line in player.log.read(TAG) if line["dir"] == "sent"]
    assert problems() == []


def test_a_second_preview_fallback_when_the_three_brought_are_known_better() -> None:
    stream = Stream(TAG)
    stream.preview(START, None)
    stream.picks = (0, 1, 2)  # CG's answer
    r0 = request(0, TURN1, picks=(3, 1, 2))
    assert stream.decision(r0, r0.live)[0]["fallback"] == {"picks": [3, 1, 2], "reason": "overruled"}
    r1 = request(1, TURN1 + TURN2, history=(entry(r0),), picks=(3, 4, 2))  # a party screen showed the back two
    assert stream.decision(r1, r1.live)[0]["fallback"] == {"picks": [3, 4, 2], "reason": "team brought"}
    r2 = request(2, TURN1 + TURN2, history=(entry(r0), entry(r1)), picks=(3, 4, 2), retry=True)
    assert all("fallback" not in m for m in stream.decision(r2, r2.live))


def test_an_unconfirmed_tap_is_a_fallback_without_an_order() -> None:
    r0 = request(0, TURN1)
    b = SimpleNamespace(index=1, live_history=[entry(r0, None, MOVE)])
    decision_record.unconfirmed(b, 0, "the party screen stayed up")  # type: ignore[arg-type]
    assert b.live_history[0]["fallback"] == {"seq": 0, "reason": "tap not confirmed"}
    assert "played" not in b.live_history[0]
    stream = Stream(TAG)
    stream.preview(START, None)
    stream.decision(r0, r0.live)
    r1 = request(1, TURN1 + TURN2, history=tuple(b.live_history), retry=True)
    first = stream.decision(r1, r1.live)[0]
    assert first["type"] == "events" and first["fallback"] == {"seq": 0, "reason": "tap not confirmed"}
    fresh = Stream(TAG, brought=(0, 1, 2))  # a CG that restarted: the request is replayed with no `played`
    replayed = [m for m in fresh.decision(r1, r1.live) if m.get("seq") == 0]
    assert replayed[0]["replay"] is True and "played" not in replayed[0]

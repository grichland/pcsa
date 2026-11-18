"""CG's model player behind the controller, by default a subprocess run by a Python with torch (tests/neural.py):
the start-up status check, a full fake-phone battle (every answer the model's, no fallback, CG's two records in the
run folder), a restart mid-battle with CG kept and with CG restarted (the same answers), a missing checkpoint (no
battle), a refused move and a controller-side timeout. Skipped when that Python or the test checkpoint is missing."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from pcsa.cg.adapter import CGPlayer
from pcsa.cg.transport import TransportFailed
from pcsa.common.clock import FakeClock
from pcsa.common.config import CGConfig, Config, load_config
from pcsa.common.errors import ConfigError, Recoverable
from pcsa.common.paths import repo_root
from pcsa.live.play import start_player
from pcsa.live.turn import TurnPipeline
from tests.cg import neural
from tests.cg.helpers import preview
from tests.cg.test_real_cg import Script, lines
from tests.live.t7_rig import action, end_of_battle, lobby_to_team_select, run_battles

pytestmark = pytest.mark.skipif(neural.missing() is not None, reason=str(neural.missing()))


@cache
def model_picks() -> tuple[int, ...]:
    """The model's preview answer for the controller's battle_start (argmax: the same every time)."""
    player = CGPlayer(neural.config())
    try:
        player.start()
        return tuple(player.teampreview(preview())[0])
    finally:
        player.close()


class Rig:
    def __init__(self, tmp_path: Path, cg: CGConfig | None = None) -> None:
        config = load_config(repo_root() / "config" / "agent.yaml", environ={})
        self.tmp, self.clock, self.script = tmp_path, FakeClock(1000.0), Script(model_picks())
        cg = cg or neural.config()
        self.config: Config = config.model_copy(update={"agent": config.agent.model_copy(update={"cg": cg})})
        self.run_dir = tmp_path / "run"
        self.player = CGPlayer(self.config, clock=self.clock, run_dir=self.run_dir)

    def play(self, monkeypatch, steps=None, **kw):
        try:
            start_player(self.player)
            player: Any = self.player
            return run_battles(self.config, steps or self.script.steps(), self.tmp, monkeypatch, player=player,
                               clock=self.clock, visits=kw.pop("visits", self.script.visits), **kw)  # fmt: skip
        finally:
            self.player.close()


def answers(played) -> dict[int, dict]:
    """seq -> the order of CG's live answer (the last one given for that seq)."""
    return {r["seq"]: r["order"] for r in lines(played, "received") if "seq" in r and "order" in r}


def assert_model_answered(rig: Rig, played, decisions: int) -> None:
    assert played.ending == "completed" and played.state.outcomes == ["win"]
    assert [r for r in lines(played, "received") if "error" in r] == []
    records = played.records()
    assert [r["degrade"] for r in records] == ["player"] * decisions
    models = [r["notes"]["cg"].get("model") for r in records if not r["notes"]["cg"].get("replayed")]
    assert models and all(m["fallback"] is False for m in models)
    summary = json.loads((played.recorder.dir / "battle-1" / "cg-summary.json").read_text(encoding="utf-8"))
    assert summary["model"]["fallbacks"] == 0 and summary["status"]["player_status"]["handed_to_fallback"] is False
    tag = played.ends[0][0]
    assert (rig.run_dir / "cg" / f"{tag}.jsonl").is_file() and (rig.run_dir / "cg" / f"{tag}.neural.jsonl").is_file()
    assert (rig.run_dir / "cg-start.json").is_file()


def test_neural_start_status(tmp_path: Path) -> None:
    player = CGPlayer(neural.config(), run_dir=tmp_path)
    try:
        status = player.start()
    finally:
        player.close()
    found = status["player_status"]
    assert found["layout_ok"] is True and found["mismatch"] == [] and found["allow_mismatch"] is False
    assert found["handed_to_fallback"] is False and Path(found["checkpoint"]) == neural.CHECKPOINT
    saved = json.loads((tmp_path / "cg-start.json").read_text(encoding="utf-8"))
    assert saved["problems"] == [] and saved["status"]["version"] == status["version"]
    made: Any = player.transport
    given = made.command if neural.TRANSPORT == "subprocess" else [made.args["log_dir"]]
    assert str(tmp_path / "cg") in given


def test_neural_missing_checkpoint_starts_no_battle(tmp_path: Path) -> None:
    player = CGPlayer(neural.config(tmp_path / "nothing.pt"), run_dir=tmp_path)
    with pytest.raises(ConfigError, match="cannot start, no battle started") as refused:
        start_player(player)
    assert "nothing.pt" in str(refused.value) and not list(tmp_path.glob("battle-*"))
    saved = json.loads((tmp_path / "cg-start.json").read_text(encoding="utf-8"))
    assert saved["status"] is None and "nothing.pt" in saved["problems"][0]


def test_neural_a_full_battle(tmp_path: Path, monkeypatch) -> None:
    rig = Rig(tmp_path)
    played = rig.play(monkeypatch)
    assert_model_answered(rig, played, 4)
    assert tuple(lines(played, "received")[0]["picks"]) == rig.script.picks
    total = played.json("summary.json") if (played.recorder.dir / "summary.json").is_file() else None
    assert total is None or "cg" not in total  # run_battles writes no run summary; play_command does


def test_neural_restart_with_cg_kept(tmp_path: Path, monkeypatch) -> None:
    uninterrupted = answers(Rig(tmp_path / "a").play(monkeypatch))
    real_save, saves = TurnPipeline._save, []

    def save(self, *args, **kw):
        saves.append(1)
        if len(saves) == 2:
            raise Recoverable("restart after the answer")
        return real_save(self, *args, **kw)

    monkeypatch.setattr(TurnPipeline, "_save", save)
    rig = Rig(tmp_path / "b")
    v = rig.script.visits
    played = rig.play(monkeypatch, visits=[v[0], v[1], v[1], v[2]], faults={("decide", 4)})
    assert played.state.restarts == 2
    assert_model_answered(rig, played, 4)
    assert answers(played) == uninterrupted


def test_neural_restarted_mid_battle(tmp_path: Path, monkeypatch) -> None:
    uninterrupted = answers(Rig(tmp_path / "a").play(monkeypatch))
    rig = Rig(tmp_path / "b")
    real = rig.player.transport.exchange

    def exchange(messages, *args):
        if any(m.get("seq") == 2 for m in messages) and not getattr(exchange, "failed", False):
            exchange.failed = True  # type: ignore[attr-defined]
            raise TransportFailed("lost")
        return real(messages, *args)

    rig.player.transport.exchange = exchange  # type: ignore[method-assign]
    played = rig.play(monkeypatch)
    assert rig.player.restarts == 1
    assert_model_answered(rig, played, 4)
    second = lines(played, "sent", cg=2)
    assert second[0]["picks"] == list(rig.script.picks) and [m["seq"] for m in second if m.get("played")] == [0, 1]
    assert {s: o for s, o in answers(played).items() if s >= 2} == {s: o for s, o in uninterrupted.items() if s >= 2}


def test_neural_a_refused_move(tmp_path: Path, monkeypatch) -> None:
    rig = Rig(tmp_path)
    lead = rig.script.lead
    steps = lobby_to_team_select() + action(lead.name) + action(lead.name) + end_of_battle("win")
    start = rig.script.visits[0]
    played = rig.play(monkeypatch, steps=steps, visits=[start, start])
    assert played.ending == "completed"
    records = played.records()
    first, second = (r["live"]["decision_request"] for r in records)
    assert "refused" not in first and second["refused"]["order"] == records[0]["live"]["played"]
    assert records[1]["notes"]["cg"]["model"]["fallback"] is False and records[1]["order"] != records[0]["order"]


def test_neural_a_controller_timeout(tmp_path: Path, monkeypatch) -> None:
    rig = Rig(tmp_path, neural.slow_config(1, 1.5, timeout_s=1.0))
    played = rig.play(monkeypatch)
    records = played.records()
    assert [r["degrade"] for r in records] == ["player", "builtin", "player", "player"]
    assert "CGTimeout" in records[1]["reason"]
    fallback = records[1]["live"]["fallback"]
    carried = [m["fallback"] for m in lines(played, "sent") if m.get("fallback")]
    assert carried == [fallback] and fallback["seq"] == 1
    assert [r for r in lines(played, "received") if "error" in r] == []

"""pcsa.cg.serve's additions never change play: a failure in `view` or `net` leaves CG's reply exactly as CG gave
it, is counted in `status.serve`, and the next request is still answered; the output loop writes one line per line
whatever happens; the built-in path's use of CG internals is optional and the neural path never touches them; and
a missing torch or champions_gym ends start-up with exit code 2 and one clear stderr line."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("champions_gym")

from champions_gym.live import Session  # noqa: E402

from pcsa.cg import serve  # noqa: E402
from pcsa.cg.serve import Server, builtin_battle, reply_line  # noqa: E402
from tests.cg.test_serve import EVENTS, PREVIEW, START, STATUS, TURN, Hooked  # noqa: E402

TURN2 = {**TURN, "seq": 1}
MESSAGES = [START, PREVIEW, EVENTS, TURN]


def bare(player) -> list:
    """CG's own replies to MESSAGES."""
    session = Session(player=player)
    return [session.handle(json.loads(json.dumps(m))) for m in MESSAGES]


def boom(*args, **kw):
    raise RuntimeError("boom")


@pytest.mark.parametrize("make", [lambda: "tower", Hooked])
def test_a_failing_view_and_net_leave_cgs_reply_as_it_was(make, monkeypatch, capsys) -> None:
    monkeypatch.setattr(serve, "view", boom)
    monkeypatch.setattr(Server, "_net", boom)
    server = Server(make())
    got = [server.handle(json.loads(json.dumps(m))) for m in MESSAGES]
    assert got == bare(make())
    assert server.failures >= 2 and "boom" in capsys.readouterr().err
    status = server.handle(STATUS)
    assert status is not None and status["status"]["serve"]["failures"] == server.failures


def test_a_failing_net_keeps_the_view(monkeypatch) -> None:
    monkeypatch.setattr(Server, "_net", boom)
    server = Server(Hooked())
    replies = [server.handle(m) for m in MESSAGES]
    decided = [r for r in replies if r is not None and ("picks" in r or "order" in r)]
    assert len(decided) == 2 and all("view" in r["notes"] and "net" not in r["notes"] for r in decided)
    assert server.failures == 2
    assert server.handle(TURN2) is not None


def test_a_reply_that_is_not_json_still_gives_one_line() -> None:
    class Broken:
        failures = 0

        def handle(self, line):
            raise RuntimeError("handle broke")

        def _failed(self, what, error):
            self.failures += 1

    line = reply_line(Broken(), json.dumps(START))  # type: ignore[arg-type]
    assert json.loads(line)["error"].startswith("pcsa.cg.serve: RuntimeError")

    class Odd(Broken):
        def handle(self, line):
            return {"seq": 0, "order": {}, "notes": {"view": {1, 2}}}

    assert json.loads(reply_line(Odd(), "x")) == {"seq": 0, "order": {}, "notes": {}}  # type: ignore[arg-type]


def test_the_builtin_path_survives_missing_cg_internals(monkeypatch) -> None:
    assert builtin_battle(SimpleNamespace()) is None and builtin_battle(SimpleNamespace(_builtin="tower")) is None
    monkeypatch.setattr(serve, "builtin_battle", lambda session: None)
    server = Server("tower")
    replies = [server.handle(m) for m in MESSAGES]
    assert replies[1] is not None and "picks" in replies[1] and "view" not in replies[1]["notes"]
    assert replies[3] is not None and "order" in replies[3] and server.handle(TURN2) is not None


def test_the_player_object_path_never_reads_cg_internals(monkeypatch) -> None:
    def touched(session):
        raise AssertionError("CG internals read on the Player object path")

    monkeypatch.setattr(serve, "builtin_battle", touched)
    server = Server(Hooked())
    assert server.session.capture_builtin is False
    replies = [server.handle(m) for m in MESSAGES]
    assert all(r is None or "error" not in r for r in replies) and replies[3] is not None and "view" in replies[3]["notes"]


def run_blocked(module: str, *args: str) -> subprocess.CompletedProcess:
    """`python -m pcsa.cg.serve ARGS` with `module` made unimportable."""
    code = (f"import runpy, sys; sys.modules[{module!r}] = None; sys.argv = ['pcsa.cg.serve', *{list(args)!r}]; "
            "runpy.run_module('pcsa.cg.serve', run_name='__main__')")  # fmt: skip
    return subprocess.run([sys.executable, "-c", code], input="", capture_output=True, text=True, timeout=60)


def test_without_torch_the_neural_player_says_how_to_get_it() -> None:
    args = json.dumps({"policy": "pcsa.network.policy:Policy", "checkpoint": "x.pt"})
    done = run_blocked("torch", "--player", serve.NEURAL, "--player-args", args)
    assert done.returncode == 2 and done.stdout == ""
    assert "needs torch: uv sync --inexact --extra torch" in done.stderr and "Traceback" not in done.stderr


def test_without_champions_gym_it_exits_2_with_a_clear_line() -> None:
    done = run_blocked("champions_gym", "--player", "tower")
    assert done.returncode == 2 and done.stdout == ""
    assert "Champions Gym (champions_gym) is not installed" in done.stderr and "Traceback" not in done.stderr

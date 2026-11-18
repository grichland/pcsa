"""pcsa.cg.serve against real CG: the protocol of `python -m champions_gym.live` (one line out per line in, `null`
for no answer, exit code 2 when it cannot start), `view` on every decided reply, the Proxy forwarding the
Session's hooks and `run`/`iteration` in player_status."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("champions_gym")

from pcsa.cg.serve import Proxy, Server  # noqa: E402
from tests.cg.helpers import START, TAG, TURN1  # noqa: E402

MON_KEYS = {"species", "form", "hp", "fainted", "status", "boosts", "effects", "moves", "item", "ability", "types",
            "active", "revealed"}  # fmt: skip
VIEW_KEYS = {"turn", "kind", "picks", "opp_preview", "anomalies", "own", "opp", "weather", "fields", "side",
             "opp_side"}  # fmt: skip
PREVIEW = {"v": "0.7", "battle": TAG, "type": "preview_request"}
EVENTS = {"v": "0.7", "battle": TAG, "type": "events", "events": list(TURN1)}
TURN = {"v": "0.7", "battle": TAG, "type": "decision_request", "seq": 0, "kind": "turn"}
STATUS = {"type": "status"}


def serve(*args: str, lines: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pcsa.cg.serve", *args], input="".join(x + "\n" for x in lines),
                          capture_output=True, text=True, encoding="utf-8", timeout=60)  # fmt: skip


def assert_view(v: dict, kind: str) -> None:
    assert set(v) == VIEW_KEYS and v["kind"] == kind and len(v["picks"]) == 3 and len(v["opp_preview"]) == 6
    assert len(v["own"]) == 6 and len(v["opp"]) == 6
    assert all(set(m) == MON_KEYS for m in v["own"] + v["opp"])
    assert all(m["revealed"] for m in v["own"]) and all(0.0 <= m["hp"] <= 1.0 for m in v["own"] + v["opp"])
    assert [m["species"] for m in v["opp"]] == START["opp_preview"]


def test_serve_speaks_the_live_protocol_with_a_builtin_player(tmp_path) -> None:
    sent = [json.dumps(m) for m in (START, PREVIEW, EVENTS, TURN, STATUS)] + ["not json", ""]
    done = serve("--player", "tower", "--seed", "3", "--log-dir", str(tmp_path), lines=sent)
    assert done.returncode == 0, done.stderr
    out = [json.loads(x) for x in done.stdout.splitlines()]
    assert len(out) == len(sent)
    started, picks, events, order, status, bad, empty = out
    assert started is None and events is None
    assert len(picks["picks"]) == 3 and "net" not in picks["notes"]
    assert_view(picks["notes"]["view"], "preview")
    assert order["seq"] == 0 and order["order"]["kind"] in ("move", "switch") and "net" not in order["notes"]
    view = order["notes"]["view"]
    assert_view(view, "turn")
    lead = view["own"][view["picks"][0]]  # the log's lead (CG corrects tower's picks to it)
    assert view["turn"] == 1 and lead["active"] and lead["species"] == "gholdengo"
    assert [m["species"] for m in view["opp"] if m["active"]] == ["glimmora"]
    assert {"action", "kind", "legal", "player", "anomalies", "corrections"} <= set(order["notes"])
    assert status["status"]["player"] == "tower" and "error" in bad and empty == {"error": "empty line"}
    assert (tmp_path / f"{TAG}.jsonl").is_file() and "pcsa.cg.serve: player tower" in done.stderr


def test_serve_cannot_start_with_a_bad_player() -> None:
    done = serve("--player", "nosuchplayer", lines=[json.dumps(START)])
    assert done.returncode == 2 and done.stdout == "" and "nosuchplayer" in done.stderr
    done = serve("--player", "tower", "--player-args", '{"x": 1}', lines=[])
    assert done.returncode == 2 and done.stdout == ""


class Hooked:
    """A Player object with every hook, recording the calls."""

    name = "hooked"
    info = {"run": "league-test", "iteration": 7}

    def __init__(self) -> None:
        self.calls: list[str] = []

    def teampreview(self, battle):
        self.calls.append("teampreview")
        return [0, 1, 2]

    def choose_move(self, battle):
        self.calls.append("choose_move")
        return battle.legal_actions()[0]

    def new_battle(self, tag, resend=False):
        self.calls.append(f"new_battle {tag} {resend}")

    def battle_end(self, result=None):
        self.calls.append(f"battle_end {result}")

    def observe(self, battle, action=None, how="played"):
        self.calls.append(f"observe {how}")

    def status(self):
        return {"name": self.name, "checkpoint": "x.pt"}

    def decision_notes(self):
        return {"fallback": False, "ms": 1.0, "value": 0.5}


def test_the_proxy_forwards_every_hook_and_keeps_the_battle() -> None:
    player = Hooked()
    server = Server(player)
    assert isinstance(server.proxy, Proxy) and server.session.player_name == "hooked"
    assert server.handle(START) is None
    picks = server.handle(PREVIEW)
    assert picks is not None and picks["picks"] == [0, 1, 2] and picks["notes"]["model"]["value"] == 0.5
    assert picks["notes"]["view"]["kind"] == "preview" and "net" not in picks["notes"]
    server.handle(EVENTS)
    order = server.handle(TURN)
    assert order is not None and order["notes"]["view"]["kind"] == "turn"
    assert server.proxy is not None and server.proxy.battle is not None and server.proxy.battle.decision == 0
    again = server.handle(TURN)  # the same request again: answered from CG's record, no Battle of a decision
    assert again is not None and again["order"] == order["order"] and "view" not in again["notes"]
    status = server.handle(STATUS)
    assert status is not None
    assert status["status"]["player_status"] == {"name": "hooked", "checkpoint": "x.pt", "run": "league-test",
                                                 "iteration": 7}  # fmt: skip
    server.handle({"v": "0.7", "battle": TAG, "type": "battle_end", "result": "win"})
    assert player.calls[:3] == [f"new_battle {TAG} False", "teampreview", "choose_move"]
    assert player.calls[-1] == "battle_end win"

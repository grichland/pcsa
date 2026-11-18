"""The shipped cg config resolved at start-up ({python}, a repo-relative checkpoint), the manifest's flat neural
player, and the preview's CG notes split for team_preview.json and the `decision` event."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pcsa.cg import startup
from pcsa.common.config import CGConfig, load_config
from pcsa.common.paths import repo_root
from pcsa.live.decision_record import preview_cg
from pcsa.run.recorder import neural_player


def test_the_shipped_command_resolves_to_this_python_and_an_absolute_checkpoint() -> None:
    cfg = load_config(repo_root() / "config" / "agent.yaml", environ={}).agent.cg
    done = startup.resolved(cfg)
    assert done.command is not None and done.command[0] == sys.executable and done.command[1:3] == ["-m", "pcsa.cg.serve"]
    args = json.loads(done.command[done.command.index("--player-args") + 1])
    want = repo_root() / "assets" / "models" / "league-20261118-h-iter-00140.pt"
    assert Path(args["checkpoint"]) == want and want.is_file()
    assert args["policy"] == "pcsa.network.policy:Policy" and args["fallback"] == "tower"
    assert startup.checkpoint(done) == str(want) and startup.expects_model(done)
    assert done.env == {"CUDA_VISIBLE_DEVICES": "-1"}


def test_resolving_leaves_other_commands_alone() -> None:
    plain = CGConfig(command=["{python}", "-m", "pcsa.cg.serve", "--player", "tower"])
    assert startup.resolved(plain).command == [sys.executable, "-m", "pcsa.cg.serve", "--player", "tower"]
    absolute = str(Path(sys.executable).parent / "x.pt")
    given = CGConfig(command=["py", "--player-args", json.dumps({"checkpoint": absolute})])
    assert json.loads((startup.resolved(given).command or [])[2])["checkpoint"] == absolute
    inprocess = CGConfig(transport="inprocess", entry="a.b:C", args={"player": "tower"})
    assert startup.resolved(inprocess) == inprocess


def test_the_manifest_names_the_neural_player_flat() -> None:
    status = {"player_status": {"checkpoint": "C:\\m\\league-x-iter-00140.pt", "run": "league-x", "iteration": 140,
                                "mode": "argmax", "fallback_player": "tower"}}  # fmt: skip
    params = {"transport": "subprocess", "start_status": {"status": status, "problems": []}}
    name, flat = neural_player("cg", params)
    assert name == "neural" and flat == {"checkpoint": "league-x-iter-00140.pt", "run": "league-x", "update": 140,
                                         "sampling": "argmax", "fallback": "tower", "cg": params}  # fmt: skip
    built_in = {"transport": "subprocess", "start_status": {"status": {"player_status": None}, "problems": []}}
    assert neural_player("cg", built_in) == ("cg", built_in) and neural_player("first", {}) == ("first", {})


def test_the_preview_notes_split() -> None:
    cg = {"anomalies": 0, "model": {"fallback": False, "ms": 8.6, "value": 0.67}, "view": {}, "net": {}}
    rest, found, flat = preview_cg({"cg_ms": 47.0, "cg": cg})
    assert rest == {"cg_ms": 47.0} and found is cg and flat == {"v": 0.67, "ms": 8.6}
    assert preview_cg({"cached": True}) == ({"cached": True}, None, {})

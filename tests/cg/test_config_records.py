"""The shipped cg config resolved at start-up ({python}, a repo-relative checkpoint, the policy's settings), the
manifest's flat neural player, and the preview's CG notes split for team_preview.json and the `decision` event."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pcsa.cg import startup
from pcsa.common.errors import ConfigError
from pcsa.common.paths import repo_root
from pcsa.live.decision_record import preview_cg
from pcsa.run.recorder import neural_player
from tests import configs


def _player_args(cfg) -> dict:
    line = startup.command_line(cfg)
    return json.loads(line[line.index("--player-args") + 1])


def test_the_shipped_command_resolves_to_this_python_and_an_absolute_checkpoint() -> None:
    done = startup.resolved(configs.shipped().agent.cg)
    line = startup.command_line(done)
    assert line[0] == sys.executable and line[1:5] == ["-m", "pcsa.cg.serve", "--player", startup.NEURAL]
    args = _player_args(done)
    want = (repo_root() / configs.shipped().agent.cg.player_args["checkpoint"]).resolve()  # type: ignore[index]
    assert Path(args["checkpoint"]) == want and want.is_file()
    assert args["policy"] == "pcsa.network.policy:Policy" and args["fallback"] == "tower"
    assert startup.checkpoint(done) == str(want) and startup.expects_model(done)
    assert done.env == {}  # the GPU stays visible for the preview search


def test_resolving_leaves_other_players_alone() -> None:
    plain = configs.cg_config(command=["{python}", "-m", "pcsa.cg.serve"], player="tower")
    assert startup.command_line(startup.resolved(plain)) == [sys.executable, "-m", "pcsa.cg.serve", "--player", "tower"]
    absolute = str(Path(sys.executable).parent / "x.pt")
    given = configs.cg_config(command=["py"], player="x:y", player_args={"checkpoint": absolute})
    assert _player_args(startup.resolved(given)) == {"checkpoint": absolute}
    inprocess = configs.cg_config(transport="inprocess", entry="a.b:C", player="tower")
    assert startup.resolved(inprocess) == inprocess
    assert startup.session_args(inprocess) == {"player": "tower"}


def test_the_player_is_set_in_one_place() -> None:
    with pytest.raises(ConfigError, match="cg.player"):
        configs.shipped("cg.command=[py, --player, tower]")
    with pytest.raises(ConfigError, match="cg.player"):
        configs.shipped("cg.transport=inprocess", "cg.entry=a.b:C", "cg.env={}", "cg.args={player: tower}")
    with pytest.raises(ConfigError, match="field required|Field required"):
        configs.shipped("cg.search={enabled: false}")


def test_the_manifest_names_the_neural_player_flat() -> None:
    status = {"player_status": {"checkpoint": "C:\\m\\league-x-iter-00140.pt", "run": "league-x", "iteration": 140,
                                "mode": "argmax", "fallback_player": "tower"}}  # fmt: skip
    params = {"transport": "subprocess", "start_status": {"status": status, "problems": []}}
    name, flat = neural_player("cg", params)
    assert name == "neural" and flat == {"checkpoint": "league-x-iter-00140.pt", "run": "league-x", "update": 140,
                                         "sampling": "argmax", "fallback": "tower", "search": None, "choice": None,
                                         "cg": params}  # fmt: skip
    built_in = {"transport": "subprocess", "start_status": {"status": {"player_status": None}, "problems": []}}
    assert neural_player("cg", built_in) == ("cg", built_in) and neural_player("first", {}) == ("first", {})


def test_the_preview_notes_split() -> None:
    cg = {"anomalies": 0, "model": {"fallback": False, "ms": 8.6, "value": 0.67}, "view": {}, "net": {}}
    rest, found, flat = preview_cg({"cg_ms": 47.0, "cg": cg})
    assert rest == {"cg_ms": 47.0} and found is cg and flat == {"v": 0.67, "ms": 8.6}
    assert preview_cg({"cached": True}) == ({"cached": True}, None, {})


def test_the_search_section_turns_search_on_in_the_players_arguments() -> None:
    no_preview = "cg.preview_search.enabled=false"  # its own wait would raise the limits too
    off = _player_args(startup.resolved(configs.shipped("cg.search.enabled=false", no_preview).agent.cg))
    shipped = configs.shipped().agent.cg.player_args or {}
    assert off["policy_args"]["search"] is None and off["timeout_s"] == shipped["timeout_s"]
    on_cfg = configs.shipped("cg.search.enabled=true", "cg.search.ms=2500", "cg.search.worlds=4", no_preview).agent.cg
    on = _player_args(startup.resolved(on_cfg))
    assert on["policy_args"]["search"] == configs.search(ms=2500.0, worlds=4)
    assert on["timeout_s"] == 3.5 and on["hard_timeout_s"] == 6.5  # the shell's limits clear the search time
    tower = configs.cg_config(command=["{python}", "-m", "pcsa.cg.serve"], player="tower",
                              search={**configs.search(), "enabled": True})  # fmt: skip
    assert startup.command_line(startup.resolved(tower))[-2:] == ["--player", "tower"]


def test_the_rules_and_choice_sections_reach_the_players_arguments() -> None:
    args = _player_args(startup.resolved(configs.shipped().agent.cg))["policy_args"]
    assert args["rules"] == configs.rules()
    assert args["choice"] == configs.shipped().agent.cg.choice.model_dump()
    plain = configs.shipped("cg.choice.rule=mean", "cg.choice.top_theirs=0", "cg.choice.top_ours=0",
                            "cg.rules.enabled=false").agent.cg  # fmt: skip
    args = _player_args(startup.resolved(plain))["policy_args"]
    assert args["choice"] is None and args["rules"] is None  # the network's own pick, no masks


def test_the_manifest_names_the_search_settings() -> None:
    status = {"status": {"player_status": {"checkpoint": "C:/m/x.pt", "search": {"ms": 1500.0, "worlds": 6}}}}
    name, params = neural_player("cg", {"start_status": status})
    assert name == "neural" and params["search"] == {"ms": 1500.0, "worlds": 6}


def test_the_preview_search_section_reaches_the_player_and_raises_the_waits() -> None:
    cg = configs.shipped("cg.search.enabled=false").agent.cg
    args = _player_args(startup.resolved(cg))
    want = cg.preview_search.model_dump(exclude={"enabled"})
    assert args["policy_args"]["preview_search"] == want and cg.preview_search.wait_s() == want["budget_s"] + 6.0
    assert args["timeout_s"] == cg.preview_search.wait_s() + 1.0 and args["hard_timeout_s"] == cg.preview_search.wait_s() + 4.0
    off = configs.shipped("cg.search.enabled=false", "cg.preview_search.enabled=false").agent.cg
    assert _player_args(startup.resolved(off))["policy_args"]["preview_search"] is None and off.preview_search.wait_s() == 0

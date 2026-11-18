"""`pcsa run` arguments, the decide.player key, OwnTeam from the sheet, the supervisor's restart bound."""

from __future__ import annotations

import pytest

from pcsa.battle.teams import team_from_record
from pcsa.common.clock import FakeClock
from pcsa.common.config import Config, load_config
from pcsa.common.errors import ConfigError, Recoverable
from pcsa.live.cli import parser
from pcsa.live.supervisor import supervise
from pcsa.live.team_read import own_team
from pcsa.run.battle_state import RunState
from tests.conftest import REPO
from tests.live.session_harness import TEAM


def test_play_and_check_parse() -> None:
    args = parser().parse_args(["run", "--player", "cg", "--battles", "2", "--set", "run.battles=3"])
    assert (args.command, args.player, args.battles, args.set) == ("run", "cg", 2, ["run.battles=3"])
    assert parser().parse_args(["check"]).command == "check"
    with pytest.raises(SystemExit):
        parser().parse_args(["run", "--player", "random"])


def test_decide_player_defaults_to_first_and_rejects_others(config: Config) -> None:
    assert config.agent.decide.player == "first"
    path = REPO / "config" / "agent.yaml"
    assert load_config(path, ["decide.player=cg", "cg.command=[cg]"], environ={}).agent.decide.player == "cg"
    with pytest.raises(ConfigError, match="cg.command"):  # cg needs a way to reach CG
        load_config(path, ["decide.player=cg"], environ={})
    with pytest.raises(ConfigError):
        load_config(path, ["decide.player=neural"], environ={})


def test_own_team_lists_the_unread_fields() -> None:
    team = own_team(team_from_record(TEAM), picks=[2, 0, 1])
    assert team.names[0] == "Garchomp" and team.brought() == ("Corviknight", "Garchomp", "Primarina")
    assert "Garchomp.ability" in team.unread and "Garchomp.move1" in team.unread  # move row 1, 0-based
    assert "Garchomp.move0" not in team.unread and team.export.startswith("Garchomp")
    bare = own_team(None, ("A", None, "C", "D", "E", "F"))
    assert bare.export == "" and bare.names[1] == "row2" and bare.unread == ("team",)


def test_supervisor_stops_after_too_many_restarts(tmp_path) -> None:
    clock = FakeClock(0.0)
    state = RunState(tmp_path, 1, 2, False, 1)

    class Failing:
        def run(self) -> None:
            raise Recoverable("boom")

    assert supervise(state, lambda s: Failing(), clock, 3) == "too_many_restarts"  # type: ignore[arg-type]
    assert state.restarts == 3

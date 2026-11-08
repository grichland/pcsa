"""Every player kind the config can name plays a battle and a forced replacement on the scripted phone."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcsa.common.config import Config
from pcsa.players.factory import KINDS, build_player
from tests.live.harness import run_script
from tests.live.test_harness import (
    PARTY,
    TURN1,
    TURN2,
    assert_played_cleanly,
    end_of_battle,
    forced_steps,
    lobby_to_team_select,
    one_turn,
)

# The scripts accept only moves on the action screen and expect the first three roster members to be brought;
# a random player (or the mixed pool when it draws one) may switch or bring others.
OFF_SCRIPT = ("random", "mixed")


@pytest.mark.parametrize("kind", KINDS)
def test_kind_plays_a_battle(config: Config, kind: str, tmp_path: Path) -> None:
    steps = lobby_to_team_select() + one_turn(1, TURN1) + one_turn(2, TURN2) + end_of_battle("win")
    state, device, recorder = run_script(config, steps, tmp_path, player=build_player(config, kind))
    assert state.outcomes == ["win"] and (kind in OFF_SCRIPT or device.rejected == []), device.rejected
    assert_played_cleanly(recorder)


@pytest.mark.parametrize("kind", KINDS)
def test_kind_makes_a_forced_replacement(config: Config, kind: str, tmp_path: Path) -> None:
    state, device, recorder = run_script(config, forced_steps(PARTY), tmp_path, player=build_player(config, kind))
    assert state.outcomes == ["loss"] and (kind in OFF_SCRIPT or device.rejected == []), device.rejected
    [path] = (recorder.dir / "battle-1" / "decisions").glob("*-forced.json")
    assert json.loads(path.read_text(encoding="utf-8"))["choice"].startswith("switch")
    assert_played_cleanly(recorder)

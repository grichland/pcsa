"""Shared fixtures. Nothing here touches a phone, the network or the simulator."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pcsa.common import paths
from pcsa.common.clock import FakeClock
from pcsa.common.config import Config, load_config

REPO = paths.repo_root()

# The players' catalog ranks the builds seen on the live ladder first (assets/sets/seen_builds.json, which grows with
# every live battle). Tests must not depend on it, so it points at a file that never exists (the plain catalog order).
os.environ["PCSA_SEEN_BUILDS"] = str(REPO / "tests" / "__no_seen_builds__.json")


@pytest.fixture(autouse=True)
def _empty_seen_builds(tmp_path: Path) -> None:
    """Every test starts with no seen builds, saved (by the live session's end-of-battle hook) under its tmp_path."""
    from pcsa.battle import seen_builds

    store = seen_builds.shared()
    store.path = tmp_path / "seen_builds.json"
    store.by_species.clear()
    store.battles.clear()
    store.version += 1


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock(start=1000.0)


# The neural player's checkpoint in config/players/neural.yaml is a local file (assets/models/*.pt, not committed).
# Tests must not depend on whether one is present, so the shared config points it at a path that never exists:
# `neural` then plays its fallback here as it does in CI; tests of the network pass their own checkpoints.
NO_CHECKPOINT = "policy.neural_checkpoint=assets/models/__absent_in_tests__.pt"


@pytest.fixture
def config() -> Config:
    return load_config(REPO / "config" / "agent.yaml", [NO_CHECKPOINT], environ={})


@pytest.fixture
def output_env(tmp_path: Path) -> dict[str, str]:
    return {paths.OUTPUT_ENV: str(tmp_path / "out")}

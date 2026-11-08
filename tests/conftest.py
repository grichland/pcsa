"""Shared fixtures. Nothing here touches a phone, the network or the simulator."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcsa.common import paths
from pcsa.common.clock import FakeClock
from pcsa.common.config import Config, load_config

REPO = paths.repo_root()


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock(start=1000.0)


@pytest.fixture
def config() -> Config:
    return load_config(REPO / "config" / "agent.yaml", environ={})


@pytest.fixture
def output_env(tmp_path: Path) -> dict[str, str]:
    return {paths.OUTPUT_ENV: str(tmp_path / "out")}

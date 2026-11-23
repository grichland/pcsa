"""Settings for the tests, taken from the shipped config/agent.yaml so that no value is repeated here."""

from __future__ import annotations

from functools import cache
from typing import Any

from pcsa.common import paths
from pcsa.common.config import AgentConfig, CGConfig, Config, load_config

SHIPPED = paths.repo_root() / "config" / "agent.yaml"


def shipped(*overrides: str) -> Config:
    """config/agent.yaml with `overrides` (section.key=value) and no environment overrides."""
    return load_config(SHIPPED, overrides, environ={})


@cache
def _agent() -> AgentConfig:
    return shipped().agent


def section(name: str, **changes: Any) -> Any:
    """One section of the shipped agent config (e.g. "degrade"), with `changes` checked by its class."""
    value = getattr(_agent(), name)
    return type(value).model_validate({**value.model_dump(), **changes})


def cg_config(**changes: Any) -> CGConfig:
    """The shipped cg section without a command, player, environment or log folder, with `changes`."""
    base = _agent().cg.model_dump()
    base.update(command=None, player=None, player_args=None, env={}, run_log_dir=None)
    return CGConfig.model_validate({**base, **changes})


def search(**changes: Any) -> dict:
    """The shipped search settings as the policy's `search` argument, with `changes`."""
    return {**_agent().cg.search.model_dump(exclude={"enabled"}), **changes}


def rules(**changes: Any) -> dict:
    """The shipped rule masks as the policy's `rules` argument, with `changes`."""
    return {**_agent().cg.rules.model_dump(exclude={"enabled"}), **changes}

"""Player by name. The only place that imports pcsa.cg, and only when asked (see .importlinter)."""

from __future__ import annotations

from typing import Any

from pcsa.common.errors import ConfigError
from pcsa.decide.api import Player
from pcsa.decide.builtin import FirstPlayer

PLAYERS = ("first", "cg")


def load_player(name: str, config: Any = None, **context: Any) -> Player:
    """`context` (clock, run_dir) is passed on to a plugin Player."""
    if name == "first":
        return FirstPlayer()
    if name == "cg":
        try:
            from pcsa.cg.adapter import CGPlayer
        except ImportError as error:
            raise ConfigError(f"player 'cg' needs pcsa/cg on the import path: {error}") from error
        return CGPlayer(config, **context)
    raise ConfigError(f"unknown player {name!r}; expected one of {', '.join(PLAYERS)}")

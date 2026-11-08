"""Build a player by kind from the agent config.

Kinds: random; simple_heuristic (the heuristic with poke-env's default numbers); custom_heuristic (the
heuristic with config/players/heuristic.yaml); max_damage (the highest-scoring attack, nothing else);
mixed (a weighted pool of the others, one drawn per game).
"""

from __future__ import annotations

from pcsa.common.config import Config
from pcsa.common.errors import ConfigError
from pcsa.players.base import Player
from pcsa.players.heuristic import HeuristicParams, HeuristicPlayer
from pcsa.players.max_damage import MaxDamagePlayer
from pcsa.players.random_player import RandomPlayer

KINDS = ("random", "simple_heuristic", "custom_heuristic", "max_damage", "mixed")


def build_player(config: Config, kind: str | None) -> Player:
    kind = kind or config.agent.policy.kind
    if kind == "random":
        return RandomPlayer()
    if kind == "simple_heuristic":
        return HeuristicPlayer(HeuristicParams(), name=kind)
    if kind == "custom_heuristic":
        path = config.directory / config.agent.policy.heuristic
        return HeuristicPlayer(HeuristicParams.load(path) if path.is_file() else HeuristicParams(), name=kind)
    if kind == "max_damage":
        return MaxDamagePlayer()
    if kind == "mixed":
        from pcsa.players.mixed import MixedPolicyPlayer

        path = config.directory / config.agent.policy.mixed
        return MixedPolicyPlayer.build(path, lambda k: build_player(config, k))
    raise ConfigError(f"unknown player kind {kind!r} (one of {', '.join(KINDS)})")

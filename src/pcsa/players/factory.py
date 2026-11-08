"""Build a player by kind from the agent config.

Live kinds (the phone can play them): random; simple_heuristic (the heuristic with poke-env's default numbers);
custom_heuristic (the heuristic with config/players/heuristic.yaml); max_damage (the highest-scoring attack,
nothing else); best_matchup_max_damage (max_damage plus a switch out of a really bad matchup,
config/players/best_matchup.yaml); mixed (a weighted pool of the others, one drawn per game).
Sim-only kinds: disciplined (max_damage plus the research rules, config/players/disciplined.yaml).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

from pcsa.battle.builds import Catalog
from pcsa.common.config import Config
from pcsa.common.errors import ConfigError
from pcsa.players.base import Player
from pcsa.players.best_matchup import BestMatchupMaxDamagePlayer, SwitchParams
from pcsa.players.heuristic import HeuristicParams, HeuristicPlayer
from pcsa.players.max_damage import MaxDamagePlayer
from pcsa.players.random_player import RandomPlayer

LIVE_KINDS = ("random", "simple_heuristic", "custom_heuristic", "max_damage", "best_matchup_max_damage", "mixed")
SIM_KINDS = ("disciplined",)  # not offered to the phone yet
SIM_KINDS += ("free_turn",)
SIM_KINDS += ("endgame",)  # disciplined plus the 1v1 chain at <= 2 members each, config/players/endgame.yaml
SIM_KINDS += ("style_weather",)  # free_turn plus a weather trio, config/players/style_weather.yaml
SIM_KINDS += ("style_hazard_lead",)  # free_turn plus a turn-1 Rocks lead, config/players/style_hazard_lead.yaml
SIM_KINDS += ("style_setup_sweeper",)  # free_turn plus a held-back sweeper, config/players/style_setup_sweeper.yaml
KINDS = LIVE_KINDS + SIM_KINDS


def build_player(config: Config, kind: str | None) -> Player:
    kind = kind or config.agent.policy.kind
    builder = BUILDERS.get(kind)
    if builder is None:
        raise ConfigError(f"unknown player kind {kind!r} (one of {', '.join(KINDS)})")
    return builder(config)


def _heuristic_params(config: Config) -> HeuristicParams:
    path = config.directory / config.agent.policy.heuristic
    return HeuristicParams.load(path) if path.is_file() else HeuristicParams()


@cache
def _catalog() -> Catalog:
    """The build catalog, read once per process."""
    return Catalog.load()


def _best_matchup(config: Config) -> Player:
    path = config.directory / config.agent.policy.best_matchup
    switch = SwitchParams.load(path) if path.is_file() else SwitchParams()
    return BestMatchupMaxDamagePlayer(_heuristic_params(config), switch)


def _mixed(config: Config) -> Player:
    from pcsa.players.mixed import MixedPlayer

    return MixedPlayer.build(config.directory / config.agent.policy.mixed, lambda k: build_player(config, k))


def _disciplined(config: Config) -> Player:
    from pcsa.players.disciplined import DisciplinedParams, DisciplinedPlayer

    path = config.directory / config.agent.policy.disciplined
    rules = DisciplinedParams.load(path) if path.is_file() else DisciplinedParams()
    return DisciplinedPlayer(rules, _catalog(), _heuristic_params(config))


def _free_turn(config: Config) -> Player:
    from pcsa.players.free_turn import build

    return build(config, _catalog(), _heuristic_params(config))


def _endgame(config: Config) -> Player:
    from pcsa.players.endgame import build

    return build(config, _catalog(), _heuristic_params(config))


def _style_weather(config: Config) -> Player:
    from pcsa.players.style_weather import build

    return build(config, _catalog(), _heuristic_params(config))


def _style_hazard_lead(config: Config) -> Player:
    from pcsa.players.style_hazard_lead import build

    return build(config, _catalog(), _heuristic_params(config))


def _style_setup_sweeper(config: Config) -> Player:
    from pcsa.players.style_setup_sweeper import build

    return build(config, _catalog(), _heuristic_params(config))


BUILDERS: dict[str, Callable[[Config], Player]] = {
    "random": lambda config: RandomPlayer(),
    "simple_heuristic": lambda config: HeuristicPlayer(HeuristicParams(), name="simple_heuristic"),
    "custom_heuristic": lambda config: HeuristicPlayer(_heuristic_params(config), name="custom_heuristic"),
    "max_damage": lambda config: MaxDamagePlayer(),
    "best_matchup_max_damage": _best_matchup,
    "mixed": _mixed,
    "disciplined": _disciplined,
    "free_turn": _free_turn,
    "endgame": _endgame,
    "style_weather": _style_weather,
    "style_hazard_lead": _style_hazard_lead,
    "style_setup_sweeper": _style_setup_sweeper,
}

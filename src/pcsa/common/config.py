"""Typed configuration: one schema, one loader, unknown keys rejected.

    config = load_config(config_dir() / "agent.yaml", overrides=["timing.max_turn_s=30"])
    config.agent.timing.max_turn_s          # 30.0
    config.screen.resolve("fight", viewport) # pixel rectangle for this device

Environment overrides use double underscores: PCSA__timing__max_turn_s=30.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pcsa.common.errors import ConfigError

ENV_PREFIX = "PCSA__"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# Agent -----------------------------------------------------------------------------------------
class RetryConfig(Strict):
    attempts: int = 4
    base_s: float = 0.5
    max_s: float = 8.0
    reconnect_wait_s: float = 20.0


class DeviceConfig(Strict):
    serial: str | None = None  # None: the single authorized device
    timeout_s: float = 15.0
    stay_awake: bool = True
    retry: RetryConfig = RetryConfig()


class RunConfig(Strict):
    battles: int = 10
    max_battles_total: int = 12
    count_forfeits: bool = False
    max_restarts_per_hour: int = 20


class TimingConfig(Strict):
    inter_tap_s: float = 0.5
    post_action_wait_s: float = 10.0
    poll_interval_s: float = 1.0
    fight_settle_s: float = 0.9
    back_settle_s: float = 0.7
    team_menu_settle_s: float = 0.8
    log_open_settle_s: float = 1.2
    log_swipe_settle_s: float = 0.6
    max_turn_s: float = 40.0
    team_budget_s: float = 45.0
    no_progress_s: float = 420.0
    screen_timeouts: dict[str, float] = Field(default_factory=dict)  # screen name -> seconds before Recoverable
    default_screen_timeout_s: float = 60.0


class DragConfig(Strict):
    x: tuple[float, float]  # viewport fractions
    y_start: tuple[float, float]
    y_end: tuple[float, float]
    steps: int = 8
    hold_s: float = 0.35


class LogCollectConfig(Strict):
    captures_max: int = 10
    seconds_max: float = 12.0
    drag: DragConfig


class TapCounts(Strict):
    focused_move_taps: int = 1
    unfocused_move_taps: int = 2
    focused_party_taps: int = 1
    unfocused_party_taps: int = 2


class NavigatorConfig(Strict):
    unknown_wait_s: float = 5.0
    unknown_max_rounds: int = 6
    edge_settle_s: float = 1.5
    max_taps_per_edge: int = 3
    disconnect_max_taps: int = 40
    disconnect_max_s: float = 180.0
    disconnect_settle_s: float = 3.0
    disconnect_blank_s: float = 45.0


class DegradeConfig(Strict):
    tap_reserve_s: float = 2.5
    decide_reserve_s: float = 3.0
    emergency_reserve_s: float = 1.0
    peek_attempts: int = 2


class PolicyConfig(Strict):
    kind: Literal[
        "random",
        "simple_heuristic",
        "custom_heuristic",
        "max_damage",
        "best_matchup_max_damage",
        "disciplined",
        "mixed",
        "neural",
        "endgame",
    ] = "custom_heuristic"
    heuristic: str = "players/heuristic.yaml"  # the custom heuristic's parameters
    best_matchup: str = "players/best_matchup.yaml"  # best_matchup_max_damage's switch thresholds
    mixed: str = "players/mixed.yaml"
    disciplined: str = "players/disciplined.yaml"  # the disciplined player's thresholds
    endgame: str = "players/endgame.yaml"  # the endgame player's chain trigger (sim only)
    neural: str = "players/neural.yaml"  # the neural player's checkpoint, sampling and fallback
    neural_checkpoint: str | None = None  # when set, replaces neural.yaml's checkpoint (one --set per run)
    neural_rules: str | None = None  # when set, replaces neural.yaml's rules: none, or names separated by commas


class SimConfig(Strict):
    format: str = "gen9championsbssregmc"


class OcrConfig(Strict):
    det_model: str = "ch_PP-OCRv5_det_mobile.onnx"
    rec_model: str = "en_PP-OCRv5_rec_mobile.onnx"
    det_limit_side_len: int = 736
    threads: int = 4
    min_confidence: float = 0.80


class AgentConfig(Strict):
    screen_config: str = "screen.yaml"
    device: DeviceConfig = DeviceConfig()
    run: RunConfig = RunConfig()
    timing: TimingConfig = TimingConfig()
    log: LogCollectConfig
    taps: TapCounts = TapCounts()
    navigator: NavigatorConfig = NavigatorConfig()
    degrade: DegradeConfig = DegradeConfig()
    policy: PolicyConfig = PolicyConfig()
    sim: SimConfig = SimConfig()
    ocr: OcrConfig = OcrConfig()


# Screen ----------------------------------------------------------------------------------------
class Region(Strict):
    """A rectangle in viewport fractions (0..1). `relative` regions are offsets from another point."""

    x: float
    y: float
    w: float
    h: float
    relative: bool = False

    @field_validator("w", "h")
    @classmethod
    def _positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("width and height must be positive")
        return value


class Circle(Strict):
    x: float
    y: float
    r: float


class Viewport(Strict):
    """Where the game draws inside the captured frame, in pixels. Default: the whole frame."""

    x: int = 0
    y: int = 0
    w: int
    h: int

    @classmethod
    def of_frame(cls, width: int, height: int) -> Viewport:
        return cls(w=width, h=height)


class Rect(Strict):
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class ScreenConfig(Strict):
    regions: dict[str, Region | list[Region]]
    forbidden: list[Circle] = Field(default_factory=list)

    def region(self, name: str, index: int | None = None) -> Region:
        value = self.regions[name]
        if isinstance(value, list):
            if index is None:
                raise KeyError(f"{name} is a list of regions; an index is required")
            return value[index]
        if index is not None:
            raise KeyError(f"{name} is a single region")
        return value

    def resolve(self, name: str, viewport: Viewport, index: int | None = None) -> Rect:
        region = self.region(name, index)
        if region.relative:
            raise ValueError(f"{name} is relative to another point; use resolve_relative")
        return self._rect(region, viewport, 0.0, 0.0)

    def resolve_relative(self, name: str, viewport: Viewport, origin: tuple[float, float]) -> Rect:
        """A relative region placed against an origin given in viewport fractions."""
        region = self.region(name)
        return self._rect(region, viewport, origin[0], origin[1])

    @staticmethod
    def _rect(region: Region, viewport: Viewport, ox: float, oy: float) -> Rect:
        return Rect(
            x1=round(viewport.x + (ox + region.x) * viewport.w),
            y1=round(viewport.y + (oy + region.y) * viewport.h),
            x2=round(viewport.x + (ox + region.x + region.w) * viewport.w),
            y2=round(viewport.y + (oy + region.y + region.h) * viewport.h),
        )


# Loader ----------------------------------------------------------------------------------------
class Config(Strict):
    path: Path
    agent: AgentConfig
    screen: ScreenConfig

    @property
    def directory(self) -> Path:
        return self.path.parent


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"configuration file must contain a mapping: {path}")
    return data


def _set_path(data: dict[str, Any], dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {dotted}: {key} is not a section")
    node[keys[-1]] = yaml.safe_load(raw)


def overrides_from_environ(environ: Mapping[str, str]) -> list[str]:
    """PCSA__section__key=value -> "section.key=value". The key is lower-cased: Windows upper-cases environment
    variable names (PCSA__policy__kind arrives as PCSA__POLICY__KIND) and every config key is lower case."""
    found = []
    for name, value in environ.items():
        if name.upper().startswith(ENV_PREFIX.upper()) and len(name) > len(ENV_PREFIX):
            found.append(name[len(ENV_PREFIX) :].replace("__", ".").lower() + "=" + value)
    return sorted(found)


def load_config(path: str | Path, overrides: Iterable[str] = (), environ: Mapping[str, str] | None = None) -> Config:
    """Read agent.yaml and the screen.yaml it names; environment overrides apply first, then `overrides`."""
    env = os.environ if environ is None else environ
    path = Path(path).resolve()
    agent = _read_yaml(path)
    for item in [*overrides_from_environ(env), *overrides]:
        if "=" not in item:
            raise ConfigError(f"override must look like section.key=value: {item}")
        dotted, raw = item.split("=", 1)
        _set_path(agent, dotted.strip(), raw)
    try:
        agent_config = AgentConfig.model_validate(agent)
        screen_config = ScreenConfig.model_validate(_read_yaml(path.parent / agent_config.screen_config))
    except ValidationError as error:
        raise ConfigError(_describe(error)) from None
    return Config(path=path, agent=agent_config, screen=screen_config)


def _describe(error: ValidationError) -> str:
    lines = []
    for problem in error.errors():
        where = ".".join(str(p) for p in problem["loc"]) or "<root>"
        lines.append(f"{where}: {problem['msg']}")
    return "invalid configuration: " + "; ".join(lines)

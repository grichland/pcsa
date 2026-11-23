"""Typed configuration: one schema, one loader, unknown and missing keys rejected.

    config = load_config(config_dir() / "agent.yaml", overrides=["timing.max_turn_s=30"])

Every value is set in config/agent.yaml; the classes here carry types and checks only. Environment overrides use
double underscores: PCSA__timing__max_turn_s=30.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from pcsa.common.config_cg import CGSearch, ChoiceSettings, PreviewSearch, RuleSettings, ValueCalibration
from pcsa.common.errors import ConfigError

ENV_PREFIX = "PCSA__"
PLAYER_FLAGS = ("--player", "--player-args")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# Agent -----------------------------------------------------------------------------------------
class RetryConfig(Strict):
    attempts: int
    base_s: float
    max_s: float
    reconnect_wait_s: float


class DeviceConfig(Strict):
    serial: str | None  # None: the single authorized device
    timeout_s: float
    stay_awake: bool
    retry: RetryConfig


class RunConfig(Strict):
    name: str = Field(pattern=r"^[\w.-]+$")  # run folder prefix
    seed: int | None  # None: a new random seed
    battles: int = Field(gt=0)
    max_battles_total: int = Field(gt=0)
    count_forfeits: bool
    max_restarts_per_hour: int


class TimingConfig(Strict):
    inter_tap_s: float
    post_action_wait_s: float
    poll_interval_s: float
    poll_cycle_s: float
    fight_settle_s: float
    back_settle_s: float
    party_settle_s: float
    move_peek: Literal["always", "when_needed"]
    team_menu_settle_s: float
    log_open_settle_s: float
    log_swipe_settle_s: float
    max_turn_s: float
    team_budget_s: float
    no_progress_s: float
    screen_timeouts: dict[str, float]  # screen name -> seconds before the session restarts
    default_screen_timeout_s: float


class DragConfig(Strict):
    x: tuple[float, float]
    y_start: tuple[float, float]
    y_end: tuple[float, float]
    steps: int
    hold_s: float


class PagingConfig(Strict):
    """Paging back to turns the log skipped past."""

    enabled: bool
    settle_s: float
    step_s: float
    page_min_s: float
    seconds_max: float


class LogCollectConfig(Strict):
    captures_max: int
    seconds_max: float
    drag: DragConfig
    paging: PagingConfig


class TapCounts(Strict):
    focused_move_taps: int
    unfocused_move_taps: int
    focused_party_taps: int
    unfocused_party_taps: int


class NavigatorConfig(Strict):
    unknown_wait_s: float
    unknown_max_rounds: int
    edge_settle_s: float
    max_taps_per_edge: int
    disconnect_max_taps: int
    disconnect_max_s: float
    disconnect_settle_s: float
    disconnect_blank_s: float


class DegradeConfig(Strict):
    tap_reserve_s: float
    decide_reserve_s: float
    peek_attempts: int


class OcrConfig(Strict):
    det_model: str
    rec_model: str
    det_limit_side_len: int
    det_limit_type: Literal["min", "max"]
    threads: int
    min_confidence: float


class DecideConfig(Strict):
    player: Literal["first", "cg"]  # pcsa.decide.load.PLAYERS


class CGConfig(Strict):
    """How Champions Gym is reached and which player it runs (pcsa.cg).

    `subprocess` runs `command` (JSON lines on stdin and stdout; `{python}` is the interpreter running pcsa);
    `inprocess` loads `entry` ("module:name", a handle(message) callable or a class with .handle) with `args`.
    `player` and `player_args` are the Session's player for both transports; a relative checkpoint in `player_args`
    is taken from the repo root. `rules`, `search`, `choice`, `value_calibration`, `preview_search`: the policy's."""

    transport: Literal["inprocess", "subprocess"]
    command: list[str] | None
    entry: str | None
    args: dict[str, Any]
    player: str | None
    player_args: dict[str, Any] | None
    python_path: list[str]  # first on sys.path (inprocess) or in the child's PYTHONPATH; relative: the repo root
    env: dict[str, str]
    run_log_dir: str | None  # a folder inside the run folder for CG's logs
    timeout_s: float = Field(gt=0)  # per call; also capped by the decision's deadline
    start_timeout_s: float = Field(gt=0)
    rules: RuleSettings
    search: CGSearch
    choice: ChoiceSettings
    value_calibration: ValueCalibration
    preview_search: PreviewSearch

    @field_validator("entry")
    @classmethod
    def _entry(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[\w.]+:[\w.]+", value):
            raise ValueError("must look like package.module:name")
        return value

    @field_validator("command")
    @classmethod
    def _command(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(flag in value for flag in PLAYER_FLAGS):
            raise ValueError("the player is set in cg.player and cg.player_args, not in the command")
        return value

    @field_validator("args")
    @classmethod
    def _args(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "player" in value or "player_args" in value:
            raise ValueError("the player is set in cg.player and cg.player_args, not in cg.args")
        return value

    @field_validator("python_path")
    @classmethod
    def _directories(cls, value: list[str]) -> list[str]:
        if any(not isinstance(p, str) or not p.strip() for p in value):
            raise ValueError("must be a list of directories")
        return value

    @field_validator("env")
    @classmethod
    def _env(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) or not isinstance(v, str) for k, v in value.items()):
            raise ValueError("must map variable names to strings")
        return value

    @field_validator("run_log_dir")
    @classmethod
    def _relative(cls, value: str | None) -> str | None:
        win = PureWindowsPath(value or "")  # either separator and a drive letter, on any system
        outside = Path(value or "").is_absolute() or win.is_absolute() or ".." in win.parts
        if value is not None and (not value.strip() or outside):
            raise ValueError("must be a folder name inside the run folder")
        return value

    @model_validator(mode="after")
    def _transport(self) -> CGConfig:
        if self.args and self.transport != "inprocess":
            raise ValueError("cg.args is for the inprocess transport")
        if self.env and self.transport != "subprocess":
            raise ValueError("cg.env is for the subprocess transport; in process CG shares the controller's")
        if self.player_args and not self.player:
            raise ValueError("cg.player_args needs cg.player")
        return self


class AgentConfig(Strict):
    screen_config: str
    decide: DecideConfig
    device: DeviceConfig
    run: RunConfig
    timing: TimingConfig
    log: LogCollectConfig
    taps: TapCounts
    navigator: NavigatorConfig
    degrade: DegradeConfig
    ocr: OcrConfig
    cg: CGConfig

    @model_validator(mode="after")
    def _consistent(self) -> AgentConfig:
        if self.run.max_battles_total < self.run.battles:
            raise ValueError("run.max_battles_total must be at least run.battles")
        if self.decide.player != "cg":
            return self
        cg = self.cg
        if cg.transport == "inprocess" and not cg.entry:
            raise ValueError("cg.entry is needed for the inprocess transport")
        if cg.transport == "subprocess" and not cg.command:
            raise ValueError("cg.command is needed for the subprocess transport")
        if cg.timeout_s >= self.timing.max_turn_s - self.degrade.tap_reserve_s:
            raise ValueError("cg.timeout_s must leave the tap reserve inside timing.max_turn_s")
        return self


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
    variable names (PCSA__run__battles arrives as PCSA__RUN__BATTLES) and every config key is lower case."""
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

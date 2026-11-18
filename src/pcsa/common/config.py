"""Typed configuration: one schema, one loader, unknown keys rejected.

    config = load_config(config_dir() / "agent.yaml", overrides=["timing.max_turn_s=30"])

Environment overrides use double underscores: PCSA__timing__max_turn_s=30.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

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
    party_settle_s: float = 0.9  # live/party_switch.py: after Pokémon and after Switch In, before reading the screen
    move_peek: Literal["always", "when_needed"] = "always"
    team_menu_settle_s: float = 0.8
    log_open_settle_s: float = 1.2
    log_swipe_settle_s: float = 0.6
    max_turn_s: float = 40.0
    team_budget_s: float = 45.0
    no_progress_s: float = 420.0
    screen_timeouts: dict[str, float] = Field(default_factory=dict)  # screen name -> seconds before Recoverable
    default_screen_timeout_s: float = 60.0


class DragConfig(Strict):
    x: tuple[float, float]
    y_start: tuple[float, float]
    y_end: tuple[float, float]
    steps: int = 8
    hold_s: float = 0.35


class PagingConfig(Strict):
    """Back-paging: reading turns whose page the log skipped past (no action screen of ours on them)."""

    enabled: bool = False
    settle_s: float = 0.8  # after an arrow tap, before the heading is read
    step_s: float = 1.6  # planning estimate for one arrow tap and its heading check
    page_min_s: float = 4.0  # planning estimate (and minimum time left) for collecting one page
    seconds_max: float = 10.0  # climb budget per missed page


class LogCollectConfig(Strict):
    captures_max: int = 10
    seconds_max: float = 12.0
    drag: DragConfig
    paging: PagingConfig = PagingConfig()


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
    peek_attempts: int = 2


class OcrConfig(Strict):
    det_model: str = "ch_PP-OCRv5_det_mobile.onnx"
    rec_model: str = "en_PP-OCRv5_rec_mobile.onnx"
    det_limit_side_len: int = 736
    threads: int = 4
    min_confidence: float = 0.80


class DecideConfig(Strict):
    player: Literal["first", "cg"] = "first"  # decide/load.py PLAYERS


class CGConfig(Strict):
    """pcsa/cg: how CG is reached. `entry` "module:attr" (a handle(message) callable, or a class or object with
    .handle) for `inprocess`; `command` (argv, JSON lines on stdin/stdout) for `subprocess`, where `{python}` is the
    interpreter running pcsa; a relative `checkpoint` in the player_args is taken from the repo root. `python_path`:
    directories put first on sys.path (inprocess) or in the child's PYTHONPATH (subprocess); relative ones from the
    repo root. `args`: keyword arguments for an entry that is a class, e.g. {player: tower, seed: 0}. `env`: extra
    environment variables for the subprocess. `run_log_dir`: a folder in the run folder given to CG as its log_dir
    (and to a model player's player_args.log_dir) when CG is made. `start_timeout_s`: the start-up status check."""

    transport: Literal["inprocess", "subprocess"] = "subprocess"
    entry: str | None = None
    command: list[str] | None = None
    python_path: list[str] = []
    args: dict[str, Any] = {}
    env: dict[str, str] = {}
    run_log_dir: str | None = None
    timeout_s: float = 8.0  # per call; also capped by the decision's deadline
    start_timeout_s: float = 60.0

    @field_validator("timeout_s", "start_timeout_s")
    @classmethod
    def _positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("entry")
    @classmethod
    def _entry(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[\w.]+:[\w.]+", value):
            raise ValueError("must look like package.module:name")
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
        if value is not None and (not value.strip() or Path(value).is_absolute() or ".." in Path(value).parts):
            raise ValueError("must be a folder name inside the run folder")
        return value

    @model_validator(mode="after")
    def _args_inprocess(self) -> CGConfig:
        if self.args and self.transport != "inprocess":
            raise ValueError("cg.args is for the inprocess transport; a subprocess takes them in cg.command")
        if self.env and self.transport != "subprocess":
            raise ValueError("cg.env is for the subprocess transport; in process CG shares the controller's")
        return self


class AgentConfig(Strict):
    screen_config: str = "screen.yaml"
    decide: DecideConfig = DecideConfig()
    device: DeviceConfig = DeviceConfig()
    run: RunConfig = RunConfig()
    timing: TimingConfig = TimingConfig()
    log: LogCollectConfig
    taps: TapCounts = TapCounts()
    navigator: NavigatorConfig = NavigatorConfig()
    degrade: DegradeConfig = DegradeConfig()
    ocr: OcrConfig = OcrConfig()
    cg: CGConfig = CGConfig()

    @model_validator(mode="after")
    def _cg_reachable(self) -> AgentConfig:
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

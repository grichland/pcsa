"""Devices and perception that replay a script, so the whole loop runs without a phone.

A script is a list of steps. Each step is one screen the phone would show, with the taps that move
it to the next step. The device stays on a step until an accepted tap arrives (or `after_s` of
clock time passes); a tap that is not accepted is a test failure, recorded in `rejected`.

    Step(Screen.LOBBY, {...}, accept=("begin_matchmaking",))
    Step(Screen.MATCHMAKING, after_s=5)
    Step(Screen.ACTION, {...}, accept=("fight",), inject="screencap_fail")  # or "tap_fail"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pcsa.common.clock import Clock
from pcsa.common.config import ScreenConfig, Viewport
from pcsa.common.errors import Transient
from pcsa.common.types import Flag, Observation, Screen
from pcsa.live.device import Transport
from pcsa.phone.adb import Frame
from pcsa.phone.ocr import REFERENCE, TextBox


@dataclass
class Step:
    screen: Screen
    fields: dict[str, Any] = field(default_factory=dict)
    flags: frozenset[Flag] = frozenset()
    texts: tuple[TextBox, ...] = ()
    accept: tuple[str, ...] = ()
    after_s: float | None = None
    inject: str | None = None
    note: str = ""
    advance_on: tuple[str, ...] | None = None  # accepted taps that move to the next step (default: all accepted)
    log_rows: tuple[str, ...] = ()  # what the battle log would show on this step (harness log reader)
    log_turn: int | None = None


class ScriptedDevice:
    """Serves the script's screens; resolves taps to region names through the screen config."""

    serial = "SCRIPTED"

    def __init__(
        self, steps: list[Step], screen: ScreenConfig, clock: Clock, size: tuple[int, int] = REFERENCE
    ) -> None:
        self.steps = steps
        self.screen = screen
        self.clock = clock
        self.size = size
        self.viewport = Viewport.of_frame(*size)
        self.index = 0
        self.entered_at = clock.now()
        self.taps: list[tuple[int, str | None]] = []  # (step index, region name or None)
        self.rejected: list[tuple[int, int, int]] = []
        self.failed_captures = 0

    @property
    def step(self) -> Step:
        return self.steps[min(self.index, len(self.steps) - 1)]

    @property
    def finished(self) -> bool:
        return self.index >= len(self.steps)

    def _advance(self) -> None:
        self.index += 1
        self.entered_at = self.clock.now()

    def _maybe_timeout(self) -> None:
        step = self.step
        if step.after_s is not None and self.clock.now() - self.entered_at >= step.after_s:
            self._advance()

    def region_at(self, x: int, y: int) -> str | None:
        """Which configured region (name, or name+index for lists) contains a device pixel.

        Regions of different screens overlap; the one the current screen allows wins."""
        from pcsa.live.taps import ALLOWED

        hits = []
        for name, value in self.screen.regions.items():
            regions = value if isinstance(value, list) else [value]
            for i, region in enumerate(regions):
                if region.relative:
                    if self._relative_hit(name, x, y):
                        hits.append((name, name))
                    continue
                rect = self.screen.resolve(name, self.viewport, i if isinstance(value, list) else None)
                if rect.contains(x, y):
                    hits.append((name, f"{name}{i}" if isinstance(value, list) else name))
        allowed = ALLOWED.get(self.step.screen, frozenset())
        preferred = [label for name, label in hits if name in allowed]
        return (preferred or [label for _, label in hits] or [None])[0]

    def _relative_hit(self, name: str, x: int, y: int) -> bool:
        """A relative region (the Switch In button) placed against every party row centre."""
        rows = self.screen.regions.get("party_rows")
        if not isinstance(rows, list):
            return False
        for i in range(len(rows)):
            centre = self.screen.resolve("party_rows", self.viewport, i).center
            origin = (centre[0] / self.viewport.w, centre[1] / self.viewport.h)
            if self.screen.resolve_relative(name, self.viewport, origin).contains(x, y):
                return True
        return False

    def capture(self) -> Frame:
        self._maybe_timeout()
        if self.step.inject == "screencap_fail" and self.failed_captures < 1:
            self.failed_captures += 1
            raise Transient("raw screenshot length differs from its dimensions")
        image = np.full((self.size[1], self.size[0], 3), 40, np.uint8)
        return Frame(image, self.size, self.clock.now(), 0.05)

    def control_at(self, x: int, y: int) -> str | None:
        """An OCR-located control of the current step whose box contains the pixel: "@<field>"."""
        for key, value in self.step.fields.items():
            if isinstance(value, dict) and "box" in value:
                left, top, right, bottom = value["box"]
                if left <= x <= right and top <= y <= bottom:
                    return f"@{key}"
        return None

    def tap(self, x: int, y: int) -> Transport:
        """Accepted names: a region ("fight", "move_rows2" or its list name "move_rows") or "@field"."""
        step = self.step
        candidates = [n for n in (self.region_at(x, y), self.control_at(x, y)) if n is not None]
        matched = [n for n in candidates if n in step.accept or n.rstrip("0123456789") in step.accept]
        name = matched[0] if matched else (candidates[0] if candidates else None)
        self.taps.append((self.index, name))
        if step.inject == "tap_fail" and not any(i == self.index for i, _ in self.taps[:-1]):
            raise Transient("adb shell input tap failed (exit 1)")
        if not matched:
            self.rejected.append((self.index, x, y))
            return Transport("tap", (x, y), 0.02)
        advance_on = step.accept if step.advance_on is None else step.advance_on
        if any(n in advance_on or n.rstrip("0123456789") in advance_on for n in matched):
            self._advance()
        return Transport("tap", (x, y), 0.02)

    def drag(self, x: int, y1: int, y2: int, steps: int = 8, hold_s: float = 0.35) -> Transport:
        return Transport("drag", (x, y1, y2), 0.1)

    def shell(self, *arguments: str) -> str:
        return ""


class ScriptedPerception:
    """Returns the current step's observation instead of running OCR."""

    def __init__(self, device: ScriptedDevice) -> None:
        self.device = device

    def observe(self, image: np.ndarray, frame_id: str = "") -> Observation:
        self.device._maybe_timeout()
        step = self.device.step
        return Observation(
            step.screen,
            self.device.size,
            step.flags,
            dict(step.fields),
            f"scripted:{step.note}",
            step.texts,
            frame_id,
        )

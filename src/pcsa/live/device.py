"""Device input for live play: taps, vertical drags and keeping the screen awake, on top of the
observation-only `phone.adb.AdbDevice`. Every call retries and reconnects on failure like the rest of adb."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pcsa.phone.adb import AdbDevice, Frame


@dataclass(frozen=True)
class Transport:
    kind: str
    args: tuple[Any, ...]
    seconds: float


class InputDevice(Protocol):
    serial: str

    def capture(self) -> Frame: ...

    def tap(self, x: int, y: int) -> Transport: ...

    def drag(self, x: int, y1: int, y2: int, steps: int, hold_s: float) -> Transport: ...

    def shell(self, *arguments: str) -> str: ...


class LiveDevice(AdbDevice):
    _stay_on_previous: str | None = None

    def _input(self, kind: str, *values: Any) -> Transport:
        started = self.clock.now()
        self._guarded(kind, lambda: self._device(["shell", "input", kind, *map(str, values)]))
        return Transport(kind, values, self.clock.now() - started)

    def tap(self, x: int, y: int) -> Transport:
        return self._input("tap", int(x), int(y))

    def drag(self, x: int, y1: int, y2: int, steps: int = 8, hold_s: float = 0.35) -> Transport:
        """Vertical drag that pauses before lifting, so a list does not fling."""
        points = [round(y1 + (y2 - y1) * i / steps) for i in range(1, steps + 1)]
        script = "; ".join(
            [
                f"input motionevent DOWN {x} {y1}",
                *(f"input motionevent MOVE {x} {y}" for y in points),
                f"sleep {hold_s}",
                f"input motionevent UP {x} {y2}",
            ]
        )
        started = self.clock.now()
        self._guarded("drag", lambda: self._device(["shell", script]))
        return Transport("drag", (x, y1, y2, steps, hold_s), self.clock.now() - started)

    def stay_awake(self, on: bool) -> None:
        if on:
            if self._stay_on_previous is None:
                self._stay_on_previous = self.shell("settings", "get", "global", "stay_on_while_plugged_in")
            self.shell("svc", "power", "stayon", "usb")
        elif self._stay_on_previous is not None:
            previous = self._stay_on_previous if self._stay_on_previous.isdigit() else "0"
            self.shell("settings", "put", "global", "stay_on_while_plugged_in", previous)
            self._stay_on_previous = None

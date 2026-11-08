"""ADB transport for observation: screenshots, shell queries and device facts. Every call retries and
reconnects on failure. Device input (taps, drags) lives in pcsa.live.

No recognition here. A failure that survives the configured attempts raises `Transient`, which the
session turns into a bounded wait and the supervisor into a restart; it never ends the run.
"""

from __future__ import annotations

import struct
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from pcsa.common import log
from pcsa.common.clock import Clock
from pcsa.common.config import DeviceConfig
from pcsa.common.errors import ConfigError, Transient
from pcsa.phone.retry import with_retry

RAW_HEADER = 16
RGBA = 1
MAX_SIDE = 10_000
DEVICE_FIELDS = 2  # "serial state ..." in `adb devices -l`


@dataclass(frozen=True)
class Frame:
    image: np.ndarray = field(repr=False)
    size: tuple[int, int]  # width, height
    captured_at: float  # clock time when the capture started
    seconds: float


class Device(Protocol):
    serial: str

    def capture(self) -> Frame: ...

    def shell(self, *arguments: str) -> str: ...


Runner = Callable[..., subprocess.CompletedProcess[bytes]]


def decode_raw(data: bytes) -> np.ndarray:
    """Decode `screencap` raw output (16-byte header + RGBA) to BGR."""
    if len(data) < RAW_HEADER:
        raise Transient("raw screenshot header is incomplete")
    width, height, pixel_format, _ = struct.unpack("<4I", data[:RAW_HEADER])
    if pixel_format != RGBA or not (0 < width <= MAX_SIDE and 0 < height <= MAX_SIDE):
        raise Transient("unsupported raw screenshot dimensions or pixel format")
    if len(data) != RAW_HEADER + width * height * 4:
        raise Transient("raw screenshot length differs from its dimensions")
    rgba = np.frombuffer(data, np.uint8, offset=RAW_HEADER).reshape(height, width, 4)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def parse_devices(listing: str) -> dict[str, str]:
    """`adb devices -l` output -> {serial: state}."""
    devices = {}
    for line in listing.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= DEVICE_FIELDS and not line.startswith("*"):
            devices[fields[0]] = fields[1]
    return devices


class AdbDevice:
    """One authorized device. Construction resolves the serial; every later call retries and reconnects."""

    def __init__(self, adb_path: Path, config: DeviceConfig, clock: Clock, runner: Runner = subprocess.run) -> None:
        if not adb_path.is_file():
            raise ConfigError(f"adb executable not found: {adb_path} (run pcsa setup)")
        self.adb = adb_path
        self.config = config
        self.clock = clock
        self.runner = runner
        self.serial = self._resolve(config.serial)

    # Transport -----------------------------------------------------------------------------------
    def _raw(self, arguments: list[str], timeout: float | None = None) -> bytes:
        command = [str(self.adb), *arguments]
        try:
            result = self.runner(command, capture_output=True, check=True, timeout=timeout or self.config.timeout_s)
        except subprocess.TimeoutExpired as error:
            raise Transient(f"adb {' '.join(arguments[:3])} timed out") from error
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or b"").decode("utf-8", "replace").strip()
            raise Transient(f"adb {' '.join(arguments[:3])} failed (exit {error.returncode}): {detail}") from error
        except OSError as error:
            raise Transient(f"cannot execute adb at {self.adb}: {error}") from error
        return result.stdout

    def _device(self, arguments: list[str], timeout: float | None = None) -> bytes:
        return self._raw(["-s", self.serial, *arguments], timeout)

    def _guarded(self, operation: str, attempt: Callable[[], Any]) -> Any:
        return with_retry(operation, attempt, self.config.retry, self.clock, lambda n, e: self.reconnect())

    # Device selection ----------------------------------------------------------------------------
    def _resolve(self, wanted: str | None) -> str:
        devices = parse_devices(self._raw(["devices", "-l"]).decode("utf-8", "replace"))
        if wanted is None:
            ready = [name for name, state in devices.items() if state == "device"]
            if len(ready) != 1:
                raise Transient(f"expected exactly one authorized device; found {devices or 'none'}")
            return ready[0]
        if devices.get(wanted) != "device":
            raise Transient(f"device {wanted!r} is {devices.get(wanted, 'absent')}; expected 'device'")
        return wanted

    def reconnect(self) -> None:
        """Ask adb to reconnect, wait for the device, and verify it is the same one."""
        log.event("device_reconnect", serial=self.serial)
        try:
            self._raw(["reconnect"], timeout=10)
        except Transient as error:
            log.warning("device_reconnect_failed", reason=str(error))
        try:
            self._raw(["-s", self.serial, "wait-for-device"], timeout=self.config.retry.reconnect_wait_s)
            self.serial = self._resolve(self.serial if self.config.serial else None)
            log.event("device_reconnected", serial=self.serial)
        except Transient as error:
            log.warning("device_still_absent", reason=str(error))

    # Operations ----------------------------------------------------------------------------------
    def capture(self) -> Frame:
        started = self.clock.now()
        image = self._guarded("screencap", lambda: decode_raw(self._device(["exec-out", "screencap"])))
        return Frame(image, (image.shape[1], image.shape[0]), started, self.clock.now() - started)

    def shell(self, *arguments: str) -> str:
        return self._guarded("shell", lambda: self._device(["shell", *arguments]).decode("utf-8", "replace").strip())

    def facts(self) -> dict[str, str]:
        """Model, Android version and screen size, for the run manifest and the device profile."""
        return {
            "serial": self.serial,
            "model": self.shell("getprop", "ro.product.model"),
            "android": self.shell("getprop", "ro.build.version.release"),
            "size": self.shell("wm", "size"),
        }

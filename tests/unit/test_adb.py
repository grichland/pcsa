import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pcsa.common.clock import FakeClock
from pcsa.common.config import DeviceConfig, RetryConfig
from pcsa.common.errors import Transient
from pcsa.phone.adb import AdbDevice, decode_raw, parse_devices


def raw_frame(width: int = 4, height: int = 2) -> bytes:
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[..., 0] = 255  # red in RGBA
    return struct.pack("<4I", width, height, 1, 0) + pixels.tobytes()


def test_decode_raw_converts_rgba_to_bgr() -> None:
    image = decode_raw(raw_frame())
    assert image.shape == (2, 4, 3)
    assert tuple(image[0, 0]) == (0, 0, 255)


def test_decode_raw_rejects_short_or_mismatched_buffers() -> None:
    with pytest.raises(Transient):
        decode_raw(b"short")
    with pytest.raises(Transient, match="length differs"):
        decode_raw(raw_frame()[:-3])


def test_parse_devices() -> None:
    listing = "List of devices attached\nABCD1234EFGH device product:x model:SM_S948U1\nemulator-5554 offline\n"
    assert parse_devices(listing) == {"ABCD1234EFGH": "device", "emulator-5554": "offline"}


class FakeRunner:
    """Scripted subprocess.run: a queue of outcomes per command kind."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.screencap_outcomes: list[bytes | Exception] = []
        self.devices = "List of devices attached\nSERIAL device model:X\n"

    def __call__(self, command, capture_output, check, timeout):
        self.calls.append(command[1:])
        args = command[1:]
        if args[:2] == ["devices", "-l"]:
            return subprocess.CompletedProcess(command, 0, self.devices.encode(), b"")
        if "screencap" in args:
            outcome = self.screencap_outcomes.pop(0) if self.screencap_outcomes else raw_frame()
            if isinstance(outcome, Exception):
                raise outcome
            return subprocess.CompletedProcess(command, 0, outcome, b"")
        return subprocess.CompletedProcess(command, 0, b"", b"")


@pytest.fixture
def device(tmp_path: Path, fake_clock: FakeClock):
    adb = tmp_path / "adb.exe"
    adb.write_bytes(b"")
    runner = FakeRunner()
    config = DeviceConfig(retry=RetryConfig(attempts=3, base_s=0.5, max_s=2.0))
    return AdbDevice(adb, config, fake_clock, runner=runner), runner, fake_clock


def test_capture_recovers_from_a_length_mismatch_then_a_timeout(device) -> None:
    dev, runner, clock = device
    runner.screencap_outcomes = [raw_frame()[:-5], subprocess.TimeoutExpired("adb", 15), raw_frame()]
    frame = dev.capture()
    assert frame.size == (4, 2)
    reconnects = [c for c in runner.calls if c == ["reconnect"]]
    assert len(reconnects) == 2
    assert clock.sleeps == [0.5, 1.0]


def test_capture_gives_up_after_the_configured_attempts(device) -> None:
    dev, runner, _ = device
    runner.screencap_outcomes = [raw_frame()[:-5]] * 3
    with pytest.raises(Transient, match="after 3 attempts"):
        dev.capture()


def test_missing_adb_is_a_config_error(tmp_path: Path, fake_clock: FakeClock) -> None:
    from pcsa.common.errors import ConfigError

    with pytest.raises(ConfigError):
        AdbDevice(tmp_path / "nope.exe", DeviceConfig(), fake_clock, runner=FakeRunner())

"""Device input for live play goes through adb with the device's serial."""

from __future__ import annotations

from pathlib import Path

from pcsa.common.clock import FakeClock
from pcsa.common.config import DeviceConfig, RetryConfig
from pcsa.live.device import LiveDevice
from tests.unit.test_adb import FakeRunner


def test_tap_and_drag_send_input_through_the_serial(tmp_path: Path, fake_clock: FakeClock) -> None:
    adb = tmp_path / "adb.exe"
    adb.write_bytes(b"")
    runner = FakeRunner()
    dev = LiveDevice(
        adb, DeviceConfig(retry=RetryConfig(attempts=3, base_s=0.5, max_s=2.0)), fake_clock, runner=runner
    )
    dev.tap(10, 20)
    dev.drag(100, 300, 900, steps=2, hold_s=0.1)
    assert ["-s", "SERIAL", "shell", "input", "tap", "10", "20"] in runner.calls
    script = next(c for c in runner.calls if c[:3] == ["-s", "SERIAL", "shell"] and "motionevent" in c[-1])[-1]
    assert script.startswith("input motionevent DOWN 100 300") and script.endswith("input motionevent UP 100 900")

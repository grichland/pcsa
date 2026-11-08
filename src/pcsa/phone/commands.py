"""Observation-only phone commands: observe and calibrate. Also builds the observing phone stack."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pcsa.common import log, paths
from pcsa.common.clock import Clock, Deadline, SystemClock
from pcsa.common.config import Config
from pcsa.common.errors import ConfigError
from pcsa.common.types import Observation
from pcsa.phone.adb import AdbDevice, Frame
from pcsa.phone.calibrate import calibrate
from pcsa.phone.ocr import RapidOcrReader
from pcsa.phone.profile import DeviceProfile, load_profile, save_profile
from pcsa.phone.screens import Perception


@dataclass
class Phone[D: AdbDevice]:
    """Screenshots in, observations out: the device, OCR, the phone's profile and the screen rules."""

    device: D
    reader: RapidOcrReader
    profile: DeviceProfile
    perception: Perception
    clock: Clock
    frames: int = 0
    save_dir: Path | None = None
    display: Callable[[np.ndarray], None] | None = None  # takes a reference to each observed frame (pcsa watch)

    def observe(self, tag: str = "poll") -> tuple[Observation, Frame]:
        frame = self.device.capture()
        self.frames += 1
        frame_id = f"{self.frames:05d}-{tag}"
        if self.display is not None:
            self.display(frame.image)
        obs = self.perception.observe(frame.image, frame_id)
        if self.save_dir is not None:
            path = self.save_dir / f"{frame_id}-{obs.screen}.jpg"
            cv2.imwrite(str(path), frame.image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        log.event(
            "observation",
            frame=frame_id,
            screen=str(obs.screen),
            reason=obs.reason,
            capture_s=round(frame.seconds, 3),
        )
        return obs, frame


def perception_for(
    config: Config, device: AdbDevice, *, require_profile: bool = True
) -> tuple[DeviceProfile, RapidOcrReader, Perception]:
    """The phone's profile (or the full frame, when allowed), a warmed OCR reader and the screen rules."""
    model = device.shell("getprop", "ro.product.model")
    profile = load_profile(model)
    if profile is None:
        if require_profile:
            raise ConfigError(f"no device profile for {model}; run `pcsa calibrate` with the game on the lobby")
        frame = device.capture()
        profile = DeviceProfile.full_frame(model, *frame.size, note="uncalibrated: full frame assumed")
    reader = RapidOcrReader(config.agent.ocr)
    reader.warm()
    return profile, reader, Perception(reader, config.screen, profile.viewport)


def build_phone(config: Config, clock: Clock | None = None, *, require_profile: bool = True) -> Phone[AdbDevice]:
    clock = clock or SystemClock()
    device = AdbDevice(paths.tool("adb") / "adb.exe", config.agent.device, clock)
    profile, reader, perception = perception_for(config, device, require_profile=require_profile)
    return Phone(device, reader, profile, perception, clock)


def session_dir(kind: str) -> Path:
    directory = paths.output_dir() / kind / f"{datetime.now():%Y%m%d-%H%M%S}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def observe_command(config: Config, seconds: float, interval_s: float) -> int:
    """Capture and classify in a loop; a smoke test of the device, OCR and the rule table."""
    directory = session_dir("observe")
    log.configure(directory)
    phone = build_phone(config, require_profile=False)
    phone.save_dir = directory
    deadline = Deadline.after(phone.clock, seconds)
    counts: dict[str, int] = {}
    while not deadline.expired():
        started = phone.clock.now()
        obs, _ = phone.observe()
        counts[str(obs.screen)] = counts.get(str(obs.screen), 0) + 1
        phone.clock.sleep(max(0.0, interval_s - (phone.clock.now() - started)))
    log.event("observe_done", frames=phone.frames, screens=counts, directory=str(directory))
    return 0


def calibrate_command(config: Config) -> int:
    log.configure(None)
    clock = SystemClock()
    device = AdbDevice(paths.tool("adb") / "adb.exe", config.agent.device, clock)
    model = device.shell("getprop", "ro.product.model")
    reader = RapidOcrReader(config.agent.ocr)
    frame = device.capture()
    profile = calibrate(reader, frame.image, model)
    path = save_profile(profile)
    log.event("profile_written", path=str(path), viewport=profile.viewport.model_dump(), note=profile.note)
    return 0

"""The phone stack for live play: the observing `Phone` plus the tap executor and the menu navigator.
Also `pcsa nav`."""

from __future__ import annotations

import random
from dataclasses import dataclass

from pcsa.common import log, paths
from pcsa.common.clock import Clock, Deadline, SystemClock
from pcsa.common.config import Config
from pcsa.common.errors import Recoverable
from pcsa.common.types import Screen
from pcsa.live.device import LiveDevice
from pcsa.live.navigate import Navigator
from pcsa.live.taps import Executor, TapPolicy
from pcsa.phone.commands import Phone, perception_for, session_dir


@dataclass(kw_only=True)
class LivePhone(Phone[LiveDevice]):
    executor: Executor
    navigator: Navigator


def build_live_phone(config: Config, clock: Clock | None = None, *, seed: int | None = None) -> LivePhone:
    clock = clock or SystemClock()
    device = LiveDevice(paths.tool("adb") / "adb.exe", config.agent.device, clock)
    profile, reader, perception = perception_for(config, device)
    executor = Executor(
        config.screen, profile.viewport, config.agent.taps, TapPolicy(config.screen), random.Random(seed)
    )
    phone = LivePhone(device, reader, profile, perception, clock, executor=executor, navigator=None)  # type: ignore[arg-type]
    phone.navigator = Navigator(
        phone.observe, executor, device, clock, config.agent.navigator, config.agent.timing.poll_interval_s
    )
    return phone


def nav_command(config: Config, target: str, seconds: float) -> int:
    directory = session_dir("nav")
    log.configure(directory)
    phone = build_live_phone(config)
    phone.save_dir = directory
    try:
        obs = phone.navigator.reach(Screen(target), Deadline.after(phone.clock, seconds))
    except Recoverable as error:
        log.warning("nav_failed", reason=str(error))
        return 1
    log.event("nav_done", screen=str(obs.screen), frames=phone.frames)
    return 0 if obs.screen == Screen(target) else 1

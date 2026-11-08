"""Run the whole loop headless: scripted phone, scripted perception, scripted battle log, fake clock."""

from __future__ import annotations

import random
from pathlib import Path

from pcsa.common import log
from pcsa.common.clock import FakeClock
from pcsa.common.config import Config
from pcsa.live.fake import ScriptedDevice, ScriptedPerception, Step
from pcsa.live.log_reader import LogVisit
from pcsa.live.navigate import Navigator
from pcsa.live.phone import LivePhone
from pcsa.live.session import Session
from pcsa.live.supervisor import supervise
from pcsa.live.taps import Executor, TapPolicy
from pcsa.phone.profile import DeviceProfile
from pcsa.players.base import Player
from pcsa.players.heuristic import HeuristicPlayer
from pcsa.run.battle_state import BattleRecord, RunState
from pcsa.run.recorder import Recorder

TEAM = {
    "name": "Team 1",
    "members": [
        {
            "roster_index": i,
            "name": n,
            "species": n,
            "ability": None,
            "item": None,
            "item_known": True,
            "moves": (None,) * 4,
            "stats": None,
            "points": None,
            "nature": None,
        }
        for i, n in enumerate(("Garchomp", "Primarina", "Corviknight", "Gholdengo", "Mimikyu", "Archaludon"))
    ],
}
ROSTER = tuple(m["name"] for m in TEAM["members"])


class ScriptedLogReader:
    """Serves the step's log rows instead of scrolling a panel."""

    def __init__(self, device: ScriptedDevice, clock: FakeClock) -> None:
        self.device = device
        self.clock = clock

    def collect_log(self, b: BattleRecord, visit: int, deadline) -> LogVisit:
        step = self.device.step
        self.clock.sleep(2.0)
        if step.log_turn is None:
            return LogVisit(stop_reason="panel_not_open")
        rows = list(step.log_rows)
        appended = rows if step.log_turn not in b.log.turns else []
        b.log.turns.setdefault(step.log_turn, rows)
        b.log.row_sprites.setdefault(step.log_turn, [None] * len(rows))
        b.last_turn = max(step.log_turn, b.last_turn or 0)
        b.start_partial = visit == 1 and step.log_turn == 1
        return LogVisit(
            turn=step.log_turn, rows=len(rows), appended=len(appended), anchor="new", complete=True, seconds=2.0
        )


def build_phone(config: Config, device: ScriptedDevice, clock: FakeClock, seed: int = 0) -> LivePhone:
    profile = DeviceProfile.full_frame("SCRIPTED", 2340, 1080)
    executor = Executor(
        config.screen, profile.viewport, config.agent.taps, TapPolicy(config.screen), random.Random(seed)
    )
    phone = LivePhone(device, None, profile, ScriptedPerception(device), clock, executor=executor, navigator=None)  # type: ignore[arg-type]
    phone.navigator = Navigator(
        phone.observe, executor, device, clock, config.agent.navigator, config.agent.timing.poll_interval_s
    )
    return phone


def run_script(
    config: Config,
    steps: list[Step],
    tmp_path: Path,
    *,
    player: Player | None = None,
    target: int = 1,
    with_team: bool = True,
    seed: int = 7,
) -> tuple[RunState, ScriptedDevice, Recorder]:
    clock = FakeClock(1000.0)
    device = ScriptedDevice(steps, config.screen, clock)
    phone = build_phone(config, device, clock, seed)
    recorder = Recorder(tmp_path / "run")
    log.configure(recorder.dir, console=False)
    state = RunState(recorder.dir, target, target + 3, False, seed, team_record=TEAM if with_team else None)
    player = player or HeuristicPlayer()

    def build(s: RunState) -> Session:
        session = Session(config, phone, player, recorder, s, clock)
        session.turns.logs = ScriptedLogReader(device, clock)  # type: ignore[assignment]
        return session

    ending = supervise(state, build, clock, config.agent.run.max_restarts_per_hour)
    recorder.close()
    log.configure(None, console=False)
    state.ending = ending  # type: ignore[attr-defined]
    return state, device, recorder

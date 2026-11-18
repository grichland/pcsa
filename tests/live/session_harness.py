"""The session shell headless: scripted phone, fake clock, the real Runner, and a stub turn pipeline.

The stub has the TurnPipeline seam (.team, .peek, .action, .forced) and taps Fight or a party row so the script
advances; it can raise Recoverable on chosen calls to force a session restart.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from pcsa.common import log
from pcsa.common.clock import FakeClock
from pcsa.common.config import Config
from pcsa.common.errors import Recoverable
from pcsa.common.types import Screen
from pcsa.decide.builtin import FirstPlayer
from pcsa.decide.runner import Runner
from pcsa.live import handlers
from pcsa.live.fake import ScriptedDevice, ScriptedPerception, Step
from pcsa.live.navigate import Navigator
from pcsa.live.phone import LivePhone
from pcsa.live.play import session_builder
from pcsa.live.supervisor import supervise
from pcsa.live.taps import Executor, TapPolicy
from pcsa.phone.profile import DeviceProfile
from pcsa.run.battle_state import RunState
from pcsa.run.recorder import Recorder

ROSTER = ("Garchomp", "Primarina", "Corviknight", "Gholdengo", "Mimikyu", "Archaludon")
TEAM = {
    "name": "Team 1",
    "members": [
        {
            "roster_index": i,
            "name": n,
            "species": n,
            "ability": None,
            "item": None,
            "moves": ("Protect", None, None, None),
            "stats": None,
            "points": None,
            "nature": None,
        }
        for i, n in enumerate(ROSTER)
    ],
}
OPPONENT = ("Snorlax", "Kingambit", "Dragonite", "Gengar", "Rotom-Wash", "Tyranitar")
TEAM_FIELDS = {"roster": ROSTER, "markers": {}, "timer": "0:45", "selected_count": 0}


@dataclass(frozen=True)
class Shown:
    name: str


class Recording(FirstPlayer):
    """The built-in Player, remembering what the runner passed it."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def battle_start(self, battle_tag, team):
        self.calls.append(("start", battle_tag))

    def teampreview(self, request):
        self.calls.append(("preview", request.battle_tag))
        return super().teampreview(request)

    def battle_end(self, battle_tag, outcome, events=(), history=()):
        self.calls.append(("end", battle_tag, outcome))


@dataclass
class StubTurns:
    phone: LivePhone
    clock: FakeClock
    fail_on: set[tuple[str, int]] = field(default_factory=set)  # (kind, call number) that raise Recoverable
    team: object = None
    calls: list[tuple] = field(default_factory=list)  # (kind, battle_id, repeated, call number)

    def peek(self, b):
        return None

    def _call(self, kind: str, b, repeated: bool) -> None:
        number = sum(1 for c in self.calls if c[0] == kind) + 1
        self.calls.append((kind, b.battle_id, repeated, number))
        if (kind, number) in self.fail_on:
            raise Recoverable(f"stub {kind} {number}")

    def action(self, b, obs, t0, repeated=False):
        self._call("action", b, repeated)
        b.action_visits += 1
        b.turns.append({"turn": len(b.turns) + 1})
        executor = self.phone.executor
        executor.execute([executor.point("fight")], Screen.ACTION, self.phone.device, self.clock, 0.0)

    def forced(self, b, obs, previous_forced):
        self._call("forced", b, previous_forced)
        executor = self.phone.executor
        executor.execute([executor.point("party_rows", 1)], obs.screen, self.phone.device, self.clock, 0.0)


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


@dataclass
class Played:
    state: RunState
    ending: str
    device: ScriptedDevice
    recorder: Recorder
    turns: StubTurns
    player: Recording

    def battle(self, index: int) -> dict:
        return json.loads((self.recorder.dir / f"battle-{index}" / "battle.json").read_text(encoding="utf-8"))

    def events(self) -> list[dict]:
        lines = (self.recorder.dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]


def run_script(
    config: Config,
    steps: list[Step],
    tmp_path: Path,
    monkeypatch,
    *,
    target: int = 1,
    fail_on: set[tuple[str, int]] | None = None,
    pending: int | None = None,
) -> Played:
    monkeypatch.setattr(handlers, "read_preview", lambda s, frame: tuple(Shown(n) for n in OPPONENT))
    clock = FakeClock(1000.0)
    device = ScriptedDevice(steps, config.screen, clock)
    phone = build_phone(config, device, clock)
    recorder = Recorder(tmp_path / "run")
    log.configure(recorder.dir, console=False)
    state = RunState(recorder.dir, target, target + 3, False, 7, team_record=TEAM)
    if pending is not None:
        state.pending_battle, state.played = pending, pending - 1
    player = Recording()
    runner = Runner(player, clock, config.agent.degrade)
    turns = StubTurns(phone, clock, fail_on or set())
    build = session_builder(config, phone, runner, recorder, clock, turns)
    try:
        ending = supervise(state, build, clock, config.agent.run.max_restarts_per_hour)
    finally:
        runner.worker.idle(5.0)
        runner.close()
        recorder.close()
        log.configure(None, console=False)
    return Played(state, ending, device, recorder, turns, player)

"""T7 rig: whole battles on the scripted phone with the real Session (through play.session_builder and the
supervisor), the real TurnPipeline, the real Runner and the built-in Player.

The log is scripted per action visit (turn_rig.FakeLogs) unless a test gives the pipeline the real LogReader.
Faults: ("decide", n) or ("teampreview", n) raise Recoverable on the n-th runner call, before the Player is asked.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pcsa.common import log
from pcsa.common.clock import FakeClock
from pcsa.common.config import Config
from pcsa.common.errors import Recoverable
from pcsa.common.types import PartyMember, Screen
from pcsa.decide.api import Answer, Order
from pcsa.decide.builtin import FIRST_PICKS, FirstPlayer
from pcsa.decide.runner import Runner
from pcsa.events.live_check import event_problems, message_problems
from pcsa.live import handlers
from pcsa.live.fake import ScriptedDevice, Step
from pcsa.live.play import session_builder
from pcsa.live.supervisor import supervise
from pcsa.live.turn import TurnPipeline
from pcsa.phone.sprites import SlotMatch
from pcsa.run.battle_state import RunState
from pcsa.run.recorder import Recorder
from tests.live.session_harness import build_phone
from tests.live.turn_rig import TEAM, FakeLogs

OPPONENT = ("Glimmora", "Lucario", "Garchomp", "Archaludon", "Kingambit", "Dragonite")
PREVIEW = tuple(SlotMatch(i, (0, 0, 1, 1), ((n, 0.9),), ((n, 0.9),)) for i, n in enumerate(OPPONENT))
TEAM_FIELDS = {"roster": TEAM.names, "markers": {}, "timer": "0:45", "selected_count": 0}
MOVE_TAPS = ("fight", "mega", "move_rows")


# Script pieces ------------------------------------------------------------------------------------------
def lobby_to_team_select() -> list[Step]:
    return [
        Step(Screen.LOBBY, {"team_label": "Team 1"}, accept=("begin_matchmaking",)),
        Step(Screen.MATCHMAKING, after_s=3),
        *team_select(),
    ]


def team_select() -> list[Step]:
    return [
        Step(Screen.TEAM_SELECT, TEAM_FIELDS, accept=("team_rows", "team_done"), advance_on=("team_done",)),
        Step(Screen.TRANSITION, after_s=2),
    ]


def action(active: str | None = None, **fields) -> list[Step]:
    """An action screen taken without the move list: Fight, the row twice (the second lands on the wait)."""
    shown = {"move_time": 40, **({"active_name": active} if active else {}), **fields}
    return [
        Step(Screen.ACTION, shown, accept=MOVE_TAPS, advance_on=("move_rows",)),
        Step(Screen.WAIT, accept=("move_rows",), advance_on=(), after_s=12),  # seen after the post-action wait
    ]


def peeked_action(active: str, moves: tuple, mega: bool) -> list[Step]:
    """An action screen where the pipeline opens the move list (Fight, read, Back), then taps the move."""
    shown = {"move_time": 40, "active_name": active}
    return [
        Step(Screen.ACTION, shown, accept=("fight",)),
        Step(Screen.MOVES, {"moves": moves, "mega_available": mega}, accept=("move_back",)),
        *action(active),
    ]


def party(*members: tuple[str, bool]) -> tuple[PartyMember, ...]:
    return tuple(
        PartyMember(i, n, (0, 100) if dead else (80, 100), fainted=dead) for i, (n, dead) in enumerate(members)
    )


def forced(*members: tuple[str, bool]) -> list[Step]:
    """A replacement screen (faint or pivot): a row tap or two, then Switch In."""
    return [
        Step(
            Screen.FORCED,
            {"party": party(*members), "no_back": True},
            accept=("party_rows", "switch_in"),
            advance_on=("switch_in",),
        ),
        Step(Screen.WAIT, after_s=12),
    ]


def end_of_battle(outcome: str = "win") -> list[Step]:
    return [
        Step(Screen.RESULT, {"outcome": outcome}, after_s=2),
        Step(Screen.REWARDS, {"own_rank": {"tier": "Master Ball", "rank": 4, "rating": 1210.0}}, after_s=2),
        Step(Screen.OPTIONS, accept=("continue_battling",)),
        Step(Screen.LOBBY, {"team_label": "Team 1"}),
    ]


# Player and runner ------------------------------------------------------------------------------------
class Script(FirstPlayer):
    """The built-in Player; `orders` (None = its own answer) are given in turn, one per choose_move call."""

    name = "first"

    def __init__(self, picks=FIRST_PICKS, orders: list[Order | None] | None = None) -> None:
        self.picks, self.orders = tuple(picks), list(orders or [])
        self.calls: list[tuple] = []

    def battle_start(self, battle_tag, team):
        self.calls.append(("start", battle_tag))

    def teampreview(self, request):
        self.calls.append(("preview", request.battle_tag))
        return self.picks, {"reason": "script"}

    def choose_move(self, request):
        order = self.orders.pop(0) if self.orders else None
        return Answer(order, {"reason": "script"}) if order is not None else super().choose_move(request)

    def battle_end(self, battle_tag, outcome, events=(), history=()):
        self.calls.append(("end", battle_tag, outcome))


@dataclass
class Sent:
    kind: str  # decide | teampreview
    request: object
    raised: bool


def inject(runner: Runner, faults: set[tuple[str, int]], sent: list[Sent]) -> None:
    """Every request the runner gets is kept; the faulted calls raise Recoverable before the Player is asked."""
    counts: Counter = Counter()
    for name in ("decide", "teampreview"):
        real = getattr(runner, name)

        def wrapped(request, _real=real, _name=name):
            counts[_name] += 1
            raised = (_name, counts[_name]) in faults
            sent.append(Sent(_name, request, raised))
            if raised:
                raise Recoverable(f"injected at {_name} {counts[_name]}")
            return _real(request)

        setattr(runner, name, wrapped)


# Running ----------------------------------------------------------------------------------------------
@dataclass
class Played:
    state: RunState
    ending: str
    device: ScriptedDevice
    recorder: Recorder
    pipeline: TurnPipeline
    player: Script
    sent: list[Sent]
    ends: list[tuple[str, str, tuple]] = field(default_factory=list)  # (battle tag, result, events at the end)

    def battle(self, index: int) -> dict:
        return self.json(f"battle-{index}/battle.json")

    def json(self, name: str) -> dict:
        return json.loads((self.recorder.dir / name).read_text(encoding="utf-8"))

    def records(self, index: int = 1) -> list[dict]:
        folder = self.recorder.dir / f"battle-{index}" / "decisions"
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]

    def requests(self, kind: str = "decide") -> list:
        """What the runner was asked, faulted calls included."""
        return [s.request for s in self.sent if s.kind == kind]

    def log_events(self) -> list[dict]:
        lines = (self.recorder.dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def taps(self) -> list[str | None]:
        return [name for _, name in self.device.taps]


def run_battles(
    config: Config,
    steps: list[Step],
    tmp_path: Path,
    monkeypatch,
    *,
    target: int = 1,
    player: Script | None = None,
    visits: list[dict] | None = None,
    faults: set[tuple[str, int]] | None = None,
    reader=None,
    logs=None,
    clock: FakeClock | None = None,
) -> Played:
    """`reader`: OCR for the real LogReader (the default when no `visits`/`logs`); `logs`: a reader of the test's."""
    monkeypatch.setattr(handlers, "read_preview", lambda s, frame: PREVIEW)
    clock = clock or FakeClock(1000.0)
    device = ScriptedDevice(steps, config.screen, clock)
    phone = build_phone(config, device, clock)
    phone.reader = reader
    recorder = Recorder(tmp_path / "run")
    log.configure(recorder.dir, console=False)
    state = RunState(recorder.dir, target, target + 3, False, 7, team_record=TEAM.record())
    player = player or Script()
    runner = Runner(player, clock, config.agent.degrade)
    sent: list[Sent] = []
    inject(runner, faults or set(), sent)
    pipeline = TurnPipeline(phone, runner, recorder, config.agent, clock, seed=state.seed)
    if logs is not None:
        pipeline.logs = logs(pipeline.logs) if callable(logs) else logs
    elif reader is None:
        pipeline.logs = FakeLogs(visits or [])  # type: ignore[assignment]
    played = Played(state, "", device, recorder, pipeline, player, sent)
    real_end = pipeline.end

    def end(b, result):
        out = real_end(b, result)
        played.ends.append((b.battle_id, result, pipeline.events_for(b).all()))
        return out

    pipeline.end = end  # type: ignore[method-assign]
    build = session_builder(config, phone, runner, recorder, clock, pipeline)
    try:
        played.ending = supervise(state, build, clock, config.agent.run.max_restarts_per_hour)
    finally:
        runner.worker.idle(5.0)
        runner.close()
        recorder.close()
        log.configure(None, console=False)
    return played


# Checks -----------------------------------------------------------------------------------------------
def live_problems(events: tuple | list) -> list[str]:
    """battle_start first and valid, then every event valid (live_check, gap included)."""
    if not events or events[0].get("type") != "battle_start":
        return ["first:not battle_start"]
    out = message_problems(events[0])
    return out + [f"{e.get('k')}:{p}" for e in events[1:] for p in event_problems(e)]


def row_events(events) -> list[dict]:
    return [e for e in events[1:] if e.get("k") not in ("turn", "gap", "refused", "end")]

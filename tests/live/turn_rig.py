"""A TurnPipeline on the scripted device: a fake log reader that merges given rows, the built-in Player."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

from pcsa.battle.teams import OwnMember, Team
from pcsa.common.clock import FakeClock
from pcsa.common.config import Config, Viewport
from pcsa.decide.builtin import FirstPlayer
from pcsa.decide.runner import Runner
from pcsa.live.fake import ScriptedDevice, ScriptedPerception, Step
from pcsa.live.log_reader import LogVisit
from pcsa.live.navigate import Navigator
from pcsa.live.taps import Executor, TapPolicy
from pcsa.live.turn import TurnPipeline
from pcsa.phone.log_merge import BattleLog
from pcsa.phone.profile import DeviceProfile
from pcsa.run.battle_state import BattleRecord
from pcsa.run.recorder import Recorder

STATS = {"hp": 180, "atk": 150, "def": 100, "spa": 90, "spd": 100, "spe": 120}


def member(i: int, name: str, item: str | None, ability: str, moves: tuple[str, ...]) -> OwnMember:
    return OwnMember(i, name, name, ability, item, moves, dict(STATS), None, "Adamant")


TEAM = Team(
    "gold",
    (
        member(0, "Gholdengo", "Choice Scarf", "Good as Gold", ("Make It Rain", "Shadow Ball", "Trick", "Nasty Plot")),
        member(1, "Garchomp", "Garchompite Z", "Rough Skin", ("Earthquake", "Dragon Claw", "Swords Dance", "Protect")),
        member(2, "Golisopod", "Golisopite", "Emergency Exit", ("First Impression", "Liquidation", "Leech Life",
                                                                "Aqua Jet")),
        member(3, "Primarina", "Leftovers", "Torrent", ("Moonblast", "Surf", "Calm Mind", "Protect")),
        member(4, "Mamoswine", "Focus Sash", "Oblivious", ("Ice Shard", "Earthquake", "Icicle Spear", "Endeavor")),
        member(5, "Cinderace", "Life Orb", "Libero", ("Pyro Ball", "U-turn", "Sucker Punch", "Court Change")),
    ),
)  # fmt: skip
PREVIEW = tuple({"top": [[n, 0.9]], "species": [[n, 0.9]]} for n in ("Glimmora", "Lucario", "Garchomp"))


class FakeLogs:
    """Stands in for LogReader: each visit merges the next scripted {turn: [text or (text, side)]}."""

    def __init__(self, visits: list[dict]) -> None:
        self.visits = list(visits)

    def collect_log(self, b: BattleRecord, visit: int, deadline) -> LogVisit:
        pages = self.visits.pop(0) if self.visits else {}
        info = LogVisit(complete=True, stop_reason="top_confirmed")
        for turn, rows in sorted(pages.items()):
            texts = [r if isinstance(r, str) else r[0] for r in rows]
            sides = [None if isinstance(r, str) else r[1] for r in rows]
            stored = b.log.turns.setdefault(turn, [])
            appended = texts[len(stored) :]
            stored.extend(appended)
            b.log.row_notes[turn] = [(None, s, ()) for s in sides]
            b.log.visits.append({"visit": visit, "turn": turn, "appended": len(appended), "warnings": []})
            info.turn, info.rows, info.appended = turn, len(texts), len(appended)
            b.last_turn = max(turn, b.last_turn or 0)
        b.start_partial = visit == 1 and info.turn == 1
        return info


class Saver:
    def __init__(self) -> None:
        self.calls = 0

    def after_visit(self, b, team) -> bool:
        self.calls += 1
        return True


class Rig:
    def __init__(self, config: Config, tmp: Path, steps: list[Step], visits: list[dict], player=None, **agent):
        self.clock = FakeClock(start=1000.0)
        self.device = ScriptedDevice(steps, config.screen, self.clock)
        perception = ScriptedPerception(self.device)
        executor = Executor(
            config.screen,
            Viewport.of_frame(2340, 1080),
            config.agent.taps,
            TapPolicy(config.screen),
            random.Random(0),
        )

        def observe(tag: str = "poll"):
            frame = self.device.capture()
            return perception.observe(frame.image, tag), frame

        navigator = Navigator(observe, executor, self.device, self.clock, config.agent.navigator, 0.5)
        profile = DeviceProfile.full_frame("fake", 2340, 1080)
        self.phone = SimpleNamespace(
            executor=executor, device=self.device, navigator=navigator, observe=observe, reader=None, profile=profile
        )
        cfg = config.agent
        if agent:
            cfg = cfg.model_copy(update={"timing": cfg.timing.model_copy(update=agent)})
        self.runner = Runner(player or FirstPlayer(), self.clock, cfg.degrade)
        self.rec = Recorder(tmp / "run")
        self.saver = Saver()
        self.pipeline = TurnPipeline(self.phone, self.runner, self.rec, cfg, self.clock, saver=self.saver)
        self.pipeline.team = TEAM
        self.pipeline.logs = FakeLogs(visits)  # type: ignore[assignment]
        self.battle = BattleRecord(
            1, "b-1", TEAM.names, BattleLog("b-1"), requested=[4, 1, 2], team_submitted=True, opponent_preview=PREVIEW
        )

    def records(self) -> list[dict]:
        folder = self.rec.dir / self.battle.dir / "decisions"
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]

    def taps(self) -> list[str | None]:
        return [name for _, name in self.device.taps]

    def close(self) -> None:
        self.runner.close()
        self.rec.close()

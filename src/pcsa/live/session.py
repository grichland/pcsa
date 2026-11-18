"""The session: observe, check the watchdog, hand the screen to its handler, repeat.

A screen that stays too long, an unknown screen or a silent device raises Recoverable (the supervisor restarts the
session on the same run state, carrying the battle in progress). Only SafetyStop ends the run. A battle this run did
not start is not adopted (SV-7): its screens are waited out without taps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from pcsa.battle.teams import Team, team_from_record
from pcsa.common import log
from pcsa.common.clock import Clock, Deadline
from pcsa.common.config import AgentConfig, Config
from pcsa.common.errors import Recoverable, SafetyStop, Transient
from pcsa.common.types import IN_BATTLE, Flag, Observation, Screen
from pcsa.decide.runner import Runner
from pcsa.live import handlers, reconnect
from pcsa.live.phone import LivePhone as Phone
from pcsa.phone.texts import Texts
from pcsa.run.battle_state import BattleRecord, RunState
from pcsa.run.recorder import Recorder

TRANSIENT_WAIT_S = 3.0
TURN_SCREENS = (Screen.ACTION, Screen.FORCED)


@dataclass
class Session:
    cfg: Config
    phone: Phone
    runner: Runner
    rec: Recorder
    state: RunState
    clock: Clock
    turns: Any = None  # live/turn.TurnPipeline; shared across restarts when given
    team: Team | None = None
    battle: BattleRecord | None = None  # carried over from the previous session of this run
    latest_rank: dict | None = None
    begin_taps: int = 0
    continue_taps: int = 0
    finished: bool = False
    previous: Observation | None = None
    screen_since: float = 0.0
    last_progress: float = 0.0
    last_screen: Screen | None = None
    unknown_rounds: int = 0
    skipped_screen: Screen | None = None
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.state.seed + self.state.restarts)
        if self.turns is None:
            from pcsa.live.turn import TurnPipeline

            self.turns = TurnPipeline(
                self.phone, self.runner, self.rec, self.cfg.agent, self.clock, seed=self.state.seed
            )
        self.screen_since = self.last_progress = self.clock.now()

    @property
    def agent(self) -> AgentConfig:
        return self.cfg.agent

    # I/O --------------------------------------------------------------------------------------------
    def observe(self, tag: str = "poll") -> tuple[Observation, Any]:
        obs, frame = self.phone.observe(tag)
        rank = obs.fields.get("own_rank")
        if rank and obs.screen in (Screen.LOBBY, Screen.REWARDS, Screen.OPTIONS):
            snapshot = {k: rank.get(k) for k in ("tier", "rank", "placement", "rating")}
            if snapshot != self.latest_rank:
                self.latest_rank = snapshot
                self.state.rank_history.append(
                    {"screen": str(obs.screen), "battle": self.battle.index if self.battle else None, **snapshot}
                )
                if self.battle is not None and obs.screen != Screen.LOBBY:
                    self.battle.rank_after = snapshot
                log.event("rank", **snapshot)
        if obs.screen == Screen.UNKNOWN:
            self.rec.save_frame(f"unknown/{obs.frame_id}.png", frame.image, png=True)
            self.rec.write_json(
                f"unknown/{obs.frame_id}.json", {"reason": obs.reason, "boxes": [b.record() for b in obs.texts]}
            )
        return obs, frame

    def tap(self, name: str, obs: Observation) -> None:
        tap = self.phone.executor.point(name)
        self.phone.executor.execute([tap], obs.screen, self.phone.device, self.clock, 0.0, Texts(obs.texts))
        self.progress()

    def execute(self, plan, screen: Screen, obs: Observation) -> None:
        self.phone.executor.execute(
            plan, screen, self.phone.device, self.clock, self.cfg.agent.timing.inter_tap_s, Texts(obs.texts)
        )
        self.progress()

    def progress(self) -> None:
        self.last_progress = self.clock.now()

    # Loop --------------------------------------------------------------------------------------------
    def run(self) -> None:
        self.start()
        while not self.finished:
            try:
                obs, frame = self.observe()
            except Transient as error:
                log.warning("device_unavailable", reason=str(error))
                self.clock.sleep(TRANSIENT_WAIT_S)
                self.watchdog(Screen.UNKNOWN)
                continue
            self.watchdog(obs.screen)
            self.flags(obs)
            try:
                self.handle(obs, frame)
            except Transient as error:
                log.warning("transient_during_handler", screen=str(obs.screen), reason=str(error))
                self.clock.sleep(TRANSIENT_WAIT_S)
            finally:
                self.previous = obs
        log.event("session_done", completed=self.state.completed, played=self.state.played)

    def start(self) -> None:
        obs, _ = self.observe("start")
        carried = self.battle is not None and not self.battle.finished
        log.event(
            "session_start",
            screen=str(obs.screen),
            restarts=self.state.restarts,
            pending_battle=self.state.pending_battle,
            carried_battle=self.battle.index if carried else None,  # type: ignore[union-attr]
        )
        if self.state.team_record and self.team is None:
            self.team = team_from_record(self.state.team_record)
        if carried:
            self.battle.team_repairs = 0  # type: ignore[union-attr]  # a fresh session may repair the lineup again
        else:
            handlers.drop_pending(self)
        if obs.screen not in IN_BATTLE and obs.screen not in (Screen.LOBBY, Screen.OPTIONS, Screen.MATCHMAKING):
            self.phone.navigator.reach(Screen.LOBBY, Deadline.after(self.clock, 240))

    def watchdog(self, screen: Screen) -> None:
        now = self.clock.now()
        if screen != self.last_screen:
            self.last_screen, self.screen_since = screen, now
        limit = self.cfg.agent.timing.screen_timeouts.get(str(screen), self.cfg.agent.timing.default_screen_timeout_s)
        if now - self.screen_since > limit:
            raise Recoverable(f"{screen} for more than {limit:.0f} s")
        if now - self.last_progress > self.cfg.agent.timing.no_progress_s:
            raise Recoverable(f"no progress for {self.cfg.agent.timing.no_progress_s:.0f} s (last screen {screen})")

    def flags(self, obs: Observation) -> None:
        if Flag.AUTO_NOTICE in obs.flags and self.battle is not None:
            log.warning("turn_timed_out", battle=self.battle.index, notice=obs.fields.get("notice"))
            b = self.battle
            if not b.finished and b.team_submitted and b.action_visits == 0 and not b.team_auto:
                b.team_auto = True  # the game chose the lineup (team_check checks it)
                log.warning("team_auto_chosen", battle=b.index, requested=list(b.names))

    def handle(self, obs: Observation, frame) -> None:
        screen = obs.screen
        turn_screen = screen in TURN_SCREENS or self.persistent_party_without_back(obs)
        if reconnect.applies(self, obs):
            reconnect.ride_out(self, obs)
        elif screen == Screen.TEAM_SELECT:
            handlers.on_team_select(self, obs, frame, frame.captured_at)
        elif (turn_screen or screen in STRAY) and not self.in_battle():
            handlers.on_not_adopted(self, obs)
        elif screen == Screen.ACTION:
            repeated = self.previous is not None and self.previous.screen == Screen.ACTION
            self.turns.action(self.battle, obs, frame.captured_at, repeated)
            self.progress()
        elif turn_screen:
            repeated = self.previous is not None and self.previous.screen in (Screen.FORCED, Screen.PARTY)
            self.turns.forced(self.battle, obs, repeated)
            self.progress()
        elif screen == Screen.UNKNOWN:
            self.unknown_rounds += 1
            self.phone.navigator.recover_unknown(obs, self.unknown_rounds)
        else:
            SIMPLE_HANDLERS.get(screen, handlers.on_waiting)(self, obs)
        if screen != Screen.UNKNOWN:
            self.unknown_rounds = 0

    def persistent_party_without_back(self, obs: Observation) -> bool:
        """A party menu without Back on two consecutive polls is a forced replacement the rule missed."""
        return (
            obs.screen == Screen.PARTY
            and bool(obs.fields.get("no_back"))
            and self.previous is not None
            and self.previous.screen == Screen.PARTY
        )

    def in_battle(self) -> bool:
        """A battle this run started is on (and its team submitted): the turn pipeline takes the screen."""
        b = self.battle
        if b is None or b.finished or not b.team_submitted:
            return False
        self.turns.team = self.team
        return True


STRAY = (Screen.MOVES, Screen.LOG, Screen.PARTY, Screen.SWITCH_POPUP)
AFTER_BATTLE = (Screen.REWARDS, Screen.RANK_UP, Screen.RATING_DIALOG, Screen.DIALOG)
MENUS = (Screen.HOME, Screen.BATTLE_MODES, Screen.RANKED_FORMATS, Screen.TITLE)
TEAM_PAGES = (Screen.TEAM_LIST, Screen.TEAM_MENU, Screen.TEAM_DETAILS, Screen.TEAM_STATS)
SIMPLE_HANDLERS = {
    Screen.LOBBY: handlers.on_lobby,
    Screen.RESULT: handlers.on_result,
    Screen.FORFEIT_MSG: handlers.on_forfeit,
    Screen.OPTIONS: handlers.on_options,
    Screen.COMM_ERROR: handlers.on_interruption,
    **dict.fromkeys(STRAY, handlers.on_stray),
    **dict.fromkeys(AFTER_BATTLE + MENUS + TEAM_PAGES, handlers.on_rank_change),
}

__all__ = ["Session", "SafetyStop"]

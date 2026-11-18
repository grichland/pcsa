"""`pcsa run`: build the phone, the Player and the run state, then supervise sessions until done.
`pcsa check`: connect to the device and classify the current screen, without a tap."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pcsa.common import log
from pcsa.common.clock import Clock, SystemClock
from pcsa.common.config import Config
from pcsa.common.errors import ConfigError
from pcsa.decide.load import load_player
from pcsa.decide.runner import Runner
from pcsa.live.session import Session
from pcsa.live.supervisor import supervise
from pcsa.run.battle_state import RunState
from pcsa.run.recorder import Recorder, manifest


def open_run(
    config: Config, resume: Path | None, name: str, battles: int | None, max_total: int | None, seed: int | None
) -> tuple[Recorder, RunState]:
    if resume is not None:
        state = RunState.load(resume)
        if state is None:
            raise ConfigError(f"{resume} has no state.json to resume")
        return Recorder(resume), state
    recorder = Recorder.new(name)
    state = RunState(
        recorder.dir,
        battles or config.agent.run.battles,
        max_total or config.agent.run.max_battles_total,
        config.agent.run.count_forfeits,
        seed if seed is not None else int(datetime.now().timestamp() * 1000) % 2**31,
    )
    state.save()
    return recorder, state


def session_builder(config: Config, phone, runner: Runner, recorder: Recorder, clock: Clock, turns=None):
    """Sessions that share the turn pipeline (`turns`, else the first session's) and carry the battle in progress
    and our team to the next one."""
    previous: list[Session] = []

    def build(state: RunState) -> Session:
        last = previous[-1] if previous else None
        session = Session(
            config,
            phone,
            runner,
            recorder,
            state,
            clock,
            turns=last.turns if last else turns,
            team=last.team if last else None,
            battle=last.battle if last and last.battle and not last.battle.finished else None,
        )
        previous[:] = [session]
        return session

    return build


def start_player(player) -> None:
    """A Player with `start` (CG) is made and checked before any battle; a failure ends the run here."""
    start = getattr(player, "start", None)
    if start is None:
        return
    try:
        status = start()
    except Exception as error:
        getattr(player, "close", lambda: None)()
        log.event("player_start_failed", player=player.name, reason=str(error))
        raise ConfigError(f"player {player.name!r} cannot start, no battle started: {error}") from error
    log.event("player_started", player=player.name, status=status)


def play_command(
    config: Config,
    *,
    player_name: str | None,
    battles: int | None,
    max_total: int | None,
    seed: int | None,
    name: str,
    resume: Path | None,
) -> int:
    from pcsa.live.phone import build_live_phone

    clock = SystemClock()
    recorder, state = open_run(config, resume, name, battles, max_total, seed)
    log.configure(recorder.dir)
    player = load_player(player_name or config.agent.decide.player, config, clock=clock, run_dir=recorder.dir)
    start_player(player)
    runner = Runner(player, clock, config.agent.degrade)
    phone = build_live_phone(config, clock, seed=state.seed)
    if config.agent.device.stay_awake:
        phone.device.stay_awake(True)
    params = getattr(player, "params", lambda: {})()
    recorder.write_json("manifest.json", manifest(state.seed, player.name, params, config.path, phone.device.facts()))
    log.event(
        "run_start",
        run=str(recorder.dir),
        player=player.name,
        battles=state.target,
        max_total=state.max_total,
        seed=state.seed,
    )
    phone.save_dir = recorder.dir / "frames"
    build = session_builder(config, phone, runner, recorder, clock)
    summary: dict = {}
    try:
        ending = supervise(state, build, clock, config.agent.run.max_restarts_per_hour)
    finally:
        try:
            phone.device.stay_awake(False)
        except Exception as error:
            log.warning("stay_awake_restore_failed", reason=str(error))
        runner.close()
        getattr(player, "close", lambda: None)()  # a plugin Player's CG process
        summary = {
            "ending": None,
            "completed": state.completed,
            "played": state.played,
            "outcomes": state.outcomes,
            "restarts": state.restarts,
            "run_dir": str(recorder.dir),
            **getattr(player, "run_summary", dict)(),
        }
        recorder.write_json("summary.json", summary)
        recorder.close()
    summary["ending"] = ending
    recorder.write_json("summary.json", summary)
    log.event("run_end", **summary)
    return 0 if ending == "completed" else 1


def check_command(config: Config) -> int:
    """The device answers and the current screen is classified; nothing is tapped."""
    from pcsa.live.phone import build_live_phone
    from pcsa.phone.commands import session_dir

    directory = session_dir("check")
    log.configure(directory)
    phone = build_live_phone(config)
    phone.save_dir = directory
    obs, frame = phone.observe("check")
    log.event(
        "check_done",
        device=phone.device.facts(),
        screen=str(obs.screen),
        reason=obs.reason,
        size=list(frame.size),
        directory=str(directory),
    )
    return 0

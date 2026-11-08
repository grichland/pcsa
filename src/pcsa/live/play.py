"""`pcsa run`: build the phone, the player and the run state, then supervise sessions until done."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pcsa.common import log
from pcsa.common.clock import SystemClock
from pcsa.common.config import Config
from pcsa.common.errors import ConfigError
from pcsa.live.phone import build_live_phone
from pcsa.live.session import Session
from pcsa.live.supervisor import supervise
from pcsa.players.factory import build_player
from pcsa.run.battle_state import RunState
from pcsa.run.display import DisplayFrames
from pcsa.run.recorder import Recorder, manifest


def _open_run(
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


def run_command(
    config: Config,
    *,
    player_kind: str | None,
    battles: int | None,
    max_total: int | None,
    seed: int | None,
    name: str,
    resume: Path | None,
) -> int:
    clock = SystemClock()
    recorder, state = _open_run(config, resume, name, battles, max_total, seed)
    log.configure(recorder.dir)
    player = build_player(config, player_kind)
    phone = build_live_phone(config, clock, seed=state.seed)
    if config.agent.device.stay_awake:
        phone.device.stay_awake(True)
    recorder.write_json(
        "manifest.json", manifest(state.seed, player.name, player.params(), config.path, phone.device.facts())
    )
    log.event(
        "run_start",
        run=str(recorder.dir),
        player=player.name,
        battles=state.target,
        max_total=state.max_total,
        seed=state.seed,
    )
    phone.save_dir = recorder.dir / "frames"
    fps = config.agent.record.display_fps
    display = DisplayFrames(recorder.dir, fps, clock) if fps > 0 else None
    phone.display = display.offer if display is not None else None

    def build(s: RunState) -> Session:
        return Session(config, phone, player, recorder, s, clock)

    try:
        ending = supervise(state, build, clock, config.agent.run.max_restarts_per_hour)
    finally:
        try:
            phone.device.stay_awake(False)
        except Exception as error:
            log.warning("stay_awake_restore_failed", reason=str(error))
        summary = {
            "ending": None,
            "completed": state.completed,
            "played": state.played,
            "outcomes": state.outcomes,
            "restarts": state.restarts,
            "run_dir": str(recorder.dir),
        }
        recorder.write_json("summary.json", summary)
        recorder.close()
        if display is not None:
            display.close()
    summary["ending"] = ending
    recorder.write_json("summary.json", summary)
    log.event("run_end", **summary)
    return 0 if ending == "completed" else 1

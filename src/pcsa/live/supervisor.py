"""Keep playing: build a session, run it, and on anything recoverable build another on the same run state."""

from __future__ import annotations

from collections.abc import Callable

from pcsa.common import log
from pcsa.common.clock import Clock
from pcsa.common.errors import Recoverable, SafetyStop
from pcsa.live.session import Session
from pcsa.run.battle_state import RunState

BACKOFF_S = (5.0, 15.0, 30.0, 60.0)
HOUR_S = 3600.0


def supervise(state: RunState, build: Callable[[RunState], Session], clock: Clock, max_restarts_per_hour: int) -> str:
    """Returns how the run ended: completed | safety_stop | interrupted | too_many_restarts."""
    restart_times: list[float] = []
    while not state.done():
        try:
            build(state).run()
            if state.done():
                break
            raise Recoverable("session finished early")
        except SafetyStop as error:
            log.warning("safety_stop", reason=str(error))
            return "safety_stop"
        except KeyboardInterrupt:
            log.event("interrupted")
            return "interrupted"
        except Recoverable as error:
            reason = str(error)
        except Exception as error:
            log.exception("session_crash", kind=type(error).__name__)
            reason = f"crash: {type(error).__name__}: {error}"
        now = clock.now()
        restart_times = [t for t in restart_times if now - t < HOUR_S] + [now]
        if len(restart_times) > max_restarts_per_hour:
            log.warning("too_many_restarts", count=len(restart_times))
            return "too_many_restarts"
        state.restarts += 1
        state.save()
        delay = BACKOFF_S[min(len(restart_times) - 1, len(BACKOFF_S) - 1)]
        log.event("session_restart", restart=state.restarts, reason=reason, delay_s=delay)
        clock.sleep(delay)
    return "completed"

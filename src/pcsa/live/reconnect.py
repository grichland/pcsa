"""Riding out a disconnect in the middle of a battle.

While a battle is in progress and the phone shows something that is not a battle screen (an unknown
frame, "Communication was interrupted", a dialog, a black or loading frame that lasts), the session
taps once per round: an OCR-confirmed safe control (OK, Close, Continue, Cancel, Not now) when the
screen offers exactly one, otherwise the screen centre. It waits `navigator.disconnect_settle_s`,
observes again, and repeats until a known screen returns: a battle screen (the turn pipeline takes
over), a post-battle screen (the usual handlers finalize), or the lobby or a menu (the battle is
finalized here, as the communication-error path does). Every tap passes the tap policy; a tap with
forbidden text near it is skipped and the round only waits. Past `disconnect_max_taps` rounds or
`disconnect_max_s` seconds it raises Recoverable and the supervisor restarts the session.
"""

from __future__ import annotations

from pcsa.common import log
from pcsa.common.errors import Recoverable
from pcsa.common.types import IN_BATTLE, Observation, Screen
from pcsa.live import handlers
from pcsa.live.navigate import box_of
from pcsa.live.taps import BLIND_FORBIDDEN_TEXT, BLIND_TEXT_MARGIN, Tap
from pcsa.phone.fields import safe_control
from pcsa.phone.texts import Texts

CENTRE = "screen_centre"
STUCK = frozenset({Screen.UNKNOWN, Screen.COMM_ERROR, Screen.DIALOG, Screen.TRANSITION})
DISMISSABLE = frozenset({Screen.UNKNOWN, Screen.COMM_ERROR, Screen.DIALOG})
POST_BATTLE = frozenset(
    {Screen.RESULT, Screen.REWARDS, Screen.OPTIONS, Screen.FORFEIT_MSG, Screen.RANK_UP, Screen.RATING_DIALOG}
)
CONTROL = r"ok|close|continue|cancel|not now"
CONTROL_REGION = (300, 150, 2040, 1000)
BLIND_MIN_CONFIDENCE = 0.5  # read more words than usual before a blind tap: a doubtful "Forfeit" still counts


def applies(s, obs: Observation) -> bool:
    """A battle is in progress and the screen is not one the battle handlers know."""
    if s.battle is None or s.battle.finished:
        return False
    if obs.screen == Screen.TRANSITION:  # a black or loading frame is normal for a while
        return s.clock.now() - s.screen_since >= s.agent.navigator.disconnect_blank_s
    return obs.screen in DISMISSABLE


def ride_out(s, obs: Observation) -> None:
    """Tap, settle, observe, until a known screen returns; Recoverable past the configured bound."""
    cfg = s.agent.navigator
    battle = s.battle.index
    started = s.clock.now()
    log.warning("disconnect_start", battle=battle, screen=str(obs.screen), reason=obs.reason, frame=obs.frame_id)
    rounds = 0
    while obs.screen in STUCK:
        elapsed = s.clock.now() - started
        if rounds >= cfg.disconnect_max_taps or elapsed >= cfg.disconnect_max_s:
            log.warning(
                "disconnect_exhausted",
                battle=battle,
                rounds=rounds,
                seconds=round(elapsed, 1),
                screen=str(obs.screen),
            )
            raise Recoverable(f"disconnect not recovered after {rounds} rounds in {elapsed:.0f} s ({obs.screen})")
        rounds += 1
        tap_once(s, obs, rounds)
        s.clock.sleep(cfg.disconnect_settle_s)
        obs, _ = s.observe("disconnect")  # unknown frames are saved by the session's observe
    resolved(s, obs, rounds, s.clock.now() - started)


def tap_once(s, obs: Observation, rounds: int) -> None:
    """The safe control if one is offered and clear of forbidden text, else the centre if clear, else nothing."""
    executor = s.phone.executor
    texts = Texts(obs.texts, BLIND_MIN_CONFIDENCE)
    reasons = []
    for kind, tap in candidates(s, obs):
        reason = executor.policy.blocked(tap.name, tap.rx, tap.ry, texts, BLIND_FORBIDDEN_TEXT, BLIND_TEXT_MARGIN)
        if reason is not None:
            reasons.append(reason)
            continue
        log.event(
            "disconnect_tap", round=rounds, screen=str(obs.screen), tap=kind, x=tap.rx, y=tap.ry, frame=obs.frame_id
        )
        executor.execute([tap], obs.screen, s.phone.device, s.clock, 0.0, texts)
        s.progress()
        return
    log.warning("disconnect_tap_skipped", round=rounds, screen=str(obs.screen), reasons=reasons, frame=obs.frame_id)


def candidates(s, obs: Observation) -> list[tuple[str, Tap]]:
    executor = s.phone.executor
    taps = []
    if obs.screen in DISMISSABLE:
        box = box_of(obs.fields.get("dismiss")) if not obs.fields.get("unsafe") else None
        if box is None:
            found = safe_control(Texts(obs.texts), CONTROL_REGION, CONTROL)
            box = box_of(found["dismiss"]) if found and not found["unsafe"] else None
        if box is not None:
            taps.append(("control", executor.point_at_box("dismiss", box)))
    return [*taps, ("centre", executor.point(CENTRE))]


def resolved(s, obs: Observation, rounds: int, seconds: float) -> None:
    """A known screen is back. Off the battle screens and the post-battle ones, the battle is over."""
    if obs.screen in IN_BATTLE:
        outcome = "battle_resumed"
    elif obs.screen in POST_BATTLE:
        outcome = "battle_over"
    else:
        outcome = "left_battle"
    log.event(
        "disconnect_recovered",
        battle=s.battle.index,
        screen=str(obs.screen),
        outcome=outcome,
        rounds=rounds,
        seconds=round(seconds, 1),
    )
    if outcome == "left_battle":
        handlers.finalize(s, None)

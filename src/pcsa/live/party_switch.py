"""A switch from the action screen: open the party, find the member by name on screen, tap it, check it went in.

The row is read from the party screen itself: the game's order differs between the party and replacement screens.
A party screen still up after Switch In is tried once more.
"""

from __future__ import annotations

from pcsa.common import log
from pcsa.common.types import Screen
from pcsa.live.taps import ExecResult
from pcsa.live.turn_sync import resync_party
from pcsa.run.battle_state import BattleRecord, same_name

SWITCH_ATTEMPTS = 2  # the party taps, and once more when still up after Switch In


def target_row(b: BattleRecord, party, slot: int) -> int | None:
    """The visible row of the member in `slot`, standing, as the party screen lists it; None when not readable."""
    name = b.names[slot] if slot < len(b.names) else None
    rows = [m.row for m in party if name and same_name(m.name, name) and not m.fainted]
    return rows[0] if len(rows) == 1 else None


def _merge(total: ExecResult, part: ExecResult) -> None:
    total.taps += part.taps
    total.first_tap_at = total.first_tap_at if total.first_tap_at is not None else part.first_tap_at
    total.last_tap_at = part.last_tap_at if part.last_tap_at is not None else total.last_tap_at
    total.error = total.error or part.error


def switch(pipeline, b: BattleRecord, slot: int) -> tuple[ExecResult, bool]:
    """Pokémon -> read the party -> the target's row -> Switch In -> the party screen must be gone."""
    phone, clock, timing = pipeline.phone, pipeline.clock, pipeline.cfg.timing
    executor, device = phone.executor, phone.device
    result = executor.execute([executor.point("pokemon")], Screen.ACTION, device, clock, 0.0)
    clock.sleep(timing.party_settle_s)
    obs, _ = phone.observe("switch")
    if obs.screen == Screen.ACTION:  # the tap did not take: the blind plan from the tracked order
        log.warning("party_not_opened", battle=b.index, slot=slot)
        blind = executor.plan_switch(b.visible_order.index(slot))
        _merge(result, executor.execute(blind, Screen.ACTION, device, clock, timing.inter_tap_s))
        return result, True
    for attempt in range(1, SWITCH_ATTEMPTS + 1):
        if obs.screen != Screen.PARTY:
            return result, True
        party = tuple(obs.fields.get("party", ()))
        resync_party(b, party)
        row = target_row(b, party, slot)
        if row is None:  # name unread: fall back to the tracked order
            guess = b.visible_order.index(slot)
            shown = next((m for m in party if m.row == guess), None)
            row = guess if shown is not None and shown.name is None and not shown.fainted else None
        if row is None or row == 0:
            log.warning("switch_target_not_on_party", battle=b.index, slot=slot, rows=[m.name for m in party])
            break
        plan = executor.plan_party_switch(row, obs.fields.get("focused_row"))
        _merge(result, executor.execute(plan, Screen.PARTY, device, clock, timing.inter_tap_s))
        clock.sleep(timing.party_settle_s)
        obs, _ = phone.observe("switch")
        if obs.screen == Screen.PARTY:
            log.warning("switch_not_taken", battle=b.index, slot=slot, row=row, attempt=attempt)
    if obs.screen != Screen.PARTY:
        return result, True
    phone.navigator.back_out(obs)
    return result, False

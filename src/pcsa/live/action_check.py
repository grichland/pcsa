"""Did the game play the action the controller chose? Checked against the log of that turn once it is read.

A difference is a warning (`action_mismatch`) in the battle record. A chosen switch must show our send-out of that
member; a chosen move must not show our active using another. A turn showing neither is not a mismatch.
"""

from __future__ import annotations

import re

from pcsa.common import log
from pcsa.run.battle_state import BattleRecord, same_name

IGNORED_MOVES = ("struggle", "recharge")


def _key(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def own_moves(rows, active: str) -> list[str]:
    """Moves the log shows our `active` using, in order."""
    found = []
    for row in rows:
        if re.search(r"\bopposing\b", row, re.I):
            continue
        match = re.search(rf"\b{re.escape(active)} used (.+?)!?$", row.strip(), re.I)
        if match:
            found.append(match[1].strip())
    return found


def own_send_outs(rows) -> list[str]:
    """Our members the log shows going in ("Go! X!"), in order."""
    return [m[1].strip() for row in rows if (m := re.search(r"\bGo! (.+?)!", row))]


def mismatch(rows, last: dict) -> str | None:
    """Why the turn's log rows show another action than `last` (b.last_action); None when they agree or can't tell."""
    if not rows:
        return None
    active = last.get("active_name")
    if last.get("kind") == "switch":
        target = last.get("switch_name")
        if not target or not active:
            return None
        # The switch comes before any move of the member it replaces; a later send-out is a fainted-member replacement.
        for row in rows:
            if any(same_name(name, target) for name in own_send_outs([row])):
                return None
            if moved := own_moves([row], active):
                return f"chose switch to {target}; the log shows {active} used {moved[0]}"
        return None
    chosen = _key(last.get("move_name"))
    used = own_moves(rows, active) if active else []
    if not chosen or not used:
        return None
    first = _key(used[0])
    if first == chosen or first in IGNORED_MOVES or chosen in IGNORED_MOVES:
        return None
    return f"chose {last.get('move_name')}; the log shows {active} used {used[0]}"


def verify_last_action(b: BattleRecord) -> str | None:
    """Compare the last chosen action with its turn's log once that turn has been read; warn on a difference."""
    last = b.last_action
    if not last or last.get("checked"):
        return None
    turn = last.get("turn")
    if turn is None or b.last_turn is None or b.last_turn < turn or turn not in b.log.turns:
        return None  # that turn's log is not in yet
    last["checked"] = True
    reason = mismatch(b.log.turns[turn], last)
    if reason is not None:
        log.warning("action_mismatch", battle=b.index, turn=turn, chosen=last.get("label"), reason=reason)
        b.failures.append(f"turn {turn}: {reason}")
        b.actions.setdefault(turn, []).append(f"executed differently: {reason}")
    return reason

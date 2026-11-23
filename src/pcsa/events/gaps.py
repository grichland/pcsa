"""`gap` events (LIVE protocol): where the log read lost rows. Silence means the log was read completely.

{"k":"gap","what":"turn","n":N}: no row of turn N was read (a failed or skipped visit, a turn with no
action screen of ours, or the final turn, which is never collectable).
{"k":"gap","what":"rows","n":N}: part of turn N's page was not read. Recorded causes: the top of the page was
not captured; a re-visit could not be anchored to the rows already read; the turn was read only before it
was played, or the visits after it failed.
"""

from __future__ import annotations

import re

START = "start of log is not captured"
UNANCHORED = "could not be anchored"


def gap(turn: int, what: str) -> dict:
    """`what`: "turn" for a whole turn; anything else is part of a page and is sent as "rows"."""
    return {"k": "gap", "what": "turn" if what == "turn" else "rows", "n": turn}


def visit_gaps(visit: dict) -> list[dict]:
    """Gaps one audited visit reports, placed before the rows it appended."""
    out = []
    for warning in visit.get("warnings") or []:
        text = str(warning)
        m = re.match(r"Turn(\d+):", text)
        turn = int(m[1]) if m else int(visit.get("turn") or 0)
        if START in text:
            out.append(gap(turn, "start"))
        elif UNANCHORED in text:
            out.append(gap(turn, "middle"))
    return out


def final_turn(coverage: dict | None, rows: list[dict], decisions: list[dict]) -> int:
    """The battle's last turn: the coverage's final_turn note, else the highest turn seen."""
    turns = [r["turn"] for r in rows if isinstance(r.get("turn"), int)]
    turns += [d["turn"] for d in decisions if isinstance(d.get("turn"), int)]
    for c in (coverage or {}).get("coverage") or []:
        if c.get("kind") == "final_turn" and isinstance(c.get("turn"), int):
            turns.append(c["turn"])
    return max(turns, default=0)


def _pre_turn_visits(decisions: list[dict]) -> set[int]:
    """Visits that read a turn's log before that turn was played (the decision's turn equals the log turn)."""
    out = set()
    for d in decisions:
        action = d.get("kind") == "action" and isinstance(d.get("visit"), int) and d.get("turn") is not None
        if action and d.get("log_turn") == d.get("turn"):
            out.add(d["visit"])
    return out


def turn_gaps(rows: list[dict], audit: dict | None, decisions: list[dict], last: int) -> dict[int, dict]:
    """Per turn 1..last: `turn` when no row was read, `end` when the turn was only read before it was played."""
    with_rows = {r["turn"] for r in rows if isinstance(r.get("turn"), int)}
    pre = _pre_turn_visits(decisions)
    closed: set[int] = set()
    for v in (audit or {}).get("visits") or []:
        if v.get("visit") not in pre and isinstance(v.get("turn"), int):
            closed.add(v["turn"])
    out = {}
    for t in range(1, last + 1):
        if t not in with_rows:
            out[t] = gap(t, "turn")
        elif t not in closed and audit is not None:
            out[t] = gap(t, "end")
    return out

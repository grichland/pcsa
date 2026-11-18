"""Choosing one row's text from every frame that showed it: readings that pass the strict wording check vote; a
row with none is snapped (unique legal name within edit distance 1-2) or marked unread with its reason."""

from __future__ import annotations

import re
from collections import Counter

from pcsa.phone import log_wording
from pcsa.phone.log_row_read import CUT_TEXT, evaluate

LABEL = re.compile(r"^([^:’']{2,24}?)(?: ([+\-])(\d) Priority)?: (.+)$")


def split_label(text: str) -> tuple[str, str | None, str]:
    """(category, priority, message) of a row text; the category is any short leading label before ": "."""
    m = LABEL.match(text)
    if m and (m[1] in log_wording.CATEGORIES or log_wording.label_snap(m[1])):
        return m[1], (m[2] + m[3]) if m[2] else None, m[4]
    return "", None, text


def text_reading(text: str) -> dict:
    """A reading for a row known only by its text (a cached view, an older record)."""
    category, priority, message = split_label(text)
    return evaluate((category, priority, message, None), "text", {})


def snapped(reading: dict) -> log_wording.Check | None:
    """The reading with its misread names (and category label) snapped, when that is unique."""
    category, priority, message = split_label(reading["text"])
    label = log_wording.label_snap(category)
    if label is None:
        return None
    result = log_wording.snap_check(label, message)
    if label != category and not result.valid:
        result = log_wording.check(label, message)
    if not result.valid:
        return None
    if label != category:
        result.snaps = [{"hole": "CATEGORY", "read": category, "to": label}, *result.snaps]
    if priority:
        result.text = result.text.replace(f"{label}: ", f"{label} {priority} Priority: ", 1)
    return result


def vote(readings: list[dict]) -> dict:
    """{text, read: ok | snapped | unread, reason, votes, snaps} for one row."""
    plain = [r for r in readings if not r.get("covered")]
    valid = Counter(r["canonical"] for r in plain if r["valid"])
    if valid:
        ranked = valid.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            texts = " | ".join(t for t, n in ranked if n == ranked[0][1])
            return _unread(ranked[0][0], f"frames disagree: {texts}", valid)
        return {"text": ranked[0][0], "read": "ok", "reason": None, "votes": dict(valid), "snaps": []}
    covered = Counter(r["canonical"] for r in readings if r["valid"])
    if covered:
        return _unread(covered.most_common(1)[0][0], "read only through a timer overlay", covered)
    snaps = {}
    for reading in plain:
        if reading["reason"] != CUT_TEXT and (result := snapped(reading)):
            snaps.setdefault(result.text, result.snaps)
    if len(snaps) == 1:
        text, how = next(iter(snaps.items()))
        return {"text": text, "read": "snapped", "reason": None, "votes": {}, "snaps": how}
    raw = Counter(r["text"] for r in (plain or readings)).most_common(1)[0][0]
    if snaps:
        return _unread(raw, "snaps disagree: " + " | ".join(snaps), {})
    reasons = [r["reason"] for r in (plain or readings) if r["reason"]]
    return _unread(raw, Counter(reasons).most_common(1)[0][0] if reasons else "no reading passes", {})


def _unread(text: str, reason: str, votes: Counter | dict) -> dict:
    return {"text": text, "read": "unread", "reason": reason, "votes": dict(votes), "snaps": []}


def record(readings: list[dict]) -> list[dict]:
    """The readings as kept in the run record."""
    keys = ("text", "how", "valid", "covered", "source", "box", "reason")
    return [{k: r.get(k) for k in keys} for r in readings]

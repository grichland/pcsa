"""Merging the views of one log visit top to bottom into one list of row texts."""

from __future__ import annotations

import difflib
import re
from dataclasses import replace

from pcsa.phone.log_panel import CLIPPED_ROW_PX, CONTENT, Row, View
from pcsa.phone.log_row_read import CUT_TEXT
from pcsa.phone.log_vote import text_reading


def row_key(text: str) -> str:
    return re.sub(r"[^0-9a-z%]", "", text.lower())


WILDCARD = "<covered>"  # a row read only through a timer box: it matches any row, never alone


def match_text(row: Row) -> str:
    """The text a row is aligned by: a row under a timer box with no clean reading matches anything."""
    clean = any(r["valid"] and not r["covered"] for r in row.readings)
    return WILDCARD if row.covered and not clean else row.text


def same_row(a: str, b: str) -> bool:
    if WILDCARD in (a, b):
        return True
    if re.findall(r"\d+", a) != re.findall(r"\d+", b):
        return False  # HP rows differ only in their numbers
    ka, kb = row_key(a), row_key(b)
    if ka == kb:
        return True
    return bool(ka and kb) and difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.9


def rows_match(a: list[str], b: list[str]) -> bool:
    pairs = list(zip(a, b, strict=False))
    solid = any(WILDCARD not in pair for pair in pairs)
    return len(a) == len(b) and solid and all(same_row(x, y) for x, y in pairs)


def overlap(left: list[str], right: list[str], minimum: int = 2) -> int:
    for count in range(min(len(left), len(right)), minimum - 1, -1):
        if rows_match(left[-count:], right[:count]):
            return count
    return 0


def find_contiguous(haystack: list[str], needle: list[str]) -> int:
    if not needle:
        return 0
    for start in range(len(haystack) - len(needle) + 1):
        if rows_match(haystack[start : start + len(needle)], needle):
            return start
    return -1


def expected_overlap(upper: View, lower: View) -> int | None:
    """Rows of `upper` the scrollbar geometry says are also visible in `lower`."""
    if not (upper.has_scrollbar and lower.has_scrollbar):
        return None
    thumb_len = ((upper.thumb[1] - upper.thumb[0]) + (lower.thumb[1] - lower.thumb[0])) / 2 + 1
    shift = (lower.thumb[0] - upper.thumb[0]) / thumb_len * (CONTENT[3] - CONTENT[1])
    return sum(1 for row in upper.rows if row.bounds[0] - shift >= CONTENT[1] - 40)


def drop_clipped_edges(view: View) -> View:
    def clipped(row: Row) -> bool:
        cut = bool(row.readings) and all(r["reason"] == CUT_TEXT for r in row.readings)
        return (
            cut or row.bounds[1] - row.bounds[0] < CLIPPED_ROW_PX or any(s["possibly_clipped"] for s in row.segments)
        )

    rows = list(view.rows)
    if rows and not view.at_top and clipped(rows[0]):
        rows = rows[1:]
    if rows and not view.at_bottom and clipped(rows[-1]):
        rows = rows[:-1]
    return replace(view, rows=rows) if len(rows) != len(view.rows) else view


def join_count(merged: list[str], lines: list[str], previous: View | None, view: View) -> int | None:
    """Rows of `lines` already in `merged`; None when no overlap is verified."""
    if not merged or find_contiguous(merged, lines) >= 0:
        return len(lines) if merged else 0
    expected = expected_overlap(previous, view) if previous else None
    if expected is not None:
        for candidate in (expected, expected - 1, expected + 1):
            if 1 <= candidate <= min(len(lines), len(merged)) and rows_match(merged[-candidate:], lines[:candidate]):
                return candidate
    if found := overlap(merged, lines):
        return found
    if not view.at_top and len(lines) > 2 and (found := overlap(merged, lines[1:])) and len(merged) > found:
        return found + 1  # the top row is cut by the edge and OCR reads it as garbage
    return None


def unverified_gaps(views: list[View]) -> list[tuple[View, View]]:
    gaps = []
    for turn in sorted({v.turn for v in views}):
        ordered = sorted((v for v in views if v.turn == turn), key=lambda v: v.scroll)
        merged: list[str] = []
        previous = None
        for view in ordered:
            lines = [match_text(row) for row in drop_clipped_edges(view).rows]
            count = join_count(merged, lines, previous, view)
            if count is None:
                gaps.append((previous, view))
                count = 0
            merged.extend(lines[count:])
            previous = view
    return gaps


def certain(view: View) -> bool:
    """Every row of the view has a reading that passes the strict wording check."""
    return bool(view.rows) and all(
        any(r["valid"] and not r["covered"] for r in row.readings or [text_reading(row.text)]) for row in view.rows
    )


def usable_views(views: list[View]) -> tuple[list[View], list[str]]:
    """The settled views, plus each unsettled one (a last capture no second capture confirmed) whose rows join
    the settled views of its turn with a verified overlap; anything else could be a frame still scrolling. With
    no settled view at all, a lone capture at the bottom of the log whose every row passes the check is kept."""
    kept = [v for v in views if v.settled]
    warnings = []
    if not kept and (lone := [v for v in views if v.at_bottom and certain(v)]):
        kept = lone[-1:]
        warnings.append(f"{kept[0].source}: lone unconfirmed capture kept (every row passes the wording check)")
    for view in (v for v in views if not v.settled and v not in kept):
        if any(v.turn == view.turn for v in kept) and len(unverified_gaps([*kept, view])) == len(
            unverified_gaps(kept)
        ):
            kept.append(view)
        else:
            warnings.append(f"{view.source}: unsettled view left out (no verified overlap with a settled view)")
    return kept, warnings


def bridges_gap(older: View, bridge: View, newer: View) -> bool:
    return not unverified_gaps([older, bridge, newer])


def merge_views(views: list[View]) -> tuple[int, list[str], list[dict], list[str]]:
    """Merge the views of one visit top to bottom: (turn, rows, merges, warnings)."""
    if not views:
        raise ValueError("no views supplied")
    views = [drop_clipped_edges(v) for v in views]
    turns = {v.turn for v in views}
    warnings: list[str] = []
    if len(turns) != 1:
        warnings.append(f"views show different turns {sorted(turns)}; using the most common")
    turn = max(turns, key=lambda t: sum(v.turn == t for v in views))
    ordered = sorted((v for v in views if v.turn == turn), key=lambda v: v.scroll)
    merged: list[str] = []
    merges = []
    previous = None
    for view in ordered:
        lines = [match_text(row) for row in view.rows]
        expected = expected_overlap(previous, view) if previous else None
        count = join_count(merged, lines, previous, view)
        if count is None:
            count = 0
            warnings.append(
                f"Turn{turn}: no verified overlap before {view.source}"
                + (f" (geometry expected {expected} rows)" if expected is not None else "")
            )
        row_start = find_contiguous(merged, lines) if merged and count == len(lines) else len(merged) - count
        merged.extend(lines[count:])
        merges.append(
            {
                "source": view.source,
                "scroll": view.scroll,
                "overlap_rows": count,
                "expected_overlap": expected,
                "has_scrollbar": view.has_scrollbar,
                "row_start": row_start,
            }
        )
        previous = view
        warnings.extend(f"{view.source}: {w}" for w in view.warnings)
    if not ordered[0].at_top:
        warnings.append(f"Turn{turn}: start of log is not captured")
    if not ordered[-1].at_bottom:
        warnings.append(f"Turn{turn}: end of log is not captured")
    return turn, merged, merges, warnings

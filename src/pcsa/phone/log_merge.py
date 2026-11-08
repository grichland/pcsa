"""Merging log views into one per-turn text, and the per-battle master log with coverage notes."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from pcsa.phone.log_panel import (
    CLIPPED_ROW_PX,
    CONTENT,
    HEADER,
    SCROLLBAR,
    SEPARATORS,
    TIMER_BOX,
    Row,
    View,
    normalize,
)
from pcsa.phone.sprites import combine_observations


def row_key(text: str) -> str:
    return re.sub(r"[^0-9a-z%]", "", text.lower())


def same_row(a: str, b: str) -> bool:
    ka, kb = row_key(a), row_key(b)
    if ka == kb:
        return True
    return bool(ka and kb) and difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.9


def rows_match(a: list[str], b: list[str]) -> bool:
    return len(a) == len(b) and all(same_row(x, y) for x, y in zip(a, b, strict=True))


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
        return row.bounds[1] - row.bounds[0] < CLIPPED_ROW_PX or any(s["possibly_clipped"] for s in row.segments)

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
            lines = [row.text for row in drop_clipped_edges(view).rows]
            count = join_count(merged, lines, previous, view)
            if count is None:
                gaps.append((previous, view))
                count = 0
            merged.extend(lines[count:])
            previous = view
    return gaps


def usable_views(views: list[View]) -> tuple[list[View], list[str]]:
    """The settled views, plus each unsettled one (a last capture no second capture confirmed) whose rows join
    the settled views of its turn with a verified overlap; anything else could be a frame still scrolling."""
    kept = [v for v in views if v.settled]
    warnings = []
    for view in (v for v in views if not v.settled):
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
        lines = [row.text for row in view.rows]
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


@dataclass
class MergeReport:
    turn: int
    rows: list[str]
    appended: list[str]
    anchor: str
    warnings: list[str]


class BattleLog:
    """Per-battle master log {turn: rows}. Coverage notes are kept apart from the text."""

    def __init__(self, battle_id: str) -> None:
        self.battle_id = battle_id
        self.turns: dict[int, list[str]] = {}
        self.visits: list[dict] = []
        self.coverage: list[dict] = []
        self.failures: list[str] = []
        self.parts: dict[str, tuple[str, str]] = {}
        self.row_sprites: dict[int, list[dict | None]] = {}

    def history(self) -> dict[int, tuple[str, ...]]:
        return {t: tuple(rows) for t, rows in self.turns.items()}

    def sprite_history(self) -> dict[int, tuple[dict | None, ...]]:
        return {t: tuple(rows) for t, rows in self.row_sprites.items()}

    def _remember_parts(self, views: list[View]) -> None:
        for view in views:
            for row in view.rows:
                category = normalize(" ".join(s["text"] for s in row.segments if s["role"] == "category"))
                message = normalize(" ".join(s["text"] for s in row.segments if s["role"] == "message"))
                self.parts.setdefault(row.text, (category, message or row.text))

    def _sprites_for(self, views: list[View], turn: int, merges: list[dict], count: int) -> list[dict | None]:
        sprites: list[dict | None] = [None] * count
        ordered = sorted((drop_clipped_edges(v) for v in views if v.turn == turn), key=lambda v: v.scroll)
        for view, merged in zip(ordered, merges, strict=True):
            for offset, row in enumerate(view.rows):
                position = merged["row_start"] + offset
                observation = {**row.sprite, "source": view.source, "sha256": view.sha256} if row.sprite else None
                sprites[position] = combine_observations(sprites[position], observation)
        return sprites

    def merge(self, views: list[View], visit: dict | None = None) -> MergeReport:
        views, left_out = usable_views(views)
        self._remember_parts(views)
        turn, rows, merges, warnings = merge_views(views)
        warnings = left_out + warnings
        sprites = self._sprites_for(views, turn, merges, len(rows))
        stored = self.turns.get(turn)
        if stored is None:
            self.turns[turn], appended, anchor = list(rows), list(rows), "new"
            self.row_sprites[turn] = sprites
        else:
            appended, anchor = self._anchor(turn, stored, rows, sprites, warnings)
        self.visits.append(
            {
                **(visit or {}),
                "turn": turn,
                "rows": len(rows),
                "appended": len(appended),
                "anchor": anchor,
                "warnings": warnings,
                "merges": merges,
            }
        )
        return MergeReport(turn, rows, appended, anchor, warnings)

    def _anchor(
        self, turn: int, stored: list[str], rows: list[str], sprites: list, warnings: list[str]
    ) -> tuple[list[str], str]:
        existing = self.row_sprites.setdefault(turn, [None] * len(stored))
        if (start := find_contiguous(stored, rows)) >= 0:
            for index, sprite in enumerate(sprites, start):
                existing[index] = combine_observations(existing[index], sprite)
            return [], "contained"
        start = find_contiguous(rows, stored)
        if start == 0:
            appended = rows[len(stored) :]
            self.row_sprites[turn] = [
                combine_observations(o, n) for o, n in zip(existing, sprites, strict=False)
            ] + sprites[len(stored) :]
            stored.extend(appended)
            return appended, "prefix"
        if start > 0:
            warnings.append(f"Turn{turn}: re-visit shows {start} earlier rows; replaced stored rows")
            for index, sprite in enumerate(existing, start):
                sprites[index] = combine_observations(sprite, sprites[index])
            self.row_sprites[turn] = sprites
            self.turns[turn] = list(rows)
            return rows[:start] + rows[start + len(stored) :], "extended"
        count = overlap(stored, rows)
        if count:
            existing[-count:] = [
                combine_observations(a, b) for a, b in zip(existing[-count:], sprites[:count], strict=True)
            ]
            existing.extend(sprites[count:])
            stored.extend(rows[count:])
            return rows[count:], f"overlap{count}"
        warnings.append(f"Turn{turn}: re-visit could not be anchored; rows appended")
        existing.extend(sprites)
        stored.extend(rows)
        return list(rows), "unanchored"

    def note(self, kind: str, detail: str, **extra) -> None:
        self.coverage.append({"kind": kind, "detail": detail, **extra})

    def fail(self, detail: str) -> None:
        self.failures.append(detail)
        self.note("failure", detail)

    def text(self) -> str:
        return (
            "\n\n".join(f"Turn{t}:\n" + "\n".join(self.turns[t]) for t in sorted(self.turns)) + "\n"
            if self.turns
            else ""
        )

    def standard_text(self, header: dict, actions: dict[int, list[str]] | None = None) -> str:
        """The readable battle log: header lines, "=== Turn N ===" blocks of "[Category] message", '#' annotations."""
        lines = ["# Pokemon Champions battle log (Ranked Singles)", "# format: pcsa-battle-log/1"]
        lines += [f"{key}: {value}" for key, value in header.items()]
        lines.append(f"turns_collected: {', '.join(map(str, sorted(self.turns))) or 'none'}")
        lines.append("")
        turns: list[int] = [*self.turns, *(actions or {})]
        last = max(turns) if turns else 0
        for turn in range(1, last + 1):
            lines.append(f"=== Turn {turn} ===")
            for text in self.turns.get(turn, []):
                category, message = self.parts.get(text, ("", text))
                lines.append(f"[{category or '-'}] {message}")
            if turn not in self.turns:
                reason = next((n["detail"] for n in self.coverage if n.get("turn") == turn), "not collected")
                lines.append(f"# (no log for this turn: {reason})")
            lines += [f"# controller: {action}" for action in (actions or {}).get(turn, [])]
            lines.append("")
        lines.append("=== Coverage notes ===")
        lines += [f"# {n['kind']}: {n['detail']}" for n in self.coverage] or ["# none"]
        return "\n".join(lines) + "\n"

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "log.jsonl").open("w", encoding="utf-8") as stream:
            for turn in sorted(self.turns):
                sprites = self.row_sprites.get(turn, [None] * len(self.turns[turn]))
                for index, row in enumerate(self.turns[turn]):
                    stream.write(
                        json.dumps(
                            {
                                "battle_id": self.battle_id,
                                "turn": turn,
                                "index": index,
                                "text": row,
                                "sprite": sprites[index],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        (directory / "log.txt").write_text(self.text(), encoding="utf-8")
        (directory / "coverage.json").write_text(
            json.dumps(
                {
                    "battle_id": self.battle_id,
                    "turns": sorted(self.turns),
                    "coverage": self.coverage,
                    "failures": self.failures,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        layout = {
            "header": HEADER,
            "content": CONTENT,
            "separators": SEPARATORS,
            "scrollbar": SCROLLBAR,
            "timer_box": TIMER_BOX,
        }
        (directory / "ocr_audit.json").write_text(
            json.dumps(
                {"battle_id": self.battle_id, "visits": self.visits, "layout": layout},
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

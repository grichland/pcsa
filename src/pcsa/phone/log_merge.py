"""The per-battle master log: each visit's merged rows anchored into their turn, every frame's reading of a row
kept and voted on, and coverage notes kept apart from the text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pcsa.phone import log_record
from pcsa.phone.log_panel import CONTENT, SCROLLBAR, View
from pcsa.phone.log_side import combine_sides
from pcsa.phone.log_views import (
    WILDCARD,
    bridges_gap,
    drop_clipped_edges,
    find_contiguous,
    merge_views,
    overlap,
    same_row,
    unverified_gaps,
    usable_views,
)
from pcsa.phone.log_vote import text_reading, vote
from pcsa.phone.sprites import combine_observations

__all__ = ["BattleLog", "MergeReport", "bridges_gap", "merge_views", "unverified_gaps"]
Note = tuple[dict | None, str | None, tuple[dict, ...]]  # per merged row: (sprite observation, pill side, readings)
EMPTY: Note = (None, None, ())
ROW_PITCH = 144  # px from one row separator to the next
ONE_ROW_SLACK = 1.0


def combine_notes(first: Note, second: Note) -> Note:
    return combine_observations(first[0], second[0]), combine_sides(first[1], second[1]), first[2] + second[2]


def key_of(note: Note, text: str) -> str:
    """The text a stored row is aligned by (a wildcard when only a timer box let it be seen)."""
    readings = note[2]
    return WILDCARD if readings and all(r.get("covered") for r in readings) else text


def rows_above(view: View) -> float | None:
    """Rows hidden above a view, from the scrollbar thumb's position and length."""
    if not view.has_scrollbar:
        return 0.0 if view.at_top else None
    thumb = view.thumb[1] - view.thumb[0] + 1
    hidden = ((SCROLLBAR[3] - SCROLLBAR[1]) / thumb - 1) * (CONTENT[3] - CONTENT[1])
    return view.scroll * hidden / ROW_PITCH


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
        self.row_notes: dict[int, list[Note]] = {}
        self.votes: dict[int, list[dict | None]] = {}

    def history(self) -> dict[int, tuple[str, ...]]:
        return {t: tuple(rows) for t, rows in self.turns.items()}

    def sprite_history(self) -> dict[int, tuple[dict | None, ...]]:
        return {t: tuple(n[0] for n in notes) for t, notes in self.row_notes.items()}

    def side_history(self) -> dict[int, tuple[str | None, ...]]:
        return {t: tuple(n[1] for n in notes) for t, notes in self.row_notes.items()}

    def _notes_for(self, views: list[View], turn: int, merges: list[dict], count: int) -> list[Note]:
        notes: list[Note] = [EMPTY] * count
        ordered = sorted((drop_clipped_edges(v) for v in views if v.turn == turn), key=lambda v: v.scroll)
        for view, merged in zip(ordered, merges, strict=True):
            for offset, row in enumerate(view.rows):
                position = merged["row_start"] + offset
                observation = {**row.sprite, "source": view.source, "sha256": view.sha256} if row.sprite else None
                box = [CONTENT[0], row.bounds[0], CONTENT[2], row.bounds[1]]
                readings = tuple(
                    {**r, "source": view.source, "box": box} for r in row.readings or [text_reading(row.text)]
                )
                notes[position] = combine_notes(notes[position], (observation, row.side, readings))
        return notes

    def merge(self, views: list[View], visit: dict | None = None) -> MergeReport:
        views, left_out = usable_views(views)
        turn, rows, merges, warnings = merge_views(views)
        warnings = left_out + warnings
        notes = self._notes_for(views, turn, merges, len(rows))
        rows = [vote(list(n[2]))["text"] if n[2] else r for r, n in zip(rows, notes, strict=True)]
        keys = [key_of(n, r) for r, n in zip(rows, notes, strict=True)]
        stored = self.turns.get(turn)
        if stored is None:
            self.turns[turn], appended, anchor = list(rows), list(rows), "new"
            self.row_notes[turn] = notes
        else:
            top = min((v for v in views if v.turn == turn), key=lambda v: v.scroll)
            appended, anchor = self._anchor(turn, stored, (rows, keys), notes, warnings, rows_above(top))
        self._revote(turn)
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

    def _revote(self, turn: int) -> None:
        """Every row of the turn takes the vote of all its readings so far."""
        notes = self.row_notes.get(turn, [])
        texts = self.turns[turn]
        self.votes[turn] = [None] * len(texts)
        for index, note in enumerate(notes[: len(texts)]):
            if note[2]:
                chosen = vote(list(note[2]))
                texts[index] = chosen["text"]
                self.votes[turn][index] = chosen

    def _anchor(
        self,
        turn: int,
        stored: list[str],
        incoming: tuple[list[str], list[str]],
        notes: list[Note],
        warnings: list[str],
        above: float | None = None,
    ) -> tuple[list[str], str]:
        """Place a re-visit's rows (texts, alignment keys) into the stored turn."""
        rows, keys = incoming
        existing = self.row_notes.setdefault(turn, [EMPTY] * len(stored))
        held = [key_of(n, t) for n, t in zip(existing, stored, strict=False)] + stored[len(existing) :]
        if (start := find_contiguous(held, keys)) >= 0:
            for index, note in enumerate(notes, start):
                existing[index] = combine_notes(existing[index], note)
            return [], "contained"
        start = find_contiguous(keys, held)
        if start == 0:
            appended = rows[len(stored) :]
            self.row_notes[turn] = [combine_notes(o, n) for o, n in zip(existing, notes, strict=False)] + notes[
                len(stored) :
            ]
            stored.extend(appended)
            return appended, "prefix"
        if start > 0:
            warnings.append(f"Turn{turn}: re-visit shows {start} earlier rows; replaced stored rows")
            for index, note in enumerate(existing, start):
                notes[index] = combine_notes(note, notes[index])
            self.row_notes[turn] = notes
            self.turns[turn] = list(rows)
            return rows[:start] + rows[start + len(stored) :], "extended"
        count = overlap(held, keys) or self._one_row_overlap(turn, held, keys, above, warnings)
        if count:
            existing[-count:] = [combine_notes(a, b) for a, b in zip(existing[-count:], notes[:count], strict=True)]
            existing.extend(notes[count:])
            stored.extend(rows[count:])
            return rows[count:], f"overlap{count}"
        gap = f" (about {above - len(stored):.0f} rows between them not captured)" if above is not None else ""
        warnings.append(f"Turn{turn}: re-visit could not be anchored; rows appended{gap}")
        existing.extend(notes)
        stored.extend(rows)
        return list(rows), "unanchored"

    @staticmethod
    def _one_row_overlap(turn: int, stored: list[str], rows: list[str], above: float | None, warnings: list) -> int:
        """One shared row is enough when the scrollbar says exactly the other stored rows lie above the view."""
        if above is None or not rows or not same_row(stored[-1], rows[0]):
            return 0
        if abs(above - (len(stored) - 1)) > ONE_ROW_SLACK:
            return 0
        warnings.append(f"Turn{turn}: one-row overlap accepted ({above:.1f} rows above the view)")
        return 1

    def note(self, kind: str, detail: str, **extra) -> None:
        self.coverage.append({"kind": kind, "detail": detail, **extra})

    def fail(self, detail: str) -> None:
        self.failures.append(detail)
        self.note("failure", detail)

    def text(self) -> str:
        return log_record.plain_text(self.turns)

    def standard_text(self, header: dict, actions: dict[int, list[str]] | None = None) -> str:
        """The readable battle log: header lines, "=== Turn N ===" blocks of "[Category] message", '#' annotations."""
        return log_record.standard_text(self, header, actions)

    def save(self, directory: Path) -> None:
        log_record.save(self, directory)

"""Row parsing helper for the battle tests (no tracked Battle)."""

from __future__ import annotations

from pcsa.battle.log_rows import RawRow, parse_row, split_category


def parse(text: str):
    category, message = split_category(text)
    return parse_row(RawRow("T1#0", 1, 0, text, category, message))

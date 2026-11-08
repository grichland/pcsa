from __future__ import annotations

import pytest

from pcsa.battle.log_rows import RawRow, parse_row, split_category
from pcsa.battle.state import replay
from pcsa.battle.translate import Member, Translator

PRELUDE = ["|player|p1|me|", "|player|p2|opponent|", "|gen|9", "|start"]


def parse(text: str):
    category, message = split_category(text)
    return parse_row(RawRow("T1#0", 1, 0, text, category, message))


def feed(translator: Translator, messages: list[str]) -> None:
    for index, text in enumerate(messages):
        category, message = split_category(text)
        translator.row(RawRow(f"T1#{index}", 1, index, text, category, message))


def translator_for(ours: tuple[str, str, int], theirs: list[tuple[str, str]], turn: int = 1) -> Translator:
    """A translator with one of ours and the first of theirs on the field."""
    name, species, hp = ours
    lines = [
        *PRELUDE,
        f"|switch|p1a: {name}|{species}, L50|{hp}/{hp}",
        f"|switch|p2a: {theirs[0][0]}|{theirs[0][1]}, L50|100/100",
        f"|turn|{turn}",
    ]
    battle = replay(lines, None)
    return Translator(battle, [Member(name, species, hp)], [Member(n, s) for n, s in theirs])


@pytest.fixture
def garchomp_vs_snorlax() -> Translator:
    return translator_for(("Garchomp", "Garchomp", 185), [("Snorlax", "Snorlax")])

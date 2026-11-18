"""ids.to_id gives the recorded id for every recorded and asset name."""

from __future__ import annotations

from pcsa.battle.ids import to_id
from tests.battle.data_files import load


def test_to_id_matches_every_recorded_id() -> None:
    cases = load("to_id_cases.json.gz")["cases"]
    assert len(cases) > 15000
    assert [(text, want) for text, want in cases if to_id(text) != want] == []


def test_to_id_keeps_unicode_letters() -> None:
    assert to_id("Flabébé") == "flabébé"
    assert to_id("Farfetch’d") == "farfetchd"
    assert to_id("Mr. Mime-Galar") == "mrmimegalar"

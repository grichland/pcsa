"""ids.to_id equals poke-env's to_id_str on every recorded and asset name."""

from __future__ import annotations

from pcsa.battle.ids import to_id
from tests.battle.data_files import load


def test_to_id_matches_poke_env_on_every_name() -> None:
    cases = load("to_id_cases.json.gz")["cases"]
    assert len(cases) > 15000
    assert [(text, want) for text, want in cases if to_id(text) != want] == []


def test_to_id_keeps_unicode_letters() -> None:
    assert to_id("Flabébé") == "flabébé"
    assert to_id("Farfetch’d") == "farfetchd"
    assert to_id("Mr. Mime-Galar") == "mrmimegalar"

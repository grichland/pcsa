"""Every distinct recorded row text parses to pcsa's event (kind, side, name, data)."""

from __future__ import annotations

from tests.battle.conftest import parse
from tests.battle.data_files import load, plain


def record(event) -> dict:
    return plain({"kind": event.kind, "side": event.side, "name": event.name, "data": event.data})


def test_every_recorded_row_parses_as_pcsa_did() -> None:
    rows = load("log_rows_parity.json.gz")["rows"]
    assert len(rows) > 10000
    assert [(text, want) for text, want in rows if record(parse(text)) != want] == []

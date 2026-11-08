import json
import logging
from pathlib import Path

from pcsa.common import log


def test_events_go_to_jsonl_and_console(tmp_path: Path, capsys) -> None:
    logger = log.configure(tmp_path)
    try:
        log.event("decision", battle=1, turn=3, action="move2")
        log.warning("turn_overrun", battle=1, by_s=1.5)
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            log.exception("session_crash", attempt=2)
    finally:
        log.configure(None, console=False)
    lines = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [line["event"] for line in lines] == ["decision", "turn_overrun", "session_crash"]
    assert lines[0]["turn"] == 3 and lines[0]["level"] == "INFO"
    assert "boom" in lines[2]["traceback"]
    assert "decision" in capsys.readouterr().err
    assert logger.name == "pcsa" and not logger.propagate


def test_reserved_field_names_are_refused() -> None:
    import pytest

    with pytest.raises(ValueError):
        log.event("data")


def test_reconfigure_closes_previous_handlers(tmp_path: Path) -> None:
    log.configure(tmp_path / "a")
    log.configure(tmp_path / "b", console=False)
    handlers = logging.getLogger("pcsa").handlers
    assert len(handlers) == 1 and isinstance(handlers[0], log.JsonlHandler)
    assert handlers[0].path == tmp_path / "b" / "events.jsonl"
    log.configure(None, console=False)

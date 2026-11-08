"""Logging: a console line for people and an `events.jsonl` line for tools, from the same call.

from pcsa.common import log
log.configure(run_dir)              # once per run (run_dir may be None for console only)
log.event("decision", battle=1, turn=3, action="move2", degrade="full")
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pcsa.common.types import jsonable

LOGGER = "pcsa"
_RESERVED = {"event", "data"}
_CONSOLE_VALUE_WIDTH = 60


class JsonlHandler(logging.Handler):
    """One JSON object per line; flushed on every record so a crash loses nothing."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        line = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
        }
        line.update(jsonable(getattr(record, "data", {})))
        if record.exc_info:
            line["traceback"] = (
                self.formatter.formatException(record.exc_info) if self.formatter else str(record.exc_info[1])
            )
        self._stream.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()
        super().close()


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        event = getattr(record, "event", None)
        if event is None:
            return f"{stamp} {record.getMessage()}"
        data = getattr(record, "data", {})
        fields = " ".join(f"{k}={_short(v)}" for k, v in data.items())
        return f"{stamp} {event:<18} {fields}".rstrip()


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= _CONSOLE_VALUE_WIDTH else text[: _CONSOLE_VALUE_WIDTH - 3] + "..."


def configure(run_dir: Path | None, level: int = logging.INFO, console: bool = True) -> logging.Logger:
    """Install the handlers on the package logger. Safe to call again: previous handlers are closed."""
    logger = logging.getLogger(LOGGER)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(level)
    logger.propagate = False
    if console:
        stream = logging.StreamHandler()
        stream.setFormatter(ConsoleFormatter())
        logger.addHandler(stream)
    if run_dir is not None:
        logger.addHandler(JsonlHandler(run_dir / "events.jsonl"))
    return logger


def event(event_name: str, /, **data: Any) -> None:
    """Record one named event with structured fields."""
    if event_name in _RESERVED:
        raise ValueError(f"{event_name!r} is a reserved field name")
    logging.getLogger(LOGGER).info(event_name, extra={"event": event_name, "data": data})


def warning(event_name: str, /, **data: Any) -> None:
    logging.getLogger(LOGGER).warning(event_name, extra={"event": event_name, "data": data})


def exception(event_name: str, /, **data: Any) -> None:
    """Record the exception being handled, with its traceback."""
    logging.getLogger(LOGGER).error(event_name, exc_info=True, extra={"event": event_name, "data": data})

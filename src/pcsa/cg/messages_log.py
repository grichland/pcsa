"""Every message sent to and received from CG, one JSON line each, in the battle's folder: battle-N/cg-messages.jsonl
(found by its battle_id.json). Before that folder exists: cg-messages/<tag>.jsonl. Without a run folder (unit
tests) the lines are kept in memory."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

FILE = "cg-messages.jsonl"


class MessageLog:
    def __init__(self, run_dir: Path | None) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.lines: list[tuple[str, dict]] = []  # (battle tag, line) when there is no run folder
        self._paths: dict[str, Path] = {}
        self._lock = threading.Lock()

    def path(self, tag: str) -> Path | None:
        if self.run_dir is None:
            return None
        if tag in self._paths:
            return self._paths[tag]
        for marker in sorted(self.run_dir.glob("battle-*/battle_id.json")):
            try:
                if json.loads(marker.read_text(encoding="utf-8")).get("battle_id") == tag:
                    self._paths[tag] = marker.parent / FILE
                    return self._paths[tag]
            except (OSError, ValueError):
                continue
        return self.run_dir / "cg-messages" / f"{tag}.jsonl"

    def write(self, tag: str, direction: str, message, generation: int) -> None:
        line = {
            "at": datetime.now().isoformat(timespec="milliseconds"),
            "dir": direction,  # sent | received | note
            "cg": generation,  # which CG start this went to or came from
            "message": message,
        }
        with self._lock:
            path = self.path(tag)
            if path is None:
                self.lines.append((tag, line))
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")

    def read(self, tag: str) -> list[dict]:
        """This battle's lines, oldest first."""
        path = self.path(tag)
        if path is None:
            return [line for t, line in self.lines if t == tag]
        if not path.is_file():
            return []
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

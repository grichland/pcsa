"""Read one recorded pcsa battle folder (read-only) into plain data for the stream builder."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# run events.jsonl kinds the stream builder reads; the rest are screen and device noise
RUN_EVENTS = frozenset({
    "log_visit", "decision", "team_preview", "turn_timed_out", "switch_not_taken", "action_mismatch",
    "log_recovery", "battle_start", "battle_end", "turn",
})  # fmt: skip


@dataclass
class Recorded:
    run: str
    folder: str
    index: int | None
    meta: dict = field(default_factory=dict)  # battle.json
    rows: list[dict] = field(default_factory=list)  # log.jsonl
    decisions: list[dict] = field(default_factory=list)  # decisions/*.json by sequence
    audit: dict | None = None  # ocr_audit.json
    coverage: dict | None = None  # coverage.json
    own_team: dict | None = None  # the run's own_team.json (sheet + team)
    run_events: list[dict] = field(default_factory=list)  # this battle's events.jsonl entries

    @property
    def tag(self) -> str:
        for source in (self.meta, *(self.rows[:1]), *(self.decisions[:1]), self.audit or {}):
            if source.get("battle_id"):
                return str(source["battle_id"])
        return f"{self.run}/{self.folder}"


def _json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def run_events(run: Path) -> dict[int, list[dict]]:
    """A run's events.jsonl, the kinds the stream reads, by battle index."""
    out: dict[int, list[dict]] = {}
    for e in _lines(run / "events.jsonl"):
        if e.get("event") in RUN_EVENTS and isinstance(e.get("battle"), int):
            out.setdefault(e["battle"], []).append(e)
    return out


def load_battle(
    folder: Path, events_by_index: dict[int, list[dict]] | None = None, rows_root: Path | None = None
) -> Recorded:
    """`rows_root`: a tree of re-read logs (<run>/<battle>/log.jsonl, ocr_audit.json, coverage.json) to use
    instead of the recorded ones when the battle has one."""
    m = re.fullmatch(r"battle-(\d+)", folder.name)
    index = int(m[1]) if m else None
    decisions = [d for p in sorted((folder / "decisions").glob("*.json")) if (d := _json(p))]
    decisions.sort(key=lambda d: d.get("sequence", 0))
    if events_by_index is None:
        events_by_index = run_events(folder.parent)
    reread = rows_root / folder.parent.name / folder.name if rows_root else None
    logs = reread if reread and (reread / "log.jsonl").exists() else folder
    return Recorded(
        run=folder.parent.name,
        folder=folder.name,
        index=index,
        meta=_json(folder / "battle.json") or {},
        rows=_lines(logs / "log.jsonl"),
        decisions=decisions,
        audit=_json(logs / "ocr_audit.json"),
        coverage=_json(logs / "coverage.json"),
        own_team=_json(folder.parent / "own_team.json"),
        run_events=events_by_index.get(index, []) if index is not None else [],
    )

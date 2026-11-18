"""Cuts tests/fixtures/watch_run from a replayed run (tools/replay_run.py output): real records, nothing invented.

Kept: manifest.json and events.jsonl as written, each battle's battle_log.txt and team_preview.json, and a few
decision records per battle (a first turn, a forced switch, one after a Mega, the last one). JSON files are written
compactly (same values) to keep the fixture small; frames and images are left out.
Run: .venv/Scripts/python.exe tests/run/make_watch_fixture.py [replay folder]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "output" / "runs" / "20261118-162900-neural-148-replay"
TARGET = REPO / "tests" / "fixtures" / "watch_run"
RECORDS = {
    1: ("0000-action", "0001-forced", "0002-action", "0006-action"),
    2: ("0000-action", "0001-action", "0005-forced", "0011-action"),
    3: ("0000-action", "0002-action", "0009-action", "0010-action"),
}


def compact(source: Path, target: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def main(source: Path) -> None:
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)
    compact(source / "manifest.json", TARGET / "manifest.json")
    shutil.copy(source / "events.jsonl", TARGET / "events.jsonl")
    for battle, names in RECORDS.items():
        folder, out = source / f"battle-{battle}", TARGET / f"battle-{battle}"
        out.mkdir()
        shutil.copy(folder / "battle_log.txt", out / "battle_log.txt")
        compact(folder / "team_preview.json", out / "team_preview.json")
        for name in names:
            compact(folder / "decisions" / f"{name}.json", out / "decisions" / f"{name}.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE)

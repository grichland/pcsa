"""Re-read every recorded log frame with the current reader and cache each view (read-only on the recordings).

Run: python -m tools.ocr_eval.reread WORKER WORKERS [THREADS]
Writes docs/investigation/ocr-fix/views/<run>__<battle>.json; a battle already cached is skipped.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import cv2

from pcsa.common.config import OcrConfig
from pcsa.phone import log_panel as lp
from pcsa.phone.ocr import RapidOcrReader
from pcsa.phone.screens import to_reference

RUNS = Path(__file__).resolve().parents[2] / "output" / "runs"
OUT = Path(__file__).resolve().parents[2] / "docs" / "investigation" / "ocr-fix" / "views"


def battles() -> list[Path]:
    found = sorted(d for d in RUNS.glob("*/battle-*") if (d / "log-frames").is_dir())
    random.Random(20261117).shuffle(found)  # spread long and short battles over the workers
    return found


def view_record(view: lp.View) -> dict:
    rows = [
        {
            "text": r.text,
            "bounds": list(r.bounds),
            "segments": r.segments,
            "side": r.side,
            "readings": r.readings,
            "covered": r.covered,
        }
        for r in view.rows
    ]
    keys = ("turn", "scroll", "at_top", "at_bottom", "has_scrollbar", "warnings", "sha256")
    return {**{k: getattr(view, k) for k in keys}, "thumb": list(view.thumb), "rows": rows}


def read_battle(directory: Path, reader: RapidOcrReader) -> dict:
    frames, previous = [], None
    for path in sorted((directory / "log-frames").glob("*.png")):
        image = to_reference(cv2.imread(str(path)))
        record = {"frame": path.name, "panel": lp.is_log_panel(image)}
        record["same_as_prev"] = bool(previous is not None and lp.same_viewport(previous, image))
        record["overlay"] = round(lp.timer_overlay_ratio(image), 3)
        try:
            record.update(
                view_record(lp.read_view(image, reader, f"{directory.name}/log-frames/{path.name}", sprites=False))
            )
        except ValueError as error:
            record["error"] = str(error)[:200]
        frames.append(record)
        previous = image
    return {"run": directory.parent.name, "battle": directory.name, "frames": frames}


def main() -> None:
    worker, workers = int(sys.argv[1]), int(sys.argv[2])
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    OUT.mkdir(parents=True, exist_ok=True)
    reader = RapidOcrReader(OcrConfig(threads=threads))
    started = time.time()
    for directory in battles()[worker::workers]:
        target = OUT / f"{directory.parent.name}__{directory.name}.json"
        if target.exists():
            continue
        record = read_battle(directory, reader)
        target.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        print(worker, "done", directory.parent.name, directory.name, len(record["frames"]), "frames",
              round(time.time() - started), "s", flush=True)  # fmt: skip
    print(worker, "finished", round(time.time() - started), "s", flush=True)


if __name__ == "__main__":
    main()

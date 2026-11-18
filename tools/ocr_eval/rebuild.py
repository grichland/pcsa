"""Rebuild each battle's rows from the cached re-read views (tools.ocr_eval.reread) with the current merge and vote.

Visits the live run merged use the same frames it used; visits it gave up on ("no readable log panel") use every
readable frame of the visit, settled when the next capture showed the same viewport.
Run: python -m tools.ocr_eval.rebuild   ->  docs/investigation/ocr-fix/rebuilt/<run>/<battle>/
"""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from pathlib import Path

from pcsa.phone.log_merge import BattleLog
from pcsa.phone.log_panel import Row, View
from pcsa.phone.log_row_read import CUT_TEXT
from pcsa.phone.log_vote import text_reading

RUNS = Path(__file__).resolve().parents[2] / "output" / "runs"
FIX = Path(__file__).resolve().parents[2] / "docs" / "investigation" / "ocr-fix"
VISIT = re.compile(r"^visit(\d+)-")


KEEP = (CUT_TEXT, "unreadable priority badge", "priority badge on")


def refresh(reading: dict) -> dict:
    """The cached reading checked again against today's wordings (pixel verdicts such as a cut edge stay)."""
    if reading["reason"] and reading["reason"].startswith(KEEP):
        return reading
    again = text_reading(reading["text"])
    return {**reading, **{k: again[k] for k in ("valid", "canonical", "wording", "reason")}}


def to_view(battle: str, frame: dict, settled: bool) -> View:
    rows = [
        Row(
            r["text"],
            tuple(r["bounds"]),
            r["segments"],
            None,
            r["side"],
            [refresh(x) for x in r["readings"]],
            r["covered"],
        )
        for r in frame["rows"]
    ]
    source = f"{battle}/log-frames/{frame['frame']}"
    keys = ("turn", "scroll", "at_top", "at_bottom", "has_scrollbar")
    return View(source, frame["sha256"], *(frame[k] for k in keys[:2]), tuple(frame["thumb"]),
                *(frame[k] for k in keys[2:]), rows, frame["warnings"], settled=settled)  # fmt: skip


def visit_views(battle: str, frames: list[dict], used: set[str] | None) -> list[View]:
    """The views of one visit: the frames the live merge used, or every readable frame of a failed visit."""
    if used is not None:
        return [to_view(battle, f, True) for f in frames if f["frame"] in used and "rows" in f]
    out = []
    for i, frame in enumerate(frames):
        if "rows" not in frame:
            continue
        settled = frame["same_as_prev"] or (i + 1 < len(frames) and frames[i + 1]["same_as_prev"])
        out.append(to_view(battle, frame, settled))
    return out


def rebuild(cache: Path) -> dict:
    record = json.loads(cache.read_text(encoding="utf-8"))
    directory = RUNS / record["run"] / record["battle"]
    audit_file = directory / "ocr_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else {"visits": []}
    audit.setdefault("battle_id", record["battle"])
    used = {v["visit"]: {Path(m["source"]).name for m in v["merges"]} for v in audit["visits"] if "visit" in v}
    by_visit: dict[int, list[dict]] = defaultdict(list)
    for frame in record["frames"]:
        if m := VISIT.match(frame["frame"]):
            by_visit[int(m[1])].append(frame)
    log, last, notes = BattleLog(audit["battle_id"]), None, []
    for number in sorted(by_visit):
        views = visit_views(record["battle"], by_visit[number], used.get(number))
        if not views:
            continue
        staged = copy.deepcopy(log)
        try:
            report = staged.merge(views, {"visit": number, "replayed": number not in used})
        except ValueError as error:
            notes.append(f"visit {number}: {error}")
            continue
        if last is None or report.turn >= last:
            log, last = staged, max(report.turn, last or 0)
            if number not in used:
                notes.append(f"visit {number}: recovered turn {report.turn} ({len(report.rows)} rows)")
    for note in notes:
        log.note("rebuild", note)
    log.save(FIX / "rebuilt" / record["run"] / record["battle"])
    return {"run": record["run"], "battle": record["battle"], "notes": notes}


def main() -> None:
    caches = sorted((FIX / "views").glob("*.json"))
    summary = [rebuild(cache) for cache in caches]
    (FIX / "rebuild_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(summary), "battles rebuilt;", sum(len(s["notes"]) for s in summary), "notes")


if __name__ == "__main__":
    main()

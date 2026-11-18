"""Side (own / opp) of every recorded battle-log row from its category pill colour (pcsa output/runs, read-only).

Each log.jsonl row names the frame it was read from (sprite.source) and its row band (sprite.box). Writes
docs/investigation/cg-events/row-sides.jsonl.gz (one battle per line, rows aligned with log.jsonl) and
row-sides-summary.json (pill vs text-side agreement, every disagreement, Held Item / Ability sides).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pcsa.battle.log_rows import split_category  # noqa: E402
from pcsa.phone import log_side as ls  # noqa: E402
from pcsa.phone.log_side import pill_stats, side_from_stats  # noqa: E402

RUNS = Path(__file__).resolve().parents[2] / "output" / "runs"
OUT = Path(__file__).resolve().parents[2] / "docs" / "investigation" / "cg-events"
WHO_CATEGORIES = ("Move", "Fainted", "Mega Evolution")  # message names the actor: "The opposing ..." or not
HP_CATEGORIES = ("HP Lost", "HP Restored")  # percent = opponent, whole numbers = ours
SIDE_ONLY_CATEGORIES = ("Held Item", "Ability", "Ability Changed")
PERCENT = re.compile(r"\d\s*%")
WHOLE = re.compile(r"HP:?\s*\d+\s*\S\s*\d+\s*$")


def text_side(category: str, message: str) -> tuple[str | None, str]:
    """(side, rule) the row text gives unambiguously; side None when it gives none."""
    head = category.split(" ")[0] if category.startswith("Move") else category
    if head in WHO_CATEGORIES:
        return ("opp" if re.match(r"^the opposing\b", message, re.IGNORECASE) else "own"), "who"
    if category in HP_CATEGORIES:
        if PERCENT.search(message):
            return "opp", "hp_percent"
        if WHOLE.search(message):
            return "own", "hp_whole"
        return None, "hp_unclear"
    return None, "none"


def load_battles() -> list[dict]:
    battles = []
    for log in sorted(RUNS.glob("*/battle-*/log.jsonl")):
        rows = [json.loads(line) for line in log.read_text(encoding="utf8").splitlines() if line.strip()]
        battles.append({"run": log.parent.parent.name, "folder": log.parent.name, "rows": rows})
    return battles


def measure_frame(job: tuple[str, list[tuple[int, int]]]) -> tuple[str, list[dict] | None]:
    path, bands = job
    frame = cv2.imread(path)
    if frame is None:
        return path, None
    return path, [pill_stats(frame, band) for band in bands]


def frame_jobs(battles: list[dict]) -> dict[str, list[tuple[int, int]]]:
    jobs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for battle in battles:
        for row in battle["rows"]:
            sprite = row.get("sprite") or {}
            if sprite.get("source") and sprite.get("box"):
                band = (sprite["box"][1], sprite["box"][3])
                path = str(RUNS / battle["run"] / sprite["source"])
                if band not in jobs[path]:
                    jobs[path].append(band)
    return jobs


def measure_all(jobs: dict[str, list[tuple[int, int]]], workers: int) -> dict[tuple[str, tuple[int, int]], dict]:
    found: dict[tuple[str, tuple[int, int]], dict] = {}
    with Pool(workers) as pool:
        for done, (path, stats) in enumerate(pool.imap_unordered(measure_frame, jobs.items(), chunksize=16), 1):
            for band, value in zip(jobs[path], stats or [None] * len(jobs[path]), strict=True):
                found[(path, band)] = value or {"error": "unreadable frame"}
            if done % 1000 == 0:
                print(f"{done}/{len(jobs)} frames", flush=True)
    return found


def row_record(battle: dict, row: dict, found: dict) -> dict:
    sprite = row.get("sprite") or {}
    category, message = split_category(row["text"])
    record = {"row": f"T{row['turn']}#{row['index']}", "category": category, "text": row["text"]}
    if not (sprite.get("source") and sprite.get("box")):
        return {**record, "side": None, "pill_present": None, "error": "no sprite source/box"}
    stats = found[(str(RUNS / battle["run"] / sprite["source"]), (sprite["box"][1], sprite["box"][3]))]
    if "error" in stats:
        return {**record, "side": None, "pill_present": None, "error": stats["error"]}
    side = side_from_stats(stats)
    return {
        **record,
        "side": side,
        "pill_present": side is not None,
        "colour": stats,
        "source": sprite["source"],
        "box": sprite["box"],
    }


class Tally:
    """Pill side against the text side, plus the presence checks."""

    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.agreement: Counter = Counter()
        self.disagreements: list[dict] = []
        self.category_no_pill: list[dict] = []
        self.plain_with_pill: list[dict] = []
        self.unmeasured: list[dict] = []
        self.text_unclear: list[dict] = []  # HP rows whose OCR'd numbers give no side
        self.side_only: dict[str, Counter] = defaultdict(Counter)
        self.by_category: dict[str, Counter] = defaultdict(Counter)
        self.hue: Counter = Counter()

    def add(self, battle: dict, record: dict) -> None:
        where = {"run": battle["run"], "folder": battle["folder"], **record}
        self.counts["rows"] += 1
        if record["pill_present"] is None:
            self.unmeasured.append(where)
            return
        category = record["category"]
        self.by_category[category or "(plain)"][str(record["side"])] += 1
        if record["side"]:
            self.hue[int(record["colour"]["hue"])] += 1
        if category and not record["pill_present"]:
            self.category_no_pill.append(where)
        if not category and record["pill_present"]:
            self.plain_with_pill.append(where)
        if category in SIDE_ONLY_CATEGORIES:
            self.side_only[category][str(record["side"])] += 1
        expected, rule = text_side(category, split_category(record["text"])[1])
        if rule == "hp_unclear":
            self.text_unclear.append(where)
        if expected is None:
            return
        self.agreement[f"{rule}:{'agree' if expected == record['side'] else 'disagree'}"] += 1
        if expected != record["side"]:
            self.disagreements.append({**where, "text_side": expected, "rule": rule})


def thresholds() -> dict:
    names = (
        "PILL_X",
        "PILL_END",
        "PILL_MIN_V",
        "PILL_MIN_S",
        "PILL_LINE_FILL",
        "PILL_MIN_LINES",
        "OPP_HUE",
        "OPP_SHARE",
    )
    return {name: getattr(ls, name) for name in names}


def summary(battles: list[dict], tally: Tally, frames: int) -> dict:
    agree = sum(n for k, n in tally.agreement.items() if k.endswith(":agree"))
    checked = sum(tally.agreement.values())
    return {
        "battles": len(battles),
        "rows": tally.counts["rows"],
        "frames_read": frames,
        "thresholds": thresholds(),
        "text_side_checked": checked,
        "text_side_agree": agree,
        "text_side_disagree": checked - agree,
        "agreement_by_rule": dict(sorted(tally.agreement.items())),
        "disagreements": tally.disagreements,
        "category_rows_without_pill": tally.category_no_pill,
        "plain_rows_with_pill": tally.plain_with_pill,
        "rows_not_measured": tally.unmeasured,
        "hp_rows_text_side_unclear": tally.text_unclear,
        "side_only_categories": {k: dict(v) for k, v in tally.side_only.items()},
        "sides_by_category": {k: dict(v) for k, v in sorted(tally.by_category.items())},
        "pill_hue_median_histogram": dict(sorted(tally.hue.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    battles = load_battles()
    jobs = frame_jobs(battles)
    print(f"{len(battles)} battles, {sum(len(b['rows']) for b in battles)} rows, {len(jobs)} frames", flush=True)
    found = measure_all(jobs, args.workers)
    tally = Tally()
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / "row-sides.jsonl.gz", "wt", encoding="utf8") as stream:
        for battle in battles:
            records = [row_record(battle, row, found) for row in battle["rows"]]
            for record in records:
                tally.add(battle, record)
            lines = [
                {k: r.get(k) for k in ("row", "side", "pill_present", "colour", "error") if k in r} for r in records
            ]
            stream.write(json.dumps({"run": battle["run"], "folder": battle["folder"], "rows": lines}) + "\n")
    result = summary(battles, tally, len(jobs))
    (OUT / "row-sides-summary.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, indent=1, ensure_ascii=False))
    print("done", flush=True)


if __name__ == "__main__":
    main()

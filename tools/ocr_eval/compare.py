"""Compare the rebuilt rows (tools.ocr_eval.rebuild) with today's log.jsonl and with the strict wording check.

Run: python -m tools.ocr_eval.compare  ->  docs/investigation/ocr-fix/compare.json, changed_rows.jsonl, samples
"""

from __future__ import annotations

import csv
import difflib
import json
import random
import re
from collections import Counter
from pathlib import Path

from pcsa.phone.log_vote import split_label
from pcsa.phone.log_wording import check

RUNS = Path(__file__).resolve().parents[2] / "output" / "runs"
FIX = Path(__file__).resolve().parents[2] / "docs" / "investigation" / "ocr-fix"
AUDIT = FIX.parent / "ocr-audit" / "failing_rows.csv"


def strict(text: str):
    category, priority, message = split_label(text)
    result = check(category, message)
    if result.valid and priority:
        result.text = result.text.replace(f"{category}: ", f"{category} {priority} Priority: ", 1)
    return result


def rewrite_kind(before: str, canonical: str) -> str:
    """What the wording's own spelling changes in a row that already passes."""
    b, c = before.replace("'", "’"), canonical
    if re.sub(r"[.…]+$", "", b) == re.sub(r"[.…]+$", "", c):
        return "ellipsis"
    if re.sub(r"\s*→\s*", "→", b) == re.sub(r"\s*→\s*", "→", c):
        return "arrow spacing"
    if b.rstrip("!.l") == c.rstrip("!.") or b + "!" == c:
        return "sentence end"
    if b.replace(" ", "") == c.replace(" ", ""):
        return "spacing"
    if re.search(r"sent out|withdrew|went back to|Omni Ring", c):
        return "trainer slot written empty"
    return "other"


def before_class(row: dict, audit: dict) -> str:
    key = (row["run"], row["battle"], row["turn"], row["index"])
    if key in audit:
        return "audit: " + audit[key]
    result = strict(row["text"])
    if not result.valid:
        return "fails: " + (result.reason or "").split(":")[0]
    return "clean" if result.text == row["text"] else "rewrite: " + rewrite_kind(row["text"], result.text)


def load(path: Path, run: str, battle: str) -> list[dict]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{**r, "run": run, "battle": battle} for r in rows]


def key(text: str) -> str:
    result = strict(text)
    return re.sub(r"[^0-9a-z%]", "", (result.text if result.valid else text).lower())


def align(before: list[dict], after: list[dict], stats: Counter, changed: list, unchanged: list, diffs: list) -> None:
    """Per turn: rows kept, changed in place, gained and lost."""
    for turn in sorted({r["turn"] for r in before} | {r["turn"] for r in after}):
        b = [r for r in before if r["turn"] == turn]
        a = [r for r in after if r["turn"] == turn]
        stats["turns only before" if not a else "turns only after" if not b else "turns both"] += 1
        matcher = difflib.SequenceMatcher(None, [key(r["text"]) for r in b], [key(r["text"]) for r in a], False)
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                for x, y in zip(b[i1:i2], a[j1:j2], strict=True):
                    (unchanged if x["text"] == y["text"] else changed).append((x, y))
            elif op == "replace" and i2 - i1 == j2 - j1:
                changed.extend(zip(b[i1:i2], a[j1:j2], strict=True))
                stats["rows replaced in place"] += i2 - i1
            else:
                kept = {key(r["text"]) for r in a}
                stats["rows lost that repeat a kept row (duplicates removed)"] += sum(
                    key(r["text"]) in kept for r in b[i1:i2]
                )
                stats["rows lost"] += i2 - i1
                stats["rows gained"] += j2 - j1
                stats.update({f"lost/gained turn blocks ({op})": 1})
                diffs.append({"turn": turn, "op": op, "before": [r["text"] for r in b[i1:i2]],
                              "after": [r["text"] for r in a[j1:j2]], "run": (b or a)[0]["run"],
                              "battle": (b or a)[0]["battle"]})  # fmt: skip


def main() -> None:
    audit = {}
    with AUDIT.open(encoding="utf-8") as stream:
        for r in csv.DictReader(stream):
            audit[(r["run"], r["battle"], int(r["turn"]), int(r["index"]))] = r["class"]
    before_stats, after_stats, align_stats = Counter(), Counter(), Counter()
    changed, unchanged, unread, diffs = [], [], [], []
    for directory in sorted((FIX / "rebuilt").glob("*/battle-*")):
        run, battle = directory.parent.name, directory.name
        before = load(RUNS / run / battle / "log.jsonl", run, battle)
        after = load(directory / "log.jsonl", run, battle)
        before_stats.update(before_class(r, audit) for r in before)
        after_stats.update(
            f"{r.get('read')}" + (f": {r['reason'].split(':')[0]}" if r.get("reason") else "") for r in after
        )
        after_stats.update(
            "after text fails the check" for r in after if r.get("read") != "unread" and not strict(r["text"]).valid
        )
        unread += [r for r in after if r.get("read") == "unread"]
        align(before, after, align_stats, changed, unchanged, diffs)
    out = {"battles": len(list((FIX / "rebuilt").glob("*/battle-*"))), "before": before_stats.most_common(),
           "after": after_stats.most_common(), "alignment": align_stats.most_common(),
           "changed_rows": len(changed), "unchanged_rows": len(unchanged), "unread_rows": len(unread)}  # fmt: skip
    (FIX / "compare.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    with (FIX / "changed_rows.jsonl").open("w", encoding="utf-8") as stream:
        for x, y in changed:
            stream.write(json.dumps({"before": x, "after": y}, ensure_ascii=False) + "\n")
    (FIX / "row_diffs.json").write_text(json.dumps(diffs, indent=1, ensure_ascii=False), encoding="utf-8")
    with (FIX / "unread_rows.jsonl").open("w", encoding="utf-8") as stream:
        stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in unread)
    rng = random.Random(20261117)
    samples = {
        "changed": rng.sample(changed, min(300, len(changed))),
        "unchanged": rng.sample(unchanged, min(100, len(unchanged))),
    }
    (FIX / "samples.json").write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

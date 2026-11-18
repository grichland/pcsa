"""Gaps in a stream file by cause: the final turn, a turn with no action screen of ours, other whole turns, part pages.

Usage: gap_causes.py <stream .jsonl.gz> [rows_root]. Writes <stream>-gaps.json next to the stream.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parents[2] / "output" / "runs"


def skipped_turns(root: Path) -> dict[tuple[str, str], set[int]]:
    """Turns noted as 'no action screen of ours' in each battle's coverage.json."""
    out: dict[tuple[str, str], set[int]] = defaultdict(set)
    for path in root.glob("*/battle-*/coverage.json"):
        for c in json.loads(path.read_text(encoding="utf8")).get("coverage") or []:
            if c.get("kind") == "no_own_action_screen":
                out[(path.parent.parent.name, path.parent.name)].add(c.get("turn"))
    return out


def main() -> None:
    stream = Path(sys.argv[1])
    skipped = skipped_turns(Path(sys.argv[2]) if len(sys.argv) > 2 else RUNS)
    causes = Counter()
    for line in gzip.open(stream, "rt", encoding="utf8"):
        b = json.loads(line)
        events = [e for m in b["messages"] if m["msg"].get("type") == "events" for e in m["msg"]["events"]]
        last = max((e["n"] for e in events if e["k"] == "turn"), default=0)
        for e in events:
            if e["k"] != "gap":
                continue
            if e["what"] == "rows":
                causes["rows: part of a page"] += 1
            elif e["n"] == last:
                causes["turn: final turn"] += 1
            elif e["n"] in skipped[(b["run"], b["folder"])]:
                causes["turn: no action screen of ours"] += 1
            else:
                causes["turn: failed or unreadable visit"] += 1
    out = stream.with_name(stream.name.replace(".jsonl.gz", "-gaps.json"))
    out.write_text(json.dumps(dict(causes.most_common()), indent=1), encoding="utf8")
    print(stream.name, dict(causes.most_common()), "total", sum(causes.values()))


if __name__ == "__main__":
    main()

"""Plain switch vs move-then-pivot: what precedes each `withdraw` in the converted battles.

For every withdraw, find the same side's last `move` earlier in the same turn (if any). Writes pivots.json.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "investigation" / "cg-events"
PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "shedtail", "chillyreception", "batonpass"}


def main() -> None:
    table, moves_before, examples = Counter(), Counter(), {}
    for line in gzip.open(OUT / "events-0.2.jsonl.gz", "rt", encoding="utf8"):
        b = json.loads(line)
        events = b["events"]["events"]
        rows = b["rows"]
        row_iter = iter(rows)
        tagged = [(e, None if e["k"] == "turn" else next(row_iter)) for e in events]
        turn_start = 0
        for i, (e, row) in enumerate(tagged):
            if e["k"] == "turn":
                turn_start = i
                continue
            if e["k"] != "withdraw":
                continue
            last = None
            for prev, _ in reversed(tagged[turn_start:i]):
                if prev["k"] == "move" and prev.get("side") == e["side"]:
                    last = prev["move"]
                    break
            if last is None:
                before = "no move by that side earlier this turn"
            elif last in PIVOT_MOVES:
                before = "pivot move"
            else:
                before = "other move"
            key = ("pivot wording" if e.get("pivot") else "plain wording", e["side"], before)
            table[key] += 1
            if e.get("pivot") or before != "no move by that side earlier this turn":
                moves_before[(key[0], last)] += 1
            examples.setdefault(key, f"{b['run']}/{b['folder']} {row}")
    out = {
        "table": [{"wording": k[0], "side": k[1], "before": k[2], "count": n, "example": examples[k]}
                  for k, n in sorted(table.items())],
        "last_move_before": [{"wording": k[0], "move": k[1], "count": n} for k, n in moves_before.most_common()],
    }
    (OUT / "pivots.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    for r in out["table"]:
        print(r)
    for r in out["last_move_before"][:40]:
        print(r)


if __name__ == "__main__":
    main()

"""Survey parse_row output over every recorded log row (kinds and their data shapes)."""
import glob, json, sys
from collections import Counter
from pcsa.battle.log_rows import RawRow, parse_row, split_category

RUNS = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "output" / "runs")
c = Counter(); ex = {}
for path in glob.glob(RUNS + r"\*\battle-*\log.jsonl"):
    for line in open(path, encoding="utf8"):
        r = json.loads(line)
        cat, msg = split_category(r["text"])
        e = parse_row(RawRow("x", r["turn"], r["index"], r["text"], cat, msg, r.get("sprite")))
        key = (e.kind, e.data.get("what") or e.data.get("cause") or e.data.get("effect") or e.data.get("reason") or "", e.side is None, e.name is None)
        c[key] += 1; ex.setdefault(key, r["text"])
for k, n in sorted(c.items(), key=lambda x: (x[0][0], -x[1])):
    print(n, k, "|", ex[k][:90])

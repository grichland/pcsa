"""Convert every recorded battle (pcsa output/runs, read-only) to LIVE.md 0.2 messages and count them.

Writes docs/investigation/cg-events/: events-0.2.jsonl.gz (one battle per line), summary.json,
templates.json (per PL id: kinds and fields produced), unread.json, per_battle.csv.
"""

from __future__ import annotations

import csv
import glob
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pcsa.battle.ids import to_id  # noqa: E402
from pcsa.events.live import convert_rows  # noqa: E402
from pcsa.events.live_check import problems  # noqa: E402
from pcsa.events.live_ids import VERSION, Teams, base_id  # noqa: E402
import templates as tpl  # noqa: E402
from convert_stream import pill_sides  # noqa: E402

RUNS = Path(__file__).resolve().parents[2] / "output" / "runs"
OUT = Path(__file__).resolve().parents[2] / "docs" / "investigation" / "cg-events"
RESULTS = {"win": "win", "loss": "loss", "opponent_forfeit": "forfeit", "unknown": "unknown"}


def battle_meta(folder: Path) -> dict:
    path = folder / "battle.json"
    return json.loads(path.read_text(encoding="utf8")) if path.exists() else {}


def teams_of(meta: dict) -> Teams:
    own = meta.get("team_played_names") or meta.get("team_requested_names") or meta.get("roster") or []
    preview = []
    for slot in meta.get("opponent_preview") or []:
        top = (slot.get("species") or slot.get("top") or [[None]])[0]
        preview.append(top[0] if top else None)
    return Teams(tuple(n for n in own if n), tuple(preview))


def shape(event: dict) -> str:
    keys = [k for k in event if k != "k"]
    return event["k"] + "(" + ",".join(keys) + ")"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    templates = tpl.load()
    kinds, probs, unread_texts = Counter(), Counter(), Counter()
    per_tpl: dict[str, dict] = defaultdict(lambda: {"rows": 0, "shapes": Counter(), "problems": Counter(), "example": None})
    no_tpl, repairs, side_unknown, errors = Counter(), [], Counter(), []
    per_battle, rows_total, battles = [], 0, 0
    sides = pill_sides()  # the category pill's colour per row, read from the recorded frames
    with gzip.open(OUT / "events-0.2.jsonl.gz", "wt", encoding="utf8") as sink:
        for log in sorted(glob.glob(str(RUNS / "*" / "battle-*" / "log.jsonl"))):
            folder = Path(log).parent
            rows = [json.loads(x) for x in open(log, encoding="utf8") if x.strip()]
            if not rows:
                continue
            battles += 1
            pills = sides.get((folder.parent.name, folder.name), {})
            for r in rows:
                r.setdefault("side", pills.get(f"T{r['turn']}#{r['index']}"))
            meta = battle_meta(folder)
            teams = teams_of(meta)
            events, conv = convert_rows(rows, teams)
            tag = meta.get("battle_id") or rows[0].get("battle_id")
            start = {"v": VERSION, "battle": tag, "type": "battle_start",
                     "own_team": [{"species": to_id(n)} for n in (meta.get("roster") or [])],
                     "opp_preview": [base_id(s) if s else None for s in teams.opp_preview]}
            end = {"v": VERSION, "battle": tag, "type": "battle_end", "result": RESULTS.get(meta.get("outcome"), "unknown")}
            row_events = [e for e in events if e["k"] != "turn"]
            assert len(row_events) == len(rows), folder  # one row, one event
            for r, e in zip(rows, row_events, strict=True):
                t = tpl.match(templates, r["text"])
                key = t.id if t else "none"
                slot = per_tpl[key]
                slot["rows"] += 1
                slot["shapes"][shape(e)] += 1
                for p in problems(e):
                    slot["problems"][p] += 1
                slot["example"] = slot["example"] or {"text": r["text"], "event": e}
                if not t:
                    no_tpl[r["text"]] += 1
            for e in events:
                kinds[e["k"]] += 1
                for p in problems(e):
                    probs[f"{e['k']}:{p}"] += 1
                if e["k"] == "unread":
                    unread_texts[e["text"]] += 1
            for x in conv.side_unknown:
                side_unknown[x] += 1
            repairs += conv.repairs
            errors.extend(conv.errors)
            n_unread = sum(e["k"] == "unread" for e in events)
            rows_total += len(rows)
            per_battle.append((folder.parent.name, folder.name, tag, len(rows), n_unread))
            sink.write(json.dumps({"run": folder.parent.name, "folder": folder.name, "battle_start": start,
                                   "events": {"v": VERSION, "battle": tag, "type": "events", "events": events},
                                   "battle_end": end,
                                   "rows": [f"T{r['turn']}#{r['index']}" for r in rows]}, ensure_ascii=False) + "\n")
    n_unread = kinds["unread"]
    summary = {
        "contract": VERSION, "battles": battles, "rows": rows_total, "events": sum(kinds.values()),
        "events_per_kind": dict(kinds.most_common()),
        "unread": n_unread, "unread_share": round(n_unread / rows_total, 5),
        "battles_with_unread": sum(1 for b in per_battle if b[4]),
        "max_unread_share_in_a_battle": max(b[4] / b[3] for b in per_battle),
        "unread_top20": unread_texts.most_common(20),
        "field_problems": dict(probs.most_common()),
        "rows_matching_no_template": sum(no_tpl.values()),
        "no_template_top": no_tpl.most_common(30),
        "ocr_name_repairs": len(repairs),
        "converter_errors": len(errors),
        "converter_errors_top": Counter(e["error"] for e in errors).most_common(10),
        "side_not_settled": sum(side_unknown.values()),
        "side_not_settled_top": side_unknown.most_common(15),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf8")
    table = {k: {"rows": v["rows"], "shapes": dict(v["shapes"].most_common()), "problems": dict(v["problems"]),
                 "example": v["example"]} for k, v in sorted(per_tpl.items())}
    (OUT / "templates.json").write_text(json.dumps(table, indent=1, ensure_ascii=False), encoding="utf8")
    (OUT / "unread.json").write_text(json.dumps(unread_texts.most_common(), indent=1, ensure_ascii=False), encoding="utf8")
    (OUT / "repairs.json").write_text(json.dumps(repairs, indent=1, ensure_ascii=False), encoding="utf8")
    with open(OUT / "per_battle.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["run", "folder", "battle_id", "rows", "unread", "unread_share"])
        for b in per_battle:
            w.writerow([*b, round(b[4] / b[3], 4)])
    print(json.dumps({k: summary[k] for k in ("battles", "rows", "events", "unread", "unread_share",
                                              "rows_matching_no_template", "side_not_settled", "ocr_name_repairs",
                                              "converter_errors")}))
    print(json.dumps(summary["field_problems"], indent=0))


if __name__ == "__main__":
    main()

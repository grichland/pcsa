"""Every recorded battle (pcsa output/runs, read-only) -> the full ordered LIVE.md message stream (VERSION).

Writes docs/investigation/cg-events/: stream-<VERSION>[-reread].jsonl.gz (one battle per line: run, folder, battle,
the ordered messages, and the row ids in the order sent) and stream-<VERSION>[-reread]-summary.json.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pcsa.battle.ids import to_id  # noqa: E402
from pcsa.events.live_check import message_problems, order_problems  # noqa: E402
from pcsa.events.live_ids import VERSION  # noqa: E402
from pcsa.events.recorded import Recorded, load_battle, run_events  # noqa: E402
from pcsa.events.stream import build_stream  # noqa: E402

RUNS = Path(__file__).resolve().parents[2] / "output" / "runs"
OUT = Path(__file__).resolve().parents[2] / "docs" / "investigation" / "cg-events"
EXPORT_STATS = {"HP": "hp", "Atk": "atk", "Def": "def", "SpA": "spa", "SpD": "spd", "Spe": "spe"}


def export_sets(text: str) -> dict[str, dict]:
    """own_export text -> {species id: {nature, points}} for the cross-check."""
    out = {}
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        if not lines:
            continue
        name = to_id(lines[0].split(" @ ")[0].split(" (")[-1].rstrip(")"))
        points = dict.fromkeys(EXPORT_STATS.values(), 0)
        nature = None
        for line in lines[1:]:
            if line.startswith("EVs: "):
                for part in line[5:].split(" / "):
                    value, stat = part.split()
                    points[EXPORT_STATS[stat]] = int(value)
            elif m := re.fullmatch(r"(\w+) Nature", line):
                nature = to_id(m[1])
        out[name] = {"nature": nature, "points": points}
    return out


def cross_check(rec: Recorded, sets: list[dict], tally: Counter) -> None:
    """The SETs against what the decision records carry: request stats and own_export points and nature."""
    by_species = {s["species"]: s for s in sets}
    for record in rec.decisions[:1]:
        for mon in ((record.get("request") or {}).get("side") or {}).get("pokemon") or []:
            species = to_id(str(mon.get("details", "")).split(",")[0])
            one = by_species.get(species)
            if one and "stats" in one and mon.get("stats"):
                tally["request_stats_" + ("same" if mon["stats"] == one["stats"] else "differ")] += 1
        for species, exported in export_sets(record.get("own_export") or "").items():
            one = by_species.get(species)
            if one:
                tally["export_nature_" + ("same" if exported["nature"] == one.get("nature") else "differ")] += 1
                if "points" in one:
                    tally["export_points_" + ("same" if exported["points"] == one["points"] else "differ")] += 1
                else:
                    tally["export_points_but_set_has_none"] += 1


def pill_sides() -> dict[tuple[str, str], dict[str, str | None]]:
    """Row pill sides another tool read from the log frames (row-sides.jsonl.gz), when present."""
    path = OUT / "row-sides.jsonl.gz"
    if not path.exists():
        return {}
    out = {}
    with gzip.open(path, "rt", encoding="utf8") as f:
        for line in f:
            b = json.loads(line)
            out[(b["run"], b["folder"])] = {r["row"]: r.get("side") for r in b["rows"]}
    return out


def folders() -> list[Path]:
    out = []
    for folder in sorted(RUNS.glob("*/battle-*")):
        log = folder / "log.jsonl"
        has_rows = log.exists() and log.stat().st_size > 0
        if has_rows or any((folder / "decisions").glob("*.json")):
            out.append(folder)
    return out


def tally_stream(stream, problems: Counter, totals: dict) -> None:
    for item in stream.messages:
        msg = item["msg"]
        if item["from"] == "controller":
            for p in message_problems(msg):
                problems[p] += 1
            if msg.get("type") == "events" and "fallback" in msg:
                fb = msg["fallback"]
                totals["fallback_orders"][str(not order_problems(fb["order"])) if "order" in fb else "none"] += 1
            for e in msg.get("events") or []:
                if e["k"] == "gap":
                    totals["gaps_by_what"][e["what"]] += 1
        elif "order" in msg:
            for p in order_problems(msg["order"]):
                problems[f"answer:{p}"] += 1


REREAD = Path(__file__).resolve().parents[2] / "docs" / "investigation" / "ocr-fix" / "rebuilt"


def main() -> None:
    """`--reread`: take each battle's rows from the re-read logs (tools/ocr_eval), which carry their own pill side."""
    reread = "--reread" in sys.argv[1:]
    name = "stream-" + VERSION + ("-reread" if reread else "")
    OUT.mkdir(parents=True, exist_ok=True)
    counts, problems, cross, no_sets, notes = Counter(), Counter(), Counter(), Counter(), Counter()
    totals = {"gaps_by_what": Counter(), "fallback_orders": Counter()}
    battles = rows = 0
    events_cache: dict[Path, dict] = {}
    sides = pill_sides()
    with gzip.open(OUT / f"{name}.jsonl.gz", "wt", encoding="utf8") as sink:
        for folder in folders():
            if folder.parent not in events_cache:
                events_cache.clear()
                events_cache[folder.parent] = run_events(folder.parent)
            rec = load_battle(folder, events_cache[folder.parent], REREAD if reread else None)
            pills = sides.get((rec.run, rec.folder), {})
            for r in rec.rows:
                if not reread:
                    r["side"] = pills.get(f"T{r['turn']}#{r['index']}")
            counts["rows_with_pill_side"] += sum(r["side"] is not None for r in rec.rows)
            stream = build_stream(rec)
            battles += 1
            rows += len(rec.rows)
            counts.update(stream.counts)
            start = stream.messages[0]["msg"]
            if not start["own_team"]:
                no_sets[stream.notes[0] if stream.notes else "unknown"] += 1
            else:
                cross_check(rec, start["own_team"], cross)
            for n in stream.notes:
                notes[n] += 1
            tally_stream(stream, problems, totals)
            sink.write(json.dumps({"run": rec.run, "folder": rec.folder, "battle": stream.tag,
                                   "messages": stream.messages, "row_ids": stream.row_ids}, ensure_ascii=False) + "\n")
            print(f"{rec.run}/{rec.folder}: {len(stream.messages)} messages, {len(rec.rows)} rows", flush=True)
    pick = lambda prefix: {k[len(prefix):]: v for k, v in counts.most_common() if k.startswith(prefix)}  # noqa: E731
    summary = {
        "contract": VERSION, "battles": battles, "rows": rows,
        "messages_per_type": {k: counts[k] for k in ("battle_start", "preview_request", "picks", "events",
                                                      "decision_request", "order", "battle_end")},
        "decision_kinds": pick("decision:"), "refusals": pick("refused:"), "retries": pick("retry:"),
        "replace_placed": pick("replace_placed:"),
        "fallbacks_by_reason": pick("fallback:"), "fallbacks_not_sent": pick("fallback_not_sent:"),
        "fallback_order_valid": dict(totals["fallback_orders"]),
        "gaps_by_what": dict(totals["gaps_by_what"].most_common()),
        "events_per_kind": pick("k:"),
        "orders_unresolved": counts["order_unresolved"],
        "rows_after_a_later_turn": counts["rows_after_a_later_turn"],
        "rows_with_pill_side": counts["rows_with_pill_side"], "rows_without_visit": counts["rows_without_visit"],
        "visit_rows_missing": counts["visit_rows_missing"],
        "battles_without_sets": dict(no_sets), "set_and_order_notes": dict(notes.most_common()),
        "sets_cross_check_first_decision": dict(sorted(cross.items())),
        "field_problems": dict(problems.most_common()),
    }
    (OUT / f"{name}-summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

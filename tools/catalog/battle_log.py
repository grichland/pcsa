"""D1 generator: measures every recorded battle-log row with pcsa's parser.

Writes docs/battle-log.md and docs/investigation/catalog/battle_log.json. Read-only on pcsa.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
PCSA = ROOT.parent / "pcsa"
if str(PCSA / "src") not in sys.path:
    sys.path.insert(0, str(PCSA / "src"))

from pcsa.battle.log_rows import parse_row, rows_from_texts, who  # noqa: E402

from battle_log_checks import never_shown  # noqa: E402
from battle_log_md import render  # noqa: E402

COMMAND = (
    "$env:PYTHONPATH='C:\\Users\\geral\\Code\\pokemon_champions\\pcsa\\src'; $env:PYTHONDONTWRITEBYTECODE=1; "
    "C:\\Users\\geral\\Code\\pokemon_champions\\pcsa\\.venv\\Scripts\\python.exe -B tools\\catalog\\battle_log.py"
)
# Data keys listed in full as sub-types (names, numbers and move names are not sub-types).
NOT_SUBTYPE = {"before", "after", "count", "amount", "trainer", "target", "target_name", "title", "move"}
# Reasons that come from the row parser itself; the rest are translator reasons.
PARSER_REASON = re.compile(r"^(no_pattern|unreadable_|mixed_hp_units|unsupported mechanic|ambiguous_template)")


def battle_dirs(runs: Path) -> list[Path]:
    return sorted(p for p in runs.glob("*/battle-*") if p.is_dir())


def load_battle(path: Path) -> list[dict]:
    """Rows of one battle's log.jsonl, parsed per turn with pcsa's parser."""
    by_turn: dict[int, list[dict]] = defaultdict(list)
    with (path / "log.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rec = json.loads(line)
                by_turn[int(rec["turn"])].append(rec)
    out = []
    key = f"{path.parent.name}/{path.name}"
    for turn in sorted(by_turn):
        recs = sorted(by_turn[turn], key=lambda r: r["index"])
        raw = rows_from_texts(turn, [r["text"] for r in recs], [r.get("sprite") for r in recs])
        for rec, row in zip(recs, raw):
            out.append({"battle": key, "turn": turn, "index": rec["index"], "text": rec["text"],
                        "sprite": rec.get("sprite"), "event": parse_row(row)})
    return out


def present(value) -> bool:
    return value is not None and value != "" and value != []


def subtype_value(value) -> str:
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value)


def measure_kinds(rows: list[dict]) -> dict:
    """Per kind: rows, battles, an example, field presence and full sub-type lists."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["event"].kind].append(r)
    kinds = {}
    for kind, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        fields: Counter = Counter()
        subtypes: dict[str, Counter] = defaultdict(Counter)
        moves: Counter = Counter()
        for r in items:
            e = r["event"]
            fields["(text says 'opposing')"] += "opposing" in r["text"].lower()
            fields["side"] += present(e.side)
            fields["name"] += present(e.name)
            for k, v in e.data.items():
                fields[k] += present(v)
                if k == "move" and present(v):
                    moves[v] += 1
                elif k not in NOT_SUBTYPE:
                    subtypes[k][subtype_value(v)] += 1
            if e.side:
                subtypes["side (parsed)"][e.side] += 1
        texts = Counter(r["text"] for r in items)
        kinds[kind] = {
            "rows": len(items),
            "battles": len({r["battle"] for r in items}),
            "example": texts.most_common(1)[0][0],
            "distinct_texts": len(texts),
            "fields": {k: round(100 * v / len(items), 1) for k, v in sorted(fields.items())},
            "subtypes": {k: dict(c.most_common()) for k, c in sorted(subtypes.items())},
            "moves": dict(moves.most_common()) if moves else None,
        }
    return kinds


def ident(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def measure_sprites(rows: list[dict]) -> dict:
    """Sprite status per kind, base species vs text name, and forms the sprite adds."""
    per_kind: dict[str, Counter] = defaultdict(Counter)
    agree: dict[str, Counter] = defaultdict(Counter)
    disagree: Counter = Counter()
    forms: Counter = Counter()
    reasons: Counter = Counter()
    keys: Counter = Counter()
    for r in rows:
        s, e = r["sprite"], r["event"]
        status = s.get("status") if isinstance(s, dict) else "no sprite record"
        per_kind[e.kind][status] += 1
        if not isinstance(s, dict):
            continue
        keys[" ".join(sorted(s))] += 1
        if s.get("reason"):
            reasons[s["reason"]] += 1
        if status != "matched":
            continue
        if s.get("name") and s.get("name") != s.get("base_name"):
            forms[f"{s['base_name']} -> {s['name']}"] += 1
        if e.name:
            text_name = who(e.name)[1]
            same = ident(text_name) == ident(s.get("base_name")) or ident(text_name) == ident(s.get("name"))
            agree[e.kind]["agree" if same else "disagree"] += 1
            if not same:
                disagree[f"{e.kind}: text {text_name!r} vs sprite {s.get('name')!r}"] += 1
        else:
            agree[e.kind]["no text name"] += 1
    matched = sum(c["matched"] for c in per_kind.values())
    return {
        "status_total": dict(sum(per_kind.values(), Counter()).most_common()),
        "per_kind": {k: dict(c.most_common()) for k, c in sorted(per_kind.items(), key=lambda kv: -sum(kv[1].values()))},
        "name_agreement": {k: dict(c) for k, c in sorted(agree.items(), key=lambda kv: -sum(kv[1].values()))},
        "disagreements": dict(disagree.most_common()),
        "forms": dict(forms.most_common()),
        "forms_rows": sum(forms.values()),
        "matched_rows": matched,
        "reasons": dict(reasons.most_common()),
        "record_keys": dict(keys.most_common()),
    }


def last_decision(path: Path) -> Path | None:
    files = sorted((path / "decisions").glob("*.json")) if (path / "decisions").is_dir() else []
    return files[-1] if files else None


def today_parse(row_id: str, text: str):
    m = re.match(r"T(\d+)#(\d+)", row_id or "")
    turn = int(m[1]) if m else 0
    return parse_row(rows_from_texts(turn, [text])[0])


def reason_class(reason: str) -> str:
    reason = re.sub(r"'[^']*'|\"[^\"]*\"", "'X'", reason)
    return reason.split(":")[0] if reason.startswith("unsupported mechanic") else reason


def measure_recorded(dirs: list[Path], today_rows: list[dict]) -> dict:
    """The last decision record's unmapped list per battle vs today's parse of the same texts."""
    recorded: Counter = Counter()
    classes: Counter = Counter()
    handled, still = [], Counter()
    battles = with_unmapped = entries = 0
    for path in dirs:
        last = last_decision(path)
        if last is None:
            continue
        battles += 1
        unmapped = json.loads(last.read_text(encoding="utf-8")).get("unmapped") or []
        with_unmapped += bool(unmapped)
        for u in unmapped:
            entries += 1
            reason = u.get("reason", "")
            recorded[reason] += 1
            origin = "parser" if PARSER_REASON.match(reason) else "translator"
            classes[f"{origin}: {reason_class(reason)}"] += 1
            if origin != "parser":
                continue
            e = today_parse(u.get("row", ""), u.get("text", ""))
            if e.kind == "unknown":
                still[e.data.get("reason", "")] += 1
            else:
                handled.append({"battle": f"{path.parent.name}/{path.name}", "row": u.get("row"),
                                "text": u.get("text"), "recorded": reason, "today": e.kind,
                                "today_data": {k: subtype_value(v) for k, v in e.data.items() if present(v)}})
    today_unknown = Counter(r["event"].data.get("reason", "") for r in today_rows if r["event"].kind == "unknown")
    return {
        "battles_with_decisions": battles, "battles_with_unmapped": with_unmapped, "entries": entries,
        "recorded_reasons": dict(recorded.most_common()), "reason_classes": dict(classes.most_common()),
        "recorded_no_pattern": recorded.get("no_pattern", 0),
        "today_no_pattern_all_rows": today_unknown.get("no_pattern", 0),
        "today_unknown_all_rows": dict(today_unknown.most_common()),
        "parser_entries_still_unknown_today": dict(still.most_common()),
        "parser_entries_handled_today": handled,
    }


def totals(dirs: list[Path], with_log: list[Path], rows: list[dict]) -> dict:
    turns = {(r["battle"], r["turn"]) for r in rows}
    per_battle = Counter(r["battle"] for r in rows)
    return {
        "runs": len({p.parent.name for p in with_log}),
        "runs_dirs": len({p.parent.name for p in dirs}),
        "battle_dirs": len(dirs),
        "battles": len(with_log),
        "battles_without_log": [f"{p.parent.name}/{p.name}" for p in dirs if p not in with_log],
        "rows": len(rows),
        "turns": len(turns),
        "max_turn": max((r["turn"] for r in rows), default=0),
        "rows_per_battle_max": max(per_battle.values(), default=0),
        "kinds": len({r["event"].kind for r in rows}),
    }


def unknown_rows(rows: list[dict]) -> list[dict]:
    return [{"battle": r["battle"], "turn": r["turn"], "index": r["index"], "text": r["text"],
             "reason": r["event"].data.get("reason")} for r in rows if r["event"].kind == "unknown"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=PCSA / "output" / "runs")
    ap.add_argument("--md", type=Path, default=ROOT / "docs" / "battle-log.md")
    ap.add_argument("--json", type=Path, default=ROOT / "docs" / "investigation" / "catalog" / "battle_log.json")
    args = ap.parse_args()
    dirs = battle_dirs(args.runs)
    with_log = [p for p in dirs if (p / "log.jsonl").is_file()]
    rows = []
    for i, path in enumerate(with_log, 1):
        rows += load_battle(path)
        if i % 50 == 0:
            print(f"parsed {i}/{len(with_log)} battles, {len(rows)} rows", flush=True)
    data = {
        "command": COMMAND,
        "totals": totals(dirs, with_log, rows),
        "kinds": measure_kinds(rows),
        "unknown_rows": unknown_rows(rows),
        "sprites": measure_sprites(rows),
        "never_shown": never_shown(rows),
        "recorded_vs_today": measure_recorded(dirs, rows),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    args.md.write_text(render(data), encoding="utf-8")
    t = data["totals"]
    print(f"done: {t['battles']} battles, {t['rows']} rows, {t['kinds']} kinds -> {args.md} and {args.json}")


if __name__ == "__main__":
    main()

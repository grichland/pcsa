"""Count each opposing species' bring and lead rate at team preview from recorded live battles, and write
assets/sets/opponent_preview_rates.json (search worlds read it with cg.search.preview_rates).

    uv run python tools/preview_rates.py [--runs output/runs] [--weight 2] [--dry-run]

rate = (times + prior * weight) / (seen + weight), with the priors 0.5 (brought) and 1/6 (lead).
- Champions Gym battles are read from <run>/cg/<battle>.jsonl: the preview and the order their Pokémon entered.
- Older battles are read from <run>/battle-N/showdown.log.
A battle's lead and the Pokémon that entered count as brought. With three seen, the other three count as left
behind; with fewer, they are not counted. The benchmark's clean battles are left out, as in tools/seen_builds.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from seen_builds import ROOT, benchmark_battles, read_jsonl, table, to_id

BROUGHT = 3
PREVIEW = 6


def from_cg(path: Path) -> tuple[list[str], list[str]] | None:
    """(their six by preview slot, the ones that entered in order) from a battle's CG messages."""
    messages = read_jsonl(path)
    start = next((m for m in messages if m.get("type") == "battle_start"), None)
    if start is None:
        return None
    entered: list[str] = []
    for m in messages:
        for e in m.get("events") or [] if m.get("type") == "events" else []:
            if isinstance(e, dict) and e.get("k") == "enter" and e.get("side") == "opp":
                mon = to_id(e.get("mon"))
                if mon and mon not in entered:
                    entered.append(mon)
    return [to_id(s) for s in start.get("opp_preview") or []], entered


def from_showdown(path: Path) -> tuple[list[str], list[str]] | None:
    """The same from an older battle's showdown.log (the side named 'opponent')."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    side = next((ln.split("|")[2] for ln in lines if ln.startswith("|player|") and ln.split("|")[3:4] == ["opponent"]),
                "p2")  # fmt: skip
    preview, entered = [], []
    for ln in lines:
        parts = ln.split("|")
        if len(parts) > 3 and parts[1] == "poke" and parts[2] == side:  # noqa: PLR2004
            preview.append(to_id(parts[3].split(",")[0]))
        elif len(parts) > 3 and parts[1] in ("switch", "drag") and parts[2].startswith(side):  # noqa: PLR2004
            mon = to_id(parts[3].split(",")[0])
            if mon and mon not in entered:
                entered.append(mon)
    return (preview, entered) if preview else None


def slot_of(mon: str, preview: list[str], base: dict[str, str]) -> int | None:
    """The preview slot of an entered form: itself, else the one preview species with its base."""
    if mon in preview:
        return preview.index(mon)
    same = [i for i, p in enumerate(preview) if base.get(p, p) == base.get(mon, mon)]
    return same[0] if len(same) == 1 else None


def battles(runs: Path) -> dict[str, tuple[list[str], list[str]]]:
    out: dict[str, tuple[list[str], list[str]]] = {}
    for f in sorted(runs.glob("*/cg/*.jsonl")):
        if not f.name.endswith(".neural.jsonl") and f.stem not in out and (got := from_cg(f)):
            out[f.stem] = got
    for rec in sorted(runs.glob("*/battle-*/battle.json")):
        log = rec.with_name("showdown.log")
        try:
            bid = json.loads(rec.read_text(encoding="utf-8")).get("battle_id")
        except ValueError:
            continue
        if bid and bid not in out and log.is_file() and (got := from_showdown(log)):
            out[bid] = got
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "output" / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "sets" / "opponent_preview_rates.json")
    ap.add_argument("--weight", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    base = {sid: to_id(s.get("baseSpecies") or sid) for sid, s in table("species").items()}
    skip = benchmark_battles()
    found = battles(args.runs)
    c: dict[str, dict] = defaultdict(lambda: {"previews": 0, "brought_seen": 0, "brought": 0, "lead_seen": 0, "lead": 0})
    used: list[str] = []
    report = {"found": len(found), "left_out_benchmark": 0, "bad_preview": 0, "unplaced": 0}
    for bid, (preview, entered) in sorted(found.items()):
        if bid in skip:
            report["left_out_benchmark"] += 1
            continue
        if len(preview) != PREVIEW:
            report["bad_preview"] += 1
            continue
        slots = [slot_of(m, preview, base) for m in entered]
        if any(s is None for s in slots):
            report["unplaced"] += 1
        seen = {s for s in slots if s is not None}
        if len(seen) > BROUGHT:
            report["bad_preview"] += 1
            continue
        used.append(bid)
        lead = slots[0] if slots else None
        for j, name in enumerate(preview):
            e = c[name]
            e["previews"] += 1
            if j in seen or len(seen) == BROUGHT:
                e["brought_seen"] += 1
                e["brought"] += int(j in seen)
            if lead is not None:
                e["lead_seen"] += 1
                e["lead"] += int(lead == j)
    w = args.weight
    species = {s: {**e, "brought_rate": round((e["brought"] + 0.5 * w) / (e["brought_seen"] + w), 4),
                   "lead_rate": round((e["lead"] + w / 6) / (e["lead_seen"] + w), 4)} for s, e in sorted(c.items())}  # fmt: skip
    doc = {"what": "opposing species' bring and lead rates at team preview, from recorded live battles",
           "rule": "rate = (times + prior * weight) / (seen + weight); priors brought 0.5, lead 1/6; species = CG ids",
           "weight": w, "battles": len(used), "first": used[0] if used else None, "last": used[-1] if used else None,
           "default": {"brought_rate": 0.5, "lead_rate": round(1 / 6, 4)}, "species": species}  # fmt: skip
    print(json.dumps({**report, **{k: doc[k] for k in ("battles", "first", "last")}, "species": len(species)}))
    if not args.dry_run:
        args.out.write_bytes((json.dumps(doc, indent=1) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

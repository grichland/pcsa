"""Learn the opposing builds that recorded Champions Gym battles revealed into assets/sets/seen_builds.json.

    uv run python tools/seen_builds.py [--runs output/runs] [--dry-run] [--report output/seen-builds-update.json]

Each battle not yet in the file's `battles` is read from its CG messages (<run>/cg/<battle>.jsonl), oldest first.
Per opposing Pokémon the revealed moves, item (one an event named, also after it is gone; a Mega's stone) and
ability (not the Mega's own) are matched as the old pcsa.battle.seen_builds did:
1. a seen build that covers it is seen again; 2. else a top build that covers it is added as "catalog";
3. else an added build it is consistent with learns the new parts; 4. else a new "added" build: the revealed
parts, the rest from the closest top build of the species.
Every battle pcsa-benchmark uses (../archive/pcsa-benchmark/*/clean_battles.txt and battles.json) is left out, as
in the team pool.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from pcsa.network.builds import Dex
from pcsa.network.evidence import STATS, Tracker

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "assets" / "sets"
GAME = ROOT / "assets" / "game" / "cg"
BENCH = ROOT.parent / "archive" / "pcsa-benchmark"
SCHEMA = "pcsa-seen-builds/1"
NOTE = "opposing builds seen on the live ladder (tools/seen_builds.py); seen builds rank first"
MOVE_SLOTS = 4
FALLBACK_NATURE = "Hardy"
FLAT_POINTS = 11
NOT_MOVES = frozenset({"struggle", "recharge"})


def to_id(s: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def table(name: str) -> dict:
    return json.loads((GAME / f"{name}.json").read_text(encoding="utf-8"))[name]


class Names:
    """CG ids -> display names, and the species facts the matching needs."""

    def __init__(self) -> None:
        self.species = table("species")
        self.moves = {k: v["name"] for k, v in table("moves").items()}
        self.items = {k: v["name"] for k, v in table("items").items()}
        self.abilities = {k: v["name"] for k, v in table("abilities").items()}
        self.natures = {n["id"]: n["name"] for n in table("natures")}

    def base(self, sid: str) -> str:
        return to_id((self.species.get(sid) or {}).get("baseSpecies") or sid)

    def legal(self, sid: str) -> set[str]:
        return {to_id(a) for a in ((self.species.get(sid) or {}).get("abilities") or {}).values()}

    def name(self, sid: str) -> str:
        return (self.species.get(sid) or {}).get("name") or sid

    def is_mega(self, sid: str) -> bool:
        return str((self.species.get(sid) or {}).get("forme") or "").startswith("Mega")


@dataclass
class Observed:
    species: str  # the key builds are kept under
    moves: tuple[str, ...]
    item: str | None
    ability: str | None

    def empty(self) -> bool:
        return not (self.moves or self.item or self.ability)


def covers(b: dict, o: Observed) -> bool:
    if not set(o.moves) <= {to_id(m) for m in b["moves"]}:
        return False
    if o.item is not None and to_id(b["item"]) != o.item:
        return False
    return o.ability is None or to_id(b["ability"]) == o.ability


def fits_added(e: dict, o: Observed) -> bool:
    if o.item is not None and e.get("known_item") and to_id(e["item"]) != o.item:
        return False
    if o.ability is not None and e.get("known_ability") and to_id(e["ability"]) != o.ability:
        return False
    return len({*e.get("known_moves", ()), *o.moves}) <= MOVE_SLOTS


class Learner:
    def __init__(self, data: dict, top: dict, names: Names) -> None:
        self.data, self.names = data, names
        self.catalog: dict[str, list[dict]] = {}
        for sp, entry in (top.get("species") or {}).items():
            builds = sorted(entry.get("builds") or [], key=lambda b: -float(b.get("score") or 0.0))
            self.catalog[to_id(sp)] = [{**b, "species": sp} for b in builds]

    def ranked(self, key: str) -> list[dict]:
        return sorted(self.data["species"].get(key, []), key=lambda e: (e["last"], e["count"]), reverse=True)

    def observe(self, o: Observed, when: str) -> str:
        if len(o.moves) > MOVE_SLOTS:
            return "skipped: more than 4 moves"
        entries = self.data["species"].setdefault(o.species, [])
        ranked = self.ranked(o.species)
        for e in ranked:
            if covers(e, o):
                self.extend(e, o)
                seen_again(e, when)
                return "seen again" if e["source"] == "catalog" else "added build seen again"
        cover = next((b for b in self.catalog.get(o.species, []) if covers(b, o)), None)
        if cover is not None:
            entries.append({"source": "catalog", "count": 1, "first": when, "last": when, **plain(cover)})
            return "covered by the catalog"
        for e in ranked:
            if e["source"] == "added" and fits_added(e, o):
                self.extend(e, o)
                seen_again(e, when)
                return "added build extended"
        entries.append({"source": "added", "count": 1, "first": when, "last": when, **self.new_build(o),
                        "known_moves": list(o.moves), "known_item": o.item is not None,
                        "known_ability": o.ability is not None})  # fmt: skip
        return "added"

    def extend(self, e: dict, o: Observed) -> None:
        if e["source"] != "added":
            return
        known = tuple(dict.fromkeys([*e.get("known_moves", ()), *o.moves]))
        item_known = bool(e.get("known_item")) or o.item is not None
        ability_known = bool(e.get("known_ability")) or o.ability is not None
        if known == tuple(e.get("known_moves", ())) and (item_known, ability_known) == (
                bool(e.get("known_item")), bool(e.get("known_ability"))):  # fmt: skip
            return
        merged = Observed(o.species, known,
                          o.item if o.item is not None else (to_id(e["item"]) if e.get("known_item") else None),
                          o.ability if o.ability is not None else (to_id(e["ability"]) if e.get("known_ability") else None))  # fmt: skip
        e.update(self.new_build(merged), known_moves=list(known), known_item=item_known, known_ability=ability_known)

    def new_build(self, o: Observed) -> dict:
        """The revealed parts, the rest from the closest top build of the species."""
        builds = self.catalog.get(o.species, [])

        def closeness(b: dict) -> tuple:
            ids = {to_id(m) for m in b["moves"]}
            return (o.item is not None and to_id(b["item"]) == o.item, len(ids & set(o.moves)),
                    o.ability is not None and to_id(b["ability"]) == o.ability)  # fmt: skip

        ordered = sorted(builds, key=closeness, reverse=True)  # stable: ladder score breaks ties
        base = ordered[0] if ordered else None
        moves = [self.names.moves.get(m, m) for m in o.moves]
        for m in (m for b in ordered for m in b["moves"]):
            if len(moves) >= MOVE_SLOTS:
                break
            if to_id(m) not in {to_id(x) for x in moves}:
                moves.append(m)
        n = self.names
        item = n.items.get(o.item, o.item) if o.item is not None else (base["item"] if base else "")
        if o.ability is not None:
            ability = n.abilities.get(o.ability, o.ability)
        elif base and to_id(base["ability"]) in n.legal(o.species):
            ability = base["ability"]
        else:
            own = list(((n.species.get(o.species) or {}).get("abilities") or {}).values())
            ability = own[0] if own else ""
        nature = base["nature"] if base else FALLBACK_NATURE
        points = dict(base["points"]) if base else dict.fromkeys(STATS, FLAT_POINTS)
        return {"species": n.name(o.species), "item": item, "ability": ability, "nature": nature,
                "points": points, "moves": moves}  # fmt: skip


def plain(b: dict) -> dict:
    return {k: b[k] for k in ("species", "item", "ability", "nature", "points", "moves")}


def seen_again(e: dict, when: str) -> None:
    e["count"] += 1
    e["first"] = min(e["first"], when) if e.get("first") else when
    e["last"] = max(e.get("last") or "", when)


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def observed(messages: list[dict], names: Names, dex: Dex) -> list[Observed]:
    """The opposing sets one battle's events revealed, keyed by their preview species."""
    start = next((m for m in messages if m.get("type") == "battle_start"), {})
    preview = [to_id(s) for s in start.get("opp_preview") or []]
    tracker = Tracker(dex, own=start.get("own_team") or [])
    moves: dict[str, list[str]] = {}
    ability: dict[str, str] = {}
    mega: dict[str, str] = {}
    for m in messages:
        if m.get("type") != "events":
            continue
        tracker.feed(m["events"])
        for e in m["events"]:
            if not isinstance(e, dict) or e.get("side") != "opp":
                continue
            mon = to_id(e.get("mon"))
            if e.get("k") == "enter":
                moves.setdefault(mon, [])
            elif e.get("k") == "move" and not e.get("called_by"):
                mv = to_id(e.get("move"))
                if mv and mv not in NOT_MOVES and mv not in moves.setdefault(mon, []):
                    moves[mon].append(mv)
            elif e.get("k") == "ability" and e.get("ability"):
                ability.setdefault(mon, to_id(e.get("ability")))
            elif e.get("k") == "mega" and e.get("form"):
                mega[mon] = to_id(e.get("form"))
    out = []
    for mon in moves.keys() | ability.keys() | tracker.had_item.keys():
        key = species_key(mon, preview, names)
        form = mega.get(mon) or (mon if names.is_mega(mon) else None)
        item = tracker.had_item.get(mon)
        if form and item is None:
            stone = (names.species.get(form) or {}).get("requiredItem")
            item = to_id(stone) or None
        ab = ability.get(mon)
        if ab and form and ab in names.legal(form) and ab not in names.legal(key):
            ab = None  # the Mega's own ability
        if ab and ab not in names.legal(key):
            ab = None
        out.append(Observed(key, tuple(moves.get(mon, ())), item, ab))
    return [o for o in out if not o.empty()]


def species_key(mon: str, preview: list[str], names: Names) -> str:
    """The preview species a battle form belongs to (a Mega or in-battle form keys under its base)."""
    if mon in preview:
        return mon
    same = [p for p in preview if names.base(p) == names.base(mon)]
    if len(same) == 1:
        return same[0]
    return names.base(mon) if names.is_mega(mon) or mon not in names.species else mon


def benchmark_battles() -> set[str]:
    """Every battle pcsa-benchmark uses: its clean battles and the battles its items come from."""
    clean, lists = sorted(BENCH.glob("*/clean_battles.txt")), sorted(BENCH.glob("*/battles.json"))
    if not clean or not lists:
        raise SystemExit(f"pcsa-benchmark's battle lists are missing under {BENCH}")
    out: set[str] = set()
    for f in clean:
        out |= {x.split()[0] for x in f.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")}
    for f in lists:
        out |= set(json.loads(f.read_text(encoding="utf-8")))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "output" / "runs")
    ap.add_argument("--out", type=Path, default=SETS / "seen_builds.json")
    ap.add_argument("--report", type=Path, default=ROOT / "output" / "seen-builds-update.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    data = json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else {"battles": [], "species": {}}
    if data.get("schema", SCHEMA) != SCHEMA:
        raise SystemExit(f"unsupported schema {data.get('schema')!r}")
    top = json.loads((SETS / "top_builds.json").read_text(encoding="utf-8"))
    names, dex = Names(), Dex.load()
    learner = Learner(data, top, names)
    learned, skip = set(data["battles"]), benchmark_battles()
    files: dict[str, Path] = {}
    for f in sorted(args.runs.glob("*/cg/*.jsonl")):
        if not f.name.endswith(".neural.jsonl"):
            files.setdefault(f.stem, f)
    report: dict = {"battles": 0, "left_out_benchmark": 0, "already": 0, "no_start": 0, "results": {}, "rows": []}
    for bid in sorted(files):
        if bid in learned:
            report["already"] += 1
            continue
        if bid in skip:
            report["left_out_benchmark"] += 1
            continue
        messages = read_jsonl(files[bid])
        if not any(m.get("type") == "battle_start" for m in messages):
            report["no_start"] += 1
            continue
        when = f"{bid[:4]}-{bid[4:6]}-{bid[6:8]}"
        for o in observed(messages, names, dex):
            result = learner.observe(o, when)
            report["results"][result] = report["results"].get(result, 0) + 1
            report["rows"].append({"battle": bid, "species": o.species, "moves": list(o.moves), "item": o.item,
                                   "ability": o.ability, "result": result})  # fmt: skip
        data["battles"].append(bid)
        report["battles"] += 1
    data["schema"], data["note"] = SCHEMA, NOTE
    data["species"] = {k: learner.ranked(k) for k in sorted(data["species"])}
    report["species"] = len(data["species"])
    report["builds"] = sum(len(v) for v in data["species"].values())
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    if not args.dry_run:
        text = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
        args.out.write_bytes(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

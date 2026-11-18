"""Compare the STOPGAP dump with CG's own values in generated.rs and print every difference (read-only)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cg_tables
from cg_tables import say

CG_DIR = Path(__file__).resolve().parents[2] / "assets" / "game" / "cg"


def table(kind: str):
    return json.loads((CG_DIR / f"{kind}.json").read_text(encoding="utf-8"))[kind]


def move_diffs(src: str) -> list[str]:
    ours, theirs = table("moves"), cg_tables.move_values(src)
    out = []
    for key, cg in theirs.items():
        mine = dict(ours[key])
        mine["basePower"] = mine["basePower"] or 0
        for field, value in cg.items():
            if mine[field] != value:
                out.append(f"move {key} {field}: stopgap {mine[field]!r} CG {value!r}")
    return out


def species_diffs(src: str) -> list[str]:
    ours = table("species")
    out = []
    for line in cg_tables.rows(src, "SPECIES")[1]:
        key = cg_tables.field(line, "id")
        stats = [int(x) for x in re.search(r"base_stats: \[([^\]]*)\]", line).group(1).split(",")]
        types = [
            t.removeprefix("Type::").lower() for t in re.search(r"types: \[([^\]]*)\]", line).group(1).split(", ")
        ]
        e = ours[key]
        mine = [e["baseStats"][s] for s in ("hp", "atk", "def", "spa", "spd", "spe")]
        if mine != stats:
            out.append(f"species {key} baseStats: stopgap {mine} CG {stats}")
        if [t.lower() for t in e["types"]] != [t for t in types if t != "typeless"]:
            out.append(f"species {key} types: stopgap {e['types']} CG {types}")
        if e["name"] != cg_tables.field(line, "name"):
            out.append(f"species {key} name: stopgap {e['name']!r} CG {cg_tables.field(line, 'name')!r}")
    return out


def chart_diffs(src: str) -> list[str]:
    ours, theirs = table("typechart"), cg_tables.type_chart(src)
    return [
        f"type {a}->{d}: stopgap {ours[a][d]} CG {v}"
        for a, row in theirs.items()
        for d, v in row.items()
        if ours[a][d] != v
    ]


def main() -> int:
    src = cg_tables.source()
    diffs = move_diffs(src) + species_diffs(src) + chart_diffs(src)
    names = {k: v["name"] for k, v in table("moves").items()}
    diffs += [
        f"move {k} name: stopgap {names[k]!r} CG {v!r}"
        for k, v in cg_tables.legal(src)["moves"].items()
        if names[k] != v
    ]
    for line in diffs:
        say(line)
    say(f"{len(diffs)} differences")
    return 0


if __name__ == "__main__":
    sys.exit(main())

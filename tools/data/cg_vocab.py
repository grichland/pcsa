"""Write assets/game/cg/vocab.json: CG's numeric embedding ids -> CG string ids, from CG's generated.rs.

    .venv/Scripts/python.exe tools/data/cg_vocab.py [--source CG_CHECKOUT]

The Champions Gym checkout is --source, else the environment variable CG_SOURCE, else ../cg-publish (the public
clone); its engine/src/sim/data/generated.rs is read (cg_tables.generated_rs).

Numbering (CG obs/src/vocab.rs, contracts/MODEL.md section 2), every list indexed by the id:
- species: id = table index + 1; id 0 is unknown (null)
- moves: id = table index; index 0 is the empty entry (null)
- items: id = table index + 1; id 0 is unknown (null), id 1 is the empty entry: "holds nothing" ("")
- abilities: id = table index; index 0 is the empty entry (null)
The table's empty placeholder is kept here, unlike cg_tables.rows(), so the indices stay CG's.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cg_tables

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "assets" / "game" / "cg" / "vocab.json"
TABLES = {"species": "SPECIES", "moves": "MOVES", "items": "ITEMS", "abilities": "ABILITIES"}


def table(src: str, name: str) -> list[str]:
    """Every entry's id in table order, the empty placeholder included; checked against the declared length."""
    m = re.search(rf"pub static {name}: \[\w+; (\d+)\] = \[\n(.*?)\n\];", src, re.S)
    if m is None:
        raise SystemExit(f"table {name} not found")
    ids = re.findall(r'^\s*\w+Data \{ id: "([^"]*)"', m.group(2), re.M)
    if len(ids) != int(m.group(1)):
        raise SystemExit(f"{name}: {len(ids)} entries read, {m.group(1)} declared")
    return ids


def vocab(src: str) -> dict[str, list[str | None]]:
    species, moves, items, abilities = (table(src, TABLES[k]) for k in ("species", "moves", "items", "abilities"))
    if moves[0] or items[0] or abilities[0] or "" in species:
        raise SystemExit("the empty placeholders are not where vocab.rs expects them")
    return {
        "species": [None, *species],
        "moves": [None, *moves[1:]],
        "items": [None, "", *items[1:]],
        "abilities": [None, *abilities[1:]],
    }


def cg_version() -> str | None:
    try:
        import champions_gym

        return champions_gym.__version__
    except ImportError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--source", type=Path, default=None, help="a Champions Gym checkout (default: CG_SOURCE, else ../cg-publish)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    src = cg_tables.source(cg_tables.generated_rs(args.source))
    commit = re.search(r'pub const SHOWDOWN_COMMIT: &str = "([0-9a-f]+)";', src)
    out = {
        "schema": "pcsa-cg-vocab/1",
        "cg_version": cg_version(),
        "showdown_commit": commit.group(1) if commit else None,
        "numbering": "species, items: table index + 1 (items: 1 = holds nothing); moves, abilities: table index; "
        "0 = unknown everywhere",
        **vocab(src),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=0, ensure_ascii=False) + "\n", encoding="utf-8")
    sizes = {k: len(out[k]) for k in TABLES}
    sys.stdout.write(f"wrote {args.out}: {sizes}, CG {out['cg_version']}, Showdown {out['showdown_commit']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

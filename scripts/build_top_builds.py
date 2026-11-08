"""Build assets/sets/top_builds.json from a usage-ranked build list (a ranked_builds.json).

    uv run python scripts/build_top_builds.py <ranked_builds.json> [--keep 8]

Every build is keyed by the species the game shows on team preview (the base species for Mega builds,
holding its stone). A Mega build that names the Mega ability gets the base species' standard ability
and is tagged `assumed_ability`. Builds are deduplicated and the `keep` highest-scoring ones per
species are written, with the species' ladder usage count for team sampling.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from poke_env.data.normalize import to_id_str

ROOT = Path(__file__).resolve().parents[1]
STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def species_table() -> dict[str, dict]:
    return json.loads((ROOT / "assets/game/species.json").read_text(encoding="utf-8"))["species"]


def as_preview_species(build: dict, dex: dict[str, dict]) -> dict:
    """Mega-form builds become the base species; the ability must be one the base enters with."""
    entry = dex.get(to_id_str(build["species"]), {})
    if "Mega" not in entry.get("forme", ""):
        return build
    base = dex[to_id_str(entry["baseSpecies"])]
    native = {to_id_str(a): a for a in base["abilities"].values()}
    out = {**build, "species": base["name"]}
    if to_id_str(build["ability"]) not in native:
        out["ability"] = base["abilities"]["0"]
        out["assumed_ability"] = True
    return out


def identity(build: dict) -> tuple:
    return (
        to_id_str(build["item"]),
        to_id_str(build["ability"]),
        to_id_str(build["nature"]),
        tuple(build["sp"][s] for s in STATS),
        tuple(sorted(to_id_str(m) for m in build["moves"])),
    )


def compact(build: dict) -> dict:
    out = {
        "item": build["item"],
        "ability": build["ability"],
        "nature": build["nature"],
        "points": {s: build["sp"][s] for s in STATS},
        "moves": list(build["moves"]),
        "score": round(build["log10_score"], 3),
        "sources": sorted({s["site"] for s in build["sources"]}),
    }
    if build.get("assumed_ability"):
        out["assumed_ability"] = True
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranked", type=Path)
    parser.add_argument("--keep", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "assets/sets/top_builds.json")
    args = parser.parse_args(argv)
    ranked = json.loads(args.ranked.read_text(encoding="utf-8"))
    dex = species_table()
    per_species: dict[str, dict] = {}
    for raw in ranked["ranked"]:
        build = as_preview_species(raw, dex)
        bucket = per_species.setdefault(build["species"], {"usage": 0, "builds": {}})
        bucket["usage"] = max(bucket["usage"], raw["species_usage"])
        key = identity(build)
        kept = bucket["builds"].get(key)
        if kept is None or build["log10_score"] > kept["log10_score"]:
            bucket["builds"][key] = {**build, "sources": build["sources"] + (kept["sources"] if kept else [])}
        else:
            kept["sources"] = kept["sources"] + build["sources"]
    species = {}
    for name in sorted(per_species):
        builds = sorted(per_species[name]["builds"].values(), key=lambda b: -b["log10_score"])[: args.keep]
        species[name] = {"usage": per_species[name]["usage"], "builds": [compact(b) for b in builds]}
    out = {
        "schema": "pcsa-top-builds/1",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": args.ranked.name,  # the file name only: no machine paths in the tracked catalog
        "ladder_battles": ranked["ladder_battles"],
        "score_note": "log10 of the product of the species' ladder shares of moves, item, ability and spread",
        "species": species,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(species)} species, {sum(len(v['builds']) for v in species.values())} builds -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

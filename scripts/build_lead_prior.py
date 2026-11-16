"""Build assets/sets/lead_prior.json: how often each species led when it was in an opposing six on the live ladder.
    uv run --no-sync python scripts/build_lead_prior.py [--runs output/runs] [--out assets/sets/lead_prior.json]
Counts per base species over recorded battles; the playout team preview (pcsa.players.preview_sampler) draws their
lead
from it. Rerun after live sessions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pcsa.players.preview_sampler import species_key  # noqa: E402

THEIR_SWITCH = re.compile(r"\|(?:switch|drag)\|p2a: [^|]*\|([^,|]+)")


def their_lead(battle_dir: Path) -> str | None:
    records = sorted((battle_dir / "decisions").glob("*.json"))
    for path in reversed(records):
        try:
            protocol = json.loads(path.read_text(encoding="utf-8")).get("protocol") or []
        except (OSError, ValueError):
            continue
        for line in protocol:
            found = THEIR_SWITCH.match(line)
            if found:
                return found[1]
    return None


def count(runs: Path) -> tuple[Counter, Counter, int]:
    lead, six, games = Counter(), Counter(), 0
    for path in sorted(runs.glob("*/battle-*/battle.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            team = [slot["species"][0][0] for slot in data.get("opponent_preview") or [] if slot.get("species")]
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
        first = their_lead(path.parent)
        keys = {species_key(name) for name in team}
        if len(team) != 6 or first is None or species_key(first) not in keys:  # noqa: PLR2004
            continue
        games += 1
        six.update(keys)
        lead[species_key(first)] += 1
    return lead, six, games


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default=str(REPO / "output" / "runs"))
    parser.add_argument("--out", default=str(REPO / "assets" / "sets" / "lead_prior.json"))
    args = parser.parse_args()
    lead, six, games = count(Path(args.runs))
    species = {key: {"lead": lead[key], "six": six[key]} for key in sorted(six)}
    out = {"built": datetime.now().isoformat(timespec="seconds"), "games": games, "species": species}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    sys.stdout.write(f"{args.out}: {games} games, {len(species)} species\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""`pcsa sets learn`: the opposing builds of recorded live battles (output/runs/<stamp>-run/battle-N, from each
battle's showdown.log and battle_final.txt) learned into the seen builds (pcsa.battle.seen_builds)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from poke_env.data.normalize import to_id_str

from pcsa.battle.builds import Catalog
from pcsa.battle.seen_builds import SeenBuilds, observed_from_battle, shared

THEIRS_ROW = re.compile(r"^\s+(\S+)\s+(?:ACTIVE|bench|fainted)\s+HP.*?ability (\S+)\s+item (\S+)\s+moves (.*)$")


def abilities_from_final_text(text: str) -> dict[str, str]:
    """species id -> revealed ability, from a recorded battle_final.txt (its THEIRS rows)."""
    out: dict[str, str] = {}
    if "THEIRS" not in text:
        return out
    for line in text.split("THEIRS", 1)[1].splitlines()[1:]:
        m = THEIRS_ROW.match(line)
        if m and m.group(2) != "?":
            out[to_id_str(m.group(1))] = m.group(2)
    return out


def learn_runs(runs: Path, store: SeenBuilds | None = None, *, dry_run: bool = False) -> list[dict]:
    """Learn every recorded live battle under `runs` (output/runs/<stamp>-run/battle-N with showdown.log)."""
    from pcsa.battle.state import replay

    store = store or shared()
    catalog = Catalog.load()
    rows: list[dict] = []
    for bdir in sorted(runs.glob("*/battle-*")):
        summary, log = bdir / "battle.json", bdir / "showdown.log"
        if not summary.is_file() or not log.is_file():
            continue
        battle_id = json.loads(summary.read_text(encoding="utf-8")).get("battle_id") or str(bdir)
        lines = [x for x in log.read_text(encoding="utf-8").splitlines() if not x.startswith(("|win|", "|tie"))]
        try:
            battle = replay(lines, None, tag=battle_id)
        except Exception as error:  # one unreadable battle never stops the rest
            rows.append({"battle": battle_id, "result": "unreadable", "why": f"{type(error).__name__}: {error}"})
            continue
        final = bdir / "battle_final.txt"
        by_species = abilities_from_final_text(final.read_text(encoding="utf-8")) if final.is_file() else {}
        abilities = {
            k: by_species[to_id_str(m.species)]
            for k, m in battle.opponent_team.items()
            if to_id_str(m.species) in by_species
        }
        stamp = re.match(r"(\d{4})(\d{2})(\d{2})", battle_id)
        when = "-".join(stamp.groups()) if stamp else None
        report = store.learn(battle_id, observed_from_battle(battle, abilities, catalog), catalog, when)
        rows += [{"battle": battle_id, **r} for r in report]
    if not dry_run:
        store.save()
    return rows

"""The battle pack of pcsa-former (the training repository) with their Pokémon named by form (Rotom Wash, Alolan
Ninetales).

    cd <pcsa-former>
    <pcsa>/.venv/Scripts/python <pcsa>/tools/forms_pack.py runs/<pack>.jsonl.gz

Runs pcsa-former's own pack step (tools/live_playon.py --pack) from its folder, with every battle's recorded
messages passed through pcsa.events.forms.with_forms first, so a recording that names a Pokémon by its base species
gets its form. The preview names come from the battle's battle.json. Nothing in pcsa-former is changed.
"""

import collections
import json
import sys
from pathlib import Path

HERE = Path.cwd()  # pcsa-former
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "tools"))
sys.path.insert(0, str(ROOT / "champions-gym" / "tools" / "phone"))
import live_playon as P  # noqa: E402
import live_rows as LR  # noqa: E402

from pcsa.events.forms import forms_of, with_forms  # noqa: E402

RUN_ROOTS = [Path(r) for r in LR.RUNS]
stats: collections.Counter = collections.Counter()
kinds: collections.Counter = collections.Counter()


def names_of(name: str) -> list:
    run, _, folder = name.partition("__")
    for root in RUN_ROOTS:
        path = root / run / folder / "battle.json"
        if path.is_file():
            slots = json.loads(path.read_text(encoding="utf-8")).get("opponent_preview") or []
            return [((s.get("top") or s.get("species") or [[None]])[0] or [None])[0] if isinstance(s, dict) else s for s in slots]
    stats["no battle.json"] += 1
    return []


def formed(battles):
    for name, msgs in battles:
        names = names_of(name)
        found = forms_of(msgs, names)
        stats["battles"] += 1
        if found:
            stats["battles with a form"] += 1
            kinds.update(found.values())
        yield name, with_forms(msgs, names)


real_stream, real_runs = LR.stream_battles, LR.run_battles
LR.stream_battles = lambda path: formed(real_stream(path))
LR.run_battles = lambda root: formed(real_runs(root))
P.pack(Path(sys.argv[1]))
print(dict(stats))
print(kinds.most_common())

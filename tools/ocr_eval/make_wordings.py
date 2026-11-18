"""Build assets/game/log_wordings.json: the clean phone-log wordings from CG's inventory (read-only source).

Damaged readings in the inventory are left out; side words fold into the {MON} hole; fixed names become holes.
Run: python -m tools.ocr_eval.make_wordings
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tools.cg_events.templates import load

from pcsa.battle.log_patterns import TERRAIN, WEATHER

OUT = Path(__file__).resolve().parents[2] / "assets" / "game" / "log_wordings.json"
# Inventory rows that are misreadings of another wording (dropped "!", "recoill", damaged trainer slot, sign lost).
DAMAGED = {92, 132, 145, 178, 185, 203, 206, 210, 215, 219, 220, 223, 238, 239, 240, 241, 250, 254, 269, 271, 274}
DAMAGED |= {280, 281, 284, 301, 306, 12, 46, 76, 88, 99}
ELLIPSIS = ("It's not very effective", "It's mostly ineffective", "It doesn't affect {MON}")
CATEGORIES = {"Move +{N} Priority": "Move", "Move -{N} Priority": "Move"}
FIXED = [
    ("PL-197", "Ability Changed", "{MON}: {ABILITY} → {ABILITY}"),
    ("EX-001", "-", "{MON} transformed into the {TYPE} type!"),
    ("EX-002", "-", "{MON} switched items with its target!"),  # Trick, seen in a battle with no saved log
    ("EX-003", "-", "{MON} obtained a {ITEM}."),
    ("EX-005", "-", "{MON} obtained a serving of {ITEM}."),
    ("EX-006", "-", "{MON} is frozen solid!"),
    ("EX-004", "-", "{TRAINER} sent out {TRAINER}'s {MON}!"),  # a team borrowed from another player
]


def generalise(text: str) -> str:
    text = re.sub(r"\b[Tt]he opposing \{MON\}", "{MON}", text)
    text = re.sub(r"(?:[Tt]he opposing side|[Yy]our side)(?= stronger|!)", "{SIDE}", text)
    text = re.sub(r"^(?:The opposing side|Your side)'s", "{SIDE}'s", text)
    text = re.sub(r"used [A-Z][A-Za-z ]+!$", "used {MOVE}!", text)
    text = re.sub(r"transformed into the [A-Z][a-z]+ type!$", "transformed into the {TYPE} type!", text)
    for stem in ELLIPSIS:
        if text.startswith(stem):
            text = stem + "..."
    return text


def build() -> list[dict]:
    seen: dict[tuple[str, str], str] = {}
    for t in load():
        number = int(t.id[3:])
        if t.cls in ("UNREADABLE", "NOT-BATTLE") or number in DAMAGED or t.id == "PL-197":
            continue
        category = CATEGORIES.get(t.category, t.category)
        key = (category, generalise(t.text))
        seen.setdefault(key, t.id)
    extra = [*WEATHER, *(f"{t}{end}" for t in TERRAIN for end in ("!", "."))]
    stems = {re.sub(r"[.!]+$", "", text) for _, text in seen}
    for n, text in enumerate(extra, 1):
        if text in WEATHER or re.sub(r"[.!]+$", "", text) not in stems:
            seen.setdefault(("-", text), f"SD-{n:03d}")
    for ident, category, text in FIXED:
        seen.setdefault((category, text), ident)
    return [{"id": i, "category": c, "text": x} for (c, x), i in sorted(seen.items(), key=lambda kv: kv[1])]


def main() -> None:
    rows = build()
    OUT.write_text(json.dumps({"source": "CG phone-log inventory, cleaned", "wordings": rows}, indent=1), "utf-8")
    print(len(rows), "wordings ->", OUT)


if __name__ == "__main__":
    main()

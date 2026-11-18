"""D1 'What the game never shows': each item backed by regex searches over every recorded row text."""

from __future__ import annotations

import re
from collections import Counter

ARROW = r"(?:→|->|—>|=>)"
# Short reading notes shown under each item.
NOTES = {
    "N4": (
        "The miss row names the target, not the attacker. The attacker is only in an earlier row (the "
        "kind of the row just before is counted below)."
    ),
    "N5": (
        "The hit-count row names no Pokemon; neighbouring rows (HP rows name a Pokemon) may suggest one."
    ),
    "N6": (
        "Move rows name the user and the move only. Some other rows name a target (counted below)."
    ),
    "N7": (
        "HP rows carry name, before, after and the HP Lost / HP Restored label. Causes come as separate "
        "rows."
    ),
    "N9": (
        "HP rows never name a side. pcsa takes the side from the unit. The counts below compare the unit "
        "with which side's send-out rows used the name; disagreements were not examined (OCR misreads, a "
        "species on both sides, Illusion or Transform are possible)."
    ),
    "N10": (
        "A Choice Scarf is shown only by Frisk (ability rows) and Poltergeist (item_reveal rows); no row "
        "matched a lock search."
    ),
    "N11": (
        "'X protected itself!' is shown. Per plan F29, pcsa's translator chooses not to write it as "
        "-activate."
    ),
}


def search(rows: list[dict], label: str, pattern: str, kinds: tuple[str, ...] | None = None) -> dict:
    """Rows whose text matches `pattern` (re.search), optionally only rows of the given parsed kinds."""
    rx = re.compile(pattern)
    hits = [r for r in rows if (kinds is None or r["event"].kind in kinds) and rx.search(r["text"])]
    return {
        "label": label,
        "regex": pattern,
        "scope": "all rows" if kinds is None else "rows parsed as " + ", ".join(kinds),
        "count": len(hits),
        "battles": len({r["battle"] for r in hits}),
        "by_kind": dict(Counter(r["event"].kind for r in hits).most_common()),
        "texts": dict(Counter(r["text"] for r in hits).most_common()),
    }


def previous_kinds(rows: list[dict], kind: str) -> dict:
    """Parsed kind of the row just before each `kind` row in the same battle and turn."""
    out: Counter = Counter()
    for prev, row in zip([None] + rows[:-1], rows):
        if row["event"].kind != kind:
            continue
        same = prev is not None and prev["battle"] == row["battle"] and prev["turn"] == row["turn"]
        out[prev["event"].kind if same else "(first row of the turn)"] += 1
    return dict(out.most_common())


def data_counts(rows: list[dict], kind: str, key: str) -> dict:
    return dict(Counter(str(r["event"].data.get(key)) for r in rows if r["event"].kind == kind).most_common())


def named(rows: list[dict], kind: str) -> dict:
    items = [r for r in rows if r["event"].kind == kind]
    return {"rows": len(items), "with_name": sum(bool(r["event"].name) for r in items)}


def choice_items(rows: list[dict]) -> dict:
    out: Counter = Counter()
    for r in rows:
        item = r["event"].data.get("item")
        if item and "choice" in str(item).lower():
            out[f"{r['event'].kind}: {item}"] += 1
    return dict(out.most_common())


def hp_unit_vs_roster(rows: list[dict]) -> dict:
    """HP rows by unit vs which side's send-out rows (Go!/sent out, come back/withdrew) named that Pokemon."""
    sides: dict[str, dict[str, set]] = {}
    for r in rows:
        e = r["event"]
        if e.kind in ("send_out", "withdraw") and e.side and e.name:
            sides.setdefault(r["battle"], {"p1": set(), "p2": set()})[e.side].add(e.name.lower())
    out: Counter = Counter()
    for r in rows:
        e = r["event"]
        if e.kind != "hp":
            continue
        seen = sides.get(r["battle"], {"p1": set(), "p2": set()})
        name = (e.name or "").lower()
        where = {"p1": "own only", "p2": "theirs only"}
        found = [s for s in ("p1", "p2") if name in seen[s]]
        label = where[found[0]] if len(found) == 1 else ("both sides" if found else "neither")
        out[f"unit {e.data.get('unit')}: name sent out by {label}"] += 1
    return dict(out.most_common())


def _checks_early(rows: list[dict]) -> list[dict]:
    return [
        {"id": "N1", "item": "Critical hit",
         "searches": [search(rows, "'critic' anywhere", r"(?i)critic"),
                      search(rows, "'crit' as a word", r"(?i)\bcrit\b")]},
        {"id": "N2", "item": "PP",
         "searches": [search(rows, "'PP' as a word", r"\bPP\b"),
                      search(rows, "'pp' as a word, any case", r"(?i)\bpp\b")],
         "shown": {"Struggle rows (trailing what=struggle)": data_counts(rows, "trailing", "what").get("struggle", 0),
                   "Pressure announced (ability rows)": search(rows, "pressure", r"is exerting its pressure")["count"]}},
        {"id": "N3", "item": "Sleep counter",
         "searches": [search(rows, "a number on a sleep row",
                             r"(?i)\b(asleep|sleep|woke|slumber)\b.*\d|\d.*\b(asleep|sleep|woke|slumber)\b")],
         "shown_searches": [search(rows, "fell asleep", r"fell asleep"),
                            search(rows, "is fast asleep", r"is fast asleep"),
                            search(rows, "woke up", r"woke up"),
                            search(rows, "grew drowsy", r"grew drowsy")]},
        {"id": "N4", "item": "Who attacked on a miss",
         "searches": [search(rows, "'miss' as a word start", r"(?i)\bmiss")],
         "shown_searches": [search(rows, "avoided the attack", r"avoided the attack")],
         "shown": {"miss rows that carry a name (the target)": named(rows, "miss"),
                   "kind of the row before each miss row": previous_kinds(rows, "miss")}},
        {"id": "N5", "item": "Who was hit on a hit count",
         "searches": [search(rows, "any 'hit N time(s)'", r"(?i)\bhit \d+ times?"),
                      search(rows, "'hit N time(s)' not said of 'The Pokémon'",
                             r"(?i)^(?!.*the pok[eé]mon was hit).*\bhit \d+ times?")],
         "shown_searches": [search(rows, "The Pokémon was hit N time(s)!", r"^The Pokémon was hit \d+ times?!$")],
         "shown": {"hit counts": data_counts(rows, "hitcount", "count"),
                   "kind of the row before each hitcount row": previous_kinds(rows, "hitcount")}},
        {"id": "N6", "item": "A move's target",
         "searches": [search(rows, "move row naming a target", r"(?i) used .+? (on|at|against|toward|into) ",
                             ("move",)),
                      search(rows, "move rows of the form 'X used Y!'", r"^(?:Move[^:]*: )?.+? used .+$", ("move",))],
         "shown": {"immune rows naming the target ('It doesn't affect X')": named(rows, "effectiveness"),
                   "miss rows naming the target": named(rows, "miss"),
                   "knock_off rows (name the target)": named(rows, "knock_off")["rows"]}},
    ]


def _checks_late(rows: list[dict]) -> list[dict]:
    hp = ("hp",)
    return [
        {"id": "N7", "item": "Damage cause on an HP row",
         "searches": [search(rows, "HP row with a cause word", r"(?i)\b(due to|from|because|by|caused)\b", hp),
                      search(rows, "HP row not exactly 'Name's HP: a -> b'",
                             rf"^(?!(?:HP (?:Lost|Restored): )?[^:]+[’']s HP: \d+ *%? *{ARROW} *\d+ *%?$)", hp)],
         "shown": {"cause rows by cause": data_counts(rows, "cause", "cause"),
                   "HP rows by category label (direction)": data_counts(rows, "hp", "direction"),
                   "kind of the row before each hp row": previous_kinds(rows, "hp")}},
        {"id": "N8", "item": "Max HP",
         "searches": [search(rows, "'a / b' numbers", r"\d+ */ *\d+"),
                      search(rows, "'max HP'", r"(?i)max(imum)? HP")],
         "shown": {"HP rows by unit": data_counts(rows, "hp", "unit")}},
        {"id": "N9", "item": "Opponent exact HP",
         "searches": [search(rows, "'opposing' on an HP row", r"(?i)opposing", hp),
                      search(rows, "HP row in exact units (no %)", rf"HP: \d+ *{ARROW} *\d+$", hp)],
         "shown_searches": [search(rows, "HP row in percent", rf"HP: \d+ *% *{ARROW} *\d+ *%$", hp)],
         "shown": {"HP rows: unit vs the side whose send-out rows named the Pokemon": hp_unit_vs_roster(rows)}},
        {"id": "N10", "item": "Choice lock",
         "searches": [search(rows, "'lock'", r"(?i)\block"),
                      search(rows, "'can't use' / 'only use'", r"(?i)(can[’']?t|cannot|only) use")],
         "shown_searches": [search(rows, "'Choice' anywhere", r"(?i)choice")],
         "shown": {"Choice items in parsed item fields": choice_items(rows)}},
        {"id": "N11", "item": "Protect activation (-activate)",
         "searches": [search(rows, "X protected itself!", r"protected itself!")],
         "shown_searches": [search(rows, "'protect' anywhere", r"(?i)protect"),
                            search(rows, "'detect' anywhere", r"(?i)\bdetect")]},
    ]


def never_shown(rows: list[dict]) -> list[dict]:
    out = _checks_early(rows) + _checks_late(rows)
    for c in out:
        c["note"] = NOTES.get(c["id"])
    return out

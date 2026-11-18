"""CG's phone-log templates (PL-001..306) as regexes, so each recorded row can be given its PL id."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

INVENTORY = Path(__file__).resolve().parents[3] / "champions-gym/docs/investigation/findings/phone-log-inventory.md"
HOLES = {"{MON}", "{MOVE}", "{ITEM}", "{ABILITY}", "{TRAINER}", "{N}", "{STATS}", "{TYPE}", "{TITLE}"}


def norm(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s*(?:→|->|—>|=>)\s*", " → ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Template:
    id: str
    rows: int
    battles: int
    category: str
    text: str
    cls: str
    kind: str
    note: str
    regex: re.Pattern
    literal: int  # characters outside holes: more is more specific


def _regex(category: str, text: str) -> tuple[re.Pattern, int]:
    whole = text if category == "-" else f"{category}: {text}"
    parts = re.split(r"(\{[A-Z]+\})", norm(whole))
    out, literal = [], 0
    for i, part in enumerate(parts):
        if part in HOLES:
            out.append(r"\d+" if part == "{N}" else ".*?")
        else:
            lit = re.escape(part)
            if i and parts[i - 1] == "{TRAINER}" and part.startswith(" "):
                lit = " ?" + re.escape(part[1:])  # an unread trainer name leaves no space
            out.append(lit)
            literal += len(part)
    return re.compile("^" + "".join(out) + "$"), literal


def load() -> list[Template]:
    table = INVENTORY.read_text(encoding="utf8").split("## Full template table", 1)[1]
    out = []
    for line in table.splitlines():
        if not line.startswith("| PL-"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split(" | ")]
        regex, literal = _regex(c[3], c[4].strip("`"))
        out.append(Template(c[0], int(c[1]), int(c[2]), c[3], c[4].strip("`"), c[5], c[6], c[7], regex, literal))
    return out


def match(templates: list[Template], text: str) -> Template | None:
    """The most specific template the row fits (most literal text), or None."""
    t = norm(text)
    found = [tp for tp in templates if tp.regex.match(t)]
    return max(found, key=lambda tp: (tp.literal, -int(tp.id[3:]))) if found else None

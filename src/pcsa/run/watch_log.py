"""The battle log of `pcsa watch`, turn by turn, read from a battle folder.

`battle_log.txt` is the standard format (`standard_text()` in pcsa.phone.log_merge): a `# format:
pcsa-battle-log/1` header, `key: value` lines, then `=== Turn N ===` blocks of `[Category] message` lines and
`# controller: ...` annotations (the action we sent). `log.txt` holds the collected rows only (`TurnN:` blocks).
Both are parsed into turns; each line carries the side it is about when the text tells (the game writes "The
opposing ..." for theirs, shows their HP as a percentage and ours as numbers). Any other text is served raw.
"""

from __future__ import annotations

import re
from pathlib import Path

from pcsa.run.watch_decisions import LOGS

FORMAT = "pcsa-battle-log/1"
TURN = re.compile(r"^=== Turn (\d+) ===$")
PLAIN_TURN = re.compile(r"^Turn\s*(\d+):$")
SECTION = re.compile(r"^=== (.+) ===$")
ROW = re.compile(r"^\[(?P<category>[^\]]*)\] (?P<message>.*)$")
HEADER = re.compile(r"^(?P<key>[a-z_]+): (?P<value>.*)$")
OWNER = re.compile(r"^(?P<name>.+?)[’']s ")
CONTROLLER = "# controller: "
OURS, THEIRS = "ours", "theirs"
OUR_CATEGORIES = ("Move", "Fainted", "Mega Evolution")
MAX_BYTES = 400_000


def _names(value: str) -> set[str]:
    """'Kingambit (roster row 5), Golisopod' -> {'kingambit', 'golisopod'}."""
    return {re.sub(r"\s*\(.*?\)", "", part).strip().lower() for part in value.split(",") if part.strip()}


def side_of(category: str, message: str, ours: set[str], theirs: set[str]) -> str | None:
    """Whose line this is: 'ours', 'theirs', or None when the text does not tell (weather, the field)."""
    low = message.lower()
    if "opposing" in low or low.startswith(("sent out ", "withdrew ")) or " sent out " in low:
        return THEIRS
    if low.startswith("go! ") or low.endswith(", come back!"):
        return OURS
    if category.startswith("HP") and "hp:" in low:
        return THEIRS if "%" in message else OURS
    if category.startswith(OUR_CATEGORIES):
        return OURS
    owner = OWNER.match(message)
    name = owner.group("name").lower() if owner else low.split(" ", 1)[0]
    if name in ours and name not in theirs:
        return OURS
    if name in theirs and name not in ours:
        return THEIRS
    return None


def _line(text: str, ours: set[str], theirs: set[str]) -> dict:
    if text.startswith(CONTROLLER):
        return {"kind": "action", "category": "", "text": text[len(CONTROLLER) :], "side": OURS}
    if text.startswith("#"):
        return {"kind": "note", "category": "", "text": text.lstrip("# "), "side": None}
    row = ROW.match(text)
    category = row.group("category").strip() if row else ""
    category = "" if category == "-" else category
    message = row.group("message") if row else text
    return {"kind": "event", "category": category, "text": message, "side": side_of(category, message, ours, theirs)}


def parse(text: str) -> dict:
    """Header, turns and closing notes of a log text; `raw` holds the text when no turn could be read."""
    header: dict[str, str] = {}
    turns: list[dict] = []
    notes: list[str] = []
    ours: set[str] = set()
    theirs: set[str] = set()
    current: list[dict] | None = None
    closing = False
    standard = False
    for raw in text.splitlines():
        line = raw.strip()
        turn = TURN.match(line) or PLAIN_TURN.match(line)
        if turn is not None:
            current, closing = [], False
            turns.append({"turn": int(turn.group(1)), "lines": current})
        elif SECTION.match(line):
            current, closing = None, True
        elif not line:
            continue
        elif closing:
            notes.append(line.lstrip("# "))
        elif current is not None:
            current.append(_line(line, ours, theirs))
        elif line.startswith("#"):
            standard = standard or FORMAT in line
        elif (pair := HEADER.match(line)) is not None:
            header[pair.group("key")] = pair.group("value")
            ours = _names(header.get("team_requested", ""))
            theirs = _names(header.get("opponent_preview", "")) | _names(header.get("opponents_seen", ""))
    kind = "standard" if standard and turns else "plain" if turns else "raw" if text.strip() else None
    return {"format": kind, "header": header, "turns": turns, "notes": notes, "raw": "" if turns else text}


def log_view(battle_dir: Path | None) -> dict:
    """The battle's log as turns: battle_log.txt when it is there, else the rows collected so far (log.txt)."""
    if battle_dir is None:
        return {"source": None, "format": None, "header": {}, "turns": [], "notes": [], "raw": ""}
    for name in LOGS:
        try:
            data = (battle_dir / name).read_bytes()[-MAX_BYTES:]
        except OSError:
            continue
        try:
            return {"source": name, **parse(data.decode("utf-8", errors="replace"))}
        except (ValueError, AttributeError, TypeError):
            return {"source": name, "format": "raw", "header": {}, "turns": [], "notes": [], "raw": ""}
    return {"source": None, "format": None, "header": {}, "turns": [], "notes": [], "raw": ""}

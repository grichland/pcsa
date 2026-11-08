"""What `pcsa watch` shows about the current battle: read from the newest `battle-N/decisions/` record.

Each decision leaves `NNNN-<kind>.json` (schema pcsa-decision/1: request, order, notes) and, for tracked
battles, `NNNN-<kind>.txt` (battle.state.battle_text). The page is read-only, so this module only parses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pcsa.run.watch_field import field_view

ROW = re.compile(
    r"^  (?P<species>\S+)\s+(?P<state>ACTIVE|bench|fainted)\s+HP (?P<hp>\S+)\s.*?"
    r"status (?P<status>\S+)\s+boosts (?P<boosts>.+?)\s+ability (?P<ability>.+?)\s+"
    r"item (?P<item>.+?)\s+moves (?P<moves>.*)$"
)
HEADER = re.compile(r"^TURN (?P<turn>\d+)\s+weather: (?P<weather>.+?)\s+fields: (?P<fields>.+)$")


@dataclass(frozen=True)
class Record:
    path: Path
    data: dict
    text: str


def _battle_dir(run_dir: Path, battle: int | None) -> Path | None:
    if battle is not None:
        return run_dir / f"battle-{battle}"
    battles = sorted(
        (p for p in run_dir.glob("battle-*") if p.name.rsplit("-", 1)[1].isdigit()),
        key=lambda p: int(p.name.rsplit("-", 1)[1]),
    )
    return battles[-1] if battles else None


def latest_record(run_dir: Path, battle: int | None = None) -> Record | None:
    """The newest decision record of a battle folder (default: the newest), or None (also when half written)."""
    battle_dir = _battle_dir(run_dir, battle)
    if battle_dir is None or not (battle_dir / "decisions").is_dir():
        return None
    files = sorted((battle_dir / "decisions").glob("*.json"))
    if not files:
        return None
    path = files[-1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        text = path.with_suffix(".txt").read_text(encoding="utf-8") if path.with_suffix(".txt").is_file() else ""
    except (OSError, json.JSONDecodeError):
        return None
    return Record(path, data, text) if isinstance(data, dict) else None


def record_key(run_dir: Path) -> tuple[str, int] | None:
    """Cheap change detector for the newest record (the writer lands files after the event line)."""
    record = latest_record(run_dir)
    return (str(record.path), record.path.stat().st_mtime_ns) if record else None


def _name(ident: str) -> str:
    return ident.split(": ", 1)[1] if ": " in ident else ident


def moves_of(data: dict) -> list[dict]:
    active = (data.get("request") or {}).get("active") or []
    if not active:
        return []
    return [
        {
            "slot": i,
            "name": m.get("move") or m.get("id") or f"Move {i + 1}",
            "pp": [m["pp"], m["maxpp"]] if "pp" in m and "maxpp" in m else None,
            "disabled": bool(m.get("disabled")),
        }
        for i, m in enumerate(active[0].get("moves") or [])
    ]


def team_of(data: dict) -> list[dict]:
    side = (data.get("request") or {}).get("side") or {}
    return [
        {
            "name": _name(p.get("ident", "")),
            "condition": p.get("condition"),
            "active": bool(p.get("active")),
            "item": p.get("item"),
            "ability": p.get("ability"),
            "moves": list(p.get("moves") or []),
        }
        for p in side.get("pokemon") or []
    ]


def own_sets(export: str | None) -> dict[str, dict]:
    """Display names from the team sheet: {name: {item, ability, moves}} (a nickname or the species)."""
    sets: dict[str, dict] = {}
    for block in re.split(r"\n\s*\n", export or ""):
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        head, _, item = lines[0].partition(" @ ")
        name = re.sub(r"\s*\((?:M|F)\)\s*$", "", head).strip()
        name = re.sub(r"^.*\((.+)\)$", r"\1", name) if "(" in name else name
        ability = next((x.split(":", 1)[1].strip() for x in lines if x.startswith("Ability:")), "")
        moves = [x[1:].strip() for x in lines if x.startswith("-")]
        sets[name] = {"item": item.strip(), "ability": ability, "moves": moves}
    return sets


def your_team(data: dict, moves: list[dict], ours: list[dict]) -> list[dict]:
    """Our picks as the request shows them, with display names from the team sheet and PP for the active."""
    sets = own_sets(data.get("own_export"))
    active_boosts = next((r["boosts"] for r in ours if r["state"] == "ACTIVE"), "-")
    out = []
    for member in team_of(data):
        condition = str(member.get("condition") or "")
        hp, _, status = condition.partition(" ")
        current, _, total = hp.partition("/")
        fraction = int(current) / int(total) if current.isdigit() and total.isdigit() and int(total) else None
        sheet = sets.get(member["name"], {})
        names = sheet.get("moves") or member["moves"]
        pp = {re.sub(r"[^a-z0-9]", "", m["name"].lower()): m["pp"] for m in moves} if member["active"] else {}
        out.append(
            {
                "name": member["name"],
                "active": member["active"],
                "fainted": status == "fnt" or hp == "0",
                "hp": hp,
                "fraction": 0.0 if status == "fnt" else fraction,
                "status": "" if status == "fnt" else status,
                "boosts": active_boosts if member["active"] and active_boosts != "-" else "",
                "item": sheet.get("item") or member.get("item") or "",
                "ability": sheet.get("ability") or member.get("ability") or "",
                "moves": [{"name": m, "pp": pp.get(re.sub(r"[^a-z0-9]", "", m.lower()))} for m in names],
            }
        )
    return out


def rows_of(text: str) -> tuple[dict, list[dict], list[dict], str]:
    """Header fields, our rows, their rows and the 'preview, not yet seen' tail of a battle_text snapshot."""
    header: dict = {}
    ours: list[dict] = []
    theirs: list[dict] = []
    unseen = ""
    side = ours
    for line in text.splitlines():
        if (h := HEADER.match(line)) is not None:
            header = h.groupdict()
        elif line.startswith("THEIRS"):
            side = theirs
        elif line.startswith("  preview, not yet seen:"):
            unseen = line.split(":", 1)[1].strip()
        elif (m := ROW.match(line)) is not None:
            row = m.groupdict()
            row["moves"] = [x.strip() for x in row["moves"].split(",")] if row["moves"] != "?" else []
            side.append(row)
    return header, ours, theirs, unseen


def battle_view(run_dir: Path, battle: dict | None, screen: str | None) -> dict:
    """Everything the page's decision and battle panels read; empty values when there is no record yet."""
    record = latest_record(run_dir, (battle or {}).get("index"))
    if record is None:
        return {}
    data, text = record.data, record.text
    moves, team = moves_of(data), team_of(data)
    header, ours, theirs, unseen = rows_of(text)
    active = next((r["species"] for r in ours if r["state"] == "ACTIVE"), None)
    opponent = next((r["species"] for r in theirs if r["state"] == "ACTIVE"), None)
    warnings = data.get("warnings") or []
    outcome = (battle or {}).get("outcome")
    return {
        "turn": data.get("turn") or header.get("turn"),
        "decision": {
            "kind": data.get("kind"),
            "turn": data.get("turn"),
            "order": data.get("order") or data.get("choice"),
            "action": data.get("choice"),
            "source": f"{data.get('degrade') or 'unknown'} · {data.get('reason') or ''}".strip(" ·"),
            "ts": datetime.fromtimestamp(record.path.stat().st_mtime, UTC).isoformat(),
        },
        "move_capture": datetime.fromtimestamp(record.path.stat().st_mtime, UTC).isoformat(),
        "moves": moves,
        "team": team,
        "your_team": your_team(data, moves, rows_of(text)[1]),
        "field": field_view(data.get("protocol"), data.get("turn")),
        "active": {"name": active, "hp": next((r["hp"] for r in ours if r["state"] == "ACTIVE"), None)},
        "opponent": {"name": opponent, "preview": [r["species"] for r in theirs] + unseen.split(", ")},
        "opponent_moves": [{"name": r["species"], "moves": r["moves"]} for r in theirs],
        "battle_state": {
            "battle": str((battle or {}).get("index", "pending")),
            "status": str(outcome or ("in progress" if battle else "waiting")),
            "screen": str(screen),
            "turn": header.get("turn") or data.get("turn") or "",
            "active": active or "",
            "opponent": opponent or "",
            "team": ", ".join(r["species"] for r in ours),
            "message": "; ".join(str(w) for w in warnings) or ((data.get("notes") or {}).get("reason") or ""),
            "battle_object": text or "(no tracked battle snapshot for this decision)",
        },
    }

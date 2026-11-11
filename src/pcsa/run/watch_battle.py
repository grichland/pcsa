"""What `pcsa watch` shows about the current battle: read from the newest `battle-N/decisions/` record.

Each decision leaves `NNNN-<kind>.json` (schema pcsa-decision/2: request, order, notes) and, for tracked
battles, `NNNN-<kind>.txt` (battle.state.battle_text). The page is read-only, so this module only parses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pcsa.run.watch_decisions import decision_files, read_json, row_of
from pcsa.run.watch_field import field_view
from pcsa.run.watch_names import names_of, own_sets, shown, to_id

ROW = re.compile(
    r"^  (?P<species>\S+)\s+(?P<state>ACTIVE|bench|fainted)\s+HP (?P<hp>\S+)\s+(?:(?P<types>\S+)\s+)?"
    r"status (?P<status>\S+)\s+boosts (?P<boosts>.+?)\s+ability (?P<ability>.+?)\s+"
    r"item (?P<item>.+?)\s+moves (?P<moves>.*)$"
)
HEADER = re.compile(r"^TURN (?P<turn>\d+)\s+weather: (?P<weather>.+?)\s+fields: (?P<fields>.+)$")


@dataclass(frozen=True)
class Record:
    path: Path
    data: dict
    text: str


def battle_dir(run_dir: Path, battle: int | None = None) -> Path | None:
    """The folder of battle N (default: the newest battle folder of the run)."""
    if battle is not None:
        return run_dir / f"battle-{battle}"
    try:
        battles = sorted(
            (p for p in run_dir.glob("battle-*") if p.name.rsplit("-", 1)[1].isdigit()),
            key=lambda p: int(p.name.rsplit("-", 1)[1]),
        )
    except OSError:
        return None
    return battles[-1] if battles else None


def latest_record(run_dir: Path, battle: int | None = None) -> Record | None:
    """The newest readable decision record of a battle folder (default: the newest folder), or None.

    A record that is still being written does not parse yet: the one before it is shown until it does.
    """
    for path in reversed(decision_files(battle_dir(run_dir, battle))):
        data = read_json(path)
        if data is None:
            continue
        try:
            text = path.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        return Record(path, data, text)
    return None


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
            "species": str(p.get("details") or "").split(",")[0],
            "condition": p.get("condition"),
            "active": bool(p.get("active")),
            "item": p.get("item"),
            "ability": p.get("ability"),
            "moves": list(p.get("moves") or []),
        }
        for p in side.get("pokemon") or []
    ]


def your_team(data: dict, moves: list[dict], ours: list[dict], names: dict[str, str]) -> list[dict]:
    """Our picks as the request shows them, with display names from the team sheet and PP for the active."""
    sets = own_sets(data.get("own_export") if isinstance(data.get("own_export"), str) else None)
    active_boosts = next((r["boosts"] for r in ours if r["state"] == "ACTIVE"), "-")
    types = {to_id(r["species"]): r.get("types") or "" for r in ours}
    out = []
    for member in team_of(data):
        condition = str(member.get("condition") or "")
        hp, _, status = condition.partition(" ")
        current, _, total = hp.partition("/")
        fraction = int(current) / int(total) if current.isdigit() and total.isdigit() and int(total) else None
        sheet = sets.get(member["name"], {})
        move_names = sheet.get("moves") or [shown(m, names) for m in member["moves"]]
        pp = {to_id(m["name"]): m["pp"] for m in moves} if member["active"] else {}
        out.append(
            {
                "name": member["name"],
                "active": member["active"],
                "fainted": status == "fnt" or hp == "0",
                "hp": hp,
                "fraction": 0.0 if status == "fnt" else fraction,
                "status": "" if status == "fnt" else status,
                "boosts": active_boosts if member["active"] and active_boosts != "-" else "",
                "types": types.get(to_id(member["species"])) or types.get(to_id(member["name"])) or "",
                "item": sheet.get("item") or shown(member.get("item") or "", names),
                "ability": sheet.get("ability") or shown(member.get("ability") or "", names),
                "moves": [{"name": m, "pp": pp.get(to_id(m))} for m in move_names],
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


def _known(value: str, names: dict[str, str]) -> str:
    """'' for the snapshot's unknown marks, else the written name."""
    return "" if value.strip() in ("?", "-", "") else shown(value.strip(), names)


def foes_of(theirs: list[dict], names: dict[str, str]) -> list[dict]:
    """The opponent's Pokémon seen so far: HP as the game shows it (a percentage) and what they revealed."""
    out = []
    for row in theirs:
        hp = row["hp"]
        fainted = row["state"] == "fainted"
        percent = hp[:-1] if hp.endswith("%") else ""
        out.append(
            {
                "name": shown(row["species"], names),
                "active": row["state"] == "ACTIVE",
                "fainted": fainted,
                "hp": "" if hp == "?" else hp,
                "fraction": 0.0 if fainted else int(percent) / 100 if percent.isdigit() else None,
                "status": "" if fainted else _known(row["status"], names),
                "boosts": "" if row["boosts"].strip() == "-" else row["boosts"].strip(),
                "types": row.get("types") or "",
                "ability": _known(row["ability"], names),
                "item": _known(row["item"], names),
                "moves": [shown(m, names) for m in row["moves"]],
            }
        )
    return out


def _on_field(members: list[dict]) -> dict:
    member = next((m for m in members if m["active"] and not m["fainted"]), None)
    keys = ("name", "hp", "fraction", "status", "boosts")
    return {k: member[k] for k in keys} if member else dict.fromkeys(keys)


def battle_view(run_dir: Path, battle: dict | None, screen: str | None) -> dict:
    """Everything the page's decision and battle panels read; empty values when there is no record yet."""
    record = latest_record(run_dir, (battle or {}).get("index"))
    if record is None:
        return {}
    data, text = record.data, record.text
    names = names_of(data)
    header, ours, theirs, unseen = rows_of(text)
    team = your_team(data, moves_of(data), ours, names)
    foes = foes_of(theirs, names)
    active, opponent = _on_field(team), _on_field(foes)
    warnings = data.get("warnings") or []
    try:
        decision = row_of(data, record.path.stat().st_mtime)
    except OSError:
        decision = row_of(data, 0.0)
    decision["source"] = f"{data.get('degrade') or 'unknown'} · {data.get('reason') or ''}".strip(" ·")
    outcome = (battle or {}).get("outcome")
    return {
        "turn": data.get("turn") or header.get("turn"),
        "decision": decision,
        "move_capture": decision["ts"],
        "your_team": team,
        "foes": foes,
        "unseen": [shown(n, names) for n in unseen.split(", ") if n and n != "-"],
        "field": field_view(data.get("protocol"), data.get("turn")),
        "active": active,
        "opponent": opponent,
        "battle_state": {
            "battle": str((battle or {}).get("index", "pending")),
            "status": str(outcome or ("in progress" if battle else "waiting")),
            "screen": str(screen),
            "turn": header.get("turn") or data.get("turn") or "",
            "active": active["name"] or "",
            "opponent": opponent["name"] or "",
            "team": ", ".join(m["name"] for m in team),
            "message": "; ".join(str(w) for w in warnings) or ((data.get("notes") or {}).get("reason") or ""),
            "battle_object": text or "(no tracked battle snapshot for this decision)",
        },
    }

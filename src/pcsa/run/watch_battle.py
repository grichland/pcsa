"""What `pcsa watch` shows about the current battle: read from the newest `battle-N/decisions/` record.

The teams and field come from CG's Battle in the record (`notes.cg.view`), the belief about their builds from the
network (`notes.cg.net.belief`); before the first decision both come from `team_preview.json` (key `cg`). A record
without a view takes them from the newest one that has it. A record of a decision schema before pcsa-decision/4
without a view is legacy: its teams, field and matrix stay empty and `legacy` says why.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pcsa.run.watch_decisions import decision_files, read_json, row_of
from pcsa.run.watch_names import names_of
from pcsa.run.watch_view import CARD_KEYS, battle_text, view_panels

SCHEMA = "pcsa-decision/4"
LOOK_BACK = 80  # records searched for the newest view or belief
NO_VIEW = "(no Champions Gym battle view for this decision)"
LEGACY = "(this run's records carry no battle view)"


@dataclass(frozen=True)
class Record:
    path: Path
    data: dict


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


def _records(run_dir: Path, battle: int | None):
    """Readable decision records of a battle folder, newest first."""
    for path in reversed(decision_files(battle_dir(run_dir, battle))):
        data = read_json(path)
        if data is not None:
            yield Record(path, data)


def latest_record(run_dir: Path, battle: int | None = None) -> Record | None:
    """The newest readable decision record of a battle folder (default: the newest folder), or None."""
    return next(_records(run_dir, battle), None)


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def cg_of(data: dict) -> dict:
    return _dict(_dict(data.get("notes")).get("cg"))


def view_of(data: dict) -> dict | None:
    view = cg_of(data).get("view")
    return view if isinstance(view, dict) else None


def belief_of(data: dict) -> list | None:
    belief = _dict(cg_of(data).get("net")).get("belief")
    return belief if isinstance(belief, list) and belief else None


def preview_of(folder: Path | None) -> dict:
    """team_preview.json of a battle folder ({} when absent or unreadable)."""
    return (read_json(folder / "team_preview.json") or {}) if folder is not None else {}


def _export(data: dict) -> str | None:
    export = data.get("own_export") or _dict(data.get("team")).get("export")
    return export if isinstance(export, str) else None


def _picks(preview: dict) -> list | None:
    picks = _dict(preview.get("decision")).get("picks")
    return picks if isinstance(picks, list) else None


def _empty_field() -> dict:
    return {"turn": None, "weather": None, "fields": [], "ours": [], "theirs": []}


def _panels(view: dict | None, export: str | None, names: dict[str, str], belief) -> dict:
    if view is None:
        empty = {"your_team": [], "foes": [], "unseen": [], "unseen_belief": [], "field": _empty_field()}
        return {**empty, "active": {}, "opponent": {}, "text": ""}
    panels = view_panels(view, export, names, belief)
    return {**panels, "text": battle_text(panels)}


def _state(battle: dict | None, screen, panels: dict, turn, message: str, empty_text: str) -> dict:
    outcome = (battle or {}).get("outcome")
    active, opponent = panels["active"] or {}, panels["opponent"] or {}
    return {
        "battle": str((battle or {}).get("index", "pending")),
        "status": str(outcome or ("in progress" if battle else "waiting")),
        "screen": str(screen),
        "turn": turn if turn is not None else "",
        "active": active.get("name") or "",
        "opponent": opponent.get("name") or "",
        "team": ", ".join(m["name"] for m in panels["your_team"]),
        "message": message,
        "battle_object": panels["text"] or empty_text,
    }


def _shown(panels: dict) -> dict:
    """The payload keys of the teams and field panels (active and opponent with every key, None when empty)."""
    out = {k: panels[k] for k in ("your_team", "foes", "unseen", "unseen_belief", "field")}
    for key in ("active", "opponent"):
        out[key] = panels[key] or dict.fromkeys(CARD_KEYS)
    return out


def _newest(run_dir: Path, battle: int | None) -> tuple[dict | None, list | None]:
    """The newest view and the newest belief among the battle's recent records (most often both the newest's)."""
    view = belief = None
    for _, record in zip(range(LOOK_BACK), _records(run_dir, battle), strict=False):
        view = view if view is not None else view_of(record.data)
        belief = belief if belief is not None else belief_of(record.data)
        if view is not None and belief is not None:
            break
    return view, belief


def preview_view(folder: Path | None, battle: dict | None, screen) -> dict:
    """The teams panel at the team preview, before any decision record: our team and their six with the belief
    (brought, lead); {} when the preview has no CG view."""
    preview = preview_of(folder)
    cg = _dict(preview.get("cg"))
    view = cg.get("view")
    if folder is None or not isinstance(view, dict):
        return {}
    request = _dict(preview.get("request"))
    export = _export(request)
    picks = view.get("picks") if any(type(i) is int for i in view.get("picks") or []) else _picks(preview)
    names = names_of({"own_export": export, "own_names": _dict(request.get("team")).get("names")})
    panels = _panels({**view, "picks": picks}, export, names, _dict(cg.get("net")).get("belief"))
    try:
        stamp = datetime.fromtimestamp((folder / "team_preview.json").stat().st_mtime, UTC).isoformat()
    except OSError:
        stamp = None
    return {
        "turn": None,
        "move_capture": stamp,
        "legacy": False,
        **_shown(panels),
        "battle_state": _state(battle, screen, panels, None, "team preview", NO_VIEW),
    }


def battle_view(run_dir: Path, battle: dict | None, screen: str | None) -> dict:
    """Everything the page's decision and battle panels read; the preview's view before the first record."""
    index = (battle or {}).get("index")
    record = latest_record(run_dir, index)
    if record is None:
        return preview_view(battle_dir(run_dir, index), battle, screen)
    data = record.data
    names = names_of(data)
    legacy = view_of(data) is None and data.get("schema") != SCHEMA
    view, belief = (None, None) if legacy else _newest(run_dir, index)
    if belief is None and not legacy:
        belief = _dict(_dict(preview_of(record.path.parent.parent).get("cg")).get("net")).get("belief")
    panels = _panels(view, _export(data), names, belief)
    try:
        decision = row_of(data, record.path.stat().st_mtime)
    except OSError:
        decision = row_of(data, 0.0)
    decision["source"] = f"{data.get('degrade') or 'unknown'} · {data.get('reason') or ''}".strip(" ·")
    warnings = data.get("warnings") or []
    message = "; ".join(str(w) for w in warnings) or str(_dict(data.get("notes")).get("reason") or "")
    turn = data.get("turn") if data.get("turn") is not None else panels["field"]["turn"]
    return {
        "turn": turn,
        "decision": decision,
        "move_capture": decision["ts"],
        "legacy": legacy,
        **_shown(panels),
        "battle_state": _state(battle, screen, panels, turn, message, LEGACY if legacy else NO_VIEW),
    }

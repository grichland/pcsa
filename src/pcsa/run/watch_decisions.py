"""The decision records of one battle as `pcsa watch` lists them: `battle-N/decisions/NNNN-<kind>.json`.

A record has the choice, order sent, legal orders, degrade rung and the player's notes. The neural player through
CG keeps CG's reply notes under `notes.cg` (model value, time, fallback; `net` with the network's outputs); they are
mapped to the names the page reads (rule, v, ms, p, h, fallback, why, matrix). Flat notes of other players are shown
as written. Missing, half-written or older records are skipped. Read-only.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from math import isfinite, log
from pathlib import Path

from pcsa.run.watch_matrix import matrix_view
from pcsa.run.watch_names import names_of, order_view

NAMED = ("rule", "why", "p", "h", "v", "ms", "matrix", "override", "override_why", "network")
CG_REST = ("cg_ms", "sent", "attempt", "dropped_replies")
MODEL_REST = ("slow", "reused")
CG_COUNTS = ("anomalies", "corrections")
CONTROLLER_FALLBACK = ("builtin", "screen", "blind")  # degrade rungs where the controller chose, not the player
VALUE_WIDTH = 120
LOGS = ("battle_log.txt", "log.txt")  # written when the battle ends; saved after every visit to the log


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if isfinite(value) else None


def _plain(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text[:VALUE_WIDTH]


def _order(text) -> str:
    return str(text or "").removeprefix("/choose ").strip()


def _strings(value) -> list[str]:
    return [_order(x) if isinstance(x, str) else _plain(x) for x in value] if isinstance(value, list) else []


def _text(notes: dict, key: str) -> str | None:
    return _plain(notes[key]) if notes.get(key) is not None else None


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def policy_numbers(pi, action) -> tuple[float | None, float | None]:
    """(p, h): the probability our policy gave the action played, and its entropy over the legal actions scaled
    to 0..1 (1: all of them equally likely)."""
    if not isinstance(pi, list):
        return None, None
    probs = [_number(x) for x in pi]
    if any(x is None for x in probs):
        return None, None
    values: list[float] = [x for x in probs if x is not None]
    p = values[action] if type(action) is int and 0 <= action < len(values) else None
    live = [x for x in values if x > 0]
    if len(live) < 2:  # noqa: PLR2004
        return p, 0.0 if live else None
    h = -sum(x * log(x) for x in live) / log(len(live))
    return p, round(min(1.0, max(0.0, h)), 4)


def _controller_fallback(record: dict) -> str | None:
    """Why the controller chose instead of the player (the record's live.fallback, else its degrade and reason)."""
    found = _dict(_dict(record.get("live")).get("fallback"))
    if found.get("reason"):
        return _plain(found["reason"])
    if record.get("degrade") in CONTROLLER_FALLBACK:
        return _plain(record.get("reason") or record["degrade"])
    return None


def _rule(player) -> str | None:
    if not isinstance(player, str) or not player:
        return None
    return "neural" if player.split(":", 1)[0] == "neural" else player


def cg_notes_view(notes: dict, record: dict, names: dict[str, str] | None, order) -> dict:
    """The page's view of CG's reply notes (`notes.cg`): model, net and the controller's own keys."""
    cg = notes["cg"]
    model, net = _dict(cg.get("model")), _dict(cg.get("net"))
    model_fallback = _dict(model.get("fallback"))
    controller = _controller_fallback(record)
    p, h = policy_numbers(net.get("pi"), cg.get("action"))
    value = _number(net.get("value"))
    why = _plain(model_fallback.get("reason") or "the model did not decide") if model.get("fallback") else controller
    rest = [{"key": k, "value": _plain(notes[k])} for k in CG_REST if notes.get(k) is not None]
    rest += [{"key": k, "value": _plain(model[k])} for k in MODEL_REST if model.get(k) is not None]
    rest += [{"key": k, "value": _plain(cg[k])} for k in CG_COUNTS if cg.get(k)]
    if model_fallback.get("player"):
        rest.append({"key": "fallback", "value": _plain(model_fallback["player"])})
    elif controller:
        rest.append({"key": "fallback_rule", "value": _plain(record.get("degrade") or "controller")})
    return {
        "rule": _rule(cg.get("player")),
        "fallback": bool(model.get("fallback")) or controller is not None,
        "why": why,
        "p": p,
        "h": h,
        "v": value if value is not None else _number(model.get("value")),
        "ms": _number(model.get("ms")),
        "matrix": matrix_view(net, names, order, cg.get("action")) if net else None,
        "override": None,
        "override_why": None,
        "network": None,
        "rest": rest,
    }


def notes_view(notes, names: dict[str, str] | None = None, order=None, record: dict | None = None) -> dict | None:
    """The neural numbers and matrix when present and every other note as a (key, value) pair. `names`, `order`
    and `record` (the decision record's) name the matrix rows, find the row played and say why the controller
    chose when it did."""
    if not isinstance(notes, dict) or not notes:
        return None
    record = record if isinstance(record, dict) else {}
    if isinstance(notes.get("cg"), dict):
        return cg_notes_view(notes, record, names, order)
    rule = notes.get("rule")
    network = _order(notes["network"]) if isinstance(notes.get("network"), str) else None
    controller = _controller_fallback(record)
    rest = [{"key": str(k), "value": _plain(v)} for k, v in notes.items() if k not in NAMED]
    if controller is not None:  # the reason is the banner's text; "played by" names the rung
        rest = [r for r in rest if r["key"] != "fallback"]
        rest.append({"key": "fallback_rule", "value": _plain(record.get("degrade") or "controller")})
    return {
        "rule": str(rule) if rule is not None else None,
        "fallback": (isinstance(rule, str) and rule.endswith(":fallback")) or controller is not None,
        "why": _text(notes, "why") or controller,
        "p": _number(notes.get("p")),
        "h": _number(notes.get("h")),
        "v": _number(notes.get("v")),
        "ms": _number(notes.get("ms")),
        "matrix": None,
        "override": _text(notes, "override"),
        "override_why": _text(notes, "override_why"),
        "network": order_view(network, names or {}) if network else None,
        "rest": rest,
    }


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def row_of(data: dict, mtime: float) -> dict:
    """One line of the decision list: orders carry the written names of their moves and Pokémon."""
    names = names_of(data)
    chosen = order_view(_order(data.get("order") or data.get("choice")), names)
    return {
        "sequence": data.get("sequence"),
        "kind": data.get("kind"),
        "turn": data.get("turn"),
        "visit": data.get("visit"),
        "choice": data.get("choice"),
        "order": chosen["order"],
        "label": chosen["label"],
        "order_kind": chosen["kind"],
        "degrade": data.get("degrade"),
        "reason": data.get("reason"),
        "legal": [order_view(o, names) for o in _strings(data.get("legal"))],
        "dropped": [order_view(o, names) for o in _strings(data.get("dropped"))],
        "warnings": _strings(data.get("warnings")),
        "notes": notes_view(data.get("notes"), names, data.get("order") or data.get("choice"), data),
        "ts": datetime.fromtimestamp(mtime, UTC).isoformat(),
        "timing": None,
    }


def decision_files(battle_dir: Path | None) -> list[Path]:
    folder = battle_dir / "decisions" if battle_dir is not None else None
    try:
        return sorted(folder.glob("*.json")) if folder is not None and folder.is_dir() else []
    except OSError:
        return []


def folder_key(battle_dir: Path | None) -> tuple:
    """Cheap change detector for a battle folder: its newest decision record, its battle log and its preview."""
    if battle_dir is None:
        return ()
    files = decision_files(battle_dir)
    stamps = []
    for path in [*files[-1:], *(battle_dir / name for name in (*LOGS, "team_preview.json"))]:
        try:
            stat = path.stat()
            stamps.append((path.name, stat.st_mtime_ns, stat.st_size))
        except OSError:
            stamps.append((path.name, 0, 0))
    return (str(battle_dir), len(files), *stamps)


class DecisionRows:
    """Decision-list rows of battle folders, each record parsed once per version of its file."""

    def __init__(self) -> None:
        self._cache: dict[Path, tuple[tuple[int, int], dict]] = {}

    def rows(self, battle_dir: Path | None) -> list[dict]:
        out = []
        for path in decision_files(battle_dir):
            try:
                stat = path.stat()
            except OSError:
                continue
            stamp = (stat.st_mtime_ns, stat.st_size)
            cached = self._cache.get(path)
            if cached is None or cached[0] != stamp:
                data = read_json(path)
                if data is None:  # half written: the next pass reads it
                    continue
                cached = (stamp, row_of(data, stat.st_mtime))
                self._cache[path] = cached
            out.append(dict(cached[1]))
        return out

"""The decision records of one battle as `pcsa watch` lists them: `battle-N/decisions/NNNN-<kind>.json`.

Each record (schema pcsa-decision/2) carries the choice, the order sent, the legal orders, the degrade rung and
the player's own notes. Notes are shown as they are: the rule name and whatever numbers the player wrote. The
neural player's are named (`p` the probability of the chosen action, `h` the normalised entropy, `v` the critic
value, `ms`, `why` under the rule `neural:fallback`, the `matrix` of the decision, and on a guard-rule override
`override`, `override_why` and `network`, the order the network wanted). A record that is missing, half written or
of an older shape is skipped. This module only reads.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pcsa.run.watch_matrix import matrix_view
from pcsa.run.watch_names import names_of, order_view

NAMED = ("rule", "why", "p", "h", "v", "ms", "matrix", "override", "override_why", "network")
VALUE_WIDTH = 120
LOGS = ("battle_log.txt", "log.txt")  # written when the battle ends; saved after every visit to the log


def _number(value) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _plain(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text[:VALUE_WIDTH]


def _order(text) -> str:
    return str(text or "").removeprefix("/choose ").strip()


def _strings(value) -> list[str]:
    return [_order(x) if isinstance(x, str) else _plain(x) for x in value] if isinstance(value, list) else []


def _text(notes: dict, key: str) -> str | None:
    return _plain(notes[key]) if notes.get(key) is not None else None


def notes_view(notes, names: dict[str, str] | None = None, order=None) -> dict | None:
    """Rule, the neural numbers and matrix when present, the override fields, and every other note as a
    (key, value) pair. `names` and `order` (the record's) name the matrix rows and find the row played."""
    if not isinstance(notes, dict) or not notes:
        return None
    rule = notes.get("rule")
    network = _order(notes["network"]) if isinstance(notes.get("network"), str) else None
    return {
        "rule": str(rule) if rule is not None else None,
        "fallback": isinstance(rule, str) and rule.endswith(":fallback"),
        "why": _text(notes, "why"),
        "p": _number(notes.get("p")),
        "h": _number(notes.get("h")),
        "v": _number(notes.get("v")),
        "ms": _number(notes.get("ms")),
        "matrix": matrix_view(notes.get("matrix"), names, order, network),
        "override": _text(notes, "override"),
        "override_why": _text(notes, "override_why"),
        "network": order_view(network, names or {}) if network else None,
        "rest": [{"key": str(k), "value": _plain(v)} for k, v in notes.items() if k not in NAMED],
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
        "notes": notes_view(data.get("notes"), names, data.get("order") or data.get("choice")),
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
    """Cheap change detector for a battle folder: its newest decision record and its battle log."""
    if battle_dir is None:
        return ()
    files = decision_files(battle_dir)
    stamps = []
    for path in [*files[-1:], *(battle_dir / name for name in LOGS)]:
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

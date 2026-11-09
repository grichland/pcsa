"""`battles.jsonl.gz` of `pcsa sim play`: one JSON object per finished battle, its full protocol included.

Each ArenaPlayer keeps a seat record per finished battle (`finished`: role, team, picks, won, turns and the
battle's replay data); `records` merges A's and B's seats into one record per battle tag, keyed by seat.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

FILE = "battles.jsonl.gz"
ROLES = ("p1", "p2")


def protocol_text(replay_data: Sequence[Sequence[str]] | None) -> str | None:
    """poke-env's `_replay_data` (split messages) as text: one "|"-joined message per line."""
    if not replay_data:
        return None
    return "\n".join("|".join(parts) for parts in replay_data)


def _winner(seats: Mapping[str, dict]) -> str | None:
    for role, seat in seats.items():
        if seat["won"] is not None:
            return role if seat["won"] else ROLES[1 - ROLES.index(role)]
    return None


def merge(index: int, tag: str, sides: Mapping[str, tuple[str, dict | None]]) -> dict[str, Any] | None:
    """One battle from each side's (kind, seat record or None); None when neither seat saw it finish."""
    seen = {side: seat for side, (_, seat) in sides.items() if seat is not None}
    if not seen:
        return None
    known_side, known = next(iter(seen.items()))
    other_side = next(s for s in sides if s != known_side)
    other_role = ROLES[1 - ROLES.index(known["role"])]
    role_of = {known_side: known["role"], other_side: other_role}
    side_at = {role: side for side, role in role_of.items()}
    seats = {role_of[side]: seat for side, seat in seen.items()}
    record: dict[str, Any] = {"tag": tag, "index": index}
    for role in ROLES:
        side = side_at[role]
        seat = seats.get(role)
        record[role] = sides[side][0]
        record[f"{role}_side"] = side
        record[f"{role}_team"] = seat["team"] if seat else None
        record[f"{role}_picks"] = seat["picks"] if seat and seat["picks"] else None
    winner = _winner(seats)
    voice = seats.get("p1") or seats["p2"]
    record.update(
        winner=winner,
        winner_side=side_at[winner] if winner else None,
        turns=max(seat["turns"] for seat in seats.values()),
        protocol_seat="p1" if "p1" in seats else "p2",
        protocol=protocol_text(voice["replay"]),
    )
    return record


def records(order: Iterable[str], sides: Mapping[str, tuple[str, Mapping[str, dict]]]) -> list[dict[str, Any]]:
    """One record per finished battle in `order` (battle tags); sides: {"a": (kind, finished), "b": ...}."""
    out = []
    for index, tag in enumerate(order):
        merged = merge(index, tag, {side: (kind, seats.get(tag)) for side, (kind, seats) in sides.items()})
        if merged is not None:
            out.append(merged)
    return out


def write(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """The records as gzipped JSON lines; returns the file's size in bytes."""
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path.stat().st_size


def read(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

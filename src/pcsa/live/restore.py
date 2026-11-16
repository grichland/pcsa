"""Picking up a battle that is already under way: its earlier decisions, and the player's memory of them.

A session that starts mid-battle (supervisor restart, `pcsa run --resume`) continues the decision numbering and
rebuilds every earlier decision's view from `decisions/NNNN-<kind>.json`, handing them to `player.restore(views)`
once. `battle-N/battle_id.json` says which battle owns a folder, so a lost battle's records are not inherited.
Restoring never breaks the battle: a record that cannot be read or rebuilt ends the list there.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from pcsa.battle import state as tracking
from pcsa.battle.adapter import ChampionsBattle
from pcsa.common import log
from pcsa.players import mirror
from pcsa.players.base import DecisionView, Degrade, ScreenFacts
from pcsa.run.battle_state import BattleRecord, same_name

SCHEMAS = ("pcsa-decision/1", "pcsa-decision/2")
SHOWN_BATTLE = (str(Degrade.FULL), str(Degrade.TRACKED))
OWNER = "battle_id.json"


@dataclass(frozen=True)
class Earlier:
    """The first part of a battle that is being picked up again."""

    battle_id: str
    records: list[dict] = field(default_factory=list)  # readable records from sequence 0, in order
    kinds: list[str] = field(default_factory=list)  # one entry per sequence number already used


def battle_tag(battle_id: str, sequence: int) -> str:
    """The tag of the Battle built for one decision; the turn flow and the rebuild name it the same way."""
    return f"{battle_id}-{sequence}"


def claim(recorder, b: BattleRecord) -> None:
    """Note that this battle owns its folder from now on."""
    try:
        recorder.write_json(f"{b.dir}/{OWNER}", {"battle_id": b.battle_id})
    except OSError as error:
        log.warning("battle_claim_failed", battle=b.index, reason=str(error))


def release(s) -> None:
    """The lobby or the post-battle options are showing: the battle the run state calls pending is over."""
    if s.state.pending_battle is not None:
        log.event("pending_battle_dropped", battle=s.state.pending_battle)
        s.state.pending_battle = None
        s.state.save()


def earlier(s, index: int) -> Earlier | None:
    """The first part of battle `index` when the run state says it was in progress; else None. Never raises."""
    if s.state.pending_battle != index:
        return None
    try:
        directory = s.rec.dir / f"battle-{index}"
        owner = json.loads((directory / OWNER).read_text(encoding="utf-8")).get("battle_id")
        if not isinstance(owner, str) or not owner:
            return None
        records, kinds = load_records(directory / "decisions", owner)
        return Earlier(owner, records, kinds)
    except (OSError, ValueError, AttributeError) as error:
        log.warning("battle_resume_unreadable", battle=index, reason=f"{type(error).__name__}: {error}")
        return None


def _read(path: Path) -> dict | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    good = isinstance(record, dict) and record.get("schema") in SCHEMAS and isinstance(record.get("sequence"), int)
    return record if good else None


def load_records(directory: Path, battle_id: str | None = None) -> tuple[list[dict], list[str]]:
    """(the records from sequence 0 up to the first missing or unreadable one, the kind of every sequence used)."""
    found: dict[int, dict | None] = {}
    kinds: dict[int, str] = {}
    for path in sorted(directory.glob("[0-9]*-*.json")):
        number, _, kind = path.stem.partition("-")
        record = _read(path)
        if record is not None and battle_id is not None and record.get("battle_id") != battle_id:
            continue
        if record is None and not number.isdigit():
            continue
        sequence = record["sequence"] if record is not None else int(number)
        if record is not None or sequence not in found:
            found[sequence], kinds[sequence] = record, kind
    records: list[dict] = []
    while (record := found.get(len(records))) is not None:
        records.append(record)
    return records, ([kinds.get(i, "") for i in range(max(found) + 1)] if found else [])


def battle_of(protocol: list[str], request: dict, tag: str):
    """The Battle of one decision, rebuilt as the turn flow built it (battle.state.build, then the request)."""
    battle = ChampionsBattle(tag, tracking.USERNAME, tracking.LOGGER, 9)
    battle.player_role = "p1"
    for line in protocol:
        try:
            battle.parse_message(line.split("|"))
        except Exception:  # the translator kept such a line as a warning; so does the rebuild
            continue
    battle.parse_request(request)
    tracking.normalize_fainted_request(battle)
    return battle


def view_of(record: dict, tag: str) -> DecisionView:
    """What the player was shown at this decision. ValueError when the record does not rebuild to it."""
    protocol, request = record.get("protocol"), record.get("request")
    if not protocol or not isinstance(request, dict):
        raise ValueError("no protocol or request in the record")
    battle = battle_of(protocol, request, tag)
    if record.get("turn") is not None and battle.turn != record["turn"]:
        raise ValueError(f"rebuilt turn {battle.turn}, recorded turn {record['turn']}")
    offered = {a.message: a for a in mirror.legal_actions(battle)}
    wanted = record.get("legal") or []
    missing = [m for m in wanted if m not in offered]
    if missing or not wanted:
        raise ValueError(f"recorded orders the rebuilt battle does not offer: {missing}")
    active = (request.get("active") or [{}])[0]
    screen = ScreenFacts((), bool(active.get("canMegaEvo")), bool(request.get("forceSwitch")), None)
    legal = tuple(offered[m] for m in wanted)
    return DecisionView(legal, screen, Degrade(record["degrade"]), random.Random(0), battle)


def rebuild_views(records: list[dict], battle_id: str) -> list[DecisionView]:
    """The views of the decisions the player was shown a Battle for, oldest first; ends at the first record
    that cannot be rebuilt. A decision made from the screen alone gave the player no Battle and is skipped."""
    views: list[DecisionView] = []
    for record in records:
        if record.get("degrade") not in SHOWN_BATTLE:
            continue
        try:
            views.append(view_of(record, battle_tag(battle_id, record["sequence"])))
        except Exception as error:
            log.warning(
                "memory_restore_stopped", sequence=record.get("sequence"), reason=f"{type(error).__name__}: {error}"
            )
            break
    return views


TEAM_SIZE = 3
GOOD_RUNGS = ("full", "tracked")  # records whose request shows our side as tracked


def _side_names(record: dict) -> list[str]:
    side = ((record.get("request") or {}).get("side") or {}).get("pokemon") or []
    return [str(p.get("ident", "")).partition(": ")[2] for p in side if isinstance(p, dict)]


def lineup(records: list[dict], roster: tuple) -> dict | None:
    """Our brought trio as the earlier records show it: `requested` = roster rows in the order of the first."""
    good = [r for r in records if r.get("degrade") in GOOD_RUNGS and len(_side_names(r)) == TEAM_SIZE]
    if not good:
        return None
    first, last = _side_names(good[0]), good[-1]

    def row(name: str) -> int | None:
        rows = [i for i, n in enumerate(roster) if same_name(n, name)]
        return rows[0] if len(rows) == 1 else None

    requested = [row(n) for n in first]
    if None in requested or len(set(requested)) != TEAM_SIZE:
        return None
    slots = [requested.index(row(n)) if row(n) in requested else None for n in _side_names(last)]
    if None in slots or sorted(slots) != list(range(TEAM_SIZE)):  # type: ignore[type-var]
        return None
    side = last["request"]["side"]["pokemon"]
    fainted = {slot for slot, p in zip(slots, side, strict=True) if str(p.get("condition", "")).endswith("fnt")}
    mega = any("mega" in str(p.get("details", "")).lower() for r in good for p in r["request"]["side"]["pokemon"])
    return {"requested": requested, "visible_order": slots, "fainted": fainted, "mega_used": mega}


def _restore_lineup(b: BattleRecord, records: list[dict]) -> None:
    try:
        found = lineup(records, b.roster)
    except Exception as error:  # never stops a resume
        log.warning("lineup_restore_failed", battle=b.index, reason=f"{type(error).__name__}: {error}")
        return
    if found is None:
        return
    b.requested, b.visible_order = found["requested"], found["visible_order"]
    b.fainted, b.mega_used = set(found["fainted"]), bool(found["mega_used"])
    log.event(
        "lineup_restored",
        battle=b.index,
        names=list(b.names),
        order=list(b.visible_order),
        fainted=sorted(b.fainted),
        mega_used=b.mega_used,
    )


def resume(s, b: BattleRecord, found: Earlier) -> None:
    """Continue the numbering of the battle's records, put our lineup back and restore the player's memory. Never raises."""
    b.decisions = list(found.kinds)
    _restore_lineup(b, found.records)
    restore = getattr(s.player, "restore", None)
    views, restored = 0, 0
    if found.records and callable(restore):
        try:
            rebuilt = rebuild_views(found.records, b.battle_id)
            views = len(rebuilt)
            restored = int(restore(rebuilt))  # pyright: ignore[reportArgumentType] - the player returns a count
        except Exception as error:
            log.exception("memory_restore_failed", battle=b.index, reason=f"{type(error).__name__}: {error}")
    log.event(
        "battle_resumed",
        battle=b.index,
        battle_id=b.battle_id,
        records=len(found.kinds),
        readable=len(found.records),
        views=views,
        restored=restored,
        player_restores=callable(restore),
    )

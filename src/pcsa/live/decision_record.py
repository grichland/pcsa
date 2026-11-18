"""The `pcsa-decision/4` record, one per decision: the fields pcsa's dashboard reads, the Request's rows, events,
offer and hints, the Player, its time and the visit's gaps, `live`: the LIVE decision_request and fallback this
decision stands for, and in `notes.cg` CG's reply notes (with `view` and `net` from pcsa.cg.serve)."""

from __future__ import annotations

from pcsa.common import log
from pcsa.common.types import jsonable
from pcsa.decide.api import Order, Request
from pcsa.decide.order_text import legal_text, move_id
from pcsa.decide.runner import Decision
from pcsa.events.live_ids import VERSION, Teams, mon_id
from pcsa.run.battle_state import BattleRecord
from pcsa.run.recorder import DECISION_SCHEMA, Recorder

SCHEMA = DECISION_SCHEMA
EVENT_LIST_LIMIT = 16
LOG_KEYS = frozenset({"battle", "kind", "choice", "degrade", "legal", "dropped"})
NOT_FALLBACK = ("player", "automatic")  # the Player answered, or the game took no choice
TAP_NOT_CONFIRMED = "tap not confirmed"


def choice_label(b: BattleRecord, order: Order) -> str:
    """pcsa's choice label: "move1", "mega+move0", or "switch<slot>" (the slot in the brought three)."""
    if order.kind == "move":
        return order.label()
    slot = b.slot_of(order.name)
    return f"switch{slot}" if slot is not None else order.label()


def live_order(order: Order, request: Request, teams: Teams | None) -> dict | None:
    """The order in LIVE shape; None when the move row has no id."""
    if order.kind == "switch":
        return {"kind": "switch", "mon": mon_id(order.name or "", "own", teams or Teams(()))}
    move = move_id(order, request.offer)
    return {"kind": "move", "move": move, "mega": bool(order.mega)} if move else None


def decision_request(request: Request, kind: str, *, refused: dict | None = None, retry: bool = False) -> dict:
    """The section-2 decision_request: `refused` when the game refused our last order (its refused event is in
    request.events), else `retry` when the screen came back without a refusal; `screen_offer` when all rows
    were read. Built before the Player is asked: it travels in Request.live."""
    out: dict = {"v": VERSION, "battle": request.battle_tag, "type": "decision_request", "seq": request.request_id,
                 "kind": "replace" if kind == "forced" else "turn"}  # fmt: skip
    if refused:
        out["refused"] = {"order": refused}
    elif retry:
        out["retry"] = True
    moves = sorted(request.offer.moves, key=lambda m: m.row)
    if request.offer.moves_read and moves and all(m.id for m in moves):
        out["screen_offer"] = {"moves": [m.id for m in moves]}
    return out


def fallback(request: Request, decision: Decision, converter) -> dict | None:
    """The section-3 fallback when the controller chose, without `order` when the row has no id (LIVE 0.6);
    None when the Player's answer stood."""
    if decision.source in NOT_FALLBACK:
        return None
    order = live_order(decision.order, request, getattr(converter, "teams", None))
    reason = f"{decision.source}: {decision.reason}" if decision.reason else decision.source
    out = {"seq": request.request_id, "reason": reason}
    return {**out, "order": order} if order else out


def live_messages(kind: str, request: Request, decision: Decision, converter) -> dict:
    """The record's `live`: the decision_request the Player was sent, the fallback when the controller chose, and
    `played`, the order tapped (a replay after a CG restart sends it, LIVE 0.7) when it has an id."""
    live: dict = {"decision_request": request.live or decision_request(request, kind)}
    found = fallback(request, decision, converter)
    if found:
        live["fallback"] = found
    played = live_order(decision.order, request, getattr(converter, "teams", None))
    if played and decision.source != "automatic":
        live["played"] = played
    return live


def history_entry(record: dict, request: Request, decision: Decision) -> dict | None:
    """What a later Request carries about this decision (Request.history); None when the Player was not asked."""
    if decision.source == "automatic":
        return None
    return {"events": len(request.events), **record["live"]}


def unconfirmed(b: BattleRecord, seq: int | None, why: str) -> None:
    """The tap for `seq` may not have registered: CG gets no `played` for it, only a fallback without an order
    (LIVE rule 31, S10-4)."""
    entry = next((e for e in reversed(b.live_history) if e["decision_request"].get("seq") == seq), None)
    if entry is None or seq is None:
        return
    entry.pop("played", None)
    entry["fallback"] = {"seq": seq, "reason": TAP_NOT_CONFIRMED}
    log.warning("tap_not_confirmed", battle=b.index, seq=seq, why=why)


def event_notes(notes: dict) -> dict:
    """Notes for the flattened `decision` log event: scalars and short lists; the record keeps them all."""
    return {
        key: value
        for key, value in notes.items()
        if key not in LOG_KEYS
        and not isinstance(value, dict)
        and not (isinstance(value, list | tuple | set | frozenset) and len(value) > EVENT_LIST_LIMIT)
    }


def preview_cg(notes: dict) -> tuple[dict, dict | None, dict]:
    """The preview's notes split: the rest, CG's reply notes (team_preview.json `cg`: view, net, model) and the flat
    `v` (value) and `ms` the `decision` event of kind team carries."""
    cg = notes.get("cg")
    if not isinstance(cg, dict):
        return notes, None, {}
    found = cg.get("model")
    model: dict = found if isinstance(found, dict) else {}
    flat = {key: model[name] for key, name in (("v", "value"), ("ms", "ms")) if model.get(name) is not None}
    return {k: v for k, v in notes.items() if k != "cg"}, cg, flat


def record(
    b: BattleRecord, kind: str, request: Request, decision: Decision, *, player: str, converter, **extra
) -> dict:
    return {
        "schema": SCHEMA,
        "battle_id": b.battle_id,
        "sequence": request.request_id,
        "request_id": request.request_id,
        "kind": kind,
        "choice": choice_label(b, decision.order),
        "order": decision.text,
        "degrade": decision.source,
        "reason": decision.reason,
        "legal": legal_text(request.offer),
        "dropped": list(decision.rejected),
        "notes": decision.notes,
        "late": list(decision.late),
        "player": player,
        "player_ms": decision.player_ms,
        "deadline": request.deadline,
        "log_quality": request.log_quality,
        "unmapped": list(getattr(converter, "unreadable", [])),
        "warnings": list(getattr(converter, "errors", [])),
        "own_export": request.team.export,
        "own_names": list(request.team.names),
        "team": jsonable(request.team),
        "opponent_preview": list(request.opponent_preview),
        "offer": jsonable(request.offer),
        "hints": jsonable(request.hints),
        "rows": jsonable(request.rows),
        "events": jsonable(request.events),
        "live": live_messages(kind, request, decision, converter),
        **extra,
    }


def save(recorder: Recorder, b: BattleRecord, kind: str, request: Request, decision: Decision, **fields) -> dict:
    """Writes battle-N/decisions/NNNN-kind.json and logs the `decision` event; returns the record."""
    b.decisions.append(kind)
    b.degrade_counts[decision.source] = b.degrade_counts.get(decision.source, 0) + 1
    out = record(b, kind, request, decision, **fields)
    entry = history_entry(out, request, decision)
    if entry is not None:
        b.live_history.append(entry)
    recorder.write_json(f"{b.dir}/decisions/{request.request_id:04d}-{kind}.json", out)
    log.event(
        "decision",
        battle=b.index,
        kind=kind,
        choice=out["choice"],
        degrade=decision.source,
        legal=len(out["legal"]),
        dropped=len(out["dropped"]),
        **event_notes(decision.notes),
    )
    return out

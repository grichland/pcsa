"""One recorded battle -> the full ordered LIVE.md message stream, in the order things happened live.

Each log visit's rows go out as one `events` message just before the decision that visit led to. A `replace`
decision is placed after the faint or pivot row it answers (LIVE.md 0.4, CF-9/CF-10), though live it came before
those rows were read; with no such row it goes at the end of its turn. Each row is sent exactly once. Messages
from CG (picks, orders) are in the stream too, marked {"from": "cg"}; the controller's are {"from": "controller"}.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from pcsa.battle.ids import to_id
from pcsa.battle.log_rows import RawRow, split_category
from pcsa.events import decisions as dec
from pcsa.events import start
from pcsa.events.gaps import final_turn, turn_gaps, visit_gaps
from pcsa.events.live import Converter
from pcsa.events.live_ids import VERSION
from pcsa.events.recorded import Recorded

RESULTS = {"win": "win", "loss": "loss", "opponent_forfeit": "forfeit", "unknown": "unknown"}


def is_trigger(e: dict) -> bool:
    """An own row a replace decision answers: a faint, a pivot withdraw, Eject Button, Emergency Exit."""
    if e.get("side") != "own":
        return False
    return (e["k"] == "faint" or (e["k"] == "withdraw" and bool(e.get("pivot")))
            or (e["k"] == "item" and e.get("item") == "ejectbutton" and e.get("what") == "used")
            or (e["k"] == "ability" and e.get("ability") == "emergencyexit"))


@dataclass
class Stream:
    tag: str
    messages: list[dict] = field(default_factory=list)  # {"from": ..., "msg": ...}
    row_ids: list[str] = field(default_factory=list)  # row ids in the order they were sent
    counts: Counter = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)  # sets left out, unresolved orders
    converter: Converter | None = None


class _Builder:
    def __init__(self, rec: Recorded) -> None:
        self.rec = rec
        self.out = Stream(rec.tag)
        self.roster = start.roster_of(rec.meta, rec.own_team)
        self.picks = start.picks_of(rec.meta, rec.run_events)
        self.brought = start.brought_names(rec.meta, self.roster, self.picks)
        names = start.preview_names(rec.meta, rec.run_events)
        self.conv = Converter(start.teams_for_rows(rec.meta, self.brought, names))
        self.out.converter = self.conv
        self.sets, set_notes = start.own_sets(rec.own_team, self.roster or None)
        self.out.notes += set_notes
        self.ctx = self._context()
        self.queues: dict[int, deque] = {}
        for r in rec.rows:
            self.queues.setdefault(r["turn"], deque()).append(r)
        self.visits = deque((rec.audit or {}).get("visits") or [])
        self.last = final_turn(rec.coverage, rec.rows, rec.decisions)
        self.turn_gaps = turn_gaps(rec.rows, rec.audit, rec.decisions, self.last)
        self.turn = 0  # the last `turn` event sent
        self.pending: list[dict] = []  # events not yet sent
        self.fallbacks: list[dict] = []  # fallbacks for the next events message
        self.active: str | None = None  # our active species, from our own `enter` events
        self.waiting: deque[dict] = deque()  # replace groups waiting for their row: {"turn", "items"}
        self.trigger_open = False  # an own trigger row was sent and no request or own enter since

    def _context(self) -> dec.OrderContext:
        species, moves = {}, {}
        for name, s in zip(self.roster, self.sets or [], strict=False):
            species[to_id(name)] = s["species"]
            moves[s["species"]] = list(s.get("moves") or [])
        return dec.OrderContext(species, moves, self.brought)

    # sending ------------------------------------------------------------------------------------
    def send(self, msg: dict, source: str = "controller") -> None:
        if msg.get("type") == "decision_request":
            self.trigger_open = False
        self.out.messages.append({"from": source, "msg": msg})
        self.out.counts[msg.get("type") or ("picks" if "picks" in msg else "order")] += 1

    def flush(self) -> None:
        """Pending events and any fallbacks as `events` messages (one fallback per message)."""
        while self.pending or self.fallbacks:
            msg = {"v": VERSION, "battle": self.out.tag, "type": "events", "events": self.pending}
            if self.fallbacks:
                msg["fallback"] = self.fallbacks.pop(0)
            self.pending = []
            self.send(msg)

    def event(self, e: dict) -> None:
        self.pending.append(e)
        self.out.counts[f"k:{e['k']}"] += 1
        if e["k"] == "enter" and e.get("side") == "own":
            self.active = e.get("mon")
            self.trigger_open = False
        elif is_trigger(e):
            self.trigger_open = True

    def send_item(self, item: tuple[str, dict]) -> None:
        what, payload = item
        if what == "event":
            self.event(payload)
        elif what == "fallback":
            self.fallbacks.append(payload)
        else:
            self.flush()
            self.send(payload, "cg" if what == "cg" else "controller")

    def release(self, how: str) -> None:
        """Send the oldest waiting replace group here."""
        group = self.waiting.popleft()
        self.out.counts[f"replace_placed:{how}"] += 1
        for item in group["items"]:
            self.send_item(item)

    def place_replace(self, record: dict, items: list, repeat: bool) -> None:
        if repeat and self.waiting:  # the same replacement screen again
            self.waiting[-1]["items"] += items
        elif repeat or self.trigger_open:
            if not repeat:
                self.out.counts["replace_placed:after_trigger"] += 1
            for item in items:
                self.send_item(item)
        else:
            self.waiting.append({"turn": record.get("turn"), "items": items})

    def new_turn(self, n: int) -> None:
        """`turn` for every turn up to n, each turn closed (with its gap, if any) before the next begins."""
        while self.turn < n:
            while self.waiting and (self.waiting[0]["turn"] or 0) <= self.turn:
                self.release("turn_end")  # its faint or pivot row was not read
            self.close_turn(self.turn)
            self.turn += 1
            self.event({"k": "turn", "n": self.turn})

    def close_turn(self, t: int) -> None:
        if t in self.turn_gaps:
            self.event(self.turn_gaps.pop(t))

    # rows ---------------------------------------------------------------------------------------
    def row(self, r: dict) -> None:
        if r["turn"] > self.turn:
            self.new_turn(r["turn"])
        elif r["turn"] < self.turn:
            self.out.counts["rows_after_a_later_turn"] += 1
        category, message = split_category(r["text"])
        rid = f"T{r['turn']}#{r['index']}"
        raw = RawRow(rid, r["turn"], r["index"], r["text"], category, message, r.get("sprite"))
        if r.get("read") == "unread":  # the reader flagged it: sent as unread with its reason (LIVE.md rule 4)
            self.event({"k": "unread", "text": r["text"], **({"category": r["reason"]} if r.get("reason") else {})})
        else:
            self.event(self.conv.row(raw, r.get("side")))  # the pill side, when the row carries one
        self.out.row_ids.append(rid)
        if self.waiting and self.trigger_open and self.waiting[0]["turn"] == r["turn"]:
            self.release("after_trigger")

    def visit(self, v: dict) -> None:
        for g in visit_gaps(v):
            self.event(g)
        queue = self.queues.get(v.get("turn"), deque())
        for _ in range(int(v.get("appended") or 0)):
            if not queue:
                self.out.counts["visit_rows_missing"] += 1
                break
            self.row(queue.popleft())

    def visits_through(self, number: int | None) -> None:
        while self.visits and (number is None or self.visits[0].get("visit", 0) <= number):
            self.visit(self.visits.popleft())

    # decisions ----------------------------------------------------------------------------------
    def decision(self, record: dict, previous: dict | None) -> None:
        refused, repeat = None, None
        if dec.refused_by(record, previous):
            before = dec.order_of(previous, self.ctx, self.active) if previous else None
            repeat = dec.repeat_of(record, previous, before)
            self.out.counts[f"{repeat}:{dec.kind_of(record)}"] += 1
            refused = before if repeat == "refused" else None
        if record.get("kind") == "action":
            self.visits_through(record.get("visit"))
            n = dec.game_turn(record)
            if n is not None:
                self.new_turn(n)  # turn n-1 has been read by now
            while self.waiting:
                self.release("before_next_request")
        items: list[tuple[str, dict]] = [("event", {"k": "refused", "order": refused})] if refused else []
        items.append(("request", dec.decision_request(self.out.tag, record, refused, repeat == "retry")))
        self.out.counts[f"decision:{dec.kind_of(record)}"] += 1
        order = dec.order_of(record, self.ctx, self.active)
        if order is None:
            self.out.counts["order_unresolved"] += 1
        if reason := dec.fallback_reason(record):
            self.out.counts[f"fallback:{reason}"] += 1
            fallback = dec.fallback_of(record, order)
            if fallback:
                items.append(("fallback", fallback))
            else:
                self.out.counts["fallback_not_sent:no_order"] += 1  # section 3 needs the order
        elif order:
            items.append(("cg", dec.answer(record, order)))
        if record.get("kind") == "forced":
            self.place_replace(record, items, repeat is not None)
        else:
            for item in items:
                self.send_item(item)

    def build(self) -> Stream:
        rec = self.rec
        self.send(start.battle_start(self.out.tag, self.sets, start.opp_preview(rec.meta, rec.run_events)))
        if self.picks is not None:
            self.send(start.preview_request(self.out.tag))
            self.send({"picks": self.picks}, "cg")
        previous = None
        for record in rec.decisions:
            self.decision(record, previous)
            previous = record
        # The game chose for us after a late answer: section 3's fallback needs an order, which was not read.
        timeouts = sum(e.get("event") == "turn_timed_out" for e in rec.run_events)
        self.out.counts["fallback_not_sent:game_timeout"] += timeouts
        self.finish()
        return self.out

    def finish(self) -> None:
        self.visits_through(None)
        for turn in sorted(self.queues):  # rows no audited visit appended (no ocr_audit.json)
            while self.queues[turn]:
                self.out.counts["rows_without_visit"] += 1
                self.row(self.queues[turn].popleft())
        while self.waiting:
            self.release("battle_end")
        self.new_turn(self.last)
        self.close_turn(self.last)
        result = RESULTS.get(self.rec.meta.get("outcome"), "unknown")
        self.event({"k": "end", "result": result})
        self.flush()
        self.send({"v": VERSION, "battle": self.out.tag, "type": "battle_end", "result": result})
        self.out.notes += self.ctx.problems


def build_stream(rec: Recorded) -> Stream:
    """Every row of the battle is sent exactly once; the assert guards it."""
    out = _Builder(rec).build()
    assert len(out.row_ids) == len(rec.rows) == len(set(out.row_ids)), rec.folder
    return out

"""One battle's LIVE message stream as sent to one CG: what went out, so nothing is sent twice, and what a CG that
has seen nothing (a new process, a new Player) needs to get the whole stream again from battle_start.

Order (LIVE sections 1-3): battle_start (with `picks` when the three brought are known), preview_request, then per
decision an `events` message with only the events new since the last one (carrying, before them, the fallbacks of
earlier decisions, as one list when there are several), then the decision_request; at the end, an `events` message
with what is left (the `end` event last), then battle_end. Decisions made before this CG saw the battle are sent
again in place from Request.history, marked `replay` with the order the game took as `played` (LIVE 0.7, S8-2).
When the team brought differs from what CG holds (its preview answer, or the last preview fallback), a preview
fallback rides on the next `events` message: the first one, and again whenever the three brought are known better
(LIVE 0.8, S10-3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pcsa.cg.transport import CGError
from pcsa.decide.api import Request, valid_picks
from pcsa.events.live_ids import VERSION


def message(tag: str, kind: str, **fields) -> dict:
    return {"v": VERSION, "battle": tag, "type": kind, **fields}


def picks_of(picks) -> tuple[int, ...] | None:
    out = tuple(picks) if picks is not None else None
    return out if out is not None and valid_picks(out) else None


@dataclass
class Stream:
    tag: str
    started: bool = False  # battle_start sent
    previewed: bool = False  # preview_request sent
    events: int = 1  # Request.events sent so far (events[0] is battle_start)
    seq: int = -1  # the last decision_request sent
    reported: set[int] = field(default_factory=set)  # seqs whose fallback was sent
    last_event: dict | None = None  # the last event sent, to see a list that changed under us
    picks: tuple[int, ...] | None = None  # CG's answer to the preview
    given: tuple[int, ...] | None = None  # picks CG holds from battle_start or a preview fallback
    brought: tuple[int, ...] | None = None  # the three brought, as last known (kept across a restart)
    answers: dict[int, dict] = field(default_factory=dict)  # seq -> CG's reply
    anomalies: int = 0  # CG's running count in its last reply

    # building messages ------------------------------------------------------------------------------
    def preview(self, start: dict | None, deadline_ms: int | None) -> list[dict]:
        out = self._start(start)
        if not self.previewed:
            out.append(self._preview_request(deadline_ms))
        return out

    def decision(self, request: Request, live: dict | None = None) -> list[dict]:
        """Everything CG has not seen up to and including this decision_request (`live`, else request.live);
        [] when it was sent already."""
        live = live or request.live
        if live is None:
            raise CGError("no decision_request for this decision")
        out, pending = self._catch_up(request.events, request.history, request.team.picks, request.request_id)
        if request.request_id <= self.seq:
            return out
        out += self._events(request.events, pending)
        out.append(live)
        self.seq = request.request_id
        return out

    def end(self, events: tuple[dict, ...], history: tuple[dict, ...], result: str) -> list[dict]:
        """At battle end: what CG has not seen (the whole stream for a new CG), the events after the last request
        with any fallback not yet reported (the `end` event last), then battle_end."""
        out, pending = self._catch_up(tuple(events), tuple(history), self.brought, None)
        return [*out, *self._events(tuple(events), pending), message(self.tag, "battle_end", result=result)]

    def changed(self, request: Request) -> bool:
        """The list no longer holds the last event sent where it was (rebuilt differently): CG may be out of step."""
        if self.last_event is None:
            return False
        return len(request.events) < self.events or request.events[self.events - 1] != self.last_event

    # pieces -----------------------------------------------------------------------------------------
    def _catch_up(self, events: tuple, history: tuple, picks, until: int | None) -> tuple[list[dict], list[dict]]:
        """battle_start and preview_request if not sent yet, then every earlier request this CG has not seen, as
        replays in place; returns them and the fallbacks still to report."""
        self.brought = picks_of(picks) or self.brought
        out = self._start(events[0] if events else None)
        if not self.previewed:
            out.append(self._preview_request(None, replay=True))  # CG answers from battle_start.picks
        pending: list[dict] = []
        if (fallback := self._picks_fallback()) is not None:
            pending.append(fallback)
        for entry in history:
            seq = entry["decision_request"]["seq"]
            if self.seq < seq and (until is None or seq < until):
                out += self._events(events[: entry["events"]], pending)
                out.append(replayed(entry))
                self.seq = seq
            fallback = entry.get("fallback")
            if fallback and fallback["seq"] not in self.reported and fallback["seq"] <= self.seq:
                pending.append(fallback)
        return out, pending

    def _start(self, start: dict | None) -> list[dict]:
        if self.started:
            return []
        if not start or start.get("type") != "battle_start":
            raise CGError("no battle_start for this battle (team sheet not read)")
        self.started, self.events = True, 1
        if self.brought is None:
            return [start]
        self.given = self.brought
        return [{**start, "picks": list(self.brought)}]

    def _preview_request(self, deadline_ms: int | None, replay: bool = False) -> dict:
        self.previewed = True
        extra: dict = {"deadline_ms": deadline_ms} if deadline_ms is not None else {}
        return message(self.tag, "preview_request", **extra, **({"replay": True} if replay else {}))

    def _picks_fallback(self) -> dict | None:
        """A preview fallback (LIVE 0.7 S8-1, 0.8 S10-3) when the team brought is known and is not what CG holds."""
        held = self.given or self.picks
        if self.brought is None or held == self.brought:
            return None
        reason = "overruled" if self.picks is not None else "no preview answer from CG"
        reason = "team brought" if self.given else reason
        self.given = self.brought
        return {"picks": list(self.brought), "reason": reason}

    def _events(self, events: tuple[dict, ...], pending: list[dict]) -> list[dict]:
        """One `events` message with what is new; the pending fallbacks ride on it, all applied before its events."""
        new = list(events[self.events :])
        if new:
            self.events, self.last_event = len(events), new[-1]
        if not new and not pending:
            return []
        out = message(self.tag, "events", events=new)
        if pending:
            out["fallback"] = pending[0] if len(pending) == 1 else list(pending)
            self.reported.update(f["seq"] for f in pending if "seq" in f)
        pending.clear()
        return [out]


def replayed(entry: dict) -> dict:
    """A request CG of this generation never answered: marked `replay`, with the order the game took when known."""
    out = {**entry["decision_request"], "replay": True}
    out.pop("deadline_ms", None)
    played = entry.get("played") or (entry.get("fallback") or {}).get("order")
    if played:
        out["played"] = played
    return out

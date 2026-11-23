"""The CG Player: speaks the LIVE protocol to CG through a transport (pcsa/cg/transport.py) and maps CG's answers
back.

Nothing in src/ imports this module except decide/load.py, lazily. Per battle a Stream keeps what this CG has been
sent; a CG that restarted (new process, new handle, new Player) gets the whole stream again from battle_start.
Anything CG answers that cannot be placed raises CGError, the runner falls back, and the next `events` message
carries that fallback (Request.history). CG's replies: picks and an order with `notes` (kept in the notes as
"cg"); `{"error"}` for a message it cannot read (the call falls back); a stale seq's error with the order CG gave
before (used, the error noted). Errors about other messages are noted as "cg_errors".
`start` makes CG before any battle and checks its status answer (pcsa/cg/startup.py). Per battle the model's
fallbacks (notes.model) are counted and written with CG's status to the battle's cg-summary.json.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from pcsa.cg import slots, startup
from pcsa.cg.messages_log import MessageLog
from pcsa.cg.stream import Stream
from pcsa.cg.transport import WALL, CGError, CGTimeout, Transport, TransportFailed, make_transport
from pcsa.common import log
from pcsa.common.config import CGConfig
from pcsa.decide.api import Answer, OwnTeam, Request, TeamPreviewRequest
from pcsa.live.battle_events import END_RESULTS
from pcsa.live.decision_record import decision_request

MARGIN_S = 0.5  # our answer must be back this long before the runner stops waiting
TAP_RESERVE_S = 2.5  # when the config gives none
ATTEMPTS = 2  # a failed CG is restarted once per call
START_TAG = "cg-start"  # the message log's name for the start-up check


def cg_config(config: Any) -> CGConfig:
    if isinstance(config, CGConfig):
        return config
    agent = getattr(config, "agent", config)
    return agent.cg


class CGPlayer:
    name = "cg"

    def __init__(self, config: Any = None, *, clock=None, run_dir: Path | None = None,
                 transport: Transport | None = None) -> None:  # fmt: skip
        self.cfg = startup.resolved(cg_config(config))
        degrade = getattr(getattr(config, "agent", config), "degrade", None)
        self.tap_reserve = getattr(degrade, "tap_reserve_s", TAP_RESERVE_S)
        self.clock = clock
        self.transport = transport or make_transport(self.cfg, run_dir)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.log = MessageLog(run_dir)
        self.streams: dict[str, Stream] = {}
        self.generation: int | None = None
        self.restarts = 0
        self.start_status: dict | None = None
        self.models: dict[str, dict] = {}  # battle tag -> the model's counts so far
        self.battles: dict[str, dict] = {}  # battle tag -> the model's counts at its end
        self.written: list[Any] = []  # summaries, when there is no run folder

    def params(self) -> dict:
        out = self.cfg.model_dump()
        return {**out, "start_status": self.start_status} if self.start_status is not None else out

    def start(self) -> dict:
        """Make CG now and check its status answer; CGError, and no battle, when it cannot play."""
        try:
            self._sync()
            reply, _ = self.transport.exchange(
                [dict(startup.STATUS)], lambda r: "status" in r or "error" in r, self.cfg.start_timeout_s,
                self._status_recorder(START_TAG),
            )  # fmt: skip
        except Exception as error:  # a bad player, a missing checkpoint, a contract mismatch
            tail = startup.stderr_tail(getattr(self.transport, "stderr_path", None))
            message = f"CG did not start: {error}" + (f"\n{tail}" if tail else "")
            self._started(None, [message])
            raise CGError(message) from error
        problems = startup.status_problems(reply, self.cfg)
        self._started(reply.get("status"), problems)
        if problems:
            raise CGError("CG's start-up check failed: " + "; ".join(problems))
        return reply["status"]

    def _started(self, status: dict | None, problems: list[str]) -> None:
        """The start-up answer, kept for the manifest and the run summary and saved as cg-start.json."""
        self.start_status = {"status": status, "problems": problems}
        self._write(self.run_dir / "cg-start.json" if self.run_dir else None, self.start_status)

    def run_summary(self) -> dict:
        """For the run's summary.json: the start-up check and each battle's model counts."""
        total = sum(b.get("fallbacks", 0) for b in self.battles.values())
        return {"cg": {"start": self.start_status, "model_fallbacks": total, "battles": self.battles}}

    # Player ----------------------------------------------------------------------------------------
    def battle_start(self, battle_tag: str, team: OwnTeam) -> None:
        self.streams.setdefault(battle_tag, Stream(battle_tag))

    def teampreview(self, request: TeamPreviewRequest) -> tuple[tuple[int, int, int], dict]:
        timeout = self._budget(request.deadline)
        start = request.events[0] if request.events else None
        stream = self._stream(request.battle_tag)
        if stream.previewed and stream.picks is not None:
            return stream.picks, {"cached": True}  # type: ignore[return-value]
        reply, notes = self._ask(
            request.battle_tag, lambda s: s.preview(start, round(timeout * 1000)),
            lambda r: "picks" in r or _error_about(r, "preview_request"), timeout,
        )  # fmt: skip
        if "picks" not in reply:
            raise CGError(f"CG error: {reply.get('error')}")
        picks = slots.picks_of(reply)
        self._stream(request.battle_tag).picks = picks
        self._watch(request.battle_tag, reply)
        return picks, {**notes, "cg": reply["notes"]} if isinstance(reply.get("notes"), dict) else notes

    def choose_move(self, request: Request) -> Answer:
        stream, seq = self._stream(request.battle_tag), request.request_id
        if seq in stream.answers:  # asked again for the same decision: the same answer, nothing sent
            return self._answer(stream.answers[seq], request, {"cached": True})
        timeout = self._budget(request.deadline)
        live = dict(request.live or decision_request(request, "forced" if request.offer.forced else "action"))
        live["deadline_ms"] = round(timeout * 1000)
        notes: dict = {}
        if stream.picks is not None and request.team.picks is not None and stream.picks != request.team.picks:
            notes["picks_overruled"] = {"cg": list(stream.picks), "brought": list(request.team.picks)}
        if stream.stale(request.events[0] if request.events else None):  # CG gets the battle again from the start
            notes["preview_corrected"] = list(request.events[0].get("opp_preview") or ())
            self._note(request.battle_tag, {"preview_corrected": notes["preview_corrected"]})
            log.warning("cg_preview_corrected", battle=request.battle_tag, preview=notes["preview_corrected"])
            stream = self.streams[request.battle_tag] = Stream(request.battle_tag, brought=stream.brought)
        if stream.changed(request):
            notes["stream_changed"] = True
            self._note(request.battle_tag, {"stream_changed": seq})
        reply, asked = self._ask(
            request.battle_tag, lambda s: s.decision(request, live),
            lambda r: r.get("seq") == seq or _error_about(r, "decision_request"), timeout,
        )  # fmt: skip
        self._stream(request.battle_tag).answers[seq] = reply
        self._watch(request.battle_tag, reply)
        return self._answer(reply, request, {**notes, **asked})

    def battle_end(self, battle_tag: str, outcome: str, events: tuple = (), history: tuple = ()) -> None:
        """The events after the last decision (the `end` event among them), then battle_end; a CG that restarted
        since gets the whole stream first."""
        stream = self.streams.get(battle_tag)
        if stream is None or not stream.started:
            self.streams.pop(battle_tag, None)
            return
        try:
            self._sync()
            stream = self.streams.pop(battle_tag)
            messages = stream.end(tuple(events), tuple(history), END_RESULTS.get(outcome, "unknown"))
            _, others = self.transport.exchange(messages, None, self.cfg.timeout_s, self._recorder(battle_tag))
        except CGError as error:
            self.streams.pop(battle_tag, None)
            self._note(battle_tag, {"battle_end_failed": str(error)})
            self._battle_summary(battle_tag, {"error": str(error)})
            return
        if errors := _errors(others):
            self._note(battle_tag, {"cg_errors": errors})
        self._battle_summary(battle_tag, self._status(battle_tag))

    def close(self) -> None:
        self.transport.close()

    # Asking ----------------------------------------------------------------------------------------
    def _ask(self, tag: str, build, wanted, timeout: float) -> tuple[dict, dict]:
        """Send what this CG has not seen and wait for the wanted reply; a broken CG is restarted once and
        gets the whole stream again."""
        began = WALL.now()
        for attempt in range(1, ATTEMPTS + 1):
            left = timeout - (WALL.now() - began)
            try:
                self._sync()
                messages = build(self._stream(tag))
                reply, others = self.transport.exchange(messages, wanted, left, self._recorder(tag))
            except TransportFailed as error:
                self.restarts += 1
                self._note(tag, {"restart": str(error), "attempt": attempt})
                log.warning("cg_restart", battle=tag, reason=str(error))
                self.transport.restart(str(error))
                if attempt == ATTEMPTS or timeout - (WALL.now() - began) <= MARGIN_S:
                    raise
                continue
            except CGTimeout as error:
                self._note(tag, {"timeout": str(error)})
                raise
            ms = round((WALL.now() - began) * 1000, 1)
            asked = {"cg_ms": ms, "sent": len(messages), "dropped_replies": len(others), "attempt": attempt}
            if errors := _errors(others):
                asked["cg_errors"] = errors
                log.warning("cg_error", battle=tag, errors=errors)
            return reply, asked
        raise CGError("unreachable")

    # Watching the model -----------------------------------------------------------------------------
    def _watch(self, tag: str, reply: dict) -> None:
        """notes.model into the battle's counts; a warning when CG's anomaly count rises."""
        notes = reply.get("notes")
        cg: dict = notes if isinstance(notes, dict) else {}
        stream, anomalies = self._stream(tag), cg.get("anomalies")
        if isinstance(anomalies, int) and anomalies > stream.anomalies:
            log.warning("cg_anomalies", battle=tag, anomalies=anomalies, new=anomalies - stream.anomalies)
            self._note(tag, {"anomalies": anomalies})
            stream.anomalies = anomalies
        model = cg.get("model")
        if not isinstance(model, dict):
            return
        counts = self.models.setdefault(tag, {"decisions": 0, "fallbacks": 0, "reasons": {}, "ms": []})
        counts["decisions"] += 1
        if isinstance(model.get("ms"), int | float):
            counts["ms"].append(model["ms"])
        if model.get("fallback") is not False:
            reason = str(model.get("fallback"))
            counts["fallbacks"] += 1
            counts["reasons"][reason] = counts["reasons"].get(reason, 0) + 1
            log.warning("cg_model_fallback", battle=tag, seq=reply.get("seq"), reason=reason)

    def _status(self, tag: str) -> dict:
        try:
            reply, _ = self.transport.exchange([{**startup.STATUS, "battle": tag}], lambda r: "status" in r,
                                               self.cfg.timeout_s, self._status_recorder(tag))  # fmt: skip
        except CGError as error:
            return {"error": str(error)}
        return reply["status"] if isinstance(reply.get("status"), dict) else {"error": str(reply)}

    def _battle_summary(self, tag: str, status: dict) -> None:
        """The battle's cg-summary.json: the model's fallbacks and times, and CG's status after battle_end."""
        counts = self.models.pop(tag, None) or {"decisions": 0, "fallbacks": 0, "reasons": {}, "ms": []}
        ms = counts.pop("ms")
        counts.update(ms_median=statistics.median(ms) if ms else None, ms_max=max(ms) if ms else None)
        self.battles[tag] = counts
        if counts["fallbacks"]:
            log.warning("cg_model_fallbacks", battle=tag, **counts)
        path = self.log.path(tag)
        if path is not None:
            named = path.name == "cg-messages.jsonl"
            path = path.with_name("cg-summary.json") if named else path.with_suffix(".summary.json")
        self._write(path, {"battle": tag, "model": counts, "status": status})

    def _write(self, path: Path | None, value: Any) -> None:
        if path is None:
            self.written.append(value)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    def _sync(self) -> None:
        """A CG that (re)started has seen nothing: every battle's stream starts again from battle_start."""
        generation = self.transport.ensure()
        if generation != self.generation:
            self.generation = generation
            self.streams = {tag: Stream(tag, brought=s.brought) for tag, s in self.streams.items()}

    def _answer(self, reply: dict, request: Request, notes: dict) -> Answer:
        error, given = reply.get("error"), reply.get("notes")
        cg: dict = given if isinstance(given, dict) else {}
        if error is not None and not isinstance(reply.get("order"), dict):
            raise CGError(f"CG error: {error}")
        try:
            order = slots.order_of(reply, request, automatic=not cg.get("last_resort"))
        except CGError as problem:
            if cg.get("last_resort"):
                raise CGError(f"CG's last resort ({cg.get('why')}) cannot be placed: {problem}") from problem
            raise
        if cg.get("last_resort") or cg.get("replaced"):
            log.warning("cg_order_flagged", battle=request.battle_tag, seq=request.request_id, notes=cg)
        out = {**notes, "cg": reply.get("notes")} if "notes" in reply else dict(notes)
        if error is not None:
            out["cg_error"] = error  # a stale seq: CG's earlier order for it
        member = slots.active_member(request.team, request.offer)
        if member is not None and request.offer.moves_read:
            mismatches = slots.row_mismatches(request.team, member, request.offer)
            if mismatches:
                out["row_mismatch"] = mismatches
        return Answer(order, out)

    def _budget(self, deadline: float) -> float:
        timeout = self.cfg.timeout_s
        if self.clock is not None:
            timeout = min(timeout, deadline - self.clock.now() - self.tap_reserve - MARGIN_S)
        if timeout <= 0:
            raise CGTimeout("no time left to ask CG")
        return timeout

    def _stream(self, tag: str) -> Stream:
        return self.streams.setdefault(tag, Stream(tag))

    def _recorder(self, tag: str):
        def record(direction: str, value) -> None:
            self.log.write(tag, direction, value, self.transport.generation)

        return record

    def _status_recorder(self, tag: str):
        """A status query is not part of the battle's stream (as in CG's own record): kept as notes."""

        def record(direction: str, value) -> None:
            self._note(tag, {"status_query" if direction == "sent" else f"status_{direction}": value})

        return record

    def _note(self, tag: str, value: dict) -> None:
        self.log.write(tag, "note", value, self.transport.generation)


def _error_about(reply: dict, kind: str) -> bool:
    """CG could not read our request of this kind (its error replies name the message type)."""
    return "error" in reply and reply.get("type") == kind and "seq" not in reply


def _errors(replies: list) -> list[dict]:
    return [r for r in replies if isinstance(r, dict) and "error" in r]

"""A fake CG for tests. It checks every message it gets against LIVE 0.7 (events/live_check, the message shapes of
sections 2-3) and the stream order rules (battle_start first, a battle_start only once per battle and CG start,
`turn` events counting up by one so no stretch of rows comes twice, seq rising, a `turn` request after its `turn`
event and only once per turn unless retry or refused, a fallback only for a seq it was asked, a preview fallback of
three different indices (at any time, LIVE 0.8), replayed requests only before the first live one, battle_end after an
`end` event with the same result), and answers with a simple legal choice from what it has seen: picks 0, 1, 2
(or battle_start's picks); the active member's first move not refused (the screen_offer's when given); on a
replacement the first brought member neither active nor fainted. A replayed request is answered with its
`played` (else this instance's earlier answer, else no order) without asking the player: `asked` lists what was.
`{"type":"status"}` is answered with `FakeCG.player_status` as the status's player_status.

In process: entry "pcsa.cg.fake_cg:FakeCG". As a process (JSON lines on stdin and stdout, one line per message,
`null` for no answer, as CG's own):
    python -m pcsa.cg.fake_cg [--log PATH] [--slow SEQ=SECONDS] [--exit-at SEQ --once MARKER]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from pcsa.common.clock import SystemClock
from pcsa.events.live_check import message_problems
from pcsa.events.live_ids import VERSION

PICKS = [0, 1, 2]


class FakeExit(Exception):
    """The fake stops without answering (a CG crash)."""


@dataclass
class Seen:
    team: list[dict]
    preview: list
    picks: list[int] | None = None
    events: list[dict] = field(default_factory=list)
    turn: int = 0
    seq: int = -1
    asked_turns: set[int] = field(default_factory=set)
    fallbacks: set[int] = field(default_factory=set)
    given: list[int] | None = None  # battle_start.picks
    live: bool = False  # a request not marked replay came
    answers: dict = field(default_factory=dict)  # seq -> order given
    end: str | None = None  # the `end` event's result


class FakeCG:
    slow: ClassVar[dict[int, float]] = {}  # seq -> seconds to wait before answering, once
    exit_at: ClassVar[int | None] = None  # seq at which to stop without answering
    exits: ClassVar[int] = 0  # stops left
    instances: ClassVar[list[FakeCG]] = []
    player_status: ClassVar[dict | None] = None

    def __init__(self, log_path: Path | None = None, **session) -> None:
        unknown = set(session) - {"player", "player_args", "seed", "log_dir"}  # CG's Session keywords, unused here
        if unknown:
            raise TypeError(f"unexpected keyword arguments {sorted(unknown)}")
        self.battles: dict[str, Seen] = {}
        self.problems: list[str] = []
        self.received: list[dict] = []
        self.statuses: list[dict] = []  # status queries
        self.replies: list[dict] = []
        self.script: dict = {}  # seq or message type -> the reply to give instead, once
        self.asked: list[dict] = []  # requests the player was asked
        self.log_path = log_path
        FakeCG.instances.append(self)

    def handle(self, message: dict) -> dict | None:
        if message.get("type") == "status":  # not part of a battle: kept apart from `received`
            self.statuses.append(message)
            held = self.battles.get(message.get("battle", ""))
            return {
                "status": {
                    "battle": message.get("battle"),
                    "held": held is not None,
                    "anomalies": [],
                    "corrections": [],
                    "player": "fake",
                    "player_status": FakeCG.player_status,
                }
            }
        self.received.append(message)
        problems = self.check(message)
        self.problems += [f"{message.get('type')}:{p}" for p in problems]
        reply, kind = None, message.get("type")
        try:
            key = message.get("seq") if kind == "decision_request" else kind
            if key in self.script:
                reply = self.script.pop(key)
            elif kind in ("preview_request", "decision_request"):
                reply = self.answer(message)
        finally:
            self._log({"pid": os.getpid(), "message": message, "problems": problems, "reply": reply})
        if reply is not None:
            self.replies.append(reply)
        return reply

    # checking -----------------------------------------------------------------------------------------
    def check(self, msg: dict) -> list[str]:
        out = message_problems(msg)
        out += [] if msg.get("v") == VERSION else ["v"]
        kind, b = msg.get("type"), self.battles.get(msg.get("battle", ""))
        if kind == "battle_start":
            out += ["order:battle_start twice"] if b is not None else []
            team, preview = list(msg.get("own_team") or []), list(msg.get("opp_preview") or [])
            given = list(msg["picks"]) if isinstance(msg.get("picks"), list) else None
            self.battles[msg.get("battle", "")] = Seen(team, preview, given=given)
        elif b is None:
            out.append("order:before battle_start")
        elif kind == "preview_request":
            out += ["order:preview twice"] if b.picks is not None and not msg.get("replay") else []
        elif kind == "events":
            out += self._check_events(b, msg)
        elif kind == "decision_request":
            out += self._check_request(b, msg)
        elif kind == "battle_end" and b.end != msg.get("result"):
            out.append(f"order:battle_end {msg.get('result')} after end {b.end}")
        return out

    @staticmethod
    def _check_events(b: Seen, msg: dict) -> list[str]:
        out = []
        for event in msg.get("events") or []:
            if event.get("k") == "end":
                b.end = event.get("result")
            if event.get("k") == "turn":
                if event.get("n") != b.turn + 1:
                    out.append(f"order:turn {event.get('n')} after {b.turn}")
                b.turn = event.get("n", b.turn)
            b.events.append(event)
        fallback = msg.get("fallback")
        for one in fallback if isinstance(fallback, list) else [fallback] if isinstance(fallback, dict) else []:
            if "picks" in one:
                if len(set(one["picks"])) != len(PICKS):
                    out.append("order:preview fallback")
                b.picks, b.given = list(one["picks"]), list(one["picks"])
                b.fallbacks.add(-1)
                continue
            seq = one.get("seq")
            if not isinstance(seq, int) or seq > b.seq or seq in b.fallbacks:
                out.append(f"order:fallback {seq}")
            b.fallbacks.add(seq)  # type: ignore[arg-type]
        return out

    @staticmethod
    def _check_request(b: Seen, msg: dict) -> list[str]:
        out, seq = [], msg.get("seq")
        if not isinstance(seq, int) or seq <= b.seq:
            out.append(f"order:seq {seq} after {b.seq}")
        b.seq = seq if isinstance(seq, int) else b.seq
        if msg.get("replay") or "played" in msg:
            out += ["order:replay after a live request"] if b.live else []
        else:
            b.live = True
        if msg.get("kind") == "turn":
            if b.turn == 0:
                out.append("order:turn request before a turn event")
            elif b.turn in b.asked_turns and not (msg.get("retry") or msg.get("refused")):
                out.append(f"order:second request for turn {b.turn}")
            b.asked_turns.add(b.turn)
        refused = (msg.get("refused") or {}).get("order")
        last = next((e for e in reversed(b.events) if e.get("k") == "refused"), None)
        if refused is not None and (last is None or last.get("order") != refused):
            out.append("order:refused without its event")
        return out

    # answering ----------------------------------------------------------------------------------------
    def answer(self, msg: dict) -> dict:
        b = self.battles.get(msg.get("battle", ""))
        if b is None:
            return {"error": "unknown battle"}
        if msg["type"] == "preview_request":
            return self._picks(b, msg)
        seq = msg.get("seq")
        if msg.get("replay") or "played" in msg:
            order = msg.get("played") or b.answers.get(seq)
            if not order:
                return {"seq": seq, "replayed": True}
            return {"seq": seq, "order": order, "notes": {"replayed": True}}
        if seq in FakeCG.slow:
            SystemClock().sleep(FakeCG.slow.pop(seq))
        if seq == FakeCG.exit_at and FakeCG.exits > 0:
            FakeCG.exits -= 1
            raise FakeExit(f"stopped at seq {seq}")
        self.asked.append(msg)
        b.answers[seq] = self.choose(b, msg)
        return {"seq": seq, "order": b.answers[seq], "notes": {"fake": True}}

    def _picks(self, b: Seen, msg: dict) -> dict:
        played = (msg.get("played") or {}).get("picks")
        if b.given is not None or played:
            b.picks = list(b.given or played)
            return {"picks": list(b.picks), "notes": {"given": True}}
        if msg.get("replay"):
            return {"picks": list(b.picks), "replayed": True} if b.picks else {"replayed": True}
        self.asked.append(msg)
        b.picks = list(PICKS)
        return {"picks": list(PICKS)}

    def choose(self, b: Seen, msg: dict) -> dict:
        picks = b.picks or PICKS
        brought = [b.team[i]["species"] for i in picks if i < len(b.team)]
        active = next((e["mon"] for e in reversed(b.events) if e.get("k") == "enter" and e.get("side") == "own"),
                      brought[0] if brought else None)  # fmt: skip
        fainted = {e.get("mon") for e in b.events if e.get("k") == "faint" and e.get("side") == "own"}
        if msg.get("kind") == "turn":
            refused = self._refused_this_turn(b, msg)
            sets = {s["species"]: s.get("moves") or [] for s in b.team}
            moves = (msg.get("screen_offer") or {}).get("moves") or sets.get(active or "", [])
            for move in moves:
                if move not in refused:
                    return {"kind": "move", "move": move, "mega": False}
        mon = next((s for s in brought if s != active and s not in fainted), None)
        return {"kind": "switch", "mon": mon} if mon else {"kind": "none"}

    @staticmethod
    def _refused_this_turn(b: Seen, msg: dict) -> set[str | None]:
        out = {((msg.get("refused") or {}).get("order") or {}).get("move")}
        for event in reversed(b.events):
            if event.get("k") == "turn":
                break
            if event.get("k") == "refused":
                out.add((event.get("order") or {}).get("move"))
        return out

    def _log(self, line: dict) -> None:
        if self.log_path is not None:
            with Path(self.log_path).open("a", encoding="utf-8") as out:
                out.write(json.dumps(line) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pcsa.cg.fake_cg")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--slow", action="append", default=[], help="SEQ=SECONDS")
    parser.add_argument("--exit-at", type=int)
    parser.add_argument("--once", type=Path, help="marker file: stop only when it does not exist yet")
    args = parser.parse_args(argv)
    FakeCG.slow = {int(k): float(v) for k, v in (s.split("=") for s in args.slow)}
    FakeCG.exit_at = args.exit_at
    FakeCG.exits = 0 if args.once is not None and args.once.exists() else 1
    cg = FakeCG(args.log)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            reply = cg.handle(json.loads(line))
        except FakeExit:
            if args.once is not None:
                args.once.write_text("stopped", encoding="utf-8")
            return 3
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""What `pcsa watch` keeps from a run's events.jsonl: record, rating, battles, warnings and the recent timeline.

The file is read incrementally (only the bytes added since the last pass, and only up to the last complete
line), so a run that is being written is followed without re-reading it and a half-written line is left for
the next pass. Every field of an event is optional: an older or unknown event shape is skipped, never an error.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

NOISY = {"observation", "disconnect_tap"}  # too frequent for the timeline
ALERT_LEVELS = {"WARNING", "ERROR", "CRITICAL"}
ALERT_EVENTS = {"disconnect_recovered", "session_restart", "device_reconnect", "device_reconnected"}
OUTCOME_KEYS = {"win": "wins", "loss": "losses", "draw": "draws", "opponent_forfeit": "forfeits"}
TIMELINE = 40
WARNINGS = 200
DETAIL_WIDTH = 160
META = ("ts", "level", "event")
TEAM_KEYS = ("battle", "kind", "team", "names")
WON = ("win", "opponent_forfeit")
RECENT = 10


def _number(value) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _rate(outcomes: list[str]) -> dict:
    wins = sum(1 for o in outcomes if o in WON)
    return {"battles": len(outcomes), "wins": wins, "rate": round(wins / len(outcomes), 4) if outcomes else None}


def detail_of(event: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in event.items() if k not in META)[:DETAIL_WIDTH]


class EventScan:
    """The running summary of events.jsonl; `refresh` folds in whatever was appended since the last call."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._reset()

    def _reset(self) -> None:
        self.offset = 0
        self.timeline: deque = deque(maxlen=TIMELINE)
        self.warnings: deque = deque(maxlen=WARNINGS)
        self.warning_count = 0
        self.disconnects = 0
        self.latest_obs: dict = {}
        self.battles: dict[int, dict] = {}
        self.record = {"wins": 0, "losses": 0, "draws": 0, "forfeits": 0}
        self.rank: dict = {}
        self.ratings: list[dict] = []
        self.timings: dict[tuple[int, int], dict] = {}
        self.last: dict = {}
        self._ended: int | None = None  # the battle a rating change now belongs to

    def refresh(self) -> None:
        try:
            size = self.path.stat().st_size
            if size < self.offset:  # the file was replaced: start over
                self._reset()
            if size == self.offset:
                return
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                data = handle.read()
        except OSError:
            return
        end = data.rfind(b"\n") + 1  # a line still being written waits for its newline
        self.offset += end
        for raw in data[:end].splitlines():
            try:
                event = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                self.take(event)

    def _battle(self, event: dict) -> dict | None:
        index = event.get("battle")
        if not isinstance(index, int) or isinstance(index, bool):
            return None
        return self.battles.setdefault(
            index,
            {
                "index": index,
                "outcome": None,
                "roster": None,
                "opponent": [],
                "picks": [],
                "preview": None,
                "turn": None,
                "decisions": 0,
                "disconnects": 0,
                "rating_before": _number(self.rank.get("rating")),
                "rating_after": None,
                "started": event.get("ts"),
            },
        )

    def take(self, event: dict) -> None:
        self.last = event
        kind = event.get("event")
        handler = getattr(self, f"_on_{kind}", None) if isinstance(kind, str) else None
        if handler is not None:
            handler(event)
        if event.get("level") in ALERT_LEVELS or kind in ALERT_EVENTS:
            self.warning_count += 1
            self.warnings.append(
                {
                    "ts": event.get("ts"),
                    "level": event.get("level"),
                    "kind": kind,
                    "battle": event.get("battle"),
                    "detail": detail_of(event),
                }
            )
        if kind not in NOISY:
            self.timeline.append({"ts": event.get("ts"), "label": kind, "detail": detail_of(event), "kind": kind})

    def _on_observation(self, event: dict) -> None:
        self.latest_obs = event

    def _on_battle_start(self, event: dict) -> None:
        battle = self._battle(event)
        if battle is not None:
            battle["roster"] = event.get("roster")
            self._ended = None

    def _on_team_preview(self, event: dict) -> None:
        battle = self._battle(event)
        if battle is not None and isinstance(event.get("opponent"), list):
            battle["opponent"] = [str(name) for name in event["opponent"]]

    def _on_decision(self, event: dict) -> None:
        battle = self._battle(event)
        if battle is None:
            return
        if event.get("kind") == "team":
            names = event.get("names")
            battle["picks"] = [str(n) for n in names] if isinstance(names, list) else []
            battle["preview"] = {
                "ts": event.get("ts"),
                "names": battle["picks"],
                "notes": {k: v for k, v in event.items() if k not in META and k not in TEAM_KEYS},
            }
        else:
            battle["decisions"] += 1

    def _on_turn(self, event: dict) -> None:
        battle = self._battle(event)
        if battle is None:
            return
        battle["turn"] = event.get("turn", battle["turn"])
        visit = event.get("visit")
        if isinstance(visit, int):
            self.timings[(battle["index"], visit)] = {
                "turn_s": _number(event.get("elapsed_s")),
                "log_s": _number(event.get("log_s")),
                "tap_s": _number(event.get("first_capture_to_last_tap_s")),
            }

    def _on_battle_end(self, event: dict) -> None:
        battle = self._battle(event)
        if battle is None:
            return
        outcome = event.get("outcome")
        battle["outcome"] = outcome
        battle["outcome_source"] = event.get("source")
        battle["degrade"] = event.get("degrade") if isinstance(event.get("degrade"), dict) else None
        self._ended = battle["index"]
        key = OUTCOME_KEYS.get(str(outcome))
        if key:
            self.record[key] += 1

    def _on_rank(self, event: dict) -> None:
        self.rank = {k: event.get(k) or self.rank.get(k) for k in ("tier", "rank", "placement", "rating")}
        rating = _number(event.get("rating"))
        if rating is None:
            return
        if self._ended is not None:
            self.battles[self._ended]["rating_after"] = rating
        if not self.ratings or self.ratings[-1]["rating"] != rating:
            self.ratings.append({"ts": event.get("ts"), "rating": rating, "battle": self._ended})

    def _on_disconnect_start(self, event: dict) -> None:
        self.disconnects += 1
        battle = self._battle(event)
        if battle is not None:
            battle["disconnects"] += 1

    def rating_view(self) -> dict | None:
        """Tier, rank and rating now, the rating the run started at, and one point per change."""
        if not self.rank and not self.ratings:
            return None
        start = self.ratings[0]["rating"] if self.ratings else None
        now = self.ratings[-1]["rating"] if self.ratings else None
        return {
            "tier": self.rank.get("tier"),
            "rank": self.rank.get("rank"),
            "rating": now,
            "start": start,
            "delta": round(now - start, 3) if now is not None and start is not None else None,
            "series": list(self.ratings),
        }

    def winrate_view(self) -> dict:
        """Finished battles and the share won (a forfeit by the opponent is a win), over the run and over its
        last 10 battles, and the current run of equal results (`streak`: +2 two wins, -3 three losses)."""
        outcomes = [str(b["outcome"]) for _, b in sorted(self.battles.items()) if b.get("outcome") in OUTCOME_KEYS]
        streak = 0
        for outcome in reversed(outcomes):
            step = 1 if outcome in WON else -1 if outcome == "loss" else 0
            if step == 0 or (streak and (step > 0) != (streak > 0)):
                break
            streak += step
        return {**_rate(outcomes), "recent": _rate(outcomes[-RECENT:]), "recent_size": RECENT, "streak": streak}

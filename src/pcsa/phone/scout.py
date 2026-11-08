"""`pcsa scout`: watch while a person walks through the game; report what differs from what we know.

Read-only on the phone: it captures and runs OCR, and sends no tap. The person drives through the
checklist shown in the console; the report lists unseen checklist items, unknown screens grouped
by their text, anchor labels that moved, roster names and items the game data does not know, and
the game version: what to adjust before a new season.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pcsa.common import log, paths
from pcsa.common.clock import Clock
from pcsa.common.config import Config
from pcsa.common.types import Observation
from pcsa.phone.ocr import REFERENCE, TextBox
from pcsa.phone.scout_vocab import Known, item_candidates, name_candidates
from pcsa.phone.texts import Texts

DUPLICATE_DIFF = 2.5
DRIFT_WARN = 0.02
ANCHORS = {
    "begin_matchmaking": (r"begin matchmaking", (1300, 905, 1900, 962)),
    "fight": (r"fight", (2045, 645, 2145, 735)),
    "pokemon_button": (r"pokemon", (1960, 855, 2055, 945)),
    "view_log": (r"view log", (160, 362, 400, 394)),
    "move_back": (r"back", (1990, 1040, 2200, 1070)),
    "log_back": (r"back", (1710, 1002, 1910, 1030)),
    "party_back": (r"back", (1880, 1000, 2060, 1045)),
    "team_done": (r"done", (560, 930, 700, 968)),
    "continue_battling": (r"continue battling", (1560, 965, 1890, 1020)),
}
VOCAB_ITEMS = ("roster", "summary", "team_details", "battle_pass")


def _has(norms: list[str], pattern: str) -> bool:
    rx = re.compile(pattern)
    return any(rx.search(n) for n in norms)


def _screen(name: str):
    return lambda s, n: s == name


def _text(pattern: str):
    return lambda s, n: _has(n, pattern)


def _popup(s: str, n: list[str]) -> bool:
    return s in ("unknown", "dialog") and _has(n, r"(season|regulation|notice|news|update|maintenance)")


def _battle_data(s: str, n: list[str]) -> bool:
    return _has(n, r"battle data") or (_has(n, r"season") and _has(n, r"(placement|final|result)"))


CHECKLIST = (
    ("title", "Title screen", _text(r"(tap|touch|press).*(start|begin|screen)")),
    ("popup", "Season / news popup (capture before dismissing)", _popup),
    ("home", "Home menu", _screen("home")),
    ("battle_modes", "Battle menu (Ranked Battles)", _screen("battle_modes")),
    ("ranked_formats", "Ranked formats (Single / Double)", _screen("ranked_formats")),
    ("lobby", "Lobby (Begin Matchmaking)", _screen("lobby")),
    ("battle_data", "Battle Data / season results", _battle_data),
    ("team_list", "Team list", _screen("team_list")),
    ("team_details", "Team details page", _screen("team_details")),
    ("team_stats", "Team stats page", _screen("team_stats")),
    ("roster", "Roster Info (scroll the whole list)", _text(r"roster")),
    ("summary", "One Pokemon's summary", _text(r"(summary|base stats|learnable|move ?set)")),
    ("battle_pass", "Battle Pass page (do not buy)", _text(r"(battle pass|premium)")),
    ("matchmaking", "Matchmaking", _screen("matchmaking")),
    ("team_select", "Team selection", _screen("team_select")),
    ("action", "Battle action screen", _screen("action")),
    ("moves", "Move list (tap Fight)", _screen("moves")),
    ("party", "Party menu (tap Pokemon)", _screen("party")),
    ("log", "Battle log (tap View Log)", _screen("log")),
    ("result", "Result (WON / LOST)", _screen("result")),
    ("rewards", "Rewards", _screen("rewards")),
    ("options", "Post-battle options", _screen("options")),
)


def anchor_drift(screen: str, boxes) -> list[dict]:
    out = []
    for key, (pattern, rect) in ANCHORS.items():
        rx = re.compile(pattern)
        cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
        for box in boxes:
            if box.confidence < 0.8 or not rx.fullmatch(box.norm):
                continue
            bx, by = box.center
            if abs(bx - cx) > REFERENCE[0] * 0.25 or abs(by - cy) > REFERENCE[1] * 0.25:
                continue
            inside = rect[0] <= bx <= rect[2] and rect[1] <= by <= rect[3]
            drift = {"dx": (bx - cx) / REFERENCE[0], "dy": (by - cy) / REFERENCE[1], "inside": inside}
            out.append({"anchor": key, "screen": screen, **drift})
    return out


def thumbnail(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (96, 44), interpolation=cv2.INTER_AREA).astype(np.int16)


def signature(boxes: tuple[TextBox, ...], limit: int = 6) -> str:
    top = sorted((b for b in boxes if b.confidence >= 0.8), key=lambda b: -b.confidence)[:limit]
    return " | ".join(sorted(b.norm for b in top)) or "(no text)"


def _rank_of(rank: dict) -> dict:
    return {k: rank.get(k) for k in ("tier", "rank", "placement", "rating")}


@dataclass
class Scout:
    out: Path
    known: Known
    seen: dict[str, str | None] = field(default_factory=lambda: {k: None for k, _, _ in CHECKLIST})
    frames: int = 0
    saved: int = 0
    screens: Counter = field(default_factory=Counter)
    first_frame: dict = field(default_factory=dict)
    unknown_groups: dict = field(default_factory=lambda: defaultdict(list))
    drift: list = field(default_factory=list)
    species_seen: set = field(default_factory=set)
    name_unmatched: Counter = field(default_factory=Counter)
    item_unmatched: Counter = field(default_factory=Counter)
    ranks: list = field(default_factory=list)
    previous_thumb: np.ndarray | None = None
    previous_screen: str | None = None

    def __post_init__(self) -> None:
        (self.out / "frames").mkdir(parents=True, exist_ok=True)
        (self.out / "unknown").mkdir(parents=True, exist_ok=True)

    def record(self, obs: Observation, image: np.ndarray) -> str | None:
        """One captured frame; returns the saved name, or None for a repeat of the previous frame."""
        self.frames += 1
        thumb = thumbnail(image)
        screen = str(obs.screen)
        repeat = (
            self.previous_thumb is not None and float(np.mean(np.abs(self.previous_thumb - thumb))) < DUPLICATE_DIFF
        )
        self.previous_thumb = thumb
        if repeat and screen == self.previous_screen:
            return None
        self.previous_screen = screen
        name = f"{self.saved:05d}-{screen}.jpg"
        cv2.imwrite(str(self.out / "frames" / name), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.saved += 1
        self.screens[screen] += 1
        self.first_frame.setdefault(screen, name)
        norms = [b.norm for b in Texts(obs.texts, 0.8).boxes]
        line = {"frame": name, **obs.record(), "boxes": [b.record() for b in obs.texts]}
        with (self.out / "ocr.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._checklist(screen, norms, name)
        if screen == "unknown":
            cv2.imwrite(str(self.out / "unknown" / name), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.unknown_groups[signature(obs.texts)].append(name)
        self.drift.extend(anchor_drift(screen, obs.texts))
        rank = obs.fields.get("own_rank")
        if rank and not any(r["rank"] == _rank_of(rank) for r in self.ranks):
            self.ranks.append({"frame": name, "screen": screen, "rank": _rank_of(rank)})
        if any(k in VOCAB_ITEMS and match(screen, norms) for k, _, match in CHECKLIST):
            matched, unmatched = name_candidates(obs.texts, self.known)
            self.species_seen |= matched
            self.name_unmatched.update(unmatched)
            self.item_unmatched.update(item_candidates(obs.texts, self.known))
        return name

    def _checklist(self, screen: str, norms: list[str], name: str) -> None:
        for key, label, match in CHECKLIST:
            if self.seen[key] is None and match(screen, norms):
                self.seen[key] = name
                log.event("scout_seen", item=label, frame=name)

    def drift_summary(self) -> list[dict]:
        by_anchor: dict = defaultdict(list)
        for d in self.drift:
            by_anchor[(d["anchor"], d["screen"])].append(d)
        rows = []
        for (anchor, screen), items in sorted(by_anchor.items()):
            dx, dy = statistics.median(r["dx"] for r in items), statistics.median(r["dy"] for r in items)
            inside = sum(r["inside"] for r in items) / len(items)
            warn = abs(dx) > DRIFT_WARN or abs(dy) > DRIFT_WARN or inside < 0.5
            rows.append(
                {
                    "anchor": anchor,
                    "screen": screen,
                    "samples": len(items),
                    "dx": round(dx, 4),
                    "dy": round(dy, 4),
                    "inside": round(inside, 2),
                    "warn": warn,
                }
            )
        return rows

    def summary(self, facts: dict) -> dict:
        groups = sorted(self.unknown_groups.items(), key=lambda kv: -len(kv[1]))
        return {
            "schema": "pcsa-scout/1",
            "captured": self.frames,
            "saved": self.saved,
            "device": facts,
            "screens": dict(self.screens),
            "checklist": [{"key": k, "label": lbl, "frame": self.seen[k]} for k, lbl, _ in CHECKLIST],
            "unknown_groups": [{"signature": s, "frames": fr} for s, fr in groups],
            "anchor_drift": self.drift_summary(),
            "species_seen": sorted(self.species_seen),
            "name_unmatched": self.name_unmatched.most_common(),
            "item_unmatched": self.item_unmatched.most_common(),
            "ranks": self.ranks,
        }

    def write_report(self, facts: dict) -> Path:
        s = self.summary(facts)
        (self.out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = [
            f"# Season scout {datetime.now():%Y-%m-%d %H:%M}",
            "",
            f"Device {facts}. Captured {self.frames} frames, saved {self.saved} distinct.",
            "",
        ]
        lines += ["## 1. Checklist", "", "| Item | Frame |", "|---|---|"]
        lines += [f"| {lbl} | {self.seen[k] or 'NOT SEEN'} |" for k, lbl, _ in CHECKLIST]
        lines += ["", "## 2. Screens seen", "", "| Screen | Frames | First |", "|---|---|---|"]
        lines += [
            f"| {k} | {v} | {self.first_frame[k]} |" for k, v in sorted(self.screens.items(), key=lambda kv: -kv[1])
        ]
        lines += ["", "## 3. Unknown screens (grouped by text)", ""]
        lines += [
            f"- {len(g['frames'])} frame(s), e.g. unknown/{g['frames'][0]}: `{g['signature']}`"
            for g in s["unknown_groups"]
        ] or ["None."]
        lines += [
            "",
            "## 4. Anchor drift",
            "",
            "| Anchor | Screen | Samples | dx | dy | Inside | Warn |",
            "|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {d['anchor']} | {d['screen']} | {d['samples']} | {d['dx']:+.3f} | {d['dy']:+.3f} | {d['inside']:.0%} | {'YES' if d['warn'] else ''} |"
            for d in s["anchor_drift"]
        ] or ["No anchors seen."]
        lines += [
            "",
            "## 5. Roster",
            "",
            f"Known species matched: {len(self.species_seen)} of {len(self.known.species)}.",
            "",
            "Unknown name-like tokens:",
            "",
        ]
        lines += [f"- {t} (x{n})" for t, n in self.name_unmatched.most_common(60)] or ["- none"]
        lines += ["", "## 6. Items not in the vocabulary", ""]
        lines += [f"- {t} (x{n})" for t, n in self.item_unmatched.most_common(40)] or ["- none"]
        lines += ["", "## 7. Rank text", ""]
        lines += [f"- {r['screen']} ({r['frame']}): {r['rank']}" for r in self.ranks] or ["- none read"]
        path = self.out / "report.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def scout_command(config: Config, interval_s: float, clock: Clock | None = None) -> int:
    from pcsa.phone.commands import build_phone

    out = paths.output_dir() / "scout" / f"{datetime.now():%Y%m%d-%H%M%S}"
    log.configure(out)
    phone = build_phone(config, clock, require_profile=False)
    scout = Scout(out, Known.load())
    facts = phone.device.facts()
    log.event("scout_start", **facts, out=str(out))
    for _, label, _ in CHECKLIST:
        log.event("scout_item", item=label)
    try:
        while True:
            started = phone.clock.now()
            obs, frame = phone.observe("scout")
            scout.record(obs, frame.image)
            phone.clock.sleep(max(0.0, interval_s - (phone.clock.now() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        path = scout.write_report(facts)
        log.event("scout_report", path=str(path), not_seen=[lbl for k, lbl, _ in CHECKLIST if scout.seen[k] is None])
    return 0

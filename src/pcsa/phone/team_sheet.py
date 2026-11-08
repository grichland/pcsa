"""Our own team, read from the six-card Moves & More and Stats pages before matchmaking.

Reading never raises for an unreadable field: the sheet carries a status per field and lists its
issues, and the caller decides how much it needs. Species and form resolution is not done here
(that is game knowledge, in `pcsa.battle`).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace

import cv2
import numpy as np

from pcsa.battle.stats import POINTS_MAX, POINTS_TOTAL, hp_value, stat_value
from pcsa.phone import fields as f
from pcsa.phone.ocr import TextBox, TextReader, normalize_text
from pcsa.phone.texts import Texts

CARD_ORIGINS = tuple((x, y) for y in (292, 510, 728) for x in (390, 1195))
FIELD_OFFSETS = {
    "name": (80, 8, 305, 58),
    "ability": (90, 58, 420, 104),
    "item": (90, 104, 420, 150),
    "moves": tuple((500, 8 + 44 * i, 738, 52 + 44 * i) for i in range(4)),
}
EMPTY = {"-", "--", "---", "none", "no item", "no held item", "empty"}
STAT_ROWS = ((51, 91), (98, 138), (143, 183))
STAT_COLUMNS = (
    {"value": (205, 271), "points": (318, 362), "label": (80, 205)},
    {"value": (550, 623), "points": (670, 714), "label": (470, 550)},
)
STAT_LAYOUT = (("hp", 0, 0), ("atk", 0, 1), ("def", 0, 2), ("spa", 1, 0), ("spd", 1, 1), ("spe", 1, 2))
ARROW_PIXELS = 30
MEMBERS = 6
NATURES = {
    ("atk", "def"): "Lonely", ("atk", "spa"): "Adamant", ("atk", "spd"): "Naughty", ("atk", "spe"): "Brave",
    ("def", "atk"): "Bold", ("def", "spa"): "Impish", ("def", "spd"): "Lax", ("def", "spe"): "Relaxed",
    ("spa", "atk"): "Modest", ("spa", "def"): "Mild", ("spa", "spd"): "Rash", ("spa", "spe"): "Quiet",
    ("spd", "atk"): "Calm", ("spd", "def"): "Gentle", ("spd", "spa"): "Careful", ("spd", "spe"): "Sassy",
    ("spe", "atk"): "Timid", ("spe", "def"): "Hasty", ("spe", "spa"): "Jolly", ("spe", "spd"): "Naive",
}  # fmt: skip
NEUTRAL_NATURE = "Serious"


@dataclass(frozen=True)
class ReadField:
    value: str | None
    status: str  # known | none | unknown
    confidence: float | None
    bounds: tuple[int, int, int, int]
    raw: str | None = None


@dataclass(frozen=True)
class MemberStats:
    name: str | None
    stats: dict[str, int | None]
    points: dict[str, int | None]
    raised: str | None
    lowered: str | None
    issues: tuple[str, ...] = ()
    inferred: tuple[str, ...] = ()

    @property
    def nature(self) -> str | None:
        if self.raised is None and self.lowered is None:
            return NEUTRAL_NATURE
        return NATURES.get((self.raised or "", self.lowered or ""))


@dataclass(frozen=True)
class RosterMember:
    roster_index: int
    name: ReadField
    ability: ReadField
    item: ReadField
    moves: tuple[ReadField, ...]
    stats: MemberStats | None = None
    form: str | None = None  # form named by the card sprite, when the name alone is ambiguous

    def record(self) -> dict:
        data = {
            "roster_index": self.roster_index,
            "name": self.name.value,
            "form": self.form,
            "ability": self.ability.value,
            "ability_status": self.ability.status,
            "item": self.item.value,
            "item_status": self.item.status,
            "moves": [{"slot": i, "name": m.value, "status": m.status} for i, m in enumerate(self.moves)],
        }
        if self.stats is not None:
            data.update(
                stats=self.stats.stats,
                stat_points=self.stats.points,
                nature=self.stats.nature,
                stat_inferences=self.stats.inferred,
            )
        return data


@dataclass(frozen=True)
class TeamSheet:
    team_name: str | None
    members: tuple[RosterMember, ...]

    @property
    def issues(self) -> tuple[str, ...]:
        problems = []
        if len(self.members) != MEMBERS:
            problems.append("expected six roster members")
        names = [normalize_text(m.name.value or "") for m in self.members]
        if not all(names) or len(set(names)) != len(names):
            problems.append("roster names are missing or duplicated")
        for m in self.members:
            row = m.roster_index + 1
            if m.name.status != "known":
                problems.append(f"row {row}: name unknown")
            if m.item.status == "unknown":
                problems.append(f"row {row}: item unknown")
            if any(move.status == "unknown" for move in m.moves):
                problems.append(f"row {row}: move position unknown")
            if not any(move.status == "known" for move in m.moves):
                problems.append(f"row {row}: no known move")
        return tuple(problems)

    @property
    def stats_issues(self) -> tuple[str, ...]:
        out = []
        for m in self.members:
            if m.stats is None:
                out.append(f"row {m.roster_index + 1}: stats not read")
            else:
                out += [f"row {m.roster_index + 1}: {issue}" for issue in m.stats.issues]
        return tuple(out)

    @property
    def completeness(self) -> float:
        """Share of the 42 name/ability/item/move fields that were read."""
        fields = [m.name for m in self.members] + [m.ability for m in self.members] + [m.item for m in self.members]
        fields += [move for m in self.members for move in m.moves]
        return sum(fld.status != "unknown" for fld in fields) / max(1, len(fields))

    def record(self) -> dict:
        return {
            "team_name": self.team_name,
            "members": [m.record() for m in self.members],
            "issues": list(self.issues),
        }


def empty_item_slot(frame: np.ndarray, origin: tuple[int, int]) -> bool:
    x, y = origin
    strip = frame[y + 104 : y + 150, x + 30 : x + 420]
    if strip.shape != (46, 390, 3):
        return False
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    background = (
        (hsv[..., 0] >= 110)
        & (hsv[..., 0] <= 160)
        & (hsv[..., 1] >= 50)
        & (hsv[..., 1] <= 180)
        & (hsv[..., 2] >= 90)
        & (hsv[..., 2] <= 240)
    )
    bright_ink = (strip.min(axis=2) > 170) & (np.ptp(strip.astype(int), axis=2) < 75)
    return bool(background.mean() >= 0.995 and np.count_nonzero(bright_ink) <= 2)


def _offset(origin: tuple[int, int], rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y = origin
    return (rect[0] + x, rect[1] + y, rect[2] + x, rect[3] + y)


class TeamReader:
    def __init__(self, reader: TextReader) -> None:
        self.reader = reader
        self.min_confidence = reader.min_confidence

    def _field(self, frame, origin, rect, allow_empty=False, item_slot=False) -> ReadField:
        bounds = _offset(origin, rect)
        left, top, right, bottom = bounds
        result = self.reader.line(frame[top:bottom, left:right])
        if item_slot and empty_item_slot(frame, origin):
            return ReadField(None, "none", None, bounds, result[0] if result else None)
        if result is None:
            return ReadField(None, "unknown", None, bounds)
        raw, confidence = result
        value = raw.strip()
        if confidence < self.min_confidence or not value:
            return ReadField(None, "unknown", confidence, bounds, raw)
        if normalize_text(value) in EMPTY:
            return ReadField(None, "none" if allow_empty else "unknown", confidence, bounds, raw)
        if not re.search(r"[A-Za-z]{2}", value):
            return ReadField(None, "unknown", confidence, bounds, raw)
        return ReadField(value, "known", confidence, bounds, raw)

    def read_details(self, frame: np.ndarray, boxes: list[TextBox]) -> TeamSheet | None:
        """The Moves & More page -> sheet, or None when the frame is not that page."""
        t = Texts(boxes, self.min_confidence)
        if f.team_page(t) != "details":
            return None
        labels = t.find(r"team\s+\d+", (600, 70, 1200, 180))
        members = tuple(
            RosterMember(
                i,
                self._field(frame, origin, FIELD_OFFSETS["name"]),
                self._field(frame, origin, FIELD_OFFSETS["ability"]),
                self._field(frame, origin, FIELD_OFFSETS["item"], True, item_slot=True),
                tuple(self._field(frame, origin, rect, True) for rect in FIELD_OFFSETS["moves"]),
            )
            for i, origin in enumerate(CARD_ORIGINS)
        )
        return TeamSheet(labels[0].text.strip() if len(labels) == 1 else None, members)

    def _digits(self, frame, rect) -> tuple[int, float] | None:
        """A 1-3 digit reading at any confidence; the caller decides whether to trust it."""
        left, top, right, bottom = rect
        result = self.reader.line(frame[top:bottom, left:right])
        if result is None or not re.fullmatch(r"\d{1,3}", result[0].strip()):
            return None
        return int(result[0].strip()), result[1]

    def _number(self, frame, rect) -> int | None:
        reading = self._digits(frame, rect)
        return reading[0] if reading and reading[1] >= self.min_confidence else None

    def read_stats(
        self,
        frame: np.ndarray,
        boxes: list[TextBox],
        sheet: TeamSheet,
        base_stats: Callable[[RosterMember], dict | None],
    ) -> TeamSheet | None:
        """Add the Stats page to a sheet from the details page; None when the frame is not that page."""
        t = Texts(boxes, self.min_confidence)
        if f.team_page(t) != "stats":
            return None
        members = []
        for member, origin in zip(sheet.members, CARD_ORIGINS, strict=True):
            members.append(replace(member, stats=self._member_stats(frame, t, member, origin, base_stats(member))))
        return replace(sheet, members=tuple(members))

    def _member_stats(self, frame, t: Texts, member: RosterMember, origin, base: dict | None) -> MemberStats:
        x, y = origin
        stats: dict[str, int | None] = {}
        points: dict[str, int | None] = {}
        doubtful: dict[str, int] = {}
        raised, lowered, issues = [], [], []
        for stat, column, row in STAT_LAYOUT:
            top, bottom = y + STAT_ROWS[row][0], y + STAT_ROWS[row][1]
            spans = STAT_COLUMNS[column]
            stats[stat] = self._number(frame, (x + spans["value"][0], top, x + spans["value"][1], bottom))
            reading = self._digits(frame, (x + spans["points"][0], top, x + spans["points"][1], bottom))
            points[stat] = reading[0] if reading and reading[1] >= self.min_confidence else None
            if reading and points[stat] is None:  # a lone "1" reads right but below the confidence floor
                doubtful[stat] = reading[0]
            direction = arrow(frame, (x + spans["label"][0], top, x + spans["label"][1], bottom))
            (raised if direction == "up" else lowered if direction == "down" else []).append(stat)
        names = [b for b in t.any_text(_offset(origin, FIELD_OFFSETS["name"]))]
        shown = names[0].text.strip() if len(names) == 1 else None
        if shown is None or normalize_text(shown) != normalize_text(member.name.value or ""):
            issues.append(f"stats card shows {shown!r}, expected {member.name.value!r}")
        inferred: list[str] = []
        if not issues:
            inferred += confirm_points(stats, points, doubtful, base, raised, lowered)
            single = recover_single_point(stats, points, base, raised, lowered)
            inferred += [single] if single else []
        unread = [s for s in stats if stats[s] is None or points[s] is None]
        if unread:
            issues.append(f"unread stats {unread}")
        elif sum(points.values()) != POINTS_TOTAL or max(points.values()) > POINTS_MAX:  # type: ignore[arg-type]
            issues.append(f"stat points {points} do not total {POINTS_TOTAL} with at most {POINTS_MAX} each")
        if len(raised) > 1 or len(lowered) > 1 or len(raised) != len(lowered):
            issues.append(f"nature arrows unclear (raised {raised}, lowered {lowered})")
        result = MemberStats(
            shown,
            stats,
            points,
            raised[0] if len(raised) == 1 else None,
            lowered[0] if len(lowered) == 1 else None,
            tuple(issues),
            tuple(inferred),
        )
        if not issues and result.nature is None:
            result = replace(result, issues=(f"no nature raises {result.raised} and lowers {result.lowered}",))
        return result


def arrow(frame: np.ndarray, rect: tuple[int, int, int, int]) -> str | None:
    left, top, right, bottom = rect
    hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV).astype(int)
    hue, bright = hsv[..., 0], (hsv[..., 1] > 70) & (hsv[..., 2] > 150)
    red = np.count_nonzero(((hue <= 10) | (hue >= 165)) & bright)
    blue = np.count_nonzero((hue >= 85) & (hue <= 110) & bright)
    if red >= ARROW_PIXELS and blue < ARROW_PIXELS:
        return "up"
    if blue >= ARROW_PIXELS and red < ARROW_PIXELS:
        return "down"
    return None


def expected_stat(stat: str, base: dict, points: int, raised, lowered) -> int:
    multiplier = 110 if stat in raised else 90 if stat in lowered else 100
    return hp_value(base[stat], points) if stat == "hp" else stat_value(base[stat], points, multiplier)


def confirm_points(stats, points, doubtful: dict[str, int], base: dict | None, raised, lowered) -> list[str]:
    """Accept low-confidence stat-point reads that reproduce their displayed stat exactly."""
    if not base or len(raised) > 1 or len(lowered) > 1 or len(raised) != len(lowered):
        return []
    notes = []
    for stat, value in doubtful.items():
        if (
            stats[stat] is not None
            and value <= POINTS_MAX
            and stats[stat] == expected_stat(stat, base, value, raised, lowered)
        ):
            points[stat] = value
            notes.append(f"{stat}={value}: low-confidence read confirmed by the displayed stat")
    return notes


def recover_single_point(stats, points, base: dict | None, raised, lowered) -> str | None:
    """Recover one unread stat-point figure when the 66-point total and all six displayed stats agree."""
    missing = [stat for stat, value in points.items() if value is None]
    if len(missing) != 1 or any(value is None for value in stats.values()) or not base:
        return None
    if len(raised) > 1 or len(lowered) > 1 or len(raised) != len(lowered):
        return None
    candidate = POINTS_TOTAL - sum(value for value in points.values() if value is not None)
    proposed = {**points, missing[0]: candidate}
    if any(not 0 <= value <= POINTS_MAX for value in proposed.values()):
        return None
    if any(
        observed != expected_stat(stat, base, proposed[stat], raised, lowered) for stat, observed in stats.items()
    ):
        return None
    points[missing[0]] = candidate
    return f"{missing[0]}={candidate}: inferred from the {POINTS_TOTAL}-point total; all six displayed stats verified"

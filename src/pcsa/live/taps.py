"""Taps: the safety policy every tap passes through, the plans for battle orders, and transmission.

Every tap lands at a uniformly random point inside its control's region. Points come in reference
coordinates (2340x1080) and are mapped to the device through the viewport in exactly one place.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from pcsa.common import log
from pcsa.common.clock import Clock
from pcsa.common.config import Rect, ScreenConfig, TapCounts, Viewport
from pcsa.common.errors import SafetyStop
from pcsa.common.types import Screen, SemanticAction
from pcsa.live.device import InputDevice as Device
from pcsa.phone.ocr import REFERENCE, TextBox
from pcsa.phone.texts import Texts

FORBIDDEN_TEXT = r".*(?:forfeit|quit|edit team|rate|submit|review|purchase|buy|double battle|shop).*"
# A blind tap (the screen centre during a disconnect) has no expected control under it, so it is refused near
# a wider set of words and with a wider margin than a tap on a known control.
BLIND_FORBIDDEN_TEXT = r".*(?:forfeit|quit|edit team|rat(?:e|ing)|submit|review|purchase|buy|double|shop|delete).*"
TEXT_MARGIN = 20
BLIND_TEXT_MARGIN = 60
MAX_PARTY_ROW = 2

# Which named regions a handler may tap on which screen. Anything else is a SafetyStop.
ALLOWED: dict[Screen, frozenset[str]] = {
    Screen.LOBBY: frozenset({"begin_matchmaking", "team_details_back", "team_selector"}),
    Screen.TEAM_LIST: frozenset({"team_details_back", "team_header"}),
    Screen.TEAM_MENU: frozenset({"team_details_back", "view_details", "cancel"}),
    Screen.TEAM_DETAILS: frozenset({"team_details_back", "stats_tab"}),
    Screen.TEAM_STATS: frozenset({"team_details_back"}),
    Screen.TEAM_SELECT: frozenset({"team_rows", "team_done"}),
    Screen.ACTION: frozenset({"fight", "pokemon", "view_log", "mega", "move_rows", "party_rows", "switch_in"}),
    Screen.MOVES: frozenset({"move_rows", "mega", "move_back"}),
    Screen.LOG: frozenset({"log_back"}),
    Screen.PARTY: frozenset({"party_rows", "switch_in", "party_back"}),
    Screen.SWITCH_POPUP: frozenset({"switch_in", "cancel"}),
    Screen.FORCED: frozenset({"party_rows", "switch_in"}),
    Screen.OPTIONS: frozenset({"continue_battling"}),
    Screen.RANK_UP: frozenset({"rank_up_close", "dismiss"}),
    Screen.RATING_DIALOG: frozenset({"dismiss"}),
    Screen.DIALOG: frozenset({"dismiss", "screen_centre"}),
    Screen.COMM_ERROR: frozenset({"dismiss", "screen_centre"}),
    Screen.TRANSITION: frozenset({"screen_centre"}),  # only while a battle is riding out a disconnect
    Screen.TITLE: frozenset({"title_continue"}),
    Screen.HOME: frozenset({"home_battle"}),
    Screen.BATTLE_MODES: frozenset({"ranked_battles"}),
    Screen.RANKED_FORMATS: frozenset({"single_battle"}),
    Screen.UNKNOWN: frozenset({"dismiss", "screen_centre"}),
}


@dataclass(frozen=True)
class Tap:
    name: str
    x: int  # device pixels
    y: int
    rx: int = 0  # reference pixels, for the record
    ry: int = 0


@dataclass
class ExecResult:
    taps: list[dict] = field(default_factory=list)
    first_tap_at: float | None = None
    last_tap_at: float | None = None
    error: str | None = None

    def record(self) -> dict:
        return asdict(self)


class TapPolicy:
    """Refuses any tap a handler has no business sending."""

    def __init__(self, screen: ScreenConfig) -> None:
        self.forbidden = [(c.x * REFERENCE[0], c.y * REFERENCE[1], c.r * REFERENCE[0]) for c in screen.forbidden]

    def check(self, screen: Screen, name: str, rx: float, ry: float, texts: Texts | None = None) -> None:
        allowed = ALLOWED.get(screen, frozenset())
        if name not in allowed:
            raise SafetyStop(f"tap {name!r} is not allowed on screen {screen}")
        reason = self.blocked(name, rx, ry, texts)
        if reason is not None:
            raise SafetyStop(reason)

    def blocked(
        self,
        name: str,
        rx: float,
        ry: float,
        texts: Texts | None,
        pattern: str = FORBIDDEN_TEXT,
        margin: float = TEXT_MARGIN,
    ) -> str | None:
        """Why a tap at this reference point would hit a forbidden area or forbidden text; None if it would not."""
        for fx, fy, radius in self.forbidden:
            if (rx - fx) ** 2 + (ry - fy) ** 2 <= radius**2:
                return f"tap {name!r} at ({rx:.0f},{ry:.0f}) is inside a forbidden area"
        if texts is not None:
            for box in texts.find(pattern):
                left, top, right, bottom = box.bounds
                if left - margin <= rx <= right + margin and top - margin <= ry <= bottom + margin:
                    return f"tap {name!r} would land on {box.text!r}"
        return None


class Executor:
    """Builds tap plans from configured regions and sends them through the policy."""

    def __init__(
        self,
        screen: ScreenConfig,
        viewport: Viewport,
        counts: TapCounts,
        policy: TapPolicy,
        rng: random.Random | None = None,
    ) -> None:
        self.screen = screen
        self.viewport = viewport
        self.counts = counts
        self.policy = policy
        self.rng = rng or random.Random()
        self.reference = Viewport.of_frame(*REFERENCE)

    # Geometry --------------------------------------------------------------------------------------
    def to_device(self, rx: float, ry: float) -> tuple[int, int]:
        vp = self.viewport
        return round(vp.x + rx / REFERENCE[0] * vp.w), round(vp.y + ry / REFERENCE[1] * vp.h)

    def rect(self, name: str, index: int | None = None) -> Rect:
        return self.screen.resolve(name, self.reference, index)

    def point_in(self, name: str, rect: Rect) -> Tap:
        rx, ry = self.rng.uniform(rect.x1, rect.x2), self.rng.uniform(rect.y1, rect.y2)
        x, y = self.to_device(rx, ry)
        return Tap(name, x, y, round(rx), round(ry))

    def point(self, name: str, index: int | None = None, label: str | None = None) -> Tap:
        return self.point_in(label or name, self.rect(name, index))

    def point_at_box(self, name: str, box: TextBox) -> Tap:
        """A tap on an OCR-located control: a random point inside the inner half of its box."""
        left, top, right, bottom = box.bounds
        w, h = right - left, bottom - top
        inner = Rect(x1=round(left + w / 4), y1=round(top + h / 4), x2=round(right - w / 4), y2=round(bottom - h / 4))
        return self.point_in(name, inner)

    def switch_in(self, row_rect: Rect) -> Tap:
        cx, cy = row_rect.center
        rect = self.screen.resolve_relative("switch_in", self.reference, (cx / REFERENCE[0], cy / REFERENCE[1]))
        return self.point_in("switch_in", rect)

    # Plans ------------------------------------------------------------------------------------------
    def plan_move(self, move: int, mega: bool, focused_row: int | None) -> list[Tap]:
        taps = [self.point("fight")]
        if mega:
            taps.append(self.point("mega"))
        count = self.counts.focused_move_taps if focused_row == move else self.counts.unfocused_move_taps
        taps += [self.point("move_rows", move, f"move_rows{move}" + ("_confirm" if i else "")) for i in range(count)]
        return taps

    def plan_switch(self, visible_row: int, focused_row: int | None = 0) -> list[Tap]:
        return [self.point("pokemon"), *self._party_taps(visible_row, "party_rows", focused_row, lowest=1)]

    def plan_forced(self, visible_row: int, focused_row: int | None) -> list[Tap]:
        return self._party_taps(visible_row, "party_rows", focused_row, lowest=0)

    def _party_taps(self, visible_row: int, key: str, focused_row: int | None, lowest: int) -> list[Tap]:
        if not lowest <= visible_row <= MAX_PARTY_ROW:
            raise ValueError(f"switch target must be row {lowest}-{MAX_PARTY_ROW}; got {visible_row}")
        count = self.counts.focused_party_taps if focused_row == visible_row else self.counts.unfocused_party_taps
        taps = [self.point(key, visible_row, f"{key}{visible_row}" + ("_open" if i else "")) for i in range(count)]
        return taps + [self.switch_in(self.rect(key, visible_row))]

    def plan_team(self, roster_rows: list[int]) -> list[Tap]:
        return [self.point("team_rows", i, f"team_rows{i}") for i in roster_rows] + [self.point("team_done")]

    def plan_action(self, action: SemanticAction, visible_order: list[int], focused_move: int | None) -> list[Tap]:
        if action.kind == "move":
            assert action.move is not None
            return self.plan_move(action.move, action.mega, focused_move)
        assert action.slot is not None
        return self.plan_switch(visible_order.index(action.slot))

    # Transmission -----------------------------------------------------------------------------------
    def execute(
        self, plan: list[Tap], screen: Screen, device: Device, clock: Clock, inter_tap_s: float, texts=None
    ) -> ExecResult:
        """Check every tap against the policy first, then send them blind with the configured spacing."""
        for tap in plan:
            self.policy.check(screen, _base_name(tap.name), tap.rx, tap.ry, texts)
        result = ExecResult()
        for index, tap in enumerate(plan):
            if index:
                clock.sleep(inter_tap_s)
            at = clock.now()
            try:
                transport = device.tap(tap.x, tap.y)
            except Exception as error:  # a transport failure ends the plan; it is recorded, not retried here
                result.error = f"{tap.name}: {error}"
                result.taps.append({"name": tap.name, "x": tap.x, "y": tap.y, "at": at, "ok": False})
                log.warning("tap_failed", tap=tap.name, reason=str(error))
                return result
            result.taps.append(
                {"name": tap.name, "x": tap.x, "y": tap.y, "at": at, "ok": True, "transport_s": transport.seconds}
            )
            result.first_tap_at = at if result.first_tap_at is None else result.first_tap_at
            result.last_tap_at = at
        return result


def _base_name(label: str) -> str:
    """'move_rows2_confirm' -> 'move_rows'; 'dismiss' -> 'dismiss'."""
    for suffix in ("_confirm", "_open"):
        label = label.removesuffix(suffix)
    return label.rstrip("0123456789")

"""Which screen is this? An ordered table of named rules over OCR boxes and a few pixel checks.

Rules are tried in order; the first that returns fields wins and its name is the observation's
`reason`. Rules marked `fallback` catch leftovers (sparse text, unknown) and are excluded from the
ambiguity test. Everything works in the 2340x1080 reference frame: `Perception` crops the device
viewport and resizes before classifying.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from pcsa.common.config import ScreenConfig, Viewport
from pcsa.common.types import Flag, Observation, Screen
from pcsa.phone import fields as f
from pcsa.phone.ocr import REFERENCE, TextBox, TextReader
from pcsa.phone.rank import read_rank
from pcsa.phone.texts import Texts, mean_brightness, skip_button, split_screen, yellow_fraction

AUTO_PATTERN = r".*(?:automatically chosen|automatically|server did not receive).*"
RATING_PATTERN = r".*(?:google play|rate this app|rate the app|rate us|rate pokemon champions).*"
RESULT_LEFT, RESULT_RIGHT = (380, 520, 920, 780), (1420, 520, 1960, 780)
SPARSE_BOXES = 6
FEW_BOXES = 4
RESULT_YELLOW = 0.2
RESULT_OTHER_HALF = 0.05
BLACK_MEAN = 18
OPPONENT_PLATE_NAME, OPPONENT_PLATE_HP = (1880, 30, 2200, 110), (2030, 100, 2200, 190)
OWN_PLATE_NAME, OWN_PLATE_HP = (240, 950, 560, 1030), (300, 1010, 600, 1080)
OWN_HP = r"\d{0,3}\s*/\s*\d{2,3}|7\d{3}"  # "149/182", "/182" (thin numerator missed), "7182" (slash read as 7)
OWN_TRAINER_PLATE, OPPONENT_TRAINER_PLATE = (450, 900, 950, 1060), (1450, 900, 1950, 1060)


@dataclass
class Ctx:
    t: Texts
    image: np.ndarray
    flags: set[Flag]
    line: Callable | None
    fields: dict = field(default_factory=dict)


Test = Callable[[Ctx], dict | None]


@dataclass(frozen=True)
class Rule:
    name: str
    screen: Screen
    test: Test
    fallback: bool = False


def flags_of(t: Texts) -> tuple[set[Flag], dict]:
    flags: set[Flag] = set()
    fields: dict = {}
    if t.has(AUTO_PATTERN):
        flags.add(Flag.AUTO_NOTICE)
        fields["notice"] = " ".join(b.text for b in t.find(AUTO_PATTERN))
    if t.has(r"move\s*time", (850, 560, 1500, 880)):
        flags.add(Flag.TIMER_WARN)
    if t.has(r".*date has changed.*"):
        flags.add(Flag.DATE_CHANGED)
    elif t.has(r"communication was interrupted\.?|.*returned to the title screen.*"):
        flags.add(Flag.COMM_ERROR)
    if t.has(r"communicating.*", (700, 200, 1650, 450)):
        flags.add(Flag.COMMUNICATING)
    return flags, fields


# Rule tests -----------------------------------------------------------------------------------------
def _interruption(c: Ctx) -> dict | None:
    if Flag.DATE_CHANGED in c.flags or Flag.COMM_ERROR in c.flags:
        control = f.safe_control(c.t, (500, 500, 1850, 1000)) or {}
        return {"kind": "date_changed" if Flag.DATE_CHANGED in c.flags else "comm_error", **control}
    return None


def _black(c: Ctx) -> dict | None:
    return {} if mean_brightness(c.image) < BLACK_MEAN and not c.t.count() else None


def _forfeit(c: Ctx) -> dict | None:
    return {} if c.t.has(r".*forfeit.*") else None


def _rating(c: Ctx) -> dict | None:
    if not c.t.has(RATING_PATTERN):
        return None
    return {"provisional": True, **(f.safe_control(c.t, (300, 200, 2040, 1000)) or {"dismiss": None, "unsafe": ()})}


def _rank_up(c: Ctx) -> dict | None:
    t = c.t
    if t.has(r"rank-?\s*up reward", (700, 30, 1650, 220)) or t.has(
        r"note: your reward has been sent.*", (700, 700, 1650, 1000)
    ):
        close = t.find(r"close", (1575, 990, 1895, 1038))
        tier = " ".join(b.text for b in t.find(r".*tier", (700, 150, 1650, 300))) or None
        return {"stage": "reward_popup", "close": f.target("close", close[0]) if close else None, "tier_text": tier}
    if t.has(r"you claimed the below:?", (700, 60, 1650, 220)):
        return {"stage": "claim_popup"}
    if skip_button(c.image):
        message = " ".join(b.text for b in t.any_text((650, 740, 1600, 990))) or None
        return {"stage": "cutscene", "skip": (2040, 41), "message": message}
    return None


def _result_banner(c: Ctx) -> dict | None:
    if c.t.count() > SPARSE_BOXES:
        return None
    left, right = yellow_fraction(c.image, RESULT_LEFT), yellow_fraction(c.image, RESULT_RIGHT)
    if max(left, right) >= RESULT_YELLOW and min(left, right) <= RESULT_OTHER_HALF and split_screen(c.image):
        return {"outcome": "win" if left > right else "loss", "banner_yellow": (round(left, 3), round(right, 3))}
    return None


def _options(c: Ctx) -> dict | None:
    cont, quit_ = c.t.joined((1470, 935, 1985, 1045)), c.t.joined((350, 935, 875, 1045))
    return {} if cont == "continue battling" and quit_ == "quit battling" else None


def _rewards(c: Ctx) -> dict | None:
    t = c.t
    if _options(c) is not None:  # the options screen shows the rank too; Continue/Quit decide
        return None
    seen = (
        t.has(r"vp earned")
        or t.has(r"rank", (1500, 330, 2200, 500))
        or t.has(r"\d+-win streak!?", (700, 40, 1650, 160))
    )
    return {} if seen else None


def _switch_popup(c: Ctx) -> dict | None:
    if c.t.has(r"switch in", (300, 300, 900, 1000)) and c.t.has(r"cancel", (300, 300, 900, 1050)):
        cancel = c.t.find(r"cancel", (300, 300, 900, 1050))
        return {"cancel": f.target("cancel", cancel[0])}
    return None


def _log(c: Ctx) -> dict | None:
    turn = c.t.find(r"turn\s*(\d+)", (950, 0, 1400, 175))
    number = re.search(r"\d+", turn[0].norm) if turn else None
    return {"turn": int(number[0])} if number else None


def _moves(c: Ctx) -> dict | None:
    return f.move_list(c.t, c.image, c.line)


def _party_or_forced(c: Ctx) -> dict | None:
    return f.party(c.t, c.image)


def _action(c: Ctx) -> dict | None:
    if (
        c.t.has(r"fight", (1850, 650, 2280, 900))
        and c.t.has(r"pokemon", (1800, 880, 2280, 1080))
        and Flag.COMMUNICATING not in c.flags
    ):
        return f.action_hud(c.t)
    return None


def _prep(c: Ctx) -> dict | None:
    return {} if c.t.has(r"preparing for battle") or c.t.has(r"standing by") else None


def _matchmaking(c: Ctx) -> dict | None:
    t = c.t
    overlay = (
        t.has(r"matchmaking in progress.*|.*opponent has been found.*|get ready.*")
        or t.has(r"matchmaking|.*in progress.*", (800, 540, 1600, 760))
        or t.has(r"cancel", (1550, 700, 1850, 800))
        or (t.has(r"matchmaking") and not t.has(r"begin matchmaking"))
    )
    return {} if overlay else None


def _lobby(c: Ctx) -> dict | None:
    if not c.t.has(r"begin matchmaking") or _matchmaking(c) is not None:  # the overlay dims the lobby behind it
        return None
    label = c.t.any_text((1150, 500, 2050, 590))
    return {"team_label": label[0].text.strip() if len(label) == 1 else None}


def _team_page(page: str) -> Test:
    return lambda c: {} if f.team_page(c.t) == page else None


def _title(c: Ctx) -> dict | None:
    return {"provisional": True} if c.t.has(r"(?:tap|touch|press)\s.*(?:start|screen|button).*") else None


def _loading(c: Ctx) -> dict | None:
    return {} if c.t.has(r"loading.*") or c.t.has(r"vs", (900, 300, 1450, 800)) else None


def _vs_cards(c: Ctx) -> dict | None:
    mode_menu = c.t.has(r"double battle") or c.t.has(r"[\d,]+ vp")
    if not mode_menu and (len(c.t.find(r"poke ball tier.*")) >= 2 or len(c.t.find(r"rank \d+")) >= 2):
        return {}
    return None


def _communicating(c: Ctx) -> dict | None:
    return {} if Flag.COMMUNICATING in c.flags else None


def _battle_message(c: Ctx) -> dict | None:
    if c.t.has(r"double battle") or c.t.has(r"[\d,]+ vp"):
        return None
    return f.battle_message(c.t)


def _hud_clocks(c: Ctx) -> dict | None:
    clocks = c.t.find(r"\d\d:\d\d", (1780, 140, 1990, 230)) + c.t.find(r"\d\d:\d\d", (300, 880, 560, 1060))
    return {} if clocks else None


def _auto_notice(c: Ctx) -> dict | None:
    return {} if Flag.AUTO_NOTICE in c.flags else None


def _hp_plates(c: Ctx) -> dict | None:
    """The battle scene between menus (attack and ability/item animations, a banner such as "Cinderace's Life
    Orb"): an HP plate shows (theirs top right: name and "41%", or ours bottom left: name and "149/182") and
    nothing else on the frame names a control. Polled like any wait; a frozen scene still meets the WAIT
    screen timeout, and a real interruption is read earlier (dialog text, Communicating, a dismiss control)."""
    t = c.t
    if t.has(f.SAFE_CONTROL) or t.has(f.UNSAFE_CONTROL) or c.flags & {Flag.COMM_ERROR, Flag.DATE_CHANGED}:
        return None
    theirs = f.name_in(t, OPPONENT_PLATE_NAME) and t.has(r"\d{0,3}\s*%", OPPONENT_PLATE_HP)
    ours = f.name_in(t, OWN_PLATE_NAME) and t.has(OWN_HP, OWN_PLATE_HP)
    if not (theirs or ours):
        return None
    return {"opponent_name": f.name_in(t, OPPONENT_PLATE_NAME), "active_name": f.name_in(t, OWN_PLATE_NAME)}


def _trainer_cards(c: Ctx) -> dict | None:
    """The split screen with both trainers' name plates (the end of a battle, before the result banner)."""
    t = c.t
    if t.has(f.SAFE_CONTROL) or t.has(f.UNSAFE_CONTROL) or not split_screen(c.image):
        return None
    return {} if t.any_text(OWN_TRAINER_PLATE) and t.any_text(OPPONENT_TRAINER_PLATE) else None


def _dialog(c: Ctx) -> dict | None:
    """A popup with one safe dismiss control and few other words; only after every named screen failed."""
    control = f.safe_control(c.t, (300, 150, 2040, 1000))
    return control if control and c.t.count() <= 2 * SPARSE_BOXES else None


def _sparse(c: Ctx) -> dict | None:
    return {} if c.t.count() <= FEW_BOXES else None


RULES: tuple[Rule, ...] = (
    Rule("interruption dialog", Screen.COMM_ERROR, _interruption),
    Rule("black frame", Screen.TRANSITION, _black),
    Rule("forfeit message", Screen.FORFEIT_MSG, _forfeit),
    Rule("WON/LOST banner", Screen.RESULT, lambda c: f.result_banner(c.t)),
    Rule("rating dialog", Screen.RATING_DIALOG, _rating),
    Rule("rank-up popup or cutscene", Screen.RANK_UP, _rank_up),
    Rule("gold banner in one half", Screen.RESULT, _result_banner),
    Rule("post-battle options", Screen.OPTIONS, _options),
    Rule("rank and VP", Screen.REWARDS, _rewards),
    Rule("Switch in / Cancel popup", Screen.SWITCH_POPUP, _switch_popup),
    Rule("turn header", Screen.LOG, _log),
    Rule("move rows with PP", Screen.MOVES, _moves),
    Rule("party rows", Screen.PARTY, _party_or_forced),
    Rule("Fight and Pokemon", Screen.ACTION, _action),
    Rule("N/3 counter and Done", Screen.TEAM_SELECT, lambda c: f.team_select(c.t)),
    Rule("Preparing / Standing By", Screen.PREP, _prep),
    Rule("matchmaking overlay", Screen.MATCHMAKING, _matchmaking),
    Rule("Begin Matchmaking", Screen.LOBBY, _lobby),
    Rule("team menu", Screen.TEAM_MENU, _team_page("menu")),
    Rule("team details", Screen.TEAM_DETAILS, _team_page("details")),
    Rule("team stats", Screen.TEAM_STATS, _team_page("stats")),
    Rule("team list", Screen.TEAM_LIST, _team_page("list")),
    Rule("home tiles", Screen.HOME, lambda c: f.home_menu(c.t)),
    Rule("battle modes", Screen.BATTLE_MODES, lambda c: f.battle_modes(c.t)),
    Rule("ranked formats", Screen.RANKED_FORMATS, lambda c: f.ranked_formats(c.t)),
    Rule("title prompt", Screen.TITLE, _title),
    Rule("loading or VS", Screen.TRANSITION, _loading, fallback=True),
    Rule("VS intro trainer cards", Screen.TRANSITION, _vs_cards, fallback=True),
    Rule("Communicating", Screen.WAIT, _communicating, fallback=True),
    Rule("battle message", Screen.WAIT, _battle_message, fallback=True),
    Rule("battle HUD clocks", Screen.WAIT, _hud_clocks, fallback=True),
    Rule("automatic choice notice", Screen.WAIT, _auto_notice, fallback=True),
    Rule("single safe control", Screen.DIALOG, _dialog, fallback=True),
    Rule("sparse text", Screen.TRANSITION, _sparse, fallback=True),
    Rule("HP plates without menus", Screen.WAIT, _hp_plates, fallback=True),
    Rule("trainer cards split screen", Screen.TRANSITION, _trainer_cards, fallback=True),
)
RANK_SCREENS = {Screen.LOBBY, Screen.REWARDS, Screen.OPTIONS}


def classify(
    boxes: list[TextBox], image: np.ndarray, min_confidence: float, line: Callable | None = None
) -> Observation:
    """Classify one reference-size BGR frame from its OCR boxes and pixels."""
    t = Texts(boxes, min_confidence)
    flags, fields = flags_of(t)
    ctx = Ctx(t, image, flags, line, fields)
    for rule in RULES:
        found = rule.test(ctx)
        if found is None:
            continue
        screen = _refine(rule.screen, found)
        fields = {**ctx.fields, **found}
        if screen in RANK_SCREENS:
            fields["own_rank"] = read_rank(t.all, lobby=screen == Screen.LOBBY, min_confidence=min_confidence)
        return Observation(screen, REFERENCE, frozenset(flags), fields, rule.name, tuple(t.all))
    return Observation(Screen.UNKNOWN, REFERENCE, frozenset(flags), ctx.fields, "no rule matched", tuple(t.all))


def _refine(screen: Screen, found: dict) -> Screen:
    """Party rows without Back and a fainted member are a forced replacement."""
    if screen == Screen.PARTY and found.get("no_back") and any(p.fainted for p in found["party"]):
        return Screen.FORCED
    return screen


def matching_rules(boxes: list[TextBox], image: np.ndarray, min_confidence: float) -> list[tuple[str, Screen]]:
    """Every non-fallback rule that matches, with its screen (for the ambiguity test)."""
    t = Texts(boxes, min_confidence)
    flags, fields = flags_of(t)
    ctx = Ctx(t, image, flags, None, fields)
    return [(r.name, r.screen) for r in RULES if not r.fallback and r.test(ctx) is not None]


def to_reference(image: np.ndarray, viewport: Viewport | None = None) -> np.ndarray:
    """Crop the game viewport out of a captured frame and resize it to the reference size."""
    if viewport is not None:
        image = image[viewport.y : viewport.y + viewport.h, viewport.x : viewport.x + viewport.w]
    if (image.shape[1], image.shape[0]) == REFERENCE:
        return image
    return cv2.resize(image, REFERENCE, interpolation=cv2.INTER_AREA)


class Perception:
    """Frame -> Observation, through the device viewport and the text reader."""

    def __init__(self, reader: TextReader, screen: ScreenConfig, viewport: Viewport | None = None) -> None:
        self.reader = reader
        self.screen = screen
        self.viewport = viewport

    def observe(self, image: np.ndarray, frame_id: str = "") -> Observation:
        reference = to_reference(image, self.viewport)
        obs = classify(self.reader.read(reference), reference, self.reader.min_confidence, self.reader.line)
        return Observation(
            obs.screen, (image.shape[1], image.shape[0]), obs.flags, obs.fields, obs.reason, obs.texts, frame_id
        )

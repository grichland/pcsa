"""Field extractors: what a recognized screen shows (moves, party, HUD, roster, menus). Reference frame."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from pcsa.common.types import MoveInfo, PartyMember
from pcsa.phone.ocr import TextBox
from pcsa.phone.texts import Rect, Texts, fraction, lime_fraction, name_in, pp_from_crop, red_fraction

MOVE_ROW_TOPS = (500, 628, 756, 884)  # move list rows, 128 px apart
PARTY_ROW_CENTERS = (287, 412, 538)
TEAM_ROW_TOP = 150  # team selection rows, 126 px apart
TEAM_ROW_PITCH = 126
TEAM_ROWS = 6
FOCUS_FRACTION = 0.25
PARTY_FOCUS_FRACTION = 0.30
RED_BADGE_FRACTION = 0.2
MIN_PP_ROWS = 2
MIN_NAMED_ROWS = 3
MIN_PARTY_HP = 1
TEAM_MENU: Rect = (600, 170, 2100, 1000)


def target(name: str, box: TextBox) -> dict:
    return {"name": name, "center": tuple(round(v) for v in box.center), "box": tuple(round(v) for v in box.bounds)}


def move_list(t: Texts, image: np.ndarray, line: Callable | None) -> dict | None:
    """The move screen's rows, or None when the frame is not a settled move list."""
    moves = []
    for slot, top in enumerate(MOVE_ROW_TOPS):
        name = name_in(t, (1740, top, 2060, top + 95))
        pp = fraction(t, (2060, top, 2235, top + 115))
        if name or pp:
            focused = lime_fraction(image, (1700, top + 15, 2190, top + 100)) > FOCUS_FRACTION
            moves.append(MoveInfo(slot, name, pp, False, focused))
    shell = t.has(r"move info", (1850, 390, 2280, 500)) or t.has(r"back", (1900, 1000, 2280, 1080))
    if not shell or (
        sum(m.pp is not None for m in moves) < MIN_PP_ROWS and sum(m.name is not None for m in moves) < MIN_NAMED_ROWS
    ):
        return None
    moves = [
        replace(m, pp=pp_from_crop(t, image, MOVE_ROW_TOPS[m.slot], line)) if m.pp is None and m.name else m
        for m in moves
    ]
    # No visual marker for a disabled row is known; a row with name and PP read is judged available.
    unknown = tuple(m.slot for m in moves if m.name is None or m.pp is None)
    return {
        "moves": tuple(moves),
        "disabled_unknown_slots": unknown,
        "focused_row": next((m.slot for m in moves if m.focused), None),
        "info_open": t.has(r"close", (850, 740, 1200, 900)) or t.has(r"category", (250, 180, 1150, 400)),
        "mega_available": t.has(r"mega\s*evolve", (1250, 860, 1700, 980)),
    }


def party(t: Texts, image: np.ndarray) -> dict | None:
    """Party rows on the party / forced-replacement screen, or None."""
    summary = t.has(r"hide summary|show summary") or t.has(r"moves\s*&\s*more", (850, 60, 1750, 200))
    rows, focused_row = [], None
    for row, center in enumerate(PARTY_ROW_CENTERS):
        hp = fraction(t, (290, center - 5, 560, center + 50))
        name = name_in(t, (290, center - 50, 560, center + 2))
        if hp or name:
            red = red_fraction(image, (275, center + 5, 320, center + 40))
            fainted = (hp is not None and hp[0] == 0) or red > RED_BADGE_FRACTION
            rows.append(PartyMember(row, name, hp, fainted))
            if lime_fraction(image, (275, center - 45, 770, center + 45)) > PARTY_FOCUS_FRACTION:
                focused_row = row
    if not summary or sum(p.hp is not None for p in rows) < MIN_PARTY_HP:
        return None
    return {"party": tuple(rows), "focused_row": focused_row, "no_back": not t.has(r"back", (1700, 940, 2280, 1080))}


def action_hud(t: Texts) -> dict:
    timer = t.find(r"\d{1,2}", (2050, 230, 2230, 320))
    return {
        "active_name": name_in(t, (240, 950, 560, 1030)),
        "active_hp": fraction(t, (300, 1010, 600, 1080)),
        "opponent_name": name_in(t, (1850, 30, 2200, 110)),
        "move_time": int(timer[0].norm) if timer else None,
        "view_log": t.has(r"view log", (100, 300, 450, 460)),
    }


def team_select(t: Texts) -> dict | None:
    counter = t.find(r"(\d)\s*/\s*3", (300, 880, 520, 1010))
    prompt = t.has(r"select \d pokemon", (800, 180, 1300, 330))
    confirmed = (counter and (t.has(r"done", (480, 880, 780, 1010)) or prompt)) or (
        prompt and t.has(r"to send into battle\.?", (800, 180, 1300, 350))
    )
    if not confirmed:
        return None
    roster, markers = [], {}
    for row in range(TEAM_ROWS):
        top = TEAM_ROW_TOP + TEAM_ROW_PITCH * row
        roster.append(name_in(t, (320, top + 5, 560, top + 72)))
        digits = t.find(r"[1-3]", (285, top + 25, 345, top + 110))
        if digits:
            markers[row] = int(digits[0].norm)
    timer = t.find(r"\d\d:\d\d", (480, 20, 700, 90))
    return {
        "selected_count": int(counter[0].norm[0]) if counter else None,
        "roster": tuple(roster),
        "markers": markers,
        "timer": timer[0].norm if timer else None,
    }


def result_banner(t: Texts) -> dict | None:
    results = [b for b in t.find(r"(won|lost)\W*") if b.bounds[3] - b.bounds[1] > 70]
    if not results:
        return None
    own = [b for b in results if b.center[0] < 1170]
    side = own or [b for b in results if b.center[0] >= 1170]
    word = side[0].norm.rstrip("!.… ")
    if own:
        return {"outcome": "win" if word.startswith("won") else "loss"}
    return {"outcome": "loss" if word.startswith("won") else "win"}  # only the opponent's half was read


def home_menu(t: Texts) -> dict | None:
    battle = t.find(r"battle", (1240, 220, 1530, 360))
    tiles = sum(t.has(p, (1240, 220, 2100, 780)) for p in (r"box", r"train", r"recruit", r"shop"))
    dock = sum(t.has(p, (500, 990, 1750, 1080)) for p in (r"missions", r"mailbox", r"style", r"submenu"))
    if battle and (tiles >= 2 or dock >= 2):
        return {"target": target("home_battle", battle[0])}
    return None


def battle_modes(t: Texts) -> dict | None:
    ranked = t.find(r"ranked(?: battles)?", (1200, 180, 1800, 300))
    if (
        ranked
        and t.has(r"casual(?: battles)?", (1200, 350, 1800, 470))
        and t.has(r"private(?: battles)?", (1200, 520, 1800, 640))
    ):
        return {"target": target("ranked_battles", ranked[0])}
    return None


def ranked_formats(t: Texts) -> dict | None:
    region: Rect = (1180, 420, 1800, 560)
    single = t.find(r"single battle", region)
    if not single:  # OCR can split the label into two boxes on the same row
        names, battles = t.find(r"single", region), t.find(r"battle", region)
        single = [
            n
            for n in names
            if any(0 < b.center[0] - n.center[0] < 240 and abs(b.center[1] - n.center[1]) < 30 for b in battles)
        ]
    tiers = t.find(r".*tier\s*\|?\s*rank\s*\d+")
    if single and t.has(r"double(?: battle)?", (1180, 690, 1800, 800)) and tiers:
        named = {
            "single": [b for b in tiers if b.center[1] < 680],
            "double": [b for b in tiers if b.center[1] >= 680],
        }
        return {
            "target": target("single_battle", single[0]),
            "tiers": {k: v[0].text.strip() for k, v in named.items() if v},
        }
    return None


def team_page(t: Texts) -> str | None:
    """Which pre-match team page is shown: list, menu, details, stats; None for none or an overlay."""
    if t.has(r"view details", TEAM_MENU) and t.has(r"cancel", TEAM_MENU):
        return "menu"
    if t.has(r"view details|edit team|select this team|cancel|yes|no|ok", (600, 170, 1800, 1000)):
        return None
    header = t.has(r"moves\s*&\s*more", (780, 200, 1210, 280)) and t.has(r"stats", (1200, 200, 1560, 280))
    if header and t.has(r"team\s+\d+", (600, 70, 1200, 180)):
        return "stats" if len(t.find(r"sp\.?\s*(?:atk|def)", (330, 280, 2000, 960))) >= TEAM_ROWS else "details"
    if len(t.find(r"team\s+\d+", (330, 120, 2100, 210))) >= 2:
        return "list"
    return None


SAFE_CONTROL = r"ok|close|cancel|not now|(?:maybe )?later|no,? thanks|back|done"
UNSAFE_CONTROL = r".*(?:rate|submit|post|review|buy|purchase|confirm purchase|forfeit|quit|edit team|delete|double).*"


def safe_control(t: Texts, region: Rect, pattern: str = SAFE_CONTROL) -> dict | None:
    """Exactly one safe dismiss label inside a region, with no unsafe control sharing it."""
    safe = t.find(pattern, region)
    if len(safe) != 1:
        return None
    unsafe = [b for b in t.find(UNSAFE_CONTROL, region) if b is not safe[0]]
    return {
        "dismiss": target(safe[0].norm, safe[0]),
        "unsafe": tuple(tuple(round(v) for v in b.bounds) for b in unsafe),
    }


def battle_message(t: Texts) -> dict | None:
    message = t.any_text((420, 760, 1800, 890))
    if not message or not any(re.search(r"[a-z]{3,}.*[!.]$", b.norm) for b in message):
        return None
    text = " ".join(b.text for b in message)
    joined = " ".join(b.norm for b in message)
    hint = (
        "win" if re.search(r"\byou defeated\b", joined) else "loss" if re.search(r"\byou lost to\b", joined) else None
    )
    return {"message": text, "outcome_hint": hint}

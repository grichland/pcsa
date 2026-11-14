"""Raw battle-log rows to typed events. Every row yields exactly one event; no pattern -> `unknown`.

Nothing here knows about game state. Sides come from the text alone ("The opposing X" is p2, a bare
name p1); the translator turns events into protocol.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from pcsa.battle import battle_text
from pcsa.battle.log_ocr import ocr_boost_row
from pcsa.battle.log_patterns import (
    APOS,
    BOOST_WORDS,
    EVENT_KINDS,
    HAZARD_END,
    HAZARD_START,
    OVERLAY_TEXTS,
    SCREENS,
    STAT_IDS,
    SUBJECT_PATTERNS,
    TEMPLATE_EVENTS,
    TERRAIN,
    WEATHER,
)
from pcsa.battle.moves import complete_move_name

STAT_RE = "(?:" + "|".join(re.escape(s) for s in STAT_IDS) + ")"
CATEGORY_RE = re.compile(
    r"^(?:Move(?:\s.*)?|HP Lost|HP Restored|Held Item|Ability Changed|Ability|Fainted|Mega Evolution)$"
)
PRIORITY_RE = re.compile(r"^Move\s*([+\-—–])\s*(\d+)\s*Priority$")
# Champions writes some items with a quantity ("obtained a serving of Leftovers", "harvested a Sitrus Berry").
ITEM_ARTICLE_RE = re.compile(
    r"^(?:(?:an?|the|one|some)\s+)?(?:(?:serving|piece|bunch|pair)s?\s+of\s+)?", re.IGNORECASE
)


def item_name(text: str) -> str:
    return ITEM_ARTICLE_RE.sub("", text.strip()).strip()


@dataclass
class RawRow:
    row_id: str
    turn: int
    index: int
    text: str
    category: str
    message: str
    sprite: dict | None = None


@dataclass
class Event:
    kind: str
    row: RawRow
    side: str | None = None
    name: str | None = None
    data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {self.kind!r}")


Maker = Callable[..., Event]


def split_category(text: str) -> tuple[str, str]:
    if ": " in text:
        head, rest = text.split(": ", 1)
        if CATEGORY_RE.match(head.strip()):
            return head.strip(), rest.strip()
    return "", text.strip()


def who(text: str) -> tuple[str, str]:
    text = text.strip()
    match = re.match(r"^[Tt]he opposing (.+)$", text)
    return ("p2", match[1].strip()) if match else ("p1", text)


def parse_stats(text: str) -> list[str] | None:
    parts = re.split(r", and |, | and ", text)
    return [STAT_IDS[p] for p in parts] if all(p in STAT_IDS for p in parts) else None


# Matchers: each returns an Event or None ------------------------------------------------------------
def _category_rows(category: str, msg: str, ev: Maker) -> Event | None:
    if category in ("HP Lost", "HP Restored") or (not category and re.search(r"[’']s HP: \d", msg)):
        m = re.fullmatch(rf"(.+?){APOS}s HP: (\d+)\s*(%?)\s*(?:→|->|—>|=>)\s*(\d+)\s*(%?)", msg)
        if not m:
            return ev("unknown", reason="unreadable_hp_row")
        name, a, pa, b, pb = m.groups()
        if bool(pa) != bool(pb):
            return ev("unknown", reason="mixed_hp_units")
        unit = "pct" if pa else "hp"
        direction = {"HP Lost": "lost", "HP Restored": "restored"}.get(category)
        return ev(
            "hp",
            "p2" if unit == "pct" else "p1",
            name.strip(),
            before=int(a),
            after=int(b),
            unit=unit,
            direction=direction,
        )
    if category == "Held Item":
        m = re.fullmatch(rf"(.+?){APOS}s (.+)", msg)
        return ev("item", None, m[1].strip(), item=m[2].strip()) if m else ev("unknown", reason="unreadable_item_row")
    if category == "Ability Changed":
        m = re.fullmatch(r"(.+?): (.+?) (?:→|->) (.+)", msg)
        return (
            ev("ability_change", None, m[1].strip(), old=m[2].strip(), new=m[3].strip())
            if m
            else ev("unknown", reason="unreadable_ability_change_row")
        )
    if category == "Ability":
        m = re.fullmatch(rf"(.+?){APOS}s (.+)", msg)
        return (
            ev("ability", None, m[1].strip(), ability=m[2].strip())
            if m
            else ev("unknown", reason="unreadable_ability_row")
        )
    return None


def _switch_rows(msg: str, ev: Maker) -> Event | None:
    if m := re.fullmatch(r"Go!\s*(.+)!", msg):  # OCR can drop the space ("Go!Gholdengo!")
        return ev("send_out", "p1", m[1].strip(), trainer=None, title=None)
    if m := re.fullmatch(r"(.*?)\s*sent out (.+?)!?", msg):
        name, title = m[2].strip(), None
        if " the " in name:  # ribbon title: "Mimikyu the Sociable"
            name, title = name.split(" the ", 1)
            title = "the " + title
        return ev("send_out", "p2", name, trainer=m[1].strip() or None, title=title)
    if m := re.fullmatch(r"(.+?), come back!", msg):
        return ev("withdraw", "p1", m[1].strip(), trainer=None)
    if m := re.fullmatch(r"(.*?)\s*withdrew\s*(.+)!", msg):
        return ev("withdraw", "p2", m[2].strip(), trainer=m[1].strip() or None)
    if m := re.fullmatch(r"(.+?) was dragged out!", msg):
        side, name = who(m[1])
        return ev("drag", side, name)
    return None


def _mega_rows(msg: str, ev: Maker) -> Event | None:
    m = re.fullmatch(rf"(.+?){APOS}s (.+?) is reacting to (?:(.*?)\s*{APOS}s )?(.+?)!", msg) or re.fullmatch(
        rf"(.+?){APOS}?s (\S+ite(?: [XYZ])?) is reacting to ?(?:(.*?)\s*(?:{APOS}|\d)?s )?(Omni Ring)!", msg
    )
    if m:
        side, name = who(m[1])
        return ev("mega_reacting", side, name, stone=m[2].strip(), trainer=(m[3] or "").strip(), ring=m[4].strip())
    if m := re.fullmatch(r"(.+?) has Mega Evolved into (.+?)!?", msg):
        side, name = who(m[1])
        return ev("mega", side, name, mega=m[2].strip())
    return None


def _move_rows(category: str, msg: str, ev: Maker) -> Event | None:
    m = re.fullmatch(r"(.+?) used (.+?)!", msg)
    if m is None and category.startswith("Move"):
        m = re.fullmatch(r"(.+?) used (.+)", msg)  # OCR merges "!" into a final "l"
        if m and not complete_move_name(m[2].strip()):
            m = None
    if not m:
        return None
    side, name = who(m[1])
    priority = None
    if category.startswith("Move") and category != "Move":
        p = PRIORITY_RE.match(category)
        priority = (("-" if p[1] in "-—–" else "+") + p[2]) if p else category
    return ev("move", side, name, move=m[2].strip(), priority=priority, label=category or None)


def _boost_rows(msg: str, ev: Maker) -> Event | None:
    boost = re.fullmatch(
        rf"(.+?){APOS}s ({STAT_RE}(?:(?:, and |, | and )(?:{STAT_RE}))*) (sharply |drastically |harshly |severely )?"
        rf"(rose|fell|fel|won{APOS}t go any (?:higher|lower))( sharply| drastically| harshly| severely)?[!.]*",
        msg,
    )
    if boost:
        side, name = who(boost[1])
        verb, mod = boost[4], (boost[3] or boost[5] or "").strip()
        if verb.startswith("won"):
            amount, sign = 0, (1 if verb.endswith("higher") else -1)
        else:
            amount, sign = BOOST_WORDS[mod], (1 if verb == "rose" else -1)
        return ev("boost", side, name, stats=parse_stats(boost[2]), amount=amount, sign=sign)
    if m := re.fullmatch(rf"(.+?){APOS}s ({STAT_RE}) was not lowered!", msg):
        side, name = who(m[1])
        return ev("boost_blocked", side, name, stat=STAT_IDS[m[2]])
    if m := re.fullmatch(r"(.+?) returned its stats to normal using its White Herb!", msg):
        side, name = who(m[1])
        return ev("clearnegativeboost", side, name)
    if m := re.fullmatch(rf"(.+?) copied (.+?){APOS}s stat changes!", msg):
        side, name = who(m[1])
        target_side, target = who(m[2])
        return ev("copyboost", side, name, target_side=target_side, target=target)
    return None


def _effect_rows(msg: str, ev: Maker) -> Event | None:
    for pattern, value, variant in (
        (rf"It{APOS}s super effective!?", "super", "super"),
        (rf"It{APOS}s extremely effective!?", "super", "extremely"),
        (rf"It{APOS}s not very effective\.*", "resisted", "not_very"),
        (rf"It{APOS}s mostly ineffective\.*", "resisted", "mostly_ineffective"),
    ):
        if re.fullmatch(pattern, msg):
            return ev("effectiveness", value=value, text_variant=variant)
    if m := re.fullmatch(rf"It doesn{APOS}t affect (.+?)\.*", msg):
        side, name = who(m[1])
        return ev("effectiveness", side, name, value="immune", text_variant="immune")
    if re.fullmatch(r"But it failed!?", msg):
        return ev("fail")
    if m := re.fullmatch(r"The Pokémon was hit (\d+) times?!", msg):
        return ev("hitcount", count=int(m[1]))
    if m := re.fullmatch(r"(.+?) fainted!", msg):
        side, name = who(m[1])
        return ev("faint", side, name)
    if re.fullmatch(r"It hurt itself in its confusion!", msg):
        return ev("confusion_damage")
    return None


def _item_rows(msg: str, ev: Maker) -> Event | None:
    if m := re.fullmatch(rf"(.+?) knocked off (.+?){APOS}s (.+?)!", msg):
        side, name = who(m[1])
        tside, tname = who(m[2])
        return ev("knock_off", side, name, target_side=tside, target_name=tname, item=m[3].strip())
    if m := re.fullmatch(rf"(.+?) stole and ate its target{APOS}s (.+?)!", msg):
        side, name = who(m[1])
        return ev("steal_eat", side, name, item=m[2].strip())
    if m := re.fullmatch(r"(.+?) floats in the air with its (.+?)!", msg):
        side, name = who(m[1])
        return ev("item_reveal", side, name, item=m[2].strip())
    if m := re.fullmatch(rf"(.+?){APOS}s (Air Balloon) popped!", msg):
        side, name = who(m[1])
        return ev("item_end", side, name, item=m[2])
    if m := re.fullmatch(r"(.+?) can act faster than normal, thanks to its (.+?)!", msg):
        side, name = who(m[1])
        return ev("trailing", side, name, what="quickclaw", item=m[2].strip())
    if re.fullmatch(r"The .+ Gem strengthened .+!", msg):
        return ev("trailing", what="gem")
    return None


def _form_rows(msg: str, ev: Maker) -> Event | None:
    if m := re.fullmatch(r"(.+?) transformed into the (.+?) type!", msg):  # Protean / Libero
        side, name = who(m[1])
        return ev("typechange", side, name, type=m[2].strip())
    if m := re.fullmatch(r"(.+?) transformed into (.+?)!", msg):  # Transform / Imposter
        side, name = who(m[1])
        target_side, target = who(m[2])
        opposing = m[2].strip().lower().startswith("the opposing")
        return ev(
            "transform",
            side,
            name,
            target_side=target_side if opposing else ("p1" if side == "p2" else "p2"),
            target=target,
        )
    if m := re.fullmatch(rf"(.+?){APOS}s disguise was busted!", msg):
        side, name = who(m[1])
        return ev("form_change", side, name, form="Mimikyu-Busted", ability="Disguise")
    if m := re.fullmatch(r"(.+?) has been afflicted with an infestation by .+!", msg):
        side, name = who(m[1])
        return ev("volatile", side, name, effect="move: Infestation")
    if m := re.fullmatch(rf"It traced (.+?){APOS}s (.+?)!", msg):
        side, name = who(m[1])
        return ev("trailing", side, name, what="trace", ability=m[2].strip())
    return None


def _field_rows(msg: str, ev: Maker) -> Event | None:
    if re.fullmatch(r"(.+?) twisted the dimensions!", msg):
        return ev("field", condition="move: Trick Room", action="start")
    if msg == "The twisted dimensions returned to normal!":
        return ev("field", condition="move: Trick Room", action="end")
    if m := re.fullmatch(rf"(.+?){APOS}s health is sapped by Leech Seed!", msg):
        side, name = who(m[1])
        return ev("cause", side, name, cause="Leech Seed", applies_to="self", hp="lost")
    if re.fullmatch(r"The effects of the weather disappeared\.?", msg):
        return ev("trailing", what="weather_suppressed")
    if msg.rstrip(".!") in TERRAIN:
        condition, action = TERRAIN[msg.rstrip(".!")]
        return ev("field", condition=condition, action=action)
    if msg in WEATHER:
        weather, upkeep = WEATHER[msg]
        return ev("weather", weather=weather, upkeep=upkeep)
    return _hazard_rows(msg, ev)


def _hazard_rows(msg: str, ev: Maker) -> Event | None:
    for table, action in ((HAZARD_START, "start"), (HAZARD_END, "end")):
        for pattern, condition in table:
            if m := re.fullmatch(pattern, msg):
                screen = re.search(SCREENS, msg)
                found = condition if condition is not None else "move: " + (screen[0] if screen else "")
                return ev("hazard", "p2" if "opposing" in m[1].lower() else "p1", condition=found, action=action)
    if m := re.fullmatch(r"Pointed stones dug into (.+?)!", msg):
        side, name = who(m[1])
        return ev("cause", side, name, cause="Stealth Rock", applies_to="self", hp="lost")
    return None


def _subject_rows(msg: str, ev: Maker) -> Event | None:
    for pattern, kind, data in SUBJECT_PATTERNS:
        if m := re.fullmatch(rf"(.+?) {pattern}", msg):
            side, name = who(m[1])
            return ev(kind, side, name, **data)
    return None


def _template_rows(msg: str, ev: Maker) -> Event | None:
    matches = battle_text.lookup(msg)
    if not matches:
        return None
    unsupported = sorted({f"{m.name} ({m.key})" for m in matches if (m.effect, m.key) not in TEMPLATE_EVENTS})
    if unsupported:
        return ev("unknown", reason="unsupported mechanic: " + " or ".join(unsupported))
    events = []
    for match in matches:
        kind, data = TEMPLATE_EVENTS[(match.effect, match.key)]
        data = dict(data)
        subject = match.groups.get("POKEMON") or match.groups.get("NICKNAME")
        side, name = who(subject) if subject else (None, None)
        if kind == "hazard":
            side, name = ("p2" if "opposing" in match.groups["TEAM"].lower() else "p1"), None
        if kind in ("item", "item_reveal", "cure") and "ITEM" in match.groups:
            data["item"] = item_name(match.groups["ITEM"])
        data = {k: v.format(**match.groups) if isinstance(v, str) and "{" in v else v for k, v in data.items()}
        events.append((kind, side, name, tuple(sorted(data.items()))))
    if len(set(events)) != 1:
        return ev("unknown", reason="ambiguous_template")
    kind, side, name, data = events[0]
    return ev(kind, side, name, **dict(data))


def parse_row(row: RawRow) -> Event:
    """One raw row to one event. Category labels decide HP/item/ability rows; the message decides the rest."""
    category, msg = row.category, re.sub(r"\s+", " ", row.message.strip())

    def ev(kind: str, side: str | None = None, name: str | None = None, **data) -> Event:
        return Event(kind, row, side, name, data)

    if not category and msg.lower().rstrip(".!") in OVERLAY_TEXTS:
        return ev("overlay")

    matchers = (
        lambda: _category_rows(category, msg, ev),
        lambda: _switch_rows(msg, ev),
        lambda: _mega_rows(msg, ev),
        lambda: _effect_rows(msg, ev),
        lambda: _field_rows(msg, ev),
        lambda: _move_rows(category, msg, ev),
        lambda: _boost_rows(msg, ev),
        lambda: _item_rows(msg, ev),
        lambda: _form_rows(msg, ev),
        lambda: _subject_rows(msg, ev),
        lambda: _template_rows(msg, ev),
    )
    for matcher in matchers:
        event = matcher()
        if event is not None:
            return event
    # OCR drops a final "!" or reads it as "l"; moves keep their own complete-name check
    repairs = (msg + "!", msg[:-1] + "!") if msg.endswith("l") else (msg + "!",) if msg[-1:].isalnum() else ()
    for repaired in repairs:
        retried = parse_row(replace(row, message=repaired))
        if retried.kind not in ("unknown", "move"):
            return replace(retried, row=row)
    if (boost := ocr_boost_row(msg, ev)) is not None:
        return boost
    return ev("unknown", reason="no_pattern")


def rows_from_texts(turn: int, texts: list[str], sprites: list[dict | None] | None = None) -> list[RawRow]:
    """RawRows for one turn's merged log texts (and the row sprites read alongside them)."""
    found: list[dict | None] = list(sprites) if sprites else [None] * len(texts)
    out = []
    for index, text in enumerate(texts):
        category, message = split_category(text)
        out.append(RawRow(f"T{turn}#{index}", turn, index, text, category, message, sprite=found[index]))
    return out

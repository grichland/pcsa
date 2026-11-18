"""The strict check of one battle-log row: a known wording (assets/game/log_wordings.json) whose every hole reads
as a legal value. A row that passes is written in the wording's own spelling: its sentence end ("!", "..."), the
" → " arrow and the spaces around names. The trainer slot is never read: it is written empty."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cache

from pcsa.battle import game
from pcsa.phone import log_vocab as vocab

HOLE = re.compile(r"\{([A-Z]+)\}")
ARROW = re.compile(r"\s*(?:→|->|—>|=>|－>)\s*")
TAIL = re.compile(r"[\s!.…|:]*$")
BANG_MISREAD = "lI1|"  # a final "!" the recogniser read as a letter
CATEGORIES = (
    "Move",
    "HP Lost",
    "HP Restored",
    "Held Item",
    "Ability",
    "Ability Changed",
    "Fainted",
    "Mega Evolution",
)


@dataclass(frozen=True)
class Wording:
    id: str
    category: str  # "" for none
    body: str  # the text without its sentence end
    tail: str
    holes: tuple[str, ...]
    patterns: tuple[re.Pattern, ...]  # lazy and greedy hole matching


@dataclass
class Check:
    valid: bool
    text: str  # the wording's spelling when valid, else the reading as read
    wording: str | None = None
    reason: str | None = None
    snaps: list[dict] = field(default_factory=list)


@dataclass
class Fill:
    wording: Wording
    values: list[str]
    bad: list[int]  # holes whose reading is not a legal value


def _literal(text: str, before: str | None, after: str | None) -> str:
    out = re.escape(text.replace("'", "’"))
    out = out.replace("’", "[’']")
    if after and out.endswith(r"\ "):
        out = out[:-2] + r"\s*"  # a space before a hole can be lost ("Go!Gholdengo!")
    if before and out.startswith(r"\ "):
        out = r"\s*" + out[2:]
    if before == "TRAINER" and text.startswith("'s"):
        out = r"\s*[’']?\s*s\s*" + re.escape(text[2:].lstrip())  # "to's", "toy's", "tos Omni Ring"
    return out


def _compile(body: str) -> tuple[tuple[str, ...], tuple[re.Pattern, ...]]:
    parts = HOLE.split(body)
    holes = tuple(parts[1::2])
    patterns = []
    for hole_rx in ("(.+?)", "(.+)"):
        out = []
        for i, part in enumerate(parts):
            if i % 2:
                out.append("(.*?)" if part == "TRAINER" else r"(\d+)" if part == "N" else hole_rx)
            else:
                before = parts[i - 1] if i else None
                after = parts[i + 1] if i + 1 < len(parts) else None
                out.append(_literal(part, before, after))
        patterns.append(re.compile("".join(out)))
    return holes, tuple(patterns)


@cache
def wordings() -> tuple[Wording, ...]:
    rows = json.loads(game.asset_path("log_wordings.json").read_text(encoding="utf-8"))["wordings"]
    out = []
    for row in rows:
        text = row["text"]
        tail = TAIL.search(text)[0]
        body = text[: len(text) - len(tail)] if tail else text
        holes, patterns = _compile(body)
        category = "" if row["category"] == "-" else row["category"]
        out.append(Wording(row["id"], category, body, tail, holes, patterns))
    return tuple(out)


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().replace("'", "’").replace("‘", "’")
    return ARROW.sub(" → ", text)


def bodies(message: str) -> list[tuple[str, bool]]:
    """(body, last letter was a misread "!") candidates with the sentence end taken off."""
    stripped = TAIL.sub("", message)
    out = [(stripped, False)]
    if stripped == message and stripped[-1:] in BANG_MISREAD and len(stripped) > 1:
        out.append((stripped[:-1].rstrip(), True))
    return out


def render(wording: Wording, values: list[str]) -> str:
    out, k = [], 0
    for i, part in enumerate(HOLE.split(wording.body)):
        if not i % 2:
            out.append(part.replace("'", "’"))
            continue
        value = "" if part == "TRAINER" else values[k]
        if part == "MON":
            side, name = vocab.split_side(value)
            value = (("The" if not "".join(out) else "the") + " opposing " if side else "") + name
        out.append(value)
        k += 1
    text = re.sub(r" {2,}", " ", "".join(out)).strip()
    return text.replace(" !", "!") + wording.tail


def fills(category: str, message: str) -> list[Fill]:
    """Every way the message fits a wording of this category, legal or not."""
    found: list[Fill] = []
    for body, bang in bodies(clean(message)):
        for wording in wordings():
            if wording.category != category or (bang and wording.tail != "!"):
                continue
            for pattern in wording.patterns:
                m = pattern.fullmatch(body)
                if m is None:
                    continue
                values = [v.strip() for v in m.groups()]
                bad = [i for i, (h, v) in enumerate(zip(wording.holes, values, strict=True)) if not vocab.valid(h, v)]
                found.append(Fill(wording, values, bad))
    return found


def check(category: str, message: str) -> Check:
    """Strict check of one row (category label without the priority badge, message text)."""
    raw = f"{category}: {message}" if category else message
    if category and category not in CATEGORIES:
        return Check(False, raw, reason=f"unknown category {category!r}")
    found = fills(category, message)
    texts = {render(f.wording, f.values): f.wording.id for f in found if not f.bad}
    if len(texts) == 1:
        text, wording = next(iter(texts.items()))
        return Check(True, f"{category}: {text}" if category else text, wording)
    if texts:
        return Check(False, raw, reason="ambiguous wording: " + " | ".join(sorted(texts)))
    if found:
        f = found[0]
        bad = ", ".join(f"{f.wording.holes[i]}={f.values[i]!r}" for i in f.bad)
        return Check(False, raw, f.wording.id, reason=f"not a legal name: {bad}")
    return Check(False, raw, reason="no known wording")


def snap_check(category: str, message: str) -> Check:
    """The reading with each illegal name snapped to its unique nearest legal name (edit distance 1-2)."""
    raw = f"{category}: {message}" if category else message
    results: dict[str, tuple[str, list[dict]]] = {}
    unsure: list[str] = []
    for f in fills(category, message):
        values, snaps = list(f.values), []
        for i in f.bad:
            hole = f.wording.holes[i]
            to, candidates = vocab.snap(hole, values[i])
            if to is None:
                unsure.append(
                    f"{hole}={values[i]!r} near {candidates[:4]}" if candidates else f"{hole}={values[i]!r}"
                )
                break
            snaps.append({"hole": hole, "read": values[i], "to": to})
            values[i] = to
        else:
            if all(vocab.valid(h, v) for h, v in zip(f.wording.holes, values, strict=True)):
                results[render(f.wording, values)] = (f.wording.id, snaps)
    if len(results) == 1:
        text, (wording, snaps) = next(iter(results.items()))
        return Check(True, f"{category}: {text}" if category else text, wording, snaps=snaps)
    if results:
        return Check(False, raw, reason="ambiguous snap: " + " | ".join(sorted(results)))
    return Check(False, raw, reason="no unique legal name: " + "; ".join(unsure) if unsure else "no known wording")


def label_snap(label: str) -> str | None:
    """The category label a misread pill text stands for, when exactly one is within edit distance 2."""
    if label in CATEGORIES or not label:
        return label
    found, _ = vocab.nearest(CATEGORIES, label)
    return found

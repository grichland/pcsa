"""Conservative repair of stat-change rows the OCR damaged: the sentence shape must hold and every word must
read as one stat ("The onnosing Volcarona’s Sn. Atk Sn. Def, and Sneed rosel")."""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from pcsa.battle.log_patterns import APOS, BOOST_WORDS

if TYPE_CHECKING:
    from pcsa.battle.log_rows import Event

Maker = Callable[..., "Event"]
OCR_STATS = {"Attack": "atk", "Defense": "def", "Speed": "spe", "accuracy": "accuracy", "evasiveness": "evasion"}
OCR_SP_STATS = {"Atk": "spa", "Def": "spd"}
OCR_AND = {"and", "ana", "anc", "ard", "arid"}
OCR_WORD_MATCH = 0.6


def _close(word: str, choices, cutoff: float) -> str | None:
    found = difflib.get_close_matches(word.lower(), [c.lower() for c in choices], n=2, cutoff=cutoff)
    if len(found) != 1 and not (found and found[0] == word.lower()):
        return None
    return next(c for c in choices if c.lower() == found[0])


def _ocr_stats(text: str) -> list[str] | None:
    """Stat names from an OCR-damaged list ("Sn. Atk Sn. Def, and Sneed"): every word must be read as a stat."""
    words = [w for w in re.split(r"[\s,]+", text) if w and w.lower() not in OCR_AND]
    out, i = [], 0
    while i < len(words):
        word = words[i]
        if re.fullmatch(r"S[a-z]?\.", word) and i + 1 < len(words):
            half = _close(words[i + 1], list(OCR_SP_STATS), OCR_WORD_MATCH) or (
                "Def" if words[i + 1] in ("Det", "Dei") else None
            )
            if half is None:
                return None
            out.append(OCR_SP_STATS[half])
            i += 2
            continue
        stat = _close(word, list(OCR_STATS), OCR_WORD_MATCH)
        if stat is None:
            return None
        out.append(OCR_STATS[stat])
        i += 1
    return out if out and len(set(out)) == len(out) else None


def ocr_boost_row(msg: str, ev: Maker) -> Event | None:
    """A stat-change row the OCR damaged ("The onnosing Volcarona’s Sn. Atk Sn. Def, and Sneed rosel"): the
    sentence shape is kept, each part must still read unambiguously, and the name is resolved later on the roster."""
    m = re.fullmatch(
        rf"(.+?){APOS}s (.+?) (sharply |harshly |drastically |severely )?(rose|fell?)l?[!.l]*", msg, re.IGNORECASE
    )
    if not m:
        return None
    words = m[1].split()
    side, name = "p1", m[1].strip()
    for k in range(min(3, len(words) - 1)):  # "The opposing" read as "The onnosing" or "I ne opposing"
        if difflib.SequenceMatcher(None, words[k].lower(), "opposing").ratio() >= OCR_WORD_MATCH:
            side, name = "p2", " ".join(words[k + 1 :])
            break
    stats = _ocr_stats(m[2])
    if stats is None or not name:
        return None
    mod = (m[3] or "").strip().lower()
    sign = 1 if m[4].lower() == "rose" else -1
    return ev("boost", side, name, stats=stats, amount=BOOST_WORDS[mod], sign=sign, ocr_repaired=True)

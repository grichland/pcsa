"""Our visible rank on the lobby and reward screens."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pcsa.phone.ocr import TextBox

TIERS = {
    "pokeball": "Poke Ball",
    "greatball": "Great Ball",
    "ultraball": "Ultra Ball",
    "masterball": "Master Ball",
    "champion": "Champion",
}
LOBBY_REGION = (1200, 300, 2050, 450)
REWARDS_REGION = (1200, 250, 2200, 550)
REWARDS_RANK_X = (1550, 1900)
REWARDS_RANK_Y = (360, 450)


def read_rank(boxes: Sequence[TextBox], *, lobby: bool, min_confidence: float = 0.80) -> dict | None:
    region = LOBBY_REGION if lobby else REWARDS_REGION
    selected = [
        b
        for b in boxes
        if b.confidence >= min_confidence
        and region[0] <= b.center[0] <= region[2]
        and region[1] <= b.center[1] <= region[3]
    ]
    values: dict[str, list] = {"tier": [], "rank": [], "placement": [], "rating": []}
    for box in selected:
        text = box.norm
        compact = re.sub(r"\s+", "", text)
        tier = re.search(r"(pokeball|greatball|ultraball|masterball|champion)tier", compact)
        if tier:
            values["tier"].append(TIERS[tier[1]])
        for key, pattern in (
            ("rank", r"\brank\s*(\d+)\b"),
            ("placement", r"\bplacement:\s*no\.?\s*([\d,]+)"),
            ("rating", r"\brating:\s*([\d,]+\.\d+)"),
        ):
            match = re.search(pattern, text)
            if match:
                number = match[1].replace(",", "")
                values[key].append(float(number) if key == "rating" else int(number))
        bare_rank = (
            not lobby
            and any(b.norm == "rank" for b in selected)
            and re.fullmatch(r"\d+", text)
            and REWARDS_RANK_X[0] <= box.center[0] <= REWARDS_RANK_X[1]
            and REWARDS_RANK_Y[0] <= box.center[1] <= REWARDS_RANK_Y[1]
        )
        if bare_rank:
            values["rank"].append(int(text))
    result = {key: entries[0] if entries and len(set(entries)) == 1 else None for key, entries in values.items()}
    if not any(value is not None for value in result.values()):
        return None
    result["raw"] = [b.record() for b in selected]
    return result


def rank_text(rank: dict | None) -> str:
    if rank is None:
        return "unknown"
    return "; ".join(
        f"{k}={rank.get(k) if rank.get(k) is not None else 'unknown'}"
        for k in ("tier", "rank", "placement", "rating")
    )

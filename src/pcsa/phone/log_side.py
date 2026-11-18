"""Which side a battle-log row belongs to, from the colour of its category pill. Reference frame.

Our pills are blue/purple, the opponent's red/pink; plain sentences have no pill. The pill is far brighter than
the translucent panel behind it, so bright saturated pixels in the pill column find it and their hue names the side.
Priority moves add a neutral purple "+1 Priority" badge under the pill (x 602-817); the badge never reaches the
pill's left end, so a line counts only when that left end is lit too.
"""

from __future__ import annotations

import cv2
import numpy as np

PILL_X = (575, 820)  # inside the pill (x 556-863), clear of its rounded right end
PILL_END = 20  # first columns of PILL_X: lit by the pill, never by the priority badge
PILL_MIN_V = 180  # recorded rows: plain rows reach 6 lines at this level, pill rows at least 49
PILL_MIN_S = 80  # drops the white label text and its anti-aliased edge
PILL_LINE_FILL = 0.5  # share of a line's pixels that must be pill-coloured
PILL_MIN_LINES = 24  # a full pill is ~50 lines tall
OPP_HUE = (145, 15)  # OpenCV hue (0-179) wraps through red: >= 145 or <= 15 is the red/pink pill; ours sits ~123
OPP_SHARE = 0.5  # share of pill pixels in the red range that makes it the opponent's


def pill_lines(mask: np.ndarray) -> np.ndarray:
    """Lines of the band that cross the pill (not the priority badge)."""
    return (mask.mean(axis=1) >= PILL_LINE_FILL) & (mask[:, :PILL_END].mean(axis=1) >= PILL_LINE_FILL)


def pill_stats(frame: np.ndarray, bounds: tuple[int, int]) -> dict:
    """Pill measurements for one row band of a reference-size BGR frame."""
    top, bottom = bounds
    if bottom <= top:
        return {"lines": 0, "pixels": 0}
    hsv = cv2.cvtColor(frame[top:bottom, PILL_X[0] : PILL_X[1]], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (v >= PILL_MIN_V) & (s >= PILL_MIN_S)
    lines = pill_lines(mask)
    picked = mask & lines[:, None]
    hue = h[picked].astype(int)
    stats: dict = {"lines": int(lines.sum()), "pixels": int(hue.size)}
    if hue.size:
        stats.update(
            hue=float(np.median(hue)),
            sat=float(np.median(s[picked])),
            val=float(np.median(v[picked])),
            opp_share=round(float(np.mean((hue >= OPP_HUE[0]) | (hue <= OPP_HUE[1]))), 4),
        )
    return stats


def side_from_stats(stats: dict) -> str | None:
    """Returns "own", "opp", or None when no pill is found."""
    if stats["lines"] < PILL_MIN_LINES:
        return None
    return "opp" if stats["opp_share"] >= OPP_SHARE else "own"


def row_side(frame: np.ndarray, bounds: tuple[int, int]) -> str | None:
    return side_from_stats(pill_stats(frame, bounds))


def combine_sides(first: str | None, second: str | None) -> str | None:
    """Keep a side seen in any view of a row; two views that disagree make it "unknown"."""
    if first is None or first == second:
        return second
    if second is None:
        return first
    return "unknown"

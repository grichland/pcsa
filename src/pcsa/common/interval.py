"""The Wilson score interval of a win rate, for simulator matches."""

from __future__ import annotations

import math

Z95 = 1.959964  # the normal quantile of a 95% interval


def wilson(wins: int, n: int) -> tuple[float, float]:
    """95% interval of a win rate of `wins` in `n` decided games ((0, 1) without a game)."""
    z = Z95
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)

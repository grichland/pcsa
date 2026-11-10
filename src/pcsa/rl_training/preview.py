# From research/pcsa-rl/rl/preview.py (sha256 d4e254a98159, 2026-11-08 19:55); keep diffable.
"""Team preview: the 60 (lead, second, third) picks and the critic-greedy nucleus sampler over them."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


def all_picks(n: int = 6, bring: int = 3) -> list[tuple[int, ...]]:
    """Every brought set with each member as lead: (lead, *rest ascending); C(6,3) x 3 = 60."""
    return [(lead, *(m for m in trio if m != lead)) for trio in combinations(range(n), bring) for lead in trio]


def preview_distribution(values: np.ndarray, *, tau: float = 0.1, top_p: float = 0.9) -> np.ndarray:
    """softmax(values / tau) cut to the smallest top set holding top_p of the mass, renormalised."""
    z = np.asarray(values, np.float64) / tau
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(-p, kind="stable")
    keep = int(np.searchsorted(np.cumsum(p[order]), top_p, side="left")) + 1
    out = np.zeros_like(p)
    out[order[:keep]] = p[order[:keep]]
    return out / out.sum()


def choose_preview(values: np.ndarray, rng: Any, *, tau: float = 0.1, top_p: float = 0.9) -> int:
    """values [60] -> index of the sampled pick; rng is anything with .random() (numpy Generator, random.Random)."""
    p = preview_distribution(values, tau=tau, top_p=top_p)
    cdf = np.cumsum(p)
    index = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
    return min(index, int(np.flatnonzero(p)[-1]))


def preview_entropy(values: np.ndarray, *, tau: float = 0.1, top_p: float = 0.9) -> float:
    """Entropy (nats) of the preview sampling distribution, for logging."""
    p = preview_distribution(values, tau=tau, top_p=top_p)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

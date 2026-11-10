# From research/pcsa-rl/rl/entropy.py (sha256 0b01578626e9, 2026-11-08 19:55); keep diffable.
# Inference only: EntropyCoef, approx_kl and cap_fraction (PPO learner) stay in research.
"""Entropy guard: the 0.99 probability cap and normalised entropy."""

from __future__ import annotations

import torch
from torch import Tensor

CAP = 0.99


def capped_probs(logits: Tensor, mask: Tensor, *, cap: float = CAP, temperature: float = 1.0) -> Tensor:
    """softmax(logits / T) over legal actions mixed with uniform: pi' = (1 - eps) pi + eps / k, eps = (1-cap) k/(k-1).

    k == 1 gives that action probability 1; rows with no legal action come back all zero.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0; take the argmax for greedy play")
    any_legal = mask.any(-1, keepdim=True)
    z = torch.where(mask, logits / temperature, torch.full_like(logits, -torch.inf))
    z = torch.where(any_legal, z, torch.zeros_like(z))
    pi = torch.where(mask, torch.softmax(z, dim=-1), torch.zeros_like(z))
    k = mask.sum(-1, keepdim=True).to(pi.dtype)
    eps = torch.where(k > 1, (1 - cap) * k / (k - 1).clamp(min=1), torch.zeros_like(k))
    mixed = (1 - eps) * pi + eps / k.clamp(min=1)
    return torch.where(mask, mixed, torch.zeros_like(mixed))


def log_prob(probs: Tensor, actions: Tensor) -> Tensor:
    """log pi'(a) per decision."""
    return probs.gather(-1, actions.long()[..., None])[..., 0].clamp(min=1e-12).log()


def normalised_entropy(probs: Tensor, mask: Tensor) -> Tensor:
    """H(pi') / log k per decision; 0 where k <= 1."""
    plogp = torch.where(probs > 0, probs * probs.clamp(min=1e-12).log(), torch.zeros_like(probs))
    h = -plogp.sum(-1)
    k = mask.sum(-1).to(probs.dtype)
    return torch.where(k > 1, h / k.clamp(min=2).log(), torch.zeros_like(h))

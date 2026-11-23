"""The decision rule and its two masks: the legal mask on our actions and the opponent prior on theirs.

score[a] = sum_b M[a, b] * p_opp[b];  policy = softmax(score / tau) over legal actions only.
No action of either side gets more than `max_prob` while another action is possible. MAX_PROB is only the
default: the network passes its checkpoint's own max_prob (model.Config).

Copied from pcsa-former (pcsa_former/decide.py); behaviour is kept identical to it.
"""

from __future__ import annotations

import torch
from torch import Tensor

NEG_INF = float("-inf")
MAX_PROB = 0.9


def cap(p: Tensor, possible: Tensor, max_prob: float = MAX_PROB) -> Tensor:
    """`p` with its largest entry held to `max_prob`; the excess goes to the other possible actions in proportion to
    what they had (evenly when they had nothing). A row with one possible action keeps probability 1 on it.
    Impossible actions stay at exactly 0."""
    top, idx = p.max(-1, keepdim=True)
    is_top = torch.zeros_like(possible).scatter_(1, idx, True)
    others = possible & ~is_top
    n_others = others.sum(-1, keepdim=True)
    rest = (p * others).sum(-1, keepdim=True)
    share = torch.where(rest > 1e-12, p * others / rest.clamp_min(1e-12), others / n_others.clamp_min(1))
    capped = torch.where(is_top, torch.full_like(p, max_prob), (1.0 - max_prob) * share)
    return torch.where((top > max_prob) & (n_others > 0), capped, p)


def safe_log(p: Tensor) -> Tensor:
    """log p, with -inf where p is exactly 0 and no NaN gradient there."""
    return torch.where(p > 0, p.clamp_min(1e-38).log(), torch.full_like(p, NEG_INF))


def opp_distribution(logits: Tensor, prior: Tensor, max_prob: float = MAX_PROB) -> tuple[Tensor, Tensor, Tensor]:
    """(p [N,13] capped, log p uncapped, all_zero [N]) from softmax(logits + log prior). A prior of 0 gives exactly 0;
    a row whose priors are all 0 falls back to uniform and is flagged."""
    possible = prior > 0
    all_zero = ~possible.any(-1)
    z = logits + torch.where(possible, prior.clamp_min(1e-30).log(), torch.full_like(prior, NEG_INF))
    z = torch.where(all_zero[:, None], torch.zeros_like(z), z)
    logp = z.log_softmax(-1)
    return cap(logp.exp(), possible | all_zero[:, None], max_prob), logp, all_zero


def expected_score(matrix: Tensor, p_opp: Tensor) -> Tensor:
    """[N,11,13], [N,13] -> [N,11]. A column of probability 0 contributes nothing, whatever it holds."""
    live = (p_opp > 0)[:, None, :]
    return (torch.where(live, matrix, torch.zeros_like(matrix)) * p_opp[:, None, :]).sum(-1)


def policy(score: Tensor, mask: Tensor, tau: Tensor | float = 1.0, max_prob: float = MAX_PROB) -> tuple[Tensor, Tensor, Tensor]:
    """(pi, legal, no_legal [N]): softmax(score / tau) over the legal actions, capped; exactly 0 on illegal ones.
    A row with no legal action (not a request of this kind) is treated as all legal so nothing is NaN, and flagged."""
    legal = mask.bool()
    no_legal = ~legal.any(-1)
    legal = legal | no_legal[:, None]
    score = torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    logits = torch.where(legal, score / tau, torch.full_like(score, NEG_INF))
    return cap(logits.softmax(-1), legal, max_prob), legal, no_legal


def entropy(pi: Tensor) -> Tensor:
    logp = safe_log(pi)
    return -(pi * torch.where(pi > 0, logp, torch.zeros_like(logp))).sum(-1)


def choose(pi: Tensor, legal: Tensor, greedy: bool = False, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """(action [N], its log-probability). With one legal action that action is returned whatever `pi` says."""
    forced = legal.sum(-1) == 1
    only = legal.float().argmax(-1)
    if greedy:
        action = torch.where(legal, pi, torch.full_like(pi, -1.0)).argmax(-1)
    else:
        safe = torch.where(forced[:, None], legal.to(pi.dtype), pi)
        action = torch.multinomial(safe, 1, generator=generator)[:, 0]
    action = torch.where(forced, only, action)
    logp = safe_log(pi).gather(1, action[:, None])[:, 0]
    return action, torch.where(forced, torch.zeros_like(logp), logp)

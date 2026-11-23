"""How our pick at a turn request is made from the network's matrix (the policy's `choice` argument).

The network scores each of our actions by the expected cell over their replies (decide.expected_score). Here only
their `top_theirs` likeliest replies (by p_opp) and our `top_ours` likeliest actions (by pi) can be kept (0: all),
and each of our rows is reduced over the kept replies by `rule`:
- mean: the expected cell, the kept replies weighted by p_opp renormalised (with no top-k: the network's own pick)
- min: the worst cell (maximin)
- mix: blend * mean + (1 - blend) * min
- regret: minus the largest shortfall against the best of our kept actions for each kept reply (minimax regret)
The returned scores are log softmax(score / tau) over our kept actions, so the shell's argmax plays the rule's pick.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pcsa.common.config_cg import ChoiceSettings

from . import layout as L

NEG_INF = float("-inf")


def config(args: Mapping[str, Any] | None) -> ChoiceSettings | None:
    """The policy's `choice` argument checked; None when it changes nothing."""
    cfg = ChoiceSettings.model_validate(args) if args else None
    return cfg if cfg is not None and cfg.active else None


def likeliest(p: Tensor, allowed: Tensor, k: int) -> Tensor:
    """[N,A] bool: the k largest entries of p among `allowed` (k <= 0: all allowed)."""
    if k <= 0 or k >= p.shape[-1]:
        return allowed
    idx = torch.where(allowed, p, torch.full_like(p, -1.0)).topk(k, dim=-1).indices
    return torch.zeros_like(allowed).scatter(-1, idx, True) & allowed


def scores(matrix: Tensor, p_opp: Tensor, pi: Tensor, legal: Tensor, cfg: ChoiceSettings) -> Tensor:
    """[N,11,13], [N,13], [N,11], [N,11] bool -> [N,11]: each of our actions' score under the rule, -inf on the
    actions not kept. A row with no prediction of theirs counts every reply equally."""
    m = matrix.float()
    ours = likeliest(pi, legal, cfg.top_ours)
    possible = p_opp > 0
    possible = possible | ~possible.any(-1, keepdim=True)
    theirs = likeliest(p_opp, possible, cfg.top_theirs)
    w = torch.where(theirs, p_opp.float(), torch.zeros_like(p_opp, dtype=torch.float))
    total = w.sum(-1, keepdim=True)
    w = torch.where(total > 0, w / total.clamp_min(1e-12), theirs.float() / theirs.sum(-1, keepdim=True).clamp_min(1))
    kept = theirs[:, None, :]
    mean = (torch.where(kept, m, torch.zeros_like(m)) * w[:, None, :]).sum(-1)
    worst = torch.where(kept, m, torch.full_like(m, float("inf"))).amin(-1)
    if cfg.rule == "mean":
        s = mean
    elif cfg.rule == "min":
        s = worst
    elif cfg.rule == "mix":
        s = cfg.blend * mean + (1.0 - cfg.blend) * worst
    else:
        best = torch.where(ours[:, :, None], m, torch.full_like(m, NEG_INF)).amax(1)  # [N,13]
        shortfall = torch.where(kept, best[:, None, :] - m, torch.full_like(m, NEG_INF))
        s = -shortfall.amax(-1)
    return torch.where(ours, s, torch.full_like(s, NEG_INF))


def _plain(row: Tensor) -> list:
    return [round(float(x), 4) if torch.isfinite(x) else None for x in row]


def apply(
    cfg: ChoiceSettings | None, kinds: np.ndarray, out: dict, scores_out: Tensor, tau: float
) -> list[dict | None]:
    """With a rule set: each turn row's first 11 scores replaced by the rule's; the note per row (else None)."""
    notes: list[dict | None] = [None] * len(kinds)
    rows = [r for r, k in enumerate(kinds) if int(k) == L.KIND_TURN]
    if cfg is None or not rows:
        return notes
    get = lambda key: out[key].detach().cpu()[rows]  # noqa: E731
    s = scores(get("matrix"), get("p_opp"), get("pi"), get("legal").bool(), cfg)
    logp = torch.log_softmax(s / max(float(tau), 1e-6), -1)
    for i, r in enumerate(rows):
        model_pick = int(scores_out[r, : L.OWN_ACTIONS].argmax())
        pick = int(s[i].argmax())
        scores_out[r, : L.OWN_ACTIONS] = logp[i]
        notes[r] = {**cfg.model_dump(), "model_pick": model_pick, "pick": pick, "changed": pick != model_pick,
                    "scores": _plain(s[i])}  # fmt: skip
    return notes

"""Guard-rule masks on the network's actions (neural.yaml `rule_mode: mask`).
`FormerPlayer.rule_mask(view, out)` returns {legal action's message: (rule, reason)}; masked rows leave the legality
mask
before the blend and softmax. All masked: the masks are dropped (noted); a rule error: no masks. `out` gains
rule_masked,
rule_legal, rule_network, rule_mask_ignored / rule_mask_error.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from pcsa.rl_training import features

ERROR_WIDTH = 300


def apply(rule_mask: Any, view, obs, out: dict) -> dict:
    """`out` with the rule masks of this decision (see the module notes); unchanged when off or on a forced switch."""
    if rule_mask is None or obs.forced:
        return out
    try:
        forbidden = rule_mask(view, out)
        masked = {}
        for action in view.legal:
            if action.message in forbidden:
                index = features.action_index(action, view.battle)
                if 0 <= index < len(obs.mask) and obs.mask[index]:
                    masked[index] = tuple(forbidden[action.message])
    except Exception as error:
        return {**out, "rule_mask_error": f"{type(error).__name__}: {error}"[:ERROR_WIDTH]}
    if not masked:
        return out
    legal = np.asarray(obs.mask, bool).copy()
    legal[list(masked)] = False
    if not legal.any():
        return {**out, "rule_mask_ignored": "every legal action was masked", "rule_masked_all": masked}
    extra: dict = {"rule_masked": masked, "rule_legal": legal}
    logits = np.asarray(out["logits"].detach().float().cpu().numpy(), float).reshape(-1)
    own = int(np.argmax(np.where(np.asarray(obs.mask, bool), logits, -np.inf)))
    if own in masked:
        with contextlib.suppress(ValueError):
            extra["rule_network"] = features.legal_action(own, view.legal, view.battle).message
    return {**out, **extra}


def legal(obs, out: dict) -> np.ndarray:
    """The rows the policy may choose: the legality mask without the masked rows."""
    return out["rule_legal"] if "rule_legal" in out else np.asarray(obs.mask, bool)


def kept(view, out: dict) -> list:
    """The legal actions that are not masked (what the matrix blend searches)."""
    masked = out.get("rule_masked")
    if not masked:
        return list(view.legal)
    return [a for a in view.legal if features.action_index(a, view.battle) not in masked]


def notes(into: dict, out: dict) -> None:
    """The decision's notes: matrix.masked {row: [rule, reason]}, mask_rules, network (the network's own pick when it
    was masked), mask_ignored / mask_error."""
    masked = out.get("rule_masked") or {}
    if masked:
        if isinstance(into.get("matrix"), dict):
            into["matrix"]["masked"] = {str(i): [r, why] for i, (r, why) in sorted(masked.items())}
        into["mask_rules"] = sorted({r for r, _ in masked.values()})
    if "rule_network" in out:
        into["network"] = out["rule_network"]
    if "rule_mask_ignored" in out:
        into["mask_ignored"] = out["rule_mask_ignored"]
        into["mask_would"] = {str(i): [r, why] for i, (r, why) in sorted(out["rule_masked_all"].items())}
    if "rule_mask_error" in out:
        into["mask_error"] = out["rule_mask_error"]

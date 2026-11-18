"""A pcsa-former checkpoint as a `champions_gym.neural` policy, with the network's outputs in each decision's extras.

    player: "champions_gym.neural:load"
    player_args: {policy: "pcsa.network.policy:Policy", checkpoint: ..., device: "cpu", policy_args: {threads: 4}}

Recreated from cg-publish docs/examples/pcsa_former_policy.py: one step of the network per request with that game's
memory, returning log(pi) and log(preview_pi), so the shell's argmax is the greedy action of decide.choose. A request
answered without asking (a restart's replay) comes to `observe`: the same forward step, so the memory after a restart
is the memory the model had. One lock per policy object around every call.

Here the belief heads run too (they change no other output) and every decision's extras carry, as plain JSON with
CG string ids (the record contract's `notes.cg.net`):
- turn and replace: tau, pi, score, p_opp, matrix [11][13], rows (our action names), cols (theirs), belief
- preview: belief (with `lead`) and picks_top, the five likeliest (lead, second, third) picks
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from . import cg_adapter, model, vocab
from . import checkpoint as ckpt
from . import layout as L

VOCAB_NAMES = {"species": "species", "item": "items", "ability": "abilities", "move": "moves"}  # ours -> CG's
TOP_ITEMS, TOP_ABILITIES, TOP_MOVES, TOP_PICKS = 3, 3, 8, 5
INPUTS = ("rows", "row_ids", "field", "last", "last_ids", "mask", "opp_prior")


def _tensor(x: Any, device: str) -> torch.Tensor:
    """CG's array (numpy, or a memoryview without numpy) as a tensor, without a copy where possible."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)
    return torch.from_numpy(np.asarray(x)).to(device)


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def decide_log(p: torch.Tensor) -> torch.Tensor:
    """log p with -inf where p is 0 (decide.safe_log)."""
    return torch.where(p > 0, p.clamp_min(1e-38).log(), torch.full_like(p, float("-inf")))


def _top(p: torch.Tensor, kind: str, k: int) -> list[list]:
    """The k largest entries of p (id 0, unknown, left out; zeros left out) as [[CG id, p]]."""
    p = p.clone()
    p[0] = 0.0
    values, ids = p.topk(min(k, p.numel()))
    return [[vocab.name(kind, int(i)), float(v)] for v, i in zip(values, ids, strict=True) if v > 0]


class Policy:
    """`checkpoint`: a .pt saved by checkpoint.save. `device`: "cpu" for live play. `threads`: torch CPU threads.
    `requery_refused`: ask the network again after a refused choice instead of the shell reusing the first answer.
    `allow_contract`: load a checkpoint made under another contract version; CG's own check still reports it."""

    def __init__(self, checkpoint: str, device: str = "cpu", threads: int = 4, requery_refused: bool = False,
                 allow_contract: bool = False):  # fmt: skip
        torch.set_num_threads(int(threads))
        self.device = device
        lay = cg_adapter.layout()  # the layout from champions_gym.FEATURES, checked against CG
        extra = {"allow_contract": True} if allow_contract else {}
        self.net, ck = ckpt.load(checkpoint, lay, cg_adapter.learnset(), device, **extra)  # LayoutMismatch
        self.net.eval()
        self.memory: model.Memory | None = None
        self.lock = threading.Lock()
        self.cols = {"species": lay.col(L.F_SPECIES), "moves": lay[L.F_MOVE_ID].columns(),
                     "active": lay.col(L.F_IS_ACTIVE)}  # fmt: skip
        lt = ck["layout"]
        w = lt["widths"]
        self.info: dict[str, Any] = {
            "contract": lt.get("contract"),
            "layout_hash": lt["hash"],
            "shapes": {
                "rows": [L.ROWS, w["rows"]],
                "row_ids": [L.ROWS, w["row_ids"]],
                "field": [w["field"]],
                "last": [w["last"]],
                "last_ids": [w["last_ids"]],
                "mask": [L.OWN_ACTIONS],
                "opp_prior": [L.OPP_ACTIONS],
            },
            "vocab": {VOCAB_NAMES[k]: int(v) for k, v in lt["vocab"].items()},
            "model_id": os.path.abspath(checkpoint),
            "checkpoint": os.path.abspath(checkpoint),
            "model_hash": _file_hash(checkpoint),
            "requery_refused": bool(requery_refused),
            "run": ck.get("extra", {}).get("run"),
            "iteration": ck.get("extra", {}).get("iteration"),
            "memory": self.net.cfg.memory,
        }

    def _grow(self, need: int) -> model.Memory:
        m = self.memory
        if m is None or m.hist.shape[0] < need:
            new = model.Memory(self.net, max(need, 2 * (m.hist.shape[0] if m is not None else 1)), self.device)
            if m is not None:
                k = m.hist.shape[0]
                new.hist[:k], new.pos[:k], new.count[:k] = m.hist, m.pos, m.count
            self.memory = m = new
        return m

    def reset(self, slots: Sequence[int]) -> None:
        with self.lock:
            if self.memory is not None:
                ok = [s for s in slots if s < self.memory.hist.shape[0]]
                if ok:
                    self.memory.reset(torch.as_tensor(ok, dtype=torch.long, device=self.device))

    def batch(self, arrays: Any) -> dict:
        """The model's inputs from CG's arrays."""
        b = {k: _tensor(getattr(arrays, k), self.device) for k in INPUTS}
        b["preview_mask"] = _tensor(arrays.action_mask, self.device)[:, L.PREVIEW_OFFSET :]
        return b

    def _forward(self, b: dict, slots: Sequence[int]) -> dict:
        """One step of each slot's game: the forward pass, which also adds this decision to the memory."""
        mem = self._grow(max(slots) + 1)
        s = torch.as_tensor(list(slots), dtype=torch.long, device=self.device)
        return self.net.forward(b, memory=mem, slots=s, beliefs=True)

    @torch.no_grad()
    def observe(self, arrays: Any, slots: Sequence[int], actions: Sequence[int | None] = ()) -> None:
        """Requests answered without asking: the memory step only (the action does not enter the memory)."""
        with self.lock:
            self._forward(self.batch(arrays), slots)

    @torch.no_grad()
    def decide(self, arrays: Any, slots: Sequence[int], battles: Any = None) -> dict[str, Any]:
        b = self.batch(arrays)
        with self.lock:
            out = self._forward(b, slots)
        scores = torch.cat([decide_log(out["pi"]), decide_log(out["preview_pi"])], dim=1).cpu()  # [B, 71]
        value = out["value"].cpu()
        try:
            extras = self.all_extras(arrays, b, out, len(slots))
        except Exception as error:  # display only: the decision stands without them
            extras = [{"error": f"extras: {type(error).__name__}: {error}"} for _ in slots]
        return {"scores": scores, "value": value, "extras": extras}

    def all_extras(self, arrays: Any, b: dict, out: dict, n: int) -> list[dict]:
        kind = _tensor(arrays.kind, self.device)
        tau = float(self.net.temperature())
        out = {k: v.cpu() for k, v in out.items()}
        ids, rows = b["row_ids"].cpu().long(), b["rows"].cpu()
        return [self.extras(out, r, int(kind[r]), ids[r], rows[r], tau) for r in range(n)]

    def extras(self, out: dict, r: int, kind: int, ids: torch.Tensor, rows: torch.Tensor, tau: float) -> dict:
        """Row r's outputs as plain JSON (module docstring). Never raises: a part that cannot be made is left out
        and named in `error`."""
        if kind == L.KIND_PREVIEW:
            parts = {
                "belief": lambda: self.belief(out, r, ids, preview=True),
                "picks_top": lambda: self.picks_top(out, r),
            }
        else:
            parts = {
                "tau": lambda: tau,
                "pi": lambda: out["pi"][r].tolist(),
                "score": lambda: out["score"][r].tolist(),
                "p_opp": lambda: out["p_opp"][r].tolist(),
                "matrix": lambda: out["matrix"][r].tolist(),
                "rows": lambda: self.own_names(ids, rows),
                "cols": lambda: self.opp_names(ids),
                "belief": lambda: self.belief(out, r, ids, preview=False),
            }
        made, errors = {}, []
        for key, make in parts.items():
            try:
                made[key] = make()
            except Exception as error:
                errors.append(f"{key}: {type(error).__name__}: {error}")
        if errors:
            made["error"] = "; ".join(errors)
        return made

    def picks_top(self, out: dict, r: int) -> list[list]:
        """The likeliest (lead, second, third) picks as [[team indices], p]."""
        top = out["preview_pi"][r].topk(TOP_PICKS)
        combos = L.preview_combos()
        return [[list(combos[int(a)]), float(p)] for p, a in zip(*top, strict=True) if p > 0]

    def _species(self, ids: torch.Tensor, row: int) -> str | None:
        return vocab.name("species", int(ids[row, self.cols["species"]]))

    def own_names(self, ids: torch.Tensor, rows: torch.Tensor) -> list[str | None]:
        """Our 11 actions: the active's moves by slot, the same with Mega, the species of rows 0-2."""
        active = int(rows[: L.OWN_PICKS, self.cols["active"]].argmax())
        moves = [vocab.name("moves", int(m)) for m in ids[active, self.cols["moves"]]]
        return moves + moves + [self._species(ids, p) for p in range(L.OWN_PICKS)]

    def opp_names(self, ids: torch.Tensor) -> list[str | None]:
        """Their 13 actions: row 6's moves by slot ("" for the first empty slot: a move not yet seen, None after),
        the same with Mega, the species of rows 7-11."""
        moves: list[str | None] = [None] * L.MOVE_SLOTS
        if self._species(ids, L.OPP_ACTIVE) is not None:
            unseen = False
            for s, m in enumerate(ids[L.OPP_ACTIVE, self.cols["moves"]]):
                moves[s] = vocab.name("moves", int(m))
                if moves[s] is None and not unseen:
                    moves[s], unseen = "", True
        bench = range(L.OPP_BENCH.start, L.OPP_BENCH.stop)
        return moves + moves + [self._species(ids, row) for row in bench]

    def belief(self, out: dict, r: int, ids: torch.Tensor, preview: bool) -> list[dict]:
        """One entry per opposing row (rows 6-11): brought, lead (preview only), top items, abilities and moves."""
        brought = out["belief_brought"][r].sigmoid()
        lead = out["belief_lead"][r].softmax(-1) if preview else None
        item, ability = out["belief_item"][r].softmax(-1), out["belief_ability"][r].softmax(-1)
        moves = out["belief_moves"][r].softmax(-1)
        result = []
        for j in range(L.ROWS - L.OPP_ROWS.start):
            result.append({
                "species": self._species(ids, L.OPP_ROWS.start + j),
                "brought": float(brought[j]),
                "lead": float(lead[j]) if lead is not None else None,
                "item": _top(item[j], "items", TOP_ITEMS),
                "ability": _top(ability[j], "abilities", TOP_ABILITIES),
                "moves": _top(moves[j], "moves", TOP_MOVES),
            })  # fmt: skip
        return result

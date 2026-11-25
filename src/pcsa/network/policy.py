"""A pcsa-former checkpoint as a `champions_gym.neural` policy, with the network's outputs in each decision's extras.

    player: "champions_gym.neural:load"
    player_args: {policy: "pcsa.network.policy:Policy", checkpoint: ..., device: "cpu", policy_args: {threads: 4}}

One step of the network per request with that game's memory, returning log(pi) and log(preview_pi), so the shell's
argmax is the greedy action of decide.choose. A request answered without asking (a restart's replay) comes to
`observe`: the same forward step, so the memory after a restart is the memory the model had. One lock per policy
object around every call.

Here the belief heads run too (they change no other output) and every decision's extras carry, as plain JSON with
CG string ids (the record contract's `notes.cg.net`):
- turn and replace: tau, pi, score, p_opp, matrix [11][13], rows (our action names), cols (theirs), belief;
  closed (our actions the rule masks closed); with a belief_slots checkpoint also opp_seen [4]: which of their
  move columns were shown (the others are named by the belief head)
- turn, with a `choice` rule set (choosing.py): choice, the rule's pick and scores beside the network's pick
- preview: belief (with `lead`) and picks_top, the five likeliest (lead, second, third) picks

The rule masks of training (rules.py) are applied to the mask the network reads, with the settings the checkpoint
was trained with (runs of one move, of switches, of setup or recovery; moves that cannot work), then pcsa's
turn rules (live_rules.py: no status moves in a mirror of a listed species; Mega Evolve whenever we can); at team
preview, pcsa's lineup rule (no pick that brings two Mega stone holders) on the preview picks. What was played
is recorded once per decision: the network's own choice when it answers, corrected by `played` (called by
pcsa.cg.serve with the action that went out, so a fallback's answer or the second answer after a refusal replaces
the first).

With `search` on (searching.py; needs Champions Gym 0.5.0 or later), a turn request is also searched: the scores
returned are the log of the search's root strategy, so the shell's argmax is the search's pick, and the extras carry
`search` (simulations, worlds, time, depth, the root strategy and Q table, the model's pick and the search's). When
the search cannot run or does too little, the model's answer stands and `search.why` says so.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pcsa.common.config_cg import SearchSettings, ValueCalibration

from . import cg_adapter, choosing, decide, live_rules, model, searching, vocab
from . import checkpoint as ckpt
from . import layout as L
from . import preview_search as previewing
from . import rules as masks

VOCAB_NAMES = {"species": "species", "item": "items", "ability": "abilities", "move": "moves"}  # ours -> CG's
TOP_ITEMS, TOP_ABILITIES, TOP_MOVES, TOP_PICKS = 3, 3, 8, 5
INPUTS = ("rows", "row_ids", "field", "last", "last_ids", "mask", "opp_prior")


def _tensor(x: Any, device: str) -> torch.Tensor:
    """CG's array (numpy, or a memoryview without numpy) as a tensor, without a copy where possible."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)
    return torch.from_numpy(np.asarray(x)).to(device)


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()[:16]


def calibrated(v: torch.Tensor, cal: ValueCalibration | None) -> torch.Tensor:
    """The value mapped so that (v + 1) / 2 is a win chance: 2 * sigmoid(a * v + b) - 1; v itself without `cal`."""
    return v if cal is None else 2.0 * torch.sigmoid(cal.a * v + cal.b) - 1.0


def _top(p: torch.Tensor, kind: str, k: int) -> list[list]:
    """The k largest entries of p (id 0, unknown, left out; zeros left out) as [[CG id, p]]."""
    p = p.clone()
    p[0] = 0.0
    values, ids = p.topk(min(k, p.numel()))
    return [[vocab.name(kind, int(i)), float(v)] for v, i in zip(values, ids, strict=True) if v > 0]


@dataclass
class _Seat:
    """One slot's rule state: what it has just done, and the last turn request (to record or record again)."""

    rules: masks.Rules
    decision: int | None = None
    obs: dict | None = None
    before: tuple | None = None  # the counters before `decision` was recorded

    def keep(self) -> None:
        self.before = self.rules.save(0)

    def restore(self) -> None:
        if self.before is not None:
            self.rules.load(0, tuple(x.copy() for x in self.before))


class Policy:
    """`checkpoint`: a .pt saved by checkpoint.save. `device`: "cpu" for live play. `threads`: torch CPU threads.
    `requery_refused`: ask the network again after a refused choice instead of the shell reusing the first answer.
    `allow_contract`: load a checkpoint made under another contract version; CG's own check still reports it.
    `rules`: RuleSettings' fields without `enabled`: training's masks and pcsa's (live_rules.py); None: off.
    `search`: search every turn request (searching.py) with these SearchSettings; None: no search.
    `value_calibration`: {a, b}: the value returned is 2 * sigmoid(a * v + b) - 1 (extras keep `value_raw`).
    `choice`: the rule that makes our pick from the matrix at turn requests (choosing.py, ChoiceSettings); the
    search's pick, when it runs, still wins. `preview_search`: PreviewSearch's fields without `enabled`: the preview
    pick from playouts (preview_search.py); None: the head's. The values come from agent.yaml's cg section."""

    def __init__(self, checkpoint: str, *, device: str, threads: int, rules: Mapping[str, Any] | None = None,  # noqa: PLR0913
                 search: Mapping[str, Any] | None = None, choice: Mapping[str, Any] | None = None,
                 value_calibration: Mapping[str, float] | None = None, requery_refused: bool = False,
                 preview_search: Mapping[str, Any] | None = None, allow_contract: bool = False):  # fmt: skip
        torch.set_num_threads(int(threads))
        self.device = device
        lay = cg_adapter.layout()  # the layout from champions_gym.FEATURES, checked against CG
        extra = {"allow_contract": True} if allow_contract else {}
        self.net, ck = ckpt.load(checkpoint, lay, cg_adapter.learnset(), device, **extra)  # LayoutMismatch
        self.net.eval()
        self.memory: model.Memory | None = None
        self.lock = threading.Lock()
        self.layout = lay
        self.rule_cfg, self.turn_rules, self.lineup = live_rules.build(rules, lay)  # training's, then pcsa's
        self.seats: dict[int, _Seat] = {}
        cfg = None if search is None else SearchSettings.model_validate(search)
        self.searcher = None if cfg is None else searching.Searcher(self, cfg)
        self.choice = choosing.config(choice)
        cal = None if value_calibration is None else {"enabled": True, **value_calibration}
        self.calibration = None if cal is None else ValueCalibration.model_validate(cal)
        self.previewer = previewing.make(self, preview_search, checkpoint, **extra)
        self.game_of: Any = None  # (battle, guess, seed) -> (Game, rejected); pcsa.cg.serve sets the Session's
        self.deadline: float | None = None  # precise_now() by which the open request must be answered
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
                "field": [w["field"]], "last": [w["last"]], "last_ids": [w["last_ids"]],
                "mask": [L.OWN_ACTIONS], "opp_prior": [L.OPP_ACTIONS],
            },
            "vocab": {VOCAB_NAMES[k]: int(v) for k, v in lt["vocab"].items()},
            "model_id": os.path.abspath(checkpoint),
            "checkpoint": os.path.abspath(checkpoint),
            "model_hash": _file_hash(checkpoint),
            "requery_refused": bool(requery_refused),
            "run": ck.get("extra", {}).get("run"),
            "iteration": ck.get("extra", {}).get("iteration"),
            "memory": self.net.cfg.memory,
            "belief_slots": bool(self.net.cfg.belief_slots),
            "rule_masks": live_rules.info(self.rule_cfg, self.turn_rules, self.lineup),
            "search": None if cfg is None else cfg.model_dump(),
            "choice": None if self.choice is None else self.choice.model_dump(),
            "value_calibration": None if self.calibration is None else self.calibration.args(),
            "preview_search": previewing.info(self.previewer),
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
            for s in slots:
                self.seats.pop(s, None)
                if self.searcher is not None:
                    self.searcher.forget(s)
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

    def _turn_obs(self, arrays: Any, r: int) -> dict | None:
        """Row r as rules.Rules reads it (one game), or None when it is not a turn request."""
        if int(np.asarray(arrays.kind).reshape(-1)[r]) != L.KIND_TURN:
            return None
        one = lambda x: np.asarray(x)[r : r + 1]  # noqa: E731
        return {"mask": one(arrays.mask), "seat": np.zeros(1, np.int64), "kind": np.full(1, L.KIND_TURN),
                "rows": one(arrays.rows), "row_ids": one(arrays.row_ids), "field": one(arrays.field)}  # fmt: skip

    def _seat(self, slot: int, decision: int, obs: dict) -> _Seat:
        """The slot's rule state at this turn request: as it was before the request when it is asked again."""
        assert self.rule_cfg is not None
        seat = self.seats.get(slot)
        if seat is None:
            seat = self.seats[slot] = _Seat(masks.Rules(self.rule_cfg, self.layout, 1))
        if seat.decision == decision:
            seat.restore()
        else:
            seat.keep()
        seat.decision, seat.obs = decision, obs
        return seat

    def _close(self, arrays: Any, b: dict, slots: Sequence[int]) -> list[list[int]]:
        """The rule masks applied to b["mask"] (a copy); per row the actions they closed."""
        closed: list[list[int]] = [[] for _ in slots]
        live_rules.close_preview(self.lineup, arrays, b, closed)
        if self.rule_cfg is None:
            return closed
        decisions = np.asarray(arrays.decision).reshape(-1)
        mask = b["mask"].clone()
        for r, slot in enumerate(slots):
            obs = self._turn_obs(arrays, r)
            if obs is None:
                continue
            new = live_rules.apply(self.turn_rules, obs, self._seat(slot, int(decisions[r]), obs).rules.mask(obs))[0]
            closed[r] = [int(a) for a in np.flatnonzero((obs["mask"][0] > 0) & (new == 0))]
            mask[r] = torch.from_numpy(new).to(mask)
        b["mask"] = mask
        return closed

    def _searched(self, arrays: Any, slots: Sequence[int], kinds: np.ndarray, b: dict, out: dict,  # noqa: PLR0913
                  battles: Any,
                  scores: torch.Tensor) -> list[dict | None]:  # fmt: skip
        """With search on: each turn row searched, its scores replaced by the search's; the notes per row."""
        notes: list[dict | None] = [None] * len(slots)
        if self.searcher is None:
            return notes
        self.searcher.note_leads(b, kinds, slots)
        decisions = np.asarray(arrays.decision).reshape(-1)
        for r, slot in enumerate(slots):
            if int(kinds[r]) == L.KIND_TURN:
                battle = battles[r] if battles is not None else None
                row, notes[r] = self.searcher.run(r, slot, int(decisions[r]), b, out, battle, scores[r])
                if row is not None:
                    scores[r, : L.OWN_ACTIONS] = row
        self.deadline = None
        return notes

    def _record(self, seat: _Seat, action: int) -> None:
        seat.restore()
        if seat.obs is not None:
            seat.rules.record(seat.obs, np.array([int(action)]))

    def played(self, decision: int, action: Any) -> None:
        """The action that went out for `decision` (the shell's choice, a fallback's, or the second answer after a
        refusal): it replaces what was recorded for that decision."""
        if isinstance(action, bool) or not isinstance(action, int):
            return
        with self.lock:
            for seat in self.seats.values():
                if seat.decision == int(decision):
                    self._record(seat, action)
            if self.searcher is not None:
                self.searcher.played(int(decision), action)

    @torch.no_grad()
    def observe(self, arrays: Any, slots: Sequence[int], actions: Sequence[int | None] = ()) -> None:
        """Requests answered without asking: the memory step (the action does not enter the memory), and the
        action, when known, into the rule state."""
        with self.lock:
            b = self.batch(arrays)
            self._close(arrays, b, slots)
            self._forward(b, slots)
            kinds = np.asarray(arrays.kind).reshape(-1)
            if self.searcher is not None:
                self.searcher.note_leads(b, kinds, slots)
            for r, (slot, action) in enumerate(zip(slots, actions, strict=False)):
                seat = self.seats.get(slot)
                if seat is not None and action is not None and int(kinds[r]) == L.KIND_TURN:
                    self._record(seat, int(action))

    @torch.no_grad()
    def decide(self, arrays: Any, slots: Sequence[int], battles: Any = None) -> dict[str, Any]:
        b = self.batch(arrays)
        with self.lock:
            closed = self._close(arrays, b, slots)
            out = self._forward(b, slots)
            scores = torch.cat([decide.safe_log(out[k]) for k in ("pi", "preview_pi")], dim=1).cpu()  # [B, 71]
            kinds = np.asarray(arrays.kind).reshape(-1)
            chosen = choosing.apply(self.choice, kinds, out, scores, float(self.net.temperature()))
            searched = self._searched(arrays, slots, kinds, b, out, battles, scores)
            followed = searching.follow(self.searcher, slots, kinds, b, battles, scores)
            previews = previewing.apply(self.previewer, kinds, b, out, scores, arrays)
            for r, slot in enumerate(slots):  # the shell's argmax, until `played` says what went out
                seat = self.seats.get(slot)
                if seat is not None and int(kinds[r]) == L.KIND_TURN:
                    self._record(seat, int(scores[r, : L.OWN_ACTIONS].argmax()))
        value = calibrated(out["value"].cpu(), self.calibration)
        try:
            extras = self.all_extras(arrays, b, out, len(slots))
            if self.rule_cfg is not None:
                for e, c in zip(extras, closed, strict=True):
                    if "pi" in e or c:
                        e["closed"] = c
            for e, *parts in zip(extras, searched, chosen, previews, followed, strict=True):
                named = zip(("search", "choice", "preview_search", "follow"), parts, strict=True)
                e.update({k: v for k, v in named if v is not None})
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
        parts: dict = {"value_raw": lambda: float(out["value"].reshape(-1)[r])} if self.calibration else {}
        if kind == L.KIND_PREVIEW:
            parts |= {
                "belief": lambda: self.belief(out, r, ids, preview=True),
                "picks_top": lambda: self.picks_top(out, r),
            }
        else:
            parts |= {
                "tau": lambda: tau,
                "pi": lambda: out["pi"][r].tolist(),
                "score": lambda: out["score"][r].tolist(),
                "p_opp": lambda: out["p_opp"][r].tolist(),
                "matrix": lambda: out["matrix"][r].tolist(),
                "rows": lambda: self.own_names(ids, rows),
                "cols": lambda: self.opp_names(ids, out["opp_named"][r] if "opp_named" in out else None),
                "belief": lambda: self.belief(out, r, ids, preview=False),
            }
            if "opp_seen" in out:
                parts["opp_seen"] = lambda: [bool(x) for x in out["opp_seen"][r]]
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

    def opp_names(self, ids: torch.Tensor, named: torch.Tensor | None = None) -> list[str | None]:
        """Their 13 actions: row 6's moves by slot ("" for the first empty slot: a move not yet seen, None after),
        the same with Mega, the species of rows 7-11. `named` (a belief_slots checkpoint's opp_named): the slots'
        move ids with the empty ones named by the belief head; a slot it left at 0 is read as without `named`."""
        moves: list[str | None] = [None] * L.MOVE_SLOTS
        if self._species(ids, L.OPP_ACTIVE) is not None:
            unseen = False
            for s, m in enumerate(ids[L.OPP_ACTIVE, self.cols["moves"]] if named is None else named):
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

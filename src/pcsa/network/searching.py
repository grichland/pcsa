"""The policy's side of the decision-time search: one turn request searched with the policy's network, memory and
belief outputs (search.py is the tree, worlds.py the guesses of their team).

`Searcher.run` never raises: when the search cannot run, runs out of time before one pass or does fewer than
`min_sims` simulations, the model's own scores stand and the note's `why` says so.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from pcsa.common.clock import precise_now
from pcsa.common.paths import assets_dir

from . import layout as L
from . import model, search, worlds

RESERVE_MS = 250.0  # of the request's deadline, kept for the reply
SEED_BITS = (1 << 63) - 1
OUTPUTS = ("pi", "p_opp", "matrix", "score", "value", "opp_named")


def game_from_battle(battle: Any, guess: list, seed: int) -> tuple:
    """(Game, rejected) from the Battle and a guess at their three; NoSearch before Champions Gym 0.5.0."""
    import champions_gym
    from champions_gym import env

    make = getattr(env, "game_from_battle", None)
    if make is None:
        raise worlds.NoSearch(f"search needs Champions Gym 0.5.0 (this is {champions_gym.__version__})")
    return make(battle, guess, seed=seed)


class Searcher:
    """`policy`: the pcsa.network.policy.Policy it searches for (its net, memory, device, `game_of`, `deadline`)."""

    def __init__(self, policy: Any, cfg: search.SearchConfig) -> None:
        self.policy, self.cfg = policy, cfg
        self.leads: dict[int, str] = {}  # per slot: the species they led with
        self._tables: worlds.Tables | None = None
        self._scratch: model.Memory | None = None

    def note_leads(self, b: dict, kinds: np.ndarray, slots: Sequence[int]) -> None:
        """Their active at a slot's first turn request is the one they led with."""
        for r, slot in enumerate(slots):
            if slot not in self.leads and int(kinds[r]) == L.KIND_TURN:
                name = self.policy._species(b["row_ids"][r].long(), L.OPP_ACTIVE)
                if name:
                    self.leads[slot] = name

    def tables(self) -> worlds.Tables:
        if self._tables is None:
            path = assets_dir() / "sets" / "top_builds.json"
            builds = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
            learnset = torch.as_tensor(self.policy.net.learnset).detach().cpu().numpy().astype(bool)
            self._tables = worlds.Tables(learnset, worlds.common_sets(builds))
        return self._tables

    def leaves(self, out: dict, ids: torch.Tensor) -> list[search.Leaf]:
        p = self.policy
        return search.leaves_of(out, ids.long(), p.cols["species"], p.cols["moves"], bool(p.net.cfg.belief_slots))

    def forward(self, nodes: list, memories: list) -> tuple[list, list]:
        """The network on our view of each node's Game, each with the memory of its path."""
        from champions_gym import env

        p, n = self.policy, len(nodes)
        b = p.batch(env.game_tensors([node.game for node in nodes], [0] * n))  # type: ignore[attr-defined]
        if self._scratch is None or self._scratch.hist.shape[0] < n:
            self._scratch = model.Memory(p.net, max(n, self.cfg.batch), p.device)
        mem = self._scratch
        for i, (hist, pos, count) in enumerate(memories):
            mem.hist[i], mem.pos[i], mem.count[i] = hist, pos, count
        out = p.net.forward(b, memory=mem, slots=torch.arange(n, device=p.device), beliefs=False)
        return self.leaves(out, b["row_ids"]), [search.memory_of(mem, i) for i in range(n)]

    def budget(self) -> float:
        """Milliseconds for this request: the setting, or what the request's deadline leaves."""
        budget, deadline = self.cfg.ms, self.policy.deadline
        if deadline is not None:
            budget = min(budget, (deadline - precise_now()) * 1000.0 - RESERVE_MS)
        if self.cfg.sims <= 0 and budget < search.MIN_BUDGET_MS:
            raise worlds.NoSearch(f"a budget of {budget:.0f} ms is too small for one pass")
        return budget

    def belief(self, r: int, ids: torch.Tensor, out: dict) -> worlds.Belief:
        rows = range(L.ROWS - L.OPP_ROWS.start)
        return worlds.Belief(
            [self.policy._species(ids, L.OPP_ROWS.start + j) for j in rows],
            out["belief_brought"][r].sigmoid().cpu().numpy(),
            out["belief_moves"][r].softmax(-1).cpu().numpy(),
            out["belief_item"][r].softmax(-1).cpu().numpy(),
            out["belief_ability"][r].softmax(-1).cpu().numpy(),
        )

    def roots(self, r: int, slot: int, b: dict, out: dict, battle: Any, rng: np.random.Generator, note: dict) -> list:
        """The worlds, each with its root expanded from the request's own forward pass."""
        p, tables = self.policy, self.tables()
        belief = self.belief(r, b["row_ids"][r].long().cpu(), out)
        make = p.game_of or game_from_battle
        built, dropped = worlds.build_worlds(battle, belief, tables, make, self.cfg.worlds, rng, self.leads.get(slot))
        note["worlds"], note["dropped"], note["dropped_why"] = len(built), len(dropped), dropped[:3]
        if not built:
            raise worlds.NoSearch("no guess of their team gave a Game" + (f": {dropped[0]}" if dropped else ""))
        leaf = self.leaves({k: out[k][r : r + 1] for k in OUTPUTS if k in out}, b["row_ids"][r : r + 1])[0]
        memory, allowed = search.memory_of(p.memory, slot), (b["mask"][r] > 0).cpu().numpy()
        for w in built:
            w.root = search.Node(w.game)
            search.expand(w.root, leaf, w, tables, self.cfg, allowed)
            w.root.mem = memory
        return built

    def answer(self, found: search.Result, allowed: np.ndarray, note: dict) -> torch.Tensor:
        """The scores for our 11 actions: the log of the root strategy over the allowed actions."""
        pick = int(found.strategy.argmax())
        note.update(sims=found.sims, depth=found.depth, nodes=found.nodes, batches=found.batches,
                    strategy=[float(x) for x in found.strategy], q=found.q, p_them=found.p_them,
                    p_other=found.p_other, visits=found.visits, stopped=found.stopped)  # fmt: skip
        if found.sims < self.cfg.min_sims:
            raise worlds.NoSearch(f"{found.sims} simulations, fewer than min_sims {self.cfg.min_sims}")
        if not found.strategy[pick] > 0 or not allowed[pick]:
            raise worlds.NoSearch("the search gave no action")
        note.update(used=True, pick=pick, changed=pick != note["model_pick"])
        row = torch.full((L.OWN_ACTIONS,), float("-inf"))
        for a in range(L.OWN_ACTIONS):
            if allowed[a] and found.strategy[a] > 0:
                row[a] = float(np.log(found.strategy[a]))
        return row

    def run(self, r: int, slot: int, decision: int, b: dict, out: dict, battle: Any,
            scores: torch.Tensor) -> tuple[torch.Tensor | None, dict]:  # fmt: skip
        """Row r's turn request searched: (the scores to return for our 11 actions, or None when the model's
        answer stands; the `search` note)."""
        start = precise_now()
        model_pick = int(scores[: L.OWN_ACTIONS].argmax())
        note: dict[str, Any] = {"used": False, "why": None, "model_pick": model_pick, "pick": model_pick,
                                "changed": False, "sims": 0}  # fmt: skip
        try:
            if battle is None:
                raise worlds.NoSearch("no Battle came with the request")
            if int((b["mask"][r] > 0).sum()) <= 1:
                raise worlds.NoSearch("one action is open")
            budget = self.budget()
            note["budget_ms"] = round(budget, 1)
            rng = np.random.default_rng([self.cfg.seed & SEED_BITS, decision & SEED_BITS, slot])
            built = self.roots(r, slot, b, out, battle, rng, note)
            tree = search.Search(self.cfg, self.tables(), self.forward, rng)
            tree.run(built, start + budget / 1000.0)
            allowed = (b["mask"][r] > 0).cpu().numpy()
            return self.answer(tree.result(built, note["dropped_why"]), allowed, note), note
        except Exception as error:
            known = isinstance(error, worlds.NoSearch)
            note["why"] = str(error) if known else f"{type(error).__name__}: {error}"
            return None, note
        finally:
            note["ms"] = round((precise_now() - start) * 1000.0, 1)

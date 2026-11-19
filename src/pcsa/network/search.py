"""Decision-time search for the neural player: simultaneous-move tree search with the network's priors and regret
matching at every node, over sampled guesses of the opponent's hidden team ("worlds").

Needs Champions Gym 0.5.0 (an engine Game from the live Battle plus a guess at their three; worlds.py builds
them, searching.py runs this for the policy). One tree per world:

- a node is a state where the seats choose (a seat with nothing to choose has the one action None). It holds, for
  our legal actions a and theirs j: visits, cumulative regrets, strategy sums and a payoff table Q[a][j] with
  counts, all from our point of view. It is expanded by one network forward on our view of the Game with the
  memory of its path: value v, our prior (pi), their prior (p_opp mapped from the 13 columns to their legal
  actions) and Q's start (the matrix cell of each pair, weight n0).
- selection, each side on its own: s[x] ~ max(R[x], 0) + c * P[x] * sqrt(N + 1) / (1 + N[x]), sampled.
- chance: at a pair's first visit up to `outcomes` outcomes are drawn (equal states merged); each visit follows
  one by its share. A simulation ends at a new node (its v), a finished game (+1, -1, 0) or a node `depth` turns
  from the root (its v).
- backup of the value u at each node on the path: Q[a][j] moves toward u (running mean with the n0 start);
  R_us[a'] += Q[a'][j] - u, R_them[j'] += u - Q[a][j'], both kept at 0 or above; the strategies used are added
  to the strategy sums.

The answer is our normalised strategy sum at the root, averaged over the worlds by their simulations; the caller
plays its largest entry among the actions allowed. Leaves are evaluated in batches of `batch`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from pcsa.common.clock import precise_now

from . import layout as L
from . import model, vocab
from .worlds import NoSearch, Tables, World

FLOOR = 0.02  # prior of a legal action of theirs that no network column stands for
KIND_ASKED = (L.KIND_TURN, L.KIND_REPLACE)
MIN_BUDGET_MS = 40.0
END_MARGIN_S = 0.015  # of the budget, for putting the answer together


@dataclass(frozen=True)
class SearchConfig:
    ms: float = 1500.0  # time for one decision's search
    worlds: int = 6
    depth: int = 3  # turns below the root
    outcomes: int = 3
    c: float = 1.5
    n0: float = 1.0
    batch: int = 16
    min_sims: int = 64  # fewer in total: the model's answer stands
    sims: int = 0  # stop after this many (0: by time only)
    seed: int = 0
    top_ours: int = 0  # at each node only our N likeliest actions by the network's policy (0: all)
    top_theirs: int = 0  # at each node only their N likeliest actions by the network's prediction (0: all)


class Node:
    __slots__ = ("N", "Nt", "Nu", "P_t", "P_u", "Q", "R_t", "R_u", "S_t", "S_u", "cols", "game", "kids", "mem",
                 "n", "pending", "terminal", "them", "turn", "us", "v")  # fmt: skip

    def __init__(self, game: Any) -> None:
        self.game = game
        self.turn = int(game.turn)
        self.terminal = bool(game.ended)
        self.pending = not self.terminal
        self.v = 0.0
        if self.terminal:
            winner = game.winner
            self.v = 0.0 if winner is None else (1.0 if int(winner) == 0 else -1.0)
        self.kids: dict = {}
        self.mem: tuple | None = None


@dataclass
class Leaf:
    """One network row for a node: what expansion reads."""

    pi: np.ndarray  # [11]
    p_opp: np.ndarray  # [13]
    matrix: np.ndarray  # [11, 13]
    score: np.ndarray  # [11]
    value: float
    named: np.ndarray  # [4] move ids of their active's columns (0: none, or the one unseen column)
    bench: list[str | None]  # species of the switch columns (rows 7-11)
    belief_slots: bool


def their_columns(leaf: Leaf, world: World, active: int, them: list, tables: Tables) -> tuple[list[int], list[float]]:
    """For each legal action of theirs: its network column (-1: none) and the share of that column's weight it
    takes. Without belief slots the first empty column stands for every move not yet seen, shared equally."""
    moves = world.moves(active)
    ids = [tables.move_id.get(m, -1) for m in moves]
    named = [int(x) for x in leaf.named]
    empty = next((c for c, x in enumerate(named) if x == 0), -1) if not leaf.belief_slots else -1
    unseen = sum(1 for k in range(len(moves)) if ids[k] not in named) or 1
    cols, share = [], []
    for j in them:
        col, part = -1, 1.0
        if j is None:
            pass
        elif j < 2 * L.MOVE_SLOTS:
            k, mega = j % L.MOVE_SLOTS, L.MOVE_SLOTS * (j // L.MOVE_SLOTS)
            if k < len(ids) and ids[k] > 0 and ids[k] in named:
                col = named.index(ids[k]) + mega
            elif empty >= 0 and k < len(ids):
                col, part = empty + mega, 1.0 / unseen
        else:
            s = j - 2 * L.MOVE_SLOTS
            name = world.species[s] if s < len(world.species) else None
            exact = [c for c, b in enumerate(leaf.bench) if b and name and b == name]
            close = [c for c, b in enumerate(leaf.bench) if b and name and (b.startswith(name) or name.startswith(b))]
            if exact or close:
                col = 2 * L.MOVE_SLOTS + (exact or close)[0]
        cols.append(col)
        share.append(part)
    return cols, share


def _normal(x: np.ndarray) -> np.ndarray:
    total = float(x.sum())
    return x / total if total > 0 else np.full(len(x), 1.0 / len(x))


def _likeliest(actions: list, weight: Sequence[float], top: int) -> list:
    """The `top` actions with the largest weight, in their own order (all of them when top is 0)."""
    if top <= 0 or len(actions) <= top:
        return actions
    keep = sorted(range(len(actions)), key=lambda k: (-float(np.nan_to_num(weight[k])), k))[:top]
    return [actions[k] for k in sorted(keep)]


def expand(
    node: Node, leaf: Leaf, world: World, tables: Tables, cfg: SearchConfig, allowed: np.ndarray | None = None
) -> None:
    """The node's actions, priors, value and Q's start from the network row. `allowed` (the root): our actions
    the request and the rule masks leave open. cfg.top_ours and cfg.top_theirs keep only each side's likeliest
    actions."""
    n0 = cfg.n0
    game = node.game
    kinds, masks = game.kinds(), np.asarray(game.masks()).reshape(2, L.OWN_ACTIONS)
    us = [int(a) for a in np.flatnonzero(masks[0])] if kinds[0] in KIND_ASKED else []
    if allowed is not None:
        us = [a for a in us if allowed[a]]
        if not us:
            raise NoSearch("the Game offers none of the request's legal actions")
    us = _likeliest(us, [leaf.pi[a] for a in us], cfg.top_ours)
    with_pi = [a for a in us if leaf.pi[a] > 0]
    node.us = us or [None]
    them = [int(a) for a in np.flatnonzero(masks[1])] if kinds[1] in KIND_ASKED else []
    active = int(np.asarray(game.readout().active).reshape(-1)[1])
    if cfg.top_theirs > 0 and len(them) > cfg.top_theirs:
        cols, share = their_columns(leaf, world, active, them, tables)
        weight = [leaf.p_opp[c] * s if c >= 0 else FLOOR for c, s in zip(cols, share, strict=True)]
        them = _likeliest(them, weight, cfg.top_theirs)
    node.them = them or [None]
    v = float(np.clip(leaf.value, -1.0, 1.0)) if np.isfinite(leaf.value) else 0.0
    node.v = v
    node.P_u = _normal(np.array([leaf.pi[a] if a in with_pi else 0.0 for a in us])) if us else np.ones(1)
    cols, share = their_columns(leaf, world, active, node.them, tables)
    node.cols = cols
    prior = np.array([leaf.p_opp[c] * s if c >= 0 else FLOOR for c, s in zip(cols, share, strict=True)], np.float64)
    node.P_t = _normal(np.nan_to_num(prior)) if them else np.ones(1)
    q = np.full((len(node.us), len(node.them)), v)
    for x, a in enumerate(node.us):
        for y, c in enumerate(cols):
            if a is None:
                continue
            cell = leaf.matrix[a][c] if c >= 0 else (leaf.score[a] if not them else v)
            if np.isfinite(cell):
                q[x, y] = cell
    node.Q = np.clip(q, -1.0, 1.0)
    node.n = np.full(q.shape, float(n0))
    node.R_u, node.R_t = np.zeros(len(node.us)), np.zeros(len(node.them))
    node.S_u, node.S_t = np.zeros(len(node.us)), np.zeros(len(node.them))
    node.Nu, node.Nt, node.N = np.zeros(len(node.us)), np.zeros(len(node.them)), 0
    node.pending = False


def strategy(regret: np.ndarray, prior: np.ndarray, visits: np.ndarray, total: int, c: float) -> np.ndarray:
    return _normal(regret + c * prior * np.sqrt(total + 1.0) / (1.0 + visits))


@dataclass
class Result:
    strategy: np.ndarray  # [11] our averaged root strategy
    sims: int
    worlds: int
    dropped: list[str]
    depth: int
    nodes: int
    batches: int
    q: list[list[float | None]]  # [11][13] the averaged root Q by network column
    p_them: list[float]  # [13] their averaged root strategy by network column
    p_other: float  # their strategy on actions no column stands for
    visits: list[float]  # [11]
    ms: float = 0.0


class Search:
    """One decision's search. `forward(tensors, memories) -> (leaves, memories after)` runs the network on a batch
    of our views, each with the memory of its path."""

    def __init__(self, cfg: SearchConfig, tables: Tables, forward: Callable, rng: np.random.Generator) -> None:
        self.cfg, self.tables, self.forward, self.rng = cfg, tables, forward, rng
        self.depth = self.nodes = self.batches = 0

    def descend(self, world: World) -> tuple[list, Node]:
        """One simulation down to a node to evaluate: (the path of (node, a, j), that node)."""
        cfg, node, path = self.cfg, world.root, []
        while not (node.terminal or node.pending or (path and node.turn - world.root.turn >= cfg.depth)):
            su = strategy(node.R_u, node.P_u, node.Nu, node.N, cfg.c)
            st = strategy(node.R_t, node.P_t, node.Nt, node.N, cfg.c)
            a = int(self.rng.choice(len(su), p=su)) if len(su) > 1 else 0
            j = int(self.rng.choice(len(st), p=st)) if len(st) > 1 else 0
            node.S_u += su
            node.S_t += st
            node.N += 1
            node.Nu[a] += 1
            node.Nt[j] += 1
            kids = node.kids.get((a, j))
            if kids is None:
                drawn = node.game.sample(node.us[a], node.them[j], cfg.outcomes, seed=int(self.rng.integers(1 << 62)))
                kids = node.kids[(a, j)] = (
                    [float(o.p) for o in drawn.outcomes],
                    [o.game for o in drawn.outcomes],
                    {},
                )
            weights, games, made = kids
            k = int(self.rng.choice(len(weights), p=_normal(np.array(weights)))) if len(weights) > 1 else 0
            path.append((node, a, j))
            child = made.get(k)
            if child is None:
                child = made[k] = Node(games[k])
                self.nodes += 1
            node = child
        self.depth = max(self.depth, len(path))
        return path, node

    @staticmethod
    def backup(path: list, u: float) -> None:
        for node, a, j in path:
            node.n[a, j] += 1
            node.Q[a, j] += (u - node.Q[a, j]) / node.n[a, j]
            node.R_u = np.maximum(node.R_u + node.Q[:, j] - u, 0.0)
            node.R_t = np.maximum(node.R_t + u - node.Q[a, :], 0.0)

    def evaluate(self, waiting: dict) -> None:
        """One forward for every new node of the batch; a node where we are not asked keeps its parent's memory."""
        nodes = list(waiting.values())
        leaves, memories = self.forward([n for n, _, _ in nodes], [parent.mem for _, parent, _ in nodes])
        for (node, parent, world), leaf, mem in zip(nodes, leaves, memories, strict=True):
            expand(node, leaf, world, self.tables, self.cfg)
            asked = node.game.kinds()[0] in KIND_ASKED
            deep = node.turn - world.root.turn >= self.cfg.depth
            node.mem = None if deep else (mem if asked else parent.mem)
        self.batches += 1

    def run(self, worlds: list[World], end: float) -> None:
        """Simulations round-robin over the worlds until `end` (precise_now) or cfg.sims."""
        cfg, sims, took = self.cfg, 0, 0.0
        while (cfg.sims <= 0 or sims < cfg.sims) and precise_now() + 2.0 * took < end - END_MARGIN_S:
            start, waiting, done = precise_now(), {}, []
            size = cfg.batch if cfg.sims <= 0 else min(cfg.batch, cfg.sims - sims)
            for k in range(size):
                world = worlds[(sims + k) % len(worlds)]
                path, node = self.descend(world)
                world.sims += 1
                if node.pending and path:
                    waiting.setdefault(id(node), (node, path[-1][0], world))
                done.append((path, node))
            if waiting:
                self.evaluate(waiting)
            for path, node in done:
                self.backup(path, node.v)
            sims += size
            took = precise_now() - start

    def result(self, worlds: list[World], dropped: list[str]) -> Result:
        total = sum(w.sims for w in worlds)
        avg, visits = np.zeros(L.OWN_ACTIONS), np.zeros(L.OWN_ACTIONS)
        q_sum, q_w = np.zeros((L.OWN_ACTIONS, L.OPP_ACTIONS)), np.zeros((L.OWN_ACTIONS, L.OPP_ACTIONS))
        p_them, other = np.zeros(L.OPP_ACTIONS), 0.0
        for w in worlds:
            root, share = w.root, (w.sims / total if total else 1.0 / len(worlds))
            ours = _normal(root.S_u) if root.S_u.sum() > 0 else root.P_u
            theirs = _normal(root.S_t) if root.S_t.sum() > 0 else root.P_t
            for x, a in enumerate(root.us):
                if a is None:
                    continue
                avg[a] += share * ours[x]
                visits[a] += root.Nu[x]
                for y, c in enumerate(root.cols):
                    if c >= 0:
                        q_sum[a, c] += root.n[x, y] * root.Q[x, y]
                        q_w[a, c] += root.n[x, y]
            for y, c in enumerate(root.cols):
                if c >= 0:
                    p_them[c] += share * theirs[y]
                else:
                    other += share * theirs[y]
        q = [[float(q_sum[a, c] / q_w[a, c]) if q_w[a, c] > 0 else None for c in range(L.OPP_ACTIONS)]
             for a in range(L.OWN_ACTIONS)]  # fmt: skip
        return Result(avg, total, len(worlds), dropped, self.depth, self.nodes, self.batches, q,
                      [float(x) for x in p_them], float(other), [float(x) for x in visits])  # fmt: skip


def settings(cfg: SearchConfig) -> dict:
    return asdict(cfg)


def leaves_of(
    out: dict, ids: torch.Tensor, species_col: int, move_cols: Sequence[int], belief_slots: bool
) -> list[Leaf]:
    """The forward's outputs as one Leaf per row."""
    get = lambda k: out[k].detach().cpu().numpy()  # noqa: E731
    pi, p_opp, matrix, score, value = get("pi"), get("p_opp"), get("matrix"), get("score"), get("value")
    ids_np = ids.detach().cpu().numpy()
    named = get("opp_named") if "opp_named" in out else ids_np[:, L.OPP_ACTIVE][:, list(move_cols)]
    species = vocab.tables()["species"]
    result = []
    for r in range(len(pi)):
        bench = [species[i] if 0 <= i < len(species) else None for i in ids_np[r, L.OPP_BENCH, species_col]]
        result.append(Leaf(pi[r], p_opp[r], matrix[r], score[r], float(value[r]), named[r], bench, belief_slots))
    return result


def memory_of(mem: model.Memory, slot: int) -> tuple:
    return mem.hist[slot].clone(), mem.pos[slot].clone(), mem.count[slot].clone()

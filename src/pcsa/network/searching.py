"""The policy's side of the decision-time search: one turn request searched with the policy's network, memory and
belief outputs (search.py is the tree, worlds.py the guesses of their team).

`Searcher.run` never raises: when the search cannot run, runs out of time before one pass or does fewer than
`min_sims` simulations, the model's own scores stand and the note's `why` says so.

With `follow_pivot`, a searched pick that the trees saw switch us out in the same turn with our active standing (a
pivot move) keeps the switch-ins they chose (pivot_follow): one for after the pivot, one for after a faint first.
The replace request of that turn plays the one the battle's events name (our `withdraw` with `pivot`, or our
`faint`; the pivot's when the rows are not read yet) when it is open, with the `follow` note. The kept switch-ins
are dropped when another action went out (`played`) or at the slot's next request.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pcsa.common.clock import precise_now
from pcsa.common.paths import assets_dir, repo_root

from . import builds, evidence, live_rules, model, search, worlds
from . import layout as L
from . import rules as masks

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


def pivot_follow(found: list[worlds.World], pick: int) -> dict | None:
    """After our `pick`, the switch-ins the trees chose in the same turn, summed over the worlds by visits:
    `target` where our active was still standing (a pivot move), `after_faint` where it had fainted first; None
    when no tree saw the pivot."""
    totals = {"pivot": np.zeros(L.OWN_ACTIONS), "faint": np.zeros(L.OWN_ACTIONS)}
    visits = {"pivot": 0, "faint": 0}
    for w in found:
        root = w.root
        if root is None or root.pending or pick not in root.us:
            continue
        x, me = root.us.index(pick), int(np.asarray(root.game.readout().active).reshape(-1)[0])
        for (a, _), (_, _, made) in root.kids.items():
            for child in made.values() if a == x else ():
                if child.pending or child.terminal or child.turn != root.turn or child.us == [None]:
                    continue
                if int(child.game.kinds()[0]) != L.KIND_REPLACE:
                    continue
                case = "faint" if bool(np.asarray(child.game.readout().fainted)[0][me]) else "pivot"
                visits[case] += max(child.N, 1)
                weight = child.S_u if child.N > 0 else child.P_u
                for y, b in enumerate(child.us):
                    totals[case][b] += weight[y]
    pivot, faint = totals["pivot"], totals["faint"]
    if not pivot.any():
        return None
    target = int(pivot.argmax())
    return {"target": target, "share": float(pivot[target] / pivot.sum()), "after_faint":
            int(faint.argmax()) if faint.any() else None, "pivot_visits": visits["pivot"],
            "faint_visits": visits["faint"]}  # fmt: skip


def follow(searcher: Searcher | None, slots: Sequence[int], kinds: np.ndarray, b: dict, battles: Any,
           scores: torch.Tensor) -> list[dict | None]:  # fmt: skip
    """Searcher.follow, or no notes without a searcher."""
    return [None] * len(slots) if searcher is None else searcher.follow(slots, kinds, b, battles, scores)


@dataclass
class Follow:
    """A searched turn's pivot move and the switch-in the search chose for after it."""

    decision: int
    turn: int
    pick: int
    target: int  # after the pivot
    after_faint: int | None = None  # if our active fainted before it could pivot


class Searcher:
    """`policy`: the pcsa.network.policy.Policy it searches for (its net, memory, device, `game_of`, `deadline`)."""

    def __init__(self, policy: Any, cfg: search.SearchSettings) -> None:
        self.policy, self.cfg = policy, cfg
        self.leads: dict[int, str] = {}  # per slot: the species they led with
        self.follows: dict[int, Follow] = {}  # per slot: the switch-in after a searched pivot move
        self._guesser: builds.Guesser | None = None
        self._rates: worlds.Rates | None = None
        self._rules: masks.Rules | None = None  # the tree's rule counters, one game, loaded per node
        self.tracker: evidence.Tracker | None = None  # this battle's evidence (pcsa.cg.serve feeds it)
        self.last_weights: dict[str, tuple] = {}  # species -> (candidates, weights, facts) at the last decision
        self._tables: worlds.Tables | None = None
        self._scratch: model.Memory | None = None

    def guesser(self) -> builds.Guesser:
        if self._guesser is None:
            self._guesser = builds.Guesser(self.tables())
        return self._guesser

    def start_battle(self, own_team: Any) -> None:
        """A new battle: its evidence starts empty, with our six sets for our speeds."""
        self.tracker = evidence.Tracker(self.guesser().dex, own=list(own_team or []))
        self.last_weights = {}

    def events(self, events: Any) -> None:
        if self.tracker is None:
            self.start_battle(None)
        self.tracker.feed(events or [])  # type: ignore[union-attr]

    def end_battle(self) -> list[dict]:
        """Per opposing Pokémon weighed this battle: the rank (1 first) and weight of the likeliest build that fits
        what was shown, by the last weights (rank None: only "other" fits)."""
        out = []
        for species, (cands, w, facts) in self.last_weights.items():
            order = np.argsort(-w)
            legal = self.guesser().dex.abilities.get(species)
            fit = [r for r, i in enumerate(order) if cands[i] is not None and builds.fits(cands[i], facts, legal)]
            hit = fit[0] if fit else None
            out.append({"species": species, "candidates": len(cands) - 1, "rank": None if hit is None else hit + 1,
                        "weight": None if hit is None else round(float(w[order[hit]]), 4),
                        "other": round(float(w[-1]), 4)})  # fmt: skip
        self.tracker, self.last_weights = None, {}
        return out

    def draw(self, battle: Any, order: list[int], belief: worlds.Belief, rng: np.random.Generator) -> tuple:
        """Their three (preview slots, lead first) as whole builds (worlds_mode builds)."""
        g, team, preview = self.guesser(), list(battle.opponent_team), [str(s) for s in battle.opponent_preview]
        had = self.tracker.had_item if self.tracker else {}
        facts = {i: builds.facts_of(team[i], had.get(builds.to_id(team[i].species))) for i in order}
        shown = [f.item for f in facts.values() if f.item]
        sets, notes = g.guess_team([(i, team[i], preview[i]) for i in order], belief, self.tracker, rng, shown)
        for i in order:
            cands, w, _ = g.weigh(team[i], preview[i], belief, self.tracker)
            self.last_weights[preview[i]] = (cands, w, facts[i])
        return sets, notes

    def forget(self, slot: int) -> None:
        self.leads.pop(slot, None)
        self.follows.pop(slot, None)

    def played(self, decision: int, action: Any) -> None:
        """What went out for `decision`: a pivot move's switch-in is kept only if the pivot move itself went out."""
        for slot, f in list(self.follows.items()):
            if f.decision == decision and f.pick != action:
                del self.follows[slot]

    def follow(self, slots: Sequence[int], kinds: np.ndarray, b: dict, battles: Any,
               scores: torch.Tensor) -> list[dict | None]:  # fmt: skip
        """Each replace row after a searched pivot move of the same turn: scores for only the kept switch-in when it
        is open; the `follow` notes per row."""
        notes: list[dict | None] = [None] * len(slots)
        for r, slot in enumerate(slots):
            if int(kinds[r]) != L.KIND_REPLACE or (f := self.follows.pop(slot, None)) is None:
                continue
            model_pick = int(scores[r, : L.OWN_ACTIONS].argmax())
            out = self.tracker.own_out() if self.tracker is not None else None  # "pivot", "faint" or None
            target = f.after_faint if out == "faint" else f.target
            note: dict[str, Any] = {"used": False, "why": None, "target": target, "out": out,
                                    "model_pick": model_pick}  # fmt: skip
            battle = battles[r] if battles is not None else None
            turn = None if battle is None else int(battle.turn)
            if turn != f.turn:
                note["why"] = f"turn {turn}, the pivot move was on turn {f.turn}"
            elif target is None:
                note["why"] = "our Pokémon fainted before the pivot and the search kept no switch-in for that"
            elif not bool(b["mask"][r, target] > 0):
                note["why"] = "the search's switch-in is not open"
            else:
                row = torch.full((L.OWN_ACTIONS,), float("-inf"))
                row[target] = 0.0
                scores[r, : L.OWN_ACTIONS] = row
                note.update(used=True, changed=target != model_pick)
            notes[r] = note
        return notes

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
        """The network rows as Leaves; with `calibrate`, the value, the matrix cells and the row scores mapped by
        the policy's value_calibration onto the scale of a finished game's +1 / -1."""
        p = self.policy
        leaves = search.leaves_of(out, ids.long(), p.cols["species"], p.cols["moves"], bool(p.net.cfg.belief_slots))
        cal = p.calibration if self.cfg.calibrate else None
        if cal is not None:
            f = lambda x: 2.0 / (1.0 + np.exp(-(cal.a * np.asarray(x, np.float64) + cal.b))) - 1.0  # noqa: E731
            for leaf in leaves:
                leaf.value, leaf.matrix, leaf.score = float(f(leaf.value)), f(leaf.matrix), f(leaf.score)
        return leaves

    def forward(self, nodes: list, memories: list) -> tuple[list, list]:
        """The network on our view of each node's Game, each with the memory of its path."""
        from champions_gym import env

        p, n = self.policy, len(nodes)
        arrays = env.game_tensors([node.game for node in nodes], [0] * n)  # type: ignore[attr-defined]
        b = p.batch(arrays)
        self.mask_nodes(nodes, arrays, b)
        if self._scratch is None or self._scratch.hist.shape[0] < n:
            self._scratch = model.Memory(p.net, max(n, self.cfg.batch), p.device)
        mem = self._scratch
        for i, (hist, pos, count) in enumerate(memories):
            mem.hist[i], mem.pos[i], mem.count[i] = hist, pos, count
        out = p.net.forward(b, memory=mem, slots=torch.arange(n, device=p.device), beliefs=False)
        return self.leaves(out, b["row_ids"]), [search.memory_of(mem, i) for i in range(n)]

    def mask_nodes(self, nodes: list, arrays: Any, b: dict) -> None:
        """Our rule masks at every new node, as at a live request: each node's rule counters are its parent's with
        our action there recorded; at our turn nodes the masks close actions in b["mask"] (the network's input)
        and in node.allowed (the tree's)."""
        p = self.policy
        if p.rule_cfg is None:
            return
        if self._rules is None:
            self._rules = masks.Rules(p.rule_cfg, p.layout, 1)
        rules, mask = self._rules, b["mask"].clone()
        for i, node in enumerate(nodes):
            parent, action = node.via if node.via is not None else (None, None)
            node.rules = None if parent is None else parent.rules
            if parent is not None and parent.obs is not None and parent.rules is not None and action is not None:
                rules.load(0, parent.rules)
                rules.record(parent.obs, np.array([int(action)]))
                node.rules = rules.save(0)
            node.obs = p._turn_obs(arrays, i)
            if node.obs is None or node.rules is None:
                continue
            rules.load(0, node.rules)
            m = live_rules.apply(p.turn_rules, node.obs, rules.mask(node.obs))[0]
            node.allowed = m > 0
            mask[i] = torch.from_numpy(m).to(mask)
        b["mask"] = mask

    def root_rules(self, root: search.Node, slot: int) -> None:
        """The root's rule counters and observation: the slot's, before this request's action is recorded."""
        seat = self.policy.seats.get(slot)
        if seat is not None and seat.obs is not None:
            root.rules, root.obs = seat.rules.save(0), seat.obs

    def budget(self) -> float:
        """Milliseconds for this request: the setting, or what the request's deadline leaves."""
        budget, deadline = self.cfg.ms, self.policy.deadline
        if deadline is not None:
            budget = min(budget, (deadline - precise_now()) * 1000.0 - RESERVE_MS)
        if self.cfg.sims <= 0 and budget < search.MIN_BUDGET_MS:
            raise worlds.NoSearch(f"a budget of {budget:.0f} ms is too small for one pass")
        return budget

    def belief(self, r: int, ids: torch.Tensor, out: dict) -> worlds.Belief:
        """The belief heads of row r; with `preview_rates`, `brought` is the live bring rate of each species."""
        rows = range(L.ROWS - L.OPP_ROWS.start)
        species = [self.policy._species(ids, L.OPP_ROWS.start + j) for j in rows]
        brought = out["belief_brought"][r].sigmoid().cpu().numpy()
        if self.cfg.preview_rates:
            if self._rates is None:
                self._rates = worlds.Rates.load(str(repo_root() / self.cfg.preview_rates))
            brought = np.array([self._rates.brought(s) for s in species])
        return worlds.Belief(
            species,
            brought,
            out["belief_moves"][r].softmax(-1).cpu().numpy(),
            out["belief_item"][r].softmax(-1).cpu().numpy(),
            out["belief_ability"][r].softmax(-1).cpu().numpy(),
        )

    def roots(self, r: int, slot: int, b: dict, out: dict, battle: Any, rng: np.random.Generator, note: dict) -> list:
        """The worlds, each with its root expanded from the request's own forward pass."""
        p, tables = self.policy, self.tables()
        belief = self.belief(r, b["row_ids"][r].long().cpu(), out)
        make = p.game_of or game_from_battle
        draw = None
        if self.cfg.worlds_mode == "builds":
            draw = lambda battle, order, rng: self.draw(battle, order, belief, rng)  # noqa: E731
        built, dropped = worlds.build_worlds(battle, belief, tables, make, self.cfg.worlds, rng, self.leads.get(slot),
                                             draw)  # fmt: skip
        note["worlds"], note["dropped"], note["dropped_why"] = len(built), len(dropped), dropped[:3]
        note["repaired"] = sum(w.repaired for w in built)
        if draw is not None:
            seen = {n["species"]: n for w in built for n in w.notes}
            note["builds"] = {"pokemon": [{k: v for k, v in n.items() if k != "drawn"} for n in seen.values()],
                              "drawn": [[n["drawn"] for n in w.notes] for w in built],
                              "orders": len(self.tracker.orders) if self.tracker else 0}  # fmt: skip
        if not built:
            raise worlds.NoSearch("no guess of their team gave a Game" + (f": {dropped[0]}" if dropped else ""))
        leaf = self.leaves({k: out[k][r : r + 1] for k in OUTPUTS if k in out}, b["row_ids"][r : r + 1])[0]
        memory, allowed = search.memory_of(p.memory, slot), (b["mask"][r] > 0).cpu().numpy()
        for w in built:
            w.root = search.Node(w.game)
            search.expand(w.root, leaf, w, tables, self.cfg, allowed)
            w.root.mem = memory
            self.root_rules(w.root, slot)
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
        self.follows.pop(slot, None)
        model_pick = int(scores[: L.OWN_ACTIONS].argmax())
        note: dict[str, Any] = {"used": False, "why": None, "model_pick": model_pick, "pick": model_pick,
                                "changed": False, "sims": 0}  # fmt: skip
        try:
            if battle is None:
                raise worlds.NoSearch("no Battle came with the request")
            if int(battle.turn) < self.cfg.from_turn:
                raise worlds.NoSearch(f"turn {int(battle.turn)}: the search starts at turn {self.cfg.from_turn}")
            if int((b["mask"][r] > 0).sum()) <= 1:
                raise worlds.NoSearch("one action is open")
            budget = self.budget()
            note["budget_ms"] = round(budget, 1)
            rng = np.random.default_rng([self.cfg.seed & SEED_BITS, decision & SEED_BITS, slot])
            built = self.roots(r, slot, b, out, battle, rng, note)
            tree = search.Search(self.cfg, self.tables(), self.forward, rng)
            tree.run(built, start + budget / 1000.0)
            allowed = (b["mask"][r] > 0).cpu().numpy()
            row = self.answer(tree.result(built, note["dropped_why"]), allowed, note)
            if self.cfg.follow_pivot and (follow := pivot_follow(built, note["pick"])):
                note["follow"] = follow
                self.follows[slot] = Follow(decision, int(battle.turn), int(note["pick"]), int(follow["target"]),
                                            follow["after_faint"])  # fmt: skip
            return row, note
        except Exception as error:
            known = isinstance(error, worlds.NoSearch)
            note["why"] = str(error) if known else f"{type(error).__name__}: {error}"
            return None, note
        finally:
            note["ms"] = round((precise_now() - start) * 1000.0, 1)

"""Team-preview search: our pick from playouts of every (our pick, their pick) cell, in place of the preview head's.

1. Guesses: `guesses` versions of their team, each Pokémon's set a whole known build (worlds_mode builds) or drawn
   part by part by worlds.guess_set from the belief heads (legacy), each version checked by Champions Gym.
2. A BatchEnv of 3,600 slots on [ours, guess 1, ...]: slot i starts on cell i (ours = i // 60, theirs = i % 60).
   Both preview picks are forced; after that one memory slot per seat. Our seat plays as live play does without
   search: the rule masks, then the choice rule over the matrix (the greedy policy when no rule is set); theirs
   plays the greedy policy without our masks.
3. Stage A: at our first turn, the win chance (calibrated value + 1) / 2. Cell value (wins + prior_games * A) /
   (games + prior_games).
4. The race (preview_race.py) cuts our picks to `race` sizes; the head's pick always stays in.
5. The best row is played when it beats the head's pick by `keep_head_margin`; otherwise, on an error, or with no
   finished game, the head's pick.

Ported from pcsa-former's tools/preview_search_test.py (search with halving, belief_teams); see NOTICE.md.
"""

from __future__ import annotations

import json
import re
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from pcsa.common.clock import precise_now
from pcsa.common.config_cg import PreviewSearch
from pcsa.common.paths import assets_dir

from . import builds, cg_adapter, choosing, model, worlds
from . import checkpoint as ckpt
from . import layout as L
from . import rules as masks
from .preview_race import P, Race, choose, table

RESERVE_S = 1.5  # of the request's deadline, kept for the record and the reply
SLOTS = P * P  # one game slot per cell
GUESS_TRIES = 6  # draws of one guess before it is given up
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
OBS = ("mask", "seat", "kind", "rows", "row_ids", "field")


def _cg():
    import champions_gym as cg

    return cg


def cg_set(m: dict) -> dict:
    """A set in CG's teams-file form (ids); gender M until CG asks for another."""
    pts = m.get("points") or {}
    return {"species": m["species"], "item": m.get("item") or "", "ability": m.get("ability") or "",
            "moves": list(m["moves"])[: L.MOVE_SLOTS], "nature": m.get("nature") or "serious",
            "evs": {s: int(pts.get(s, 0)) for s in STATS}, "level": 50, "gender": m.get("gender") or "M"}  # fmt: skip


def checked(teams: list[list[dict]]) -> list[str]:
    """CG's problems with `teams`, after switching any set CG says needs another gender to F."""
    for _ in range(2):
        problems = _cg().check_teams(json.dumps(teams))[1]
        fixed = False
        for t, reason in problems:
            found = re.search(r"set (\d+)", reason)
            if "gender" in reason and found and teams[t][int(found[1])].get("gender") != "F":
                teams[t][int(found[1])]["gender"], fixed = "F", True
        if not fixed:
            break
    return [f"team {t}: {reason}" for t, reason in problems]


class Previewer:
    """The search for one Policy (its layout, rule masks, deadline); its own copy of the net on `cfg.device`."""

    def __init__(self, policy: Any, cfg: PreviewSearch, checkpoint: str, **load: Any) -> None:
        self.policy, self.cfg = policy, cfg
        self.net: Any = None
        self.error: str | None = None
        self.own_team: list[dict] | None = None  # set by pcsa.cg.serve from each battle_start
        self._tables: worlds.Tables | None = None
        self._guesser: builds.Guesser | None = None
        try:
            if cfg.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available to this process")
            self.net, _ = ckpt.load(checkpoint, policy.layout, cg_adapter.learnset(), cfg.device, **load)
            self.net.eval()
        except Exception as error:  # every preview then plays the head's pick and says why
            self.error = f"start: {type(error).__name__}: {error}"

    def tables(self) -> worlds.Tables:
        if self._tables is None:
            path = assets_dir() / "sets" / "top_builds.json"
            builds = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
            learnset = torch.as_tensor(self.policy.net.learnset).detach().cpu().numpy().astype(bool)
            self._tables = worlds.Tables(learnset, worlds.common_sets(builds))
        return self._tables

    def run(self, r: int, b: dict, out: dict, preview_scores: torch.Tensor, decision: int) -> dict:
        """Row r's preview request: the record, with `chosen` the pick to play (the head's unless the search says)."""
        t0 = precise_now()
        head = int(preview_scores.argmax())
        note: dict[str, Any] = {"head": head, "chosen": head}
        try:
            if self.error:
                raise RuntimeError(self.error)
            if not self.own_team:
                raise RuntimeError("no own team came with battle_start")
            stop = t0 + self.cfg.budget_s
            if self.policy.deadline is not None:
                stop = min(stop, self.policy.deadline - RESERVE_S)
            seed = (int(self.cfg.seed) * 1_000_003 + int(decision)) % 2**32
            own = [cg_set(m) for m in self.own_team]
            teams, refused = self.guesses(r, b, out, own, np.random.default_rng(seed))
            note |= {"seed": seed, "guesses": teams, "refused": refused}
            if not teams:
                raise RuntimeError("no guess of their team passed CG's check")
            found = self.play([own, *teams], head, stop, seed, np.flatnonzero(b["preview_mask"][r].cpu().numpy()))
            note |= found
            if found["games"] == 0:
                raise RuntimeError("no game finished before the deadline")
            c = {k: np.asarray(v, float) for k, v in found["cells"].items()}
            T = table(c["a_sum"], c["a_cnt"], c["wins"], c["games"], self.cfg.prior_games)
            pick, why = choose(T, np.asarray(found["rows"]), head, self.cfg.choose, self.cfg.keep_head_margin)
            note |= {**why, "chosen": pick}
        except Exception as error:
            where = traceback.extract_tb(error.__traceback__)[-1]
            note["error"] = f"{type(error).__name__}: {error} ({Path(where.filename).name}:{where.lineno})"
        note["seconds"] = round(precise_now() - t0, 2)
        return note

    def guesses(self, r: int, b: dict, out: dict, own: list[dict], rng: np.random.Generator) -> tuple[list, list]:
        """Up to `cfg.guesses` versions of their team CG accepts, and the reasons of the draws it refused."""
        ids = b["row_ids"][r].long()
        species = [self.policy._species(ids, L.OPP_ROWS.start + j) for j in range(L.ROWS - L.OPP_ROWS.start)]
        soft = lambda k: out[k][r].softmax(-1).cpu().numpy()  # noqa: E731
        brought = out["belief_brought"][r].sigmoid().cpu().numpy()
        belief = worlds.Belief(species, brought, soft("belief_moves"), soft("belief_item"), soft("belief_ability"))
        tables, teams, refused = self.tables(), [], []
        shown = [SimpleNamespace(form=None, species=sid, moves=(), item=None, ability=None) for sid in species]
        for _ in range(self.cfg.guesses):
            for _ in range(GUESS_TRIES):
                if self.cfg.worlds_mode == "builds":
                    if self._guesser is None:
                        self._guesser = builds.Guesser(tables)
                    drawn_sets, _ = self._guesser.guess_team(
                        [(slot, shown[slot], str(sid)) for slot, sid in enumerate(species)], belief, None, rng)
                else:
                    drawn_sets = [worlds.guess_set(shown[slot], slot, str(sid), belief, tables, rng)
                                  for slot, sid in enumerate(species)]  # fmt: skip
                team = [cg_set({"species": sid, **d}) for sid, d in zip(species, drawn_sets, strict=True)]
                problems = checked([own, team])
                if not problems:
                    teams.append(team)
                    break
                refused.append(problems)
        return teams, refused

    def play(self, teams: list[list[dict]], head: int, stop: float, seed: int, rows: Any = None) -> dict:
        """The playouts of our open picks `rows` (default all) until `stop`: per-cell sums, the race's cuts and final
        rows, games and steps."""
        n, dev = SLOTS, self.cfg.device
        env = open_env(teams, n, seed, self.cfg.threads)
        mem, state = model.Memory(self.net, 2 * n, dev), self.start(n, head, rows)
        t_start, steps = precise_now(), 0
        while precise_now() < stop:
            self._step(env, mem, state)
            steps += 1
            self._results(env, mem, state, precise_now() - t_start, stop - t_start)
        race = state["race"]
        cells = {k: [round(float(x), 4) for x in state[k]] for k in ("a_sum", "a_cnt", "wins", "games")}
        return {"cells": cells, "cuts": race.cuts, "rows": [int(x) for x in race.rows],
                "games": int(state["games"].sum()), "steps": steps, "play_seconds": round(precise_now() - t_start, 2)}

    def start(self, n: int, head: int, rows: Any = None) -> dict:
        """The playouts' state: our seat's rule masks, the race, per-cell sums, whether a slot's turn 1 was scored."""
        cfg = self.policy.rule_cfg
        rules = None if cfg is None else masks.Rules(cfg, self.policy.layout, n)
        sums = {k: np.zeros(P * P) for k in ("a_sum", "a_cnt", "wins", "games")}  # per cell
        return {"rules": rules, "race": Race(n, head, self.cfg.race, rows), "scored": np.zeros(n, bool), **sums}

    @torch.no_grad()
    def _step(self, env: Any, mem: model.Memory, state: dict) -> None:
        """One decision in every slot: forced preview picks, greedy turns with our seat's masks."""
        dev, t = self.cfg.device, env.observe_tensors()
        obs = {k: np.asarray(getattr(t, k)) for k in OBS}
        kind, seat = obs["kind"].reshape(-1), obs["seat"].reshape(-1).astype(np.int64)
        put = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)  # noqa: E731
        b = {k: put(getattr(t, k)) for k in model.INPUTS if k != "preview_mask"}
        b["preview_mask"] = put(np.asarray(t.action_mask)[:, L.PREVIEW_OFFSET :])
        rules = state["rules"]
        if rules is not None:
            m = live_masks(self.policy, rules, obs)
            m[seat == 1] = obs["mask"][seat == 1]  # their seat plays without our masks
            b["mask"] = torch.from_numpy(m).to(b["mask"])
        slots = torch.arange(len(kind), device=dev) * 2 + torch.from_numpy(seat).to(dev)
        out = self.net.act(b, torch.from_numpy(kind).to(dev), greedy=True, memory=mem, slots=slots)
        action = out["action"].cpu().numpy()
        ours = np.flatnonzero((kind == L.KIND_TURN) & (seat == 0))
        if self.policy.choice is not None and len(ours):  # our seat picks as live play does: the choice rule
            rows = torch.from_numpy(ours).to(dev)
            ruled = choosing.scores(out["matrix"][rows], out["p_opp"][rows], out["pi"][rows],
                                    out["legal"][rows].bool(), self.policy.choice)  # fmt: skip
            action[ours] = ruled.argmax(-1).cpu().numpy()
        cell = state["race"].cell
        pre = kind == L.KIND_PREVIEW
        action[pre & (seat == 0)] = L.PREVIEW_OFFSET + cell[pre & (seat == 0)] // P
        action[pre & (seat == 1)] = L.PREVIEW_OFFSET + cell[pre & (seat == 1)] % P
        first = np.flatnonzero((kind == L.KIND_TURN) & (seat == 0) & ~state["scored"])
        if len(first):  # stage A: the win chance of our first turn, (calibrated value + 1) / 2
            v, cal = out["value"][first].float().reshape(-1), self.policy.calibration
            v = v if cal is None else 2.0 * torch.sigmoid(cal.a * v + cal.b) - 1.0
            np.add.at(state["a_sum"], cell[first], ((v + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy())
            np.add.at(state["a_cnt"], cell[first], 1)
            state["scored"][first] = True
        if rules is not None:
            rules.record(obs, action)
        env.act(action)

    def _results(self, env: Any, mem: model.Memory, state: dict, elapsed: float, budget: float) -> None:
        """Games that ended: counted in their cells, their slots reset and moved on by the race."""
        res = env.results()
        done = np.flatnonzero(np.asarray(res.done))
        if not len(done):
            return
        race, winner = state["race"], np.asarray(res.winner)[done]
        np.add.at(state["wins"], race.cell[done], np.where(winner < 0, 0.5, (winner == 0).astype(float)))
        np.add.at(state["games"], race.cell[done], 1)
        state["scored"][done] = False
        ended = torch.as_tensor(done, device=self.cfg.device)
        mem.reset(torch.cat([ended * 2, ended * 2 + 1]))
        if state["rules"] is not None:
            state["rules"].reset(done)
        while race.cut_due(elapsed, budget):
            T = table(state["a_sum"], state["a_cnt"], state["wins"], state["games"], self.cfg.prior_games)
            race.cut(T, elapsed)
        race.reassign(done)


def open_env(teams: list[list[dict]], n: int, seed: int, threads: int) -> Any:
    """A BatchEnv of n games: seat 0 always teams[0], seat 1 one of the others, drawn evenly."""
    g = len(teams) - 1
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "teams.json"
        path.write_text(json.dumps(teams), encoding="utf-8")
        return _cg().BatchEnv(str(path), n, seed, threads=threads, strict_teams=True,
                              team_weights=[[1] + [0] * g, [0] + [1] * g], agent_seat=[0] * n)  # fmt: skip


def live_masks(policy: Any, rules: masks.Rules, obs: dict) -> np.ndarray:
    """Training's rule masks, then pcsa's turn rules, as the policy applies them live."""
    m = rules.mask(obs)
    turn = getattr(policy, "turn_rules", None)
    return m if turn is None else turn.mask(obs, m)


def apply(previewer: Previewer | None, kinds: np.ndarray, b: dict, out: dict, scores: torch.Tensor,
          arrays: Any) -> list[dict | None]:  # fmt: skip
    """The search on every preview row; its pick made the shell's argmax. Per row the record, or None."""
    notes: list[dict | None] = [None] * len(kinds)
    if previewer is None:
        return notes
    decisions = np.asarray(arrays.decision).reshape(-1)
    for r in np.flatnonzero(kinds == L.KIND_PREVIEW):
        note = notes[r] = previewer.run(int(r), b, out, scores[r, L.PREVIEW_OFFSET :], int(decisions[r]))
        if note["chosen"] != note["head"]:
            scores[r, L.PREVIEW_OFFSET + note["chosen"]] = scores[r, L.PREVIEW_OFFSET :].max() + 1.0
    return notes


def make(policy: Any, settings: Any, checkpoint: str, **load: Any) -> Previewer | None:
    """The policy's Previewer from its `preview_search` argument (None: off)."""
    if settings is None:
        return None
    return Previewer(policy, PreviewSearch.model_validate({"enabled": True, **settings}), checkpoint, **load)


def info(previewer: Previewer | None) -> dict | None:
    """The settings as the policy reports them, with a start-up error if there was one."""
    if previewer is None:
        return None
    return {**previewer.cfg.model_dump(exclude={"enabled"}), "start_error": previewer.error}

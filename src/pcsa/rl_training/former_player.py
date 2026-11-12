"""The policy player over the poke-singles-former network (former.py): propose, then commit.

The network remembers a game through one stored token per earlier decision and layer (former.Net.step_cached).
On the phone a decision is not always final when the network answers, so the two steps are separate:

  propose(view)  the network's answer for this view. Changes nothing: it steps a copy of the game's memory.
  commit(p)      makes a proposal the game's latest decision (its memory entry is kept) and returns its action.

A proposal that is never committed (a late answer the caller dropped) leaves no trace. The same turn asked again
with the same observation (a re-read screen, a rejected tap) replaces the last decision instead of adding one,
so the memory holds one entry per decision, as in training. `restore` rebuilds the memory of a battle that is
resumed in a new process from the views of its earlier decisions.

The notes of a decision carry the whole pair matrix (`matrix`: our 11 actions against their kept actions) for
the run records and the dashboard. Research's training player (rl2/player.py: episodes, rewards, exploration)
is not copied.

Feature modes (`features`, research rl2/player.py): "base" is the observation the network was trained on before
the belief tracker; "belief" and "belief+mega" (pcsa.rl_training.belief_features) replace some columns with what
one belief per battle (pcsa.belief.fast.BattleBelief, kept with the game's memory) says. The mode must be the one
the checkpoint was trained with (checkpoint_features); pcsa.players.neural enforces it.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pcsa.common.clock import Deadline
from pcsa.players.base import DecisionView, LegalAction, TeamPreviewView, random_team
from pcsa.rl_training import entropy, features, preview
from pcsa.rl_training import former_features as fx
from pcsa.rl_training.features_vocab import tables
from pcsa.rl_training.former import Config, Net, stack
from pcsa.rl_training.spec import VOCAB

PREVIEW_TAU, PREVIEW_TOP_P = 0.1, 0.9
VERSION = "v2"  # the "version" field of a poke-singles-former checkpoint
FEATURES = ("base", "belief", "belief+mega")  # feature modes (belief_features.MODES)
MOVES, MEGA_AT, SWITCH_AT = 4, 4, 8  # action rows: 0-3 move, 4-7 move with Mega, 8-10 switch to bench
THEIR_ACTIVE, THEIR_BENCH = 6, 7  # observation rows of their active and of their first bench member
MASKED = -1e8  # a logit below this is a masked one


@dataclass
class _Game:
    turn: int = -1
    tag: str | None = None
    cache: dict = field(default_factory=dict)  # the memory after the latest committed decision
    before: dict = field(default_factory=dict)  # the memory before it (a same-turn retry starts here again)
    last: Any = None  # the latest committed decision's observation
    belief: Any = None  # pcsa.belief.fast.BattleBelief of this battle (feature modes other than "base")


@dataclass(frozen=True)
class Proposal:
    """One decision before it is committed."""

    obs: fx.Obs
    index: int
    probs: torch.Tensor  # [1, ACTIONS], capped
    value: float
    turn: int
    tag: str | None
    replaces: bool  # a same-turn retry of the latest committed decision
    chosen: LegalAction
    base: dict  # the memory this decision was computed from
    cache: dict  # the memory with this decision in it
    out: dict  # the network's outputs (matrix, kept_idx, kept_p, opp_mega, ...)
    belief: Any = None  # the battle's belief (feature modes other than "base"), carried into the committed game


def as_numpy_rng(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, random.Random):
        return np.random.default_rng(rng.getrandbits(64))
    return np.random.default_rng(rng)


def load_checkpoint(path: Path | str) -> dict:
    """A torch checkpoint (our own training output: it holds numpy rng state, so not weights_only)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state if isinstance(state, dict) and "net" in state else {"net": state}


def checkpoint_config(state: dict) -> Config:
    """The checkpoint's model Config; ValueError when it is not a poke-singles-former checkpoint."""
    model = state.get("model")
    if state.get("version") != VERSION or not isinstance(model, dict):
        raise ValueError(f"not a poke-singles-former checkpoint (version {state.get('version')!r}, need {VERSION!r})")
    return Config(**model)


def checkpoint_features(state: dict) -> str:
    """The feature mode a checkpoint was trained with: its run config's `features` (research rl2/train.py Run2),
    "base" when the config has none (every run before the belief tracker)."""
    mode = (state.get("config") or {}).get("features") or "base"
    if mode not in FEATURES:
        raise ValueError(f"checkpoint trained with unknown feature mode {mode!r} (known: {', '.join(FEATURES)})")
    return mode


def _copy(cache: dict) -> dict:
    """A memory that can be stepped without touching `cache` (the stored tokens themselves are never changed)."""
    return {"t": cache.get("t", 0), "mem": [list(layer) for layer in cache.get("mem", [])]} if cache else {}


def _same(a: fx.Obs, b: fx.Obs) -> bool:
    return all(
        np.array_equal(getattr(a, k), getattr(b, k)) for k in ("rows", "ids", "field", "last", "last_ids", "mask")
    )


@lru_cache(maxsize=4)
def _names(kind: str) -> dict[int, str]:
    return {i: name for name, i in tables()[kind].items()}


def matrix_notes(obs: fx.Obs, out: dict, probs: torch.Tensor, index: int) -> dict:
    """The pair matrix of one decision as plain lists: our 11 actions (rows) against their kept actions (columns),
    with each column's probability and our final distribution. Names are vocabulary ids ("" for an empty slot)."""
    move, species = _names("move"), _names("species")
    ours = [move.get(int(i), "") for i in obs.ids[0, 3 : 3 + MOVES]]
    bench = [species.get(int(obs.ids[1 + k, 0]), "") for k in range(len(obs.mask) - SWITCH_AT)]
    shown = {int(i) for i in obs.ids[THEIR_ACTIVE, 3 : 3 + MOVES] if int(i)}
    kept = [int(i) for i in out["kept_idx"].tolist()]
    v = VOCAB["move"]
    cols = [move.get(i, "") if i < v else species.get(int(obs.ids[THEIR_BENCH + i - v, 0]), "") for i in kept]
    mega = float(out["opp_mega"])
    return {
        "rows": [*ours, *(f"{m} + mega" if m else "" for m in ours), *bench],
        "legal": [bool(x) for x in obs.mask],
        "cols": cols,
        "col_kind": ["move" if i < v else "switch" for i in kept],
        "shown": [i in shown for i in kept],
        "col_legal": [bool(out["opp_logits"][i] > MASKED) for i in kept],  # False: fainted or absent
        "p_cols": [round(float(x), 3) for x in out["kept_p"].tolist()],
        "cells": [[round(float(x), 3) for x in row] for row in out["matrix"].tolist()],
        "probs": [round(float(x), 3) for x in probs[0].tolist()],
        "chosen": index,
        "mega_p": round(float(torch.sigmoid(out["opp_mega"])), 3) if mega > MASKED else None,
    }


def belief_notes(belief) -> dict:
    """A short summary of the battle's belief for the decision record (feature modes other than "base")."""
    t = belief.tracker
    return {
        "members": len(t.members),
        "updates": int(belief.updates),
        "rebuilds": int(belief.rebuilds),
        "evidence": {k: int(v) for k, v in sorted(t.stats.items())},
    }


class FormerPlayer:
    """pcsa Player protocol over a former.Net; never raises on a decision (falls back to the first legal one)."""

    def __init__(
        self,
        net: Net,
        cfg: Config,
        *,
        device: str = "cpu",
        temperature: float = 1.0,
        rng=None,
        tag: str = "policy",
        greedy_preview: bool = False,
        opponent_temperature: float = 1.0,
        opponent_top: int = 0,
        features: str = "base",
        belief_options: dict | None = None,
    ) -> None:
        if features not in FEATURES:
            raise ValueError(f"unknown feature mode {features!r} (one of {', '.join(FEATURES)})")
        self.net, self.cfg, self.device = net.to(device).eval(), cfg, device
        self.features = features
        # new_battle_belief's options (crit_reported, empty_unknown); research's defaults when not given
        self.belief_options = dict(belief_options or {})
        # stage 3 (see _blend): logits + matrix_blend x Q from `matrix_values`; 0 / None = the network as trained
        self.matrix_blend = 0.0
        self.matrix_values: Any = None
        self.matrix_errors = 0
        # the temperature of their predicted actions where they weight the matrix (1 = as trained)
        self.net.opp_temperature = float(opponent_temperature)
        self.net.opp_top = int(opponent_top)  # weigh only their N likeliest kept actions (0 = all, as trained)
        self.temperature, self.rng = temperature, as_numpy_rng(rng)
        self.greedy_preview = greedy_preview
        self.name = f"rl:{tag}"
        self.game = _Game()
        self.fallbacks = 0
        self._notes: dict = {}

    @classmethod
    def from_checkpoint(cls, path: Path | str, **kw) -> FormerPlayer:
        state = load_checkpoint(path)
        cfg = checkpoint_config(state)
        player = cls(Net(cfg), cfg, **kw)
        player.load_state_dict(state["net"])
        return player

    def load_state_dict(self, sd: dict) -> None:
        self.net.load_state_dict(sd)
        self.net.eval()

    def params(self) -> dict:
        c = self.cfg
        out = {"temperature": self.temperature, "d": c.d, "layers": c.layers, "top_k": c.top_k, "memory": c.memory}
        return out | {"features": self.features}

    # Observations -------------------------------------------------------------------------------------
    def _belief(self, game: _Game):
        """The battle's belief (created on first use), or None in feature mode "base"."""
        if self.features == "base":
            return None
        if game.belief is None:
            from pcsa.rl_training.belief_features import new_battle_belief

            game.belief = new_battle_belief(**self.belief_options)
        return game.belief

    def _observe(self, battle, legal, game: _Game) -> fx.Obs:
        if self.features == "base":
            return fx.observe(battle, legal)
        return fx.observe(battle, legal, belief=self._belief(game), feature_mode=self.features)

    def _preview_observations(self, battle, picks) -> list[fx.Obs]:
        if self.features == "base":
            return fx.preview_observations(battle, picks)
        return fx.preview_observations(battle, picks, belief=self._belief(self.game), feature_mode=self.features)

    def notes(self) -> dict:
        notes, self._notes = self._notes, {}
        return notes

    # Team preview -------------------------------------------------------------------------------------
    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        """The pick sampled from the critic's top-p nucleus over all 60 (lead, second, third) options."""
        self.game = _Game()
        if view.battle is None:
            return self._team_fallback(view, "no battle at preview")
        try:
            picks = preview.all_picks(len(view.battle.team) or 6, view.battle.max_team_size or 3)
            values = self.preview_values(view.battle, picks)
            if self.greedy_preview:
                index = int(np.argmax(values))
            else:
                index = preview.choose_preview(values, self.rng, tau=PREVIEW_TAU, top_p=PREVIEW_TOP_P)
            nats = preview.preview_entropy(values, tau=PREVIEW_TAU, top_p=PREVIEW_TOP_P)
            h = nats / np.log(len(picks)) if len(picks) > 1 else 0.0
            self._notes = {"rule": "rl", "v": round(float(values[index]), 4), "h": round(float(h), 4)}
            return [int(i) for i in picks[index]]
        except Exception as error:  # a broken preview must not end the game
            return self._team_fallback(view, f"{type(error).__name__}: {error}")

    @torch.no_grad()
    def preview_values(self, battle, picks) -> np.ndarray:
        """The value of each pick's turn-0 observation, each pick its own one-decision game."""
        obs = self._preview_observations(battle, picks)
        out = self.net(stack([[o] for o in obs], self.device))
        return out["value"][:, 0].float().cpu().numpy()

    def _team_fallback(self, view: TeamPreviewView, why: str) -> list[int]:
        self.fallbacks += 1
        self._notes = {"rule": "rl:fallback", "why": why}
        return random_team(view.rng, len(view.roster) or 6)

    # Decisions ----------------------------------------------------------------------------------------
    def choose_action(self, view: DecisionView, deadline: Deadline) -> LegalAction:
        if not view.legal:
            raise ValueError("no legal action")
        if view.battle is None:
            return self._fallback(view, "no battle (screen-level decision)")
        try:
            return self.commit(self.propose(view))
        except Exception as error:  # any failure plays the first legal action
            return self._fallback(view, f"{type(error).__name__}: {error}")

    def reset(self) -> None:
        """Forget the game (a new battle)."""
        self.game = _Game()

    def _game(self, turn: int) -> _Game:
        """The game so far, or an empty one when the turn count started over. The battle's tag is not compared:
        the live loop builds a new Battle, with a new tag, for every decision of one battle."""
        return self.game if turn >= self.game.turn else _Game()

    def _step(self, view: DecisionView) -> tuple[fx.Obs, int, bool, dict, dict, dict, Any]:
        """(observation, turn, replaces, memory before, memory after, outputs, belief) of this view. Changes nothing
        but the battle's belief, which reads the view's protocol (the same protocol read again changes nothing)."""
        battle = view.battle
        turn = int(battle.turn or 0)  # type: ignore[union-attr]
        g = self._game(turn)
        obs = self._observe(battle, view.legal, g)
        replaces = g.last is not None and g.turn == turn and _same(g.last, obs)
        base = g.before if replaces else g.cache
        cache = _copy(base)
        return obs, turn, replaces, base, cache, self.net.step_cached(obs, cache), g.belief

    def propose(self, view: DecisionView) -> Proposal:
        """The decision for this view; changes nothing (the game's memory is stepped on a copy)."""
        obs, turn, replaces, base, cache, out, belief = self._step(view)
        out = self._blend(view, obs, out, belief)
        mask = torch.as_tensor(np.asarray(obs.mask, bool), device=out["logits"].device)
        index, probs = self._sample(out["logits"][None], mask[None])
        chosen = features.legal_action(index, view.legal, view.battle)
        tag = getattr(view.battle, "battle_tag", None)
        return Proposal(obs, index, probs, float(out["value"]), turn, tag, replaces, chosen, base, cache, out, belief)

    def _blend(self, view: DecisionView, obs: fx.Obs, out: dict, belief: Any) -> dict:
        """The place for stage 3 (research/pcsa-belief stage 3: rl2/matrix.py and rl2/player.py's `matrix_blend`,
        default off; not ported yet). There: Q_i = sum_j p_j V(i, j), a one-turn value per (our action, their kept
        action) cell from the belief's damage calculator, and the logits become logits + matrix_blend x Q (masked
        actions stay masked), never on a forced switch, the network's logits on any error. Here `matrix_values(battle,
        legal, out, belief) -> (q [ACTIONS], info)` stands for rl2.matrix.action_values; in feature mode "base" the
        matrix would need a belief of its own (research keeps one for the matrix only). Off: `out` untouched."""
        if not self.matrix_blend or self.matrix_values is None or obs.forced:
            return out
        try:
            q, info = self.matrix_values(view.battle, view.legal, out, belief)
            with torch.no_grad():
                q = torch.as_tensor(np.asarray(q, np.float32), dtype=out["logits"].dtype, device=out["logits"].device)
                logits = out["logits"] + float(self.matrix_blend) * q
        except Exception:  # the matrix never stops a decision (research: the same)
            self.matrix_errors += 1
            return out
        return {**out, "logits": logits, "network_logits": out["logits"], "matrix_q": q, "matrix_info": info}

    def commit(self, p: Proposal) -> LegalAction:
        """Make a proposal the game's latest decision (later decisions remember it) and return its action."""
        self.game = _Game(turn=p.turn, tag=p.tag, cache=p.cache, before=p.base, last=p.obs, belief=p.belief)
        mask = torch.as_tensor(np.asarray(p.obs.mask, bool), device=p.probs.device)
        h = float(entropy.normalised_entropy(p.probs, mask[None])[0])
        # v: the critic's value of this position (expected return: +1 win / -1 loss, +-0.1 per faint)
        self._notes = {"rule": "rl", "p": round(float(p.probs.max()), 4), "h": round(h, 4), "v": round(p.value, 4)}
        self._notes["matrix"] = matrix_notes(p.obs, p.out, p.probs, p.index)
        # tau: `probs` is softmax over legal rows of (sum_j p_cols[j] * cells[i][j]) / tau. It is the network's
        # learned temperature times the player's sampling temperature (1 when it plays greedy).
        self._notes["matrix"]["opp_temperature"] = round(float(self.net.opp_temperature), 4)  # of p_cols
        # opp_top: only their N likeliest columns weighed our actions (0 = all); w_cols are the weights used
        self._notes["matrix"]["opp_top"] = int(self.net.opp_top)
        self._notes["matrix"]["w_cols"] = [round(float(x), 3) for x in p.out["kept_w"].tolist()]
        scale = self.temperature if self.temperature > 0 else 1.0
        self._notes["matrix"]["tau"] = round(float(self.net.log_tau.detach().exp()) * scale, 4)
        self._notes["decisions"] = int(p.cache.get("t", 0))
        if p.belief is not None:
            self._notes["belief"] = belief_notes(p.belief)
        if p.replaces:
            self._notes["retry"] = True
        return p.chosen

    def restore(self, views: Iterable[DecisionView]) -> int:
        """Rebuild the game's memory from the views of a battle's earlier decisions, oldest first (a battle resumed
        in a new process). Stops at the first view that cannot be replayed; returns how many decisions are in the
        memory. Chooses nothing and writes no notes."""
        self.game = _Game()
        for view in views:
            try:
                obs, turn, _, base, cache, _, belief = self._step(view)
            except Exception:  # keep what was rebuilt
                break
            self.game = _Game(
                turn=turn,
                tag=getattr(view.battle, "battle_tag", None),
                cache=cache,
                before=base,
                last=obs,
                belief=belief,
            )
        return int(self.game.cache.get("t", 0))

    def _sample(self, logits: torch.Tensor, mask: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Capped probabilities at the player's temperature (T = 1 for logging at T = 0) and the action taken."""
        probs = entropy.capped_probs(logits, mask, temperature=self.temperature if self.temperature > 0 else 1.0)
        p = probs[0].double().cpu().numpy()
        if self.temperature <= 0:
            return int(np.argmax(p)), probs
        return int(self.rng.choice(len(p), p=p / p.sum())), probs

    def _fallback(self, view: DecisionView, why: str) -> LegalAction:
        self.fallbacks += 1
        self._notes = {"rule": "rl:fallback", "why": why}
        return view.legal[0]

"""The policy player over the poke-singles-former network (former.py): propose, then commit.
propose(view) answers for a view and changes nothing (it steps a copy of the game's memory); commit(p) makes a
proposal
the latest decision. The same turn asked again replaces the last decision (one memory entry per decision, as in
training); `restore` rebuilds it for a resumed battle. Feature mode must match the checkpoint; rule masks leave the
legality mask before the softmax. Notes carry the pair matrix.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pcsa.common.clock import Deadline
from pcsa.players.base import DecisionView, LegalAction, TeamPreviewView, random_team
from pcsa.rl_training import entropy, features, preview, rule_masks, value_calibration
from pcsa.rl_training import former_features as fx
from pcsa.rl_training.former import Config, Net, stack
from pcsa.rl_training.former_notes import belief_notes, matrix_notes

PREVIEW_TAU, PREVIEW_TOP_P = 0.1, 0.9
VERSION = "v2"  # the "version" field of a poke-singles-former checkpoint
FEATURES = ("base", "belief", "belief+mega")
__all__ = ["FormerPlayer", "Proposal", "belief_notes", "matrix_notes"]


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
    cache: dict
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
        self.rule_mask: Any = None  # (view, out) -> {message: (rule, reason)} of forbidden actions (rule_masks.py)
        # the temperature of their predicted actions where they weight the matrix (1 = as trained)
        self.net.opp_temperature = float(opponent_temperature)
        self.net.opp_top = int(opponent_top)  # weigh only their N likeliest kept actions (0 = all, as trained)
        # their active's move options narrowed to its listed builds' moves (pcsa.rl_training.opponent_moves)
        self.opponent_moves_from_builds = False
        self.temperature, self.rng = temperature, as_numpy_rng(rng)
        self.greedy_preview = greedy_preview
        # preview rules (pcsa.players.preview_rules.keeper): (battle, picks) -> (keep per pick, notes); None = off
        self.preview_keep: Any = None
        # robust preview (pcsa.rl_training.preview_leads)
        self.preview_robust_worst = 0
        self.value_map: Any = None  # pcsa.rl_training.value_calibration.mapper; None = the raw value
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
        out |= {"opponent_moves_from_builds": bool(self.opponent_moves_from_builds)}
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
            if self.preview_robust_worst > 0:
                return self._robust_team(view, picks)
            values = self.preview_values(view.battle, picks)
            keep, rules = self.preview_keep(view.battle, picks) if self.preview_keep else ([True] * len(picks), {})
            rows = [i for i, k in enumerate(keep) if k] or list(range(len(picks)))
            kept = values[rows]
            if self.greedy_preview:
                index = rows[int(np.argmax(kept))]
            else:
                index = rows[preview.choose_preview(kept, self.rng, tau=PREVIEW_TAU, top_p=PREVIEW_TOP_P)]
            nats = preview.preview_entropy(kept, tau=PREVIEW_TAU, top_p=PREVIEW_TOP_P)
            h = nats / np.log(len(rows)) if len(rows) > 1 else 0.0
            self._notes = {"rule": "rl", "v": round(float(values[index]), 4), "h": round(float(h), 4)}
            if rules:  # what the rules removed, and the critic's best pick without them
                best = int(np.argmax(values))
                self._notes["preview_rules"] = {
                    **rules,
                    "best_any": list(picks[best]),
                    "v_any": round(float(values[best]), 4),
                }
            return [int(i) for i in picks[index]]
        except Exception as error:  # a broken preview must not end the game
            return self._team_fallback(view, f"{type(error).__name__}: {error}")

    def _robust_team(self, view: TeamPreviewView, picks) -> list[int]:
        from pcsa.rl_training.preview_leads import robust_team

        return robust_team(self, view.battle, picks)

    def lead_values(self, battle, picks) -> tuple[np.ndarray, list[str]]:
        """[picks, their six] turn-1 values (pcsa.rl_training.preview_leads.lead_values), and their six."""
        from pcsa.rl_training.preview_leads import lead_values

        return lead_values(self, battle, picks)

    @torch.no_grad()
    def preview_values(self, battle, picks) -> np.ndarray:
        """The value of each pick's turn-0 observation, each pick its own one-decision game."""
        obs = self._preview_observations(battle, picks)
        out = self.net(stack([[o] for o in obs], self.device))
        return value_calibration.apply(self.value_map, out["value"][:, 0].float().cpu().numpy())

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
        """(observation, turn, replaces, memory before, memory after, outputs, belief) of this view."""
        battle = view.battle
        turn = int(battle.turn or 0)  # type: ignore[union-attr]
        g = self._game(turn)
        obs = self._observe(battle, view.legal, g)
        replaces = g.last is not None and g.turn == turn and _same(g.last, obs)
        base = g.before if replaces else g.cache
        cache = _copy(base)
        allowed = None
        if self.opponent_moves_from_builds:
            from pcsa.rl_training.opponent_moves import allowed_mask

            allowed = allowed_mask(battle)
        return obs, turn, replaces, base, cache, self.net.step_cached(obs, cache, allowed), g.belief

    def propose(self, view: DecisionView) -> Proposal:
        """The decision for this view; changes nothing (the game's memory is stepped on a copy)."""
        obs, turn, replaces, base, cache, out, belief = self._step(view)
        out = rule_masks.apply(self.rule_mask, view, obs, out)
        mask = torch.as_tensor(rule_masks.legal(obs, out), device=out["logits"].device)
        index, probs = self._sample(out["logits"][None], mask[None])
        chosen = features.legal_action(index, view.legal, view.battle)
        tag = getattr(view.battle, "battle_tag", None)
        return Proposal(obs, index, probs, float(out["value"]), turn, tag, replaces, chosen, base, cache, out, belief)

    def commit(self, p: Proposal) -> LegalAction:
        """Make a proposal the game's latest decision (later decisions remember it) and return its action."""
        self.game = _Game(turn=p.turn, tag=p.tag, cache=p.cache, before=p.base, last=p.obs, belief=p.belief)
        mask = torch.as_tensor(rule_masks.legal(p.obs, p.out), device=p.probs.device)
        h = float(entropy.normalised_entropy(p.probs, mask[None])[0])
        # v: the critic's value of this position (expected return: +1 win / -1 loss, +-0.1 per faint)
        self._notes = {"rule": "rl", "p": round(float(p.probs.max()), 4), "h": round(h, 4)}
        self._notes.update(value_calibration.notes(self.value_map, p.value))  # v (calibrated when set), v_raw
        self._notes["matrix"] = matrix_notes(p.obs, p.out, p.probs, p.index)
        # tau: `probs` is softmax over legal rows of (sum_j p_cols[j] * cells[i][j]) / tau
        self._notes["matrix"]["opp_temperature"] = round(float(self.net.opp_temperature), 4)
        # opp_top: only their N likeliest columns weighed our actions (0 = all); w_cols are the weights used
        self._notes["matrix"]["opp_top"] = int(self.net.opp_top)
        self._notes["matrix"]["w_cols"] = [round(float(x), 3) for x in p.out["kept_w"].tolist()]
        scale = self.temperature if self.temperature > 0 else 1.0
        self._notes["matrix"]["tau"] = round(float(self.net.log_tau.detach().exp()) * scale, 4)
        self._notes["decisions"] = int(p.cache.get("t", 0))
        if p.belief is not None:
            self._notes["belief"] = belief_notes(p.belief)
        rule_masks.notes(self._notes, p.out)
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
            except Exception:
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

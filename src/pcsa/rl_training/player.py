# From research/pcsa-rl/rl/player.py (sha256 7929704ccfc1, 2026-11-08 22:33); keep diffable.
# Inference only: the episode buffer and rewards (EpisodeData, finish, episode, rewards, build_episode) stay in
# research. Changes for the phone, none of which alters a decision in the simulator:
# - a new seat-game starts at team preview or when the turn goes backwards, not when the battle tag changes
#   (the live loop rebuilds the Battle with a new tag for every decision);
# - asking again at the same turn with the same observation (a live same-turn retry) replaces the last
#   buffered decision instead of adding one;
# - a decision is split into propose() (no state change) and commit(), so a caller with a deadline can drop a
#   late proposal; switches are resolved against the decision's own Battle (thread-safe);
# - greedy play and greedy preview are options; from_checkpoint reads the size from the checkpoint.
"""The PPO policy as a pcsa Player: critic-scored team preview, window-attention decisions, a seat-game buffer."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from pcsa.battle.builds import Catalog
from pcsa.common.clock import Deadline
from pcsa.players.base import DecisionView, LegalAction, TeamPreviewView, random_team
from pcsa.players.opponent import Tracker
from pcsa.rl_training import entropy, features, preview
from pcsa.rl_training.model import SIZES, ModelConfig, Net
from pcsa.rl_training.spec import WINDOW

PREVIEW_TAU, PREVIEW_TOP_P = 0.1, 0.9


@dataclass
class _Game:
    turn: int = -1
    obs: list = field(default_factory=list)  # the last WINDOW - 1 decisions' observations
    decisions: int = 0


@dataclass(frozen=True)
class Proposal:
    """One decision before it is committed: the observation, the chosen index, its capped distribution."""

    obs: features.Obs
    index: int
    probs: torch.Tensor  # [1, ACTIONS]
    value: float
    pos: int
    turn: int
    replaces: bool  # a same-turn retry of the last buffered decision
    chosen: LegalAction


def as_numpy_rng(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, random.Random):
        return np.random.default_rng(rng.getrandbits(64))
    return np.random.default_rng(rng)


def checkpoint_config(state: dict, size: str | None = None) -> tuple[ModelConfig, str]:
    """The checkpoint's ModelConfig ("model" fields, else its run config's size) and its size name; ValueError
    when `size` is given and differs, or when the checkpoint names no size."""
    model = state.get("model")
    run_size = (state.get("config") or {}).get("size")
    if isinstance(model, dict):
        cfg = ModelConfig(**model)
        name = next((k for k, v in SIZES.items() if v == cfg), run_size or "custom")
    elif run_size in SIZES:
        cfg, name = SIZES[run_size], run_size
    elif size in SIZES:
        cfg, name = SIZES[size], size
    else:
        raise ValueError("the checkpoint names no model size; set `size`")
    if size is not None and size != name:
        raise ValueError(f"size mismatch: the checkpoint is {name}, the config asks for {size}")
    return cfg, name


def load_checkpoint(path: Path | str) -> dict:
    """A torch checkpoint (our own training output: it holds numpy rng state, so not weights_only)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state if isinstance(state, dict) and "net" in state else {"net": state}


class PolicyPlayer:
    """pcsa Player protocol over a Net; never raises on a decision (falls back to the first legal one)."""

    def __init__(
        self,
        net: Net,
        cfg: ModelConfig,
        *,
        catalog: Catalog,
        device: str = "cpu",
        temperature: float = 1.0,
        rng=None,
        tag: str = "policy",
        greedy_preview: bool = False,
    ) -> None:
        self.net, self.cfg, self.catalog, self.device = net.to(device).eval(), cfg, catalog, device
        self.temperature, self.rng = temperature, as_numpy_rng(rng)
        self.greedy_preview = greedy_preview
        self.name = f"rl:{tag}"
        self.tracker = Tracker(catalog)
        self.game = _Game()
        self.fallbacks = 0
        self._notes: dict = {}

    @classmethod
    def from_checkpoint(
        cls, path: Path | str, size: str | None = None, *, catalog: Catalog | None = None, **kw
    ) -> PolicyPlayer:
        state = load_checkpoint(path)
        cfg, _ = checkpoint_config(state, size)
        player = cls(Net(cfg), cfg, catalog=catalog or Catalog.load(), **kw)
        player.load_state_dict(state["net"])
        return player

    def load_state_dict(self, sd: dict) -> None:
        self.net.load_state_dict(sd)
        self.net.eval()

    def set_temperature(self, t: float) -> None:
        self.temperature = t

    def params(self) -> dict:
        return {"temperature": self.temperature, "d": self.cfg.d, "layers": self.cfg.layers}

    def notes(self) -> dict:
        notes, self._notes = self._notes, {}
        return notes

    # Team preview -------------------------------------------------------------------------------------
    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        """The pick sampled from the critic's top-p nucleus over all 60 (lead, second, third) options."""
        self.game = _Game()
        self.tracker.reset()
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
        """Critic value of each pick's turn-0 observation; one batch, each its own sequence."""
        obs = features.preview_observations(battle, picks, catalog=self.catalog, tracker=self.tracker)
        rows, ids, fld = Net._stack(obs, torch.device(self.device))
        n = len(obs)
        mask = torch.as_tensor(np.stack([o.mask for o in obs]), dtype=torch.bool, device=self.device)
        index = torch.arange(n, device=self.device)
        _, value = self.net(rows, ids, fld, mask, index, torch.zeros(n, dtype=torch.long, device=self.device))
        return value.float().cpu().numpy()

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

    def propose(self, view: DecisionView) -> Proposal:
        """The decision for this view; changes nothing but the opponent tracker's evidence."""
        battle = view.battle
        turn = int(battle.turn or 0)  # type: ignore[union-attr]
        g = self.game if turn >= self.game.turn else _Game()
        self.tracker.observe(battle)
        obs = features.observe(battle, view.legal, catalog=self.catalog, tracker=self.tracker)
        replaces = bool(g.obs) and g.turn == turn and _same(g.obs[-1], obs)
        past = g.obs[:-1] if replaces else g.obs
        pos = g.decisions - 1 if replaces else g.decisions
        window = [*past[-(WINDOW - 1) :], obs]
        logits, value = self.net.step(window, pos=pos)
        mask = torch.as_tensor(np.asarray(obs.mask, bool), device=logits.device)
        index, probs = self._sample(logits[None], mask[None])
        chosen = features.legal_action(index, view.legal, battle)
        return Proposal(obs, index, probs, float(value), pos, turn, replaces, chosen)

    def commit(self, p: Proposal) -> LegalAction:
        """Buffer a proposal's observation (the next decisions attend over it) and return its action."""
        g = self.game if p.turn >= self.game.turn else _Game()
        if p.replaces and g.obs:
            g.obs.pop()
        g.obs = [*g.obs, p.obs][-(WINDOW - 1) :]
        g.decisions, g.turn = p.pos + 1, p.turn
        self.game = g
        mask = torch.as_tensor(np.asarray(p.obs.mask, bool), device=p.probs.device)
        h = float(entropy.normalised_entropy(p.probs, mask[None])[0])
        self._notes = {"rule": "rl", "p": round(float(p.probs.max()), 4), "h": round(h, 4)}
        if p.replaces:
            self._notes["retry"] = True
        return p.chosen

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


def _same(a: features.Obs, b: features.Obs) -> bool:
    return (
        np.array_equal(a.rows, b.rows)
        and np.array_equal(a.ids, b.ids)
        and np.array_equal(a.field, b.field)
        and np.array_equal(a.mask, b.mask)
    )

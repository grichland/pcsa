# From research/pcsa-rl/rl/model.py (sha256 4a3ef609b004, 2026-11-08 20:19); keep diffable.
"""Policy/value network: per-decision token encoders, a windowed transformer over the seat-game, flatten heads.

A decision's output depends only on the decisions of its seat-game with seq_pos in [pos - window + 1, pos]: the
transformer runs over that window alone (block-causal inside it), so `step` on a buffer reproduces `forward`.
Decisions whose window starts at their game's first decision share one pass; later ones get their own window.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pcsa.rl_training.spec import ACTIONS, FIELD_FLOATS, POKEMON_FLOATS, POKEMON_IDS, ROWS, VOCAB, WINDOW

TOKENS = ROWS + 1
NEG = -1e9


@dataclass(frozen=True)
class ModelConfig:
    d: int = 64
    layers: int = 4
    heads: int = 4
    ffn_mult: float = 8 / 3
    window: int = WINDOW
    readout: int = 512
    emb_species: int = 32
    emb_item: int = 16
    emb_ability: int = 16
    emb_move: int = 16
    max_positions: int = 64


SIZES = {"S": ModelConfig(64, 4, 4), "M": ModelConfig(192, 6, 6), "L": ModelConfig(256, 8, 8)}


@dataclass(frozen=True)
class Plan:
    """Window layout of a packed batch: gather [B, Lw] decision indices (N = pad), win/slot [N] readout place."""

    gather: np.ndarray
    win: np.ndarray
    slot: np.ndarray


def plan_windows(seq_index: np.ndarray, seq_pos: np.ndarray, window: int) -> Plan:
    """Group packed decisions into attention windows of at most `window` decisions each."""
    n = len(seq_index)
    order = np.lexsort((seq_pos, seq_index))
    g, p = seq_index[order].astype(np.int64), seq_pos[order].astype(np.int64)
    idx = np.arange(n)
    new = np.ones(n, bool)
    new[1:] = g[1:] != g[:-1]
    game = np.cumsum(new) - 1
    start = idx[new][game]
    key = g * (1 << 32) + p
    lo = np.maximum(np.searchsorted(key, key - window + 1, side="left"), start)
    head = lo == start
    tail = idx[~head]
    w_start = np.concatenate([idx[new], lo[tail]])
    w_len = np.concatenate([np.bincount(game, weights=head).astype(np.int64), tail - lo[tail] + 1])
    win = game.copy()
    win[tail] = new.sum() + np.arange(len(tail))
    slot = np.where(head, idx - start, w_len[win] - 1)
    ar = np.arange(w_len.max())
    at = np.minimum(w_start[:, None] + ar[None, :], n - 1)
    gather = np.where(ar[None, :] < w_len[:, None], order[at], n)
    win_o, slot_o = np.empty(n, np.int64), np.empty(n, np.int64)
    win_o[order], slot_o[order] = win, slot
    return Plan(gather, win_o, slot_o)


def _mlp(width_in: int, hidden: Sequence[int], width_out: int, norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = width_in
    for h in hidden:
        layers += [nn.Linear(last, h), nn.GELU()]
        last = h
    layers.append(nn.Linear(last, width_out))
    if norm:
        layers.append(nn.LayerNorm(width_out))
    return nn.Sequential(*layers)


class Block(nn.Module):
    """Pre-LN block: multi-head self-attention under a boolean mask, then a SwiGLU feed-forward."""

    def __init__(self, d: int, heads: int, hidden: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.gate_up, self.down = nn.Linear(d, 2 * hidden), nn.Linear(hidden, d)

    def forward(self, x: Tensor, allowed: Tensor) -> Tensor:
        b, length, d = x.shape
        qkv = self.qkv(self.ln1(x)).view(b, length, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=allowed)
        x = x + self.proj(a.transpose(1, 2).reshape(b, length, d))
        gate, up = self.gate_up(self.ln2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


class Net(nn.Module):
    """forward(rows, ids, field, mask, seq_index, seq_pos) -> (logits [N, ACTIONS] masked to -1e9, value [N])."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.species = nn.Embedding(VOCAB["species"], cfg.emb_species, padding_idx=0)
        self.item = nn.Embedding(VOCAB["item"], cfg.emb_item, padding_idx=0)
        self.ability = nn.Embedding(VOCAB["ability"], cfg.emb_ability, padding_idx=0)
        self.move = nn.Embedding(VOCAB["move"], cfg.emb_move, padding_idx=0)
        emb = cfg.emb_species + cfg.emb_item + cfg.emb_ability + (POKEMON_IDS - 3) * cfg.emb_move
        self.pokemon_net = _mlp(POKEMON_FLOATS + emb, (512, 256), d)
        self.field_net = _mlp(FIELD_FLOATS, (256,), d)
        self.token_type = nn.Parameter(torch.randn(TOKENS, d) * 0.02)
        self.position = nn.Parameter(torch.randn(cfg.max_positions, d) * 0.02)
        hidden = max(8, round(cfg.ffn_mult * d / 8) * 8)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, hidden) for _ in range(cfg.layers)])
        self.final = nn.LayerNorm(d)
        self.policy = _mlp(TOKENS * d, (cfg.readout,), ACTIONS, norm=False)
        self.value = _mlp(TOKENS * d, (cfg.readout,), 1, norm=False)

    def encode(self, rows: Tensor, ids: Tensor, field: Tensor, seq_pos: Tensor | None = None) -> Tensor:
        """rows [N,12,F], ids [N,12,7], field [N,126], seq_pos [N] -> tokens [N,13,d] (no position if None)."""
        emb = torch.cat(
            [
                self.species(ids[..., 0]),
                self.item(ids[..., 1]),
                self.ability(ids[..., 2]),
                self.move(ids[..., 3:]).flatten(-2),
            ],
            dim=-1,
        )
        pokemon = self.pokemon_net(torch.cat([rows, emb], dim=-1))
        x = torch.cat([self.field_net(field)[:, None], pokemon], dim=1) + self.token_type
        return x if seq_pos is None else self.place(x, seq_pos)

    def place(self, x: Tensor, seq_pos: Tensor) -> Tensor:
        """Add the absolute decision-position embedding to every token of each decision."""
        return x + self.position[seq_pos.clamp(0, self.cfg.max_positions - 1)][:, None]

    def attend(self, x: Tensor, plan: Plan) -> Tensor:
        """Run the blocks over each window of tokens; return every decision's own 13 output tokens."""
        d = x.shape[-1]
        gather = torch.as_tensor(plan.gather, device=x.device)
        b, lw = gather.shape
        h = torch.cat([x, x.new_zeros(1, TOKENS, d)])[gather].reshape(b, lw * TOKENS, d)
        dec = torch.arange(lw * TOKENS, device=x.device) // TOKENS
        allowed = dec[None, :] <= dec[:, None]
        for block in self.blocks:
            h = block(h, allowed)
        h = self.final(h).reshape(b, lw, TOKENS, d)
        win = torch.as_tensor(plan.win, device=x.device)
        slot = torch.as_tensor(plan.slot, device=x.device)
        return h[win, slot]

    def heads(self, z: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        flat = z.flatten(1)
        logits = self.policy(flat)
        logits = torch.where(mask, logits, torch.full_like(logits, NEG))
        return logits, self.value(flat)[:, 0]

    def forward(
        self, rows: Tensor, ids: Tensor, field: Tensor, mask: Tensor, seq_index: Tensor, seq_pos: Tensor
    ) -> tuple[Tensor, Tensor]:
        x = self.encode(rows, ids, field, seq_pos)
        plan = plan_windows(seq_index.cpu().numpy(), seq_pos.cpu().numpy(), self.cfg.window)
        return self.heads(self.attend(x, plan), mask)

    @torch.no_grad()
    def step(
        self, obs_window: Sequence[Any], pos: int | None = None, cache: dict | None = None
    ) -> tuple[Tensor, Tensor]:
        """Buffered Obs of one seat-game (oldest first) -> (logits [11], value []) for the last decision.

        Only the last `window` entries are used. `pos` is the last decision's index in the game; it defaults to
        len(obs_window) - 1, which is right when the buffer holds the whole seat-game. `cache` (one dict per
        seat-game, cleared when the weights change) keeps each Obs's encoded tokens so only new ones are encoded.
        """
        obs = list(obs_window)[-self.cfg.window :]
        last = len(obs_window) - 1 if pos is None else pos
        device = next(self.parameters()).device
        seq_pos = torch.arange(last - len(obs) + 1, last + 1, device=device)
        x = self.place(self._tokens(obs, device, {} if cache is None else cache), seq_pos)
        plan = Plan(np.arange(len(obs))[None], np.zeros(1, np.int64), np.full(1, len(obs) - 1))
        mask = torch.as_tensor(np.asarray(obs[-1].mask, bool), device=device)[None]
        logits, value = self.heads(self.attend(x, plan)[-1:], mask)
        return logits[0], value[0]

    def _tokens(self, obs: list[Any], device: torch.device, cache: dict) -> Tensor:
        fresh = [o for o in obs if cache.get(id(o), (None,))[0] is not o]
        if fresh:
            for o, t in zip(fresh, self.encode(*self._stack(fresh, device)), strict=True):
                cache[id(o)] = (o, t)
        return torch.stack([cache[id(o)][1] for o in obs])

    @staticmethod
    def _stack(obs: Sequence[Any], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        rows = torch.as_tensor(np.stack([o.rows for o in obs]), dtype=torch.float32, device=device)
        ids = torch.as_tensor(np.stack([o.ids for o in obs]), dtype=torch.long, device=device)
        field = torch.as_tensor(np.stack([o.field for o in obs]), dtype=torch.float32, device=device)
        return rows, ids, field

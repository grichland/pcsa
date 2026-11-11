# From research/pcsa-rl/rl/model.py (sha256 4a3ef609b004, 2026-11-08 20:19); keep diffable.
# pcsa: the first network (ModelConfig, SIZES, window planning, Net) is removed; these two building blocks are
# what former.py uses.
"""Building blocks of the policy network (former.py): the MLP helper and the transformer block."""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn

from pcsa.rl_training.spec import ROWS

TOKENS = ROWS + 1
NEG = -1e9


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

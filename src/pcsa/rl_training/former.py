# From research/pcsa-rl/rl2/model.py (sha256 7202f21a923a, 2026-11-11 13:12); keep diffable.
"""poke-singles-former (plan 1, research/pcsa-rl-s-v2): predict the opponent, then choose through a pair matrix.

Batch layout: games [G, T] padded (valid [G, T]). Per decision 14 state tokens: field, our 6 rows, their 6 rows,
the last turn. A token sees its own decision's 14 tokens and the field token of up to `memory` earlier decisions of
the same game (ps-ppo's turn memory); so one pass over a game equals step-by-step play. Three readouts cross-attend:
opp-action (to the 14), our-action (to the 14 and the opp readout), value (to the 14).

Heads:
  opponent: logits over every move id (masked to their active's learnset plus revealed moves) and their 5 bench
            rows; a separate Mega logit
  matrix:   M[i, j] = c_i + A_i . B_j for our 11 actions x their kept actions (top-K moves by p, then 5 switches);
            our logits = sum_j stopgrad(p_kept_j) M[i, j] / tau
  value:    51-bin distribution over [-1.6, 1.6]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pcsa.rl_training import former_features as fx
from pcsa.rl_training.features_vocab import index
from pcsa.rl_training.model import Block, _mlp
from pcsa.rl_training.spec import ACTIONS, VOCAB

NEG = -1e9
STATE = 14  # field, 12 rows, last turn
THEIR_ACTIVE, THEIR_BENCH = 7, slice(8, 13)  # token positions
OUR_ACTIVE, OUR_BENCH = 1, slice(2, 5)
SWITCHES = 5
BINS, VMIN, VMAX = 51, -1.6, 1.6
HP_COL = 34
COULD_MEGA_COL = fx.ROW + 12


@dataclass(frozen=True)
class Config:
    d: int = 128
    layers: int = 4
    heads: int = 4
    ffn_mult: float = 8 / 3
    rank: int = 32
    top_k: int = 4  # opposing move columns: every revealed move (at most 4), then the most likely (8 until 11 Nov)
    memory: int = 64
    max_positions: int = 128
    emb_species: int = 32
    emb_item: int = 16
    emb_ability: int = 16
    emb_move: int = 32


class Readout(nn.Module):
    """A learned query cross-attending to a decision's tokens, then a feed-forward."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln, self.ffn = nn.LayerNorm(d), _mlp(d, (2 * d,), d)

    def forward(self, kv: Tensor) -> Tensor:  # kv [B, L, d] -> [B, d]
        q = self.query.expand(kv.shape[0], 1, -1)
        a, _ = self.attn(q, kv, kv, need_weights=False)
        h = q + a
        return (h + self.ffn(self.ln(h)))[:, 0]


def learnset_table() -> Tensor:
    """[species vocab, move vocab] bool: the moves each species (its base form) can learn in Champions."""
    from pcsa.battle import game

    table = torch.zeros(VOCAB["species"], VOCAB["move"], dtype=torch.bool)
    learn = game.learnsets()
    for name in game.dex():
        s = index("species", name)
        moves = learn.get(name) or learn.get(game.base_name(name)) or ()
        for m in moves:
            mi = index("move", m)
            if s and mi:
                table[s, mi] = True
    return table


class Net(nn.Module):
    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or Config()  # pcsa: no call in the default (research: cfg: Config = Config())
        d = cfg.d
        self.species = nn.Embedding(VOCAB["species"], cfg.emb_species, padding_idx=0)
        self.item = nn.Embedding(VOCAB["item"], cfg.emb_item, padding_idx=0)
        self.ability = nn.Embedding(VOCAB["ability"], cfg.emb_ability, padding_idx=0)
        self.move = nn.Embedding(VOCAB["move"], cfg.emb_move, padding_idx=0)
        emb = cfg.emb_species + cfg.emb_item + 4 * cfg.emb_ability + 4 * cfg.emb_move
        self.pokemon_net = _mlp(fx.WIDTH + emb, (2 * d,), d)
        self.field_net = _mlp(126, (2 * d,), d)
        self.last_net = _mlp(fx.LAST + 2 * cfg.emb_move + 2 * cfg.emb_species, (2 * d,), d)
        self.token_type = nn.Parameter(torch.randn(STATE, d) * 0.02)
        self.position = nn.Parameter(torch.randn(cfg.max_positions, d) * 0.02)
        hidden = max(8, round(cfg.ffn_mult * d / 8) * 8)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, hidden) for _ in range(cfg.layers)])
        self.final = nn.LayerNorm(d)
        self.read_opp, self.read_us, self.read_v = (Readout(d, cfg.heads) for _ in range(3))
        self.move_vec = _mlp(cfg.emb_move, (d,), d)  # every move id -> d (opponent logits and action vectors)
        self.opp_move_q, self.opp_switch_q = nn.Linear(d, d), nn.Linear(d, d)
        self.opp_mega = nn.Linear(d, 1)
        self.a_move, self.b_move = _mlp(2 * d, (d,), cfg.rank, norm=False), _mlp(2 * d, (d,), cfg.rank, norm=False)
        self.a_switch, self.b_switch = nn.Linear(d, cfg.rank), nn.Linear(d, cfg.rank)
        self.mega_vec = nn.Parameter(torch.zeros(cfg.rank))
        self.base = _mlp(d, (d,), ACTIONS, norm=False)
        self.log_tau = nn.Parameter(torch.zeros(()))
        # Inference only (training keeps 1): the temperature of their action distribution where it weights the
        # matrix columns. Above 1 hedges against their less likely actions, below 1 trusts the favourite more.
        self.opp_temperature = 1.0
        # Inference only (training keeps 0 = all): weigh our actions against only their N likeliest kept actions.
        self.opp_top = 0
        self.value = _mlp(d, (d,), BINS, norm=False)
        self.register_buffer("support", torch.linspace(VMIN, VMAX, BINS), persistent=False)
        self.register_buffer("learnset", learnset_table(), persistent=False)

    def move_vectors(self) -> Tensor:
        """Every move id's vector [V, d]; cached while not training (the weights cannot change then)."""
        if self.training or torch.is_grad_enabled():
            self._moves = None
            return self.move_vec(self.move.weight)
        moves = getattr(self, "_moves", None)
        if moves is None:
            moves = self._moves = self.move_vec(self.move.weight)
        return moves

    def load_state_dict(self, *args, **kwargs):
        self._moves = None
        return super().load_state_dict(*args, **kwargs)

    # Tokens ----------------------------------------------------------------------------------------------------
    def encode(self, b: dict) -> Tensor:
        """b: rows [G,T,12,W], ids [G,T,12,10], field [G,T,126], last [G,T,20], last_ids [G,T,4] -> [G,T,14,d]."""
        ids = b["ids"]
        emb = torch.cat(
            [
                self.species(ids[..., 0]),
                self.item(ids[..., 1]),
                self.ability(ids[..., 2]),
                self.ability(ids[..., 7:10]).flatten(-2),
                self.move(ids[..., 3:7]).flatten(-2),
            ],
            dim=-1,
        )
        rows = self.pokemon_net(torch.cat([b["rows"], emb], dim=-1))
        li = b["last_ids"]
        last_emb = torch.cat(
            [self.move(li[..., 0]), self.species(li[..., 1]), self.move(li[..., 2]), self.species(li[..., 3])], dim=-1
        )
        last = self.last_net(torch.cat([b["last"], last_emb], dim=-1))
        x = torch.cat([self.field_net(b["field"])[..., None, :], rows, last[..., None, :]], dim=-2)
        t = x.shape[1]
        pos = self.position[torch.arange(t, device=x.device).clamp(max=self.cfg.max_positions - 1)]
        return x + self.token_type + pos[None, :, None, :]

    def attend(self, x: Tensor, valid: Tensor) -> Tensor:
        """[G,T,14,d] -> contextualised [G,T,14,d] under the turn-memory mask."""
        g, t, s, d = x.shape
        dec = torch.arange(t * s, device=x.device) // s
        slot = torch.arange(t * s, device=x.device) % s
        same = dec[:, None] == dec[None, :]
        memory = (
            (dec[None, :] < dec[:, None]) & (slot[None, :] == 0) & (dec[:, None] - dec[None, :] <= self.cfg.memory)
        )
        allowed = (same | memory)[None, None] & valid.repeat_interleave(s, dim=1)[:, None, None, :]
        allowed = allowed | torch.eye(t * s, dtype=torch.bool, device=x.device)[None, None]
        h = x.reshape(g, t * s, d)
        for block in self.blocks:
            h = block(h, allowed)
        return self.final(h).reshape(g, t, s, d)

    # Heads -----------------------------------------------------------------------------------------------------
    def forward(self, b: dict) -> dict:
        h = self.attend(self.encode(b), b["valid"])
        g, t, s, d = h.shape
        flat = h.reshape(g * t, s, d)
        r_opp = self.read_opp(flat)
        r_us = self.read_us(torch.cat([flat, r_opp[:, None]], dim=1))
        r_v = self.read_v(flat)
        out = self._opponent(flat, r_opp, b)
        out.update(self._matrix(flat, r_us, out, b))
        vlog = self.value(r_v)
        out["value_logits"] = vlog
        out["value"] = (vlog.softmax(-1) * self.support).sum(-1)
        return {k: v.reshape(g, t, *v.shape[1:]) for k, v in out.items()}  # every output is [G*T, ...] here

    def _opponent(self, h: Tensor, r: Tensor, b: dict) -> dict:
        ids = b["ids"].reshape(-1, *b["ids"].shape[2:])
        rows = b["rows"].reshape(-1, *b["rows"].shape[2:])
        moves = self.move_vectors()  # [V, d]
        scale = self.cfg.d**-0.5
        move_logits = (self.opp_move_q(r) @ moves.T) * scale  # [B, V]
        allowed = cast(Tensor, self.learnset)[ids[:, 6, 0]].clone()
        narrowed = b.get("opp_allowed")  # pcsa (inference): their moves narrowed by builds (opponent_moves.py)
        if narrowed is not None:
            narrowed = allowed & narrowed
            allowed = torch.where(narrowed[:, 1:].any(-1, keepdim=True), narrowed, allowed)
        allowed.scatter_(1, ids[:, 6, 3:7], True)
        allowed[:, 0] = False
        bench = h[:, THEIR_BENCH]
        switch_logits = (self.opp_switch_q(r)[:, None] * bench).sum(-1) * scale  # [B, 5]
        switch_ok = (ids[:, 7:12, 0] > 0) & (rows[:, 7:12, HP_COL] > 0)
        logits = torch.cat([move_logits, switch_logits], dim=-1)
        ok = torch.cat([allowed, switch_ok], dim=-1)
        none = ~ok.any(-1)
        ok[none, 0] = True  # no legal opponent action known: keep one placeholder
        logits = torch.where(ok, logits, torch.full_like(logits, NEG))
        mega = self.opp_mega(r)[:, 0]
        mega = torch.where(rows[:, 6, COULD_MEGA_COL] > 0, mega, torch.full_like(mega, NEG))
        return {"opp_logits": logits, "opp_mega": mega, "moves": moves}

    def _matrix(self, h: Tensor, r: Tensor, opp: dict, b: dict) -> dict:
        ids = b["ids"].reshape(-1, *b["ids"].shape[2:])
        mask = b["mask"].reshape(-1, ACTIONS)
        moves, v = opp.pop("moves"), VOCAB["move"]
        p = (opp["opp_logits"].detach() / self.opp_temperature).softmax(-1)
        # their active's revealed moves always get a column; the rest of the top_k are filled by probability
        shown = torch.zeros_like(p[:, :v]).scatter_(1, ids[:, 6, 3:7], 2.0)
        shown[:, 0] = 0.0  # id 0 = no move revealed in that slot
        picked = (p[:, :v] + shown).topk(self.cfg.top_k, dim=-1).indices
        kept_idx = torch.cat([picked, v + torch.arange(SWITCHES, device=p.device).expand(p.shape[0], -1)], -1)
        kept_p = torch.cat([p[:, :v].gather(1, picked), p[:, v:]], dim=-1)
        kept_p = kept_p / kept_p.sum(-1, keepdim=True).clamp_min(1e-9)
        their_active = h[:, THEIR_ACTIVE][:, None].expand(-1, self.cfg.top_k, -1)
        b_moves = self.b_move(torch.cat([their_active, moves[picked]], dim=-1))
        b_vec = torch.cat([b_moves, self.b_switch(h[:, THEIR_BENCH])], dim=1)  # [B, K+5, r]
        ours = h[:, OUR_ACTIVE][:, None].expand(-1, 4, -1)
        a_moves = self.a_move(torch.cat([ours, moves[ids[:, 0, 3:7]]], dim=-1))  # [B, 4, r]
        a_vec = torch.cat([a_moves, a_moves + self.mega_vec, self.a_switch(h[:, OUR_BENCH])], dim=1)  # [B, 11, r]
        m = self.base(r)[:, :, None] + a_vec @ b_vec.transpose(1, 2)  # [B, 11, K+5]
        used = kept_p
        if self.opp_top:  # their N likeliest kept actions only, renormalised
            best = kept_p.topk(min(int(self.opp_top), kept_p.shape[-1]), dim=-1).indices
            used = kept_p * torch.zeros_like(kept_p).scatter_(1, best, 1.0)
            used = used / used.sum(-1, keepdim=True).clamp_min(1e-9)
        logits = (m * used[:, None, :]).sum(-1) / self.log_tau.exp()
        logits = torch.where(mask, logits, torch.full_like(logits, NEG))
        return {"logits": logits, "matrix": m, "kept_idx": kept_idx, "kept_p": kept_p, "kept_w": used}

    # One game, step by step (inference) --------------------------------------------------------------------------
    @torch.no_grad()
    def step_cached(self, obs, cache: dict, opp_allowed=None) -> dict:
        """The next decision of a game, given `cache` (one dict per game, empty at its start): each layer keeps the
        inputs of earlier decisions' field tokens (the only tokens later decisions see), so only this decision's 14
        tokens run. Equal to the full pass. `opp_allowed` ([move vocab] bool, pcsa): their active's moves narrowed
        to these (and its revealed ones) in the opponent head; None = every learnable move, as trained."""
        device = next(self.parameters()).device
        t = cache.get("t", 0)
        b = stack([[obs]], device)
        if opp_allowed is not None:
            b["opp_allowed"] = torch.as_tensor(np.asarray(opp_allowed, bool), device=device)[None]
        x = self.encode(b)[0, 0]  # [14, d]; position 0 from encode, replaced by position t
        x = x - self.position[0] + self.position[min(t, self.cfg.max_positions - 1)]
        mems = cache.setdefault("mem", [[] for _ in self.blocks])
        for layer, block in enumerate(cast("list[Block]", self.blocks)):
            past = mems[layer][-self.cfg.memory :]
            mems[layer].append(x[0].clone())
            x = _block_step(block, x, torch.stack(past) if past else x.new_zeros(0, x.shape[-1]))
        cache["t"] = t + 1
        h = self.final(x)[None]  # [1, 14, d]
        r_opp = self.read_opp(h)
        r_us = self.read_us(torch.cat([h, r_opp[:, None]], dim=1))
        out = self._opponent(h, r_opp, b)
        out.update(self._matrix(h, r_us, out, b))
        vlog = self.value(self.read_v(h))
        out["value_logits"] = vlog
        out["value"] = (vlog.softmax(-1) * self.support).sum(-1)
        return {k: v[0] for k, v in out.items()}

    @torch.no_grad()
    def step_batch(self, obs: list, caches: list[dict], value_only: bool = False) -> dict:
        """The next decision of B different games at once (obs[i] with its game's `caches[i]`, as step_cached):
        each game's tokens attend to themselves and to that game's own earlier field tokens (padded and masked).
        Returns batched outputs [B, ...]; equal to step_cached per game."""
        device = next(self.parameters()).device
        n = len(obs)
        b = stack([[o] for o in obs], device)
        t = torch.as_tensor([c.get("t", 0) for c in caches], device=device).clamp(max=self.cfg.max_positions - 1)
        x = self.encode(b)[:, 0] - self.position[0] + self.position[t][:, None, :]  # [B, 14, d]
        d = x.shape[-1]
        for c in caches:
            c.setdefault("mem", [[] for _ in self.blocks])
        for layer, block in enumerate(cast("list[Block]", self.blocks)):
            past = [c["mem"][layer][-self.cfg.memory :] for c in caches]
            width = max(len(p) for p in past)
            mem, ok = x.new_zeros(n, width, d), torch.zeros(n, width, dtype=torch.bool, device=device)
            for i, p in enumerate(past):
                if p:
                    mem[i, : len(p)] = torch.stack(p)
                    ok[i, : len(p)] = True
            for i, c in enumerate(caches):
                c["mem"][layer].append(x[i, 0].clone())
                del c["mem"][layer][: -self.cfg.memory - 1]
            x = _block_step_batch(block, x, mem, ok)
        for c in caches:
            c["t"] = c.get("t", 0) + 1
        h = self.final(x)
        out: dict = {}
        if not value_only:  # team preview needs only the value
            r_opp = self.read_opp(h)
            r_us = self.read_us(torch.cat([h, r_opp[:, None]], dim=1))
            out = self._opponent(h, r_opp, b)
            out.update(self._matrix(h, r_us, out, b))
        vlog = self.value(self.read_v(h))
        out["value_logits"] = vlog
        out["value"] = (vlog.softmax(-1) * self.support).sum(-1)
        return out

    @torch.no_grad()
    def step(self, game: list) -> dict:
        """rl2.features.Obs of one game so far (oldest first) -> the last decision's outputs (batch of one)."""
        obs = game[-(self.cfg.memory + 1) :]
        b = stack([obs], next(self.parameters()).device)
        out = self.forward(b)
        return {k: v[0, -1] for k, v in out.items()}


def _block_step(block: Block, x: Tensor, mem: Tensor) -> Tensor:
    """One Block for a single decision: its tokens x [14, d] attend to themselves and to `mem` [M, d] (earlier field
    tokens' inputs to this block), as in the full pass."""
    s, d = x.shape
    heads = block.heads
    allx = torch.cat([x, mem], dim=0)
    qkv = block.qkv(block.ln1(allx)).view(-1, 3, heads, d // heads)
    q = qkv[:s, 0].transpose(0, 1)[None]
    k, v = qkv[:, 1].transpose(0, 1)[None], qkv[:, 2].transpose(0, 1)[None]
    a = F.scaled_dot_product_attention(q, k, v)
    x = x + block.proj(a[0].transpose(0, 1).reshape(s, d))
    gate, up = block.gate_up(block.ln2(x)).chunk(2, dim=-1)
    return x + block.down(F.silu(gate) * up)


def _block_step_batch(block: Block, x: Tensor, mem: Tensor, ok: Tensor) -> Tensor:
    """_block_step for a batch: x [B, 14, d], mem [B, M, d] padded, ok [B, M] marks the real memory entries."""
    n, s, d = x.shape
    heads = block.heads
    qkv = block.qkv(block.ln1(torch.cat([x, mem], dim=1))).view(n, -1, 3, heads, d // heads)
    q = qkv[:, :s, 0].transpose(1, 2)
    k, v = qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
    allowed = torch.cat([torch.ones(n, s, dtype=torch.bool, device=x.device), ok], dim=1)[:, None, None, :]
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
    x = x + block.proj(a.transpose(1, 2).reshape(n, s, d))
    gate, up = block.gate_up(block.ln2(x)).chunk(2, dim=-1)
    return x + block.down(F.silu(gate) * up)


def stack(games: list[list], device) -> dict:
    """Games of rl2.features.Obs -> a padded batch dict [G, T, ...] with `valid`."""
    t = max(len(g) for g in games)
    pad = games[0][0]

    def arr(name: str, dtype) -> Tensor:
        out = np.zeros((len(games), t, *np.shape(getattr(pad, name))), dtype=dtype)
        for i, g in enumerate(games):
            out[i, : len(g)] = np.stack([getattr(o, name) for o in g])
        return torch.as_tensor(out, device=device)

    valid = np.zeros((len(games), t), dtype=bool)
    for i, g in enumerate(games):
        valid[i, : len(g)] = True
    return {
        "rows": arr("rows", np.float32),
        "ids": arr("ids", np.int64),
        "field": arr("field", np.float32),
        "last": arr("last", np.float32),
        "last_ids": arr("last_ids", np.int64),
        "mask": arr("mask", bool),
        "valid": torch.as_tensor(valid, device=device),
    }


def parameter_count(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

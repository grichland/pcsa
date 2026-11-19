# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false
"""The decision-matrix network on the CG model contract (MODEL.md 0.6), on the poke-singles-former trunk.

Tokens: field, 12 Pokemon, last turn (14); optionally the 48 move slots as tokens too. Trunk, as in
research/pcsa-rl/rl2/model.py: transformer blocks in which a decision's tokens attend to each other and to the field
and last-turn tokens of up to `memory` earlier decisions of the game, so one pass over a game equals step-by-step play. Heads:
opponent (13, pointing at their active's move slots and their bench rows), payoff matrix M [11, 13], value, team
preview (60, computed only at preview requests), all reading the field token as the state summary, and the
training-only belief heads (brought, lead, item, ability, moves of each opposing Pokemon).

Source: pcsa-former/pcsa_former/model.py, copied verbatim; only the imports are package-relative (and the
pyright line above: torch modules indexed and called the way the source does).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import decide
from . import layout as L

TOKENS = L.ROWS + 2  # field, 12 Pokemon, last turn
LOG_TAU_MIN, LOG_TAU_MAX = -7.0, 1.0  # temperature between about 0.001 and 2.7
MEMORY_TOKENS = (0, TOKENS - 1)  # what later decisions of the game can see: the field and last-turn tokens
INPUTS = ("rows", "row_ids", "field", "last", "last_ids", "mask", "preview_mask", "opp_prior")


@dataclass(frozen=True)
class Config:
    d: int = 128
    layers: int = 4
    heads: int = 4
    ffn_mult: float = 2.0  # gated feed-forward width / d; at most 2, so its way back down only halves
    rank: int = 32  # width of the action vectors whose inner product fills the matrix
    move_tokens: bool = False  # True: the 48 move slots are tokens of their own (62 tokens instead of 14)
    emb_species: int = 32
    emb_item: int = 16
    emb_ability: int = 16
    emb_move: int = 32
    detach_opp: bool = True  # the policy gradient does not flow into the opponent head
    max_prob: float = 0.9  # no action of either side is given more than this while another is possible
    tau: float = 0.05  # starting policy temperature: the matrix is in game-outcome units (-1..1), so small
    # differences in expected outcome must be able to make a sharp policy
    memory: int = 32  # earlier decisions of the game whose field and last-turn tokens a decision can see; 0 = none
    # the empty move slots of their active are named by the belief head, most likely move first (user, 19 Nov);
    # False: the first empty slot stands for every move not yet seen
    belief_slots: bool = False


def _steps(n_in: int, n_out: int, act: bool) -> list[nn.Module]:
    """Linear layers from n_in to n_out, none with more than twice as many inputs as outputs (user's rule): the
    width is halved until the last step fits. `act`: GELU between them; without it the chain is one linear map,
    initialised to keep the signal's size through its depth."""
    layers: list[nn.Module] = []
    cur = n_in
    while True:
        nxt = n_out if cur <= 2 * n_out else (cur + 1) // 2
        lin = nn.Linear(cur, nxt)
        if not act:
            nn.init.normal_(lin.weight, std=cur**-0.5)
            nn.init.zeros_(lin.bias)
        layers.append(lin)
        if nxt == n_out:
            return layers
        if act:
            layers.append(nn.GELU())
        cur = nxt


def mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    """n_in -> hidden with GELU, then a linear read-out to n_out; every step at most halves the width."""
    return nn.Sequential(*_steps(n_in, hidden, True), nn.GELU(), *_steps(hidden, n_out, False))


def taper(n_in: int, n_out: int) -> nn.Sequential:
    """A linear map n_in -> n_out as a chain of steps that each at most halve the width."""
    return nn.Sequential(*_steps(n_in, n_out, False))


def rotate(x: Tensor, pos: Tensor) -> Tensor:
    """Rotary position embedding: x [..., c] (c even) turned by its position; `pos` broadcasts to x's leading shape."""
    half = x.shape[-1] // 2
    freq = 10000.0 ** (-torch.arange(half, device=x.device, dtype=x.dtype) / half)
    angle = pos.to(x.dtype)[..., None] * freq
    cos, sin = angle.cos(), angle.sin()
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class Block(nn.Module):
    """Pre-norm block of the poke-singles-former trunk: a decision's tokens attend to each other and to the field
    and last-turn tokens of earlier decisions of the game; gated feed-forward. Position is rotary, by decision number:
    a decision's own tokens share one position, so only how many decisions back a remembered token is matters."""

    def __init__(self, d: int, heads: int, hidden: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.gate_up, self.down = nn.Linear(d, 2 * hidden), nn.Linear(hidden, d)

    def _finish(self, x: Tensor, a: Tensor) -> Tensor:
        x = x + self.proj(a)
        gate, up = self.gate_up(self.ln2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)

    def forward(self, x: Tensor, mem_ok: Tensor) -> Tensor:
        """A game at once. x [G, T, S, d]: T decisions of S tokens (MEMORY_TOKENS = field, last turn);
        mem_ok [G, T, T]: decision i may see those tokens of decision j. The remembered tokens are kept once per game
        and scored against every decision, not copied per decision."""
        g, t, s, d = x.shape
        h, c = self.heads, d // self.heads
        q, k, v = self.qkv(self.ln1(x)).view(g, t, s, 3, h, c).unbind(3)  # each [G, T, S, h, c]
        m = len(MEMORY_TOKENS)
        turn = torch.arange(t, device=x.device)
        own = torch.einsum("gtshc,gtuhc->gthsu", q, k)  # a decision's tokens share one position: no rotation needed
        mk = rotate(k[:, :, list(MEMORY_TOKENS)], turn[None, :, None, None]).reshape(g, t * m, h, c)
        mv = v[:, :, list(MEMORY_TOKENS)].reshape(g, t * m, h, c)
        far = torch.einsum("gtshc,gkhc->gthsk", rotate(q, turn[None, :, None, None]), mk)
        seen = mem_ok[..., None].expand(g, t, t, m).reshape(g, t, 1, 1, t * m)
        att = torch.cat([own, far.masked_fill(~seen, float("-inf"))], dim=-1).mul(c**-0.5).softmax(-1)
        a = torch.einsum("gthsu,gtuhc->gtshc", att[..., :s], v) + torch.einsum("gthsk,gkhc->gtshc", att[..., s:], mv)
        return self._finish(x, a.reshape(g, t, s, d))

    def step(self, x: Tensor, pos: Tensor, mem: Tensor, mpos: Tensor, ok: Tensor) -> Tensor:
        """The newest decision alone. x [B, S, d], decision number pos [B]; mem [B, M, d]: this layer's input memory
        tokens of earlier decisions, their numbers mpos [B, M]; ok [B, M]."""
        b, s, d = x.shape
        h, c = self.heads, d // self.heads
        qkv = self.qkv(self.ln1(torch.cat([x, mem], dim=1))).view(b, -1, 3, h, c)
        q = qkv[:, :s, 0].transpose(1, 2)
        k, v = qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
        allowed = torch.cat([torch.ones(b, s, dtype=torch.bool, device=x.device), ok], dim=1)[:, None, None, :]
        qpos = pos[:, None].expand(b, s)
        kpos = torch.cat([qpos, mpos], dim=1)
        a = F.scaled_dot_product_attention(rotate(q, qpos[:, None]), rotate(k, kpos[:, None]), v, attn_mask=allowed)
        return self._finish(x, a.transpose(1, 2).reshape(b, s, d))


class Memory:
    """What step-by-step play remembers: per slot (one seat of one game) the field and last-turn tokens of its last
    `memory` decisions, as each block received them. Gives the same result as passing the whole game at once.
    Size: slots x layers x memory x 2 x d floats (0.13 MB per slot at the defaults)."""

    def __init__(self, net: Net, slots: int, device=None) -> None:
        cfg = net.cfg
        self.size = max(cfg.memory, 1)
        self.hist = torch.zeros(slots, max(cfg.layers, 1), self.size, len(MEMORY_TOKENS), cfg.d, device=device)
        self.pos = torch.zeros(slots, self.size, dtype=torch.long, device=device)  # decision number of each entry
        self.count = torch.zeros(slots, dtype=torch.long, device=device)

    def reset(self, slots) -> None:
        """A new game for these slots."""
        self.count[torch.as_tensor(slots, dtype=torch.long, device=self.count.device)] = 0


class Net(nn.Module):
    def __init__(self, layout: L.Layout, cfg: Config | None = None, learnset: Tensor | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or Config()
        self.layout = layout
        d, w = cfg.d, layout.widths
        self.species = nn.Embedding(L.VOCAB["species"], cfg.emb_species, padding_idx=0)
        self.item = nn.Embedding(L.VOCAB["item"], cfg.emb_item, padding_idx=0)
        self.ability = nn.Embedding(L.VOCAB["ability"], cfg.emb_ability, padding_idx=0)
        self.move = nn.Embedding(L.VOCAB["move"], cfg.emb_move, padding_idx=0)

        # where things are, all from the layout
        self.register_buffer("plain_cols", torch.tensor(layout.row_plain_columns()), persistent=False)
        self.slot_base, self.slot_stride = layout.slot_base, layout.slot_stride
        self.is_active_col = layout.col(L.F_IS_ACTIVE)
        self.i_species, self.i_item, self.i_ability = (layout.col(f) for f in (L.F_SPECIES, L.F_ITEM, L.F_ABILITY))
        self.i_moves, self.i_possible = layout[L.F_MOVE_ID].columns(), layout[L.F_POSSIBLE_ABILITY].columns()
        self.i_last_moves = [layout.col(f) for f in L.F_LAST_MOVES]
        self.i_last_species = [layout.col(f) for f in L.F_LAST_SPECIES]

        # input encoders
        self.slot_net = mlp(self.slot_stride + cfg.emb_move, 2 * d, d)
        self.slot_pos = nn.Parameter(torch.randn(L.MOVE_SLOTS, d) * 0.02)
        self.empty_slot = nn.Parameter(torch.randn(d) * 0.02)  # id 0: no move, or theirs not yet seen
        emb = cfg.emb_species + cfg.emb_item + (1 + len(self.i_possible)) * cfg.emb_ability
        self.pokemon_net = mlp(len(self.plain_cols) + emb, 2 * d, d)
        self.moves_to_pokemon = nn.Linear(d, d)
        self.field_net = mlp(w["field"], 2 * d, d)
        self.last_net = mlp(w["last"] + 2 * cfg.emb_move + 2 * cfg.emb_species, 2 * d, d)
        self.token_type = nn.Parameter(torch.randn(L.ROWS + 2, d) * 0.02)
        self.move_token_type = nn.Parameter(torch.randn(d) * 0.02)

        hidden = max(8, round(min(cfg.ffn_mult, 2.0) * d / 8) * 8)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, hidden) for _ in range(cfg.layers)])
        self.final = nn.LayerNorm(d)

        # action vectors: one per own action and one per opponent action, each pointing at input tokens
        self.own_move, self.opp_move = mlp(2 * d + 1, 2 * d, d), mlp(2 * d + 1, 2 * d, d)  # +1: the Mega flag
        self.own_switch, self.opp_switch = mlp(d, d, d), mlp(d, d, d)
        self.opp_query = nn.Linear(d, d)
        self.a_proj, self.b_proj = taper(d, cfg.rank), taper(d, cfg.rank)
        self.a_bias, self.b_bias = taper(d, 1), taper(d, 1)
        self.base = mlp(d, d, L.OWN_ACTIONS)  # one baseline per own action from the state summary, as in the original
        self.log_tau = nn.Parameter(torch.tensor(float(cfg.tau)).log())
        self.value = mlp(d, d, 1)

        # team preview: one score per (lead, back, back)
        self.register_buffer("combos", torch.tensor(L.preview_combos()), persistent=False)
        self.lead_proj, self.back_proj, self.preview_ctx = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.preview_out = mlp(d, d, 1)

        # belief heads (training only), per opposing row
        self.belief_brought, self.belief_lead = taper(d, 1), taper(d, 1)
        self.belief_item = nn.Linear(d, L.VOCAB["item"])
        self.belief_ability = nn.Linear(d, L.VOCAB["ability"])
        self.belief_moves = nn.Linear(d, L.VOCAB["move"])
        if learnset is None:
            learnset = torch.ones(L.VOCAB["species"], L.VOCAB["move"], dtype=torch.bool)
        learnset = learnset.bool().clone()
        learnset[:, 0] = False  # move id 0 is never a real move
        self.register_buffer("learnset", learnset, persistent=False)

    # Encoder -----------------------------------------------------------------------------------------------------
    def embed(self, b: dict) -> tuple[Tensor, Tensor]:
        """-> (tokens [N, S, d] before the trunk, S = 14 or 62; move-slot vectors [N, 12, 4, d])."""
        rows, ids = b["rows"].float(), b["row_ids"].long()
        n, d = rows.shape[0], self.cfg.d
        lo, hi = self.slot_base, self.slot_base + self.slot_stride * L.MOVE_SLOTS
        move_ids = ids[..., self.i_moves]
        slots = rows[..., lo:hi].reshape(n, L.ROWS, L.MOVE_SLOTS, self.slot_stride)
        moves = self.slot_net(torch.cat([slots, self.move(move_ids)], dim=-1)) + self.slot_pos
        moves = moves + (move_ids == 0)[..., None] * self.empty_slot
        emb = torch.cat(
            [
                self.species(ids[..., self.i_species]),
                self.item(ids[..., self.i_item]),
                self.ability(ids[..., self.i_ability]),
                self.ability(ids[..., self.i_possible]).flatten(-2),
            ],
            dim=-1,
        )
        pokemon = self.pokemon_net(torch.cat([rows[..., self.plain_cols], emb], dim=-1))
        pokemon = pokemon + self.moves_to_pokemon(moves.mean(-2))
        li = b["last_ids"].long()
        last_emb = [self.move(li[:, i]) for i in self.i_last_moves] + [self.species(li[:, i]) for i in self.i_last_species]
        last = self.last_net(torch.cat([b["last"].float(), *last_emb], dim=-1))
        x = torch.cat([self.field_net(b["field"].float())[:, None], pokemon, last[:, None]], dim=1) + self.token_type
        if self.cfg.move_tokens:
            own_row = self.token_type[1 : 1 + L.ROWS, None, :]
            x = torch.cat([x, (moves + own_row + self.move_token_type).reshape(n, -1, d)], dim=1)
        return x, moves

    # Trunk -------------------------------------------------------------------------------------------------------
    def _trunk(self, x: Tensor, seq: Tensor) -> Tensor:
        """Tokens [N, S, d] and games `seq` [G, T] (row numbers, oldest first, -1 = padding) -> [N, S, d]: each
        decision's tokens after the blocks, having seen the field tokens of up to `memory` earlier decisions."""
        g, t = seq.shape
        valid = seq >= 0
        pos = torch.arange(t, device=x.device)
        h = x[seq.clamp_min(0)] * valid[..., None, None]
        back = pos[:, None] - pos[None, :]
        mem_ok = ((back > 0) & (back <= self.cfg.memory))[None] & valid[:, None, :]
        for block in self.blocks:
            h = block(h, mem_ok)
        h = self.final(h)
        out = torch.zeros_like(x)
        out[seq[valid]] = h[valid]
        return out

    def _trunk_step(self, x: Tensor, memory: Memory, slots: Tensor) -> Tensor:
        """The same for the newest decision of each slot, from `memory`, which is then updated."""
        count = memory.count[slots]
        at = count % memory.size
        h = x
        m = len(MEMORY_TOKENS)
        ok = torch.arange(memory.size, device=x.device)[None, :] < count.clamp(max=self.cfg.memory)[:, None]
        ok, mpos = ok.repeat_interleave(m, dim=1), memory.pos[slots].repeat_interleave(m, dim=1)
        for layer, block in enumerate(self.blocks):
            mem = memory.hist[slots, layer].flatten(1, 2)  # [B, M * 2, d]
            memory.hist[slots, layer, at] = h[:, list(MEMORY_TOKENS)]
            h = block.step(h, count, mem, mpos, ok)
        memory.pos[slots, at] = count
        memory.count[slots] = count + 1
        return self.final(h)

    # Heads -------------------------------------------------------------------------------------------------------
    def _action_vectors(self, pokemon: Tensor, moves: Tensor, bench: Tensor, move_net, switch_net) -> Tensor:
        """[N, d] acting Pokemon, [N, 4, d] its move slots, [N, K, d] its bench -> [N, 8 + K, d]:
        4 moves, the same 4 with Mega, K switches."""
        p = pokemon[:, None].expand(-1, L.MOVE_SLOTS, -1)
        zero = moves.new_zeros(*moves.shape[:2], 1)
        plain = move_net(torch.cat([p, moves, zero], dim=-1))
        mega = move_net(torch.cat([p, moves, zero + 1], dim=-1))
        return torch.cat([plain, mega, switch_net(bench)], dim=1)

    def _believed_slots(self, b: dict, token: Tensor, moves: Tensor, prior: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Their active's empty move slots named by the belief head: the most likely moves it has not shown, in
        that order. -> (slot vectors [N, 4, d], opponent prior [N, 13], move ids [N, 4]). A named slot's prior is
        how likely that move is among the hidden ones (times what the Battle gave the first empty slot, and its
        Mega share); a slot that cannot be named (unknown species) stays as the Battle gave it."""
        ids = b["row_ids"][:, L.OPP_ACTIVE].long()
        known = ids[:, self.i_moves]  # seen moves first, then 0
        logits = self.belief_moves(token.detach()).masked_fill(~self.learnset[ids[:, self.i_species]], decide.NEG_INF)
        logits = logits.scatter(1, known, decide.NEG_INF)  # a seen move is not a hidden one (also closes id 0)
        can = (logits > decide.NEG_INF).any(-1, keepdim=True)
        top_p, top_id = torch.where(can, logits, torch.zeros_like(logits)).softmax(-1).topk(L.MOVE_SLOTS, -1)
        empty = known == 0
        rank = (empty.cumsum(-1) - 1).clamp_min(0)  # the first empty slot takes the most likely move
        cand, cand_p = top_id.gather(1, rank), top_p.gather(1, rank)
        use = empty & can & (cand > 0)
        feats = moves.new_zeros(*known.shape, self.slot_stride)
        vec = self.slot_net(torch.cat([feats, self.move(cand)], dim=-1)) + self.slot_pos + self.empty_slot
        first = empty.float().argmax(-1, keepdim=True)  # the first empty slot: where the Battle put "not yet seen"
        base, mega = prior.gather(1, first), prior.gather(1, first + L.MOVE_SLOTS)
        share = (cand_p * empty.sum(-1, keepdim=True)).clamp(0.02, 1.0)  # chance the move is one of the hidden ones
        new = prior.clone()
        new[:, : L.MOVE_SLOTS] = torch.where(use, base * share, prior[:, : L.MOVE_SLOTS])
        new[:, L.MOVE_SLOTS : 2 * L.MOVE_SLOTS] = torch.where(use, mega * share, prior[:, L.MOVE_SLOTS : 2 * L.MOVE_SLOTS])
        return torch.where(use[..., None], vec, moves), new, torch.where(use, cand, known)

    def forward(self, b: dict, seq: Tensor | None = None, memory: Memory | None = None, slots: Tensor | None = None,
                beliefs: bool = True) -> dict:  # fmt: skip
        """Reads only the keys in INPUTS. Every output is per decision; which ones apply depends on the request kind.
        History: `seq` [G, T] lists the batch rows of each game, oldest first (training); or `memory` with one slot per
        row (step-by-step play). With neither, every row is a game's first decision. `beliefs` False leaves out the
        training-only belief heads."""
        x, moves = self.embed(b)
        if memory is not None:
            x = self._trunk_step(x, memory, slots)
        else:
            x = self._trunk(x, torch.arange(len(x), device=x.device)[:, None] if seq is None else seq)
        if self.cfg.move_tokens:
            moves = x[:, TOKENS:].reshape(len(x), L.ROWS, L.MOVE_SLOTS, self.cfg.d)
        tokens = x[:, :TOKENS]
        pokemon = tokens[:, 1 : 1 + L.ROWS]
        n = tokens.shape[0]
        state = tokens[:, 0]  # the field token: it has attended to every token in every block, and is what is remembered

        # our active: whichever of rows 0-2 is marked (row 0 when none is: a preview, or all three fainted)
        active = b["rows"][:, : L.OWN_PICKS, self.is_active_col].argmax(-1)
        at = torch.arange(n, device=tokens.device)
        u_own = self._action_vectors(
            pokemon[at, active], moves[at, active], pokemon[:, : L.OWN_PICKS], self.own_move, self.own_switch
        )  # [N, 11, d]
        their_moves, opp_prior, named = moves[:, L.OPP_ACTIVE], b["opp_prior"].float(), None
        if self.cfg.belief_slots:
            their_moves, opp_prior, named = self._believed_slots(b, pokemon[:, L.OPP_ACTIVE], their_moves, opp_prior)
        u_opp = self._action_vectors(
            pokemon[:, L.OPP_ACTIVE], their_moves, pokemon[:, L.OPP_BENCH], self.opp_move, self.opp_switch
        )  # [N, 13, d]

        opp_logits = (self.opp_query(state)[:, None] * u_opp).sum(-1) * self.cfg.d**-0.5
        p_opp, opp_logp, prior_all_zero = decide.opp_distribution(opp_logits, opp_prior, self.cfg.max_prob)

        a, bb = self.a_proj(u_own), self.b_proj(u_opp)
        matrix = (
            self.base(state)[:, :, None]
            + self.a_bias(u_own)
            + self.b_bias(u_opp).transpose(1, 2)
            + (a @ bb.transpose(1, 2)) * self.cfg.rank**-0.5
        )  # [N, 11, 13]
        score = decide.expected_score(matrix, p_opp.detach() if self.cfg.detach_opp else p_opp)
        tau = self.temperature()
        pi, legal, no_legal = decide.policy(score, b["mask"], tau, self.cfg.max_prob)

        at_preview = b["preview_mask"].bool().any(-1)  # the 60 picks are scored only where a pick is being asked for
        preview = tokens.new_zeros(n, L.PREVIEW_ACTIONS)
        if at_preview.any():
            c, team = self.combos, pokemon[at_preview]
            triple = self.lead_proj(team[:, c[:, 0]]) + self.back_proj(team[:, c[:, 1]] + team[:, c[:, 2]])
            preview[at_preview] = self.preview_out(F.gelu(triple + self.preview_ctx(state[at_preview])[:, None]))[..., 0]
        preview_pi, preview_legal, _ = decide.policy(preview, b["preview_mask"], 1.0, self.cfg.max_prob)

        out = {
            "opp_logits": opp_logits,
            "opp_logp": opp_logp,
            "p_opp": p_opp,
            "prior_all_zero": prior_all_zero,
            "matrix": matrix,
            "score": score,
            "pi": pi,
            "legal": legal,
            "no_legal": no_legal,
            "value": self.value(state)[:, 0],
            "preview_pi": preview_pi,
            "preview_legal": preview_legal,
        }
        if named is not None:
            out["opp_named"] = named  # [N, 4] move ids of their active's slots, seen or believed; 0 = none
            out["opp_seen"] = b["row_ids"][:, L.OPP_ACTIVE].long()[:, self.i_moves] != 0  # [N, 4] the seen ones
        if beliefs:
            theirs = pokemon[:, L.OPP_ROWS]
            species = b["row_ids"][:, L.OPP_ROWS, self.i_species].long()
            belief_moves = self.belief_moves(theirs).masked_fill(~self.learnset[species], decide.NEG_INF)
            belief_moves[..., 0] = torch.where(self.learnset[species].any(-1), belief_moves[..., 0], 0.0)  # unknown species
            out.update(
                belief_brought=self.belief_brought(theirs)[..., 0],
                belief_lead=self.belief_lead(theirs)[..., 0],
                belief_item=self.belief_item(theirs),
                belief_ability=self.belief_ability(theirs),
                belief_moves=belief_moves,
            )
        return out

    # Acting ------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, b: dict, kind: Tensor, greedy: bool = False, generator: torch.Generator | None = None,
            memory: Memory | None = None, slots: Tensor | None = None) -> dict:  # fmt: skip
        """One decision per row of the batch. `action` is the environment's index: 0-10 for a turn or replace request,
        11-70 for a preview request. Calling again with a smaller mask (a refused choice) gives the best of the rest.
        `memory` and `slots` carry each game's earlier turns; every call adds this decision to them."""
        out = self.forward(b, memory=memory, slots=slots, beliefs=False)
        turn, turn_logp = decide.choose(out["pi"], out["legal"], greedy, generator)
        prev, prev_logp = decide.choose(out["preview_pi"], out["preview_legal"], greedy, generator)
        is_preview = kind.to(turn.device) == L.KIND_PREVIEW
        out["action"] = torch.where(is_preview, prev + L.PREVIEW_OFFSET, turn)
        out["logp"] = torch.where(is_preview, prev_logp, turn_logp)
        return out

    def temperature(self) -> Tensor:
        return self.log_tau.clamp(LOG_TAU_MIN, LOG_TAU_MAX).exp()

    def describe(self) -> dict:
        return {"config": asdict(self.cfg), "parameters": sum(p.numel() for p in self.parameters())}


def inputs(batch: dict, device: torch.device | str | None = None) -> dict:
    """Only the model's inputs from a batch that may also hold labels, as tensors."""
    out = {}
    for k in INPUTS:
        v = batch[k]
        v = v if isinstance(v, Tensor) else torch.from_numpy(v)
        out[k] = v.to(device) if device is not None else v
    return out

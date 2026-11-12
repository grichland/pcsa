# From research/pcsa-rl/rl2/belief_features.py (sha256 60dacfe21d34, 2026-11-12 14:28); keep diffable.
# pcsa changes are marked "pcsa:".
"""Belief feature mode (research/pcsa-belief, stage 2): v2 input columns read from the belief tracker.

Off by default: with feature mode "base" nothing here is imported and former_features is exactly what it was. The
other modes replace, in the same columns (research/pcsa-belief/belief/report.md, "For stage 2"):

  318-323 / 324-329  opposing stat low / high (/400): the belief's 99% interval in the current form (members seen only
                     at team preview: their species' prior)
  148 / 149          surely faster / surely slower: P(their speed > ours) >= 0.999 or P(their speed >= ours) <= 0.001
                     (Trick Room still swaps the two, as today)
  150 / 151          log2 speed ratios against the belief's 0.5% / 99.5% speed quantiles (no x1.5 unknown-item factor)
  move blocks 158 + 40j, +31..+39   damage both ways from the belief: mean = posterior mean roll x accuracy x hits,
                     top = 99% high x hits (no ITEM_UNKNOWN_UPPER); 152 / 153 follow (they read the same hits)
  their active is taken as it will be in the coming turn: rows holding its Mega Stone in the Mega form (speed too),
  an Aegislash attacking in Blade Forme

Mode "belief+mega" adds the columns of our Mega actions (they share the move blocks with the plain moves): while our
active can Mega Evolve this turn, the four boosted columns (+36..+39: boosted mean, boosted top, boosted KO mean,
boosted KO top; otherwise 1.5 x mean / top, redundant) of our active's move blocks hold its Mega form's mean, top and
KO flags against their active, and those of their active's move blocks the same against our Mega form.

Not added: unrevealed-move slots (their own gate). pcsa: the tracker is pcsa.belief.fast (ported from
research/pcsa-belief/belief); live, critical hits are not in the protocol (crit_reported=False) and a listed build's
empty item or ability is unknown (empty_unknown=True); research's behaviour is crit_reported=True,
empty_unknown=False (the equivalence tests).
"""

from __future__ import annotations

import math
from typing import Any

from poke_env.battle import Field

from pcsa.rl_training import features_moves as fm
from pcsa.rl_training import features_rows as fr

MODES = ("base", "belief", "belief+mega")
SURE = 0.001  # P(faster) at or below / at or above 1 - SURE: "surely"
SCARF_FLOOR = (
    0.15  # the hidden Choice Scarf prior floor chosen in stage 2 (research/pcsa-belief/belief/stage2-prep.md)
)
_STATE: dict[Any, Any] = {}


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"unknown feature mode {mode!r} (one of {', '.join(MODES)})")
    return mode


def _fast():
    """pcsa.belief.fast (pcsa: imported only in the belief modes)."""
    if "fast" not in _STATE:
        from pcsa.belief import fast

        _STATE["fast"] = fast
    return _STATE["fast"]


def new_battle_belief(crit_reported: bool = True, empty_unknown: bool = False):
    """One battle's belief (pcsa.belief.fast.BattleBelief). The catalog (with the ladder's seen builds, read once per
    process), the species spaces and the preview-only priors are shared by every battle of the process (pcsa: one
    set per `empty_unknown`, which changes the spaces)."""
    fast = _fast()
    state = _STATE.setdefault(("spaces", bool(empty_unknown)), {})
    if "spaces" not in state:
        from pcsa.battle.builds import Catalog

        if "catalog" not in _STATE:
            _STATE["catalog"] = Catalog.load(seen=True)
        priors = fast.FastPriors(scarf_floor=SCARF_FLOOR, empty_unknown=bool(empty_unknown))
        state["spaces"] = fast.FastSpaces(_STATE["catalog"], priors)
        state["prior"] = {}
    bb = fast.BattleBelief(_STATE["catalog"], spaces=state["spaces"], crit_reported=crit_reported)
    bb._prior = state["prior"]  # prior members never change: one per species and process
    return bb


def _their_side(battle, row) -> bool:
    """Whether this opposing row is their active as the coming turn will see it (Mega, Stance Change)."""
    return row.key is not None and row.mon is battle.opponent_active_pokemon and not row.mon.fainted


# Columns --------------------------------------------------------------------------------------------------------
def hits(ctx, att, dfn, our_form: str | None = None) -> list:
    """rl.features_moves.hits from the belief: per move (effectiveness /4, mean x accuracy x hits, 99% top x hits).
    Effectiveness and accuracy as today (current types, revealed abilities)."""
    b, bel = ctx.battle, ctx.belief
    if att.ours == dfn.ours:
        return []
    ours, theirs = (att, dfn) if att.ours else (dfn, att)
    weather = frozenset(w.name.lower() for w in b.weather)
    stage = max(-6, min(6, att.mon.boosts.get("accuracy", 0) - dfn.mon.boosts.get("evasion", 0)))
    types = (dfn.mon.type_1, dfn.mon.type_2)
    # the abilities today's accuracy reads: ours as held, theirs as revealed (Row.resolve: = the fighters' abilities)
    fa_ability, fd_ability = att.ability, dfn.ability
    coming = _their_side(b, theirs)
    out = []
    for move in att.moves[:4]:
        static = fm._static(move.id) if move is not None else None
        if static is None:
            out.append(fm.NO_HIT)
            continue
        move_type, n_hits, base_accuracy = static
        r = bel.damage(b, ours.mon, theirs.mon, move.id, not att.ours, coming=coming, our_form=our_form)  # pyright: ignore[reportOptionalMemberAccess]
        if not r or not r.get("hi", 0.0) > 0:
            out.append(fm.NO_HIT)
            continue
        accuracy = fm._accuracy(move.id, base_accuracy, weather, stage, fa_ability, fd_ability)  # pyright: ignore[reportOptionalMemberAccess]
        mean, top = r["point"] * accuracy * n_hits, r["hi"] * n_hits
        out.append((min(1.0, fm._effectiveness(move_type, types) / fm.EFFECT_DIV), fm._finite(mean), fm._finite(top)))
    return out


def _speed(ctx, row, opp, f) -> None:
    """Columns 148-151 of `row` against `opp` (one ours, one theirs), as rl.features_rows.matchup lays them out."""
    b = ctx.battle
    if opp is None or opp.mon is None or opp.mon.fainted or row.mon.fainted or row.ours == opp.ours:
        return
    ours, theirs = (row, opp) if row.ours else (opp, row)
    our_speed = ctx.speed(ours)[0]  # ours is exact (speed_bounds gives (s, s))
    sp = ctx.belief.speed(b, theirs.mon, our_speed, coming=_their_side(b, theirs))
    if sp is None:
        return
    p_gt, p_ge = sp["faster"], sp["faster"] + sp["tie"]
    if row.ours:
        faster, slower = p_ge <= SURE, p_gt >= 1 - SURE
    else:
        faster, slower = p_gt >= 1 - SURE, p_ge <= SURE
    trick_room = Field.TRICK_ROOM in b.fields
    f[148], f[149] = (float(slower), float(faster)) if trick_room else (float(faster), float(slower))
    lo, hi = float(sp["lo"]), float(sp["hi"])
    if row.ours:
        f[150], f[151] = fr._ratio(our_speed, lo), fr._ratio(our_speed, hi)
    else:
        mid = math.sqrt(max(lo, 1e-6) * max(hi, 1e-6))
        f[150] = f[151] = fr._ratio(mid, our_speed)


def _stats(ctx, row, e) -> None:
    """EXTRA f[0:12] (columns 318-329) of an opposing row: the belief's 99% stat interval / 400."""
    got = ctx.belief.stats(ctx.battle, row.mon)
    if got is None:
        return
    lo, hi = got
    e[0:6] = [min(2.0, float(v) / fr.STAT_DIV) for v in lo]
    e[6:12] = [min(2.0, float(v) / fr.STAT_DIV) for v in hi]


def _mega(ctx, row, opp, f) -> None:
    """belief+mega: the boosted damage columns (+36..+39) of our active's and their active's move blocks hold the
    damage with our active in the Mega form it can take this turn."""
    b = ctx.battle
    if opp is None or row.mon is None or opp.mon is None or row.ours == opp.ours:
        return
    ours, theirs = (row, opp) if row.ours else (opp, row)
    if not ours.active or ours.mon.fainted or not _their_side(b, theirs):
        return
    form = _fast().our_mega_form(b, ours.mon)
    if form is None:
        return
    mega_hits = hits(ctx, row, opp, our_form=form)
    hp = opp.hp
    for j, move in enumerate(row.moves[:4]):
        if move is None or j >= len(mega_hits):
            continue
        cols = fm.damage_columns(*mega_hits[j], hp)  # effect, mean, top, KO mean, KO top, ...  # pyright: ignore[reportCallIssue]
        at = 158 + j * fm.MOVE_WIDTH + fm.FACTS + fm.STATE
        f[at + 5 : at + 9] = cols[1:5]


def patch(ctx, row, opp, f, e) -> None:
    """Overwrite one encoded row (`f`: the 318 S1 floats, `e`: EXTRA) with the belief's columns."""
    if row.mon is None:
        return
    _speed(ctx, row, opp, f)
    if not row.ours:
        _stats(ctx, row, e)
    if ctx.feature_mode == "belief+mega":
        _mega(ctx, row, opp, f)

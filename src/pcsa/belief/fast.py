# From research/pcsa-belief/belief/fast.py (sha256 dcd849036429, 2026-11-12 14:38), split into keys, grids, board,
# member_state, member_grids, member_out, member, fast (pcsa's 400-line file limit); pcsa changes are marked "pcsa:".
"""Stage 2: stage 1's belief tracker (belief/tracker.py), made cheap enough for training rollouts.

Same model (rows, priors, evidence, likelihoods, outputs; README.md), other bookkeeping:

- The free component (every spread x item x ability) is a dense array [items, abilities, spreads] of log-weights
  instead of ~110k rows; the listed components (list / spread / item) stay rows (a few thousand). Certain reveals
  shrink the item and ability axes; a used move multiplies the free part by one number.
- Every calculation runs on a grid (item class x ability class x spread). Items and abilities that a calculation
  cannot tell apart share a class: the calculator (pcsa.battle.damage, belief.calc, pcsa.players.opponent.speed)
  reads an item or ability only through the tables named in the *_key functions below, so the classes are exact
  (tests/test_fast.py compares every row with stage 1). The scalar parts of a calculation run once per class pair,
  the stat arithmetic once per (class pair, spread) in one numpy pass.
- Protocol lines are read once (belief.hits.Reader: hits, turn orders, Mega lines). The board of the moment comes from
  one private Battle advanced line by line, with the decision's request applied at each decision (IncBoard), instead
  of replaying the protocol prefix for every hit and turn order. A board_at callback (stage 1's interface) still
  works.
- Posterior weights, their class aggregates and the outputs are cached per member version.

Additions over stage 1 (stage-2 brief):
- the Mega form in the coming turn: `coming=True` on their active takes the rows holding one of its Mega Stones in
  that Mega form (types, base stats, ability, the stone, the weather a Drought / Drizzle / Sand Stream / Snow
  Warning Mega sets), the others in the current form, and an Aegislash's damaging moves in Blade Forme (Stance
  Change); `our_form=<Mega form>` takes our member in that Mega form (exact stats from the solved points, the Mega's
  ability);
- a floor on the hidden Choice Scarf prior (FastPriors.scarf_floor), for the item prior of the free and item
  components.
"""

from __future__ import annotations

import collections

from pcsa.battle import game
from pcsa.battle.builds import Catalog
from pcsa.battle.damage import Fighter
from pcsa.battle.seen_builds import species_key
from pcsa.battle.stats import form_of
from pcsa.belief import calc as K
from pcsa.belief.board import FastTracker, IncBoard
from pcsa.belief.grids import (
    Grid,
    our_mega_form,
    own_fighter,
)
from pcsa.belief.keys import (
    COMPONENTS,
    FastPriors,
    FastSpaces,
    FastSpecies,
    Interval,
    wquantile,
    wquantiles,
)
from pcsa.belief.member import FastMember
from pcsa.belief.space import STATS
from pcsa.players import opponent as opp

__all__ = [
    "FastPriors",
    "FastSpaces",
    "FastSpecies",
    "Interval",
    "wquantile",
    "wquantiles",
    "own_fighter",
    "our_mega_form",
    "Grid",
    "FastMember",
    "IncBoard",
    "FastTracker",
    "BattleBelief",
    "COMPONENTS",
]


# ------------------------------------------------------------------------------------------------ per battle
class BattleBelief:
    """One battle's belief, as the network's features read it: update from the decision's protocol and request,
    then per member the 99% stat interval, speed against one of ours, and damage ranges both ways (cached until
    the next update). Preview-only members (seen at team preview, not yet in battle) get their species' prior."""

    MASS = 0.99

    def __init__(
        self,
        catalog: Catalog | None = None,
        spaces: FastSpaces | None = None,
        *,
        crit_reported: bool = True,
        fixes: K.Fixes = K.ALL_FIXES,
        tag: str = "belief",
    ) -> None:
        self.tracker = FastTracker(catalog, None, fixes, crit_reported, spaces, tag)
        self._args = (catalog, fixes, crit_reported, spaces, tag)  # pcsa: for a rebuild
        self._lines: list[str] = []  # pcsa: the protocol lines read so far
        self.rebuilds = 0
        self._requests: dict[int, dict] = {}  # pcsa: the first request seen at each turn (for a rebuild)
        self._prior: dict[str, FastMember] = {}
        self._cache: dict = {}  # this decision (cleared by update)
        self._across: dict = {}  # damage ranges by value key, across decisions
        self._species: dict[str, str] = {}
        self.updates = 0
        self.counts: collections.Counter = collections.Counter()

    def update(self, lines: list[str], end: int | None = None, request: dict | None = None) -> None:
        """Read the decision's protocol (lines[:end]) and request. pcsa: the tracker reads only the lines it has not
        read; when this protocol does not extend the one read before (live: a battle log re-read differently, 169
        of 2,798 recorded decisions in stage 1), the tracker starts again (`_rebuild`)."""
        end = len(lines) if end is None else end
        n = len(self._lines)
        for m in self._prior.values():  # pcsa: memos keyed by id(battle); live builds a Battle per decision
            m._ctx_memo.clear()
        if n and (end < n or lines[:n] != self._lines):
            self._rebuild(lines, end, request)
        else:
            self.tracker.observe(lines, end, request=request)
        if request:
            self._requests.setdefault(_turn_of(lines, end), request)
        self._lines = list(lines[:end])
        self._cache.clear()
        self.updates += 1

    def _rebuild(self, lines: list[str], end: int, request: dict | None) -> None:
        """pcsa: a new tracker reads the whole protocol as the decisions would have: up to each turn's start, then
        that turn's request (the first one seen at that turn, else the latest earlier one, else this decision's),
        so the boards of earlier hits have our exact stats as in the incremental flow."""
        _, fixes, crit_reported, _, tag = self._args
        self.tracker = FastTracker(self.tracker.catalog, None, fixes, crit_reported, self.tracker.spaces, tag)
        self._across.clear()  # keyed by member version, which starts again
        self.rebuilds += 1
        self.counts["rebuilds"] += 1
        known = sorted(self._requests.items())
        for i in range(end):
            if lines[i].startswith("|turn|"):
                turn = _turn_of(lines, i + 1)
                earlier = [r for t, r in known if t <= turn]
                self.tracker.observe(lines, i + 1, request=earlier[-1] if earlier else request)
        self.tracker.observe(lines, end, request=request)

    def update_from(self, battle) -> None:
        """From a decision's Battle (p1 view: a live p1 battle or a replayed, mirrored p2 one)."""
        lines = ["|".join(p) for p in battle._replay_data]
        self.update(lines, len(lines), battle.last_request)

    def member_of(self, battle, mon) -> FastMember | None:
        key = next((k for k, m in battle.opponent_team.items() if m is mon and ": " in k), None)
        if key is not None:
            m = self.tracker.member(key)
            if m is not None:
                return m
        sp = self._species.get(mon.species)
        if sp is None:
            sp = self._species[mon.species] = species_key(
                game.base_name(mon.species) if game.is_mega(mon.species) else mon.species, self.tracker.catalog
            )
        m = self._prior.get(sp)
        if m is None:
            m = self._prior[sp] = FastMember(
                "prior:" + sp, sp, self.tracker.spaces.get(sp), self.tracker.priors, self.tracker.fixes
            )
        return m

    def stats(self, battle, mon) -> tuple[list[float], list[float]] | None:
        m = self.member_of(battle, mon)
        if m is None:
            return None
        iv = m.stat_intervals(battle, mon, self.MASS)
        return [iv[s].lo for s in STATS], [iv[s].hi for s in STATS]

    def speed(self, battle, mon, our_speed: float, coming: bool = False) -> dict | None:
        m = self.member_of(battle, mon)
        if m is None:
            return None
        key = ("spe", id(mon), round(float(our_speed), 4), coming)
        if key not in self._cache:
            self._cache[key] = m.speed_vs(battle, mon, our_speed, coming=coming, mass=self.MASS)
        return self._cache[key]

    def damage(
        self,
        battle,
        our_mon,
        their_mon,
        move_id: str,
        theirs_attacking: bool,
        *,
        coming: bool = False,
        our_form: str | None = None,
    ) -> dict | None:
        m = self.member_of(battle, their_mon)
        if m is None:
            return None
        key = ("dmg", id(our_mon), id(their_mon), move_id, theirs_attacking, coming, our_form)
        if key not in self._cache:
            own = self.own(battle, our_mon, our_form)
            vkey = self._value_key(battle, m, our_mon, their_mon, own, move_id, theirs_attacking, coming, our_form)
            hit = self._across.get(vkey)
            if hit is None:
                if len(self._across) > 4000:
                    self._across.clear()
                hit = self._across[vkey] = m.damage_range(
                    battle,
                    our_mon,
                    their_mon,
                    move_id,
                    theirs_attacking,
                    (0.99,),
                    coming=coming,
                    our_form=our_form,
                    own=own,
                    ko=False,
                )
                self.counts["damage_computed"] += 1
            else:
                self.counts["damage_reused"] += 1
            self._cache[key] = hit
        return self._cache[key]

    def _value_key(self, battle, m, our_mon, their_mon, own: Fighter, move_id, theirs_attacking, coming, our_form):
        """Everything a damage range reads, by value: the member's posterior version, our Fighter, their visible
        state, the conditions on the defender's side, the HP thresholds, the Mega state. The same key across
        decisions (and across a p2 seat's per-decision Battles) gives the same range."""
        frac = their_mon.current_hp_fraction if their_mon.current_hp is not None else 1.0
        our_frac = (our_mon.current_hp or 0) / max(1, our_mon.max_hp or 1)
        ctx = m._context(battle, their_mon, coming)
        where = opp.conditions(battle, our_mon if theirs_attacking else their_mon)
        return (
            m.key,
            m.version,
            m.vocab,
            move_id,
            theirs_attacking,
            coming,
            our_form,
            tuple(own.stats.items()),
            own.types,
            own.ability,
            own.item,
            tuple(sorted(own.boosts.items())),
            own.status,
            form_of(our_mon, battle)[0],
            our_mon.species,
            ctx[0],
            ctx[1],
            tuple(sorted(ctx[2].items())),
            ctx[3],
            tuple(sorted(dict(their_mon.boosts).items())),
            opp._status(their_mon),
            opp._types(their_mon),
            their_mon.species,
            frac >= 1.0,
            frac * 3 <= 1.0,
            our_frac >= 1.0,
            our_frac * 3 <= 1.0,
            where.weather,
            where.terrain,
            where.defender_screens,
            our_frac > 0,
            frac > 0,
        )

    def own(self, battle, mon, form: str | None = None) -> Fighter:
        """Our member's Fighter (fast.own_fighter), once per decision."""
        key = ("own", id(battle), id(mon), form)
        if key not in self._cache:
            self._cache[key] = own_fighter(battle, mon, form)
        return self._cache[key]

    def bench_outputs(self, battle) -> int:
        """What the belief features need at a decision (stage2/bench.py): stats per opposing member, speed and
        damage both ways for their active against each of ours and each of theirs against our active."""
        n = 0
        active, ours = battle.opponent_active_pokemon, battle.active_pokemon
        for key, mon in battle.opponent_team.items():
            if ": " not in key:
                continue
            self.stats(battle, mon)
            n += 1
            if ours is not None and not ours.fainted and not mon.fainted:
                is_active = mon is active
                self.speed(battle, mon, opp.speed(battle, ours, opp.own_fighter(battle, ours)), coming=is_active)
                for mv in opp.revealed_moves(mon)[:4]:
                    self.damage(battle, ours, mon, mv, True, coming=is_active)
                    n += 1
        if active is not None and not active.fainted and any(m is active for m in battle.opponent_team.values()):
            for mine in battle.team.values():
                if mine.fainted:
                    continue
                if mine is not ours:
                    self.speed(battle, active, opp.speed(battle, mine, opp.own_fighter(battle, mine)), coming=True)
                for mv in list(mine.moves)[:4]:
                    self.damage(battle, mine, active, mv, False, coming=True)
                    n += 1
        return n


def _turn_of(lines: list[str], end: int) -> int:
    """pcsa: the turn of lines[:end] (its last |turn| line; 0 before the first)."""
    for i in range(end - 1, -1, -1):
        if lines[i].startswith("|turn|"):
            try:
                return int(lines[i].split("|")[2])
            except (IndexError, ValueError):
                return 0
    return 0

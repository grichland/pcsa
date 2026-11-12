# From research/pcsa-belief/belief/fast.py (sha256 dcd849036429, 2026-11-12 14:38), split into keys, grids, board,
# member_state, member_grids, member_out, member, fast (pcsa's 400-line file limit); pcsa changes are marked "pcsa:".
"""The incremental board and the tracker (overview: pcsa.belief.fast)."""

from __future__ import annotations

import collections

from poke_env.battle import Field
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.builds import Catalog
from pcsa.battle.moves import base_data
from pcsa.battle.seen_builds import species_key
from pcsa.belief import calc as K
from pcsa.belief import evidence as E
from pcsa.belief import hits as H
from pcsa.belief.keys import (
    HP_POWER,
    SKIP_POWER,
    SPEED_POWER,
    FastPriors,
    FastSpaces,
)
from pcsa.belief.member import FastMember
from pcsa.belief.space import Priors
from pcsa.players import opponent as opp


# ------------------------------------------------------------------------------------------------ the board
class IncBoard:
    """One p1 Battle advanced line by line (pcsa.battle.state.replay's per-line handling); the decision's request is
    applied at each decision, so our side has its exact stats and items as in a live battle."""

    def __init__(self, tag: str = "belief") -> None:
        from pcsa.battle.state import LOGGER, USERNAME, ChampionsBattle

        self.battle = ChampionsBattle(tag, USERNAME, LOGGER, 9)
        self.battle.player_role = "p1"
        self.at = 0
        self.ended: set[str] = set()
        self.errors = 0
        self._moves: dict[tuple[str, str], list] = {}  # (side, name) -> form-changing lines (observed_form's)
        self._forms: dict[int, tuple] = {}  # id(mon) -> (transitions applied, form, species it started from)
        self._request_sig: tuple = ()

    def advance(self, lines: list[str], i: int) -> None:
        b = self.battle
        for k in range(self.at, i):
            line = lines[k]
            if line.startswith("|player|p1|"):
                line = "|player|p1|me||"
            elif line.startswith("|player|p2|"):
                line = "|player|p2|opponent||"
            try:
                parts = line.split("|")
                if len(parts) > 2 and parts[1] == "-fieldstart":
                    self.ended.discard(parts[2])
                elif len(parts) > 2 and parts[1] == "-fieldend":
                    if parts[2] in self.ended:
                        continue
                    self.ended.add(parts[2])
                if len(parts) > 5 and parts[1] == "move" and "[notarget]" in parts[5:]:
                    parts[4] = ""
                b.parse_message(parts)
                self._form_line(parts)
            except (KeyError, IndexError, ValueError, AttributeError):
                self.errors += 1
        self.at = max(self.at, i)

    def _form_line(self, parts: list[str]) -> None:
        if len(parts) > 4 and parts[1] == "-mega":
            who = parts[2].split(": ", 1)
            if len(who) == 2 and who[0][2:] == "a":
                self._moves.setdefault((who[0][:2], who[1]), []).append(parts)
        elif len(parts) > 3 and parts[1] in ("switch", "drag", "replace", "detailschange", "-formechange"):
            who = parts[2].split(": ", 1)
            if len(who) == 2:
                self._moves.setdefault((who[0][:2], who[1]), []).append(parts)

    def prime_forms(self) -> None:
        """pcsa.battle.moves.observed_form for every member, computed from the lines kept per member and put in
        observed_form's own cache for the current protocol length (it would rescan the whole protocol: the length
        changes at every line here). Same transitions, same order, same result."""
        from pcsa.battle.game import mega_form_for_stone, species_entry
        from pcsa.battle.moves import reconcile_form

        b = self.battle
        cache = b.__dict__.setdefault("_pcsa_forms", {})  # pyright: ignore[reportAttributeAccessIssue]
        n = len(b._replay_data)
        for team in (b.team, b.opponent_team):
            for ident, mon in team.items():
                if ": " not in ident:
                    continue
                side, name = ident.split(": ", 1)
                trans = self._moves.get((side, name), [])
                done, form, start = self._forms.get(id(mon), (0, mon.species, mon.species))
                if start != mon.species:  # observed_form starts from the current species: start again
                    done, form, start = 0, mon.species, mon.species
                for parts in trans[done:]:
                    if parts[1] == "-mega":
                        if parts[2].split(": ", 1) == [side + "a", name]:
                            resolved = mega_form_for_stone(parts[3], parts[4])
                            if resolved:
                                form = to_id_str(resolved)
                        continue
                    candidate = to_id_str(parts[3].split(",", 1)[0])
                    current = species_entry(form)
                    persistent = "Mega" in current.get("forme", "") or form == "mimikyubusted"
                    same_base = to_id_str(current.get("baseSpecies") or form) == to_id_str(
                        species_entry(candidate).get("baseSpecies") or candidate
                    )
                    if parts[1] in ("switch", "drag") and persistent and same_base:
                        continue
                    form = reconcile_form(form, candidate)
                self._forms[id(mon)] = (len(trans), form, start)
                cache[(id(mon), n)] = form

    def request(self, request: dict) -> None:
        """Apply the decision's request: our members' exact stats, items and abilities. Skipped when those are what
        the last applied request said (HP, status, stages, switches and items used are in the protocol)."""
        from pcsa.battle.state import normalize_fainted_request

        side = (request or {}).get("side") or {}
        sig = tuple(
            (
                p.get("ident"),
                p.get("details"),
                tuple(sorted((p.get("stats") or {}).items())),
                p.get("item"),
                p.get("ability"),
                p.get("baseAbility"),
            )
            for p in side.get("pokemon") or []
        )
        if sig and sig == self._request_sig:
            return
        self._request_sig = sig
        try:
            self.battle.parse_request(request)
            normalize_fainted_request(self.battle)
        except Exception:  # noqa: BLE001 - a bad request leaves the protocol-only board
            self.errors += 1


# ------------------------------------------------------------------------------------------------ the tracker
class FastTracker:
    """tracker.BeliefTracker with FastMember members. `observe(lines, end, board_at=None, request=None)`: with
    board_at, stage 1's interface (the Battle just before line i); without, the private incremental board, then the
    request (the decision's) applied to it."""

    def __init__(
        self,
        catalog: Catalog | None = None,
        priors: Priors | None = None,
        fixes: K.Fixes = K.ALL_FIXES,
        crit_reported: bool = True,
        spaces: FastSpaces | None = None,
        tag: str = "belief",
    ) -> None:
        self.catalog = catalog or Catalog.load(seen=True)
        self.priors = priors or FastPriors()
        self.spaces = spaces or FastSpaces(self.catalog, self.priors)
        self.priors = self.spaces.priors
        self.fixes = fixes
        self.crit_reported = crit_reported
        self.members: dict[str, FastMember] = {}
        self.team_items: set[str] = set()
        self._read = 0
        self._hits_done: set[int] = set()
        self._orders_done: set[int] = set()
        self.stats: collections.Counter = collections.Counter()
        self.reader = H.Reader(opp.EARLY_MOVERS)
        self.board = IncBoard(tag)
        self._n_hits = 0

    def member(self, key: str) -> FastMember | None:
        return self.members.get(key)

    def _new(self, key: str, species: str) -> FastMember:
        sp = species_key(game.base_name(species) if game.is_mega(species) else species, self.catalog)
        m = FastMember(key, sp, self.spaces.get(sp), self.priors, self.fixes)
        for item in self.team_items:
            m.forbid_item(item)
        self.members[key] = m
        return m

    def observe(self, lines: list[str], end: int, board_at=None, request: dict | None = None) -> None:
        if end > self._read:
            self._reveals(lines, end)
            self.reader.feed(lines, end)
            events = []
            for h in self.reader.hits[self._n_hits :]:
                if h.line < end and h.line not in self._hits_done:
                    events.append((h.line, 0, h))
            self._n_hits = len(self.reader.hits)
            for o in self.reader.orders_until():
                if o["turn"] in self._orders_done or o["second_line"] >= end:
                    continue
                events.append((o["first_line"], 1, o))
            events.sort(key=lambda e: (e[0], e[1]))
            at = board_at or self._board_at(lines)
            for _, kind, ev in events:
                if kind == 0:
                    self._hits_done.add(ev.line)
                    self._hit(ev, at)
                else:
                    self._orders_done.add(ev["turn"])
                    self._order(ev, at)
            self._read = end
        if board_at is None:
            self.board.advance(lines, end)
            if request:
                self.board.request(request)

    def _board_at(self, lines):
        def at(i: int):
            self.board.advance(lines, i)
            self.board.prime_forms()
            return self.board.battle

        return at

    def _reveals(self, lines: list[str], end: int) -> None:
        for r in E.reveals(lines, self._read, end):
            if r.kind == "switch":
                if r.key not in self.members:
                    self._new(r.key, r.value)
                continue
            m = self.members.get(r.key)
            if m is None:
                continue
            if r.kind == "move":
                m.reveal_move(r.value)
            elif r.kind == "spoil":
                m.spoiled = True
            elif r.kind == "ability":
                m.reveal_ability(r.value)
            elif r.kind in ("item", "mega"):
                had = m.item_shown
                if r.kind == "mega":
                    m.reveal_mega(r.value)
                else:
                    m.reveal_item(r.value)
                if m.item_shown and had is None and not m.spoiled:
                    self.team_items.add(m.item_shown)
                    for other in self.members.values():
                        if other is not m:
                            other.forbid_item(m.item_shown)

    def _hit(self, h, board_at) -> None:
        theirs_attacking = h.side == "p2"
        key = h.attacker if theirs_attacking else h.target
        m = self.members.get(key)
        st = self.stats
        if m is None:
            return
        if h.crit and self.crit_reported:
            st["hit_skip_crit"] += 1
            return
        if not H.clean(h):
            st["hit_skip_unclean"] += 1
            return
        if h.move in SKIP_POWER or h.move in SPEED_POWER or h.move in HP_POWER:
            st["hit_skip_power"] += 1
            return
        if not float(base_data(h.move).get("power") or 0) > 0:
            return
        if theirs_attacking and any(t == h.turn and line < h.line for line, t, _ in self.reader.megas):
            st["hit_skip_mega_turn"] += 1
            return
        b = board_at(h.line)
        their = b.opponent_team.get(key)
        ours = b.team.get(h.target if theirs_attacking else h.attacker)
        if their is None or ours is None or not ours.max_hp:
            st["hit_skip_board"] += 1
            return
        own_ability = to_id_str(ours.ability or "")
        if own_ability in (K.UNMODELED_DEFEND if theirs_attacking else K.UNMODELED_ATTACK):
            st["hit_skip_own_ability"] += 1
            return
        d = H.dealt(h)
        if theirs_attacking:
            scale = float(h.scale)
            g = m.damage_grid(b, ours, their, h.move, True, our_full=h.before == h.scale, our_third=False)
        else:
            scale = 100.0
            g = m.damage_grid(b, ours, their, h.move, False, their_full=h.before == h.scale)
        if g is None or g.zero:
            st["hit_skip_no_damage"] += 1
            return
        seen = h.before if h.ko else d
        m.observe_hit(
            g,
            float(seen),  # pyright: ignore[reportArgumentType]
            max(1, h.hits),
            h.ko,
            scale,
            not theirs_attacking,
            crit_possible=not self.crit_reported,
        )
        st["hit_used_" + ("taken" if theirs_attacking else "dealt")] += 1

    def _order(self, o, board_at) -> None:
        st = self.stats
        if o["early"]:
            st["order_skip_early"] += 1
            return
        their_key, our_key = H._key(o["theirs"][1]), H._key(o["ours"][1])
        m = self.members.get(their_key)
        if m is None:
            return
        a, b_ = opp.move_object(o["theirs"][2]), opp.move_object(o["ours"][2])
        if a is None or b_ is None or a.priority != b_.priority:
            st["order_skip_priority"] += 1
            return
        if {o["theirs"][2], o["ours"][2]} & E.ORDER_MOVES:
            st["order_skip_move"] += 1
            return
        b = board_at(o["first_line"])
        if Field.TRICK_ROOM in b.fields:
            st["order_skip_trick_room"] += 1
            return
        mon, mine = b.opponent_team.get(their_key), b.team.get(our_key)
        if mon is None or mine is None:
            return
        if to_id_str(mine.ability or "") in E.ORDER_ABILITIES:
            st["order_skip_own_ability"] += 1
            return
        our_speed = opp.speed(b, mine, opp.own_fighter(b, mine))
        m.observe_order(b, mon, our_speed, o["they_first"])
        st["order_used"] += 1

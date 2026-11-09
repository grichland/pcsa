"""The disciplined player: max_damage's attacks, team preview and forced picks, plus rules from the ladder research.

At each decision the first rule that applies decides (thresholds in config/players/disciplined.yaml):
1. Sweeper denier (R2): the opposing active is boosted or has used a setup move. No voluntary switch unless the
   incoming member takes < `denier_switch_in_max` of its HP from their best hit and can KO them; else, in order:
   a KO now (we move first or survive), Haze/Clear Smog/Encore/Taunt/a phaze/Perish Song, a priority KO, the
   hardest hit.
2. Mega first (R8): when Mega Evolution is offered only the Mega versions of the moves are considered.
3. Futile attack: our best *legal* hit is under `futile_hit` of their max HP (a Choice lock into an immunity, a
   resisted last move): switch to the bench member that hits them hardest, back-to-back or not.
4. Bad trade: threatened, unable to KO, and our best legal hit under `trade_hit_min`: switch to the bench member
   taking the least share of its HP if that is under `switch_in_relaxed`.
5. Disciplined switching (R9): switch only when threatened (their best hit >= `threat_threshold`), unable to KO,
   and a bench member takes < `switch_in_max` of its HP and hits back for >= `switch_in_threatens`; the one
   taking least comes in. best_matchup's back-to-back rule applies to every voluntary switch but the futile one.
6. Recovery (R5): below `recover_below` HP, when the healed HP survives their best hit and we cannot KO them.
7. Phase (R11): ahead in members, no non-attacking move but the denier's; recovery is off.
8. Otherwise max_damage's best attack.
Damage and speed estimates are pcsa.players.threat's on the sets pcsa.players.opponent assumes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import yaml
from poke_env.battle import Effect, Move, Target, Weather
from poke_env.data.normalize import to_id_str

from pcsa.battle.builds import Catalog
from pcsa.common.clock import Deadline
from pcsa.common.errors import ConfigError
from pcsa.players import opponent, threat
from pcsa.players.base import DecisionView, LegalAction, TeamPreviewView
from pcsa.players.best_matchup import BestMatchupMaxDamagePlayer
from pcsa.players.heuristic import HeuristicParams
from pcsa.players.max_damage import MaxDamagePlayer

BOOST_STATS = ("atk", "spa", "spe", "def", "spd")
SETUP_WITHOUT_BOOSTS = frozenset({"bellydrum"})  # setup moves whose boosts poke-env does not list
ANTI_SETUP = ("haze", "clearsmog", "encore", "taunt", "roar", "whirlwind", "dragontail", "perishsong")
CLEARS_BOOSTS = frozenset({"haze", "clearsmog", "roar", "whirlwind", "dragontail"})
PHAZES = frozenset({"roar", "whirlwind", "dragontail"})
PERISH = frozenset({Effect.PERISH0, Effect.PERISH1, Effect.PERISH2, Effect.PERISH3})
RECOVERY = frozenset(
    "recover roost slackoff softboiled milkdrink shoreup wish synthesis moonlight morningsun".split()
)  # Rest is not one: it puts the user to sleep
WEATHER_HEALS = frozenset({"synthesis", "moonlight", "morningsun"})
SUN = frozenset({Weather.SUNNYDAY, Weather.DESOLATELAND})
HALF, SUN_HEAL, WEATHER_HEAL = 0.5, 2 / 3, 0.25
MIN_HP = 1e-3
PHAZE_TARGETS = 2  # a phaze needs an opposing member left besides the active


@dataclass
class DisciplinedParams:
    denier_switch_in_max: float = 0.25  # vs a boosted foe, a switch-in may take at most this share of its HP...
    threat_threshold: float = 0.4  # their best hit (share of our max HP) that counts as a threat
    switch_in_max: float = 0.25  # a voluntary switch-in takes less than this share of its HP...
    switch_in_threatens: float = 0.3  # ...and hits them for at least this share of their max HP
    recover_below: float = 0.6  # recovery only below this HP fraction
    futile_hit: float = 0.05  # our best legal hit under this share of their max HP: switch regardless
    trade_hit_min: float = 0.15  # threatened and hitting for less than this: a bad trade, switch if...
    switch_in_relaxed: float = 0.6  # ...a bench member takes less than this share of its HP

    @classmethod
    def load(cls, path: Path) -> DisciplinedParams:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(f"unknown disciplined parameters in {path}: {sorted(unknown)}")
        return cls(**data)


def boosted(mon) -> bool:
    """A positive stage on atk/spa/spe/def/spd."""
    return sum(max(0, mon.boosts.get(s, 0)) for s in BOOST_STATS) > 0


def is_setup(move: Move) -> bool:
    return move.id in SETUP_WITHOUT_BOOSTS or (
        move.target == Target.SELF and bool(move.boosts) and sum(move.boosts.values()) > 0
    )


def mega_first(view: DecisionView) -> DecisionView:
    """The view without the plain moves that have a Mega version on offer."""
    megas = {move.id for a in view.legal if (move := _move(a)) is not None and a.order.mega}  # type: ignore[union-attr]
    if not megas:
        return view
    plain = [a for a in view.legal if (move := _move(a)) is not None and not a.order.mega and move.id in megas]  # type: ignore[union-attr]
    return replace(view, legal=tuple(a for a in view.legal if a not in plain))


def heal_fraction(battle, move_id: str) -> float:
    if move_id not in WEATHER_HEALS or not battle.weather:
        return HALF
    return SUN_HEAL if set(battle.weather) & SUN else WEATHER_HEAL


def _move(action: LegalAction) -> Move | None:
    order = action.order.order if action.order is not None else None
    return order if isinstance(order, Move) else None


class DisciplinedPlayer(BestMatchupMaxDamagePlayer):
    name = "disciplined"

    def __init__(
        self,
        rules: DisciplinedParams | None = None,
        catalog: Catalog | None = None,
        heuristic: HeuristicParams | None = None,
    ) -> None:
        super().__init__(heuristic)
        self.rules = rules or DisciplinedParams()
        self.catalog = catalog if catalog is not None else Catalog.load()
        self.tracker = opponent.Tracker(self.catalog)

    def params(self) -> dict:
        return {"stab": self.stab, **asdict(self.rules)}

    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        self.tracker.reset()
        return super().choose_team(view, deadline)

    def choose_action(self, view: DecisionView, deadline: Deadline) -> LegalAction:
        battle = view.battle
        if battle is None or not view.legal:
            return MaxDamagePlayer.choose_action(self, view, deadline)
        turn = int(battle.turn or 0)
        if battle.battle_tag != self._tag or turn < self._turn:
            self._reset(battle.battle_tag)
            self.tracker.reset()
        self._turn = turn
        self.tracker.observe(battle)
        view = mega_first(view)
        chosen = self._decide(view)
        if chosen is None:
            chosen = MaxDamagePlayer.choose_action(self, self._phase_filtered(view), deadline)
        if chosen.order is not None and chosen.order.mega:
            self._notes["mega"] = "first turn offered"
        return chosen

    # Estimates ------------------------------------------------------------------------------------------
    @property
    def _known(self) -> tuple:
        """The catalog and tracker the threat estimates read."""
        return self.catalog, self.tracker

    def _hit(self, battle, attacker, defender) -> tuple[float, str | None]:
        return threat.best_hit(battle, attacker, defender, self.catalog, tracker=self.tracker)

    def _ko(self, battle, attacker, defender) -> bool:
        return threat.can_ko(battle, attacker, defender, self.catalog, tracker=self.tracker)

    def _taken(self, battle, attacker, member) -> float:
        """The share of `member`'s current HP that `attacker`'s best hit takes."""
        return self._hit(battle, attacker, member)[0] / max(member.current_hp_fraction, MIN_HP)

    def _ahead(self, battle) -> bool:
        return _left(battle, battle.team) > _left(battle, battle.opponent_team)

    # Rules ----------------------------------------------------------------------------------------------
    def _decide(self, view: DecisionView) -> LegalAction | None:
        battle = view.battle
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon  # type: ignore[union-attr]
        moves = [a for a in view.legal if _move(a) is not None]
        switches = [a for a in view.legal if a.order is not None and _move(a) is None]
        if not moves or mine is None or theirs is None:
            self._notes["rule"] = "forced"
            return None  # max_damage's pick
        if boosted(theirs) or any(is_setup(m) for m in theirs.moves.values()):
            return self._deny(battle, moves, switches)
        chosen = (
            self._futile_switch(battle, moves, switches)
            or self._trade_switch(battle, moves, switches)
            or self._disciplined_switch(battle, switches)
            or self._recover(battle, moves)
        )
        if chosen is None:
            self._notes["rule"] = "attack"
        return chosen

    def _deny(self, battle, moves: list[LegalAction], switches: list[LegalAction]) -> LegalAction | None:
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        steps = (
            ("switch", lambda: self._denier_switch(battle, switches)),
            ("ko", lambda: self._ko_now(battle, moves)),
            ("anti_setup", lambda: next((a for a in moves if self._denies(battle, _move(a))), None)),
            ("priority_ko", lambda: _by_id(moves, threat.priority_ko(battle, mine, theirs, *self._known))),
        )
        for rule, step in steps:
            chosen = step()
            if chosen is not None:
                self._notes["rule"] = f"denier:{rule}"
                return chosen
        self._notes["rule"] = "denier:hardest_hit"
        return None

    def _denier_switch(self, battle, switches: list[LegalAction]) -> LegalAction | None:
        theirs = battle.opponent_active_pokemon
        taken = {id(a): self._taken(battle, theirs, a.order.order) for a in switches}  # type: ignore[union-attr]
        fits = [
            a
            for a in switches
            if taken[id(a)] < self.rules.denier_switch_in_max and self._ko(battle, a.order.order, theirs)  # type: ignore[union-attr]
        ]
        if not fits:
            return None
        best = min(fits, key=lambda a: taken[id(a)])
        reason = f"boosted foe: {best.order.order.species} takes {taken[id(best)]:.2f} and KOs"  # type: ignore[union-attr]
        return self._take_switch(battle, best, reason)

    def _ko_now(self, battle, moves: list[LegalAction]) -> LegalAction | None:
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        if not self._ko(battle, mine, theirs):
            return None
        first = threat.outspeeds(battle, mine, theirs, self.catalog, tracker=self.tracker)
        if not first and self._ko(battle, theirs, mine):
            return None
        return _by_id(moves, self._hit(battle, mine, theirs)[1])

    def _denies(self, battle, move: Move | None) -> bool:
        """Whether `move` is an anti-setup move that can work on the opposing active now."""
        theirs = battle.opponent_active_pokemon
        if move is None or move.id not in ANTI_SETUP:
            return False
        if move.id in CLEARS_BOOSTS and not boosted(theirs):
            return False
        if move.id in PHAZES and _left(battle, battle.opponent_team) < PHAZE_TARGETS:
            return False
        blocked = {"taunt": {Effect.TAUNT}, "encore": {Effect.ENCORE}, "perishsong": PERISH}.get(move.id, set())
        if blocked & set(theirs.effects):
            return False
        return move.category.name == "STATUS" or theirs.damage_multiplier(move) > 0

    def _legal_hit(self, battle, moves: list[LegalAction]) -> float:
        """Our best hit among the legal moves (a Choice lock or Encore leaves fewer than we know)."""
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        hits = [threat.hit_fraction(battle, mine, theirs, _move(a), *self._known) for a in moves]
        return max(hits, default=0.0)

    def _futile_switch(self, battle, moves: list[LegalAction], switches: list[LegalAction]) -> LegalAction | None:
        if not switches or self._legal_hit(battle, moves) >= self.rules.futile_hit:
            return None
        theirs = battle.opponent_active_pokemon
        best = max(switches, key=lambda a: self._hit(battle, a.order.order, theirs)[0])  # type: ignore[union-attr]
        reason = f"futile: {battle.active_pokemon.species}'s legal moves do nothing -> {best.order.order.species}"  # type: ignore[union-attr]
        return self._take_switch(battle, best, reason, force=True)

    def _trade_switch(self, battle, moves: list[LegalAction], switches: list[LegalAction]) -> LegalAction | None:
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        if not switches or self._legal_hit(battle, moves) >= self.rules.trade_hit_min:
            return None
        threshold = self.rules.threat_threshold
        if not threat.threatened(battle, mine, theirs, self.catalog, tracker=self.tracker, threshold=threshold):
            return None
        taken = {id(a): self._taken(battle, theirs, a.order.order) for a in switches}  # type: ignore[union-attr]
        fits = [a for a in switches if taken[id(a)] < self.rules.switch_in_relaxed]
        if not fits:
            self._notes["switch_blocked"] = f"bad trade: {mine.species}, every bench member takes too much"
            return None
        best = min(fits, key=lambda a: taken[id(a)])
        reason = f"bad trade: {mine.species} -> {best.order.order.species} takes {taken[id(best)]:.2f}"  # type: ignore[union-attr]
        return self._take_switch(battle, best, reason)

    def _disciplined_switch(self, battle, switches: list[LegalAction]) -> LegalAction | None:
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        if not switches:
            return None
        threshold = self.rules.threat_threshold
        if not threat.threatened(battle, mine, theirs, self.catalog, tracker=self.tracker, threshold=threshold):
            return None
        if self._ko(battle, mine, theirs):
            return None
        taken = {id(a): self._taken(battle, theirs, a.order.order) for a in switches}  # type: ignore[union-attr]
        fits = [
            a
            for a in switches
            if taken[id(a)] < self.rules.switch_in_max
            and self._hit(battle, a.order.order, theirs)[0] >= self.rules.switch_in_threatens  # type: ignore[union-attr]
        ]
        if not fits:
            self._notes["switch_blocked"] = f"threatened: {mine.species}, no bench member safe and threatening"
            return None
        best = min(fits, key=lambda a: taken[id(a)])
        reason = f"threatened: {mine.species} -> {best.order.order.species} takes {taken[id(best)]:.2f}"  # type: ignore[union-attr]
        return self._take_switch(battle, best, reason)

    def _take_switch(self, battle, action: LegalAction, reason: str, *, force: bool = False) -> LegalAction | None:
        blocked = None if force else self._blocked(battle)
        if blocked:
            self._notes["switch_blocked"] = f"{reason}; {blocked}"
            return None
        back_to_back = self._last_switch == self._turn - 1
        self._streak = self._streak + 1 if back_to_back else 1
        self._last_switch = self._turn
        self._notes["rule"] = "switch"
        self._notes["switch"] = reason + (" (back-to-back: they switched too)" if back_to_back else "")
        return action

    def _recover(self, battle, moves: list[LegalAction]) -> LegalAction | None:
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        heal = next((a for a in moves if _move(a).id in RECOVERY), None)  # type: ignore[union-attr]
        if heal is None or mine.current_hp_fraction >= self.rules.recover_below:
            return None
        healed = min(1.0, mine.current_hp_fraction + heal_fraction(battle, _move(heal).id))  # type: ignore[union-attr]
        if self._hit(battle, theirs, mine)[0] >= healed or self._ko(battle, mine, theirs):
            return None
        if self._ahead(battle):
            self._notes["recover_blocked"] = "phase: ahead in members, attacks only"
            return None
        self._notes["rule"] = "recover"
        return heal

    def _phase_filtered(self, view: DecisionView) -> DecisionView:
        """Ahead in members: the status moves are dropped while an attack remains."""
        if view.battle is None or not self._ahead(view.battle):
            return view
        attacks = [a for a in view.legal if _move(a) is not None and _move(a).category.name != "STATUS"]  # type: ignore[union-attr]
        if not attacks:
            return view
        return replace(view, legal=tuple(a for a in view.legal if _move(a) is None or a in attacks))


def _by_id(moves: list[LegalAction], move_id: str | None) -> LegalAction | None:
    if not move_id:
        return None
    key = to_id_str(move_id)
    return next((a for a in moves if _move(a).id == key), None)  # type: ignore[union-attr]


def _left(battle, team: dict) -> int:
    """Members a side has left: the team size less its fainted ones."""
    return (battle.max_team_size or 3) - sum(1 for m in team.values() if m.fainted)

"""The endgame player: disciplined until both sides are down to `max_members`, then the 1v1 chain decides.

With at most `max_members` (config/players/endgame.yaml) left on each side, `pcsa.players.chain` plays out the
rest as 1v1 duels and this player takes its first step: stay and use the line's move (its recovery move for
chain.RECOVER), switch to the line's first member, or send it after a faint. Everything else is disciplined's.

Inputs, per member: HP fraction; speed from `opponent.speed` on threat's fighter (negated under Trick Room);
attacks from `threat.hit_fraction` on every known damaging move (ours exact, theirs `opponent.assumed`), with the
accuracy divided back out, and as the KO roll the mean over 0.925 (rolls are 85-100%: threat exposes only the
mean); recovery 1/2 (weather heals as disciplined's `heal_fraction`); known residuals (poison and toxic -1/8,
burn -1/16, Leftovers and Black Sludge, sandstorm, Poison Heal, Magic Guard); Rocky Helmet 1/6 and Rough
Skin / Iron Barbs 1/8 per hit taken. No free switch after a KO (ranked has none).

Their unseen members (Champions brings 3: alive = 3 - fainted, less the revealed alive) are each set of
unrevealed team-preview species in turn, assumed sets at full HP; every option is scored against the worst set
for us (minimax). Our active's attacks are its legal moves; the Mega it may take this turn is not modelled.
A pivot (U-turn's switch request) lets the leaving active be swapped even after a KO.
Falls back to disciplined, with a note, on a tie between the top two legal options, a drawn best line (no win,
margin >= 0: a loss always has a negative margin), a line it cannot play, or an error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from itertools import combinations
from pathlib import Path

import yaml
from poke_env.battle import Field, MoveCategory, Weather

from pcsa.battle import game
from pcsa.battle.builds import Catalog
from pcsa.common.errors import ConfigError
from pcsa.players import opponent, threat
from pcsa.players.base import DecisionView, LegalAction
from pcsa.players.chain import RECOVER, Attack, ChainResult, Combatant, Matchups, chain_options
from pcsa.players.disciplined import RECOVERY, DisciplinedParams, DisciplinedPlayer, _by_id, _move, heal_fraction
from pcsa.players.heuristic import HeuristicParams
from pcsa.players.tactics import effective_accuracy

TEAM_SIZE = 3  # Champions brings three
KO_ROLL = 1 / 0.925  # max roll over the mean roll
POISON, BURN, LEFTOVERS, SAND = 1 / 8, 1 / 16, 1 / 16, 1 / 16
HELMET, BARBS = 1 / 6, 1 / 8
SAND_PROOF_TYPES = frozenset({"rock", "ground", "steel"})
SAND_PROOF = frozenset({"overcoat", "sandveil", "sandrush", "sandforce", "magicguard"})
TIE = 1e-6
WHY_CHARS = 80


@dataclass
class EndgameParams:
    max_members: int = 2  # the chain decides once both sides have at most this many members left

    @classmethod
    def load(cls, path: Path) -> EndgameParams:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(f"unknown endgame parameters in {path}: {sorted(unknown)}")
        return cls(**data)


# Inputs ---------------------------------------------------------------------------------------------------
def residual(battle, fighter) -> float:
    """Known per-turn HP change: status, Leftovers / Black Sludge, sandstorm."""
    status, item, ability = fighter.status, fighter.item, fighter.ability
    change = 0.0
    if status in ("psn", "tox"):
        change += POISON if ability == "poisonheal" else -POISON
    elif status == "brn":
        change -= BURN
    if item == "leftovers" or (item == "blacksludge" and "poison" in fighter.types):
        change += LEFTOVERS
    elif item == "blacksludge":
        change -= POISON
    sand = Weather.SANDSTORM in battle.weather
    if sand and not (SAND_PROOF_TYPES & set(fighter.types) or ability in SAND_PROOF or item == "safetygoggles"):
        change -= SAND
    return max(change, 0.0) if ability == "magicguard" else change


def passive_on_hit(fighter) -> float:
    """What an attacker loses per hit on this member: Rocky Helmet, Rough Skin / Iron Barbs."""
    return (HELMET if fighter.item == "rockyhelmet" else 0.0) + (
        BARBS if fighter.ability in ("roughskin", "ironbarbs") else 0.0
    )


def member_speed(battle, mon, fighter) -> int:
    """Speed for the chain's order (higher moves first): negated under Trick Room."""
    value = round(opponent.speed(battle, mon, fighter))
    return -value if Field.TRICK_ROOM in battle.fields else value


class Inputs:
    """Chain inputs for one decision; members and attacks are built once each. A None member is a fainted
    active (hp 0, no attacks). `pivot`: our active leaving after U-turn and the like, which may be swapped
    out even after a KO."""

    def __init__(self, battle, known: tuple, legal_moves: frozenset[str] | None, pivot=None) -> None:
        self.battle, self.known, self.legal_moves, self.pivot = battle, known, legal_moves, pivot
        self._units: dict[int, Combatant] = {}
        self._attacks: dict[tuple[int, int], tuple[Attack, ...]] = {}

    def moves(self, mon) -> list:
        moves = threat.known_moves(self.battle, mon, *self.known)
        if self.legal_moves is not None and mon is self.battle.active_pokemon:
            moves = [m for m in moves if m.id in self.legal_moves]
        return moves

    def unit(self, mon, hp: float | None = None) -> Combatant:
        if mon is None:
            return Combatant("fainted", 0.0, 0, can_switch_on_faint=False)
        if id(mon) not in self._units:
            battle = self.battle
            fighter = threat.fighters(battle, mon, *self.known)
            heals = [heal_fraction(battle, m.id) for m in self.moves(mon) if m.id in RECOVERY]
            self._units[id(mon)] = Combatant(
                game.base_name(mon.species),
                mon.current_hp_fraction if hp is None else hp,
                member_speed(battle, mon, fighter),
                recover=max(heals, default=0.0),
                residual=residual(battle, fighter),
                passive_on_hit=passive_on_hit(fighter),
                can_switch_on_faint=mon is self.pivot,
            )
        return self._units[id(mon)]

    def attacks(self, attacker, defender) -> tuple[Attack, ...]:
        if attacker is None or defender is None:
            return ()
        key = (id(attacker), id(defender))
        if key not in self._attacks:
            self._attacks[key] = tuple(self._attack_list(attacker, defender))
        return self._attacks[key]

    def _attack_list(self, attacker, defender) -> list[Attack]:
        battle = self.battle
        ability = threat.fighters(battle, attacker, *self.known).ability
        out = []
        for move in self.moves(attacker):
            if move.category == MoveCategory.STATUS:
                continue
            hit = threat.hit_fraction(battle, attacker, defender, move, *self.known)
            accuracy = effective_accuracy(battle, attacker, defender, move, ability)
            if hit <= 0 or accuracy <= 0:
                continue
            mean = hit / accuracy
            out.append(Attack(move.id, mean, move.priority, accuracy, mean * KO_ROLL))
        return out

    def options(self, ours: Sequence, theirs: Sequence, active: tuple[int, int], unseen: tuple = ()) -> dict:
        """chain_options for these members (`unseen`: their members to put at full HP)."""
        fresh = [1.0 if any(m is u for u in unseen) else None for m in theirs]
        units = [self.unit(m) for m in ours], [self.unit(m, hp) for m, hp in zip(theirs, fresh, strict=True)]
        pairs = [(i, j, o, t) for i, o in enumerate(ours) for j, t in enumerate(theirs)]
        matchups = Matchups(
            {(i, j): self.attacks(o, t) for i, j, o, t in pairs}, {(i, j): self.attacks(t, o) for i, j, o, t in pairs}
        )
        return chain_options(*units, matchups, active_ours=active[0], active_theirs=active[1])


# Who is left ----------------------------------------------------------------------------------------------
def _live(mon) -> bool:
    return mon is not None and not mon.fainted and mon.current_hp_fraction > 0


def side_members(team: Sequence, active) -> tuple[list, int]:
    """(alive members, plus None for a fainted active; the active's index)."""
    members = [m for m in team if _live(m)]
    if _live(active):
        return members, members.index(active)
    return [*members, None], len(members)


def our_team(battle) -> list:
    """Our brought members: the request's side when battle.team still holds the preview six."""
    team = list(battle.team.values())
    size = battle.max_team_size or TEAM_SIZE
    side = ((battle.last_request or {}).get("side") or {}).get("pokemon") or []
    if len(team) > size and side:
        idents = {p.get("ident", "").replace("a: ", ": ", 1) for p in side}
        team = [m for key, m in battle.team.items() if key in idents]
    return team


def unseen_candidates(battle) -> list:
    """Preview members not yet revealed (candidates for their one unseen member)."""
    revealed = {game.base_name(m.species) for m in battle.opponent_team.values()}
    return [m for m in battle.teampreview_opponent_team if game.base_name(m.species) not in revealed]


def score(result: ChainResult) -> tuple[bool, float]:
    return result.wins, result.margin


def build(config, catalog: Catalog, heuristic: HeuristicParams) -> EndgamePlayer:
    """The player with config/players/endgame.yaml and disciplined.yaml (defaults when missing)."""
    policy = config.agent.policy
    endgame_path, rules_path = config.directory / policy.endgame, config.directory / policy.disciplined
    endgame = EndgameParams.load(endgame_path) if endgame_path.is_file() else EndgameParams()
    rules = DisciplinedParams.load(rules_path) if rules_path.is_file() else DisciplinedParams()
    return EndgamePlayer(endgame, rules, catalog, heuristic)


class EndgamePlayer(DisciplinedPlayer):
    name = "endgame"

    def __init__(
        self,
        endgame: EndgameParams | None = None,
        rules: DisciplinedParams | None = None,
        catalog: Catalog | None = None,
        heuristic: HeuristicParams | None = None,
    ) -> None:
        super().__init__(rules, catalog, heuristic)
        self.endgame = endgame or EndgameParams()

    def params(self) -> dict:
        return {**super().params(), **asdict(self.endgame)}

    def _decide(self, view: DecisionView) -> LegalAction | None:
        try:
            if not self._in_endgame(view.battle):
                return super()._decide(view)
            chosen, why = self._chain_pick(view)
        except Exception as error:  # any surprise in the inputs: play disciplined
            chosen, why = None, f"{type(error).__name__}: {error}"
        if chosen is not None:
            return chosen
        fallback = super()._decide(view)
        self._notes.update({"then": self._notes.get("rule"), "rule": "endgame:fallback", "why": why[:WHY_CHARS]})
        return fallback

    def _in_endgame(self, battle) -> bool:
        size = battle.max_team_size or TEAM_SIZE
        ours = sum(_live(m) for m in our_team(battle))
        theirs = size - sum(m.fainted for m in battle.opponent_team.values())
        return 0 < ours <= self.endgame.max_members and 0 < theirs <= self.endgame.max_members

    def _chain_pick(self, view: DecisionView) -> tuple[LegalAction | None, str]:
        battle = view.battle
        if battle is None:
            return None, "no battle"
        moves = [a for a in view.legal if _move(a) is not None]
        legal_ids = frozenset(_move(a).id for a in moves)  # type: ignore[union-attr]
        ours, i = side_members(our_team(battle), battle.active_pokemon)
        inputs = Inputs(battle, self._known, legal_ids if moves else None, None if moves else ours[i])
        theirs, j = side_members(list(battle.opponent_team.values()), battle.opponent_active_pokemon)
        unseen = (battle.max_team_size or TEAM_SIZE) - sum(m.fainted for m in battle.opponent_team.values())
        unseen -= sum(m is not None for m in theirs)
        worst = self._worst(inputs, ours, theirs, (i, j), unseen)
        picks = self._playable(view, moves, ours, i, worst)
        if not picks:
            return None, "no playable line"
        picks.sort(key=lambda p: score(p[1][0]), reverse=True)
        (action, (best, cand)), rest = picks[0], picks[1:]
        if not best.wins and best.margin >= 0:
            return None, "draw"
        if rest and abs(best.margin - rest[0][1][0].margin) < TIE and best.wins == rest[0][1][0].wins:
            return None, "tie"
        self._note(battle, action, (best, cand), ours, forced=not moves or ours[i] is None)
        return action, ""

    def _worst(self, inputs: Inputs, ours: list, theirs: list, active: tuple, unseen: int) -> dict:
        """Our option k -> (its result against the worst set of unseen candidates, that set)."""
        cands = list(combinations(unseen_candidates(inputs.battle), unseen))
        if not cands:
            raise ValueError(f"{unseen} unseen members, too few preview candidates")
        worst: dict[int, tuple[ChainResult, tuple]] = {}
        for cand in cands:
            for k, result in inputs.options(ours, [*theirs, *cand], active, cand).items():
                if k not in worst or score(result) < score(worst[k][0]):
                    worst[k] = (result, cand)
        return worst

    def _playable(self, view: DecisionView, moves: list, ours: list, i: int, worst: dict) -> list:
        """(legal action, (result, candidate)) for each option that maps to a legal action."""
        switches = [a for a in view.legal if a.order is not None and _move(a) is None]
        voluntary = bool(moves) and ours[i] is not None
        blocked = voluntary and self._blocked(view.battle) is not None
        out = []
        for k, pair in worst.items():
            if k == i and ours[i] is not None:
                action = self._stay(moves, pair[0].first_move)
            elif blocked:
                continue
            else:
                action = next((a for a in switches if a.order.order is ours[k]), None)  # type: ignore[union-attr]
            if action is not None:
                out.append((action, pair))
        return out

    def _stay(self, moves: list, first_move: str | None) -> LegalAction | None:
        if first_move == RECOVER:
            return next((a for a in moves if _move(a).id in RECOVERY), None)  # type: ignore[union-attr]
        return _by_id(moves, first_move)

    def _note(self, battle, action: LegalAction, pick: tuple, ours: list, forced: bool) -> None:
        """The notes for the chosen line; a voluntary switch also counts for the back-to-back rule."""
        best, cand = pick
        stay = _move(action) is not None
        line = ">".join(ours[k].species for k in best.order if ours[k] is not None)
        line += f" {best.first_move}" if stay else ""
        if not stay and not forced:
            self._take_switch(battle, action, f"endgame line {line}")
        rule = "stay" if stay else "switch"
        self._notes.update({"rule": f"endgame:{rule}", "wins": best.wins, "margin": round(best.margin, 2)})
        self._notes["line"] = line + (f" (worst unseen: {'+'.join(m.species for m in cand)})" if cand else "")
        if forced:
            self._notes["forced"] = True

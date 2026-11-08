"""The weather-style player: free_turn, plus bringing and playing a weather trio (research R14, Cackleford967).

Preview: `styles.fit_mons("weather", our six)`; no fit -> exactly free_turn (note {"style": "no fit"}). A fit
brings its setter and two abusers (a setter of another weather only when no other abuser is left), the setter
leading; a setter in styles.yaml `back` (Politoed) goes second
behind the abuser with the best summed heuristic matchup against their six. A setter of another weather in their
preview is noted ({"weather_clash": ...}); our setter still leads (or Politoed stays back) so ours is up first.
In battle, after disciplined's sweeper denier and before its switching (thresholds in style_weather.yaml):
1. weather:reset: our weather not up, the setter alive on the bench, the active not the setter and threatened
   -> the setter comes in (best_matchup's back-to-back rule applies);
2. weather:to_abuser: weather up, the setter in front, unable to KO and threatened -> the bench abuser whose
   share of HP taken is least, when below `pivot_max_hit`;
3. weather up and an abuser in front: no voluntary switch; after the free turn and recovery, weather:boosted
   picks the best weather move (its STAB in the weather, or one of the weather's moves) when its expected hit is
   within `boosted_move_slack` of the best;
4. Solar Beam / Solar Blade only in sun, unless nothing else is left.
Forced switches: the setter while our weather is down, else (weather up) the abuser hitting their active hardest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import yaml
from poke_env.battle import Move, Weather
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.builds import Catalog
from pcsa.common.clock import Deadline
from pcsa.common.errors import ConfigError
from pcsa.players import styles, threat
from pcsa.players.base import DecisionView, LegalAction, TeamPreviewView
from pcsa.players.disciplined import DisciplinedParams, _move, boosted, is_setup
from pcsa.players.free_turn import FreeTurnParams, FreeTurnPlayer
from pcsa.players.heuristic import HeuristicParams

STYLE = "weather"
STATES = {
    "sun": frozenset({Weather.SUNNYDAY, Weather.DESOLATELAND}),
    "rain": frozenset({Weather.RAINDANCE, Weather.PRIMORDIALSEA}),
    "sand": frozenset({Weather.SANDSTORM}),
    "snow": frozenset({Weather.SNOWSCAPE, Weather.HAIL}),
}
SOLAR = frozenset({"solarbeam", "solarblade"})


@dataclass
class WeatherParams:
    pivot_max_hit: float = 0.5  # weather:to_abuser only to an abuser taking less than this share of its HP
    boosted_move_slack: float = 0.10  # a weather move within this share of the best expected hit is preferred

    @classmethod
    def load(cls, path: Path) -> WeatherParams:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(f"unknown style_weather parameters in {path}: {sorted(unknown)}")
        return cls(**data)


@dataclass(frozen=True)
class Plan:
    """The weather trio of one battle; species are base ids."""

    tag: str
    weather: styles.Weather
    setter: str
    abusers: frozenset[str]


def base(mon) -> str:
    return game.base_name(to_id_str(mon.species))


def weather_up(battle, name: str) -> bool:
    return bool(set(battle.weather or {}) & STATES.get(name, frozenset()))


def their_weathers(rules: styles.Styles, opponents) -> set[str]:
    """Weathers a member of their preview can set (any of its possible abilities)."""
    found = set()
    for mon in opponents:
        abilities = {to_id_str(a) for a in (mon.possible_abilities or [])}
        found |= {name for name, w in rules.weathers.items() if abilities & w.setters}
    return found


def weather_move(move: Move, mon, weather: styles.Weather) -> bool:
    """One of the weather's moves, or a move of the weather's type that is also the user's own type."""
    if move.id in weather.moves:
        return True
    move_type = to_id_str(move.type.name)
    return move_type in weather.stab and move_type in {to_id_str(t.name) for t in mon.types if t is not None}


class StyleWeatherPlayer(FreeTurnPlayer):
    name = "style_weather"

    def __init__(
        self,
        rules: DisciplinedParams | None = None,
        free: FreeTurnParams | None = None,
        weather: WeatherParams | None = None,
        catalog: Catalog | None = None,
        heuristic: HeuristicParams | None = None,
    ) -> None:
        self.weather = weather or WeatherParams()
        self.styles = styles.load()
        self._plan: Plan | None = None
        super().__init__(rules, free, catalog, heuristic)

    def params(self) -> dict:
        return {**super().params(), **asdict(self.weather)}

    # Team preview ---------------------------------------------------------------------------------------
    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        picks = super().choose_team(view, deadline)
        self._plan = None
        battle = view.battle
        if battle is None:
            return picks
        team = list(battle.team.values())
        fit = self.styles.fit_mons(STYLE, team)
        if fit is None or fit.weather is None:
            self._notes["style"] = "no fit"
            return picks
        weather = self.styles.weathers[fit.weather]
        setter, abusers = fit.roles["setter"][0], self._abusers(fit, team)
        order = [setter, *abusers]
        opponents = list(battle.teampreview_opponent_team or [])
        if base(team[setter]) in self.styles.style(STYLE).back:
            score = {i: sum(self.heuristic.matchup(team[i], o, battle) for o in opponents) for i in abusers}
            lead = max(abusers, key=score.__getitem__)
            order = [lead, setter, *(i for i in abusers if i != lead)]
        clash = their_weathers(self.styles, opponents) - {fit.weather}
        if clash:
            self._notes["weather_clash"] = ",".join(sorted(clash))
        self._plan = Plan(battle.battle_tag, weather, base(team[setter]), frozenset(base(team[i]) for i in abusers))
        self._notes["style"] = f"weather:{fit.weather}"
        return order[: battle.max_team_size or 3]

    def _abusers(self, fit: styles.Fit, team: list) -> list[int]:
        """The fit's abuser count from its candidates, a setter of another weather (it would replace ours) last."""
        others = [w for name, w in self.styles.weathers.items() if name != fit.weather]
        pool = [i for i in fit.candidates["abuser"] if i not in fit.roles["setter"]]
        rival = {i for i in pool if any(w.set_by(styles.facts(styles.Member.of_mon(team[i]))) for w in others)}
        return sorted(pool, key=rival.__contains__)[: len(fit.roles["abuser"])]

    def _plan_for(self, battle) -> Plan | None:
        plan = self._plan
        return plan if battle is not None and plan is not None and plan.tag == battle.battle_tag else None

    # Battle ---------------------------------------------------------------------------------------------
    def _decide(self, view: DecisionView) -> LegalAction | None:
        battle = view.battle
        plan = self._plan_for(battle)
        if plan is None:
            return super()._decide(view)
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon  # type: ignore[union-attr]
        moves = [a for a in view.legal if _move(a) is not None]
        switches = [a for a in view.legal if a.order is not None and _move(a) is None]
        if not moves or mine is None or theirs is None:
            return self._forced(battle, plan, switches)
        if boosted(theirs) or any(is_setup(m) for m in theirs.moves.values()):
            return self._deny(battle, moves, switches)
        chosen = self._weather_rules(battle, plan, moves, switches)
        if chosen is None:
            self._notes["rule"] = "attack"
        return chosen

    def _weather_rules(self, battle, plan: Plan, moves, switches) -> LegalAction | None:
        mine = battle.active_pokemon
        up = weather_up(battle, plan.weather.name)
        if not up and base(mine) != plan.setter:
            chosen = self._pivot(battle, self._reset_weather(battle, plan, switches), "weather:reset")
            if chosen is not None:
                return chosen
        if up and base(mine) == plan.setter:
            chosen = self._pivot(battle, self._to_abuser(battle, plan, switches), "weather:to_abuser")
            if chosen is not None:
                return chosen
        if up and base(mine) in plan.abusers:
            return self._recover(battle, moves) or self._boosted(battle, plan, moves)
        return self._disciplined_switch(battle, switches) or self._recover(battle, moves)

    def _pivot(self, battle, found: tuple[LegalAction, str] | None, rule: str) -> LegalAction | None:
        if found is None:
            return None
        chosen = self._take_switch(battle, *found)
        if chosen is not None:
            self._notes["rule"] = rule
        return chosen

    def _threatened(self, battle) -> bool:
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        threshold = self.rules.threat_threshold
        return threat.threatened(battle, mine, theirs, self.catalog, tracker=self.tracker, threshold=threshold)

    def _reset_weather(self, battle, plan: Plan, switches) -> tuple[LegalAction, str] | None:
        setter = next((a for a in switches if base(a.order.order) == plan.setter), None)  # type: ignore[union-attr]
        if setter is None or not self._threatened(battle):
            return None
        return setter, f"weather down: {battle.active_pokemon.species} -> {setter.order.order.species}"  # type: ignore[union-attr]

    def _to_abuser(self, battle, plan: Plan, switches) -> tuple[LegalAction, str] | None:
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        abusers = [a for a in switches if base(a.order.order) in plan.abusers]  # type: ignore[union-attr]
        if not abusers or self._ko(battle, mine, theirs) or not self._threatened(battle):
            return None
        taken = {id(a): self._taken(battle, theirs, a.order.order) for a in abusers}  # type: ignore[union-attr]
        best = min(abusers, key=lambda a: taken[id(a)])
        if taken[id(best)] >= self.weather.pivot_max_hit:
            self._notes["switch_blocked"] = f"setter threatened: every abuser takes >= {self.weather.pivot_max_hit}"
            return None
        return best, f"setter out: {mine.species} -> {best.order.order.species} takes {taken[id(best)]:.2f}"  # type: ignore[union-attr]

    def _boosted(self, battle, plan: Plan, moves: list[LegalAction]) -> LegalAction | None:
        """The weather move with the highest expected hit, when within the slack of the best attack."""
        mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        attacks = [a for a in moves if _move(a).category.name != "STATUS"]  # type: ignore[union-attr]
        hit = {
            id(a): threat.hit_fraction(battle, mine, theirs, _move(a), self.catalog, tracker=self.tracker)
            for a in attacks
        }
        boosted_moves = [a for a in attacks if weather_move(_move(a), mine, plan.weather)]  # type: ignore[arg-type]
        if not boosted_moves:
            return None
        best = max(hit.values())
        pick = max(boosted_moves, key=lambda a: hit[id(a)])
        if hit[id(pick)] <= 0 or hit[id(pick)] < best * (1 - self.weather.boosted_move_slack):
            return None
        self._notes["rule"] = "weather:boosted"
        return pick

    def _forced(self, battle, plan: Plan, switches: list[LegalAction]) -> LegalAction | None:
        """The setter while our weather is down, else the abuser hitting hardest; None: max_damage's pick."""
        self._notes["rule"] = "forced"
        theirs = battle.opponent_active_pokemon
        if not switches or theirs is None:
            return None
        if not weather_up(battle, plan.weather.name):
            setter = next((a for a in switches if base(a.order.order) == plan.setter), None)  # type: ignore[union-attr]
            if setter is not None:
                self._notes["rule"] = "weather:forced_setter"
            return setter
        abusers = [a for a in switches if base(a.order.order) in plan.abusers]  # type: ignore[union-attr]
        if not abusers:
            return None
        self._notes["rule"] = "weather:forced_abuser"
        return max(abusers, key=lambda a: self._hit(battle, a.order.order, theirs)[0])  # type: ignore[union-attr]

    def _phase_filtered(self, view: DecisionView) -> DecisionView:
        """Disciplined's filter; with a plan, Solar Beam / Solar Blade also drop out of sun while a move is left."""
        view = super()._phase_filtered(view)
        if self._plan_for(view.battle) is None or weather_up(view.battle, "sun"):
            return view
        solar = [a for a in view.legal if (m := _move(a)) is not None and m.id in SOLAR]
        if not solar or len(solar) == sum(1 for a in view.legal if _move(a) is not None):
            return view
        return replace(view, legal=tuple(a for a in view.legal if a not in solar))


def build(config, catalog: Catalog, heuristic: HeuristicParams) -> StyleWeatherPlayer:
    """The player with disciplined's, free_turn's and the weather thresholds from the agent config's paths."""
    policy = config.agent.policy
    paths = (config.directory / p for p in (policy.disciplined, policy.free_turn, policy.style_weather))
    rules_path, free_path, weather_path = paths
    rules = DisciplinedParams.load(rules_path) if rules_path.is_file() else DisciplinedParams()
    free = FreeTurnParams.load(free_path) if free_path.is_file() else FreeTurnParams()
    weather = WeatherParams.load(weather_path) if weather_path.is_file() else WeatherParams()
    return StyleWeatherPlayer(rules, free, weather, catalog, heuristic)

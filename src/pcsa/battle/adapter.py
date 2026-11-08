"""A poke-env Battle that understands Champions: pinned form abilities, Mega stones, phone HP rows."""

from __future__ import annotations

import copy

from poke_env.battle import Battle, Field
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.game import mega_ability, mega_form_for_stone, species_entry
from pcsa.battle.moves import observed_form
from pcsa.battle.stats import hp_value, solve_points, stat_value


class ChampionsBattle(Battle):
    """poke-env's tracking with Champions form abilities and the phone's HP-row conventions."""

    compatibility_repairs: list[str]
    replay_errors: list[str]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compatibility_repairs = []
        self.replay_errors = []

    def _normalize(self, split_message: list[str]) -> list[str] | None:
        """The message as poke-env should see it, or None to drop it."""
        if len(split_message) > 3 and split_message[1] in ("-item", "-enditem"):
            normalized = game.canonical_item(split_message[3])
            if normalized is None:
                return None
            split_message = [*split_message[:3], normalized, *split_message[4:]]
        if len(split_message) > 3 and split_message[1] in ("-damage", "-heal", "switch", "drag"):
            column = 4 if split_message[1] in ("switch", "drag") else 3
            if len(split_message) > column and ": " in split_message[2]:
                mon = self._mon(split_message[2])
                condition = split_message[column]
                if (
                    mon is not None
                    and mon.status
                    and mon.status.name != "FNT"
                    and len(condition.split()) == 1
                    and "/" in condition
                ):
                    split_message = list(split_message)  # phone HP rows omit the status suffix; absence is not a cure
                    split_message[column] += " " + mon.status.name.lower()
        if (
            len(split_message) > 2
            and split_message[1] == "-fieldend"
            and Field.from_showdown_message(split_message[2]) not in self.fields
        ):
            self.compatibility_repairs.append("Observed field end without a known start: " + split_message[2])
            return None
        return split_message

    def _mon(self, actor: str):
        side, name = actor.split(": ", 1)
        team = self.team if side[:2] == self.player_role else self.opponent_team
        return team.get(f"{side[:2]}: {name}")

    def parse_message(self, split_message: list[str]) -> None:
        normalized = self._normalize(split_message)
        if normalized is None:
            return
        if len(normalized) > 4 and normalized[1] == "-mega":
            form = mega_form_for_stone(normalized[3], normalized[4])
            if form:
                super().parse_message(["", "detailschange", normalized[2], form + ", L50"])
        super().parse_message(normalized)
        if (
            len(normalized) < 4
            or normalized[1] not in ("switch", "drag", "detailschange", "-formechange", "-mega", "-ability")
            or ": " not in normalized[2]
        ):
            return
        mon = self._mon(normalized[2])
        if mon is None or mon.transformed:
            return
        if normalized[1] == "-ability":
            mon.temporary_ability = to_id_str(
                normalized[3]
            )  # poke-env can write the base ability beneath a form override
            return
        ability = mega_ability(species_entry(observed_form(self, mon)))
        if ability:
            mon.forme_change_ability = ability

    def parse_request(self, request: dict) -> None:
        normalized = current_request(self, request)
        super().parse_request(normalized)
        for member in normalized.get("side", {}).get("pokemon", []):
            mon = self.team.get(member["ident"])
            if mon is None or mon.transformed:
                continue
            ability = mega_ability(species_entry(observed_form(self, mon)))
            if ability:
                mon.forme_change_ability = ability


def current_request(battle, request: dict) -> dict:
    """The request with observed forms and abilities applied, without touching the caller's copy."""
    result = copy.deepcopy(request)
    for member in result.get("side", {}).get("pokemon", []):
        mon = battle.team.get(member["ident"])
        if mon is None or mon.transformed:
            continue
        form = observed_form(battle, mon)
        entry = species_entry(form)
        details = member["details"].split(",", 1)
        if entry and to_id_str(details[0]) != form:
            if "Mega" in entry.get("forme", "") and member.get("stats"):
                member["stats"] = _mega_stats(details[0], entry, member["stats"], member["ident"])
            member["details"] = entry.get("name", form) + ("," + details[1] if len(details) > 1 else "")
        base_ability = mega_ability(entry) or mon.base_ability
        if base_ability:
            member["baseAbility"] = base_ability
        ability = mon.temporary_ability or base_ability
        if ability:
            member["ability"] = ability
    return result


def _mega_stats(old_species: str, entry: dict, stats: dict, ident: str) -> dict:
    old_base = species_entry(to_id_str(old_species)).get("baseStats")
    solved = solve_points(old_base, stats) if old_base else None
    if solved is None:
        raise ValueError(f"cannot convert {ident} stats to its Mega form")
    points, nature = solved
    return {
        stat: hp_value(entry["baseStats"][stat], points[stat])
        if stat == "hp"
        else stat_value(entry["baseStats"][stat], points[stat], nature[stat])
        for stat in stats
        if points.get(stat) is not None
    }

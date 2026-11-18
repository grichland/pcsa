"""Read our own team from the lobby's team pages, then return to the same lobby. Also the Player's OwnTeam."""

from __future__ import annotations

import re

from pcsa.battle import game
from pcsa.battle.ids import to_id
from pcsa.battle.names import showdown_species
from pcsa.battle.stats import nature_name, solve_points
from pcsa.battle.teams import OwnMember, Team, export_text, teambuilder_team
from pcsa.common import log
from pcsa.common.clock import Deadline
from pcsa.common.errors import Recoverable
from pcsa.common.types import Observation, Screen
from pcsa.decide.api import OwnTeam
from pcsa.phone import fields as f
from pcsa.phone.adb import Frame
from pcsa.phone.screens import to_reference
from pcsa.phone.team_sheet import RosterMember, TeamReader, TeamSheet
from pcsa.phone.texts import Texts

TEAM_LABEL_REGION = (1400, 450, 1900, 720)
TEAM_HEADER_REGION = (330, 120, 2100, 210)
STATS_TAB_REGION = (1200, 200, 1560, 280)
PAGE_SCREENS = {
    Screen.TEAM_LIST: "list",
    Screen.TEAM_MENU: "menu",
    Screen.TEAM_DETAILS: "details",
    Screen.TEAM_STATS: "stats",
}
CAPTURE_ATTEMPTS = 4
READ_BUDGET_S = 90.0
MEMBER_FIELDS = ("ability", "item", "stats", "points", "nature")
STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def resolve_species(member: RosterMember) -> str:
    """The sprite's form when it is a form of the named species; the card name shows no forms."""
    named = showdown_species(member.name.value or "").species
    if member.form and game.base_name(to_id(member.form)) == game.base_name(to_id(named)):
        return member.form
    return named


def base_stats_of(member: RosterMember) -> dict | None:
    return game.species_entry(to_id(resolve_species(member))).get("baseStats")


def _complete(values: dict) -> dict | None:
    return dict(values) if values and all(v is not None for v in values.values()) else None


def _points(m: RosterMember, stats: dict | None) -> tuple[dict | None, str | None]:
    """The read stat points and nature; unread points are solved from the stats (a spread-less export gets the
    wrong max HP)."""
    read = _complete(m.stats.points) if m.stats else None
    nature = m.stats.nature if m.stats else None
    if read is not None or stats is None:
        return read, nature
    base = base_stats_of(m)
    solved = solve_points(base, stats) if base else None
    if solved is None or any(v is None for v in solved[0].values()):
        return None, nature
    return dict(solved[0]), nature or nature_name(solved[1])


def from_sheet(sheet: TeamSheet) -> Team:
    """The battle's view of the sheet: species resolved, unread fields as None."""
    members = []
    for m in sheet.members:
        stats = m.stats
        complete = _complete(stats.stats) if stats else None
        points, nature = _points(m, complete)
        members.append(
            OwnMember(
                roster_index=m.roster_index,
                name=m.name.value or f"row{m.roster_index + 1}",
                species=resolve_species(m),
                ability=m.ability.value if m.ability.status == "known" else None,
                item=m.item.value if m.item.status == "known" else None,
                moves=tuple(move.value if move.status == "known" else None for move in m.moves),
                stats=complete,
                points=points,
                nature=nature,
            )
        )
    return Team(sheet.team_name, tuple(members))


def own_team(team: Team | None, roster: tuple[str | None, ...] = (), picks: list[int] | None = None) -> OwnTeam:
    """The Player's view of our team. Without a sheet read only the selection screen's names are known."""
    chosen = tuple(picks) if picks else None
    if team is None:
        names = tuple(n or f"row{i + 1}" for i, n in enumerate(roster))
        return OwnTeam("", names, chosen, ("team",))
    unread = [f"{m.name}.{key}" for m in team.members for key in MEMBER_FIELDS if getattr(m, key) is None]
    unread += [f"{m.name}.move{k}" for m in team.members for k, move in enumerate(m.moves) if not move]  # row k
    unread += [f"{m.name}.stats" for m in team.members
               if m.stats and any(not isinstance(m.stats.get(s), int) for s in STATS)]  # fmt: skip
    return OwnTeam(export_text(teambuilder_team(team)), team.names, chosen, tuple(unread))


class TeamReadSession:
    def __init__(self, phone, settle_s: float) -> None:
        self.phone = phone
        self.settle_s = settle_s
        self.reader = TeamReader(phone.reader)

    def _capture(self, expected: set[str]) -> tuple[Observation, Frame, str]:
        for _ in range(CAPTURE_ATTEMPTS):
            self.phone.clock.sleep(self.settle_s)
            obs, frame = self.phone.observe("team-read")
            stage = (
                "lobby"
                if obs.screen == Screen.LOBBY
                else PAGE_SCREENS.get(obs.screen) or f.team_page(Texts(obs.texts))
            )
            if stage in expected:
                return obs, frame, stage
        raise Recoverable(f"team pages did not reach {sorted(expected)}")

    def _tap_label(self, obs: Observation, pattern: str, region, name: str) -> str:
        found = Texts(obs.texts, self.phone.reader.min_confidence).find(pattern, region)
        if len(found) != 1:
            raise Recoverable(f"team page control {name} is missing or ambiguous")
        tap = self.phone.executor.point_at_box(name, found[0])
        self.phone.executor.execute([tap], obs.screen, self.phone.device, self.phone.clock, 0.0, Texts(obs.texts))
        return found[0].text.strip()

    def read(self) -> tuple[TeamSheet, Team]:
        obs, _, _ = self._capture({"lobby"})
        team_name = self._tap_label(obs, r"team\s+\d+", TEAM_LABEL_REGION, "team_selector")
        try:
            obs, _, stage = self._capture({"list", "menu"})
        except Recoverable:  # the lobby can ignore the first tap
            obs, _, _ = self._capture({"lobby"})
            self._tap_label(obs, r"team\s+\d+", TEAM_LABEL_REGION, "team_selector")
            obs, _, stage = self._capture({"list", "menu"})
        if stage == "list":
            for _ in range(2):
                self._tap_label(obs, re.escape(team_name.casefold()), TEAM_HEADER_REGION, "team_header")
                obs, _, stage = self._capture({"list", "menu"})
                if stage == "menu":
                    break
            if stage != "menu":
                raise Recoverable("the selected team's menu did not open")
        self._tap_label(obs, r"view details", f.TEAM_MENU, "view_details")
        obs, frame, _ = self._capture({"details"})
        sheet = self.reader.read_details(to_reference(frame.image, self.phone.profile.viewport), list(obs.texts))
        if sheet is None:
            raise Recoverable("the team details page could not be read")
        self._tap_label(obs, r"stats", STATS_TAB_REGION, "stats_tab")
        obs, frame, _ = self._capture({"stats"})
        with_stats = self.reader.read_stats(
            to_reference(frame.image, self.phone.profile.viewport), list(obs.texts), sheet, base_stats_of
        )
        sheet = with_stats or sheet
        self._back_to_lobby()
        log.event(
            "team_read",
            name=sheet.team_name,
            completeness=round(sheet.completeness, 2),
            issues=list(sheet.issues),
            stats_issues=list(sheet.stats_issues),
        )
        return sheet, from_sheet(sheet)

    def _back_to_lobby(self) -> None:
        self.phone.navigator.reach(Screen.LOBBY, Deadline.after(self.phone.clock, READ_BUDGET_S))

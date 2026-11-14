"""Getting to the lobby from anywhere outside a battle, and backing out of stray surfaces.

A graph of known screens and the one safe tap that moves toward the lobby from each. Every tap
goes through the tap policy. In-battle screens are never navigated away from: the turn pipeline
owns them. An unknown screen is waited on, backed out of with an OCR-confirmed safe control, and
after a bounded number of rounds raises Recoverable (the supervisor restarts the session).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pcsa.common import log
from pcsa.common.clock import Clock, Deadline
from pcsa.common.config import NavigatorConfig
from pcsa.common.errors import Recoverable
from pcsa.common.types import IN_BATTLE, Observation, Screen
from pcsa.live.taps import Executor
from pcsa.phone.ocr import TextBox
from pcsa.phone.texts import Texts

Observe = Callable[[str], tuple[Observation, object]]  # tag -> (observation, frame)


@dataclass(frozen=True)
class Edge:
    tap: str  # region name, or "dismiss" for the OCR-located control in fields["dismiss"]/["close"]/["target"]
    field: str | None = None  # observation field holding an OCR box target for OCR-located taps


# From each screen, the one tap that leads toward the lobby.
GRAPH: dict[Screen, Edge] = {
    Screen.TITLE: Edge("title_continue", "target"),
    Screen.HOME: Edge("home_battle", "target"),
    Screen.BATTLE_MODES: Edge("ranked_battles", "target"),
    Screen.RANKED_FORMATS: Edge("single_battle", "target"),
    Screen.OPTIONS: Edge("continue_battling"),
    Screen.RANK_UP: Edge("dismiss", "close"),
    Screen.RATING_DIALOG: Edge("dismiss", "dismiss"),
    Screen.COMM_ERROR: Edge("dismiss", "dismiss"),
    Screen.DIALOG: Edge("dismiss", "dismiss"),
    Screen.MOVES: Edge("move_back"),
    Screen.LOG: Edge("log_back"),
    Screen.PARTY: Edge("party_back"),
    Screen.SWITCH_POPUP: Edge("cancel", "cancel"),
    Screen.TEAM_MENU: Edge("team_details_back"),
    Screen.TEAM_DETAILS: Edge("team_details_back"),
    Screen.TEAM_STATS: Edge("team_details_back"),
    Screen.TEAM_LIST: Edge("team_details_back"),
}
DISMISS_FROM_ROUND = 2  # the first unknown round only waits; a dismiss control is tapped from the second
WAITING = frozenset(
    {
        Screen.TRANSITION,
        Screen.WAIT,
        Screen.MATCHMAKING,
        Screen.PREP,
        Screen.REWARDS,
        Screen.RESULT,
        Screen.FORFEIT_MSG,
    }
)


class Navigator:
    def __init__(
        self, observe: Observe, executor: Executor, device, clock: Clock, config: NavigatorConfig, poll_s: float
    ) -> None:
        self.observe = observe
        self.executor = executor
        self.device = device
        self.clock = clock
        self.config = config
        self.poll_s = poll_s

    def reach(self, target: Screen, deadline: Deadline) -> Observation:
        """Walk the graph until `target` shows (or a battle screen does: the caller takes over)."""
        taps_on: dict[Screen, int] = {}
        unknown_rounds = 0
        while not deadline.expired():
            obs, _ = self.observe("nav")
            if obs.screen == target:
                return obs
            if obs.screen in WAITING:  # transitions and waits happen between menus too; poll through them
                self.clock.sleep(self.poll_s)
                continue
            if obs.screen in IN_BATTLE:
                return obs
            if obs.screen == Screen.UNKNOWN:
                unknown_rounds += 1
                self.recover_unknown(obs, unknown_rounds)
                continue
            edge = GRAPH.get(obs.screen)
            if edge is None:
                raise Recoverable(f"no navigation edge from {obs.screen}")
            taps_on[obs.screen] = taps_on.get(obs.screen, 0) + 1
            if taps_on[obs.screen] > self.config.max_taps_per_edge:
                raise Recoverable(f"{obs.screen} did not change after {self.config.max_taps_per_edge} taps")
            self.follow(obs, edge)
            self.clock.sleep(self.config.edge_settle_s)
        raise Recoverable(f"could not reach {target} in time")

    def follow(self, obs: Observation, edge: Edge) -> None:
        log.event("navigate", frm=str(obs.screen), tap=edge.tap)
        if edge.field is None:
            tap = self.executor.point(edge.tap)
        else:
            box = _box_from_field(obs, edge.field)
            if box is None:
                log.warning("navigate_no_control", screen=str(obs.screen), field=edge.field)
                self.clock.sleep(self.poll_s)
                return
            tap = self.executor.point_at_box(edge.tap, box)
        self.executor.execute([tap], obs.screen, self.device, self.clock, 0.0, Texts(obs.texts))

    def recover_unknown(self, obs: Observation, rounds: int) -> None:
        """Wait, then tap the single safe control if the screen offers one; give up after the bound."""
        if rounds > self.config.unknown_max_rounds:
            raise Recoverable(f"unknown screen for {rounds} rounds: {obs.reason}")
        log.warning("unknown_screen", round=rounds, reason=obs.reason, frame=obs.frame_id)
        self.clock.sleep(self.config.unknown_wait_s)
        box = _box_from_field(obs, "dismiss")
        if box is not None and rounds >= DISMISS_FROM_ROUND:
            tap = self.executor.point_at_box("dismiss", box)
            self.executor.execute([tap], Screen.UNKNOWN, self.device, self.clock, 0.0, Texts(obs.texts))
            self.clock.sleep(self.config.edge_settle_s)

    def back_out(self, obs: Observation) -> None:
        """One back edge from a stray in-battle surface (moves, log, party, popup)."""
        obs = self.close_move_info(obs)
        edge = GRAPH.get(obs.screen)
        if edge is not None:
            self.follow(obs, edge)
            self.clock.sleep(self.config.edge_settle_s)

    def close_move_info(self, obs: Observation) -> Observation:
        """The moves screen with its Move Info view off. With the view on (13 Nov), the panel covered the screen
        and neither Back nor a move tap took: the turn timed out. The toggle is tapped first, then the panel's
        Close; returns the screen after, or `obs` when the view was not on."""
        for tap in ("move_info", "info_close"):
            if obs.screen != Screen.MOVES or not obs.fields.get("info_open"):
                return obs
            log.event("move_info_close", tap=tap)
            self.executor.execute([self.executor.point(tap)], Screen.MOVES, self.device, self.clock, 0.0)
            self.clock.sleep(self.config.edge_settle_s)
            obs, _ = self.observe("info")
        if obs.screen == Screen.MOVES and obs.fields.get("info_open"):
            log.warning("move_info_still_open")
        return obs


def _box_from_field(obs: Observation, name: str) -> TextBox | None:
    return box_of(obs.fields.get(name), name)


def box_of(value: object, name: str = "dismiss") -> TextBox | None:
    """The OCR box of a located control (a `fields.target` dict), or None when there is none."""
    if not isinstance(value, dict) or "box" not in value:
        return None
    left, top, right, bottom = value["box"]
    return TextBox(value.get("name", name), 1.0, (left, top, right, bottom))

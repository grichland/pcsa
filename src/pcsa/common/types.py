"""Shared value types: screens, flags, observations, actions. No behaviour beyond small derived properties."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Screen(StrEnum):
    # Outside a battle
    TITLE = "title"  # tap-to-start
    HOME = "home"  # main menu with the Battle tile
    BATTLE_MODES = "battle_modes"  # Ranked / Casual / ...
    RANKED_FORMATS = "ranked_formats"  # Single Battle / Double Battle
    LOBBY = "lobby"  # Begin Matchmaking
    TEAM_LIST = "team_list"
    TEAM_MENU = "team_menu"
    TEAM_DETAILS = "team_details"
    TEAM_STATS = "team_stats"
    MATCHMAKING = "matchmaking"
    # Inside a battle
    TRANSITION = "transition"  # black, loading, VS intro, send-out
    TEAM_SELECT = "team_select"
    PREP = "prep"  # Preparing for Battle / Standing By
    ACTION = "action"  # Fight + Pokemon + View Log
    MOVES = "moves"
    LOG = "log"
    PARTY = "party"
    SWITCH_POPUP = "switch_popup"
    FORCED = "forced"  # forced replacement
    WAIT = "wait"  # Communicating / resolution text / animations
    FORFEIT_MSG = "forfeit_msg"
    RESULT = "result"
    REWARDS = "rewards"
    OPTIONS = "options"  # Continue Battling / Quit Battling
    # Interruptions
    RANK_UP = "rank_up"
    RATING_DIALOG = "rating_dialog"
    DIALOG = "dialog"  # any popup with a single safe control
    COMM_ERROR = "comm_error"  # communication interrupted / returned to title
    UNKNOWN = "unknown"


IN_BATTLE = frozenset(
    {
        Screen.TEAM_SELECT,
        Screen.PREP,
        Screen.ACTION,
        Screen.MOVES,
        Screen.LOG,
        Screen.PARTY,
        Screen.SWITCH_POPUP,
        Screen.FORCED,
        Screen.WAIT,
        Screen.TRANSITION,
    }
)


class Flag(StrEnum):
    AUTO_NOTICE = "auto_notice"  # "automatically chosen": a turn timed out
    TIMER_WARN = "timer_warn"
    COMMUNICATING = "communicating"
    COMM_ERROR = "comm_error"
    DATE_CHANGED = "date_changed"


@dataclass(frozen=True)
class MoveInfo:
    slot: int  # 0-3, top to bottom
    name: str | None
    pp: tuple[int, int] | None  # (current, maximum); None when unread
    disabled: bool = False
    focused: bool = False


@dataclass(frozen=True)
class PartyMember:
    row: int  # visible row 0-2, top is active
    name: str | None
    hp: tuple[int, int] | None
    fainted: bool = False


@dataclass(frozen=True)
class Observation:
    screen: Screen
    size: tuple[int, int]
    flags: frozenset[Flag] = frozenset()
    fields: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""  # the rule that matched
    texts: tuple[Any, ...] = ()  # OCR boxes, for audit and text-anchored taps
    frame_id: str = ""

    def record(self) -> dict[str, Any]:
        return {
            "screen": str(self.screen),
            "size": list(self.size),
            "flags": sorted(map(str, self.flags)),
            "fields": jsonable(self.fields),
            "reason": self.reason,
            "frame_id": self.frame_id,
        }


MOVE_SLOTS = 4  # moves per Pokémon


@dataclass(frozen=True)
class SemanticAction:
    """A battle order as the phone sees it: move row k (0-3, with or without Mega) or selected-team slot s (0-2)."""

    kind: str  # "move" | "switch"
    move: int | None = None
    mega: bool = False
    slot: int | None = None

    def label(self) -> str:
        if self.kind == "move":
            return f"{'mega+' if self.mega else ''}move{self.move}"
        return f"switch{self.slot}"


def jsonable(value: Any) -> Any:
    """Plain JSON-compatible data from dataclasses, enums, mappings and sequences."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {k: jsonable(getattr(value, k)) for k in value.__dataclass_fields__}
    if isinstance(value, StrEnum):
        return str(value)
    return value

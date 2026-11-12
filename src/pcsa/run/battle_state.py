"""Per-battle bookkeeping and the run state that survives a session restart."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pcsa.common.types import SemanticAction, jsonable
from pcsa.phone.log_merge import BattleLog

PIVOT_MOVES = frozenset(
    {"U-turn", "Volt Switch", "Flip Turn", "Parting Shot", "Teleport", "Baton Pass", "Shed Tail", "Chilly Reception"}
)
NAME_SIMILARITY = 0.7
ROSTER_SIZE = 6


def same_name(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False

    def clean(s: str) -> str:
        return "".join(c for c in s.lower() if c.isalpha())

    return difflib.SequenceMatcher(None, clean(a), clean(b)).ratio() >= NAME_SIMILARITY


def same_name_in(row: str, name: str | None) -> bool:
    if not name:
        return False
    return any(same_name(word, name) for word in re.findall(r"[A-Za-z][A-Za-z.\-’']+", row))


def reconcile_roster(
    expected: tuple[str, ...], observed: tuple[str | None, ...]
) -> tuple[tuple[str, ...], int] | None:
    """Fill unreadable selection rows from the verified team; None on a readable conflict."""
    if len(expected) != ROSTER_SIZE or len(observed) != ROSTER_SIZE:
        return None
    if any(seen and not same_name(wanted, seen) for wanted, seen in zip(expected, observed, strict=True)):
        return None
    missing = sum(not seen for seen in observed)
    return tuple(seen or wanted for wanted, seen in zip(expected, observed, strict=True)), missing


@dataclass
class BattleRecord:
    index: int
    battle_id: str
    roster: tuple[str | None, ...]
    log: BattleLog
    started_at: str = ""
    adopted: bool = False  # the session started (or restarted) in the middle of this battle
    requested: list[int] | None = None  # roster rows in selected-slot order
    team_submitted: bool = False
    team_repairs: int = 0
    team_deadline: float | None = None
    visible_order: list[int] = field(default_factory=lambda: [0, 1, 2])  # selected slots as the party shows them
    fainted: set[int] = field(default_factory=set)
    mega_used: bool = False
    mega_pending: bool = False
    mega_spent: bool = False  # live/move_peek: our Mega Evolution is over for this battle (log or move list)
    pp_offsets: dict[str, int] = field(default_factory=dict)  # live/move_peek: PP uses the log count missed
    log_doubts: int = 0  # live/move_peek: unusable log rows up to the last decision
    last_action: dict | None = None
    last_turn: int | None = None
    start_partial: bool = False
    action_visits: int = 0
    turns: list[dict] = field(default_factory=list)
    outcome: str | None = None
    outcome_hint: str | None = None
    outcome_source: str | None = None
    observed: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    finished: bool = False
    forced_attempts: int = 0
    forced_choice: dict | None = None
    actions: dict[int, list[str]] = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)
    opponents: list[str] = field(default_factory=list)
    opponent_preview: tuple = ()
    preview_frame: str = ""
    rank_before: dict | None = None
    rank_after: dict | None = None
    degrade_counts: dict[str, int] = field(default_factory=dict)

    def current_turn(self) -> int:
        """The turn being chosen now: the log shows the latest completed turn, except at the first
        action screen, where it already shows Turn 1 with only the send-outs."""
        return (self.last_turn or 0) + (0 if self.start_partial else 1)

    @property
    def names(self) -> tuple[str | None, ...]:
        return tuple(self.roster[i] if i < len(self.roster) else None for i in (self.requested or []))

    @property
    def dir(self) -> str:
        return f"battle-{self.index}"

    def slot_of(self, name: str | None) -> int | None:
        slots = [s for s, n in enumerate(self.names) if same_name(n, name)]
        return slots[0] if len(slots) == 1 else None

    def record_switch(self, slot: int) -> None:
        order = self.visible_order
        i = order.index(slot)
        order[0], order[i] = order[i], order[0]

    def summary(self, seed: int, counts: bool) -> dict:
        return {
            "index": self.index,
            "battle_id": self.battle_id,
            "outcome": self.outcome,
            "outcome_source": self.outcome_source,
            "seed": seed,
            "adopted": self.adopted,
            "rank_before": self.rank_before,
            "rank_after": self.rank_after,
            "roster": self.roster,
            "team_requested": self.requested,
            "team_requested_names": list(self.names),
            "opponent_preview": jsonable(self.opponent_preview),
            "preview_frame": self.preview_frame,
            "team_observed": self.observed,
            "turns": self.turns,
            "coverage": self.log.coverage,
            "log_turns": sorted(self.log.turns),
            "counts_toward_target": counts,
            "degrade": self.degrade_counts,
            "failures": self.failures,
        }


@dataclass
class RunState:
    """What a restarted session must know. Persisted to run/state.json after every change."""

    run_dir: Path
    target: int
    max_total: int
    count_forfeits: bool
    seed: int
    completed: int = 0
    played: int = 0
    outcomes: list[str | None] = field(default_factory=list)
    rank_history: list[dict] = field(default_factory=list)
    restarts: int = 0
    team_record: dict | None = None
    pending_battle: int | None = None  # a battle in progress when the session last stopped

    def done(self) -> bool:
        return self.completed >= self.target or self.played >= self.max_total

    def save(self) -> None:
        data = {k: v for k, v in self.__dict__.items() if k != "run_dir"}
        (self.run_dir / "state.json").write_text(json.dumps(jsonable(data), indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, run_dir: Path) -> RunState | None:
        path = run_dir / "state.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(run_dir=run_dir, **data)


def action_detail(action: SemanticAction, move_name: str | None, switch_name: str | None) -> str:
    if action.kind == "move":
        return (
            f"chose move {(action.move or 0) + 1}"
            + (f" ({move_name})" if move_name else "")
            + (" with Mega Evolution" if action.mega else "")
        )
    return f"chose switch to {switch_name}"

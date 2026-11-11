"""The battle's log files, written while the battle is on and again when it ends.

`battle-N/log.jsonl`, `log.txt` and `coverage.json` (BattleLog.save) and the readable `battle_log.txt`. The turn
flow saves them after every visit to the log that changed it, so the dashboard shows the log of a battle in
progress and a battle that crashes keeps what was read; `finalize` writes the final versions.
"""

from __future__ import annotations

from pcsa.battle.teams import Team
from pcsa.common import log
from pcsa.phone.rank import rank_text
from pcsa.run.battle_state import BattleRecord
from pcsa.run.recorder import Recorder


def standard_header(b: BattleRecord, team: Team | None, seed: int) -> dict:
    picked = ", ".join(
        f"{name} (roster row {row + 1})" for name, row in zip(b.names, b.requested or [], strict=False)
    )
    return {
        "battle_id": b.battle_id,
        "battle_number": b.index,
        "started": b.started_at,
        "own_team": team.name if team else "unknown",
        "own_rank_before": rank_text(b.rank_before),
        "own_rank_after": rank_text(b.rank_after),
        "team_requested": picked or "unknown",
        "lead_observed": b.observed.get("lead") or "unknown",
        "opponents_seen": ", ".join(b.opponents) or "none",
        "opponent_preview": ", ".join(m.name or "?" for m in b.opponent_preview) or "unknown",
        "outcome": b.outcome or "unknown",
        "outcome_source": b.outcome_source or "none",
        "seed": seed,
    }


def write_log(recorder: Recorder, b: BattleRecord, team: Team | None, seed: int) -> None:
    """Every log file of the battle, from what has been collected so far."""
    b.log.save(recorder.dir / b.dir)
    recorder.write_text(f"{b.dir}/battle_log.txt", b.log.standard_text(standard_header(b, team, seed), b.actions))


class LogSaver:
    """Writes the log files after a visit to the log, only when the log is not what was last written."""

    def __init__(self, recorder: Recorder, seed: int) -> None:
        self.rec = recorder
        self.seed = seed
        self._written: int | None = None

    def after_visit(self, b: BattleRecord, team: Team | None) -> bool:
        """True when the files were written. Never raises: a log that cannot be saved does not stop the turn."""
        try:
            sprites = [len(rows) for _, rows in sorted(b.log.row_sprites.items())]
            mark = hash((b.battle_id, b.log.text(), len(b.log.coverage), len(b.log.failures), tuple(sprites)))
            if mark == self._written:
                return False
            write_log(self.rec, b, team, self.seed)
            self._written = mark
            return True
        except Exception as error:
            log.warning("log_save_failed", battle=b.index, reason=f"{type(error).__name__}: {error}")
            return False

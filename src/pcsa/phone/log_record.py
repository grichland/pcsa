"""Writing a battle's master log: log.jsonl (one row per line, with its readings and vote), log.txt, the readable
log, coverage.json (with every unread row and the frames it came from) and ocr_audit.json. Turns read by paging
back carry "back_paged": true on their rows and are listed in coverage.json "back_paged_turns"."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pcsa.phone.log_panel import CONTENT, HEADER, SCROLLBAR, SEPARATORS, TIMER_BOX
from pcsa.phone.log_row_read import BATTLE_TIMER_BOX
from pcsa.phone.log_vote import record, split_label

if TYPE_CHECKING:
    from pcsa.phone.log_merge import BattleLog


def plain_text(turns: dict[int, list[str]]) -> str:
    if not turns:
        return ""
    return "\n\n".join(f"Turn{t}:\n" + "\n".join(turns[t]) for t in sorted(turns)) + "\n"


def back_paged(log: BattleLog) -> list[int]:
    """Turns read by paging back from a later page (their visits are marked "page": "back")."""
    return sorted({v["turn"] for v in log.visits if v.get("page") == "back"})


def row_record(log: BattleLog, turn: int, index: int, paged: frozenset[int] = frozenset()) -> dict:
    notes = log.row_notes.get(turn, [])
    note = notes[index] if index < len(notes) else (None, None, ())
    votes = log.votes.get(turn, [])
    chosen = votes[index] if index < len(votes) else None
    out = {
        "battle_id": log.battle_id,
        "turn": turn,
        "index": index,
        "text": log.turns[turn][index],
        "sprite": note[0],
        "side": note[1],
    }
    if turn in paged:
        out["back_paged"] = True
    if chosen is not None:
        out.update(read=chosen["read"], reason=chosen["reason"], votes=chosen["votes"], snaps=chosen["snaps"])
        out["readings"] = record(list(note[2]))
    return out


def unread_rows(log: BattleLog) -> list[dict]:
    """Rows no frame read with certainty, with the frame and box of every reading."""
    out = []
    for turn in sorted(log.turns):
        for index in range(len(log.turns[turn])):
            row = row_record(log, turn, index)
            if row.get("read") == "unread":
                notes = log.row_notes[turn][index][2]
                frames = [{"frame": r.get("source"), "box": r.get("box"), "text": r["text"]} for r in notes]
                out.append(
                    {"turn": turn, "index": index, "text": row["text"], "reason": row["reason"], "frames": frames}
                )
    return out


def standard_text(log: BattleLog, header: dict, actions: dict[int, list[str]] | None = None) -> str:
    lines = ["# Pokemon Champions battle log (Ranked Singles)", "# format: pcsa-battle-log/1"]
    lines += [f"{key}: {value}" for key, value in header.items()]
    lines.append(f"turns_collected: {', '.join(map(str, sorted(log.turns))) or 'none'}")
    lines.append("")
    turns: list[int] = [*log.turns, *(actions or {})]
    paged = back_paged(log)
    for turn in range(1, (max(turns) if turns else 0) + 1):
        lines.append(f"=== Turn {turn} ===")
        if turn in paged:
            lines.append("# read by paging back (no action screen of ours on this turn)")
        for index, text in enumerate(log.turns.get(turn, [])):
            category, priority, message = split_label(text)
            label = f"{category} {priority} Priority" if priority else category
            lines.append(f"[{label or '-'}] {message}")
            row = row_record(log, turn, index)
            if row.get("read") == "unread":
                lines.append(f"# unread row: {row['reason']}")
        if turn not in log.turns:
            reason = next((n["detail"] for n in log.coverage if n.get("turn") == turn), "not collected")
            lines.append(f"# (no log for this turn: {reason})")
        lines += [f"# controller: {action}" for action in (actions or {}).get(turn, [])]
        lines.append("")
    lines.append("=== Coverage notes ===")
    lines += [f"# {n['kind']}: {n['detail']}" for n in log.coverage] or ["# none"]
    return "\n".join(lines) + "\n"


def save(log: BattleLog, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    paged = back_paged(log)
    with (directory / "log.jsonl").open("w", encoding="utf-8") as stream:
        for turn in sorted(log.turns):
            for index in range(len(log.turns[turn])):
                row = row_record(log, turn, index, frozenset(paged))
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (directory / "log.txt").write_text(plain_text(log.turns), encoding="utf-8")
    coverage = {
        "battle_id": log.battle_id,
        "turns": sorted(log.turns),
        "back_paged_turns": paged,
        "coverage": log.coverage,
        "failures": log.failures,
        "unread_rows": unread_rows(log),
    }
    (directory / "coverage.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    layout = {
        "header": HEADER,
        "content": CONTENT,
        "separators": SEPARATORS,
        "scrollbar": SCROLLBAR,
        "timer_box": TIMER_BOX,
        "battle_timer_box": BATTLE_TIMER_BOX,
    }
    audit = {"battle_id": log.battle_id, "visits": log.visits, "layout": layout}
    (directory / "ocr_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

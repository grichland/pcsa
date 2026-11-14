"""Re-translate every saved live battle with the current log translator, into a separate reconciled tree.

    cd <repo>; PYTHONPATH=src .venv/Scripts/python.exe scripts/reconcile_live.py
        [--runs DIR ...] [--out-root DIR | --out DIR] [--tree-file FILE] [--only RUN[/battle-N] ...]

The live turn flow builds each decision's Battle from the battle log as read so far (live/turn.py `_build`:
battle.state.build over `b.log.history()` and `b.log.sprite_history()`), saves the protocol it built in
`battle-N/decisions/NNNN-<kind>.json`, and keeps the whole log in `log.jsonl` with the merge of every log visit in
`ocr_audit.json` (phone/log_merge.py BattleLog). The fine-tune rebuilds every decision from the record's protocol and
request, so rows the translator of the day dropped are missing from its data. This tool translates each decision
again from exactly the rows the bot had read when it made it, and writes the result to

    <out-root>/<translator commit>[-dirty]/<runs folder tag>/<run>/battle-N/

(a fresh folder each time: `-2`, `-3` ... when the name is taken), with report.md and report.json at the top. The
original run folders are only read.

Which rows a decision saw. An action decision made at log visit V (its record's `visit`) saw the log after every
accepted merge up to V; a forced replacement does not read the log, so it saw the log of the last action decision
before it (no rows of the rest of its turn). A merge only appends to a turn's rows (anchors new / prefix / overlap /
unanchored / contained; `extended`, which rewrites a turn, has never been recorded and makes the battle unbounded),
so the turn's rows after visit V are the first sum(appended of that turn's visits <= V) rows of log.jsonl. A row's
sprite is the combination of its observations over the visits (sprites.combine_observations, which never replaces a
matched observation except by a conflict), so the observation the bot had at V is the stored one when its source
frame is from a visit <= V; otherwise the bot had no matched sprite for that row then, which the translator treats as
no sprite (None here). A record whose bound cannot be checked keeps its original content: the folder's log does not
add up (the visits' appended rows against log.jsonl), the action visits restart (a resumed battle, whose earlier
rows another log held), the record belongs to another battle (a reused folder), the original's unmapped or inferred
rows are not the bounded rows, or the rebuilt protocol's `|turn|` lines differ from the original's.

Consistency checks, each falling back to the original record: our own moves and switches as in the original
protocol; the request still applies to the rebuilt Battle; `pcsa.battle.state.replay(protocol, request)` succeeds with
no replay error the original did not have, at the same turn, offering every recorded legal order the original
offered. Per battle, the regenerated showdown.log must keep the original's |win| line and its faint count per side
(more faints are allowed and reported: the original missed them); when it does not, the whole battle is kept as it
was.

Each reconciled record keeps everything else from the original and gets a `reconciled` block: the status, the bound
(visit and rows per turn), the protocol lines added and removed, and the original unmapped, warnings, inferred and
revealed abilities. Kept records are copied byte for byte. battle.json, battle_id.json and the run's manifest.json,
state.json and own_team.json are copied; showdown.log and battle_final.txt are regenerated from the final rows.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pcsa.common.clock import SystemClock

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [Path(r"C:\Users\geral\Code\pcsa\output\runs")]  # this repo's games only (the old repo's: --runs)
DEFAULT_OUT_ROOT = Path(r"C:\Users\geral\Code\research\pcsa-rl\runs\live-reconciled")
SCHEMA = "pcsa-reconciled/1"
RUN_FILES = ("manifest.json", "state.json", "own_team.json", "own_team.txt")
BATTLE_COPIES = ("battle.json", "battle_id.json")
APPEND_ANCHORS = re.compile(r"^(new|prefix|contained|unanchored|overlap\d+)$")
VISIT_IN_SOURCE = re.compile(r"visit(\d+)-")
TAG_PARTS = ("<tag>", "output", "runs")
MOVE_FIELDS, IDENT_FIELDS = 4, 3  # |move|<ident>|<move>, |switch|<ident>
WINNER = {"win": "me", "opponent_forfeit": "me", "loss": "opponent"}  # live/handlers.py finalize


# Translator state ------------------------------------------------------------------------------------------------
def git(*args: str) -> str:
    out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, encoding="utf-8")
    return out.stdout.strip() if out.returncode == 0 else ""


def translator_state() -> dict:
    """The code this run translates with: commit, the files changed under src/ (dirty), and where pcsa came from."""
    import pcsa

    dirty = [line[3:] for line in git("status", "--porcelain", "--", "src").splitlines() if line.strip()]
    return {
        "repo": str(REPO),
        "commit": git("rev-parse", "--short", "HEAD") or "nogit",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
        "pcsa_imported_from": str(Path(pcsa.__file__).resolve().parent),
    }


def runs_tag(folder: Path) -> str:
    """`.../pcsa/output/runs` -> `pcsa` (rl2.scripts.live_finetune folder_tag)."""
    parts = folder.resolve().parts
    return parts[-3] if folder.name == "runs" and len(parts) >= len(TAG_PARTS) else folder.name


def fresh(path: Path) -> Path:
    if not path.exists():
        return path
    n = 2
    while path.with_name(f"{path.name}-{n}").exists():
        n += 1
    return path.with_name(f"{path.name}-{n}")


# Reading a battle folder -----------------------------------------------------------------------------------------
def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def visit_of(observation: dict | None) -> int | None:
    match = VISIT_IN_SOURCE.search(str((observation or {}).get("source") or ""))
    return int(match[1]) if match else None


def sprite_at(sprite: dict | None, visit: int) -> tuple[dict | None, bool]:
    """(the sprite observation the bot held for this row after log visit `visit`, exact). Not exact when a matched
    observation names no source frame."""
    if not sprite:
        return None, True
    if sprite.get("reason") == "conflicting_sprites":
        first, second = (sprite.get("observations") or [None, None])[:2]
        late, early = visit_of(second), visit_of(first)
        if late is None or early is None:
            return sprite, False
        if late <= visit:
            return sprite, True
        return (first, True) if early <= visit else (None, True)
    seen = visit_of(sprite)
    if seen is None:
        return sprite, sprite.get("status") != "matched"
    return (sprite, True) if seen <= visit else (None, True)


@dataclass
class Log:
    rows: dict[int, list[dict]]  # turn -> log.jsonl rows in index order
    visits: list[dict]  # ocr_audit.json visits: the accepted merges, in order
    problem: str | None = None

    def bounded(self, visit: int) -> tuple[dict[int, tuple[str, ...]], dict[int, tuple], dict[int, int], bool]:
        """(rows_by_turn, sprites_by_turn, row counts, sprites exact) as the BattleLog held them after `visit`."""
        counts: dict[int, int] = {}
        for v in self.visits:
            if v["visit"] <= visit:
                counts[v["turn"]] = counts.get(v["turn"], 0) + int(v["appended"])
        texts, sprites, exact = {}, {}, True
        for turn, n in sorted(counts.items()):
            rows = self.rows.get(turn, [])[:n]
            texts[turn] = tuple(r["text"] for r in rows)
            held = []
            for r in rows:
                sprite, ok = sprite_at(r.get("sprite"), visit)
                held.append(sprite)
                exact = exact and ok
            sprites[turn] = tuple(held)
        return texts, sprites, counts, exact

    def final(self) -> tuple[dict[int, tuple[str, ...]], dict[int, tuple]]:
        turns = {v["turn"] for v in self.visits} | set(self.rows)
        return (
            {t: tuple(r["text"] for r in self.rows.get(t, [])) for t in sorted(turns)},
            {t: tuple(r.get("sprite") for r in self.rows.get(t, [])) for t in sorted(turns)},
        )


def load_log(bdir: Path) -> Log:
    rows: dict[int, list[dict]] = {}
    if not (bdir / "log.jsonl").is_file() or not (bdir / "ocr_audit.json").is_file():
        return Log({}, [], "no log.jsonl or ocr_audit.json")
    for line in (bdir / "log.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.setdefault(int(r["turn"]), []).append(r)
    for turn, found in rows.items():
        found.sort(key=lambda r: r["index"])
        if [r["index"] for r in found] != list(range(len(found))):
            return Log(rows, [], f"turn {turn}: log.jsonl row indexes are not 0..n-1")
    visits = read_json(bdir / "ocr_audit.json").get("visits") or []
    log = Log(rows, visits)
    numbers = [v.get("visit") for v in visits]
    if any(not isinstance(n, int) for n in numbers) or numbers != sorted(set(numbers)):
        log.problem = "ocr_audit visits are not numbered in increasing order"
    elif bad := [v["anchor"] for v in visits if not APPEND_ANCHORS.match(str(v.get("anchor")))]:
        log.problem = f"a merge that rewrites stored rows (anchor {bad[0]})"
    else:
        added = Counter()
        for v in visits:
            added[v["turn"]] += int(v["appended"])
        if wrong := [t for t in set(added) | set(rows) if added[t] != len(rows.get(t, []))]:
            t = min(wrong)
            log.problem = f"turn {t}: visits append {added[t]} rows, log.jsonl has {len(rows.get(t, []))}"
    return log


def preview_of(summary: dict):
    from pcsa.phone.sprites import SlotMatch

    return tuple(
        SlotMatch(
            int(s["slot"]),
            tuple(s.get("box") or (0, 0, 0, 0)),
            tuple((str(n), float(p)) for n, p in s.get("top") or ()),
            tuple((str(n), float(p)) for n, p in s.get("species") or ()),
        )
        for s in summary.get("opponent_preview") or ()
    )


def team_of(run: Path):
    from pcsa.battle.teams import team_from_record

    state = run / "state.json"
    if state.is_file() and (record := read_json(state).get("team_record")):
        return team_from_record(record)
    own = run / "own_team.json"
    if own.is_file() and (record := read_json(own).get("team")):
        return team_from_record(record)
    return None


# Protocol comparisons --------------------------------------------------------------------------------------------
def own_actions(lines: list[str]) -> list[tuple[str, ...]]:
    out = []
    for line in lines or ():
        p = line.split("|")
        if len(p) >= MOVE_FIELDS and p[1] == "move" and p[2].startswith("p1"):
            out.append(("move", p[2], p[3]))
        elif len(p) >= IDENT_FIELDS and p[1] in ("switch", "drag") and p[2].startswith("p1"):
            out.append((p[1], p[2]))
    return out


def own_actions_added(protocol: list[str], out, unmapped_before) -> tuple[list[tuple[str, ...]], str | None]:
    """(our own moves and switches the rebuilt protocol adds, why the change is refused or None).

    The original's own actions must all be there, in order. An added one must come from a row the original left
    unmapped and the rebuild maps (a "Go! X!" whose icon was misread as Ditto): a row of the original's unmapped list,
    no longer unmapped, whose text names the member (and the move, for a move)."""
    old, new = own_actions(protocol), own_actions(out.lines)
    if old == new:
        return [], None
    added, i = [], 0
    for action in new:
        if i < len(old) and action == old[i]:
            i += 1
        else:
            added.append(action)
    if i < len(old):
        return added, f"check: our own action {old[i]} of the original is not in the rebuilt protocol"
    still = {u.get("row") for u in out.unmapped}
    fixed = [letters(u.get("text")) for u in unmapped_before if isinstance(u, dict) and u.get("row") not in still]
    for action in added:
        wanted = [letters(action[1].split(": ", 1)[-1])] + ([letters(action[2])] if action[0] == "move" else [])
        if not any(all(w in text for w in wanted) for text in fixed):
            return added, f"check: the rebuilt protocol adds own action {action} from no newly mapped row"
    return added, None


def letters(text) -> str:
    return re.sub(r"[^a-z]", "", str(text or "").lower())


def turn_lines(lines: list[str]) -> list[str]:
    return [x for x in lines or () if x.startswith("|turn|")]


def faints(lines: list[str]) -> tuple[int, int]:
    own = theirs = 0
    for line in lines:
        p = line.split("|")
        if len(p) >= IDENT_FIELDS and p[1] == "faint":
            own, theirs = (own + 1, theirs) if p[2].startswith("p1") else (own, theirs + 1)
    return own, theirs


def diff(before: list[str], after: list[str]) -> tuple[list[str], list[str]]:
    added, removed = [], []
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if op in ("replace", "delete"):
            removed += before[a0:a1]
        if op in ("replace", "insert"):
            added += after[b0:b1]
    return added, removed


def row_refs(record: dict) -> list[tuple[str, str | None]]:
    refs = [(u.get("row"), u.get("text")) for u in record.get("unmapped") or () if isinstance(u, dict)]
    refs += [(i.get("row"), i.get("text")) for i in record.get("inferred") or () if isinstance(i, dict)]
    return [(r, t) for r, t in refs if isinstance(r, str)]


def refs_in_bound(refs, texts: dict[int, tuple[str, ...]]) -> str | None:
    """None when every row an original record names is one of the bounded rows with the same text."""
    for ref, text in refs:
        m = re.fullmatch(r"T(\d+)#(\d+)", ref)
        if not m:
            continue
        turn, index = int(m[1]), int(m[2])
        rows = texts.get(turn, ())
        if index >= len(rows):
            return f"the original names row {ref}, beyond the {len(rows)} rows of turn {turn} read by then"
        if text is not None and rows[index] != text:
            return f"the original's row {ref} reads {text!r}, log.jsonl {rows[index]!r}"
    return None


def reason_key(reason: str) -> str:
    return re.sub(r"'[^']*'|\"[^\"]*\"", "...", str(reason))


# Rebuilding ---------------------------------------------------------------------------------------------------
@dataclass
class Battle:
    run: Path
    bdir: Path
    out: Path
    team: object
    summary: dict | None
    log: Log
    records: list[tuple[Path, dict | None]] = field(default_factory=list)
    report: dict = field(default_factory=dict)


def replay_check(protocol: list[str], lines: list[str], request: dict, legal) -> tuple[str | None, dict]:
    """(why the rebuilt protocol fails the replay checks or None, replay error counts before and after)."""
    from pcsa.battle import state as tracking
    from pcsa.players import mirror

    try:
        old = tracking.replay(protocol, request, "original")
        new = tracking.replay(lines, request, "reconciled")
    except Exception as error:  # noqa: BLE001 - a failed replay keeps the original
        return f"check: replay failed ({type(error).__name__}: {str(error)[:120]})", {}
    errors = {"before": len(old.replay_errors), "after": len(new.replay_errors)}
    if new_errors := [e for e in new.replay_errors if e not in set(old.replay_errors)]:
        return f"check: new replay errors ({new_errors[0][:120]})", errors
    if new.turn != old.turn:
        return f"check: replay turn {new.turn}, original {old.turn}", errors
    try:
        offered_old = {a.message for a in mirror.legal_actions(old)}
        offered_new = {a.message for a in mirror.legal_actions(new)}
    except Exception as error:  # noqa: BLE001
        return f"check: legal actions failed ({type(error).__name__}: {str(error)[:120]})", errors
    if lost := [m for m in legal if m in offered_old and m not in offered_new]:
        return f"check: the rebuilt Battle no longer offers {lost[0]}", errors
    return None, errors


def rebuild_record(b: Battle, record: dict, visit: int, translator: dict) -> tuple[dict | None, str | None, dict]:
    """(the reconciled record or None, why it is kept, details)."""
    from pcsa.battle import state as tracking
    from pcsa.live.restore import battle_tag

    protocol, request = record.get("protocol"), record.get("request")
    if not protocol or not isinstance(request, dict):
        return None, "untracked decision (no protocol or request in the record)", {}
    texts, sprites, counts, exact = b.log.bounded(visit)
    info = {"visit": visit, "rows": {str(t): n for t, n in counts.items()}}
    if not exact:
        return None, "unbounded: a matched row sprite names no source frame", info
    if (why := refs_in_bound(row_refs(record), texts)) is not None:
        return None, f"unbounded: {why}", info
    summary = b.summary or {}
    battle, out = tracking.build(
        battle_tag(summary["battle_id"], int(record.get("sequence", 0))),
        b.team,
        preview_of(summary),
        picks=list(summary["team_requested"]),
        rows_by_turn=texts,
        decision_turn=int(record.get("turn") or 0),
        sprites_by_turn=sprites,
    )
    tracking.apply_request(battle, out, request)
    if any("rejected the request" in w for w in out.warnings):
        return None, "check: the request does not apply to the rebuilt Battle", info
    if turn_lines(out.lines) != turn_lines(protocol):
        return None, "unbounded: the rebuilt |turn| lines differ from the original's", info
    own_added, why = own_actions_added(protocol, out, record.get("unmapped") or ())
    if why is not None:
        return None, why, info
    why, errors = replay_check(protocol, out.lines, request, record.get("legal") or ())
    if why is not None:
        return None, why, info
    added, removed = diff(list(protocol), list(out.lines))
    reconciled = dict(record)
    reconciled.update(
        protocol=out.lines,
        unmapped=out.unmapped,
        warnings=out.warnings,
        revealed_abilities=out.revealed_abilities,
        inferred=out.inferred,
    )
    reconciled["reconciled"] = {
        "schema": SCHEMA,
        "status": "reconciled",
        "translator": {k: translator[k] for k in ("commit", "branch", "dirty")},
        "bound": dict(info),
        "identical": list(protocol) == list(out.lines),
        "lines_added": added,
        "lines_removed": removed,
        "own_actions_added": [list(x) for x in own_added],
        "unmapped_before": record.get("unmapped"),
        "warnings_before": record.get("warnings"),
        "inferred_before": record.get("inferred"),
        "revealed_abilities_before": record.get("revealed_abilities"),
        "replay_errors": errors,
    }
    info["text"] = tracking.battle_text(battle, out.revealed_abilities)
    return reconciled, None, info


def final_check(b: Battle) -> tuple[list[str] | None, str | None, dict]:
    """(the regenerated showdown.log lines, why the battle is kept as it was, details)."""
    from pcsa.battle import state as tracking

    summary = b.summary or {}
    original = b.bdir / "showdown.log"
    if not original.is_file():
        return None, "no showdown.log (the battle did not finish in the session)", {}
    old = [x for x in original.read_text(encoding="utf-8").splitlines() if x]
    texts, sprites = b.log.final()
    battle, out = tracking.build(
        f"{summary['battle_id']}-final",
        b.team,
        preview_of(summary),
        picks=list(summary["team_requested"]),
        rows_by_turn=texts,
        sprites_by_turn=sprites,
    )
    winner = WINNER.get(summary.get("outcome"))
    lines = out.lines + ([f"|win|{winner}"] if winner else [])
    info = {
        "faints_before": faints(old),
        "faints_after": faints(lines),
        "win_before": [x for x in old if x.startswith("|win|")],
        "win_after": [x for x in lines if x.startswith("|win|")],
        "final_unmapped": out.unmapped,
        "final_text": tracking.battle_text(battle, out.revealed_abilities),
    }
    info["lines_added"], info["lines_removed"] = (len(x) for x in diff(old, lines))
    if info["win_before"] != info["win_after"]:
        return None, f"check: |win| {info['win_before']} -> {info['win_after']}", info
    before, after = info["faints_before"], info["faints_after"]
    if after[0] < before[0] or after[1] < before[1]:
        return None, f"check: faints per side {before} -> {after}", info
    if after != before:
        info["faints_added"] = [after[0] - before[0], after[1] - before[1]]
    return lines, None, info


def load_records(bdir: Path) -> list[tuple[Path, dict | None]]:
    out = []
    for path in sorted((bdir / "decisions").glob("*.json")):
        try:
            out.append((path, read_json(path)))
        except (OSError, ValueError):
            out.append((path, None))
    return out


def battle_problem(b: Battle) -> str | None:
    if b.summary is None:
        return "unfinished: no battle.json (the session ended or crashed before the result)"
    if b.team is None:
        return "no team record in the run's state.json or own_team.json"
    if not b.summary.get("battle_id") or not b.summary.get("team_requested"):
        return "battle.json has no battle_id or team_requested"
    if b.log.problem:
        return f"log not boundable: {b.log.problem}"
    visits = [r.get("visit") for _, r in b.records if r and r.get("kind") == "action"]
    if any(not isinstance(v, int) for v in visits):
        return "an action record has no visit number"
    if any(later < earlier for earlier, later in zip(visits, visits[1:], strict=False)):
        return "the action visits restart (a resumed battle: its earlier rows were in another log)"
    return None


def final_rebuild(b: Battle) -> tuple[list[str] | None, str | None, dict]:
    """The battle-level step: (regenerated showdown.log lines, why the whole battle is kept, details)."""
    rep = b.report
    problem = battle_problem(b)
    if problem is not None:
        rep["battle_kept"] = problem
        return None, problem, {}
    try:
        lines, problem, final = final_check(b)
    except Exception as error:  # noqa: BLE001 - a battle that cannot be rebuilt is kept as it was
        lines, problem, final = None, f"error: final rebuild failed ({type(error).__name__}: {error})", {}
        rep["traceback"] = traceback.format_exc()
    rep["final"] = {k: v for k, v in final.items() if k not in ("final_text", "final_unmapped")}
    rep["final_unmapped"] = Counter(reason_key(u["reason"]) for u in final.get("final_unmapped") or [])
    if problem is not None:
        rep["battle_kept"] = problem
    return lines, problem, final


def record_outcome(b: Battle, path: Path, record: dict | None, visit: int, problem: str | None, translator: dict):
    """(the reconciled record and its battle text, or None and why the original is kept)."""
    if record is None:
        return None, "unreadable record"
    if problem is not None:
        return None, problem
    if (rid := record.get("battle_id")) is not None and rid != (b.summary or {}).get("battle_id"):
        return None, "the record belongs to another battle (reused folder)"
    try:
        new, why, info = rebuild_record(b, record, visit, translator)
    except Exception as error:  # noqa: BLE001 - one record that cannot be rebuilt keeps its original
        b.report.setdefault("errors", []).append(f"{path.name}: {traceback.format_exc()[-600:]}")
        return None, f"error: {type(error).__name__}: {str(error)[:160]}"
    return ((new, info.pop("text")), None) if new is not None else (None, why or "?")


def write_record(b: Battle, path: Path, record: dict | None, done, why: str | None) -> dict:
    """Writes the record's reconciled or original files; returns the record as written."""
    rep, target = b.report, b.out / "decisions" / path.name
    if done is not None:
        new, battle_text = done
        target.write_text(json.dumps(new, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        target.with_suffix(".txt").write_text(battle_text, encoding="utf-8")
        r = new["reconciled"]
        rep["reconciled"] += 1
        rep["identical"] += int(r["identical"])
        rep["lines_added"] += len(r["lines_added"])
        rep["lines_removed"] += len(r["lines_removed"])
        return new
    shutil.copy2(path, target)
    if path.with_suffix(".txt").is_file():
        shutil.copy2(path.with_suffix(".txt"), target.with_suffix(".txt"))
    rep["kept"][why] = rep["kept"].get(why, 0) + 1
    return record or {}


def unmapped_rows(record: dict | None, into: dict[str, str]) -> None:
    for u in (record or {}).get("unmapped") or ():
        into[str(u.get("row"))] = reason_key(u.get("reason", "?"))


def reconcile_battle(b: Battle, translator: dict) -> None:
    (b.out / "decisions").mkdir(parents=True, exist_ok=True)
    for name in BATTLE_COPIES:
        if (b.bdir / name).is_file():
            shutil.copy2(b.bdir / name, b.out / name)
    rep = b.report
    rep.update(records=len(b.records), reconciled=0, identical=0, kept={}, lines_added=0, lines_removed=0)
    lines, problem, final = final_rebuild(b)
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    visit = 0  # a forced replacement saw the log of the last action decision before it
    for path, record in b.records:
        if record is not None and record.get("kind") == "action" and isinstance(record.get("visit"), int):
            visit = record["visit"]
        unmapped_rows(record, before)
        done, why = record_outcome(b, path, record, visit, problem, translator)
        unmapped_rows(write_record(b, path, record, done, why), after)
    rep["unmapped_before"], rep["unmapped_after"] = Counter(before.values()), Counter(after.values())
    rep["unmapped_rows_before"], rep["unmapped_rows_after"] = len(before), len(after)
    if problem is None and lines is not None:
        (b.out / "showdown.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (b.out / "battle_final.txt").write_text(final["final_text"], encoding="utf-8")
        return
    for name in ("showdown.log", "battle_final.txt"):
        if (b.bdir / name).is_file():
            shutil.copy2(b.bdir / name, b.out / name)


# Report -------------------------------------------------------------------------------------------------------
def newest_per_run(battles: list[dict]) -> dict[str, str]:
    """run -> its highest-numbered battle whose records were reconciled (not kept as a whole)."""
    newest: dict[str, str] = {}
    for r in battles:
        if r.get("in_progress") or not r.get("battle_id") or r.get("battle_kept"):
            continue
        run, number = r["battle"].rsplit("/", 1)[0], int(r["battle"].rsplit("-", 1)[1])
        if run not in newest or number > int(newest[run].rsplit("-", 1)[1]):
            newest[run] = r["battle"]
    return newest


def write_report(tree: Path, header: dict, battles: list[dict]) -> dict:
    total: dict = {
        "battles": len(battles),
        "records": 0,
        "reconciled": 0,
        "identical": 0,
        "lines_added": 0,
        "lines_removed": 0,
        "kept": Counter(),
        "battles_kept": Counter(),
        "unmapped_before": Counter(),
        "unmapped_after": Counter(),
        "unmapped_rows_before": 0,
        "unmapped_rows_after": 0,
        "faints_added": [],
        "final_unmapped": Counter(),
    }
    total["in_progress"] = [r["battle"] for r in battles if r.get("in_progress")]
    newest = newest_per_run(battles)
    total["newest_reconciled_per_run"] = newest
    battles = [r for r in battles if not r.get("in_progress")]
    total["battles"] = len(battles)
    for rep in battles:
        for key in (
            "records",
            "reconciled",
            "identical",
            "lines_added",
            "lines_removed",
            "unmapped_rows_before",
            "unmapped_rows_after",
        ):
            total[key] += rep.get(key, 0)
        total["kept"].update(rep.get("kept", {}))
        total["unmapped_before"].update(rep.get("unmapped_before", {}))
        total["unmapped_after"].update(rep.get("unmapped_after", {}))
        total["final_unmapped"].update(rep.get("final_unmapped", {}))
        if rep.get("battle_kept"):
            total["battles_kept"][rep["battle_kept"]] += 1
        if (added := (rep.get("final") or {}).get("faints_added")) is not None:
            total["faints_added"].append({"battle": rep["battle"], "added": added})
    result = {**header, "total": total, "battles": battles}
    (tree / "report.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    t = header["translator"]
    md = [
        "# Reconciled live battles",
        "",
        f"Translator: `{t['repo']}` {t['branch']} {t['commit']}"
        + (f", dirty: {', '.join(t['dirty'])}" if t["dirty"] else ", clean"),
        f"pcsa imported from `{t['pcsa_imported_from']}`. Runs: {', '.join('`' + r + '`' for r in header['runs'])}.",
        f"Started {header['started']}, {header['seconds']:.0f} s.",
        "",
        "## Total",
        "",
        f"- battles {total['battles']}, records {total['records']}: reconciled {total['reconciled']} "
        f"(protocol identical to the original: {total['identical']}), kept as original "
        f"{total['records'] - total['reconciled']}",
        f"- protocol lines added {total['lines_added']}, removed {total['lines_removed']} (summed over records)",
        f"- unmapped rows (unique rows per battle over its records): {total['unmapped_rows_before']} -> "
        f"{total['unmapped_rows_after']}",
        f"- battles whose final log adds faints: {len(total['faints_added'])} {total['faints_added']}",
        f"- in progress, skipped (not written): {', '.join(total['in_progress']) or 'none'}",
        "",
        "### Unmapped rows by reason",
        "",
        "| reason | before | after |",
        "|---|---|---|",
    ]
    for reason in sorted(
        set(total["unmapped_before"]) | set(total["unmapped_after"]),
        key=lambda r: -total["unmapped_before"].get(r, 0),
    ):
        md.append(
            f"| {reason} | {total['unmapped_before'].get(reason, 0)} | {total['unmapped_after'].get(reason, 0)} |"
        )
    md += ["", "### Newest battle reconciled per run", "", "| run | battle |", "|---|---|"]
    md += [
        f"| {run} | {name.rsplit('/', 1)[1]} |" for run, name in sorted(total["newest_reconciled_per_run"].items())
    ]
    md += ["", "### Records kept as original", "", "| why | records |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in total["kept"].most_common()]
    md += ["", "### Battles kept as original", "", "| why | battles |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in total["battles_kept"].most_common()]
    md += [
        "",
        "### Final-log rows the current translator leaves unmapped (incl. turns no decision read)",
        "",
        "| reason | rows |",
        "|---|---|",
    ]
    md += [f"| {k} | {v} |" for k, v in total["final_unmapped"].most_common()]
    md += [
        "",
        "## Per battle",
        "",
        "| battle | records | reconciled | identical | kept | +lines | -lines | unmapped before -> after | note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rep in battles:
        kept = sum(rep.get("kept", {}).values())
        note = rep.get("battle_kept") or ("; ".join(f"{k} x{v}" for k, v in rep.get("kept", {}).items()) or "")
        if (added := (rep.get("final") or {}).get("faints_added")) is not None:
            note = f"faints added {added}. " + note
        md.append(
            f"| {rep['battle']} | {rep.get('records', 0)} | {rep.get('reconciled', 0)} | "
            f"{rep.get('identical', 0)} | {kept} | {rep.get('lines_added', 0)} | {rep.get('lines_removed', 0)} | "
            f"{rep.get('unmapped_rows_before', 0)} -> {rep.get('unmapped_rows_after', 0)} | {note} |"
        )
    (tree / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return total


def selected(run: Path, bdir: Path, only: list[str]) -> bool:
    return not only or any(o in (run.name, f"{run.name}/{bdir.name}") for o in only)


def safe_log(bdir: Path) -> Log:
    try:
        return load_log(bdir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return Log({}, [], f"unreadable log files ({type(error).__name__}: {error})")


def pending_battle(run: Path) -> int | None:
    try:
        pending = read_json(run / "state.json").get("pending_battle")
    except (OSError, ValueError, AttributeError):
        return None
    return pending if isinstance(pending, int) else None


def reconcile_run(run: Path, out_run: Path, only: list[str], translator: dict, newest: bool = False) -> list[dict]:
    """Every battle folder the run holds now. In the newest run, the battle its state calls pending and that has no
    battle.json yet is still being played: it is skipped (not written) and reported as in progress."""
    bdirs = [p for p in run.glob("battle-*") if p.is_dir() and p.name.rsplit("-", 1)[1].isdigit()]
    bdirs = [p for p in bdirs if selected(run, p, only)]
    if not bdirs:
        return []
    out_run.mkdir(parents=True, exist_ok=True)
    for f in RUN_FILES:
        if (run / f).is_file():
            shutil.copy2(run / f, out_run / f)
    try:
        team = team_of(run)
    except Exception:  # noqa: BLE001 - no team: every battle of the run is kept as it was
        team = None
    reports = []
    pending = pending_battle(run) if newest else None
    for bdir in sorted(bdirs, key=lambda p: int(p.name.rsplit("-", 1)[1])):
        name = f"{out_run.parent.name}:{run.name}/{bdir.name}"
        if int(bdir.name.rsplit("-", 1)[1]) == pending and not (bdir / "battle.json").is_file():
            reports.append({"battle": name, "battle_id": None, "in_progress": True})
            print(f"{name}: in progress (no result yet), skipped", flush=True)
            continue
        try:
            summary = read_json(bdir / "battle.json") if (bdir / "battle.json").is_file() else None
        except (OSError, ValueError):
            summary = None
        b = Battle(run, bdir, out_run / bdir.name, team, summary, safe_log(bdir), load_records(bdir))
        b.report = {"battle": name, "battle_id": (summary or {}).get("battle_id")}
        reconcile_battle(b, translator)
        reports.append(b.report)
        rep = b.report
        print(
            f"{rep['battle']}: {rep['reconciled']}/{rep['records']} reconciled ({rep['identical']} identical), "
            f"unmapped rows {rep['unmapped_rows_before']} -> {rep['unmapped_rows_after']}"
            + (f"; kept: {rep['battle_kept']}" if rep.get("battle_kept") else ""),
            flush=True,
        )
    return reports


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, default=DEFAULT_RUNS, help="runs folders (read only)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--out", type=Path, default=None, help="the tree itself (default <out-root>/<commit>[-dirty])")
    ap.add_argument("--tree-file", type=Path, default=None, help="write the tree's path to this file")
    ap.add_argument("--only", nargs="*", default=[], help="RUN or RUN/battle-N names to reconcile (default all)")
    ap.add_argument("--allow-foreign-pcsa", action="store_true", help="translate with a pcsa outside this repo")
    a = ap.parse_args(argv)
    translator = translator_state()
    if not Path(translator["pcsa_imported_from"]).is_relative_to(REPO / "src") and not a.allow_foreign_pcsa:
        print(
            f"pcsa is imported from {translator['pcsa_imported_from']}, not {REPO / 'src'}: run with "
            "PYTHONPATH=src from the repo",
            file=sys.stderr,
        )
        return 2
    clock = SystemClock()
    started, t0 = time.strftime("%Y-%m-%d %H:%M:%S"), clock.now()
    name = translator["commit"] + ("-dirty" if translator["dirty"] else "")
    tree = a.out if a.out is not None else fresh(a.out_root / name)
    for folder in a.runs:
        if tree.resolve().is_relative_to(folder.resolve()):
            print(f"the tree {tree} is inside the runs folder {folder}", file=sys.stderr)
            return 2
    tree.mkdir(parents=True, exist_ok=a.out is not None)
    if a.tree_file is not None:
        a.tree_file.write_text(str(tree) + "\n", encoding="utf-8")
    print(f"reconciling into {tree} with {name} (dirty: {translator['dirty']})", flush=True)
    reports = []
    for folder in a.runs:
        runs = sorted(p for p in folder.glob("*-run") if p.is_dir())  # discovered now: every run and battle there is
        for run in runs:
            reports += reconcile_run(run, tree / runs_tag(folder) / run.name, a.only, translator, run == runs[-1])
    header = {
        "schema": SCHEMA,
        "tree": str(tree),
        "translator": translator,
        "runs": [str(r) for r in a.runs],
        "only": a.only,
        "started": started,
        "seconds": clock.now() - t0,
    }
    total = write_report(tree, header, reports)
    print(
        f"done: {total['reconciled']}/{total['records']} records reconciled ({total['identical']} identical); "
        f"unmapped rows {total['unmapped_rows_before']} -> {total['unmapped_rows_after']}",
        flush=True,
    )
    print(f"TREE {tree}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

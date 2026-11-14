"""scripts/reconcile_live.py: each decision re-translated from the rows read by then, originals untouched."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from pcsa.battle import state as tracking
from pcsa.battle.teams import team_from_record

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "reconcile_live.py"


@pytest.fixture(scope="module")
def rl():
    spec = importlib.util.spec_from_file_location("reconcile_live", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["reconcile_live"] = module
    spec.loader.exec_module(module)
    return module


def member(i, name, moves, hp=150):
    stats = {"hp": hp, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}
    return {
        "roster_index": i,
        "name": name,
        "species": name,
        "ability": None,
        "item": None,
        "moves": moves,
        "stats": stats,
        "points": None,
        "nature": None,
    }


TEAM = {
    "name": "t",
    "members": [
        member(0, "Gholdengo", ["Make It Rain", "Shadow Ball"]),
        member(1, "Garchomp", ["Earthquake", "Dragon Claw"]),
        member(2, "Golisopod", ["Leech Life", "Liquidation"]),
        member(3, "Primarina", ["Moonblast", "Sparkling Aria"]),
        member(4, "Mamoswine", ["Icicle Crash", "Earthquake"]),
        member(5, "Cinderace", ["Pyro Ball", "U-turn"], hp=156),
    ],
}
PREVIEW = [
    {"slot": i, "box": [0, 0, 1, 1], "top": [[n, 0.9]], "species": [[n, 0.9]]}
    for i, n in enumerate(["Glimmora", "Corviknight", "Kingambit", "Hippowdon", "Rotom-Wash", "Volcarona"])
]
PICKS = [5, 2, 3]
TURN1 = [
    "sent out Glimmora!",
    "Go! Cinderace!",
    "Move: Cinderace used Pyro Ball!",
    "Move: The opposing Glimmora used Power Gem!",
]
TURN1_LATE = ["Move: The opposing Glimmora used Earth Power!"]  # read at visit 2 (the rest of the turn)
TURN2 = ["Move: Cinderace used U-turn!", "Fainted: The opposing Glimmora fainted!"]


def sprite(visit, status="none", name=None):
    return {
        "status": status,
        "name": name,
        "base_name": name,
        "source": f"battle-1/log-frames/visit{visit:02d}-0.png",
    }


def make_battle(root: Path) -> Path:
    run = root / "runs" / "20990101-000000-run"
    bdir = run / "battle-1"
    (bdir / "decisions").mkdir(parents=True)
    (run / "state.json").write_text(json.dumps({"team_record": TEAM}), encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"player": "neural"}), encoding="utf-8")
    summary = {"battle_id": "B1", "outcome": "win", "team_requested": PICKS, "opponent_preview": PREVIEW}
    (bdir / "battle.json").write_text(json.dumps(summary), encoding="utf-8")
    (bdir / "battle_id.json").write_text(json.dumps({"battle_id": "B1"}), encoding="utf-8")
    # visit 1: turn 1 with 4 rows; visit 2: turn 1 again, one more row; visit 3: turn 2
    visits = [
        {"visit": 1, "turn": 1, "rows": 4, "appended": 4, "anchor": "new"},
        {"visit": 2, "turn": 1, "rows": 5, "appended": 1, "anchor": "prefix"},
        {"visit": 3, "turn": 2, "rows": 2, "appended": 2, "anchor": "new"},
    ]
    (bdir / "ocr_audit.json").write_text(json.dumps({"visits": visits}), encoding="utf-8")
    rows = [(1, t, sprite(1)) for t in TURN1] + [(1, t, sprite(2)) for t in TURN1_LATE]
    rows += [(2, t, sprite(3)) for t in TURN2]
    lines, index = [], {}
    for turn, text, s in rows:
        lines.append(json.dumps({"turn": turn, "index": index.setdefault(turn, 0), "text": text, "sprite": s}))
        index[turn] += 1
    (bdir / "log.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return bdir, team_from_record(TEAM), summary


def original(team, summary, rows_by_turn, turn, seq, kind, visit=None, forced=False):
    from pcsa.live.restore import battle_tag
    from pcsa.phone.sprites import SlotMatch

    preview = tuple(
        SlotMatch(s["slot"], tuple(s["box"]), tuple(map(tuple, s["top"])), tuple(map(tuple, s["species"])))
        for s in summary["opponent_preview"]
    )
    battle, out = tracking.build(
        battle_tag("B1", seq), team, preview, picks=PICKS, rows_by_turn=rows_by_turn, decision_turn=turn
    )
    request = tracking.battle_request(team, PICKS, battle, active=0, screen_hp={}, fainted=set(), forced=forced)
    tracking.apply_request(battle, out, request)
    rec = {
        "schema": "pcsa-decision/2",
        "battle_id": "B1",
        "sequence": seq,
        "kind": kind,
        "turn": battle.turn,
        "order": "/choose switch Golisopod",
        "legal": [],
        "degrade": "full",
        "request": request,
        "protocol": out.lines,
        "unmapped": out.unmapped,
        "warnings": out.warnings,
        "revealed_abilities": out.revealed_abilities,
        "inferred": out.inferred,
    }
    if visit is not None:
        rec["visit"] = visit
    return rec


def write(bdir, rec):
    path = bdir / "decisions" / f"{rec['sequence']:04d}-{rec['kind']}.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


def test_sprite_at_keeps_only_what_was_read(rl):
    early = {"status": "matched", "name": "Glimmora", "source": "b/log-frames/visit01-0.png"}
    late = {"status": "matched", "name": "Glimmora", "source": "b/log-frames/visit03-recovery01.png"}
    assert rl.sprite_at(early, 1) == (early, True)
    assert rl.sprite_at(late, 2) == (None, True)
    conflict = {"status": "unknown", "reason": "conflicting_sprites", "observations": [early, late]}
    assert rl.sprite_at(conflict, 2) == (early, True)
    assert rl.sprite_at(conflict, 3) == (conflict, True)
    assert rl.sprite_at({"status": "matched", "name": "X"}, 3)[1] is False


def test_bounded_rows_follow_the_visits(rl, tmp_path):
    bdir, _, _ = make_battle(tmp_path)
    log = rl.load_log(bdir)
    assert log.problem is None
    texts, _, counts, exact = log.bounded(1)
    assert texts == {1: tuple(TURN1)} and counts == {1: 4} and exact
    texts, _, _, _ = log.bounded(3)
    assert texts == {1: tuple(TURN1 + TURN1_LATE), 2: tuple(TURN2)}


def test_log_that_does_not_add_up_is_unbounded(rl, tmp_path):
    bdir, _, _ = make_battle(tmp_path)
    with (bdir / "log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"turn": 2, "index": 2, "text": "extra", "sprite": None}) + "\n")
    assert "visits append 2 rows" in rl.load_log(bdir).problem


def test_reconcile_end_to_end(rl, tmp_path):
    bdir, team, summary = make_battle(tmp_path)
    t1, t12 = {1: tuple(TURN1)}, {1: tuple(TURN1 + TURN1_LATE)}
    write(bdir, original(team, summary, t1, 1, 0, "action", visit=1))
    write(bdir, original(team, summary, t1, 1, 1, "forced", forced=True))  # mid-turn: visit 1's rows only
    future = original(team, summary, {**t12, 2: tuple(TURN2)}, 3, 2, "action", visit=2)  # saw rows of visit 3
    write(bdir, future)
    write(bdir, original(team, summary, {**t12, 2: tuple(TURN2)}, 3, 3, "action", visit=3))
    full = original(team, summary, {**t12, 2: tuple(TURN2)}, 0, 9, "action")
    (bdir / "showdown.log").write_text("\n".join(full["protocol"] + ["|win|me"]) + "\n", encoding="utf-8")
    before = {p: p.read_bytes() for p in bdir.rglob("*") if p.is_file()}
    tree = tmp_path / "tree"
    assert rl.main(["--runs", str(tmp_path / "runs"), "--out", str(tree)]) == 0
    assert {p: p.read_bytes() for p in bdir.rglob("*") if p.is_file()} == before  # originals untouched
    out = tree / rl.runs_tag(tmp_path / "runs") / "20990101-000000-run" / "battle-1"
    recs = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in (out / "decisions").glob("*.json")}
    assert recs["0000-action.json"]["reconciled"]["identical"] is True
    forced = recs["0001-forced.json"]
    assert forced["reconciled"]["bound"]["rows"] == {"1": 4}
    assert not any("Earth Power" in x for x in forced["protocol"])
    # the record that saw visit 3's rows cannot have been made at visit 2: its |turn| lines say so
    assert "reconciled" not in recs["0002-action.json"]
    assert recs["0003-action.json"]["reconciled"]["identical"] is True
    report = json.loads((tree / "report.json").read_text(encoding="utf-8"))
    assert report["total"]["reconciled"] == 3
    assert (out / "showdown.log").read_text(encoding="utf-8").splitlines()[-1] == "|win|me"
    assert (out / "battle.json").is_file() and (out.parent / "manifest.json").is_file()


def test_own_actions_added_only_from_newly_mapped_rows(rl):
    from types import SimpleNamespace

    old = ["|switch|p1a: Cinderace|Cinderace, L50|156/156", "|move|p1a: Cinderace|Pyro Ball|p2a: Glimmora"]
    new = [*old, "|switch|p1a: Primarina|Primarina, L50|170/170"]
    fixed = [{"row": "T2#1", "text": "Go! Primarina!", "reason": "cannot assign sprite 'Ditto' to one roster member"}]
    out = SimpleNamespace(lines=new, unmapped=[])
    assert rl.own_actions_added(old, out, fixed) == ([("switch", "p1a: Primarina")], None)
    assert "from no newly mapped row" in rl.own_actions_added(old, out, [])[1]
    assert (
        "from no newly mapped row" in rl.own_actions_added(old, SimpleNamespace(lines=new, unmapped=fixed), fixed)[1]
    )
    assert (
        "is not in the rebuilt protocol"
        in rl.own_actions_added(old, SimpleNamespace(lines=old[:1], unmapped=[]), [])[1]
    )


def test_battle_in_progress_is_skipped(rl, tmp_path):
    bdir, team, summary = make_battle(tmp_path)
    write(bdir, original(team, summary, {1: tuple(TURN1)}, 1, 0, "action", visit=1))
    full = original(team, summary, {1: tuple(TURN1 + TURN1_LATE), 2: tuple(TURN2)}, 0, 9, "action")
    (bdir / "showdown.log").write_text("\n".join([*full["protocol"], "|win|me"]) + "\n", encoding="utf-8")
    run = bdir.parent
    playing = run / "battle-2" / "decisions"
    playing.mkdir(parents=True)
    (playing / "0000-action.json").write_text(json.dumps({"schema": "pcsa-decision/2", "sequence": 0}), "utf-8")
    (run / "state.json").write_text(json.dumps({"team_record": TEAM, "pending_battle": 2}), encoding="utf-8")
    tree = tmp_path / "tree"
    assert rl.main(["--runs", str(tmp_path / "runs"), "--out", str(tree)]) == 0
    out_run = tree / rl.runs_tag(tmp_path / "runs") / run.name
    assert (out_run / "battle-1" / "decisions" / "0000-action.json").is_file()
    assert not (out_run / "battle-2").exists()
    total = json.loads((tree / "report.json").read_text(encoding="utf-8"))["total"]
    assert total["in_progress"] == [f"{rl.runs_tag(tmp_path / 'runs')}:{run.name}/battle-2"]
    assert total["newest_reconciled_per_run"][f"{rl.runs_tag(tmp_path / 'runs')}:{run.name}"].endswith("battle-1")

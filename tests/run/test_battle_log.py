"""battles.jsonl.gz: protocol formatting and seat merging without a simulator, and a 2-battle `pcsa sim play`
when the Showdown engine is installed."""

from __future__ import annotations

import pytest

from pcsa.common import log, paths
from pcsa.common.config import SimConfig, load_config
from pcsa.common.errors import ConfigError
from pcsa.run import battle_log
from pcsa.sim.engine import Engine

REPLAY = [
    ["", "player", "p1", "a1"],
    ["", "turn", "1"],
    ["", "move", "p1a: Garchomp", "Earthquake"],
    ["", "win", "a1"],
]


def _engine() -> Engine | None:
    try:
        return Engine.locate(SimConfig().format)
    except ConfigError:
        return None


ENGINE = _engine()


def _seat(role: str, won: bool | None, team: str, picks: list[str]) -> dict:
    return {"role": role, "team": team, "picks": picks, "won": won, "turns": 1, "replay": REPLAY}


def test_protocol_text_joins_each_message_on_its_own_line() -> None:
    text = battle_log.protocol_text(REPLAY)
    assert text == "|player|p1|a1\n|turn|1\n|move|p1a: Garchomp|Earthquake\n|win|a1"
    assert battle_log.protocol_text([]) is None and battle_log.protocol_text(None) is None


def test_records_merge_both_seats_into_one_per_battle(tmp_path) -> None:
    a = {"t1": _seat("p1", True, "TEAM A", ["X", "Y", "Z"]), "t2": _seat("p2", False, "TEAM A2", ["Q"])}
    b = {"t1": _seat("p2", False, "TEAM B", ["U"]), "t2": _seat("p1", True, "TEAM B2", [])}
    rows = battle_log.records(["t1", "t2", "t3"], {"a": ("heuristic", a), "b": ("random", b)})
    assert [r["tag"] for r in rows] == ["t1", "t2"]  # t3 never finished: no record
    first, second = rows
    assert (first["p1"], first["p2"], first["p1_team"], first["p2_team"]) == (
        "heuristic",
        "random",
        "TEAM A",
        "TEAM B",
    )
    assert first["p1_picks"] == ["X", "Y", "Z"] and first["winner"] == "p1" and first["winner_side"] == "a"
    assert (second["index"], second["p1"], second["p1_side"], second["winner_side"]) == (1, "random", "b", "b")
    assert second["p1_picks"] is None and second["p2_picks"] == ["Q"]
    assert first["protocol"].endswith("|win|a1") and first["protocol_seat"] == "p1"
    only_b = battle_log.records(["t4"], {"a": ("heuristic", {}), "b": ("random", {"t4": _seat("p2", True, "B", [])})})
    assert only_b[0]["p1"] == "heuristic" and only_b[0]["p1_team"] is None and only_b[0]["winner"] == "p2"
    path = tmp_path / battle_log.FILE
    assert battle_log.write(path, rows) == path.stat().st_size > 0
    assert battle_log.read(path) == rows


@pytest.mark.skipif(ENGINE is None, reason="the simulator is not installed; run `pcsa setup --with sim`")
def test_sim_play_writes_one_battle_record_per_battle(monkeypatch, tmp_path) -> None:
    from pcsa.run.sim_command import play_command

    monkeypatch.setenv(paths.OUTPUT_ENV, str(tmp_path))
    config = load_config(paths.config_dir() / "agent.yaml")
    try:
        outcome = play_command(config, "random", "max_damage", 2, 7, noisy=False, port=None)
    finally:
        log.configure(None, console=False)  # closes events.jsonl
    (out_dir,) = (tmp_path / "sim").iterdir()
    rows = battle_log.read(out_dir / battle_log.FILE)
    assert outcome.code == 0 and len(rows) == 2 and len({r["tag"] for r in rows}) == 2
    assert all("\n|win|" in r["protocol"] and r["p1_team"] and r["p2_team"] for r in rows)
    assert {(r["p1_side"], r["p2_side"]) for r in rows} == {("a", "b"), ("b", "a")}
    assert {r["p1"] for r in rows} == {"random", "max_damage"}

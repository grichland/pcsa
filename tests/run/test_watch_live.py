"""`pcsa watch` on a run that is being written, and over HTTP: events arrive line by line, files land late."""

from __future__ import annotations

import json
import shutil
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from pcsa.common import paths
from pcsa.run.watch import RunReader, make_server
from pcsa.run.watch_events import EventScan

FIXTURE = paths.repo_root() / "tests" / "fixtures" / "watch_run"
# What app.js, matrix.js and panels.js write to: the top bar, the frame and the four panels.
PAGE_IDS = (
    "run-status",
    "top-battle",
    "top-turn",
    "session-record",
    "top-rating",
    "top-player",
    "params",
    "warn",
    "warn-count",
    "warnings",
    "theme",
    "t-frame",
    "phone-frame",
    "t-matrix",
    "mx-action",
    "mx-stats",
    "mx-banner",
    "mx-grid",
    "mx-alt",
    "mx-empty",
    "t-win",
    "win-record",
    "win-rate",
    "win-recent",
    "win-rating",
    "spark",
    "strip",
    "follow",
    "chance",
    "vspark",
    "t-teams",
    "team-ours",
    "field-strip",
    "team-theirs",
    "unseen",
    "belief-key",
    "t-log",
    "battle-log",
    "tip",
)
# Every key of /api/state the page reads (app.js, matrix.js, panels.js).
STATE_KEYS = (
    "run_name",
    "player",
    "player_view",
    "screen",
    "turn",
    "frame_id",
    "battle",
    "pinned",
    "battles",
    "session_record",
    "rating",
    "decision",
    "decisions",
    "preview",
    "winrate",
    "log",
    "warnings",
    "warning_count",
    "disconnects",
    "status",
    "age_s",
    "move_capture",
    "your_team",
    "foes",
    "unseen",
    "unseen_belief",
    "field",
    "active",
    "opponent",
    "legacy",
)
MATRIX_KEYS = ("rows", "row_labels", "legal", "cols", "col_labels", "col_kind", "shown", "col_legal", "p_cols")
MATRIX_KEYS += ("cells", "probs", "chosen", "played", "mega_p", "tau", "opp_temperature", "masked")


def line(**event) -> str:
    return json.dumps({"ts": "2026-11-18T22:00:00.000+00:00", "level": "INFO", **event}) + "\n"


def test_events_are_read_incrementally_and_a_half_line_waits(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    scan = EventScan(path)
    scan.refresh()
    assert scan.offset == 0 and scan.battles == {}
    start = line(event="battle_start", battle=1, roster=["Garchomp"])
    end = line(event="battle_end", battle=1, outcome="win", source="result screen")
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(start + end[:25])
        stream.flush()
        scan.refresh()
        assert scan.offset == len(start) and scan.battles[1]["outcome"] is None and scan.record["wins"] == 0
        stream.write(end[25:])
        stream.flush()
        scan.refresh()
    assert scan.offset == len(start) + len(end) and scan.battles[1]["outcome"] == "win" and scan.record["wins"] == 1


def test_bad_lines_and_odd_events_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "not json\n"
        + "[1, 2]\n"
        + line(event="battle_end")
        + line(event="decision", battle="one", kind="action")
        + line(event="turn", battle=3, visit="x")
        + line(event="device_retry", level="WARNING", op="screencap")
        + line(event="rank", tier="Master Ball", rating=1500.5),
        encoding="utf-8",
    )
    scan = EventScan(path)
    scan.refresh()
    assert list(scan.battles) == [3] and scan.warning_count == 1
    view = scan.rating_view()
    assert view is not None and view["rating"] == 1500.5 and view["tier"] == "Master Ball"


def test_state_follows_a_run_as_its_files_appear(tmp_path: Path) -> None:
    run = tmp_path / "live-run"
    run.mkdir()
    reader = RunReader(run)
    assert reader.state()["status"] == "WAITING"
    events = (FIXTURE / "events.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    first_end = next(i for i, raw in enumerate(events) if '"battle_end"' in raw)
    with (run / "events.jsonl").open("a", encoding="utf-8", newline="") as stream:
        stream.writelines(events[:first_end])
        stream.flush()
        state = reader.state()
        assert state["status"] == "LIVE" and state["battle"]["index"] == 1 and state["battle"]["outcome"] is None
        assert state["decision"] is None and state["preview"]["names"] == ["Hippowdon", "Metagross", "Primarina"]
        assert state["player"] is None and "your_team" not in state
        shutil.copy(FIXTURE / "manifest.json", run / "manifest.json")
        (run / "battle-1").mkdir()
        shutil.copy(FIXTURE / "battle-1" / "team_preview.json", run / "battle-1" / "team_preview.json")
        state = reader.state()  # the preview: our team and their six with the belief, no decision yet
        assert state["decision"] is None and len(state["your_team"]) == 3 and state["foes"] == []
        assert len(state["unseen"]) == 6 and all(u["belief"]["lead"] is not None for u in state["unseen_belief"])
        decisions = run / "battle-1" / "decisions"
        decisions.mkdir()
        shutil.copy(FIXTURE / "battle-1" / "decisions" / "0000-action.json", decisions / "0000-action.json")
        state = reader.state()
        assert state["player"] == "neural" and state["decision"]["label"] == "Yawn" and state["turn"] == 1
        assert state["opponent"]["name"] == "Meganium" and len(state["decisions"]) == 1
        stream.writelines(events[first_end:])
        stream.flush()
        state = reader.state()
    assert state["battle"]["index"] == 3 and state["session_record"]["losses"] == 2
    assert state["decision"] is None and state["decisions"] == []
    assert reader.state(battle=1)["decision"]["label"] == "Yawn"


# HTTP ----------------------------------------------------------------------------------------------


@pytest.fixture
def server(tmp_path: Path) -> Iterator[str]:
    run = Path(shutil.copytree(FIXTURE, tmp_path / "20261118-162900-neural-148"))
    (run / "frames").mkdir()
    (run / "frames" / "00001-poll-lobby.jpg").write_bytes(b"\xff\xd8 not a real picture \xff\xd9")
    httpd = make_server(run, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def get(url: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read()


def test_server_serves_the_page_and_its_data(server: str) -> None:
    status, kind, body = get(server + "/")
    page = body.decode("utf-8")
    assert status == 200 and kind.startswith("text/html") and "PCSA Singles Agent" in page
    for wanted in (f'id="{name}"' for name in PAGE_IDS):
        assert wanted in page
    status, kind, body = get(server + "/style.css")
    assert status == 200 and kind.startswith("text/css") and b".hpbar" in body and b".chip.is-believed" in body
    status, kind, body = get(server + "/app.js")
    assert status == 200 and kind.startswith("text/javascript") and b"/api/state" in body
    for script, marker in (("/matrix.js", b"function renderMatrix"), ("/panels.js", b"function renderLog")):
        status, kind, body = get(server + script)
        assert status == 200 and kind.startswith("text/javascript") and marker in body and f'src="{script}"' in page

    status, kind, body = get(server + "/api/state")
    state = json.loads(body)
    assert status == 200 and kind.startswith("application/json")
    assert all(key in state for key in STATE_KEYS)
    assert state["player"] == "neural" and state["battle"]["index"] == 3 and len(state["battles"]) == 3
    assert state["decision"]["notes"]["rule"] == "neural" and state["frame_id"].startswith("00001-poll-lobby.jpg:")
    matrix = state["decision"]["notes"]["matrix"]
    assert all(key in matrix for key in MATRIX_KEYS) and len(matrix["cells"]) == 11 and len(matrix["cells"][0]) == 9
    assert all("belief" in f for f in state["foes"]) and state["winrate"]["battles"] == 3
    pinned = json.loads(get(server + "/api/state?battle=1")[2])
    assert pinned["battle"]["index"] == 1 and pinned["pinned"] == 1 and len(pinned["decisions"]) == 4
    assert json.loads(get(server + "/api/state?battle=abc")[2])["battle"]["index"] == 3

    status, kind, body = get(server + "/api/frame")
    assert status == 200 and kind == "image/jpeg" and body.startswith(b"\xff\xd8")
    assert get(server + "/api/nothing")[0] == 404 and get(server + "/../manifest.json")[0] == 404


def test_server_without_frames_answers_404_for_the_frame(tmp_path: Path) -> None:
    httpd = make_server(tmp_path / "no-run-yet", "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_port}"
        assert get(base + "/api/frame")[0] == 404
        assert json.loads(get(base + "/api/state")[2])["status"] == "WAITING"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

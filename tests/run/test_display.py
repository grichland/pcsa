"""The dashboard frame: the newest captured frame, written at most `display_fps` times a second, never a capture."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from pcsa.common.clock import FakeClock
from pcsa.run.display import DisplayFrames, Throttle
from pcsa.run.watch import RunReader, display_frame
from tests.run.test_watch import write_run


def frame(value: int) -> np.ndarray:
    return np.full((9, 20, 3), value, dtype=np.uint8)


def test_throttle_allows_one_write_per_period_without_bursts() -> None:
    clock = FakeClock(100.0)
    throttle = Throttle(2.0, clock)  # every 0.5 s
    assert throttle.due() and not throttle.due()
    clock.advance(0.3)
    assert not throttle.due() and abs(throttle.wait_s() - 0.2) < 1e-9
    clock.advance(0.2)
    assert throttle.due()
    clock.advance(10.0)  # a quiet stretch earns one slot, not twenty
    assert throttle.due() and not throttle.due()


def test_only_a_new_frame_inside_an_open_slot_is_taken(tmp_path: Path) -> None:
    clock = FakeClock()
    display = DisplayFrames(tmp_path, 1.0, clock, start=False)
    assert display.take() is None  # nothing offered yet
    display.offer(frame(1))
    display.offer(frame(2))
    taken = display.take()
    assert taken is not None and taken[0, 0, 0] == 2  # the newest, the older one is skipped
    display.offer(frame(3))
    assert display.take() is None  # too soon
    clock.advance(1.0)
    taken = display.take()
    assert taken is not None and taken[0, 0, 0] == 3
    clock.advance(1.0)
    assert display.take() is None  # no new frame: no rewrite
    assert json.loads((tmp_path / "display.json").read_text(encoding="utf-8")) == {"fps": 1.0}


def test_the_writer_thread_saves_the_frame_and_the_watch_serves_it(tmp_path: Path) -> None:
    run = write_run(tmp_path)
    display = DisplayFrames(run, 5.0, FakeClock())
    display.offer(frame(7))
    landed = threading.Event()
    for _ in range(40):
        if (run / "display.jpg").is_file():
            break
        landed.wait(0.05)
    display.close()
    assert (run / "display.jpg").is_file() and not (run / "display.tmp.jpg").exists()
    found = display_frame(run)
    assert found is not None and found[0].startswith("display:")
    state = RunReader(run).state()
    assert state["frame_id"] == found[0] and state["display_fps"] == 5.0


def test_without_the_setting_the_watch_keeps_the_frames_archive(tmp_path: Path) -> None:
    run = write_run(tmp_path)
    (run / "frames").mkdir(exist_ok=True)
    (run / "frames" / "00001-poll-action.jpg").write_bytes(b"jpg")
    state = RunReader(run).state()
    assert display_frame(run) is None and state["display_fps"] is None
    assert state["frame_id"].startswith("00001-poll-action.jpg")

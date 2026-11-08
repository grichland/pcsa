"""The dashboard's view of the phone: the newest frame the loop already captured, saved at most `fps` times a second.

It never captures on its own (screencaps are slow and the decision loop owns the device). The loop hands over a
reference to each frame it observes; a background thread encodes the newest one to `display.jpg` (written to a
temporary file and swapped in, so `pcsa watch` never reads half a file) when the throttle allows and the frame is
new. `display.json` tells the page the rate. With `record.display_fps: 0` nothing here is built.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from pcsa.common import log
from pcsa.common.clock import Clock

NAME, META = "display.jpg", "display.json"
QUALITY = 80
MAX_FPS = 10.0


class Throttle:
    """At most one `due` per period; a quiet stretch does not build up a burst."""

    def __init__(self, fps: float, clock: Clock) -> None:
        self.period = 1.0 / min(max(fps, 1e-6), MAX_FPS)
        self.clock = clock
        self._next = float("-inf")

    def due(self) -> bool:
        now = self.clock.now()
        if now < self._next:
            return False
        self._next = now + self.period
        return True

    def wait_s(self) -> float:
        """Seconds until the next slot (0 when one is open)."""
        return max(0.0, self._next - self.clock.now())


class DisplayFrames:
    def __init__(self, run_dir: Path, fps: float, clock: Clock, *, start: bool = True) -> None:
        self.path = run_dir / NAME
        self.throttle = Throttle(fps, clock)
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._sequence = 0
        self._written = 0
        self._wake = threading.Event()
        self._stop = threading.Event()
        (run_dir / META).write_text(json.dumps({"fps": min(fps, MAX_FPS)}), encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="pcsa-display", daemon=True)
        if start:  # tests drive take/write directly
            self._thread.start()

    def offer(self, image: np.ndarray) -> None:
        """Called by the loop with each captured frame: keeps a reference, never blocks on disk."""
        with self._lock:
            self._latest = image
            self._sequence += 1
        self._wake.set()

    def take(self) -> np.ndarray | None:
        """The newest frame when it is new and the throttle allows a write, else None."""
        with self._lock:
            if self._sequence == self._written or self._latest is None or not self.throttle.due():
                return None
            self._written = self._sequence
            return self._latest

    def write(self, image: np.ndarray) -> None:
        temporary = self.path.with_suffix(".tmp.jpg")
        try:
            if cv2.imwrite(str(temporary), image, [cv2.IMWRITE_JPEG_QUALITY, QUALITY]):
                os.replace(temporary, self.path)
        except OSError as error:  # the page holding the file open on Windows: skip this frame
            log.warning("display_frame_failed", reason=str(error))

    def _run(self) -> None:
        while not self._stop.is_set():
            pause = self.throttle.wait_s()
            if pause > 0:  # frames arriving faster than the rate wait here, not in a busy loop
                self._stop.wait(pause)
                continue
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            image = self.take()
            if image is not None:
                self.write(image)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

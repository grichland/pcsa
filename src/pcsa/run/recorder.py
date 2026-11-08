"""The run directory: frames on a background writer, per-battle files, the manifest, `latest`."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pcsa.common import log, paths
from pcsa.common.types import jsonable

QUEUE_LIMIT = 64


class Recorder:
    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "frames").mkdir(exist_ok=True)
        (self.dir / "unknown").mkdir(exist_ok=True)
        self._queue: queue.Queue[tuple[Path, np.ndarray, list[int]] | None] = queue.Queue(maxsize=QUEUE_LIMIT)
        self._thread = threading.Thread(target=self._writer, name="pcsa-frames", daemon=True)
        self._thread.start()
        self.dropped = 0

    @classmethod
    def new(cls, name: str = "run") -> Recorder:
        run_dir = paths.runs_dir() / f"{datetime.now():%Y%m%d-%H%M%S}-{name}"
        recorder = cls(run_dir)
        recorder.point_latest()
        return recorder

    def point_latest(self) -> None:
        """output/latest -> this run (a directory junction on Windows, a symlink elsewhere)."""
        link = paths.output_dir() / "latest"
        try:
            if link.is_symlink() or link.exists():
                if link.is_dir() and not link.is_symlink():
                    os.rmdir(link)
                else:
                    link.unlink()
            os.symlink(self.dir, link, target_is_directory=True)
        except OSError as error:
            log.warning("latest_link_failed", reason=str(error))

    def save_frame(self, relative: str, image: np.ndarray, png: bool = False) -> str:
        path = self.dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1] if png else [cv2.IMWRITE_JPEG_QUALITY, 85]
        try:
            self._queue.put_nowait((path, image, params))
        except queue.Full:
            self.dropped += 1
            log.warning("frame_dropped", relative=relative, dropped=self.dropped)
        return relative

    def _writer(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            path, image, params = item
            try:
                cv2.imwrite(str(path), image, params)
            except Exception as error:
                log.warning("frame_write_failed", path=str(path), reason=str(error))

    def write_json(self, relative: str, value: Any) -> Path:
        path = self.dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def write_text(self, relative: str, text: str) -> Path:
        path = self.dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30)


def manifest(seed: int, player_name: str, player_params: dict, config_path: Path, device: dict) -> dict:
    """What produced this run: code and asset hashes, pins, player, device."""
    src = paths.repo_root() / "src" / "pcsa"
    digest = hashlib.sha256()
    for path in sorted(src.rglob("*.py")):
        digest.update(path.read_bytes())
    assets = {}
    for path in sorted((paths.assets_dir()).rglob("*")):
        if path.is_file() and path.suffix in (".json", ".npz", ".onnx"):
            assets[path.relative_to(paths.assets_dir()).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()[
                :16
            ]
    lock = json.loads(paths.tools_lock().read_text(encoding="utf-8"))
    return {
        "schema": "pcsa-run-manifest/1",
        "seed": seed,
        "player": player_name,
        "player_params": player_params,
        "config": str(config_path),
        "source_sha256": digest.hexdigest(),
        "assets": assets,
        "showdown_commit": lock["showdown"]["commit"],
        "device": device,
    }

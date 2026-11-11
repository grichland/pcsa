"""`pcsa watch`: a read-only page over a run directory (events, decisions, battle log, latest frame). No controls.

The server only reads the run folder, and it tolerates a run that is being written: files that are missing,
empty or half written show as empty panels until they land. It listens on localhost.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PureWindowsPath
from urllib.parse import parse_qs, urlsplit

from pcsa.common import paths
from pcsa.run.watch_battle import battle_dir, battle_view
from pcsa.run.watch_decisions import DecisionRows, folder_key, notes_view, read_json
from pcsa.run.watch_events import EventScan
from pcsa.run.watch_log import log_view

STATIC = Path(__file__).with_name("static")
ASSETS = {
    "/style.css": "text/css",
    "/app.js": "text/javascript",
    "/matrix.js": "text/javascript",
    "/panels.js": "text/javascript",
}
STALE_S = 35
SETTLE_S = 1.0  # a frame younger than this may still be on its way to disk: the one before it is shown
SUMMARY_KEYS = ("checkpoint", "update", "sampling")
VIEW_ERRORS = (AttributeError, IndexError, KeyError, TypeError, ValueError, OSError)


def latest_run() -> Path | None:
    runs = (
        [p for p in paths.runs_dir().glob("*") if (p / "events.jsonl").is_file()] if paths.runs_dir().is_dir() else []
    )
    return max(runs, key=lambda p: (p / "events.jsonl").stat().st_mtime_ns) if runs else None


def latest_frame(run: Path) -> tuple[str, Path] | None:
    """(id, path) of the newest frame the run archived under frames/ (never a capture of its own)."""
    try:
        names = sorted(e.name for e in os.scandir(run / "frames") if e.name.endswith(".jpg"))[-2:]
    except OSError:
        return None
    found = []
    for name in reversed(names):
        path = run / "frames" / name
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size:
            found.append((f"{name}:{stat.st_mtime_ns}", path, stat.st_mtime))
    if not found:
        return None
    settled = datetime.now(UTC).timestamp() - found[0][2] >= SETTLE_S
    return found[0][:2] if settled or len(found) == 1 else found[1][:2]


def player_view(manifest: dict) -> dict:
    """The player kind and its parameters from the run manifest; file paths are shown by name."""
    raw = manifest.get("player_params")
    params = []
    for key, value in (raw if isinstance(raw, dict) else {}).items():
        if value is None:
            continue
        full = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        # PureWindowsPath splits on both separators on every OS (a manifest written on Windows read on Linux).
        short = PureWindowsPath(full).name if isinstance(value, str) and ("/" in full or "\\" in full) else full
        params.append({"key": str(key), "value": short, "full": full})
    by_key = {p["key"]: p["value"] for p in params}
    parts = [f"update {by_key[k]}" if k == "update" else by_key[k] for k in SUMMARY_KEYS if k in by_key]
    return {"kind": manifest.get("player"), "params": params, "summary": " · ".join(parts)}


class RunReader:
    def __init__(self, run_dir: Path) -> None:
        self.run = run_dir
        self.events = run_dir / "events.jsonl"
        self._lock = threading.RLock()
        self._scan = EventScan(self.events)
        self._rows = DecisionRows()
        self._key: tuple | None = None
        self._state: dict = {}

    def state(self, battle: int | None = None) -> dict:
        """The page's whole view; `battle` pins one battle of the run (default: the newest)."""
        with self._lock:
            self._scan.refresh()
            frame = latest_frame(self.run)
            current = self._current(battle)
            folder = battle_dir(self.run, current["index"] if current else None)
            key = (self._scan.offset, battle, folder_key(folder), frame[0] if frame else None)
            if key != self._key or not self._state:
                self._key, self._state = key, self._build(current, folder, frame, battle)
            self._state["age_s"] = self._age()
            self._state["status"] = self._status(self._state["age_s"])
            return self._state

    def payload(self, battle: int | None = None) -> bytes:
        """`state` as JSON, encoded while the run summary cannot change under it."""
        with self._lock:
            return json.dumps(self.state(battle), ensure_ascii=False, default=str).encode("utf-8")

    def frame_path(self) -> Path | None:
        frame = latest_frame(self.run)
        return frame[1] if frame else None

    def _current(self, battle: int | None) -> dict | None:
        battles = self._scan.battles
        if battle is not None:
            return battles.get(battle) or {"index": battle, "outcome": None}
        return battles[max(battles)] if battles else None

    def _age(self) -> float:
        try:
            return round(max(0.0, datetime.now(UTC).timestamp() - self.events.stat().st_mtime), 1)
        except OSError:
            return 0.0

    def _status(self, age: float) -> str:
        if not self._scan.last:
            return "WAITING"
        if self._scan.last.get("event") == "run_end":
            return "STOPPED"
        return "LIVE" if age <= STALE_S else "STALE"

    def _decisions(self, current: dict | None, folder: Path | None) -> list[dict]:
        rows = self._rows.rows(folder) if current else []
        for row in rows:
            if row["notes"] and row["notes"].get("matrix"):  # the list stays light: only the latest decision's matrix
                row["notes"] = {**row["notes"], "matrix": None}
            if row["kind"] == "action" and isinstance(row["visit"], int):
                row["timing"] = self._scan.timings.get((current["index"], row["visit"])) if current else None
        return rows

    def _build(self, current: dict | None, folder: Path | None, frame, pinned: int | None) -> dict:
        scan = self._scan
        manifest = read_json(self.run / "manifest.json") or {}
        screen = scan.latest_obs.get("screen")
        preview = (current or {}).get("preview")
        try:
            view = battle_view(self.run, current, screen)
        except VIEW_ERRORS:  # a record of an unexpected shape: the battle panels stay empty
            view = {}
        return {
            "run_name": self.run.name,
            "player": manifest.get("player"),
            "player_view": player_view(manifest),
            "screen": screen,
            "turn": None,
            "frame_id": frame[0] if frame else None,
            "battle": current,
            "pinned": pinned,
            "battles": [scan.battles[i] for i in sorted(scan.battles)],
            "session_record": scan.record,
            "rating": scan.rating_view(),
            "decision": None,
            "decisions": self._decisions(current, folder),
            "preview": {**preview, "notes": notes_view(preview.get("notes"))} if preview else None,
            "winrate": scan.winrate_view(),
            "log": log_view(folder if current else None),
            "battle_state": {"battle": str(current["index"]) if current else "pending", "screen": str(screen)},
            "timeline": list(scan.timeline),
            "warnings": list(scan.warnings),
            "warning_count": scan.warning_count,
            "disconnects": scan.disconnects,
            "last_event_at": scan.last.get("ts"),
            **view,
        }


class Handler(BaseHTTPRequestHandler):
    reader: RunReader

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # the http.server API names it
        url = urlsplit(self.path)
        path = url.path
        if path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path in ASSETS:
            return self._send(200, (STATIC / path.lstrip("/")).read_bytes(), f"{ASSETS[path]}; charset=utf-8")
        if path == "/api/state":
            wanted = parse_qs(url.query).get("battle", [""])[0]
            body = self.reader.payload(int(wanted) if wanted.isdigit() else None)
            return self._send(200, body, "application/json; charset=utf-8")
        if path == "/api/frame":
            frame = self.reader.frame_path()
            try:
                return self._send(200, frame.read_bytes(), "image/jpeg") if frame else self._no_frame()
            except OSError:
                return self._no_frame()
        self._send(404, b"Not found", "text/plain")

    def _no_frame(self) -> None:
        self._send(404, b"No frame yet", "text/plain")

    def log_message(self, format: str, *args) -> None:  # the http.server API names it
        pass


class Server(ThreadingHTTPServer):
    allow_reuse_address = False  # a second dashboard on a taken port fails instead of sharing it (Windows)
    daemon_threads = True


def make_server(run_dir: Path, host: str, port: int) -> Server:
    handler = type("BoundHandler", (Handler,), {"reader": RunReader(run_dir)})
    return Server((host, port), handler)


def serve(run_dir: Path, host: str, port: int, open_browser: bool) -> int:
    server = make_server(run_dir, host, port)
    if open_browser:
        webbrowser.open(f"http://{host}:{server.server_port}/")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0

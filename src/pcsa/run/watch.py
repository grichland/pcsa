"""`pcsa watch`: a read-only page over a run directory (events, latest frame, battle table). No controls."""

from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from collections import deque
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from pcsa.common import paths
from pcsa.run.watch_battle import battle_view, record_key

STATIC = Path(__file__).with_name("static")
IMPORTANT = {
    "battle_start",
    "battle_end",
    "decision",
    "turn",
    "log_visit",
    "navigate",
    "unknown_screen",
    "session_restart",
    "device_retry",
    "device_reconnect",
    "team_read",
    "rank",
    "run_end",
    "safety_stop",
    "turn_overrun",
    "tracking_failed",
}
DISPLAY, DISPLAY_META = "display.jpg", "display.json"
STALE_S = 35
TIMELINE = 40
DECISIONS = 12


def latest_run() -> Path | None:
    runs = (
        [p for p in paths.runs_dir().glob("*") if (p / "events.jsonl").is_file()] if paths.runs_dir().is_dir() else []
    )
    return max(runs, key=lambda p: (p / "events.jsonl").stat().st_mtime_ns) if runs else None


def display_frame(run: Path) -> tuple[str, Path] | None:
    """The dashboard frame the loop saves at `record.display_fps` (display.jpg), as (id, path)."""
    path = run / DISPLAY
    try:
        return f"display:{path.stat().st_mtime_ns}", path
    except OSError:
        return None


def latest_frame(run: Path) -> tuple[str | None, str | None]:
    """(id, path) of the frame to show: display.jpg when the run saves one, else the newest archived frame."""
    display = display_frame(run)
    if display is not None:
        return display[0], str(display[1])
    frames = sorted((run / "frames").glob("*.jpg")) if (run / "frames").is_dir() else []
    if not frames:
        return None, None
    return f"{frames[-1].name}:{frames[-1].stat().st_mtime_ns}", str(frames[-1])


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class _Scan:
    """What one pass over events.jsonl accumulates."""

    def __init__(self) -> None:
        self.timeline: deque = deque(maxlen=TIMELINE)
        self.decisions: deque = deque(maxlen=DECISIONS)
        self.latest_obs: dict = {}
        self.latest_moves: dict = {}
        self.battles: dict[int, dict] = {}
        self.record = {"wins": 0, "losses": 0, "draws": 0, "forfeits": 0}
        self.last: dict = {}

    def take(self, event: dict) -> None:
        self.last = event
        kind = event.get("event")
        if kind == "observation":
            self.latest_obs = event
            if (event.get("fields") or {}).get("moves"):
                self.latest_moves = event
        elif kind == "decision":
            self.decisions.append(event)
        elif kind == "battle_start":
            self.battles[event["battle"]] = {"index": event["battle"], "outcome": None, "roster": event.get("roster")}
        elif kind == "battle_end":
            outcome = event.get("outcome")
            self.battles.setdefault(event["battle"], {"index": event["battle"]})["outcome"] = outcome
            key = {"win": "wins", "loss": "losses", "draw": "draws", "opponent_forfeit": "forfeits"}.get(str(outcome))
            if key:
                self.record[key] += 1
        if kind in IMPORTANT:
            detail = ", ".join(f"{k}={v}" for k, v in event.items() if k not in ("ts", "level", "event"))
            self.timeline.append({"ts": event.get("ts"), "label": kind, "detail": detail[:160], "kind": kind})


class RunReader:
    def __init__(self, run_dir: Path) -> None:
        self.run = run_dir
        self._lock = threading.Lock()
        self._key: tuple | None = None
        self._state: dict = {}

    def state(self) -> dict:
        with self._lock:
            events_path = self.run / "events.jsonl"
            key = (events_path.stat().st_mtime_ns if events_path.is_file() else 0, record_key(self.run))
            if key == self._key and self._state:
                self._state["age_s"] = self._age(events_path)
                return self._state
            self._key, self._state = key, self._build(events_path)
            return self._state

    @staticmethod
    def _age(events_path: Path) -> float:
        return (
            round(max(0.0, datetime.now(UTC).timestamp() - events_path.stat().st_mtime), 1)
            if events_path.is_file()
            else 0.0
        )

    def _build(self, events_path: Path) -> dict:
        scan = _Scan()
        if events_path.is_file():
            with events_path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        scan.take(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        age = self._age(events_path)
        last = scan.last
        stopped = last.get("event") == "run_end"
        battles, decisions, latest_obs, latest_moves, record, timeline = (
            scan.battles,
            scan.decisions,
            scan.latest_obs,
            scan.latest_moves,
            scan.record,
            scan.timeline,
        )
        current = max(battles.values(), key=lambda b: b["index"]) if battles else None
        frame_id, frame_path = latest_frame(self.run)
        manifest = _read_json(self.run / "manifest.json") or {}
        return {
            "status": "STOPPED" if stopped else "LIVE" if age <= STALE_S else "STALE",
            "run_name": self.run.name,
            "age_s": age,
            "policy": manifest.get("player"),
            "live_stream": False,
            "screen": latest_obs.get("screen"),
            "screen_reason": latest_obs.get("reason"),
            "turn": None,
            "frame_id": frame_id,
            "frame_path": frame_path,
            "display_fps": (_read_json(self.run / DISPLAY_META) or {}).get("fps"),
            "frame_type": "image/jpeg",
            "battle": current,
            "session_record": record,
            "active": {"name": None, "hp": None},
            "opponent": {"name": None, "preview": []},
            "moves": (latest_moves.get("fields") or {}).get("moves") or [],
            "decision": None,
            "decisions": list(decisions),
            "team": [],
            "battle_count": len(battles),
            "completed_battles": sum(1 for b in battles.values() if b.get("outcome")),
            "summary": _read_json(self.run / "summary.json"),
            "battle_state": {
                "battle": str(current["index"]) if current else "pending",
                "screen": str(latest_obs.get("screen")),
            },
            "timeline": list(timeline),
            "last_event_at": last.get("ts"),
            **battle_view(self.run, current, latest_obs.get("screen")),
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
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path in ("/style.css", "/app.js"):
            asset = STATIC / path.lstrip("/")
            kind = "text/css" if asset.suffix == ".css" else "text/javascript"
            return self._send(200, asset.read_bytes(), f"{kind}; charset=utf-8")
        if path == "/api/state":
            return self._send(
                200,
                json.dumps(self.reader.state(), ensure_ascii=False, default=str).encode("utf-8"),
                "application/json; charset=utf-8",
            )
        if path == "/api/frame-id":  # cheap: the page polls it at the display rate
            display = display_frame(self.reader.run)
            body = {"id": display[0] if display else self.reader.state().get("frame_id")}
            return self._send(200, json.dumps(body).encode("utf-8"), "application/json; charset=utf-8")
        if path == "/api/frame":
            display = display_frame(self.reader.run)
            frame = str(display[1]) if display else self.reader.state().get("frame_path")
            if not frame or not Path(frame).is_file():
                return self._send(404, b"No frame yet", "text/plain")
            return self._send(200, Path(frame).read_bytes(), mimetypes.guess_type(frame)[0] or "image/jpeg")
        self._send(404, b"Not found", "text/plain")

    def log_message(self, format: str, *args) -> None:  # the http.server API names it
        pass


def serve(run_dir: Path, host: str, port: int, open_browser: bool) -> int:
    reader = RunReader(run_dir)
    handler = type("BoundHandler", (Handler,), {"reader": reader})
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_port}/"
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0

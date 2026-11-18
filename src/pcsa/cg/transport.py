"""How LIVE messages reach CG: in process (a Python entry point) or a subprocess speaking JSON lines.

`exchange` sends the messages in order and returns the first reply `wanted` accepts, with every other reply (an
answer to an older seq, or to a request sent again after a restart). `generation` rises on every (re)start: a CG
that started after a message was sent has not seen it.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from pcsa.cg.startup import with_log_dir
from pcsa.common.clock import SystemClock
from pcsa.common.paths import repo_root

REQUESTS = ("preview_request", "decision_request")
WALL = SystemClock()  # waits for CG are real time, whatever clock the controller runs on
Record = Callable[[str, Any], None]  # (direction "sent" | "received" | "note", message)
Wanted = Callable[[dict], bool] | None  # None: no reply expected
Handle = Callable[[dict], dict | None]


class CGError(Exception):
    """CG's answer cannot be used; the runner falls back."""


class CGTimeout(CGError):
    pass


class TransportFailed(CGError):
    """CG is gone or broken: restart it and send the stream again from battle_start."""


class Transport(Protocol):
    kind: str
    generation: int

    def ensure(self) -> int: ...

    def exchange(self, messages: list[dict], wanted: Wanted, timeout: float, record: Record) -> tuple: ...

    def restart(self, reason: str) -> None: ...

    def close(self) -> None: ...


def search_path(python_path: list[str] | None) -> list[str]:
    """cg.python_path as absolute directories; a relative one is taken from the repo root."""
    return [str((repo_root() / p).resolve()) for p in python_path or ()]


def resolve(entry: str, args: dict | None = None) -> Handle:
    """ "module:name": a class is instantiated with `args` as keyword arguments; an object with .handle gives
    that; else the callable itself."""
    module, _, name = entry.partition(":")
    target: Any = importlib.import_module(module)
    for part in name.split("."):
        target = getattr(target, part)
    if isinstance(target, type):
        try:
            target = target(**(args or {}))
        except (TypeError, ValueError) as error:
            raise CGError(f"{entry}({args or {}}): {error}") from error
    elif args:
        raise CGError(f"cg.args needs an entry that is a class: {entry}")
    handle = getattr(target, "handle", None)
    if callable(handle):
        return cast(Handle, handle)
    if callable(target):
        return cast(Handle, target)
    raise CGError(f"{entry} is neither callable nor has handle()")


def _copy(message: dict) -> dict:
    return json.loads(json.dumps(message))  # CG gets plain JSON, as over a pipe


def _collect(box: queue.Queue, wanted: Wanted, timeout: float, record: Record) -> tuple[dict | None, list[dict]]:
    """Replies off `box` until the wanted one; ("done", None) ends the messages; ("failed", reason) ends CG."""
    until, others = WALL.now() + timeout, []
    while True:
        try:
            kind, value = box.get(timeout=max(0.0, until - WALL.now()))
        except queue.Empty:
            raise CGTimeout(f"no answer within {timeout:.1f}s") from None
        if kind == "failed":
            raise TransportFailed(value)
        if kind == "done":
            if wanted is None:
                return None, others
            errors = [o["error"] for o in others if isinstance(o, dict) and "error" in o]
            raise CGError("CG sent no answer to the request" + (f": {errors[-1]}" if errors else ""))
        if kind == "unparsed":
            record("note", {"unparsed": value})
            continue
        record("received", value)
        if wanted is not None and isinstance(value, dict) and wanted(value):
            return value, others
        if value is not None:
            others.append(value)


class InProcess:
    """CG's object lives on one thread of its own, created there and only ever called there (CG's Rust core may
    not be used from another thread). A timeout or a crash leaves that thread and starts a new one."""

    kind = "inprocess"

    def __init__(self, entry: str, args: dict | None = None, python_path: list[str] | None = None) -> None:
        self.entry, self.args = entry, dict(args or {})
        self.python_path = search_path(python_path)
        self.generation = 0
        self._jobs: queue.Queue | None = None

    def ensure(self) -> int:
        if self._jobs is None:
            jobs: queue.Queue = queue.Queue()
            ready: queue.Queue = queue.Queue()
            threading.Thread(target=self._serve, args=(jobs, ready), name="cg-inprocess", daemon=True).start()
            error = ready.get()
            if error is not None:
                raise error
            self._jobs = jobs
            self.generation += 1
        return self.generation

    def _serve(self, jobs: queue.Queue, ready: queue.Queue) -> None:
        try:
            sys.path[:0] = [p for p in self.python_path if p not in sys.path]
            handle = resolve(self.entry, self.args)
        except Exception as error:
            ready.put(error)
            return
        ready.put(None)
        while (job := jobs.get()) is not None:
            messages, box, record = job
            try:
                for message in messages:
                    record("sent", message)
                    reply = handle(_copy(message))
                    if reply is not None:
                        box.put(("reply", _copy(reply)))
                box.put(("done", None))
            except BaseException as error:  # a Rust panic is a BaseException
                box.put(("failed", f"CG raised {type(error).__name__}: {error}"))
                return

    def exchange(self, messages: list[dict], wanted: Wanted, timeout: float, record: Record) -> tuple:
        assert self._jobs is not None, "ensure() first"
        box: queue.Queue = queue.Queue()
        self._jobs.put((messages, box, record))
        try:
            return _collect(box, wanted, timeout, record)
        except CGTimeout:
            self.restart("timeout")  # the old object may still be running: never call it again
            raise

    def restart(self, reason: str) -> None:
        jobs, self._jobs = self._jobs, None
        if jobs is not None:
            jobs.put(None)  # its thread ends once it is free

    def close(self) -> None:
        self.restart("close")


class Subprocess:
    kind = "subprocess"

    def __init__(
        self,
        command: list[str],
        stderr_path: Path | None = None,
        python_path: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.command = list(command)
        self.stderr_path = stderr_path
        self.env: dict[str, str] | None = None  # None: the controller's own environment
        if python_path or env:
            self.env = merged = {**os.environ, **(env or {})}
            if python_path:
                inherited = [merged["PYTHONPATH"]] if merged.get("PYTHONPATH") else []
                merged["PYTHONPATH"] = os.pathsep.join(search_path(python_path) + inherited)
        self.generation = 0
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue = queue.Queue()

    def ensure(self) -> int:
        if self.proc is not None and self.proc.poll() is None:
            return self.generation
        self._stop()
        stderr: Any = subprocess.DEVNULL
        if self.stderr_path is not None:
            self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
            stderr = self.stderr_path.open("a", encoding="utf-8")
        try:
            self.proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, encoding="utf-8", bufsize=1, env=self.env,
            )  # fmt: skip
        except OSError as error:
            raise TransportFailed(f"cannot start {self.command}: {error}") from error
        finally:
            if stderr is not subprocess.DEVNULL:
                stderr.close()
        self.lines = queue.Queue()
        threading.Thread(target=self._read, args=(self.proc, self.lines), name="cg-stdout", daemon=True).start()
        self.generation += 1
        return self.generation

    @staticmethod
    def _read(proc: subprocess.Popen, lines: queue.Queue) -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except ValueError:
                    lines.put(("unparsed", line.rstrip("\n")))
                    continue
                if value is not None:  # `null`: the message had no answer, as None in process
                    lines.put(("reply", value))
        except (OSError, ValueError):
            pass  # the pipe was closed by a restart
        lines.put(("failed", f"CG exited (code {proc.wait()})"))

    def exchange(self, messages: list[dict], wanted: Wanted, timeout: float, record: Record) -> tuple:
        proc = self.proc
        assert proc is not None and proc.stdin is not None, "ensure() first"
        late = self._drain(record)
        try:
            for message in messages:
                record("sent", message)
                proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        except OSError as error:
            raise TransportFailed(f"CG stopped reading: {error}") from error
        if wanted is None:
            return None, late
        reply, others = _collect(self.lines, wanted, timeout, record)
        return reply, late + others

    def _drain(self, record: Record) -> list[dict]:
        """Replies that came after their wait ended: kept in the record, never taken as a new answer."""
        late = []
        while True:
            try:
                kind, value = self.lines.get_nowait()
            except queue.Empty:
                return late
            if kind == "failed":
                self.lines.put((kind, value))  # read again by the wait that follows
                return late
            record("received" if kind == "reply" else "note", value if kind == "reply" else {"unparsed": value})
            late.append(value)

    def restart(self, reason: str) -> None:
        self._stop()

    def _stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass

    def close(self) -> None:
        self._stop()


def make_transport(cfg: Any, run_dir: Path | None) -> Transport:
    """From the config's cg section (common.config.CGConfig); CG's log_dir goes in the run folder when
    cg.run_log_dir is set."""
    if run_dir is not None and getattr(cfg, "run_log_dir", None):
        cfg = with_log_dir(cfg, Path(run_dir) / cfg.run_log_dir)
    if cfg.transport == "inprocess":
        if not cfg.entry:
            raise CGError("cg.entry is needed for the inprocess transport")
        return InProcess(cfg.entry, cfg.args, cfg.python_path)
    if not cfg.command:
        raise CGError("cg.command is needed for the subprocess transport")
    stderr = run_dir / "cg-stderr.log" if run_dir is not None else None
    return Subprocess(cfg.command, stderr, cfg.python_path, getattr(cfg, "env", None))

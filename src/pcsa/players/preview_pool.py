"""The playout workers of the playout team preview: child processes (pcsa.players.preview_worker), each with its
own Showdown server, started once and kept for the session at below-normal priority. Tasks go in as JSON lines,
results come back on one queue. A worker that is still playing an abandoned game is simply busy until it answers.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

BELOW_NORMAL = 0x00004000  # Windows BELOW_NORMAL_PRIORITY_CLASS (the Showdown child inherits it)
NEW_GROUP = 0x00000200  # CREATE_NEW_PROCESS_GROUP


class WorkerProcess:
    def __init__(self, index: int, spec: dict, folder: Path, results: queue.Queue) -> None:
        self.index, self.results = index, results
        self.ready = False
        self.error: str | None = None
        self.busy: object = None  # the id of the task it is playing
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / f"worker-{os.getpid()}-{index}"
        self._err = open(f"{stem}.err", "a", encoding="utf-8")  # noqa: SIM115 - lives as long as the worker
        flags = (BELOW_NORMAL | NEW_GROUP) if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "pcsa.players.preview_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._err,
            text=True,
            encoding="utf-8",
            creationflags=flags,
        )
        self._write({**spec, "log": f"{stem}.server.log"})
        threading.Thread(target=self._read, name=f"playout-worker-{index}", daemon=True).start()

    def _write(self, record: dict) -> bool:
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(record) + "\n")
            self.proc.stdin.flush()
            return True
        except (OSError, ValueError):
            return False

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if "ready" in record:
                self.ready, self.error = bool(record["ready"]), record.get("error")
                continue
            self.busy = None
            self.results.put({**record, "worker": self.index})
        self.ready = False  # the process is gone

    def alive(self) -> bool:
        return self.proc.poll() is None

    def idle(self) -> bool:
        return self.ready and self.busy is None and self.alive()

    def send(self, task: dict) -> bool:
        self.busy = task["id"]
        if self._write(task):
            return True
        self.busy, self.ready = None, False
        return False

    def close(self) -> None:
        self._write({"quit": True})
        with contextlib.suppress(Exception):
            assert self.proc.stdin is not None
            self.proc.stdin.close()
        try:
            self.proc.wait(5.0)
        except subprocess.TimeoutExpired:
            if os.name == "nt":  # the tree: the worker and its Showdown server
                subprocess.run(
                    ["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                self.proc.kill()
        with contextlib.suppress(Exception):
            self._err.close()


class Pool:
    """`n` workers started at once without waiting; `idle()` lists the ones that can take a task now."""

    def __init__(self, n: int, spec: dict, folder: Path) -> None:
        self.results: queue.Queue = queue.Queue()
        self.workers = [WorkerProcess(i, spec, folder, self.results) for i in range(n)]
        atexit.register(self.close)

    def idle(self) -> list[WorkerProcess]:
        return [w for w in self.workers if w.idle()]

    def ready(self) -> int:
        return sum(w.ready and w.alive() for w in self.workers)

    def errors(self) -> list[str]:
        return [w.error for w in self.workers if w.error]

    def close(self) -> None:
        workers, self.workers = self.workers, []
        for w in workers:
            w.close()

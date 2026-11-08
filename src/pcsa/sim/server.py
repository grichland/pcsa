"""A local Showdown server for poke-env players (no login, any username), as a child process."""

from __future__ import annotations

import os
import socket
import subprocess
import threading
from pathlib import Path

from poke_env import ServerConfiguration

from pcsa.common.clock import Clock, SystemClock
from pcsa.sim.engine import Engine

LOCALHOST = "127.0.0.1"
POLL_S = 0.1
KEPT_LINES = 500


def server_configuration(port: int) -> ServerConfiguration:
    return ServerConfiguration(f"ws://localhost:{port}/showdown/websocket", None)  # type: ignore[arg-type]


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((LOCALHOST, port)) == 0


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((LOCALHOST, 0))
        return s.getsockname()[1]


class ShowdownServer:
    def __init__(
        self,
        engine: Engine,
        port: int | None = None,
        clock: Clock | None = None,
        startup_timeout_s: float = 60.0,
        log_path: Path | None = None,
    ) -> None:
        self.engine = engine
        self.port = port or free_port()
        self.clock = clock or SystemClock()
        self.startup_timeout_s = startup_timeout_s
        self.log_path = log_path
        self.process: subprocess.Popen | None = None
        self.output: list[str] = []
        self._ready = threading.Event()
        self._reader: threading.Thread | None = None

    @property
    def configuration(self) -> ServerConfiguration:
        return server_configuration(self.port)

    def start(self) -> ShowdownServer:
        if self.process is not None:
            return self
        if port_open(self.port):
            raise RuntimeError(f"port {self.port} is already in use")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.process = subprocess.Popen(
            [str(self.engine.node), "pokemon-showdown", "start", "--skip-build", "--no-security", str(self.port)],
            cwd=self.engine.showdown,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        started = self.clock.now()
        while self.clock.now() - started < self.startup_timeout_s:
            if self.process.poll() is not None:
                raise RuntimeError("Showdown exited during startup:\n" + "\n".join(self.output[-30:]))
            if self._ready.is_set() and port_open(self.port):
                return self
            self.clock.sleep(POLL_S)
        self.stop()
        raise TimeoutError(f"Showdown did not listen on port {self.port} within {self.startup_timeout_s}s")

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        log = open(self.log_path, "a", encoding="utf-8") if self.log_path else None  # noqa: SIM115
        try:
            for raw in self.process.stdout:
                line = raw.rstrip("\n")
                self.output.append(line)
                del self.output[:-KEPT_LINES]
                if log:
                    log.write(line + "\n")
                    log.flush()
                if "now listening on" in line:
                    self._ready.set()
        finally:
            if log:
                log.close()

    def stop(self, timeout_s: float = 10.0) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.terminate()
        try:
            process.wait(timeout_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout_s)
        if self._reader:
            self._reader.join(timeout_s)
        self._ready.clear()

    def __enter__(self) -> ShowdownServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

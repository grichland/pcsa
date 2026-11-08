"""Where the pinned Node and Showdown live, and the one check that they are installed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pcsa.common import paths
from pcsa.common.errors import ConfigError
from pcsa.common.tools import load_lock

JS_DIR = Path(__file__).with_name("js")


@dataclass(frozen=True)
class Engine:
    node: Path
    showdown: Path
    format: str

    @classmethod
    def locate(cls, fmt: str) -> Engine:
        """The installed engine, or a ConfigError naming the setup command."""
        lock = load_lock()
        node = paths.tool(lock["node"]["into"]) / "node.exe"
        showdown = paths.tool(lock["showdown"]["into"])
        marker = showdown / ".pcsa-built"
        if not node.is_file() or not (showdown / "dist" / "sim").is_dir() or not marker.is_file():
            raise ConfigError("the simulator is not installed; run `pcsa setup --with sim`")
        commit = marker.read_text(encoding="utf-8").strip()
        if commit != lock["showdown"]["commit"]:
            raise ConfigError(f"Showdown build {commit[:10]} is not the pinned {lock['showdown']['commit'][:10]}")
        return cls(node, showdown, fmt)

    def script(self, name: str) -> Path:
        return JS_DIR / name

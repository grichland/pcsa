"""Team legality through Showdown's own validator (js/validate.cjs)."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from pcsa.sim.engine import Engine

TIMEOUT_S = 120.0


def validate(engine: Engine, exports: list[str]) -> list[dict]:
    """One {index, ok, problems} per export text."""
    if not exports:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        teams_path = Path(tmp) / "teams.json"
        teams_path.write_text(json.dumps(exports), encoding="utf-8")
        done = subprocess.run(
            [
                str(engine.node),
                str(engine.script("validate.cjs")),
                str(engine.showdown),
                engine.format,
                str(teams_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=TIMEOUT_S,
        )
    return json.loads(done.stdout)

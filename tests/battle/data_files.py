"""Saved reference results (recorded before the Champions Gym port), loaded once."""

from __future__ import annotations

import gzip
import json
from functools import cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


@cache
def load(name: str) -> dict:
    with gzip.open(DATA / name, "rt", encoding="utf-8") as f:
        return json.load(f)


def plain(value):
    """Tuples and sets as JSON gives them back."""
    return json.loads(json.dumps(value, default=sorted))

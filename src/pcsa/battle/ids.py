"""Showdown ids: the same rule as poke-env's to_id_str, so ids match pcsa's records."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2**13)
def to_id(text: str) -> str:
    """Letters and digits only (Unicode isalnum, so "é" stays), lower-cased."""
    return "".join(char for char in text if char.isalnum()).lower()

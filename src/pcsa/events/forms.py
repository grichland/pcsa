"""Their Pokémon named by the form they were brought in (Rotom Wash, Alolan Ninetales). The log's rows leave the
form out; the team preview shows it, and a move only one form learns names it where the preview did not.

`with_forms` gives recorded controller messages the form ids, for battles recorded before the live side sent them.
"""

from __future__ import annotations

import copy
from typing import Any

from pcsa.events.live_ids import base_id, form_by_move, preview_id


def rename(value: Any, was: str, now: str) -> None:
    """Their Pokémon `was` is `now` in an event and in the Pokémon it names inside (`of`)."""
    if isinstance(value, dict):
        if value.get("mon") == was and value.get("side") != "own":
            value["mon"] = now
        value = list(value.values())
    if isinstance(value, list):
        for inner in value:
            rename(inner, was, now)


def forms_of(messages: list[dict], preview_names: list[str | None]) -> dict[str, str]:
    """Species id as sent -> form id: from the preview names, else from a move only one form learns."""
    forms = {base_id(n): preview_id(n) for n in preview_names if n and preview_id(n) != base_id(n)}
    for message in messages:
        for e in message.get("events") or () if message.get("type") == "events" else ():
            if e.get("k") == "move" and e.get("side") == "opp" and e.get("mon"):
                found = form_by_move(forms.get(e["mon"], e["mon"]), str(e.get("move") or ""))
                if found is not None:
                    forms[e["mon"]] = found
    return forms


def with_forms(messages: list[dict], preview_names: list[str | None]) -> list[dict]:
    """The messages with their Pokémon named by form in battle_start's preview and in every event; the same
    list when no form is known. Our own Pokémon and CG's answers are left as they are."""
    forms = forms_of(messages, preview_names)
    if not forms:
        return messages
    out = copy.deepcopy(messages)
    for message in out:
        if message.get("type") == "battle_start":
            message["opp_preview"] = [forms.get(i, i) if i else i for i in message.get("opp_preview") or ()]
        elif message.get("type") == "events":
            for was, now in forms.items():
                rename(message.get("events"), was, now)
    return out

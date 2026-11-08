"""Moves: base move data with Champions PP, move names and observed forms."""

from __future__ import annotations

import json
from functools import lru_cache

from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.game import mega_form_for_stone, species_entry


def reconcile_form(current: str, candidate: str) -> str:
    """A generic Mimikyu icon cannot restore a Disguise that was already busted."""
    current, candidate = to_id_str(current), to_id_str(candidate)
    intact = {"mimikyubusted": "mimikyu", "mimikyubustedtotem": "mimikyutotem"}
    return current if intact.get(current) == candidate else candidate


def observed_form(battle, mon) -> str:
    """The latest form in this Battle's received protocol (poke-env keeps the base species). Cached on the
    Battle per member and protocol length (the protocol only grows, so a longer one is a new key)."""
    cache = battle.__dict__.setdefault("_pcsa_forms", {})
    key = (id(mon), len(battle._replay_data))
    if key not in cache:
        cache[key] = _observed_form(battle, mon)
    return cache[key]


def _observed_form(battle, mon) -> str:
    ident = next(
        (key for team in (battle.team, battle.opponent_team) for key, member in team.items() if member is mon), None
    )
    form = mon.species
    if ident is None or ": " not in ident:  # team preview keys the opponent's members by bare species
        return form
    side, name = ident.split(": ", 1)
    for parts in battle._replay_data:
        if len(parts) > 4 and parts[1] == "-mega" and parts[2].split(": ", 1) == [side + "a", name]:
            resolved = mega_form_for_stone(parts[3], parts[4])
            if resolved:
                form = to_id_str(resolved)
        if len(parts) > 3 and parts[1] in ("switch", "drag", "replace", "detailschange", "-formechange"):
            actor = parts[2].split(": ", 1)
            if len(actor) != 2 or actor[0][:2] != side or actor[1] != name:
                continue
            candidate = to_id_str(parts[3].split(",", 1)[0])
            current = species_entry(form)
            persistent = "Mega" in current.get("forme", "") or form == "mimikyubusted"
            same_base = to_id_str(current.get("baseSpecies") or form) == to_id_str(
                species_entry(candidate).get("baseSpecies") or candidate
            )
            if parts[1] in ("switch", "drag") and persistent and same_base:
                continue  # phone send-out rows name the base species after a persistent form change
            form = reconcile_form(form, candidate)
    return form


@lru_cache(maxsize=1)
def champions_moves() -> dict[str, dict]:
    """Move id -> {type, category, power, accuracy, pp} from the Champions tables in vocab.json."""
    return json.loads(game.asset_path("vocab.json").read_text(encoding="utf-8")).get("champions_moves", {})


def base_data(move_id: str) -> dict:
    """Base move data: the Champions table where it has the move, else poke-env with Champions PP."""
    champ = champions_moves().get(move_id)
    if champ:
        return dict(champ)
    entry = GenData.from_gen(9).moves.get(move_id) or {}
    category = to_id_str(entry.get("category", ""))
    power = None if category == "status" else max(1, int(entry.get("basePower", 0)))
    accuracy = entry.get("accuracy", True)
    pp = int(entry.get("pp", 0))
    fallback = {int(k): int(v) for k, v in game.rule("fallback_pp_by_gen9_base_pp").items()}
    cap = int(game.rule("fallback_pp_cap"))
    return {
        "type": to_id_str(entry.get("type", "")),
        "category": category,
        "power": power,
        "accuracy": 101 if accuracy is True else int(accuracy),
        "pp": fallback.get(pp, min(pp, cap)),
    }


def max_pp(move_id: str) -> int:
    return base_data(move_id)["pp"]


@lru_cache(maxsize=1)
def move_names() -> frozenset[str]:
    return frozenset(e["name"] for e in GenData.from_gen(9).moves.values() if e.get("name"))


def complete_move_name(name: str) -> bool:
    """A real move name that is not the start of a longer one."""
    names = move_names()
    return name in names and not any(other != name and other.startswith(name) for other in names)

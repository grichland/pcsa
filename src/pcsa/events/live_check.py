"""Check messages and events against LIVE.md 0.7: required and allowed fields, listed values, section 6 never-sent."""

from __future__ import annotations

MON = ("side", "mon")
# kind: (required, optional). Taken from LIVE.md section 4.
SCHEMA: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "turn": (("n",), ()),
    "refused": (("order",), ()),
    "end": (("result",), ()),
    "enter": (MON, ("forced",)),
    "withdraw": (MON, ("pivot",)),
    "move": ((*MON, "move"), ("priority", "called_by")),
    "prepare": (MON, ("move",)),
    "cant": (("reason",), MON),  # no Pokemon named: "no PP left" (0.6)
    "protect": (MON, ()),
    "miss": (MON, ()),
    "fail": ((), (*MON, "reason")),
    "result": (("how",), MON),
    "hits": (("n",), ()),
    "hp": ((*MON, "dir"), ("before", "after", "before_pct", "after_pct")),
    "cause": (("effect",), (*MON, "of")),
    "faint": (MON, ()),
    "status": ((*MON, "status"), ("effect",)),
    "cure": ((*MON, "status"), ("item",)),
    "volatile": ((*MON, "effect"), ("detail",)),
    "volatile_end": ((*MON, "effect"), ()),
    "volatile_hit": (("effect",), MON),
    "boost": ((*MON, "stats", "by"), ()),
    "boost_limit": ((*MON, "stats", "dir"), ()),
    "boost_reset": (("what",), (*MON, "of", "stats")),
    "blocked": (("effect",), (*MON, "what")),
    "ability": ((*MON, "ability"), ()),
    "ability_change": ((*MON, "from", "to"), ()),
    "item": (("item", "what"), (*MON, "of")),
    "mega_stone": ((*MON, "stone"), ()),
    "mega": (MON, ("form",)),
    "form": ((*MON, "form"), ()),
    "transform": ((*MON, "into"), ()),
    "types": (MON, ("type", "lose", "add")),
    "weather": (("weather", "what"), ()),
    "field": (("effect", "what"), ()),
    "side": (("side", "effect", "what"), ("mon",)),
    "once": (("effect",), MON),
    "revive": (MON, ()),
    "pp": ((*MON, "move", "lost"), ()),
    "gap": (("what",), ("n",)),
    "unread": (("text",), ("category",)),
}
NEVER = ("crit", "pp", "maxhp", "max_hp", "turns", "turns_left", "setter", "from")  # section 6; "from" ok only in ability_change


STATUSES = ("brn", "par", "slp", "frz", "psn", "tox")
RESULTS = ("win", "loss", "tie", "forfeit", "unknown")
# (kind, field): the values section 4 lists
VALUES: dict[tuple[str, str], tuple] = {
    ("result", "how"): ("super", "extreme", "resisted", "mostly_ineffective", "immune", "ohko"),
    ("hp", "dir"): ("lost", "restored"),
    ("item", "what"): ("activated", "used", "removed", "received", "found"),
    ("boost_reset", "what"): ("all", "negative", "positive", "copy", "swap", "invert"),
    ("boost_limit", "dir"): ("up", "down"),
    ("weather", "what"): ("start", "end"),
    ("field", "what"): ("start", "end"),
    ("side", "what"): ("start", "end", "acted"),
    ("status", "status"): STATUSES,
    ("cure", "status"): STATUSES,
    ("volatile_hit", "effect"): ("substitute", "confusion_self_hit", "endured_hit", "confusion", "attract"),
    ("gap", "what"): ("turn", "rows"),
    ("end", "result"): RESULTS,
}


def problems(event: dict) -> list[str]:
    """Problems of one event, as short codes ('missing:side', 'extra:stats', 'kind:x', 'value:how')."""
    kind = event.get("k")
    if kind not in SCHEMA:
        return [f"kind:{kind}"]
    required, optional = SCHEMA[kind]
    keys = set(event) - {"k"}
    out = [f"missing:{f}" for f in required if f not in keys]
    out += [f"extra:{f}" for f in sorted(keys - set(required) - set(optional))]
    out += [f"never:{f}" for f in NEVER if f in keys and not (kind == "ability_change" and f == "from")]
    out += [f"value:{f}" for (k, f), allowed in VALUES.items() if k == kind and f in keys and event[f] not in allowed]
    if "side" in keys and event["side"] not in ("own", "opp"):
        out.append("value:side")
    if "of" in keys and not (isinstance(event["of"], dict) and event["of"].get("side") in ("own", "opp")):
        out.append("value:of")
    if kind == "hp":
        own, opp = {"before", "after"} <= keys, {"before_pct", "after_pct"} <= keys
        if own == opp:
            out.append("hp:units")
    if kind == "gap" and event.get("what") == "turn" and not isinstance(event.get("n"), int):
        out.append("gap:n")
    return out


SET_FIELDS = (("species", "moves", "stats"), ("item", "ability", "nature", "points"))
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
MOVES_PER_SET, TEAM_SIZE, PICKS = 4, 6, 3
# type: (required, optional) beyond v, battle, type
MESSAGES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "battle_start": (("own_team", "opp_preview"), ("picks",)),
    "preview_request": ((), ("deadline_ms", "replay", "played")),
    "events": (("events",), ("fallback",)),
    "decision_request": (("seq", "kind"), ("deadline_ms", "refused", "retry", "screen_offer", "replay", "played")),
    "battle_end": (("result",), ()),
}


def event_problems(event: dict) -> list[str]:
    """Kept for callers of 0.3: gap is a section-4 kind now."""
    return problems(event)


def order_problems(order: dict | None) -> list[str]:
    if not isinstance(order, dict):
        return ["order:missing"]
    if order.get("kind") == "move":  # `mega` may be left out (0.7)
        ok = isinstance(order.get("move"), str) and isinstance(order.get("mega", False), bool)
        return [] if ok and set(order) <= {"kind", "move", "mega"} else ["order:move"]
    if order.get("kind") == "switch":
        return [] if isinstance(order.get("mon"), str) and set(order) == {"kind", "mon"} else ["order:switch"]
    return ["order:kind"]


def set_problems(one: dict) -> list[str]:
    required, optional = SET_FIELDS
    out = [f"set:missing:{f}" for f in required if f not in one]
    out += [f"set:extra:{f}" for f in sorted(set(one) - set(required) - set(optional))]
    if len(one.get("moves") or []) != MOVES_PER_SET:
        out.append("set:moves")
    for key in ("stats", "points"):
        if key in one and set(one[key]) != set(STAT_KEYS):
            out.append(f"set:{key}")
    return out


def picks_problems(picks, where: str) -> list[str]:
    """Three different indices into own_team, lead first (0.7)."""
    ok = isinstance(picks, list) and len(picks) == PICKS and len(set(picks)) == PICKS
    ok = ok and all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < TEAM_SIZE for i in picks)
    return [] if ok else [f"{where}:picks"]


def fallback_problems(fallback) -> list[str]:
    """Section 3: {"seq": N, "order": ORDER, "reason": "..."}; no `order` when it could not be read (0.6); a preview
    fallback {"picks": [...], "reason": "..."} (0.7); one object or a list (0.7)."""
    if isinstance(fallback, list):
        return [p for one in fallback for p in fallback_problems(one)] if fallback else ["fallback:empty"]
    if not isinstance(fallback, dict):
        return ["fallback:shape"]
    if "picks" in fallback:
        out = [f"fallback:extra:{f}" for f in sorted(set(fallback) - {"picks", "reason"})]
        out += [] if isinstance(fallback.get("reason", ""), str) else ["fallback:reason"]
        return out + picks_problems(fallback["picks"], "fallback")
    out = [f"fallback:extra:{f}" for f in sorted(set(fallback) - {"seq", "order", "reason"})]
    out += [] if isinstance(fallback.get("seq"), int) else ["fallback:seq"]
    out += [] if isinstance(fallback.get("reason", ""), str) else ["fallback:reason"]
    if "order" not in fallback:
        return out
    return out + [f"fallback:{p}" for p in order_problems(fallback["order"])]


def message_problems(msg: dict) -> list[str]:
    """Problems of one controller message: its own fields, then what it carries."""
    kind = msg.get("type")
    if kind not in MESSAGES:
        return [f"type:{kind}"]
    required, optional = MESSAGES[kind]
    keys = set(msg) - {"v", "battle", "type"}
    out = [f"{kind}:missing:{f}" for f in ("v", "battle") if f not in msg]
    out += [f"{kind}:missing:{f}" for f in required if f not in keys]
    out += [f"{kind}:extra:{f}" for f in sorted(keys - set(required) - set(optional))]
    if kind == "battle_start":
        team = msg.get("own_team") or []
        out += ["battle_start:own_team:size"] if len(team) != TEAM_SIZE else []
        out += [p for one in team for p in set_problems(one)]
        out += ["battle_start:opp_preview:size"] if len(msg.get("opp_preview") or []) != TEAM_SIZE else []
        out += picks_problems(msg["picks"], "battle_start") if "picks" in msg else []
    elif kind == "preview_request":
        out += replay_problems(msg, lambda played: picks_problems((played or {}).get("picks"), "played"))
    elif kind == "events":
        events = msg.get("events") or []
        out += [f"{e.get('k')}:{p}" for e in events for p in event_problems(e)]
        out += [f"refused:{p}" for e in events if e.get("k") == "refused" for p in order_problems(e.get("order"))]
        out += fallback_problems(msg["fallback"]) if "fallback" in msg else []
    elif kind == "decision_request":
        out += [] if msg.get("kind") in ("turn", "replace") else ["decision_request:kind"]
        if "refused" in msg:
            out += order_problems((msg["refused"] or {}).get("order"))
        if "retry" in msg and (msg["retry"] is not True or "refused" in msg):
            out.append("decision_request:retry")  # true only, and never with refused (CF-8)
        out += replay_problems(msg, order_problems)
    elif kind == "battle_end":
        out += [] if msg.get("result") in RESULTS else ["battle_end:result"]
    return out


def replay_problems(msg: dict, check) -> list[str]:
    """`replay` is true only; `played` (0.7) checked by `check`."""
    out = [] if msg.get("replay", True) is True else [f"{msg.get('type')}:replay"]
    return out + ([f"played:{p}" for p in check(msg["played"])] if "played" in msg else [])

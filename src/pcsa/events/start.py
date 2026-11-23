"""battle_start, preview_request and the picks answer from a recorded battle (LIVE protocol).

Sets come from the run's team sheet (own_team.json), which pcsa read from the game's Moves & More and Stats
pages: item, ability, moves, the shown stats, the stat points and the nature arrows. A point figure pcsa
inferred from the 66-point total (not read) drops `points` for that member: unknown fields are left out.
"""

from __future__ import annotations

from pcsa.battle.ids import to_id
from pcsa.events.live_ids import VERSION, Teams, ability_id, item_id, move_id, preview_id, species_id

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
INFERRED = "inferred from"  # the wording for a stat point that was computed, not read
MOVES_PER_SET = 4


def _member_set(sheet: dict, team: dict) -> tuple[dict, list[str]]:
    """One SET and the reasons fields were left out."""
    notes: list[str] = []
    name = sheet.get("name") or team.get("name") or ""
    out: dict = {"species": species_id(team.get("species") or name)}
    if sheet.get("item_status") == "known" and sheet.get("item"):
        out["item"] = item_id(sheet["item"])[0]
    elif sheet.get("item_status") != "none":
        notes.append(f"{name}: item not read")
    if sheet.get("ability_status") == "known" and sheet.get("ability"):
        out["ability"] = ability_id(sheet["ability"], out["species"])[0]
    else:
        notes.append(f"{name}: ability not read")
    moves = [m for m in sheet.get("moves") or [] if m.get("status") == "known" and m.get("name")]
    out["moves"] = [move_id(m["name"]) for m in sorted(moves, key=lambda m: m.get("slot", 0))]
    if len(out["moves"]) != MOVES_PER_SET:
        notes.append(f"{name}: {len(out['moves'])} moves read")
    if sheet.get("nature"):
        out["nature"] = to_id(sheet["nature"])
    points = sheet.get("stat_points") or {}
    inferred = [s for s in sheet.get("stat_inferences") or [] if INFERRED in s]
    if inferred:
        notes.append(f"{name}: points left out, {'; '.join(inferred)}")
    elif all(isinstance(points.get(s), int) for s in STATS):
        out["points"] = {s: points[s] for s in STATS}
    stats = sheet.get("stats") or {}
    if all(isinstance(stats.get(s), int) for s in STATS):
        out["stats"] = {s: stats[s] for s in STATS}
    else:
        notes.append(f"{name}: stats not read")
    return out, notes


def own_sets(own_team: dict | None, roster: list[str] | None) -> tuple[list[dict] | None, list[str]]:
    """Six SETs in team-select order (battle.json roster, else sheet order); None with the reason when no sheet."""
    if not own_team or not (own_team.get("sheet") or {}).get("members"):
        return None, ["no own_team.json sheet"]
    sheet = own_team["sheet"]["members"]
    team = (own_team.get("team") or {}).get("members") or [{} for _ in sheet]
    by_name = {to_id(s.get("name") or ""): (s, t) for s, t in zip(sheet, team, strict=False)}
    order = [to_id(n) for n in roster] if roster else list(by_name)
    sets, notes = [], []
    for key in order:
        if key not in by_name:
            return None, [f"roster member {key} not on the sheet"]
        one, why = _member_set(*by_name[key])
        sets.append(one)
        notes += why
    return sets, notes


def roster_of(meta: dict, own_team: dict | None) -> list[str]:
    if meta.get("roster"):
        return list(meta["roster"])
    return [m.get("name") or "" for m in ((own_team or {}).get("sheet") or {}).get("members") or []]


def opp_preview(meta: dict, run_events: list[dict]) -> list[str | None]:
    """Six species ids as shown at preview (the top sprite match), None where unread."""
    return [preview_id(n) if n else None for n in preview_names(meta, run_events)]


def picks_of(meta: dict, run_events: list[dict]) -> list[int] | None:
    if meta.get("team_requested") is not None:
        return list(meta["team_requested"])
    team = next((e for e in run_events if e.get("event") == "decision" and e.get("kind") == "team"), None)
    return list(team["team"]) if team and team.get("team") is not None else None


def brought_names(meta: dict, roster: list[str], picks: list[int] | None) -> list[str]:
    """Our three brought, in pick order: what `switchN` labels and own rows refer to."""
    names = meta.get("team_played_names") or meta.get("team_requested_names")
    if names:
        return [n for n in names if n]
    if picks and roster:
        return [roster[i] for i in picks if 0 <= i < len(roster)]
    return [n for n in roster if n]


def teams_for_rows(meta: dict, brought: list[str], preview_names: list[str | None]) -> Teams:
    """What the row converter is given: our brought members (else the roster) and their preview."""
    own = brought or meta.get("roster") or []
    return Teams(tuple(n for n in own if n), tuple(preview_names))


def preview_names(meta: dict, run_events: list[dict]) -> list[str | None]:
    """Display names at preview, form included (`top`; `species` is the same match without its form)."""
    slots = meta.get("opponent_preview") or []
    if slots:
        out = []
        for slot in slots:
            top = (slot.get("top") or slot.get("species") or [[None]])[0]
            out.append(top[0] if top else None)
        return out
    return list(next((e.get("opponent") for e in run_events if e.get("event") == "team_preview"), None) or [])


def battle_start(tag: str, sets: list[dict] | None, preview: list[str | None]) -> dict:
    return {"v": VERSION, "battle": tag, "type": "battle_start", "own_team": sets or [], "opp_preview": preview}


def preview_request(tag: str) -> dict:
    """The team-select timer was read live but not recorded, so no deadline_ms (unknown fields are left out)."""
    return {"v": VERSION, "battle": tag, "type": "preview_request"}

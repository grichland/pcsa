"""One poke-env Battle per decision, reconstructed from everything known so far.

Inputs: our team sheet, the preview sprite slots, our three picks, the collected log rows, and what
the screens show now (HP, PP, Mega, forced replacement). Nothing carries over between decisions,
so a corrected log row or a late reveal applies to every later decision.
"""

from __future__ import annotations

import logging

from poke_env.battle import Battle
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.battle.adapter import ChampionsBattle, current_request
from pcsa.battle.moves import max_pp
from pcsa.battle.names import showdown_species
from pcsa.battle.teams import OwnMember, Team
from pcsa.battle.translate import LEVEL, Member, Translation, Translator

USERNAME = "me"
PICKED = 3
LOGGER = logging.getLogger("pcsa.battle")
LOGGER.setLevel(logging.ERROR)


def our_members(team: Team, picks: list[int] | None = None) -> list[Member]:
    rows = team.members if picks is None else [team.members[i] for i in picks]
    return [Member(m.name, m.species, m.stats.get("hp") if m.stats else None) for m in rows]


def their_members(preview, replaced: dict[int, Member] | None = None) -> list[Member]:
    """Preview sprite slots (the top candidate decides the form); unread slots are skipped."""
    out = []
    for i, slot in enumerate(preview):
        if replaced and i in replaced:
            out.append(replaced[i])
        elif slot.top:
            base = slot.species[0][0] if slot.species else slot.top[0][0]
            out.append(Member(base, showdown_species(slot.top[0][0]).species))
    return out


def misread_slots(
    preview, replaced: dict[int, Member], unpreviewed: list[Member], seen: set[str]
) -> dict[int, Member]:
    """A sent-out opponent no preview slot showed takes a slot not yet seen in battle."""
    out: dict[int, Member] = {}
    taken = set(replaced)
    for member in unpreviewed:
        species = to_id_str(member.species)
        slots = [
            (not any(to_id_str(showdown_species(name).species) == species for name, _ in slot.top), slot.top[0][1], i)
            for i, slot in enumerate(preview)
            if slot.top and i not in taken and (slot.species[0][0] if slot.species else slot.top[0][0]) not in seen
        ]
        if not slots:
            break
        *_, i = min(slots)
        out[i] = member
        taken.add(i)
    return out


def _entry(member: OwnMember, condition: str, active: bool, item: str | None) -> dict:
    stats = member.stats or {}
    ability = member.ability
    return {
        "ident": f"p1: {member.name}",
        "details": f"{member.species}, L{LEVEL}",
        "condition": condition,
        "active": active,
        "stats": {s: stats[s] for s in ("hp", "atk", "def", "spa", "spd", "spe") if stats.get(s) is not None},
        "moves": [to_id_str(m) for m in member.moves if m],
        "baseAbility": to_id_str(ability) if ability else "",
        "ability": to_id_str(ability) if ability else "",
        "item": to_id_str(item) if item else "",
    }


def _full(member: OwnMember) -> str:
    hp = (member.stats or {}).get("hp")
    return f"{hp}/{hp}" if hp else "100/100"


def team_preview_request(team: Team) -> dict:
    return {
        "teamPreview": True,
        "maxTeamSize": PICKED,
        "side": {"name": USERNAME, "id": "p1", "pokemon": [_entry(m, _full(m), False, m.item) for m in team.members]},
    }


def battle_request(
    team: Team,
    picks: list[int],
    battle: Battle,
    *,
    active: int | None,
    screen_hp: dict,
    fainted: set,
    moves=(),
    mega_available=False,
    forced=False,
) -> dict:
    """Our side's request now. On an action screen the log's exact HP wins over screen OCR; on a
    forced screen the faint happened after the last log read, so the party screen wins."""
    order = ([active] if active is not None else []) + [s for s in range(len(picks)) if s != active]
    entries = []
    for slot in order:
        member = team.members[picks[slot]]
        mon = battle.team.get(f"p1: {member.name}")
        logged = (
            (mon.current_hp, mon.max_hp) if mon is not None and mon.current_hp is not None and mon.max_hp else None
        )
        shown = screen_hp.get(slot)
        hp = (shown or logged) if forced else (logged or shown)
        if slot in fainted or (hp is not None and hp[0] <= 0):
            condition = "0 fnt"
        else:
            condition = f"{hp[0]}/{hp[1]}" if hp is not None else _full(member)
        if mon is not None and mon.status and mon.status.name != "FNT" and condition != "0 fnt":
            condition += " " + mon.status.name.lower()
        item = member.item
        if mon is not None and mon.item != GenData.UNKNOWN_ITEM:
            item = mon.item  # latest revealed, changed, consumed, or knocked-off item
        entries.append(_entry(member, condition, slot == active, item))
    side = {"name": USERNAME, "id": "p1", "pokemon": entries}
    if forced:
        return current_request(battle, {"forceSwitch": [True], "side": side})
    rows = _move_rows(team, picks, battle, active, moves)
    if entries and rows:  # the screen's moves are the active member's moves when the sheet had none
        known = entries[0]["moves"]
        entries[0]["moves"] = known + [r["id"] for r in rows if r["id"] not in known]
    active_block = {"moves": rows, **({"canMegaEvo": True} if mega_available else {})}
    return current_request(battle, {"active": [active_block], "side": side})


CHOICE_ITEMS = frozenset({"choicescarf", "choiceband", "choicespecs"})


def choice_lock(battle: Battle, name: str, item: str | None) -> str | None:
    """The move a Choice item locks our member into: its last move since it was last sent out, if any."""
    if to_id_str(item or "") not in CHOICE_ITEMS:
        return None
    actor = f"p1a: {name}"
    for parts in reversed(getattr(battle, "_replay_data", [])):
        if len(parts) < 3 or parts[2] != actor:
            continue
        if parts[1] in ("switch", "drag"):
            return None
        if parts[1] == "move" and len(parts) > 3 and to_id_str(parts[3]) != "struggle":
            return to_id_str(parts[3])
    return None


FIRST_TURN_MOVES = frozenset({"fakeout", "firstimpression", "matblock"})


def acted_since_entry(battle: Battle, name: str) -> bool:
    """Our member has taken a move action (a move line or a `cant`) since it was last sent out.

    The simulator's request then disables Fake Out, First Impression and Mat Block; the game's move screen does
    not grey them out, so the live request has to. Without this the network was offered First Impression on a
    later turn, a choice it never had in training, and clicked it (run 20261109-205551, battle 2, turn 6)."""
    actor = f"p1a: {name}"
    for parts in reversed(getattr(battle, "_replay_data", [])):
        if len(parts) < 3 or parts[2] != actor:
            continue
        if parts[1] in ("switch", "drag"):
            return False
        if parts[1] == "cant" or (parts[1] == "move" and not any("[from]" in p for p in parts[4:])):
            return True
    return False


def _move_rows(team: Team, picks: list[int], battle: Battle, active: int | None, moves) -> list[dict]:
    """Move names come from the team sheet by slot; the move screen supplies PP and availability."""
    if active is None:
        return []
    member = team.members[picks[active]]
    known = list(member.moves)
    active_mon = battle.team.get(f"p1: {member.name}")
    encore_move = (
        active_mon.last_move if active_mon and any(effect.name == "ENCORE" for effect in active_mon.effects) else None
    )
    encore_id = to_id_str(encore_move.id) if encore_move is not None else None
    shown = active_mon.item if active_mon is not None else None
    held = shown if shown not in (None, "", GenData.UNKNOWN_ITEM) else member.item
    locked_id = choice_lock(battle, member.name, held)
    spent = acted_since_entry(battle, member.name)  # first-turn-only moves are over, as in the simulator's request
    rows = []
    for m in moves:
        name = known[m.slot] if m.slot < len(known) and known[m.slot] else m.name
        if not name:
            continue
        pp = tuple(m.pp) if m.pp else logged_pp(active_mon, name)
        held = (encore_id is not None and to_id_str(name) != encore_id) or (
            locked_id is not None and to_id_str(name) != locked_id
        )
        disabled = bool(m.disabled or pp[0] == 0 or held or (spent and to_id_str(name) in FIRST_TURN_MOVES))
        rows.append(
            {
                "move": name,
                "id": to_id_str(name),
                "pp": pp[0],
                "maxpp": pp[1],
                "target": "normal",
                "disabled": disabled,
            }
        )
    return rows


def logged_pp(mon, move: str) -> tuple[int, int]:
    """(PP left, max PP) for a move whose PP the move screen could not read: Champions max minus uses counted."""
    move_id = to_id_str(move)
    maximum = max_pp(move_id)
    tracked = mon.moves.get(move_id) if mon is not None else None
    used = tracked.max_pp - tracked.current_pp if tracked is not None else 0
    return max(0, maximum - used), maximum


def build(
    tag: str,
    team: Team,
    preview,
    *,
    picks=None,
    rows_by_turn=None,
    decision_turn: int = 0,
    sprites_by_turn=None,
) -> tuple[Battle, Translation]:
    """The Battle for one decision. Without `picks` it is the team preview.

    An Illusion user found partway means earlier rows named its disguise: the log is translated again
    with that send-out known. A sent-out opponent the preview missed replaces a misread slot the same way."""
    disguises: dict[str, str] = {}
    replaced: dict[int, Member] = {}
    battle, t = None, None
    for _ in range(2 * PICKED + 1):
        battle = ChampionsBattle(tag, USERNAME, LOGGER, 9)
        battle.player_role = "p1"
        members = their_members(preview, replaced)
        t = Translator(battle, our_members(team, picks), members, disguises=disguises)
        for line in ["|player|p1|me|", "|player|p2|opponent|", "|gen|9", "|clearpoke"]:
            t.apply(line)
        for member in members:
            t.apply(f"|poke|p2|{member.species}, L{LEVEL}|")
        t.apply(f"|teampreview|{PICKED}")
        if picks is not None:
            for line in [f"|teamsize|p1|{len(picks)}", f"|teamsize|p2|{PICKED}", "|start"]:
                t.apply(line)
            t.turns(rows_by_turn or {}, decision_turn, sprites_by_turn=sprites_by_turn)
        seen = {key.split(": ", 1)[1] for key in battle.opponent_team if ": " in key}  # sent out, not preview-only
        misread = misread_slots(preview, replaced, t.out.unpreviewed, seen)
        if t.out.illusions.items() <= disguises.items() and not misread:
            break
        disguises = {**disguises, **t.out.illusions}
        replaced = {**replaced, **misread}
    assert battle is not None and t is not None
    if replaced:
        t.out.warnings.append(
            "preview corrected from the log: "
            + ", ".join(f"slot {i + 1} is {m.name}" for i, m in sorted(replaced.items()))
        )
    if picks is None:
        apply_request(battle, t.out, team_preview_request(team))
    return battle, t.out


def apply_request(battle: Battle, out: Translation, request: dict) -> None:
    try:
        battle.parse_request(request)
        normalize_fainted_request(battle)
    except Exception as error:
        out.warnings.append(f"poke-env rejected the request: {type(error).__name__}: {error}")


def normalize_fainted_request(battle) -> None:
    """A fainted request member is not an active combatant, even in its old field slot."""
    for mon in battle.team.values():
        if mon.fainted:
            mon.faint()
            mon._active = False
            mon.clear_boosts()
            if mon.stats.get("hp"):
                mon._max_hp = mon.stats["hp"]


def replay(protocol: list[str], request: dict | None, tag: str = "replay") -> Battle:
    """A p1 Battle rebuilt from protocol lines, then a request (a p2 seat mirrored into p1's view)."""
    battle = ChampionsBattle(tag, USERNAME, LOGGER, 9)
    battle.player_role = "p1"
    ended_fields: set[str] = set()
    for line in protocol:
        try:
            parts = line.split("|")
            if len(parts) > 2 and parts[1] == "-fieldstart":
                ended_fields.discard(parts[2])
            elif len(parts) > 2 and parts[1] == "-fieldend":
                if parts[2] in ended_fields:
                    battle.compatibility_repairs.append(f"duplicate observed field end: {parts[2]}")
                    continue
                ended_fields.add(parts[2])
            if len(parts) > 5 and parts[1] == "move" and "[notarget]" in parts[5:]:
                parts[4] = ""
            battle.parse_message(parts)
        except (KeyError, IndexError, ValueError, AttributeError) as error:
            battle.replay_errors.append(f"{line}: {type(error).__name__}: {error}")
    if request is not None:
        battle.parse_request(request)
        normalize_fainted_request(battle)
    return battle


def battle_text(battle: Battle, revealed: dict[str, str]) -> str:
    """Readable snapshot: turn and field, then every known Pokémon of both sides."""

    def mon(key, m, ours):
        hp = f"{m.current_hp}/{m.max_hp}" if ours else (f"{m.current_hp}%" if m.current_hp or m.fainted else "?")
        state = "fainted" if m.fainted else ("ACTIVE" if m.active else "bench")
        boosts = " ".join(f"{k}{v:+d}" for k, v in m.boosts.items() if v) or "-"
        ability = m.ability if ours else revealed.get(key, "?")
        item = "?" if m.item == "unknown_item" else (m.item or "-")
        types = "/".join(t.name.title() for t in m.types)
        status = m.status.name if m.status else "-"
        head = f"  {m.species:14} {state:7} HP {hp:9} {types:16} status {status:4} boosts {boosts:10}"
        return f"{head} ability {ability or '?':14} item {item or '-':14} moves {', '.join(m.moves) or '?'}"

    weather = ", ".join(w.name for w in battle.weather) or "-"
    fields = ", ".join(f.name for f in battle.fields) or "-"
    lines = [f"TURN {battle.turn}   weather: {weather}   fields: {fields}"]
    lines.append(f"OURS   side: {', '.join(s.name for s in battle.side_conditions) or '-'}")
    lines += [mon(k, m, True) for k, m in battle.team.items()]
    lines.append(f"THEIRS side: {', '.join(s.name for s in battle.opponent_side_conditions) or '-'}")
    lines += [mon(k, m, False) for k, m in battle.opponent_team.items()]
    seen = {m.species for m in battle.opponent_team.values()}
    lines.append(
        "  preview, not yet seen: "
        + (", ".join(m.species for m in battle.teampreview_opponent_team if m.species not in seen) or "-")
    )
    return "\n".join(lines) + "\n"

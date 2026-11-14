"""Protocol for the less common mechanics: item ends and reveals, revival, full heals, single-turn and
field effects, Haze, Burn Up, and charge turns whose move row was not captured."""

from __future__ import annotations

from poke_env.data.normalize import to_id_str

from pcsa.battle import battle_text, game
from pcsa.battle.log_rows import Event, RawRow
from pcsa.battle.translate import Member, Translator


def _one_line(t: Translator, row: RawRow, e: Event, template: str) -> None:
    if (member := t.require(row, e)) is None or e.side is None:
        return
    t.apply(template.format(ident=t.ident(e.side, member), **e.data))


def item_reveal_or_end(t: Translator, row: RawRow, e: Event) -> None:
    if (member := t.require(row, e)) is None or e.side is None:
        return
    ident = t.ident(e.side, member)
    if e.kind == "item_end":
        # Red Card / Eject Button: the "Held Item" row just before usually ended the item already
        if f"|-enditem|{ident}|{e.data['item']}" not in this_turn(t):
            t.apply(f"|-enditem|{ident}|{e.data['item']}")
        return
    if ability := e.data.get("ability"):  # Harvest: the berry is back and the ability is shown
        t.out.revealed_abilities[f"{e.side}: {member.name}"] = ability
        mon = t.tracked(e.side, member)
        if mon is None or mon.ability != to_id_str(ability):
            t.apply(f"|-ability|{ident}|{ability}")
        return t.apply(f"|-item|{ident}|{e.data['item']}|[from] ability: {ability}")
    t.apply(f"|-item|{ident}|{e.data['item']}")


def this_turn(t: Translator) -> list[str]:
    """Lines written since the latest `|turn|` line."""
    out: list[str] = []
    for line in reversed(t.out.lines):
        if line.startswith("|turn|"):
            break
        out.append(line)
    return out


def revive(t: Translator, row: RawRow, e: Event) -> None:
    """Revival Blessing: a fainted party member comes back with half its HP (Showdown writes it with the bench
    identifier, `p2: X`); poke-env's -heal clears the fainted status."""
    if (member := t.require(row, e)) is None or e.side is None:
        return
    t.apply(
        f"|-heal|{e.side}: {member.name}|{half_or_full(t, e.side, member, half=True)}|[from] move: Revival Blessing"
    )


def half_or_full(t: Translator, side: str, member: Member, *, half: bool) -> str:
    mon = t.tracked(side, member)
    maximum = (member.max_hp or (mon.max_hp if mon is not None and mon.max_hp else 100)) if side == "p1" else 100
    return f"{maximum // 2 if half else maximum}/{maximum}"


def full_heal(t: Translator, row: RawRow, e: Event) -> None:
    """Healing Wish: full HP and no status. The status cure goes first: an HP line without a status suffix keeps
    the old status here (phone HP rows never show it)."""
    if (member := t.require(row, e)) is None or e.side is None:
        return
    ident = t.ident(e.side, member)
    mon = t.tracked(e.side, member)
    if mon is not None and mon.status is not None and mon.status.name != "FNT":
        t.apply(f"|-curestatus|{ident}|{mon.status.name.lower()}|[msg]")
    t.apply(f"|-heal|{ident}|{half_or_full(t, e.side, member, half=False)}|[from] {e.data['source']}")


def single(t: Translator, row: RawRow, e: Event) -> None:
    _one_line(t, row, e, "|{verb}|{ident}|{effect}")


def activate(t: Translator, row: RawRow, e: Event) -> None:
    _one_line(t, row, e, "|-activate|{ident}|{effect}")


def field_activate(t: Translator, row: RawRow, e: Event) -> None:
    t.apply(f"|-fieldactivate|{e.data['effect']}")


def clear_all_boost(t: Translator, row: RawRow, e: Event) -> None:
    t.apply("|-clearallboost")


def typechange(t: Translator, row: RawRow, e: Event) -> None:
    if "lose" not in e.data:
        return _one_line(t, row, e, "|-start|{ident}|typechange|{type}")
    # Burn Up: the user loses its Fire type ("???" when nothing is left)
    if (member := t.require(row, e)) is None or e.side is None:
        return
    mon = t.tracked(e.side, member)
    types = [ty.name.title() for ty in (mon.types if mon is not None else []) if ty.name.title() != e.data["lose"]]
    t.apply(f"|-start|{t.ident(e.side, member)}|typechange|{'/'.join(types) or '???'}|[from] {e.data['source']}")


def prepare(t: Translator, row: RawRow, e: Event) -> None:
    member = t.member(e.side, e.name) if e.side else None
    if member is None or e.side is None:
        return
    ident = t.ident(e.side, member)
    move = next((line.split("|")[3] for line in reversed(t.out.lines) if line.startswith(f"|move|{ident}|")), None)
    if move is None:
        move = charged_move(member.species, row.message)
        if move is None:
            return t.unmapped(row, "charge turn without its move")
        # the move row was not captured: the charge text and the species' learnset name the move
        t.out.inferred.append({"kind": "charge_move", "row": row.row_id, "name": member.name, "move": move})
        t.ensure_active(e.side, member, row)
        t.apply(f"|move|{ident}|{move}||[still]")
    t.apply(f"|-prepare|{ident}|{move}")


def charged_move(species: str, message: str) -> str | None:
    """The one charge move whose prepare text this is and that the species learns."""
    names = {m.name for m in battle_text.lookup(message.strip()) if m.key == "prepare"}
    learnable = sorted(n for n in names if game.can_learn(species, n))
    return learnable[0] if len(learnable) == 1 else None

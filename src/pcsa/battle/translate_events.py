"""What each log event writes into the protocol. One small function per event kind."""

from __future__ import annotations

from collections.abc import Callable

from poke_env.battle.effect import Effect
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.log_rows import Event, RawRow
from pcsa.battle.moves import observed_form
from pcsa.battle.names import showdown_species
from pcsa.battle.translate import CONSUMED_IDS, INFORMATIONAL_KINDS, LEVEL, Member, Translator, has_illusion, letters
from pcsa.battle.translate_megas import mega, mega_reacting, resolve_pending_mega

Handler = Callable[[Translator, RawRow, Event], None]


def _need(t: Translator, row: RawRow, e: Event) -> Member | None:
    return t.require(row, e)


def _confusion_damage(t: Translator, row: RawRow, e: Event) -> None:
    previous = t.previous_event
    if previous is None or previous.kind != "volatile_active" or previous.data.get("effect") != "confusion":
        return t.unmapped(row, "Confusion damage has no immediately observed subject")
    member = t.member(previous.side, previous.name)
    if member is None or previous.side is None:
        return t.unmapped(row, "Confusion subject cannot be assigned")
    t.apply(f"|cant|{t.ident(previous.side, member)}|confusion")


def _volatile_active(t: Translator, row: RawRow, e: Event) -> None:
    if (member := _need(t, row, e)) is None or e.side is None:
        return
    mon = t.tracked(e.side, member)
    if mon is None or not any(to_id_str(effect.name) == to_id_str(e.data["effect"]) for effect in mon.effects):
        t.apply(f"|-start|{t.ident(e.side, member)}|{e.data['effect']}")
    t.apply(f"|-activate|{t.ident(e.side, member)}|{e.data['effect']}")


def _fail(t: Translator, row: RawRow, e: Event) -> None:
    move = next((parts for parts in reversed(t.battle._replay_data) if len(parts) > 2 and parts[1] == "move"), None)
    if move is None:
        return t.unmapped(row, "Move failure has no observed move")
    t.apply(f"|-fail|{move[2]}")


def _send_out(t: Translator, row: RawRow, e: Event) -> None:
    side = e.side or "p1"
    member = t.member(side, e.name)
    if member is None and side == "p2" and game.species_entry(to_id_str(e.name or "")):
        mapped = showdown_species(e.name or "")  # the send-out row outranks a preview read that missed it
        species = (
            mapped.species
            if mapped.status == "mapped"
            else game.species_entry(to_id_str(e.name or "")).get("name", e.name)
        )
        member = t.theirs[e.name] = Member(e.name, species)  # type: ignore[index]
        t.out.warnings.append(f"opponent sent out {e.name}, which the team preview did not show")
        t.out.unpreviewed.append(member)
    if member is None:
        return t.unmapped(row, f"unknown {side} Pokemon {e.name!r}")
    t.alias.pop(side, None)
    t.sent_out[side] = row.row_id
    if (real := t.theirs.get(t.disguises.get(row.row_id, ""))) is not None:
        t.alias[side] = (member.name, real)
        t.out.revealed_abilities[f"{side}: {real.name}"] = "Illusion"
        t.out.inferred.append({"kind": "illusion", "row": row.row_id, "shown": member.name, "real": real.name})
        member = real
    verb = "switch" if e.kind == "send_out" else "drag"
    t.apply(f"|{verb}|{t.ident(side, member)}|{member.species}, L{LEVEL}|{t.hp_field(side, member)}")
    if side in t.passed_substitute:  # Shed Tail: the decoy stays for the replacement
        t.passed_substitute.discard(side)
        t.apply(f"|-start|{t.ident(side, member)}|Substitute")


def _move(t: Translator, row: RawRow, e: Event) -> None:
    if (member := _need(t, row, e)) is None or e.side is None:
        return
    if (
        e.side == "p2"
        and e.side not in t.alias
        and e.side in t.sent_out
        and not game.can_learn(member.species, e.data["move"])
    ):
        users = [
            m
            for m in t.theirs.values()
            if m is not member and has_illusion(m.species) and game.can_learn(m.species, e.data["move"])
        ]
        if len(users) == 1:  # an Illusion user was sent out in disguise; build() translates again
            t.out.illusions[t.sent_out[e.side]] = users[0].name
    t.ensure_active(e.side, member, row)
    target = t.active(t.other(e.side)) or ""
    t.apply(f"|move|{t.ident(e.side, member)}|{e.data['move']}|{target}")
    mon = t.tracked(e.side, member)
    move = GenData.from_gen(9).moves.get(to_id_str(e.data["move"]), {})
    if (
        mon is not None
        and Effect.CHARGE in mon.effects
        and move.get("type") == "Electric"
        and to_id_str(e.data["move"]) != "charge"
    ):
        t.apply(f"|-end|{t.ident(e.side, member)}|Charge|[silent]")


def _hp(t: Translator, row: RawRow, e: Event) -> None:
    if (member := _need(t, row, e)) is None or e.side is None:
        return
    d = e.data
    direction = d["direction"] or ("lost" if d["after"] < d["before"] else "restored")
    verb = "-damage" if direction == "lost" else "-heal"
    t.apply(f"|{verb}|{t.ident(e.side, member)}|{t.hp_field(e.side, member, d['after'])}")


def _faint(t: Translator, row: RawRow, e: Event) -> None:
    if (member := _need(t, row, e)) is None or e.side is None:
        return
    if (alias := t.alias.get(e.side)) and alias[1] is member:
        t.alias.pop(e.side)
    t.apply(f"|faint|{t.ident(e.side, member)}")


def _illusion_end(t: Translator, row: RawRow, e: Event) -> None:
    if (member := _need(t, row, e)) is None or e.side is None:
        return
    if (alias := t.alias.get(e.side)) and alias[1] is member:
        t.alias.pop(e.side)
    elif e.side in t.sent_out:
        t.out.illusions[t.sent_out[e.side]] = member.name
    else:
        t.unmapped(row, "Illusion ended with no observed send-out")


def _ability(t: Translator, row: RawRow, e: Event) -> None:
    found = t.located(e)
    if found is None:
        return t.unmapped(row, f"cannot tell whose Pokemon {e.name!r} is")
    side, member = found
    ability = e.data["ability"] if e.kind == "ability" else e.data["new"]
    # A changed ability (Trace, Skill Swap, Mummy) need not belong to the species: fall back to every ability.
    normalized = game.canonical_ability(ability, member.species) or game.canonical_ability(ability)
    if normalized is None:
        return t.unmapped(row, f"ambiguous ability OCR {ability!r}")
    resolve_pending_mega(t, side, member, normalized)
    if normalized != ability:
        t.out.inferred.append(
            {"kind": "ability_ocr_correction", "raw": ability, "normalized": normalized, "row": row.row_id}
        )
    t.out.revealed_abilities[f"{side}: {member.name}"] = normalized
    ident = t.ident(side, member)
    changed = "|[from] observed ability change" if e.kind == "ability_change" else ""
    t.apply(f"|-ability|{ident}|{normalized}{changed}")
    if e.kind == "ability" and letters(normalized) == "hungerswitch":
        mon = t.tracked(side, member)
        current = observed_form(t.battle, mon) if mon is not None else None
        if current in ("morpeko", "morpekohangry"):
            t.apply(f"|detailschange|{ident}|{'Morpeko-Hangry' if current == 'morpeko' else 'Morpeko'}, L{LEVEL}")


def _item(t: Translator, row: RawRow, e: Event) -> None:
    found = t.located(e)
    if found is None:
        return t.unmapped(row, f"cannot tell whose Pokemon {e.name!r} is")
    side, member = found
    item = e.data["item"]
    if to_id_str(item).endswith("berry"):
        return t.apply(f"|-enditem|{t.ident(side, member)}|{item}|[eat]")
    if to_id_str(item) in CONSUMED_IDS:
        return t.apply(f"|-enditem|{t.ident(side, member)}|{item}")
    t.apply(f"|-item|{t.ident(side, member)}|{item}")


def _item_reveal_or_end(t: Translator, row: RawRow, e: Event) -> None:
    if (member := _need(t, row, e)) is None or e.side is None:
        return
    verb = "-item" if e.kind == "item_reveal" else "-enditem"
    t.apply(f"|{verb}|{t.ident(e.side, member)}|{e.data['item']}")


def _boost(t: Translator, row: RawRow, e: Event) -> None:
    member = t.member(e.side, e.name)
    if member is None or not e.data.get("stats") or e.side is None:
        return t.unmapped(row, f"unknown Pokemon {e.name!r} or stats")
    if e.data["amount"]:
        verb = "-boost" if e.data["sign"] > 0 else "-unboost"
        for stat in e.data["stats"]:
            t.apply(f"|{verb}|{t.ident(e.side, member)}|{stat}|{e.data['amount']}")


def _status_like(t: Translator, row: RawRow, e: Event) -> None:
    if (member := _need(t, row, e)) is None or e.side is None:
        return
    d = e.data
    if e.kind == "cant":
        t.ensure_active(e.side, member, row)
    ident = t.ident(e.side, member)
    if e.kind == "cure" and d.get("item"):
        t.apply(f"|-enditem|{ident}|{d['item']}|[eat]")
    line = {
        "status": f"|-status|{ident}|{d.get('status')}",
        "cure": f"|-curestatus|{ident}|{d.get('status')}",
        "cant": f"|cant|{ident}|{d.get('reason')}",
        "volatile": f"|-start|{ident}|{d.get('effect')}",
    }[e.kind]
    if e.kind == "volatile" and to_id_str(d.get("effect")) == "attract":
        source = t.active(t.other(e.side))
        if source is None:
            return t.unmapped(row, "Attract has no observed source")
        line += f"|[of] {source}"
    if e.kind == "volatile" and d.get("passes"):
        t.passed_substitute.add(e.side)
    t.apply(line)


def _typechange(t: Translator, row: RawRow, e: Event) -> None:
    _simple(t, row, e, "|-start|{ident}|typechange|{type}")


def _clear_negative(t: Translator, row: RawRow, e: Event) -> None:
    _simple(t, row, e, "|-clearnegativeboost|{ident}")


def _volatile_end(t: Translator, row: RawRow, e: Event) -> None:
    _simple(t, row, e, "|-end|{ident}|{effect}")


def _simple(t: Translator, row: RawRow, e: Event, template: str) -> None:
    """One protocol line about the row's own Pokémon."""
    if (member := _need(t, row, e)) is None or e.side is None:
        return
    t.apply(template.format(ident=t.ident(e.side, member), **e.data))


def _transform(t: Translator, row: RawRow, e: Event) -> None:
    member, target = t.member(e.side, e.name), t.member(e.data["target_side"], e.data["target"])
    if member is None or target is None or e.side is None:
        return t.unmapped(row, "unknown Transform user or target")
    t.apply(f"|-transform|{t.ident(e.side, member)}|{t.ident(e.data['target_side'], target)}")


def _form_change(t: Translator, row: RawRow, e: Event) -> None:
    if (member := _need(t, row, e)) is None or e.side is None:
        return
    ident = t.ident(e.side, member)
    t.out.revealed_abilities[f"{e.side}: {member.name}"] = e.data["ability"]
    t.apply(f"|-activate|{ident}|ability: {e.data['ability']}")
    t.apply(f"|detailschange|{ident}|{e.data['form']}, L{LEVEL}")


def _steal_eat(t: Translator, row: RawRow, e: Event) -> None:
    member = t.member(e.side, e.name)
    target = t.active(t.other(e.side)) if e.side else None
    if member is None or target is None or e.side is None:
        return t.unmapped(row, "unknown Bug Bite / Pluck user or target")
    t.apply(f"|-enditem|{target}|{e.data['item']}|[from] stealeat|[of] {t.ident(e.side, member)}")


def _field(t: Translator, row: RawRow, e: Event) -> None:
    t.apply(f"|-field{'start' if e.data['action'] == 'start' else 'end'}|{e.data['condition']}")


def _weather(t: Translator, row: RawRow, e: Event) -> None:
    t.apply(f"|-weather|{e.data['weather']}" + ("|[upkeep]" if e.data["upkeep"] else ""))


def _hazard(t: Translator, row: RawRow, e: Event) -> None:
    verb = "-sidestart" if e.data["action"] == "start" else "-sideend"
    t.apply(f"|{verb}|{e.side}: {'me' if e.side == 'p1' else 'opponent'}|{e.data['condition']}")


def _knock_off(t: Translator, row: RawRow, e: Event) -> None:
    target = t.member(e.data["target_side"], e.data["target_name"])
    if target is None:
        return t.unmapped(row, f"unknown target {e.data['target_name']!r}")
    t.apply(f"|-enditem|{t.ident(e.data['target_side'], target)}|{e.data['item']}|[from] move: Knock Off")


def _copyboost(t: Translator, row: RawRow, e: Event) -> None:
    member, target = t.member(e.side, e.name), t.member(e.data["target_side"], e.data["target"])
    if member is None or target is None or e.side is None:
        return t.unmapped(row, "unknown Psych Up user or target")
    t.apply(f"|-copyboost|{t.ident(e.side, member)}|{t.ident(e.data['target_side'], target)}")


def _last_move(t: Translator) -> list[str] | None:
    """The fields of this turn's latest `|move|` line, or None before any move this turn."""
    for line in reversed(t.out.lines):
        if line.startswith("|turn|"):
            return None
        if line.startswith("|move|"):
            return line.split("|")
    return None


def _miss(t: Translator, row: RawRow, e: Event) -> None:
    """The game's only miss text is "<target> avoided the attack!": the attacker is the latest mover. Without
    this turn's move row (the start of the turn was not captured) the row stays informational."""
    if (member := _need(t, row, e)) is None or e.side is None:
        return
    move = _last_move(t)
    if move is None or move[2].startswith(e.side):
        t.out.informational += 1
        return
    t.apply(f"|-miss|{move[2]}|{t.ident(e.side, member)}")


def _hitcount(t: Translator, row: RawRow, e: Event) -> None:
    """The hit count row ("The Pokémon was hit N times!") names no one: the latest move's target took the hits."""
    move = _last_move(t)
    if move is None or len(move) < 5 or not move[4]:
        t.out.informational += 1
        return
    t.apply(f"|-hitcount|{move[4]}|{e.data['count']}")


def _prepare(t: Translator, row: RawRow, e: Event) -> None:
    member = t.member(e.side, e.name) if e.side else None
    if member is None or e.side is None:
        t.out.informational += 1
        return
    ident = t.ident(e.side, member)
    move = next((line.split("|")[3] for line in reversed(t.out.lines) if line.startswith(f"|move|{ident}|")), None)
    if move is None:
        return t.unmapped(row, "charge turn without its move")
    t.apply(f"|-prepare|{ident}|{move}")


HANDLERS: dict[str, Handler] = {
    "confusion_damage": _confusion_damage,
    "volatile_active": _volatile_active,
    "fail": _fail,
    "send_out": _send_out,
    "drag": _send_out,
    "move": _move,
    "hp": _hp,
    "faint": _faint,
    "illusion_end": _illusion_end,
    "ability": _ability,
    "ability_change": _ability,
    "item": _item,
    "item_reveal": _item_reveal_or_end,
    "item_end": _item_reveal_or_end,
    "mega_reacting": mega_reacting,
    "mega": mega,
    "boost": _boost,
    "clearnegativeboost": _clear_negative,
    "status": _status_like,
    "cure": _status_like,
    "cant": _status_like,
    "volatile": _status_like,
    "typechange": _typechange,
    "transform": _transform,
    "form_change": _form_change,
    "steal_eat": _steal_eat,
    "volatile_end": _volatile_end,
    "field": _field,
    "weather": _weather,
    "hazard": _hazard,
    "knock_off": _knock_off,
    "copyboost": _copyboost,
    "prepare": _prepare,
    "miss": _miss,
    "hitcount": _hitcount,
}


def dispatch(t: Translator, row: RawRow, e: Event) -> None:
    """Normalize item text, remember the event for confusion damage, and run the handler."""
    if "item" in e.data:
        normalized = game.canonical_item(e.data["item"])
        if normalized is None:
            t.previous_event = e
            return t.unmapped(row, "Item text cannot be resolved uniquely")
        if normalized != e.data["item"]:
            t.out.inferred.append(
                {"kind": "item_ocr_correction", "raw": e.data["item"], "normalized": normalized, "row": row.row_id}
            )
            e.data["item"] = normalized
    previous, t.previous_event = t.previous_event, e
    handler = HANDLERS.get(e.kind)
    if handler is not None:
        t.previous_event = previous if e.kind == "confusion_damage" else e
        handler(t, row, e)
        t.previous_event = e
    elif e.kind in INFORMATIONAL_KINDS:
        t.out.informational += 1
    else:
        t.unmapped(row, e.data.get("reason", e.kind))

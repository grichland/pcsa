"""The battle's LIVE event list, grown as the log is read: battle_start, a `turn` event per turn, one event per new
row (through events/live.Converter, the pill side from the reader's rows), `gap` events from each visit, an
`enter` for an opposing switch the read lost (events/inferred), and `refused`/`end`. The list only grows; a
rebuild from the same BattleLog (after a restart) gives the same rows again. battle_start's `opp_preview` is
corrected when an opposing Pokémon outside its six names enters (the player's side then resends the battle).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pcsa.battle.ids import to_id
from pcsa.battle.log_rows import RawRow, parse_row, split_category
from pcsa.battle.teams import OwnMember, Team
from pcsa.decide.api import LogRow
from pcsa.events.forms import rename
from pcsa.events.gaps import gap, visit_gaps
from pcsa.events.inferred import OppActive, misread_slot
from pcsa.events.live import Converter
from pcsa.events.live_ids import (
    Teams,
    ability_id,
    base_id,
    form_by_move,
    item_id,
    mon_id,
    move_id,
    preview_id,
    species_id,
)
from pcsa.events.start import STATS, battle_start
from pcsa.phone.log_record import row_record
from pcsa.run.battle_state import BattleRecord

END_RESULTS = {"win": "win", "loss": "loss", "opponent_forfeit": "forfeit", "unknown": "unknown"}


def set_of(member: OwnMember) -> dict:
    """One LIVE SET from the team sheet; fields the sheet did not read are left out."""
    out: dict = {"species": species_id(member.species or member.name)}
    if member.item:
        out["item"] = item_id(member.item)[0]
    if member.ability:
        out["ability"] = ability_id(member.ability, out["species"])[0]
    out["moves"] = [move_id(m) for m in member.moves if m]
    if member.nature:
        out["nature"] = to_id(member.nature)
    for key, values in (("points", member.points), ("stats", member.stats)):
        if values and all(isinstance(values.get(s), int) for s in STATS):
            out[key] = {s: values[s] for s in STATS}
    return out


def preview_names(preview) -> list[str | None]:
    """Display names of the preview slots (the top sprite match, form included); None where unread."""
    out: list[str | None] = []
    for slot in preview or ():
        top = slot.get("top") if isinstance(slot, dict) else getattr(slot, "top", None)
        out.append(top[0][0] if top and top[0] and top[0][0] else None)
    return out


def preview_ids(preview) -> tuple[str | None, ...]:
    return tuple(preview_id(n) if n else None for n in preview_names(preview))


def sprite_name(sprite: dict | None) -> tuple[str | None, float | None]:
    if not sprite or sprite.get("status") != "matched":
        return None, None
    return sprite.get("name"), sprite.get("score")


@dataclass
class BattleEvents:
    tag: str
    team: Team | None
    preview: tuple = ()
    start: dict = field(init=False)
    events: list[dict] = field(default_factory=list)
    converter: Converter = field(init=False)
    opp: OppActive = field(init=False)
    turn: int = 0  # the last `turn` event sent
    sent: dict[int, int] = field(default_factory=dict)  # turn -> rows sent
    visits: int = 0  # b.log.visits already scanned for gaps
    late_rows: int = 0  # rows of a turn older than the last `turn` event
    preview_fixes: list[dict] = field(default_factory=list)  # preview slots renamed by an entering Pokémon
    from_screen: int = 0  # events added from the screen for rows the log read lost
    _kinds: dict[tuple[int, int, str], str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sets = [set_of(m) for m in self.team.members] if self.team else []
        self.start = battle_start(self.tag, sets, list(preview_ids(self.preview)))
        self.converter = Converter(Teams((), tuple(preview_names(self.preview))))
        self.opp = OppActive(preview_ids(self.preview))

    # what a Request carries ---------------------------------------------------------------------
    def all(self) -> tuple[dict, ...]:
        return (self.start, *self.events)

    def gaps(self) -> list[dict]:
        return [e for e in self.events if e["k"] == "gap"]

    def quality(self, b: BattleRecord) -> str:
        if not b.log.turns:
            return "none"
        return "gaps" if self.gaps() or b.log.failures else "full"

    def rows(self, b: BattleRecord) -> tuple[LogRow, ...]:
        """The whole battle log as read so far (texts as the latest vote has them)."""
        out = []
        for turn in sorted(b.log.turns):
            for index, text in enumerate(b.log.turns[turn]):
                rec = row_record(b.log, turn, index)
                sprite, score = sprite_name(rec.get("sprite"))
                key = (turn, index, text)
                if key not in self._kinds:
                    self._kinds[key] = parse_row(_raw(turn, index, text, rec.get("sprite"))).kind
                out.append(LogRow(f"{turn}-{index}", turn, index, text, sprite, score, self._kinds[key]))
        return tuple(out)

    # growing the list ---------------------------------------------------------------------------
    def update(self, b: BattleRecord) -> list[dict]:
        """Everything new in b.log since the last call, appended; returns the new events."""
        before = len(self.events)
        brought = tuple(m.species for m in self._brought(b))
        if brought != self.converter.teams.own:
            self.converter.teams = Teams(brought, self.converter.teams.opp_preview)
        found = []
        for visit in b.log.visits[self.visits :]:
            found += visit_gaps(visit)
        self.visits = len(b.log.visits)
        gaps_by_turn: dict[int, list[dict]] = {}
        for g in found:
            gaps_by_turn.setdefault(g["n"], []).append(g)
        for turn in sorted(set(b.log.turns) | set(gaps_by_turn)):
            rows = b.log.turns.get(turn, [])
            start = self.sent.get(turn, 0)
            if len(rows) <= start and turn not in gaps_by_turn:
                continue
            if turn > self.turn:
                self.to_turn(turn, b)
            elif turn < self.turn and len(rows) > start:
                self.late_rows += len(rows) - start
            self.events += gaps_by_turn.pop(turn, [])
            for index in range(start, len(rows)):
                event = self._event(b, turn, index)
                if turn < self.turn and _own(event, "enter") and event.get("mon") == self.own_active():
                    continue  # an old row of an entry the screen already gave
                self._fix_preview(event)
                self._fix_form(event)
                self.events += self.opp.step(event, late=turn < self.turn)
            self.sent[turn] = max(start, len(rows))
        return self.events[before:]

    def own_active(self) -> str | None:
        """Our Pokémon on the field as the events so far have it."""
        for e in reversed(self.events):
            if e.get("side") == "own" and e.get("k") in ("enter", "withdraw"):
                return e.get("mon") if e["k"] == "enter" else None
        return None

    def screen_shows(self, mon: str | None, mega: bool = False) -> list[dict]:
        """What the action screen shows and the log read lost: our active's `enter`, our Mega Evolution."""
        added: list[dict] = []
        if mon and self.own_active() != mon:
            added.append({"k": "enter", "side": "own", "mon": mon})
        if mon and mega and not any(_own(e, "mega") for e in self.events):
            added.append({"k": "mega", "side": "own", "mon": mon})
        self.events += added
        self.from_screen += len(added)
        return added

    def to_turn(self, n: int, b: BattleRecord) -> None:
        """`turn` events up to n; a turn that closes with no row read gets a `gap` first."""
        while self.turn < n:
            if self.turn >= 1 and not b.log.turns.get(self.turn):
                self.events.append(gap(self.turn, "turn"))
            self.turn += 1
            self.events.append({"k": "turn", "n": self.turn})

    def _fix_preview(self, event: dict) -> None:
        """An opposing Pokémon enters that is none of the six preview names: the misread slot takes its name."""
        ids: list = list(self.start.get("opp_preview") or ())
        if event.get("k") != "enter" or event.get("side") != "opp" or not ids or None in ids:
            return
        mon = base_id(event["mon"])
        if mon in {base_id(i) for i in ids if i}:
            return
        slot = misread_slot(self.preview, ids, self.opp.entered, mon)
        if slot is None:
            return
        self.preview_fixes.append({"slot": slot, "was": ids[slot], "now": mon})
        ids[slot] = mon
        self.start = {**self.start, "opp_preview": ids}
        self.opp.preview = tuple(ids)

    def _fix_form(self, event: dict) -> None:
        """Their Pokémon uses a move only another form of it learns: the preview left the form out or misread it.
        The slot, the rows' names and the events so far take the form; CG then gets the battle again."""
        if event.get("k") != "move" or event.get("side") != "opp" or not event.get("mon"):
            return
        was = event["mon"]
        now = form_by_move(was, str(event.get("move") or ""))
        if now is None:
            return
        for e in (*self.events, event):
            rename(e, was, now)
        ids: list = list(self.start.get("opp_preview") or ())
        at = [i for i, ident in enumerate(ids) if ident and base_id(ident) == base_id(was)] or [
            i for i, ident in enumerate(ids) if ident is None
        ]
        if at:
            ids[at[0]] = now
            names = list(self.converter.teams.opp_preview)
            names[at[0] : at[0] + 1] = [now]
            self.converter.teams = Teams(self.converter.teams.own, tuple(names))
            self.start = {**self.start, "opp_preview": ids}
            self.opp.preview = tuple(ids)
        self.opp.active = now if self.opp.active == was else self.opp.active
        self.opp.fainted = {now if m == was else m for m in self.opp.fainted}
        self.preview_fixes.append({"slot": at[0] if at else None, "was": was, "now": now, "by": event.get("move")})

    def refused(self, order: dict) -> None:
        """Only for a choice the game refused; a repeated tap is `retry` on the request."""
        self.events.append({"k": "refused", "order": order})

    def end(self, result: str) -> None:
        """`result` is a run outcome; LIVE's `end` mirrors battle_end, where the opponent's forfeit is "forfeit"."""
        self.events.append({"k": "end", "result": END_RESULTS.get(result, "unknown")})

    def _event(self, b: BattleRecord, turn: int, index: int) -> dict:
        rec = row_record(b.log, turn, index)
        if rec.get("read") == "unread":  # the reader flagged it: sent as unread with its reason
            return {"k": "unread", "text": rec["text"], **({"category": rec["reason"]} if rec.get("reason") else {})}
        return self.converter.row(_raw(turn, index, rec["text"], rec.get("sprite")), rec.get("side"))

    def _brought(self, b: BattleRecord) -> list[OwnMember]:
        if self.team is None or not b.requested:
            return []
        return [self.team.members[i] for i in b.requested if 0 <= i < len(self.team.members)]

    def mon(self, member: OwnMember | None) -> str | None:
        """Our member's species id as the events name it."""
        if member is None:
            return None
        return mon_id(member.species, "own", self.converter.teams)


def _own(event: dict, kind: str) -> bool:
    return event.get("k") == kind and event.get("side") == "own"


def _raw(turn: int, index: int, text: str, sprite: dict | None) -> RawRow:
    category, message = split_category(text)
    return RawRow(f"T{turn}#{index}", turn, index, text, category, message, sprite)

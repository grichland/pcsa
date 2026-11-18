"""One log row -> one LIVE.md event. Rows are never merged and no field is filled from another row."""

from __future__ import annotations

import re

from pcsa.battle.log_patterns import APOS
from pcsa.battle.log_rows import Event, RawRow, parse_row, split_category
from pcsa.events.live_ids import (
    ALREADY,
    CAUSES,
    EFFECTIVENESS,
    FORM_IDS,
    TRAILING_CAUSES,
    TRAILING_FAIL,
    TRAILING_ONCE,
    WEATHER_ROWS,
    Teams,
    ability_id,
    effect_id,
    item_id,
    legal,
    mega_form,
    mon_id,
    move_id,
    priority_of,
    species_id,
)
from pcsa.events.live_more import special
from pcsa.events.live_more import who as _who

SIDES = {"p1": "own", "p2": "opp"}


class Unreadable(Exception):
    """A name with no single nearest legal id: the row is sent as `unread` (LIVE.md CT-15)."""


class Converter:
    """Stateless per row except for the counters the report reads."""

    def __init__(self, teams: Teams) -> None:
        self.teams = teams
        self.repairs: list[dict] = []  # OCR-repaired item/ability names
        self.side_unknown: list[str] = []  # sideless rows whose pill colour was not read
        self._pill: str | None = None  # the current row's pill side
        self.errors: list[dict] = []  # rows the mapping raised on; sent as unread
        self.unreadable: list[dict] = []  # rows naming something with no single nearest legal id

    # helpers ------------------------------------------------------------------------------------
    def _mon(self, side: str | None, name: str | None) -> dict:
        if name is None:
            return {}
        ident = mon_id(name, side, self.teams)
        if ident not in legal("species"):
            raise Unreadable(name)
        out = {"mon": ident}
        return {"side": side, **out} if side else out

    def _named(self, name: str, text: str) -> dict:
        """Side for a row whose text names no side (Held Item, Ability): the colour of its category pill."""
        if self._pill is None:
            self.side_unknown.append(text)
        return self._mon(self._pill, name)

    def _item(self, name: str, text: str) -> str:
        ident, repaired = item_id(name)
        if ident not in legal("items"):
            raise Unreadable(name)
        if repaired:
            self.repairs.append({"text": text, "item": ident})
        return ident

    def _ability(self, name: str, mon: str | None, text: str) -> str:
        ident, repaired = ability_id(name, mon)
        if ident not in legal("abilities"):
            raise Unreadable(name)
        if repaired:
            self.repairs.append({"text": text, "ability": ident})
        return ident

    # rows the shared parser does not split the way LIVE.md needs ----------------------------------
    def _special(self, msg: str, text: str) -> dict | None:  # noqa: C901, PLR0911, PLR0912
        if (out := special(self, msg, text)) is not None:
            return out
        if m := re.fullmatch(r"(.+?) lost (\d+) PP from (.+?)!", msg):  # Spite, Eerie Spell
            if move_id(m[3]) not in legal("moves"):
                raise Unreadable(m[3])
            return {"k": "pp", **self._mon(*_who(m[1])), "move": move_id(m[3]), "lost": int(m[2])}
        if m := re.fullmatch(r"(.+?) frisked (.+?) and found its (.+?)!", msg):
            fs, fn = _who(m[1])
            ts, tn = _who(m[2])
            return {"k": "item", **self._mon(ts, tn), "item": self._item(m[3], text), "what": "found",
                    "of": self._mon(fs, fn)}
        if m := re.fullmatch(r"(.+?) is about to be attacked by its (.+?)!", msg):
            return {"k": "item", **self._mon(*_who(m[1])), "item": self._item(m[2], text), "what": "found"}
        if m := re.fullmatch(r"(.+?) harvested (?:a |an |one )?(.+?)!", msg):
            return {"k": "item", **self._mon(*_who(m[1])), "item": self._item(m[2], text), "what": "found"}
        if m := re.fullmatch(rf"The power of (.+?){APOS}s (\w+)-type moves rose!", msg):
            return {"k": "volatile", **self._mon(*_who(m[1])), "effect": "type_power_rose", "detail": m[2].lower()}
        if m := re.fullmatch(r"(.+?) endured the hit!", msg):
            return {"k": "volatile_hit", **self._mon(*_who(m[1])), "effect": "endured_hit"}
        if m := re.fullmatch(r"(.+?) braced itself!", msg):
            return {"k": "volatile", **self._mon(*_who(m[1])), "effect": "endure"}
        if m := re.fullmatch(r"(.+?) slept and restored its HP!", msg):
            return {"k": "status", **self._mon(*_who(m[1])), "status": "slp", "effect": "slept_and_restored"}
        if m := re.fullmatch(r"(.+?) is already (poisoned|badly poisoned|burned|paralyzed|asleep)[.!]?", msg):
            return {"k": "fail", **self._mon(*_who(m[1])), "reason": ALREADY[m[2]]}
        if m := re.fullmatch(r"(.+?) (burned itself out|used up all its electricity)!", msg):
            lost = "fire" if m[2].startswith("burned") else "electric"
            return {"k": "types", **self._mon(*_who(m[1])), "lose": lost}
        if re.fullmatch(r"All Pok[ée]mon that heard the song will ?faint in three turns!", msg):
            return {"k": "once", "effect": "perishsong"}
        if m := re.fullmatch(r"Quick Draw made (.+?) move faster!", msg):
            return {"k": "ability", **self._mon(*_who(m[1])), "ability": "quickdraw"}
        if m := re.fullmatch(r"(Your side|The opposing side|Your team|The opposing team) is too nervous to eat Berries!", msg):
            return {"k": "blocked", "side": "opp" if "opposing" in m[1] else "own", "effect": "unnerve", "what": "berries"}
        if m := re.fullmatch(r"(.+?) fell straight down!", msg):
            return {"k": "volatile", **self._mon(*_who(m[1])), "effect": "smackdown"}
        if m := re.fullmatch(r"(.+?) is preparing to tell a chillingly bad joke!", msg):
            return {"k": "once", **self._mon(*_who(m[1])), "effect": "chillyreception"}
        if m := re.fullmatch(rf"(.+?){APOS}s (.+?) (?:cured its \w+|woke it up|defrosted it)!", msg):
            status = {"poisoning": "psn", "paralysis": "par", "burn": "brn"}.get(msg.rsplit(" ", 1)[-1].rstrip("!"))
            out = {"k": "cure", **self._mon(*_who(m[1])), "item": self._item(m[2], text)}
            if "woke" in msg:
                status = "slp"
            elif "defrost" in msg:
                status = "frz"
            return {**out, "status": status} if status else out
        if m := re.fullmatch(rf"The (.+?) Gem strengthened (.+?){APOS}s power!", msg):
            return {"k": "item", "item": self._item(m[1] + " Gem", text), "what": "used"}
        return None

    # main ------------------------------------------------------------------------------------------
    def row(self, row: RawRow, pill: str | None = None) -> dict:
        """`pill`: "own"/"opp" from the category pill's colour, None when there is no pill or it was not read."""
        self._pill = pill if pill in ("own", "opp") else None
        msg = re.sub(r"\s+", " ", row.message.strip())
        try:
            out = self._special(msg, row.text)
            if out is None:
                out = self._from_event(parse_row(row), msg)
        except Unreadable as name:
            self.unreadable.append({"text": row.text, "name": str(name)})
            out = None
        except (KeyError, TypeError, ValueError) as error:  # counted, never silent (rule 4)
            self.errors.append({"text": row.text, "error": repr(error)})
            out = None
        if out is None:
            out = {"k": "unread", "text": row.text}
            if row.category:
                out["category"] = row.category
        return out

    def _from_event(self, e: Event, msg: str) -> dict | None:
        side = SIDES.get(e.side or "", None)
        d, text = e.data, e.row.text
        mon = self._mon(side, e.name) if e.name else {}
        kind = e.kind
        for group in (self._switch_move, self._hp_status, self._boost_ability_item, self._form_field):
            if (out := group(e, msg, kind, side, d, mon, text)) is not None:
                return out
        return None  # unknown, overlay and anything unmapped -> unread

    def _switch_move(self, e: Event, msg: str, kind: str, side: str | None, d: dict, mon: dict, text: str) -> dict | None:  # noqa: C901, PLR0911, PLR0912
        if kind == "send_out":
            return {"k": "enter", **mon}
        if kind == "drag":
            return {"k": "enter", **mon, "forced": True}
        if kind == "withdraw":
            return {"k": "withdraw", **mon}
        if kind == "trailing" and d.get("what") == "recall":
            return {"k": "withdraw", **mon, "pivot": True}
        if kind == "move":
            if move_id(d["move"]) not in legal("moves"):
                raise Unreadable(d["move"])
            out = {"k": "move", **mon, "move": move_id(d["move"])}
            if (p := priority_of(d.get("label"))) is not None:
                out["priority"] = p
            return out
        if kind == "prepare":
            return {"k": "prepare", **mon}
        if kind == "cant":
            return {"k": "cant", **mon, "reason": effect_id(d["reason"])}
        if kind == "trailing" and d.get("what") == "protect":
            return {"k": "protect", **mon}
        if kind == "miss":
            return {"k": "miss", **mon}
        if kind == "fail":
            return {"k": "fail"}
        if kind == "trailing" and d.get("what") in TRAILING_FAIL:  # the row names why (X1)
            reason = TRAILING_FAIL[d["what"]] if d["what"] != "heal_failed" or "HP is full" in msg else None
            return {"k": "fail", **mon, **({"reason": reason} if reason else {})}
        if kind == "effectiveness":
            return {"k": "result", "how": EFFECTIVENESS[d["text_variant"]], **mon}
        if kind == "trailing" and d.get("what") == "ohko":
            return {"k": "result", "how": "ohko"}
        if kind == "hitcount":
            return {"k": "hits", "n": d["count"]}
        return None

    def _hp_status(self, e: Event, msg: str, kind: str, side: str | None, d: dict, mon: dict, text: str) -> dict | None:  # noqa: C901, PLR0911, PLR0912
        if kind == "hp":
            out = {"k": "hp", **mon}
            if d.get("direction"):
                out["dir"] = d["direction"]
            if side == "own":
                return {**out, "before": d["before"], "after": d["after"]}
            return {**out, "before_pct": d["before"], "after_pct": d["after"]}
        if kind == "cause":
            effect = CAUSES.get(d["cause"])
            if effect is None:  # "X is hurt by MOVE!": the trapping move names the cause
                m = re.search(r"is hurt by (.+?)!", msg)
                effect = move_id(m[1]) if m else effect_id(d["cause"])
            return {"k": "cause", "effect": effect, **mon}
        if kind == "trailing" and d.get("what") in TRAILING_CAUSES:
            return {"k": "cause", "effect": TRAILING_CAUSES[d["what"]], **mon}
        if kind == "full_heal":
            return {"k": "cause", "effect": "healingwish", **mon}
        if kind == "faint":
            return {"k": "faint", **mon}
        if kind == "status":
            return {"k": "status", **mon, "status": d["status"]}
        if kind == "cure":
            out = {"k": "cure", **mon, "status": d["status"]}
            return {**out, "item": self._item(d["item"], text)} if d.get("item") else out
        if kind == "volatile":
            effect = effect_id(d["effect"])
            if effect.startswith("perish"):
                n = effect[len("perish"):].replace("o", "0")
                return {"k": "volatile", **mon, "effect": "perishsong", **({"detail": int(n)} if n.isdigit() else {})}
            out = {"k": "volatile", **mon, "effect": effect}
            return {**out, "detail": move_id(d["detail"])} if d.get("detail") else out
        if kind == "volatile_active":
            return {"k": "volatile_hit", **mon, "effect": effect_id(d["effect"])}
        if kind == "confusion_damage":
            return {"k": "volatile_hit", "effect": "confusion_self_hit"}
        if kind == "volatile_end":
            return {"k": "volatile_end", **mon, "effect": effect_id(d["effect"])}
        if kind == "illusion_end":
            return {"k": "volatile_end", **mon, "effect": "illusion"}
        return None

    def _boost_ability_item(self, e: Event, msg: str, kind: str, side: str | None, d: dict, mon: dict, text: str) -> dict | None:  # noqa: C901, PLR0911, PLR0912
        if kind == "boost":
            stats = d.get("stats")
            if not stats:
                return None
            many = {"stats": list(stats)}
            if d["amount"] == 0:
                return {"k": "boost_limit", **mon, **many, "dir": "up" if d["sign"] > 0 else "down"}
            return {"k": "boost", **mon, **many, "by": d["amount"] * d["sign"]}
        if kind == "boost_blocked":
            return {"k": "blocked", **mon, "effect": "stat_drop", "what": d["stat"]}
        if kind == "trailing" and d.get("what") == "stats_not_lowered":
            return {"k": "blocked", **mon, "effect": "stat_drop"}
        if kind == "trailing" and d.get("what") == "terrain_block":
            return {"k": "blocked", **mon, "effect": "psychicterrain"}
        if kind == "clearallboost":
            return {"k": "boost_reset", "what": "all"}
        if kind == "clearnegativeboost":
            return {"k": "boost_reset", **mon, "what": "negative"}
        if kind == "copyboost":
            target = self._mon(SIDES.get(d["target_side"]), d["target"])
            return {"k": "boost_reset", **mon, "what": "copy", "of": target}
        if kind == "ability":
            where = mon if e.side else self._named(e.name, text)
            return {"k": "ability", **where, "ability": self._ability(d["ability"], where.get("mon"), text)}
        if kind == "trailing" and d.get("what") == "trace":
            return {"k": "ability", **mon, "ability": self._ability(d["ability"], mon.get("mon"), text)}
        if kind == "ability_change":
            where = self._named(e.name, text)
            return {"k": "ability_change", **where, "from": self._ability(d["old"], None, text),
                    "to": self._ability(d["new"], None, text)}
        if kind == "item" and e.side is None:  # "Held Item:" row
            return {"k": "item", **self._named(e.name, text), "item": self._item(d["item"], text), "what": "activated"}
        if kind == "item":  # "X's BERRY lessened the damage" and similar
            return {"k": "item", **mon, "item": self._item(d["item"], text), "what": "activated"}
        if kind == "trailing" and d.get("what") == "quickclaw":
            return {"k": "item", **mon, "item": self._item(d["item"], text), "what": "activated"}
        if kind == "item_reveal":  # "floats in the air with its Air Balloon"
            return {"k": "item", **mon, "item": self._item(d["item"], text), "what": "activated"}
        if kind == "trailing" and d.get("what") == "focus_sash":
            m = re.search(r"hung on using its (.+?)!", msg)
            return {"k": "item", **mon, "item": self._item(m[1] if m else "Focus Sash", text), "what": "used"}
        if kind == "item_end":
            what = "removed" if "popped" in msg else "used"
            out = {"k": "item", **mon, "item": self._item(d["item"], text), "what": what}
            m = re.search(r" against (.+?)!", msg)
            return {**out, "of": self._mon(*_who(m[1]))} if m else out
        if kind == "knock_off":
            target = self._mon(SIDES.get(d["target_side"]), d["target_name"])
            return {"k": "item", **target, "item": self._item(d["item"], text), "what": "removed", "of": mon}
        if kind == "steal_eat":  # names the eater only; whose Berry it was is left to the builder
            return {"k": "item", "item": self._item(d["item"], text), "what": "removed", "of": mon}
        if kind == "mega_reacting":
            return {"k": "mega_stone", **mon, "stone": self._item(d["stone"], text)}
        if kind == "mega":
            out = {"k": "mega", **mon}
            form = mega_form(d["mega"], e.name)
            return {**out, "form": form} if form else out
        return None

    def _form_field(self, e: Event, msg: str, kind: str, side: str | None, d: dict, mon: dict, text: str) -> dict | None:  # noqa: C901, PLR0911, PLR0912
        if kind == "form_change":
            return {"k": "form", **mon, "form": FORM_IDS.get(d["form"], species_id(d["form"]))}
        if kind == "transform":
            return {"k": "transform", **mon, "into": species_id(d["target"])}
        if kind == "typechange":
            if not d.get("type"):
                return None
            return {"k": "types", **mon, "type": d["type"].lower()}
        if kind == "weather":
            found = WEATHER_ROWS.get(msg)
            return {"k": "weather", "weather": found[0], "what": found[1]} if found else None
        if kind == "field":
            return {"k": "field", "effect": effect_id(d["condition"]), "what": d["action"]}
        if kind == "hazard":
            return {"k": "side", "side": side, "effect": effect_id(d["condition"]), "what": d["action"]}
        if kind == "activate":  # Sticky Web acting on the Pokemon that came in
            return {"k": "side", **mon, "effect": effect_id(d["effect"]), "what": "acted"}
        if kind == "field_activate":
            return {"k": "once", "effect": effect_id(d["effect"])}
        if kind == "single":
            return {"k": "once", **mon, "effect": effect_id(d["effect"])}
        if kind == "trailing" and d.get("what") == "item_swap":  # Trick: the `received` rows follow
            return {"k": "once", **mon, "effect": "trick"}
        if kind == "trailing" and d.get("what") in TRAILING_ONCE:
            return {"k": "once", **mon, "effect": TRAILING_ONCE[d["what"]]}
        if kind == "trailing" and d.get("what") == "endure":
            return {"k": "volatile", **mon, "effect": "endure"}
        if kind == "revive":
            return {"k": "revive", **mon}
        return None


def convert_rows(rows: list[dict], teams: Teams) -> tuple[list[dict], Converter]:
    """log.jsonl rows (turn, index, text, side) -> events, with a `turn` event at each new turn heading."""
    conv, events, turn = Converter(teams), [], None
    for r in rows:
        if r["turn"] != turn:
            turn = r["turn"]
            events.append({"k": "turn", "n": turn})
        category, message = split_category(r["text"])
        raw = RawRow(f"T{r['turn']}#{r['index']}", r["turn"], r["index"], r["text"], category, message, r.get("sprite"))
        if r.get("read") == "unread":  # the reader flagged it
            events.append({"k": "unread", "text": r["text"], **({"category": r["reason"]} if r.get("reason") else {})})
        else:
            events.append(conv.row(raw, r.get("side")))
    return events, conv

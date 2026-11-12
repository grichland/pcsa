# From research/pcsa-belief/belief/hits.py (sha256 c18a386adf7e, 2026-11-12 14:37); keep diffable.
# pcsa changes are marked "pcsa:".
"""Hits read from protocol lines (p1's view): stage 0's reader (stage0/common.py Hit, read_hits, clean, dealt),
copied verbatim so the belief package needs no stage-0 import, plus an incremental reader (`Reader`) that reads each
line once: hits, turn orders (evidence.order_events) and Mega lines (evidence.mega_lines), with the same results.
"""

from __future__ import annotations

from dataclasses import dataclass

from poke_env.data.normalize import to_id_str


@dataclass
class Hit:
    """One damaging move use read from a turn's protocol lines."""

    turn: int
    line: int  # index of the |move| line in the full protocol
    side: str  # attacker side, "p1" / "p2"
    attacker: str  # team key
    target: str  # team key
    move: str  # id
    before: float | None = None  # target HP before (ours: HP; theirs: percent)
    after: float | None = None
    scale: float = 100.0  # the denominator of the HP readings
    hits: int = 0
    crit: bool = False
    miss: bool = False
    immune: bool = False
    blocked: bool = False  # Protect, Substitute, a failed move
    item_event: bool = False  # a berry, Focus Sash, Air Balloon ... during the hit
    survived_by: bool = False  # Sturdy / Endure / Focus Sash / Disguise
    ko: bool = False
    called: bool = False  # a move called by another, or a locked/charging continuation
    events_before: bool = False  # something changed the board between the decision and this hit
    first: bool = False  # the first move of the turn


def _key(ident: str) -> str:
    side, _, name = ident.partition(": ")
    return f"{side[:2]}: {name}"


def _hp(text: str) -> tuple[float, float] | None:
    token = text.split(" ", 1)[0]
    if "/" in token:
        a, b = token.split("/", 1)
        try:
            return float(a), float(b.rstrip("ygr"))  # "50/100g": the simulator's forced bar colour
        except ValueError:
            return None
    return (0.0, 0.0) if token == "0" else None


CHANGE_KINDS = (
    "-boost",
    "-unboost",
    "-setboost",
    "-status",
    "-curestatus",
    "-weather",
    "-fieldstart",
    "-fieldend",
    "-sidestart",
    "-sideend",
    "detailschange",
    "-formechange",
    "-mega",
    "switch",
    "drag",
    "-item",
    "-enditem",
    "-ability",
    "-clearboost",
    "-clearallboost",
    "-swapboost",
    "-invertboost",
    "-copyboost",
    "-transform",
    "-start",
    "-end",
)
SHIELDS = ("protect", "substitute", "shield", "bunker", "detect", "obstruct", "silk trap", "bulwark")


def read_hits(lines: list[str]) -> list[Hit]:
    """Every move aimed at the other side with its HP effect. HP is tracked per member over the whole protocol.
    pcsa: stage 0's full scan is the incremental Reader fed every line (the same code; research's
    tests/test_stage2.py checks the two agree), so the hit rules live in one place."""
    reader = Reader(())
    reader.feed(lines, len(lines))
    return reader.hits


def dealt(h: Hit) -> float | None:
    if h.before is None or h.after is None or not h.hits:
        return None
    return h.before - h.after


def clean(h: Hit) -> bool:
    """A hit whose damage is a plain reading of the formula."""
    d = dealt(h)
    return (
        h.hits > 0
        and not h.crit
        and not h.miss
        and not h.immune
        and not h.blocked
        and not h.item_event
        and not h.survived_by
        and not h.called
        and d is not None
        and d > 0
    )


# ------------------------------------------------------------------------------------------------ incremental
EARLY_TEXT = ("Lagging Tail", "Full Incense")


class Reader:
    """read_hits, evidence.order_events and evidence.mega_lines over a growing protocol, each line read once.

    `feed(lines, end)` reads lines[read:end]. Then `hits` holds every hit whose move line was read (the last one may
    still be completed by later lines: stage 1 used it as read so far, and so does the tracker), `orders` every
    finished turn's order event plus, through `orders_until(end)`, the current turn's when both sides have moved
    (order_events' end-of-lines sentinel), and `megas` every Mega Evolution / forme change line."""

    def __init__(self, early_movers: tuple[str, ...]) -> None:
        self.early_movers = early_movers
        self.read = 0
        # read_hits state
        self.hp: dict[str, tuple[float, float]] = {}
        self.hits: list[Hit] = []
        self.turn, self.cur, self.changed, self.first = 0, None, False, True
        # order_events state
        self.o_turn, self.o_cur, self.o_start = 0, [], 0
        self.orders: list[dict] = []
        # mega_lines
        self.megas: list[tuple[int, int, str]] = []
        self._lines: list[str] = []

    def feed(self, lines: list[str], end: int) -> None:
        self._lines = lines
        for i in range(self.read, end):
            self._line(i, lines[i])
        self.read = max(self.read, end)

    def _order_event(self, turn: int, cur: list, start: int) -> dict | None:
        mine = next((m for m in cur if m[1].startswith("p1")), None)
        theirs = next((m for m in cur if m[1].startswith("p2")), None)
        if mine is None or theirs is None:
            return None
        second = max(mine[0], theirs[0])
        text = "\n".join(self._lines[start : second + 1])
        early = any(x in text for x in self.early_movers) or any(x in text for x in EARLY_TEXT)
        return {
            "turn": turn,
            "first_line": min(mine[0], theirs[0]),
            "second_line": second,
            "ours": mine,
            "theirs": theirs,
            "they_first": theirs[0] < mine[0],
            "early": early,
        }

    def orders_until(self) -> list[dict]:
        """Finished turns' order events, then the current turn's if both sides have moved."""
        out = list(self.orders)
        if self.o_turn and self.o_cur:
            o = self._order_event(self.o_turn, self.o_cur, self.o_start)
            if o is not None:
                out.append(o)
        return out

    def _line(self, i: int, line: str) -> None:
        p = line.split("|")
        # mega_lines (the turn number is the one read_hits keeps: both count |turn| lines)
        if line.startswith(("|-mega|", "|detailschange|")):
            self.megas.append((i, self.turn, line.split("|")[2][:2]))
        # order_events
        if len(p) > 1 and p[1] == "turn":
            if self.o_turn and self.o_cur:
                o = self._order_event(self.o_turn, self.o_cur, self.o_start)
                if o is not None:
                    self.orders.append(o)
            self.o_turn, self.o_cur, self.o_start = int(p[2]), [], i
        elif len(p) > 3 and p[1] == "move" and not any(x.startswith("[from]") for x in p[5:]):
            self.o_cur.append((i, p[2], to_id_str(p[3])))
        # read_hits
        self._hit_line(i, p, line)

    def _hit_line(self, i: int, p: list[str], line: str) -> None:
        hp, out = self.hp, self.hits
        if len(p) < 2:
            return
        kind = p[1]
        if kind == "turn":
            self.turn, self.cur, self.changed, self.first = int(p[2]), None, False, True
            return
        if kind == "upkeep":
            self.cur = None
            return
        if kind == "move":
            self.cur = None
            actor, target = p[2], p[4] if len(p) > 4 else ""
            called = any(x.startswith("[from]") for x in p[5:])
            if target and ": " in target and target[:2] != actor[:2]:
                cur = Hit(
                    self.turn,
                    i,
                    actor[:2],
                    _key(actor),
                    _key(target),
                    to_id_str(p[3]),
                    called=called,
                    events_before=self.changed,
                    first=self.first,
                )
                cur.miss = any(x.startswith("[miss]") for x in p[5:])
                held = hp.get(cur.target)
                if held is not None:
                    cur.before, cur.scale = held
                out.append(cur)
                self.cur = cur
            self.first = False
            self.changed = True
            return
        if kind in CHANGE_KINDS:
            self.changed = True
        cur = self.cur
        if kind in ("switch", "drag", "replace"):
            self.cur = None
            v = _hp(p[4]) if len(p) > 4 else None
            if v is not None:
                hp[_key(p[2])] = v
            return
        if kind in ("-damage", "-heal", "-sethp"):
            _hp_line(hp, cur, kind, p)
            return
        if cur is not None:
            _flag(cur, kind, p, line)


def _flag(cur: Hit, kind: str, p: list[str], line: str) -> None:
    """The flags a line sets on the hit being read (pcsa: split out of Reader._hit_line, unchanged)."""
    who = _key(p[2]) if len(p) > 2 and ": " in p[2] else ""
    low = line.lower()
    if kind == "-crit" and who == cur.target:
        cur.crit = True
    elif kind == "-miss":
        cur.miss = True
    elif kind == "-immune" and who == cur.target:
        cur.immune = True
    elif kind in ("-fail", "cant"):
        cur.blocked = True
    elif kind == "-activate" and who == cur.target:
        if any(x in low for x in SHIELDS):
            cur.blocked = True
        if "endure" in low or "sturdy" in low or "disguise" in low:
            cur.survived_by = True
    elif kind in ("-enditem", "-item") and who == cur.target:
        cur.item_event = True
        if "focus sash" in low or "focus band" in low:
            cur.survived_by = True
    elif kind == "-ability" and who == cur.target and "sturdy" in low:
        cur.survived_by = True
    elif kind == "-end" and who == cur.target and "substitute" in low:
        cur.blocked = True
    elif (
        kind in ("-start", "detailschange", "-formechange")
        and who == cur.target
        and ("disguise" in low or "busted" in low)
    ):
        cur.survived_by = True


def _hp_line(hp: dict, cur: Hit | None, kind: str, p: list[str]) -> None:
    """A -damage / -heal / -sethp line: the member's HP, and the hit being read (pcsa: split out of
    Reader._hit_line, unchanged)."""
    v = _hp(p[3]) if len(p) > 3 else None
    k = _key(p[2])
    if v is None:
        return
    if v[1] == 0.0 and k in hp:
        v = (0.0, hp[k][1])
    tagged = any(x.startswith("[from]") for x in p[4:])
    if kind == "-damage" and cur is not None and k == cur.target and not tagged:
        cur.hits += 1
        cur.after = v[0]
        if v[0] <= 0:
            cur.ko = True
    hp[k] = v

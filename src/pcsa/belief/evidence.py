# From research/pcsa-belief/belief/evidence.py (sha256 b8b7c2d5721a, 2026-11-12 14:37); keep diffable.
# pcsa changes are marked "pcsa:".
"""Evidence read from protocol lines (p1's view): reveals, turn orders, hits.

Hits come from stage 0's reader (stage0/common.py read_hits, copied verbatim into belief/hits.py so the package
needs no stage-0 import); turn orders follow stage 0's order_events. Reveals are read here: moves used (not called
by another move), items named by the protocol (a Mega Evolution names its stone; an item that arrived by Trick or
the like spoils the member's item evidence), abilities named by the protocol (the holder is the subject of the line,
or the [of] member for lines whose subject is someone else).
"""

from __future__ import annotations

from dataclasses import dataclass

from poke_env.data.normalize import to_id_str

from pcsa.belief import hits as C  # stage 0's read_hits, Hit, clean, _key (a verbatim copy of stage0/common.py's)
from pcsa.players import opponent as opp

THEM = "p2"
TRADE = (
    "trick",
    "switcheroo",
    "thief",
    "covet",
    "bestow",
    "magician",
    "pickpocket",
    "pickup",
    "recycle",
    "harvest",
    "symbiosis",
)
OF_IS_HOLDER = frozenset(
    {
        "-damage",
        "-unboost",
        "-weather",
        "-fieldstart",
        "-sidestart",
        "-clearallboost",
        "-block",
        "-fail",
        "-boost",
        "-setboost",
        "-curestatus",
        "-status",
    }
)
NOT_MOVES = frozenset({"struggle", "recharge"})
ORDER_ABILITIES = frozenset(
    {
        "prankster",
        "galewings",
        "triage",
        "quickdraw",
        "stall",
        "myceliummight",
        "unburden",
        "slowstart",
        "surgesurfer",
        "protosynthesis",
        "quarkdrive",
    }
)
ORDER_MOVES = frozenset({"grassyglide", "pursuit", "focuspunch"})


@dataclass
class Reveal:
    line: int
    kind: str  # "switch" | "move" | "item" | "ability" | "mega" | "spoil"
    key: str  # "p2: Name"
    value: str = ""  # species / move / item / ability / mega form


def _of(parts: list[str]) -> str | None:
    return next((x[5:] for x in parts if x.startswith("[of] ")), None)


def _from_ability(parts: list[str]) -> str | None:
    tag = next((x for x in parts if x.startswith("[from] ability: ")), None)
    return tag[len("[from] ability: ") :] if tag else None


def reveals(lines: list[str], start: int = 0, end: int | None = None) -> list[Reveal]:
    """Opposing reveals in lines[start:end], in line order."""
    out: list[Reveal] = []
    end = len(lines) if end is None else end
    for i in range(start, end):
        line = lines[i]
        p = line.split("|")
        if len(p) < 3:
            continue
        kind = p[1]
        low = line.lower()
        if kind in ("switch", "drag", "replace") and p[2].startswith(THEM) and len(p) > 3:
            out.append(Reveal(i, "switch", C._key(p[2]), p[3].split(",", 1)[0]))
            continue
        if kind == "move" and p[2].startswith(THEM) and len(p) > 3:
            if not any(x.startswith("[from]") for x in p[4:]):
                m = to_id_str(p[3])
                if m not in NOT_MOVES:
                    out.append(Reveal(i, "move", C._key(p[2]), m))
            continue
        if kind == "-mega" and p[2].startswith(THEM) and len(p) > 4:
            out.append(Reveal(i, "mega", C._key(p[2]), to_id_str(p[4])))
            continue
        if kind in ("-item", "-enditem") and len(p) > 3:
            trade = kind == "-item" and any(
                ("[from] move: " + m) in low or ("[from] ability: " + m) in low for m in TRADE
            )
            if trade:
                for who in (p[2], _of(p) or ""):
                    if who.startswith(THEM):
                        out.append(Reveal(i, "spoil", C._key(who)))
                continue
            if p[2].startswith(THEM):
                out.append(Reveal(i, "item", C._key(p[2]), to_id_str(p[3])))
            continue
        if "[from] item: " in line:
            holder = _of(p) or p[2]
            if holder.startswith(THEM):
                item = next(x for x in p if x.startswith("[from] item: "))[len("[from] item: ") :]
                out.append(Reveal(i, "item", C._key(holder), to_id_str(item)))
        if kind == "-ability" and len(p) > 3:
            src = _from_ability(p)
            if src is None and p[2].startswith(THEM):
                out.append(Reveal(i, "ability", C._key(p[2]), to_id_str(p[3])))
            elif src is not None and to_id_str(src) == "trace" and p[2].startswith(THEM):
                out.append(Reveal(i, "ability", C._key(p[2]), "trace"))
            continue
        if kind == "-activate" and p[2].startswith(THEM) and len(p) > 3 and p[3].startswith("ability: "):
            out.append(Reveal(i, "ability", C._key(p[2]), to_id_str(p[3][len("ability: ") :])))
            continue
        src = _from_ability(p)
        if src is not None:
            holder = (_of(p) if kind in OF_IS_HOLDER else None) or p[2]
            if holder.startswith(THEM) and ": " in holder:
                out.append(Reveal(i, "ability", C._key(holder), to_id_str(src)))
    return out


def order_events(lines: list[str], end: int) -> list[dict]:
    """Per turn finished before `end`: the first move of each side when both moved (stage 0's order_events)."""
    out, cur, turn, start = [], [], 0, 0
    for i, line in enumerate(lines[:end] + ["|turn|999999"]):
        p = line.split("|")
        if len(p) > 1 and p[1] == "turn":
            if turn and cur:
                mine = next((m for m in cur if m[1].startswith("p1")), None)
                theirs = next((m for m in cur if m[1].startswith("p2")), None)
                if mine is not None and theirs is not None:
                    second = max(mine[0], theirs[0])
                    text = "\n".join(lines[start : second + 1])
                    early = (
                        any(x in text for x in opp.EARLY_MOVERS) or "Lagging Tail" in text or "Full Incense" in text
                    )
                    out.append(
                        {
                            "turn": turn,
                            "first_line": min(mine[0], theirs[0]),
                            "second_line": second,
                            "ours": mine,
                            "theirs": theirs,
                            "they_first": theirs[0] < mine[0],
                            "early": early,
                        }
                    )
            turn, cur, start = int(p[2]), [], i
            continue
        if len(p) > 3 and p[1] == "move" and not any(x.startswith("[from]") for x in p[5:]):
            cur.append((i, p[2], to_id_str(p[3])))
    return out


def mega_lines(lines: list[str]) -> list[tuple[int, int, str]]:
    """(line, turn, side) of every Mega Evolution / forme change."""
    out, turn = [], 0
    for i, line in enumerate(lines):
        if line.startswith("|turn|"):
            turn = int(line.split("|")[2])
        elif line.startswith(("|-mega|", "|detailschange|")):
            out.append((i, turn, line.split("|")[2][:2]))
    return out


read_hits = C.read_hits
clean = C.clean
dealt = C.dealt
Hit = C.Hit

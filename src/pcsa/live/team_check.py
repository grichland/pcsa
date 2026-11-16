"""Is the battle played with the three members the team selection asked for? If not, play the team brought.

15 Nov 2026 (run 20261115-034548, battles 31-32): a communication error during team selection brought the
selection screen back with 2 seconds left; the lineup taps came too late, the game chose Gholdengo, Garchomp and
Golisopod by itself ("Pokémon to send into battle were automatically chosen for you") and every decision of the
battle was made for Mamoswine, Cinderace and Primarina, the trio requested.

Every own member the battle shows (the active on the action screen, the party rows, the log's "Go! X!") is
compared with the requested trio. A member outside it means the lineup differs: the record switches to the team
actually brought (`b.requested`), keeps what was asked in `b.requested_at_preview`, and logs `team_mismatch`.
Members not seen yet are filled from the game's automatic lineup (the first three roster rows, lead first) when
the selection timed out, else from the requested trio; a later sighting corrects the guess.
"""

from __future__ import annotations

from pcsa.common import log
from pcsa.live.action_check import own_send_outs
from pcsa.run.battle_state import BattleRecord, same_name

TEAM_SIZE = 3
AUTOMATIC_LINEUP = (0, 1, 2)  # the game's pick when the selection timer runs out (battle 32: rows 1-3, row 1 led)


def roster_row(b: BattleRecord, name: str | None) -> int | None:
    rows = [i for i, member in enumerate(b.roster) if name and same_name(member, name)]
    return rows[0] if len(rows) == 1 else None


def _seen(b: BattleRecord, names) -> list[int]:
    """Roster rows of every own member seen so far in this battle (kept in b.observed["own_seen"])."""
    seen: list[str] = b.observed.setdefault("own_seen", [])
    for name in names:
        if name and roster_row(b, name) is not None and not any(same_name(name, s) for s in seen):
            seen.append(name)
    rows = [roster_row(b, name) for name in seen]
    return [r for r in rows if r is not None]


def check_team(b: BattleRecord, *, active: str | None = None, party=()) -> bool:
    """Adopt the team actually brought when what the battle shows is not the requested trio. True when it changed."""
    if b.requested is None or len(b.roster) < TEAM_SIZE:
        return False
    log_names = [name for turn in sorted(b.log.turns) for name in own_send_outs(b.log.turns[turn])]
    seen = _seen(b, [active, *party, *log_names])
    if all(row in b.requested for row in seen):
        return False
    lead = roster_row(b, b.observed.get("lead"))
    first = [lead] if lead is not None and lead in seen else []
    fill = AUTOMATIC_LINEUP if b.team_auto else tuple(b.requested)
    members = list(dict.fromkeys([*first, *seen, *fill, *range(len(b.roster))]))[:TEAM_SIZE]
    # Slot order is the party order assumed until a party screen shows it: the lead, then the lineup's own order.
    played = first + [r for r in (*fill, *members) if r in members and r not in first]
    played = list(dict.fromkeys(played))
    if len(played) < TEAM_SIZE or played == b.requested:
        return False
    if b.requested_at_preview is None:
        b.requested_at_preview = list(b.requested)
    before = list(b.names)
    b.requested = played
    b.visible_order = [0, 1, 2]
    b.fainted = set()
    log.warning(
        "team_mismatch",
        battle=b.index,
        requested=[b.roster[i] for i in b.requested_at_preview],
        tracked=before,
        seen=[b.roster[i] for i in seen],
        playing=list(b.names),
        automatic=b.team_auto,
    )
    b.failures.append(f"team mismatch: requested {before}, playing {list(b.names)}")
    return True

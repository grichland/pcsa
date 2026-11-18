"""assets/game/cg/vocab.json against CG: the sizes are champions_gym.VOCAB, and on a recorded battle the ids in each
observation's row_ids name what the Battle itself reports for that Pokémon (species or form, moves, item,
ability)."""

from __future__ import annotations

import pytest

cg = pytest.importorskip("champions_gym")

from champions_gym.live import Session  # noqa: E402

from pcsa.network import vocab  # noqa: E402
from tests.cg.recorded import Replayer, sent  # noqa: E402

COLS = {"species": "F1-3", "items": "F2-6", "abilities": "F2-7", "moves": "F7-13"}


def test_sizes_are_cgs() -> None:
    assert {k: len(v) for k, v in vocab.tables().items()} == dict(cg.VOCAB)
    assert vocab.name("items", 0) is None and vocab.name("items", 1) == "" and vocab.name("moves", 0) is None
    assert vocab.name("species", 0) is None and vocab.name("abilities", 0) is None
    assert vocab.name("species", len(vocab.tables()["species"])) is None


def decoded(ids, row: int) -> dict:
    col = {k: [int(c) for c in cg.FEATURES[f].columns()] for k, f in COLS.items()}
    return {
        "species": vocab.name("species", int(ids[row][col["species"][0]])),
        "item": vocab.name("items", int(ids[row][col["items"][0]])),
        "ability": vocab.name("abilities", int(ids[row][col["abilities"][0]])),
        "moves": [vocab.name("moves", int(ids[row][c])) for c in col["moves"]],
    }


def reported(p) -> dict:
    moves = [m for m, _ in p.moves]
    return {"names": {p.species, p.form}, "item": p.item, "ability": p.ability, "moves": moves + [None] * (4 - len(moves))}


def test_row_ids_name_what_the_battle_reports() -> None:
    player = Replayer()
    session = Session(player=player)
    for message in sent():
        session.handle(message)
    turns = [b for b in player.battles if b.kind != cg.KIND_PREVIEW]
    assert len(turns) == 7
    checked = 0
    for battle in turns:
        ids = cg.tensors(battle).row_ids
        picks = [p for p in battle.picks if p is not None]
        own = [battle.team[p] for p in picks] + [m for i, m in enumerate(battle.team) if i not in picks]
        theirs = {m.form: m for m in battle.opponent_team if m.form} | {m.species: m for m in battle.opponent_team}
        for row in range(12):
            got = decoded(ids, row)
            if got["species"] is None:
                continue
            member = own[row] if row < 6 else theirs[got["species"]]
            want = reported(member)
            assert got["species"] in want["names"], (battle.decision, row)
            assert got["moves"] == want["moves"], (battle.decision, row, got, want)
            assert got["item"] == want["item"] and got["ability"] == want["ability"], (battle.decision, row, got, want)
            checked += 1
    assert checked > 7 * 6

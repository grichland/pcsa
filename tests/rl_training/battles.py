"""A team-preview Battle built the way the arena and the live loop build one: preview protocol, then our six."""

from __future__ import annotations

from pcsa.battle.state import replay

OURS = {
    "Azumarill": (175, ["playrough", "aquajet", "bellydrum", "knockoff"], "sitrusberry", "hugepower"),
    "Slowbro": (170, ["surf", "slackoff", "psychic", "haze"], "leftovers", "regenerator"),
    "Charizard": (153, ["airslash", "flamethrower", "roost", "dragonpulse"], "charizarditex", "blaze"),
    "Garchomp": (183, ["earthquake", "dragonclaw", "swordsdance", "stoneedge"], "choicescarf", "roughskin"),
    "Gengar": (135, ["shadowball", "sludgebomb", "focusblast", "nastyplot"], "focussash", "cursedbody"),
    "Kingambit": (175, ["kowtowcleave", "suckerpunch", "ironhead", "swordsdance"], "blackglasses", "supremeoverlord"),
}
THEIRS = ["Tyranitar", "Corviknight", "Gardevoir", "Dragonite", "Rotom-Wash", "Volcarona"]


def preview_battle():
    lines = ["|player|p1|me|", "|player|p2|opponent|", "|gen|9", "|clearpoke"]
    lines += [f"|poke|p1|{name}, L50|" for name in OURS] + [f"|poke|p2|{name}, L50|" for name in THEIRS]
    lines.append("|teampreview|3")
    side = [
        {
            "ident": f"p1: {name}",
            "details": f"{name}, L50",
            "condition": f"{hp}/{hp}",
            "active": False,
            "stats": {"atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 100},
            "moves": moves,
            "baseAbility": ability,
            "ability": ability,
            "item": item,
        }
        for name, (hp, moves, item, ability) in OURS.items()
    ]
    request = {"teamPreview": True, "maxTeamSize": 3, "side": {"name": "me", "id": "p1", "pokemon": side}, "rqid": 1}
    return replay(lines, request, "preview-test")

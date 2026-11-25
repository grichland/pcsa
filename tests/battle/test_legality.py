"""Every item, ability, move, species and nature named in a source constant is in CG's legal tables.

A new constant whose name ends in _ITEMS, _ABILITIES, _MOVES or _SPECIES must be added to CONSTANTS below.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

from pcsa.battle import battle_text, game, log_patterns, names
from pcsa.battle.ids import to_id
from pcsa.common import paths
from pcsa.live import action_check
from pcsa.network import evidence
from pcsa.phone import team_sheet
from pcsa.run import battle_state

GAME_ID_SUFFIX = re.compile(r"_(ITEMS|ABILITIES|MOVES|SPECIES)$")
NOT_MOVES = {"recharge"}  # Showdown's pseudo-choice, never a move
NOT_GAME_IDS = {"VOCAB_ITEMS"}  # scout.py: names of screens to scout, not items
TEXT_SECTIONS = {"default", "par", "slp", "frz", "heal", "unboost", "trapped"}  # battle_text keys, not game ids
# Illegal names that are known and allowed, listed exactly so a new one fails. The display tables in names.py also
# cover species outside Champions; the game never shows them. The other three are candidates for removal.
KNOWN_ILLEGAL = {
    "run.battle_state.PIVOT_MOVES": ["Teleport"],
    "log_patterns effects": ["Heal Block", "Magma Storm"],
    "log_patterns template effects": ["healblock", "magmastorm"],
    "names.MEGA": [
        "Crucibelle-Mega", "Darkrai-Mega", "Diancie-Mega", "Heatran-Mega", "Latias-Mega", "Latios-Mega",
        "Mewtwo-Mega-X", "Mewtwo-Mega-Y", "Rayquaza-Mega", "Zeraora-Mega", "Zygarde-Mega",
    ],
    "names.AMBIGUOUS": [
        "Magearna-Mega", "Magearna-Original-Mega", "Tatsugiri-Curly-Mega", "Tatsugiri-Droopy-Mega",
        "Tatsugiri-Stretchy-Mega",
    ],
    "names.FORMS": [
        "Articuno-Galar", "Braviary-Hisui", "Corsola-Galar", "Darmanitan-Galar", "Dugtrio-Alola", "Electrode-Hisui",
        "Exeggutor-Alola", "Golem-Alola", "Lilligant-Hisui", "Marowak-Alola", "Moltres-Galar", "Muk-Alola",
        "Nidoran-F", "Nidoran-M", "Qwilfish-Hisui", "Rapidash-Galar", "Raticate-Alola", "Sandslash-Alola",
        "Sliggoo-Hisui", "Weezing-Galar", "Zapdos-Galar",
    ],
    "names.SPACED_SPECIES": [
        "Brute Bonnet", "Flutter Mane", "Gouging Fire", "Great Tusk", "Iron Boulder", "Iron Bundle", "Iron Crown",
        "Iron Hands", "Iron Jugulis", "Iron Leaves", "Iron Moth", "Iron Thorns", "Iron Treads", "Iron Valiant",
        "Mime Jr.", "Raging Bolt", "Roaring Moon", "Sandy Shocks", "Scream Tail", "Slither Wing", "Tapu Bulu",
        "Tapu Fini", "Tapu Koko", "Tapu Lele", "Type: Null", "Walking Wake",
    ],
}  # fmt: skip


def _moves_in(values: Iterable[str]) -> list[str]:
    return [v.removeprefix("move: ") for v in values]


def _template_values(key: str) -> list[str]:
    return [
        d[key] for _, d in log_patterns.TEMPLATE_EVENTS.values() if isinstance(d.get(key), str) and "{" not in d[key]
    ]


def _effects() -> list[str]:
    found = [
        d["effect"] for _, d in log_patterns.TEMPLATE_EVENTS.values() if str(d.get("effect", "")).startswith("move: ")
    ]
    found += [c for _, c in log_patterns.HAZARD_START + log_patterns.HAZARD_END if c and c.startswith("move: ")]
    found += [e for e, _ in log_patterns.TERRAIN.values()]
    return [e for e in found if "{" not in e]


def _rules(kind: str) -> list[str]:
    return list(game.rule(f"known_stat_multipliers.{kind}"))


CONSTANTS: dict[str, tuple[str, list[str]]] = {
    "run.battle_state.PIVOT_MOVES": ("moves", sorted(battle_state.PIVOT_MOVES)),
    "live.action_check.IGNORED_MOVES": ("moves", [m for m in action_check.IGNORED_MOVES if m not in NOT_MOVES]),
    "log_patterns.SCREENS": ("moves", log_patterns.SCREENS.split("|")),
    "log_patterns effects": ("moves", _moves_in(_effects())),
    "log_patterns causes": ("moves", _template_values("cause")),
    "log_patterns items": ("items", _template_values("item")),
    "log_patterns abilities": ("abilities", _template_values("ability")),
    "log_patterns template effects": ("any", [k for k, _ in log_patterns.TEMPLATE_EVENTS if k not in TEXT_SECTIONS]),
    "battle_text.CHAMPIONS_TEXT": (
        "any",
        [row[1] for row in battle_text.CHAMPIONS_TEXT if row[1] not in TEXT_SECTIONS],
    ),
    "network.evidence.ORDER_ITEMS": ("items", sorted(evidence.ORDER_ITEMS)),
    "network.evidence.SPEED_ITEMS": ("items", sorted(evidence.SPEED_ITEMS)),
    "network.evidence.ORDER_ABILITIES": ("abilities", sorted(evidence.ORDER_ABILITIES)),
    "rules.json items": ("items", _rules("items")),
    "rules.json abilities": ("abilities", _rules("abilities")),
    "team_sheet.NATURES": ("natures", [*team_sheet.NATURES.values(), team_sheet.NEUTRAL_NATURE]),
    "names.MEGA": ("species", list(names.MEGA.values())),
    "names.AMBIGUOUS": ("species", [s for forms in names.AMBIGUOUS.values() for s in forms]),
    "names.FORMS": ("species", list(names.FORMS.values())),
    "names.SPACED_SPECIES": ("species", sorted(names.SPACED_SPECIES)),
    "names.COSMETIC_BASES": ("species", sorted(names.COSMETIC_BASES)),
}


def legal(kind: str) -> set[str]:
    if kind == "natures":
        return {n["id"] for n in game.cg_table("natures")}
    if kind == "species":
        return set(game.dex())
    if kind == "any":
        return set().union(*(legal(k) for k in ("moves", "items", "abilities")))
    return set(game.cg_table(kind))


def illegal() -> dict[str, list[str]]:
    out = {}
    for name, (kind, values) in CONSTANTS.items():
        bad = sorted({v for v in values if to_id(v) not in legal(kind)})
        if bad:
            out[name] = bad
    return out


def test_named_ids_are_legal_except_the_listed_ones() -> None:
    assert illegal() == KNOWN_ILLEGAL


def test_every_game_id_constant_is_registered() -> None:
    registered = {key.rsplit(".", 1)[-1] for key in CONSTANTS}
    missing = []
    for path in sorted((paths.repo_root() / "src" / "pcsa").rglob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            for t in targets:
                if isinstance(t, ast.Name) and GAME_ID_SUFFIX.search(t.id) and t.id not in registered | NOT_GAME_IDS:
                    missing.append(f"{path.name}:{t.id}")
    assert missing == []

"""Game display names to Showdown species names: hand-written tables checked against the pinned dex.

Base species names pass through unchanged. A name that looks like a forme but is not in these tables
is kept as written and reported; it is never replaced by a different forme.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MEGA = {
    f"Mega {name}": f"{name}-Mega"
    for name in (
        "Venusaur Blastoise Beedrill Pidgeot Clefable Alakazam Victreebel Slowbro Gengar Kangaskhan Starmie Pinsir "
        "Gyarados Aerodactyl Dragonite Meganium Feraligatr Ampharos Steelix Scizor Heracross Skarmory Houndoom "
        "Tyranitar Sceptile Blaziken Swampert Gardevoir Sableye Mawile Aggron Medicham Manectric Sharpedo Camerupt "
        "Altaria Banette Chimecho Absol Glalie Salamence Metagross Latias Latios Rayquaza Staraptor Lopunny Garchomp "
        "Lucario Abomasnow Gallade Froslass Heatran Darkrai Emboar Excadrill Audino Scolipede Scrafty Eelektross "
        "Chandelure Golurk Chesnaught Delphox Greninja Pyroar Floette Malamar Barbaracle Dragalge Hawlucha Zygarde "
        "Diancie Crabominable Golisopod Drampa Zeraora Falinks Scovillain Glimmora Baxcalibur Crucibelle"
    ).split()
}
MEGA.update(
    {
        "Mega Charizard X": "Charizard-Mega-X",
        "Mega Charizard Y": "Charizard-Mega-Y",
        "Mega Raichu X": "Raichu-Mega-X",
        "Mega Raichu Y": "Raichu-Mega-Y",
        "Mega Mewtwo X": "Mewtwo-Mega-X",
        "Mega Mewtwo Y": "Mewtwo-Mega-Y",
        "Mega Absol Z": "Absol-Mega-Z",
        "Mega Garchomp Z": "Garchomp-Mega-Z",
        "Mega Lucario Z": "Lucario-Mega-Z",
    }
)
AMBIGUOUS = {
    "Mega Charizard": ("Charizard-Mega-X", "Charizard-Mega-Y"),
    "Mega Meowstic": ("Meowstic-M-Mega", "Meowstic-F-Mega"),
    "Mega Tatsugiri": ("Tatsugiri-Curly-Mega", "Tatsugiri-Droopy-Mega", "Tatsugiri-Stretchy-Mega"),
    "Mega Magearna": ("Magearna-Mega", "Magearna-Original-Mega"),
}
REGIONAL = {
    "Alolan": ("Alola", "Raichu Ninetales Persian Sandslash Marowak Muk Exeggutor Golem Dugtrio Raticate"),
    "Galarian": ("Galar", "Slowbro Slowking Stunfisk Weezing Rapidash Corsola Moltres Zapdos Articuno Darmanitan"),
    "Hisuian": (
        "Hisui",
        "Arcanine Avalugg Decidueye Goodra Samurott Typhlosion Zoroark Lilligant Braviary Electrode Sliggoo Qwilfish",
    ),
}
FORMS = {
    f"{prefix} {name}": f"{name}-{suffix}" for prefix, (suffix, names) in REGIONAL.items() for name in names.split()
}
FORMS.update(
    {
        "Paldean Tauros Combat Breed": "Tauros-Paldea-Combat",
        "Paldean Tauros Blaze Breed": "Tauros-Paldea-Blaze",
        "Paldean Tauros Aqua Breed": "Tauros-Paldea-Aqua",
        **{f"Rotom {k}": f"Rotom-{k}" for k in ("Heat", "Wash", "Frost", "Fan", "Mow")},
        **{f"{k} Rotom": f"Rotom-{k}" for k in ("Heat", "Wash", "Frost", "Fan", "Mow")},
        "Basculegion Male": "Basculegion",
        "Basculegion Female": "Basculegion-F",
        "Meowstic Male": "Meowstic",
        "Meowstic Female": "Meowstic-F",
        "Indeedee Male": "Indeedee",
        "Indeedee Female": "Indeedee-F",
        "Lycanroc Midday Form": "Lycanroc",
        "Lycanroc Midnight Form": "Lycanroc-Midnight",
        "Lycanroc Dusk Form": "Lycanroc-Dusk",
        "Aegislash Shield Forme": "Aegislash",
        "Aegislash Blade Forme": "Aegislash-Blade",
        "Palafin Zero Form": "Palafin",
        "Palafin Hero Form": "Palafin-Hero",
        "Toxtricity Amped Form": "Toxtricity",
        "Toxtricity Low Key Form": "Toxtricity-Low-Key",
        "Gourgeist Average Variety": "Gourgeist",
        "Gourgeist Small Variety": "Gourgeist-Small",
        "Gourgeist Large Variety": "Gourgeist-Large",
        "Gourgeist Jumbo Variety": "Gourgeist-Super",
        "Maushold Family of Three": "Maushold",
        "Maushold Family of Four": "Maushold-Four",
        "Squawkabilly Green Plumage": "Squawkabilly",
        "Squawkabilly Blue Plumage": "Squawkabilly-Blue",
        "Squawkabilly Yellow Plumage": "Squawkabilly-Yellow",
        "Squawkabilly White Plumage": "Squawkabilly-White",
        "Polteageist Phony Form": "Polteageist",
        "Polteageist Antique Form": "Polteageist-Antique",
        "Sinistcha Unremarkable Form": "Sinistcha",
        "Sinistcha Masterpiece Form": "Sinistcha-Masterpiece",
        "Floette": "Floette-Eternal",  # Champions' only Floette is the Eternal Flower one
        "Floette Eternal Flower": "Floette-Eternal",
        "Alcremie Vanilla Cream": "Alcremie",
        "Alcremie Ruby Cream": "Alcremie-Ruby-Cream",
        "Alcremie Matcha Cream": "Alcremie-Matcha-Cream",
        "Alcremie Mint Cream": "Alcremie-Mint-Cream",
        "Alcremie Lemon Cream": "Alcremie-Lemon-Cream",
        "Alcremie Salted Cream": "Alcremie-Salted-Cream",
        "Alcremie Ruby Swirl": "Alcremie-Ruby-Swirl",
        "Alcremie Caramel Swirl": "Alcremie-Caramel-Swirl",
        "Alcremie Rainbow Swirl": "Alcremie-Rainbow-Swirl",
        "Florges Red Flower": "Florges",
        "Florges Yellow Flower": "Florges-Yellow",
        "Florges Orange Flower": "Florges-Orange",
        "Florges Blue Flower": "Florges-Blue",
        "Florges White Flower": "Florges-White",
        "Furfrou Natural Form": "Furfrou",
        "Furfrou Heart Trim": "Furfrou-Heart",
        "Furfrou Dandy Trim": "Furfrou-Dandy",
        "Vivillon Meadow Pattern": "Vivillon",
        "Vivillon High Plains Pattern": "Vivillon-High Plains",
        "Nidoran♀": "Nidoran-F",
        "Nidoran♂": "Nidoran-M",
    }
)
SPACED_SPECIES = frozenset(
    "Mr. Mime|Mr. Rime|Mime Jr.|Type: Null|Tapu Koko|Tapu Lele|Tapu Bulu|Tapu Fini|Great Tusk|Scream Tail|"
    "Brute Bonnet|Flutter Mane|Slither Wing|Sandy Shocks|Roaring Moon|Walking Wake|Gouging Fire|Raging Bolt|"
    "Iron Treads|Iron Bundle|Iron Hands|Iron Jugulis|Iron Moth|Iron Thorns|Iron Valiant|Iron Leaves|Iron Boulder|"
    "Iron Crown".split("|")
)
COSMETIC_BASES = {"Alcremie", "Florges", "Furfrou", "Vivillon", "Polteageist", "Sinistcha", "Maushold"}
FORMES_OF: dict[str, list[str]] = {}
for _display, _target in FORMS.items():
    _base = _target.split("-")[0]
    if _base not in COSMETIC_BASES and _target != _base:
        FORMES_OF.setdefault(_base, []).append(_target)
FORM_HINT = re.compile(
    r"^(Mega|Alolan|Galarian|Hisuian|Paldean|Primal|Origin|Heat|Wash|Frost|Fan|Mow)\b"
    r"|\b(Form|Forme|Breed|Male|Female|Style|Mode|Cloak|Size|Variety|Trim|Pattern|Plumage|"
    r"Flower|Cream|Swirl|Family|Mask|Rider|Crowned|Unbound|Therian|Incarnate|[XYZ])$"
)


@dataclass
class NameResult:
    display: str
    species: str  # Showdown species, or the display name kept as written
    status: str  # base | mapped | ambiguous | unmapped_form
    candidates: tuple = ()
    warnings: list = field(default_factory=list)


def showdown_species(display: str) -> NameResult:
    """Map one game display name. Never substitutes a different forme for an unmapped one."""
    name = re.sub(r"\s+", " ", display.strip())
    if name in MEGA:
        return NameResult(display, MEGA[name], "mapped")
    if name in FORMS:
        return NameResult(display, FORMS[name], "mapped")
    if name in AMBIGUOUS:
        return NameResult(
            display, name, "ambiguous", AMBIGUOUS[name], [f"'{name}' could be any of {', '.join(AMBIGUOUS[name])}"]
        )
    if name in SPACED_SPECIES:
        return NameResult(display, name, "base")
    if " " in name or FORM_HINT.search(name):
        return NameResult(
            display, name, "unmapped_form", (), [f"'{name}' looks like a forme with no mapping; kept as written"]
        )
    if name in FORMES_OF:
        return NameResult(
            display, name, "base", tuple(FORMES_OF[name]), [f"'{name}' also has formes {', '.join(FORMES_OF[name])}"]
        )
    return NameResult(display, name, "base")

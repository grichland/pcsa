"""Tables that turn battle-log sentences into events: weather, terrain, hazards, subject patterns, templates."""

from __future__ import annotations

APOS = "[’']"
STAT_IDS = {
    "Attack": "atk",
    "Defense": "def",
    "Sp. Atk": "spa",
    "Sp. Def": "spd",
    "Speed": "spe",
    "accuracy": "accuracy",
    "evasiveness": "evasion",
}
WEATHER = {
    "A sandstorm kicked up!": ("Sandstorm", False),
    "The sandstorm subsided.": ("none", False),
    "The sandstorm is raging.": ("Sandstorm", True),
    "It started to rain!": ("RainDance", False),
    "The rain stopped.": ("none", False),
    "Rain continues to fall.": ("RainDance", True),
    "The sunlight turned harsh!": ("SunnyDay", False),
    "The harsh sunlight faded.": ("none", False),
    "The sunlight is strong.": ("SunnyDay", True),
    "It started to snow!": ("Snowscape", False),
    "The snow stopped.": ("none", False),
    "Snow continues to fall.": ("Snowscape", True),
}
TERRAIN = {
    "An electric current ran across the battlefield": ("move: Electric Terrain", "start"),
    "Grass grew to cover the battlefield": ("move: Grassy Terrain", "start"),
    "Mist swirled around the battlefield": ("move: Misty Terrain", "start"),
    "The battlefield got weird": ("move: Psychic Terrain", "start"),
    "The electricity disappeared from the battlefield": ("move: Electric Terrain", "end"),
    "The grass disappeared from the battlefield": ("move: Grassy Terrain", "end"),
    "The mist disappeared from the battlefield": ("move: Misty Terrain", "end"),
    "The weirdness disappeared from the battlefield": ("move: Psychic Terrain", "end"),
}
SCREENS = r"Light Screen|Reflect|Aurora Veil|Tailwind"
HAZARD_START = [
    (r"^Pointed stones float in the air on (the opposing side|your side)!$", "move: Stealth Rock"),
    (r"^Spikes were scattered on the ground all around (the opposing team|your team)!$", "Spikes"),
    (r"^Poison spikes were scattered on the ground all around (the opposing team|your team)!$", "move: Toxic Spikes"),
    (r"^A sticky web has been laid out on the ground around (the opposing team|your team)!$", "move: Sticky Web"),
    (rf"^(?:{SCREENS}) made (the opposing side|the opposing team|your side|your team) stronger.*$", None),
]
HAZARD_END = [
    (r"^The pointed stones disappeared from (the opposing side|your side)!$", "move: Stealth Rock"),
    (r"^The pointed stones disappeared from around (the opposing team|your team)!$", "move: Stealth Rock"),
    (r"^The spikes disappeared from the ground around (the opposing team|your team)!$", "Spikes"),
    (r"^The poison spikes disappeared from the ground around (the opposing team|your team)!$", "move: Toxic Spikes"),
    (r"^The sticky web has disappeared from the ground around (the opposing team|your team)!$", "move: Sticky Web"),
    (rf"^(The opposing side|The opposing team|Your side|Your team){APOS}s (?:{SCREENS}) wore off!$", None),
]
# "<who> ..." sentences: (regex on the remainder after the name, kind, data)
SUBJECT_PATTERNS = [
    (r"fell asleep!", "status", {"status": "slp"}),
    (r"was poisoned!", "status", {"status": "psn"}),
    (r"was badly poisoned!", "status", {"status": "tox"}),
    (r"was burned!", "status", {"status": "brn"}),
    (r"was frozen solid!", "status", {"status": "frz"}),
    (r"is paralyzed(?:,| !|!) so it may be unable to move!", "status", {"status": "par"}),
    (r"is paralyzed! It may be unable to move!", "status", {"status": "par"}),
    (r"is fast asleep\.?", "cant", {"reason": "slp"}),
    (r"is paralyzed! It can[’']t move!", "cant", {"reason": "par"}),
    (r"is frozen solid!", "cant", {"reason": "frz"}),
    (r"flinched and couldn[’']t move!", "cant", {"reason": "flinch"}),
    (r"must recharge!", "cant", {"reason": "recharge"}),
    (r"woke up!", "cure", {"status": "slp"}),
    (r"thawed out!", "cure", {"status": "frz"}),
    (r"grew drowsy!", "volatile", {"effect": "move: Yawn"}),
    (r"fell for the taunt!", "volatile", {"effect": "move: Taunt"}),
    (r"was prevented from healing!", "volatile", {"effect": "move: Heal Block"}),
    (r"became confused!", "volatile", {"effect": "confusion"}),
    (r"became confused due to fatigue!", "volatile", {"effect": "confusion"}),
    (r"is confused!", "volatile_active", {"effect": "confusion"}),
    (r"snapped out of its confusion!", "volatile_end", {"effect": "confusion"}),
    (r"was seeded!", "volatile", {"effect": "Leech Seed"}),
    (r"fell in love!", "volatile", {"effect": "Attract"}),
    (r"protected itself!", "trailing", {"what": "protect"}),
    (r"had its HP restored\.", "trailing", {"what": "hp_restored"}),
    (r"vanished instantly!", "prepare", {}),
    (r"absorbed light!", "prepare", {}),
    (r"flew up high!", "prepare", {}),
    (r"burrowed its way under the ground!", "prepare", {}),
    (r"avoided the attack!", "miss", {}),
    (r"is buffeted by the sandstorm!", "cause", {"cause": "Sandstorm", "applies_to": "self", "hp": "lost"}),
    (r"was damaged by the recoil!", "cause", {"cause": "Recoil", "applies_to": "self", "hp": "lost"}),
    (r"was hurt by its poisoning!", "cause", {"cause": "psn", "applies_to": "self", "hp": "lost"}),
    (r"was hurt by its burn!", "cause", {"cause": "brn", "applies_to": "self", "hp": "lost"}),
    (r"kept going and crashed!", "cause", {"cause": "crash", "applies_to": "self", "hp": "lost"}),
    (r"had its energy drained!", "cause", {"cause": "drain", "applies_to": "foe", "hp": "restored"}),
    (r"lost some of its HP!", "trailing", {"what": "lifeorb"}),
    (r"must do an encore!", "volatile", {"effect": "Encore"}),
    (r"is exerting its pressure!", "ability", {"ability": "Pressure"}),
    (r"breaks the mold!", "ability", {"ability": "Mold Breaker"}),
    (r"is hurt by .+!", "cause", {"cause": "partiallytrapped", "applies_to": "self", "hp": "lost"}),
    (r"went back to(?: .*)?!?", "trailing", {"what": "recall"}),
    (r"hung on using its .+!", "trailing", {"what": "focus_sash"}),
    (r"endured the hit!", "trailing", {"what": "endure"}),
    (r"ended its encore!", "volatile_end", {"effect": "Encore"}),
]
# What a recognized Showdown template does, keyed by (effect, message key).
TEMPLATE_EVENTS = {
    ("taunt", "end"): ("volatile_end", {"effect": "move: Taunt"}),
    ("heal", "fail"): ("trailing", {"what": "heal_failed"}),
    ("psychicterrain", "block"): ("trailing", {"what": "terrain_block"}),
    ("default", "ohko"): ("trailing", {"what": "ohko"}),
    ("default", "addItem"): ("item_reveal", {}),
    ("trick", "activate"): ("trailing", {"what": "item_swap"}),
    ("switcheroo", "activate"): ("trailing", {"what": "item_swap"}),
    ("spikes", "damage"): ("cause", {"cause": "Spikes", "applies_to": "self", "hp": "lost"}),
    ("stealthrock", "damage"): ("cause", {"cause": "Stealth Rock", "applies_to": "self", "hp": "lost"}),
    ("wish", "heal"): ("cause", {"cause": "Wish", "applies_to": "self", "hp": "restored"}),
    ("spikes", "start"): ("hazard", {"condition": "Spikes", "action": "start"}),
    ("toxicspikes", "start"): ("hazard", {"condition": "move: Toxic Spikes", "action": "start"}),
    ("attract", "cant"): ("cant", {"reason": "attract"}),
    ("attract", "activate"): ("volatile_active", {"effect": "Attract"}),
    ("par", "cant"): ("cant", {"reason": "par"}),
    ("taunt", "cant"): ("cant", {"reason": "move: Taunt"}),
    ("default", "activateWeaken"): ("item", {}),
    ("default", "eatItemWeaken"): ("item", {}),
    ("slp", "endFromItem"): ("cure", {"status": "slp"}),
    ("meteorbeam", "prepare"): ("prepare", {}),
    ("bounce", "prepare"): ("prepare", {}),
    ("substitute", "start"): ("volatile", {"effect": "Substitute"}),
    ("substitute", "activate"): ("volatile_active", {"effect": "Substitute"}),
    ("substitute", "end"): ("volatile_end", {"effect": "Substitute"}),
    ("shedtail", "start"): ("volatile", {"effect": "Substitute", "passes": True}),
    ("dive", "prepare"): ("prepare", {}),
    ("charge", "start"): ("volatile", {"effect": "Charge"}),
    ("illusion", "end"): ("illusion_end", {}),
    ("endure", "start"): ("trailing", {"what": "endure"}),
    ("destinybond", "activate"): ("trailing", {"what": "destiny_bond"}),
    ("frisk", "activate"): ("ability", {"ability": "Frisk"}),
    ("supremeoverlord", "activate"): ("ability", {"ability": "Supreme Overlord"}),
}
EVENT_KINDS = frozenset(
    "send_out withdraw drag move hp faint ability ability_change item mega_reacting mega boost clearnegativeboost "
    "boost_blocked effectiveness fail miss status cant cure volatile prepare hitcount knock_off weather hazard cause "
    "trailing copyboost item_reveal item_end typechange transform form_change field steal_eat volatile_end "
    "volatile_active confusion_damage illusion_end unknown".split()
)

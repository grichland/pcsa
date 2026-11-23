# pyright: reportArgumentType=false
"""Rule masks used while training: what a seat may not choose because of what it has just done or because it does
nothing.

  - the same move of the same Pokémon at most `max_same_move` turns in a row: then that move is closed while another
    move is open (a Pokémon with one usable move, as when locked into it, is left alone)
  - at most `max_switches` switches by choice in a row: then switching is closed while a move is open
  - optional rules, all off by default (see RuleConfig): runs of setup or recovery, moves that do nothing against
    their active, setup into a KO, status that cannot land, hazards / weather / terrain already up, Protect twice,
    and self moves that are wasted

A replacement after a faint is not a choice: it neither counts nor ends a run. Only turn decisions are masked. A rule
closes something only while another action stays open, so the mask is never all closed. The rules are applied to
the mask before a net chooses, and the closed mask is the one stored, so the update sees what the net saw. The live
player needs the same rules to play as trained.

Copied from pcsa-former (pcsa_former/rules.py); behaviour is kept identical to it
(tests/network/test_rules_parity.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import layout as L


@dataclass(frozen=True)
class RuleConfig:
    max_same_move: int = 4  # 0 = no such rule
    max_switches: int = 3  # 0 = no such rule
    max_boosts: int = 0  # setup moves by the same Pokémon in a row
    max_heals: int = 0  # pure recovery (status move with heal) by the same Pokémon in a row
    no_null_moves: bool = False  # damaging moves that do nothing to their active
    no_boost_when_ko: bool = False  # setup while their best hit is a KO on our current HP
    no_failing_status: bool = False  # status moves that cannot land
    no_redundant_field: bool = False  # hazards at their maximum, weather / terrain already up
    no_double_protect: bool = False  # Protect family by the same Pokémon two turns in a row
    no_wasted_self: bool = False  # recovery at full HP, setup at +6, Substitute when it fails


# CG's ids (contracts/MODEL.md section 2): moves = index in the move table (champions-gym/data/moves.json order,
# 0 = none), abilities = index in the ability table, items = index + 1. tests/test_rules_extra.py checks them.
MOVE = {
    "thunderwave": 469, "glare": 186, "stunspore": 438, "willowisp": 505, "toxic": 475, "poisonpowder": 316,
    "sleeppowder": 397, "spore": 421, "hypnosis": 221, "sing": 391, "yawn": 512, "leechseed": 252,
    "stealthrock": 422, "spikes": 415, "toxicspikes": 476, "stickyweb": 426,
    "sunnyday": 441, "raindance": 347, "sandstorm": 370, "snowscape": 407,
    "electricterrain": 128, "grassyterrain": 189, "mistyterrain": 282, "psychicterrain": 333,
    "protect": 329, "detect": 99, "kingsshield": 242, "spikyshield": 416, "banefulbunker": 31, "substitute": 439,
    "recover": 350, "roost": 365, "slackoff": 395, "softboiled": 409, "milkdrink": 278, "moonlight": 284,
    "morningsun": 285, "synthesis": 452, "lifedew": 253, "swallow": 447, "rest": 354,
    "counter": 84, "mirrorcoat": 280, "metalburst": 273, "comeuppance": 77,
    "bellydrum": 37, "acidarmor": 2, "agility": 8, "amnesia": 13, "bulkup": 57, "calmmind": 63, "clangoroussoul": 72,
    "coil": 76, "cosmicpower": 81, "cottonguard": 82, "doubleteam": 108, "dragondance": 112, "growth": 192,
    "howl": 215, "irondefense": 237, "minimize": 279, "nastyplot": 292, "noretreat": 297, "quiverdance": 342,
    "rockpolish": 360, "shellsmash": 387, "shelter": 388, "shiftgear": 389, "swordsdance": 451,
}  # fmt: skip
ABILITY = {
    "levitate": 130, "goodasgold": 90, "moldbreaker": 153, "teravolt": 282, "turboblaze": 295, "corrosion": 38,
    "limber": 134, "insomnia": 121, "vitalspirit": 302, "waterveil": 308, "immunity": 116, "purifyingsalt": 198,
    "comatose": 33, "overcoat": 170, "prankster": 188, "infiltrator": 118,
}  # fmt: skip
ITEM = {"airballoon": 7}
# status moves that target the foe (Good as Gold, Prankster into Dark); Curse left out (a non-Ghost's Curse is setup)
FOE_STATUS = (7, 24, 30, 45, 67, 78, 80, 83, 96, 97, 102, 126, 129, 133, 137, 146, 147, 165, 175, 181, 186, 193, 194,
              205, 221, 236, 252, 257, 263, 269, 272, 274, 296, 299, 303, 305, 316, 324, 325, 335, 339, 353, 358, 364,
              374, 376, 390, 391, 392, 397, 408, 413, 414, 419, 421, 433, 434, 438, 446, 448, 449, 450, 456, 457, 459,
              469, 470, 472, 474, 475, 477, 479, 481, 482, 502, 505, 509, 512)  # fmt: skip
# the stat stages a setup move raises: atk def spa spd spe accuracy evasion (Charge and Belly Drum apart)
SETUP_BOOSTS = {
    "acidarmor": "def", "agility": "spe", "amnesia": "spd", "bulkup": "atk def", "calmmind": "spa spd",
    "clangoroussoul": "atk def spa spd spe", "coil": "atk def accuracy", "cosmicpower": "def spd",
    "cottonguard": "def", "doubleteam": "evasion", "dragondance": "atk spe", "growth": "atk spa", "howl": "atk",
    "irondefense": "def", "minimize": "evasion", "nastyplot": "spa", "noretreat": "atk def spa spd spe",
    "quiverdance": "spa spd spe", "rockpolish": "spe", "shellsmash": "atk spa spe", "shelter": "def",
    "shiftgear": "atk spe", "swordsdance": "atk", "bellydrum": "atk",
}  # fmt: skip
STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
TYPES = ("bug dark dragon electric fairy fighting fire flying ghost grass ground ice normal poison psychic rock steel "
         "water").split()  # fmt: skip
VOLATILE = {"substitute": 5, "leechseed": 6, "yawn": 7, "magnetrise": 22, "smackdown": 24}  # bits of F3-5
PAR, SLP, PSN, TOX, BRN, DROWSY = 1, 2, 3, 4, 5, 6  # what a status move inflicts
NEW_RULES = ("boosts", "heals", "null", "boost_ko", "status", "field", "protect", "self")


def _table(names, value=True, dtype=bool) -> np.ndarray:
    t = np.zeros(L.VOCAB["move"], dtype)
    t[[MOVE[n] for n in names]] = value
    return t


class Rules:
    """What each seat of each game has just done, and the masks that follow from it."""

    def __init__(self, cfg: RuleConfig, layout: L.Layout, games: int) -> None:
        self.cfg = cfg
        self.active_col, self.species_col = layout.col(L.F_IS_ACTIVE), layout.col(L.F_SPECIES)
        self.move_cols = layout[L.F_MOVE_ID].columns()
        self.key = np.zeros((games, 2, 2), np.int64)  # (species, move) of the run of one move
        self.same = np.zeros((games, 2), np.int64)  # its length
        self.switches = np.zeros((games, 2), np.int64)  # switches by choice in a row
        self.closed_moves = self.closed_switches = 0  # decisions a rule closed something for
        # runs of a move class by one Pokémon: [setup, heal, protect] x (species, length)
        self.run_species = np.zeros((games, 2, 3), np.int64)
        self.run_len = np.zeros((games, 2, 3), np.int64)
        self.closed = dict.fromkeys(NEW_RULES, 0)
        self._extra = any(getattr(cfg, f) for f in ("max_boosts", "max_heals", "no_null_moves", "no_boost_when_ko",
                                                     "no_failing_status", "no_redundant_field", "no_double_protect",
                                                     "no_wasted_self"))  # fmt: skip
        if self._extra:
            self._columns(layout)
            self._tables()

    def _columns(self, layout: L.Layout) -> None:
        by = {f.name: f for f in layout.features.values()}

        def c(name):
            return np.asarray(by[name].columns())

        self.c_type, self.c_status, self.c_boost = c("pokemon.identity.type"), c("pokemon.condition.status"), np.concatenate(
            [c(f"pokemon.condition.boost.{s}") for s in STATS])  # fmt: skip
        self.c_vol, self.c_hp, self.c_incoming = c("pokemon.condition.volatile"), c("pokemon.condition.hp_fraction")[0], c("pokemon.matchup.incoming_hit.mean")[0]
        self.c_mtype, self.c_mcat, self.c_prio = c("pokemon.move.fact.type"), c("pokemon.move.fact.category"), c("pokemon.move.fact.priority")
        self.c_setup, self.c_heal, self.c_max = c("pokemon.move.fact.is_setup"), c("pokemon.move.fact.is_heal_or_drain"), c("pokemon.move.damage.max")
        self.c_item, self.c_ability = layout.col(L.F_ITEM), layout.col(L.F_ABILITY)
        self.c_possible = np.asarray(layout[L.F_POSSIBLE_ABILITY].columns())
        self.c_weather, self.c_strong, self.c_terrain = c("field.weather.kind"), c("field.weather.is_strong")[0], c("field.terrain.kind")
        self.c_gravity, self.c_safeguard = c("field.room.gravity")[0], c("field.side.opp.safeguard")[0]
        self.c_hazard = np.concatenate([c(f"field.side.opp.{h}") for h in ("stealth_rock", "spikes_layers", "toxic_spikes_layers", "sticky_web")])

    def _tables(self) -> None:
        self.t_status = np.zeros(L.VOCAB["move"], np.int8)
        for names, kind in ((("thunderwave", "glare", "stunspore"), PAR), (("sleeppowder", "spore", "hypnosis", "sing"), SLP),
                            (("poisonpowder",), PSN), (("toxic",), TOX), (("willowisp",), BRN), (("yawn",), DROWSY)):  # fmt: skip
            self.t_status[[MOVE[n] for n in names]] = kind
        self.t_powder = _table(("sleeppowder", "spore", "stunspore", "poisonpowder"))
        self.t_twave, self.t_seed, self.t_sound = _table(("thunderwave",)), _table(("leechseed",)), _table(("sing",))
        self.t_foe = np.zeros(L.VOCAB["move"], bool)
        self.t_foe[list(FOE_STATUS)] = True
        self.t_reactive = _table(("counter", "mirrorcoat", "metalburst", "comeuppance"))  # their damage is not computed
        self.t_hazard = np.zeros(L.VOCAB["move"], np.int8)  # 1 + index into c_hazard
        self.t_hazard[[MOVE[n] for n in ("stealthrock", "spikes", "toxicspikes", "stickyweb")]] = [1, 2, 3, 4]
        self.t_weather = np.zeros(L.VOCAB["move"], np.int8)  # 1 + index into field.weather.kind (sun rain sand snow)
        self.t_weather[[MOVE[n] for n in ("sunnyday", "raindance", "sandstorm", "snowscape")]] = [1, 2, 3, 4]
        self.t_terrain = np.zeros(L.VOCAB["move"], np.int8)  # 1 + index into field.terrain.kind
        self.t_terrain[[MOVE[n] for n in ("electricterrain", "grassyterrain", "mistyterrain", "psychicterrain")]] = [1, 2, 3, 4]
        self.t_protect = _table(("protect", "detect", "kingsshield", "spikyshield", "banefulbunker"))
        self.t_recover = _table(("recover", "roost", "slackoff", "softboiled", "milkdrink", "moonlight", "morningsun",
                                 "synthesis", "lifedew", "swallow", "rest"))  # fmt: skip
        self.t_sub, self.t_drum = _table(("substitute",)), _table(("bellydrum",))
        self.t_boost = np.zeros((L.VOCAB["move"], len(STATS)), bool)
        for name, stats in SETUP_BOOSTS.items():
            self.t_boost[MOVE[name], [STATS.index(s) for s in stats.split()]] = True
        self.t_def_stats = np.array([s in ("def", "spd", "evasion") for s in STATS])
        self.boost_top = np.array([1, 1, 1, 1, 1, np.log2(3) / 2, np.log2(3) / 2]) - 1e-3  # +6 as F3-4 encodes it
        self.ty = {t: i for i, t in enumerate(TYPES)}

    def _active(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """(species [N], move ids [N, 4]) of the asked seat's Pokémon on the field."""
        e = np.arange(len(obs["kind"]))
        row = np.asarray(obs["rows"])[:, : L.OWN_PICKS, self.active_col].argmax(-1)
        ids = np.asarray(obs["row_ids"])[e, row]
        return ids[:, self.species_col].astype(np.int64), ids[:, self.move_cols].astype(np.int64)

    def _close(self, m: np.ndarray, bad: np.ndarray, turn: np.ndarray, name: str) -> None:
        """Close move slots `bad` [N, 4] (both copies) of turn rows, where another action stays open."""
        close = np.concatenate([bad, bad], 1) & turn[:, None] & (m[:, :8] > 0)
        rest = (m > 0).copy()
        rest[:, :8] &= ~close
        close &= rest.any(1)[:, None]
        hit = close.any(1)
        self.closed[name] += int(hit.sum())
        m[:, :8][close] = 0

    def _extra_masks(self, obs: dict, m: np.ndarray, turn: np.ndarray) -> None:
        cfg, n = self.cfg, len(m)
        e, s = np.arange(n), np.asarray(obs["seat"]).astype(np.int64)
        rows, ids, fld = np.asarray(obs["rows"]), np.asarray(obs["row_ids"]), np.asarray(obs["field"])
        row = rows[:, : L.OWN_PICKS, self.active_col].argmax(-1)
        me, me_ids, foe, foe_ids = rows[e, row], ids[e, row], rows[:, L.OPP_ACTIVE], ids[:, L.OPP_ACTIVE]
        species, mid = me_ids[:, self.species_col].astype(np.int64), me_ids[:, self.move_cols].astype(np.int64)
        mid = np.clip(mid, 0, L.VOCAB["move"] - 1)
        cat = me[:, self.c_mcat].reshape(n, 4, 3)
        damaging, status_move = (cat[..., 0] + cat[..., 1]) > 0, cat[..., 2] > 0
        setup = me[:, self.c_setup] > 0
        hp = me[:, self.c_hp]
        ty = self.ty

        def their(ability):  # known, or the only ability their form can have
            known = foe_ids[:, self.c_ability]
            only = (foe_ids[:, self.c_possible[0]] == ability) & (foe_ids[:, self.c_possible[1:]] == 0).all(1)
            return (known == ability) | ((known == 0) & only)

        our_ability = me_ids[:, self.c_ability]
        breaker = np.isin(our_ability, [ABILITY[a] for a in ("moldbreaker", "teravolt", "turboblaze")])
        foe_type = foe[:, self.c_type] > 0
        vol = foe[:, self.c_vol] > 0
        airborne = (their(ABILITY["levitate"]) & ~breaker) | (foe_ids[:, self.c_item] == ITEM["airballoon"]) | vol[:, VOLATILE["magnetrise"]]
        airborne &= ~(vol[:, VOLATILE["smackdown"]] | (fld[:, self.c_gravity] > 0))
        grounded = ~airborne & (~foe_type[:, ty["flying"]] | (fld[:, self.c_gravity] > 0) | vol[:, VOLATILE["smackdown"]])
        terrain = fld[:, self.c_terrain] > 0  # electric grassy misty psychic

        if cfg.no_null_moves:
            mtype = me[:, self.c_mtype].reshape(n, 4, 18)
            # the calculator's max already has the type chart, abilities and terrain; the type chart alone (F7-9)
            # misses moves that change type (Pixilate, Weather Ball), so a 0 there is not used on its own
            bad = (me[:, self.c_max] <= 0) & ~self.t_reactive[mid]
            bad |= (mtype[..., ty["ground"]] > 0) & airborne[:, None]
            bad |= (me[:, self.c_prio] > 0) & (terrain[:, 3] & grounded)[:, None]
            self._close(m, bad & damaging & (mid > 0), turn, "null")
        if cfg.no_failing_status:
            kind = self.t_status[mid]
            ctype = lambda t: foe_type[:, ty[t]][:, None]  # noqa: E731
            no_break = ~breaker[:, None]
            bad = (kind == PAR) & (ctype("electric") | (their(ABILITY["limber"])[:, None] & no_break))
            bad |= self.t_twave[mid] & ctype("ground")
            bad |= (kind == BRN) & (ctype("fire") | (their(ABILITY["waterveil"])[:, None] & no_break))
            poison = (kind == PSN) | (kind == TOX)
            bad |= poison & (ctype("steel") | ctype("poison")) & (our_ability != ABILITY["corrosion"])[:, None]
            bad |= poison & their(ABILITY["immunity"])[:, None] & no_break
            sleep = (kind == SLP) | (kind == DROWSY)
            sleep_ab = their(ABILITY["insomnia"]) | their(ABILITY["vitalspirit"])
            bad |= sleep & ((sleep_ab[:, None] & no_break) | (terrain[:, 0] & grounded)[:, None])
            bad |= (self.t_powder[mid] | self.t_seed[mid]) & ctype("grass")
            bad |= self.t_powder[mid] & their(ABILITY["overcoat"])[:, None] & no_break
            bad |= self.t_seed[mid] & vol[:, VOLATILE["leechseed"]][:, None]
            bad |= (kind == DROWSY) & vol[:, VOLATILE["yawn"]][:, None]
            any_status = (foe[:, self.c_status] > 0).any(1)
            no_status = any_status | ((their(ABILITY["purifyingsalt"]) | their(ABILITY["comatose"])) & ~breaker)
            no_status |= terrain[:, 2] & grounded  # Misty Terrain
            infiltrate = our_ability == ABILITY["infiltrator"]
            no_status |= (fld[:, self.c_safeguard] > 0) & ~infiltrate
            bad |= (kind > 0) & no_status[:, None]
            sub = vol[:, VOLATILE["substitute"]] & ~infiltrate
            bad |= ((kind > 0) | self.t_seed[mid]) & ~self.t_sound[mid] & sub[:, None]
            foe_move = self.t_foe[mid]
            bad |= foe_move & (their(ABILITY["goodasgold"]) & ~breaker)[:, None]
            bad |= foe_move & ((our_ability == ABILITY["prankster"]) & foe_type[:, ty["dark"]])[:, None]
            self._close(m, bad & status_move & (mid > 0), turn, "status")
        if cfg.no_redundant_field:
            full = np.column_stack([fld[:, self.c_hazard] >= 1 - 1e-3, np.zeros(n, bool)])  # last column: no hazard
            h = self.t_hazard[mid].astype(np.int64) - 1
            bad = full[e[:, None], np.where(h < 0, 4, h)]
            weather = np.column_stack([fld[:, self.c_weather] > 0, np.zeros(n, bool)])
            w = self.t_weather[mid].astype(np.int64) - 1
            bad |= np.where(w < 0, False, weather[e[:, None], np.where(w < 0, 4, w)] | (fld[:, self.c_strong] > 0)[:, None])
            t = self.t_terrain[mid].astype(np.int64) - 1
            bad |= np.column_stack([terrain, np.zeros(n, bool)])[e[:, None], np.where(t < 0, 4, t)]
            self._close(m, bad & (mid > 0), turn, "field")
        if cfg.no_wasted_self:
            boost = me[:, self.c_boost]
            raises = self.t_boost[mid]  # [N, 4, 7]
            topped = (boost >= self.boost_top)[:, None, :]
            bad = raises.any(-1) & (~raises | topped).all(-1)
            bad |= self.t_drum[mid] & (hp <= 0.5)[:, None]
            bad |= self.t_recover[mid] & (hp >= 1 - 1e-6)[:, None]
            bad |= self.t_sub[mid] & ((me[:, self.c_vol[VOLATILE["substitute"]]] > 0) | (hp <= 0.25))[:, None]
            self._close(m, bad & (mid > 0), turn, "self")
        if cfg.no_double_protect:
            last = (self.run_len[e, s, 2] > 0) & (self.run_species[e, s, 2] == species)
            self._close(m, self.t_protect[mid] & last[:, None], turn, "protect")
        if cfg.no_boost_when_ko:  # defensive setup (only Def, Sp. Def or evasion) may be what stops the KO: left open
            r = self.t_boost[mid]
            defensive = r.any(-1) & ~(r & ~self.t_def_stats).any(-1)
            self._close(m, setup & ~defensive & (me[:, self.c_incoming] >= 1.0)[:, None], turn, "boost_ko")
        if cfg.max_boosts:
            due = (self.run_len[e, s, 0] >= cfg.max_boosts) & (self.run_species[e, s, 0] == species)
            self._close(m, setup & due[:, None], turn, "boosts")
        if cfg.max_heals:
            heal = status_move & (me[:, self.c_heal] > 0)
            due = (self.run_len[e, s, 1] >= cfg.max_heals) & (self.run_species[e, s, 1] == species)
            self._close(m, heal & due[:, None], turn, "heals")

    def mask(self, obs: dict) -> np.ndarray:
        """obs["mask"] [N, 11] with the rules applied."""
        m = np.array(obs["mask"], copy=True)
        e, s = np.arange(len(m)), np.asarray(obs["seat"]).astype(np.int64)
        turn = np.asarray(obs["kind"]) == L.KIND_TURN
        if self._extra:
            self._extra_masks(obs, m, turn)
        moves_open = m[:, :8] > 0
        if self.cfg.max_same_move:
            species, moves = self._active(obs)
            due = turn & (self.same[e, s] >= self.cfg.max_same_move) & (self.key[e, s, 0] == species)
            slot = (moves == self.key[e, s, 1][:, None]) & (moves != 0) & due[:, None]
            close = np.concatenate([slot, slot], 1)  # with and without Mega
            close &= (moves_open & ~close).any(1)[:, None]  # only while another move is open
            self.closed_moves += int((close & moves_open).any(1).sum())
            m[:, :8][close] = 0
            moves_open = m[:, :8] > 0
        if self.cfg.max_switches:
            due = turn & (self.switches[e, s] >= self.cfg.max_switches) & moves_open.any(1)
            self.closed_switches += int((due & (m[:, 8:] > 0).any(1)).sum())
            m[due, 8:] = 0
        return m

    def record(self, obs: dict, action: np.ndarray) -> None:
        """The actions chosen for `obs` (every game's asked seat)."""
        action = np.asarray(action).astype(np.int64)
        e, s = np.arange(len(action)), np.asarray(obs["seat"]).astype(np.int64)
        turn = np.asarray(obs["kind"]) == L.KIND_TURN
        move, switch = turn & (action < 8), turn & (action >= 8)
        species, moves = self._active(obs)
        slot = np.clip(action, 0, 7) % L.MOVE_SLOTS
        used = moves[e, slot]
        again = move & (self.key[e, s, 0] == species) & (self.key[e, s, 1] == used)
        self.same[e, s] = np.where(move, np.where(again, self.same[e, s] + 1, 1), np.where(switch, 0, self.same[e, s]))
        self.key[e, s, 0] = np.where(move, species, self.key[e, s, 0])
        self.key[e, s, 1] = np.where(move, used, self.key[e, s, 1])
        self.switches[e, s] = np.where(switch, self.switches[e, s] + 1, np.where(move, 0, self.switches[e, s]))
        if self._extra:
            rows = np.asarray(obs["rows"])
            me = rows[e, rows[:, : L.OWN_PICKS, self.active_col].argmax(-1)]
            heal = (me[:, self.c_mcat[2::3]] > 0) & (me[:, self.c_heal] > 0)
            kinds = np.stack([me[:, self.c_setup] > 0, heal, self.t_protect[np.clip(moves, 0, L.VOCAB["move"] - 1)]], -1)
            this = kinds[e, slot]  # [N, 3]
            run_len, run_sp = self.run_len[e, s], self.run_species[e, s]
            cont = this & (run_sp == species[:, None])
            new_len = np.where(cont, run_len + 1, np.where(this, 1, 0))
            self.run_len[e, s] = np.where((move | switch)[:, None], np.where(move[:, None], new_len, 0), run_len)
            self.run_species[e, s] = np.where(move[:, None], species[:, None], run_sp)

    def counts(self) -> dict:
        """Decisions each rule closed something for."""
        return {"moves": self.closed_moves, "switches": self.closed_switches, **self.closed}

    def save(self, g: int) -> tuple:
        """Game g's counts, for a copy of the game."""
        return (self.key[g].copy(), self.same[g].copy(), self.switches[g].copy(), self.run_species[g].copy(),
                self.run_len[g].copy())  # fmt: skip

    def load(self, g: int, saved: tuple) -> None:
        self.key[g], self.same[g], self.switches[g], self.run_species[g], self.run_len[g] = saved

    def reset(self, games: np.ndarray) -> None:
        """Games that ended: nothing done yet in the next ones."""
        self.key[games], self.same[games], self.switches[games] = 0, 0, 0
        self.run_species[games], self.run_len[games] = 0, 0

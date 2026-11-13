"""Team preview against each of their possible leads (13 Nov 2026, the `robust` preview).

The critic scores a pick from the turn-0 view (our trio with our lead out, their six), an average over whatever the
training opponents led with. Here each pick is scored once per lead of theirs instead: the turn-1 Battle right after
both leads are sent out (nothing revealed yet, full HP), built from the preview Battle the same way the live loop
builds a decision's Battle (protocol lines, then our request). That is the position every training game's first
decision saw, so the value head is on familiar ground. A pick's score is the mean of its `worst` lowest values
over their leads: it does not walk into the lead that counters it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle import state as tracking
from pcsa.battle.adapter import ChampionsBattle
from pcsa.battle.moves import max_pp
from pcsa.battle.translate import LEVEL

MEGA_STONE_ENDINGS = ("ite", "itex", "itey", "itez")


def their_species(battle) -> list[str]:
    """Their six at preview, as protocol species names (e.g. Rotom-Wash)."""
    out = []
    for mon in battle.teampreview_opponent_team:
        entry = game.species_entry(to_id_str(mon.species)) or {}
        out.append(str(entry.get("name") or mon.species))
    return out


def _is_mega_stone(item: str) -> bool:
    item = to_id_str(item or "")
    return item != "eviolite" and item.endswith(MEGA_STONE_ENDINGS)


def lead_battle(preview_battle, pick: Sequence[int], their_lead: str, species: Sequence[str], tag: str):
    """The turn-1 Battle of `pick` (our team indices, lead first) against `their_lead` (one of `species`)."""
    request = preview_battle.last_request or {}
    side = request.get("side") or {}
    members = side.get("pokemon") or []
    entries = [{**members[i], "active": n == 0} for n, i in enumerate(pick)]
    lead = entries[0]
    moves = [
        {"move": m, "id": m, "pp": max_pp(m), "maxpp": max_pp(m), "target": "normal", "disabled": False}
        for m in lead.get("moves") or []
    ]
    active = {"moves": moves, **({"canMegaEvo": True} if _is_mega_stone(lead.get("item", "")) else {})}
    lines = ["|player|p1|me|", "|player|p2|opponent|", "|gen|9", "|clearpoke"]
    lines += [f"|poke|p2|{s}, L{LEVEL}|" for s in species]
    lines += [f"|teampreview|{len(pick)}", f"|teamsize|p1|{len(pick)}", f"|teamsize|p2|{len(pick)}", "|start"]
    lines += [
        f"|switch|p2a: {their_lead}|{their_lead}, L{LEVEL}|100/100",
        f"|switch|{lead['ident'].replace('p1: ', 'p1a: ')}|{lead['details']}|{lead['condition']}",
        "|turn|1",
    ]
    battle = ChampionsBattle(tag, tracking.USERNAME, tracking.LOGGER, 9)
    battle.player_role = "p1"
    for line in lines:
        battle.parse_message(line.split("|"))
    battle.parse_request({"active": [active], "side": {**side, "pokemon": entries}, "rqid": 1})
    tracking.normalize_fainted_request(battle)
    return battle


def robust_scores(table: np.ndarray, worst: int) -> np.ndarray:
    """table [picks, their leads] -> per pick the mean of its `worst` lowest values (worst >= number of leads: the
    plain mean)."""
    k = max(1, min(int(worst), table.shape[1]))
    return np.sort(table, axis=1)[:, :k].mean(axis=1)


@torch.no_grad()
def lead_values(policy, battle, picks) -> tuple[np.ndarray, list[str]]:
    """[picks, their six] values of the turn-1 view after our lead and their lead come out (each its own game's
    first decision, belief fresh), and their six."""
    from pcsa.players.mirror import legal_actions
    from pcsa.rl_training.former import stack
    from pcsa.rl_training.former_player import _Game

    species = their_species(battle)
    obs = []
    for pick in picks:
        for s in species:
            b = lead_battle(battle, pick, s, species, f"{battle.battle_tag}-lead")
            obs.append(policy._observe(b, legal_actions(b), _Game()))
    out = policy.net(stack([[o] for o in obs], policy.device))
    return out["value"][:, 0].float().cpu().numpy().reshape(len(picks), len(species)), species


def robust_team(policy, battle, picks) -> list[int]:
    """The kept pick (the preview rules still remove picks first) whose mean over its `policy.preview_robust_worst`
    worst leads of theirs is the highest; sets the policy's notes."""
    table, species = lead_values(policy, battle, picks)
    keep, rules = policy.preview_keep(battle, picks) if policy.preview_keep else ([True] * len(picks), {})
    rows = [i for i, k in enumerate(keep) if k] or list(range(len(picks)))
    scores = robust_scores(table, policy.preview_robust_worst)
    index = rows[int(np.argmax(scores[rows]))]
    mean = table.mean(axis=1)
    policy._notes = {
        "rule": "rl",
        "preview": "robust",
        "v": round(float(scores[index]), 4),
        "worst": policy.preview_robust_worst,
        "their_leads": species,
        "values": [round(float(x), 3) for x in table[index]],
        "best_mean": list(picks[rows[int(np.argmax(mean[rows]))]]),
        **({"preview_rules": rules} if rules else {}),
    }
    return [int(i) for i in picks[index]]

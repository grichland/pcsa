# The neural player

The player is Champions Gym's neural player shell running this repository's policy (`src/pcsa/network/policy.py`)
over a checkpoint from `assets/models/`. It runs on the CPU in its own process, `pcsa.cg.serve`, which `pcsa run`
starts from the `cg` section of `config/agent.yaml`. It needs torch (`pcsa setup --with torch`).

The network code in `src/pcsa/network/` (`model.py`, `decide.py`, `layout.py`, `checkpoint.py`, `cg_adapter.py`,
`rules.py`) is copied from pcsa-former, the training repository, for inference only.
`tests/network/test_parity.py` checks that it gives the same outputs as pcsa-former's code when that repository
is checked out next to this one (it is optional; the test skips without it).

## Checkpoints

`cg.player_args.checkpoint` names the checkpoint played (relative to the repository root, or absolute); the
other files in `assets/models/` are earlier snapshots of the same training, named by run and iteration, kept for
comparison and for the tests. A
checkpoint stores the layout of its inputs and its own settings (the policy cap, belief slots). Before the first
battle Champions Gym reports whether the layout and contract match the installed version and which checkpoint it
loaded; on any mismatch the run does not start.

## The network

About 1.6 M parameters (width 128, 4 layers, 4 heads). It predicts the opponent, then chooses.

- **Inputs.** The field, each of the 12 Pokémon and the previous turn become one token each, 14 tokens per
  decision, in the layout Champions Gym's model contract defines (the same arrays as in training). The
  opponent's rows hold only what the battle has shown.
- **Memory.** A decision also sees the field and previous-turn tokens of up to 32 earlier decisions of the
  battle. After a restart the memory is rebuilt by replaying the earlier requests.
- **Their action.** One head gives a probability to each of their 13 actions: the four move slots of their active
  Pokémon (revealed moves, with one slot standing for a move not yet seen), the same four with Mega Evolution,
  and five switches. A checkpoint trained with belief slots names each empty slot with the likeliest unshown move.
- **The matrix.** Our 11 actions (4 moves, the same 4 with Mega Evolution, 3 switches) are the rows and their
  13 actions the columns; each cell scores our action against theirs. A row's score is its cells weighted by
  their probabilities, and a softmax over the scores gives the policy, capped at the checkpoint's maximum
  probability while another action is possible.
- **Value.** A head estimates the outcome (+1 a win, −1 a loss). Another scores the 60 lead, second and third
  picks at team preview.
- **Belief.** For each opposing Pokémon: how likely it was brought, how likely it leads, and its item, ability and
  moves. They do not change the network's own scores; they name the empty move slots of a belief-slot
  checkpoint, supply the search's guesses at the hidden team, and are shown on the dashboard.

The player takes the policy's most likely action, after the steps below. Every output of every decision is
written to the run's decision records.

## The choice rule (`cg.choice`)

At turn requests the pick can be made from the matrix by another rule than the network's own expectation:

| `rule` | Score of each of our actions over their kept replies |
|---|---|
| `mean` | the expected cell, the kept replies weighted by their renormalised probabilities |
| `min` | the worst cell (maximin) |
| `mix` | `blend × mean + (1 − blend) × min` |
| `regret` | minus the largest shortfall against the best kept action for each reply (minimax regret) |

`top_theirs` keeps only their N likeliest replies, `top_ours` only our N likeliest actions (0: all). With
`rule: mean` and both at 0 the rule is the network's own pick and nothing is passed to the policy. Each turn
record notes the rule's pick next to the network's. The shipped config hedges with `regret` over our 8 likeliest
actions and their 8 likeliest replies.

## Rule masks (`cg.rules`)

The masks are training's (`src/pcsa/network/rules.py`, a copy of pcsa-former's checked by
`tests/network/test_rules_parity.py`), and they must be set as the checkpoint was trained: a checkpoint does not
store them. They close actions at turn requests only, and only while another action stays open:

- `max_same_move`, `max_switches`: the same move used that many turns in a row, or that many voluntary switches in
  a row (a replacement after a faint does not count);
- `max_boosts`, `max_heals`: setup or pure recovery by the same Pokémon that many turns in a row;
- `no_null_moves`, `no_failing_status`: damaging moves that do nothing to their active, status moves that cannot
  land;
- `no_boost_when_ko`: setup, other than Def, Sp. Def or evasion, while their best hit KOs us;
- `no_redundant_field`, `no_double_protect`, `no_wasted_self`: hazards at their maximum or weather and terrain
  already up, Protect twice in a row, recovery at full HP, setup at +6, a Substitute that would fail.

Three rules are pcsa's own, not training's (`src/pcsa/network/live_rules.py`, applied after the others):
- `status_free_mirrors` lists species (CG ids, shipped `[primarina]`); when our active and theirs are the same
  listed species, our status moves and their Mega copies are closed, while another action stays open. An empty
  list turns it off.
- `always_mega` (shipped `true`): at turn requests, a plain move is closed while the same move with Mega Evolution
  is open, so we Mega Evolve the first time we can; switches stay open. It runs after the mirror rule.
- `one_mega_lineup` (shipped `true`): at team preview, every pick of three that brings two or more Mega stone
  holders is closed, since only one can Mega Evolve, while another pick stays open. The stones are the items a
  species' Mega form needs in CG's species table (`mega_stones` in `assets/game/cg/vocab.json`). Preview search
  races only the picks left open; the decision's extras list the closed picks under `closed` (11 + pick).

The masks hold wherever we choose: at the request itself; in the search tree at every one of our turn nodes (each
node's counters are its parent's with our action there recorded, and the mask closes actions both for the network
and for the tree); in preview search's playouts of our seat; and on the action that goes out, a fallback player's
too (`live_rules.guard`, called by pcsa.cg.serve: a closed action becomes the same move with Mega, another move,
another switch, else any open action, and the log says so).

The shipped checkpoint, like every from-scratch checkpoint (`scratch-20261122-*`, `gold-20261123-*`), was
trained with every rule: `max_same_move` 3, `max_switches` 2, `max_boosts` 3, `max_heals` 3 and all six `no_*`
rules on. The earlier league checkpoints (`league-*`) were trained with the first two only, at 4 and 3.

## Value calibration (`cg.value_calibration`)

The from-scratch checkpoints were trained on a shaped reward (faints and long games count as well as the result),
so their raw value is not a win probability. With `cg.value_calibration` on, the value reported is
`2 * sigmoid(a * v + b) - 1`, so `(v + 1) / 2` reads as a win chance; the records keep the raw value as
`value_raw`. The map changes no decision. `a` and `b` belong to one checkpoint: `tools/value_calibration.py` fits
them on recorded ladder battles, each decision's value against the battle's result (forfeits and unknown results
count as wins), and writes its report and every sample to `output/value-calibration/`. For the shipped checkpoint,
on 195 battles, the raw value averaged a 64% win chance against a 45% win rate; fitted, the held-out calibration
error is 0.03 (raw 0.21) and the Brier score 0.218 (raw 0.264). The search reads the network's raw values and
has not been re-tuned for these checkpoints, so it stays off.

## Team-preview search (`cg.preview_search`)

At team preview the pick comes from playouts, not from the preview head
(`src/pcsa/network/preview_search.py`, ported from pcsa-former's `tools/preview_search_test.py`):

1. Three guesses of their team: all six species, each set a whole known build (`worlds_mode: builds`) or drawn
   part by part from the belief heads (`legacy`); Champions Gym checks each guess and a refused one is drawn again.
2. A BatchEnv of 3,600 games on the GPU, one per cell (our pick x their pick, numbered as
   `layout.preview_combos`). Both preview picks are forced; after that our seat plays as live play does without
   search (the live rule masks, then the `choice` rule over the matrix) and theirs plays the network's greedy
   policy. Our six sets come from `battle_start`.
3. Each cell scores `(wins + 2 * A) / (games + 2)`, A being the win chance at our first turn: (calibrated value +
   1) / 2.
4. A race over our picks: at 1/4, 1/2 and 3/4 of the budget the best 20, 6 and 2 rows by mean stay in (the head's
   pick always stays); finished games move to the next cell of the rows still in.
5. The best row is played when its mean beats the head's pick by `keep_head_margin` (0.03); otherwise, on any
   error, or with no finished game, the head's pick is played and the record says why.

The search stops at `budget_s` or at the preview deadline less 1.5 s, whichever comes first; with it on, the
controller waits `budget_s` + 6 s for the preview answer, and CG's shell limits are raised to match. The GPU must be
visible to the serve process (`cg.env` without `CUDA_VISIBLE_DEVICES: '-1'`); the live player itself stays on the
CPU. On this machine (RTX 4060 Laptop) 30 s gives about 21,000-26,000 games. Each preview's record keeps the guesses,
the per-cell sums, every cut, the head's pick and the chosen one (`cg.net.preview_search` in
`battle-N/team_preview.json`).

## Decision-time search (`cg.search`, off by default)

With `cg.search.enabled: true` every turn request is also searched before it is answered. The player samples
`worlds` completions of the opponent's hidden team from the belief heads, builds an engine game for each from the
live battle, and grows one tree per completion: both sides choose at once at every node by regret matching started
from the network's policy, its prediction and its matrix, and new positions are valued by the network. The action
played is the one the averaged root strategy favours most, inside the legal and rule masks. At every position the
tree keeps only each side's `top_*_moves` likeliest moves (a move and its Mega copy count as one) and
`top_*_switches` likeliest switches (0: all); the shipped config keeps their 3 moves and 2 switches and all of
ours.

With `worlds_mode: builds` (also `cg.preview_search.worlds_mode`) each hidden Pokémon's set is one whole build from
assets/sets/seen_builds.json and top_builds.json, plus an "other" candidate drawn part by part from the belief
head. Candidates that contradict what the battle showed (moves, an item even after it was used or knocked off, the
ability, a Mega Evolution) or have an ability the species cannot have are left out. The rest are weighted by
sightings x the belief head's probabilities of the build's item, ability and hidden moves x this battle's turn
orders (pcsa.network.evidence: a build whose speed contradicts an order keeps 5%), and a team is drawn build by
build under Item Clause. pcsa.cg.serve feeds the controller's events to the evidence and writes, at each battle's
end, the rank of the build that fits what was shown to `cg/builds-<battle>.json`; each searched turn's note has
`builds` (candidates per Pokémon, the top three with weights, the build drawn per world). Damage is not used yet:
Champions Gym has no damage calculation in its Python API. `legacy` draws item, ability and moves separately with
the species' commonest spread.

Both tables are rebuilt from the recorded runs after a session: `tools/seen_builds.py` learns the sets new battles
revealed into seen_builds.json, and `tools/preview_rates.py` counts the bring and lead rates per species into
assets/sets/opponent_preview_rates.json (read when `preview_rates` names it). Both leave out every battle
pcsa-benchmark uses.

With `calibrate` the network's value, matrix cells and row scores go through `value_calibration` before the search
uses them, so they sit on the scale of a finished game (+1 a win, -1 a loss) like the tree's terminal positions.
The search stops at `ms` or the request's deadline, whichever is sooner, and earlier once the answer has settled
(`settle_sims`, `settle_share`, `settle_gap`). Turns before `from_turn`, requests with a single open action, team
preview and replacements are not searched. With `follow_pivot`, when the pick is a move the trees saw switch us out
in the same turn (U-turn, Volt Switch and the like), two switch-ins are kept: the one the search chose after the
pivot and the one it chose if our Pokémon fainted first. That turn's replace request plays the one its events name
(our `withdraw` with `pivot`, or our `faint`; the pivot's when the rows are not read yet) when it is open. When the
search cannot run or runs fewer than `min_sims` simulations, the network's own answer is played and the record says
why. The search needs Champions Gym 0.5.0 or later, which builds engine games from a live battle; with an earlier
version every turn records that it could not run.

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

## Decision-time search (`cg.search`, off by default)

With `cg.search.enabled: true` every turn request is also searched before it is answered. The player samples
`worlds` completions of the opponent's hidden team from the belief heads, builds an engine game for each from the
live battle, and grows one tree per completion: both sides choose at once at every node by regret matching
started from the network's policy, its prediction and its matrix, and new positions are valued by the network.
The action played is the one the averaged root strategy favours most, inside the legal and rule masks.

The search stops at `ms` or the request's deadline, whichever is sooner, and earlier once the answer has settled
(`settle_sims`, `settle_share`, `settle_gap`). Turns before `from_turn`, requests with a single open action, team
preview and replacements are not searched. When the search cannot run or runs fewer than `min_sims`
simulations, the network's own answer is played and the record says why. The search needs Champions Gym 0.5.0 or later,
which builds engine games from a live battle; with an earlier version every turn records that it could not run.

# pcsa.belief: what each opposing Pokémon could be

The belief tracker of research/pcsa-belief (stages 1-2), ported for the neural player's belief feature modes.
Per opposing member: a weighted set of complete builds (the listed builds, the listed builds with any spread or
any item, and a free spread x item x ability component), narrowed at every decision by the evidence in the
protocol (reveals, moves used, turn order, damage dealt and taken), with 99% stat intervals, speed probabilities
and damage ranges read from the posterior. Design and numbers: research/pcsa-belief/belief/README.md,
report.md, stage2-prep.md.

numpy only (no torch). Only the neural player reaches it (pcsa.rl_training.belief_features; .importlinter).

## Modules

From research/pcsa-belief/belief (each file names the research file and version; pcsa changes are marked
`pcsa:`). research's fast.py is split over several modules for pcsa's 400-line limit; `pcsa.belief.fast` re-exports
the names research code uses.

| module | what |
|---|---|
| `space.py` | species spaces: the spread table, listed builds, item / ability / move priors (stage 1's dense rows not ported) |
| `calc.py` | pcsa.battle.damage vectorised over rows, with the local fixes (weather defences, Multiscale, pinch, Unaware, resist berries, Protean, Muscle Band, Iron Ball) |
| `hits.py`, `evidence.py` | hits, turn orders, Mega lines and reveals read from protocol lines (one incremental reader) |
| `keys.py` | item / ability classes per calculation, `FastPriors`, `FastSpecies`, `FastSpaces` |
| `grids.py` | the vectorised roll arithmetic, `own_fighter`, `our_mega_form`, `Grid` |
| `member_state.py`, `member_grids.py`, `member_out.py`, `member.py` | one member (`FastMember`): state and weights, calculation grids, outputs, evidence |
| `board.py` | `IncBoard` (one private Battle advanced line by line) and `FastTracker` |
| `fast.py` | `BattleBelief`: one battle's belief as the features read it |

## The setting (config/players/neural.yaml)

- `features: checkpoint` (default): the feature mode the network was trained with, read from the checkpoint's own
  run config (`config.features`, written by research rl2/train.py; missing = `base`, which is every network before
  the belief tracker, gold940 included). So the live default is unchanged: gold940 plays `base`, no belief code runs.
- `features: base | belief | belief+mega`: an explicit mode. On a `base` checkpoint anything but `base` is refused
  at build (ConfigError): an old network is never fed belief columns. Between belief modes it is allowed and logged
  (`neural_features_override`).
- `belief_crit_lines: false`: whether the protocol prints critical hits. The phone's battle log does not, so live
  every hit may have been a crit (the damage likelihood mixes in a x1.5 part, research's `crit_reported=False`).
  Only a simulator run should set true (research's rollouts: crit lines shown, crit hits left out).
- Always on in pcsa: a seen build with an empty item or ability (the ladder never showed it; seen_builds.json has a
  dozen) holds an unknown one (the species' item / ability prior) instead of "no item" / the species' first ability
  (`FastPriors.empty_unknown`; research reads them as given, so the equivalence checks run with it off).

The run manifest's `player_params` show `features` (the setting) and `features_used` (the mode played); in a belief
mode every decision's notes carry `belief` (members tracked, updates, rebuilds, evidence counts).

## Live fixes

- Critical hits: above (`belief_crit_lines`).
- A decision's protocol that does not extend the one read before (a battle log re-read differently; 169 of 2,798
  recorded live decisions in stage 1): the tracker starts again and reads the whole protocol turn by turn, each turn
  with the request first seen at it, so it ends where a tracker that saw the new protocol all along would
  (`BattleBelief._rebuild`; tests/belief). Dry run: 128 such decisions in the 3,574 recorded ones, all rebuilt.
- One Battle per decision (the live loop): the tracker keeps its own board and reads only the protocol text and the
  request; memos keyed by object ids are cleared on every update.
- Our own new Mega: in the belief modes our member's Fighter has its Mega ability (Tough Claws, ...) once evolved
  (`grids.own_fighter`), and `belief+mega` adds the damage of our Mega form in the coming turn.
- Not changed in `base` (documented, kept so gold940 reads what it was trained on): `pcsa.players.opponent.
  own_fighter` keeps the base ability of our evolved Mega in the damage columns (stage 0 "Looks wrong" 1), and the
  damage columns ignore a Mega Evolution in the coming turn (stage 0: 65-85% of Mega-turn hits inside the range).

## Cost (live machine, recorded live decisions, research/pcsa-belief/stage4/runs)

3,574 decisions of 355 battles, measured while a research training run used every core. Per decision, the player's
own `ms` (observation, network, guard rules): base 9.6 ms mean (p90 13); belief+mega 17.3 ms mean, p90 26 ms, worst
153 ms (the first decisions of a battle build species spaces); the observation alone 2.9 vs 9.9 ms. Team preview
47 ms base, 57 ms belief+mega. The decision budget is `max_decision_s: 5.0`
(minus `reserve_s`), so the belief modes use well under 1% of it; a slow decision would go to the fallback anyway.

## Equivalence with research (research/pcsa-belief/stage4)

`equiv_pcsa.py` rebuilds archived simulator decisions (both seats) once per implementation: pcsa's observation
equals research rl2's bit for bit in all three modes (1,579 decisions, 111 team previews; v2-teams-1111 u00845 /
u00848 and research's own belief+mega rollouts v2-belief u00002), and pcsa's `base` equals pcsa main's. A short
version runs in the tests (`tests/rl_training/test_belief_parity.py`, skipped without the research folders).

## Stage 3

`FormerPlayer.matrix_blend` (0.0) and `FormerPlayer.matrix_values` (None), applied in `FormerPlayer._blend`: the
place for research's matrix-cell blend (research/pcsa-belief stage 3: rl2/matrix.py, rl2/player.py `matrix_blend`,
default off, still in progress on 12 Nov). As there: logits + matrix_blend x Q, with Q_i the belief's one-turn value
of our action i against the opponent head's kept actions; never on a forced switch; the network's logits on any
error. Not ported: off. Porting it means copying rl2/matrix.py's `action_values` as `matrix_values` and, in feature
mode `base`, keeping a belief for the matrix only (research does).

## A live trial

1. A network trained in a belief mode: research `runs/v2-belief/ckpt/<update>.pt` (its run config says
   `features: belief+mega`) once its validations pass; copy it to `assets/models/` (or `$PCSA_NETWORKS`).
2. This branch in the code the phone runs: `feature/belief` merged into main (the live session and research's
   training both run pcsa main's editable install; merge only when no training run is going).
3. `pcsa run --config config/agent.yaml --player neural --set policy.neural_checkpoint=assets/models/<file>.pt ...`
   The mode follows the checkpoint; the manifest says `features_used: belief+mega`.
4. Back to today: drop the `--set` (gold940, `base`, exactly as before).

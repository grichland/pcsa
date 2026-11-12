# Architecture

## Packages and the one rule between them

```
common  <-  battle  <-  phone | sim  <-  players  <-  belief  <-  run  <-  live  <-  cli
                                                     sprite_training (the sprite network's trainer; only cli imports it)
                                                     rl_training (the neural player's network; only players.neural imports it)
```

| Package | Question it answers |
|---|---|
| `common` | paths, config schema, clock, logging, errors, shared types |
| `battle` | what is going on in the fight: game data, team text, log rows to protocol, battle state, the usage-ranked set catalog the simulator's teams come from |
| `phone` | reading the device: adb screenshots, screens, OCR, sprites, the log panel, team sheets, calibration |
| `sim` | the Showdown engine: the local server and the team validator `pcsa sim play` runs on |
| `players` | how it decides: random, the two heuristics, max_damage, best_matchup_max_damage, disciplined, the mixed pool, neural (and the sim-only endgame) |
| `belief` | what each opposing member could be: the belief tracker of research/pcsa-belief (weighted builds narrowed by the protocol's evidence; stat, speed and damage ranges), read by the neural player's belief feature modes through `rl_training`; numpy only (`src/pcsa/belief/README.md`) |
| `run` | run records and their read-only view: battle bookkeeping, the recorder, `pcsa sim play`, the `pcsa watch` dashboard |
| `live` | live play on the phone: device input, taps, navigation, session, turn pipeline, handlers, supervisor, `pcsa run` |
| `sprite_training` | the sprite network's trainer, `pcsa train sprites` (needs the `train` extra; nothing at runtime imports it) |
| `rl_training` | the neural player's network (the poke-singles-former), inference only, copied from research/pcsa-rl/rl and rl2 (each file names its source version): guess-free observations, the turn parser, the transformer with its memory of earlier decisions, the opponent head and pair matrix, the 0.99 cap, team preview, and the policy player (propose, commit, restore). Needs torch (`train` extra); only `players.neural` imports it, lazily |

The direction is enforced by `import-linter` (`.importlinter`) in CI. `phone` never imports `battle` state
construction; `battle` never imports `phone`; that boundary is what makes the loop testable without a phone.
No package imports `live` except `cli`.

## Rules enforced by tooling

- Every timeout, retry count and screen region lives in `config/`. Unknown keys are rejected. Measured pixel
  thresholds stay named constants next to the vision code that uses them (`phone/screens.py`, `fields.py`, ...).
- `time.*` only in `common/clock.py`; `Path(...).parents` only in `common/paths.py`; `print` only in `cli.py` and
  `common/setup.py`. Files stay under 400 lines and functions under 60
  (tests/unit/test_code_limits.py).
- No side channels: a caller reads returned values or declared protocol methods, never attributes set on the
  callee afterwards. No attribute injection after construction.
- Live quirks are recorded in `docs/quirks.md`, not in comments scattered through handlers.

## Resilience model (live play)

Four exception classes, four meanings (`common/errors.py`):

| Class | Meaning | Who handles it |
|---|---|---|
| `Transient` | retry in place with backoff | the device client, the turn pipeline |
| `Recoverable` | restart the session from whatever screen the phone shows | the supervisor |
| `SafetyStop` | the run ends; the next action could leave Ranked Singles or cost the account something | nobody: it propagates |
| `ConfigError` | nothing was done | the CLI |

There is no "refuse to act" path. A turn whose battle log could not be read is decided from the screen
alone (the `screen` rung of the degrade ladder, `live/decide.py`); an unknown screen is waited on, backed out
of, and finally restarted from; a dropped USB connection is reconnected; the title screen after a network
error is a node in the navigation graph like any other. Only a safety violation ends a run.

## Output

Everything the package writes lives under `output/` (`common/paths.py`): `runs/<timestamp>-<name>/` with
`manifest.json`, `events.jsonl`, `state.json`, `summary.json`, frames and per-battle folders; `observe/`,
`nav/` and `scout/` sessions; `sim/` results; `sprite_net/` from the sprite network's trainer.
`pcsa sim play` keeps every battle (teams, picks, winner, protocol) in `sim/<stamp>/battles.jsonl.gz`.

## The dashboard (`pcsa watch`)

A local page over one run folder, in `run/watch*.py` with its page in `run/static/` (plain HTML, CSS and
JavaScript; standard-library HTTP server on 127.0.0.1; no extra dependency, no torch, no phone or adb import).
It only reads what a run already writes: `events.jsonl` (read incrementally, up to the last complete line),
`manifest.json`, the newest file in `frames/`, and per battle `decisions/NNNN-<kind>.json` + `.txt`,
`battle_log.txt` (or `log.txt`), which the live loop rewrites after every visit to the log. A file that is
missing or half written leaves its panel empty until it lands. `watch_events.py` keeps the record, win rates,
rating, battle list and warnings; `watch_decisions.py` the decision rows and the players' notes;
`watch_matrix.py` the neural player's pair matrix of a decision (names, the row played, a guard-rule
override); `watch_log.py` the battle log as turns; `watch_battle.py` the two teams from the newest decision
record (the opponent's side shows revealed information only); `watch_field.py` weather, terrain and side
conditions from its protocol; `watch_names.py` the written names of ids, taken from the record itself. The
page has four panels: the phone frame over the decision matrix on the left; win rate, the teams (ours blue,
theirs red) and the battle log on the right. A new decision plays once as an animation of the matrix. The page polls `/api/state`
(`?battle=N` pins a battle) once a second and `/api/frame` when the frame changes.

## Device independence

Regions are fractions of the game viewport (`config/screen.yaml`). The per-phone profile from
`pcsa calibrate` says where the viewport sits in the captured frame and how rows are pitched. Labelled
controls are located by OCR at runtime; the configured rectangle is the search window and the fallback.
Verified on one device so far; a second phone's frames must join the perception corpus before support for
other aspect ratios counts as verified.

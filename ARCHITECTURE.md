# Architecture

## Packages and the one rule between them

```
common  <-  battle  <-  phone | sim  <-  players  <-  run  <-  live  <-  cli
                                                     sprite_training (the sprite network's trainer; only cli imports it)
```

| Package | Question it answers |
|---|---|
| `common` | paths, config schema, clock, logging, errors, shared types |
| `battle` | what is going on in the fight: game data, team text, log rows to protocol, battle state, build estimates |
| `phone` | reading the device: adb screenshots, screens, OCR, sprites, the log panel, team sheets, calibration |
| `sim` | the Showdown engine: the local server and the team validator `pcsa sim play` runs on |
| `players` | how it decides: five kinds (random, the two heuristics, max_damage, the mixed pool) |
| `run` | run records and their read-only views: battle bookkeeping, the recorder, `pcsa watch`, `pcsa sim play` |
| `live` | live play on the phone: device input, taps, navigation, session, turn pipeline, handlers, supervisor, `pcsa run` (private: not in the public repository) |
| `sprite_training` | the sprite network's trainer, `pcsa train sprites` (needs the `train` extra; nothing at runtime imports it) |

The direction is enforced by `import-linter` (`.importlinter`) in CI. `phone` never imports `battle` state
construction; `battle` never imports `phone`; that boundary is what makes the loop testable without a phone.
No package imports `live`: `cli` loads it only when it is present, so the public repository, which leaves
it out, is complete without it.

## Rules enforced by tooling

- Every timeout, retry count and screen region lives in `config/`. Unknown keys are rejected. Measured pixel
  thresholds stay named constants next to the vision code that uses them (`phone/screens.py`, `fields.py`, ...).
- `time.*` only in `common/clock.py`; `Path(...).parents` only in `common/paths.py`; `print` only in `cli.py`,
  `train_cli.py` and `common/setup.py`. Files stay under 400 lines and functions under 60
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
`nav/` and `scout/` sessions; `sim/` results; `sprite_net/` from the sprite network's trainer. `output/latest` points at the newest run.

## Device independence

Regions are fractions of the game viewport (`config/screen.yaml`). The per-phone profile from
`pcsa calibrate` says where the viewport sits in the captured frame and how rows are pitched. Labelled
controls are located by OCR at runtime; the configured rectangle is the search window and the fallback.
Verified on one device so far; a second phone's frames must join the perception corpus before support for
other aspect ratios counts as verified.

# Architecture

## What runs where

```
phone screen --adb--> phone/ --rows--> battle/ + events/ --LIVE events--> cg/ --JSON lines--> pcsa.cg.serve (own process)
                                                                                               Champions Gym Session
     taps <--adb-- live/ <--order-- decide/ <------------- order, Battle, network outputs ---  + pcsa.network.policy
```

The controller process reads the phone and taps. It holds no battle state of its own beyond the rows it has
read: Champions Gym (CG) builds the Battle from the events, decides what is legal, and asks the player. The
player process is this repo's `pcsa.cg.serve`, a thin wrapper around CG's `Session` that runs CG's neural
player shell with the policy in `pcsa.network`; it is the only process that loads torch.

## Packages

| Package | Question it answers |
|---|---|
| `common` | paths, config schema, clock, logging, errors, shared types; setup and doctor |
| `battle` | game knowledge with no phone and no engine: ids, names, stats, the team text, parsing a log row, hints for the request (PP counted, Choice lock, Encore, Taunt) |
| `phone` | reading the device: adb screenshots, screens, OCR, sprites, the log panel, team sheets, calibration |
| `events` | log rows to Champions Gym's LIVE events, and the whole-battle stream from recordings |
| `decide` | the one interface to whatever decides: Request, Offer, Order, Player; the runner that asks the Player on a worker thread and falls back so there is always an order |
| `cg` | the Champions Gym Player: messages, transport (a subprocess speaking JSON lines, or in process), start-up check, the resend after a restart; `serve.py` and `view.py` run inside the player process |
| `network` | the neural player's network, inference only, copied from the training code (each file names its source), and the policy CG's shell runs |
| `live` | live play on the phone: device input, taps, navigation, session, turn pipeline, handlers, supervisor |
| `run` | run records and their read-only view: battle bookkeeping, the recorder, the `pcsa watch` dashboard |
| `sprite_training` | the sprite network's trainer, `pcsa train sprites` |
| `cli` | the `pcsa` command; each command imports its code only when it runs |

Rules enforced by `import-linter` (`.importlinter`) in CI:

- nothing imports poke-env;
- torch is imported only by `network` and `sprite_training` (the CLI reaches the trainer lazily);
- `champions_gym` is imported only by `cg/serve.py`, `cg/view.py` and `network`, the code that runs in the
  player process;
- `cg` is imported only by `decide/load.py`, and only when the `cg` Player is asked for;
- nothing imports `network`: the player process loads the policy by name;
- the `run/watch*` modules import no phone, live, Champions Gym or network code.

`phone` never imports `live`; `battle` never imports `phone`; that boundary is what makes the loop testable
without a phone. `common/setup.py` loads `champions_gym` and torch by name to report their versions; it is the
one place outside those rules.

## Rules enforced by tooling

- Every timeout, retry count and screen region lives in `config/`. Unknown keys are rejected. Measured pixel
  thresholds stay named constants next to the vision code that uses them (`phone/screens.py`, `fields.py`, ...).
- `time.*` only in `common/clock.py`; `Path(...).parents` only in `common/paths.py`; `print` only in `cli.py`
  and `common/setup.py`. Files stay under 400 lines and functions under 60 (`tests/unit/test_code_limits.py`;
  the five network files copied from the training code are exempt, so they stay diffable with their source).
- No side channels: a caller reads returned values or declared protocol methods, never attributes set on the
  callee afterwards.
- Live quirks are recorded in `docs/quirks.md`, not in comments scattered through handlers.

## Resilience model (live play)

Four exception classes, four meanings (`common/errors.py`):

| Class | Meaning | Who handles it |
|---|---|---|
| `Transient` | retry in place with backoff | the device client, the turn pipeline |
| `Recoverable` | restart the session from whatever screen the phone shows | the supervisor |
| `SafetyStop` | the run ends; the next action could leave Ranked Singles or cost the account something | nobody: it propagates |
| `ConfigError` | nothing was done | the CLI |

There is no "refuse to act" path. A decision comes from the first of these that produces one
(`decide/runner.py`), and the record's `degrade` says which:

| `degrade` | Who chose |
|---|---|
| `player` | the Player (Champions Gym) answered in time with a choice the screen offers |
| `builtin` | the Player failed, was late or answered something the screen does not offer: the built-in Player takes the first legal choice |
| `screen` | the built-in Player's choice was not valid either: the first action the screen offers |
| `blind` | the screen offered nothing usable: the first of a fixed list of actions |
| `automatic` | the game plays the turn itself (a locked move); nothing was asked |

Inside Champions Gym there is a second fallback: when the network errs or takes longer than its time limit,
CG's scripted `tower` player decides, and the reply says so (`notes.cg.model.fallback`).

An unknown screen is waited on, backed out of, and finally restarted from; a dropped USB connection is
reconnected; a lost connection to the game is ridden out (`live/reconnect.py`); a player process that dies is
started again and sent the whole battle so far, with the choices already played marked as such. Only a safety
violation ends a run.

## Champions Gym

The dependency is the release wheel pinned in `pyproject.toml` and `tools.lock.json` (0.4.0; LIVE contract
0.9, model contract 0.6). `cg/adapter.py` speaks LIVE messages: `battle_start` with our six sets and their
preview, `preview_request`, `events`, `decision_request`, `battle_end`. Before the first battle it asks for
CG's status and refuses to start if the model's input layout or contract does not match the installed CG, or
if the checkpoint loaded is not the one configured (`cg/startup.py`).

`cg/serve.py` replaces `python -m champions_gym.live` as the player process, with the same flags and the same
one-line-in, one-line-out protocol. It builds the player, wraps it so that it keeps the exact Battle each
request was decided on, and adds two things to each reply's notes: `view`, a snapshot of that Battle, and
`net`, the network's outputs for the decision. It uses CG's public `Session` and `neural.load`; to capture the
Battle a built-in player was asked about it overrides one private method of `Session`, which must be rechecked
when the pinned CG version changes.

`network/policy.py` is the policy CG's neural shell calls. It runs one step of the network per request with
that battle's memory and returns the policy; the belief heads run in the same forward pass and change no
other output.

## Output

Everything the package writes lives under `output/` (`common/paths.py`). A run is
`runs/<timestamp>-<name>/`:

| File | What |
|---|---|
| `manifest.json` | seed, player and its parameters, config, source and asset hashes, device |
| `events.jsonl` | one line per thing that happened: observations, decisions, turns, battle starts and ends, rank |
| `state.json`, `summary.json` | counts and rank history |
| `frames/` | every captured frame |
| `cg/`, `cg-start.json`, `cg-stderr.log` | Champions Gym's own logs and its start-up status |
| `battle-N/battle.json`, `battle_log.txt`, `log.jsonl`, `log-frames/` | the battle's outcome and teams, the log as text, the OCR'd rows, the log captures |
| `battle-N/events.json`, `cg-messages.jsonl` | the LIVE events sent, and every message to and from CG |
| `battle-N/team_preview.json` | the preview read, the picks, and under `cg` the network's preview outputs |
| `battle-N/decisions/NNNN-<kind>.json` | one record per decision |

### The decision record (`pcsa-decision/4`)

`live/decision_record.py` writes it: the choice and the order (`"/choose move meteormash mega"`,
`"/choose switch Primarina"`), the legal list, `degrade` and its reason, timings, what the screen offered
(`offer`), the hints, the rows read and the LIVE events sent for this decision, and `notes`. For a Champions
Gym decision `notes.cg` is the reply's notes:

```
action, kind, legal, player, turn, anomalies, corrections
model   {fallback, ms, reused, slow, value}
view    the Battle the player decided on:
        turn, kind, picks, opp_preview, own[6], opp[6], weather, fields, side, opp_side
        each member: species, form, hp (0..1), fainted, status, boosts, effects, moves [[id, pp]],
                     item (null unknown, "" gone), ability, types, active, revealed
net     the network's outputs (absent on a fallback or a replayed request):
        value, tau, pi[11], score[11], rows[11], cols[13], p_opp[13], matrix[11][13],
        belief[6]: species, brought, lead (preview only), item [[id, p] x3], ability [[id, p] x3],
                   moves [[id, p] up to 8]
```

All ids are Champions Gym's string ids, the ones `assets/game/cg/*.json` is keyed by. The numeric ids the
network uses are decoded with `assets/game/cg/vocab.json`, written from CG's data tables by
`tools/data/cg_vocab.py`.

`tools/replay_run.py` rebuilds these records for a recorded run from its saved CG messages, with no phone.

## The dashboard (`pcsa watch`)

A local page over one run folder, in `run/watch*.py` with its page in `run/static/` (plain HTML, CSS and
JavaScript; standard-library HTTP server on 127.0.0.1; no torch, no Champions Gym, no phone or adb import).
It only reads what a run already writes: `events.jsonl` (read incrementally, up to the last complete line),
`manifest.json`, the newest file in `frames/`, and per battle `decisions/NNNN-<kind>.json`,
`team_preview.json` and `battle_log.txt` (or `log.txt`). A file that is missing or half written leaves its
panel empty until it lands.

| Module | What it gives the page |
|---|---|
| `watch.py` | the server, the run and frame discovery, the top bar |
| `watch_events.py` | the record, win rates, rating, battle list and warnings |
| `watch_decisions.py` | the decision rows and the player's notes in the shape the page reads |
| `watch_matrix.py` | the decision matrix: the network's 11 x 13 with each of their moves folded with its Mega variant into the page's 11 x 9, weighted so that every row's score is unchanged |
| `watch_view.py` | the two teams, the field and the battle summary from `view` |
| `watch_belief.py` | the belief per opposing member; a revealed fact replaces it |
| `watch_names.py` | written names from the team sheet and Champions Gym's tables |
| `watch_log.py` | the battle log as turns |
| `watch_battle.py` | puts the battle's panels together |

The page polls `/api/state` (`?battle=N` pins a battle) once a second and `/api/frame` when the frame changes.
A run recorded before this format (no `view` in its records) shows its record, rating, log and decision list;
the teams, field and matrix panels say so in one line.

## Device independence

Regions are fractions of the game viewport (`config/screen.yaml`). The per-phone profile from
`pcsa calibrate` says where the viewport sits in the captured frame and how rows are pitched. Labelled
controls are located by OCR at runtime; the configured rectangle is the search window and the fallback.
Verified on one device so far; a second phone's frames must join the perception corpus before support for
other aspect ratios counts as verified.

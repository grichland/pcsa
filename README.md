# pcsa

pcsa plays Pokémon Champions singles from the screen of a real phone. Everything it knows is read from
screenshots: text by OCR, Pokémon by a sprite network trained for this project, screens by a rule table. The
battle log is rebuilt as a Pokémon Showdown protocol so the agent holds the same battle state the Showdown
engine would, and the decision is made by one of the player kinds in `players/` from that tracked battle.
The Showdown engine doubles as the simulator the players are measured in.

> **Unofficial fan project**, not affiliated with or endorsed by Nintendo, The Pokémon Company, Creatures or
> GAME FREAK. Automated play on the live ranked ladder may break the game's terms of service, affects real
> players' matches and can get an account banned. Please run experiments in the simulator (`pcsa sim play`)
> rather than on the live ladder. Credits and licenses: `NOTICE.md`.

## What is here

1. **Perception** (`phone/`): adb screenshots, OCR for text, the sprite network for Pokémon
   (`docs/sprites.md`; its trainer is `sprite_training/`, `pcsa train sprites`), and a rule table that names
   each screen.
2. **State reconstruction** (`battle/`): the battle log is read every turn and rebuilt as a Pokémon Showdown
   protocol, so the agent holds the same battle state the Showdown engine would.
3. **Decision** (`players/`): nine player kinds, all usable live and in the simulator: `random`, `simple_heuristic` (poke-env's numbers),
   `custom_heuristic` (the tuned `config/players/heuristic.yaml`), `max_damage` (the strongest attack, nothing
   else), `best_matchup_max_damage` (max_damage plus a switch out of a really bad matchup,
   `config/players/best_matchup.yaml`), `disciplined` (best_matchup plus damage-race rules for priority, Mega,
   replacements and switching, `config/players/disciplined.yaml`), `mixed` (a weighted pool of the others,
   one drawn per game, `config/players/mixed.yaml`), `endgame` (disciplined plus a 1v1 chain search once both
   sides are down to two members, `config/players/endgame.yaml`) and `neural` (a PPO / imitation network
   trained in the research repo, `config/players/neural.yaml`, network code in `rl_training/`; see "Neural
   player" below).
4. **Simulator** (`sim/`): the Showdown engine as a local server, with its team validator, on which
   `pcsa sim play` plays games between the player kinds.
5. **Action** (`live/`): taps through a fixed menu map, and the session loop that plays battles (`pcsa run`).
   The interface is restricted by construction: there is no path to forfeiting, quitting, editing teams,
   purchases or rating dialogs.

`ARCHITECTURE.md` describes the packages. `docs/quirks.md` records what live play turned up and how each
issue was fixed, and the known defects of the battle-log reader.

## Scope and limitations

- **One device, one game version.** Everything was built on a single phone (the profile in
  `config/devices/`). Screen layouts, the menu map and the perception rules are tuned to that setup; another
  phone needs `pcsa calibrate` and will likely need fixes, and game updates can break the screen rules.
- **Singles only.** The opponents' hidden builds are not inferred from play: `disciplined` and `endgame` assume
  each opponent runs its species' top usage build (`players/opponent.py`), corrected by what the battle has
  revealed; the other players read only what has been revealed.

**Requirements:** Windows 10/11 and an Android phone with Pokémon Champions and USB debugging on; an NVIDIA GPU
speeds up training the sprite network. Python, adb, scrcpy, the OCR models and the battle engine are installed for you.

## Setup

```
winget install astral-sh.uv                       # once per machine
git clone https://github.com/grichland/pcsa.git && cd pcsa
uv run pcsa setup --with sim --with train         # Python env, battle engine, torch, OCR models, output folder
```

`uv run pcsa doctor` repeats the checks without installing anything.

**Simulator** (no phone needed). Any two player kinds:

```
uv run pcsa sim play --a max_damage --b custom_heuristic --battles 100          # both teams from top builds
uv run pcsa sim play --a max_damage --b custom_heuristic --battles 100 --noisy  # both teams from any catalog build
```

Both sides' teams are drawn from the usage-ranked set catalog (`assets/sets/top_builds.json`): each species'
top build by default, with `--noisy` any of its catalog builds weighted by usage score.

Results land in `output/sim/<timestamp>-max_damage-vs-custom_heuristic-seed1/results.json` (`-noisy` appended with
`--noisy`) with a 95% Wilson interval. `max_damage` picks the attack the heuristic's damage estimate scores
highest and nothing else; `mixed` draws one member per game from `config/players/mixed.yaml`.

**Perception on a phone** (read-only: these commands capture the screen and send no input). With the phone
on USB (USB debugging on) and the game open:

```
uv run pcsa calibrate                # measures this phone once; writes config/devices/<model>.yaml
uv run pcsa observe                  # capture and classify the screen in a loop
uv run pcsa scout                    # record a manual walk through the game and report what changed
```

**Live play** (sends input to the phone). With the game open on the lobby:

```
uv run pcsa run --player custom_heuristic --battles 10
uv run pcsa nav --to lobby           # walk the menus to the lobby
```

`--player` takes a player kind (default: `policy.kind` in `config/agent.yaml`, `custom_heuristic`).

For longer evaluation sessions, `scripts\run_agent.ps1` holds the settings (player, seed, run name,
overrides) at the top of the file, stops any agent left over from a run that ended abruptly, syncs the
environment, checks the phone and starts the run. With `$Mirror = $true` (the default) it also opens a phone
mirror window through [scrcpy](https://github.com/Genymobile/scrcpy), which `pcsa setup` installs into
`.tools\scrcpy\` (a `scrcpy` on PATH works as a fallback); when it is missing the script prints a warning and runs
without the mirror:

```
powershell -ExecutionPolicy Bypass -File scripts\run_agent.ps1 -Battles 25 -MaxBattlesTotal 30
```

**Neural player.** `neural` plays a checkpoint from the research runs (`research/pcsa-rl`: `runs/<run>/ckpt/*.pt`,
any size S/M/L, read from the checkpoint). It needs torch (`uv sync --inexact --extra train`; `pcsa setup --with
train` installs it too) and runs on the CPU. Point it at a network with one override (or set `checkpoint` in
`config/players/neural.yaml`; relative paths are looked up in `$PCSA_NETWORKS`, then in the repo, e.g. a copy in
`networks/`, which git ignores):

```
uv run pcsa sim play --a neural --b disciplined --battles 100 --set policy.neural_checkpoint=C:/nets/s1-u1401.pt
uv run pcsa run --player neural --battles 10 --set policy.neural_checkpoint=C:/nets/s1-u1401.pt
```

Live play defaults to the most likely action (`sampling: greedy`; `sample` draws from training's capped 0.99
distribution) and training's team preview (the critic scores all 60 lead/second/third picks, softmax at 0.1,
top-p 0.9; `preview: greedy` takes the best). Whenever the network cannot decide (no or unreadable checkpoint,
a size mismatch, no tracked Battle, an error, slower than `max_decision_s` or the turn deadline) the
`fallback` kind (`disciplined`) decides, and the decision's notes say why (`rule: neural:fallback`, `why`).

## Where things go

| What | Where |
|---|---|
| Everything the agent writes (runs, frames, simulator results, sprite network training) | `output/` (override with `PCSA_OUTPUT`) |
| Every simulator battle (both teams, picks, winner, full protocol), one JSON line each | `output/sim/<stamp>/battles.jsonl.gz` |
| Installed tools (adb, scrcpy, OCR models, Node, Showdown) | `.tools/` (pinned in `tools.lock.json`) |
| Settings | `config/agent.yaml`, `config/screen.yaml`, `config/devices/*.yaml`; players: `config/players/heuristic.yaml`, `best_matchup.yaml`, `disciplined.yaml`, `mixed.yaml`, `endgame.yaml`, `neural.yaml` |
| Network checkpoints for the neural player (optional copies; any path works) | `networks/` (or `PCSA_NETWORKS`; ignored by git) |
| Game data, the sprite network (see `docs/sprites.md`), the simulator's set catalog | `assets/` |

## Development

```
uv sync --group dev
uv run poe check                     # ruff, pyright, import-linter, pytest
uv run pre-commit install
```

`pytest` skips the tests marked `torch` (needs the train extra) and `ocr` (runs the real OCR engine instead of
cached boxes); select them with `uv run pytest -m torch` or `-m ocr`.

Design: `ARCHITECTURE.md`. Live quirks, the runs that motivated them and the known reader defects:
`docs/quirks.md`. The sprite network: `docs/sprites.md`.

## Acknowledgements

- [championsbattledata.com](https://championsbattledata.com): the Pokémon Champions ranked-usage statistics
  that rank the simulator's set catalog.
- [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) and [poke-env](https://github.com/hsahovic/poke-env):
  the battle engine, game data and battle model everything else is built on.

Full credits and licenses: `NOTICE.md`.

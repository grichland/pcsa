# pcsa

pcsa plays Pokémon Champions singles from the screen of a real phone. Everything it knows is read from
screenshots: text by OCR, Pokémon by a sprite network trained for this project, screens by a rule table. The
battle log is rebuilt as a Pokémon Showdown protocol so the agent holds the same battle state the Showdown
engine would, the opponent's hidden builds are estimated from ranked usage statistics, and the decision is made
by one of the player kinds in `players/`. The Showdown engine doubles as the simulator the players are
measured in.

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
3. **Decision** (`players/`): five player kinds: `random`, `simple_heuristic` (poke-env's numbers),
   `custom_heuristic` (the tuned `config/players/heuristic.yaml`), `max_damage` (the strongest attack, nothing
   else) and `mixed` (a weighted pool of the others, one drawn per game, `config/players/mixed.yaml`).
4. **Simulator** (`sim/`): the Showdown engine as a local server, with its team validator, on which
   `pcsa sim play` plays games between the player kinds.
5. **Action** (`live/`, not published): taps through a fixed menu map. The interface is restricted by
   construction: there is no path to forfeiting, quitting, editing teams, purchases or rating dialogs.

`ARCHITECTURE.md` describes the packages. `docs/quirks.md` records what live play turned up and how each
issue was fixed, and the known defects of the battle-log reader.

## Scope and limitations

- **One device, one game version.** Everything was built on a single phone (the profile in
  `config/devices/`). Screen layouts, the menu map and the perception rules are tuned to that setup; another
  phone needs `pcsa calibrate` and will likely need fixes, and game updates can break the screen rules.
- **Singles only**, with the opponents' hidden builds estimated from public usage statistics.

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
uv run pcsa sim play --a max_damage --b custom_heuristic --battles 100          # opponents run top builds
uv run pcsa sim play --a max_damage --b custom_heuristic --battles 100 --noisy  # opponents run any catalog build
```

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

**Live play is not published.** The layer that sends input to the phone and plays battles (device input,
taps, menu navigation and the session loop, `pcsa run`) is kept in a private package and is not part of this
repository. Everything it relies on (perception, state reconstruction and the players) is here.

## Where things go

| What | Where |
|---|---|
| Everything the agent writes (runs, frames, simulator results, sprite network training) | `output/` (override with `PCSA_OUTPUT`) |
| Installed tools (adb, scrcpy, OCR models, Node, Showdown) | `.tools/` (pinned in `tools.lock.json`) |
| Settings | `config/agent.yaml`, `config/screen.yaml`, `config/players/*.yaml`, `config/devices/*.yaml` |
| Game data, the sprite network (see `docs/sprites.md`), compact set catalog | `assets/` |

## Development

```
uv sync --group dev
uv run poe check                     # ruff, pyright, import-linter, pytest
uv run pre-commit install
```

Design: `ARCHITECTURE.md`. Live quirks, the runs that motivated them and the known reader defects:
`docs/quirks.md`. The sprite network: `docs/sprites.md`.

## Acknowledgements

- [championsbattledata.com](https://championsbattledata.com): the Pokémon Champions ranked-usage statistics
  behind the opponent build estimates.
- [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) and [poke-env](https://github.com/hsahovic/poke-env):
  the battle engine, game data and battle model everything else is built on.

Full credits and licenses: `NOTICE.md`.

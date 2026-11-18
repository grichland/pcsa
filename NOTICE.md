# Notices

This project's own code, its trained networks and its documentation are under the MIT License (`LICENSE`).
It builds on the work below.

## Included in this repository

| What | Where | Source and license |
|---|---|---|
| Species, learnsets, battle text and name tables | `assets/game/species.json`, `learnsets.json`, `battle_text.json`, `vocab.json`, `rules.json` | Generated from [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) at the commit named in `tools.lock.json` (the name tables through poke-env's gen 9 data at the time they were made). MIT License, Copyright (c) 2011-2026 Guangcong Luo and other contributors. Neither Showdown nor poke-env is installed, imported or run by this project. |
| Champions Gym's legal tables and id numbering | `assets/game/cg/` | Written out from [Champions Gym](https://github.com/grichland/poke-champions-gym) 0.4.0 (MIT), whose game data is generated from Pokémon Showdown's `champions` mod at the same commit. |
| Champions item effects, move PP and learnable moves (`vocab.json`: Champions items, `champions_moves`) | `assets/game/vocab.json` | From the Pokémon Champions battle data published by [championsbattledata.com](https://championsbattledata.com). |
| The wordings of the game's battle log | `assets/game/log_wordings.json` | Collected from this project's own recorded battles. |
| Usage-ranked builds per species, and the opposing builds this project met on the ladder | `assets/sets/top_builds.json`, `assets/sets/seen_builds.json` | The first is built from the Pokémon Champions ranked-usage statistics published by [championsbattledata.com](https://championsbattledata.com); thanks to its maintainers for collecting and sharing the data. The second is from this project's own recorded battles. Data only: nothing here reads them at run time. |
| Sprite network weights and reference vectors | `assets/sprites/` | Trained by this project (`pcsa train sprites`, `docs/sprites.md`). No sprite images are distributed. |
| The neural player's checkpoints and network code | `assets/models/`, `src/pcsa/network/` | Trained and written by this project's author by self-play on Champions Gym. |
| A screenshot of this project's live dashboard after a ranked battle (no trainer names on screen; it links to a demo video on YouTube), and one screenshot of its ranked result after a battle (no trainer name or ID on screen) | `docs/images/demo-poster.jpg`, `docs/images/master-ball-rank2.jpg` | A screen from Pokémon Champions, © Nintendo / Creatures Inc. / GAME FREAK inc. / The Pokémon Company, shown for illustration only. |

## Downloaded by `pcsa setup`, not redistributed

| Tool | License |
|---|---|
| [scrcpy](https://github.com/Genymobile/scrcpy) | Apache-2.0, Copyright (C) Genymobile |
| [RapidOCR](https://github.com/RapidAI/RapidOCR) PP-OCRv5 models | Apache-2.0 |
| Android SDK Platform-Tools (adb) | Android Software Development Kit License |

## Python dependencies

Installed under their own licenses: [Champions Gym](https://github.com/grichland/poke-champions-gym) (MIT; its
wheel carries its own third-party notices, among them Pokémon Showdown for the game data and the Rust crates it
is built with), [RapidOCR](https://github.com/RapidAI/RapidOCR) (Apache-2.0), numpy, OpenCV, pydantic, PyYAML,
onnxruntime and, for the neural player and for training the sprite network, PyTorch.

## Trademarks

Pokémon and Pokémon character names are trademarks of Nintendo, Creatures Inc. and GAME FREAK inc. This is
an unofficial fan project, not affiliated with or endorsed by Nintendo, The Pokémon Company, Creatures Inc.
or GAME FREAK inc.

# Notices

This project's own code, its trained network and its documentation are under the MIT License (`LICENSE`).
It builds on the work below.

## Included in this repository

| What | Where | Source and license |
|---|---|---|
| Species, learnsets, battle text, fallback PP rules, item/move/ability name tables | `assets/game/` | Generated from [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) at the commit pinned in `tools.lock.json` (name tables via poke-env's gen 9 data). MIT License, Copyright (c) 2011-2026 Guangcong Luo and other contributors. |
| Champions item effects, move PP and learnable moves (`vocab.json`: Champions items, `champions_moves`) | `assets/game/vocab.json` | From the Pokémon Champions battle data published by [championsbattledata.com](https://championsbattledata.com). |
| The simulator's team catalog (usage-ranked builds per species) | `assets/sets/top_builds.json` | Built from the Pokémon Champions ranked-usage statistics published by [championsbattledata.com](https://championsbattledata.com). Thanks to its maintainers for collecting and sharing the data. |
| Sprite network weights and reference vectors | `assets/sprites/` | Trained by this project (`pcsa train sprites`, `docs/sprites.md`). No sprite images are distributed. |

## Downloaded by `pcsa setup`, not redistributed

| Tool | License |
|---|---|
| [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) (the battle engine) | MIT |
| [Node.js](https://nodejs.org) | MIT |
| [scrcpy](https://github.com/Genymobile/scrcpy) | Apache-2.0, Copyright (C) Genymobile |
| [RapidOCR](https://github.com/RapidAI/RapidOCR) PP-OCRv5 models | Apache-2.0 |
| Android SDK Platform-Tools (adb) | Android Software Development Kit License |

## Python dependencies

Installed from PyPI under their own licenses, among them [poke-env](https://github.com/hsahovic/poke-env)
(MIT), [RapidOCR](https://github.com/RapidAI/RapidOCR) (Apache-2.0), numpy, OpenCV, pydantic, PyYAML,
onnxruntime and, for training the sprite network, PyTorch.

## Trademarks

Pokémon and Pokémon character names are trademarks of Nintendo, Creatures Inc. and GAME FREAK inc. This is
an unofficial fan project, not affiliated with or endorsed by Nintendo, The Pokémon Company, Creatures Inc.
or GAME FREAK inc.

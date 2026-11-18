# Notices

This project's own code, its trained sprite network and its documentation are under the MIT License. Most of
the code is copied from pcsa (the same author). It builds on the work below.

## Included in this repository

| What | Where | Source and license |
|---|---|---|
| Species and item/move/ability name tables | `assets/game/species.json`, `assets/game/vocab.json` | Copied from pcsa, where they were generated from [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) at commit c0845f33d (name tables via poke-env's gen 9 data at the time). MIT License, Copyright (c) 2011-2026 Guangcong Luo and other contributors. |
| Champions item effects, move PP and learnable moves (`vocab.json`: Champions items, `champions_moves`) | `assets/game/vocab.json` | From the Pokémon Champions battle data published by [championsbattledata.com](https://championsbattledata.com). |
| Sprite network weights and reference vectors | `assets/sprites/` | Trained by pcsa (`pcsa train sprites`). No sprite images are distributed. |

poke-env is no longer used: this project does not depend on it or import it. It is credited above only because
pcsa used its gen 9 data when the copied name tables were generated.

## Not redistributed

| Tool | License |
|---|---|
| [RapidOCR](https://github.com/RapidAI/RapidOCR) PP-OCRv5 models (`.tools/ocr/`) | Apache-2.0 |
| Android SDK Platform-Tools (adb, `.tools/adb/`) | Android Software Development Kit License |

## Python dependencies

Installed from PyPI under their own licenses: [RapidOCR](https://github.com/RapidAI/RapidOCR) (Apache-2.0),
numpy, OpenCV, pydantic, PyYAML and onnxruntime.

## Trademarks

Pokémon and Pokémon character names are trademarks of Nintendo, Creatures Inc. and GAME FREAK inc. This is
an unofficial fan project, not affiliated with or endorsed by Nintendo, The Pokémon Company, Creatures Inc.
or GAME FREAK inc.

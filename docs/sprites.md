# Sprite recognition

The agent names Pokémon from their sprites in two places: the six opponent cards on team selection
("preview") and the small sprite at the left of each battle-log row ("log"). It does this with a small
neural network the project trained itself; the repository ships the network's weights and one vector per
known sprite view, and no game imagery.

## How a sprite is matched

1. `phone/sprites.py` cuts the sprite from its background and normalizes its silhouette to a 40x40
   square, with the Lab colour under it and the silhouette's aspect ratio (`feature`).
2. `phone/sprite_net.py` runs the network in numpy (six 3x3 convolutions, three max-pools, global
   average pooling, two dense layers) and returns a unit vector.
3. The vector is compared by cosine with the reference vectors in `assets/sprites/preview_refs.npz` or
   `log_refs.npz`. Team preview takes the best three; a log row is accepted when the best score and its
   margin over the runner-up clear the thresholds in `assets/sprites/sprite_net.json`.

| File | What it holds |
|---|---|
| `assets/sprites/sprite_net.json` | schema, file hashes, the log thresholds, the training summary |
| `assets/sprites/sprite_net.npz` | the network's weights (batch norm folded into the convolutions) |
| `assets/sprites/preview_refs.npz`, `log_refs.npz` | names, base names and 64-number vectors, one per sprite view |

Only numpy runs at play time; torch is needed for training alone.

## Training: `pcsa train sprites`

```
uv run pcsa train sprites                 # train, measure, write output/sprite_net/<stamp>/
uv run pcsa train sprites --install       # the same, then copy the result into assets/sprites
uv run pcsa train sprites --refresh-data  # extract the crops again instead of using the cache
```

It needs the torch extra (`uv sync --extra torch`) and two local inputs that are not in the repository:

- **Reference views**: `output/sprite_source/roster_index.*` and `log_index.*` (`--source`), the
  silhouette indexes of the earlier thumbnail matcher (one view per sprite, from in-game Roster Info
  screenshots and the Champions menu sprites). They are game imagery, so they stay local.
- **Recorded runs**: `output/runs/` (`--runs`). The crops come from each battle's `log.jsonl` (the sprite
  box and frame of every log row) and `battle.json` (the team-preview frame and the six recorded cards).

The extracted crops are cached in `output/sprite_net/data.npz` and `data.json`; `data.json` also lists,
per battle, the opponent's sent-out species and which file they came from.

### Labels

- **Battle-log rows** are labelled by the species their own text names ("Move: The opposing Tauros used
  ..."). The form is the one the recorded matcher picked when it agreed on the species, else any form of
  that species. Rows on which the run saw no sprite, and rows whose text names no single species, stay
  unlabelled.
- **Team-preview cards** are labelled by the opponent later sending that species out. The sent-out
  species come from one of two files in the battle folder:
  - `events.json` (runs recorded through Champions Gym): the `enter` events on the `opp` side
    (`contracts/LIVE.md` in Champions Gym). These name a Pokémon by the species id the log text gives,
    which leaves out the form where forms share a name (Basculegion, Rotom, Tauros); such a card accepts
    any non-Mega form of the species.
  - `showdown.log` (runs recorded before the Champions Gym port): the `switch` and `drag` lines of player
    2, read as plain text. Their details carry the form. A Pokémon that came back in after Mega Evolving
    is listed with its Mega form too.

  `events.json` is used when a battle has both. A battle with neither (an unfinished battle) gives log
  crops but no preview labels.

### What a run of the trainer does

Every batch mixes heavily augmented reference views with lightly augmented real crops. The augmentations
include palette swaps for shiny Pokémon (hue rotation, inverted lightness, colour faded to grey), so
colour helps a match but never outweighs the shape. After training, each reference view is embedded, plus
the centre of the real crops for every form the runs have shown.

The newest two runs (`--test-runs`) are held out and never trained on. Both matchers run through the
same matching code on them; the log thresholds are calibrated on a validation slice of the training runs,
allowing no more wrong matches than the old matcher made there. The training also fails if the numpy
forward pass differs from torch's by more than 1e-3. The result folder holds the export, `report.json`
(the measurements) and `train_state.pt`.

There is no separate measuring command: the measurement is part of a training run.

## Measured on the two held-out runs

Carried over from the previous version of this page: these figures come from the training run that
produced the shipped network, on runs recorded before the Champions Gym port. They have not been
measured again since.

| Held-out runs (77 battles) | Network | Old thumbnail matcher |
|---|---|---|
| Battle-log rows matched correctly (of 2,339) | 2,336 | 2,316 |
| Battle-log rows matched to the wrong species | 0 | 0 |
| Battle-log rows left unknown | 3 | 23 |
| Opponent species sent out that team preview had read (of 204) | 197 (96.6%) | 191 (93.6%) |
| Best-guess species, 600 log crops recoloured (hue, lightness, part, grey) | 100% each | 99.0-99.2% |

Recorded runs include shiny opponents (a white Corviknight, a recoloured Mega Lucario Z); an earlier
version of the network without the palette augmentations misread them.

## Adding a sprite

A new Pokémon or form needs a reference view in the local source indexes, then `pcsa train sprites
--install`. Forms that have appeared in recorded runs also get the centre of their real crops as a view.

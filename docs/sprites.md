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

## Training: `pcsa train sprites`

```
uv run pcsa train sprites                 # train, measure, write output/sprite_net/<stamp>/
uv run pcsa train sprites --install       # the same, then copy the result into assets/sprites
```

It needs the train extra (`pcsa setup --with train`) and two local inputs that are not in the repository:

- **Reference views**: `output/sprite_source/roster_index.*` and `log_index.*`, the silhouette indexes of
  the earlier thumbnail matcher (one view per sprite, from in-game Roster Info screenshots and the
  Champions menu sprites). They are game imagery, so they stay local.
- **Recorded runs**: `output/runs/`. A battle-log row is labelled by the species its own text names
  ("Move: The opposing Tauros used ..."); a team-preview card by the opponent later sending that species
  out (`showdown.log`). Rows the run saw no sprite on are left out.

Every batch mixes heavily augmented reference views with lightly augmented real crops. The augmentations
include palette swaps for shiny Pokémon (hue rotation, inverted lightness, colour faded to grey), so
colour helps a match but never outweighs the shape. After training, each reference view is embedded, plus
the centre of the real crops for every form the runs have shown.

The newest two runs are held out and never trained on. Both matchers run through the same matching
code on them; the log thresholds are calibrated on a validation slice of the training runs, allowing no
more wrong matches than the old matcher made there. The training also fails if the numpy forward pass
differs from torch's by more than 1e-3.

## Measured on the two held-out runs

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

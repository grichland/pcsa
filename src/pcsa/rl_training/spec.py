# From research/pcsa-rl/rl/spec.py; keep diffable.
"""Shapes and constants shared by every rl module."""

ROWS = 12
POKEMON_FLOATS = 318
POKEMON_IDS = 7
FIELD_FLOATS = 126
ACTIONS = 11
# embedding table sizes (index 0 = unknown/pad); mapping in data/vocab.json (built by research rl.features_vocab)
VOCAB = {"species": 1519, "item": 238, "ability": 318, "move": 955}

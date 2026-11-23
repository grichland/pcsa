# Development

```
uv sync --group dev --extra torch
uv run poe check        # ruff, pyright, import-linter, pytest
```

CI runs the same checks on Windows (with Champions Gym and torch) and on Ubuntu (without Champions Gym, whose
wheel is Windows x64 only; the tests that need it skip there).

## Tests

`pytest` leaves out two markers by default: `torch` (the sprite trainer's end-to-end run) and `ocr` (the real
OCR engine instead of cached boxes). Select them with `uv run pytest -m torch` or `-m ocr`. Tests that need local
data not in the repository (screenshots, recorded runs, a pcsa-former checkout next to this
one) skip when it is missing.

Tests build their settings from `config/agent.yaml` through `tests/configs.py`, so no value is repeated in a test.

## Rules the checks enforce

- Every setting's value is written in `config/agent.yaml` only; the settings classes in `common/config.py` and
  `common/config_cg.py` carry types and checks, no defaults (`tests/unit/test_config_source.py`). Measured pixel
  thresholds stay named constants next to the vision code that uses them.
- `time.*` only in `common/clock.py`, `Path(...).parents` only in `common/paths.py`, `print` only in `cli.py` and
  `common/setup.py`. Files stay under 400 lines and functions under 60 (`tests/unit/test_code_limits.py`); the
  network files copied from pcsa-former are exempt so they stay comparable with their source.
- The import rules in `.importlinter` (listed in `ARCHITECTURE.md`).
- `docs/configuration.md` is generated from `config/agent.yaml` and must be current.
- Lines are at most 118 characters (ruff). The repository is linted with `ruff check`, not reformatted.

## Tools

Each tool states its usage in its docstring and reads its settings from `config/agent.yaml` (`--set` to change
one).

| Tool | What it does |
|---|---|
| `tools/replay_run.py` | Replays a run's saved Champions Gym messages through the neural player, with no phone, and writes a copy of the run with fresh decision records; reports whether every answer matches the recorded one. |
| `tools/choice_replay.py` | Scores the choice rules on recorded turn decisions, offline. |
| `tools/value_calibration.py` | Fits `cg.value_calibration` for a checkpoint on recorded battles and reports its calibration before and after. |
| `tools/search_eval.py` | Plays the network with and without search in the simulator from the same seeds. |
| `tools/search_recorded.py` | Runs the search on recorded ladder turns and scores its picks against play-on results. |
| `tools/config_reference.py` | Writes `docs/configuration.md` from `config/agent.yaml`. |
| `tools/catalog/battle_log.py` | Measures every recorded battle-log row and writes `docs/reference/battle-log.md`. |
| `tools/data/cg_vocab.py`, `cg_tables.py`, `check_stopgap.py` | Write and check `assets/game/cg/` from a Champions Gym checkout. |
| `tools/forms_pack.py` | Writes pcsa-former's battle pack with opposing Pokémon named by form. |
| `scripts/curate_frames.py` | Builds the perception test corpus from saved run frames. |

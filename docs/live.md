# Live play (private)

This file and `src/pcsa/live/`, `tests/live/`, `scripts/run_agent.ps1` and `scripts/export_public.py` are
private: `scripts/export_public.py` leaves them out of the public repository.

## Running on the phone

With the phone on USB (USB debugging on) and the game open on the lobby:

```
uv run pcsa calibrate                # measures this phone once; writes config/devices/<model>.yaml
uv run pcsa run --player custom_heuristic --battles 10
uv run pcsa watch                    # read-only view of the latest run
uv run pcsa nav --to lobby           # walk the menus to the lobby
```

`--player` takes a player kind: `random`, `simple_heuristic`, `custom_heuristic` (default), `max_damage` or
`mixed` (`config/players/mixed.yaml`).

For longer evaluation sessions, `scripts\run_agent.ps1` holds the settings (player, seed, run name,
overrides) at the top of the file, stops any agent or dashboard left over from a run that ended abruptly,
syncs the environment, checks the phone, starts the run and opens the dashboard on it. With `$Mirror = $true`
(the default) it also opens a phone mirror window through
[scrcpy](https://github.com/Genymobile/scrcpy), which `pcsa setup` installs into `.tools\scrcpy\` (a `scrcpy`
on PATH works as a fallback); when it is missing the script prints a warning and runs without the mirror:

```
powershell -ExecutionPolicy Bypass -File scripts\run_agent.ps1 -Battles 25 -MaxBattlesTotal 30
```

## Publishing the public repository

```
uv run python scripts/export_public.py --to ..\pcsa            # copy the public files, then check them there
```

The export copies every tracked and new (not ignored) file except the private paths above, removes the
`live` layer and contract from `.importlinter` and the private files' lint exemptions from `pyproject.toml`,
refuses to finish if any exported file still imports `pcsa.live`, and with `--check` runs `uv sync` and the
full `poe check` inside the export. Commit and push from the export folder.

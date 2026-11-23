# Setup and commands

## Install

```
winget install astral-sh.uv                       # once per machine
git clone https://github.com/grichland/pcsa.git && cd pcsa
uv run pcsa setup --with torch                    # Python environment, Champions Gym, torch, adb, scrcpy, OCR models, output folder
```

`uv run pcsa doctor` repeats the checks without installing anything. Without `--with torch` everything works
except the neural player and the sprite trainer; `--set decide.player=first` then plays the first legal choice,
which is enough to test the phone side.

Tools are pinned in `tools.lock.json` and installed into `.tools/`. Champions Gym is pinned in `pyproject.toml`.

## Phone commands that send no input

With the phone on USB (USB debugging on) and the game open:

```
uv run pcsa check        # connect and name the current screen once
uv run pcsa calibrate    # measure this phone once; writes config/devices/<model>.yaml
uv run pcsa observe      # capture and classify the screen in a loop
uv run pcsa scout        # record a manual walk through the game and report what changed
```

## Live play

With the game open on the Ranked Singles lobby ("Begin Matchmaking"):

```
uv run pcsa run                                    # battles and player as config/agent.yaml sets them
uv run pcsa run --set run.battles=25               # one setting changed for this run
uv run pcsa run --resume output/runs/<stamp>-run   # continue a run
uv run pcsa nav --to lobby                         # walk the menus to the lobby
```

A run ends after `run.battles` battles with a decided outcome, or after `run.max_battles_total` battles started.
Every command that reads the configuration takes `--config <file>` and any number of `--set section.key=value`;
the environment variable `PCSA__section__key` does the same.

An action turn reads the battle log, asks the player and taps Fight and the move row. It opens the move list
first only when a turn needs it (a move nearly out of PP by the log's count, a log read that lost rows, taps that
did not register, an unconfirmed Mega Evolution); `timing.move_peek: always` looks before every turn.

## The session launcher

`scripts\run_agent.ps1` starts a longer session:

```
powershell -ExecutionPolicy Bypass -File scripts\run_agent.ps1
```

It stops any agent left over from a run that ended abruptly, syncs the environment, checks that exactly one phone
is connected with the game in front, runs `pcsa check`, opens a phone mirror through
[scrcpy](https://github.com/Genymobile/scrcpy) and the dashboard, and starts `pcsa run`. The settings block at the
top of the script chooses the config file, one-off overrides (`$Overrides`, passed as `--set`), the mirror, the
dashboard and a run to resume.

## Where things go

| What | Where |
|---|---|
| Everything the agent writes (runs, frames, sprite training) | `output/` (override with `PCSA_OUTPUT`) |
| One run: manifest, events, frames, and per battle the log, the LIVE events, the Champions Gym messages and one record per decision | `output/runs/<stamp>-<name>/` |
| Installed tools (adb, scrcpy, OCR models) | `.tools/` |
| Settings | `config/agent.yaml`, `config/screen.yaml`, `config/devices/*.yaml` |
| Network checkpoints | `assets/models/` |
| Game data, Champions Gym's tables and id numbering, the sprite network | `assets/` |
| Usage-ranked builds and the opposing builds met on the ladder (data only; nothing reads them at run time) | `assets/sets/` |

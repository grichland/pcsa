param(
    [int]$Battles = 100,
    [int]$MaxBattlesTotal = 112
)

# Launch the Ranked Singles agent with the settings below.
# Usage (from any folder):  powershell -ExecutionPolicy Bypass -File <repo>\scripts\run_agent.ps1 [-Battles 25 -MaxBattlesTotal 30]
# Before starting: phone on USB with debugging on, the game open on the Ranked Singles lobby ("Begin Matchmaking").
# Player settings live in config\players\ (heuristic.yaml, best_matchup.yaml, disciplined.yaml, mixed.yaml, neural.yaml); controller timings in config\agent.yaml.
# $Player may also be "endgame" (disciplined plus the 1v1 endgame search, config\players\endgame.yaml) or
# "neural" (a research network; needs torch: uv sync --inexact --extra train). Pick its network with
#   $Overrides = @("policy.neural_checkpoint=C:/path/to/ckpt.pt")   (else the checkpoint in config\players\neural.yaml)
# ============================== SETTINGS (edit these) ==============================
$Player     = "neural"                   # neural (config/players/neural.yaml) | random | simple_heuristic | custom_heuristic | max_damage | best_matchup_max_damage | disciplined | mixed | endgame
$Seed       = $null               # $null = new random seed; a number makes choices and taps reproducible
$RunName    = "run"               # run folder prefix under output\runs\
$Config     = "config/agent.yaml" # controller settings; player files are named inside it
$Overrides  = @()                 # one-off settings, e.g. @("timing.max_turn_s=35", "run.count_forfeits=true")
$Watch      = $true               # $true opens the read-only dashboard (pcsa watch) on this run in a second window
$Mirror     = $true               # open a phone mirror window (scrcpy) next to the run; skipped with a warning when scrcpy is missing
$Resume     = ""                  # an existing run folder to continue, or "" for a new run
# ===================================================================================

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

# Leftovers from a run that ended abruptly: a phone agent started from THIS repo would still hold pcsa.exe (so the
# environment cannot be synced). Training runs (pcsa train ...) are left alone. With $Watch on, a dashboard left
# over from an earlier launch is stopped too (it would keep port 8765); with it off, dashboards are left alone.
$repoPattern = [regex]::Escape($Repo + '\')  # the trailing separator leaves out sibling folders such as pcsa-public
$agentPattern = 'pcsa\.exe"? run\b|pcsa\.cli"? run\b'
if ($Watch) { $agentPattern = 'pcsa\.exe"? (run|watch)\b|pcsa\.cli"? (run|watch)\b' }
$all = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -match $repoPattern -and $_.ProcessId -ne $PID }
$leftovers = @($all | Where-Object { $_.CommandLine -match $agentPattern })
foreach ($p in $leftovers) {
    $shown = $p.CommandLine.Substring(0, [Math]::Min(100, $p.CommandLine.Length))
    Write-Host "Stopping leftover process $($p.ProcessId): $shown"
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
if ($leftovers.Count -gt 0) { Start-Sleep -Seconds 2 }

# Runner: sync the environment with uv when it is installed, then launch the project's own executable directly.
# (Launching through `uv run` is what fails with "Access is denied" when something still holds pcsa.exe.)
$Pcsa = Join-Path $Repo ".venv\Scripts\pcsa.exe"
$Uv = Get-Command uv -ErrorAction SilentlyContinue
if ($Uv) {
    & $Uv.Source sync --inexact --quiet   # keeps torch and other installed extras; live play needs none
    if ($LASTEXITCODE -ne 0) { Write-Host "uv sync failed; continuing with the existing environment" }
}
if (-not (Test-Path -LiteralPath $Pcsa)) { throw "pcsa.exe not found at $Pcsa. Run: winget install astral-sh.uv; uv run pcsa setup" }

# Phone: exactly one authorized device, with the game in front and no screen sharing stealing taps.
$Adb = Join-Path $Repo ".tools\adb\adb.exe"
if (-not (Test-Path -LiteralPath $Adb)) { throw "adb not found at $Adb. Run: uv run pcsa setup" }
$devices = & $Adb devices | Select-String "\tdevice$"
if (@($devices).Count -ne 1) { throw "Expected one authorized ADB device, found $(@($devices).Count). Connect and authorize the phone." }
$Serial = ($devices[0] -split "`t")[0]
Write-Host "Phone: $Serial"
$focus = (& $Adb shell dumpsys window | Select-String "mCurrentFocus=Window" | Select-Object -First 1) -as [string]
if ($focus -notmatch "pokemonchampions") { throw "The game is not in front ($focus). Open it on Begin Matchmaking." }
if ((& $Adb shell dumpsys window windows | Select-String "WakeLockScreenLayout")) {
    throw "Link to Windows screen sharing is blocking taps. Stop it (notification shade or Phone Link) and retry."
}


# Arguments for `pcsa run`.
$arguments = @("run", "--config", $Config, "--player", $Player, "--battles", $Battles, "--max-total", $MaxBattlesTotal, "--name", $RunName)
foreach ($setting in $Overrides) { $arguments += @("--set", $setting) }
if ($null -ne $Seed) { $arguments += @("--seed", $Seed) }
$resumeDir = if ($Resume) { (Resolve-Path $Resume).Path.TrimEnd('\') } else { $null }
if ($resumeDir) { $arguments += @("--resume", $resumeDir) }

if ($Mirror) {
    # Phone mirror: `pcsa setup` installs the pinned scrcpy release into .tools\scrcpy\ (scrcpy.exe next to its
    # scrcpy-server); a scrcpy on PATH is the fallback. It runs as its own process and never blocks or fails the run.
    $Scrcpy = Join-Path $Repo ".tools\scrcpy\scrcpy.exe"
    if (-not (Test-Path -LiteralPath $Scrcpy)) {
        $found = Get-Command scrcpy -ErrorAction SilentlyContinue
        if ($found) { $Scrcpy = $found.Source } else { $Scrcpy = $null }
    }
    $open = @(Get-Process -Name scrcpy -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -eq "pcsa phone" })
    if (-not $Scrcpy) {
        Write-Host "Mirror skipped: scrcpy not found (.tools\scrcpy\scrcpy.exe or on PATH). Run: uv run pcsa setup. Set `$Mirror = `$false to silence this."
    } elseif ($open.Count -gt 0) {
        Write-Host "Mirror already open (scrcpy process $($open[0].Id))"
    } else {
        # scrcpy ships its own adb; a different adb version would restart the ADB server under the agent.
        # ADB is scrcpy's environment variable for the adb executable to use, so it shares the repo's one.
        $env:ADB = $Adb
        # Start-Process joins -ArgumentList with spaces and does not quote, so the title carries its own quotes;
        # unquoted, scrcpy sees "phone" as a stray argument and exits at once.
        $mirrorArguments = @("--serial", $Serial, "--window-title", '"pcsa phone"', "--no-audio")
        Start-Process -FilePath $Scrcpy -ArgumentList $mirrorArguments -WorkingDirectory $Repo | Out-Null
        Write-Host "Mirror: $Scrcpy"
    }
}

$started = Get-Date
if ($Watch) {
    # A second window waits for this run's folder to appear (a resumed run's folder already exists), then opens
    # the dashboard on it. If port 8765 is still held by something else, the dashboard takes the next free port.
    $port = 8765
    while (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { $port++ }
    if ($port -ne 8765) { Write-Host "Port 8765 is busy; the dashboard will use port $port" }
    $runsDir = Join-Path $Repo "output\runs"
    $wanted = "`$_.CreationTime -gt [datetime]'$($started.ToString('o'))'"
    if ($resumeDir) {
        $runsDir = Split-Path -Parent $resumeDir
        $wanted = "`$_.FullName -eq '$($resumeDir.Replace("'", "''"))'"
    }
    $script = "while (`$true) { `$d = Get-ChildItem -LiteralPath '$runsDir' -Directory -ErrorAction SilentlyContinue | " +
              "Where-Object { $wanted -and (Test-Path (Join-Path `$_.FullName 'events.jsonl')) } | " +
              "Sort-Object CreationTime | Select-Object -Last 1; if (`$d) { & '$Pcsa' watch --port $port --run `"`$(`$d.FullName)`"; break }; Start-Sleep 2 }"
    Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $script) `
        -WorkingDirectory $Repo -WindowStyle Minimized | Out-Null
    Write-Host "Dashboard: http://127.0.0.1:$port/ (opens when the run folder appears)"
}

# Python logs to stderr; Windows PowerShell would treat each such line as an error.
$ErrorActionPreference = "Continue"
Write-Host "Starting: $Pcsa $($arguments -join ' ')"
& $Pcsa @arguments 2>&1 | ForEach-Object { "$_" } | Out-Host
$exitCode = $LASTEXITCODE
Write-Host ("Agent finished with exit code {0} after {1:N0} minutes. Runs: {2}" -f $exitCode, ((Get-Date) - $started).TotalMinutes, (Join-Path $Repo "output\runs"))
exit $exitCode

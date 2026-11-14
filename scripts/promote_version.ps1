# Promote a live fine-tune job's version (research/pcsa-rl rl2/scripts/live_finetune_job.py) to the live player.
#
#   scripts\promote_version.ps1 <job folder> [-NoCommit]
#
# Copies the job's model to assets/models/, its version.yaml to config/players/versions/neural-<stamp>.yaml (the
# history) and to config/players/neural.yaml (the live config), then commits the three files. Refuses while a live
# phone session or a training run is going (pcsa is an editable install). Pushing stays a separate step.
# Rolling back = promoting an older version: scripts\promote_version.ps1 -Version neural-<stamp>
param(
    [string]$Job = "",
    [string]$Version = "",
    [switch]$NoCommit
)
$ErrorActionPreference = "Stop"
$pcsa = Split-Path -Parent $PSScriptRoot
$live = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'pcsa(\.exe)?"? run |rl_training|live_finetune' }
if ($live) { throw "a live session or training/fine-tune run is going (pids $($live.ProcessId -join ', ')); promote when it is over" }
$versions = Join-Path $pcsa "config\players\versions"
New-Item -ItemType Directory -Force $versions | Out-Null
$target = Join-Path $pcsa "config\players\neural.yaml"

if ($Version) {
    $source = Join-Path $versions "$Version.yaml"
    if (-not (Test-Path $source)) { throw "no version $source" }
    $text = Get-Content $source -Raw
    $model = [regex]::Match($text, '(?m)^checkpoint:\s*(\S+)').Groups[1].Value
    if (-not (Test-Path (Join-Path $pcsa $model))) { throw "the version's model $model is not in the repository" }
    Copy-Item $source $target -Force
    $files = @($target)
    $what = "Live player: switch back to version $Version"
} else {
    if (-not $Job -or -not (Test-Path (Join-Path $Job "version.yaml"))) { throw "give a job folder that holds version.yaml" }
    $state = Get-Content (Join-Path $Job "job.json") -Raw | ConvertFrom-Json
    if (-not $state.finished) { throw "that job did not finish: $($state.error)" }
    if ($state.args.smoke) { throw "that is a smoke-test job; promote a real one" }
    $text = Get-Content (Join-Path $Job "version.yaml") -Raw
    $model = [regex]::Match($text, '(?m)^checkpoint:\s*(\S+)').Groups[1].Value
    $name = Split-Path $model -Leaf
    $stamp = [regex]::Match($name, 'live-(\d{8}-\d{4})').Groups[1].Value
    Copy-Item (Join-Path $Job $name) (Join-Path $pcsa $model) -Force
    $versionFile = Join-Path $versions "neural-$stamp.yaml"
    Copy-Item (Join-Path $Job "version.yaml") $versionFile -Force
    Copy-Item (Join-Path $Job "version.yaml") $target -Force
    $files = @((Join-Path $pcsa $model), $versionFile, $target)
    $what = "Live player: promote version neural-$stamp (job $(Split-Path $Job -Leaf))"
}
Write-Output "copied: $($files -join ', ')"
if (-not $NoCommit) {
    git -C $pcsa add -- $files
    git -C $pcsa commit -m "$what`n`nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- $files
    Write-Output "committed (not pushed)"
}

# Go back to an earlier live-player version (config/players/versions/neural-<stamp>.yaml and its model).
#
#   scripts\rollback_version.ps1            back one version: the one before the live one
#   scripts\rollback_version.ps1 -List      every version, the live one marked
#   scripts\rollback_version.ps1 -To neural-20261113-1215
#
# The live config's first line names its version (promote_version.ps1 writes it); a live config without one is the
# first version. The rollback is promote_version.ps1 -Version (it refuses during a live session, commits, never pushes).
# Nothing is deleted: rolling forward again is another -To.
param([switch]$List, [string]$To = "")
$ErrorActionPreference = "Stop"
$pcsa = Split-Path -Parent $PSScriptRoot
$versions = Get-ChildItem (Join-Path $pcsa "config\players\versions") -Filter "neural-*.yaml" | Sort-Object Name
if (-not $versions) { throw "no versions in config\players\versions" }
$first = Get-Content (Join-Path $pcsa "config\players\neural.yaml") -TotalCount 1
$m = [regex]::Match($first, '^# version (neural-\d{8}-\d{4})')
$current = if ($m.Success) { $m.Groups[1].Value } else { $versions[0].BaseName }

if ($List) {
    foreach ($v in $versions) {
        $model = [regex]::Match((Get-Content $v.FullName -Raw), '(?m)^checkpoint:\s*(\S+)').Groups[1].Value
        $mark = if ($v.BaseName -eq $current) { "* live" } else { "      " }
        Write-Output ("{0} {1}  {2}" -f $mark, $v.BaseName, $model)
    }
    return
}
if (-not $To) {
    $names = @($versions | ForEach-Object { $_.BaseName })
    $at = [array]::IndexOf($names, $current)
    if ($at -le 0) { throw "$current is the oldest version; nothing to roll back to (-List shows them)" }
    $To = $names[$at - 1]
}
if ($To -eq $current) { throw "$To is already live" }
Write-Output "live: $current -> $To"
& (Join-Path $PSScriptRoot "promote_version.ps1") -Version $To

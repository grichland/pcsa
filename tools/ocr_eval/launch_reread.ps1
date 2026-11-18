# Start the frame re-read in visible windows, one log per worker under docs/investigation/ocr-fix/logs.
param([int]$Workers = 16, [int]$Threads = 2)
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$py = Join-Path $repo '.venv\Scripts\python.exe'
$logs = Join-Path $repo 'docs\investigation\ocr-fix\logs'
for ($i = 0; $i -lt $Workers; $i++) {
    $c = "`$host.UI.RawUI.WindowTitle='ocr-eval reread $i'; Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue; `$env:PYTHONUTF8='1'; `$env:PYTHONPATH='$repo;$repo\src'; Set-Location '$repo'; & '$py' -m tools.ocr_eval.reread $i $Workers $Threads 2>&1 | Tee-Object -FilePath '$logs\reread.$i.log'"
    $p = Start-Process powershell -ArgumentList '-NoProfile', '-Command', $c -PassThru
    $p.PriorityClass = 'BelowNormal'
}

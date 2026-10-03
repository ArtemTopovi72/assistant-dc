# One-command setup for Windows:  powershell -ExecutionPolicy Bypass -File setup.ps1
# Options: -no-models -no-start -cpu -dev (passed through to scripts\setup.py as --no-models ...)
# Safe to run again: every step skips what is already done.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$uvDirs = @("$env:USERPROFILE\.local\bin", "$env:USERPROFILE\.cargo\bin")
foreach ($d in $uvDirs) { if ((Test-Path $d) -and ($env:Path -notlike "*$d*")) { $env:Path = "$d;$env:Path" } }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host '== installing uv (Python package manager)'
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    foreach ($d in $uvDirs) { if ((Test-Path $d) -and ($env:Path -notlike "*$d*")) { $env:Path = "$d;$env:Path" } }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv did not install; see https://docs.astral.sh/uv/' }
}

if (-not (Test-Path 'venv\Scripts\python.exe')) {
    Write-Host '== creating venv (Python 3.13)'
    uv venv -p 3.13 venv
    if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' }
}

# PowerShell-style switches (-no-models) become setup.py flags (--no-models).
$pass = foreach ($a in $args) { if ($a -match '^-[a-z]') { "-$a" -replace '^---', '--' } else { $a } }
& 'venv\Scripts\python.exe' 'scripts\setup.py' @pass
exit $LASTEXITCODE

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Python = Join-Path $Backend ".venv/Scripts/python.exe"
if (-not (Test-Path $Python)) { $Python = Join-Path $Backend ".venv/bin/python" }
if (-not (Test-Path $Python)) { throw "backend/.venv is missing. Run scripts/setup.ps1 first." }
Push-Location $Backend
try { & $Python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 }
finally { Pop-Location }

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Venv = Join-Path $Backend ".venv"

if (-not (Test-Path $Venv)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $Venv
    }
    else {
        & python -m venv $Venv
    }
}

$Python = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = Join-Path $Venv "bin/python"
}
if (-not (Test-Path $Python)) {
    throw "Python virtual environment was not created successfully."
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $Backend "requirements-dev.txt")

if (-not (Test-Path (Join-Path $Backend ".env"))) {
    Copy-Item (Join-Path $Backend ".env.example") (Join-Path $Backend ".env")
}

Push-Location $Root
try {
    & npm ci
    if (-not (Test-Path ".env.local")) {
        Copy-Item ".env.example" ".env.local"
    }
}
finally {
    Pop-Location
}

$OllamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
$OllamaExe = if ($OllamaCommand) {
    $OllamaCommand.Source
}
else {
    "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
}

if (Test-Path $OllamaExe) {
    $Models = (& $OllamaExe list 2>$null | Out-String)
    foreach ($Model in @("bge-m3", "qwen3:4b")) {
        if ($Models -notmatch [regex]::Escape($Model)) {
            Write-Warning "Missing Ollama model '$Model'. Run: ollama pull $Model"
        }
    }
}
else {
    Write-Warning "Ollama was not found. Install it from https://ollama.com/download/windows"
}

Write-Host "Setup complete. Pull bge-m3 and qwen3:4b, then press F5 in VS Code."

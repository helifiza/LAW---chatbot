param(
    [string]$EmbeddingModel = "bge-m3",
    [string]$GenerationModel = "qwen3:4b"
)

$ErrorActionPreference = "Stop"

function Resolve-OllamaExecutable {
    $Command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }

    $Candidates = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "$env:ProgramFiles\Ollama\ollama.exe"
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate) {
            return $Candidate
        }
    }
    throw "Ollama was not found. Install it from https://ollama.com/download/windows"
}

function Test-ModelAvailable {
    param(
        [string[]]$InstalledModels,
        [string]$RequiredModel
    )
    if ($InstalledModels -contains $RequiredModel) {
        return $true
    }
    if (($RequiredModel -notmatch ":") -and ($InstalledModels -contains "${RequiredModel}:latest")) {
        return $true
    }
    return $false
}

$OllamaExe = Resolve-OllamaExecutable

try {
    $Tags = Invoke-RestMethod `
        -Method Get `
        -Uri "http://127.0.0.1:11434/api/tags" `
        -TimeoutSec 5
}
catch {
    throw "Cannot connect to Ollama at http://127.0.0.1:11434. Open Ollama or run: ollama serve"
}

$Installed = @($Tags.models | ForEach-Object { $_.name })
foreach ($Model in @($EmbeddingModel, $GenerationModel)) {
    if (-not (Test-ModelAvailable -InstalledModels $Installed -RequiredModel $Model)) {
        throw "Ollama model '$Model' is missing. Run: & '$OllamaExe' pull $Model"
    }
}

$EmbedBody = @{
    model = $EmbeddingModel
    input = "Vietnamese embedding test"
} | ConvertTo-Json
$EmbedResult = Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:11434/api/embed" `
    -ContentType "application/json" `
    -Body $EmbedBody `
    -TimeoutSec 120

if (-not $EmbedResult.embeddings -or $EmbedResult.embeddings.Count -eq 0) {
    throw "Embedding model did not return a vector."
}

$ChatBody = @{
    model = $GenerationModel
    messages = @(
        @{ role = "user"; content = "Reply with exactly OK" }
    )
    stream = $false
    think = $false
    options = @{
        temperature = 0
        num_predict = 16
    }
} | ConvertTo-Json -Depth 6
$ChatResult = Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:11434/api/chat" `
    -ContentType "application/json" `
    -Body $ChatBody `
    -TimeoutSec 300

if (-not $ChatResult.message -or -not $ChatResult.message.content) {
    throw "Generation model did not return chat content."
}

$Dimensions = $EmbedResult.embeddings[0].Count
Write-Host "OK: Ollama is ready. $EmbeddingModel -> $Dimensions dimensions; $GenerationModel -> chat response received."

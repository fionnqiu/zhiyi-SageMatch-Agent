# Start the private loopback API only after the current real RAG gate passes.
param(
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [string]$EvalDir = 'data/eval',
    [string]$Python = '',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$backendDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $backendDir '..'))
$pythonPath = if ($Python) { $Python } else { Join-Path $repoRoot '.venv/Scripts/python.exe' }

# The gate checks the same PostgreSQL ready index the API will use. A failed
# or stale private report must stop startup before any durable worker claims jobs.
& (Join-Path $PSScriptRoot 'verify_private_rag_gate.ps1') -EvalDir $EvalDir -Python $pythonPath
if ($CheckOnly) {
    return
}

Push-Location $backendDir
try {
    & $pythonPath run.py --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) {
        throw "Private API exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

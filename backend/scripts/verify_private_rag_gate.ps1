# Verify a private, reviewed RAG run without copying its material into Git.
param(
    [string]$EvalDir = 'data/eval',
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
$backendDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $backendDir '..'))
$evalPath = if ([System.IO.Path]::IsPathRooted($EvalDir)) {
    [System.IO.Path]::GetFullPath($EvalDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $EvalDir))
}
$pythonPath = if ($Python) { $Python } else { Join-Path $repoRoot '.venv/Scripts/python.exe' }

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python executable is missing: $pythonPath"
}

$dataset = Join-Path $evalPath 'reviewed-cases.json'
$report = Join-Path $evalPath 'rag-run-final.json'
$review = Join-Path $evalPath 'claim-reviews.json'
$reviewed = Join-Path $evalPath 'rag-reviewed-final.json'
foreach ($file in @($dataset, $report, $review, $reviewed)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Private RAG gate input is missing: $file"
    }
}

# The verifier checks the live ready-index fingerprint and recomputes every
# reviewed metric. No provider request or report write occurs in this script.
Push-Location $backendDir
try {
    & $pythonPath -m app.services.materials.rag.live_eval `
        --dataset $dataset --report $report --review $review --verify-reviewed $reviewed
    if ($LASTEXITCODE -ne 0) {
        throw "Private RAG quality gate failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

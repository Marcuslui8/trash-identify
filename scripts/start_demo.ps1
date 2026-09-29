param(
    [string]$HostAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Missing .venv. Run scripts\setup_demo.ps1 first."
}
foreach ($modelName in @("waste_detector.pt", "waste_classifier.pt")) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot "models\$modelName") -PathType Leaf)) {
        throw "Missing models\$modelName. Run scripts\setup_demo.ps1 first."
    }
}

Write-Host "Scanner: http://${HostAddress}:$Port/scanner" -ForegroundColor Green
Write-Host "Health : http://${HostAddress}:$Port/health" -ForegroundColor Green
Write-Host "Keep this window open. Press Ctrl+C to stop." -ForegroundColor Yellow
Push-Location $projectRoot
try {
    & $python -m uvicorn waste_system.api.main:app --host $HostAddress --port $Port --env-file .env
} finally {
    Pop-Location
}

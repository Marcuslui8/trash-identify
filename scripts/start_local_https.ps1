param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8443
)

$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$certificateDirectory = Join-Path $projectRoot "certs"
$serverKey = Join-Path $certificateDirectory "server.key"
$serverCertificate = Join-Path $certificateDirectory "server.crt"
$ipFile = Join-Path $certificateDirectory "ip-address.txt"
$environmentFile = Join-Path $projectRoot ".env"

foreach ($requiredFile in @($python, $serverKey, $serverCertificate, $ipFile)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Missing $requiredFile. Run scripts\setup_local_https.ps1 first."
    }
}

$ipAddress = (Get-Content -LiteralPath $ipFile -Raw).Trim()
Write-Host "HTTPS scanner: https://${ipAddress}:$Port/scanner" -ForegroundColor Green
Write-Host "Keep this window open. Press Ctrl+C to stop the service."

$arguments = @(
    "-m", "uvicorn", "waste_system.api.main:app",
    "--host", "0.0.0.0", "--port", $Port,
    "--ssl-keyfile", $serverKey,
    "--ssl-certfile", $serverCertificate
)
if (Test-Path -LiteralPath $environmentFile) {
    $arguments += @("--env-file", $environmentFile)
}

Push-Location $projectRoot
try {
    & $python @arguments
} finally {
    Pop-Location
}

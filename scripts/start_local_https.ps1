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

$portInUse = $false
$tcpClient = New-Object System.Net.Sockets.TcpClient
try {
    $connectTask = $tcpClient.ConnectAsync("127.0.0.1", $Port)
    $portInUse = $connectTask.Wait(750) -and $tcpClient.Connected
} catch {
    $portInUse = $false
} finally {
    $tcpClient.Dispose()
}

if ($portInUse) {
    $healthResponse = $null
    try {
        $healthJson = & curl.exe --silent --show-error --insecure --max-time 3 `
            "https://127.0.0.1:$Port/health" 2>$null
        if ($LASTEXITCODE -eq 0 -and $healthJson) {
            $healthResponse = $healthJson | ConvertFrom-Json
        }
    } catch {
        $healthResponse = $null
    }

    if ($null -ne $healthResponse -and $healthResponse.status -eq "ok") {
        Write-Host "Waste identification HTTPS service is already running on port $Port." -ForegroundColor Yellow
        Write-Host "HTTPS scanner: https://${ipAddress}:$Port/scanner" -ForegroundColor Green
        Write-Host "HTTPS collector: https://${ipAddress}:$Port/collector" -ForegroundColor Green
        Write-Host "Health: version $($healthResponse.version), detector=$($healthResponse.detector_available), classifier=$($healthResponse.classifier_available)"
        exit 0
    }

    throw "Port $Port is already used by another program. Run: netstat -ano | findstr :$Port"
}

Write-Host "HTTPS scanner: https://${ipAddress}:$Port/scanner" -ForegroundColor Green
Write-Host "HTTPS collector: https://${ipAddress}:$Port/collector" -ForegroundColor Green
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

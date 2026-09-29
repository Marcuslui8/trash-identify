param(
    [string]$Repository = "Marcuslui8/trash-identify",
    [string]$ReleaseTag = "v0.9.0",
    [string]$ExpectedSha256 = "",
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
$modelDirectory = Join-Path $projectRoot "models"
$archiveName = "waste-models-$ReleaseTag.zip"
$archiveUrl = "https://github.com/$Repository/releases/download/$ReleaseTag/$archiveName"
$checksumName = "waste-models-$ReleaseTag.sha256"
$checksumUrl = "https://github.com/$Repository/releases/download/$ReleaseTag/$checksumName"
$downloadDirectory = Join-Path $projectRoot ".tmp\demo-models"
$archivePath = Join-Path $downloadDirectory $archiveName
$checksumPath = Join-Path $downloadDirectory $checksumName

function Invoke-ProjectPython([string[]]$Arguments) {
    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}

 $venvUsable = $false
if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $venvUsable = $true
    try {
        & $venvPython --version *> $null
        if (-not $?) { $venvUsable = $false }
    } catch {
        $venvUsable = $false
    }
    $pythonVersionInfo = (Get-Item -LiteralPath $venvPython).VersionInfo
    if ($pythonVersionInfo.OriginalFilename -eq "py.exe") { $venvUsable = $false }
    $venvConfig = Join-Path $projectRoot ".venv\pyvenv.cfg"
    if (Test-Path -LiteralPath $venvConfig -PathType Leaf) {
        $homeLine = Select-String -LiteralPath $venvConfig -Pattern '^home\s*=\s*(.+)$' | Select-Object -First 1
        if ($null -ne $homeLine) {
            $basePython = Join-Path $homeLine.Matches[0].Groups[1].Value.Trim() "python.exe"
            if (-not (Test-Path -LiteralPath $basePython -PathType Leaf)) { $venvUsable = $false }
        }
    }
}
if (-not $venvUsable) {
    if (Test-Path -LiteralPath (Join-Path $projectRoot ".venv") -PathType Container) {
        Write-Host "Existing .venv is not usable; recreating it." -ForegroundColor Yellow
        Remove-Item -LiteralPath (Join-Path $projectRoot ".venv") -Recurse -Force
    }
    $pythonCommand = Get-Command python -CommandType Application -ErrorAction SilentlyContinue |
        Where-Object { (Get-Item -LiteralPath $_.Source).VersionInfo.OriginalFilename -ne "py.exe" } |
        Select-Object -First 1
    if ($null -ne $pythonCommand) {
        & $pythonCommand.Source -m venv --copies (Join-Path $projectRoot ".venv")
    } else {
        $pyCommand = Get-Command py -ErrorAction SilentlyContinue
        if ($null -eq $pyCommand) { throw "Python 3.11+ is required but was not found on PATH." }
        & $pyCommand.Source -3 -m venv --copies (Join-Path $projectRoot ".venv")
    }
    if ($LASTEXITCODE -ne 0) { throw "Unable to create .venv" }
}

Write-Host "Installing project dependencies..." -ForegroundColor Cyan
Invoke-ProjectPython @("-m", "pip", "install", "-e", ".[detection]")

$environmentFile = Join-Path $projectRoot ".env"
if (-not (Test-Path -LiteralPath $environmentFile -PathType Leaf)) {
    Copy-Item -LiteralPath (Join-Path $projectRoot ".env.example") -Destination $environmentFile
}

if (-not $SkipModelDownload) {
    New-Item -ItemType Directory -Force -Path $downloadDirectory | Out-Null
    Write-Host "Downloading model release $ReleaseTag..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $archiveUrl -OutFile $archivePath -UseBasicParsing
    if (-not $ExpectedSha256) {
        Invoke-WebRequest -Uri $checksumUrl -OutFile $checksumPath -UseBasicParsing
        $ExpectedSha256 = ((Get-Content -LiteralPath $checksumPath -Raw) -split "\s+")[0].Trim()
    }
    if (-not $ExpectedSha256 -or $ExpectedSha256.Length -ne 64) {
        throw "A valid SHA-256 checksum is required for the model release."
    }
    $actualSha256 = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $ExpectedSha256.ToLowerInvariant()) {
        throw "Model checksum mismatch. Expected $ExpectedSha256, received $actualSha256"
    }
    Write-Host "Model checksum verified." -ForegroundColor Green
    $extractDirectory = Join-Path $downloadDirectory "extracted"
    Remove-Item -LiteralPath $extractDirectory -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -LiteralPath $archivePath -DestinationPath $extractDirectory -Force
    foreach ($modelName in @("waste_detector.pt", "waste_classifier.pt")) {
        $source = Join-Path $extractDirectory $modelName
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Release is missing $modelName" }
        New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null
        Copy-Item -LiteralPath $source -Destination (Join-Path $modelDirectory $modelName) -Force
    }
}

foreach ($modelName in @("waste_detector.pt", "waste_classifier.pt")) {
    if (-not (Test-Path -LiteralPath (Join-Path $modelDirectory $modelName) -PathType Leaf)) {
        throw "Missing $modelName. Run setup_demo.ps1 again or provide the model release assets."
    }
}

Write-Host "Demo setup complete." -ForegroundColor Green
Write-Host "Start with: powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1" -ForegroundColor Yellow

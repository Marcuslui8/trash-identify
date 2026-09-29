param(
    [string]$DetectorPath = "runs\detection\phone_s640_mju_w0\weights\best.pt",
    [string]$ClassifierPath = "models\waste_classifier.pt",
    [string]$OutputDirectory = "release-assets",
    [string]$Tag = "v0.9.0"
)

$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$detector = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $DetectorPath))
$classifier = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $ClassifierPath))
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))

foreach ($path in @($detector, $classifier)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing model file: $path"
    }
}

$stage = Join-Path $outputRoot "model-stage"
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Remove-Item -LiteralPath (Join-Path $stage "*") -Force -Recurse -ErrorAction SilentlyContinue
Copy-Item -LiteralPath $detector -Destination (Join-Path $stage "waste_detector.pt") -Force
Copy-Item -LiteralPath $classifier -Destination (Join-Path $stage "waste_classifier.pt") -Force

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$archive = Join-Path $outputRoot "waste-models-$Tag.zip"
Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $archive -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $(Split-Path $archive -Leaf)" | Set-Content -LiteralPath (Join-Path $outputRoot "waste-models-$Tag.sha256") -Encoding ascii

Write-Host "Created: $archive" -ForegroundColor Green
Write-Host "SHA256 : $hash" -ForegroundColor Green
Write-Host "Upload the ZIP and SHA256 file to the GitHub Release $Tag." -ForegroundColor Yellow

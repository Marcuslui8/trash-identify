param(
    [Parameter(Mandatory = $true)]
    [string]$IpAddress
)

$ErrorActionPreference = "Stop"
$parsedAddress = $null
if (-not [System.Net.IPAddress]::TryParse($IpAddress, [ref]$parsedAddress) -or
    $parsedAddress.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) {
    throw "Provide a valid IPv4 address, for example: 10.131.28.216"
}

$opensslCommand = Get-Command openssl -ErrorAction SilentlyContinue
if (-not $opensslCommand) {
    throw "OpenSSL was not found. Add the Anaconda or Git OpenSSL directory to PATH."
}

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$certificateDirectory = Join-Path $projectRoot "certs"
$configPath = Join-Path $certificateDirectory "server-openssl.cnf"
$caKey = Join-Path $certificateDirectory "trash-identify-ca.key"
$caCertificate = Join-Path $certificateDirectory "trash-identify-ca.crt"
$serverKey = Join-Path $certificateDirectory "server.key"
$serverRequest = Join-Path $certificateDirectory "server.csr"
$serverCertificate = Join-Path $certificateDirectory "server.crt"
$serialFile = Join-Path $certificateDirectory "trash-identify-ca.srl"
$ipFile = Join-Path $certificateDirectory "ip-address.txt"

New-Item -ItemType Directory -Force -Path $certificateDirectory | Out-Null

$config = @"
[req]
prompt = no
distinguished_name = distinguished_name
req_extensions = server_extensions

[distinguished_name]
CN = $IpAddress

[server_extensions]
subjectAltName = @alternative_names
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth

[ca_extensions]
basicConstraints = critical, CA:TRUE, pathlen:0
keyUsage = critical, keyCertSign, cRLSign
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always, issuer

[alternative_names]
IP.1 = $IpAddress
IP.2 = 127.0.0.1
DNS.1 = localhost
"@
Set-Content -LiteralPath $configPath -Value $config -Encoding ascii

if (-not (Test-Path -LiteralPath $caKey) -or -not (Test-Path -LiteralPath $caCertificate)) {
    & $opensslCommand.Source genrsa -out $caKey 4096
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the local CA key." }
    & $opensslCommand.Source req -x509 -new -sha256 -days 3650 -key $caKey -out $caCertificate -subj "/CN=Trash Identify Local CA" -config $configPath -extensions ca_extensions
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the local CA certificate." }
}

& $opensslCommand.Source genrsa -out $serverKey 2048
if ($LASTEXITCODE -ne 0) { throw "Failed to create the HTTPS server key." }
& $opensslCommand.Source req -new -key $serverKey -out $serverRequest -config $configPath
if ($LASTEXITCODE -ne 0) { throw "Failed to create the HTTPS certificate request." }

$signArguments = @(
    "x509", "-req", "-in", $serverRequest,
    "-CA", $caCertificate, "-CAkey", $caKey,
    "-out", $serverCertificate, "-days", "825", "-sha256",
    "-extfile", $configPath, "-extensions", "server_extensions"
)
if (Test-Path -LiteralPath $serialFile) {
    $signArguments += @("-CAserial", $serialFile)
} else {
    $signArguments += "-CAcreateserial"
}
& $opensslCommand.Source @signArguments
if ($LASTEXITCODE -ne 0) { throw "Failed to sign the HTTPS certificate." }

Set-Content -LiteralPath $ipFile -Value $IpAddress -Encoding ascii
Remove-Item -LiteralPath $serverRequest, $configPath -Force

Write-Host "Local HTTPS certificates created." -ForegroundColor Green
Write-Host "Install and trust this certificate on the phone: $caCertificate" -ForegroundColor Yellow
Write-Host "After installation, open: https://${IpAddress}:8443/scanner"
Write-Host "Data collector: https://${IpAddress}:8443/collector"
Write-Host "Keep trash-identify-ca.key private. Never copy it to the phone." -ForegroundColor Yellow

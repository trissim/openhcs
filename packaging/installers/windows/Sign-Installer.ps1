[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ArtifactPath,
    [string]$TimestampUrl = "http://timestamp.digicert.com"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RequiredEnvironmentValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Required Windows installer signing secret is missing: $Name"
    }
    return $value
}

function Resolve-SignToolPath {
    $command = Get-Command "signtool.exe" -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    $programFilesX86 = [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
    if (-not [string]::IsNullOrWhiteSpace($programFilesX86)) {
        $windowsKitsRoot = [IO.Path]::Combine(
            $programFilesX86,
            "Windows Kits",
            "10",
            "bin"
        )
        $candidatePattern = [IO.Path]::Combine(
            $windowsKitsRoot,
            "*",
            "x64",
            "signtool.exe"
        )
        $candidates = @(
            Get-ChildItem -Path $candidatePattern -File `
                -ErrorAction SilentlyContinue |
                Sort-Object -Property @{
                    Expression = {
                        [Version]$_.Directory.Parent.Name
                    }
                } -Descending
        )
        if ($candidates.Count -gt 0) {
            return $candidates[0].FullName
        }
    }

    throw "SignTool.exe was not found in PATH or the Windows 10 SDK."
}

$resolvedArtifact = [IO.Path]::GetFullPath($ArtifactPath)
if (-not (Test-Path -LiteralPath $resolvedArtifact -PathType Leaf)) {
    throw "Windows installer artifact not found: $resolvedArtifact"
}

$parsedTimestampUrl = $null
if (
    -not [Uri]::TryCreate(
        $TimestampUrl,
        [UriKind]::Absolute,
        [ref]$parsedTimestampUrl
    ) -or
    $parsedTimestampUrl.Scheme -notin @(
        [Uri]::UriSchemeHttp,
        [Uri]::UriSchemeHttps
    )
) {
    throw "TimestampUrl must be an absolute HTTP or HTTPS URL."
}

$certificateBase64 = Get-RequiredEnvironmentValue `
    "OPENHCS_WINDOWS_SIGNING_CERTIFICATE_BASE64"
$certificatePassword = Get-RequiredEnvironmentValue `
    "OPENHCS_WINDOWS_SIGNING_CERTIFICATE_PASSWORD"
$temporaryCertificatePath = [IO.Path]::Combine(
    [IO.Path]::GetTempPath(),
    "openhcs-authenticode-$([Guid]::NewGuid().ToString('N')).pfx"
)

try {
    try {
        $certificateBytes = [Convert]::FromBase64String($certificateBase64)
    }
    catch [FormatException] {
        throw "OPENHCS_WINDOWS_SIGNING_CERTIFICATE_BASE64 is not valid base64."
    }
    [IO.File]::WriteAllBytes($temporaryCertificatePath, $certificateBytes)
    $certificateBytes = $null
    $certificateBase64 = $null

    $signToolPath = Resolve-SignToolPath
    $signArguments = @(
        "sign",
        "/fd",
        "SHA256",
        "/tr",
        $parsedTimestampUrl.AbsoluteUri,
        "/td",
        "SHA256",
        "/f",
        $temporaryCertificatePath,
        "/p",
        $certificatePassword,
        $resolvedArtifact
    )
    & $signToolPath @signArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Authenticode signing failed with exit code $LASTEXITCODE."
    }

    & $signToolPath "verify" "/pa" "/tw" "/v" $resolvedArtifact
    if ($LASTEXITCODE -ne 0) {
        throw "Authenticode verification failed with exit code $LASTEXITCODE."
    }

    $authenticodeSignature = Get-AuthenticodeSignature `
        -LiteralPath $resolvedArtifact
    if (
        $authenticodeSignature.Status -ne
        [System.Management.Automation.SignatureStatus]::Valid
    ) {
        throw (
            "Authenticode status is not valid: " +
            $authenticodeSignature.StatusMessage
        )
    }
    if ($null -eq $authenticodeSignature.SignerCertificate) {
        throw "Authenticode verification returned no signer certificate."
    }
    if ($null -eq $authenticodeSignature.TimeStamperCertificate) {
        throw "The Authenticode signature has no timestamp certificate."
    }
}
finally {
    $certificatePassword = $null
    Remove-Item Env:OPENHCS_WINDOWS_SIGNING_CERTIFICATE_BASE64 `
        -ErrorAction SilentlyContinue
    Remove-Item Env:OPENHCS_WINDOWS_SIGNING_CERTIFICATE_PASSWORD `
        -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $temporaryCertificatePath) {
        Remove-Item -LiteralPath $temporaryCertificatePath -Force `
            -ErrorAction SilentlyContinue
    }
}

Write-Host "Signed and verified $resolvedArtifact"

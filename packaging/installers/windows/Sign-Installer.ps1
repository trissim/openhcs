[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ArtifactPath,
    [string]$CertificateThumbprint,
    [string]$TimestampUrl = "http://time.certum.pl"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Normalize-CertificateThumbprint {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Thumbprint
    )

    $normalized = ($Thumbprint -replace "[\s:]", "").ToUpperInvariant()
    if ($normalized -notmatch "^[0-9A-F]{40}$") {
        throw (
            "Certificate thumbprint must contain exactly 40 hexadecimal " +
            "characters."
        )
    }
    return $normalized
}

function Get-CodeSigningCertificate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Thumbprint
    )

    $certificatePath = "Cert:\CurrentUser\My\$Thumbprint"
    $certificate = Get-Item -LiteralPath $certificatePath `
        -ErrorAction SilentlyContinue
    if ($null -eq $certificate) {
        throw (
            "Code-signing certificate is absent from CurrentUser\\My: " +
            $Thumbprint
        )
    }
    if (-not $certificate.HasPrivateKey) {
        throw (
            "The selected certificate has no accessible private key. " +
            "Connect SimplySign or the certificate's hardware provider."
        )
    }

    $utcNow = [DateTime]::UtcNow
    if ($certificate.NotBefore.ToUniversalTime() -gt $utcNow) {
        throw "The selected code-signing certificate is not valid yet."
    }
    if ($certificate.NotAfter.ToUniversalTime() -le $utcNow) {
        throw "The selected code-signing certificate has expired."
    }

    $codeSigningEkuOid = "1.3.6.1.5.5.7.3.3"
    $hasCodeSigningEku = $false
    foreach ($extension in $certificate.Extensions) {
        if (
            $extension -is
            [System.Security.Cryptography.X509Certificates.X509EnhancedKeyUsageExtension]
        ) {
            foreach ($usage in $extension.EnhancedKeyUsages) {
                if ($usage.Value -eq $codeSigningEkuOid) {
                    $hasCodeSigningEku = $true
                }
            }
        }
    }
    if (-not $hasCodeSigningEku) {
        throw "The selected certificate does not permit code signing."
    }

    return $certificate
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

$configuredThumbprint = $CertificateThumbprint
if ([string]::IsNullOrWhiteSpace($configuredThumbprint)) {
    $configuredThumbprint = [Environment]::GetEnvironmentVariable(
        "OPENHCS_WINDOWS_SIGNING_CERTIFICATE_THUMBPRINT"
    )
}
if ([string]::IsNullOrWhiteSpace($configuredThumbprint)) {
    throw (
        "Set CertificateThumbprint or " +
        "OPENHCS_WINDOWS_SIGNING_CERTIFICATE_THUMBPRINT."
    )
}
$normalizedThumbprint = Normalize-CertificateThumbprint $configuredThumbprint
$null = Get-CodeSigningCertificate $normalizedThumbprint

$signToolPath = Resolve-SignToolPath
$signArguments = @(
    "sign",
    "/sha1",
    $normalizedThumbprint,
    "/s",
    "My",
    "/tr",
    $parsedTimestampUrl.AbsoluteUri,
    "/td",
    "SHA256",
    "/fd",
    "SHA256",
    "/v",
    $resolvedArtifact
)
& $signToolPath @signArguments
if ($LASTEXITCODE -ne 0) {
    throw "Authenticode signing failed with exit code $LASTEXITCODE."
}

& $signToolPath "verify" "/pa" "/all" "/tw" "/v" $resolvedArtifact
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
$actualSignerThumbprint = Normalize-CertificateThumbprint `
    $authenticodeSignature.SignerCertificate.Thumbprint
if ($actualSignerThumbprint -ne $normalizedThumbprint) {
    throw (
        "Authenticode signer does not match the selected certificate. " +
        "Expected $normalizedThumbprint; received $actualSignerThumbprint."
    )
}
if ($null -eq $authenticodeSignature.TimeStamperCertificate) {
    throw "The Authenticode signature has no timestamp certificate."
}

Write-Host "Signed and verified $resolvedArtifact"

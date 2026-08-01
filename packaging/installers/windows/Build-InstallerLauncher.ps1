[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,
    [Parameter(Mandatory = $true)]
    [string]$ContractPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resolvedOutput = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($resolvedOutput) | Out-Null
$resolvedContract = [IO.Path]::GetFullPath($ContractPath)
if (-not (Test-Path -LiteralPath $resolvedContract -PathType Leaf)) {
    throw "Rendered installer contract not found: $resolvedContract"
}
$brandIconPath = [IO.Path]::GetFullPath(
    [IO.Path]::Combine(
        $PSScriptRoot,
        "..",
        "..",
        "..",
        "openhcs",
        "resources",
        "assets",
        "openhcs.ico"
    )
)
if (-not (Test-Path -LiteralPath $brandIconPath -PathType Leaf)) {
    throw "OpenHCS brand icon not found: $brandIconPath"
}
$brandLogoPath = [IO.Path]::GetFullPath(
    [IO.Path]::Combine(
        $PSScriptRoot,
        "..",
        "..",
        "..",
        "openhcs",
        "resources",
        "assets",
        "openhcs-icon-square.png"
    )
)
if (-not (Test-Path -LiteralPath $brandLogoPath -PathType Leaf)) {
    throw "OpenHCS brand logo not found: $brandLogoPath"
}
$temporaryRoot = [IO.Path]::Combine(
    [IO.Path]::GetTempPath(),
    "openhcs-installer-launcher-$([Guid]::NewGuid().ToString('N'))"
)
$sourceRoot = [IO.Path]::Combine($temporaryRoot, "source")
$windowsSourceRoot = [IO.Path]::Combine($sourceRoot, "windows")
$buildRoot = [IO.Path]::Combine($temporaryRoot, "build")
$artifactsRoot = [IO.Path]::Combine($temporaryRoot, "artifacts")

try {
    [IO.Directory]::CreateDirectory($windowsSourceRoot) | Out-Null
    foreach ($sourceName in @(
        "InstallerLauncher.cs",
        "InstallerLauncher.csproj",
        "Install-OpenHCS.ps1"
    )) {
        Copy-Item -LiteralPath ([IO.Path]::Combine($PSScriptRoot, $sourceName)) `
            -Destination ([IO.Path]::Combine($windowsSourceRoot, $sourceName))
    }
    Copy-Item -LiteralPath $resolvedContract -Destination (
        [IO.Path]::Combine($sourceRoot, "installer_contract.json")
    )
    Copy-Item -LiteralPath $brandIconPath -Destination (
        [IO.Path]::Combine($windowsSourceRoot, "OpenHCS.ico")
    )
    Copy-Item -LiteralPath $brandLogoPath -Destination (
        [IO.Path]::Combine($windowsSourceRoot, "OpenHCS.png")
    )
    $projectPath = [IO.Path]::Combine(
        $windowsSourceRoot,
        "InstallerLauncher.csproj"
    )
    & dotnet build $projectPath `
        --configuration Release `
        --output $buildRoot `
        --artifacts-path $artifactsRoot `
        --nologo
    if ($LASTEXITCODE -ne 0) {
        throw "The Windows installer launcher build failed with exit code $LASTEXITCODE."
    }

    $launcherName = "OpenHCS-Windows-Installer.exe"
    $launcherPath = [IO.Path]::Combine($buildRoot, $launcherName)
    if (-not (Test-Path -LiteralPath $launcherPath -PathType Leaf)) {
        throw "The Windows installer build did not produce $launcherName."
    }
    Copy-Item -LiteralPath $launcherPath -Destination (
        [IO.Path]::Combine($resolvedOutput, $launcherName)
    ) -Force
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force `
            -ErrorAction SilentlyContinue
    }
}

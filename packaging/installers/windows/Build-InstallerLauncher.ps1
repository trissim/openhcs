[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resolvedOutput = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($resolvedOutput) | Out-Null

$projectPath = [IO.Path]::Combine(
    $PSScriptRoot,
    "InstallerLauncher.csproj"
)
$temporaryRoot = [IO.Path]::Combine(
    [IO.Path]::GetTempPath(),
    "openhcs-installer-launcher-$([Guid]::NewGuid().ToString('N'))"
)
$buildRoot = [IO.Path]::Combine($temporaryRoot, "build")
$artifactsRoot = [IO.Path]::Combine($temporaryRoot, "artifacts")

try {
    & dotnet build $projectPath `
        --configuration Release `
        --output $buildRoot `
        --artifacts-path $artifactsRoot `
        --nologo
    if ($LASTEXITCODE -ne 0) {
        throw "The Windows installer launcher build failed with exit code $LASTEXITCODE."
    }

    $launcherPath = [IO.Path]::Combine($buildRoot, "Install-OpenHCS.exe")
    if (-not (Test-Path -LiteralPath $launcherPath -PathType Leaf)) {
        throw "The Windows installer launcher build did not produce Install-OpenHCS.exe."
    }
    Copy-Item -LiteralPath $launcherPath -Destination (
        [IO.Path]::Combine($resolvedOutput, "Install-OpenHCS.exe")
    ) -Force
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force `
            -ErrorAction SilentlyContinue
    }
}

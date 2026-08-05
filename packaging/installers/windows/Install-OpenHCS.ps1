[CmdletBinding()]
param(
    [switch]$Worker,
    [switch]$RegisterMcpClients,
    [string]$InstallRoot,
    [string]$CancellationPath,
    [string]$BrandIconPath,
    [string]$BrandLogoPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:SupportedContractSchema = "openhcs.installer.v2"
$script:ManagedEnvironmentNamePattern = "^env-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{32}$"
$script:LogPath = $null
$script:LogWriter = $null

function Get-EmergencyLogPath {
    $localData = [Environment]::GetFolderPath("LocalApplicationData")
    if ([string]::IsNullOrWhiteSpace($localData)) {
        $localData = [IO.Path]::GetTempPath()
    }
    return [IO.Path]::Combine($localData, "OpenHCS Installer", "bootstrap.log")
}

function Write-EmergencyLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    $path = Get-EmergencyLogPath
    try {
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($path)) | Out-Null
        Add-Content -LiteralPath $path -Encoding UTF8 -Value (
            "{0:u} {1}" -f [DateTime]::Now, $Message
        )
    }
    catch {
        # The message box still carries the original failure when even the
        # user-local emergency log cannot be written.
    }
    return $path
}

function Get-WindowsPowerShellExecutable {
    $windowsDirectory = [Environment]::GetFolderPath("Windows")
    if ([string]::IsNullOrWhiteSpace($windowsDirectory)) {
        throw "Windows did not provide its system directory."
    }
    $executable = [IO.Path]::Combine(
        $windowsDirectory,
        "System32",
        "WindowsPowerShell",
        "v1.0",
        "powershell.exe"
    )
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Windows PowerShell is unavailable at '$executable'."
    }
    return $executable
}

function Replace-FileDiscardingPrevious {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )

    # File.Replace accepts a null backup on Windows, but alternate CLR hosts
    # can reject it as an empty path. Use a real temporary backup for the
    # atomic swap, then discard that backup.
    $discardedPath = (
        "$DestinationPath.discarded-$([Guid]::NewGuid().ToString('N'))"
    )
    try {
        [IO.File]::Replace(
            $SourcePath, $DestinationPath, $discardedPath, $true
        )
    }
    finally {
        if (Test-Path -LiteralPath $discardedPath) {
            Remove-Item -LiteralPath $discardedPath -Force `
                -ErrorAction SilentlyContinue
        }
    }
}

function Get-RequiredTextProperty {
    param(
        [Parameter(Mandatory = $true)][object]$InputObject,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $property = $InputObject.PSObject.Properties[$Name]
    if ($null -eq $property -or -not ($property.Value -is [string])) {
        throw "Installer contract property '$Name' must be a string."
    }
    $value = [string]$property.Value
    if ([string]::IsNullOrWhiteSpace($value) -or $value -match "[\x00-\x1f]") {
        throw "Installer contract property '$Name' is empty or contains control characters."
    }
    return $value
}

function Resolve-ContractPath {
    $repositoryParent = Split-Path -Parent $PSScriptRoot
    $candidates = @(
        [IO.Path]::Combine($PSScriptRoot, "installer_contract.json"),
        [IO.Path]::Combine($repositoryParent, "installer_contract.json")
    ) | Select-Object -Unique
    $existing = @($candidates | Where-Object {
        Test-Path -LiteralPath $_ -PathType Leaf
    })
    if ($existing.Count -ne 1) {
        throw (
            "Expected exactly one installer_contract.json beside the installer " +
            "or in its repository parent; found {0}." -f $existing.Count
        )
    }
    return $existing[0]
}

function Read-InstallerContract {
    $contractPath = Resolve-ContractPath
    try {
        $contract = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8 |
            ConvertFrom-Json
    }
    catch {
        throw "Could not parse installer contract '$contractPath': $($_.Exception.Message)"
    }

    $schemaVersion = Get-RequiredTextProperty $contract "schema_version"
    $productName = Get-RequiredTextProperty $contract "product_name"
    $pythonVersion = Get-RequiredTextProperty $contract "python_version"
    $packageRequirement = Get-RequiredTextProperty $contract "package_requirement"
    $entryPoint = Get-RequiredTextProperty $contract "entry_point"
    $guiEntryPoint = Get-RequiredTextProperty $contract "gui_entry_point"

    if ($schemaVersion -ne $script:SupportedContractSchema) {
        throw "Unsupported installer contract schema '$schemaVersion'."
    }
    if ($productName -notmatch "^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$" -or
        $productName.EndsWith(" ") -or $productName.EndsWith(".")) {
        throw "Installer contract product_name has an unsafe Windows path format."
    }
    if ($pythonVersion -notmatch "^3\.[0-9]+$") {
        throw "Installer contract python_version must select one Python 3 minor."
    }
    if ($packageRequirement -notmatch (
        "^[A-Za-z0-9][A-Za-z0-9_.-]*" +
        "(\[[A-Za-z0-9_.-]+(,[A-Za-z0-9_.-]+)*\])?" +
        "([<>=!~]=?[A-Za-z0-9.*+!_-]+)?$"
    )) {
        throw "Installer contract package_requirement has an unsafe format."
    }
    if ($entryPoint -notmatch "^[A-Za-z0-9][A-Za-z0-9_.-]*$") {
        throw "Installer contract entry_point has an unsafe executable-name format."
    }
    if ($guiEntryPoint -notmatch "^[A-Za-z0-9][A-Za-z0-9_.-]*$") {
        throw (
            "Installer contract gui_entry_point has an unsafe " +
            "executable-name format."
        )
    }

    $uvReleaseProperty = $contract.PSObject.Properties["uv_release"]
    if ($null -eq $uvReleaseProperty -or $null -eq $uvReleaseProperty.Value) {
        throw "Installer contract property 'uv_release' is required."
    }
    $uvVersion = Get-RequiredTextProperty $uvReleaseProperty.Value "version"
    $uvBaseUrl = Get-RequiredTextProperty $uvReleaseProperty.Value "base_url"
    if ($uvVersion -notmatch "^[0-9]+\.[0-9]+\.[0-9]+$") {
        throw "Installer contract uv_release.version must be stable SemVer."
    }
    $parsedUvBaseUrl = $null
    if (-not [Uri]::TryCreate(
            $uvBaseUrl,
            [UriKind]::Absolute,
            [ref]$parsedUvBaseUrl
        ) -or
        $parsedUvBaseUrl.Scheme -ne [Uri]::UriSchemeHttps -or
        -not [string]::Equals(
            $parsedUvBaseUrl.IdnHost,
            "astral.sh",
            [StringComparison]::OrdinalIgnoreCase
        ) -or
        $parsedUvBaseUrl.AbsolutePath.TrimEnd("/") -ne "/uv" -or
        -not [string]::IsNullOrEmpty($parsedUvBaseUrl.Query) -or
        -not [string]::IsNullOrEmpty($parsedUvBaseUrl.Fragment)) {
        throw (
            "Installer contract uv_release.base_url must be the official " +
            "https://astral.sh/uv endpoint."
        )
    }
    $uvInstallerUrl = "{0}/{1}/install.ps1" -f (
        $parsedUvBaseUrl.AbsoluteUri.TrimEnd("/"),
        $uvVersion
    )

    return [PSCustomObject]@{
        ProductName = $productName
        PythonVersion = $pythonVersion
        PackageRequirement = $packageRequirement
        EntryPoint = $entryPoint
        GuiEntryPoint = $guiEntryPoint
        UvVersion = $uvVersion
        UvInstallerUrl = $uvInstallerUrl
    }
}

function Resolve-InstallRoot {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path) -or $Path -match "[\x00-\x1f]") {
        throw "Choose a non-empty installation folder."
    }
    $resolved = [IO.Path]::GetFullPath($Path)
    if (-not [IO.Path]::IsPathRooted($resolved)) {
        throw "The installation folder must be an absolute path."
    }
    $volumeRoot = [IO.Path]::GetPathRoot($resolved)
    if ($resolved.TrimEnd("\", "/") -eq $volumeRoot.TrimEnd("\", "/")) {
        throw "A filesystem root cannot be used as the installation folder."
    }
    return $resolved.TrimEnd("\", "/")
}

function Resolve-ManagedEnvironmentPath {
    param(
        [Parameter(Mandatory = $true)][string]$EnvironmentPath,
        [Parameter(Mandatory = $true)][string]$EnvironmentsRoot
    )

    $resolvedRoot = [IO.Path]::GetFullPath($EnvironmentsRoot).TrimEnd("\", "/")
    $resolvedEnvironment = [IO.Path]::GetFullPath($EnvironmentPath).TrimEnd("\", "/")
    $environmentParent = [IO.Path]::GetDirectoryName($resolvedEnvironment)
    $environmentName = [IO.Path]::GetFileName($resolvedEnvironment)
    if (-not [string]::Equals(
            $environmentParent,
            $resolvedRoot,
            [StringComparison]::OrdinalIgnoreCase
        )) {
        throw (
            "Refusing to remove an environment outside the managed environment " +
            "directory: '$resolvedEnvironment'."
        )
    }
    if ($environmentName -notmatch $script:ManagedEnvironmentNamePattern) {
        throw "Refusing to remove an environment with an unmanaged name."
    }
    return $resolvedEnvironment
}

function ConvertTo-WindowsExtendedPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = [IO.Path]::GetFullPath($Path)
    if ($resolved.StartsWith("\\?\")) {
        return $resolved
    }
    if ($resolved.StartsWith("\\")) {
        return "\\?\UNC\$($resolved.TrimStart('\'))"
    }
    return "\\?\$resolved"
}

function Remove-ManagedEnvironmentDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$EnvironmentPath,
        [Parameter(Mandatory = $true)][string]$EnvironmentsRoot
    )

    $resolvedEnvironment = Resolve-ManagedEnvironmentPath `
        $EnvironmentPath $EnvironmentsRoot
    if (-not (Test-Path -LiteralPath $resolvedEnvironment)) {
        return
    }

    $powerShellRemovalError = $null
    try {
        Remove-Item -LiteralPath $resolvedEnvironment -Recurse -Force
    }
    catch {
        $powerShellRemovalError = $_.Exception.Message
    }
    if (-not (Test-Path -LiteralPath $resolvedEnvironment)) {
        return
    }

    # Windows PowerShell 5.1 can fail partway through deep package trees even
    # when uv and Python created them successfully. Native rd accepts the
    # extended-length path prefix. Pass the validated path through one private
    # environment variable so no path text is interpreted as a command.
    $commandProcessor = [IO.Path]::Combine(
        $env:SystemRoot, "System32", "cmd.exe"
    )
    if (-not (Test-Path -LiteralPath $commandProcessor -PathType Leaf)) {
        throw (
            "Windows PowerShell could not remove '$resolvedEnvironment' and " +
            "the native command processor is unavailable. " +
            "PowerShell error: $powerShellRemovalError"
        )
    }
    $previousDeleteTarget = $env:OPENHCS_INSTALLER_DELETE_TARGET
    try {
        $env:OPENHCS_INSTALLER_DELETE_TARGET = ConvertTo-WindowsExtendedPath `
            $resolvedEnvironment
        & $commandProcessor /D /S /C `
            'rd /S /Q "%OPENHCS_INSTALLER_DELETE_TARGET%"'
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        [Environment]::SetEnvironmentVariable(
            "OPENHCS_INSTALLER_DELETE_TARGET",
            $previousDeleteTarget,
            "Process"
        )
    }
    if ($nativeExitCode -ne 0 -or
        (Test-Path -LiteralPath $resolvedEnvironment)) {
        throw (
            "Both Windows PowerShell and native long-path cleanup failed for " +
            "'$resolvedEnvironment' (native exit code $nativeExitCode). " +
            "PowerShell error: $powerShellRemovalError"
        )
    }
}

function Write-InstallLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    if ([string]::IsNullOrWhiteSpace($script:LogPath) -or
        $null -eq $script:LogWriter) {
        throw "Installer log writer was not initialized."
    }
    $line = (
        "{0:u} {1}" -f [DateTime]::Now, $Message
    )
    $script:LogWriter.WriteLine($line)
    if ($Worker) {
        Write-Host $line
    }
}

function Open-InstallLog {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ($null -ne $script:LogWriter) {
        throw "Installer log writer was already initialized."
    }
    $stream = [IO.FileStream]::new(
        $Path,
        [IO.FileMode]::Create,
        [IO.FileAccess]::Write,
        [IO.FileShare]::Read
    )
    $writer = $null
    try {
        $encoding = [Text.UTF8Encoding]::new($true)
        $writer = [IO.StreamWriter]::new($stream, $encoding)
        $writer.AutoFlush = $true
        $script:LogWriter = $writer
    }
    catch {
        if ($null -ne $writer) {
            $writer.Dispose()
        }
        else {
            $stream.Dispose()
        }
        throw
    }
}

function Close-InstallLog {
    if ($null -eq $script:LogWriter) {
        return
    }
    try {
        $script:LogWriter.Dispose()
    }
    finally {
        $script:LogWriter = $null
    }
}

function Resolve-InstallerCancellationPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path) -or $Path -match "[\x00-\x1f]") {
        throw "The installer cancellation path is invalid."
    }
    $resolved = [IO.Path]::GetFullPath($Path)
    $temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    $resolvedParent = [IO.Path]::GetDirectoryName($resolved)
    if (-not [string]::Equals(
        $resolvedParent.TrimEnd("\", "/"),
        $temporaryRoot.TrimEnd("\", "/"),
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "The installer cancellation marker must be in the temporary folder."
    }
    if ([IO.Path]::GetFileName($resolved) -notmatch (
        "^openhcs-installer-cancel-[a-f0-9]{32}\.marker$"
    )) {
        throw "The installer cancellation marker name is invalid."
    }
    return $resolved
}

function New-InstallerCancellationPath {
    $path = [IO.Path]::Combine(
        [IO.Path]::GetTempPath(),
        "openhcs-installer-cancel-$([Guid]::NewGuid().ToString('N')).marker"
    )
    return Resolve-InstallerCancellationPath $path
}

function Request-InstallerCancellation {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = Resolve-InstallerCancellationPath $Path
    [IO.File]::WriteAllText(
        $resolved,
        [DateTime]::UtcNow.ToString("O"),
        [Text.Encoding]::UTF8
    )
}

function Remove-InstallerCancellationMarker {
    param([AllowNull()][string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return
    }
    $resolved = Resolve-InstallerCancellationPath $Path
    if (Test-Path -LiteralPath $resolved -PathType Leaf) {
        Remove-Item -LiteralPath $resolved -Force -ErrorAction SilentlyContinue
    }
}

function Test-InstallerCancellationRequested {
    param([Parameter(Mandatory = $true)][string]$Path)

    return Test-Path -LiteralPath $Path -PathType Leaf
}

function Assert-InstallerCancellationNotRequested {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (Test-InstallerCancellationRequested $Path) {
        throw [OperationCanceledException]::new(
            "Installation was cancelled before publication."
        )
    }
}

function Stop-InstallerChildProcess {
    param([Parameter(Mandatory = $true)][Diagnostics.Process]$Process)

    if ($Process.HasExited) {
        return
    }
    $taskkill = [IO.Path]::Combine($env:SystemRoot, "System32", "taskkill.exe")
    & $taskkill "/PID" ([string]$Process.Id) "/T" "/F" 2>$null | Out-Null
    $Process.WaitForExit(5000) | Out-Null
    if (-not $Process.HasExited) {
        $Process.Kill()
        $Process.WaitForExit(5000) | Out-Null
    }
    if (-not $Process.HasExited) {
        throw "The active installer command did not stop within ten seconds."
    }
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][string]$CancellationPath,
        [switch]$CaptureOutput
    )

    Assert-InstallerCancellationNotRequested $CancellationPath
    Write-InstallLog "START: $Description"
    $payload = [PSCustomObject]@{
        FilePath = $FilePath
        ArgumentList = @($ArgumentList)
    } | ConvertTo-Json -Compress
    $payloadBase64 = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes($payload)
    )
    $childCommand = @"
`$payload = [Text.Encoding]::UTF8.GetString(
    [Convert]::FromBase64String('$payloadBase64')
) | ConvertFrom-Json
& ([string]`$payload.FilePath) @([string[]]`$payload.ArgumentList)
exit `$LASTEXITCODE
"@
    $encodedCommand = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($childCommand)
    )
    $startInfo = New-Object Diagnostics.ProcessStartInfo
    $startInfo.FileName = Get-WindowsPowerShellExecutable
    $startInfo.Arguments = (
        "-NoProfile -ExecutionPolicy Bypass -EncodedCommand {0}" -f
        $encodedCommand
    )
    $startInfo.WorkingDirectory = $PSScriptRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $startInfo
    $capturedOutput = [Collections.Generic.List[string]]::new()
    $cancelled = $false
    try {
        if (-not $process.Start()) {
            throw "Windows could not start: $Description"
        }
        $standardOutput = $process.StandardOutput.ReadLineAsync()
        $standardError = $process.StandardError.ReadLineAsync()
        $standardOutputEnded = $false
        $standardErrorEnded = $false
        $pollCount = 0
        while ($true) {
            if (-not $standardOutputEnded -and $standardOutput.IsCompleted) {
                $line = $standardOutput.GetAwaiter().GetResult()
                if ($null -eq $line) {
                    $standardOutputEnded = $true
                }
                else {
                    if (-not [string]::IsNullOrWhiteSpace($line)) {
                        Write-InstallLog $line
                        if ($CaptureOutput) {
                            $capturedOutput.Add($line)
                        }
                    }
                    $standardOutput = $process.StandardOutput.ReadLineAsync()
                }
            }
            if (-not $standardErrorEnded -and $standardError.IsCompleted) {
                $line = $standardError.GetAwaiter().GetResult()
                if ($null -eq $line) {
                    $standardErrorEnded = $true
                }
                else {
                    if (-not [string]::IsNullOrWhiteSpace($line)) {
                        Write-InstallLog $line
                    }
                    $standardError = $process.StandardError.ReadLineAsync()
                }
            }

            $processExited = $process.WaitForExit(100)
            if ($processExited -and $standardOutputEnded -and $standardErrorEnded) {
                break
            }
            if (-not $processExited -and -not $cancelled -and
                (Test-InstallerCancellationRequested $CancellationPath)) {
                Write-InstallLog "CANCELLING: $Description"
                Stop-InstallerChildProcess $process
                $cancelled = $true
            }
            if (-not $processExited) {
                $pollCount++
                if (($pollCount % 100) -eq 0) {
                    Write-InstallLog "WAITING: $Description is still running."
                }
            }
            elseif (-not $standardOutputEnded -or -not $standardErrorEnded) {
                Start-Sleep -Milliseconds 10
            }
        }
        $process.WaitForExit()
        $exitCode = $process.ExitCode
    }
    finally {
        $process.Dispose()
    }
    if ($cancelled) {
        throw [OperationCanceledException]::new(
            "Installation was cancelled while running: $Description"
        )
    }
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode."
    }
    Assert-InstallerCancellationNotRequested $CancellationPath
    Write-InstallLog "DONE: $Description"
    if ($CaptureOutput) {
        return $capturedOutput.ToArray()
    }
}

function Get-StableLauncherPath {
    param(
        [Parameter(Mandatory = $true)][object]$Contract,
        [Parameter(Mandatory = $true)][string]$ResolvedInstallRoot
    )

    $launcherName = "Launch-{0}.ps1" -f ($Contract.ProductName -replace " ", "-")
    return [IO.Path]::Combine($ResolvedInstallRoot, $launcherName)
}

function Get-DesktopShortcutPath {
    param([Parameter(Mandatory = $true)][object]$Contract)

    $desktop = [Environment]::GetFolderPath("DesktopDirectory")
    if ([string]::IsNullOrWhiteSpace($desktop)) {
        throw "Windows did not provide a user Desktop folder."
    }
    return [IO.Path]::Combine($desktop, "$($Contract.ProductName).lnk")
}

function Publish-LaunchAdapterAndShortcut {
    param(
        [Parameter(Mandatory = $true)][object]$Contract,
        [Parameter(Mandatory = $true)][string]$ResolvedInstallRoot,
        [Parameter(Mandatory = $true)][string]$EnvironmentName,
        [Parameter(Mandatory = $true)][string]$CancellationPath
    )

    if ($EnvironmentName -notmatch $script:ManagedEnvironmentNamePattern) {
        throw "Internal environment name has an unsafe format."
    }

    $launcherPath = Get-StableLauncherPath $Contract $ResolvedInstallRoot
    $environmentPath = [IO.Path]::Combine(
        $ResolvedInstallRoot, "environments", $EnvironmentName
    )
    $environmentPython = [IO.Path]::Combine(
        $environmentPath, "Scripts", "python.exe"
    )
    if (-not (Test-Path -LiteralPath $environmentPython -PathType Leaf)) {
        throw "Installed Python executable is unavailable: $environmentPython"
    }
    $uvExecutable = [IO.Path]::Combine(
        $ResolvedInstallRoot, "bootstrap", "uv", "uv.exe"
    )
    if (-not (Test-Path -LiteralPath $uvExecutable -PathType Leaf)) {
        throw "Managed uv executable is unavailable: $uvExecutable"
    }
    $previousUvExecutable = $env:OPENHCS_UV_EXECUTABLE
    try {
        $env:OPENHCS_UV_EXECUTABLE = $uvExecutable
        $output = @(
            Invoke-LoggedCommand `
                -FilePath $environmentPython `
                -ArgumentList @(
                    "-I",
                    "-m", "openhcs.desktop_deployment_cli",
                    "--installation-pointer=$launcherPath",
                    "--json"
                ) `
                -Description "Publish desktop application, launchers, and shortcut" `
                -CancellationPath $CancellationPath `
                -CaptureOutput
        )
    }
    finally {
        [Environment]::SetEnvironmentVariable(
            "OPENHCS_UV_EXECUTABLE", $previousUvExecutable, "Process"
        )
    }
    $jsonLines = @(
        $output | Where-Object { ([string]$_).TrimStart().StartsWith("{") }
    )
    if ($jsonLines.Count -ne 1) {
        throw "Installed desktop deployment did not return one JSON report."
    }
    $report = ([string]$jsonLines[0]) | ConvertFrom-Json
    $shortcutPath = [IO.Path]::GetFullPath([string]$report.desktop_shortcut_path)
    if (-not (Test-Path -LiteralPath $launcherPath -PathType Leaf)) {
        throw "Installed desktop deployment did not publish the stable launcher."
    }
    if (-not (Test-Path -LiteralPath $shortcutPath -PathType Leaf)) {
        throw "Installed desktop deployment did not publish the Desktop shortcut."
    }
    Write-InstallLog "Desktop shortcut: $shortcutPath"
}

function Register-InstalledMcpClients {
    param(
        [Parameter(Mandatory = $true)][object]$Contract,
        [Parameter(Mandatory = $true)][string]$ResolvedInstallRoot,
        [Parameter(Mandatory = $true)][string]$EnvironmentName
    )

    $statusPath = [IO.Path]::Combine(
        $ResolvedInstallRoot, "agent-registration-status"
    )
    $registrationExecutable = [IO.Path]::Combine(
        $ResolvedInstallRoot,
        "environments",
        $EnvironmentName,
        "Scripts",
        "openhcs-mcp-register.exe"
    )
    if (-not (Test-Path -LiteralPath $registrationExecutable -PathType Leaf)) {
        Write-InstallLog (
            "WARNING: Agent registration entry point is unavailable: " +
            $registrationExecutable
        )
        Set-Content -LiteralPath $statusPath -Encoding ASCII -Value "warning" `
            -ErrorAction SilentlyContinue
        return $false
    }

    $launcherPath = Get-StableLauncherPath $Contract $ResolvedInstallRoot
    $powerShellExecutable = Get-WindowsPowerShellExecutable
    $launcherArguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $launcherPath,
        "mcp"
    )
    $registrationArguments = @(
        "--command", $powerShellExecutable
    )
    foreach ($launcherArgument in $launcherArguments) {
        $registrationArguments += (
            "--launcher-argument={0}" -f $launcherArgument
        )
    }
    $registrationArguments += @(
        "--register", "codex",
        "--register-detected",
        "--json"
    )
    $reportPath = [IO.Path]::Combine(
        $ResolvedInstallRoot, "agent-registration.json"
    )
    $reportCandidate = "$reportPath.candidate-$([Guid]::NewGuid().ToString('N'))"

    Write-InstallLog "START: Connect OpenHCS to local agent clients"
    try {
        $output = @(
            & $registrationExecutable @registrationArguments 2>&1
        )
        $exitCode = $LASTEXITCODE
        foreach ($line in $output) {
            if (-not [string]::IsNullOrWhiteSpace([string]$line)) {
                Write-InstallLog ([string]$line)
            }
        }
        $jsonText = ($output -join [Environment]::NewLine)
        $report = $null
        if (-not [string]::IsNullOrWhiteSpace($jsonText)) {
            $report = $jsonText | ConvertFrom-Json
            Set-Content -LiteralPath $reportCandidate -Encoding UTF8 -Value $jsonText
            if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
                Replace-FileDiscardingPrevious `
                    -SourcePath $reportCandidate `
                    -DestinationPath $reportPath
            }
            else {
                [IO.File]::Move($reportCandidate, $reportPath)
            }
        }
        if ($exitCode -ne 0 -or $null -eq $report -or
            $report.ok -ne $true) {
            Write-InstallLog (
                "WARNING: One or more agent client registrations did not complete " +
                "(exit code $exitCode). OpenHCS itself remains installed."
            )
            Set-Content -LiteralPath $statusPath -Encoding ASCII -Value "warning" `
                -ErrorAction SilentlyContinue
            return $false
        }
        Set-Content -LiteralPath $statusPath -Encoding ASCII -Value "connected" `
            -ErrorAction SilentlyContinue
        Write-InstallLog "DONE: Connect OpenHCS to local agent clients"
        return $true
    }
    catch {
        Write-InstallLog (
            "WARNING: Could not finish agent client registration: " +
            $_.Exception.Message
        )
        Set-Content -LiteralPath $statusPath -Encoding ASCII -Value "warning" `
            -ErrorAction SilentlyContinue
        return $false
    }
    finally {
        if (Test-Path -LiteralPath $reportCandidate) {
            Remove-Item -LiteralPath $reportCandidate -Force -ErrorAction SilentlyContinue
        }
    }
}

function Remove-SupersededEnvironments {
    param(
        [Parameter(Mandatory = $true)][string]$EnvironmentsRoot,
        [Parameter(Mandatory = $true)][string]$CurrentEnvironmentPath
    )

    $currentFullPath = [IO.Path]::GetFullPath($CurrentEnvironmentPath)
    Get-ChildItem -LiteralPath $EnvironmentsRoot -Directory | ForEach-Object {
        $supersededEnvironmentPath = $_.FullName
        if ([IO.Path]::GetFullPath($supersededEnvironmentPath) -eq $currentFullPath) {
            return
        }
        try {
            Remove-ManagedEnvironmentDirectory `
                $supersededEnvironmentPath $EnvironmentsRoot
            Write-InstallLog "Removed superseded environment: $supersededEnvironmentPath"
        }
        catch {
            Write-InstallLog (
                "WARNING: Could not remove superseded environment " +
                "'$supersededEnvironmentPath': " +
                $_.Exception.Message
            )
        }
    }
}

function Remove-UnpublishedCandidateEnvironment {
    param(
        [AllowNull()][string]$CandidatePath,
        [AllowNull()][string]$EnvironmentsRoot,
        [Parameter(Mandatory = $true)][string]$Outcome
    )

    if ([string]::IsNullOrWhiteSpace($CandidatePath) -or
        [string]::IsNullOrWhiteSpace($EnvironmentsRoot) -or
        -not (Test-Path -LiteralPath $CandidatePath)) {
        return
    }
    try {
        Remove-ManagedEnvironmentDirectory $CandidatePath $EnvironmentsRoot
        Write-InstallLog "Removed $Outcome candidate environment: $CandidatePath"
    }
    catch {
        try {
            Write-InstallLog (
                "WARNING: Could not remove $Outcome candidate environment " +
                "'$CandidatePath': $($_.Exception.Message)"
            )
        }
        catch {
            # The durable outcome was already recorded by the caller.
        }
    }
}

function Invoke-WorkerInstall {
    param(
        [Parameter(Mandatory = $true)][object]$Contract,
        [Parameter(Mandatory = $true)][string]$RequestedInstallRoot,
        [Parameter(Mandatory = $true)][string]$RequestedCancellationPath
    )

    $temporaryUvInstaller = $null
    $newEnvironmentPath = $null
    $environmentsRoot = $null
    $publicationStarted = $false
    try {
        $resolvedRoot = Resolve-InstallRoot $RequestedInstallRoot
        $resolvedCancellationPath = Resolve-InstallerCancellationPath `
            $RequestedCancellationPath
        [IO.Directory]::CreateDirectory($resolvedRoot) | Out-Null
        $script:LogPath = [IO.Path]::Combine($resolvedRoot, "installer.log")
        Open-InstallLog $script:LogPath
        Write-InstallLog "Starting $($Contract.ProductName) installation."

        $bootstrapRoot = [IO.Path]::Combine($resolvedRoot, "bootstrap")
        $uvInstallRoot = [IO.Path]::Combine($bootstrapRoot, "uv")
        $environmentsRoot = [IO.Path]::Combine($resolvedRoot, "environments")
        $environmentName = "env-{0}-{1}" -f (
            [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ"),
            [Guid]::NewGuid().ToString("N")
        )
        $newEnvironmentPath = [IO.Path]::Combine(
            $environmentsRoot, $environmentName
        )
        [IO.Directory]::CreateDirectory($bootstrapRoot) | Out-Null
        [IO.Directory]::CreateDirectory($uvInstallRoot) | Out-Null
        [IO.Directory]::CreateDirectory($environmentsRoot) | Out-Null

        Assert-InstallerCancellationNotRequested $resolvedCancellationPath
        Write-InstallLog (
            "Downloading the pinned official uv $($Contract.UvVersion) " +
            "installer over HTTPS."
        )
        $temporaryUvInstaller = [IO.Path]::Combine(
            [IO.Path]::GetTempPath(),
            "openhcs-uv-installer-$([Guid]::NewGuid().ToString('N')).ps1"
        )
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri $Contract.UvInstallerUrl `
            -OutFile $temporaryUvInstaller -TimeoutSec 120
        Assert-InstallerCancellationNotRequested $resolvedCancellationPath

        $previousUvInstallDir = $env:UV_INSTALL_DIR
        $previousUvNoModifyPath = $env:UV_NO_MODIFY_PATH
        try {
            $env:UV_INSTALL_DIR = $uvInstallRoot
            $env:UV_NO_MODIFY_PATH = "1"
            Invoke-LoggedCommand `
                -FilePath (Get-WindowsPowerShellExecutable) `
                -ArgumentList @(
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", $temporaryUvInstaller
                ) `
                -Description "Install uv" `
                -CancellationPath $resolvedCancellationPath
        }
        finally {
            [Environment]::SetEnvironmentVariable(
                "UV_INSTALL_DIR", $previousUvInstallDir, "Process"
            )
            [Environment]::SetEnvironmentVariable(
                "UV_NO_MODIFY_PATH", $previousUvNoModifyPath, "Process"
            )
        }

        $uvExecutable = [IO.Path]::Combine($uvInstallRoot, "uv.exe")
        if (-not (Test-Path -LiteralPath $uvExecutable -PathType Leaf)) {
            throw (
                "The pinned official uv $($Contract.UvVersion) installer completed " +
                "without creating '$uvExecutable'. Security software may have " +
                "quarantined uv.exe. Do not disable protection or add a broad " +
                "folder exclusion; report the exact detection as a false positive " +
                "and retry after the file is cleared."
            )
        }

        Invoke-LoggedCommand -FilePath $uvExecutable -ArgumentList @(
            "--no-config", "python", "install", $Contract.PythonVersion
        ) -Description "Install managed Python $($Contract.PythonVersion)" `
            -CancellationPath $resolvedCancellationPath

        Invoke-LoggedCommand -FilePath $uvExecutable -ArgumentList @(
            "--no-config", "venv", "--python",
            $Contract.PythonVersion, $newEnvironmentPath
        ) -Description "Create a candidate virtual environment" `
            -CancellationPath $resolvedCancellationPath

        Invoke-LoggedCommand -FilePath $uvExecutable -ArgumentList @(
            "--no-config", "pip", "install", "--python", $newEnvironmentPath,
            "--prerelease", "explicit", "--upgrade", $Contract.PackageRequirement
        ) -Description "Install $($Contract.PackageRequirement)" `
            -CancellationPath $resolvedCancellationPath

        Invoke-LoggedCommand -FilePath $uvExecutable -ArgumentList @(
            "--no-config", "pip", "check", "--python", $newEnvironmentPath
        ) -Description "Verify installed dependencies" `
            -CancellationPath $resolvedCancellationPath

        $entryExecutable = [IO.Path]::Combine(
            $newEnvironmentPath, "Scripts", "$($Contract.EntryPoint).exe"
        )
        if (-not (Test-Path -LiteralPath $entryExecutable -PathType Leaf)) {
            throw "Installation did not create the declared GUI entry point."
        }

        Assert-InstallerCancellationNotRequested $resolvedCancellationPath
        $publicationStarted = $true
        Publish-LaunchAdapterAndShortcut `
            $Contract $resolvedRoot $environmentName $resolvedCancellationPath
        if ($RegisterMcpClients) {
            $null = Register-InstalledMcpClients `
                $Contract $resolvedRoot $environmentName
        }
        Write-InstallLog "SUCCESS: Installation completed."
        if (Test-InstallerCancellationRequested $resolvedCancellationPath) {
            Write-InstallLog (
                "Cancellation arrived after publication; the verified installation " +
                "remains committed."
            )
        }
        Remove-SupersededEnvironments $environmentsRoot $newEnvironmentPath
        return 0
    }
    catch [OperationCanceledException] {
        if ($publicationStarted) {
            $message = (
                "FAILED: An unexpected cancellation error occurred during publication: " +
                $_.Exception.Message
            )
            try {
                Write-InstallLog $message
            }
            catch {
                Write-EmergencyLog $message | Out-Null
            }
            Remove-UnpublishedCandidateEnvironment `
                $newEnvironmentPath $environmentsRoot "failed"
            return 1
        }
        $message = "CANCELLED: $($_.Exception.Message)"
        try {
            Write-InstallLog $message
        }
        catch {
            Write-EmergencyLog $message | Out-Null
        }
        Remove-UnpublishedCandidateEnvironment `
            $newEnvironmentPath $environmentsRoot "cancelled"
        return 2
    }
    catch {
        $message = "FAILED: $($_.Exception.Message)"
        try {
            Write-InstallLog $message
            Write-InstallLog ([string]$_)
        }
        catch {
            Write-EmergencyLog $message | Out-Null
        }
        Remove-UnpublishedCandidateEnvironment `
            $newEnvironmentPath $environmentsRoot "failed"
        return 1
    }
    finally {
        if ($null -ne $temporaryUvInstaller -and
            (Test-Path -LiteralPath $temporaryUvInstaller)) {
            Remove-Item -LiteralPath $temporaryUvInstaller -Force -ErrorAction SilentlyContinue
        }
        Close-InstallLog
    }
}

function Start-InstallerWorker {
    param(
        [Parameter(Mandatory = $true)][string]$ResolvedInstallRoot,
        [Parameter(Mandatory = $true)][string]$ResolvedCancellationPath,
        [Parameter(Mandatory = $true)][bool]$ShouldRegisterMcpClients
    )

    $escapedScriptPath = $PSCommandPath.Replace("'", "''")
    $escapedInstallRoot = $ResolvedInstallRoot.Replace("'", "''")
    $escapedCancellationPath = $ResolvedCancellationPath.Replace("'", "''")
    $registrationLiteral = if ($ShouldRegisterMcpClients) { '$true' } else { '$false' }
    $workerCommand = (
        "& '{0}' -Worker -InstallRoot '{1}' -CancellationPath '{2}' " +
        "-RegisterMcpClients:{3}"
    ) -f (
        $escapedScriptPath,
        $escapedInstallRoot,
        $escapedCancellationPath,
        $registrationLiteral
    )
    $encodedCommand = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($workerCommand)
    )
    $startInfo = New-Object Diagnostics.ProcessStartInfo
    $startInfo.FileName = Get-WindowsPowerShellExecutable
    $startInfo.Arguments = (
        "-NoProfile -ExecutionPolicy Bypass -EncodedCommand {0}" -f $encodedCommand
    )
    $startInfo.WorkingDirectory = $PSScriptRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        $process.Dispose()
        throw "Windows could not start the background installer."
    }
    return $process
}

function Show-InstallerWindow {
    param(
        [Parameter(Mandatory = $true)][object]$Contract,
        [Parameter(Mandatory = $true)][string]$BrandIconPath,
        [Parameter(Mandatory = $true)][string]$BrandLogoPath
    )

    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [Windows.Forms.Application]::EnableVisualStyles()

    $resolvedBrandIconPath = [IO.Path]::GetFullPath($BrandIconPath)
    if (-not (Test-Path -LiteralPath $resolvedBrandIconPath -PathType Leaf)) {
        throw (
            "The OpenHCS installer brand icon is missing: " +
            "'$resolvedBrandIconPath'."
        )
    }
    $resolvedBrandLogoPath = [IO.Path]::GetFullPath($BrandLogoPath)
    if (-not (Test-Path -LiteralPath $resolvedBrandLogoPath -PathType Leaf)) {
        throw (
            "The OpenHCS installer brand logo is missing: " +
            "'$resolvedBrandLogoPath'."
        )
    }
    $installerIcon = New-Object Drawing.Icon($resolvedBrandIconPath)
    $installerLogo = [Drawing.Image]::FromFile($resolvedBrandLogoPath)

    $localData = [Environment]::GetFolderPath("LocalApplicationData")
    if ([string]::IsNullOrWhiteSpace($localData)) {
        throw "Windows did not provide a user Local Application Data folder."
    }
    $defaultRoot = [IO.Path]::Combine($localData, $Contract.ProductName)
    $script:LogPath = [IO.Path]::Combine($defaultRoot, "installer.log")
    $script:WorkerProcess = $null
    $script:WorkerStandardOutput = $null
    $script:WorkerStandardError = $null
    $script:InstallerProgressLines = [Collections.Generic.Queue[string]]::new()
    $script:CancellationPath = $null
    $script:InstallSucceeded = $false
    $script:ActiveInstallRoot = $null
    $script:ExitCode = 0
    $script:WizardPage = "Welcome"

    $form = New-Object Windows.Forms.Form
    $form.Text = "$($Contract.ProductName) Setup"
    $form.StartPosition = "CenterScreen"
    $form.ClientSize = New-Object Drawing.Size(640, 430)
    $form.MinimumSize = New-Object Drawing.Size(640, 430)
    $form.MaximumSize = New-Object Drawing.Size(760, 560)
    $form.MaximizeBox = $false
    $form.Icon = $installerIcon

    $brandPicture = New-Object Windows.Forms.PictureBox
    $brandPicture.Image = $installerLogo
    $brandPicture.SizeMode = [Windows.Forms.PictureBoxSizeMode]::Zoom
    $brandPicture.Location = New-Object Drawing.Point(18, 9)
    $brandPicture.Size = New-Object Drawing.Size(40, 40)
    $form.Controls.Add($brandPicture)

    $title = New-Object Windows.Forms.Label
    $title.Text = "$($Contract.ProductName) Setup"
    $title.Font = New-Object Drawing.Font("Segoe UI", 17, [Drawing.FontStyle]::Bold)
    $title.AutoSize = $true
    $title.Location = New-Object Drawing.Point(70, 18)
    $form.Controls.Add($title)

    $separator = New-Object Windows.Forms.Label
    $separator.BorderStyle = "Fixed3D"
    $separator.Location = New-Object Drawing.Point(0, 58)
    $separator.Size = New-Object Drawing.Size(640, 2)
    $separator.Anchor = "Top,Left,Right"
    $form.Controls.Add($separator)

    $welcomePanel = New-Object Windows.Forms.Panel
    $welcomePanel.Name = "WelcomePage"
    $welcomePanel.Location = New-Object Drawing.Point(0, 61)
    $welcomePanel.Size = New-Object Drawing.Size(640, 309)
    $welcomePanel.Anchor = "Top,Bottom,Left,Right"
    $form.Controls.Add($welcomePanel)

    $welcomeHeading = New-Object Windows.Forms.Label
    $welcomeHeading.Text = "Welcome to the $($Contract.ProductName) Setup Wizard"
    $welcomeHeading.Font = New-Object Drawing.Font(
        "Segoe UI", 13, [Drawing.FontStyle]::Bold
    )
    $welcomeHeading.Location = New-Object Drawing.Point(28, 35)
    $welcomeHeading.Size = New-Object Drawing.Size(580, 34)
    $welcomePanel.Controls.Add($welcomeHeading)

    $welcomeSummary = New-Object Windows.Forms.Label
    $welcomeSummary.Text = (
        "This wizard installs the complete desktop application for your Windows " +
        "account. It manages its own Python $($Contract.PythonVersion) environment, " +
        "so you do not need Python or command-line tools.`r`n`r`n" +
        "Close $($Contract.ProductName) before updating an existing installation."
    )
    $welcomeSummary.Location = New-Object Drawing.Point(31, 87)
    $welcomeSummary.Size = New-Object Drawing.Size(570, 105)
    $welcomePanel.Controls.Add($welcomeSummary)

    $welcomePrompt = New-Object Windows.Forms.Label
    $welcomePrompt.Text = "Click Next to continue."
    $welcomePrompt.Location = New-Object Drawing.Point(31, 222)
    $welcomePrompt.Size = New-Object Drawing.Size(570, 24)
    $welcomePanel.Controls.Add($welcomePrompt)

    $optionsPanel = New-Object Windows.Forms.Panel
    $optionsPanel.Name = "OptionsPage"
    $optionsPanel.Location = New-Object Drawing.Point(0, 61)
    $optionsPanel.Size = New-Object Drawing.Size(640, 309)
    $optionsPanel.Anchor = "Top,Bottom,Left,Right"
    $form.Controls.Add($optionsPanel)

    $optionsHeading = New-Object Windows.Forms.Label
    $optionsHeading.Text = "Installation options"
    $optionsHeading.Font = New-Object Drawing.Font(
        "Segoe UI", 13, [Drawing.FontStyle]::Bold
    )
    $optionsHeading.Location = New-Object Drawing.Point(28, 26)
    $optionsHeading.Size = New-Object Drawing.Size(580, 32)
    $optionsPanel.Controls.Add($optionsHeading)

    $folderLabel = New-Object Windows.Forms.Label
    $folderLabel.Text = "Install for this account in:"
    $folderLabel.Location = New-Object Drawing.Point(31, 78)
    $folderLabel.AutoSize = $true
    $optionsPanel.Controls.Add($folderLabel)

    $folderText = New-Object Windows.Forms.TextBox
    $folderText.Text = $defaultRoot
    $folderText.Location = New-Object Drawing.Point(34, 104)
    $folderText.Size = New-Object Drawing.Size(457, 25)
    $folderText.Anchor = "Top,Left,Right"
    $optionsPanel.Controls.Add($folderText)

    $browseButton = New-Object Windows.Forms.Button
    $browseButton.Text = "Browse..."
    $browseButton.Location = New-Object Drawing.Point(500, 102)
    $browseButton.Size = New-Object Drawing.Size(104, 29)
    $browseButton.Anchor = "Top,Right"
    $optionsPanel.Controls.Add($browseButton)

    $optionsSummary = New-Object Windows.Forms.Label
    $optionsSummary.Text = (
        "Setup creates a desktop shortcut. If $($Contract.ProductName) is already " +
        "installed here, Next safely updates it after the replacement is verified."
    )
    $optionsSummary.Location = New-Object Drawing.Point(31, 154)
    $optionsSummary.Size = New-Object Drawing.Size(570, 42)
    $optionsPanel.Controls.Add($optionsSummary)

    $agentConnectionCheck = New-Object Windows.Forms.CheckBox
    $agentConnectionCheck.Text = (
        "Connect OpenHCS to ChatGPT, Codex, and local AI agent apps"
    )
    $agentConnectionCheck.Checked = $true
    $agentConnectionCheck.Location = New-Object Drawing.Point(31, 201)
    $agentConnectionCheck.Size = New-Object Drawing.Size(570, 28)
    $optionsPanel.Controls.Add($agentConnectionCheck)

    $optionsPrompt = New-Object Windows.Forms.Label
    $optionsPrompt.Text = "Click Next to begin installation."
    $optionsPrompt.Location = New-Object Drawing.Point(31, 246)
    $optionsPrompt.Size = New-Object Drawing.Size(570, 24)
    $optionsPanel.Controls.Add($optionsPrompt)

    $progressPanel = New-Object Windows.Forms.Panel
    $progressPanel.Name = "ProgressPage"
    $progressPanel.Location = New-Object Drawing.Point(0, 61)
    $progressPanel.Size = New-Object Drawing.Size(640, 309)
    $progressPanel.Anchor = "Top,Bottom,Left,Right"
    $form.Controls.Add($progressPanel)

    $progressHeading = New-Object Windows.Forms.Label
    $progressHeading.Text = "Installing $($Contract.ProductName)"
    $progressHeading.Font = New-Object Drawing.Font(
        "Segoe UI", 13, [Drawing.FontStyle]::Bold
    )
    $progressHeading.Location = New-Object Drawing.Point(28, 15)
    $progressHeading.Size = New-Object Drawing.Size(580, 30)
    $progressPanel.Controls.Add($progressHeading)

    $progressStatusLabel = New-Object Windows.Forms.Label
    $progressStatusLabel.Text = "Preparing the private application environment..."
    $progressStatusLabel.Location = New-Object Drawing.Point(31, 49)
    $progressStatusLabel.Size = New-Object Drawing.Size(570, 24)
    $progressPanel.Controls.Add($progressStatusLabel)

    $progressBar = New-Object Windows.Forms.ProgressBar
    $progressBar.Location = New-Object Drawing.Point(34, 78)
    $progressBar.Size = New-Object Drawing.Size(570, 19)
    $progressBar.Style = [Windows.Forms.ProgressBarStyle]::Marquee
    $progressBar.MarqueeAnimationSpeed = 30
    $progressBar.Anchor = "Top,Left,Right"
    $progressPanel.Controls.Add($progressBar)

    $logBox = New-Object Windows.Forms.TextBox
    $logBox.Multiline = $true
    $logBox.ReadOnly = $true
    $logBox.ScrollBars = "Vertical"
    $logBox.BackColor = [Drawing.Color]::White
    $logBox.Location = New-Object Drawing.Point(34, 112)
    $logBox.Size = New-Object Drawing.Size(570, 166)
    $logBox.Anchor = "Top,Bottom,Left,Right"
    $progressPanel.Controls.Add($logBox)

    $finishPanel = New-Object Windows.Forms.Panel
    $finishPanel.Name = "FinishPage"
    $finishPanel.Location = New-Object Drawing.Point(0, 61)
    $finishPanel.Size = New-Object Drawing.Size(640, 309)
    $finishPanel.Anchor = "Top,Bottom,Left,Right"
    $form.Controls.Add($finishPanel)

    $finishHeading = New-Object Windows.Forms.Label
    $finishHeading.Text = "Installation complete"
    $finishHeading.Font = New-Object Drawing.Font(
        "Segoe UI", 13, [Drawing.FontStyle]::Bold
    )
    $finishHeading.Location = New-Object Drawing.Point(28, 35)
    $finishHeading.Size = New-Object Drawing.Size(580, 34)
    $finishPanel.Controls.Add($finishHeading)

    $finishSummary = New-Object Windows.Forms.Label
    $finishSummary.Text = "$($Contract.ProductName) is ready to use."
    $finishSummary.Location = New-Object Drawing.Point(31, 87)
    $finishSummary.Size = New-Object Drawing.Size(570, 64)
    $finishPanel.Controls.Add($finishSummary)

    $launchCheck = New-Object Windows.Forms.CheckBox
    $launchCheck.Text = "Launch $($Contract.ProductName) after setup"
    $launchCheck.Checked = $true
    $launchCheck.Location = New-Object Drawing.Point(31, 165)
    $launchCheck.Size = New-Object Drawing.Size(570, 28)
    $finishPanel.Controls.Add($launchCheck)

    $finishLogHint = New-Object Windows.Forms.Label
    $finishLogHint.Text = "The installation log remains available below."
    $finishLogHint.Location = New-Object Drawing.Point(31, 216)
    $finishLogHint.Size = New-Object Drawing.Size(570, 28)
    $finishPanel.Controls.Add($finishLogHint)

    $openLogButton = New-Object Windows.Forms.Button
    $openLogButton.Text = "Open log"
    $openLogButton.Location = New-Object Drawing.Point(22, 384)
    $openLogButton.Size = New-Object Drawing.Size(100, 34)
    $openLogButton.Anchor = "Bottom,Left"
    $form.Controls.Add($openLogButton)

    $backButton = New-Object Windows.Forms.Button
    $backButton.Text = "< Back"
    $backButton.Location = New-Object Drawing.Point(310, 384)
    $backButton.Size = New-Object Drawing.Size(96, 34)
    $backButton.Anchor = "Bottom,Right"
    $form.Controls.Add($backButton)

    $nextButton = New-Object Windows.Forms.Button
    $nextButton.Text = "Next >"
    $nextButton.Location = New-Object Drawing.Point(412, 384)
    $nextButton.Size = New-Object Drawing.Size(96, 34)
    $nextButton.Anchor = "Bottom,Right"
    $form.AcceptButton = $nextButton
    $form.Controls.Add($nextButton)

    $cancelButton = New-Object Windows.Forms.Button
    $cancelButton.Text = "Cancel"
    $cancelButton.Location = New-Object Drawing.Point(514, 384)
    $cancelButton.Size = New-Object Drawing.Size(104, 34)
    $cancelButton.Anchor = "Bottom,Right"
    $form.CancelButton = $cancelButton
    $form.Controls.Add($cancelButton)

    $pagePanels = @{
        Welcome = $welcomePanel
        Options = $optionsPanel
        Progress = $progressPanel
        Finish = $finishPanel
    }

    function Add-InstallerProgressLine {
        param([AllowNull()][string]$Line)

        if ([string]::IsNullOrWhiteSpace($Line)) {
            return
        }
        $script:InstallerProgressLines.Enqueue($Line)
        while ($script:InstallerProgressLines.Count -gt 14) {
            $script:InstallerProgressLines.Dequeue() | Out-Null
        }
    }

    function Show-InstallerProgressLines {
        $logBox.Lines = @($script:InstallerProgressLines.ToArray())
        $logBox.SelectionStart = $logBox.TextLength
        $logBox.ScrollToCaret()
    }

    function New-InstallerProgressStream {
        param(
            [Parameter(Mandatory = $true)]
            [IO.StreamReader]$Reader
        )

        return [PSCustomObject]@{
            Reader = $Reader
            PendingRead = $Reader.ReadLineAsync()
            Ended = $false
        }
    }

    function Read-InstallerProgressStream {
        param(
            [Parameter(Mandatory = $true)]
            [object]$Stream
        )

        while (-not $Stream.Ended -and $Stream.PendingRead.IsCompleted) {
            $line = $Stream.PendingRead.GetAwaiter().GetResult()
            if ($null -eq $line) {
                $Stream.Ended = $true
                return
            }
            Add-InstallerProgressLine $line
            $Stream.PendingRead = $Stream.Reader.ReadLineAsync()
        }
    }

    function Set-WizardPage {
        param(
            [Parameter(Mandatory = $true)]
            [ValidateSet("Welcome", "Options", "Progress", "Finish")]
            [string]$Page
        )

        foreach ($panel in $pagePanels.Values) {
            $panel.Visible = $false
        }
        $pagePanels[$Page].Visible = $true
        $script:WizardPage = $Page
        $backButton.Visible = $false
        $backButton.Enabled = $true
        $nextButton.Visible = $true
        $nextButton.Enabled = $true
        $nextButton.Text = "Next >"
        $cancelButton.Visible = $true
        $cancelButton.Enabled = $true
        $cancelButton.Text = "Cancel"
        $openLogButton.Visible = $false

        switch ($Page) {
            "Welcome" {
                $form.AcceptButton = $nextButton
            }
            "Options" {
                $backButton.Visible = $true
                $form.AcceptButton = $nextButton
                $folderText.Focus()
            }
            "Progress" {
                $nextButton.Visible = $false
                $cancelButton.Text = "Cancel install"
                $form.AcceptButton = $null
            }
            "Finish" {
                $nextButton.Text = "Finish"
                $cancelButton.Visible = $false
                $openLogButton.Visible = $true
                $form.AcceptButton = $nextButton
                $nextButton.Focus()
            }
        }
    }

    function Show-InstallerResult {
        param(
            [Parameter(Mandatory = $true)][string]$Heading,
            [Parameter(Mandatory = $true)][string]$Message,
            [Parameter(Mandatory = $true)][bool]$Succeeded
        )

        $script:InstallSucceeded = $Succeeded
        $finishHeading.Text = $Heading
        $finishSummary.Text = $Message
        $launchCheck.Visible = $Succeeded
        $launchCheck.Checked = $Succeeded
        if ($Succeeded) {
            $finishLogHint.Text = (
                "A desktop shortcut is ready. The installation log remains available."
            )
        }
        else {
            $finishLogHint.Text = (
                "Open the durable log for details. You can safely run Setup again."
            )
        }
        $progressBar.MarqueeAnimationSpeed = 0
        Set-WizardPage "Finish"
    }

    $browseButton.Add_Click({
        $dialog = New-Object Windows.Forms.FolderBrowserDialog
        $dialog.Description = "Choose a private installation folder"
        $dialog.SelectedPath = $folderText.Text
        if ($dialog.ShowDialog($form) -eq [Windows.Forms.DialogResult]::OK) {
            $folderText.Text = $dialog.SelectedPath
        }
        $dialog.Dispose()
    })

    $openLogButton.Add_Click({
        $visibleLog = $script:LogPath
        if (-not (Test-Path -LiteralPath $visibleLog -PathType Leaf)) {
            $visibleLog = Get-EmergencyLogPath
        }
        if (Test-Path -LiteralPath $visibleLog -PathType Leaf) {
            Start-Process -FilePath $visibleLog
        }
        else {
            [Windows.Forms.MessageBox]::Show(
                $form, "No installer log has been created yet.", $form.Text,
                [Windows.Forms.MessageBoxButtons]::OK,
                [Windows.Forms.MessageBoxIcon]::Information
            ) | Out-Null
        }
    })

    $backButton.Add_Click({
        if ($script:WizardPage -eq "Options") {
            Set-WizardPage "Welcome"
        }
    })

    $nextButton.Add_Click({
        if ($script:WizardPage -eq "Welcome") {
            Set-WizardPage "Options"
            return
        }

        if ($script:WizardPage -eq "Finish") {
            if ($script:InstallSucceeded -and $launchCheck.Checked) {
                try {
                    Start-Process -FilePath (Get-DesktopShortcutPath $Contract)
                }
                catch {
                    $emergencyPath = Write-EmergencyLog (
                        "Could not launch $($Contract.ProductName): " +
                        $_.Exception.Message
                    )
                    [Windows.Forms.MessageBox]::Show(
                        $form,
                        "$($Contract.ProductName) is installed, but could not be " +
                        "opened automatically.`r`n`r`nLog: $emergencyPath",
                        $form.Text,
                        [Windows.Forms.MessageBoxButtons]::OK,
                        [Windows.Forms.MessageBoxIcon]::Warning
                    ) | Out-Null
                    return
                }
            }
            $form.Close()
            return
        }

        if ($script:WizardPage -ne "Options") {
            return
        }

        try {
            $resolvedRoot = Resolve-InstallRoot $folderText.Text
        }
        catch {
            [Windows.Forms.MessageBox]::Show(
                $form,
                $_.Exception.Message,
                $form.Text,
                [Windows.Forms.MessageBoxButtons]::OK,
                [Windows.Forms.MessageBoxIcon]::Warning
            ) | Out-Null
            return
        }

        $existingLauncher = Get-StableLauncherPath $Contract $resolvedRoot
        if (Test-Path -LiteralPath $existingLauncher -PathType Leaf) {
            $choice = [Windows.Forms.MessageBox]::Show(
                $form,
                "Close $($Contract.ProductName) before continuing. Setup will " +
                "replace it only after the update is fully verified. Continue?",
                "Update $($Contract.ProductName)",
                [Windows.Forms.MessageBoxButtons]::YesNo,
                [Windows.Forms.MessageBoxIcon]::Question
            )
            if ($choice -ne [Windows.Forms.DialogResult]::Yes) {
                return
            }
        }

        $script:LogPath = [IO.Path]::Combine($resolvedRoot, "installer.log")
        $script:ActiveInstallRoot = $resolvedRoot
        Remove-InstallerCancellationMarker $script:CancellationPath
        $script:CancellationPath = New-InstallerCancellationPath
        $script:InstallSucceeded = $false
        $script:ExitCode = 0
        $script:InstallerProgressLines.Clear()
        $logBox.Clear()
        $progressStatusLabel.Text = (
            "Installing. This can take several minutes on the first run."
        )
        $progressBar.MarqueeAnimationSpeed = 30
        Set-WizardPage "Progress"
        try {
            $script:WorkerProcess = Start-InstallerWorker `
                $resolvedRoot `
                $script:CancellationPath `
                $agentConnectionCheck.Checked
            $script:WorkerStandardOutput = New-InstallerProgressStream `
                $script:WorkerProcess.StandardOutput
            $script:WorkerStandardError = New-InstallerProgressStream `
                $script:WorkerProcess.StandardError
        }
        catch {
            $script:ExitCode = 1
            Remove-InstallerCancellationMarker $script:CancellationPath
            $script:CancellationPath = $null
            Write-EmergencyLog $_.Exception.Message | Out-Null
            Show-InstallerResult `
                -Heading "Installation could not start" `
                -Message (
                    "Setup could not start its background installer. Open the log " +
                    "for details."
                ) `
                -Succeeded $false
        }
    })

    $cancelButton.Add_Click({
        if ($null -ne $script:WorkerProcess -and
            $script:WorkerProcess.HasExited) {
            $cancelButton.Enabled = $false
            $progressStatusLabel.Text = "Finishing installation..."
            return
        }
        if ($null -ne $script:WorkerProcess) {
            $choice = [Windows.Forms.MessageBox]::Show(
                $form,
                "Cancel the active installation? You can run this installer again safely.",
                $form.Text,
                [Windows.Forms.MessageBoxButtons]::YesNo,
                [Windows.Forms.MessageBoxIcon]::Question
            )
            if ($choice -eq [Windows.Forms.DialogResult]::Yes) {
                try {
                    Request-InstallerCancellation $script:CancellationPath
                    $cancelButton.Enabled = $false
                    $progressStatusLabel.Text = (
                        "Cancelling safely. Setup is finishing cleanup..."
                    )
                }
                catch {
                    [Windows.Forms.MessageBox]::Show(
                        $form,
                        "Setup could not request cancellation. It will continue " +
                        "so the installation remains consistent.",
                        $form.Text,
                        [Windows.Forms.MessageBoxButtons]::OK,
                        [Windows.Forms.MessageBoxIcon]::Warning
                    ) | Out-Null
                    $cancelButton.Enabled = $true
                }
            }
        }
        else {
            $form.Close()
        }
    })

    $timer = New-Object Windows.Forms.Timer
    $timer.Interval = 350
    $timer.Add_Tick({
        if ($null -ne $script:WorkerProcess) {
            Read-InstallerProgressStream $script:WorkerStandardOutput
            Read-InstallerProgressStream $script:WorkerStandardError
            Show-InstallerProgressLines
        }

        if ($null -eq $script:WorkerProcess -or
            -not $script:WorkerProcess.HasExited -or
            -not $script:WorkerStandardOutput.Ended -or
            -not $script:WorkerStandardError.Ended) {
            return
        }

        $workerExitCode = $script:WorkerProcess.ExitCode
        $script:WorkerProcess.Dispose()
        $script:WorkerProcess = $null
        $script:WorkerStandardOutput = $null
        $script:WorkerStandardError = $null
        Remove-InstallerCancellationMarker $script:CancellationPath
        $script:CancellationPath = $null

        if ($workerExitCode -eq 0) {
            $script:ExitCode = 0
            $completionMessage = (
                "$($Contract.ProductName) is installed and ready to use. " +
                "Click Finish to close Setup."
            )
            if ($agentConnectionCheck.Checked) {
                $registrationStatusPath = [IO.Path]::Combine(
                    $script:ActiveInstallRoot, "agent-registration-status"
                )
                $registrationStatus = $null
                if (Test-Path -LiteralPath $registrationStatusPath -PathType Leaf) {
                    $registrationStatus = (
                        Get-Content -LiteralPath $registrationStatusPath -Raw
                    ).Trim()
                }
                if ($registrationStatus -eq "connected") {
                    $connectedClients = (
                        "ChatGPT desktop, Codex, and detected local agent apps"
                    )
                    $registrationReportPath = [IO.Path]::Combine(
                        $script:ActiveInstallRoot, "agent-registration.json"
                    )
                    if (Test-Path -LiteralPath $registrationReportPath -PathType Leaf) {
                        try {
                            $registrationReport = (
                                Get-Content -LiteralPath $registrationReportPath -Raw |
                                    ConvertFrom-Json
                            )
                            $registeredDisplayNames = @(
                                $registrationReport.results |
                                    Where-Object { $_.status -ne "failed" } |
                                    ForEach-Object { [string]$_.display_name }
                            )
                            if ($registeredDisplayNames.Count -gt 0) {
                                $connectedClients = $registeredDisplayNames -join ", "
                            }
                        }
                        catch {
                            Write-EmergencyLog (
                                "WARNING: Could not read agent registration summary: " +
                                $_.Exception.Message
                            ) | Out-Null
                        }
                    }
                    $completionMessage = (
                        "$($Contract.ProductName) is connected to $connectedClients. " +
                        "Restart ChatGPT desktop, Codex, and other listed apps, " +
                        "then ask them to use OpenHCS."
                    )
                }
                else {
                    $completionMessage = (
                        "$($Contract.ProductName) is installed, but one or more " +
                        "agent connections need attention. Open the log for details."
                    )
                }
            }
            Show-InstallerResult `
                -Heading "Installation complete" `
                -Message $completionMessage `
                -Succeeded $true
        }
        elseif ($workerExitCode -eq 2) {
            $script:ExitCode = 2
            Show-InstallerResult `
                -Heading "Installation cancelled" `
                -Message (
                    "No replacement was published. Run Setup again whenever " +
                    "you are ready."
                ) `
                -Succeeded $false
        }
        else {
            $script:ExitCode = 1
            Show-InstallerResult `
                -Heading "Installation failed" `
                -Message (
                    "Setup could not complete the installation. The existing " +
                    "installation, if any, was left in place."
                ) `
                -Succeeded $false
        }
    })
    $timer.Start()

    $form.Add_FormClosing({
        if ($null -ne $script:WorkerProcess -and
            -not $script:WorkerProcess.HasExited) {
            $_.Cancel = $true
            [Windows.Forms.MessageBox]::Show(
                $form,
                "Cancel the installation first, then close this window.",
                $form.Text,
                [Windows.Forms.MessageBoxButtons]::OK,
                [Windows.Forms.MessageBoxIcon]::Information
            ) | Out-Null
        }
    })

    $form.Add_Shown({
        Set-WizardPage "Welcome"
        $form.Activate()
    })
    $form.ShowDialog() | Out-Null
    $timer.Stop()
    $timer.Dispose()
    Remove-InstallerCancellationMarker $script:CancellationPath
    $script:CancellationPath = $null
    $form.Dispose()
    $installerLogo.Dispose()
    $installerIcon.Dispose()
    return $script:ExitCode
}

try {
    $installerContract = Read-InstallerContract
}
catch {
    $emergencyLog = Write-EmergencyLog $_.Exception.Message
    if (-not $Worker) {
        Add-Type -AssemblyName System.Windows.Forms
        [Windows.Forms.MessageBox]::Show(
            "This installer cannot continue because its shared contract is missing " +
            "or invalid.`r`n`r`n$($_.Exception.Message)`r`n`r`nLog: $emergencyLog",
            "Installer contract error",
            [Windows.Forms.MessageBoxButtons]::OK,
            [Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
    exit 2
}

if ($Worker) {
    if ([string]::IsNullOrWhiteSpace($InstallRoot) -or
        [string]::IsNullOrWhiteSpace($CancellationPath)) {
        Write-EmergencyLog (
            "Worker mode requires explicit installation and cancellation paths."
        ) | Out-Null
        exit 2
    }
    exit (Invoke-WorkerInstall $installerContract $InstallRoot $CancellationPath)
}

try {
    exit (Show-InstallerWindow `
        $installerContract $BrandIconPath $BrandLogoPath)
}
catch {
    $emergencyLog = Write-EmergencyLog $_.Exception.Message
    Add-Type -AssemblyName System.Windows.Forms
    [Windows.Forms.MessageBox]::Show(
        "The installer could not open.`r`n`r`n$($_.Exception.Message)" +
        "`r`n`r`nLog: $emergencyLog",
        "Installer error",
        [Windows.Forms.MessageBoxButtons]::OK,
        [Windows.Forms.MessageBoxIcon]::Error
    ) | Out-Null
    exit 1
}

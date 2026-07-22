[CmdletBinding()]
param(
    [switch]$Worker,
    [string]$InstallRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:SupportedContractSchema = "openhcs.installer.v1"
$script:LogPath = $null

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

    $urlsProperty = $contract.PSObject.Properties["uv_installer_urls"]
    if ($null -eq $urlsProperty -or $null -eq $urlsProperty.Value) {
        throw "Installer contract property 'uv_installer_urls' is required."
    }
    $windowsUrl = Get-RequiredTextProperty $urlsProperty.Value "windows"
    $parsedUrl = $null
    if (-not [Uri]::TryCreate($windowsUrl, [UriKind]::Absolute, [ref]$parsedUrl) -or
        $parsedUrl.Scheme -ne [Uri]::UriSchemeHttps -or
        -not [string]::Equals(
            $parsedUrl.IdnHost,
            "astral.sh",
            [StringComparison]::OrdinalIgnoreCase
        )) {
        throw "Installer contract Windows uv installer URL must use HTTPS on astral.sh."
    }

    return [PSCustomObject]@{
        ProductName = $productName
        PythonVersion = $pythonVersion
        PackageRequirement = $packageRequirement
        EntryPoint = $entryPoint
        UvInstallerUrl = $parsedUrl.AbsoluteUri
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

function Write-InstallLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    if ([string]::IsNullOrWhiteSpace($script:LogPath)) {
        throw "Installer log path was not initialized."
    }
    Add-Content -LiteralPath $script:LogPath -Encoding UTF8 -Value (
        "{0:u} {1}" -f [DateTime]::Now, $Message
    )
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$Description
    )

    Write-InstallLog "START: $Description"
    & $FilePath @ArgumentList 2>&1 | ForEach-Object {
        Write-InstallLog ([string]$_)
    }
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode."
    }
    Write-InstallLog "DONE: $Description"
}

function Get-StableLauncherPath {
    param(
        [Parameter(Mandatory = $true)][object]$Contract,
        [Parameter(Mandatory = $true)][string]$ResolvedInstallRoot
    )

    $launcherName = "Launch-{0}.ps1" -f ($Contract.ProductName -replace " ", "-")
    return [IO.Path]::Combine($ResolvedInstallRoot, $launcherName)
}

function Publish-LaunchAdapterAndShortcut {
    param(
        [Parameter(Mandatory = $true)][object]$Contract,
        [Parameter(Mandatory = $true)][string]$ResolvedInstallRoot,
        [Parameter(Mandatory = $true)][string]$EnvironmentName
    )

    if ($EnvironmentName -notmatch "^env-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{32}$") {
        throw "Internal environment name has an unsafe format."
    }

    $launcherPath = Get-StableLauncherPath $Contract $ResolvedInstallRoot
    $transactionId = [Guid]::NewGuid().ToString("N")
    $launcherCandidate = "$launcherPath.candidate-$transactionId"
    $launcherBackup = "$launcherPath.backup-$transactionId"
    $launcherLines = @(
        '$env:OPENHCS_CPU_ONLY = "true"',
        (
            '& (Join-Path $PSScriptRoot "environments\{0}\Scripts\{1}.exe")' -f
            $EnvironmentName, $Contract.EntryPoint
        ),
        'exit $LASTEXITCODE'
    )
    Set-Content -LiteralPath $launcherCandidate -Encoding UTF8 -Value $launcherLines

    $desktop = [Environment]::GetFolderPath("DesktopDirectory")
    if ([string]::IsNullOrWhiteSpace($desktop)) {
        throw "Windows did not provide a user Desktop folder."
    }
    $shortcutPath = [IO.Path]::Combine($desktop, "$($Contract.ProductName).lnk")
    $shortcutCandidate = "$shortcutPath.candidate-$transactionId.lnk"
    $shortcutBackup = "$shortcutPath.backup-$transactionId.lnk"
    $powerShellExecutable = [IO.Path]::Combine($PSHOME, "powershell.exe")

    $shell = New-Object -ComObject WScript.Shell
    try {
        $shortcut = $shell.CreateShortcut($shortcutCandidate)
        try {
            $shortcut.TargetPath = $powerShellExecutable
            $shortcut.Arguments = (
                '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}"' -f
                $launcherPath
            )
            $shortcut.WorkingDirectory = $ResolvedInstallRoot
            $shortcut.Description = "Launch $($Contract.ProductName)"
            $shortcut.IconLocation = "$powerShellExecutable,0"
            $shortcut.Save()
        }
        finally {
            if ($null -ne $shortcut) {
                [Runtime.InteropServices.Marshal]::FinalReleaseComObject($shortcut) |
                    Out-Null
            }
        }
    }
    finally {
        [Runtime.InteropServices.Marshal]::FinalReleaseComObject($shell) | Out-Null
    }

    $launcherBackedUp = $false
    $launcherPublished = $false
    $shortcutBackedUp = $false
    $shortcutPublished = $false
    try {
        if (Test-Path -LiteralPath $launcherPath -PathType Leaf) {
            [IO.File]::Replace(
                $launcherCandidate, $launcherPath, $launcherBackup, $true
            )
            $launcherBackedUp = $true
        }
        else {
            [IO.File]::Move($launcherCandidate, $launcherPath)
        }
        $launcherPublished = $true

        if (Test-Path -LiteralPath $shortcutPath -PathType Leaf) {
            [IO.File]::Replace(
                $shortcutCandidate, $shortcutPath, $shortcutBackup, $true
            )
            $shortcutBackedUp = $true
        }
        else {
            [IO.File]::Move($shortcutCandidate, $shortcutPath)
        }
        $shortcutPublished = $true
    }
    catch {
        if ($shortcutBackedUp -and (Test-Path -LiteralPath $shortcutBackup)) {
            if (Test-Path -LiteralPath $shortcutPath) {
                [IO.File]::Replace($shortcutBackup, $shortcutPath, $null, $true)
            }
            else {
                [IO.File]::Move($shortcutBackup, $shortcutPath)
            }
        }
        elseif ($shortcutPublished -and (Test-Path -LiteralPath $shortcutPath)) {
            Remove-Item -LiteralPath $shortcutPath -Force -ErrorAction SilentlyContinue
        }
        if ($launcherBackedUp -and (Test-Path -LiteralPath $launcherBackup)) {
            if (Test-Path -LiteralPath $launcherPath) {
                [IO.File]::Replace($launcherBackup, $launcherPath, $null, $true)
            }
            else {
                [IO.File]::Move($launcherBackup, $launcherPath)
            }
        }
        elseif ($launcherPublished -and (Test-Path -LiteralPath $launcherPath)) {
            Remove-Item -LiteralPath $launcherPath -Force -ErrorAction SilentlyContinue
        }
        throw
    }
    finally {
        foreach ($temporaryPath in @(
            $launcherCandidate,
            $shortcutCandidate,
            $launcherBackup,
            $shortcutBackup
        )) {
            if (Test-Path -LiteralPath $temporaryPath) {
                Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
            }
        }
    }
    Write-InstallLog "Desktop shortcut: $shortcutPath"
}

function Remove-SupersededEnvironments {
    param(
        [Parameter(Mandatory = $true)][string]$EnvironmentsRoot,
        [Parameter(Mandatory = $true)][string]$CurrentEnvironmentPath
    )

    $currentFullPath = [IO.Path]::GetFullPath($CurrentEnvironmentPath)
    Get-ChildItem -LiteralPath $EnvironmentsRoot -Directory | ForEach-Object {
        if ([IO.Path]::GetFullPath($_.FullName) -eq $currentFullPath) {
            return
        }
        try {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force
            Write-InstallLog "Removed superseded environment: $($_.FullName)"
        }
        catch {
            Write-InstallLog (
                "WARNING: Could not remove superseded environment '$($_.FullName)': " +
                $_.Exception.Message
            )
        }
    }
}

function Invoke-WorkerInstall {
    param(
        [Parameter(Mandatory = $true)][object]$Contract,
        [Parameter(Mandatory = $true)][string]$RequestedInstallRoot
    )

    $temporaryUvInstaller = $null
    $newEnvironmentPath = $null
    try {
        $resolvedRoot = Resolve-InstallRoot $RequestedInstallRoot
        [IO.Directory]::CreateDirectory($resolvedRoot) | Out-Null
        $script:LogPath = [IO.Path]::Combine($resolvedRoot, "installer.log")
        Set-Content -LiteralPath $script:LogPath -Encoding UTF8 -Value (
            "{0:u} Starting $($Contract.ProductName) installation." -f [DateTime]::Now
        )

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

        Write-InstallLog "Downloading the official uv installer over HTTPS."
        $temporaryUvInstaller = [IO.Path]::Combine(
            [IO.Path]::GetTempPath(),
            "openhcs-uv-installer-$([Guid]::NewGuid().ToString('N')).ps1"
        )
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri $Contract.UvInstallerUrl `
            -OutFile $temporaryUvInstaller

        $previousUvInstallDir = $env:UV_INSTALL_DIR
        $previousUvNoModifyPath = $env:UV_NO_MODIFY_PATH
        try {
            $env:UV_INSTALL_DIR = $uvInstallRoot
            $env:UV_NO_MODIFY_PATH = "1"
            Invoke-LoggedCommand `
                -FilePath ([IO.Path]::Combine($PSHOME, "powershell.exe")) `
                -ArgumentList @(
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", $temporaryUvInstaller
                ) `
                -Description "Install uv"
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
            throw "The uv installer completed without creating '$uvExecutable'."
        }

        Invoke-LoggedCommand -FilePath $uvExecutable -ArgumentList @(
            "--no-config", "python", "install", $Contract.PythonVersion
        ) -Description "Install managed Python $($Contract.PythonVersion)"

        Invoke-LoggedCommand -FilePath $uvExecutable -ArgumentList @(
            "--no-config", "venv", "--python",
            $Contract.PythonVersion, $newEnvironmentPath
        ) -Description "Create a candidate virtual environment"

        Invoke-LoggedCommand -FilePath $uvExecutable -ArgumentList @(
            "--no-config", "pip", "install", "--python", $newEnvironmentPath,
            "--upgrade", $Contract.PackageRequirement
        ) -Description "Install $($Contract.PackageRequirement)"

        Invoke-LoggedCommand -FilePath $uvExecutable -ArgumentList @(
            "--no-config", "pip", "check", "--python", $newEnvironmentPath
        ) -Description "Verify installed dependencies"

        $entryExecutable = [IO.Path]::Combine(
            $newEnvironmentPath, "Scripts", "$($Contract.EntryPoint).exe"
        )
        if (-not (Test-Path -LiteralPath $entryExecutable -PathType Leaf)) {
            throw "Installation did not create the declared GUI entry point."
        }

        Publish-LaunchAdapterAndShortcut `
            $Contract $resolvedRoot $environmentName
        Write-InstallLog "SUCCESS: Installation completed."
        Remove-SupersededEnvironments $environmentsRoot $newEnvironmentPath
        return 0
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
        if ($null -ne $newEnvironmentPath -and
            (Test-Path -LiteralPath $newEnvironmentPath)) {
            try {
                Remove-Item -LiteralPath $newEnvironmentPath -Recurse -Force
            }
            catch {
                try {
                    Write-InstallLog (
                        "WARNING: Could not remove failed candidate environment: " +
                        $_.Exception.Message
                    )
                }
                catch {
                    # The durable failure was already recorded above.
                }
            }
        }
        return 1
    }
    finally {
        if ($null -ne $temporaryUvInstaller -and
            (Test-Path -LiteralPath $temporaryUvInstaller)) {
            Remove-Item -LiteralPath $temporaryUvInstaller -Force -ErrorAction SilentlyContinue
        }
    }
}

function Start-InstallerWorker {
    param([Parameter(Mandatory = $true)][string]$ResolvedInstallRoot)

    $escapedScriptPath = $PSCommandPath.Replace("'", "''")
    $escapedInstallRoot = $ResolvedInstallRoot.Replace("'", "''")
    $workerCommand = (
        "& '{0}' -Worker -InstallRoot '{1}'" -f
        $escapedScriptPath, $escapedInstallRoot
    )
    $encodedCommand = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($workerCommand)
    )
    $startInfo = New-Object Diagnostics.ProcessStartInfo
    $startInfo.FileName = [IO.Path]::Combine($PSHOME, "powershell.exe")
    $startInfo.Arguments = (
        "-NoProfile -ExecutionPolicy Bypass -EncodedCommand {0}" -f $encodedCommand
    )
    $startInfo.WorkingDirectory = $PSScriptRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    return [Diagnostics.Process]::Start($startInfo)
}

function Stop-InstallerWorker {
    param([Parameter(Mandatory = $true)][Diagnostics.Process]$Process)

    if ($Process.HasExited) {
        return
    }
    $taskkill = [IO.Path]::Combine($env:SystemRoot, "System32", "taskkill.exe")
    & $taskkill "/PID" ([string]$Process.Id) "/T" "/F" 2>$null | Out-Null
    $Process.WaitForExit(5000) | Out-Null
    if (-not $Process.HasExited) {
        $Process.Kill()
    }
}

function Show-InstallerWindow {
    param([Parameter(Mandatory = $true)][object]$Contract)

    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [Windows.Forms.Application]::EnableVisualStyles()

    $localData = [Environment]::GetFolderPath("LocalApplicationData")
    if ([string]::IsNullOrWhiteSpace($localData)) {
        throw "Windows did not provide a user Local Application Data folder."
    }
    $defaultRoot = [IO.Path]::Combine($localData, $Contract.ProductName)
    $script:LogPath = [IO.Path]::Combine($defaultRoot, "installer.log")
    $script:WorkerProcess = $null
    $script:CancelRequested = $false
    $script:ExitCode = 0

    $form = New-Object Windows.Forms.Form
    $form.Text = "$($Contract.ProductName) Installer"
    $form.StartPosition = "CenterScreen"
    $form.ClientSize = New-Object Drawing.Size(660, 440)
    $form.MinimumSize = New-Object Drawing.Size(620, 420)
    $form.MaximizeBox = $false

    $title = New-Object Windows.Forms.Label
    $title.Text = "Install $($Contract.ProductName)"
    $title.Font = New-Object Drawing.Font("Segoe UI", 17, [Drawing.FontStyle]::Bold)
    $title.AutoSize = $true
    $title.Location = New-Object Drawing.Point(20, 18)
    $form.Controls.Add($title)

    $summary = New-Object Windows.Forms.Label
    $summary.Text = (
        "This installs a private Python $($Contract.PythonVersion) environment " +
        "and the desktop application for your Windows account."
    )
    $summary.Location = New-Object Drawing.Point(23, 58)
    $summary.Size = New-Object Drawing.Size(610, 42)
    $summary.Anchor = "Top,Left,Right"
    $form.Controls.Add($summary)

    $folderLabel = New-Object Windows.Forms.Label
    $folderLabel.Text = "Installation folder"
    $folderLabel.Location = New-Object Drawing.Point(23, 106)
    $folderLabel.AutoSize = $true
    $form.Controls.Add($folderLabel)

    $folderText = New-Object Windows.Forms.TextBox
    $folderText.Text = $defaultRoot
    $folderText.Location = New-Object Drawing.Point(26, 128)
    $folderText.Size = New-Object Drawing.Size(500, 25)
    $folderText.Anchor = "Top,Left,Right"
    $form.Controls.Add($folderText)

    $browseButton = New-Object Windows.Forms.Button
    $browseButton.Text = "Browse..."
    $browseButton.Location = New-Object Drawing.Point(535, 126)
    $browseButton.Size = New-Object Drawing.Size(96, 29)
    $browseButton.Anchor = "Top,Right"
    $form.Controls.Add($browseButton)

    $statusLabel = New-Object Windows.Forms.Label
    $statusLabel.Text = "Ready to install. Existing installations are safely refreshed."
    $statusLabel.Location = New-Object Drawing.Point(23, 168)
    $statusLabel.Size = New-Object Drawing.Size(610, 24)
    $statusLabel.Anchor = "Top,Left,Right"
    $form.Controls.Add($statusLabel)

    $logBox = New-Object Windows.Forms.TextBox
    $logBox.Multiline = $true
    $logBox.ReadOnly = $true
    $logBox.ScrollBars = "Vertical"
    $logBox.BackColor = [Drawing.Color]::White
    $logBox.Location = New-Object Drawing.Point(26, 196)
    $logBox.Size = New-Object Drawing.Size(605, 172)
    $logBox.Anchor = "Top,Bottom,Left,Right"
    $form.Controls.Add($logBox)

    $installButton = New-Object Windows.Forms.Button
    $installButton.Text = "Install / Update"
    $installButton.Location = New-Object Drawing.Point(380, 384)
    $installButton.Size = New-Object Drawing.Size(122, 34)
    $installButton.Anchor = "Bottom,Right"
    $form.AcceptButton = $installButton
    $form.Controls.Add($installButton)

    $cancelButton = New-Object Windows.Forms.Button
    $cancelButton.Text = "Close"
    $cancelButton.Location = New-Object Drawing.Point(511, 384)
    $cancelButton.Size = New-Object Drawing.Size(120, 34)
    $cancelButton.Anchor = "Bottom,Right"
    $form.Controls.Add($cancelButton)

    $openLogButton = New-Object Windows.Forms.Button
    $openLogButton.Text = "Open log"
    $openLogButton.Location = New-Object Drawing.Point(26, 384)
    $openLogButton.Size = New-Object Drawing.Size(100, 34)
    $openLogButton.Anchor = "Bottom,Left"
    $form.Controls.Add($openLogButton)

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

    $installButton.Add_Click({
        try {
            $resolvedRoot = Resolve-InstallRoot $folderText.Text
        }
        catch {
            [Windows.Forms.MessageBox]::Show(
                $form, $_.Exception.Message, $form.Text,
                [Windows.Forms.MessageBoxButtons]::OK,
                [Windows.Forms.MessageBoxIcon]::Warning
            ) | Out-Null
            return
        }

        $existingLauncher = Get-StableLauncherPath $Contract $resolvedRoot
        if (Test-Path -LiteralPath $existingLauncher -PathType Leaf) {
            $choice = [Windows.Forms.MessageBox]::Show(
                $form,
                "Close $($Contract.ProductName) before continuing. The dedicated " +
                "environment will be refreshed to the current release. Continue?",
                "Update $($Contract.ProductName)",
                [Windows.Forms.MessageBoxButtons]::YesNo,
                [Windows.Forms.MessageBoxIcon]::Question
            )
            if ($choice -ne [Windows.Forms.DialogResult]::Yes) {
                return
            }
        }

        $script:LogPath = [IO.Path]::Combine($resolvedRoot, "installer.log")
        $script:CancelRequested = $false
        $script:ExitCode = 0
        $logBox.Clear()
        $statusLabel.Text = "Installing. You can cancel without closing this window."
        $installButton.Enabled = $false
        $browseButton.Enabled = $false
        $folderText.Enabled = $false
        $cancelButton.Text = "Cancel install"
        try {
            $script:WorkerProcess = Start-InstallerWorker $resolvedRoot
        }
        catch {
            $script:ExitCode = 1
            $statusLabel.Text = "Could not start the installer worker."
            $installButton.Enabled = $true
            $browseButton.Enabled = $true
            $folderText.Enabled = $true
            $cancelButton.Text = "Close"
            $emergencyPath = Write-EmergencyLog $_.Exception.Message
            [Windows.Forms.MessageBox]::Show(
                $form,
                "The installer could not start. Log: $emergencyPath",
                $form.Text,
                [Windows.Forms.MessageBoxButtons]::OK,
                [Windows.Forms.MessageBoxIcon]::Error
            ) | Out-Null
        }
    })

    $cancelButton.Add_Click({
        if ($null -ne $script:WorkerProcess -and
            -not $script:WorkerProcess.HasExited) {
            $choice = [Windows.Forms.MessageBox]::Show(
                $form,
                "Cancel the active installation? You can run this installer again safely.",
                $form.Text,
                [Windows.Forms.MessageBoxButtons]::YesNo,
                [Windows.Forms.MessageBoxIcon]::Question
            )
            if ($choice -eq [Windows.Forms.DialogResult]::Yes) {
                $script:CancelRequested = $true
                $script:ExitCode = 2
                Stop-InstallerWorker $script:WorkerProcess
                $statusLabel.Text = "Installation cancelled. Run Install / Update to retry."
            }
        }
        else {
            $form.Close()
        }
    })

    $timer = New-Object Windows.Forms.Timer
    $timer.Interval = 350
    $timer.Add_Tick({
        if (Test-Path -LiteralPath $script:LogPath -PathType Leaf) {
            try {
                $logBox.Lines = @(Get-Content -LiteralPath $script:LogPath -Tail 14)
                $logBox.SelectionStart = $logBox.TextLength
                $logBox.ScrollToCaret()
            }
            catch {
                # A concurrent append can briefly make the tail unavailable.
            }
        }

        if ($null -eq $script:WorkerProcess -or
            -not $script:WorkerProcess.HasExited) {
            return
        }

        $workerExitCode = $script:WorkerProcess.ExitCode
        $script:WorkerProcess.Dispose()
        $script:WorkerProcess = $null
        $installButton.Enabled = $true
        $browseButton.Enabled = $true
        $folderText.Enabled = $true
        $cancelButton.Text = "Close"

        if ($script:CancelRequested) {
            $script:CancelRequested = $false
            return
        }
        if ($workerExitCode -eq 0) {
            $script:ExitCode = 0
            $statusLabel.Text = "Installation complete. A desktop shortcut is ready."
            $choice = [Windows.Forms.MessageBox]::Show(
                $form,
                "$($Contract.ProductName) is installed. Launch it now?",
                $form.Text,
                [Windows.Forms.MessageBoxButtons]::YesNo,
                [Windows.Forms.MessageBoxIcon]::Information
            )
            if ($choice -eq [Windows.Forms.DialogResult]::Yes) {
                $shortcutPath = [IO.Path]::Combine(
                    [Environment]::GetFolderPath("DesktopDirectory"),
                    "$($Contract.ProductName).lnk"
                )
                Start-Process -FilePath $shortcutPath
            }
        }
        else {
            $script:ExitCode = 1
            $statusLabel.Text = "Installation failed. The previous lines identify the failed step."
            $failureLog = $script:LogPath
            if (-not (Test-Path -LiteralPath $failureLog -PathType Leaf)) {
                $failureLog = Get-EmergencyLogPath
            }
            [Windows.Forms.MessageBox]::Show(
                $form,
                "Installation failed. Review the durable log at:`r`n$failureLog",
                $form.Text,
                [Windows.Forms.MessageBoxButtons]::OK,
                [Windows.Forms.MessageBoxIcon]::Error
            ) | Out-Null
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

    $form.Add_Shown({ $form.Activate() })
    $form.ShowDialog() | Out-Null
    $timer.Stop()
    $timer.Dispose()
    $form.Dispose()
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
    if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
        Write-EmergencyLog "Worker mode requires an explicit installation root." | Out-Null
        exit 2
    }
    exit (Invoke-WorkerInstall $installerContract $InstallRoot)
}

try {
    exit (Show-InstallerWindow $installerContract)
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

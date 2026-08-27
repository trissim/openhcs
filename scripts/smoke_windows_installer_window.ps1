param(
    [Parameter(Mandatory = $true)][string]$LauncherPath,
    [Parameter(Mandatory = $true)][string]$ExpectedTitle,
    [Parameter(Mandatory = $true)][string]$EvidenceDirectory,
    [Parameter(Mandatory = $true)][string]$CompletionLogPath
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$resolvedLauncher = (Resolve-Path -LiteralPath $LauncherPath).Path
$resolvedEvidenceDirectory = [IO.Path]::GetFullPath($EvidenceDirectory)
$resolvedCompletionLog = [IO.Path]::GetFullPath($CompletionLogPath)
[IO.Directory]::CreateDirectory($resolvedEvidenceDirectory) | Out-Null

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName UIAutomationClient
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class OpenHCSInstallerWindowProbe
{
    [StructLayout(LayoutKind.Sequential)]
    public struct WindowRectangle
    {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    [DllImport("user32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool GetWindowRect(
        IntPtr window,
        out WindowRectangle rectangle
    );
}
'@

function Save-InstallerWindowEvidence {
    param(
        [Parameter(Mandatory = $true)][string]$EvidenceName,
        [Parameter(Mandatory = $true)][string]$ScreenshotName,
        [Parameter(Mandatory = $true)][string]$Stage
    )

    $rectangle = New-Object OpenHCSInstallerWindowProbe+WindowRectangle
    if (-not [OpenHCSInstallerWindowProbe]::GetWindowRect(
        $windowProcess.MainWindowHandle,
        [ref]$rectangle
    )) {
        throw "Windows could not read the native installer window bounds."
    }
    $width = $rectangle.Right - $rectangle.Left
    $height = $rectangle.Bottom - $rectangle.Top
    if ($width -lt 600 -or $height -lt 400) {
        throw (
            "The native installer window is unexpectedly small: " +
            "${width}x${height}."
        )
    }

    $screenshotPath = Join-Path `
        $resolvedEvidenceDirectory `
        "$ScreenshotName.png"
    $bitmap = New-Object Drawing.Bitmap($width, $height)
    $graphics = [Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CopyFromScreen(
            $rectangle.Left,
            $rectangle.Top,
            0,
            0,
            (New-Object Drawing.Size($width, $height))
        )
        $bitmap.Save($screenshotPath, [Drawing.Imaging.ImageFormat]::Png)
    }
    finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }
    if ((Get-Item -LiteralPath $screenshotPath).Length -le 0) {
        throw "The native installer screenshot is empty."
    }

    [pscustomobject]@{
        stage = $Stage
        launcher_process_id = $launcherProcess.Id
        process_id = $windowProcess.Id
        title = $windowProcess.MainWindowTitle
        left = $rectangle.Left
        top = $rectangle.Top
        width = $width
        height = $height
        screenshot_path = $screenshotPath
    } |
        ConvertTo-Json |
        Set-Content -LiteralPath (
            Join-Path $resolvedEvidenceDirectory "$EvidenceName.json"
        ) -Encoding UTF8
}

function Wait-InstallerAutomationElement {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][DateTime]$Deadline
    )

    $nameCondition = [Windows.Automation.PropertyCondition]::new(
        [Windows.Automation.AutomationElement]::NameProperty,
        $Name
    )
    while ([DateTime]::UtcNow -lt $Deadline) {
        if ($windowProcess.HasExited) {
            throw "The native installer exited before showing '$Name'."
        }
        $root = [Windows.Automation.AutomationElement]::FromHandle(
            $windowProcess.MainWindowHandle
        )
        $element = $root.FindFirst(
            [Windows.Automation.TreeScope]::Descendants,
            $nameCondition
        )
        if ($null -ne $element -and
            $element.Current.IsEnabled -and
            -not $element.Current.IsOffscreen) {
            return $element
        }
        Start-Sleep -Milliseconds 250
    }
    throw "The native installer did not show '$Name' before its deadline."
}

function Invoke-InstallerButton {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][DateTime]$Deadline
    )

    $button = Wait-InstallerAutomationElement -Name $Name -Deadline $Deadline
    $invokePattern = $button.GetCurrentPattern(
        [Windows.Automation.InvokePattern]::Pattern
    )
    $invokePattern.Invoke()
}

function Wait-InstallerLogLine {
    param(
        [Parameter(Mandatory = $true)][string]$ExpectedLine,
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][DateTime]$Deadline
    )

    while ([DateTime]::UtcNow -lt $Deadline) {
        if ($windowProcess.HasExited) {
            throw "The native installer exited before $Description."
        }
        if (Test-Path -LiteralPath $resolvedCompletionLog -PathType Leaf) {
            $logText = [IO.File]::ReadAllText($resolvedCompletionLog)
            if ($logText.Contains($ExpectedLine)) {
                return
            }
        }
        Start-Sleep -Milliseconds 250
    }
    throw "The native installer did not reach $Description within 20 minutes."
}

$launcherProcess = Start-Process -FilePath $resolvedLauncher -PassThru
$windowProcess = $null
$windowDeadline = [DateTime]::UtcNow.AddSeconds(30)
try {
    while ([DateTime]::UtcNow -lt $windowDeadline) {
        $childProcessRecords = @(
            Get-CimInstance Win32_Process -Filter (
                "ParentProcessId = {0}" -f $launcherProcess.Id
            )
        )
        foreach ($childProcessRecord in $childProcessRecords) {
            $candidateProcess = Get-Process `
                -Id ([int]$childProcessRecord.ProcessId) `
                -ErrorAction SilentlyContinue
            if ($null -eq $candidateProcess) {
                continue
            }
            if ($candidateProcess.MainWindowHandle -ne [IntPtr]::Zero -and
                $candidateProcess.MainWindowTitle -eq $ExpectedTitle) {
                $windowProcess = $candidateProcess
                break
            }
            $candidateProcess.Dispose()
        }
        if ($null -ne $windowProcess) {
            break
        }
        if ($launcherProcess.HasExited) {
            throw (
                "The native installer exited before showing '$ExpectedTitle' " +
                "with code $($launcherProcess.ExitCode)."
            )
        }
        Start-Sleep -Milliseconds 250
    }
    if ($null -eq $windowProcess) {
        throw "The native installer did not show '$ExpectedTitle' within 30 seconds."
    }

    Save-InstallerWindowEvidence `
        -EvidenceName "installer-window" `
        -ScreenshotName "installer-welcome" `
        -Stage "welcome"

    $installationDeadline = [DateTime]::UtcNow.AddMinutes(20)
    Invoke-InstallerButton -Name "Next >" -Deadline $installationDeadline
    $null = Wait-InstallerAutomationElement `
        -Name "Installation options" `
        -Deadline $installationDeadline
    Invoke-InstallerButton -Name "Next >" -Deadline $installationDeadline

    Wait-InstallerLogLine `
        -ExpectedLine "START:" `
        -Description "visible installation progress" `
        -Deadline $installationDeadline
    Start-Sleep -Milliseconds 500
    Save-InstallerWindowEvidence `
        -EvidenceName "installer-progress" `
        -ScreenshotName "installer-progress" `
        -Stage "progress"

    Wait-InstallerLogLine `
        -ExpectedLine "SUCCESS: Installation completed." `
        -Description "successful completion" `
        -Deadline $installationDeadline
    $null = Wait-InstallerAutomationElement `
        -Name "Installation complete" `
        -Deadline $installationDeadline
    Save-InstallerWindowEvidence `
        -EvidenceName "installer-finished" `
        -ScreenshotName "installer-finished" `
        -Stage "finished"

    if (-not $windowProcess.CloseMainWindow()) {
        throw "Windows could not close the completed native installer window."
    }
    if (-not $windowProcess.WaitForExit(15000)) {
        throw "The completed native installer window did not close cleanly."
    }
    if (-not $launcherProcess.WaitForExit(15000)) {
        throw "The native installer launcher did not finish after its window closed."
    }
    if ($launcherProcess.ExitCode -ne 0) {
        throw (
            "The native installer launcher exited with " +
            "$($launcherProcess.ExitCode) after its completed window closed."
        )
    }
}
finally {
    if ($null -ne $windowProcess -and -not $windowProcess.HasExited) {
        $windowProcess.Kill()
        $windowProcess.WaitForExit()
    }
    if (-not $launcherProcess.HasExited) {
        $launcherProcess.Kill()
        $launcherProcess.WaitForExit()
    }
    $launcherProcess.Dispose()
}

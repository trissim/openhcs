param(
    [Parameter(Mandatory = $true)][string]$LauncherPath,
    [Parameter(Mandatory = $true)][string]$ExpectedTitle,
    [Parameter(Mandatory = $true)][string]$InstallRoot,
    [Parameter(Mandatory = $true)][string]$EvidenceDirectory,
    [Parameter(Mandatory = $true)][string]$CompletionLogPath
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$resolvedLauncher = (Resolve-Path -LiteralPath $LauncherPath).Path
$resolvedEvidenceDirectory = [IO.Path]::GetFullPath($EvidenceDirectory)
$resolvedCompletionLog = [IO.Path]::GetFullPath($CompletionLogPath)
$resolvedInstallRoot = [IO.Path]::GetFullPath($InstallRoot)
[IO.Directory]::CreateDirectory($resolvedEvidenceDirectory) | Out-Null

Add-Type -AssemblyName System.Drawing
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;

public static class OpenHCSInstallerWindowProbe
{
    private const uint ButtonClick = 0x00F5;

    private delegate bool EnumerateWindowCallback(IntPtr window, IntPtr state);

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

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool EnumChildWindows(
        IntPtr parent,
        EnumerateWindowCallback callback,
        IntPtr state
    );

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetWindowTextLengthW(IntPtr window);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetWindowTextW(
        IntPtr window,
        StringBuilder text,
        int maximumCount
    );

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool IsWindowEnabled(IntPtr window);

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool IsWindowVisible(IntPtr window);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern IntPtr SendMessageW(
        IntPtr window,
        uint message,
        IntPtr wordParameter,
        IntPtr longParameter
    );

    public static bool HasVisibleChildText(IntPtr parent, string expectedText)
    {
        return FindUniqueVisibleChild(parent, expectedText) != IntPtr.Zero;
    }

    public static bool ClickVisibleChildByText(
        IntPtr parent,
        string expectedText
    )
    {
        IntPtr child = FindUniqueVisibleChild(parent, expectedText);
        if (child == IntPtr.Zero)
        {
            return false;
        }
        SendMessageW(child, ButtonClick, IntPtr.Zero, IntPtr.Zero);
        return true;
    }

    private static IntPtr FindUniqueVisibleChild(
        IntPtr parent,
        string expectedText
    )
    {
        IntPtr match = IntPtr.Zero;
        int matchCount = 0;
        EnumChildWindows(
            parent,
            delegate(IntPtr child, IntPtr state)
            {
                if (!IsWindowVisible(child) || !IsWindowEnabled(child))
                {
                    return true;
                }
                int textLength = GetWindowTextLengthW(child);
                if (textLength != expectedText.Length)
                {
                    return true;
                }
                StringBuilder text = new StringBuilder(textLength + 1);
                GetWindowTextW(child, text, text.Capacity);
                if (string.Equals(
                    text.ToString(),
                    expectedText,
                    StringComparison.Ordinal
                ))
                {
                    match = child;
                    matchCount++;
                }
                return true;
            },
            IntPtr.Zero
        );
        return matchCount == 1 ? match : IntPtr.Zero;
    }
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

function Wait-InstallerVisibleText {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][DateTime]$Deadline
    )

    while ([DateTime]::UtcNow -lt $Deadline) {
        if ($windowProcess.HasExited) {
            throw "The native installer exited before showing '$Name'."
        }
        if ([OpenHCSInstallerWindowProbe]::HasVisibleChildText(
            $windowProcess.MainWindowHandle,
            $Name
        )) {
            return
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

    Wait-InstallerVisibleText -Name $Name -Deadline $Deadline
    if (-not [OpenHCSInstallerWindowProbe]::ClickVisibleChildByText(
        $windowProcess.MainWindowHandle,
        $Name
    )) {
        throw "The native installer could not activate '$Name'."
    }
}

function Wait-InstallerLogLine {
    param(
        [Parameter(Mandatory = $true)][string]$ExpectedLine,
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][DateTime]$Deadline,
        [Parameter(Mandatory = $true)][string]$TimeoutDescription
    )

    while ([DateTime]::UtcNow -lt $Deadline) {
        if ($windowProcess.HasExited) {
            throw "The native installer exited before $Description."
        }
        if (Test-Path -LiteralPath $resolvedCompletionLog -PathType Leaf) {
            $logStream = $null
            $logReader = $null
            $logText = $null
            try {
                $logStream = [IO.File]::Open(
                    $resolvedCompletionLog,
                    [IO.FileMode]::Open,
                    [IO.FileAccess]::Read,
                    [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
                )
                $logReader = [IO.StreamReader]::new($logStream)
                $logText = $logReader.ReadToEnd()
            }
            catch [IO.IOException] {
                $logText = $null
            }
            finally {
                if ($null -ne $logReader) {
                    $logReader.Dispose()
                }
                elseif ($null -ne $logStream) {
                    $logStream.Dispose()
                }
            }
            if ($null -ne $logText -and $logText.Contains($ExpectedLine)) {
                return
            }
        }
        Start-Sleep -Milliseconds 250
    }
    throw (
        "The native installer did not reach $Description within " +
        "$TimeoutDescription."
    )
}

$launcherStartInfo = [Diagnostics.ProcessStartInfo]::new()
$launcherStartInfo.FileName = $resolvedLauncher
$launcherStartInfo.UseShellExecute = $false
[void]$launcherStartInfo.ArgumentList.Add("-InstallRoot")
[void]$launcherStartInfo.ArgumentList.Add($resolvedInstallRoot)
$launcherProcess = [Diagnostics.Process]::Start($launcherStartInfo)
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

    $interactionDeadline = [DateTime]::UtcNow.AddSeconds(30)
    Invoke-InstallerButton -Name "Next >" -Deadline $interactionDeadline
    Wait-InstallerVisibleText `
        -Name "Installation options" `
        -Deadline $interactionDeadline
    Wait-InstallerVisibleText `
        -Name $resolvedInstallRoot `
        -Deadline $interactionDeadline
    Invoke-InstallerButton -Name "Next >" -Deadline $interactionDeadline

    $progressDeadline = [DateTime]::UtcNow.AddSeconds(30)
    Wait-InstallerLogLine `
        -ExpectedLine "START:" `
        -Description "visible installation progress" `
        -Deadline $progressDeadline `
        -TimeoutDescription "30 seconds"
    Start-Sleep -Milliseconds 500
    Save-InstallerWindowEvidence `
        -EvidenceName "installer-progress" `
        -ScreenshotName "installer-progress" `
        -Stage "progress"

    $installationDeadline = [DateTime]::UtcNow.AddMinutes(20)
    Wait-InstallerLogLine `
        -ExpectedLine "SUCCESS: Installation completed." `
        -Description "successful completion" `
        -Deadline $installationDeadline `
        -TimeoutDescription "20 minutes"
    Wait-InstallerVisibleText `
        -Name "Installation complete" `
        -Deadline $installationDeadline
    Wait-InstallerVisibleText `
        -Name "Open log" `
        -Deadline $installationDeadline
    Wait-InstallerVisibleText `
        -Name "Finish" `
        -Deadline $installationDeadline
    Save-InstallerWindowEvidence `
        -EvidenceName "installer-finished" `
        -ScreenshotName "installer-finished" `
        -Stage "finished"
    Copy-Item `
        -LiteralPath $resolvedCompletionLog `
        -Destination (Join-Path $resolvedEvidenceDirectory "installer.log")

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
catch {
    $_ | Out-String | Set-Content -LiteralPath (
        Join-Path $resolvedEvidenceDirectory "installer-smoke-error.txt"
    ) -Encoding UTF8
    if ($null -ne $windowProcess -and -not $windowProcess.HasExited) {
        try {
            Save-InstallerWindowEvidence `
                -EvidenceName "installer-failure" `
                -ScreenshotName "installer-failure" `
                -Stage "failure"
        }
        catch {
            $_ | Out-String | Set-Content -LiteralPath (
                Join-Path $resolvedEvidenceDirectory "installer-capture-error.txt"
            ) -Encoding UTF8
        }
    }
    throw
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

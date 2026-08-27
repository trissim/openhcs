param(
    [Parameter(Mandatory = $true)][string]$LauncherPath,
    [Parameter(Mandatory = $true)][string]$ExpectedTitle,
    [Parameter(Mandatory = $true)][string]$EvidenceDirectory
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$resolvedLauncher = (Resolve-Path -LiteralPath $LauncherPath).Path
$resolvedEvidenceDirectory = [IO.Path]::GetFullPath($EvidenceDirectory)
[IO.Directory]::CreateDirectory($resolvedEvidenceDirectory) | Out-Null

Add-Type -AssemblyName System.Drawing
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

    $screenshotPath = Join-Path $resolvedEvidenceDirectory "installer-welcome.png"
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
            Join-Path $resolvedEvidenceDirectory "installer-window.json"
        ) -Encoding UTF8

    if (-not $windowProcess.CloseMainWindow()) {
        throw "Windows could not close the native installer welcome window."
    }
    if (-not $windowProcess.WaitForExit(15000)) {
        throw "The native installer welcome window did not close cleanly."
    }
    if (-not $launcherProcess.WaitForExit(15000)) {
        throw "The native installer launcher did not finish after its window closed."
    }
    if ($launcherProcess.ExitCode -ne 0) {
        throw (
            "The native installer launcher exited with " +
            "$($launcherProcess.ExitCode) after its welcome window closed."
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

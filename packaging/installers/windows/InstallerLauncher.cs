using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

internal static class InstallerLauncher
{
    private const uint MessageBoxOk = 0x00000000;
    private const uint MessageBoxIconError = 0x00000010;

    [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern int MessageBoxW(
        IntPtr window,
        string message,
        string caption,
        uint type
    );

    [STAThread]
    private static int Main()
    {
        try
        {
            string launcherDirectory = AppContext.BaseDirectory;
            string installerScript = Path.Combine(
                launcherDirectory,
                "Install-OpenHCS.ps1"
            );
            string installerContract = Path.Combine(
                launcherDirectory,
                "installer_contract.json"
            );
            RequireSiblingFile(installerScript);
            RequireSiblingFile(installerContract);

            string windowsDirectory = Environment.GetFolderPath(
                Environment.SpecialFolder.Windows
            );
            string powerShell = Path.Combine(
                windowsDirectory,
                "System32",
                "WindowsPowerShell",
                "v1.0",
                "powershell.exe"
            );
            if (!File.Exists(powerShell))
            {
                throw new FileNotFoundException(
                    "Windows PowerShell is unavailable.",
                    powerShell
                );
            }

            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                FileName = powerShell,
                WorkingDirectory = launcherDirectory,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden,
            };
            startInfo.Arguments = (
                "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "
                + QuoteWindowsArgument(installerScript)
            );

            Process process = Process.Start(startInfo);
            if (process == null)
            {
                throw new InvalidOperationException(
                    "Windows could not start the setup wizard."
                );
            }
            process.Dispose();
            return 0;
        }
        catch (Exception exception)
        {
            MessageBoxW(
                IntPtr.Zero,
                "OpenHCS Setup could not start.\n\n" + exception.Message,
                "OpenHCS Setup",
                MessageBoxOk | MessageBoxIconError
            );
            return 1;
        }
    }

    private static void RequireSiblingFile(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                "The installer archive is incomplete. Extract all files together.",
                path
            );
        }
    }

    private static string QuoteWindowsArgument(string value)
    {
        StringBuilder quoted = new StringBuilder(value.Length + 2);
        quoted.Append('"');
        int pendingBackslashes = 0;
        foreach (char character in value)
        {
            if (character == '\\')
            {
                pendingBackslashes++;
                continue;
            }
            if (character == '"')
            {
                quoted.Append('\\', (pendingBackslashes * 2) + 1);
                quoted.Append('"');
                pendingBackslashes = 0;
                continue;
            }
            quoted.Append('\\', pendingBackslashes);
            quoted.Append(character);
            pendingBackslashes = 0;
        }
        quoted.Append('\\', pendingBackslashes * 2);
        quoted.Append('"');
        return quoted.ToString();
    }
}

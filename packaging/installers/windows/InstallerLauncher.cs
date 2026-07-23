using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Reflection;
using System.Text;

internal static class InstallerLauncher
{
    private const uint MessageBoxOk = 0x00000000;
    private const uint MessageBoxIconError = 0x00000010;
    private const string WorkerResourceName =
        "OpenHCS.Installer.Install-OpenHCS.ps1";
    private const string ContractResourceName =
        "OpenHCS.Installer.installer_contract.json";

    [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern int MessageBoxW(
        IntPtr window,
        string message,
        string caption,
        uint type
    );

    [STAThread]
    private static int Main(string[] arguments)
    {
        string temporaryDirectory = null;
        try
        {
            if (Environment.Is64BitOperatingSystem && !Environment.Is64BitProcess)
            {
                throw new PlatformNotSupportedException(
                    "The OpenHCS installer must use native 64-bit Windows PowerShell."
                );
            }
            temporaryDirectory = Path.Combine(
                Path.GetTempPath(),
                "OpenHCS Installer",
                Guid.NewGuid().ToString("N")
            );
            Directory.CreateDirectory(temporaryDirectory);
            string installerScript = Path.Combine(
                temporaryDirectory,
                "Install-OpenHCS.ps1"
            );
            string installerContract = Path.Combine(
                temporaryDirectory,
                "installer_contract.json"
            );
            ExtractEmbeddedFile(WorkerResourceName, installerScript);
            ExtractEmbeddedFile(ContractResourceName, installerContract);

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
                WorkingDirectory = temporaryDirectory,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden,
            };
            StringBuilder powerShellArguments = new StringBuilder(
                "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "
            );
            powerShellArguments.Append(QuoteWindowsArgument(installerScript));
            foreach (string argument in arguments)
            {
                powerShellArguments.Append(' ');
                powerShellArguments.Append(QuoteWindowsArgument(argument));
            }
            startInfo.Arguments = powerShellArguments.ToString();

            Process process = Process.Start(startInfo);
            if (process == null)
            {
                throw new InvalidOperationException(
                    "Windows could not start the setup wizard."
                );
            }
            using (process)
            {
                process.WaitForExit();
                return process.ExitCode;
            }
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
        finally
        {
            TryDeleteTemporaryDirectory(temporaryDirectory);
        }
    }

    private static void ExtractEmbeddedFile(string resourceName, string outputPath)
    {
        Assembly assembly = Assembly.GetExecutingAssembly();
        using (Stream input = assembly.GetManifestResourceStream(resourceName))
        {
            if (input == null)
            {
                throw new InvalidDataException(
                    "The installer is missing embedded resource " + resourceName + "."
                );
            }
            using (
                FileStream output = new FileStream(
                    outputPath,
                    FileMode.CreateNew,
                    FileAccess.Write,
                    FileShare.None
                )
            )
            {
                input.CopyTo(output);
            }
        }
    }

    private static void TryDeleteTemporaryDirectory(string path)
    {
        if (string.IsNullOrEmpty(path))
        {
            return;
        }
        try
        {
            if (Directory.Exists(path))
            {
                Directory.Delete(path, true);
            }
        }
        catch
        {
            // A scanner can briefly retain an extracted file. The random
            // user-temporary directory contains no credentials or user data.
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

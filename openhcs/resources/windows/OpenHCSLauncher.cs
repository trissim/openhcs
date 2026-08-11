using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Windows.Forms;

internal static class OpenHCSLauncher
{
    private const string ProductName = __OPENHCS_PRODUCT_NAME__;
    private const string CurrentEnvironmentPointerName =
        __OPENHCS_CURRENT_ENVIRONMENT_POINTER_NAME__;
    private const string McpLauncherName = __OPENHCS_MCP_LAUNCHER_NAME__;
    private const string EnvironmentContainerRelativePath =
        __OPENHCS_ENVIRONMENT_CONTAINER_RELATIVE_PATH__;
    private const string GuiModule = __OPENHCS_GUI_MODULE__;
    private const string UvRelativePath = __OPENHCS_UV_RELATIVE_PATH__;
    private const string CpuOnlyEnvironmentVariable =
        __OPENHCS_CPU_ONLY_ENVIRONMENT__;
    private const string NumbaCacheEnvironmentVariable =
        __OPENHCS_NUMBA_CACHE_ENVIRONMENT__;
    private const string NumbaCachePath = __OPENHCS_NUMBA_CACHE_PATH__;
    private const string UvEnvironmentVariable = __OPENHCS_UV_ENVIRONMENT__;
    private const string RestartExecutableEnvironmentVariable =
        __OPENHCS_RESTART_EXECUTABLE_ENVIRONMENT__;
    private const string InstallationPointerEnvironmentVariable =
        __OPENHCS_MCP_INSTALLATION_POINTER_ENVIRONMENT__;
    private const string StableCommandEnvironmentVariable =
        __OPENHCS_MCP_STABLE_COMMAND_ENVIRONMENT__;
    private const string StartupHandoffEnvironmentVariable =
        __OPENHCS_STARTUP_HANDOFF_EVENT__;
    private const string StableMcpCommandJson =
        __OPENHCS_STABLE_MCP_COMMAND_JSON__;

    [STAThread]
    private static int Main(string[] arguments)
    {
        try
        {
            string installRoot = Path.GetFullPath(
                AppDomain.CurrentDomain.BaseDirectory
            );
            string environmentRoot = ResolveCurrentEnvironmentRoot(installRoot);
            string pythonExecutable = ResolvePythonExecutable(environmentRoot);
            string uvExecutable = Path.Combine(installRoot, UvRelativePath);
            string installationPointer = Path.Combine(
                installRoot,
                McpLauncherName
            );
            RequireFile(uvExecutable, "managed uv executable");
            RequireFile(installationPointer, "stable MCP launcher");

            string eventName = "Local\\OpenHCS.Startup."
                + Process.GetCurrentProcess().Id.ToString()
                + "."
                + Guid.NewGuid().ToString("N");
            bool created;
            using (
                EventWaitHandle handoffEvent = new EventWaitHandle(
                    false,
                    EventResetMode.ManualReset,
                    eventName,
                    out created
                )
            )
            {
                if (!created)
                {
                    throw new InvalidOperationException(
                        "Windows could not create the startup handoff event."
                    );
                }
                ProcessStartInfo startInfo = new ProcessStartInfo();
                startInfo.FileName = pythonExecutable;
                startInfo.Arguments = ModuleArguments(arguments);
                startInfo.WorkingDirectory = installRoot;
                startInfo.UseShellExecute = false;
                startInfo.CreateNoWindow = true;
                startInfo.WindowStyle = ProcessWindowStyle.Hidden;
                startInfo.EnvironmentVariables[CpuOnlyEnvironmentVariable] = "true";
                startInfo.EnvironmentVariables[NumbaCacheEnvironmentVariable] =
                    NumbaCachePath;
                startInfo.EnvironmentVariables[UvEnvironmentVariable] = uvExecutable;
                startInfo.EnvironmentVariables[RestartExecutableEnvironmentVariable] =
                    Application.ExecutablePath;
                startInfo.EnvironmentVariables[
                    InstallationPointerEnvironmentVariable
                ] = installationPointer;
                startInfo.EnvironmentVariables[StableCommandEnvironmentVariable] =
                    StableMcpCommandJson;
                startInfo.EnvironmentVariables[StartupHandoffEnvironmentVariable] =
                    eventName;

                using (Process process = Process.Start(startInfo))
                {
                    if (process == null)
                    {
                        throw new InvalidOperationException(
                            "Windows could not start the OpenHCS GUI process."
                        );
                    }
                    while (!handoffEvent.WaitOne(100))
                    {
                        if (process.HasExited)
                        {
                            throw new InvalidOperationException(
                                "The OpenHCS GUI process ended before its startup "
                                + "window became ready."
                            );
                        }
                    }
                }
            }
            return 0;
        }
        catch (Exception exception)
        {
            MessageBox.Show(
                ProductName + " could not start.\n\n" + exception.Message,
                ProductName,
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
            return 1;
        }
    }

    private static string ResolveCurrentEnvironmentRoot(string installRoot)
    {
        string pointer = Path.Combine(
            installRoot,
            CurrentEnvironmentPointerName
        );
        RequireFile(pointer, "current environment pointer");
        string environmentName = File.ReadAllText(pointer, Encoding.UTF8).Trim();
        string environmentContainer = Path.GetFullPath(
            Path.Combine(installRoot, EnvironmentContainerRelativePath)
        ).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string environmentRoot = Path.GetFullPath(
            Path.Combine(environmentContainer, environmentName)
        ).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        DirectoryInfo parent = Directory.GetParent(environmentRoot);
        if (
            string.IsNullOrWhiteSpace(environmentName)
            || parent == null
            || !string.Equals(
                parent.FullName.TrimEnd(
                    Path.DirectorySeparatorChar,
                    Path.AltDirectorySeparatorChar
                ),
                environmentContainer,
                StringComparison.OrdinalIgnoreCase
            )
        )
        {
            throw new InvalidDataException(
                "The installed current-environment pointer is invalid. "
                + "Re-run the official OpenHCS installer to repair it."
            );
        }
        return environmentRoot;
    }

    private static string ResolvePythonExecutable(string environmentRoot)
    {
        string scripts = Path.Combine(environmentRoot, "Scripts");
        string windowedPython = Path.Combine(scripts, "pythonw.exe");
        if (File.Exists(windowedPython))
        {
            return windowedPython;
        }
        string consolePython = Path.Combine(scripts, "python.exe");
        RequireFile(consolePython, "Python interpreter");
        return consolePython;
    }

    private static void RequireFile(string path, string description)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                "The installed " + description + " is unavailable.",
                path
            );
        }
    }

    private static string ModuleArguments(string[] arguments)
    {
        StringBuilder commandLine = new StringBuilder();
        commandLine.Append(QuoteWindowsArgument("-m"));
        commandLine.Append(' ');
        commandLine.Append(QuoteWindowsArgument(GuiModule));
        string forwarded = QuoteArguments(arguments);
        if (forwarded.Length > 0)
        {
            commandLine.Append(' ');
            commandLine.Append(forwarded);
        }
        return commandLine.ToString();
    }

    private static string QuoteArguments(string[] arguments)
    {
        StringBuilder commandLine = new StringBuilder();
        foreach (string argument in arguments)
        {
            if (commandLine.Length > 0)
            {
                commandLine.Append(' ');
            }
            commandLine.Append(QuoteWindowsArgument(argument));
        }
        return commandLine.ToString();
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

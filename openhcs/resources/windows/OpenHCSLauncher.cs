using System;
using System.Diagnostics;
using System.Drawing;
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
    private const string EnvironmentsRelativePath =
        __OPENHCS_ENVIRONMENTS_RELATIVE_PATH__;
    private const string GuiRelativePath = __OPENHCS_GUI_RELATIVE_PATH__;
    private const string UvRelativePath = __OPENHCS_UV_RELATIVE_PATH__;
    private const string CpuOnlyEnvironmentVariable =
        __OPENHCS_CPU_ONLY_ENVIRONMENT__;
    private const string UvEnvironmentVariable = __OPENHCS_UV_ENVIRONMENT__;
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
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            using (StartupWindow window = new StartupWindow(arguments))
            {
                Application.Run(window);
                return window.ExitCode;
            }
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

    private sealed class StartupWindow : Form
    {
        private readonly string[] _arguments;
        private readonly Label _status;
        private readonly ProgressBar _progress;
        private readonly Button _closeButton;
        private EventWaitHandle _handoffEvent;
        private RegisteredWaitHandle _handoffWait;
        private Process _process;
        private volatile bool _handoffCompleted;
        private bool _failed;

        internal StartupWindow(string[] arguments)
        {
            _arguments = arguments;
            ExitCode = 1;
            Text = "Starting " + ProductName;
            StartPosition = FormStartPosition.CenterScreen;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            MinimizeBox = false;
            ShowInTaskbar = true;
            ClientSize = new Size(500, 142);
            BackColor = Color.FromArgb(30, 30, 30);
            ForeColor = Color.White;

            Icon associatedIcon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
            if (associatedIcon != null)
            {
                Icon = associatedIcon;
            }

            Label title = new Label();
            title.AutoSize = true;
            title.Font = new Font(Font.FontFamily, 18, FontStyle.Bold);
            title.ForeColor = Color.FromArgb(0, 170, 255);
            title.Location = new Point(22, 18);
            title.Text = ProductName;
            Controls.Add(title);

            _status = new Label();
            _status.AutoEllipsis = true;
            _status.Location = new Point(24, 61);
            _status.Size = new Size(452, 24);
            _status.Text = "Preparing the high-content screening workspace";
            Controls.Add(_status);

            _progress = new ProgressBar();
            _progress.Location = new Point(24, 94);
            _progress.Size = new Size(452, 9);
            _progress.Style = ProgressBarStyle.Marquee;
            _progress.MarqueeAnimationSpeed = 25;
            Controls.Add(_progress);

            _closeButton = new Button();
            _closeButton.Location = new Point(376, 104);
            _closeButton.Size = new Size(100, 27);
            _closeButton.Text = "Close";
            _closeButton.Visible = false;
            _closeButton.Click += delegate { Close(); };
            Controls.Add(_closeButton);

            Shown += delegate { BeginLaunch(); };
            FormClosing += PreventPrematureClose;
            FormClosed += delegate { DisposeLaunchResources(); };
        }

        internal int ExitCode { get; private set; }

        private void BeginLaunch()
        {
            try
            {
                string installRoot = Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory);
                string environmentRoot = ResolveCurrentEnvironmentRoot(installRoot);
                string guiExecutable = Path.Combine(environmentRoot, GuiRelativePath);
                string uvExecutable = Path.Combine(installRoot, UvRelativePath);
                string installationPointer = Path.Combine(
                    installRoot,
                    McpLauncherName
                );
                RequireFile(guiExecutable, "GUI entry point");
                RequireFile(uvExecutable, "managed uv executable");
                RequireFile(installationPointer, "stable MCP launcher");

                string eventName = "Local\\OpenHCS.Startup."
                    + Process.GetCurrentProcess().Id.ToString()
                    + "."
                    + Guid.NewGuid().ToString("N");
                bool created;
                _handoffEvent = new EventWaitHandle(
                    false,
                    EventResetMode.ManualReset,
                    eventName,
                    out created
                );
                if (!created)
                {
                    throw new InvalidOperationException(
                        "Windows could not create the startup handoff event."
                    );
                }

                ProcessStartInfo startInfo = new ProcessStartInfo();
                startInfo.FileName = guiExecutable;
                startInfo.Arguments = QuoteArguments(_arguments);
                startInfo.WorkingDirectory = installRoot;
                startInfo.UseShellExecute = false;
                startInfo.CreateNoWindow = true;
                startInfo.WindowStyle = ProcessWindowStyle.Hidden;
                startInfo.EnvironmentVariables[CpuOnlyEnvironmentVariable] = "true";
                startInfo.EnvironmentVariables[UvEnvironmentVariable] = uvExecutable;
                startInfo.EnvironmentVariables[InstallationPointerEnvironmentVariable] =
                    installationPointer;
                startInfo.EnvironmentVariables[StableCommandEnvironmentVariable] =
                    StableMcpCommandJson;
                startInfo.EnvironmentVariables[StartupHandoffEnvironmentVariable] =
                    eventName;

                _process = Process.Start(startInfo);
                if (_process == null)
                {
                    throw new InvalidOperationException(
                        "Windows could not start the OpenHCS GUI process."
                    );
                }
                _process.EnableRaisingEvents = true;
                _process.Exited += ProcessExited;
                _handoffWait = ThreadPool.RegisterWaitForSingleObject(
                    _handoffEvent,
                    StartupHandoffCompleted,
                    null,
                    Timeout.Infinite,
                    true
                );
                ExitCode = 0;
            }
            catch (Exception exception)
            {
                ShowFailure(exception.Message);
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
            string environmentsRoot = Path.GetFullPath(
                Path.Combine(installRoot, EnvironmentsRelativePath)
            ).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string environmentRoot = Path.GetFullPath(
                Path.Combine(environmentsRoot, environmentName)
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
                    environmentsRoot,
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

        private void StartupHandoffCompleted(
            object state,
            bool timedOut
        )
        {
            if (timedOut)
            {
                return;
            }
            _handoffCompleted = true;
            if (IsHandleCreated)
            {
                BeginInvoke((MethodInvoker)delegate { Close(); });
            }
        }

        private void ProcessExited(object sender, EventArgs eventArguments)
        {
            if (_handoffCompleted || !IsHandleCreated)
            {
                return;
            }
            BeginInvoke(
                (MethodInvoker)delegate
                {
                    if (!_handoffCompleted)
                    {
                        ShowFailure(
                            "The OpenHCS GUI process ended before its startup "
                            + "window became ready."
                        );
                    }
                }
            );
        }

        private void ShowFailure(string message)
        {
            _failed = true;
            ExitCode = 1;
            _progress.Style = ProgressBarStyle.Blocks;
            _progress.Value = 0;
            _status.ForeColor = Color.FromArgb(255, 85, 85);
            _status.Text = message;
            _closeButton.Visible = true;
        }

        private void PreventPrematureClose(
            object sender,
            FormClosingEventArgs eventArguments
        )
        {
            if (
                eventArguments.CloseReason == CloseReason.UserClosing
                && !_failed
                && !_handoffCompleted
            )
            {
                eventArguments.Cancel = true;
                WindowState = FormWindowState.Minimized;
            }
        }

        private void DisposeLaunchResources()
        {
            if (_handoffWait != null)
            {
                _handoffWait.Unregister(null);
                _handoffWait = null;
            }
            if (_handoffEvent != null)
            {
                _handoffEvent.Dispose();
                _handoffEvent = null;
            }
            if (_process != null)
            {
                _process.Dispose();
                _process = null;
            }
        }
    }
}

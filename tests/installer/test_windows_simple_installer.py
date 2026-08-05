"""Static contract tests for the native Windows installer sources."""

from __future__ import annotations

import json
from pathlib import Path
import re

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INSTALLER_ROOT = REPOSITORY_ROOT / "packaging" / "installers"
WINDOWS_ROOT = INSTALLER_ROOT / "windows"
POWERSHELL_PATH = WINDOWS_ROOT / "Install-OpenHCS.ps1"
CMD_PATH = WINDOWS_ROOT / "Install-OpenHCS.cmd"
LAUNCHER_PATH = WINDOWS_ROOT / "InstallerLauncher.cs"
LAUNCHER_PROJECT_PATH = WINDOWS_ROOT / "InstallerLauncher.csproj"
LAUNCHER_BUILD_PATH = WINDOWS_ROOT / "Build-InstallerLauncher.ps1"
CONTRACT_PATH = INSTALLER_ROOT / "installer_contract.json"
INTEGRATION_WORKFLOW_PATH = (
    REPOSITORY_ROOT / ".github" / "workflows" / "integration-tests.yml"
)
PUBLISH_WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "publish.yml"


def _source() -> str:
    return POWERSHELL_PATH.read_text(encoding="utf-8")


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_windows_installer_has_stable_double_click_entrypoint() -> None:
    cmd = CMD_PATH.read_text(encoding="utf-8")
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    project = LAUNCHER_PROJECT_PATH.read_text(encoding="utf-8")
    build = LAUNCHER_BUILD_PATH.read_text(encoding="utf-8")

    # The CMD remains a developer fallback. The release-facing WinExe never
    # creates a console window.
    assert "powershell.exe -NoProfile -WindowStyle Hidden" in cmd
    assert '-ExecutionPolicy Bypass -File "%~dp0Install-OpenHCS.ps1"' in cmd
    assert '-BrandIconPath "%~dp0..\\..\\..\\openhcs\\resources\\assets\\openhcs.ico"' in cmd
    assert '-BrandLogoPath "%~dp0..\\..\\..\\openhcs\\resources\\assets\\openhcs-icon-square.png"' in cmd
    assert 'start ""' in cmd
    assert "<OutputType>WinExe</OutputType>" in project
    assert "<TargetFramework>net48</TargetFramework>" in project
    assert "<Prefer32Bit>false</Prefer32Bit>" in project
    assert "<RuntimeIdentifier>" not in project
    assert "<SelfContained>" not in project
    assert "<PublishSingleFile>" not in project
    assert "<AssemblyName>OpenHCS-Windows-Installer</AssemblyName>" in project
    assert "<ApplicationIcon>OpenHCS.ico</ApplicationIcon>" in project
    assert 'EmbeddedResource Include="Install-OpenHCS.ps1"' in project
    assert 'EmbeddedResource Include="OpenHCS.ico"' in project
    assert 'EmbeddedResource Include="OpenHCS.png"' in project
    assert 'EmbeddedResource Include="..\\installer_contract.json"' in project
    assert "OpenHCS.Installer.Install-OpenHCS.ps1" in project
    assert "OpenHCS.Installer.OpenHCS.ico" in project
    assert "OpenHCS.Installer.OpenHCS.png" in project
    assert "OpenHCS.Installer.installer_contract.json" in project
    assert "dotnet build $projectPath" in build
    assert "--runtime" not in build
    assert "--self-contained" not in build
    assert "[string]$ContractPath" in build
    assert '"OpenHCS-Windows-Installer.exe"' in build
    assert '"installer_contract.json"' in build
    assert '"openhcs.ico"' in build
    assert '"openhcs-icon-square.png"' in build
    assert '"OpenHCS.ico"' in build
    assert '"OpenHCS.png"' in build

    assert "Assembly.GetExecutingAssembly()" in launcher
    assert "Environment.Is64BitOperatingSystem" in launcher
    assert "!Environment.Is64BitProcess" in launcher
    assert "native 64-bit Windows PowerShell" in launcher
    assert "GetManifestResourceStream(resourceName)" in launcher
    assert 'Guid.NewGuid().ToString("N")' in launcher
    assert "Path.GetTempPath()" in launcher
    assert '"Install-OpenHCS.ps1"' in launcher
    assert '"installer_contract.json"' in launcher
    assert '"OpenHCS.ico"' in launcher
    assert '"OpenHCS.png"' in launcher
    assert "AppContext.BaseDirectory" not in launcher
    assert "RequireSiblingFile" not in launcher
    assert "ExtractEmbeddedFile(WorkerResourceName, installerScript)" in launcher
    assert "ExtractEmbeddedFile(ContractResourceName, installerContract)" in launcher
    assert "ExtractEmbeddedFile(BrandIconResourceName, installerBrandIcon)" in launcher
    assert "ExtractEmbeddedFile(BrandLogoResourceName, installerBrandLogo)" in launcher
    assert "UseShellExecute = false" in launcher
    assert "CreateNoWindow = true" in launcher
    assert "WindowStyle = ProcessWindowStyle.Hidden" in launcher
    assert 'startInfo.EnvironmentVariables.Remove("PSModulePath")' in launcher
    assert "startInfo.ArgumentList" not in launcher
    assert (
        "powerShellArguments.Append(QuoteWindowsArgument(installerScript))" in launcher
    )
    assert 'powerShellArguments.Append(" -BrandIconPath ")' in launcher
    assert "powerShellArguments.Append(QuoteWindowsArgument(installerBrandIcon))" in launcher
    assert 'powerShellArguments.Append(" -BrandLogoPath ")' in launcher
    assert "powerShellArguments.Append(QuoteWindowsArgument(installerBrandLogo))" in launcher
    assert "foreach (string argument in arguments)" in launcher
    assert "process.WaitForExit()" in launcher
    assert "return process.ExitCode" in launcher
    assert "TryDeleteTemporaryDirectory(temporaryDirectory)" in launcher
    assert "private static string QuoteWindowsArgument(string value)" in launcher
    assert "pendingBackslashes * 2" in launcher
    assert "(pendingBackslashes * 2) + 1" in launcher
    assert 'DllImport("user32.dll"' in launcher
    assert "IntPtr.Zero" in launcher
    assert "MessageBoxW(" in launcher


def test_windows_installer_fails_closed_on_validated_shared_contract() -> None:
    source = _source()
    contract = _contract()

    assert "Resolve-ContractPath" in source
    assert '"installer_contract.json"' in source
    assert "ConvertFrom-Json" in source
    assert 'Get-RequiredTextProperty $contract "entry_point"' in source
    assert 'Get-RequiredTextProperty $contract "gui_entry_point"' in source
    assert '"openhcs.installer.v2"' in source
    assert "Expected exactly one installer_contract.json" in source
    assert "Uri]::TryCreate" in source
    assert "UriSchemeHttps" in source
    assert "$parsedUvBaseUrl.IdnHost," in source
    assert '"astral.sh"' in source
    assert '"^3\\.[0-9]+$"' in source
    assert '"^[0-9]+\\.[0-9]+\\.[0-9]+$"' in source
    assert '"{0}/{1}/install.ps1"' in source

    # Shared semantic values are data, never fallback constants in the script.
    for value in (
        contract["python_version"],
        contract["package_requirement"],
        contract["uv_release"]["version"],
    ):
        assert value not in source


def test_windows_installer_uses_uv_for_python_and_pip_for_packages() -> None:
    source = _source()

    assert "function Get-WindowsPowerShellExecutable" in source
    assert '"System32"' in source
    assert '"WindowsPowerShell"' in source
    assert '"v1.0"' in source
    assert "$PSHOME" not in source
    assert "Invoke-WebRequest" in source
    assert "GetTempPath" in source
    assert "openhcs-uv-installer-$([Guid]::NewGuid()" in source
    assert ".ps1" in source
    assert "-OutFile $temporaryUvInstaller" in source
    assert "-TimeoutSec 120" in source
    assert "Invoke-Expression" not in source
    assert re.search(r'"--no-config", "python", "install"', source)
    assert re.search(r'"--no-config", "venv", "--python"', source)
    assert '"--seed"' in source
    assert '"venv", "--clear"' not in source
    assert '"-m", "pip", "install"' in source
    assert '"-m", "pip", "check"' in source
    assert '"--prerelease"' not in source
    assert '"--prepare-capabilities"' in source
    assert '-Description "Prepare the execution catalog"' in source
    assert "$env:UV_INSTALL_DIR" in source
    assert "$env:UV_NO_MODIFY_PATH" in source
    assert "pinned official uv $($Contract.UvVersion)" in source
    assert "Do not disable protection or add a broad" in source

    # Contract values remain individual native arguments even when paths contain spaces.
    assert "[string[]]$ArgumentList" in source
    assert "FilePath = $FilePath" in source
    assert "ArgumentList = @($ArgumentList)" in source
    assert "ConvertTo-Json -Compress" in source
    assert "[Text.Encoding]::UTF8.GetBytes($payload)" in source
    assert (
        "& ([string]`$payload.FilePath) @([string[]]`$payload.ArgumentList)" in source
    )
    assert "$startInfo.RedirectStandardOutput = $true" in source
    assert "$startInfo.RedirectStandardError = $true" in source
    assert "$process.StandardOutput.ReadLineAsync()" in source
    assert "$process.StandardError.ReadLineAsync()" in source
    assert "$standardOutput.IsCompleted" in source
    assert "$standardError.IsCompleted" in source
    assert "$standardOutput.GetAwaiter().GetResult()" in source
    assert "$standardError.GetAwaiter().GetResult()" in source
    assert "ReadToEndAsync" not in source
    assert "$exitCode = $process.ExitCode" in source
    command = source[
        source.index("function Invoke-LoggedCommand") : source.index(
            "function Get-StableLauncherPath"
        )
    ]
    assert "cmd.exe" not in command
    assert "/c " not in command.lower()


def test_windows_installer_delegates_desktop_projection_to_installed_authority() -> None:
    source = _source()

    assert '"Scripts"' in source
    assert '"$($Contract.EntryPoint).exe"' in source
    assert '"-m", "openhcs.desktop_deployment_cli"' in source
    assert '"--installation-pointer=$launcherPath"' in source
    assert "$env:OPENHCS_UV_EXECUTABLE = $uvExecutable" in source
    assert "WScript.Shell" not in source
    assert "CreateShortcut" not in source
    assert "SHChangeNotify" not in source
    assert "openhcs.resources.brand" not in source
    assert '"environments"' in source
    assert "Publish-LaunchAdapterAndShortcut" in source
    assert "Remove-SupersededEnvironments" in source
    assert "function Remove-ManagedEnvironmentDirectory" in source
    assert "Resolve-ManagedEnvironmentPath" in source
    assert "ConvertTo-WindowsExtendedPath" in source
    assert '"OPENHCS_INSTALLER_DELETE_TARGET"' in source
    assert "'rd /S /Q" in source
    cleanup = source[
        source.index("function Remove-SupersededEnvironments") :
        source.index("function Remove-UnpublishedCandidateEnvironment")
    ]
    assert "$supersededEnvironmentPath = $_.FullName" in cleanup
    assert (
        "'$supersededEnvironmentPath': " in cleanup
    )
    assert "$($_.FullName)" not in cleanup
    assert "Remove-Item -LiteralPath $supersededEnvironmentPath" not in cleanup
    unpublished_cleanup = source[
        source.index("function Remove-UnpublishedCandidateEnvironment") :
        source.index("function Invoke-WorkerInstall")
    ]
    assert "Remove-ManagedEnvironmentDirectory" in unpublished_cleanup
    assert "Remove-Item -LiteralPath $CandidatePath" not in unpublished_cleanup
    assert source.index('"pip", "check"') < source.index(
        "Publish-LaunchAdapterAndShortcut `"
    )
    assert "openhcs.pyqt_gui" not in source


def test_windows_installer_keeps_ui_responsive_and_failures_visible() -> None:
    source = _source()
    write_log = source[
        source.index("function Write-InstallLog") : source.index(
            "function Resolve-InstallerCancellationPath"
        )
    ]

    assert "System.Windows.Forms" in source
    assert "Start-InstallerWorker" in source
    assert "-EncodedCommand" in source
    assert '"taskkill.exe"' in source
    assert '"/T"' in source
    assert "Cancel install" in source
    assert "installer.log" in source
    assert "bootstrap.log" in source
    assert "if ($Worker)" in write_log
    assert "Write-Host $line" in write_log
    assert "[IO.FileShare]::Read" in write_log
    assert "[IO.StreamWriter]::new" in write_log
    assert "$writer.AutoFlush = $true" in write_log
    assert "$script:LogWriter.WriteLine($line)" in write_log
    assert "Add-Content" not in write_log
    assert "Show-InstallerResult" in source
    assert '-Heading "Installation failed"' in source
    assert "Open the durable log for details" in source


def test_windows_installer_is_a_four_page_next_next_finish_wizard() -> None:
    source = _source()
    window = source[
        source.index("function Show-InstallerWindow") : source.index(
            "\ntry {\n    $installerContract"
        )
    ]

    for page in ("Welcome", "Options", "Progress", "Finish"):
        panel = page.lower()
        assert f'${panel}Panel.Name = "{page}Page"' in window
        assert f"{page} = ${panel}Panel" in window

    assert '[ValidateSet("Welcome", "Options", "Progress", "Finish")]' in window
    assert '$nextButton.Text = "Next >"' in window
    assert '$nextButton.Text = "Finish"' in window
    assert '$welcomePrompt.Text = "Click Next to continue."' in window
    assert '$optionsPrompt.Text = "Click Next to begin installation."' in window
    assert window.index('Set-WizardPage "Options"') < window.index(
        'Set-WizardPage "Progress"'
    )
    assert window.index('Set-WizardPage "Progress"') < window.index(
        "$script:WorkerProcess = Start-InstallerWorker `"
    )
    assert '-Heading "Installation complete"' in window
    assert 'Set-WizardPage "Finish"' in window


def test_windows_wizard_owns_liveness_failure_and_optional_launch_ui() -> None:
    source = _source()
    worker_start = source[
        source.index("function Start-InstallerWorker") : source.index(
            "function Show-InstallerWindow"
        )
    ]
    window = source[
        source.index("function Show-InstallerWindow") : source.index(
            "\ntry {\n    $installerContract"
        )
    ]

    assert "Windows.Forms.ProgressBar" in source
    assert "[Windows.Forms.ProgressBarStyle]::Marquee" in source
    assert "$progressBar.MarqueeAnimationSpeed = 30" in source
    assert "$timer.Interval = 350" in source
    assert "$startInfo.RedirectStandardOutput = $true" in worker_start
    assert "$startInfo.RedirectStandardError = $true" in worker_start
    assert "New-InstallerProgressStream" in window
    assert "$Reader.ReadLineAsync()" in window
    assert "Read-InstallerProgressStream" in window
    assert '$openLogButton.Text = "Open log"' in source
    assert '$launchCheck.Text = "Launch $($Contract.ProductName) after setup"' in source
    assert "$launchCheck.Checked = $true" in source
    assert '"Connect OpenHCS to ChatGPT, Codex, and local AI agent apps"' in source
    assert "$agentConnectionCheck.Checked = $true" in source
    assert "Get-DesktopShortcutPath $Contract" in source
    assert "Start-Process -FilePath (Get-DesktopShortcutPath $Contract)" in source

    # Completion is an actual Finish page, not a modal launch question.
    assert "$($Contract.ProductName) is installed. Launch it now?" not in source


def test_windows_wizard_never_reopens_the_worker_owned_log_during_install() -> None:
    source = _source()
    window = source[
        source.index("function Show-InstallerWindow") : source.index(
            "\ntry {\n    $installerContract"
        )
    ]
    timer = window[
        window.index("$timer = New-Object Windows.Forms.Timer") :
        window.index("$timer.Start()")
    ]
    page_switch = window[
        window.index("switch ($Page)") : window.index(
            "function Show-InstallerResult"
        )
    ]
    progress_page = page_switch[
        page_switch.index('"Progress"') : page_switch.index('"Finish"')
    ]
    finish_page = page_switch[
        page_switch.index('"Finish"') : page_switch.index("        }\n    }")
    ]

    # The worker is the sole durable-log writer. The wizard consumes the
    # worker's existing stdout projection and opens the durable file only after
    # terminal completion.
    assert "Write-InstallLog" not in window
    assert "Open-InstallLog" not in window
    assert "$script:LogPath" not in timer
    assert "Get-Content -LiteralPath $script:LogPath" not in timer
    assert "Read-InstallerProgressStream" in timer
    assert "$openLogButton.Visible = $true" not in progress_page
    assert "$openLogButton.Visible = $true" in finish_page

    worker = source[
        source.index("function Invoke-WorkerInstall") : source.index(
            "function Start-InstallerWorker"
        )
    ]
    assert "Open-InstallLog $script:LogPath" in worker
    assert 'Write-InstallLog "Starting $($Contract.ProductName) installation."' in worker
    assert "Close-InstallLog" in worker


def test_windows_installer_registers_agent_clients_through_stable_launcher() -> None:
    source = _source()

    assert "function Replace-FileDiscardingPrevious" in source
    assert '"$DestinationPath.discarded-' in source
    assert "[IO.File]::Replace($reportCandidate, $reportPath, $null" not in source
    assert "[IO.File]::Replace($shortcutBackup, $shortcutPath, $null" not in source
    assert "[IO.File]::Replace($launcherBackup, $launcherPath, $null" not in source
    assert "[switch]$RegisterMcpClients" in source
    assert '"openhcs-mcp-register.exe"' in source
    assert '"--command", $powerShellExecutable' in source
    assert '"--launcher-argument={0}" -f $launcherArgument' in source
    assert "& $registrationExecutable @registrationArguments 2>&1" in source
    assert '"--args-json" $launcherArguments' not in source
    assert '"--register", "codex"' in source
    assert '"--register-detected"' in source
    assert '"--json"' in source
    assert '"mcp"' in source
    assert "OPENHCS_UV_EXECUTABLE" in source
    assert '"bootstrap", "uv", "uv.exe"' in source
    assert "openhcs.desktop_deployment_cli" in source
    assert "agent-registration.json" in source
    assert "agent-registration-status" in source
    assert "$registrationReport.results" in source
    assert "[string]$_.display_name" in source
    assert "Register-InstalledMcpClients" in source
    assert "Restart ChatGPT desktop, Codex, and other listed apps" in source
    assert "$exitCode -ne 0" in source
    assert "$report.ok -ne $true" in source
    assert source.index("Publish-LaunchAdapterAndShortcut `") < source.index(
        "Register-InstalledMcpClients `"
    )


def test_windows_wizard_preserves_cancel_and_transactional_update_boundaries() -> None:
    source = _source()
    window = source[
        source.index("function Show-InstallerWindow") : source.index(
            "\ntry {\n    $installerContract"
        )
    ]

    assert '"Cancel install"' in source
    assert '"Cancelling safely. Setup is finishing cleanup..."' in source
    assert "Request-InstallerCancellation $script:CancellationPath" in window
    assert "Stop-InstallerWorker" not in source
    assert '"taskkill.exe"' not in window
    assert '"/PID"' in source
    assert '"/T"' in source
    assert '"/F"' in source
    assert '-Heading "Installation cancelled"' in source
    assert "replace it only after the update is fully verified" in source
    assert source.index('"pip", "check"') < source.index(
        "Publish-LaunchAdapterAndShortcut `"
    )


def test_windows_precommit_cancellation_is_worker_owned_and_cleans_candidate() -> None:
    source = _source()
    command = source[
        source.index("function Invoke-LoggedCommand") : source.index(
            "function Get-StableLauncherPath"
        )
    ]
    worker = source[
        source.index("function Invoke-WorkerInstall") : source.index(
            "function Start-InstallerWorker"
        )
    ]

    assert "Test-InstallerCancellationRequested $CancellationPath" in command
    assert "$process.WaitForExit(100)" in command
    assert "Stop-InstallerChildProcess $process" in command
    assert "The active installer command did not stop within ten seconds." in source
    assert "catch [OperationCanceledException]" in worker
    cancelled_cleanup = (
        "Remove-UnpublishedCandidateEnvironment `\n"
        '            $newEnvironmentPath $environmentsRoot "cancelled"'
    )
    assert cancelled_cleanup in worker
    assert worker.index(cancelled_cleanup) < worker.index("return 2")

    # Every native install command receives the one validated worker marker.
    assert worker.count("-CancellationPath $resolvedCancellationPath") == worker.count(
        "Invoke-LoggedCommand"
    )
    assert "Resolve-InstallerCancellationPath" in worker
    assert "-TimeoutSec 120" in worker


def test_windows_postcommit_cancellation_reports_installed_without_killing() -> None:
    source = _source()
    worker = source[
        source.index("function Invoke-WorkerInstall") : source.index(
            "function Start-InstallerWorker"
        )
    ]
    window = source[
        source.index("function Show-InstallerWindow") : source.index(
            "\ntry {\n    $installerContract"
        )
    ]

    checkpoint = worker.rindex(
        "Assert-InstallerCancellationNotRequested $resolvedCancellationPath",
        0,
        worker.index("Publish-LaunchAdapterAndShortcut `"),
    )
    publication = worker.index("$publicationStarted = $true", checkpoint)
    publish_call = worker.index("Publish-LaunchAdapterAndShortcut `", publication)
    success = worker.index('Write-InstallLog "SUCCESS:', publish_call)
    success_return = worker.index("return 0", success)
    committed_region = worker[publication:success_return]

    assert checkpoint < publication < publish_call < success < success_return
    assert "Assert-InstallerCancellationNotRequested" not in committed_region
    assert "Stop-InstallerChildProcess" not in committed_region
    assert "Cancellation arrived after publication" in committed_region

    # Worker exit status, not the earlier button click, owns terminal truth.
    success_branch = window.index("if ($workerExitCode -eq 0)")
    cancelled_branch = window.index("elseif ($workerExitCode -eq 2)")
    assert success_branch < cancelled_branch
    terminal_region = window[success_branch:cancelled_branch]
    assert '-Heading "Installation complete"' in terminal_region
    assert "$script:CancelRequested" not in terminal_region


def test_windows_installer_ci_has_an_absolute_safety_ceiling() -> None:
    workflow = INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    smoke_step = workflow[
        workflow.index(
            "      - name: Execute and verify Windows installer"
        ) : workflow.index("      - name: Show Windows installer log on failure")
    ]

    assert "        timeout-minutes: 30" in smoke_step
    assert "Build-InstallerLauncher.ps1" in smoke_step
    assert "$env:PIP_FIND_LINKS = $env:UV_FIND_LINKS" in smoke_step
    assert '"OpenHCS-Windows-Installer.exe"' in smoke_step
    assert "GUI-subsystem executable" in smoke_step
    assert "Length -gt 2MB" in smoke_step
    assert "[Drawing.Icon]::ExtractAssociatedIcon($launcher)" in smoke_step
    assert "Windows could not extract the installer executable icon." in smoke_step
    assert "openhcs/resources/assets/openhcs.ico" in smoke_step
    assert "Windows installer executable icon differs from the brand asset." in (
        smoke_step
    )
    assert '"openhcs-installer-cancel-{0}.marker"' in smoke_step
    assert '"-CancellationPath", $CancellationMarker' in smoke_step
    assert '"-RegisterMcpClients"' in smoke_step
    assert "$summary.application_path" in smoke_step
    assert '"current-environment"' in smoke_step
    assert "$shortcut.TargetPath -ne $summary.application_path" in smoke_step
    assert "Desktop shortcut target is not a GUI-subsystem executable." in smoke_step
    assert "-I -m openhcs.resources.brand windows_icon" in smoke_step
    assert "$shortcut.IconLocation -ne" in smoke_step
    assert "match '$expectedIconLocation'." in smoke_step
    assert '$env:CODEX_HOME = Join-Path $env:RUNNER_TEMP "codex-home"' in smoke_step
    assert "Windows installer did not register the stable OpenHCS MCP launcher." in (
        smoke_step
    )
    assert "function Invoke-OpenHcsInstallerWorker" in smoke_step
    assert "$installerStartInfo.ArgumentList.Add([string]$argument)" in smoke_step
    assert "$installerProcess.WaitForExit()" in smoke_step
    assert "$installerProcess.ExitCode" in smoke_step
    assert '$installerUv = Join-Path $installRoot "bootstrap\\uv\\uv.exe"' in (
        smoke_step
    )
    assert "--no-config pip install" in smoke_step
    assert "--dry-run" in smoke_step
    assert '--upgrade "openhcs==$releaseVersion"' in smoke_step
    assert "Installer-owned uv could not resolve the stable OpenHCS update." in (
        smoke_step
    )
    assert "--prerelease" not in smoke_step


def test_windows_installer_ci_exercises_long_path_update_cleanup() -> None:
    workflow = INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    smoke_step = workflow[
        workflow.index(
            "      - name: Execute and verify Windows installer"
        ) : workflow.index("      - name: Show Windows installer log on failure")
    ]

    assert "env-20000101T000000Z-00000000000000000000000000000000" in smoke_step
    assert "packages_base_lib_index_js-webpack_sharing_consume_default_jquery" in (
        smoke_step
    )
    assert '$deepFile.Length -le 260' in smoke_step
    assert '[IO.Directory]::CreateDirectory("\\\\?\\$deepDirectory")' in smoke_step
    assert '[IO.File]::WriteAllText("\\\\?\\$deepFile"' in smoke_step
    assert "$updateExitCode = Invoke-OpenHcsInstallerWorker" in smoke_step
    assert "Windows installer update left the long-path stale environment" in smoke_step
    assert "Updated Windows desktop smoke failed." in smoke_step
    assert '"Initial Windows install did not create \'$shortcutPath\'."' in smoke_step
    assert "Remove-Item -LiteralPath $shortcutPath -Force" in smoke_step
    assert '"Could not remove the shortcut repair fixture \'$shortcutPath\'."' in (
        smoke_step
    )
    assert smoke_step.index("Remove-Item -LiteralPath $shortcutPath -Force") < (
        smoke_step.index("$updateExitCode = Invoke-OpenHcsInstallerWorker")
    )


def test_windows_desktop_refresh_reuses_cancellable_process_authority() -> None:
    source = _source()

    assert "[switch]$CaptureOutput" in source
    assert "-CaptureOutput" in source
    assert '"-m", "openhcs.desktop_deployment_cli"' in source
    assert '-Description "Publish desktop application, launchers, and shortcut"' in source


def test_windows_release_is_one_directly_runnable_file() -> None:
    workflow = PUBLISH_WORKFLOW_PATH.read_text(encoding="utf-8")
    windows_job = workflow[
        workflow.index("  build-windows-installer:") : workflow.index(
            "  build-macos-installer:"
        )
    ]

    assert "Build release-pinned single-file Windows installer" in windows_job
    assert "OpenHCS-Windows-Installer.exe" in windows_job
    assert "path: OpenHCS-Windows-Installer.exe" in windows_job
    assert "Compress-Archive" not in windows_job
    assert "OpenHCS-Windows-Installer.zip" not in windows_job


def test_windows_installer_ci_uses_napari_tested_software_opengl() -> None:
    workflow = INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    desktop_job = workflow[
        workflow.index("  desktop-installer-source-test:") : workflow.index(
            "  wheel-integration-test:"
        )
    ]

    assert "      - name: Set up Windows software OpenGL" in desktop_job
    assert "        if: matrix.platform == 'windows'" in desktop_job
    assert (
        "        uses: pyvista/setup-headless-display-action@"
        "5bc8de3bc71fcda7a96439571287a554901541a0 # v4.3"
    ) in desktop_job
    assert "          qt: true" in desktop_job
    assert "          wm: herbstluftwm" in desktop_job


def test_windows_installer_ci_surfaces_detached_viewer_logs_on_failure() -> None:
    workflow = INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    failure_step = workflow[
        workflow.index(
            "      - name: Show Windows installer log on failure"
        ) : workflow.index("      - name: Validate macOS installer sources")
    ]

    assert 'Join-Path $HOME ".local\\share\\openhcs\\logs"' in failure_step
    assert 'Filter "*_detached_port_*.log"' in failure_step
    assert "Get-Content -LiteralPath $_.FullName" in failure_step

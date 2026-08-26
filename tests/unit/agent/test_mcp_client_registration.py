from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import tomlkit

from openhcs.agent.runtime_platform import AgentRuntimePlatformKey
from openhcs.mcp import client_registration
from openhcs.mcp.client_registration import (
    CLIENT_REGISTRATION_SCHEMA_VERSION,
    ClientRegistrationEnvironment,
    ClientRegistrationStatus,
    ClaudeDesktopClientRegistrationTarget,
    CodexClientRegistrationTarget,
    CursorClientRegistrationTarget,
    GeminiCliClientRegistrationTarget,
    McpClientRegistrationTarget,
    McpLauncherSpec,
    VsCodeClientRegistrationTarget,
    WindsurfClientRegistrationTarget,
    register_mcp_clients,
)


def _environment(
    home: Path,
    *,
    platform_key: AgentRuntimePlatformKey = AgentRuntimePlatformKey.LINUX,
    environ: dict[str, str] | None = None,
    executables: dict[str, str] | None = None,
    process_runner=None,
) -> ClientRegistrationEnvironment:
    resolved_executables = executables or {}

    def resolve(candidate: str) -> str | None:
        return resolved_executables.get(candidate)

    def default_runner(
        command: list[str],
        **kwargs,
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return ClientRegistrationEnvironment(
        home=home,
        environ=environ or {},
        platform_key=platform_key,
        executable_resolver=resolve,
        process_runner=process_runner or default_runner,
    )


def _launcher(tmp_path: Path) -> McpLauncherSpec:
    return McpLauncherSpec(
        command=str(tmp_path / "stable" / "Launch-OpenHCS-MCP"),
        arguments=("--stdio",),
    )


def test_registered_client_targets_are_the_client_semantic_authority() -> None:
    assert McpClientRegistrationTarget.__registry__ == {
        "codex": CodexClientRegistrationTarget,
        "claude-desktop": ClaudeDesktopClientRegistrationTarget,
        "cursor": CursorClientRegistrationTarget,
        "gemini-cli": GeminiCliClientRegistrationTarget,
        "windsurf": WindsurfClientRegistrationTarget,
        "vscode": VsCodeClientRegistrationTarget,
    }
    assert (
        CodexClientRegistrationTarget.display_name == "ChatGPT desktop and OpenAI Codex"
    )


def test_codex_registration_preserves_unrelated_toml_and_creates_backup(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir()
    original = (
        "# keep this user comment\n"
        'model = "gpt-5"\n\n'
        "[mcp_servers.other]\n"
        'command = "other-server"\n'
        'args = ["--keep"]\n\n'
        "[mcp_servers.openhcs]\n"
        'command = "/old/launcher"\n'
        'args = ["old"]\n'
    )
    config_path.write_text(original, encoding="utf-8")

    report = register_mcp_clients(
        _launcher(tmp_path),
        required_target_ids=("codex",),
        environment=environment,
    )

    assert report.ok
    assert report.required_ok
    (result,) = report.results
    assert result.status == ClientRegistrationStatus.UPDATED.value
    assert result.config_path == str(config_path)
    assert result.backup_path == f"{config_path}.openhcs.bak"
    assert Path(result.backup_path).read_text(encoding="utf-8") == original
    rendered = config_path.read_text(encoding="utf-8")
    assert "# keep this user comment" in rendered
    document = tomlkit.parse(rendered)
    assert document["model"] == "gpt-5"
    assert document["mcp_servers"]["other"]["args"] == ["--keep"]
    assert document["mcp_servers"]["openhcs"] == {
        "command": _launcher(tmp_path).command,
        "args": ["--stdio"],
    }


def test_codex_unchanged_registration_does_not_rewrite_or_backup(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir()
    launcher = _launcher(tmp_path)
    original = (
        "# stable formatting\n"
        "[mcp_servers.openhcs]\n"
        f'command = "{launcher.command}"\n'
        'args = ["--stdio"]\n'
    )
    config_path.write_text(original, encoding="utf-8")

    report = register_mcp_clients(
        launcher,
        required_target_ids=("codex",),
        environment=environment,
    )

    assert report.results[0].status == ClientRegistrationStatus.UNCHANGED.value
    assert config_path.read_text(encoding="utf-8") == original
    assert not Path(f"{config_path}.openhcs.bak").exists()


def test_codex_inline_table_config_remains_inline_and_preserves_other_servers(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir()
    config_path.write_text(
        'mcp_servers = { other = { command = "other" } }\n',
        encoding="utf-8",
    )
    launcher = _launcher(tmp_path)

    report = register_mcp_clients(
        launcher,
        required_target_ids=("codex",),
        environment=environment,
    )

    assert report.ok
    document = tomlkit.parse(config_path.read_text(encoding="utf-8"))
    assert document["mcp_servers"]["other"] == {"command": "other"}
    assert document["mcp_servers"]["openhcs"] == launcher.stdio_server_entry()
    assert config_path.read_text(encoding="utf-8").startswith("mcp_servers = {")


def test_new_cursor_config_uses_strict_mcp_servers_json(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    launcher = _launcher(tmp_path)

    report = register_mcp_clients(
        launcher,
        required_target_ids=("cursor",),
        environment=environment,
    )

    result = report.results[0]
    assert result.status == ClientRegistrationStatus.REGISTERED.value
    assert result.backup_path is None
    config_path = tmp_path / ".cursor" / "mcp.json"
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "mcpServers": {
            "openhcs": {
                "command": launcher.command,
                "args": ["--stdio"],
            }
        }
    }


@pytest.mark.parametrize(
    ("target_id", "relative_path"),
    (
        ("gemini-cli", Path(".gemini/settings.json")),
        ("windsurf", Path(".codeium/windsurf/mcp_config.json")),
    ),
)
def test_additional_home_json_clients_use_documented_user_config(
    tmp_path: Path,
    target_id: str,
    relative_path: Path,
) -> None:
    environment = _environment(tmp_path)
    launcher = _launcher(tmp_path)
    config_path = tmp_path / relative_path
    config_path.parent.mkdir(parents=True)
    original = {
        "theme": "dark",
        "mcpServers": {
            "other": {"command": "other"},
        },
    }
    config_path.write_text(
        f"{json.dumps(original, indent=2)}\n",
        encoding="utf-8",
    )

    report = register_mcp_clients(
        launcher,
        required_target_ids=(target_id,),
        environment=environment,
    )

    assert report.ok
    (result,) = report.results
    assert result.status == ClientRegistrationStatus.UPDATED.value
    assert result.config_path == str(config_path)
    updated = json.loads(config_path.read_text(encoding="utf-8"))
    assert updated["theme"] == "dark"
    assert updated["mcpServers"]["other"] == {"command": "other"}
    assert updated["mcpServers"]["openhcs"] == launcher.stdio_server_entry()


@pytest.mark.parametrize(
    ("target_id", "relative_path"),
    (
        ("cursor", Path(".cursor/mcp.json")),
        ("gemini-cli", Path(".gemini/settings.json")),
        ("windsurf", Path(".codeium/windsurf/mcp_config.json")),
    ),
)
def test_home_json_registration_is_idempotent_without_backup_churn(
    tmp_path: Path,
    target_id: str,
    relative_path: Path,
) -> None:
    environment = _environment(tmp_path)
    launcher = _launcher(tmp_path)
    config_path = tmp_path / relative_path
    config_path.parent.mkdir(parents=True)
    original = (
        f"{json.dumps({'mcpServers': {'openhcs': launcher.stdio_server_entry()}}, indent=3)}"
        "\n"
    )
    config_path.write_text(original, encoding="utf-8")

    report = register_mcp_clients(
        launcher,
        required_target_ids=(target_id,),
        environment=environment,
    )

    (result,) = report.results
    assert result.status == ClientRegistrationStatus.UNCHANGED.value
    assert result.backup_path is None
    assert config_path.read_text(encoding="utf-8") == original
    assert not Path(f"{config_path}.openhcs.bak").exists()


def test_json_update_replaces_only_openhcs_and_preserves_unrelated_entries(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    config_path = tmp_path / ".cursor" / "mcp.json"
    config_path.parent.mkdir()
    original_document = {
        "theme": "dark",
        "mcpServers": {
            "other": {"command": "other"},
            "openhcs": {"command": "old", "env": {"KEEP": "no"}},
        },
    }
    original = f"{json.dumps(original_document, indent=4)}\n"
    config_path.write_text(original, encoding="utf-8")
    launcher = _launcher(tmp_path)

    report = register_mcp_clients(
        launcher,
        required_target_ids=("cursor",),
        environment=environment,
    )

    result = report.results[0]
    assert result.status == ClientRegistrationStatus.UPDATED.value
    assert Path(result.backup_path).read_text(encoding="utf-8") == original
    updated = json.loads(config_path.read_text(encoding="utf-8"))
    assert updated["theme"] == "dark"
    assert updated["mcpServers"]["other"] == {"command": "other"}
    assert updated["mcpServers"]["openhcs"] == launcher.stdio_server_entry()


@pytest.mark.parametrize(
    ("target_id", "relative_path"),
    (
        ("codex", Path(".codex/config.toml")),
        ("cursor", Path(".cursor/mcp.json")),
        ("gemini-cli", Path(".gemini/settings.json")),
        ("windsurf", Path(".codeium/windsurf/mcp_config.json")),
    ),
)
def test_malformed_config_fails_without_mutation_or_backup(
    tmp_path: Path,
    target_id: str,
    relative_path: Path,
) -> None:
    environment = _environment(tmp_path)
    config_path = tmp_path / relative_path
    config_path.parent.mkdir(parents=True)
    malformed = "[not valid" if config_path.suffix == ".toml" else "{not valid"
    config_path.write_text(malformed, encoding="utf-8")

    report = register_mcp_clients(
        _launcher(tmp_path),
        required_target_ids=(target_id,),
        environment=environment,
    )

    assert not report.ok
    assert not report.required_ok
    assert report.results[0].status == ClientRegistrationStatus.FAILED.value
    assert "left unchanged" in report.results[0].message
    assert config_path.read_text(encoding="utf-8") == malformed
    assert not Path(f"{config_path}.openhcs.bak").exists()


def test_detected_clients_register_and_nondetected_clients_are_ignored(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / "custom-codex"
    codex_home.mkdir()
    cursor_home = tmp_path / ".cursor"
    cursor_home.mkdir()
    gemini_home = tmp_path / ".gemini"
    gemini_home.mkdir()
    windsurf_home = tmp_path / ".codeium" / "windsurf"
    windsurf_home.mkdir(parents=True)
    environment = _environment(
        tmp_path,
        environ={"CODEX_HOME": str(codex_home)},
    )

    report = register_mcp_clients(
        _launcher(tmp_path),
        register_detected=True,
        environment=environment,
    )

    assert [result.target_id for result in report.results] == [
        "codex",
        "cursor",
        "gemini-cli",
        "windsurf",
    ]
    assert (codex_home / "config.toml").exists()
    assert (cursor_home / "mcp.json").exists()
    assert (gemini_home / "settings.json").exists()
    assert (windsurf_home / "mcp_config.json").exists()


@pytest.mark.parametrize(
    ("executable_name", "expected_target"),
    (
        ("gemini", "gemini-cli"),
        ("windsurf", "windsurf"),
    ),
)
def test_additional_client_executables_trigger_detected_registration(
    tmp_path: Path,
    executable_name: str,
    expected_target: str,
) -> None:
    environment = _environment(
        tmp_path,
        executables={executable_name: str(tmp_path / "bin" / executable_name)},
    )

    report = register_mcp_clients(
        _launcher(tmp_path),
        register_detected=True,
        environment=environment,
    )

    assert [result.target_id for result in report.results] == [expected_target]


def test_detected_failure_preserves_success_and_does_not_fail_required_contract(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    cursor_home = tmp_path / ".cursor"
    cursor_home.mkdir()
    cursor_config = cursor_home / "mcp.json"
    cursor_config.write_text("{malformed", encoding="utf-8")
    environment = _environment(tmp_path)

    report = register_mcp_clients(
        _launcher(tmp_path),
        required_target_ids=("codex",),
        register_detected=True,
        environment=environment,
    )

    assert not report.ok
    assert report.required_ok
    assert [
        (result.target_id, result.status, result.required) for result in report.results
    ] == [
        ("codex", ClientRegistrationStatus.REGISTERED.value, True),
        ("cursor", ClientRegistrationStatus.FAILED.value, False),
    ]


@pytest.mark.parametrize(
    ("platform_key", "environ", "expected_relative"),
    (
        (
            AgentRuntimePlatformKey.WINDOWS,
            {"APPDATA": "APP_DATA"},
            Path("APP_DATA/Claude/claude_desktop_config.json"),
        ),
        (
            AgentRuntimePlatformKey.MACOS,
            {},
            Path("Library/Application Support/Claude/claude_desktop_config.json"),
        ),
    ),
)
def test_claude_config_path_is_owned_by_platform_specific_leaf_hook(
    tmp_path: Path,
    platform_key: AgentRuntimePlatformKey,
    environ: dict[str, str],
    expected_relative: Path,
) -> None:
    if "APPDATA" in environ:
        environ = {"APPDATA": str(tmp_path / environ["APPDATA"])}
    environment = _environment(
        tmp_path,
        platform_key=platform_key,
        environ=environ,
    )

    assert (
        ClaudeDesktopClientRegistrationTarget.config_path(environment)
        == tmp_path / expected_relative
    )


def test_claude_windows_leaf_owns_desktop_installation_forms(
    tmp_path: Path,
) -> None:
    local_app_data = tmp_path / "LocalAppData"
    program_files = tmp_path / "ProgramFiles"
    environment = _environment(
        tmp_path,
        platform_key=AgentRuntimePlatformKey.WINDOWS,
        environ={
            "LOCALAPPDATA": str(local_app_data),
            "PROGRAMFILES": str(program_files),
        },
    )

    assert ClaudeDesktopClientRegistrationTarget.installation_paths(environment) == (
        local_app_data / "AnthropicClaude" / "Claude.exe",
        local_app_data / "Programs" / "Claude" / "Claude.exe",
        program_files / "Claude" / "Claude.exe",
    )

    msix_package = local_app_data / "Packages" / "Claude_test"
    msix_package.mkdir(parents=True)
    assert ClaudeDesktopClientRegistrationTarget.desktop_app_installed(environment)


def test_claude_desktop_is_explicitly_unsupported_on_linux(tmp_path: Path) -> None:
    environment = _environment(
        tmp_path,
        platform_key=AgentRuntimePlatformKey.LINUX,
    )

    assert ClaudeDesktopClientRegistrationTarget.config_path(environment) is None
    assert ClaudeDesktopClientRegistrationTarget.installation_paths(environment) == ()
    assert not ClaudeDesktopClientRegistrationTarget.desktop_app_installed(environment)


def test_claude_code_cli_alone_does_not_detect_claude_desktop(
    tmp_path: Path,
) -> None:
    environment = _environment(
        tmp_path,
        platform_key=AgentRuntimePlatformKey.MACOS,
        executables={"claude": str(tmp_path / ".local" / "bin" / "claude")},
    )

    assert not ClaudeDesktopClientRegistrationTarget.detected(environment)

    desktop_app = tmp_path / "Applications" / "Claude.app"
    desktop_app.mkdir(parents=True)
    assert ClaudeDesktopClientRegistrationTarget.detected(environment)


def test_vscode_registration_uses_documented_add_mcp_cli_contract(
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], dict]] = []

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="added", stderr="")

    environment = _environment(
        tmp_path,
        executables={"code": "/usr/bin/code"},
        process_runner=run,
    )
    launcher = _launcher(tmp_path)

    report = register_mcp_clients(
        launcher,
        register_detected=True,
        environment=environment,
    )

    assert report.ok
    (result,) = report.results
    assert result.target_id == "vscode"
    assert calls[0][0][:2] == ["/usr/bin/code", "--add-mcp"]
    assert json.loads(calls[0][0][2]) == {
        "name": "openhcs",
        "command": launcher.command,
        "args": ["--stdio"],
    }
    assert calls[0][1] == {
        "capture_output": True,
        "text": True,
        "check": False,
    }


def test_cli_emits_structured_json_and_requires_explicit_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    environment = _environment(tmp_path)
    monkeypatch.setattr(
        ClientRegistrationEnvironment,
        "current",
        classmethod(lambda cls: environment),
    )
    launcher = _launcher(tmp_path)

    exit_code = client_registration.main(
        [
            "--command",
            launcher.command,
            "--args-json",
            '["--stdio"]',
            "--register",
            "codex",
            "--register-detected",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == CLIENT_REGISTRATION_SCHEMA_VERSION
    assert payload["ok"]
    assert payload["required_ok"]
    assert payload["results"][0]["target_id"] == "codex"


def test_cli_accepts_repeated_launcher_arguments_without_json_quoting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    monkeypatch.setattr(
        ClientRegistrationEnvironment,
        "current",
        classmethod(lambda cls: environment),
    )
    launcher_path = str(tmp_path / "launcher with spaces.ps1")

    exit_code = client_registration.main(
        [
            "--command",
            str(tmp_path / "powershell.exe"),
            "--launcher-argument=-NoProfile",
            "--launcher-argument=-File",
            f"--launcher-argument={launcher_path}",
            "--launcher-argument=mcp",
            "--register",
            "codex",
            "--json",
        ]
    )

    assert exit_code == 0
    document = tomlkit.parse(
        (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
    )
    assert document["mcp_servers"]["openhcs"]["args"] == [
        "-NoProfile",
        "-File",
        launcher_path,
        "mcp",
    ]


def test_cli_returns_nonzero_for_unregistrable_required_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    environment = _environment(tmp_path)
    monkeypatch.setattr(
        ClientRegistrationEnvironment,
        "current",
        classmethod(lambda cls: environment),
    )

    exit_code = client_registration.main(
        [
            "--command",
            str(tmp_path / "launcher"),
            "--register",
            "claude-desktop",
            "--json",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert not payload["required_ok"]
    assert payload["results"][0]["status"] == "failed"

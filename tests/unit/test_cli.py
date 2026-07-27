"""Tests for the package-level OpenHCS command dispatcher."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import openhcs.cli as cli


def _recording_entrypoint(calls, label):
    def main():
        calls.append((label, tuple(cli.sys.argv[1:])))
        return 17

    return SimpleNamespace(main=main)


def test_cli_defaults_to_gui_and_preserves_gui_options(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli,
        "import_module",
        lambda name: _recording_entrypoint(calls, name),
    )

    result = cli.main(["--log-level", "INFO"])

    assert result == 17
    assert calls == [
        ("openhcs.gui_startup", ("--log-level", "INFO")),
    ]


def test_cli_explicit_gui_command_is_not_forwarded(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli,
        "import_module",
        lambda name: _recording_entrypoint(calls, name),
    )

    cli.main(["gui", "--log-level", "SILENT"])

    assert calls == [
        ("openhcs.gui_startup", ("--log-level", "SILENT")),
    ]


def test_cli_mcp_command_uses_headless_entrypoint(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli,
        "import_module",
        lambda name: _recording_entrypoint(calls, name),
    )

    cli.main(["mcp"])

    assert calls == [("openhcs.mcp.bootstrap", ())]


def test_cli_mcp_options_never_route_through_gui_startup(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli,
        "import_module",
        lambda name: _recording_entrypoint(calls, name),
    )

    cli.main(["mcp", "--transport", "stdio"])

    assert calls == [
        ("openhcs.mcp.bootstrap", ("--transport", "stdio")),
    ]


def test_cli_mcp_route_clean_process_never_imports_gui_startup_or_pyqt() -> None:
    checkout = Path(__file__).resolve().parents[2]
    script = """
import sys
from types import ModuleType

bootstrap = ModuleType("openhcs.mcp.bootstrap")
bootstrap.main = lambda: 31
sys.modules["openhcs.mcp.bootstrap"] = bootstrap

from openhcs.cli import main
result = main(["mcp", "--transport", "stdio"])
assert result == 31
assert sys.argv[1:] == []
assert "openhcs.gui_startup" not in sys.modules
assert not any(name == "PyQt6" or name.startswith("PyQt6.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr

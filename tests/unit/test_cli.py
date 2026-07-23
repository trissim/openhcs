"""Tests for the package-level OpenHCS command dispatcher."""

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
        ("openhcs.pyqt_gui.__main__", ("--log-level", "INFO")),
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
        ("openhcs.pyqt_gui.__main__", ("--log-level", "SILENT")),
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

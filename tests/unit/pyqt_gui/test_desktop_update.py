from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt6.QtNetwork import QNetworkReply
from PyQt6.QtWidgets import QMessageBox

import openhcs.pyqt_gui.main as main_module
from openhcs.pyqt_gui.services.desktop_update import (
    LATEST_RELEASE_API_URL,
    DesktopRuntimeEnvironment,
    DesktopUpdateCommandPlan,
    DesktopUpdateError,
    DesktopUpdateService,
    DesktopUpdateSession,
    parse_latest_release,
)


def _release_payload(version: str = "0.7.0") -> dict[str, object]:
    tag = f"v{version}"
    base = f"https://github.com/OpenHCSDev/openhcs/releases/download/{tag}"
    return {
        "tag_name": tag,
        "html_url": f"https://github.com/OpenHCSDev/openhcs/releases/tag/{tag}",
        "draft": False,
        "prerelease": False,
        "assets": [
            {
                "name": "OpenHCS-Windows-Installer.exe",
                "browser_download_url": f"{base}/OpenHCS-Windows-Installer.exe",
            },
            {
                "name": "OpenHCS-macOS-Installer.dmg",
                "browser_download_url": f"{base}/OpenHCS-macOS-Installer.dmg",
            },
            {
                "name": "openhcs-0.7.0-py3-none-any.whl",
                "browser_download_url": f"{base}/openhcs-0.7.0-py3-none-any.whl",
            },
        ],
    }


@pytest.mark.parametrize(
    ("system_name", "expected_suffix", "has_native_installer"),
    [
        ("Windows", ".exe", True),
        ("Darwin", ".dmg", True),
        ("Linux", "/tag/v0.7.0", False),
    ],
)
def test_newer_release_routes_to_platform_handoff(
    system_name: str,
    expected_suffix: str,
    has_native_installer: bool,
) -> None:
    update = parse_latest_release(
        _release_payload(),
        installed_version="0.6.2",
        system_name=system_name,
    )

    assert update.update_available
    assert update.has_native_installer is has_native_installer
    assert update.handoff_url.endswith(expected_suffix)


def test_current_release_is_not_reported_as_update() -> None:
    update = parse_latest_release(
        _release_payload("0.6.2"),
        installed_version="0.6.2",
        system_name="Windows",
    )

    assert not update.update_available


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(tag_name="not a version"),
        lambda payload: payload.update(
            html_url="https://example.invalid/OpenHCSDev/openhcs/releases/tag/v0.7.0"
        ),
        lambda payload: payload.update(prerelease=True),
        lambda payload: payload.pop("tag_name"),
    ],
)
def test_malformed_or_untrusted_release_is_rejected(mutation) -> None:
    payload = _release_payload()
    mutation(payload)

    with pytest.raises(DesktopUpdateError):
        parse_latest_release(
            payload,
            installed_version="0.6.2",
            system_name="Windows",
        )


def test_untrusted_or_ambiguous_native_asset_falls_back_to_release_page() -> None:
    payload = _release_payload()
    assets = payload["assets"]
    assert isinstance(assets, list)
    windows_asset = assets[0]
    assert isinstance(windows_asset, dict)
    windows_asset["browser_download_url"] = (
        "https://example.invalid/OpenHCS-Windows-Installer.exe"
    )

    update = parse_latest_release(
        payload,
        installed_version="0.6.2",
        system_name="Windows",
    )

    assert not update.has_native_installer
    assert update.handoff_url == update.release_url


class _Signal:
    def __init__(self) -> None:
        self.callback = None

    def connect(self, callback) -> None:
        self.callback = callback

    def emit(self) -> None:
        assert self.callback is not None
        self.callback()


class _UnavailableReply:
    def __init__(self) -> None:
        self.finished = _Signal()
        self.deleted = False

    def error(self):
        return QNetworkReply.NetworkError.HostNotFoundError

    def errorString(self) -> str:
        return "Host not found"

    def readAll(self) -> bytes:
        return b""

    def deleteLater(self) -> None:
        self.deleted = True


class _NetworkManager:
    def __init__(self, reply) -> None:
        self.reply = reply
        self.request = None

    def get(self, request):
        self.request = request
        return self.reply


def test_unavailable_release_service_emits_failure_without_blocking() -> None:
    reply = _UnavailableReply()
    manager = _NetworkManager(reply)
    service = DesktopUpdateService(
        installed_version="0.6.2",
        system_name="Windows",
        network_manager=manager,
    )
    failures = []
    service.check_failed.connect(failures.append)

    assert service.check_for_updates()
    assert manager.request.url().toString() == LATEST_RELEASE_API_URL
    assert failures == []

    reply.finished.emit()

    assert failures == ["Host not found"]
    assert reply.deleted


def test_open_update_refuses_a_mutated_untrusted_destination() -> None:
    update = parse_latest_release(
        _release_payload(),
        installed_version="0.6.2",
        system_name="Windows",
    )
    opened = []
    service = DesktopUpdateService(
        installed_version="0.6.2",
        system_name="Windows",
        network_manager=_NetworkManager(_UnavailableReply()),
        url_opener=lambda url: opened.append(url.toString()) or True,
    )

    assert service.open_update(update)
    assert opened == [update.handoff_url]

    update = replace(update, handoff_url="https://example.invalid/update.exe")
    with pytest.raises(DesktopUpdateError):
        service.open_update(update)


class _Action:
    def __init__(self) -> None:
        self.enabled = True

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled


class _StatusSignal:
    def __init__(self) -> None:
        self.messages = []

    def emit(self, message: str) -> None:
        self.messages.append(message)


class _UpdateService:
    def __init__(self, *, starts: bool = True) -> None:
        self.starts = starts
        self.opened = []
        self.started = []

    def check_for_updates(self) -> bool:
        return self.starts

    def open_update(self, update) -> bool:
        self.opened.append(update)
        return True

    def start_update(self, update, *, runtime, session) -> bool:
        self.started.append((update, runtime, session))
        return True


def test_main_window_starts_check_only_from_explicit_action() -> None:
    action = _Action()
    status = _StatusSignal()
    main_like = SimpleNamespace(
        desktop_update_service=_UpdateService(),
        check_for_updates_action=action,
        status_message=status,
    )

    main_module.OpenHCSMainWindow.check_for_updates(main_like)

    assert not action.enabled
    assert status.messages == ["Checking for OpenHCS updates…"]


def test_main_window_available_update_saves_and_starts_restart(monkeypatch) -> None:
    update = parse_latest_release(
        _release_payload(),
        installed_version="0.6.2",
        system_name="Windows",
    )
    action = _Action()
    action.enabled = False
    status = _StatusSignal()
    service = _UpdateService()
    dialog_calls = []
    runtime = object()
    session = SimpleNamespace(discard=lambda: None)
    closed = []

    class _DialogProbe:
        StandardButton = QMessageBox.StandardButton

        @staticmethod
        def question(*args):
            dialog_calls.append(args)
            return QMessageBox.StandardButton.Yes

        @staticmethod
        def warning(*_args):
            raise AssertionError("successful handoff must not show a warning")

    monkeypatch.setattr(main_module, "QMessageBox", _DialogProbe)
    monkeypatch.setattr(
        main_module.DesktopRuntimeEnvironment,
        "current",
        classmethod(lambda cls: runtime),
    )
    monkeypatch.setattr(
        main_module.DesktopUpdateSession,
        "capture",
        classmethod(lambda cls, window: session),
    )
    main_like = SimpleNamespace(
        desktop_update_service=service,
        check_for_updates_action=action,
        status_message=status,
        close=lambda: closed.append(True),
    )

    main_module.OpenHCSMainWindow._on_update_check_completed(main_like, update)

    assert action.enabled
    assert status.messages == [
        "OpenHCS 0.7.0 is available",
        "OpenHCS update prepared; restarting…",
    ]
    assert service.started == [(update, runtime, session)]
    assert closed == [True]
    assert len(dialog_calls) == 1
    update_prompt = dialog_calls[0][2]
    assert "working session and edit history" in update_prompt
    assert "ObjectState" not in update_prompt


def test_update_command_prefers_configured_uv(tmp_path: Path) -> None:
    uv = tmp_path / "uv"
    uv.touch()
    python = tmp_path / "python"

    command = DesktopUpdateCommandPlan.for_environment(
        python_executable=python,
        latest_version=parse_latest_release(
            _release_payload(),
            installed_version="0.6.2",
            system_name="Linux",
        ).latest_version,
        environment={"OPENHCS_UV_EXECUTABLE": str(uv)},
    )

    assert command.executable == uv.resolve()
    assert command.arguments == (
        "--no-config",
        "pip",
        "install",
        "--python",
        str(python),
        "--upgrade",
        "openhcs==0.7.0",
    )


def test_update_command_falls_back_to_running_interpreter_pip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.shutil.which",
        lambda _name: None,
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.find_spec",
        lambda name: object() if name == "pip" else None,
    )
    python = tmp_path / "python"

    command = DesktopUpdateCommandPlan.for_environment(
        python_executable=python,
        latest_version=parse_latest_release(
            _release_payload(),
            installed_version="0.6.2",
            system_name="Linux",
        ).latest_version,
        environment={},
    )

    assert command.executable == python
    assert command.arguments[:3] == ("-m", "pip", "install")
    assert command.arguments[-1] == "openhcs==0.7.0"


def test_update_command_rejects_environment_without_uv_or_pip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.shutil.which",
        lambda _name: None,
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.find_spec",
        lambda _name: None,
    )

    with pytest.raises(DesktopUpdateError, match="neither an available uv"):
        DesktopUpdateCommandPlan.for_environment(
            python_executable=tmp_path / "python",
            latest_version=parse_latest_release(
                _release_payload(),
                installed_version="0.6.2",
                system_name="Linux",
            ).latest_version,
            environment={},
        )


def test_service_starts_worker_with_unambiguous_argument_vectors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    update = parse_latest_release(
        _release_payload(),
        installed_version="0.6.2",
        system_name="Linux",
    )
    python = tmp_path / "python"
    worker_python = tmp_path / "base-python"
    uv = tmp_path / "uv"
    runtime = DesktopRuntimeEnvironment(
        python_executable=python,
        worker_python_executable=worker_python,
        environment_root=tmp_path,
        restart_executable=tmp_path / "openhcs",
        restart_arguments=("--log-level", "DEBUG"),
    )
    session = DesktopUpdateSession(tmp_path / "pending")
    session.directory.mkdir()
    session.worker_document.write_text("worker", encoding="utf-8")
    launched = []
    monkeypatch.setattr(
        DesktopRuntimeEnvironment,
        "update_command",
        lambda self, version: DesktopUpdateCommandPlan(
            executable=uv,
            arguments=("--no-config", "pip", "install", f"openhcs=={version}"),
        ),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.subprocess.Popen",
        lambda command, **kwargs: launched.append((command, kwargs)),
    )
    service = DesktopUpdateService(
        installed_version="0.6.2",
        system_name="Linux",
        network_manager=_NetworkManager(_UnavailableReply()),
    )

    assert service.start_update(
        update,
        runtime=runtime,
        session=session,
        parent_pid=42,
    )

    command, launch_kwargs = launched[0]
    assert command[0] == str(worker_python)
    arguments = command[1:]
    assert arguments[0] == str(session.worker_document)
    assert "--update-argument=--no-config" in arguments
    assert "--restart-argument=--log-level" in arguments
    assert arguments[arguments.index("--verification-executable") + 1] == str(python)
    assert "--restore-option=--restore-update-session" in arguments
    assert arguments[arguments.index("--parent-pid") + 1] == "42"
    assert "--background-creationflags=0" in arguments
    assert "--detached-creationflags=0" in arguments
    assert "--detached-start-new-session" in arguments
    assert launch_kwargs["start_new_session"] is True
    assert launch_kwargs["stdin"] is subprocess.DEVNULL
    assert launch_kwargs["stdout"] is subprocess.DEVNULL
    assert launch_kwargs["stderr"] is subprocess.DEVNULL


def test_runtime_environment_rejects_distribution_outside_virtual_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment_root = tmp_path / "venv"
    environment_root.mkdir()
    source_root = tmp_path / "source"
    source_root.mkdir()
    python = environment_root / "python"
    python.touch()
    base_python = tmp_path / "base-python"
    base_python.touch()
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.executable",
        str(python),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.prefix",
        str(environment_root),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.base_prefix",
        str(tmp_path / "base"),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys._base_executable",
        str(base_python),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.distribution",
        lambda _name: SimpleNamespace(
            locate_file=lambda _path: source_root,
            read_text=lambda _name: None,
        ),
    )

    with pytest.raises(DesktopUpdateError, match="editable or source"):
        DesktopRuntimeEnvironment.current()


def test_runtime_environment_rejects_worker_interpreter_inside_target_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment_root = tmp_path / "venv"
    distribution_root = environment_root / "lib" / "site-packages"
    distribution_root.mkdir(parents=True)
    python = environment_root / "bin" / "python"
    python.parent.mkdir()
    python.touch()
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.executable",
        str(python),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys._base_executable",
        str(python),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.prefix",
        str(environment_root),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.base_prefix",
        str(tmp_path / "base"),
    )

    with pytest.raises(DesktopUpdateError, match="base Python interpreter outside"):
        DesktopRuntimeEnvironment.current()


def test_runtime_environment_derives_restart_from_installed_entry_point(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment_root = tmp_path / "venv"
    distribution_root = environment_root / "lib" / "site-packages"
    distribution_root.mkdir(parents=True)
    python = environment_root / "bin" / "python"
    python.parent.mkdir()
    python.touch()
    base_python = tmp_path / "base-python"
    base_python.touch()
    entry_point = environment_root / "bin" / "openhcs"
    entry_point.touch()
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.executable",
        str(python),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.prefix",
        str(environment_root),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.base_prefix",
        str(tmp_path / "base"),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys._base_executable",
        str(base_python),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.argv",
        [
            str(entry_point),
            "--log-level",
            "DEBUG",
            "--restore-update-session",
            str(tmp_path / "old-session"),
            f"--restore-update-session={tmp_path / 'older-session'}",
            "--config=--leading-dash-value",
        ],
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.distribution",
        lambda _name: SimpleNamespace(
            locate_file=lambda _path: distribution_root,
            read_text=lambda _name: None,
        ),
    )

    runtime = DesktopRuntimeEnvironment.current()

    assert runtime.python_executable == python.resolve()
    assert runtime.worker_python_executable == base_python.resolve()
    assert runtime.environment_root == environment_root.resolve()
    assert runtime.restart_executable == entry_point.resolve()
    assert runtime.restart_arguments == (
        "--log-level",
        "DEBUG",
        "--config=--leading-dash-value",
    )


def test_runtime_environment_preserves_virtual_environment_python_symlink(
    monkeypatch,
    tmp_path: Path,
) -> None:
    uv_executable = tmp_path / "uv"
    uv_executable.touch()
    monkeypatch.setenv("OPENHCS_UV_EXECUTABLE", str(uv_executable))
    environment_root = tmp_path / "venv"
    distribution_root = environment_root / "lib" / "site-packages"
    distribution_root.mkdir(parents=True)
    base_python = tmp_path / "managed-python"
    base_python.touch()
    python = environment_root / "bin" / "python"
    python.parent.mkdir()
    python.symlink_to(base_python)
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.executable",
        str(python),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys._base_executable",
        str(base_python),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.prefix",
        str(environment_root),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.base_prefix",
        str(tmp_path / "base"),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.argv",
        ["openhcs-gui"],
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.distribution",
        lambda _name: SimpleNamespace(
            locate_file=lambda _path: distribution_root,
            read_text=lambda _name: None,
        ),
    )

    runtime = DesktopRuntimeEnvironment.current()

    assert runtime.python_executable == python
    assert runtime.worker_python_executable == base_python.resolve()
    assert runtime.update_command(
        parse_latest_release(
            _release_payload(),
            installed_version="0.6.99",
            system_name="Linux",
        ).latest_version
    ).arguments[4] == str(python)


def test_runtime_environment_rejects_read_only_install(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment_root = tmp_path / "venv"
    distribution_root = environment_root / "lib" / "site-packages"
    distribution_root.mkdir(parents=True)
    python = environment_root / "bin" / "python"
    python.parent.mkdir()
    python.touch()
    base_python = tmp_path / "base-python"
    base_python.touch()
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.executable",
        str(python),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.prefix",
        str(environment_root),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys.base_prefix",
        str(tmp_path / "base"),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.sys._base_executable",
        str(base_python),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.distribution",
        lambda _name: SimpleNamespace(
            locate_file=lambda _path: distribution_root,
            read_text=lambda _name: None,
        ),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.os.access",
        lambda *_args: False,
    )

    with pytest.raises(DesktopUpdateError, match="not writable"):
        DesktopRuntimeEnvironment.current()


def test_capture_uses_canonical_plate_source_and_objectstate_history(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plate_manager = SimpleNamespace(
        is_any_plate_running=lambda: False,
        orchestrator_code_document_context=lambda **_kwargs: SimpleNamespace(
            source="canonical session source"
        ),
    )
    main_window = SimpleNamespace(
        embedded_widgets=SimpleNamespace(
            require_plate_manager=lambda: plate_manager,
        ),
        runtime_context=SimpleNamespace(ui_config=object()),
    )
    monkeypatch.setattr(
        "openhcs.core.xdg_paths.get_openhcs_cache_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.config.save_ui_config_sync",
        lambda _config: True,
    )
    monkeypatch.setattr(
        "objectstate.object_state.ObjectStateRegistry.save_history_to_file",
        lambda path: Path(path).write_text("canonical history", encoding="utf-8"),
    )

    session = DesktopUpdateSession.capture(main_window)

    assert session.session_document.read_text(encoding="utf-8") == (
        "canonical session source"
    )
    assert session.history_document.read_text(encoding="utf-8") == ("canonical history")
    assert session.worker_document.read_text(encoding="utf-8").startswith(
        '"""Out-of-process OpenHCS environment update'
    )
    session.discard()


def test_saved_update_session_restores_through_existing_authorities(
    monkeypatch,
    tmp_path: Path,
) -> None:
    session = DesktopUpdateSession(tmp_path)
    session.session_document.write_text("plate_paths = []", encoding="utf-8")
    session.history_document.write_text("{}", encoding="utf-8")
    session.update_error_document.write_text("install failed", encoding="utf-8")
    calls = []
    plate_manager = SimpleNamespace(
        apply_code_document_source=lambda source: calls.append(("source", source)),
        update_item_list=lambda: calls.append(("refresh", None)),
    )
    time_travel = SimpleNamespace(refresh=lambda: calls.append(("history-ui", None)))
    main_window = SimpleNamespace(
        embedded_widgets=SimpleNamespace(
            require_plate_manager=lambda: plate_manager,
        ),
        time_travel_widget=time_travel,
    )
    monkeypatch.setattr(
        "objectstate.object_state.ObjectStateRegistry.load_history_from_file",
        lambda path: calls.append(("history", path)),
    )

    update_error = session.restore(main_window)

    assert update_error == "install failed"
    assert calls == [
        ("source", "plate_paths = []"),
        ("history", str(session.history_document)),
        ("history-ui", None),
        ("refresh", None),
    ]
    assert not tmp_path.exists()

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt6.QtNetwork import QNetworkReply
from PyQt6.QtWidgets import QMessageBox, QWidget

import openhcs.pyqt_gui.main as main_module
from openhcs import __version__ as OPENHCS_VERSION
from openhcs.pyqt_gui.services.service_adapter import PyQtServiceAdapter
from openhcs.pyqt_gui.services.desktop_update import (
    LATEST_RELEASE_API_URL,
    DesktopRuntimeEnvironment,
    DesktopUpdateCheckFailure,
    DesktopUpdateCheckOrigin,
    DesktopUpdateCheckResult,
    DesktopUpdateCommandPlan,
    DesktopUpdateDialogPresenter,
    DesktopUpdateError,
    DesktopUpdateService,
    DesktopUpdateSession,
    parse_latest_release,
)
from openhcs.pyqt_gui.services.desktop_update_worker import DesktopUpdateProgressTheme
from openhcs.resources.brand import BrandAsset, brand_asset_bytes
from pyqt_reactive.process_launch import BackgroundProcessPlatform
from pyqt_reactive.theming import ColorScheme


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


def test_update_service_defaults_to_source_version_authority(qapp) -> None:
    service = DesktopUpdateService()

    assert service._installed_version == OPENHCS_VERSION


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

    assert service.check_for_updates(DesktopUpdateCheckOrigin.STARTUP)
    assert manager.request.url().toString() == LATEST_RELEASE_API_URL
    assert failures == []

    reply.finished.emit()

    assert failures == [
        DesktopUpdateCheckFailure(
            message="Host not found",
            origin=DesktopUpdateCheckOrigin.STARTUP,
        )
    ]
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
        self.origins = []

    def check_for_updates(self, origin) -> bool:
        self.origins.append(origin)
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
    assert main_like.desktop_update_service.origins == [
        DesktopUpdateCheckOrigin.EXPLICIT
    ]
    assert status.messages == ["Checking for OpenHCS updates…"]


@pytest.mark.parametrize(
    ("enabled", "expected_origins", "expected_action_enabled"),
    [
        (True, [DesktopUpdateCheckOrigin.STARTUP], False),
        (False, [], True),
    ],
)
def test_main_window_startup_check_obeys_ui_config_without_blocking(
    enabled,
    expected_origins,
    expected_action_enabled,
) -> None:
    service = _UpdateService()
    action = _Action()
    main_like = SimpleNamespace(
        runtime_context=SimpleNamespace(
            ui_config=SimpleNamespace(check_for_updates_on_startup=enabled)
        ),
        desktop_update_service=service,
        check_for_updates_action=action,
    )

    main_module.OpenHCSMainWindow._check_for_updates_on_startup(main_like)

    assert service.origins == expected_origins
    assert action.enabled is expected_action_enabled


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
    runtime = object()
    session = SimpleNamespace(discard=lambda: None)
    closed = []
    presenter = SimpleNamespace(
        confirm_update=lambda candidate: candidate is update,
        show_up_to_date=lambda _candidate: pytest.fail("update is available"),
        show_warning=lambda _message: pytest.fail("handoff must succeed"),
    )
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
        desktop_update_presenter=presenter,
        close=lambda: closed.append(True),
    )

    main_module.OpenHCSMainWindow._on_update_check_completed(
        main_like,
        DesktopUpdateCheckResult(
            update=update,
            origin=DesktopUpdateCheckOrigin.STARTUP,
        ),
    )

    assert action.enabled
    assert status.messages == [
        "OpenHCS 0.7.0 is available",
        "OpenHCS update prepared; restarting…",
    ]
    assert service.started == [(update, runtime, session)]
    assert closed == [True]


def test_startup_check_is_quiet_when_current_or_unavailable() -> None:
    current = parse_latest_release(
        _release_payload("0.6.2"),
        installed_version="0.6.2",
        system_name="Windows",
    )
    action = _Action()
    action.enabled = False
    status = _StatusSignal()
    presented = []
    main_like = SimpleNamespace(
        check_for_updates_action=action,
        status_message=status,
        desktop_update_presenter=SimpleNamespace(
            show_up_to_date=lambda update: presented.append(update),
            show_warning=presented.append,
        ),
    )

    main_module.OpenHCSMainWindow._on_update_check_completed(
        main_like,
        DesktopUpdateCheckResult(
            update=current,
            origin=DesktopUpdateCheckOrigin.STARTUP,
        ),
    )
    main_module.OpenHCSMainWindow._on_update_check_failed(
        main_like,
        DesktopUpdateCheckFailure(
            message="offline",
            origin=DesktopUpdateCheckOrigin.STARTUP,
        ),
    )

    assert action.enabled
    assert status.messages == []
    assert presented == []


def test_update_presenter_scopes_shared_dark_theme_to_message_box(qapp) -> None:
    parent = QWidget()
    scheme = ColorScheme()
    dialog_service = object.__new__(PyQtServiceAdapter)
    dialog_service.main_window = parent
    dialog_service.theme_manager = SimpleNamespace(color_scheme=scheme)
    message_box = dialog_service.create_message_box(
        icon=QMessageBox.Icon.Question,
        title="OpenHCS Update Available",
        text="Install the update now?",
        buttons=(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel),
        default_button=QMessageBox.StandardButton.Yes,
    )

    try:
        message_box.show()
        qapp.processEvents()
        stylesheet = message_box.styleSheet()
        assert "QDialog" in stylesheet
        assert "QPushButton" in stylesheet
        assert scheme.to_hex(scheme.window_bg) in stylesheet
        assert scheme.to_hex(scheme.button_normal_bg) in stylesheet
        rendered_colors = {
            message_box.grab().toImage().pixelColor(x, y).name()
            for x in range(message_box.width())
            for y in range(message_box.height())
        }
        assert scheme.to_hex(scheme.window_bg) in rendered_colors
    finally:
        message_box.close()
        parent.close()


def test_update_presenter_owns_update_wording_not_dialog_construction() -> None:
    update = parse_latest_release(
        _release_payload(),
        installed_version="0.6.2",
        system_name="Windows",
    )
    calls = []

    class _MessageBox:
        def exec(self):
            return QMessageBox.StandardButton.Yes.value

    dialog_service = SimpleNamespace(
        create_message_box=lambda **kwargs: calls.append(kwargs) or _MessageBox()
    )
    presenter = DesktopUpdateDialogPresenter(dialog_service)

    assert presenter.confirm_update(update)
    assert len(calls) == 1
    assert "working session and edit history" in calls[0]["text"]
    assert "ObjectState" not in calls[0]["text"]
    assert calls[0]["default_button"] is QMessageBox.StandardButton.Yes


def test_update_command_uses_environment_pip(monkeypatch, tmp_path: Path) -> None:
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
    )

    assert command.executable == python
    assert command.arguments == (
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--upgrade",
        "openhcs==0.7.0",
    )


def test_update_command_uses_running_interpreter_pip(
    monkeypatch,
    tmp_path: Path,
) -> None:
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
    )

    assert command.executable == python
    assert command.arguments[:3] == ("-m", "pip", "install")
    assert command.arguments[-1] == "openhcs==0.7.0"


def test_update_command_rejects_environment_without_pip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.find_spec",
        lambda _name: None,
    )

    with pytest.raises(DesktopUpdateError, match="has no pip module"):
        DesktopUpdateCommandPlan.for_environment(
            python_executable=tmp_path / "python",
            latest_version=parse_latest_release(
                _release_payload(),
                installed_version="0.6.2",
                system_name="Linux",
            ).latest_version,
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
        installation_pointer=tmp_path / "Launch-OpenHCS.ps1",
    )
    session = DesktopUpdateSession(tmp_path / "pending")
    session.directory.mkdir()
    session.worker_document.write_text("worker", encoding="utf-8")
    session.progress_theme_document.write_text("{}", encoding="utf-8")
    session.progress_brand_document.write_bytes(b"brand")
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
    assert command[:2] == [str(worker_python), "-I"]
    arguments = command[2:]
    assert arguments[0] == str(session.worker_document)
    assert "--update-argument=--no-config" in arguments
    assert "--restart-argument=--log-level" in arguments
    assert arguments[arguments.index("--verification-executable") + 1] == str(python)
    assert arguments[arguments.index("--progress-theme-file") + 1] == str(
        session.progress_theme_document
    )
    assert arguments[arguments.index("--progress-brand-file") + 1] == str(
        session.progress_brand_document
    )
    assert "--restore-option=--restore-update-session" in arguments
    assert (
        f"--installation-pointer={tmp_path / 'Launch-OpenHCS.ps1'}" in arguments
    )
    assert arguments[arguments.index("--parent-pid") + 1] == "42"
    assert "--background-creationflags=0" in arguments
    assert "--detached-creationflags=0" in arguments
    assert "--detached-start-new-session" in arguments
    assert launch_kwargs["start_new_session"] is True
    assert launch_kwargs["stdin"] is subprocess.DEVNULL
    assert launch_kwargs["stdout"] is subprocess.DEVNULL
    assert launch_kwargs["stderr"] is subprocess.DEVNULL


def test_windows_update_worker_uses_windowed_interpreter_and_no_console(
    monkeypatch,
    tmp_path: Path,
) -> None:
    update = parse_latest_release(
        _release_payload(),
        installed_version="0.6.2",
        system_name="Windows",
    )
    worker_python = tmp_path / "base" / "python.exe"
    worker_python.parent.mkdir()
    worker_python.touch()
    worker_pythonw = worker_python.with_name("pythonw.exe")
    worker_pythonw.touch()
    runtime = DesktopRuntimeEnvironment(
        python_executable=tmp_path / "environment" / "python.exe",
        worker_python_executable=worker_python,
        environment_root=tmp_path / "environment",
        restart_executable=tmp_path / "environment" / "openhcs-gui.exe",
        restart_arguments=(),
        installation_pointer=tmp_path / "Launch-OpenHCS.ps1",
    )
    session = DesktopUpdateSession(tmp_path / "pending")
    session.directory.mkdir()
    session.worker_document.write_text("worker", encoding="utf-8")
    session.progress_theme_document.write_text("{}", encoding="utf-8")
    session.progress_brand_document.write_bytes(b"brand")
    uv = tmp_path / "uv.exe"
    launched = []
    create_no_window = 0x08000000
    create_new_process_group = 0x00000200
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", create_no_window, raising=False)
    monkeypatch.setattr(
        subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        create_new_process_group,
        raising=False,
    )
    monkeypatch.setattr(
        BackgroundProcessPlatform,
        "current",
        classmethod(lambda cls: cls.WINDOWS),
    )
    monkeypatch.setattr(
        DesktopRuntimeEnvironment,
        "update_command",
        lambda self, version: DesktopUpdateCommandPlan(
            executable=uv,
            arguments=("pip", "install", f"openhcs=={version}"),
        ),
    )
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.subprocess.Popen",
        lambda command, **kwargs: launched.append((command, kwargs)),
    )
    service = DesktopUpdateService(
        installed_version="0.6.2",
        system_name="Windows",
        network_manager=_NetworkManager(_UnavailableReply()),
    )

    assert service.start_update(
        update,
        runtime=runtime,
        session=session,
        parent_pid=42,
    )

    command, launch_kwargs = launched[0]
    assert command[:2] == [str(worker_pythonw), "-I"]
    arguments = command[2:]
    assert f"--background-creationflags={create_no_window}" in arguments
    assert (
        f"--detached-creationflags={create_no_window | create_new_process_group}"
        in arguments
    )
    assert launch_kwargs["creationflags"] == (
        create_no_window | create_new_process_group
    )
    assert "start_new_session" not in launch_kwargs


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
    installation_pointer = tmp_path / "Launch-OpenHCS.ps1"
    monkeypatch.setenv(
        "OPENHCS_MCP_INSTALLATION_POINTER",
        str(installation_pointer),
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
    assert runtime.installation_pointer == installation_pointer


def test_runtime_environment_preserves_virtual_environment_python_symlink(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_update.find_spec",
        lambda name: object() if name == "pip" else None,
    )
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
    ).executable == python


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
        window_services=SimpleNamespace(
            get_current_color_scheme=lambda: ColorScheme(),
        ),
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
    assert DesktopUpdateProgressTheme.read(session.progress_theme_document) == (
        DesktopUpdateProgressTheme(
            window_bg="#2b2b2b",
            panel_bg="#1e1e1e",
            text_primary="#ffffff",
            text_secondary="#cccccc",
            text_accent="#00aaff",
            border_color="#555555",
            button_bg="#404040",
            button_text="#ffffff",
            error_color="#ff0000",
            progress_color="#0078d4",
        )
    )
    assert session.progress_brand_document.read_bytes() == brand_asset_bytes(
        BrandAsset.ICON_RASTER
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

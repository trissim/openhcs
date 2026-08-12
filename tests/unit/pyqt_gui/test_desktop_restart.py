"""Desktop session restart handoff tests."""

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

from PyQt6.QtWidgets import QMessageBox, QWidget
from openhcs.pyqt_gui.services.desktop_restart import DesktopSessionRestart
from openhcs.pyqt_gui.services.desktop_update import (
    DesktopRestartEnvironment,
    DesktopRestartPurpose,
    DesktopRestartSession,
    UPDATE_SESSION_ARGUMENT,
)
from openhcs.pyqt_gui.services.service_adapter import PyQtServiceAdapter
from openhcs.pyqt_gui.services.zmq_version_restart import (
    ZMQVersionRestartDialogPresenter,
)
from openhcs.runtime.zmq_application import OPENHCS_ENDPOINT_APPLICATION
from openhcs.runtime.zmq_application import OpenHCSEndpointCompatibility
from pyqt_reactive.theming import ColorScheme
from zmqruntime import EndpointApplication


def test_desktop_restart_worker_receives_session_and_restart_arguments(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = DesktopRestartEnvironment(
        worker_python_executable=tmp_path / "base-python",
        restart_executable=tmp_path / "openhcs-gui",
        restart_arguments=("--log-level", "DEBUG"),
    )
    session = DesktopRestartSession(tmp_path / "pending")
    session.directory.mkdir()
    session.session_document.write_text("source", encoding="utf-8")
    session.history_document.write_text("history", encoding="utf-8")
    launched = []
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.desktop_restart.subprocess.Popen",
        lambda command, **kwargs: launched.append((command, kwargs)),
    )

    assert DesktopSessionRestart(runtime, session).start(parent_pid=42)

    command, launch_kwargs = launched[0]
    assert command[command.index("--parent-pid") + 1] == "42"
    assert f"--restart-argument={UPDATE_SESSION_ARGUMENT}" in command
    assert f"--restart-argument={session.directory}" in command
    assert "--restart-argument=--log-level" in command
    assert "--restart-argument=DEBUG" in command
    assert launch_kwargs["stdin"] is subprocess.DEVNULL
    assert launch_kwargs["stdout"] is subprocess.DEVNULL
    assert launch_kwargs["stderr"] is subprocess.DEVNULL


def test_restart_purpose_owns_the_post_restore_message(tmp_path: Path) -> None:
    session = DesktopRestartSession(tmp_path)
    session.purpose_document.write_text(
        DesktopRestartPurpose.ZMQ_VERSION.value,
        encoding="utf-8",
    )

    assert session.purpose is DesktopRestartPurpose.ZMQ_VERSION
    assert "matching execution server" in session.purpose.success_message


def test_version_restart_capture_omits_update_only_assets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plate_manager = SimpleNamespace(
        is_any_plate_running=lambda: False,
        orchestrator_code_document_context=lambda **_kwargs: SimpleNamespace(
            source="plate_paths = []"
        ),
    )
    main_window = SimpleNamespace(
        embedded_widgets=SimpleNamespace(
            require_plate_manager=lambda: plate_manager,
        ),
        runtime_context=SimpleNamespace(ui_config=object()),
        window_services=SimpleNamespace(
            get_current_color_scheme=lambda: (_ for _ in ()).throw(
                AssertionError("version restart must not prepare updater assets")
            ),
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
        lambda path: Path(path).write_text("history", encoding="utf-8"),
    )

    session = DesktopRestartSession.capture(
        main_window,
        purpose=DesktopRestartPurpose.ZMQ_VERSION,
    )

    assert session.is_complete
    assert session.purpose is DesktopRestartPurpose.ZMQ_VERSION
    assert not session.worker_document.exists()
    assert not session.progress_theme_document.exists()
    assert not session.progress_brand_document.exists()


def test_version_mismatch_dialog_is_themed_and_reports_both_versions(
    qapp,
    tmp_path: Path,
) -> None:
    parent = QWidget()
    scheme = ColorScheme()
    dialog_service = object.__new__(PyQtServiceAdapter)
    dialog_service.main_window = parent
    dialog_service.theme_manager = SimpleNamespace(color_scheme=scheme)
    presenter = ZMQVersionRestartDialogPresenter(dialog_service)
    compatibility = OpenHCSEndpointCompatibility(
        expected=OPENHCS_ENDPOINT_APPLICATION,
        observed=EndpointApplication(identifier="openhcs", version="0.7.20"),
    )
    captured = []
    original = dialog_service.create_message_box

    def create_message_box(**kwargs):
        message_box = original(**kwargs)
        captured.append(message_box)
        return message_box

    dialog_service.create_message_box = create_message_box
    qapp.processEvents()
    from PyQt6.QtCore import QTimer

    QTimer.singleShot(
        0,
        lambda: captured[0].button(QMessageBox.StandardButton.Cancel).click(),
    )
    try:
        assert presenter.confirm_restart(compatibility) is False
        message_box = captured[0]
        assert OPENHCS_ENDPOINT_APPLICATION.version in message_box.text()
        assert "0.7.20" in message_box.text()
        assert scheme.to_hex(scheme.window_bg) in message_box.styleSheet()
        image = message_box.grab().toImage()
        assert image.save(str(tmp_path / "version-mismatch-dialog.png"))
    finally:
        for message_box in captured:
            message_box.close()
        parent.close()

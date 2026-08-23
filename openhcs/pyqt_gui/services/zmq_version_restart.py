"""Presentation for OpenHCS endpoint version mismatches."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from openhcs.core.execution_state import ManagerExecutionState
from openhcs.pyqt_gui.services.desktop_restart import DesktopSessionRestart

if TYPE_CHECKING:
    from zmqruntime import EndpointApplicationCompatibility

    from openhcs.pyqt_gui.main import OpenHCSMainWindow
    from openhcs.pyqt_gui.services.service_adapter import PyQtServiceAdapter
    from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import (
        ZMQClientService,
    )


class ZMQVersionRestartDialogPresenter:
    """Present the decision required by one endpoint version mismatch."""

    def __init__(self, dialog_service: "PyQtServiceAdapter") -> None:
        self._dialog_service = dialog_service

    def confirm_restart(
        self,
        compatibility: "EndpointApplicationCompatibility",
    ) -> bool:
        response = QMessageBox.StandardButton(
            self._dialog_service.create_message_box(
                icon=QMessageBox.Icon.Warning,
                title="OpenHCS Version Mismatch",
                text=(
                    f"This OpenHCS UI is version {compatibility.expected.version}, "
                    "but the connected execution server reports "
                    f"{compatibility.observed_version_label}.\n\n"
                    "Restart the execution server and OpenHCS with the UI version? "
                    "The current session and edit history will be restored."
                ),
                buttons=(
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
                ),
                default_button=QMessageBox.StandardButton.Yes,
            ).exec()
        )
        return response == QMessageBox.StandardButton.Yes

    def show_failure(self, message: str) -> None:
        self._dialog_service.create_message_box(
            icon=QMessageBox.Icon.Warning,
            title="OpenHCS Version Restart",
            text=message,
            buttons=QMessageBox.StandardButton.Ok,
            default_button=QMessageBox.StandardButton.Ok,
        ).exec()


class ZMQVersionRestartWorkflow(QObject):
    """Own the complete state-preserving response to endpoint incompatibility."""

    replacement_completed = pyqtSignal()
    replacement_failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        main_window: OpenHCSMainWindow,
        client_service: ZMQClientService,
        execution_state: Callable[[], ManagerExecutionState],
        execute_async: Callable[..., object],
        publish_status: Callable[[str], None],
        presenter: ZMQVersionRestartDialogPresenter,
    ) -> None:
        super().__init__(main_window)
        self._main_window = main_window
        self._client_service = client_service
        self._execution_state = execution_state
        self._execute_async = execute_async
        self._publish_status = publish_status
        self._presenter = presenter
        self._pending_session: DesktopSessionRestart | None = None
        self._deferred_compatibility: EndpointApplicationCompatibility | None = None
        self.replacement_completed.connect(self._complete)
        self.replacement_failed.connect(self._fail)

    def observe_compatibility(
        self,
        compatibility: EndpointApplicationCompatibility,
    ) -> None:
        """Admit a match or begin exactly one user-approved replacement."""

        if compatibility.matches:
            self._deferred_compatibility = None
            return
        if self._pending_session is not None:
            return
        if self._execution_state().busy:
            self._deferred_compatibility = compatibility
            self._publish_status(
                "ZMQ version replacement will be offered after the current "
                "operation finishes"
            )
            return
        self._deferred_compatibility = None
        if not self._presenter.confirm_restart(compatibility):
            return
        try:
            self._pending_session = DesktopSessionRestart.capture(self._main_window)
        except Exception as error:
            self._presenter.show_failure(
                f"OpenHCS could not save the current session for restart.\n\n{error}"
            )
            return
        self._publish_status("Replacing the mismatched ZMQ execution server…")
        self._execute_async(self._replace_endpoint, compatibility)

    def observe_execution_state(self, state: ManagerExecutionState) -> None:
        """Resume a deferred replacement after the batch owner becomes idle."""

        if state.busy or self._deferred_compatibility is None:
            return
        compatibility = self._deferred_compatibility
        self._deferred_compatibility = None
        self.observe_compatibility(compatibility)

    async def _replace_endpoint(
        self,
        compatibility: EndpointApplicationCompatibility,
    ) -> None:
        try:
            await self._client_service.restart_endpoint(
                expected_compatibility=compatibility,
                persistent=True,
            )
        except Exception as error:
            self.replacement_failed.emit(str(error))
        else:
            self.replacement_completed.emit()

    def _complete(self) -> None:
        session = self._pending_session
        self._pending_session = None
        if session is None:
            return
        if session.start():
            self._publish_status("ZMQ server matched; restarting OpenHCS…")
            self._main_window.close()
            return
        session.discard()
        self._presenter.show_failure(
            "The matching ZMQ server started, but OpenHCS could not launch its "
            "session restart. The current application remains open."
        )

    def _fail(self, message: str) -> None:
        session = self._pending_session
        self._pending_session = None
        if session is not None:
            session.discard()
        self._publish_status("ZMQ server restart failed")
        self._presenter.show_failure(
            "OpenHCS could not replace the mismatched ZMQ server. The current "
            f"application and session remain open.\n\n{message}"
        )

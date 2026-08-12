"""Presentation for OpenHCS endpoint version mismatches."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox

if TYPE_CHECKING:
    from openhcs.pyqt_gui.services.service_adapter import PyQtServiceAdapter
    from openhcs.runtime.zmq_application import OpenHCSEndpointCompatibility


class ZMQVersionRestartDialogPresenter:
    """Present the decision required by one endpoint version mismatch."""

    def __init__(self, dialog_service: "PyQtServiceAdapter") -> None:
        self._dialog_service = dialog_service

    def confirm_restart(self, compatibility: "OpenHCSEndpointCompatibility") -> bool:
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
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.Cancel
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

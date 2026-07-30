"""Execution-server status line presenter for batch workflow UI."""

from __future__ import annotations

from dataclasses import dataclass
from openhcs.core.progress.projection import ExecutionRuntimeProjection
from pyqt_reactive.services.zmq_server_info import (
    ExecutionServerInfo,
)


@dataclass(frozen=True)
class ExecutionServerStatusView:
    """Rendered status text for plate-manager status bar."""

    text: str


class ExecutionServerStatusPresenter:
    """Build status text from runtime projection."""

    def build_status_text(
        self,
        *,
        projection: ExecutionRuntimeProjection,
        server_info: ExecutionServerInfo | None,
    ) -> ExecutionServerStatusView:
        plate_count = len(projection.by_plate_latest)
        if plate_count == 0:
            return ExecutionServerStatusView(text="Ready")

        parts = projection.count_status_labels()
        status_text = ", ".join(parts) if parts else "idle"
        return ExecutionServerStatusView(
            text=(
                f"Server: {status_text} | "
                f"{plate_count} plates | avg {projection.overall_percent:.1f}%"
            )
        )

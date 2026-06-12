"""Progress projection and server-status polling for PyQt batch workflows."""

from __future__ import annotations

import logging
from typing import Callable

from PyQt6.QtCore import QTimer

from openhcs.core.progress import ProgressEvent
from openhcs.core.progress.projection import (
    ExecutionRuntimeProjection,
    build_execution_runtime_projection,
)
from openhcs.pyqt_gui.config import ProgressUIConfig
from openhcs.pyqt_gui.widgets.shared.services.debug_progress_service import (
    DebugProgressNotificationService,
)
from openhcs.core.progress.live_measurements import LiveMeasurementPayloadError
from openhcs.pyqt_gui.widgets.shared.services.live_measurement_progress_service import (
    LiveMeasurementProgressNotificationService,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_server_status_presenter import (
    ExecutionServerStatusPresenter,
)
from openhcs.pyqt_gui.widgets.shared.services.progress_batch_reset import (
    reset_progress_views_for_new_batch,
)
from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import ZMQClientService
from pyqt_reactive.services import (
    CallbackIntervalSnapshotPollerPolicy,
    ExecutionServerInfo,
    IntervalSnapshotPoller,
    ServerInfoParserABC,
)

logger = logging.getLogger(__name__)


class ProgressWorkflowService:
    """Owns progress event registration, projection refresh, and server status."""

    def __init__(
        self,
        *,
        host,
        client_service: ZMQClientService,
        server_info_parser: ServerInfoParserABC,
        debug_notifications: DebugProgressNotificationService,
        status_presenter: ExecutionServerStatusPresenter,
        live_measurements: LiveMeasurementProgressNotificationService | None = None,
        on_dirty: Callable[[], None] | None = None,
        start_timer: bool = True,
    ) -> None:
        self._host = host
        self._client_service = client_service
        self._server_info_parser = server_info_parser
        self._debug_notifications = debug_notifications
        self._live_measurements = (
            live_measurements or LiveMeasurementProgressNotificationService()
        )
        self._status_presenter = status_presenter
        self._runtime_projection = ExecutionRuntimeProjection()
        self._progress_dirty = False
        self._on_dirty = on_dirty
        self._server_info_poller = IntervalSnapshotPoller[ExecutionServerInfo](
            CallbackIntervalSnapshotPollerPolicy(
                fetch_snapshot_fn=self._fetch_server_info_snapshot,
                clone_snapshot_fn=lambda snapshot: snapshot,
                poll_interval_seconds_value=1.0,
                on_snapshot_changed_fn=lambda _snapshot: self.mark_dirty(),
                on_poll_error_fn=lambda error: logger.debug(
                    "Server info poll failed: %s", error
                ),
            )
        )
        self._progress_coalesce_timer = None
        if start_timer:
            self._progress_coalesce_timer = QTimer()
            self._progress_coalesce_timer.timeout.connect(self.coalesced_update)
            self._progress_coalesce_timer.start(ProgressUIConfig().update_interval_ms)

    def cleanup(self) -> None:
        if self._progress_coalesce_timer is None:
            return
        self._progress_coalesce_timer.stop()
        self._progress_coalesce_timer.deleteLater()
        self._progress_coalesce_timer = None

    def reset_for_new_batch(self) -> None:
        self._runtime_projection = reset_progress_views_for_new_batch(
            self._host,
            projection=ExecutionRuntimeProjection(),
        )
        self._server_info_poller.reset()
        self.mark_dirty()

    def coalesced_update(self) -> None:
        if self._client_service.zmq_client is not None:
            self._server_info_poller.tick()
        if not self._progress_dirty:
            return
        self._progress_dirty = False
        progress_tracker = self._host._progress_tracker
        events_by_execution = {
            execution_id: progress_tracker.get_events(execution_id)
            for execution_id in progress_tracker.get_execution_ids()
        }
        self._runtime_projection = build_execution_runtime_projection(
            events_by_execution
        )
        self._host.runtime_progress_projection = self._runtime_projection
        self._host.execution_server_info = self.server_info_snapshot()
        self._emit_execution_server_status()
        self._host.update_item_list()

    def on_progress(self, message: dict) -> None:
        try:
            event = ProgressEvent.from_dict(message)
            self._host._progress_tracker.register_event(event.execution_id, event)
            self._debug_notifications.notify_from_progress_event(
                event,
                zmq_client=self._client_service.zmq_client,
            )
            try:
                self._live_measurements.notify_from_progress_event(event)
            except LiveMeasurementPayloadError as error:
                logger.warning(
                    "Malformed live measurement progress context for "
                    "execution_id=%s axis_id=%s step_name=%s: %s",
                    event.execution_id,
                    event.axis_id,
                    event.step_name,
                    error,
                )
        except Exception as error:
            logger.warning("Failed to parse/register progress event: %s", error)
        finally:
            self.mark_dirty()

    def mark_dirty(self, *_listener_event: object) -> None:
        self._progress_dirty = True
        if self._on_dirty is not None:
            self._on_dirty()

    def server_info_snapshot(self) -> ExecutionServerInfo | None:
        return self._server_info_poller.get_snapshot_copy()

    def _emit_execution_server_status(self) -> None:
        status_view = self._status_presenter.build_status_text(
            projection=self._runtime_projection,
            server_info=self.server_info_snapshot(),
        )
        self._host.emit_status(status_view.text)

    def _fetch_server_info_snapshot(self) -> ExecutionServerInfo:
        if self._client_service.zmq_client is None:
            raise RuntimeError("ZMQ client is not connected")
        pong = self._client_service.zmq_client.get_server_info_snapshot()
        parsed = self._server_info_parser.parse(pong.to_dict())
        if not isinstance(parsed, ExecutionServerInfo):
            raise ValueError(
                f"Expected ExecutionServerInfo, got {type(parsed).__name__}"
            )
        return parsed


def is_progress_workflow_service_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_progress_workflow_service_export(name, value)
)

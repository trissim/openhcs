"""Progress projection and server-status polling for PyQt batch workflows."""

from __future__ import annotations

import logging
from typing import Callable

from PyQt6.QtCore import QCoreApplication, QThread, QTimer
from pyqt_reactive.services.interval_snapshot_poller import (
    CallbackIntervalSnapshotPollerPolicy,
    IntervalSnapshotPoller,
)
from pyqt_reactive.services.zmq_server_info import (
    BaseServerInfo,
    ExecutionServerInfo,
)
from zmqruntime.progress import EventRegistryMutation

from openhcs.core.debug_session_projection import DebugSessionProjectionContext
from openhcs.core.progress import ProgressEvent
from openhcs.core.progress.debug_projection import (
    RuntimeProjectionBuilder,
    RuntimeProjectionSource,
)
from openhcs.core.progress.live_measurements import LiveMeasurementPayloadError
from openhcs.core.progress.projection import ExecutionRuntimeProjection
from openhcs.core.progress.runtime_artifacts import RuntimeArtifactPayloadError
from openhcs.pyqt_gui.config import ProgressUIConfig
from openhcs.pyqt_gui.widgets.shared.services.batch_context import (
    BatchWorkflowContext,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_progress_service import (
    DebugProgressNotificationService,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_server_status_presenter import (
    ExecutionServerStatusPresenter,
)
from openhcs.pyqt_gui.widgets.shared.services.live_measurement_progress_service import (
    LiveMeasurementProgressNotificationService,
)
from openhcs.pyqt_gui.widgets.shared.services.progress_batch_reset import (
    reset_progress_views_for_new_batch,
)
from openhcs.pyqt_gui.widgets.shared.services.runtime_artifact_progress_service import (
    RuntimeArtifactProgressNotificationService,
)

logger = logging.getLogger(__name__)


class ProgressWorkflowService:
    """Owns progress event registration, projection refresh, and server status."""

    def __init__(
        self,
        *,
        host,
        context: BatchWorkflowContext,
        debug_notifications: DebugProgressNotificationService,
        status_presenter: ExecutionServerStatusPresenter,
        config: ProgressUIConfig,
        live_measurements: LiveMeasurementProgressNotificationService | None = None,
        runtime_artifacts: RuntimeArtifactProgressNotificationService | None = None,
        debug_session_context_provider: (
            Callable[[], DebugSessionProjectionContext | None] | None
        ) = None,
        on_dirty: Callable[[], None] | None = None,
    ) -> None:
        self._host = host
        self._context = context
        self._debug_notifications = debug_notifications
        self._live_measurements = (
            live_measurements or LiveMeasurementProgressNotificationService()
        )
        self._runtime_artifacts = (
            runtime_artifacts or RuntimeArtifactProgressNotificationService()
        )
        self._status_presenter = status_presenter
        self._config = config
        self._debug_session_context_provider = debug_session_context_provider
        self._runtime_projection_builder = RuntimeProjectionBuilder()
        self._progress_dirty = False
        self._on_dirty = on_dirty
        self._registry_listener = self._on_registry_mutation
        self._host._progress_tracker.add_mutation_listener(self._registry_listener)
        self._registry_listener_registered = True
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

    def start(self) -> None:
        """Start UI projection refresh on the Qt application thread."""

        if self._progress_coalesce_timer is not None:
            return
        application = QCoreApplication.instance()
        if application is None:
            raise RuntimeError(
                "ProgressWorkflowService requires a running Qt application."
            )
        if QThread.currentThread() != application.thread():
            raise RuntimeError(
                "ProgressWorkflowService must start on the Qt application thread."
            )
        self._progress_coalesce_timer = QTimer()
        self._progress_coalesce_timer.timeout.connect(self.coalesced_update)
        self._progress_coalesce_timer.start(self._config.update_interval_ms)

    def update_config(self, config: ProgressUIConfig) -> None:
        """Apply the exact process configuration to the live coalescing timer."""

        self._config = config
        if self._progress_coalesce_timer is not None:
            self._progress_coalesce_timer.setInterval(config.update_interval_ms)

    def cleanup(self) -> None:
        if self._registry_listener_registered:
            removed = self._host._progress_tracker.remove_mutation_listener(
                self._registry_listener
            )
            if not removed:
                raise RuntimeError(
                    "ProgressWorkflowService listener removal failed: "
                    "listener not registered"
                )
            self._registry_listener_registered = False
        if self._progress_coalesce_timer is not None:
            self._progress_coalesce_timer.stop()
            self._progress_coalesce_timer.deleteLater()
            self._progress_coalesce_timer = None

    def reset_for_new_batch(self) -> None:
        reset_progress_views_for_new_batch(self._host)
        self._server_info_poller.reset()
        self.mark_dirty()

    def coalesced_update(self) -> None:
        if self._context.zmq.has_client():
            self._server_info_poller.tick()
        if not self._progress_dirty:
            return
        self._progress_dirty = False
        self.rebuild_runtime_projection()

    def clear_execution(self, execution_id: str) -> None:
        """Remove one execution's progress and immediately refresh projections."""
        self._host._progress_tracker.clear_execution(execution_id)
        self.rebuild_runtime_projection()

    def rebuild_runtime_projection(self) -> None:
        """Rebuild the host-facing runtime projection from tracked events."""
        self._progress_dirty = False
        progress_tracker = self._host._progress_tracker
        events_by_execution = {
            execution_id: progress_tracker.get_events(execution_id)
            for execution_id in progress_tracker.get_execution_ids()
        }
        server_info = self.server_info_snapshot()
        debug_context = self._debug_session_context()
        runtime_projection_bundle = self._runtime_projection_builder.build(
            RuntimeProjectionSource(
                events_by_execution=events_by_execution,
                running_executions=(
                    () if server_info is None else server_info.running_execution_entries
                ),
                queued_executions=(
                    () if server_info is None else server_info.queued_execution_entries
                ),
                session=None if debug_context is None else debug_context.active_session,
                terminal_summary=(
                    None if debug_context is None else debug_context.terminal_summary
                ),
                snapshots=() if debug_context is None else debug_context.snapshots,
            )
        )
        self._host.apply_runtime_projection(runtime_projection_bundle)
        self._emit_execution_server_status(runtime_projection_bundle.execution)
        self._host.update_item_list()

    def on_progress(self, message: dict) -> None:
        try:
            event = ProgressEvent.from_dict(message)
            accepted = self._host._progress_tracker.register_event(
                event.execution_id,
                event,
            )
            if not accepted:
                return
            self._debug_notifications.notify_from_progress_event(
                event,
                zmq_client=self._context.zmq.current_client,
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
            try:
                self._runtime_artifacts.notify_from_progress_event(event)
            except RuntimeArtifactPayloadError as error:
                logger.warning(
                    "Malformed runtime artifact progress context for "
                    "execution_id=%s axis_id=%s step_name=%s: %s",
                    event.execution_id,
                    event.axis_id,
                    event.step_name,
                    error,
                )
        except Exception as error:
            logger.warning("Failed to parse/register progress event: %s", error)

    def _on_registry_mutation(
        self,
        _mutation: EventRegistryMutation[ProgressEvent],
    ) -> None:
        self.mark_dirty()

    def mark_dirty(self) -> None:
        self._progress_dirty = True
        if self._on_dirty is not None:
            self._on_dirty()

    def server_info_snapshot(self) -> ExecutionServerInfo | None:
        return self._server_info_poller.get_snapshot_copy()

    def _debug_session_context(self) -> DebugSessionProjectionContext | None:
        if self._debug_session_context_provider is None:
            return None
        return self._debug_session_context_provider()

    def _emit_execution_server_status(
        self,
        projection: ExecutionRuntimeProjection,
    ) -> None:
        status_view = self._status_presenter.build_status_text(
            projection=projection,
        )
        self._host.emit_status(status_view.text)

    def _fetch_server_info_snapshot(self) -> ExecutionServerInfo:
        pong = self._context.zmq.require_client().get_server_info_snapshot()
        parsed = BaseServerInfo.from_response(pong)
        if not isinstance(parsed, ExecutionServerInfo):
            raise ValueError(
                f"Expected ExecutionServerInfo, got {type(parsed).__name__}"
            )
        return parsed


def is_progress_workflow_service_export(name: str, value) -> bool:
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

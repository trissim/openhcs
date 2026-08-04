"""PlateManager batch workflow orchestration."""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

import asyncio

from openhcs.core.orchestrator.orchestrator import OrchestratorState
from openhcs.core.debug import (
    DebugArtifactRef,
    DebugArtifactExportResponse,
    DebugCommandType,
    DebugPausedWorkerStatus,
    DebugReplayMode,
)
from openhcs.pyqt_gui.widgets.shared.services.batch_workflow_components import (
    BatchWorkflowComponents,
)
from openhcs.pyqt_gui.widgets.shared.services.batch_context import (
    BatchWorkflowContext,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_progress_service import (
    DebugSnapshotAvailableNotification,
)
from openhcs.pyqt_gui.widgets.shared.services.live_measurement_progress_service import (
    LiveMeasurementAvailableNotification,
)
from openhcs.pyqt_gui.widgets.shared.services.runtime_artifact_progress_service import (
    RuntimeArtifactAvailableNotification,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_workflow_service import (
    DebugPlateRunRequest,
)
from openhcs.core.execution_state import (
    ManagerExecutionState,
)
from openhcs.pyqt_gui.services.plate_manager_row import PlateManagerRow
from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import (
    ZMQExecutionClientBoundary,
)
from openhcs.pyqt_gui.config import ProgressUIConfig

logger = logging.getLogger(__name__)
T = TypeVar("T")


class PlateManagerBatchWorkflow:
    """Single owner of PlateManager compilation and execution workflow."""

    def __init__(
        self,
        host,
        *,
        zmq: ZMQExecutionClientBoundary,
        progress_config: ProgressUIConfig,
    ) -> None:
        self.host = host
        self.port = zmq.config.default_port
        self._execution_zmq = zmq
        self._context = BatchWorkflowContext(
            zmq=self._execution_zmq,
            global_config_provider=lambda: self.host.global_config,
            run_blocking=self._run_blocking,
            connect_progress_client=self._connect_progress_client,
        )
        self.components = BatchWorkflowComponents(
            host=self.host,
            context=self._context,
            port=self.port,
            progress_config=progress_config,
        )
        self._cleaned_up = False

    def start_progress_updates(self) -> None:
        """Start the Qt-affine progress projection lifecycle."""

        self.components.start_progress_updates()

    def update_progress_config(self, config: ProgressUIConfig) -> None:
        """Apply progress timing to the workflow's exact timer owner."""

        self.components.update_progress_config(config)

    def cleanup(self) -> None:
        """Release timers/listeners owned by this service."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        self.components.cleanup()

    def clear_progress_execution(self, execution_id: str) -> None:
        """Clear one ZMQ execution from the shared progress projection."""
        self.components.progress_workflow.clear_execution(execution_id)

    async def compile_plates(self, selected_items: list[PlateManagerRow]) -> None:
        """Compile pipelines for selected plates."""
        self.components.progress_workflow.reset_for_new_batch()
        await self.components.compile_batch.compile_plates(selected_items)

    def add_debug_snapshot_listener(
        self,
        listener: Callable[[DebugSnapshotAvailableNotification], None],
    ) -> None:
        """Subscribe to debug snapshot availability announced through progress."""

        self.components.debug_notifications.add_listener(listener)

    def add_live_measurement_listener(
        self,
        listener: Callable[[LiveMeasurementAvailableNotification], None],
    ) -> None:
        """Subscribe to live measurement previews announced through progress."""

        self.components.live_measurements.add_listener(listener)

    def add_runtime_artifact_listener(
        self,
        listener: Callable[[RuntimeArtifactAvailableNotification], None],
    ) -> None:
        """Subscribe to generic runtime artifact addresses from progress."""

        self.components.runtime_artifacts.add_listener(listener)

    async def run_plates(self, ready_items: list[PlateManagerRow]) -> None:
        """Run selected plates using compile-all then execute-all workflow."""
        loop = asyncio.get_event_loop()
        try:
            plate_paths = [row.scope_id for row in ready_items]
            logger.info("Starting ZMQ execution for %d plates", len(plate_paths))

            self.components.progress_workflow.reset_for_new_batch()
            self.host.reset_live_measurements()
            self.host.emit_clear_logs()

            await self._connect_progress_client()

            self.host.supersede_debug_terminal_summaries_for_standard_run(
                plate_paths
            )
            self.host.plate_execution_ids.clear()
            self.host.plate_terminal_activity_status.begin_batch(plate_paths)

            from objectstate import ObjectStateRegistry

            for row in ready_items:
                plate_path = row.scope_id
                orchestrator = ObjectStateRegistry.get_object(plate_path)
                if orchestrator is not None:
                    orchestrator._state = OrchestratorState.EXECUTING
                    self.host.emit_orchestrator_state(
                        plate_path,
                        OrchestratorState.EXECUTING,
                    )

            self.host.execution_state = ManagerExecutionState.RUNNING
            self.host.emit_status(
                f"Compiling {len(ready_items)} plate(s) before execution..."
            )
            self.host.update_button_states()
            self.host.update_item_list()

            run_specs = [
                self.components.plate_request_builder.build_run_spec(plate_path)
                for plate_path in plate_paths
            ]
            compile_artifacts = (
                await self.components.compile_batch.compile_before_execution(
                    run_specs=run_specs,
                    loop=loop,
                )
            )

            self.host.emit_status(
                f"Compilation complete. Submitting {len(run_specs)} plate(s) for execution..."
            )
            for run_spec in run_specs:
                await self.components.execution_submission.submit_plate(
                    run_spec=run_spec,
                    compile_artifact_id=compile_artifacts[run_spec.plate_path],
                    loop=loop,
                )
        except Exception as error:
            logger.error("Failed to execute plates via ZMQ: %s", error, exc_info=True)
            self.host.emit_error(f"Failed to execute: {error}")
            await self.components.execution_control.handle_execution_failure(loop)

    async def run_debug_plate(
        self,
        *,
        plate_path: str,
        debug_session_id: str,
        snapshot_store_ref: str,
        command_type,
        snapshot_store_backend: str | None = None,
        selected_source_group: str | None = None,
        pause_step_indices: tuple[int, ...] = (),
        start_step_index: int = 0,
        start_after_invocation_key: str | None = None,
        replay_mode: DebugReplayMode = DebugReplayMode.WARM_ARTIFACT,
    ) -> None:
        """Compile one plate and submit a bounded debug execution."""

        loop = asyncio.get_event_loop()
        try:
            await self._connect_progress_client()
            self.components.progress_workflow.reset_for_new_batch()
            self.host.plate_terminal_activity_status.begin_batch([plate_path])
            self.host.execution_state = ManagerExecutionState.RUNNING
            self.host.emit_status(f"Compiling debug run for {plate_path}...")
            self.host.update_button_states()
            self.host.update_item_list()

            debug_request = DebugPlateRunRequest(
                debug_session_id=debug_session_id,
                snapshot_store_ref=snapshot_store_ref,
                snapshot_store_backend=snapshot_store_backend,
                command_type=DebugCommandType(command_type),
                selected_source_group=selected_source_group,
                pause_step_indices=tuple(pause_step_indices),
                start_step_index=start_step_index,
                start_after_invocation_key=start_after_invocation_key,
                replay_mode=replay_mode,
            )
            run_spec = self.components.plate_request_builder.build_run_spec(plate_path)
            compile_artifact_id = (
                await self.components.debug_workflow.compile_artifact_id(
                    run_spec=run_spec,
                    debug_request=debug_request,
                    loop=loop,
                )
            )
            self.host.emit_status(f"Submitting debug run for {plate_path}...")
            await self.components.debug_workflow.submit_debug_plate(
                run_spec=run_spec,
                compile_artifact_id=compile_artifact_id,
                debug_request=debug_request,
                loop=loop,
            )
        except Exception as error:
            logger.error(
                "Failed to execute debug run via ZMQ: %s", error, exc_info=True
            )
            self.host.emit_error(f"Failed to execute debug run: {error}")
            await self.components.execution_control.handle_execution_failure(loop)

    async def send_debug_worker_command(
        self,
        *,
        debug_session_id: str,
        command_type: DebugCommandType,
    ) -> DebugPausedWorkerStatus:
        """Send a control command to an already-running persistent debug worker."""

        loop = asyncio.get_event_loop()
        await self._connect_progress_client()
        return await self.components.debug_workflow.send_worker_command(
            debug_session_id=debug_session_id,
            command_type=command_type,
            loop=loop,
        )

    async def inspect_debug_runtime(
        self,
        *,
        debug_session_id: str,
    ):
        """Read the live runtime inspection view from a paused debug worker."""

        loop = asyncio.get_event_loop()
        await self._connect_progress_client()
        return await self.components.debug_workflow.inspect_runtime(
            debug_session_id=debug_session_id,
            loop=loop,
        )

    async def export_debug_artifact(
        self,
        *,
        debug_session_id: str,
        artifact_ref: DebugArtifactRef,
        export_root: str,
        snapshot_store_ref: str | None = None,
        snapshot_store_backend: str | None = None,
    ) -> DebugArtifactExportResponse:
        """Materialize one debug artifact through the execution server namespace."""

        loop = asyncio.get_event_loop()
        await self._connect_progress_client()
        return await self.components.debug_workflow.export_artifact(
            debug_session_id=debug_session_id,
            artifact_ref=artifact_ref,
            export_root=export_root,
            snapshot_store_ref=snapshot_store_ref,
            snapshot_store_backend=snapshot_store_backend,
            loop=loop,
        )

    async def _connect_progress_client(self):
        """Connect the shared ZMQ client with the standard progress callback."""

        return await self._execution_zmq.connect(
            progress_callback=self.components.progress_workflow.on_progress,
            persistent=True,
            timeout=15,
        )

    @staticmethod
    async def _run_blocking(loop, func: Callable[[], T]) -> T:
        return await loop.run_in_executor(None, func)

    def stop_execution(self, force: bool = False) -> None:
        self.components.execution_control.stop_execution(force=force)

    def disconnect(self) -> None:
        self.components.execution_control.disconnect()

    def disconnect_async(self) -> None:
        self.components.execution_control.disconnect_async()


def is_plate_manager_batch_workflow_export(name: str, value) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_plate_manager_batch_workflow_export(name, value)
)

"""Lazy component registry for batch workflow services."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from openhcs.pyqt_gui.widgets.shared.services.compile_batch_workflow_service import (
    CompileBatchWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.compile_workflow_service import (
    CompileWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_progress_service import (
    DebugProgressNotificationService,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_workflow_service import (
    DebugWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_control_service import (
    ExecutionControlService,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_server_status_presenter import (
    ExecutionServerStatusPresenter,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_submission_service import (
    ExecutionSubmissionService,
)
from openhcs.pyqt_gui.widgets.shared.services.live_measurement_progress_service import (
    LiveMeasurementProgressNotificationService,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_pipeline_request_builder import (
    PlatePipelineRequestBuilder,
)
from openhcs.pyqt_gui.widgets.shared.services.progress_workflow_service import (
    ProgressWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.terminal_result_builder import (
    TerminalExecutionResultBuilder,
)
from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import ZMQClientService
from pyqt_reactive.services.zmq_server_info_parser import ServerInfoParserABC
from zmqruntime.execution import ExecutionStatusPoller

T = TypeVar("T")
RunBlockingCallable = Callable[[object, Callable[[], T]], Any]
ProgressClientConnector = Callable[[], Any]


class BatchWorkflowComponents:
    """Owns lazy construction of services used by the batch workflow facade."""

    def __init__(
        self,
        *,
        host,
        client_service: ZMQClientService,
        port: int,
        server_info_parser: ServerInfoParserABC,
        run_blocking: RunBlockingCallable,
        connect_progress_client: ProgressClientConnector,
    ) -> None:
        self.host = host
        self.client_service = client_service
        self.port = port
        self.server_info_parser = server_info_parser
        self.run_blocking = run_blocking
        self.connect_progress_client = connect_progress_client
        self.server_status_presenter = ExecutionServerStatusPresenter()
        self.execution_status_poller = ExecutionStatusPoller()

        self._compile_workflow: CompileWorkflowService | None = None
        self._compile_batch: CompileBatchWorkflowService | None = None
        self._debug_notifications: DebugProgressNotificationService | None = None
        self._live_measurements: LiveMeasurementProgressNotificationService | None = None
        self._plate_request_builder: PlatePipelineRequestBuilder | None = None
        self._terminal_result_builder: TerminalExecutionResultBuilder | None = None
        self._execution_control: ExecutionControlService | None = None
        self._execution_submission: ExecutionSubmissionService | None = None
        self._debug_workflow: DebugWorkflowService | None = None
        self._progress_workflow: ProgressWorkflowService | None = None

    @property
    def compile_workflow(self) -> CompileWorkflowService:
        if self._compile_workflow is None:
            self._compile_workflow = CompileWorkflowService(
                global_config_provider=lambda: self.host.global_config,
                run_blocking=self.run_blocking,
            )
        return self._compile_workflow

    @property
    def compile_batch(self) -> CompileBatchWorkflowService:
        if self._compile_batch is None:
            self._compile_batch = CompileBatchWorkflowService(
                host=self.host,
                client_service=self.client_service,
                global_config_provider=lambda: self.host.global_config,
                run_blocking=self.run_blocking,
                connect_progress_client=self.connect_progress_client,
                compile_workflow=self.compile_workflow,
                plate_request_builder=self.plate_request_builder,
            )
        return self._compile_batch

    @property
    def debug_notifications(self) -> DebugProgressNotificationService:
        if self._debug_notifications is None:
            self._debug_notifications = DebugProgressNotificationService()
        return self._debug_notifications

    @property
    def live_measurements(self) -> LiveMeasurementProgressNotificationService:
        if self._live_measurements is None:
            self._live_measurements = LiveMeasurementProgressNotificationService()
        return self._live_measurements

    @property
    def plate_request_builder(self) -> PlatePipelineRequestBuilder:
        if self._plate_request_builder is None:
            self._plate_request_builder = PlatePipelineRequestBuilder(self.host)
        return self._plate_request_builder

    @property
    def terminal_result_builder(self) -> TerminalExecutionResultBuilder:
        if self._terminal_result_builder is None:
            self._terminal_result_builder = TerminalExecutionResultBuilder()
        return self._terminal_result_builder

    @property
    def execution_control(self) -> ExecutionControlService:
        if self._execution_control is None:
            self._execution_control = ExecutionControlService.openhcs_default(
                host=self.host,
                client_service=self.client_service,
                port=self.port,
            )
        return self._execution_control

    @property
    def execution_submission(self) -> ExecutionSubmissionService:
        if self._execution_submission is None:
            self._execution_submission = ExecutionSubmissionService(
                host=self.host,
                client_service=self.client_service,
                run_blocking=self.run_blocking,
                completion_poller=self.execution_status_poller,
                terminal_result_builder=self.terminal_result_builder,
                on_completion_update=self.execution_control.check_all_completed,
            )
        return self._execution_submission

    @property
    def debug_workflow(self) -> DebugWorkflowService:
        if self._debug_workflow is None:
            self._debug_workflow = DebugWorkflowService(
                host=self.host,
                client_service=self.client_service,
                run_blocking=self.run_blocking,
                compile_before_execution=self.compile_batch.compile_before_execution,
                execution_submission=self.execution_submission,
            )
        return self._debug_workflow

    @property
    def progress_workflow(self) -> ProgressWorkflowService:
        if self._progress_workflow is None:
            self._progress_workflow = ProgressWorkflowService(
                host=self.host,
                client_service=self.client_service,
                server_info_parser=self.server_info_parser,
                debug_notifications=self.debug_notifications,
                live_measurements=self.live_measurements,
                status_presenter=self.server_status_presenter,
            )
        return self._progress_workflow


__all__ = ("BatchWorkflowComponents",)

"""OpenHCS extensions for zmqruntime server hooks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable

from zmqruntime.messages import (
    ExecutionStatus,
    MessageFields,
    PongResponse,
    QueuedExecutionInfo,
    ResponseType,
)

from openhcs.core.execution_state import ExecutionOutputPlateSummary
from openhcs.core.orchestrator.compiled_plate_execution import (
    CompiledPlateExecutionExtras,
)
from openhcs.serialization.json import to_jsonable

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ZMQPongResponseEnricher:
    """Adds OpenHCS queue and compile-status data to base pong responses."""

    active_executions: dict[str, Any]
    compile_status: Callable[[], tuple[str | None, str | None]]

    def enrich(self, response: PongResponse) -> PongResponse:
        compile_status, compile_message = self.compile_status()

        queued = [
            (execution_id, record)
            for execution_id, record in self.active_executions.items()
            if record.status == ExecutionStatus.QUEUED.value
        ]
        queued_executions = tuple(
            QueuedExecutionInfo(
                execution_id=execution_id,
                plate_id=str(record.plate_id),
                queue_position=index + 1,
            )
            for index, (execution_id, record) in enumerate(queued)
        )
        return replace(
            response,
            compile_status=compile_status,
            compile_message=compile_message,
            queued_executions=queued_executions,
        )


@dataclass(frozen=True, slots=True)
class ZMQResultsSummaryEnricher:
    """Projects OpenHCS output-plate metadata into execution summaries."""

    active_executions: dict[str, Any]

    def attach(
        self,
        *,
        execution_id: str,
        record: Any,
        execution_payload: dict | None = None,
    ) -> None:
        if record.status != ExecutionStatus.COMPLETE.value:
            return

        summary = record.results_summary
        if not isinstance(summary, dict):
            summary = {}
            record.results_summary = summary

        output_plate_summary = record.get_extra(
            ExecutionOutputPlateSummary.EXECUTION_RECORD_KEY
        )
        if output_plate_summary is None:
            output_plate_summary = ExecutionOutputPlateSummary()
        elif not isinstance(output_plate_summary, ExecutionOutputPlateSummary):
            raise TypeError(
                "Execution output-plate metadata must use "
                f"{ExecutionOutputPlateSummary.__name__}."
            )
        observation_export_path = record.get_extra("runtime_observation_export_path")
        compiled_execution_extras = record.get_extra(
            CompiledPlateExecutionExtras.EXECUTION_RECORD_KEY
        )
        if compiled_execution_extras is not None and not isinstance(
            compiled_execution_extras,
            CompiledPlateExecutionExtras,
        ):
            raise TypeError(
                "Compiled execution metadata must use "
                f"{CompiledPlateExecutionExtras.__name__}."
            )
        summary.update(output_plate_summary.results_summary_fields())
        if observation_export_path:
            summary["runtime_observation_export_path"] = str(observation_export_path)
        if (
            compiled_execution_extras is not None
            and compiled_execution_extras.viewer_states_by_port
        ):
            summary[CompiledPlateExecutionExtras.RESULTS_SUMMARY_KEY] = to_jsonable(
                compiled_execution_extras.viewer_states_by_port
            )
        if isinstance(execution_payload, dict):
            execution_payload[MessageFields.RESULTS_SUMMARY] = summary

        logger.info(
            "[%s] Attached results_summary extras: output_plate_root=%s auto_add=%s observation=%s",
            execution_id,
            output_plate_summary.output_plate_root,
            output_plate_summary.auto_add_output_plate_to_plate_manager,
            summary.get("runtime_observation_export_path"),
        )

    def attach_to_status_response(
        self,
        *,
        execution_id: str | None,
        response: dict[str, Any],
    ) -> dict[str, Any]:
        if response.get(MessageFields.STATUS) != ResponseType.OK.value:
            return response
        if not execution_id:
            return response

        record = self.active_executions[execution_id]
        execution_payload = response.get(MessageFields.EXECUTION)
        self.attach(
            execution_id=execution_id,
            record=record,
            execution_payload=(
                execution_payload if isinstance(execution_payload, dict) else None
            ),
        )
        return response


@dataclass(frozen=True, slots=True)
class ZMQWorkerCleanup:
    """Gracefully cancels OpenHCS orchestrators before base worker cleanup."""

    active_executions: dict[str, Any]

    def cancel_execution(self, execution_id: str) -> None:
        """Request cooperative cancellation for one execution orchestrator."""

        record = self.active_executions.get(execution_id)
        if record is None:
            return
        orchestrator = record.get_extra("orchestrator")
        if orchestrator is None:
            return
        logger.info("[%s] Requesting graceful cancellation...", execution_id)
        orchestrator.cancel_execution()

    def cancel_orchestrators(self) -> None:
        for execution_id, record in self.active_executions.items():
            orchestrator = record.get_extra("orchestrator")
            if orchestrator is None:
                continue
            try:
                logger.info("[%s] Requesting graceful cancellation...", execution_id)
                orchestrator.cancel_execution()
            except Exception as error:
                logger.warning(
                    "[%s] Graceful cancellation failed: %s",
                    execution_id,
                    error,
                )

"""OpenHCS extensions for zmqruntime server hooks."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable

from zmqruntime.messages import ExecutionStatus, MessageFields, ResponseType


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ZMQPongResponseEnricher:
    """Adds OpenHCS queue and compile-status data to base pong responses."""

    active_executions: dict[str, Any]
    compile_status: Callable[[], tuple[str | None, str | None]]

    def enrich(self, response: dict[str, Any]) -> dict[str, Any]:
        compile_status, compile_message = self.compile_status()
        if compile_status is not None:
            response[MessageFields.COMPILE_STATUS] = compile_status
        if compile_message is not None:
            response[MessageFields.COMPILE_MESSAGE] = compile_message

        queued = [
            (execution_id, record)
            for execution_id, record in self.active_executions.items()
            if record.status == ExecutionStatus.QUEUED.value
        ]
        response["queued_executions"] = [
            {
                MessageFields.EXECUTION_ID: execution_id,
                MessageFields.PLATE_ID: str(record.plate_id),
                "queue_position": index + 1,
            }
            for index, (execution_id, record) in enumerate(queued)
        ]
        return response


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

        output_plate_root = record.get_extra("output_plate_root")
        auto_add_output_plate = record.get_extra("auto_add_output_plate")
        if output_plate_root:
            summary["output_plate_root"] = str(output_plate_root)
        if auto_add_output_plate is not None:
            summary["auto_add_output_plate_to_plate_manager"] = bool(
                auto_add_output_plate
            )
        if isinstance(execution_payload, dict):
            execution_payload[MessageFields.RESULTS_SUMMARY] = summary

        logger.info(
            "[%s] Attached results_summary extras: output_plate_root=%s auto_add=%s",
            execution_id,
            summary.get("output_plate_root"),
            summary.get("auto_add_output_plate_to_plate_manager"),
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
            execution_payload=execution_payload
            if isinstance(execution_payload, dict)
            else None,
        )
        return response


@dataclass(frozen=True, slots=True)
class ZMQWorkerCleanup:
    """Gracefully cancels OpenHCS orchestrators before base worker cleanup."""

    active_executions: dict[str, Any]

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

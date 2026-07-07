"""Runtime-observation export for ZMQ executions."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.orchestrator.execution_result import ExecutionResult
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionObservation,
)
from openhcs.core.runtime_exports import RuntimeExportObservation
from openhcs.core.runtime_stores import StoredRuntimeValue


ZMQ_RUNTIME_OBSERVATION_EXPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ZMQRuntimeExecutionObservationExport:
    """Pickle-safe runtime observation emitted by a ZMQ server execution."""

    schema_version: int
    records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]]
    exports: RuntimeExportObservation
    output_roots: tuple[Path, ...]
    execution_success_by_axis: Mapping[str, bool]

    @classmethod
    def from_execution(
        cls,
        *,
        compiled_contexts: Mapping[str, ProcessingContext],
        execution_results: Mapping[str, ExecutionResult],
        output_roots: tuple[Path, ...],
    ) -> "ZMQRuntimeExecutionObservationExport":
        observation = RuntimeArtifactExecutionObservation.from_contexts(
            compiled_contexts,
            output_roots[0] if len(output_roots) == 1 else None,
        )
        return cls(
            schema_version=ZMQ_RUNTIME_OBSERVATION_EXPORT_SCHEMA_VERSION,
            records_by_axis=dict(observation.records_by_axis),
            exports=observation.exports,
            output_roots=tuple(Path(root) for root in output_roots),
            execution_success_by_axis={
                str(axis_id): result.is_success()
                for axis_id, result in execution_results.items()
            },
        )

    @classmethod
    def read(cls, path: Path) -> "ZMQRuntimeExecutionObservationExport":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, cls):
            raise TypeError(
                "ZMQ runtime observation export must contain "
                f"{cls.__name__}, got {type(payload).__name__}."
            )
        if payload.schema_version != ZMQ_RUNTIME_OBSERVATION_EXPORT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported ZMQ runtime observation export schema version "
                f"{payload.schema_version!r}."
            )
        return payload

    def write(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def observation(self) -> RuntimeArtifactExecutionObservation:
        return RuntimeArtifactExecutionObservation(
            records_by_axis={
                str(axis_id): tuple(records)
                for axis_id, records in self.records_by_axis.items()
            },
            exports=self.exports,
        )

    @property
    def axis_count(self) -> int:
        return len(self.execution_success_by_axis)

    def execution_failures(self) -> tuple[str, ...]:
        failed = tuple(
            axis_id
            for axis_id, success in self.execution_success_by_axis.items()
            if not success
        )
        if not failed:
            return ()
        return (f"unsuccessful execution axes: {failed!r}",)

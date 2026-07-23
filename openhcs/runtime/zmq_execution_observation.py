"""Runtime-observation export for ZMQ executions."""

from __future__ import annotations

import gzip
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.orchestrator.execution_result import ExecutionResult
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionExpectation,
    RuntimeArtifactExecutionObservation,
    runtime_artifact_execution_failures,
)
from openhcs.core.runtime_exports import RuntimeExportObservation
from openhcs.core.source_matching import SourceImageSetIdentityPolicy
from openhcs.core.runtime_stores import StoredRuntimeValue


ZMQ_RUNTIME_OBSERVATION_EXPORT_SCHEMA_VERSION = 5


@dataclass(frozen=True, slots=True)
class ZMQRuntimeExecutionObservationExport:
    """Pickle-safe runtime observation emitted by a ZMQ server execution."""

    schema_version: int
    expectation: RuntimeArtifactExecutionExpectation
    records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]]
    exports: RuntimeExportObservation
    output_roots: tuple[Path, ...]
    execution_success_by_axis: Mapping[str, bool]
    source_image_set_identity_policy: SourceImageSetIdentityPolicy = (
        SourceImageSetIdentityPolicy()
    )

    @classmethod
    def from_execution(
        cls,
        *,
        compiled_contexts: Mapping[str, ProcessingContext],
        execution_results: Mapping[str, ExecutionResult],
        output_roots: tuple[Path, ...],
    ) -> "ZMQRuntimeExecutionObservationExport":
        observation = RuntimeArtifactExecutionObservation.from_contexts(
            compiled_contexts
        )
        return cls(
            schema_version=ZMQ_RUNTIME_OBSERVATION_EXPORT_SCHEMA_VERSION,
            expectation=RuntimeArtifactExecutionExpectation.from_compiled_contexts(
                compiled_contexts
            ),
            records_by_axis=dict(observation.records_by_axis),
            exports=observation.exports,
            output_roots=tuple(Path(root) for root in output_roots),
            execution_success_by_axis={
                str(axis_id): result.is_success()
                for axis_id, result in execution_results.items()
            },
            source_image_set_identity_policy=(
                observation.source_image_set_identity_policy
            ),
        )

    @classmethod
    def read(cls, path: Path) -> "ZMQRuntimeExecutionObservationExport":
        with gzip.open(Path(path), "rb") as handle:
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
        with gzip.open(target, "wb", compresslevel=1) as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def observation(self) -> RuntimeArtifactExecutionObservation:
        return RuntimeArtifactExecutionObservation(
            records_by_axis={
                str(axis_id): tuple(records)
                for axis_id, records in self.records_by_axis.items()
            },
            exports=self.exports,
            source_image_set_identity_policy=(
                self.source_image_set_identity_policy
            ),
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

    def require_valid_observation(self) -> RuntimeArtifactExecutionObservation:
        """Return the observation after validating execution and artifact outputs."""

        observation = self.observation()
        failures = (
            *self.execution_failures(),
            *runtime_artifact_execution_failures(self.expectation, observation),
        )
        if failures:
            raise RuntimeError(
                "ZMQ runtime execution violated compiled expectations:\n"
                + "\n".join(f"- {failure}" for failure in failures)
            )
        return observation

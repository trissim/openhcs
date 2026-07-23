"""Shared support for CellProfiler measurement execution flows."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from functools import partial
import time
from typing import TYPE_CHECKING

from openhcs.core.aligned_image_payload import (
    ImagePayloadExecutionMode,
    payload_slice_count,
)
from openhcs.core.artifacts import ArtifactOutputPlan, ArtifactSpec
from openhcs.core.callable_contract import CallableContract
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
)
from openhcs.core.runtime_profile import RuntimeProfileTimer
from openhcs.core.runtime_tabular_values import (
    ColumnarRows,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
)
from openhcs.core.runtime_batch_contracts import RuntimeBatchInvocationRequest
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.function_contract_execution import (
    _execute_runtime_batch_invocation,
)
from openhcs.interop.cellprofiler.runtime.invocation import CellProfilerMeasurementImage
from openhcs.interop.cellprofiler.runtime.object_measurement_execution import (
    CellProfilerObjectMeasurementExecutionPolicy,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    measurement_table_rows,
)
from openhcs.core.runtime_output_matching import RuntimeReturnedOutputMatcher
from openhcs.core.steps.function_runtime import RuntimeCallableArgument, RuntimeFunctionOutput
from openhcs.interop.cellprofiler.runtime.profile_fields import (
    dense_label_argument_stage_profile_fields,
    object_label_stage_profile_fields,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileEvent,
    CellProfilerRuntimeProfileLogger,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
    from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
        CellProfilerObjectMeasurementRowPolicy,
        ObjectMeasurementInvocation,
    )

ObjectMeasurementBatchExecutor = Callable[
    [
        Callable[..., RuntimeFunctionOutput],
        tuple[RuntimeBatchInvocationRequest, ...],
        Callable[
            [Callable[..., RuntimeFunctionOutput], RuntimeBatchInvocationRequest],
            RuntimeCallableArgument,
        ],
    ],
    Sequence[RuntimeCallableArgument],
]


def object_measurement_runtime_inputs(
    *,
    object_label_execution: ObjectLabelInputExecutionMode,
    measurement_image: CellProfilerMeasurementImage,
    object_spec: ArtifactSpec,
    label_payload: RuntimeCallableArgument,
    adapter: CellProfilerRuntimeAdapter,
) -> tuple[
    CellProfilerMeasurementImage,
    RuntimeCallableArgument,
    RuntimeCallableArgument,
    ImagePayloadExecutionMode,
    tuple["CellProfilerRuntimeProfileEvent", ...],
    float,
    float,
]:
    """Prepare image, labels, payload, and execution mode for object measurement."""
    profile_enabled = CellProfilerRuntimeProfileLogger.enabled()
    profile_events: list[CellProfilerRuntimeProfileEvent] = []
    label_payload_timer = RuntimeProfileTimer.start()
    if not isinstance(label_payload, ObjectLabelValue):
        raise TypeError(
            "CellProfiler object measurement requires ObjectLabelValue labels, "
            f"got {type(label_payload).__name__}."
        )
    prepared_labels = measurement_image.prepare_object_labels(
        label_payload,
        plane_projector=adapter,
    )
    if profile_enabled:
        profile_events.append(
            object_label_stage_event(
                "raw",
                measurement_image,
                object_spec,
                prepared_labels.source_projected_payload,
            )
        )
        profile_events.append(
            dense_label_stage_event(
                "measurement_image",
                measurement_image,
                object_spec,
                prepared_labels.source_projected_labels,
            )
        )
    label_payload_seconds = label_payload_timer.elapsed()

    label_align_timer = RuntimeProfileTimer.start()
    if profile_enabled:
        profile_events.append(
            dense_label_stage_event(
                "source_projected_labels",
                measurement_image,
                object_spec,
                prepared_labels.source_projected_labels,
            )
        )
    aligned_measurement_image = prepared_labels.aligned_source
    if not isinstance(aligned_measurement_image, CellProfilerMeasurementImage):
        raise TypeError(
            "CellProfiler object measurement preparation must preserve "
            "CellProfilerMeasurementImage, got "
            f"{type(aligned_measurement_image).__name__}."
        )
    measurement_labels = prepared_labels.measurement_labels
    if profile_enabled:
        profile_events.append(
            dense_label_stage_event(
                "final_labels",
                measurement_image,
                object_spec,
                measurement_labels,
            )
        )
    completion_label_payload = prepared_labels.completion_payload
    if profile_enabled:
        profile_events.append(
            object_label_stage_event(
                "completion_payload",
                measurement_image,
                object_spec,
                completion_label_payload,
            )
        )
    execution_policy = CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
        object_label_execution
    )
    semantic_label_payload = execution_policy.semantic_label_payload(
        prepared_labels.source_projected_payload,
        completion_label_payload,
    )
    executable_labels = semantic_label_payload
    if profile_enabled:
        profile_events.append(
            dense_label_stage_event(
                "executable_labels",
                measurement_image,
                object_spec,
                executable_labels,
            )
        )
    label_align_seconds = label_align_timer.elapsed()
    execution_mode = execution_policy.image_execution_mode(
        semantic_label_payload,
        measurement_image.execution_mode,
        runtime_slice_count=payload_slice_count(aligned_measurement_image.payload),
    )
    return (
        aligned_measurement_image,
        executable_labels,
        completion_label_payload,
        execution_mode,
        tuple(profile_events),
        label_payload_seconds,
        label_align_seconds,
    )


def object_measurement_batch_group_key(
    *,
    object_spec: ArtifactSpec,
    labels: RuntimeCallableArgument,
) -> tuple[Hashable, ...] | None:
    """Return a batch key only for labels with explicit object-label semantics."""
    if not isinstance(labels, ObjectLabelValue):
        return None
    return (
        ("object_artifact", object_spec.ref()),
        ("object_labels", labels.object_label_semantic_identity()),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedObjectMeasurementInvocation(RuntimeBatchInvocationRequest):
    """One prepared object-measurement invocation plus output ownership context."""

    measurement_image: CellProfilerMeasurementImage
    object_spec: ArtifactSpec
    invocation: "ObjectMeasurementInvocation"
    completion_label_payload: RuntimeCallableArgument

    def record_output(
        self,
        output_recorder: "ObjectMeasurementOutputRecorder",
        raw_output: RuntimeCallableArgument,
    ) -> None:
        """Record one raw invocation output through the shared recorder."""
        output_recorder.record(
            raw_output,
            measurement_image=self.measurement_image,
            object_spec=self.object_spec,
            completion_label_payload=self.completion_label_payload,
            invocation=self.invocation,
        )


@dataclass(frozen=True, slots=True)
class PreparedObjectMeasurementInvocationBatch:
    """Execute prepared object-measurement invocations in declared batch order."""

    callable_contract: CallableContract
    func: Callable[..., RuntimeFunctionOutput]
    function_name: str
    invocations: tuple[PreparedObjectMeasurementInvocation, ...]
    batch_executor: ObjectMeasurementBatchExecutor | None

    def execute(
        self,
        output_recorder: "ObjectMeasurementOutputRecorder",
    ) -> float:
        """Execute all invocations, record outputs, and return contract seconds."""
        if self.batch_executor is not None:
            return self._execute_batched(output_recorder)
        return self._execute_serial(output_recorder)

    def _execute_serial(
        self,
        output_recorder: "ObjectMeasurementOutputRecorder",
    ) -> float:
        contract_execute_seconds = 0.0
        for prepared_invocation in self.invocations:
            contract_started_at = time.perf_counter()
            raw_output = _execute_runtime_batch_invocation(
                self.callable_contract,
                self.func,
                prepared_invocation,
            )
            contract_execute_seconds += time.perf_counter() - contract_started_at
            prepared_invocation.record_output(output_recorder, raw_output)
        return contract_execute_seconds

    def _execute_batched(
        self,
        output_recorder: "ObjectMeasurementOutputRecorder",
    ) -> float:
        batch_requests = tuple(
            invocation.batch_executor_request() for invocation in self.invocations
        )
        if any(request is None for request in batch_requests):
            return self._execute_serial(output_recorder)
        executable_requests = tuple(
            request for request in batch_requests if request is not None
        )
        contract_started_at = time.perf_counter()
        raw_outputs = tuple(
            self.require_batch_executor()(
                self.func,
                executable_requests,
                partial(
                    _execute_runtime_batch_invocation,
                    self.callable_contract,
                ),
            )
        )
        contract_execute_seconds = time.perf_counter() - contract_started_at
        if len(raw_outputs) != len(self.invocations):
            raise ValueError(
                f"{self.function_name} measurement-image batch executor returned "
                f"{len(raw_outputs)} outputs for {len(self.invocations)} requests."
            )

        ordered_batch_outputs = {
            prepared_invocation.batch_index: (
                raw_output,
                prepared_invocation,
            )
            for raw_output, prepared_invocation in zip(
                raw_outputs,
                self.invocations,
                strict=True,
            )
        }
        for order_index in range(len(ordered_batch_outputs)):
            raw_output, prepared_invocation = ordered_batch_outputs[order_index]
            prepared_invocation.record_output(output_recorder, raw_output)
        return contract_execute_seconds

    def require_batch_executor(self) -> ObjectMeasurementBatchExecutor:
        """Return the declared batch executor for the batched path."""
        if self.batch_executor is None:
            raise RuntimeError(
                "PreparedObjectMeasurementInvocationBatch requires a batch executor "
                "for batched execution."
            )
        return self.batch_executor


def object_label_stage_event(
    stage: str,
    measurement_image: CellProfilerMeasurementImage,
    object_spec: ArtifactSpec,
    value: RuntimeCallableArgument,
) -> "CellProfilerRuntimeProfileEvent":
    return CellProfilerRuntimeProfileEvent(
        "cp_per_object_label_stage",
        0.0,
        object_label_stage_profile_fields(
            stage,
            measurement_image,
            object_spec,
            value,
        ),
    )


def dense_label_stage_event(
    stage: str,
    measurement_image: CellProfilerMeasurementImage,
    object_spec: ArtifactSpec,
    value: RuntimeCallableArgument,
) -> "CellProfilerRuntimeProfileEvent":
    return CellProfilerRuntimeProfileEvent(
        "cp_per_object_label_stage",
        0.0,
        dense_label_argument_stage_profile_fields(
            stage,
            measurement_image,
            object_spec,
            value,
        ),
    )


@dataclass(frozen=True, slots=True)
class CellProfilerRuntimeProfiler:
    """Module/function-scoped profile event writer."""

    module_name: str
    function_name: str

    def record(
        self, event: str, elapsed: float, **fields: RuntimeCallableArgument
    ) -> None:
        CellProfilerRuntimeProfileLogger.log_module_profile(
            event,
            elapsed,
            module=self.module_name,
            function=self.function_name,
            **fields,
        )

    def record_events(
        self,
        events: Sequence["CellProfilerRuntimeProfileEvent"],
    ) -> None:
        for event in events:
            self.record(event.name, event.elapsed, **dict(event.fields))


@dataclass(slots=True)
class ObjectMeasurementOutputTimings:
    """Mutable timings for per-object measurement output handling."""

    split_seconds: float = 0.0
    complete_rows_seconds: float = 0.0
    annotate_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ObjectMeasurementOutputRecorder:
    """Record one per-object CellProfiler measurement output."""

    callable_contract: CallableContract
    measurement_output_plan: ArtifactOutputPlan
    row_policy: "CellProfilerObjectMeasurementRowPolicy"
    module_type: type["CellProfilerModule"]
    func: Callable[..., RuntimeFunctionOutput]
    adapter: CellProfilerRuntimeAdapter
    measurement_images: tuple["CellProfilerMeasurementImage", ...]
    object_inputs: tuple[ArtifactSpec, ...]
    image_measurement_rows: list[ColumnarRows]
    columnar_rows: list[ColumnarRows]
    timings: ObjectMeasurementOutputTimings

    def record(
        self,
        raw_output: RuntimeCallableArgument,
        *,
        measurement_image: "CellProfilerMeasurementImage",
        object_spec: ArtifactSpec,
        completion_label_payload: RuntimeCallableArgument,
        invocation: "ObjectMeasurementInvocation",
    ) -> None:
        split_started_at = time.perf_counter()
        _returned_values, matched_outputs = RuntimeReturnedOutputMatcher(
            callable_contract=self.callable_contract,
            returned_output=raw_output,
        ).resolve_plan_values((self.measurement_output_plan,))
        self.timings.split_seconds += time.perf_counter() - split_started_at
        _output_plan, _output_spec, output_value = matched_outputs[0]
        emitted_measurement_rows = measurement_table_rows(output_value)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_per_object_output_rows",
            0.0,
            artifact_count=len(matched_outputs),
            emitted_type=type(emitted_measurement_rows).__name__,
            emitted_rows=len(emitted_measurement_rows),
        )
        raw_measurement_rows = self.row_policy.project_rows(
            emitted_measurement_rows,
            invocation,
        )
        object_measurement_rows, non_object_measurement_rows = (
            self.row_policy.split_scoped_rows(raw_measurement_rows)
        )
        measurement_rows = self.completed_measurement_rows(
            object_measurement_rows,
            completion_label_payload,
        )
        self.record_non_object_rows(
            non_object_measurement_rows,
            measurement_image,
        )
        self.record_object_rows(
            measurement_rows,
            object_spec,
            measurement_image,
        )

    def completed_measurement_rows(
        self,
        object_measurement_rows: ColumnarRows,
        completion_label_payload: RuntimeCallableArgument,
    ) -> ColumnarRows:
        complete_rows_started_at = time.perf_counter()
        measurement_rows = self.row_policy.complete_rows(
            object_measurement_rows,
            label_payload=completion_label_payload,
        )
        self.timings.complete_rows_seconds += (
            time.perf_counter() - complete_rows_started_at
        )
        return measurement_rows

    def project_owned_rows(
        self,
        rows: ColumnarRows,
        *,
        measurement_image: "CellProfilerMeasurementImage",
        row_object_name: str | None,
    ) -> ColumnarRows:
        ownership = self.row_policy.row_ownership(
            measurement_image=measurement_image,
            measurement_images=self.measurement_images,
            object_name=row_object_name,
            object_inputs=self.object_inputs,
            contains_image_measurement_rows=bool(self.image_measurement_rows),
        )
        owned_rows = ownership.annotate_rows(rows)
        owned_rows = self.module_type.prepare_measurement_record_rows(
            owned_rows,
            source_image_name=measurement_image.source_image_name,
        )
        return owned_rows

    def record_non_object_rows(
        self,
        rows: ColumnarRows,
        measurement_image: "CellProfilerMeasurementImage",
    ) -> None:
        if not rows.row_count():
            return
        annotate_started_at = time.perf_counter()
        projected_rows = self.project_owned_rows(
            rows,
            measurement_image=measurement_image,
            row_object_name=None,
        )
        self.image_measurement_rows.append(projected_rows)
        self.timings.annotate_seconds += time.perf_counter() - annotate_started_at

    def record_object_rows(
        self,
        rows: ColumnarRows,
        object_spec: ArtifactSpec,
        measurement_image: "CellProfilerMeasurementImage",
    ) -> None:
        annotate_started_at = time.perf_counter()
        projected_rows = self.project_owned_rows(
            rows,
            measurement_image=measurement_image,
            row_object_name=object_spec.name,
        )
        self.columnar_rows.append(projected_rows)
        self.timings.annotate_seconds += time.perf_counter() - annotate_started_at

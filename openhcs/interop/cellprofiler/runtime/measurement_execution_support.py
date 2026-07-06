"""Shared support for CellProfiler measurement execution flows."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
import time

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode, payload_slice_count
from openhcs.core.artifacts import ArtifactSpec
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    object_label_measurement_execution_from_callable,
)
from openhcs.core.runtime_profile import RuntimeProfileTimer
from openhcs.core.runtime_values import (
    ColumnarRows,
    ObjectLabelValue,
)
from openhcs.core.runtime_invocation import RuntimeBatchInvocationRequest
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.function_contract_execution import (
    _execute_runtime_batch_invocation,
)
from openhcs.interop.cellprofiler.runtime.invocation import CellProfilerMeasurementImage
from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
    CellProfilerMeasurementRecord,
    MeasurementRowColumnarMaterialization,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_execution import (
    CellProfilerObjectMeasurementExecutionDomainPolicy,
    CellProfilerObjectMeasurementLabelArgumentPolicy,
    CellProfilerObjectMeasurementLabelArgumentRequest,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    _measurement_rows_from_output,
    _split_cellprofiler_output,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValues,
    CellProfilerRuntimeValueSequence,
    MeasurementRowsInput,
)
from openhcs.interop.cellprofiler.runtime.profile_fields import (
    dense_label_argument_stage_profile_fields,
    object_label_stage_profile_fields,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileEvent,
    CellProfilerRuntimeProfileLogger,
)

ObjectMeasurementBatchExecutor = Callable[
    [
        CellProfilerFunction,
        tuple[RuntimeBatchInvocationRequest, ...],
        Callable[[CellProfilerFunction, RuntimeBatchInvocationRequest], CellProfilerRuntimeValue],
    ],
    Sequence[CellProfilerRuntimeValue],
]


def project_object_label_payload_for_measurement_image(
    measurement_image: CellProfilerMeasurementImage,
    payload: CellProfilerRuntimeValue,
    *,
    adapter: CellProfilerRuntimeAdapter | None = None,
) -> CellProfilerRuntimeValue:
    """Return payload labels aligned to the measurement image's local pixels."""
    if isinstance(payload, ObjectLabelValue):
        return measurement_image.prepare_object_labels(
            payload,
            plane_projector=adapter,
        ).source_projected_payload
    return payload


def object_measurement_runtime_inputs(
    *,
    module_name: str,
    func: CellProfilerFunction,
    measurement_image: CellProfilerMeasurementImage,
    object_spec: ArtifactSpec,
    label_payload: CellProfilerRuntimeValue,
    adapter: CellProfilerRuntimeAdapter,
) -> tuple[
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValue,
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
    aligned_image = prepared_labels.aligned_image
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
    object_label_execution = object_label_measurement_execution_from_callable(func)
    semantic_label_payload = (
        prepared_labels.source_projected_payload
        if object_label_execution is ObjectLabelMeasurementExecution.FULL_STACK
        else completion_label_payload
    )
    executable_labels = (
        CellProfilerObjectMeasurementLabelArgumentPolicy.for_enum_member(
            object_label_execution
        ).label_argument(
            CellProfilerObjectMeasurementLabelArgumentRequest(
                dense_labels=measurement_labels,
                label_payload=semantic_label_payload,
                measurement_image_payload=measurement_image.payload,
            )
        )
    )
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
    execution_mode = (
        CellProfilerObjectMeasurementExecutionDomainPolicy.for_module(
            module_name
        ).execution_mode(
            func,
            semantic_label_payload,
            measurement_image.execution_mode,
            runtime_slice_count=payload_slice_count(measurement_image.payload),
        )
    )
    return (
        aligned_image,
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
    labels: CellProfilerRuntimeValue,
) -> tuple[Hashable, ...] | None:
    """Return a batch key only for labels with explicit object-label semantics."""
    if not isinstance(labels, ObjectLabelValue):
        return None
    return (
        ("object_artifact", (object_spec.name, object_spec.artifact_type)),
        ("object_labels", labels.object_label_semantic_identity()),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedObjectMeasurementInvocation(RuntimeBatchInvocationRequest):
    """One prepared object-measurement invocation plus output ownership context."""

    measurement_image: CellProfilerMeasurementImage
    object_spec: ArtifactSpec
    invocation: "ObjectMeasurementInvocation"
    completion_label_payload: CellProfilerRuntimeValue

    def record_output(
        self,
        output_recorder: "ObjectMeasurementOutputRecorder",
        raw_output: CellProfilerRuntimeValue,
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

    func: CellProfilerFunction
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
        contract_started_at = time.perf_counter()
        raw_outputs = tuple(
            self.require_batch_executor()(
                self.func,
                self.invocations,
                _execute_runtime_batch_invocation,
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
    value: CellProfilerRuntimeValue,
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
    value: CellProfilerRuntimeValue,
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

    def record(self, event: str, elapsed: float, **fields: CellProfilerRuntimeValue) -> None:
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


@dataclass(frozen=True, slots=True)
class PerImageMeasurementProfile:
    """Profile event facade for per-image measurement execution."""

    profiler: CellProfilerRuntimeProfiler

    def measurement_images(self, elapsed: float, images: int) -> None:
        self.profiler.record(
            "cp_per_image_measurement_images",
            elapsed,
            images=images,
        )

    def prepare_kwargs(self, elapsed: float) -> None:
        self.profiler.record("cp_per_image_prepare_kwargs", elapsed)

    def contract_execute(self, elapsed: float) -> None:
        self.profiler.record("cp_per_image_contract_execute", elapsed)

    def split_rows(self, elapsed: float, rows: int) -> None:
        self.profiler.record(
            "cp_per_image_split_rows",
            elapsed,
            rows=rows,
        )

    def record_measurements(self, elapsed: float, rows: int) -> None:
        self.profiler.record(
            "cp_per_image_record_measurements",
            elapsed,
            rows=rows,
        )


@dataclass(slots=True)
class ObjectMeasurementOutputTimings:
    """Mutable timings for per-object measurement output handling."""

    split_seconds: float = 0.0
    complete_rows_seconds: float = 0.0
    annotate_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ObjectMeasurementOutputRecorder:
    """Record one per-object CellProfiler measurement output."""

    row_policy: "CellProfilerObjectMeasurementRowPolicy"
    func: CellProfilerFunction
    adapter: CellProfilerRuntimeAdapter
    measurement_images: tuple["CellProfilerMeasurementImage", ...]
    object_inputs: tuple[ArtifactSpec, ...]
    contains_image_measurement_rows: bool
    combined_rows: list[CellProfilerRuntimeValue]
    columnar_rows: list[ColumnarRows]
    timings: ObjectMeasurementOutputTimings

    def record(
        self,
        raw_output: CellProfilerRuntimeValue,
        *,
        measurement_image: "CellProfilerMeasurementImage",
        object_spec: ArtifactSpec,
        completion_label_payload: CellProfilerRuntimeValue,
        invocation: "ObjectMeasurementInvocation",
    ) -> None:
        artifact_values = self.artifact_values(raw_output)
        emitted_measurement_rows = _measurement_rows_from_output(artifact_values)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_per_object_output_rows",
            0.0,
            artifact_count=len(artifact_values),
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

    def artifact_values(
        self,
        raw_output: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValues:
        split_started_at = time.perf_counter()
        _ignored_main_output, artifact_values = _split_cellprofiler_output(raw_output)
        self.timings.split_seconds += time.perf_counter() - split_started_at
        return artifact_values

    def completed_measurement_rows(
        self,
        object_measurement_rows: MeasurementRowsInput,
        completion_label_payload: CellProfilerRuntimeValue,
    ) -> MeasurementRowsInput:
        complete_rows_started_at = time.perf_counter()
        measurement_rows = self.row_policy.complete_rows(
            object_measurement_rows,
            label_payload=completion_label_payload,
            func=self.func,
        )
        self.timings.complete_rows_seconds += (
            time.perf_counter() - complete_rows_started_at
        )
        return measurement_rows

    def project_owned_rows(
        self,
        rows: MeasurementRowsInput,
        *,
        measurement_image: "CellProfilerMeasurementImage",
        row_object_name: str | None,
        record_object_name: str | None,
    ) -> MeasurementRowsInput:
        ownership = self.row_policy.row_ownership(
            measurement_image=measurement_image,
            measurement_images=self.measurement_images,
            object_name=row_object_name,
            object_inputs=self.object_inputs,
            contains_image_measurement_rows=self.contains_image_measurement_rows,
        )
        owned_rows = ownership.annotate_rows(rows)
        if isinstance(rows, ColumnarRows) and not isinstance(owned_rows, ColumnarRows):
            raise TypeError(
                "Columnar measurement ownership annotation must preserve "
                f"ColumnarRows, got {type(owned_rows).__name__}."
            )
        return owned_rows

    def record_non_object_rows(
        self,
        rows: CellProfilerRuntimeValueSequence,
        measurement_image: "CellProfilerMeasurementImage",
    ) -> None:
        if not rows:
            return
        annotate_started_at = time.perf_counter()
        projected_rows = self.project_owned_rows(
            rows,
            measurement_image=measurement_image,
            row_object_name=None,
            record_object_name=None,
        )
        self.combined_rows.extend(projected_rows)
        self.timings.annotate_seconds += time.perf_counter() - annotate_started_at

    def record_object_rows(
        self,
        rows: MeasurementRowsInput,
        object_spec: ArtifactSpec,
        measurement_image: "CellProfilerMeasurementImage",
    ) -> None:
        annotate_started_at = time.perf_counter()
        if not isinstance(rows, ColumnarRows):
            rows, _fields = MeasurementRowColumnarMaterialization.from_rows(rows).table()
        projected_rows = self.project_owned_rows(
            rows,
            measurement_image=measurement_image,
            row_object_name=object_spec.name,
            record_object_name=object_spec.name,
        )
        if isinstance(projected_rows, ColumnarRows):
            self.columnar_rows.append(projected_rows)
            self.timings.annotate_seconds += time.perf_counter() - annotate_started_at
            return
        self.combined_rows.extend(projected_rows)
        self.timings.annotate_seconds += time.perf_counter() - annotate_started_at

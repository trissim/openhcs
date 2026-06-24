"""Shared support for CellProfiler measurement execution flows."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
import time

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode, payload_slice_count
from openhcs.core.artifacts import ArtifactSpec
from openhcs.core.measurement_image_alignment import PreparedMeasurementObjectLabels
from openhcs.core.pipeline.function_contracts import (
    object_label_measurement_execution_from_callable,
)
from openhcs.core.runtime_values import (
    ColumnarRows,
    ObjectLabelValue,
)
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
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

def project_object_label_payload_for_measurement_image(
    measurement_image: CellProfilerMeasurementImage,
    payload: CellProfilerRuntimeValue,
    *,
    adapter: CellProfilerRuntimeAdapter | None = None,
) -> CellProfilerRuntimeValue:
    """Return payload labels aligned to the measurement image's local pixels."""
    if isinstance(payload, ObjectLabelValue):
        return PreparedMeasurementObjectLabels.from_source(
            measurement_image,
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
    current_image: CellProfilerRuntimeValue,
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
    profile_events: list[CellProfilerRuntimeProfileEvent] = []
    label_payload_started_at = time.perf_counter()
    if not isinstance(label_payload, ObjectLabelValue):
        raise TypeError(
            "CellProfiler object measurement requires ObjectLabelValue labels, "
            f"got {type(label_payload).__name__}."
        )
    prepared_labels = PreparedMeasurementObjectLabels.from_source(
        measurement_image,
        label_payload,
        plane_projector=adapter,
        align_image_to_labels=measurement_image.align_to_labels,
    )
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
    label_payload_seconds = time.perf_counter() - label_payload_started_at

    label_align_started_at = time.perf_counter()
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
    profile_events.append(
        dense_label_stage_event(
            "final_labels",
            measurement_image,
            object_spec,
            measurement_labels,
        )
    )
    completion_label_payload = prepared_labels.completion_payload
    profile_events.append(
        object_label_stage_event(
            "completion_payload",
            measurement_image,
            object_spec,
            completion_label_payload,
        )
    )
    executable_labels = (
        CellProfilerObjectMeasurementLabelArgumentPolicy.for_enum_member(
            object_label_measurement_execution_from_callable(func)
        ).label_argument(
            CellProfilerObjectMeasurementLabelArgumentRequest(
                dense_labels=measurement_labels,
                label_payload=completion_label_payload,
                measurement_image_payload=measurement_image.payload,
            )
        )
    )
    profile_events.append(
        dense_label_stage_event(
            "executable_labels",
            measurement_image,
            object_spec,
            executable_labels,
        )
    )
    label_align_seconds = time.perf_counter() - label_align_started_at
    execution_mode = (
        CellProfilerObjectMeasurementExecutionDomainPolicy.for_module(
            module_name
        ).execution_mode(
            func,
            completion_label_payload,
            measurement_image.execution_mode,
            runtime_slice_count=payload_slice_count(current_image),
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
        ("object_artifact", (object_spec.name, object_spec.kind)),
        ("object_labels", labels.object_label_semantic_identity()),
    )


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

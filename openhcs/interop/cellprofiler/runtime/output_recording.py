"""CellProfiler output recording and measurement record builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from functools import lru_cache
import time
from types import MappingProxyType
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec, ArtifactSpecCollection
from openhcs.core.callable_contract import CallableContract
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.runtime_semantics import MeasurementScope, ParentChildRelationshipPayload
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_values import (
    SpatialGrid,
    image_payload_metadata,
)
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerImageRequest,
    CellProfilerInvocationRequest,
)
from openhcs.interop.cellprofiler.runtime.measurement_image_sources import (
    CellProfilerImageMeasurementSource,
    ProducedArtifactImageMeasurementSource,
    UnqualifiedRuntimeImageMeasurementSource,
)
from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
    CellProfilerMeasurementFieldSchema,
    CellProfilerMeasurementMaterializer,
    CellProfilerMeasurementRecord,
    CellProfilerMeasurementSourceContext,
    MeasurementRowColumnarMaterialization,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    AlignMeasurementRows,
    ClassifyObjectsMeasurementRows,
    ObjectLocationMeasurementRows,
    ThresholdMeasurementRows,
    _measurement_object_name,
    measurement_table_rows,
)
from openhcs.interop.cellprofiler.runtime.measurement_source_names import (
    measurement_source_name_for_specs,
)
from openhcs.interop.cellprofiler.runtime.module_names import (
    CELLPROFILER_CROP_MODULE,
    CELLPROFILER_MEASURE_COLOCALIZATION_MODULE,
    CELLPROFILER_MEASURE_OBJECT_NEIGHBORS_MODULE,
    CELLPROFILER_MEASURE_OBJECT_SIZE_SHAPE_MODULE,
    CELLPROFILER_RELATE_OBJECTS_MODULE,
    CELLPROFILER_TRACK_OBJECTS_MODULE,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    CellProfilerObjectMeasurementRowPolicy,
)
from openhcs.interop.cellprofiler.runtime.output_contexts import (
    CellProfilerImageOutputContextStrategy,
    CellProfilerImageOutputSourcePayloadPolicy,
    CellProfilerImageOutputValuePolicy,
    CellProfilerObjectLabelOutputContextStrategy,
)
from openhcs.interop.cellprofiler.runtime.output_record_request import (
    CellProfilerOutputRecordRequest,
)
from openhcs.interop.cellprofiler.runtime.output_value_resolution import (
    CellProfilerResolvedOutputValues,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargs,
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValues,
    CellProfilerRuntimeValueSequence,
)
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    CellProfilerModulePolicyAutoRegisterMeta,
    CellProfilerModulePolicyLeafSpec,
    CellProfilerModulePolicyLookupMixin,
    CellProfilerModulePolicyRegistryKey,
)
from openhcs.interop.cellprofiler.runtime.profile_fields import (
    cellprofiler_profile_payload_fields,
)
from openhcs.interop.cellprofiler.runtime.pure2d_output_aggregation import (
    ImagePayloadPure2DOutputAggregator,
)
from openhcs.interop.cellprofiler.runtime.relationship_measurement_rows import (
    RelationshipMeasurementRows,
)
from openhcs.interop.cellprofiler.runtime.relationship_endpoints import (
    RelationshipEndpointResolver,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)

_MEASURE_OBJECT_SIZE_SHAPE_MODULE = CELLPROFILER_MEASURE_OBJECT_SIZE_SHAPE_MODULE
_MEASURE_COLOCALIZATION_MODULE = CELLPROFILER_MEASURE_COLOCALIZATION_MODULE
_MEASURE_OBJECT_NEIGHBORS_MODULE = CELLPROFILER_MEASURE_OBJECT_NEIGHBORS_MODULE
_CROP_MODULE = CELLPROFILER_CROP_MODULE
_RELATE_OBJECTS_MODULE = CELLPROFILER_RELATE_OBJECTS_MODULE
_TRACK_OBJECTS_MODULE = CELLPROFILER_TRACK_OBJECTS_MODULE

class CellProfilerMeasurementRecordBuilder(
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Nominal module-specific measurement-row enrichment."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)

    @abstractmethod
    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        """Return measurement rows plus the object set they describe."""


class DefaultMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Use the emitted rows and infer object ownership from declared inputs."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        rows = [
            *measurement_table_rows(request.output_value),
            *RelationshipMeasurementRows.for_request(request).rows(),
        ]
        rows_declare_object_name = (
            CellProfilerMeasurementFieldSchema.rows_declare_object_name(rows)
        )
        object_name = _measurement_object_name(
            request.declared_input_specs
        )
        source_image_name = request.source.source_image_name
        if source_image_name is None:
            source_image_name = measurement_source_name_for_specs(
                request.primary_image_input_policy.primary_image_inputs(
                    request.module_name,
                    request.func,
                    request.declared_input_specs,
                )
            )
        if rows_declare_object_name:
            object_name = None
            source_image_name = None
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=object_name,
            source_context=CellProfilerMeasurementSourceContext(
                source_image_name=source_image_name,
                source_image_payload=request.source.payload,
            ),
            fields=request.fields_for_rows(rows),
        )


class SourcePairMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Preserve source-pair measurements with their table-level source identity."""

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        source_rows = measurement_table_rows(request.output_value)
        return CellProfilerMeasurementRecord(
            rows=source_rows,
            object_name=None,
            source_context=CellProfilerMeasurementSourceContext(
                source_image_name=request.source.source_image_name,
                source_image_payload=request.source.payload,
            ),
            fields=request.fields_for_rows(source_rows),
        )


CellProfilerModulePolicyLeafSpec(
    class_name="MeasureColocalizationMeasurementRecordBuilder",
    base_type=SourcePairMeasurementRecordBuilder,
    module_name=_MEASURE_COLOCALIZATION_MODULE,
).declare_in(globals())


class ObjectTopologyMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Object-only topology measurements are not qualified by image source."""

    module_name = _MEASURE_OBJECT_NEIGHBORS_MODULE

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        rows, fields = MeasurementRowColumnarMaterialization.from_rows(
            measurement_table_rows(request.output_value),
        ).table()
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=_measurement_object_name(
                request.declared_input_specs
            ),
            fields=fields,
        )


class ProducedImageMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Base for diagnostics whose semantic owner is the produced image artifact."""

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        rows = measurement_table_rows(request.output_value)
        source_image = self._primary_image_measurement_source(request)
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=None,
            source_context=CellProfilerMeasurementSourceContext(
                source_image_name=source_image.source_image_name(request),
                source_image_payload=source_image.source_image_payload(request),
            ),
            fields=request.fields_for_rows(rows),
        )

    def _primary_image_measurement_source(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerImageMeasurementSource:
        image_specs = tuple(
            spec
            for spec in request.contract.declared_output_collection().of_kind(
                ArtifactKind.IMAGE
            )
            if spec.sidecar_role is None
        )
        if not image_specs:
            return UnqualifiedRuntimeImageMeasurementSource()
        if len(image_specs) == 1:
            return ProducedArtifactImageMeasurementSource(image_specs[0])
        accepted_image_output_types = (
            ImagePayloadPure2DOutputAggregator.accepted_output_types()
        )
        retained_image_names = {
            name
            for name, value in request.output_values.items()
            if accepted_image_output_types
            and isinstance(value, accepted_image_output_types)
        }
        retained_specs = tuple(
            spec for spec in image_specs if spec.name in retained_image_names
        )
        if len(retained_specs) == 1:
            return ProducedArtifactImageMeasurementSource(retained_specs[0])
        raise ValueError(
            "Produced-image measurement ownership requires exactly one primary image "
            f"output spec, got {[spec.name for spec in image_specs]!r}."
        )


CellProfilerModulePolicyLeafSpec(
    class_name="CropMeasurementRecordBuilder",
    base_type=ProducedImageMeasurementRecordBuilder,
    module_name=_CROP_MODULE,
).declare_in(globals())


class ThresholdMeasurementRecordBuilder(ProducedImageMeasurementRecordBuilder):
    """Threshold diagnostics describe the produced binary image artifact."""

    module_name = "Threshold"

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        source_image = self._primary_image_measurement_source(
            request
        ).require_produced_artifact()
        return CellProfilerMeasurementRecord(
            rows=ThresholdMeasurementRows(
                request.output_value,
                object_name=source_image.artifact_spec.name,
            ).rows(),
            object_name=None,
            source_context=CellProfilerMeasurementSourceContext(
                source_image_name=None,
                source_image_payload=source_image.source_image_payload(request),
            ),
        )


class AlignMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Expose Align shifts as image-scoped measurements for each output image."""

    module_name = "Align"

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        output_names = tuple(
            spec.name
            for spec in ArtifactSpecCollection(
                request.declared_outputs
            ).of_kind(ArtifactKind.IMAGE)
        )
        return CellProfilerMeasurementRecord(
            rows=AlignMeasurementRows(request.output_value, output_names=output_names).rows(),
            object_name=None,
        )


class RelateObjectsMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Expose CellProfiler parent-scoped relationship measurements."""

    module_name = _RELATE_OBJECTS_MODULE

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        table_started_at = time.perf_counter()
        table_rows = measurement_table_rows(request.output_value)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "relate_measurement_table_rows",
            time.perf_counter() - table_started_at,
            module=self.module_name,
            rows=len(table_rows),
        )
        relationship_started_at = time.perf_counter()
        relationship_rows = RelationshipMeasurementRows.for_request(request).rows()
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "relate_relationship_rows",
            time.perf_counter() - relationship_started_at,
            module=self.module_name,
            rows=len(relationship_rows),
        )
        rows, fields = MeasurementRowColumnarMaterialization.from_rows(
            [*table_rows, *relationship_rows],
        ).table()
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=None,
            source_context=CellProfilerMeasurementSourceContext(
                source_image_name=request.source.source_image_name,
            ),
            fields=fields,
        )


class IdentifyObjectRelationshipsMeasurementRecordBuilder(
    CellProfilerMeasurementRecordBuilder
):
    """Expose object-creation relationships as generic measurement facts."""

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        return CellProfilerMeasurementRecord(
            rows=[
                *ThresholdMeasurementRows(
                    request.output_value,
                    object_name=request.single_output_object_name(),
                ).rows(),
                *RelationshipMeasurementRows.for_request(request).rows(),
            ],
            object_name=None,
        )


class ClassifyObjectsMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Expose classification bins as image and object measurement facts."""

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        object_name = _measurement_object_name(
            request.declared_input_specs
        )
        rows, fields = MeasurementRowColumnarMaterialization.from_rows(
            ClassifyObjectsMeasurementRows(
                request.output_value,
                object_name=object_name,
            ).rows()
        ).table()
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=None,
            fields=fields,
        )


for _record_builder_spec in (
    CellProfilerModulePolicyLeafSpec(
        class_name="ClassifyObjectsSingleMeasurementRecordBuilder",
        base_type=ClassifyObjectsMeasurementRecordBuilder,
        module_name="ClassifyObjectsSingleMeasurement",
    ),
    CellProfilerModulePolicyLeafSpec(
        class_name="ClassifyObjectsTwoMeasurementsRecordBuilder",
        base_type=ClassifyObjectsMeasurementRecordBuilder,
        module_name="ClassifyObjectsTwoMeasurements",
    ),
):
    _record_builder_spec.declare_in(globals())
del _record_builder_spec


class CalculateMathMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Expose named math results without inherited image-source qualification."""

    module_name = "CalculateMath"

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        rows = measurement_table_rows(request.output_value)
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=_measurement_object_name(
                request.declared_input_specs
            ),
            fields=request.fields_for_rows(rows),
        )


class MeasureObjectSizeShapeMeasurementRecordBuilder(
    CellProfilerMeasurementRecordBuilder
):
    """Expose object shape rows without image-source feature qualification."""

    module_name = _MEASURE_OBJECT_SIZE_SHAPE_MODULE

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        rows = measurement_table_rows(request.output_value)
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=_measurement_object_name(
                request.declared_input_specs
            ),
            fields=request.fields_for_rows(rows),
            source_context=CellProfilerMeasurementSourceContext(
                source_image_name=None,
                source_image_payload=request.source.payload,
            ),
        )


class IdentifyObjectsMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Expose segmentation threshold diagnostics as image-scope measurements."""

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        object_name = request.single_output_object_name()
        return CellProfilerMeasurementRecord(
            rows=ThresholdMeasurementRows(
                request.output_value,
                object_name=object_name,
            ).rows(),
            object_name=None,
            source_context=CellProfilerMeasurementSourceContext(
                source_image_name=None,
                source_image_payload=request.source.payload,
            ),
        )


class IdentifyObjectsInGridMeasurementRecordBuilder(
    CellProfilerMeasurementRecordBuilder
):
    """Expose grid object location facts emitted by CellProfiler object creation."""

    module_name = "IdentifyObjectsInGrid"

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        object_name = request.single_output_object_name()
        rows = [
            *measurement_table_rows(request.output_value),
            *ObjectLocationMeasurementRows(
                request.output_values[object_name],
                object_name=object_name,
            ).rows(),
        ]
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=None,
            fields=request.fields_for_rows(rows),
        )


for _record_builder_spec in (
    CellProfilerModulePolicyLeafSpec(
        class_name="IdentifyPrimaryObjectsMeasurementRecordBuilder",
        base_type=IdentifyObjectsMeasurementRecordBuilder,
        module_name="IdentifyPrimaryObjects",
    ),
    CellProfilerModulePolicyLeafSpec(
        class_name="IdentifySecondaryObjectsMeasurementRecordBuilder",
        base_type=IdentifyObjectRelationshipsMeasurementRecordBuilder,
        module_name="IdentifySecondaryObjects",
    ),
):
    _record_builder_spec.declare_in(globals())
del _record_builder_spec


class IdentifyTertiaryObjectsMeasurementRecordBuilder(
    CellProfilerMeasurementRecordBuilder
):
    """Expose tertiary parent-child relationships as generic measurement facts."""

    module_name = "IdentifyTertiaryObjects"

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        rows, fields = MeasurementRowColumnarMaterialization.from_rows(
            RelationshipMeasurementRows.for_request(request).rows(),
        ).table()
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=None,
            fields=fields,
        )


class TrackObjectsMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Expose TrackObjects long-form image and object measurements."""

    module_name = _TRACK_OBJECTS_MODULE

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
            request.module_name
        )
        rows = row_policy.annotate_record_rows(
            measurement_table_rows(request.output_value),
            object_name=_measurement_object_name(
                request.declared_input_specs
            ),
            source_image_name=request.source.source_image_name or MeasurementScope.IMAGE.value,
        )
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=None,
            fields=CellProfilerMeasurementFieldSchema.from_row_mappings(rows),
        )


@dataclass(frozen=True, slots=True)
class CellProfilerOutputRecordingPlan:
    """Prepared output order and recorders for one module contract."""

    ordered_outputs: tuple[ArtifactSpec, ...]
    recorders: Mapping[ArtifactKind, CellProfilerOutputRecorder]

    @classmethod
    def from_outputs(
        cls,
        outputs: tuple[ArtifactSpec, ...],
    ) -> "CellProfilerOutputRecordingPlan":
        ordered_outputs = _output_recording_order(outputs)
        return cls(
            ordered_outputs=ordered_outputs,
            recorders=MappingProxyType(
                {
                    kind: CellProfilerOutputRecorder.for_kind(kind)
                    for kind in {spec.kind for spec in ordered_outputs}
                }
            ),
        )


class CellProfilerOutputRecorder(
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal output writer selected by artifact kind."""

    __registry_key__ = "kind"
    __skip_if_no_key__ = True
    stable_key_axis: ClassVar[str] = "kind"
    kind: ClassVar[ArtifactKind | None] = None

    @classmethod
    @lru_cache(maxsize=None)
    def for_kind(cls, kind: ArtifactKind) -> "CellProfilerOutputRecorder":
        recorder_type = cls.__registry__.get(kind)
        if recorder_type is None:
            raise TypeError(f"Unsupported CellProfiler output kind {kind.value}.")
        return recorder_type()

    @classmethod
    def recording_dependency_depth(cls) -> int:
        """Return dependency order from the recorder inheritance chain."""
        return cls.mro().index(CellProfilerOutputRecorder)

    @classmethod
    def record_module_outputs(
        cls,
        *,
        contract: ModuleArtifactContract,
        recording_plan: CellProfilerOutputRecordingPlan,
        primary_image_input_policy: "CellProfilerPrimaryImageInputPolicy",
        adapter: CellProfilerRuntimeAdapter,
        func: CellProfilerFunction,
        main_output: CellProfilerRuntimeValue,
        artifact_values: CellProfilerRuntimeValues,
        invocation: CellProfilerInvocationRequest,
        image_request: CellProfilerImageRequest,
        current_image: CellProfilerRuntimeValue,
    ) -> None:
        """Record one module invocation's returned artifacts."""
        if not contract.outputs:
            return

        function_name = CallableContract.from_callable(func).function_name
        values_started_at = time.perf_counter()
        resolved_values = CellProfilerResolvedOutputValues.from_returned_outputs(
            recorded_specs=contract.outputs,
            context_specs=contract.declared_outputs or contract.outputs,
            main_output=main_output,
            artifact_values=artifact_values,
            func=func,
            declared_output_specs=contract.declared_outputs,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_output_values_by_kind",
            time.perf_counter() - values_started_at,
            module=contract.module_name,
            function=function_name,
            outputs=len(contract.outputs),
        )

        output_source = replace(
            image_request,
            payload=invocation.image,
            source_image_name=invocation.source_image_name,
            image_count=invocation.image_count,
            execution_mode=invocation.execution_mode,
        )
        for spec in recording_plan.ordered_outputs:
            record_started_at = time.perf_counter()
            output_value = resolved_values.recorded_value(spec)
            recording_plan.recorders[spec.kind].record(
                CellProfilerOutputRecordRequest(
                    contract=contract,
                    primary_image_input_policy=primary_image_input_policy,
                    adapter=adapter,
                    spec=spec,
                    output_value=output_value,
                    output_values=resolved_values.context_values,
                    source=output_source,
                    func=func,
                    call_kwargs=invocation.kwargs,
                    current_image=current_image,
                )
            )
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_output_record_one",
                time.perf_counter() - record_started_at,
                module=contract.module_name,
                function=function_name,
                artifact=spec.name,
                kind=spec.kind.value,
                **cellprofiler_profile_payload_fields("value", output_value),
            )

    @abstractmethod
    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        """Record one output artifact through the runtime adapter."""


class ImmediateOutputRecorder(CellProfilerOutputRecorder):
    """Recorder family for artifacts that create no later recording dependency."""


class RelationshipDependentOutputRecorder(ImmediateOutputRecorder):
    """Recorder family for artifacts that require object endpoints to exist."""


class MeasurementDependentOutputRecorder(RelationshipDependentOutputRecorder):
    """Recorder family for artifacts that may require prior relationships."""


class ImageOutputRecorder(ImmediateOutputRecorder):
    """Record image outputs."""

    kind = ArtifactKind.IMAGE

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        output_value = CellProfilerImageOutputValuePolicy.for_module(
            request.module_name
        ).output_value(request)
        source_payload = CellProfilerImageOutputSourcePayloadPolicy.for_module(
            request.module_name
        ).source_payload(request)
        value = CellProfilerImageOutputContextStrategy.for_value(
            output_value
        ).runtime_image_value(
            output_value,
            source_payload,
        )
        request.adapter.add_image(
            request.spec.name,
            value,
            source_image_name=request.source.source_image_name,
        )


class ObjectLabelsOutputRecorder(ImmediateOutputRecorder):
    """Record object-label outputs."""

    kind = ArtifactKind.OBJECT_LABELS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        source_payload = request.object_label_output_source_payload()
        value = CellProfilerObjectLabelOutputContextStrategy.for_value(
            request.output_value
        ).runtime_object_label_value(
            request.output_value,
            source_payload,
            request.object_label_output_domain_scope(),
        )
        source_metadata = image_payload_metadata(source_payload)
        request.adapter.add_objects(
            request.spec.name,
            value,
            source_image_name=request.source.source_image_name,
            source_image_names=(
                request.source.source_aliases or source_metadata.source_image_names
            ),
            source_image_payload=source_payload,
        )


class MeasurementsOutputRecorder(MeasurementDependentOutputRecorder):
    """Record measurement outputs with inferred image/object ownership."""

    kind = ArtifactKind.MEASUREMENTS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        function_name = CallableContract.from_callable(request.func).function_name
        build_started_at = time.perf_counter()
        measurement_record = CellProfilerMeasurementRecordBuilder.for_module(
            request.module_name
        ).build(request)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_measurement_record_build",
            time.perf_counter() - build_started_at,
            module=request.module_name,
            function=function_name,
            artifact=request.spec.name,
            rows=len(measurement_record.rows),
        )
        if self._records_runtime_artifact(request):
            materialize_started_at = time.perf_counter()
            CellProfilerMeasurementMaterializer.record(
                measurement_record.materialization_request(
                    adapter=request.adapter,
                    name=request.spec.name,
                    output_values=request.output_values,
                    current_image=request.current_image,
                )
            )
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_measurement_record_materialize",
                time.perf_counter() - materialize_started_at,
                module=request.module_name,
                function=function_name,
                artifact=request.spec.name,
                rows=len(measurement_record.rows),
            )
            return

        row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
            request.module_name
        )
        partitions_started_at = time.perf_counter()
        partitions = row_policy.record_partitions(measurement_record)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_measurement_record_partitions",
            time.perf_counter() - partitions_started_at,
            module=request.module_name,
            function=function_name,
            artifact=request.spec.name,
            rows=len(measurement_record.rows),
            partitions=len(partitions),
        )
        for partition in partitions:
            materialize_started_at = time.perf_counter()
            CellProfilerMeasurementMaterializer.record(
                partition.materialization_request(
                    adapter=request.adapter,
                    name=request.spec.name,
                    output_values=request.output_values,
                    current_image=request.current_image,
                )
            )
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_measurement_record_materialize",
                time.perf_counter() - materialize_started_at,
                module=request.module_name,
                function=function_name,
                artifact=request.spec.name,
                rows=len(partition.rows),
            )

    @staticmethod
    def _records_runtime_artifact(request: CellProfilerOutputRecordRequest) -> bool:
        """Return whether the adapter has one compiled output slot for this table."""
        return (
            isinstance(request.adapter, CellProfilerRuntimeAdapter)
            and request.spec.name in request.adapter.artifact_outputs
        )


class RelationshipsOutputRecorder(RelationshipDependentOutputRecorder):
    """Record parent-child relationship artifacts."""

    kind = ArtifactKind.RELATIONSHIPS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        if not isinstance(request.output_value, ParentChildRelationshipPayload):
            raise TypeError(
                f"{request.module_name} relationship output "
                f"'{request.spec.name}' must be ParentChildRelationshipPayload, "
                f"got {type(request.output_value).__name__}."
            )
        parent_spec, child_spec = RelationshipEndpointResolver(request).endpoint_specs(
            request.spec
        )
        source_metadata = image_payload_metadata(request.source.payload)
        request.adapter.add_relationship(
            request.spec.name,
            parent_object_name=parent_spec.name,
            child_object_name=child_spec.name,
            parent_ids=request.output_value.parent_ids,
            child_ids=request.output_value.child_ids,
            slice_indices=request.output_value.explicit_slice_indices(),
            slice_count=request.output_value.slice_count,
            source_path=source_metadata.source_path,
            source_component_metadata=source_metadata.source_component_metadata,
            source_image_provenance_planes=(
                source_metadata.source_image_provenance_planes
            ),
        )


class SpatialGridOutputRecorder(ImmediateOutputRecorder):
    """Record spatial-grid outputs."""

    kind = ArtifactKind.SPATIAL_GRID

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        request.adapter.add_spatial_grid(
            request.spec.name,
            _coerce_spatial_grid(request.output_value, request.spec.name),
        )


def _output_recording_order(
    output_specs: tuple[ArtifactSpec, ...],
) -> tuple[ArtifactSpec, ...]:
    return tuple(
        sorted(
            output_specs,
            key=lambda spec: type(
                CellProfilerOutputRecorder.for_kind(spec.kind)
            ).recording_dependency_depth(),
        )
    )


def _coerce_spatial_grid(
    value: CellProfilerRuntimeValue,
    name: str,
) -> SpatialGrid | CellProfilerKwargs | RuntimeSliceAlignedValues[CellProfilerRuntimeValue]:
    match value:
        case RuntimeSliceAlignedValues(slices=slices):
            return RuntimeSliceAlignedValues(
                slices=tuple(_coerce_spatial_grid(item, name) for item in slices)
            )
        case SpatialGrid() as grid:
            return grid.with_name(name)
        case Mapping() as mapping:
            return mapping
        case _ if is_dataclass(value):
            return asdict(value)
        case _:
            raise TypeError(
                f"Spatial grid output '{name}' must be SpatialGrid or mapping-backed, "
                f"got {type(value).__name__}."
            )

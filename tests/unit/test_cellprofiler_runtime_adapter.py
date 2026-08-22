import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
import pytest

from openhcs.constants.constants import AllComponents, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.alias_property import AliasProperty
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecRef,
    ArtifactType,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    RelationshipsArtifactType,
    SpatialGridArtifactType,
)
from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
from openhcs.core.callable_contract import (
    CallableContract,
    runtime_image_execution_mode,
)
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
    RuntimeExecutionAxisScope,
)
from openhcs.core.config import DtypeConfig
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
)
from openhcs.core.measurement_feature_queries import (
    MeasurementObjectFeatureVectorBatchQuery,
    measurement_values_for_feature,
)
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    DataclassMeasurementColumnarRows,
    MeasurementSparseColumnarRows,
    measurement_rows,
)
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    composed_image_payload,
    object_label_input_execution_mode,
    runtime_bound_parameters,
    special_inputs,
)
from openhcs.core.runtime_artifact_queries import (
    MeasurementLabelSliceFeatureQuery,
    MeasurementTableAxisProjection,
    MeasurementTableUnion,
)
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.runtime_batch_contracts import SliceIndexRuntimeParameter
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
)
from openhcs.core.runtime_measurements import (
    MeasurementTable,
    RuntimeMeasurementFeatureOwner,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectLabelVariantData,
    object_label_dense_array,
)
from openhcs.processing.backends.cellprofiler.classification import (
    ClassificationBinChoice,
    SingleMeasurementClassificationRule,
)
from openhcs.core.runtime_sparse_labels import SparseIJVLabelRows
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelRepresentation,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneProjection,
)
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_spatial_grid import (
    SpatialGrid,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_tabular_values import (
    ColumnarRows,
)
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    NamedSourceBinding,
    SourceBindingOrigin,
    SourceBindingRuntimeContext,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceProjectionRole,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.source_image_provenance import (
    SourceImageProvenance,
    SourceImageProvenancePlanes,
)
from openhcs.core.source_matching import (
    SourceImageSetIdentity,
    SourceImageSetIdentityCompatibility,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    MeasurementFeatureRecord,
)
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerModuleExecutor,
)
from openhcs.interop.cellprofiler.runtime.object_label_measurements import (
    ObjectFeatureMeasurementContext,
    ObjectLabelMeasurementSliceRequest,
    RelationshipPlaneProjectionResolution,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_tables import (
    ObjectMeasurementTableIndex,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    MeasurementImageOperandVectorResolution,
)
from openhcs.microscopes.imagexpress import ImageXpressFilenameParser
from openhcs.processing.backends.cellprofiler.image_math import ImageMathOperation
from openhcs.processing.backends.cellprofiler.colocalization import (
    ObjectColocalizationMetricArrays,
)
from openhcs.processing.backends.cellprofiler.measurement_math import (
    CalculateMathModule,
    calculate_math,
)
from openhcs.processing.backends.cellprofiler.relationships import (
    RelateObjectsDistanceMethod,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from tests.unit.cellprofiler_runtime_test_support import (
    cellprofiler_runtime_adapter_for_test,
    cellprofiler_runtime_input_edge_for_test,
    object_measurement_tables_for_test,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)

AXIS_ID = "A01"
DNA_IMAGE = "DNA"
NUCLEI = "Nuclei"
CELLS = "Cells"
PARENT_CHILD = "ParentChild"
MEASUREMENTS = "Measurements"
NUCLEI_MEASUREMENTS = "NucleiMeasurements"


class _FixtureMeasurementFeatureOwner(RuntimeMeasurementFeatureOwner):
    """Test-only owner for a table containing several declared feature families."""

    feature_names: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def owns_measurement_feature_name(cls, feature_name: str) -> bool:
        return feature_name in cls.feature_names

    @classmethod
    def owns_primary_measurement_feature_name(cls, feature_name: str) -> bool:
        return cls.owns_measurement_feature_name(feature_name)


def _runtime_request_input_edge(
    *,
    input_index: int = 0,
) -> InvocationArtifactInputEdgePlan:
    return cellprofiler_runtime_input_edge_for_test(
        ArtifactInputPlan(
            name=DNA_IMAGE,
            path="/memory/DNA.pkl",
            artifact_type=ImageArtifactType,
        ),
        input_index=input_index,
        invocation_scope=ComponentGroupScope.ungrouped(),
        producer_selection_scope=ComponentGroupScope.ungrouped(),
        component_scopes=(),
        consumer_variable_components=(),
    )


def test_runtime_adapter_request_rejects_non_occurrence_input_key() -> None:
    edge = _runtime_request_input_edge()
    request = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        axis_scope=runtime_axis_scope(AXIS_ID),
    ).request

    with pytest.raises(
        TypeError,
        match="InvocationArtifactInputProjectionKey keys",
    ):
        replace(request, artifact_inputs={DNA_IMAGE: edge})


def test_runtime_adapter_request_rejects_non_edge_input_value() -> None:
    edge = _runtime_request_input_edge()
    request = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        axis_scope=runtime_axis_scope(AXIS_ID),
    ).request

    with pytest.raises(
        TypeError,
        match="InvocationArtifactInputEdgePlan values",
    ):
        replace(request, artifact_inputs={edge.key: object()})


def test_runtime_adapter_request_rejects_mismatched_input_occurrence_key() -> None:
    edge = _runtime_request_input_edge()
    mismatched_key = InvocationArtifactInputProjectionKey(
        invocation_key=edge.key.invocation_key,
        input_index=edge.key.input_index + 1,
    )
    request = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        axis_scope=runtime_axis_scope(AXIS_ID),
    ).request

    with pytest.raises(ValueError, match="conflicts with compiled edge key"):
        replace(request, artifact_inputs={mismatched_key: edge})


def _selected_output_plan(
    adapter: CellProfilerRuntimeAdapter,
    name: str,
    artifact_type: type[ArtifactType],
) -> ArtifactOutputPlan:
    spec = adapter.request.require_callable_contract().artifact_outputs.require_by_name_and_artifact_type(
        name, artifact_type
    )
    return adapter.request.require_artifact_output_plan(spec.ref())


def _output_objects(adapter: CellProfilerRuntimeAdapter, name: str) -> ObjectLabelSet:
    records = adapter.artifact_output_records(
        _selected_output_plan(adapter, name, ObjectLabelsArtifactType)
    )
    assert len(records) == 1
    value = records[0].value.data
    assert isinstance(value, ObjectLabelSet)
    return value


def _output_measurements(
    adapter: CellProfilerRuntimeAdapter,
    name: str,
) -> MeasurementTable:
    records = adapter.artifact_output_records(
        _selected_output_plan(adapter, name, MeasurementsArtifactType)
    )
    tables = tuple(record.value.data for record in records)
    assert all(isinstance(table, MeasurementTable) for table in tables)
    return MeasurementTableUnion(name, tables).as_table()


def _output_relationship(
    adapter: CellProfilerRuntimeAdapter,
    name: str,
) -> ObjectRelationship | RuntimeSliceAlignedValues[ObjectRelationship]:
    records = adapter.artifact_output_records(
        _selected_output_plan(adapter, name, RelationshipsArtifactType)
    )
    relationships = tuple(record.value.data for record in records)
    assert all(isinstance(value, ObjectRelationship) for value in relationships)
    if len(relationships) == 1:
        return relationships[0]
    return RuntimeSliceAlignedValues(relationships)


def _output_spatial_grid(
    adapter: CellProfilerRuntimeAdapter,
    name: str,
) -> SpatialGrid | RuntimeSliceAlignedValues[SpatialGrid]:
    records = adapter.artifact_output_records(
        _selected_output_plan(adapter, name, SpatialGridArtifactType)
    )
    value = RuntimeValue.compose(tuple(record.value for record in records))
    assert isinstance(value, (SpatialGrid, RuntimeSliceAlignedValues))
    return value


IDENTIFY_PRIMARY_OBJECTS = "IdentifyPrimaryObjects"
IDENTIFY_SECONDARY_OBJECTS = "IdentifySecondaryObjects"
IDENTIFY_TERTIARY_OBJECTS = "IdentifyTertiaryObjects"
MEASURE_OBJECT_INTENSITY = "MeasureObjectIntensity"
MEASURE_OBJECT_NEIGHBORS = "MeasureObjectNeighbors"
MEASURE_OBJECT_SIZE_SHAPE = "MeasureObjectSizeShape"


def _parent_child_relationship(
    *,
    name: str,
    parent_object_name: str,
    child_object_name: str,
    payload: DirectedObjectRelationshipPayload,
    source_provenance: SourceImageProvenance = SourceImageProvenance(),
) -> ObjectRelationship:
    declaration = ObjectRelationshipDeclaration(
        source=ArtifactSpec.input(
            parent_object_name,
            ObjectLabelsArtifactType,
        ).ref(),
        target=ArtifactSpec.input(
            child_object_name,
            ObjectLabelsArtifactType,
        ).ref(),
        producer_module_number=1,
        relationship_type="parent_child",
        source_role="parent",
        target_role="child",
        source_id_field="parent_id",
        target_id_field="child_id",
        source_runtime_slice_offset=0,
        target_runtime_slice_offset=0,
    )
    return ObjectRelationship.from_payload(
        name=name,
        declaration=declaration,
        payload=payload,
        source_provenance=source_provenance,
    )


def runtime_axis_scope(
    axis_id: str = AXIS_ID,
    component: str | None = None,
    value: str | None = None,
) -> RuntimeExecutionAxisScope:
    return RuntimeExecutionAxisScope.from_raw(
        axis_id,
        component=component,
        value=value,
    )


@dataclass(frozen=True)
class StaticFilenameParser:
    metadata: Mapping[str, object]
    identity: tuple[object, ...] = ("static",)

    def semantic_identity(self) -> tuple[object, ...]:
        return self.identity

    def parse_filename(self, filename: str) -> Mapping[str, object]:
        del filename
        return self.metadata


def measurement_values_for_label_slices(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    labels: object,
    *,
    domain: ObjectLabelDomain,
    plane_axis: RuntimePlaneAxis | None,
    plane_projector: RuntimePlaneProjection,
    object_name: str | None = None,
    row_axis: MeasurementRowAxisField = MeasurementRowAxisField.SLICE_INDEX,
    dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
) -> tuple[object, ...]:
    return MeasurementLabelSliceFeatureQuery(
        measurement_tables=measurement_tables,
        feature_name=feature_name,
        object_name=object_name,
        dialect=dialect,
        row_axis=row_axis,
        plane_projector=plane_projector,
    ).values_for_labels(
        ObjectLabelSet(
            name="Labels",
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=domain,
            plane_axis=plane_axis,
        )
    )


def adapter_label_measurement_values(
    adapter: CellProfilerRuntimeAdapter,
    object_name: str,
    feature_name: str,
    labels: object,
    *,
    domain: ObjectLabelDomain,
    plane_axis: RuntimePlaneAxis | None,
    slice_index: int | None = None,
) -> tuple[object, ...]:
    return ObjectLabelMeasurementSliceRequest(
        object_name=object_name,
        feature_name=feature_name,
        group_key=adapter.request.group_key,
        slice_index=slice_index,
        labels=ObjectLabelSet(
            name=object_name,
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=domain,
            plane_axis=plane_axis,
        ),
    ).values(adapter)


def _measurement_rows_for_assertion(measurements):
    return [dict(row) for row in measurements.rows.iter_row_mappings()]


def _wide_measurement_feature_values(
    rows,
    feature_names: tuple[str, ...],
    *,
    identity_fields: tuple[str, ...] = (),
):
    """Return exact values from schema-owned wide measurement columns."""

    return {
        (
            *(row[field_name] for field_name in identity_fields),
            feature_name,
            row[feature_name],
        )
        for row in rows.iter_row_mappings()
        for feature_name in feature_names
        if feature_name in row
    }


def test_source_identity_compatibility_matches_partial_channel_scope():
    record_identity = SourceImageSetIdentity((("channel", "1"),))
    current_identity = SourceImageSetIdentity(
        (
            ("site", "001"),
            ("channel", "1"),
            ("z", "001"),
            ("timepoint", "001"),
        )
    )
    other_current = SourceImageSetIdentity(
        (
            ("site", "001"),
            ("channel", "2"),
            ("z", "001"),
            ("timepoint", "001"),
        )
    )

    assert SourceImageSetIdentityCompatibility(
        record_identity,
        current_identity,
    ).matches()
    assert not SourceImageSetIdentityCompatibility(
        record_identity,
        other_current,
    ).matches()


@dataclass(frozen=True, slots=True)
class SimpleColumnarRows(ColumnarRows):
    data: Mapping[str, tuple[object, ...]]
    fields: tuple[FieldSpec, ...]

    columns: ClassVar[AliasProperty[Mapping[str, tuple[object, ...]]]] = AliasProperty(
        "data"
    )

    def __post_init__(self) -> None:
        self.validate_fields()


@dataclass(frozen=True, slots=True)
class AreaMeasurementRow:
    object_id: int
    Area: float


@dataclass(frozen=True, slots=True)
class SliceAreaMeasurementRow(MeasurementFeatureRecord):
    object_id: Annotated[int, MeasurementRowAxisField.OBJECT_ID]
    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    Area: float


@dataclass(frozen=True, slots=True)
class IntensityMeasurementRow(MeasurementFeatureRecord):
    mean_intensity: float
    object_label: Annotated[int, MeasurementRowAxisField.OBJECT_ID]


@dataclass(frozen=True, slots=True)
class ObjectCountMeasurementRow:
    object_count: int
    label: int


MEASURE_IMAGE_INTENSITY = "MeasureImageIntensity"
RELATE_OBJECTS = "RelateObjects"
CALCULATE_MATH = "CalculateMath"


def declared_processing_contract(contract: ProcessingContract):
    def decorator(func):
        func.__processing_contract__ = contract
        return func

    return decorator


class FileManagerStub:
    def __init__(self):
        self.registry = {}
        self.saved = {}
        self.directories = []
        self.loaded_batches = []
        self.deleted = []

    def save(self, data, path, backend):
        self.saved[(backend, path)] = data

    def exists(self, path, backend):
        return (backend, path) in self.saved

    def delete(self, path, backend):
        self.deleted.append((backend, path))
        self.saved.pop((backend, path))

    def ensure_directory(self, path, backend):
        self.directories.append((backend, path))

    def resolve_address(self, backend_address, backend, *, base_path):
        return backend_address

    physical_source_path = resolve_address

    def load_batch(self, paths, backend, **kwargs):
        self.loaded_batches.append((tuple(paths), backend, dict(kwargs)))
        return [self.saved[(backend, path)] for path in paths]


class ContextStub:
    def __init__(self, filemanager):
        self.filemanager = filemanager
        self.input_dir = "/plate/Images"
        self.global_config = SimpleNamespace(zarr_config=None)
        self.microscope_handler = SimpleNamespace(
            parser=ImageXpressFilenameParser(),
            get_primary_backend=lambda plate_path, filemanager: "memory",
        )


def _plan(name, kind, *, group_component=None):
    return ArtifactOutputPlan(
        name=name,
        path=f"/memory/{name}.pkl",
        artifact_type=kind,
        group_component=group_component,
    )


def _output_binding(
    name,
    kind,
    *,
    plan: ArtifactOutputPlan | None = None,
    group_component=None,
) -> tuple[ArtifactSpec, ArtifactOutputPlan]:
    spec = ArtifactSpec.output(name, kind)
    return _output_binding_for_spec(
        spec,
        plan=plan,
        group_component=group_component,
    )


def _output_binding_for_spec(
    spec: ArtifactSpec,
    *,
    plan: ArtifactOutputPlan | None = None,
    group_component=None,
) -> tuple[ArtifactSpec, ArtifactOutputPlan]:
    output_plan = (
        plan
        if plan is not None
        else _plan(spec.name, spec.artifact_type, group_component=group_component)
    )
    if output_plan.ref() != spec.ref():
        raise ValueError(
            f"Test output declaration {spec.ref()!r} conflicts with "
            f"plan {output_plan.ref()!r}."
        )
    return spec, output_plan


def _compiled_source_binding_plan(
    source_bindings: StepSourceBindingsConfig,
) -> CompiledSourceBindingPlan:
    return CompiledSourceBindingPlan.from_config(
        source_bindings,
    )


def _adapter(
    output_bindings,
    *,
    source_bindings=StepSourceBindingsConfig(
        bindings=(NamedSourceBinding(alias=DNA_IMAGE),)
    ),
    source_binding_context=SourceBindingRuntimeContext.empty(),
    processing_context=None,
    plane_projection=RuntimePlaneProjection.stack(1),
    callable_contract=None,
):
    filemanager = FileManagerStub()
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=output_bindings,
        source_binding_plan=_compiled_source_binding_plan(source_bindings),
        source_binding_context=source_binding_context,
        plane_projection=plane_projection,
        callable_contract=callable_contract,
        microscope_handler=(
            processing_context.microscope_handler
            if processing_context is not None
            else None
        ),
        filemanager=filemanager,
    )
    return adapter, filemanager


def _pipeline_start_contains_binding(alias):
    return NamedSourceBinding(
        alias=alias,
        selector=SourceSelector(
            filters=(
                SourceFilterClause(
                    SourceFilterSubject.FILE,
                    SourceFilterMatchType.CONTAINS,
                    alias,
                ),
            )
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )


def _source_bound_image_adapter(output_bindings, images):
    filemanager = FileManagerStub()
    paths = tuple(f"/src/{alias}.tif" for alias in images)
    for alias, image in images.items():
        filemanager.saved[("memory", f"/src/{alias}.tif")] = image
    context = ContextStub(filemanager)
    return cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=output_bindings,
        plane_projection=RuntimePlaneProjection.stack(),
        source_binding_plan=_compiled_source_binding_plan(
            StepSourceBindingsConfig(
                bindings=tuple(
                    _pipeline_start_contains_binding(alias) for alias in images
                )
            )
        ),
        source_binding_context=SourceBindingRuntimeContext(
            step_input_files=paths,
            step_input_dir="/src",
            pipeline_input_files=paths,
            pipeline_input_backend="memory",
        ),
        microscope_handler=(context).microscope_handler,
        filemanager=filemanager,
    )


def _source_bound_image_stack(images):
    aliases = tuple(images)
    return ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(f"/src/{alias}.tif" for alias in aliases),
        ),
        source_image_names=aliases,
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    ).payload_with(np.stack(tuple(images.values())), None)


def _source_image_payload(image, source_image_name=DNA_IMAGE):
    return ImagePayloadMetadata(
        source_image_names=(source_image_name,),
    ).payload_with(image, None)


@pytest.fixture
def declaration_owned_cellprofiler_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Install a local test callable at its declaration-owned module boundary."""

    def install(func: Callable[..., object]) -> Callable[..., object]:
        module_type = CellProfilerModule.for_backend_function_name(func.__name__)
        assert module_type is not None
        implementation_module = importlib.import_module(module_type.__module__)
        monkeypatch.setattr(func, "__module__", module_type.__module__)
        monkeypatch.setitem(vars(implementation_module), func.__name__, func)
        assert module_type.require_callable(func.__name__) is func
        return func

    return install


def _executor(
    func,
    adapter: CellProfilerRuntimeAdapter,
    outputs,
    *,
    main_flow_inputs=(ArtifactSpec.input(DNA_IMAGE, ImageArtifactType),),
    source_artifact_inputs=(),
    runtime_artifact_inputs=(),
):
    module_type = CellProfilerModule.for_backend_function_name(func.__name__)
    assert module_type is not None
    return _executor_for_contract(
        func,
        adapter,
        _compiled_callable_contract(
            func,
            artifact_inputs=(
                *main_flow_inputs,
                *source_artifact_inputs,
                *runtime_artifact_inputs,
            ),
            artifact_outputs=outputs,
        ),
        main_flow_input_refs=frozenset(spec.ref() for spec in main_flow_inputs),
    )


def _compiled_callable_contract(
    func,
    *,
    artifact_inputs: tuple[ArtifactSpec, ...] = (),
    artifact_outputs: tuple[ArtifactSpec, ...] = (),
) -> CallableContract:
    raw_contract = CallableContract.from_callable(func)
    return replace(
        raw_contract,
        metadata=replace(
            raw_contract.metadata,
            artifact_inputs=artifact_inputs,
            artifact_outputs=artifact_outputs,
        ),
    )


def _executor_for_contract(
    func,
    adapter: CellProfilerRuntimeAdapter,
    contract: CallableContract,
    *,
    main_flow_input_refs: frozenset[ArtifactSpecRef] = frozenset(),
) -> CellProfilerModuleExecutor:
    contract = replace(
        contract,
        metadata=replace(
            contract.metadata,
            runtime_adapter=CellProfilerRuntimeAdapter.runtime_adapter_spec(),
        ),
    )
    available_plans = adapter.request.artifact_outputs
    invocation_key = FunctionInvocationKey(func.__name__, DEFAULT_GROUP_KEY, 0)
    input_edges = []
    for input_index, spec in enumerate(contract.artifact_inputs):
        source_output_ref = spec.ref().for_plan_type(ArtifactOutputPlan)
        if (
            spec.ref() in main_flow_input_refs
            or source_output_ref not in available_plans
        ):
            input_edges.append(
                InvocationArtifactInputEdgePlan(
                    key=InvocationArtifactInputProjectionKey(
                        invocation_key=invocation_key,
                        input_index=input_index,
                    ),
                    spec=spec,
                    storage_plan=None,
                    projection=None,
                    consumes_main_flow=spec.ref() in main_flow_input_refs,
                )
            )
            continue
        output_plan = available_plans[source_output_ref]
        input_edges.append(
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=spec.name,
                    path=output_plan.path,
                    artifact_type=spec.artifact_type,
                ),
                spec=spec,
                input_index=input_index,
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            )
        )
    artifact_inputs = {edge.key: edge for edge in input_edges}
    artifact_outputs = {
        spec.ref(): replace(
            available_plans[spec.ref()],
            relations=spec.relations,
        )
        for spec in contract.artifact_outputs
        if spec.ref() in available_plans
    }
    adapter.request = replace(
        adapter.request,
        callable_contract=contract,
        artifact_inputs=artifact_inputs,
        artifact_outputs=artifact_outputs,
    )
    return CellProfilerModuleExecutor(
        raw_func=func,
        callable_contract=contract,
    )


def _calculate_math_contract(
    *,
    output_name: str,
    operand1_feature: str,
    operand2_feature: str,
    operand1_object_name: str | None = None,
    operand2_object_name: str | None = None,
) -> CallableContract:
    from openhcs.core.artifacts import (
        ArtifactSpecCollection,
        GroupLineageSourceRelation,
    )
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
    from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
    from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting

    def setting(setting_name, value: object) -> ModuleSetting:
        return ModuleSetting(setting_name.names[0], str(value))

    module = ModuleBlock(
        name=CalculateMathModule.module_name,
        module_num=1,
        setting_records=[
            setting(CalculateMathModule.output_measurement_setting, output_name),
            setting(CalculateMathModule.operation_setting, "Divide"),
            setting(
                CalculateMathModule.numerator_objects_setting,
                operand1_object_name or "None",
            ),
            setting(
                CalculateMathModule.numerator_measurement_setting,
                operand1_feature,
            ),
            setting(
                CalculateMathModule.denominator_objects_setting,
                operand2_object_name or "None",
            ),
            setting(
                CalculateMathModule.denominator_measurement_setting,
                operand2_feature,
            ),
        ],
    )
    object_names = tuple(
        dict.fromkeys(
            name
            for name in (operand1_object_name, operand2_object_name)
            if name is not None
        )
    )
    object_outputs = tuple(
        ArtifactSpec.output(name, ObjectLabelsArtifactType) for name in object_names
    )
    measurement_relations = tuple(
        GroupLineageSourceRelation(
            source=ArtifactSpec.input(
                name,
                ObjectLabelsArtifactType,
            ).ref()
        )
        for name in object_names
    )
    feature_owner = type(
        "FixtureMeasurementFeatureOwner",
        (_FixtureMeasurementFeatureOwner,),
        {
            "feature_names": frozenset((operand1_feature, operand2_feature)),
        },
    )
    prior_measurements = ArtifactSpec.output(
        "PriorMeasurements",
        MeasurementsArtifactType,
        measurement_feature_owner=feature_owner,
        relations=measurement_relations,
    )
    available_artifacts = ArtifactSpecCollection(
        (
            prior_measurements,
            *object_outputs,
        )
    )
    prior_measurement_producers = artifact_producers_for_outputs(
        (prior_measurements,),
        groups=(None,),
        invocation_keys=(
            FunctionInvocationKey(
                "fixture_measurements",
                DEFAULT_GROUP_KEY,
                0,
            ),
        ),
    )
    object_producers = artifact_producers_for_outputs(
        object_outputs,
        groups=(None,),
        invocation_keys=(
            FunctionInvocationKey(
                "fixture_object_producer",
                DEFAULT_GROUP_KEY,
                0,
            ),
        ),
    )
    return CalculateMathModule.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            str(CalculateMathModule.function_name),
            DEFAULT_GROUP_KEY,
            0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_name=CalculateMathModule.module_name,
            step_index=0,
            available_artifact_producers=(
                *object_producers,
                *prior_measurement_producers,
            ),
            available_artifacts=available_artifacts,
            main_flow_artifacts=ArtifactSpecCollection(()),
        ),
    )


def _measurement_output_name(contract: CallableContract) -> str:
    names = contract.artifact_outputs.names_of_artifact_type(MeasurementsArtifactType)
    assert len(names) == 1
    return names[0]


def test_cellprofiler_adapter_adds_and_reads_objects_through_runtime_store():
    adapter, filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
        )
    )
    labels = np.zeros((2, 2), dtype=np.int32)

    record = adapter.add_objects(
        NUCLEI,
        ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
        source_image_name=DNA_IMAGE,
        dimensions=("y", "x"),
    )
    objects = _output_objects(adapter, NUCLEI)

    assert isinstance(record.value.data, ObjectLabelSet)
    assert record.value.data.name == NUCLEI
    assert objects.labels is labels
    assert objects.source_image_name == DNA_IMAGE
    assert objects.dimensions == ("y", "x")
    saved_payload = filemanager.saved[("memory", "/memory/Nuclei.pkl")]
    assert isinstance(saved_payload, ObjectLabelSet)
    assert saved_payload.labels is labels


def test_cellprofiler_adapter_contextualizes_source_aligned_object_label_stack():
    adapter, filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
        )
    )
    source_image = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/src/A01_s001_w1_z001_t001.tif",
                "/src/A01_s002_w1_z001_t001.tif",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "1"},
                {"well": "A01", "site": "2", "channel": "1"},
            ),
        )
    ).payload_with(np.zeros((2, 5, 6), dtype=np.float32), None)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((2, 5, 6), dtype=np.int32))
    )

    record = adapter.add_objects(
        NUCLEI,
        labels,
        source_image_name=DNA_IMAGE,
        source_image_payload=source_image,
    )

    assert isinstance(record.value.data, ObjectLabelSet)
    assert record.value.data.source_image_provenance_planes.paths == (
        "/src/A01_s001_w1_z001_t001.tif",
        "/src/A01_s002_w1_z001_t001.tif",
    )
    assert tuple(
        dict(metadata)
        for metadata in record.value.data.source_image_provenance_planes.component_metadata
        if metadata is not None
    ) == (
        {"well": "A01", "site": "1", "channel": "1"},
        {"well": "A01", "site": "2", "channel": "1"},
    )
    saved_payload = filemanager.saved[("memory", "/memory/Nuclei.pkl")]
    assert isinstance(saved_payload, ObjectLabelSet)
    assert (
        saved_payload.source_image_provenance_planes.paths
        == record.value.data.source_image_provenance_planes.paths
    )


def test_cellprofiler_adapter_contextualizes_single_source_aligned_label_plane():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
        )
    )
    source_path = "/src/A01_s001_w1_z001_t001.tif"
    source_image = ImagePayloadMetadata(
        source_path=source_path,
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
            "z_index": "1",
        },
    ).payload_with(np.zeros((5, 6), dtype=np.float32), None)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((1, 5, 6), dtype=np.int32)),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,),),
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )

    record = adapter.add_objects(
        NUCLEI,
        labels,
        source_image_name=DNA_IMAGE,
        source_image_payload=source_image,
    )

    object_labels = record.value.data
    assert isinstance(object_labels, ObjectLabelSet)
    object_labels.validate_source_alignment(NUCLEI)
    assert object_labels.source_image_provenance_planes.paths == (source_path,)


def test_cellprofiler_adapter_preserves_sparse_ijv_object_value_representation():
    adapter, filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
        )
    )
    sparse_rows = SparseIJVLabelRows(
        np.array(
            [
                [0, 1, 2],
                [1, 0, 3],
            ],
            dtype=np.int32,
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=sparse_rows),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )

    record = adapter.add_objects(NUCLEI, labels)

    assert isinstance(record.value.data, ObjectLabelSet)
    assert record.value.data.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert record.value.data.labels is sparse_rows
    objects = _output_objects(adapter, NUCLEI)
    assert objects.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert objects.labels is sparse_rows
    saved_payload = filemanager.saved[("memory", "/memory/Nuclei.pkl")]
    assert isinstance(saved_payload, ObjectLabelSet)
    assert saved_payload.labels is sparse_rows


def test_cellprofiler_adapter_rejects_unaddressable_source_aligned_label_stack():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
        )
    )
    source_image = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(None, None)
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    ).payload_with(np.zeros((2, 5, 6), dtype=np.float32), None)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((2, 5, 6), dtype=np.int32)),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (1,)),
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )

    with pytest.raises(ValueError, match="neither source_path"):
        adapter.add_objects(
            NUCLEI,
            labels,
            source_image_name=DNA_IMAGE,
            source_image_payload=source_image,
        )


def test_cellprofiler_adapter_does_not_cache_current_image_object_selection():
    store = RuntimeValueStore()
    outputs = {
        NUCLEI: _plan(
            NUCLEI, ObjectLabelsArtifactType, group_component=AllComponents.SITE
        )
    }
    output_bindings = (
        _output_binding(
            NUCLEI,
            ObjectLabelsArtifactType,
            plan=outputs[NUCLEI],
        ),
    )
    source_paths = (
        "/src/A01_s001_w1_z001_t001.tif",
        "/src/A01_s002_w1_z001_t001.tif",
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=source_paths,
        step_input_dir="/src",
        pipeline_input_files=source_paths,
    )
    filemanager = FileManagerStub()
    processing_context = ContextStub(filemanager)

    for group_key, source_path, label_value in (
        ("1", source_paths[0], 1),
        ("2", source_paths[1], 2),
    ):
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            artifact_output_bindings=output_bindings,
            source_binding_context=source_binding_context,
            group_key=group_key,
            microscope_handler=(
                processing_context.microscope_handler
                if processing_context is not None
                else None
            ),
            filemanager=filemanager,
        )
        producer.add_objects(
            NUCLEI,
            ObjectLabelSet(
                name=NUCLEI,
                variant_data=ObjectLabelVariantData(
                    labels=np.full((2, 2), label_value, dtype=np.int32)
                ),
                source_path=source_path,
                domain=ObjectLabelDomain(declared_object_ids=(label_value,)),
            ),
        )

    object_spec = ArtifactSpec.input(NUCLEI, ObjectLabelsArtifactType)
    contract = _compiled_callable_contract(
        calculate_math,
        artifact_inputs=(object_spec,),
    )
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        callable_contract=contract,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs={
            edge.key: edge
            for edge in (
                cellprofiler_runtime_input_edge_for_test(
                    ArtifactInputPlan(
                        name=NUCLEI,
                        path=outputs[NUCLEI].path,
                        artifact_type=ObjectLabelsArtifactType,
                        group_component=AllComponents.SITE,
                    ),
                    invocation_scope=ComponentGroupScope.ungrouped(),
                    producer_selection_scope=ComponentGroupScope.dynamic(
                        AllComponents.SITE
                    ),
                    component_scopes=(ComponentGroupScope.dynamic(AllComponents.SITE),),
                    consumer_variable_components=(AllComponents.SITE,),
                ),
            )
        },
        variable_components=(VariableComponents.SITE,),
        source_binding_context=source_binding_context,
        microscope_handler=(
            processing_context.microscope_handler
            if processing_context is not None
            else None
        ),
        filemanager=filemanager,
    )
    tuple(
        ImagePayloadMetadata(source_path=source_path).payload_with(
            np.zeros((2, 2), dtype=np.float32), None
        )
        for source_path in source_paths
    )

    first = consumer.get_objects(
        NUCLEI,
    )
    second = consumer.get_objects(
        NUCLEI,
    )

    expected = np.stack(
        (
            np.full((2, 2), 1, dtype=np.int32),
            np.full((2, 2), 2, dtype=np.int32),
        )
    )
    np.testing.assert_array_equal(first.labels, expected)
    np.testing.assert_array_equal(second.labels, expected)


def test_cellprofiler_adapter_does_not_select_relationship_from_current_source_plane():
    store = RuntimeValueStore()
    relationship_name = "Nuclei_Cells_relationships"
    group_paths = {
        "1": "/memory/relationship_site_1.pkl",
        "2": "/memory/relationship_site_2.pkl",
    }
    output_plan = ArtifactOutputPlan(
        name=relationship_name,
        path="/memory/relationships.pkl",
        artifact_type=RelationshipsArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.SITE,
        paths_by_group=group_paths,
    )
    filemanager = FileManagerStub()
    for group_key in group_paths:
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            artifact_output_bindings=(
                _output_binding(
                    relationship_name,
                    RelationshipsArtifactType,
                    plan=output_plan,
                ),
            ),
            group_key=group_key,
            filemanager=filemanager,
        )
        producer.add_relationship(
            _parent_child_relationship(
                name=relationship_name,
                parent_object_name=NUCLEI,
                child_object_name=CELLS,
                payload=DirectedObjectRelationshipPayload(
                    source_ids=(int(group_key),),
                    target_ids=(int(group_key),),
                ),
                source_provenance=SourceImageProvenance(
                    source_path=f"/src/A01_s00{group_key}_w1.tif",
                    source_component_metadata={"well": AXIS_ID, "site": group_key},
                ),
            )
        )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=relationship_name,
                    path=output_plan.path,
                    artifact_type=RelationshipsArtifactType,
                    group_keys=output_plan.group_keys,
                    group_component=output_plan.group_component,
                    paths_by_group=output_plan.paths_by_group,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.from_raw(
                    ("1", "2"), component=AllComponents.SITE
                ),
                component_scopes=(
                    ComponentGroupScope.from_raw(
                        ("1", "2"), component=AllComponents.SITE
                    ),
                ),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        variable_components=(VariableComponents.SITE,),
        filemanager=filemanager,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )
    ImagePayloadMetadata(
        source_path="/src/A01_s002_w1.tif",
        source_component_metadata={"well": AXIS_ID, "site": "2"},
    ).payload_with(np.zeros((2, 2), dtype=np.float32), None)

    relationship = consumer.get_relationship(
        relationship_name,
    )

    assert isinstance(relationship, RuntimeSliceAlignedValues)
    assert tuple(
        tuple(relationship.value_for_slice(index).payload.source_ids)
        for index in range(relationship.slice_count)
    ) == ((1,), (2,))
    assert tuple(
        tuple(relationship.value_for_slice(index).payload.target_ids)
        for index in range(relationship.slice_count)
    ) == ((1,), (2,))


def test_cellprofiler_adapter_aligns_grouped_relationships_to_runtime_slices():
    store = RuntimeValueStore()
    relationship_name = "Nuclei_Cells_relationships"
    group_paths = {
        "1": "/memory/relationship_site_1.pkl",
        "2": "/memory/relationship_site_2.pkl",
    }
    output_plan = ArtifactOutputPlan(
        name=relationship_name,
        path="/memory/relationships.pkl",
        artifact_type=RelationshipsArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.SITE,
        paths_by_group=group_paths,
    )
    filemanager = FileManagerStub()
    for group_key in reversed(tuple(group_paths)):
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            artifact_output_bindings=(
                _output_binding(
                    relationship_name,
                    RelationshipsArtifactType,
                    plan=output_plan,
                ),
            ),
            group_key=group_key,
            filemanager=filemanager,
        )
        producer.add_relationship(
            _parent_child_relationship(
                name=relationship_name,
                parent_object_name=NUCLEI,
                child_object_name=CELLS,
                payload=DirectedObjectRelationshipPayload(
                    source_ids=(int(group_key),),
                    target_ids=(int(group_key),),
                ),
            )
        )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=relationship_name,
                    path=output_plan.path,
                    artifact_type=RelationshipsArtifactType,
                    group_keys=output_plan.group_keys,
                    group_component=output_plan.group_component,
                    paths_by_group=output_plan.paths_by_group,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.from_raw(
                    ("1", "2"), component=AllComponents.SITE
                ),
                component_scopes=(
                    ComponentGroupScope.from_raw(
                        ("1", "2"), component=AllComponents.SITE
                    ),
                ),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    relationships = consumer.get_relationship(relationship_name)

    assert isinstance(relationships, RuntimeSliceAlignedValues)
    assert tuple(
        tuple(relationships.value_for_slice(index).payload.source_ids)
        for index in range(relationships.slice_count)
    ) == ((1,), (2,))


def test_cellprofiler_adapter_does_not_source_scope_default_image_records():
    store = RuntimeValueStore()
    outputs = {
        DNA_IMAGE: _plan(
            DNA_IMAGE, ImageArtifactType, group_component=AllComponents.SITE
        )
    }
    output_bindings = (
        _output_binding(DNA_IMAGE, ImageArtifactType, plan=outputs[DNA_IMAGE]),
    )
    filemanager = FileManagerStub()

    for group_key, source_path, value in (
        ("1", "/src/A01_s001_w1.tif", 1.0),
        ("2", "/src/A01_s002_w1.tif", 2.0),
    ):
        source_metadata = {"site": group_key}
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            artifact_output_bindings=output_bindings,
            group_key=group_key,
            filemanager=filemanager,
        )
        producer.add_image(
            DNA_IMAGE,
            ImagePayloadMetadata(
                source_path=source_path,
                source_component_metadata=source_metadata,
            ).payload_with(np.full((2, 2), value, dtype=np.float32), None),
        )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=DNA_IMAGE,
                    path=outputs[DNA_IMAGE].path,
                    artifact_type=ImageArtifactType,
                    group_component=AllComponents.SITE,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.dynamic(
                    AllComponents.SITE
                ),
                component_scopes=(ComponentGroupScope.dynamic(AllComponents.SITE),),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        variable_components=(VariableComponents.SITE,),
        source_binding_context=SourceBindingRuntimeContext(
            step_input_files=("/src/A01_s003_w1.tif",),
        ),
        filemanager=filemanager,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )
    ImagePayloadMetadata(
        source_path="/src/A01_s003_w1.tif",
        source_component_metadata={"site": "3"},
    ).payload_with(np.zeros((2, 2), dtype=np.float32), None)

    image = consumer.get_image(
        DNA_IMAGE,
    )

    assert image_payload_data(image).shape == (2, 2, 2)
    np.testing.assert_array_equal(image_payload_data(image)[0], np.full((2, 2), 1.0))
    np.testing.assert_array_equal(image_payload_data(image)[1], np.full((2, 2), 2.0))


def test_cellprofiler_adapter_stacks_declared_default_image_input_runtime_groups():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    context = ContextStub(filemanager)
    image_name = "Hoechst"
    image_path = "/memory/Hoechst.pkl"
    source_paths = (
        "/plate/Images/A01_s001_w1_z001_t001.tif",
        "/plate/Images/A01_s002_w1_z001_t001.tif",
    )
    output_plan = ArtifactOutputPlan(
        name=image_name,
        path=image_path,
        artifact_type=ImageArtifactType,
        group_keys=(None,),
        group_component=AllComponents.SITE,
    )

    for group_key, source_path, value in (
        ("1", source_paths[0], 1.0),
        ("2", source_paths[1], 2.0),
    ):
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            group_key=group_key,
            artifact_output_bindings=(
                _output_binding(image_name, ImageArtifactType, plan=output_plan),
            ),
            filemanager=filemanager,
        )
        producer.add_image(
            image_name,
            ImagePayloadMetadata(
                source_path=source_path,
                source_component_metadata={"site": group_key},
            ).payload_with(np.full((2, 2), value, dtype=np.float32), None),
        )

    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=source_paths,
        current_step_input_files=source_paths,
        pipeline_input_files=source_paths,
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=image_name,
                    path=image_path,
                    artifact_type=ImageArtifactType,
                    group_keys=(None,),
                    group_component=AllComponents.SITE,
                ),
                invocation_scope=ComponentGroupScope.from_raw(
                    (AXIS_ID,), component=AllComponents.WELL
                ),
                producer_selection_scope=ComponentGroupScope.dynamic(
                    AllComponents.SITE
                ),
                component_scopes=(
                    ComponentGroupScope.from_raw(
                        (AXIS_ID,), component=AllComponents.WELL
                    ),
                    ComponentGroupScope.dynamic(AllComponents.SITE),
                ),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID, "well", AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        source_binding_context=source_binding_context,
        microscope_handler=(context).microscope_handler,
        filemanager=filemanager,
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )
    ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=source_paths,
            component_metadata=(
                {"site": "1"},
                {"site": "2"},
            ),
        )
    ).payload_with(np.zeros((2, 2, 2), dtype=np.float32), None)

    image = consumer.get_image(
        image_name,
    )

    assert image_payload_data(image).shape == (2, 2, 2)
    np.testing.assert_array_equal(image_payload_data(image)[0], np.full((2, 2), 1.0))
    np.testing.assert_array_equal(image_payload_data(image)[1], np.full((2, 2), 2.0))


def test_cellprofiler_adapter_keeps_multisource_current_image_grouped_when_files_are_narrow():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    context = ContextStub(filemanager)
    image_name = "Hoechst"
    image_path = "/memory/Hoechst.pkl"
    source_paths = (
        "/plate/Images/A01_s1_w1.tif",
        "/plate/Images/A01_s2_w1.tif",
    )
    source_metadata_by_path = {
        source_paths[0]: {"Site": "1", "ChannelNumber": "1"},
        source_paths[1]: {"Site": "2", "ChannelNumber": "1"},
    }
    output_plan = ArtifactOutputPlan(
        name=image_name,
        path=image_path,
        artifact_type=ImageArtifactType,
        group_keys=(None,),
        group_component=AllComponents.SITE,
    )

    for group_key, source_path, value in (
        ("1", source_paths[0], 1.0),
        ("2", source_paths[1], 2.0),
    ):
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            group_key=group_key,
            artifact_output_bindings=(
                _output_binding(image_name, ImageArtifactType, plan=output_plan),
            ),
            filemanager=filemanager,
        )
        producer.add_image(
            image_name,
            ImagePayloadMetadata(
                source_path=source_path,
                source_component_metadata=source_metadata_by_path[source_path],
            ).payload_with(np.full((2, 2), value, dtype=np.float32), None),
        )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=image_name,
                    path=image_path,
                    artifact_type=ImageArtifactType,
                    group_keys=(None,),
                    group_component=AllComponents.SITE,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.dynamic(
                    AllComponents.SITE
                ),
                component_scopes=(ComponentGroupScope.dynamic(AllComponents.SITE),),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        source_binding_context=SourceBindingRuntimeContext(
            step_input_files=source_paths,
            current_step_input_files=(source_paths[0],),
            pipeline_input_files=source_paths,
            source_metadata_by_path=source_metadata_by_path,
        ),
        microscope_handler=(context).microscope_handler,
        filemanager=filemanager,
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )
    ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=source_paths
        )
    ).payload_with(np.zeros((2, 2, 2), dtype=np.float32), None)

    image = consumer.get_image(
        image_name,
    )
    metadata = image_payload_metadata(image)

    assert image_payload_data(image).shape == (2, 2, 2)
    np.testing.assert_array_equal(image_payload_data(image)[0], np.full((2, 2), 1.0))
    np.testing.assert_array_equal(image_payload_data(image)[1], np.full((2, 2), 2.0))
    assert metadata.source_image_provenance_planes.paths == source_paths


def test_cellprofiler_adapter_stacks_declared_image_input_for_pattern_group():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    context = ContextStub(filemanager)
    image_name = "Hoechst"
    image_path = "/memory/Hoechst.pkl"
    source_paths = (
        "/plate/Images/A01_s001_w1_z001_t001.tif",
        "/plate/Images/A01_s002_w1_z001_t001.tif",
    )
    output_plan = ArtifactOutputPlan(
        name=image_name,
        path=image_path,
        artifact_type=ImageArtifactType,
        group_keys=(None,),
        group_component=AllComponents.SITE,
    )

    for group_key, source_path, value in (
        ("1", source_paths[0], 1.0),
        ("2", source_paths[1], 2.0),
    ):
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            group_key=group_key,
            artifact_output_bindings=(
                _output_binding(image_name, ImageArtifactType, plan=output_plan),
            ),
            filemanager=filemanager,
        )
        producer.add_image(
            image_name,
            ImagePayloadMetadata(
                source_path=source_path,
                source_component_metadata={"site": group_key},
            ).payload_with(np.full((2, 2), value, dtype=np.float32), None),
        )

    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=source_paths,
        current_step_input_files=source_paths,
        pipeline_input_files=source_paths,
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=image_name,
                    path=image_path,
                    artifact_type=ImageArtifactType,
                    group_keys=(None,),
                    group_component=AllComponents.SITE,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.dynamic(
                    AllComponents.SITE
                ),
                component_scopes=(ComponentGroupScope.dynamic(AllComponents.SITE),),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        group_key="A01_s{iii}_w1_z001_t001.tif",
        artifact_inputs=_compiled_artifact_inputs,
        source_binding_context=source_binding_context,
        microscope_handler=(context).microscope_handler,
        filemanager=filemanager,
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )
    ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=source_paths,
            component_metadata=(
                {"site": "1"},
                {"site": "2"},
            ),
        )
    ).payload_with(np.zeros((2, 2, 2), dtype=np.float32), None)

    image = consumer.get_image(
        image_name,
    )

    assert image_payload_data(image).shape == (2, 2, 2)
    np.testing.assert_array_equal(image_payload_data(image)[0], np.full((2, 2), 1.0))
    np.testing.assert_array_equal(image_payload_data(image)[1], np.full((2, 2), 2.0))


def test_cellprofiler_adapter_records_output_in_declared_invocation_group():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    output_name = "MembMasked"
    source_path = "/plate/Images/3d_monolayer_xy1_ch3.tif"
    mask_path = "/plate/Images/3d_monolayer_xy1_ch1.tif"
    output_plan = ArtifactOutputPlan(
        name=output_name,
        path=f"/memory/{output_name}.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("0", "3"),
        group_component=AllComponents.CHANNEL,
        paths_by_group={
            "0": f"/memory/{output_name}_w0.pkl",
            "3": f"/memory/{output_name}_w3.pkl",
        },
    )
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            AXIS_ID,
            component=AllComponents.CHANNEL,
            value="3",
            fixed_component_values=(
                (AllComponents.Z_INDEX, "1"),
                (AllComponents.TIMEPOINT, "1"),
            ),
        ),
        group_key="3",
        artifact_output_bindings=(
            _output_binding(output_name, ImageArtifactType, plan=output_plan),
        ),
        source_binding_context=SourceBindingRuntimeContext(
            step_input_files=(mask_path,),
            current_step_input_files=(mask_path,),
            pipeline_input_files=(
                "/plate/Images/3d_monolayer_xy1_ch0.tif",
                mask_path,
                source_path,
            ),
        ),
        microscope_handler=(ContextStub(filemanager)).microscope_handler,
        filemanager=filemanager,
    )
    output_payload = ImagePayloadMetadata(
        source_path=source_path,
        source_component_metadata={
            "well": "A01",
            "site": "001",
            "channel": "3",
            "z_index": "001",
            "time_index": "001",
        },
    ).payload_with(np.full((2, 2), 3.0, dtype=np.float32), None)

    stored = adapter.add_image(output_name, output_payload)

    assert stored.key.scope.value_text == "3"
    assert stored.key.scope.fixed_component_values == (
        (AllComponents.Z_INDEX, "1"),
        (AllComponents.TIMEPOINT, "1"),
    )
    assert ("memory", f"/memory/{output_name}_w3.pkl") in filemanager.saved

    with pytest.raises(RuntimeError, match="no selected artifact output plan"):
        adapter.artifact_output_value(
            replace(output_plan, artifact_type=ObjectLabelsArtifactType)
        )


def test_cellprofiler_adapter_projects_source_bound_runtime_image_to_group_plane():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    image_name = "MaskedMito"
    image_path = "/memory/MaskedMito.pkl"
    source_paths = (
        "/plate/Images/A01_s1_w5.tif",
        "/plate/Images/A01_s2_w5.tif",
    )
    source_metadata = (
        {"site": "1", "channel": "5"},
        {"site": "2", "channel": "5"},
    )
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                image_name,
                ImageArtifactType,
                plan=ArtifactOutputPlan(
                    name=image_name,
                    path=image_path,
                    artifact_type=ImageArtifactType,
                    group_keys=(None,),
                    variable_components=(AllComponents.SITE,),
                    paths_by_group={None: image_path},
                ),
            ),
        ),
        filemanager=filemanager,
    )
    producer.add_image(
        image_name,
        ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=source_paths, component_metadata=source_metadata
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(
            np.stack(
                (
                    np.full((2, 3), 1.0, dtype=np.float32),
                    np.full((2, 3), 2.0, dtype=np.float32),
                ),
                axis=0,
            ),
            None,
        ),
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=image_name,
                    path=image_path,
                    artifact_type=ImageArtifactType,
                    group_keys=(None,),
                    paths_by_group={None: image_path},
                    variable_components=(AllComponents.SITE,),
                ),
                invocation_scope=ComponentGroupScope.from_raw(
                    ("2",), component=AllComponents.SITE
                ),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(
                    ComponentGroupScope.from_raw(("2",), component=AllComponents.SITE),
                ),
                consumer_variable_components=(),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID, "site", "2"),
        group_key="2",
        plane_projection=RuntimePlaneProjection.selected(1, 2),
        artifact_inputs=_compiled_artifact_inputs,
        source_binding_context=SourceBindingRuntimeContext(
            source_metadata_by_path=dict(
                zip(source_paths, source_metadata, strict=True)
            ),
        ),
        filemanager=filemanager,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    image = consumer.get_image(image_name)
    metadata = image_payload_metadata(image)

    assert image_payload_data(image).shape == (2, 3)
    np.testing.assert_array_equal(
        image_payload_data(image),
        np.full((2, 3), 2.0, dtype=np.float32),
    )
    assert metadata.source_path == source_paths[1]
    assert metadata.source_component_metadata == source_metadata[1]


def test_cellprofiler_adapter_deduplicates_grouped_runtime_image_input_locations():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    image_name = "MaskedMito"
    image_path = "/memory/MaskedMito.pkl"
    source_paths = (
        "/plate/Images/A01_s1_w5.tif",
        "/plate/Images/A01_s2_w5.tif",
    )
    source_metadata = (
        {"site": "1", "channel": "5"},
        {"site": "2", "channel": "5"},
    )
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                image_name,
                ImageArtifactType,
                plan=ArtifactOutputPlan(
                    name=image_name,
                    path=image_path,
                    artifact_type=ImageArtifactType,
                    group_keys=(None,),
                    variable_components=(AllComponents.SITE,),
                    paths_by_group={None: image_path},
                ),
            ),
        ),
        filemanager=filemanager,
    )
    producer.add_image(
        image_name,
        ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=source_paths, component_metadata=source_metadata
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(
            np.stack(
                (
                    np.full((2, 3), 1.0, dtype=np.float32),
                    np.full((2, 3), 2.0, dtype=np.float32),
                ),
                axis=0,
            ),
            None,
        ),
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=image_name,
                    path=image_path,
                    artifact_type=ImageArtifactType,
                    group_keys=(None,),
                    paths_by_group={None: image_path},
                    variable_components=(AllComponents.SITE,),
                ),
                invocation_scope=ComponentGroupScope.from_raw(
                    ("2",), component=AllComponents.SITE
                ),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(
                    ComponentGroupScope.from_raw(("2",), component=AllComponents.SITE),
                ),
                consumer_variable_components=(),
            ),
        )
    }
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID, "site", "2"),
        plane_projection=RuntimePlaneProjection.selected(1, 2),
        artifact_inputs=_compiled_artifact_inputs,
        source_binding_context=SourceBindingRuntimeContext(
            source_metadata_by_path=dict(
                zip(source_paths, source_metadata, strict=True)
            ),
        ),
        filemanager=filemanager,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    image = consumer.get_image(image_name)
    metadata = image_payload_metadata(image)

    assert image_payload_data(image).shape == (2, 3)
    np.testing.assert_array_equal(
        image_payload_data(image),
        np.full((2, 3), 2.0, dtype=np.float32),
    )
    assert metadata.source_path == source_paths[1]
    assert metadata.source_component_metadata == source_metadata[1]


def test_cellprofiler_adapter_uses_identity_record_for_collapsed_grouped_input():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    image_name = "MaskedMito"
    image_path = "/memory/MaskedMito.pkl"
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                image_name,
                ImageArtifactType,
                plan=ArtifactOutputPlan(
                    name=image_name,
                    path=image_path,
                    artifact_type=ImageArtifactType,
                    group_keys=(None,),
                    paths_by_group={None: image_path},
                ),
            ),
        ),
        filemanager=filemanager,
    )
    producer.add_image(
        image_name,
        ImagePayloadMetadata(source_path="/plate/Images/A01_s1_w5.tif").payload_with(
            np.full((2, 3), 7.0, dtype=np.float32), None
        ),
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=image_name,
                    path=image_path,
                    artifact_type=ImageArtifactType,
                    group_keys=(None,),
                    paths_by_group={None: image_path},
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        )
    }
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    image = consumer.get_image(image_name)

    np.testing.assert_array_equal(
        image_payload_data(image),
        np.full((2, 3), 7.0, dtype=np.float32),
    )


def test_cellprofiler_adapter_uses_grouped_input_when_consumer_group_is_different_component():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    image_name = "Mito"
    image_path = "/memory/Mito.pkl"
    output_plan = ArtifactOutputPlan(
        name=image_name,
        path=image_path,
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.SITE,
        paths_by_group={
            "1": "/memory/A01_w1_Mito.pkl",
            "2": "/memory/A01_w2_Mito.pkl",
        },
    )
    for group_key, value in (("1", 1.0), ("2", 2.0)):
        cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID, "site", group_key),
            group_key=group_key,
            artifact_output_bindings=(
                _output_binding(image_name, ImageArtifactType, plan=output_plan),
            ),
            filemanager=filemanager,
        ).add_image(
            image_name,
            ImagePayloadMetadata().payload_with(
                np.full((2, 3), value, dtype=np.float32), None
            ),
        )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=image_name,
                    path=image_path,
                    artifact_type=ImageArtifactType,
                    group_keys=("1", "2"),
                    group_component=AllComponents.SITE,
                    paths_by_group={
                        "1": "/memory/A01_w1_Mito.pkl",
                        "2": "/memory/A01_w2_Mito.pkl",
                    },
                ),
                invocation_scope=ComponentGroupScope.from_raw(
                    ("1",), component=AllComponents.CHANNEL
                ),
                producer_selection_scope=ComponentGroupScope.from_raw(
                    ("1", "2"), component=AllComponents.SITE
                ),
                component_scopes=(
                    ComponentGroupScope.from_raw(
                        ("1",), component=AllComponents.CHANNEL
                    ),
                    ComponentGroupScope.from_raw(
                        ("1", "2"), component=AllComponents.SITE
                    ),
                ),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID, "channel", "1"),
        group_key="1",
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    image = consumer.get_image(image_name)

    assert image_payload_data(image).shape == (2, 2, 3)
    np.testing.assert_array_equal(
        image_payload_data(image)[0],
        np.full((2, 3), 1.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        image_payload_data(image)[1],
        np.full((2, 3), 2.0, dtype=np.float32),
    )


def test_cellprofiler_adapter_does_not_project_channel_stack_for_site_group():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    image_name = "Corrected"
    image_path = "/memory/Corrected.pkl"
    source_paths = (
        "/plate/Images/A01_s2_w1.tif",
        "/plate/Images/A01_s2_w2.tif",
    )
    source_metadata = (
        {"site": "2", "channel": "1"},
        {"site": "2", "channel": "2"},
    )
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                image_name,
                ImageArtifactType,
                plan=ArtifactOutputPlan(
                    name=image_name,
                    path=image_path,
                    artifact_type=ImageArtifactType,
                    group_keys=(None,),
                    variable_components=(AllComponents.CHANNEL,),
                    paths_by_group={None: image_path},
                ),
            ),
        ),
        filemanager=filemanager,
        variable_components=(VariableComponents.CHANNEL,),
    )
    producer.add_image(
        image_name,
        ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=source_paths, component_metadata=source_metadata
            )
        ).payload_with(
            np.stack(
                (
                    np.full((2, 3), 1.0, dtype=np.float32),
                    np.full((2, 3), 2.0, dtype=np.float32),
                ),
                axis=0,
            ),
            None,
        ),
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=image_name,
                    path=image_path,
                    artifact_type=ImageArtifactType,
                    group_keys=(None,),
                    paths_by_group={None: image_path},
                    variable_components=(AllComponents.CHANNEL,),
                ),
                invocation_scope=ComponentGroupScope.from_raw(
                    ("2",), component=AllComponents.SITE
                ),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(
                    ComponentGroupScope.from_raw(("2",), component=AllComponents.SITE),
                    ComponentGroupScope.from_raw(
                        ("1", "2"), component=AllComponents.CHANNEL
                    ),
                ),
                consumer_variable_components=(AllComponents.CHANNEL,),
            ),
        )
    }
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID, "site", "2"),
        group_key="2",
        plane_projection=RuntimePlaneProjection.selected(1, 2),
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        variable_components=(VariableComponents.CHANNEL,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    image = consumer.get_image(image_name)
    metadata = image_payload_metadata(image)

    assert image_payload_data(image).shape == (2, 2, 3)
    np.testing.assert_array_equal(
        image_payload_data(image)[0],
        np.full((2, 3), 1.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        image_payload_data(image)[1],
        np.full((2, 3), 2.0, dtype=np.float32),
    )
    assert metadata.source_image_provenance_planes.paths == source_paths


def test_cellprofiler_adapter_projects_stack_without_replacing_artifact_provenance():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    image_name = "Masked"
    image_path = "/memory/Masked.pkl"
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                image_name,
                ImageArtifactType,
                plan=ArtifactOutputPlan(
                    name=image_name,
                    path=image_path,
                    artifact_type=ImageArtifactType,
                    group_keys=(None,),
                    variable_components=(AllComponents.SITE,),
                    paths_by_group={None: image_path},
                ),
            ),
        ),
        filemanager=filemanager,
    )
    producer.add_image(
        image_name,
        ImagePayloadMetadata(
            source_path="/plate/Images/A01_s1_w1.tif",
            source_component_metadata={"site": "1", "channel": "1"},
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/plate/Images/A01_s1_w1.tif",
                    "/plate/Images/A01_s1_w1.tif",
                ),
                component_metadata=(
                    {"site": "1", "channel": "1"},
                    {"site": "2", "channel": "1"},
                ),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(
            np.stack(
                (
                    np.full((2, 3), 1.0, dtype=np.float32),
                    np.full((2, 3), 2.0, dtype=np.float32),
                ),
                axis=0,
            ),
            None,
        ),
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=image_name,
                    path=image_path,
                    artifact_type=ImageArtifactType,
                    group_keys=(None,),
                    paths_by_group={None: image_path},
                    variable_components=(AllComponents.SITE,),
                ),
                invocation_scope=ComponentGroupScope.from_raw(
                    ("2",), component=AllComponents.SITE
                ),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(
                    ComponentGroupScope.from_raw(("2",), component=AllComponents.SITE),
                ),
                consumer_variable_components=(),
            ),
        )
    }
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID, "site", "2"),
        group_key="2",
        plane_projection=RuntimePlaneProjection.selected(1, 2),
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )
    ImagePayloadMetadata(
        source_path="/plate/Images/A01_s2_w1.tif",
        source_component_metadata={"site": "2", "channel": "1"},
    ).payload_with(np.zeros((2, 3), dtype=np.float32), None)

    image = consumer.get_image(
        image_name,
    )
    metadata = image_payload_metadata(image)

    assert image_payload_data(image).shape == (2, 3)
    np.testing.assert_array_equal(
        image_payload_data(image),
        np.full((2, 3), 2.0, dtype=np.float32),
    )
    assert metadata.source_path == "/plate/Images/A01_s1_w1.tif"
    assert metadata.source_component_metadata == {"site": "2", "channel": "1"}


def test_cellprofiler_adapter_does_not_select_image_record_from_current_source_scope():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    image_name = "Mito"
    image_paths = {
        "1": "/memory/Mito_site1.pkl",
        "2": "/memory/Mito_site2.pkl",
    }
    output_plan = ArtifactOutputPlan(
        name=image_name,
        path="/memory/Mito.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.SITE,
        paths_by_group=image_paths,
    )
    for group_key, site, value in (
        ("1", "1", 1.0),
        ("2", "2", 2.0),
    ):
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            group_key=group_key,
            artifact_output_bindings=(
                _output_binding(image_name, ImageArtifactType, plan=output_plan),
            ),
            filemanager=filemanager,
        )
        producer.add_image(
            image_name,
            ImagePayloadMetadata(
                source_path=f"/plate/Images/A01_s{site}_w5.tif",
                source_component_metadata={
                    "well": "A01",
                    "site": site,
                    "channel": "5",
                },
            ).payload_with(np.full((2, 3), value, dtype=np.float32), None),
        )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=image_name,
                    path="/memory/Mito.pkl",
                    artifact_type=ImageArtifactType,
                    group_keys=("1", "2"),
                    group_component=AllComponents.SITE,
                    paths_by_group=image_paths,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.from_raw(
                    ("1", "2"), component=AllComponents.SITE
                ),
                component_scopes=(
                    ComponentGroupScope.from_raw(
                        ("1", "2"), component=AllComponents.SITE
                    ),
                ),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )
    ImagePayloadMetadata(
        source_path="/plate/Images/A01_s2_w3.tif",
        source_component_metadata={
            "well": "A01",
            "site": "2",
            "channel": "3",
        },
    ).payload_with(np.zeros((2, 3), dtype=np.float32), None)

    image = consumer.get_image(
        image_name,
    )
    metadata = image_payload_metadata(image)

    np.testing.assert_array_equal(
        image_payload_data(image),
        np.stack(
            (
                np.full((2, 3), 1.0, dtype=np.float32),
                np.full((2, 3), 2.0, dtype=np.float32),
            )
        ),
    )
    assert metadata.source_path is None
    assert metadata.source_image_provenance_planes.paths == (
        "/plate/Images/A01_s1_w5.tif",
        "/plate/Images/A01_s2_w5.tif",
    )
    assert tuple(
        dict(component_metadata)
        for component_metadata in metadata.source_image_provenance_planes.component_metadata
        if component_metadata is not None
    ) == (
        {"well": "A01", "site": "1", "channel": "5"},
        {"well": "A01", "site": "2", "channel": "5"},
    )


def test_cellprofiler_adapter_keeps_template_scoped_object_records_grouped():
    store = RuntimeValueStore()
    outputs = {
        NUCLEI: _plan(
            NUCLEI, ObjectLabelsArtifactType, group_component=AllComponents.SITE
        )
    }
    output_bindings = (
        _output_binding(
            NUCLEI,
            ObjectLabelsArtifactType,
            plan=outputs[NUCLEI],
        ),
    )
    filemanager = FileManagerStub()

    for group_key, source_path, value in (
        ("1", "/src/A01_s001_w2.tif", 1),
        ("2", "/src/A01_s002_w2.tif", 2),
    ):
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            artifact_output_bindings=output_bindings,
            group_key=group_key,
            filemanager=filemanager,
        )
        producer.add_objects(
            NUCLEI,
            ObjectLabelSet(
                name=NUCLEI,
                variant_data=ObjectLabelVariantData(
                    labels=np.full((2, 2), value, dtype=np.int32)
                ),
                source_path=source_path,
                domain=ObjectLabelDomain(declared_object_ids=(value,)),
            ),
        )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI,
                    path=outputs[NUCLEI].path,
                    artifact_type=ObjectLabelsArtifactType,
                    group_component=AllComponents.SITE,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.dynamic(
                    AllComponents.SITE
                ),
                component_scopes=(ComponentGroupScope.dynamic(AllComponents.SITE),),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        variable_components=(VariableComponents.SITE,),
        source_binding_context=SourceBindingRuntimeContext(
            step_input_files=("/src/A01_s{iii}_w1.tif",),
        ),
        filemanager=filemanager,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )
    ImagePayloadMetadata(source_path="/src/A01_s{iii}_w1.tif").payload_with(
        np.zeros((2, 2), dtype=np.float32), None
    )

    objects = consumer.get_objects(
        NUCLEI,
    )

    assert objects.labels.shape == (2, 2, 2)
    np.testing.assert_array_equal(objects.labels[0], np.full((2, 2), 1))
    np.testing.assert_array_equal(objects.labels[1], np.full((2, 2), 2))


def test_cellprofiler_adapter_does_not_project_stacked_objects_from_source_metadata():
    output_spec = ArtifactSpec.output(NUCLEI, ObjectLabelsArtifactType)
    output_plan = _plan(NUCLEI, ObjectLabelsArtifactType)
    adapter, _filemanager = _adapter(
        (_output_binding_for_spec(output_spec, plan=output_plan),)
    )
    artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI,
                    path=output_plan.path,
                    artifact_type=ObjectLabelsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        )
    }
    adapter.request = replace(
        adapter.request,
        artifact_inputs=artifact_inputs,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(edge.spec for edge in artifact_inputs.values()),
            artifact_outputs=(output_spec,),
        ),
    )
    source_paths = (
        "/plate/Images/A01_s1_w1.tif",
        "/plate/Images/A01_s2_w1.tif",
    )
    labels = np.stack(
        (
            np.full((2, 2), 1, dtype=np.int32),
            np.full((2, 2), 2, dtype=np.int32),
        ),
        axis=0,
    )
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=labels),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=source_paths, component_metadata=({"site": "1"}, {"site": "2"})
            ),
            source_image_names=(DNA_IMAGE,),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1,), (2,)),
            ),
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        ),
    )
    ImagePayloadMetadata(
        source_path=source_paths[1],
        source_component_metadata={"site": "2"},
    ).payload_with(np.zeros((2, 2), dtype=np.float32), None)

    objects = adapter.get_objects(
        NUCLEI,
    )

    np.testing.assert_array_equal(objects.labels, labels)
    assert objects.domain.declared_object_id_domains == ((1,), (2,))
    assert objects.source_image_names == (DNA_IMAGE,)


def test_cellprofiler_adapter_preserves_nominal_object_label_spatial_domains():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    nuclei_output = ArtifactSpec.output(NUCLEI, ObjectLabelsArtifactType)
    cells_output = ArtifactSpec.output(CELLS, ObjectLabelsArtifactType)
    source_plan = _plan(DNA_IMAGE, ImageArtifactType)
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(DNA_IMAGE, ImageArtifactType, plan=source_plan),
        ),
        filemanager=filemanager,
    )
    source_image = ImagePayloadMetadata(
        source_path="/plate/A01_s001_w1_z001_t001.tif",
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(3, 5),
            source_shape_yx=(16, 16),
        ),
    ).payload_with(np.zeros((4, 4), dtype=np.float32), None)
    producer.add_image(DNA_IMAGE, source_image)
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=DNA_IMAGE,
                    path=source_plan.path,
                    artifact_type=ImageArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        )
    }
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        artifact_output_bindings=(
            _output_binding_for_spec(nuclei_output),
            _output_binding_for_spec(cells_output),
        ),
        filemanager=filemanager,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
            artifact_outputs=(nuclei_output, cells_output),
        ),
    )

    source_aligned_labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((2, 2), dtype=np.int32)),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(3, 5),
            source_shape_yx=(16, 16),
        ),
    )
    consumer.add_objects(
        NUCLEI,
        source_aligned_labels,
        source_image_name=DNA_IMAGE,
    )
    source_aligned_objects = _output_objects(consumer, NUCLEI)
    assert source_aligned_objects.spatial_origin_yx == (3, 5)
    assert source_aligned_objects.source_spatial_shape_yx == (16, 16)

    transformed_labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((1, 1), dtype=np.int32))
    )
    consumer.add_objects(CELLS, transformed_labels, source_image_name=DNA_IMAGE)
    transformed_objects = _output_objects(consumer, CELLS)
    assert transformed_objects.spatial_origin_yx is None
    assert transformed_objects.source_spatial_shape_yx is None


@pytest.mark.parametrize(
    "labels",
    (
        np.ones((2, 2), dtype=np.int32),
        SparseIJVLabelRows(np.array([[0, 0, 1]], dtype=np.int32)),
    ),
)
def test_cellprofiler_adapter_rejects_non_nominal_object_label_values(labels):
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    source_plan = ArtifactOutputPlan(
        name=DNA_IMAGE,
        path="/memory/DNA.pkl",
        artifact_type=ImageArtifactType,
    )
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(DNA_IMAGE, ImageArtifactType, plan=source_plan),
        ),
        filemanager=filemanager,
    )
    producer.add_image(DNA_IMAGE, np.zeros((2, 2), dtype=np.float32))
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=DNA_IMAGE,
                    path=source_plan.path,
                    artifact_type=ImageArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        )
    }
    nuclei_output = ArtifactSpec.output(NUCLEI, ObjectLabelsArtifactType)
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        artifact_output_bindings=(_output_binding_for_spec(nuclei_output),),
        filemanager=filemanager,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
            artifact_outputs=(nuclei_output,),
        ),
    )

    with pytest.raises(TypeError, match="requires an ObjectLabelValue"):
        consumer.add_objects(
            NUCLEI,
            labels,
            source_image_name=DNA_IMAGE,
        )


def test_cellprofiler_adapter_reads_declared_inputs_by_compiled_location():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
        ),
        filemanager=filemanager,
    )
    labels = np.zeros((2, 2), dtype=np.int32)
    producer.add_objects(
        NUCLEI, ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels))
    )

    input_plan = ArtifactInputPlan(
        name=NUCLEI,
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=(None,),
    )
    input_edge = cellprofiler_runtime_input_edge_for_test(
        input_plan,
        invocation_scope=ComponentGroupScope.ungrouped(),
        producer_selection_scope=ComponentGroupScope.ungrouped(),
        component_scopes=(),
        consumer_variable_components=(),
    )
    _compiled_artifact_inputs = {input_edge.key: input_edge}
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    assert consumer.get_objects(NUCLEI).labels is labels


def test_cellprofiler_adapter_availability_accepts_grouped_runtime_inputs():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    group_paths = {
        "1": "/memory/Nuclei_s1.pkl",
        "2": "/memory/Nuclei_s2.pkl",
    }
    for group_key, path in group_paths.items():
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            group_key=group_key,
            artifact_output_bindings=(
                _output_binding(
                    NUCLEI,
                    ObjectLabelsArtifactType,
                    plan=ArtifactOutputPlan(
                        name=NUCLEI,
                        path=path,
                        artifact_type=ObjectLabelsArtifactType,
                        group_keys=(group_key,),
                        group_component=AllComponents.SITE,
                        paths_by_group={group_key: path},
                    ),
                ),
            ),
            filemanager=filemanager,
        )
        producer.add_objects(
            NUCLEI,
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(
                    labels=np.full((2, 2), int(group_key), dtype=np.int32)
                )
            ),
        )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI,
                    path="/memory/Nuclei.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    group_keys=tuple(group_paths),
                    group_component=AllComponents.SITE,
                    paths_by_group=group_paths,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.from_raw(
                    ("1", "2"), component=AllComponents.SITE
                ),
                component_scopes=(
                    ComponentGroupScope.from_raw(
                        ("1", "2"), component=AllComponents.SITE
                    ),
                ),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    consumer.require_artifact_available(
        name=NUCLEI,
        kind=ObjectLabelsArtifactType,
    )


def test_cellprofiler_adapter_discovers_single_realized_dynamic_grouped_input():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    realized_labels = np.full((2, 2), 1, dtype=np.int32)
    realized_path = "/memory/Nuclei_s1.pkl"
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        group_key="1",
        artifact_output_bindings=(
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=ArtifactOutputPlan(
                    name=NUCLEI,
                    path=realized_path,
                    artifact_type=ObjectLabelsArtifactType,
                    group_keys=("1",),
                    group_component=AllComponents.CHANNEL,
                    paths_by_group={"1": realized_path},
                ),
            ),
        ),
        filemanager=filemanager,
    )
    producer.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=realized_labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI,
                    path="/memory/Nuclei.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    group_keys=(None,),
                    group_component=AllComponents.CHANNEL,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.dynamic(
                    AllComponents.CHANNEL
                ),
                component_scopes=(ComponentGroupScope.dynamic(AllComponents.CHANNEL),),
                consumer_variable_components=(AllComponents.CHANNEL,),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        variable_components=(VariableComponents.CHANNEL,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    objects = consumer.get_objects(NUCLEI)

    np.testing.assert_array_equal(objects.labels, realized_labels[None, ...])
    assert objects.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert objects.domain.declared_object_id_domains == ((1,),)


def test_cellprofiler_adapter_discovers_realized_dynamic_grouped_object_inputs():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    group_paths = {
        "1": "/memory/Cells_w1.pkl",
        "2": "/memory/Cells_w2.pkl",
        "3": "/memory/Cells_w3.pkl",
    }
    realized_labels = {
        "1": np.full((2, 2), 1, dtype=np.int32),
        "3": np.full((2, 2), 3, dtype=np.int32),
    }
    for group_key, labels in realized_labels.items():
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            group_key=group_key,
            artifact_output_bindings=(
                _output_binding(
                    CELLS,
                    ObjectLabelsArtifactType,
                    plan=ArtifactOutputPlan(
                        name=CELLS,
                        path=group_paths[group_key],
                        artifact_type=ObjectLabelsArtifactType,
                        group_keys=(group_key,),
                        group_component=AllComponents.CHANNEL,
                        paths_by_group={group_key: group_paths[group_key]},
                    ),
                ),
            ),
            filemanager=filemanager,
        )
        producer.add_objects(
            CELLS,
            ObjectLabelSet(
                name=CELLS,
                variant_data=ObjectLabelVariantData(labels=labels),
                domain=ObjectLabelDomain(
                    declared_object_ids=(int(group_key),),
                ),
            ),
        )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=CELLS,
                    path="/memory/Cells.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    group_keys=(None,),
                    group_component=AllComponents.CHANNEL,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.dynamic(
                    AllComponents.CHANNEL
                ),
                component_scopes=(ComponentGroupScope.dynamic(AllComponents.CHANNEL),),
                consumer_variable_components=(AllComponents.CHANNEL,),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        variable_components=(VariableComponents.CHANNEL,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    objects = consumer.get_objects(CELLS)

    np.testing.assert_array_equal(
        objects.labels,
        np.stack((realized_labels["1"], realized_labels["3"])),
    )


def test_cellprofiler_adapter_composes_object_input_across_declared_site_axis():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    first_labels = np.full((2, 2), 1, dtype=np.int32)
    second_labels = np.full((2, 2), 2, dtype=np.int32)
    group_paths = {
        "1": "/memory/Nuclei_s1.pkl",
        "2": "/memory/Nuclei_s2.pkl",
    }
    for group_key, labels in (
        ("1", first_labels),
        ("2", second_labels),
    ):
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            group_key=group_key,
            artifact_output_bindings=(
                _output_binding(
                    NUCLEI,
                    ObjectLabelsArtifactType,
                    plan=ArtifactOutputPlan(
                        name=NUCLEI,
                        path=group_paths[group_key],
                        artifact_type=ObjectLabelsArtifactType,
                        group_keys=(group_key,),
                        group_component=AllComponents.SITE,
                        paths_by_group={group_key: group_paths[group_key]},
                    ),
                ),
            ),
            filemanager=filemanager,
        )
        producer.add_objects(
            NUCLEI,
            ObjectLabelSet(
                name=NUCLEI,
                variant_data=ObjectLabelVariantData(labels=labels),
                domain=ObjectLabelDomain(
                    declared_object_ids=(int(group_key),),
                ),
            ),
        )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI,
                    path="/memory/Nuclei.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    group_keys=("1", "2"),
                    group_component=AllComponents.SITE,
                    paths_by_group=group_paths,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.from_raw(
                    ("1", "2"), component=AllComponents.SITE
                ),
                component_scopes=(
                    ComponentGroupScope.from_raw(
                        ("1", "2"), component=AllComponents.SITE
                    ),
                ),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )
    ImagePayloadMetadata(
        source_path="/plate/Images/A01_s001_w1_z001_t001.tif",
    ).payload_with(np.zeros((2, 2), dtype=np.float32), None)

    objects = consumer.get_objects(
        NUCLEI,
    )

    np.testing.assert_array_equal(
        objects.labels,
        np.stack((first_labels, second_labels)),
    )


def test_cellprofiler_adapter_does_not_resolve_object_input_from_source_context():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    first_labels = np.full((2, 2), 1, dtype=np.int32)
    second_labels = np.full((2, 2), 2, dtype=np.int32)
    group_paths = {
        "1": "/memory/Nuclei_s1.pkl",
        "2": "/memory/Nuclei_s2.pkl",
    }
    for group_key, labels in (("1", first_labels), ("2", second_labels)):
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            group_key=group_key,
            artifact_output_bindings=(
                _output_binding(
                    NUCLEI,
                    ObjectLabelsArtifactType,
                    plan=ArtifactOutputPlan(
                        name=NUCLEI,
                        path=group_paths[group_key],
                        artifact_type=ObjectLabelsArtifactType,
                        group_keys=(group_key,),
                        group_component=AllComponents.SITE,
                        paths_by_group={group_key: group_paths[group_key]},
                    ),
                ),
            ),
            filemanager=filemanager,
        )
        producer.add_objects(
            NUCLEI,
            ObjectLabelSet(
                name=NUCLEI,
                variant_data=ObjectLabelVariantData(labels=labels),
                domain=ObjectLabelDomain(
                    declared_object_ids=(int(group_key),),
                ),
            ),
        )

    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "/plate/Images/A01_s002_w1_z001_t001.tif",
            "/plate/Images/A01_s002_w2_z001_t001.tif",
        ),
        pipeline_input_files=(
            "/plate/Images/A01_s001_w1_z001_t001.tif",
            "/plate/Images/A01_s001_w2_z001_t001.tif",
            "/plate/Images/A01_s002_w1_z001_t001.tif",
            "/plate/Images/A01_s002_w2_z001_t001.tif",
        ),
        current_step_input_files=(
            "/plate/Images/A01_s002_w1_z001_t001.tif",
            "/plate/Images/A01_s002_w2_z001_t001.tif",
        ),
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI,
                    path="/memory/Nuclei.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    group_keys=("1", "2"),
                    group_component=AllComponents.SITE,
                    paths_by_group=group_paths,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.from_raw(
                    ("1", "2"), component=AllComponents.SITE
                ),
                component_scopes=(
                    ComponentGroupScope.from_raw(
                        ("1", "2"), component=AllComponents.SITE
                    ),
                ),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        source_binding_context=source_binding_context,
        microscope_handler=(ContextStub(filemanager)).microscope_handler,
        filemanager=filemanager,
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    objects = consumer.get_objects(
        NUCLEI,
    )

    np.testing.assert_array_equal(
        objects.labels,
        np.stack((first_labels, second_labels)),
    )


def test_cellprofiler_adapter_preserves_ungrouped_runtime_slice_output_stack():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=ArtifactOutputPlan(
                    name=NUCLEI,
                    path="/memory/Nuclei.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    variable_components=(AllComponents.SITE,),
                ),
            ),
        ),
        filemanager=filemanager,
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.stack(
                (
                    np.full((2, 2), 1, dtype=np.int32),
                    np.full((2, 2), 2, dtype=np.int32),
                )
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    producer.add_objects(NUCLEI, labels)

    assert ("memory", "/memory/Nuclei.pkl") in filemanager.saved
    assert len(filemanager.saved) == 1
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI,
                    path="/memory/Nuclei.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    variable_components=(AllComponents.SITE,),
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(
                    ComponentGroupScope.from_raw(
                        ("1", "2"), component=AllComponents.SITE
                    ),
                ),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    objects = consumer.get_objects(NUCLEI)

    assert objects.labels.shape == (2, 2, 2)
    assert objects.domain.declared_object_id_domains == ((1,), (2,))
    np.testing.assert_array_equal(objects.labels[0], np.full((2, 2), 1))
    np.testing.assert_array_equal(objects.labels[1], np.full((2, 2), 2))


def test_cellprofiler_adapter_stacks_dynamic_compiled_grouped_images():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    first = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        group_key="1",
        artifact_output_bindings=(
            _output_binding(
                DNA_IMAGE,
                ImageArtifactType,
                plan=ArtifactOutputPlan(
                    name=DNA_IMAGE,
                    path="/memory/DNA_s1.pkl",
                    artifact_type=ImageArtifactType,
                    group_keys=("1",),
                    group_component=AllComponents.SITE,
                    paths_by_group={"1": "/memory/DNA_s1.pkl"},
                ),
            ),
        ),
        filemanager=filemanager,
    )
    second = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        group_key="2",
        artifact_output_bindings=(
            _output_binding(
                DNA_IMAGE,
                ImageArtifactType,
                plan=ArtifactOutputPlan(
                    name=DNA_IMAGE,
                    path="/memory/DNA_s2.pkl",
                    artifact_type=ImageArtifactType,
                    group_keys=("2",),
                    group_component=AllComponents.SITE,
                    paths_by_group={"2": "/memory/DNA_s2.pkl"},
                ),
            ),
        ),
        filemanager=filemanager,
    )
    first.add_image(DNA_IMAGE, np.full((2, 3), 1.0, dtype=np.float32))
    second.add_image(DNA_IMAGE, np.full((2, 3), 2.0, dtype=np.float32))
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=DNA_IMAGE,
                    path="/memory/DNA.pkl",
                    artifact_type=ImageArtifactType,
                    group_keys=(None,),
                    group_component=AllComponents.SITE,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.dynamic(
                    AllComponents.SITE
                ),
                component_scopes=(ComponentGroupScope.dynamic(AllComponents.SITE),),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    image = consumer.get_image(DNA_IMAGE)

    assert image_payload_data(image).shape == (2, 2, 3)
    np.testing.assert_array_equal(image_payload_data(image)[0], np.full((2, 3), 1.0))
    np.testing.assert_array_equal(image_payload_data(image)[1], np.full((2, 3), 2.0))


def test_cellprofiler_adapter_relationships_validate_declared_inputs_by_location():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    relationship_output = ArtifactSpec.output(
        PARENT_CHILD,
        RelationshipsArtifactType,
    )
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                CELLS,
                ObjectLabelsArtifactType,
                plan=_plan(CELLS, ObjectLabelsArtifactType),
            ),
        ),
        filemanager=filemanager,
    )
    producer.add_objects(
        NUCLEI,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32))
        ),
    )
    producer.add_objects(
        CELLS,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32))
        ),
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI,
                    path="/memory/Nuclei.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    group_keys=(None,),
                ),
                input_index=0,
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=CELLS,
                    path="/memory/Cells.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    group_keys=(None,),
                ),
                input_index=1,
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        artifact_output_bindings=(_output_binding_for_spec(relationship_output),),
        filemanager=filemanager,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
            artifact_outputs=(relationship_output,),
        ),
    )

    relationship = consumer.add_relationship(
        _parent_child_relationship(
            name=PARENT_CHILD,
            parent_object_name=NUCLEI,
            child_object_name=CELLS,
            payload=DirectedObjectRelationshipPayload(
                source_ids=(1,),
                target_ids=(2,),
            ),
        )
    )

    assert relationship.value.artifact_type is RelationshipsArtifactType


def test_cellprofiler_adapter_declared_relationship_allows_pruned_child_endpoint():
    filemanager = FileManagerStub()
    relationship_name = "Nuclei_FilteredCells_relationships"
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                relationship_name,
                RelationshipsArtifactType,
                plan=_plan(relationship_name, RelationshipsArtifactType),
            ),
        ),
        filemanager=filemanager,
    )

    relationship = adapter.add_relationship(
        _parent_child_relationship(
            name=relationship_name,
            parent_object_name=NUCLEI,
            child_object_name="FilteredCells",
            payload=DirectedObjectRelationshipPayload(
                source_ids=(1,),
                target_ids=(2,),
            ),
        )
    )

    assert relationship.value.artifact_type is RelationshipsArtifactType
    assert isinstance(relationship.value.data, ObjectRelationship)
    assert relationship.value.data.declaration.target.name == "FilteredCells"


def test_cellprofiler_adapter_relationships_accept_grouped_parent_inputs():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    first = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        group_key="1",
        artifact_output_bindings=(
            _output_binding(
                CELLS,
                ObjectLabelsArtifactType,
                plan=ArtifactOutputPlan(
                    name=CELLS,
                    path="/memory/Cells_s1.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    group_keys=("1",),
                    group_component=AllComponents.SITE,
                    paths_by_group={"1": "/memory/Cells_s1.pkl"},
                ),
            ),
        ),
        filemanager=filemanager,
    )
    second = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        group_key="2",
        artifact_output_bindings=(
            _output_binding(
                CELLS,
                ObjectLabelsArtifactType,
                plan=ArtifactOutputPlan(
                    name=CELLS,
                    path="/memory/Cells_s2.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    group_keys=("2",),
                    group_component=AllComponents.SITE,
                    paths_by_group={"2": "/memory/Cells_s2.pkl"},
                ),
            ),
        ),
        filemanager=filemanager,
    )
    first.add_objects(
        CELLS,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=np.zeros((2, 3), dtype=np.int32))
        ),
    )
    second.add_objects(
        CELLS,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=np.ones((2, 3), dtype=np.int32))
        ),
    )

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                PARENT_CHILD,
                RelationshipsArtifactType,
                plan=_plan(PARENT_CHILD, RelationshipsArtifactType),
            ),
        ),
        filemanager=filemanager,
    )
    consumer.add_objects(
        NUCLEI,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(
                labels=np.zeros((2, 2, 3), dtype=np.int32)
            )
        ),
    )

    relationship = consumer.add_relationship(
        _parent_child_relationship(
            name=PARENT_CHILD,
            parent_object_name=CELLS,
            child_object_name=NUCLEI,
            payload=DirectedObjectRelationshipPayload(
                source_ids=(1,),
                target_ids=(2,),
            ),
        )
    )

    assert relationship.value.artifact_type is RelationshipsArtifactType


def test_cellprofiler_adapter_relationships_allow_same_invocation_child_output():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    child_output = ArtifactSpec.output(CELLS, ObjectLabelsArtifactType)
    relationship_output = ArtifactSpec.output(
        PARENT_CHILD,
        RelationshipsArtifactType,
    )
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
        ),
        filemanager=filemanager,
    )
    producer.add_objects(
        NUCLEI,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32))
        ),
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI,
                    path="/memory/Nuclei.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    group_keys=(None,),
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        artifact_output_bindings=(
            _output_binding_for_spec(child_output),
            _output_binding_for_spec(relationship_output),
        ),
        filemanager=filemanager,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
            artifact_outputs=(child_output, relationship_output),
        ),
    )

    relationship = consumer.add_relationship(
        _parent_child_relationship(
            name=PARENT_CHILD,
            parent_object_name=NUCLEI,
            child_object_name=CELLS,
            payload=DirectedObjectRelationshipPayload(
                source_ids=(1,),
                target_ids=(2,),
            ),
        )
    )

    assert relationship.value.artifact_type is RelationshipsArtifactType


def test_cellprofiler_adapter_adds_and_reads_spatial_grid_artifacts():
    adapter, filemanager = _adapter(
        (
            _output_binding(
                "Grid",
                SpatialGridArtifactType,
                plan=_plan("Grid", SpatialGridArtifactType),
            ),
        )
    )
    grid = SpatialGrid(
        name="grid_info",
        rows=30,
        columns=30,
        x_spacing=55.0,
        y_spacing=55.0,
        x_origin=27.0,
        y_origin=27.0,
    )

    adapter.add_spatial_grid("Grid", grid)
    stored = _output_spatial_grid(adapter, "Grid")

    assert stored.name == "Grid"
    assert stored.rows == 30
    assert stored.columns == 30
    assert stored.x_origin == 27.0
    saved_grid = filemanager.saved[("memory", "/memory/Grid.pkl")]
    assert isinstance(saved_grid, SpatialGrid)
    assert saved_grid.rows == 30


def test_cellprofiler_adapter_adds_and_reads_slice_aligned_spatial_grids():
    adapter, filemanager = _adapter(
        (
            _output_binding(
                "Grid",
                SpatialGridArtifactType,
                plan=_plan("Grid", SpatialGridArtifactType),
            ),
        )
    )
    grids = [
        SpatialGrid(
            name="grid_info",
            rows=2,
            columns=2,
            x_spacing=8.0,
            y_spacing=8.0,
            x_origin=1.0,
            y_origin=4.0,
        ),
        SpatialGrid(
            name="grid_info",
            rows=2,
            columns=2,
            x_spacing=8.0,
            y_spacing=8.0,
            x_origin=2.0,
            y_origin=4.0,
        ),
    ]

    adapter.add_spatial_grid("Grid", grids)
    stored = _output_spatial_grid(adapter, "Grid")

    assert isinstance(stored, RuntimeSliceAlignedValues)
    assert stored.slice_count == 2
    assert [
        stored.value_for_slice(index).name for index in range(stored.slice_count)
    ] == ["Grid", "Grid"]
    assert [
        stored.value_for_slice(index).x_origin for index in range(stored.slice_count)
    ] == [1.0, 2.0]
    saved_grids = filemanager.saved[("memory", "/memory/Grid.pkl")]
    assert isinstance(saved_grids, RuntimeSliceAlignedValues)
    assert [
        saved_grids.value_for_slice(index).x_origin
        for index in range(saved_grids.slice_count)
    ] == [
        1.0,
        2.0,
    ]


def test_cellprofiler_adapter_replaces_existing_payload_with_latest_binding():
    adapter, filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
        )
    )
    first = np.ones((2, 2), dtype=np.uint16)
    second = np.full((2, 2), 2, dtype=np.uint16)

    adapter.add_objects(
        NUCLEI, ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=first))
    )
    record = adapter.add_objects(
        NUCLEI, ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=second))
    )

    assert isinstance(record.value.data, ObjectLabelSet)
    assert record.value.data.labels is second
    assert filemanager.deleted == [("memory", "/memory/Nuclei.pkl")]
    saved_objects = filemanager.saved[("memory", "/memory/Nuclei.pkl")]
    assert isinstance(saved_objects, ObjectLabelSet)
    assert saved_objects.labels is second


def test_cellprofiler_adapter_allows_measurements_for_source_bound_objects():
    source_bindings = StepSourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias=NUCLEI,
                artifact_kind=ObjectLabelsArtifactType,
                projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
            ),
        )
    )
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI_MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(
                    NUCLEI_MEASUREMENTS,
                    MeasurementsArtifactType,
                ),
            ),
        ),
        source_bindings=source_bindings,
    )
    rows = MeasurementSparseColumnarRows.from_rows(
        ({"object_id": 1, "area": 42.0},),
        fields=(FieldSpec("object_id", int), FieldSpec("area", float)),
    )

    adapter.add_measurements(
        MeasurementTable(
            name=NUCLEI_MEASUREMENTS,
            rows=rows,
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/plate/Images/A01_s001_w1_z001_t001.tif",),
            ),
        )
    )

    measurements = _output_measurements(adapter, NUCLEI_MEASUREMENTS)
    assert measurements.subject.object_name == NUCLEI
    assert measurements.source_image_provenance_planes.paths == (
        "/plate/Images/A01_s001_w1_z001_t001.tif",
    )


def test_cellprofiler_adapter_records_ungrouped_measurements_once():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=ArtifactOutputPlan(
                    name=MEASUREMENTS,
                    path="/memory/A01_Measurements.pkl",
                    artifact_type=MeasurementsArtifactType,
                ),
            ),
        ),
        filemanager=filemanager,
    )

    adapter.add_measurements(
        MeasurementTable(
            name=MEASUREMENTS,
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"image_area": 100.0},), fields=(FieldSpec("image_area", float),)
            ),
            subject=MeasurementSubject(
                MeasurementScope.IMAGE,
                MeasurementScope.IMAGE.value,
            ),
        )
    )

    records = store.find(
        name=MEASUREMENTS,
        artifact_type=MeasurementsArtifactType,
        axis_id=AXIS_ID,
        group_key=None,
        match_group=True,
    )

    assert len(records) == 1
    assert records[0].path == "/memory/A01_Measurements.pkl"


def test_cellprofiler_adapter_uses_static_output_scope():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    output_path = "/memory/A01_w3_Cytoplasm.pkl"
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        group_key=None,
        artifact_output_bindings=(
            _output_binding(
                CELLS,
                ObjectLabelsArtifactType,
                plan=ArtifactOutputPlan(
                    name=CELLS,
                    path=output_path,
                    artifact_type=ObjectLabelsArtifactType,
                    group_keys=("3",),
                    group_component=AllComponents.SITE,
                    paths_by_group={"3": output_path},
                ),
            ),
        ),
        filemanager=filemanager,
    )

    record = adapter.add_objects(
        CELLS,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=np.array([[1]], dtype=np.int32))
        ),
    )

    assert record.key.scope.value_text == "3"
    assert (
        len(
            store.find(
                name=CELLS,
                artifact_type=ObjectLabelsArtifactType,
                axis_id=AXIS_ID,
                group_key="3",
                match_group=True,
            )
        )
        == 1
    )
    assert (
        store.find(
            name=CELLS,
            artifact_type=ObjectLabelsArtifactType,
            axis_id=AXIS_ID,
            group_key="1",
            match_group=True,
        )
        == ()
    )


def test_cellprofiler_adapter_preserves_same_artifact_measurement_subjects():
    measurement_output_plan = _plan(MEASUREMENTS, MeasurementsArtifactType)
    adapter, _filemanager = _adapter(
        (
            _output_binding(NUCLEI, ObjectLabelsArtifactType),
            _output_binding(CELLS, ObjectLabelsArtifactType),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=measurement_output_plan,
            ),
        )
    )
    adapter.add_objects(
        NUCLEI,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=np.array([[1]], dtype=np.int32))
        ),
    )
    adapter.add_objects(
        CELLS,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=np.array([[1]], dtype=np.int32))
        ),
    )
    adapter.add_measurements(
        MeasurementTable(
            name=MEASUREMENTS,
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"object_name": NUCLEI, "object_label": 1, "area": 10.0},),
                fields=(
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("area", float),
                ),
            ),
            subject=MeasurementSubject(
                MeasurementScope.OBJECT,
                NUCLEI,
                MeasurementRowAxisField.OBJECT_LABEL.value,
            ),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/plate/Images/A01_s001_w1_z001_t001.tif",),
            ),
        )
    )
    adapter.add_measurements(
        MeasurementTable(
            name=MEASUREMENTS,
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"object_name": CELLS, "object_label": 1, "area": 20.0},),
                fields=(
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("area", float),
                ),
            ),
            subject=MeasurementSubject(
                MeasurementScope.OBJECT,
                CELLS,
                MeasurementRowAxisField.OBJECT_LABEL.value,
            ),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/plate/Images/A01_s001_w1_z001_t001.tif",),
            ),
        )
    )

    tables = tuple(
        record.value.data
        for record in adapter.artifact_output_records(measurement_output_plan)
    )

    assert tuple(table.subject for table in tables) == (
        MeasurementSubject(
            MeasurementScope.OBJECT,
            NUCLEI,
            MeasurementRowAxisField.OBJECT_LABEL.value,
        ),
        MeasurementSubject(
            MeasurementScope.OBJECT,
            CELLS,
            MeasurementRowAxisField.OBJECT_LABEL.value,
        ),
    )
    assert tuple(table.rows.row_mappings() for table in tables) == (
        ({"object_name": NUCLEI, "object_label": 1, "area": 10.0},),
        ({"object_name": CELLS, "object_label": 1, "area": 20.0},),
    )


def test_cellprofiler_adapter_does_not_select_measurement_record_from_current_source():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    object_group_paths = {
        "1": "/memory/A01_s001_w1_z001_t001_Nuclei.pkl",
        "2": "/memory/A01_s002_w1_z001_t001_Nuclei.pkl",
    }
    group_paths = {
        "1": "/memory/A01_s001_w1_z001_t001_NucleiMeasurements.pkl",
        "2": "/memory/A01_s002_w1_z001_t001_NucleiMeasurements.pkl",
    }
    for group_key, rows in (
        ("1", [{"object_id": 1, "area": 10.0}]),
        ("2", [{"object_id": 1, "area": 20.0}]),
    ):
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            group_key=group_key,
            artifact_output_bindings=(
                _output_binding(
                    NUCLEI,
                    ObjectLabelsArtifactType,
                    plan=ArtifactOutputPlan(
                        name=NUCLEI,
                        path=object_group_paths[group_key],
                        artifact_type=ObjectLabelsArtifactType,
                        group_keys=(group_key,),
                        group_component=AllComponents.SITE,
                        paths_by_group={group_key: object_group_paths[group_key]},
                    ),
                ),
                _output_binding(
                    NUCLEI_MEASUREMENTS,
                    MeasurementsArtifactType,
                    plan=ArtifactOutputPlan(
                        name=NUCLEI_MEASUREMENTS,
                        path=group_paths[group_key],
                        artifact_type=MeasurementsArtifactType,
                        group_keys=(group_key,),
                        group_component=AllComponents.SITE,
                        paths_by_group={group_key: group_paths[group_key]},
                    ),
                ),
            ),
            filemanager=filemanager,
        )
        producer.add_objects(
            NUCLEI,
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(
                    labels=np.array([[1]], dtype=np.int32)
                )
            ),
        )
        producer.add_measurements(
            MeasurementTable(
                name=NUCLEI_MEASUREMENTS,
                rows=MeasurementSparseColumnarRows.from_rows(
                    rows, fields=(FieldSpec("object_id", int), FieldSpec("area", float))
                ),
                subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=(f"/plate/Images/A01_s00{group_key}_w1_z001_t001.tif",),
                ),
            )
        )

    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=("/plate/Images/A01_s002_w1_z001_t001.tif",),
        pipeline_input_files=(
            "/plate/Images/A01_s001_w1_z001_t001.tif",
            "/plate/Images/A01_s002_w1_z001_t001.tif",
        ),
        current_step_input_files=("/plate/Images/A01_s002_w1_z001_t001.tif",),
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI_MEASUREMENTS,
                    path="/memory/NucleiMeasurements.pkl",
                    artifact_type=MeasurementsArtifactType,
                    group_keys=("1", "2"),
                    group_component=AllComponents.SITE,
                    paths_by_group=group_paths,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.from_raw(
                    ("1", "2"), component=AllComponents.SITE
                ),
                component_scopes=(
                    ComponentGroupScope.from_raw(
                        ("1", "2"), component=AllComponents.SITE
                    ),
                ),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        source_binding_context=source_binding_context,
        microscope_handler=(ContextStub(filemanager)).microscope_handler,
        filemanager=filemanager,
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    measurements = consumer.get_measurements(
        NUCLEI_MEASUREMENTS,
    )

    assert measurements.rows.row_mappings() == (
        {"object_id": 1, "area": 10.0},
        {"object_id": 1, "area": 20.0},
    )
    assert measurements.subject.object_name == NUCLEI


def test_cellprofiler_adapter_adds_measurements_after_object_reference_exists():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                NUCLEI_MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(
                    NUCLEI_MEASUREMENTS,
                    MeasurementsArtifactType,
                ),
            ),
        )
    )
    adapter.add_objects(
        NUCLEI,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32))
        ),
    )
    rows = MeasurementSparseColumnarRows.from_rows(
        ({"object_id": 1, "area": 42.0},),
        fields=(FieldSpec("object_id", int), FieldSpec("area", float)),
    )

    adapter.add_measurements(
        MeasurementTable(
            name=NUCLEI_MEASUREMENTS,
            rows=rows,
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI, "object_id"),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/plate/Images/A01_s001_w1_z001_t001.tif",),
            ),
        )
    )
    measurements = _output_measurements(adapter, NUCLEI_MEASUREMENTS)

    assert measurements.rows is rows
    assert measurements.subject.object_name == NUCLEI
    assert measurements.subject.object_id_field == "object_id"
    assert measurements.rows.fields == (
        FieldSpec("object_id", int),
        FieldSpec("area", float),
    )


def test_cellprofiler_adapter_uses_schema_owned_by_measurement_rows():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(
                    MEASUREMENTS,
                    MeasurementsArtifactType,
                ),
            ),
        )
    )

    adapter.add_measurements(
        MeasurementTable(
            name=MEASUREMENTS,
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"slice_index": 0, "area": 42.0},),
                fields=(FieldSpec("slice_index", int), FieldSpec("area", float)),
            ),
            subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        )
    )

    assert _output_measurements(adapter, MEASUREMENTS).rows.fields == (
        FieldSpec("slice_index", int),
        FieldSpec("area", float),
    )


def test_cellprofiler_adapter_does_not_list_undeclared_measurement_outputs():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                NUCLEI_MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(
                    NUCLEI_MEASUREMENTS,
                    MeasurementsArtifactType,
                ),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        )
    )
    adapter.add_objects(
        NUCLEI,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32))
        ),
    )
    rows = [{"slice_index": 0, "object_id": 1, "area": 42.0}]
    adapter.add_measurements(
        MeasurementTable(
            name=NUCLEI_MEASUREMENTS,
            rows=MeasurementSparseColumnarRows.from_rows(
                rows,
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_id", int),
                    FieldSpec("area", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/plate/Images/A01_s001_w1_z001_t001.tif",),
            ),
        )
    )
    adapter.add_measurements(
        MeasurementTable(
            name=MEASUREMENTS,
            rows=MeasurementSparseColumnarRows.from_rows(
                [{"slice_index": 0, "image_area": 100.0}],
                fields=(FieldSpec("slice_index", int), FieldSpec("image_area", float)),
            ),
            subject=MeasurementSubject(
                MeasurementScope.IMAGE,
                MeasurementScope.IMAGE.value,
            ),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/plate/Images/A01_s001_w1_z001_t001.tif",),
            ),
        )
    )

    tables = object_measurement_tables_for_test(adapter, NUCLEI)

    assert tables == ()


def test_cellprofiler_adapter_hides_undeclared_same_name_object_table_occurrence():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    source_binding_plan = _compiled_source_binding_plan(
        StepSourceBindingsConfig(bindings=(NamedSourceBinding(alias=DNA_IMAGE),))
    )
    output_plans = []
    produced_tables = []

    for index, value in enumerate((11.0, 13.0), start=1):
        output_plan = ArtifactOutputPlan(
            name=NUCLEI_MEASUREMENTS,
            path=f"/memory/{NUCLEI_MEASUREMENTS}_{index}.pkl",
            artifact_type=MeasurementsArtifactType,
        )
        output_plans.append(output_plan)
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            artifact_output_bindings=(
                _output_binding(
                    NUCLEI_MEASUREMENTS,
                    MeasurementsArtifactType,
                    plan=output_plan,
                ),
            ),
            source_binding_plan=source_binding_plan,
            filemanager=filemanager,
        )
        produced_tables.append(
            producer.add_measurements(
                MeasurementTable(
                    name=NUCLEI_MEASUREMENTS,
                    rows=MeasurementSparseColumnarRows.from_rows(
                        [
                            {
                                "slice_index": 0,
                                "object_label": 1,
                                "object_name": NUCLEI,
                                "area": value,
                            }
                        ],
                        fields=(
                            FieldSpec("slice_index", int),
                            FieldSpec("object_label", int),
                            FieldSpec("object_name", str),
                            FieldSpec("area", float),
                        ),
                    ),
                    subject=MeasurementSubject(
                        MeasurementScope.ARTIFACT,
                        NUCLEI_MEASUREMENTS,
                    ),
                )
            ).value.data
        )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI_MEASUREMENTS,
                    path=output_plans[0].path,
                    artifact_type=MeasurementsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        )
    }

    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        source_binding_plan=source_binding_plan,
        filemanager=filemanager,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    declared_tables = adapter.measurement_tables()
    object_tables = object_measurement_tables_for_test(adapter, NUCLEI)

    assert declared_tables == (produced_tables[0],)
    assert object_tables == (produced_tables[0],)
    assert object_tables[0] is not produced_tables[1]


def test_declared_measurement_inputs_require_producer_declared_slice_indexes():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    source_binding_plan = _compiled_source_binding_plan(
        StepSourceBindingsConfig(bindings=(NamedSourceBinding(alias=DNA_IMAGE),))
    )
    group_paths = {
        "1": f"/memory/{MEASUREMENTS}_1.pkl",
        "2": f"/memory/{MEASUREMENTS}_2.pkl",
    }
    output_plan = ArtifactOutputPlan(
        name=MEASUREMENTS,
        path=f"/memory/{MEASUREMENTS}.pkl",
        artifact_type=MeasurementsArtifactType,
        group_component=AllComponents.SITE,
        group_keys=tuple(group_paths),
        paths_by_group=group_paths,
    )
    for index, value in enumerate((11.0, 13.0), start=1):
        group_key = str(index)
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID, AllComponents.SITE, group_key),
            artifact_output_bindings=(
                _output_binding(
                    MEASUREMENTS,
                    MeasurementsArtifactType,
                    plan=output_plan,
                ),
            ),
            group_key=group_key,
            source_binding_plan=source_binding_plan,
            filemanager=filemanager,
        )
        producer.add_measurements(
            MeasurementTable(
                name=MEASUREMENTS,
                rows=MeasurementSparseColumnarRows.from_rows(
                    [
                        (
                            {"slice_index": 0, "area_occupied": value}
                            if index == 1
                            else {"area_occupied": value}
                        )
                    ],
                    fields=(
                        FieldSpec("slice_index", int, required=False),
                        FieldSpec("area_occupied", float),
                    ),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.IMAGE,
                    MeasurementScope.IMAGE.value,
                ),
            )
        )
    measurement_spec = ArtifactSpec.input(MEASUREMENTS, MeasurementsArtifactType)
    contract = _compiled_callable_contract(
        calculate_math,
        artifact_inputs=(measurement_spec,),
    )
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        callable_contract=contract,
        axis_scope=runtime_axis_scope(AXIS_ID, AllComponents.CHANNEL, "1"),
        artifact_inputs={
            edge.key: edge
            for edge in (
                cellprofiler_runtime_input_edge_for_test(
                    ArtifactInputPlan(
                        name=MEASUREMENTS,
                        path=output_plan.path,
                        artifact_type=MeasurementsArtifactType,
                        group_component=AllComponents.SITE,
                        group_keys=output_plan.group_keys,
                        paths_by_group=output_plan.paths_by_group,
                    ),
                    invocation_scope=ComponentGroupScope.from_raw(
                        ("1",), component=AllComponents.CHANNEL
                    ),
                    producer_selection_scope=ComponentGroupScope.from_raw(
                        output_plan.group_keys, component=AllComponents.SITE
                    ),
                    component_scopes=(
                        ComponentGroupScope.from_raw(
                            ("1",), component=AllComponents.CHANNEL
                        ),
                        ComponentGroupScope.from_raw(
                            output_plan.group_keys, component=AllComponents.SITE
                        ),
                    ),
                    consumer_variable_components=(AllComponents.SITE,),
                ),
            )
        },
        artifact_outputs={},
        variable_components=(VariableComponents.SITE,),
        source_binding_plan=source_binding_plan,
        filemanager=filemanager,
    )
    with pytest.raises(
        ValueError,
        match="mixes declared and axisless 'slice_index' row domains",
    ):
        RuntimeInputBindingRequest(
            adapter=consumer,
            kwargs={},
            current_image=np.zeros((2, 2), dtype=np.float32),
            selected_object_inputs=(),
        ).declared_measurement_tables()


def test_cellprofiler_adapter_aligns_multiplane_measurements_across_groups():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    outputs = {
        NUCLEI: _plan(
            NUCLEI, ObjectLabelsArtifactType, group_component=AllComponents.SITE
        ),
        NUCLEI_MEASUREMENTS: _plan(
            NUCLEI_MEASUREMENTS,
            MeasurementsArtifactType,
            group_component=AllComponents.SITE,
        ),
    }
    output_bindings = (
        _output_binding(
            NUCLEI,
            ObjectLabelsArtifactType,
            plan=outputs[NUCLEI],
        ),
        _output_binding(
            NUCLEI_MEASUREMENTS,
            MeasurementsArtifactType,
            plan=outputs[NUCLEI_MEASUREMENTS],
        ),
    )
    source_binding_plan = _compiled_source_binding_plan(
        StepSourceBindingsConfig(bindings=(NamedSourceBinding(alias=DNA_IMAGE),))
    )

    for group_key, slice_index, value in (
        ("site1", 0, 5.0),
        ("site2", 1, 7.0),
    ):
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            artifact_output_bindings=output_bindings,
            source_binding_plan=source_binding_plan,
            filemanager=filemanager,
            group_key=group_key,
        )
        producer.add_measurements(
            MeasurementTable(
                name=NUCLEI_MEASUREMENTS,
                rows=MeasurementSparseColumnarRows.from_rows(
                    [
                        {
                            "slice_index": slice_index,
                            "object_label": 1,
                            "mean_intensity": value,
                            "object_name": NUCLEI,
                        }
                    ],
                    fields=(
                        FieldSpec("slice_index", int),
                        FieldSpec("object_label", int),
                        FieldSpec("mean_intensity", float),
                        FieldSpec("object_name", str),
                    ),
                ),
                source_image_name="rawGFP",
                subject=MeasurementSubject(MeasurementScope.IMAGE, "rawGFP"),
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=(f"/plate/Images/A01_{group_key}_rawGFP.tif",),
                ),
            )
        )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI_MEASUREMENTS,
                    path=outputs[NUCLEI_MEASUREMENTS].path,
                    artifact_type=MeasurementsArtifactType,
                    group_component=AllComponents.SITE,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.dynamic(
                    AllComponents.SITE
                ),
                component_scopes=(ComponentGroupScope.dynamic(AllComponents.SITE),),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        artifact_outputs={},
        source_binding_plan=source_binding_plan,
        filemanager=filemanager,
        group_key="collapsed",
        plane_projection=RuntimePlaneProjection.stack(2),
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )
    labels = np.array([[[1]], [[1]]], dtype=np.int32)

    values = adapter_label_measurement_values(
        consumer,
        NUCLEI,
        "Intensity_MeanIntensity_rawGFP",
        labels,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (1,)),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    np.testing.assert_allclose(values[0], [5.0])
    np.testing.assert_allclose(values[1], [7.0])


def test_cellprofiler_adapter_rejects_single_slice_measurements_for_repeated_labels():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    measurement_output = _plan(NUCLEI_MEASUREMENTS, MeasurementsArtifactType)
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                NUCLEI_MEASUREMENTS,
                MeasurementsArtifactType,
                plan=measurement_output,
            ),
        ),
        filemanager=filemanager,
        plane_projection=RuntimePlaneProjection.stack(3),
    )
    producer.add_measurements(
        MeasurementTable(
            name=NUCLEI_MEASUREMENTS,
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": 0,
                        "object_label": 1,
                        "mean_intensity": 5.0,
                        "object_name": NUCLEI,
                    }
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_label", int),
                    FieldSpec("mean_intensity", float),
                    FieldSpec("object_name", str),
                ),
            ),
            source_image_name="rawGFP",
            subject=MeasurementSubject(MeasurementScope.IMAGE, "rawGFP"),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/plate/Images/A01_s001_rawGFP.tif",),
            ),
        )
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI_MEASUREMENTS,
                    path=measurement_output.path,
                    artifact_type=MeasurementsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        )
    }
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        artifact_outputs={},
        filemanager=filemanager,
        plane_projection=RuntimePlaneProjection.stack(3),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )
    labels = np.array([[[1]], [[1]], [[1]]], dtype=np.int32)

    with pytest.raises(ValueError, match="does not match the declared label domain"):
        adapter_label_measurement_values(
            consumer,
            NUCLEI,
            "Intensity_MeanIntensity_rawGFP",
            labels,
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1,), (1,), (1,)),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )


def test_measurement_lookup_uses_table_source_for_source_qualified_object_rows():
    values = measurement_values_for_feature(
        (
            MeasurementTable(
                name=MEASURE_OBJECT_INTENSITY,
                rows=MeasurementSparseColumnarRows.from_rows(
                    (
                        {
                            "object_name": NUCLEI,
                            "object_label": 1,
                            "mean_intensity": 9.0,
                        },
                    ),
                    fields=(
                        FieldSpec("object_name", str),
                        FieldSpec("object_label", int),
                        FieldSpec("mean_intensity", float),
                    ),
                ),
                source_image_name=DNA_IMAGE,
                subject=MeasurementSubject(MeasurementScope.IMAGE, DNA_IMAGE),
            ),
            MeasurementTable(
                name=MEASURE_OBJECT_INTENSITY,
                rows=MeasurementSparseColumnarRows.from_rows(
                    (
                        {
                            "object_name": NUCLEI,
                            "object_label": 1,
                            "mean_intensity": 5.0,
                        },
                    ),
                    fields=(
                        FieldSpec("object_name", str),
                        FieldSpec("object_label", int),
                        FieldSpec("mean_intensity", float),
                    ),
                ),
                source_image_name="rawGFP",
                subject=MeasurementSubject(MeasurementScope.IMAGE, "rawGFP"),
            ),
        ),
        "Intensity_MeanIntensity_rawGFP",
        object_count=1,
        object_ids=(1,),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [5.0])


def test_measurement_lookup_row_source_owns_columnar_source_domain():
    table = MeasurementTable(
        name=MEASURE_OBJECT_INTENSITY,
        rows=SimpleColumnarRows(
            {
                "object_name": (NUCLEI,),
                "object_label": (1,),
                "source_image_name": ("rowGFP",),
                "mean_intensity": (5.0,),
            },
            fields=(
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("source_image_name", str),
                FieldSpec("mean_intensity", float),
            ),
        ),
        source_image_name="rawGFP",
        subject=MeasurementSubject(MeasurementScope.IMAGE, "rawGFP"),
    )

    with pytest.raises(ValueError, match="Could not resolve measurement feature"):
        measurement_values_for_feature(
            (table,),
            "Intensity_MeanIntensity_rawGFP",
            object_count=1,
            object_ids=(1,),
            object_name=NUCLEI,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )

    values = measurement_values_for_feature(
        (table,),
        "Intensity_MeanIntensity_rowGFP",
        object_count=1,
        object_ids=(1,),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [5.0])


def test_source_qualified_object_feature_ignores_unrelated_runtime_slice_scope():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    measurement_output = _plan(MEASURE_OBJECT_INTENSITY, MeasurementsArtifactType)
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                MEASURE_OBJECT_INTENSITY,
                MeasurementsArtifactType,
                plan=measurement_output,
            ),
        ),
        filemanager=filemanager,
    )
    producer.add_measurements(
        MeasurementTable(
            name=MEASURE_OBJECT_INTENSITY,
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "mean_intensity": 5.0,
                    }
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("mean_intensity", float),
                ),
            ),
            source_image_name="rawGFP",
            subject=MeasurementSubject(MeasurementScope.IMAGE, "rawGFP"),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/plate/Images/A01_s001_rawGFP.tif",),
            ),
        )
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=MEASURE_OBJECT_INTENSITY,
                    path=measurement_output.path,
                    artifact_type=MeasurementsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        )
    }
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        artifact_outputs={},
        filemanager=filemanager,
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )
    context = ObjectFeatureMeasurementContext(
        object_name=NUCLEI,
        feature_name="Intensity_MeanIntensity_rawGFP",
        group_key=None,
        slice_index=2,
    )

    tables = context.measurement_tables(consumer, match_group=False)
    values = measurement_values_for_feature(
        tables,
        "Intensity_MeanIntensity_rawGFP",
        object_count=1,
        object_ids=(1,),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [5.0])


def test_measurement_lookup_normalizes_columnar_object_domain():
    values = measurement_values_for_feature(
        (
            MeasurementTable(
                name=MEASURE_OBJECT_INTENSITY,
                rows=SimpleColumnarRows(
                    {
                        "object_name": ("nuclei",),
                        "object_label": (1,),
                        "mean_intensity": (5.0,),
                    },
                    fields=(
                        FieldSpec("object_name", str),
                        FieldSpec("object_label", int),
                        FieldSpec("mean_intensity", float),
                    ),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT, MEASURE_OBJECT_INTENSITY
                ),
            ),
        ),
        "Intensity_MeanIntensity_rawGFP",
        object_count=1,
        object_ids=(1,),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [5.0])


def test_cellprofiler_adapter_multiplane_measurement_alignment_is_feature_scoped():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    outputs = {
        NUCLEI_MEASUREMENTS: _plan(
            NUCLEI_MEASUREMENTS,
            MeasurementsArtifactType,
            group_component=AllComponents.SITE,
        ),
    }
    output_bindings = (
        _output_binding(
            NUCLEI_MEASUREMENTS,
            MeasurementsArtifactType,
            plan=outputs[NUCLEI_MEASUREMENTS],
        ),
    )
    source_binding_plan = _compiled_source_binding_plan(
        StepSourceBindingsConfig(bindings=(NamedSourceBinding(alias=DNA_IMAGE),))
    )

    for group_key, slice_index, feature_value in (
        ("dna_site1", 0, 90.0),
        ("gfp_site1", 0, 5.0),
        ("dna_site2", 1, 95.0),
        ("gfp_site2", 1, 7.0),
    ):
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            artifact_output_bindings=output_bindings,
            source_binding_plan=source_binding_plan,
            filemanager=filemanager,
            group_key=group_key,
        )
        producer.add_measurements(
            MeasurementTable(
                name=NUCLEI_MEASUREMENTS,
                rows=MeasurementSparseColumnarRows.from_rows(
                    (
                        {
                            "slice_index": slice_index,
                            "object_name": NUCLEI,
                            "object_label": 1,
                            "mean_intensity": feature_value,
                        },
                    ),
                    fields=(
                        FieldSpec("slice_index", int),
                        FieldSpec("object_name", str),
                        FieldSpec("object_label", int),
                        FieldSpec("mean_intensity", float),
                    ),
                ),
                source_image_name=(
                    DNA_IMAGE if group_key.startswith("dna") else "rawGFP"
                ),
                subject=MeasurementSubject(
                    MeasurementScope.IMAGE,
                    DNA_IMAGE if group_key.startswith("dna") else "rawGFP",
                ),
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=(f"/plate/Images/A01_{group_key}.tif",),
                ),
            )
        )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI_MEASUREMENTS,
                    path=outputs[NUCLEI_MEASUREMENTS].path,
                    artifact_type=MeasurementsArtifactType,
                    group_component=AllComponents.SITE,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.dynamic(
                    AllComponents.SITE
                ),
                component_scopes=(ComponentGroupScope.dynamic(AllComponents.SITE),),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }

    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        artifact_outputs={},
        source_binding_plan=source_binding_plan,
        filemanager=filemanager,
        group_key="collapsed",
        plane_projection=RuntimePlaneProjection.stack(2),
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    values = adapter_label_measurement_values(
        consumer,
        NUCLEI,
        "Intensity_MeanIntensity_rawGFP",
        np.array([[[1]], [[1]]], dtype=np.int32),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (1,)),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    np.testing.assert_allclose(values[0], [5.0])
    np.testing.assert_allclose(values[1], [7.0])


def test_cellprofiler_adapter_measurement_query_cache_is_store_scoped():
    labels = np.array([[1, 2], [0, 0]], dtype=np.int32)
    consumers = []
    for values in ((20.0, 80.0), (4.0, 12.0)):
        store = RuntimeValueStore()
        filemanager = FileManagerStub()
        object_output = _plan(NUCLEI, ObjectLabelsArtifactType)
        measurement_output = _plan(NUCLEI_MEASUREMENTS, MeasurementsArtifactType)
        producer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            axis_scope=runtime_axis_scope(AXIS_ID),
            artifact_output_bindings=(
                _output_binding(
                    NUCLEI,
                    ObjectLabelsArtifactType,
                    plan=object_output,
                ),
                _output_binding(
                    NUCLEI_MEASUREMENTS,
                    MeasurementsArtifactType,
                    plan=measurement_output,
                ),
            ),
            filemanager=filemanager,
        )
        producer.add_objects(
            NUCLEI,
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(labels=labels),
                domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
                plane_axis=None,
            ),
        )
        producer.add_measurements(
            MeasurementTable(
                name=NUCLEI_MEASUREMENTS,
                rows=MeasurementSparseColumnarRows.from_rows(
                    [
                        {
                            "slice_index": 0,
                            "object_name": NUCLEI,
                            "object_label": object_label,
                            "feature_name": "AreaShape_Area",
                            "result_value": value,
                        }
                        for object_label, value in enumerate(values, start=1)
                    ],
                    fields=(
                        FieldSpec("slice_index", int),
                        FieldSpec("object_name", str),
                        FieldSpec("object_label", int),
                        FieldSpec("feature_name", str),
                        FieldSpec("result_value", float),
                    ),
                ),
                subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=("/plate/Images/A01_s001_w1_z001_t001.tif",),
                ),
            )
        )
        _compiled_artifact_inputs = {
            edge.key: edge
            for edge in (
                cellprofiler_runtime_input_edge_for_test(
                    ArtifactInputPlan(
                        name=NUCLEI_MEASUREMENTS,
                        path=measurement_output.path,
                        artifact_type=MeasurementsArtifactType,
                    ),
                    invocation_scope=ComponentGroupScope.ungrouped(),
                    producer_selection_scope=ComponentGroupScope.ungrouped(),
                    component_scopes=(),
                    consumer_variable_components=(),
                ),
            )
        }
        consumers.append(
            cellprofiler_runtime_adapter_for_test(
                runtime_value_store=store,
                axis_scope=runtime_axis_scope(AXIS_ID),
                artifact_inputs=_compiled_artifact_inputs,
                artifact_outputs={},
                filemanager=filemanager,
                callable_contract=_compiled_callable_contract(
                    calculate_math,
                    artifact_inputs=tuple(
                        edge.spec for edge in _compiled_artifact_inputs.values()
                    ),
                ),
            )
        )
    first, second = consumers

    np.testing.assert_allclose(
        adapter_label_measurement_values(
            first,
            NUCLEI,
            "AreaShape_Area",
            labels[np.newaxis, ...],
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1, 2),),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )[0],
        [20.0, 80.0],
    )
    np.testing.assert_allclose(
        adapter_label_measurement_values(
            second,
            NUCLEI,
            "AreaShape_Area",
            labels[np.newaxis, ...],
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1, 2),),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )[0],
        [4.0, 12.0],
    )


def test_label_measurement_cache_distinguishes_single_plane_from_stack() -> None:
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    object_output = _plan(NUCLEI, ObjectLabelsArtifactType)
    single_measurement_name = f"{NUCLEI_MEASUREMENTS}SinglePlane"
    single_measurement_output = _plan(
        single_measurement_name,
        MeasurementsArtifactType,
    )
    measurement_output = _plan(NUCLEI_MEASUREMENTS, MeasurementsArtifactType)
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(NUCLEI, ObjectLabelsArtifactType, plan=object_output),
            _output_binding(
                single_measurement_name,
                MeasurementsArtifactType,
                plan=single_measurement_output,
            ),
            _output_binding(
                NUCLEI_MEASUREMENTS,
                MeasurementsArtifactType,
                plan=measurement_output,
            ),
        ),
        filemanager=filemanager,
    )
    producer.add_objects(
        NUCLEI,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(
                labels=np.array([[1, 2]], dtype=np.int32)
            ),
            domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
            plane_axis=None,
        ),
    )
    producer.add_measurements(
        MeasurementTable(
            name=single_measurement_name,
            rows=MeasurementSparseColumnarRows.from_rows(
                (
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "feature_name": "AreaShape_Area",
                        "result_value": 10.0,
                    },
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 2,
                        "feature_name": "AreaShape_Area",
                        "result_value": 20.0,
                    },
                ),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("feature_name", str),
                    FieldSpec("result_value", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/plate/Images/A01_s001_w1_z001_t001.tif",),
            ),
        )
    )
    producer.add_measurements(
        MeasurementTable(
            name=NUCLEI_MEASUREMENTS,
            rows=MeasurementSparseColumnarRows.from_rows(
                (
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "feature_name": "AreaShape_Area",
                        "result_value": 10.0,
                    },
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 2,
                        "feature_name": "AreaShape_Area",
                        "result_value": 20.0,
                    },
                    {
                        "slice_index": 1,
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "feature_name": "AreaShape_Area",
                        "result_value": 30.0,
                    },
                ),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("feature_name", str),
                    FieldSpec("result_value", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/plate/Images/A01_s001_w1_z001_t001.tif",
                    "/plate/Images/A01_s002_w1_z001_t001.tif",
                ),
            ),
        )
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=single_measurement_name,
                    path=single_measurement_output.path,
                    artifact_type=MeasurementsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        )
    }
    single_plane_adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        artifact_outputs={},
        filemanager=filemanager,
        plane_projection=RuntimePlaneProjection.stack(1),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI_MEASUREMENTS,
                    path=measurement_output.path,
                    artifact_type=MeasurementsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        )
    }
    stack_adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        artifact_outputs={},
        filemanager=filemanager,
        plane_projection=RuntimePlaneProjection.stack(2),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )
    first_plane = np.array([[[1, 2]]], dtype=np.int32)
    label_stack = np.array(([[1, 2]], [[1, 0]]), dtype=np.int32)
    single_plane_values = adapter_label_measurement_values(
        single_plane_adapter,
        NUCLEI,
        "AreaShape_Area",
        first_plane,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1, 2),),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    stack_values = adapter_label_measurement_values(
        stack_adapter,
        NUCLEI,
        "AreaShape_Area",
        label_stack,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1, 2), (1,)),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    np.testing.assert_allclose(single_plane_values[0], [10.0, 20.0])
    assert len(stack_values) == 2
    np.testing.assert_allclose(stack_values[0], [10.0, 20.0])
    np.testing.assert_allclose(stack_values[1], [30.0])


def test_cellprofiler_adapter_projects_duplicate_object_labels_to_current_runtime_slice():
    filemanager = FileManagerStub()
    nuclei_output = ArtifactSpec.output(NUCLEI, ObjectLabelsArtifactType)
    measurement_output = ArtifactSpec.output(
        NUCLEI_MEASUREMENTS,
        MeasurementsArtifactType,
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=NUCLEI_MEASUREMENTS,
                    path=f"/memory/{NUCLEI_MEASUREMENTS}.pkl",
                    artifact_type=MeasurementsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        )
    }
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        artifact_output_bindings=(
            _output_binding_for_spec(nuclei_output),
            _output_binding_for_spec(measurement_output),
        ),
        source_binding_plan=_compiled_source_binding_plan(
            StepSourceBindingsConfig(bindings=(NamedSourceBinding(alias=DNA_IMAGE),))
        ),
        filemanager=filemanager,
        plane_projection=RuntimePlaneProjection.selected(1, 2),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
            artifact_outputs=(nuclei_output, measurement_output),
        ),
    )
    labels = np.array([[1, 2], [0, 0]], dtype=np.int32)
    adapter.add_objects(
        NUCLEI, ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels))
    )
    adapter.add_measurements(
        MeasurementTable(
            name=NUCLEI_MEASUREMENTS,
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": slice_index,
                        "object_name": NUCLEI,
                        "object_label": object_label,
                        "feature_name": "AreaShape_Area",
                        "result_value": value,
                    }
                    for slice_index, values in enumerate(
                        ((100.0, 200.0), (500.0, 600.0))
                    )
                    for object_label, value in enumerate(values, start=1)
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("feature_name", str),
                    FieldSpec("result_value", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        )
    )

    value_slices = adapter_label_measurement_values(
        adapter,
        NUCLEI,
        "AreaShape_Area",
        labels,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PAYLOAD,
            declared_object_ids=(1, 2),
        ),
        plane_axis=None,
    )

    assert len(value_slices) == 1
    np.testing.assert_allclose(value_slices[0], [500.0, 600.0])


def test_cellprofiler_adapter_adds_relationships_after_objects_exist():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                CELLS,
                ObjectLabelsArtifactType,
                plan=_plan(CELLS, ObjectLabelsArtifactType),
            ),
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                PARENT_CHILD,
                RelationshipsArtifactType,
                plan=_plan(PARENT_CHILD, RelationshipsArtifactType),
            ),
        )
    )
    adapter.add_objects(
        CELLS,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32))
        ),
    )
    adapter.add_objects(
        NUCLEI,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32))
        ),
    )

    adapter.add_relationship(
        _parent_child_relationship(
            name=PARENT_CHILD,
            parent_object_name=CELLS,
            child_object_name=NUCLEI,
            payload=DirectedObjectRelationshipPayload(
                source_ids=(10, 11),
                target_ids=(1, 2),
            ),
        )
    )
    relationship = _output_relationship(adapter, PARENT_CHILD)

    assert relationship.declaration.source.name == CELLS
    assert relationship.declaration.target.name == NUCLEI
    assert relationship.payload.source_ids == (10, 11)
    assert relationship.payload.target_ids == (1, 2)


def test_cellprofiler_adapter_write_requires_compiled_output_plan():
    output_spec = ArtifactSpec.output(NUCLEI, ObjectLabelsArtifactType)
    adapter, _filemanager = _adapter(
        (),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_outputs=(output_spec,),
        ),
    )

    with pytest.raises(RuntimeError, match="no selected artifact output plan"):
        adapter.add_objects(
            NUCLEI,
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(
                    labels=np.zeros((2, 2), dtype=np.int32)
                )
            ),
        )


def test_cellprofiler_relationship_write_requires_compiled_output_plan(
    monkeypatch: pytest.MonkeyPatch,
):
    output_spec = ArtifactSpec.output(PARENT_CHILD, RelationshipsArtifactType)
    adapter, _filemanager = _adapter(
        (),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_outputs=(output_spec,),
        ),
    )
    relationship = _parent_child_relationship(
        name=PARENT_CHILD,
        parent_object_name=CELLS,
        child_object_name=NUCLEI,
        payload=DirectedObjectRelationshipPayload(
            source_ids=(1,),
            target_ids=(1,),
        ),
    )

    def fail_endpoint_fallback(*_args, **_kwargs):
        pytest.fail("relationship writes must not probe endpoint availability")

    monkeypatch.setattr(
        CellProfilerRuntimeAdapter,
        "require_artifact_available",
        fail_endpoint_fallback,
    )

    with pytest.raises(
        RuntimeError,
        match=r"no selected artifact output plan.*ParentChild",
    ):
        adapter.add_relationship(relationship)


def test_cellprofiler_adapter_write_rejects_undeclared_output_kind():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                MeasurementsArtifactType,
                plan=_plan(NUCLEI, MeasurementsArtifactType),
            ),
        )
    )

    with pytest.raises(ValueError, match="No object_labels artifact named 'Nuclei'"):
        adapter.add_objects(
            NUCLEI,
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(
                    labels=np.zeros((2, 2), dtype=np.int32)
                )
            ),
        )


def test_cellprofiler_adapter_write_requires_filemanager_vfs_boundary():
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="filemanager is required for writes"):
        adapter.add_objects(
            NUCLEI,
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(
                    labels=np.zeros((2, 2), dtype=np.int32)
                )
            ),
        )


def test_cellprofiler_adapter_measurements_require_object_reference():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI_MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(
                    NUCLEI_MEASUREMENTS,
                    MeasurementsArtifactType,
                ),
            ),
        )
    )
    adapter.request = replace(
        adapter.request,
        callable_contract=_compiled_callable_contract(calculate_math),
    )

    with pytest.raises(
        ValueError,
        match="No object_labels artifact named 'Nuclei' is declared",
    ):
        adapter.add_measurements(
            MeasurementTable(
                name=NUCLEI_MEASUREMENTS,
                rows=MeasurementSparseColumnarRows.from_rows(
                    [{"object_id": 1}], fields=(FieldSpec("object_id", int),)
                ),
                subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
            )
        )


def test_cellprofiler_module_executor_records_and_publishes_object_output(
    declaration_owned_cellprofiler_callable,
):
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
        ),
        plane_projection=RuntimePlaneProjection.stack(),
    )
    image = np.zeros((2, 2), dtype=np.float32)
    labels = np.ones((2, 2), dtype=np.int32)

    @declared_processing_contract(ProcessingContract.PURE_2D)
    def identify_primary_objects(image_arg, *, min_diameter):
        np.testing.assert_array_equal(image_payload_data(image_arg), image)
        assert min_diameter == 8
        return ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        )

    executor = _executor(
        declaration_owned_cellprofiler_callable(identify_primary_objects),
        adapter,
        (
            ArtifactSpec.output_inheriting_group_scope(
                NUCLEI,
                ObjectLabelsArtifactType,
                ArtifactSpec.input(DNA_IMAGE, ImageArtifactType),
            ),
        ),
    )
    result = executor(
        _source_image_payload(image),
        cellprofiler_runtime=adapter,
        min_diameter=8,
    )

    np.testing.assert_array_equal(object_label_dense_array(result), labels)
    np.testing.assert_array_equal(_output_objects(adapter, NUCLEI).labels, labels)


@pytest.mark.parametrize(
    "module_name",
    [
        IDENTIFY_PRIMARY_OBJECTS,
        IDENTIFY_SECONDARY_OBJECTS,
        IDENTIFY_TERTIARY_OBJECTS,
        MEASURE_OBJECT_INTENSITY,
        MEASURE_OBJECT_NEIGHBORS,
        MEASURE_OBJECT_SIZE_SHAPE,
        MEASURE_IMAGE_INTENSITY,
    ],
)
def test_core_cellprofiler_functions_resolve_with_numpy_memory_contract(module_name):
    func = CellProfilerModule.require_module(module_name).require_callable()

    assert callable(func)
    assert func.input_memory_type == "numpy"
    assert func.output_memory_type == "numpy"


def test_cellprofiler_module_executor_runs_resolved_identify_primary_objects():
    adapter, filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        )
    )
    image = np.zeros((64, 64), dtype=np.float32)
    image[18:28, 18:28] = 0.95
    image[40:50, 40:50] = 0.85
    identify_primary_objects = CellProfilerModule.require_module(
        IDENTIFY_PRIMARY_OBJECTS
    ).require_callable()
    executor = _executor(
        identify_primary_objects,
        adapter,
        (
            ArtifactSpec.output(MEASUREMENTS, MeasurementsArtifactType),
            ArtifactSpec.output_inheriting_group_scope(
                NUCLEI,
                ObjectLabelsArtifactType,
                ArtifactSpec.input(DNA_IMAGE, ImageArtifactType),
            ),
        ),
    )

    result = executor(
        _source_image_payload(image),
        cellprofiler_runtime=adapter,
        dtype_config=DtypeConfig(),
        min_diameter=4,
        max_diameter=20,
        exclude_border_objects=False,
    )

    objects = _output_objects(adapter, NUCLEI)
    assert result.shape == image.shape
    assert objects.labels.shape == image.shape
    assert objects.labels.max() == 2
    saved_objects = filemanager.saved[("memory", "/memory/Nuclei.pkl")]
    assert isinstance(saved_objects, ObjectLabelSet)
    assert saved_objects.labels.shape == image.shape


def test_cellprofiler_module_executor_reads_objects_for_measurements(
    declaration_owned_cellprofiler_callable,
):
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                NUCLEI_MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(
                    NUCLEI_MEASUREMENTS,
                    MeasurementsArtifactType,
                ),
            ),
        ),
        plane_projection=RuntimePlaneProjection.stack(),
    )
    image = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/src/DNA.tif",),
        ),
        source_image_names=(DNA_IMAGE,),
    ).payload_with(np.zeros((2, 2), dtype=np.float32), None)
    labels_array = np.array([[1, 0], [0, 0]], dtype=np.int32)
    rows = DataclassMeasurementColumnarRows(
        (AreaMeasurementRow(object_id=1, Area=12.0),),
        row_type=AreaMeasurementRow,
    )
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=labels_array),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
    )

    @declared_processing_contract(ProcessingContract.FLEXIBLE)
    def measure_object_size_shape(
        image_arg, *, labels: ObjectLabelValue
    ) -> tuple[object, DataclassMeasurementColumnarRows]:
        np.testing.assert_array_equal(
            image_payload_data(image_arg), image_payload_data(image)
        )
        np.testing.assert_array_equal(object_label_dense_array(labels), labels_array)
        return image_arg, rows

    executor = _executor(
        declaration_owned_cellprofiler_callable(measure_object_size_shape),
        adapter,
        (ArtifactSpec.output(NUCLEI_MEASUREMENTS, MeasurementsArtifactType),),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
        ),
    )
    executor(image, cellprofiler_runtime=adapter)
    measurements = _output_measurements(adapter, NUCLEI_MEASUREMENTS)

    assert _measurement_rows_for_assertion(measurements) == [
        {"object_id": 1, "AreaShape_Area": 12.0},
    ]
    assert measurements.subject.object_name == NUCLEI
    assert measurements.source_image_name is None


def test_cellprofiler_object_only_measurement_uses_label_domain_reference_image(
    declaration_owned_cellprofiler_callable,
):
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                NUCLEI_MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(
                    NUCLEI_MEASUREMENTS,
                    MeasurementsArtifactType,
                ),
            ),
        ),
        plane_projection=RuntimePlaneProjection.stack(),
    )
    image = np.zeros((1006, 1000), dtype=np.float32)
    label_pixels = np.ones((199, 199), dtype=np.int32)
    rows = DataclassMeasurementColumnarRows(
        (
            AreaMeasurementRow(
                object_id=1,
                Area=float(label_pixels.size),
            ),
        ),
        row_type=AreaMeasurementRow,
    )
    seen = []
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=label_pixels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        ),
    )

    @object_label_input_execution_mode(ObjectLabelInputExecutionMode.SLICE_ALIGNED)
    @declared_processing_contract(ProcessingContract.PURE_2D)
    def measure_object_size_shape(
        image_arg, *, labels: ObjectLabelValue
    ) -> tuple[object, DataclassMeasurementColumnarRows]:
        image_pixels = np.asarray(image_payload_data(image_arg))
        seen.append((image_pixels.copy(), object_label_dense_array(labels).copy()))
        return image_arg, rows

    executor = _executor(
        declaration_owned_cellprofiler_callable(measure_object_size_shape),
        adapter,
        (ArtifactSpec.output(NUCLEI_MEASUREMENTS, MeasurementsArtifactType),),
        main_flow_inputs=(),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
        ),
    )
    executor(image, cellprofiler_runtime=adapter)
    measurements = _output_measurements(adapter, NUCLEI_MEASUREMENTS)

    assert len(seen) == 1
    measurement_image, measurement_labels = seen[0]
    assert measurement_image.shape == label_pixels.shape
    assert measurement_image.dtype == image.dtype
    np.testing.assert_array_equal(
        measurement_image, np.zeros_like(label_pixels, dtype=image.dtype)
    )
    np.testing.assert_array_equal(measurement_labels, label_pixels)
    assert _measurement_rows_for_assertion(measurements) == [
        {"object_id": 1, "AreaShape_Area": float(label_pixels.size)},
    ]
    assert measurements.subject.object_name == NUCLEI


def test_cellprofiler_object_only_pure_2d_module_executes_label_runtime_slices(
    declaration_owned_cellprofiler_callable,
):
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                NUCLEI_MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(
                    NUCLEI_MEASUREMENTS,
                    MeasurementsArtifactType,
                ),
            ),
        ),
        plane_projection=RuntimePlaneProjection.stack(2),
    )
    label_planes = np.stack(
        (
            np.full((5, 6), 1, dtype=np.int32),
            np.full((5, 6), 2, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name=NUCLEI,
        variant_data=ObjectLabelVariantData(labels=label_planes),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/src/Nuclei_t0.tif", "/src/Nuclei_t1.tif"),
        ),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (2,)),
        ),
    )
    image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.zeros((2, 20, 21), dtype=np.float32), None)
    seen: list[tuple[tuple[int, ...], int]] = []
    adapter.add_objects(NUCLEI, labels)

    @object_label_input_execution_mode(ObjectLabelInputExecutionMode.SLICE_ALIGNED)
    @runtime_bound_parameters(SliceIndexRuntimeParameter)
    @declared_processing_contract(ProcessingContract.PURE_2D)
    def measure_object_size_shape(
        image_arg, *, labels: ObjectLabelValue, slice_index: int | None = None
    ) -> tuple[object, DataclassMeasurementColumnarRows]:
        image_pixels = np.asarray(image_payload_data(image_arg))
        label_pixels = object_label_dense_array(labels)
        seen.append((tuple(image_pixels.shape), int(label_pixels.max())))
        assert slice_index is not None
        return image_arg, DataclassMeasurementColumnarRows(
            (
                SliceAreaMeasurementRow(
                    object_id=int(label_pixels.max()),
                    slice_index=slice_index,
                    Area=float(np.count_nonzero(label_pixels)),
                ),
            ),
            row_type=SliceAreaMeasurementRow,
        )

    executor = _executor(
        declaration_owned_cellprofiler_callable(measure_object_size_shape),
        adapter,
        (ArtifactSpec.output(NUCLEI_MEASUREMENTS, MeasurementsArtifactType),),
        main_flow_inputs=(),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
        ),
    )
    executor(image, cellprofiler_runtime=adapter)
    measurements = _output_measurements(adapter, NUCLEI_MEASUREMENTS)

    assert seen == [((5, 6), 1), ((5, 6), 2)]
    assert _measurement_rows_for_assertion(measurements) == [
        {
            "object_id": 1,
            "AreaShape_Area": float(label_planes[0].size),
            "slice_index": 0,
        },
        {
            "object_id": 2,
            "AreaShape_Area": float(label_planes[1].size),
            "slice_index": 1,
        },
    ]
    assert measurements.subject.object_name == NUCLEI


def test_cellprofiler_object_only_full_stack_measurement_preserves_label_runtime_slices(
    declaration_owned_cellprofiler_callable,
):
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                NUCLEI_MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(
                    NUCLEI_MEASUREMENTS,
                    MeasurementsArtifactType,
                ),
            ),
        ),
        plane_projection=RuntimePlaneProjection.stack(2),
    )
    label_planes = np.stack(
        (
            np.full((5, 6), 1, dtype=np.int32),
            np.full((5, 6), 2, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name=NUCLEI,
        variant_data=ObjectLabelVariantData(labels=label_planes),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (2,)),
        ),
    )
    image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.zeros((2, 20, 21), dtype=np.float32), None)
    seen: list[tuple[int, ...]] = []
    adapter.add_objects(NUCLEI, labels)

    @object_label_input_execution_mode(ObjectLabelInputExecutionMode.FULL_STACK)
    @declared_processing_contract(ProcessingContract.FLEXIBLE)
    def measure_object_size_shape(
        image_arg, *, labels: ObjectLabelValue
    ) -> tuple[object, DataclassMeasurementColumnarRows]:
        seen.append(tuple(object_label_dense_array(labels).shape))
        return image_arg, DataclassMeasurementColumnarRows(
            (
                SliceAreaMeasurementRow(
                    object_id=1,
                    slice_index=0,
                    Area=float(label_planes[0].size),
                ),
                SliceAreaMeasurementRow(
                    object_id=2,
                    slice_index=1,
                    Area=float(label_planes[1].size),
                ),
            ),
            row_type=SliceAreaMeasurementRow,
        )

    executor = _executor(
        declaration_owned_cellprofiler_callable(measure_object_size_shape),
        adapter,
        (ArtifactSpec.output(NUCLEI_MEASUREMENTS, MeasurementsArtifactType),),
        main_flow_inputs=(),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
        ),
    )

    executor(image, cellprofiler_runtime=adapter)
    measurements = _output_measurements(adapter, NUCLEI_MEASUREMENTS)

    assert seen == [label_planes.shape]
    assert _measurement_rows_for_assertion(measurements) == [
        {
            "object_id": 1,
            "slice_index": 0,
            "AreaShape_Area": float(label_planes[0].size),
        },
        {
            "object_id": 2,
            "slice_index": 1,
            "AreaShape_Area": float(label_planes[1].size),
        },
    ]


def test_cellprofiler_module_executor_measures_each_declared_image_for_single_object(
    declaration_owned_cellprofiler_callable,
):
    dna = np.full((4, 5), 3.0, dtype=np.float32)
    ph3 = np.full((4, 5), 9.0, dtype=np.float32)
    nuclei = np.ones((4, 5), dtype=np.int32)
    adapter = _source_bound_image_adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        ),
        {DNA_IMAGE: dna, "PH3": ph3},
    )
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=nuclei),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(0, 0),
                source_shape_yx=nuclei.shape,
            ),
        ),
    )
    seen = []

    @declared_processing_contract(ProcessingContract.PURE_2D)
    def measure_object_intensity(
        image_arg, *, labels: ObjectLabelValue
    ) -> tuple[object, DataclassMeasurementColumnarRows]:
        image_pixels = image_payload_data(image_arg)
        label_pixels = object_label_dense_array(labels)
        seen.append((float(image_pixels.mean()), int(label_pixels.max())))
        return image_pixels, DataclassMeasurementColumnarRows(
            (
                IntensityMeasurementRow(
                    mean_intensity=float(image_pixels.mean()),
                    object_label=int(label_pixels.max()),
                ),
            ),
            row_type=IntensityMeasurementRow,
        )

    executor = _executor(
        declaration_owned_cellprofiler_callable(measure_object_intensity),
        adapter,
        (ArtifactSpec.output(MEASUREMENTS, MeasurementsArtifactType),),
        main_flow_inputs=(
            ArtifactSpec.input(DNA_IMAGE, ImageArtifactType),
            ArtifactSpec.input("PH3", ImageArtifactType),
        ),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
        ),
    )
    source_stack = _source_bound_image_stack({DNA_IMAGE: dna, "PH3": ph3})
    result = executor(
        source_stack,
        cellprofiler_runtime=adapter,
    )
    measurements = _output_measurements(adapter, MEASUREMENTS)

    np.testing.assert_array_equal(result, np.stack((dna, ph3)))
    assert seen == [(3.0, 1), (9.0, 1)]
    assert _measurement_rows_for_assertion(measurements) == [
        {
            "Intensity_MeanIntensity_DNA": 3.0,
            "object_label": 1,
            "object_name": NUCLEI,
            "source_image_name": DNA_IMAGE,
        },
        {
            "Intensity_MeanIntensity_PH3": 9.0,
            "object_label": 1,
            "object_name": NUCLEI,
            "source_image_name": "PH3",
        },
    ]
    assert measurements.subject.object_name == NUCLEI
    assert measurements.source_image_name is None


def test_cellprofiler_module_executor_keeps_coupled_measurement_images_composed(
    declaration_owned_cellprofiler_callable,
):
    dna = np.full((4, 5), 3.0, dtype=np.float32)
    ph3 = np.full((4, 5), 9.0, dtype=np.float32)
    nuclei = np.ones((4, 5), dtype=np.int32)
    adapter = _source_bound_image_adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        ),
        {DNA_IMAGE: dna, "PH3": ph3},
    )
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=nuclei),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(0, 0),
                source_shape_yx=nuclei.shape,
            ),
        ),
    )
    seen = []

    @declared_processing_contract(ProcessingContract.PURE_2D)
    @runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
    @composed_image_payload
    def measure_colocalization(
        image_arg, *, labels: ObjectLabelValue
    ) -> tuple[object, ColumnarRows]:
        image_pixels = image_payload_data(image_arg)
        label_pixels = object_label_dense_array(labels)
        seen.append((image_pixels.shape, label_pixels.shape))
        return image_pixels[0], ObjectColocalizationMetricArrays.empty(1).rows_for(
            np.asarray((int(label_pixels.max()),), dtype=np.int32)
        )

    executor = _executor(
        declaration_owned_cellprofiler_callable(measure_colocalization),
        adapter,
        (ArtifactSpec.output(MEASUREMENTS, MeasurementsArtifactType),),
        main_flow_inputs=(
            ArtifactSpec.input(DNA_IMAGE, ImageArtifactType),
            ArtifactSpec.input("PH3", ImageArtifactType),
        ),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
        ),
    )
    result = executor(
        _source_bound_image_stack({DNA_IMAGE: dna, "PH3": ph3}),
        cellprofiler_runtime=adapter,
    )
    measurements = _output_measurements(adapter, MEASUREMENTS)

    np.testing.assert_array_equal(result, np.stack((dna, ph3)))
    assert seen == [((2, 4, 5), (4, 5))]
    assert _measurement_rows_for_assertion(measurements) == [
        {
            "slice_index": 0,
            "object_label": 1,
            "Correlation_Correlation_DNA_PH3": 0.0,
            "Correlation_Overlap_DNA_PH3": 0.0,
            "Correlation_K_DNA_PH3": 0.0,
            "Correlation_K_PH3_DNA": 0.0,
            "Correlation_Manders_DNA_PH3": 0.0,
            "Correlation_Manders_PH3_DNA": 0.0,
            "Correlation_RWC_DNA_PH3": 0.0,
            "Correlation_RWC_PH3_DNA": 0.0,
            "Correlation_Costes_DNA_PH3": 0.0,
            "Correlation_Costes_PH3_DNA": 0.0,
        }
    ]
    assert measurements.subject.object_name == NUCLEI
    assert measurements.source_image_name == f"{DNA_IMAGE}__PH3"


def test_cellprofiler_module_executor_combines_multi_object_measurements(
    declaration_owned_cellprofiler_callable,
):
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                CELLS,
                ObjectLabelsArtifactType,
                plan=_plan(CELLS, ObjectLabelsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        ),
        plane_projection=RuntimePlaneProjection.stack(),
    )
    image = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/src/DNA.tif",),
        ),
        source_image_names=(DNA_IMAGE,),
    ).payload_with(np.zeros((2, 2), dtype=np.float32), None)
    nuclei = np.array([[1, 0], [0, 0]], dtype=np.int32)
    cells = np.array([[0, 0], [0, 1]], dtype=np.int32)
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=nuclei),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
    )
    adapter.add_objects(
        CELLS,
        ObjectLabelSet(
            name=CELLS,
            variant_data=ObjectLabelVariantData(labels=cells),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
    )

    @declared_processing_contract(ProcessingContract.PURE_2D)
    def measure_object_size_shape(
        image_arg, *, labels: ObjectLabelValue
    ) -> tuple[object, DataclassMeasurementColumnarRows]:
        if labels.name == NUCLEI:
            return image_arg, DataclassMeasurementColumnarRows(
                (AreaMeasurementRow(1, 1.0),),
                row_type=AreaMeasurementRow,
            )
        if labels.name == CELLS:
            return image_arg, DataclassMeasurementColumnarRows(
                (AreaMeasurementRow(1, 1.0),),
                row_type=AreaMeasurementRow,
            )
        raise AssertionError("unexpected labels")

    executor = _executor(
        declaration_owned_cellprofiler_callable(measure_object_size_shape),
        adapter,
        (ArtifactSpec.output(MEASUREMENTS, MeasurementsArtifactType),),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
            ArtifactSpec.input(
                CELLS, ObjectLabelsArtifactType, parameter_name="labels"
            ),
        ),
    )
    executor(image, cellprofiler_runtime=adapter)
    measurements = _output_measurements(adapter, MEASUREMENTS)

    assert _measurement_rows_for_assertion(measurements) == [
        {
            "object_id": 1,
            "AreaShape_Area": 1.0,
            "object_name": NUCLEI,
        },
        {
            "object_id": 1,
            "AreaShape_Area": 1.0,
            "object_name": CELLS,
        },
    ]
    assert measurements.subject.object_name is None
    assert measurements.source_image_name is None
    assert object_measurement_tables_for_test(adapter, NUCLEI) == ()
    assert object_measurement_tables_for_test(adapter, CELLS) == ()


def test_measurement_lookup_filters_mixed_object_measurement_rows():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                CELLS,
                ObjectLabelsArtifactType,
                plan=_plan(CELLS, ObjectLabelsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        )
    )
    adapter.add_objects(
        NUCLEI,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(
                labels=np.array([[1, 2], [0, 0]], dtype=np.int32)
            )
        ),
    )
    adapter.add_objects(
        CELLS,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(
                labels=np.array([[1, 0], [0, 0]], dtype=np.int32)
            )
        ),
    )
    adapter.add_measurements(
        MeasurementTable(
            name=MEASUREMENTS,
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "mean_intensity": 5.0,
                    },
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 2,
                        "mean_intensity": 7.0,
                    },
                    {
                        "slice_index": 0,
                        "object_name": CELLS,
                        "object_label": 1,
                        "mean_intensity": 11.0,
                    },
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("mean_intensity", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.ARTIFACT, MEASUREMENTS),
        )
    )

    values = measurement_values_for_feature(
        (_output_measurements(adapter, MEASUREMENTS),),
        "Intensity_MeanIntensity_CropBlue",
        object_count=2,
        object_name=NUCLEI,
    )

    np.testing.assert_array_equal(values, np.array([5.0, 7.0]))


def test_measurement_lookup_reads_slotted_dataclass_rows():
    @dataclass(frozen=True, slots=True)
    class MeasurementRow:
        object_name: str
        object_label: int
        mean_intensity: float

    values = measurement_values_for_feature(
        (
            MeasurementTable(
                name=MEASUREMENTS,
                rows=DataclassMeasurementColumnarRows(
                    (
                        MeasurementRow(NUCLEI, 1, 5.0),
                        MeasurementRow(CELLS, 1, 11.0),
                    ),
                    row_type=MeasurementRow,
                ),
                subject=MeasurementSubject(MeasurementScope.ARTIFACT, MEASUREMENTS),
            ),
        ),
        "Intensity_MeanIntensity_CropBlue",
        object_count=1,
        object_name=NUCLEI,
    )

    np.testing.assert_array_equal(values, np.array([5.0]))


def test_measurement_lookup_reads_long_form_feature_value_rows():
    values = measurement_values_for_feature(
        (
            MeasurementTable(
                name=MEASUREMENTS,
                rows=MeasurementSparseColumnarRows.from_rows(
                    (
                        {
                            "object_name": NUCLEI,
                            "object_label": 1,
                            "feature_name": "Math_Ratio",
                            "result_value": 0.5,
                        },
                        {
                            "object_name": CELLS,
                            "object_label": 1,
                            "feature_name": "Math_Ratio",
                            "result_value": 0.25,
                        },
                    ),
                    fields=(
                        FieldSpec("object_name", str),
                        FieldSpec("object_label", int),
                        FieldSpec("feature_name", str),
                        FieldSpec("result_value", float),
                    ),
                ),
                subject=MeasurementSubject(MeasurementScope.ARTIFACT, MEASUREMENTS),
            ),
        ),
        "Math_Ratio",
        object_count=1,
        object_name=NUCLEI,
    )

    np.testing.assert_array_equal(values, np.array([0.5]))


def test_measurement_lookup_aligns_values_to_label_slices():
    value_slices = measurement_values_for_label_slices(
        (
            MeasurementTable(
                name=MEASUREMENTS,
                rows=MeasurementSparseColumnarRows.from_rows(
                    (
                        {
                            "slice_index": 0,
                            "object_label": 10,
                            "area": 100.0,
                            "object_name": NUCLEI,
                        },
                        {
                            "slice_index": 0,
                            "object_label": 20,
                            "area": 200.0,
                            "object_name": NUCLEI,
                        },
                        {
                            "slice_index": 1,
                            "object_label": 30,
                            "area": 300.0,
                            "object_name": NUCLEI,
                        },
                    ),
                    fields=(
                        FieldSpec("slice_index", int),
                        FieldSpec("object_label", int),
                        FieldSpec("area", float),
                        FieldSpec("object_name", str),
                    ),
                ),
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=("/src/plane_0.tif", "/src/plane_1.tif"),
                ),
                subject=MeasurementSubject(MeasurementScope.ARTIFACT, MEASUREMENTS),
            ),
        ),
        "AreaShape_Area",
        np.array(
            [
                [[10, 20], [0, 0]],
                [[30, 0], [0, 0]],
            ],
            dtype=np.int32,
        ),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((10, 20), (30,)),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        plane_projector=RuntimePlaneProjection.stack(2),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert len(value_slices) == 2
    np.testing.assert_array_equal(value_slices[0], np.array([100.0, 200.0]))
    np.testing.assert_array_equal(value_slices[1], np.array([300.0]))


def test_measurement_lookup_does_not_stop_on_broad_cellprofiler_alias():
    value_slices = measurement_values_for_label_slices(
        (
            MeasurementTable(
                name=MEASUREMENTS,
                rows=MeasurementSparseColumnarRows.from_rows(
                    (
                        {
                            "slice_index": 0,
                            "object_name": NUCLEI,
                            "object_label": 1,
                            "Location_MaxIntensity_X_OrigGreen": 477.0,
                        },
                        {
                            "slice_index": 0,
                            "object_name": NUCLEI,
                            "object_label": 1,
                            "Intensity_MaxIntensity_OrigGreen": 0.0549019612,
                        },
                        {
                            "slice_index": 0,
                            "object_name": NUCLEI,
                            "object_label": 2,
                            "Intensity_MaxIntensity_OrigGreen": 0.9607843161,
                        },
                    ),
                    fields=(
                        FieldSpec("slice_index", int),
                        FieldSpec("object_name", str),
                        FieldSpec("object_label", int),
                        FieldSpec(
                            "Location_MaxIntensity_X_OrigGreen", float, required=False
                        ),
                        FieldSpec(
                            "Intensity_MaxIntensity_OrigGreen", float, required=False
                        ),
                    ),
                ),
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=("/src/plane_0.tif",),
                ),
                subject=MeasurementSubject(MeasurementScope.ARTIFACT, MEASUREMENTS),
            ),
        ),
        "Intensity_MaxIntensity_OrigGreen",
        np.array([[[1, 2]]], dtype=np.int32),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1, 2),),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        plane_projector=RuntimePlaneProjection.stack(1),
        object_name=NUCLEI,
    )

    np.testing.assert_allclose(value_slices[0], [0.0549019612, 0.9607843161])


def test_measurement_lookup_uses_canonical_runtime_identifier_for_numbered_features():
    values = measurement_values_for_feature(
        (
            MeasurementTable(
                name=MEASUREMENTS,
                rows=MeasurementSparseColumnarRows.from_rows(
                    (
                        {"Children_PH3_Count": 2.0},
                        {"Children_PH3_Count": 5.0},
                    ),
                    fields=(FieldSpec("Children_PH3_Count", float),),
                ),
                subject=MeasurementSubject(MeasurementScope.ARTIFACT, MEASUREMENTS),
            ),
        ),
        "Children_PH3_Count",
        object_count=2,
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_array_equal(values, np.array([2.0, 5.0]))


def test_cellprofiler_child_count_lookup_uses_parent_row_domain():
    lookup = CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT.feature_lookup(
        "Children_PH3_Count"
    )

    assert lookup.query_object_name(NUCLEI) is None


def test_adapter_batch_child_count_lookup_uses_parent_row_domain():
    table = MeasurementTable(
        name=MEASUREMENTS,
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"object_name": NUCLEI, "object_label": 1, "Children_PH3_Count": 2.0},
                {"object_name": NUCLEI, "object_label": 2, "Children_PH3_Count": 5.0},
            ),
            fields=(
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("Children_PH3_Count", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI, "object_label"),
    )

    values_by_label, positional_values = MeasurementObjectFeatureVectorBatchQuery(
        "Children_PH3_Count",
        ("PH3",),
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    ).value_indexes({"PH3": (table,)})["PH3"]

    assert positional_values == []
    assert values_by_label == {1: 2.0, 2: 5.0}


def test_relationship_plane_records_use_compiled_input_projection_order():
    store = RuntimeValueStore()
    relationship_name = "Nuclei_PH3_relationships"
    group_paths = {
        "site_a": "/memory/relationship_site_a.pkl",
        "site_b": "/memory/relationship_site_b.pkl",
    }
    declaration = ObjectRelationshipDeclaration.parent_child(
        source=ArtifactSpec.output(NUCLEI, ObjectLabelsArtifactType).ref(),
        target=ArtifactSpec.output("PH3", ObjectLabelsArtifactType).ref(),
        producer_module_number=1,
    )
    for group_key, parent_id in (("site_b", 2), ("site_a", 1)):
        relationship = ObjectRelationship(
            name=relationship_name,
            declaration=declaration,
            payload=DirectedObjectRelationshipPayload(
                source_ids=(parent_id,),
                target_ids=(parent_id,),
                slice_indices=(),
                slice_count=None,
            ),
        )
        value = RuntimeValue.normalize(
            ArtifactOutputPlan(
                name=relationship_name,
                path=group_paths[group_key],
                artifact_type=RelationshipsArtifactType,
                group_keys=(group_key,),
                group_component=AllComponents.SITE,
                paths_by_group={group_key: group_paths[group_key]},
            ),
            relationship,
            axis_id=AXIS_ID,
        )
        store.record(value, path=group_paths[group_key], backend="memory")
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=relationship_name,
                    path="/memory/relationships.pkl",
                    artifact_type=RelationshipsArtifactType,
                    group_keys=("site_a", "site_b"),
                    group_component=AllComponents.SITE,
                    paths_by_group=group_paths,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.from_raw(
                    ("site_a", "site_b"), component=AllComponents.SITE
                ),
                component_scopes=(
                    ComponentGroupScope.from_raw(
                        ("site_a", "site_b"), component=AllComponents.SITE
                    ),
                ),
                consumer_variable_components=(AllComponents.SITE,),
            ),
        )
    }
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        variable_components=(VariableComponents.SITE,),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    relationship_value = adapter.get_relationship(relationship_name)
    assert isinstance(relationship_value, RuntimeSliceAlignedValues)
    plane_resolution = RelationshipPlaneProjectionResolution.from_value(
        relationship_name,
        relationship_value,
        label_plane_count=2,
    )
    plane_records = plane_resolution.records

    assert [record.plane_index for record in plane_records] == [0, 1]
    assert [
        tuple(record.relationship.payload.source_ids) for record in plane_records
    ] == [
        (1,),
        (2,),
    ]


def test_relationship_plane_projection_rejects_mismatched_runtime_slice_count():
    declaration = ObjectRelationshipDeclaration.parent_child(
        source=ArtifactSpec.output(NUCLEI, ObjectLabelsArtifactType).ref(),
        target=ArtifactSpec.output("PH3", ObjectLabelsArtifactType).ref(),
        producer_module_number=1,
    )
    relationship = ObjectRelationship(
        name="Nuclei_PH3_relationships",
        declaration=declaration,
        payload=DirectedObjectRelationshipPayload(
            source_ids=(1,), target_ids=(1,), slice_indices=(), slice_count=None
        ),
    )

    with pytest.raises(ValueError, match="carries 1 runtime slices for 2 label planes"):
        RelationshipPlaneProjectionResolution.from_value(
            "Nuclei_PH3_relationships",
            RuntimeSliceAlignedValues((relationship,)),
            label_plane_count=2,
        )


def test_object_measurement_table_index_uses_declared_subject_for_unnamed_rows():
    table = MeasurementTable(
        name="Shape",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "object_label": 1, "form_factor": 0.9},
                {"slice_index": 0, "object_label": 2, "form_factor": 1.1},
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("form_factor", float),
            ),
        ),
        subject=MeasurementSubject(
            MeasurementScope.OBJECT,
            CELLS,
            MeasurementRowAxisField.OBJECT_LABEL.value,
        ),
    )

    tables = ObjectMeasurementTableIndex.from_tables((table,)).for_object_feature(
        "Cells",
        "AreaShape_FormFactor",
    )

    assert tables == (table,)


def test_child_count_lookup_tolerates_heterogeneous_relationship_summary_rows():
    table = MeasurementTable(
        name=RELATE_OBJECTS,
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "child_object_count": 2},
                {
                    "slice_index": 0,
                    "object_name": NUCLEI,
                    "object_label": 1,
                    "Children_PH3_Count": 2.0,
                },
                {
                    "slice_index": 0,
                    "object_name": NUCLEI,
                    "object_label": 2,
                    "Children_PH3_Count": 0.0,
                },
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("child_object_count", int, required=False),
                FieldSpec("object_name", str, required=False),
                FieldSpec("object_label", int, required=False),
                FieldSpec("Children_PH3_Count", float, required=False),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, RELATE_OBJECTS),
    )
    projected_table = MeasurementTableAxisProjection(
        MeasurementRowAxisField.SLICE_INDEX,
        0,
    ).tables((table,))[0]

    values = measurement_values_for_feature(
        (projected_table,),
        "Children_PH3_Count",
        object_count=2,
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [2.0, 0.0])


def test_adapter_feature_lookup_rejects_undeclared_measurement_output():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        )
    )
    labels = np.array([[[1, 2]]], dtype=np.int32)
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1, 2),),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    )
    adapter.add_measurements(
        MeasurementTable(
            name=MEASUREMENTS,
            rows=MeasurementSparseColumnarRows.from_rows(
                (
                    {
                        "object_label": 1,
                        "Location_MaxIntensity_X_OrigGreen": 477.0,
                    },
                    {
                        "object_label": 1,
                        "Intensity_MaxIntensity_OrigGreen": 0.0549019612,
                    },
                    {
                        "object_label": 2,
                        "Intensity_MaxIntensity_OrigGreen": 0.9607843161,
                    },
                ),
                fields=(
                    FieldSpec("object_label", int),
                    FieldSpec(
                        "Location_MaxIntensity_X_OrigGreen", float, required=False
                    ),
                    FieldSpec(
                        "Intensity_MaxIntensity_OrigGreen", float, required=False
                    ),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI, "object_label"),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/plane_0.tif",),
            ),
        )
    )

    with pytest.raises(
        ValueError, match="requires a declared relationship or measurement"
    ):
        adapter_label_measurement_values(
            adapter,
            NUCLEI,
            "Intensity_MaxIntensity_OrigGreen",
            labels,
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1, 2),),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )


def test_adapter_measurement_vector_scope_uses_feature_bearing_axis_projection():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    measurement_output = _plan(MEASUREMENTS, MeasurementsArtifactType)
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(NUCLEI, ObjectLabelsArtifactType),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=measurement_output,
            ),
        ),
        filemanager=filemanager,
        plane_projection=RuntimePlaneProjection.stack(1),
    )
    labels = np.array([[[1, 2]]], dtype=np.int32)
    producer.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1, 2),),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    )
    producer.add_measurements(
        MeasurementTable(
            name=MEASUREMENTS,
            rows=MeasurementSparseColumnarRows.from_rows(
                (
                    {
                        "slice_index": 0,
                        "object_label": 1,
                        "Intensity_MaxIntensity_OrigGreen": 0.25,
                    },
                    {
                        "slice_index": 0,
                        "object_label": 2,
                        "Intensity_MaxIntensity_OrigGreen": 0.75,
                    },
                ),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_label", int),
                    FieldSpec("Intensity_MaxIntensity_OrigGreen", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI, "object_label"),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/plane_0.tif",),
            ),
        )
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=MEASUREMENTS,
                    path=measurement_output.path,
                    artifact_type=MeasurementsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        )
    }
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        plane_projection=RuntimePlaneProjection.stack(1),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    value_slices = adapter_label_measurement_values(
        adapter,
        NUCLEI,
        "Intensity_MaxIntensity_OrigGreen",
        labels,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1, 2),),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        slice_index=99,
    )

    np.testing.assert_allclose(value_slices[0], [0.25, 0.75])


def test_adapter_multiplane_label_lookup_prefers_runtime_slice_axis():
    store = RuntimeValueStore()
    filemanager = FileManagerStub()
    measurement_output = _plan(MEASUREMENTS, MeasurementsArtifactType)
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_output_bindings=(
            _output_binding(NUCLEI, ObjectLabelsArtifactType),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=measurement_output,
            ),
        ),
        filemanager=filemanager,
        plane_projection=RuntimePlaneProjection.stack(2),
    )
    labels = np.array(
        [
            [[1, 0], [0, 0]],
            [[2, 0], [0, 0]],
        ],
        dtype=np.int32,
    )
    producer.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1,), (2,)),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    )
    producer.add_measurements(
        MeasurementTable(
            name=MEASUREMENTS,
            rows=MeasurementSparseColumnarRows.from_rows(
                (
                    {
                        "slice_index": 0,
                        "object_label": 1,
                        "Intensity_MaxIntensity_OrigGreen": 0.25,
                    },
                    {
                        "slice_index": 1,
                        "object_label": 2,
                        "Intensity_MaxIntensity_OrigGreen": 0.75,
                    },
                ),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_label", int),
                    FieldSpec("Intensity_MaxIntensity_OrigGreen", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI, "object_label"),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/plane_0.tif", "/src/plane_1.tif"),
            ),
        )
    )
    _compiled_artifact_inputs = {
        edge.key: edge
        for edge in (
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=MEASUREMENTS,
                    path=measurement_output.path,
                    artifact_type=MeasurementsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        )
    }
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        artifact_inputs=_compiled_artifact_inputs,
        filemanager=filemanager,
        plane_projection=RuntimePlaneProjection.stack(2),
        callable_contract=_compiled_callable_contract(
            calculate_math,
            artifact_inputs=tuple(
                edge.spec for edge in _compiled_artifact_inputs.values()
            ),
        ),
    )

    value_slices = adapter_label_measurement_values(
        adapter,
        NUCLEI,
        "Intensity_MaxIntensity_OrigGreen",
        labels,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (2,)),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        slice_index=99,
    )

    np.testing.assert_allclose(value_slices[0], [0.25])
    np.testing.assert_allclose(value_slices[1], [0.75])


def test_adapter_measurement_vector_scope_rejects_undeclared_axis_table():
    filemanager = FileManagerStub()
    store = RuntimeValueStore()
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        group_key="producer",
        artifact_output_bindings=(
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(
                    NUCLEI, ObjectLabelsArtifactType, group_component=AllComponents.SITE
                ),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(
                    MEASUREMENTS,
                    MeasurementsArtifactType,
                    group_component=AllComponents.SITE,
                ),
            ),
        ),
        filemanager=filemanager,
    )
    labels = np.array([[[1, 2]]], dtype=np.int32)
    producer.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1, 2),),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    )
    producer.add_measurements(
        MeasurementTable(
            name=MEASUREMENTS,
            rows=MeasurementSparseColumnarRows.from_rows(
                (
                    {
                        "object_label": 1,
                        "Intensity_MaxIntensity_OrigGreen": 0.25,
                    },
                    {
                        "object_label": 2,
                        "Intensity_MaxIntensity_OrigGreen": 0.75,
                    },
                ),
                fields=(
                    FieldSpec("object_label", int),
                    FieldSpec("Intensity_MaxIntensity_OrigGreen", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI, "object_label"),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/plane_0.tif",),
            ),
        )
    )
    consumer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=runtime_axis_scope(AXIS_ID),
        group_key="consumer",
        filemanager=filemanager,
    )

    with pytest.raises(
        ValueError, match="requires a declared relationship or measurement"
    ):
        adapter_label_measurement_values(
            consumer,
            NUCLEI,
            "Intensity_MaxIntensity_OrigGreen",
            labels,
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1, 2),),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )


def test_measurement_lookup_filters_source_qualified_columnar_feature_rows():
    values = measurement_values_for_feature(
        (
            MeasurementTable(
                name=MEASURE_OBJECT_INTENSITY,
                rows=SimpleColumnarRows(
                    {
                        "object_name": (NUCLEI, NUCLEI, NUCLEI, NUCLEI),
                        "object_label": (1, 2, 1, 2),
                        "source_image_name": (
                            DNA_IMAGE,
                            DNA_IMAGE,
                            "OrigGreen",
                            "OrigGreen",
                        ),
                        "max_intensity": (0.90, 0.95, 0.05, 0.80),
                    },
                    fields=(
                        FieldSpec("object_name", str),
                        FieldSpec("object_label", int),
                        FieldSpec("source_image_name", str),
                        FieldSpec("max_intensity", float),
                    ),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT, MEASURE_OBJECT_INTENSITY
                ),
            ),
        ),
        "Intensity_MaxIntensity_OrigGreen",
        object_count=2,
        object_ids=(1, 2),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [0.05, 0.80])


def test_measurement_lookup_preserves_heterogeneous_columnar_batch_features():
    rows = ConcatenatedColumnarRows(
        (
            SimpleColumnarRows(
                {
                    "object_name": (NUCLEI, NUCLEI),
                    "object_label": (1, 2),
                    "source_image_name": (DNA_IMAGE, DNA_IMAGE),
                    "mean_intensity": (0.90, 0.95),
                },
                fields=(
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("source_image_name", str),
                    FieldSpec("mean_intensity", float),
                ),
            ),
            SimpleColumnarRows(
                {
                    "object_name": (NUCLEI, NUCLEI),
                    "object_label": (1, 2),
                    "source_image_name": ("rawGFP", "rawGFP"),
                    "MeanIntensity_rawGFP": (0.05, 0.80),
                },
                fields=(
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("source_image_name", str),
                    FieldSpec("MeanIntensity_rawGFP", float),
                ),
            ),
        )
    )

    values = measurement_values_for_feature(
        (
            MeasurementTable(
                name=MEASURE_OBJECT_INTENSITY,
                rows=rows,
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT, MEASURE_OBJECT_INTENSITY
                ),
            ),
        ),
        "Intensity_MeanIntensity_rawGFP",
        object_count=2,
        object_ids=(1, 2),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [0.05, 0.80])


def test_measurement_slice_projection_keeps_axisless_columnar_rows():
    rows = ConcatenatedColumnarRows(
        (
            SimpleColumnarRows(
                {
                    "object_name": (NUCLEI,),
                    "object_label": (1,),
                    "slice_index": (23,),
                    "source_image_name": (DNA_IMAGE,),
                    "mean_intensity": (0.90,),
                },
                fields=(
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("slice_index", int),
                    FieldSpec("source_image_name", str),
                    FieldSpec("mean_intensity", float),
                ),
            ),
            SimpleColumnarRows(
                {
                    "object_name": (NUCLEI,),
                    "object_label": (1,),
                    "source_image_name": ("rawGFP",),
                    "mean_intensity": (0.05,),
                },
                fields=(
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("source_image_name", str),
                    FieldSpec("mean_intensity", float),
                ),
            ),
        )
    )
    table = MeasurementTable(
        name=MEASURE_OBJECT_INTENSITY,
        rows=rows,
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, MEASURE_OBJECT_INTENSITY),
    )

    values = measurement_values_for_feature(
        MeasurementTableAxisProjection(
            MeasurementRowAxisField.SLICE_INDEX,
            23,
        ).tables((table,)),
        "Intensity_MeanIntensity_rawGFP",
        object_count=1,
        object_ids=(1,),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    np.testing.assert_allclose(values, [0.05])


def test_measurement_slice_projection_requires_exact_singleton_axis_match():
    rows = SimpleColumnarRows(
        {
            "object_name": (NUCLEI,),
            "object_label": (1,),
            "slice_index": (1,),
            "source_image_name": ("rawGFP",),
            "mean_intensity": (0.05,),
        },
        fields=(
            FieldSpec("object_name", str),
            FieldSpec("object_label", int),
            FieldSpec("slice_index", int),
            FieldSpec("source_image_name", str),
            FieldSpec("mean_intensity", float),
        ),
    )
    table = MeasurementTable(
        name=MEASURE_OBJECT_INTENSITY,
        rows=rows,
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, MEASURE_OBJECT_INTENSITY),
    )

    projected = MeasurementTableAxisProjection(
        MeasurementRowAxisField.SLICE_INDEX,
        23,
    ).tables((table,))

    assert measurement_rows(projected) == ()


def test_measurement_slice_projection_does_not_treat_row_sequence_singleton_as_invariant():
    table = MeasurementTable(
        name=MEASURE_OBJECT_INTENSITY,
        rows=MeasurementSparseColumnarRows.from_rows(
            [
                {
                    "object_name": NUCLEI,
                    "object_label": 1,
                    "slice_index": 1,
                    "mean_intensity": 0.05,
                },
            ],
            fields=(
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("slice_index", int),
                FieldSpec("mean_intensity", float),
            ),
            object_row_identity=MeasurementObjectRowIdentity.ROW_SEQUENCE,
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, MEASURE_OBJECT_INTENSITY),
    )

    projected = MeasurementTableAxisProjection(
        MeasurementRowAxisField.SLICE_INDEX,
        23,
    ).tables((table,))

    assert measurement_rows(projected) == ()


def test_measurement_lookup_allows_empty_multiplane_label_planes():
    labels = np.array(
        [
            [[0, 0]],
            [[1, 2]],
        ],
        dtype=np.int32,
    )
    measurement_rows = SimpleColumnarRows(
        {
            "slice_index": (1, 1),
            "object_name": (NUCLEI, NUCLEI),
            "object_label": (1, 2),
            "source_image_name": ("rawGFP", "rawGFP"),
            "mean_intensity": (0.25, 0.75),
        },
        fields=(
            FieldSpec("slice_index", int),
            FieldSpec("object_name", str),
            FieldSpec("object_label", int),
            FieldSpec("source_image_name", str),
            FieldSpec("mean_intensity", float),
        ),
    )

    value_slices = measurement_values_for_label_slices(
        (
            MeasurementTable(
                name=MEASURE_OBJECT_INTENSITY,
                rows=measurement_rows,
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=("/src/plane_0.tif", "/src/plane_1.tif"),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT, MEASURE_OBJECT_INTENSITY
                ),
            ),
        ),
        "Intensity_MeanIntensity_rawGFP",
        labels,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((), (1, 2)),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        plane_projector=RuntimePlaneProjection.stack(2),
        object_name=NUCLEI,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert value_slices[0].size == 0
    np.testing.assert_allclose(value_slices[1], [0.25, 0.75])


def test_measurement_lookup_rejects_singleton_columnar_slice_for_label_stack():
    with pytest.raises(ValueError, match="does not match the declared label domain"):
        measurement_values_for_label_slices(
            (
                MeasurementTable(
                    name=MEASURE_OBJECT_INTENSITY,
                    rows=SimpleColumnarRows(
                        {
                            "slice_index": (0, 0),
                            "object_name": (NUCLEI, NUCLEI),
                            "object_label": (1, 2),
                            "source_image_name": ("rawGFP", "rawGFP"),
                            "mean_intensity": (0.25, 0.75),
                        },
                        fields=(
                            FieldSpec("slice_index", int),
                            FieldSpec("object_name", str),
                            FieldSpec("object_label", int),
                            FieldSpec("source_image_name", str),
                            FieldSpec("mean_intensity", float),
                        ),
                    ),
                    source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                        paths=("/src/plane_0.tif",),
                    ),
                    subject=MeasurementSubject(
                        MeasurementScope.ARTIFACT, MEASURE_OBJECT_INTENSITY
                    ),
                ),
            ),
            "Intensity_MeanIntensity_rawGFP",
            np.array(
                [
                    [[1, 0], [0, 0]],
                    [[2, 0], [0, 0]],
                ],
                dtype=np.int32,
            ),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1,), (2,)),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_projector=RuntimePlaneProjection.stack(2),
            object_name=NUCLEI,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )


def test_measurement_lookup_rejects_singleton_indexed_slice_for_label_stack():
    with pytest.raises(ValueError, match="does not match the declared label domain"):
        measurement_values_for_label_slices(
            (
                MeasurementTable(
                    name=MEASUREMENTS,
                    rows=MeasurementSparseColumnarRows.from_rows(
                        (
                            {
                                "slice_index": 0,
                                "object_label": 1,
                                "mean_intensity": 0.25,
                                "object_name": NUCLEI,
                            },
                            {
                                "slice_index": 0,
                                "object_label": 2,
                                "mean_intensity": 0.5,
                                "object_name": NUCLEI,
                            },
                        ),
                        fields=(
                            FieldSpec("slice_index", int),
                            FieldSpec("object_label", int),
                            FieldSpec("mean_intensity", float),
                            FieldSpec("object_name", str),
                        ),
                    ),
                    source_image_name="rawGFP",
                    source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                        paths=("/src/plane_0.tif",),
                    ),
                    subject=MeasurementSubject(MeasurementScope.IMAGE, "rawGFP"),
                ),
            ),
            "Intensity_MeanIntensity_rawGFP",
            np.array(
                [
                    [[1, 0], [0, 0]],
                    [[2, 0], [0, 0]],
                ],
                dtype=np.int32,
            ),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1,), (2,)),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_projector=RuntimePlaneProjection.stack(2),
            object_name=NUCLEI,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )


def test_measurement_lookup_rejects_shifted_slice_domain_for_local_label_stack():
    with pytest.raises(ValueError, match="does not match the declared label domain"):
        measurement_values_for_label_slices(
            (
                MeasurementTable(
                    name=MEASUREMENTS,
                    rows=MeasurementSparseColumnarRows.from_rows(
                        (
                            {
                                "slice_index": 8,
                                "object_label": 1,
                                "mean_intensity": 0.25,
                                "object_name": NUCLEI,
                            },
                            {
                                "slice_index": 9,
                                "object_label": 2,
                                "mean_intensity": 0.5,
                                "object_name": NUCLEI,
                            },
                        ),
                        fields=(
                            FieldSpec("slice_index", int),
                            FieldSpec("object_label", int),
                            FieldSpec("mean_intensity", float),
                            FieldSpec("object_name", str),
                        ),
                    ),
                    source_image_name="rawGFP",
                    source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                        paths=("/src/plane_0.tif", "/src/plane_1.tif"),
                    ),
                    subject=MeasurementSubject(MeasurementScope.IMAGE, "rawGFP"),
                ),
            ),
            "Intensity_MeanIntensity_rawGFP",
            np.array(
                [
                    [[1, 0], [0, 0]],
                    [[2, 0], [0, 0]],
                ],
                dtype=np.int32,
            ),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1,), (2,)),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_projector=RuntimePlaneProjection.stack(2),
            object_name=NUCLEI,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )


def test_measurement_lookup_rejects_smaller_axis_domain_for_label_stack():
    with pytest.raises(ValueError, match="does not match the declared label domain"):
        measurement_values_for_label_slices(
            (
                MeasurementTable(
                    name=MEASUREMENTS,
                    rows=SimpleColumnarRows(
                        {
                            "slice_index": (0, 1),
                            "object_name": (NUCLEI, NUCLEI),
                            "object_label": (1, 1),
                            "source_image_name": ("rawGFP", "rawGFP"),
                            "mean_intensity": (0.25, 0.5),
                        },
                        fields=(
                            FieldSpec("slice_index", int),
                            FieldSpec("object_name", str),
                            FieldSpec("object_label", int),
                            FieldSpec("source_image_name", str),
                            FieldSpec("mean_intensity", float),
                        ),
                    ),
                    source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                        paths=("/src/plane_0.tif", "/src/plane_1.tif"),
                    ),
                    subject=MeasurementSubject(MeasurementScope.ARTIFACT, MEASUREMENTS),
                ),
            ),
            "Intensity_MeanIntensity_rawGFP",
            np.array(
                [
                    [[1, 0], [0, 0]],
                    [[1, 0], [0, 0]],
                    [[1, 0], [0, 0]],
                    [[1, 0], [0, 0]],
                ],
                dtype=np.int32,
            ),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1,), (1,), (1,), (1,)),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_projector=RuntimePlaneProjection.stack(4),
            object_name=NUCLEI,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )


def test_measurement_lookup_returns_empty_slices_for_empty_objects():
    value_slices = measurement_values_for_label_slices(
        (),
        "AreaShape_FormFactor",
        np.zeros((2, 3, 4), dtype=np.int32),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((), ()),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        plane_projector=RuntimePlaneProjection.stack(2),
        object_name=NUCLEI,
    )

    assert len(value_slices) == 2
    assert all(value_slice.size == 0 for value_slice in value_slices)


def test_measurement_lookup_rejects_undeclared_row_axis_for_nonempty_objects():
    with pytest.raises(ValueError, match="row axis does not match"):
        measurement_values_for_label_slices(
            (),
            "AreaShape_FormFactor",
            np.array([[[1, 0], [0, 0]]], dtype=np.int32),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1,),),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_projector=RuntimePlaneProjection.stack(1),
            object_name=NUCLEI,
        )


def test_calculate_math_records_object_indexed_measurements():
    output_name = "Ratio"
    contract = _calculate_math_contract(
        output_name=output_name,
        operand1_feature="Intensity_MeanIntensity_CropBlue",
        operand2_feature="AreaShape_Area",
        operand1_object_name=NUCLEI,
        operand2_object_name=NUCLEI,
    )
    measurement_output = _measurement_output_name(contract)
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                "PriorMeasurements",
                MeasurementsArtifactType,
                plan=_plan("PriorMeasurements", MeasurementsArtifactType),
            ),
            _output_binding(
                measurement_output,
                MeasurementsArtifactType,
                plan=_plan(measurement_output, MeasurementsArtifactType),
            ),
        )
    )
    labels = np.array([[[1, 2], [0, 0]]], dtype=np.int32)
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1, 2),),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        ),
    )
    adapter.add_measurements(
        MeasurementTable(
            name="PriorMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "mean_intensity": 10.0,
                        "area": 20.0,
                    },
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 2,
                        "mean_intensity": 20.0,
                        "area": 80.0,
                    },
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("mean_intensity", float),
                    FieldSpec("area", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        )
    )
    calculate_math = CellProfilerModule.require_module(
        CALCULATE_MATH
    ).require_callable()
    executor = _executor_for_contract(
        calculate_math,
        adapter,
        contract,
    )

    result = executor(
        np.zeros((2, 2), dtype=np.float32),
        cellprofiler_runtime=adapter,
        operation=ImageMathOperation.DIVIDE,
        operand1_feature="Intensity_MeanIntensity_CropBlue",
        operand2_feature="AreaShape_Area",
        output_name=output_name,
        dtype_config=DtypeConfig(),
    )
    measurements = _output_measurements(adapter, measurement_output)
    measurement_rows = tuple(measurements.rows.iter_row_mappings())

    np.testing.assert_array_equal(result, np.zeros((2, 2), dtype=np.float32))
    assert measurements.subject.object_name == NUCLEI
    assert [row["object_name"] for row in measurement_rows] == [NUCLEI, NUCLEI]
    assert [row["object_label"] for row in measurement_rows] == [1, 2]
    assert [row["feature_name"] for row in measurement_rows] == [
        "Math_Ratio",
        "Math_Ratio",
    ]
    np.testing.assert_allclose(
        [row["result_value"] for row in measurement_rows],
        [0.5, 0.25],
    )
    np.testing.assert_allclose(
        measurement_values_for_feature(
            (measurements,),
            "Math_Ratio",
            object_count=2,
            object_name=NUCLEI,
        ),
        np.array([0.5, 0.25]),
    )


def test_calculate_math_feature_name_does_not_replace_artifact_identity():
    contract = _calculate_math_contract(
        output_name="PercentPositive",
        operand1_feature="AreaOccupied_AreaOccupied_Objects1",
        operand2_feature="AreaOccupied_AreaOccupied_Objects2",
    )

    assert _measurement_output_name(contract) == "CalculateMath_1_measurements"


def test_calculate_math_pads_missing_same_object_operand_values():
    output_name = "Ratio"
    contract = _calculate_math_contract(
        output_name=output_name,
        operand1_feature="Intensity_MeanIntensity_CropBlue",
        operand2_feature="AreaShape_Area",
        operand1_object_name=NUCLEI,
        operand2_object_name=NUCLEI,
    )
    measurement_output = _measurement_output_name(contract)
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                "PriorMeasurements",
                MeasurementsArtifactType,
                plan=_plan("PriorMeasurements", MeasurementsArtifactType),
            ),
            _output_binding(
                measurement_output,
                MeasurementsArtifactType,
                plan=_plan(measurement_output, MeasurementsArtifactType),
            ),
        )
    )
    labels = np.array([[[1, 2], [0, 0]]], dtype=np.int32)
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1, 2),),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        ),
    )
    adapter.add_measurements(
        MeasurementTable(
            name="PriorMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "mean_intensity": 10.0,
                        "area": 20.0,
                    },
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 2,
                        "mean_intensity": 20.0,
                    },
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("mean_intensity", float),
                    FieldSpec("area", float, required=False),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        )
    )
    calculate_math = CellProfilerModule.require_module(
        CALCULATE_MATH
    ).require_callable()
    executor = _executor_for_contract(
        calculate_math,
        adapter,
        contract,
    )

    executor(
        np.zeros((2, 2), dtype=np.float32),
        cellprofiler_runtime=adapter,
        operation=ImageMathOperation.DIVIDE,
        operand1_feature="Intensity_MeanIntensity_CropBlue",
        operand2_feature="AreaShape_Area",
        output_name=output_name,
        dtype_config=DtypeConfig(),
    )

    measurements = _output_measurements(adapter, measurement_output)
    measurement_rows = _measurement_rows_for_assertion(measurements)
    assert [row["object_label"] for row in measurement_rows] == [1, 2]
    np.testing.assert_allclose(
        [row["result_value"] for row in measurement_rows],
        [0.5, np.nan],
        equal_nan=True,
    )


def test_calculate_math_resolves_image_scoped_measurements_via_core_query():
    output_name = "Stain1Colocalized"
    contract = _calculate_math_contract(
        output_name=output_name,
        operand1_feature="AreaOccupied_AreaOccupied_ColocalizedRegion",
        operand2_feature="AreaOccupied_AreaOccupied_Objects1",
    )
    measurement_output = _measurement_output_name(contract)
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                "PriorMeasurements",
                MeasurementsArtifactType,
                plan=_plan("PriorMeasurements", MeasurementsArtifactType),
            ),
            _output_binding(
                measurement_output,
                MeasurementsArtifactType,
                plan=_plan(measurement_output, MeasurementsArtifactType),
            ),
        )
    )
    adapter.add_measurements(
        MeasurementTable(
            name="PriorMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": 0,
                        "area_occupied": 17809.0,
                        "source_image_name": "ColocalizedRegion",
                    },
                    {
                        "slice_index": 0,
                        "area_occupied": 30324.0,
                        "source_image_name": "Objects1",
                    },
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("area_occupied", float),
                    FieldSpec("source_image_name", str),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.ARTIFACT, "PriorMeasurements"),
        )
    )
    calculate_math = CellProfilerModule.require_module(
        CALCULATE_MATH
    ).require_callable()
    executor = _executor_for_contract(
        calculate_math,
        adapter,
        contract,
    )

    executor(
        np.zeros((2, 2), dtype=np.float32),
        cellprofiler_runtime=adapter,
        operation=ImageMathOperation.DIVIDE,
        operand1_feature="AreaOccupied_AreaOccupied_ColocalizedRegion",
        operand2_feature="AreaOccupied_AreaOccupied_Objects1",
        output_name=output_name,
        dtype_config=DtypeConfig(),
    )
    measurements = _output_measurements(adapter, measurement_output)
    measurement_rows = _measurement_rows_for_assertion(measurements)

    assert measurements.subject.object_name is None
    assert len(measurements.rows) == 1
    row = measurement_rows[0]
    assert row["feature_name"] == "Math_Stain1Colocalized"
    assert row["result_value"] == pytest.approx(17809.0 / 30324.0)
    assert "operand1_value" not in row
    assert "operand2_value" not in row
    assert "operation" not in row


def test_calculate_math_aligns_image_scoped_measurements_by_slice():
    output_name = "Stain1Colocalized"
    contract = _calculate_math_contract(
        output_name=output_name,
        operand1_feature="AreaOccupied_AreaOccupied_ColocalizedRegion",
        operand2_feature="AreaOccupied_AreaOccupied_Objects1",
    )
    measurement_output = _measurement_output_name(contract)
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                "PriorMeasurements",
                MeasurementsArtifactType,
                plan=_plan("PriorMeasurements", MeasurementsArtifactType),
            ),
            _output_binding(
                measurement_output,
                MeasurementsArtifactType,
                plan=_plan(measurement_output, MeasurementsArtifactType),
            ),
        )
    )
    adapter.add_measurements(
        MeasurementTable(
            name="PriorMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": 0,
                        "area_occupied": 10.0,
                        "source_image_name": "ColocalizedRegion",
                    },
                    {
                        "slice_index": 1,
                        "area_occupied": 15.0,
                        "source_image_name": "ColocalizedRegion",
                    },
                    {
                        "slice_index": 0,
                        "area_occupied": 20.0,
                        "source_image_name": "Objects1",
                    },
                    {
                        "slice_index": 1,
                        "area_occupied": 30.0,
                        "source_image_name": "Objects1",
                    },
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("area_occupied", float),
                    FieldSpec("source_image_name", str),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.ARTIFACT, "PriorMeasurements"),
        )
    )
    calculate_math = CellProfilerModule.require_module(
        CALCULATE_MATH
    ).require_callable()
    executor = _executor_for_contract(
        calculate_math,
        adapter,
        contract,
    )

    result = executor(
        np.zeros((2, 2, 2), dtype=np.float32),
        cellprofiler_runtime=adapter,
        operation=ImageMathOperation.DIVIDE,
        operand1_feature="AreaOccupied_AreaOccupied_ColocalizedRegion",
        operand2_feature="AreaOccupied_AreaOccupied_Objects1",
        output_name=output_name,
        dtype_config=DtypeConfig(),
    )
    measurements = _output_measurements(adapter, measurement_output)
    measurement_rows = _measurement_rows_for_assertion(measurements)

    np.testing.assert_array_equal(result, np.zeros((2, 2, 2), dtype=np.float32))
    assert measurements.subject.object_name is None
    assert [row["slice_index"] for row in measurement_rows] == [0, 1]
    assert [row["object_label"] for row in measurement_rows] == [None, None]
    np.testing.assert_allclose(
        [row["result_value"] for row in measurement_rows],
        [0.5, 0.5],
    )


def test_calculate_math_preserves_declared_group_measurement_axis(
    monkeypatch,
):
    tables = tuple(
        MeasurementTable(
            name="MeasureImageAreaOccupied_13_measurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": slice_index,
                        "area_occupied": value,
                        "source_image_name": "ColocalizedRegion",
                    }
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("area_occupied", float),
                    FieldSpec("source_image_name", str),
                ),
            ),
            subject=MeasurementSubject(
                MeasurementScope.ARTIFACT, "MeasureImageAreaOccupied_13_measurements"
            ),
        )
        for slice_index, value in enumerate((10.0, 15.0))
    )

    monkeypatch.setattr(
        MeasurementImageOperandVectorResolution,
        "runtime_feature_tables",
        classmethod(
            lambda cls, adapter, query, **kwargs: (
                tables if not kwargs["match_group"] else ()
            )
        ),
    )
    adapter = SimpleNamespace(measurement_tables=lambda **kwargs: ())

    aligned = MeasurementImageOperandVectorResolution.runtime_axis_scope_tables(
        adapter,
        "AreaOccupied_AreaOccupied_ColocalizedRegion",
        group_key="1",
    )

    assert [
        next(iter(table.rows.iter_row_mappings()))["slice_index"] for table in aligned
    ] == [0, 1]


def test_classify_objects_binds_runtime_measurement_values():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                "PriorMeasurements",
                MeasurementsArtifactType,
                plan=_plan("PriorMeasurements", MeasurementsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        )
    )
    labels = np.array([[[1, 2], [0, 0]]], dtype=np.int32)
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1, 2),),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        ),
    )
    adapter.add_measurements(
        MeasurementTable(
            name="PriorMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "feature_name": "Math_Ratio",
                        "result_value": 0.5,
                    },
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 2,
                        "feature_name": "Math_Ratio",
                        "result_value": 0.8,
                    },
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("feature_name", str),
                    FieldSpec("result_value", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        )
    )
    classify_objects = CellProfilerModule.require_module(
        "ClassifyObjects"
    ).require_callable()
    executor = _executor(
        classify_objects,
        adapter,
        (ArtifactSpec.output(MEASUREMENTS, MeasurementsArtifactType),),
        main_flow_inputs=(),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
            ArtifactSpec.input("PriorMeasurements", MeasurementsArtifactType),
        ),
    )

    result = executor(
        ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(np.zeros((1, 2, 2), dtype=np.float32), None),
        cellprofiler_runtime=adapter,
        measurement_feature="Math_Ratio",
        bin_choice=ClassificationBinChoice.EVEN,
        bin_count=2,
        low_threshold=0.0,
        high_threshold=1.0,
        dtype_config=DtypeConfig(),
    )
    measurements = _output_measurements(adapter, MEASUREMENTS)

    np.testing.assert_array_equal(result, np.zeros((1, 2, 2), dtype=np.float32))
    assert measurements.subject.object_name is None
    assert _wide_measurement_feature_values(
        measurements.rows,
        (
            "Classify_Bin_1_NumObjectsPerBin",
            "Classify_Bin_2_NumObjectsPerBin",
        ),
    ) == {
        ("Classify_Bin_1_NumObjectsPerBin", 1),
        ("Classify_Bin_2_NumObjectsPerBin", 1),
    }
    assert _wide_measurement_feature_values(
        measurements.rows,
        ("Classify_Bin_1", "Classify_Bin_2"),
        identity_fields=("object_label",),
    ) == {
        (1, "Classify_Bin_1", 1),
        (1, "Classify_Bin_2", 0),
        (2, "Classify_Bin_1", 0),
        (2, "Classify_Bin_2", 1),
    }


def test_classify_objects_uses_declared_area_shape_measurements():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                "PriorMeasurements",
                MeasurementsArtifactType,
                plan=_plan("PriorMeasurements", MeasurementsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        )
    )
    labels = np.array([[[1, 2], [0, 2]]], dtype=np.int32)
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1, 2),),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        ),
    )
    adapter.add_measurements(
        MeasurementTable(
            name="PriorMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "feature_name": "AreaShape_Area",
                        "result_value": 0.5,
                    },
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 2,
                        "feature_name": "AreaShape_Area",
                        "result_value": 1.5,
                    },
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("feature_name", str),
                    FieldSpec("result_value", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        )
    )
    classify_objects = CellProfilerModule.require_module(
        "ClassifyObjects"
    ).require_callable()
    executor = _executor(
        classify_objects,
        adapter,
        (ArtifactSpec.output(MEASUREMENTS, MeasurementsArtifactType),),
        main_flow_inputs=(),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
            ArtifactSpec.input("PriorMeasurements", MeasurementsArtifactType),
        ),
    )

    executor(
        ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(np.zeros((1, 2, 2), dtype=np.float32), None),
        cellprofiler_runtime=adapter,
        measurement_feature="AreaShape_Area",
        bin_choice=ClassificationBinChoice.EVEN,
        bin_count=2,
        low_threshold=0.0,
        high_threshold=2.0,
        dtype_config=DtypeConfig(),
    )

    assert _wide_measurement_feature_values(
        _output_measurements(adapter, MEASUREMENTS).rows,
        (
            "Classify_Bin_1_NumObjectsPerBin",
            "Classify_Bin_2_NumObjectsPerBin",
        ),
    ) == {
        ("Classify_Bin_1_NumObjectsPerBin", 1),
        ("Classify_Bin_2_NumObjectsPerBin", 1),
    }


def test_classify_objects_binds_custom_threshold_and_named_low_high_bins():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                "PriorMeasurements",
                MeasurementsArtifactType,
                plan=_plan("PriorMeasurements", MeasurementsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        )
    )
    labels = np.array([[[1, 2], [3, 0]]], dtype=np.int32)
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1, 2, 3),),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        ),
    )
    adapter.add_measurements(
        MeasurementTable(
            name="PriorMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "feature_name": "Intensity_MaxIntensity_OrigGreen",
                        "result_value": 0.05,
                    },
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 2,
                        "feature_name": "Intensity_MaxIntensity_OrigGreen",
                        "result_value": 0.15,
                    },
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 3,
                        "feature_name": "Intensity_MaxIntensity_OrigGreen",
                        "result_value": 0.80,
                    },
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("feature_name", str),
                    FieldSpec("result_value", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        )
    )
    classify_objects = CellProfilerModule.require_module(
        "ClassifyObjects"
    ).require_callable()
    executor = _executor(
        classify_objects,
        adapter,
        (ArtifactSpec.output(MEASUREMENTS, MeasurementsArtifactType),),
        main_flow_inputs=(),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
            ArtifactSpec.input("PriorMeasurements", MeasurementsArtifactType),
        ),
    )

    executor(
        ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(np.zeros((1, 2, 2), dtype=np.float32), None),
        cellprofiler_runtime=adapter,
        measurement_feature="Intensity_MaxIntensity_OrigGreen",
        bin_choice=ClassificationBinChoice.CUSTOM,
        bin_count=3,
        low_threshold=0.0,
        high_threshold=1.0,
        wants_low_bin=True,
        wants_high_bin=True,
        custom_thresholds=(0.2,),
        bin_names=("PH3Neg", "PH3Pos"),
        dtype_config=DtypeConfig(),
    )
    rows = _output_measurements(adapter, MEASUREMENTS).rows

    assert _wide_measurement_feature_values(
        rows,
        (
            "Classify_PH3Neg_NumObjectsPerBin",
            "Classify_PH3Pos_NumObjectsPerBin",
        ),
    ) == {
        ("Classify_PH3Neg_NumObjectsPerBin", 2),
        ("Classify_PH3Pos_NumObjectsPerBin", 1),
    }
    assert {
        value
        for value in _wide_measurement_feature_values(
            rows,
            ("Classify_PH3Neg", "Classify_PH3Pos"),
            identity_fields=("object_label",),
        )
        if value[-1] == 1
    } == {
        (1, "Classify_PH3Neg", 1),
        (2, "Classify_PH3Neg", 1),
        (3, "Classify_PH3Pos", 1),
    }


def test_classify_objects_binds_repeated_single_measurement_rules():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                "PriorMeasurements",
                MeasurementsArtifactType,
                plan=_plan("PriorMeasurements", MeasurementsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        )
    )
    labels = np.array([[[1, 2], [0, 0]]], dtype=np.int32)
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1, 2),),
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        ),
    )
    adapter.add_measurements(
        MeasurementTable(
            name="PriorMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "feature_name": "AreaShape_Area",
                        "result_value": 4.0,
                    },
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 2,
                        "feature_name": "AreaShape_Area",
                        "result_value": 12.0,
                    },
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 1,
                        "feature_name": "Intensity_MeanIntensity_DNA",
                        "result_value": 0.02,
                    },
                    {
                        "slice_index": 0,
                        "object_name": NUCLEI,
                        "object_label": 2,
                        "feature_name": "Intensity_MeanIntensity_DNA",
                        "result_value": 0.2,
                    },
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("feature_name", str),
                    FieldSpec("result_value", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        )
    )
    classify_objects = CellProfilerModule.require_module(
        "ClassifyObjects"
    ).require_callable()
    executor = _executor(
        classify_objects,
        adapter,
        (ArtifactSpec.output(MEASUREMENTS, MeasurementsArtifactType),),
        main_flow_inputs=(),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
            ArtifactSpec.input("PriorMeasurements", MeasurementsArtifactType),
        ),
    )

    executor(
        ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(np.zeros((1, 2, 2), dtype=np.float32), None),
        cellprofiler_runtime=adapter,
        classification_rules=(
            SingleMeasurementClassificationRule(
                measurement_feature="AreaShape_Area",
                bin_choice=ClassificationBinChoice.CUSTOM,
                custom_thresholds=(0.0, 5.0, 20.0),
                bin_names=("Small", "Large"),
            ),
            SingleMeasurementClassificationRule(
                measurement_feature="Intensity_MeanIntensity_DNA",
                bin_choice=ClassificationBinChoice.CUSTOM,
                custom_thresholds=(0.05,),
                wants_low_bin=True,
                wants_high_bin=True,
                bin_names=("White", "Red"),
            ),
        ),
        dtype_config=DtypeConfig(),
    )
    rows = _output_measurements(adapter, MEASUREMENTS).rows

    assert _wide_measurement_feature_values(
        rows,
        (
            "Classify_Small_NumObjectsPerBin",
            "Classify_Large_NumObjectsPerBin",
            "Classify_White_NumObjectsPerBin",
            "Classify_Red_NumObjectsPerBin",
        ),
    ) == {
        ("Classify_Small_NumObjectsPerBin", 1),
        ("Classify_Large_NumObjectsPerBin", 1),
        ("Classify_White_NumObjectsPerBin", 1),
        ("Classify_Red_NumObjectsPerBin", 1),
    }
    assert _wide_measurement_feature_values(
        rows,
        (
            "Classify_Small",
            "Classify_Large",
            "Classify_White",
            "Classify_Red",
        ),
        identity_fields=("object_label",),
    ) == {
        (1, "Classify_Small", 1),
        (1, "Classify_Large", 0),
        (1, "Classify_White", 1),
        (1, "Classify_Red", 0),
        (2, "Classify_Small", 0),
        (2, "Classify_Large", 1),
        (2, "Classify_White", 0),
        (2, "Classify_Red", 1),
    }


def test_classify_objects_slices_runtime_measurements_with_label_stack():
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                "PriorMeasurements",
                MeasurementsArtifactType,
                plan=_plan("PriorMeasurements", MeasurementsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        ),
        plane_projection=RuntimePlaneProjection.stack(2),
    )
    labels = np.array(
        [
            [[1, 2], [0, 0]],
            [[3, 4], [0, 0]],
        ],
        dtype=np.int32,
    )
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(
                declared_object_id_domains=((1, 2), (3, 4)),
                scope=ObjectLabelDomainScope.PLANE,
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei_t0.tif", "/src/Nuclei_t1.tif"),
            ),
        ),
    )
    adapter.add_measurements(
        MeasurementTable(
            name="PriorMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": 0 if label in (1, 2) else 1,
                        "object_name": NUCLEI,
                        "object_label": label,
                        "area": float(label),
                    }
                    for label in (1, 2, 3, 4)
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("area", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, NUCLEI),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei_t0.tif", "/src/Nuclei_t1.tif"),
            ),
        )
    )
    classify_objects = CellProfilerModule.require_module(
        "ClassifyObjects"
    ).require_callable()
    executor = _executor(
        classify_objects,
        adapter,
        (ArtifactSpec.output(MEASUREMENTS, MeasurementsArtifactType),),
        main_flow_inputs=(),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
            ArtifactSpec.input("PriorMeasurements", MeasurementsArtifactType),
        ),
    )

    result = executor(
        ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(np.zeros((2, 2, 2), dtype=np.float32), None),
        cellprofiler_runtime=adapter,
        measurement_feature="AreaShape_Area",
        bin_choice=ClassificationBinChoice.EVEN,
        bin_count=2,
        low_threshold=0.0,
        high_threshold=4.0,
        dtype_config=DtypeConfig(),
    )
    measurements = _output_measurements(adapter, MEASUREMENTS)

    assert result.shape == (2, 2, 2)
    assert _wide_measurement_feature_values(
        measurements.rows,
        (
            "Classify_Bin_1_NumObjectsPerBin",
            "Classify_Bin_2_NumObjectsPerBin",
        ),
        identity_fields=("slice_index",),
    ) == {
        (0, "Classify_Bin_1_NumObjectsPerBin", 2),
        (0, "Classify_Bin_2_NumObjectsPerBin", 0),
        (1, "Classify_Bin_1_NumObjectsPerBin", 0),
        (1, "Classify_Bin_2_NumObjectsPerBin", 2),
    }
    assert {
        value
        for value in _wide_measurement_feature_values(
            measurements.rows,
            ("Classify_Bin_1", "Classify_Bin_2"),
            identity_fields=("object_label",),
        )
        if value[-1] == 1
    } == {
        (1, "Classify_Bin_1", 1),
        (2, "Classify_Bin_1", 1),
        (3, "Classify_Bin_2", 1),
        (4, "Classify_Bin_2", 1),
    }


def test_object_only_measurements_use_each_object_owned_reference_image(
    declaration_owned_cellprofiler_callable,
):
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                CELLS,
                ObjectLabelsArtifactType,
                plan=_plan(CELLS, ObjectLabelsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        ),
        plane_projection=RuntimePlaneProjection.stack(),
    )
    image = np.stack(
        [
            np.full((4, 5), 3.0, dtype=np.float32),
            np.full((4, 5), 9.0, dtype=np.float32),
        ]
    )
    nuclei = np.ones((4, 5), dtype=np.int32)
    cells = np.full((4, 5), 2, dtype=np.int32)
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=nuclei),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Nuclei.tif",),
            ),
        ),
    )
    adapter.add_objects(
        CELLS,
        ObjectLabelSet(
            name=CELLS,
            variant_data=ObjectLabelVariantData(labels=cells),
            domain=ObjectLabelDomain(declared_object_ids=(2,)),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/src/Cells.tif",),
            ),
        ),
    )
    seen_images = []

    @declared_processing_contract(ProcessingContract.PURE_2D)
    def measure_object_size_shape(
        image_arg, *, labels: ObjectLabelValue
    ) -> tuple[object, DataclassMeasurementColumnarRows]:
        image_pixels = np.asarray(image_payload_data(image_arg))
        label_pixels = object_label_dense_array(labels)
        seen_images.append((image_pixels.copy(), label_pixels.copy()))
        return image_arg, DataclassMeasurementColumnarRows(
            (
                AreaMeasurementRow(
                    object_id=int(label_pixels.max()),
                    Area=float(np.count_nonzero(label_pixels)),
                ),
            ),
            row_type=AreaMeasurementRow,
        )

    executor = _executor(
        declaration_owned_cellprofiler_callable(measure_object_size_shape),
        adapter,
        (ArtifactSpec.output(MEASUREMENTS, MeasurementsArtifactType),),
        main_flow_inputs=(),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
            ArtifactSpec.input(
                CELLS, ObjectLabelsArtifactType, parameter_name="labels"
            ),
        ),
    )
    result = executor(image, cellprofiler_runtime=adapter)
    measurements = _output_measurements(adapter, MEASUREMENTS)

    assert len(seen_images) == 2
    for measurement_image, measurement_labels in seen_images:
        assert measurement_image.shape == measurement_labels.shape == (4, 5)
    np.testing.assert_array_equal(result, image)
    assert _measurement_rows_for_assertion(measurements) == [
        {
            "object_id": 1,
            "AreaShape_Area": 20.0,
            "object_name": NUCLEI,
        },
        {
            "object_id": 1,
            "AreaShape_Area": 20.0,
            "object_name": CELLS,
        },
    ]


def test_cellprofiler_module_executor_measures_each_declared_image_and_object(
    declaration_owned_cellprofiler_callable,
):
    dna = np.full((4, 5), 3.0, dtype=np.float32)
    ph3 = np.full((4, 5), 9.0, dtype=np.float32)
    nuclei = np.ones((4, 5), dtype=np.int32)
    cells = np.full((4, 5), 2, dtype=np.int32)
    adapter = _source_bound_image_adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                CELLS,
                ObjectLabelsArtifactType,
                plan=_plan(CELLS, ObjectLabelsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        ),
        {DNA_IMAGE: dna, "PH3": ph3},
    )
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=nuclei),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(0, 0),
                source_shape_yx=nuclei.shape,
            ),
        ),
    )
    adapter.add_objects(
        CELLS,
        ObjectLabelSet(
            name=CELLS,
            variant_data=ObjectLabelVariantData(labels=cells),
            domain=ObjectLabelDomain(declared_object_ids=(2,)),
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(0, 0),
                source_shape_yx=cells.shape,
            ),
        ),
    )
    seen = []

    @declared_processing_contract(ProcessingContract.FLEXIBLE)
    def measure_object_intensity(
        image_arg, *, labels: ObjectLabelValue
    ) -> tuple[object, DataclassMeasurementColumnarRows]:
        image_pixels = image_payload_data(image_arg)
        label_pixels = object_label_dense_array(labels)
        seen.append((float(image_pixels.mean()), int(label_pixels.max())))
        return image_pixels, DataclassMeasurementColumnarRows(
            (
                IntensityMeasurementRow(
                    mean_intensity=float(image_pixels.mean()),
                    object_label=int(label_pixels.max()),
                ),
            ),
            row_type=IntensityMeasurementRow,
        )

    executor = _executor(
        declaration_owned_cellprofiler_callable(measure_object_intensity),
        adapter,
        (ArtifactSpec.output(MEASUREMENTS, MeasurementsArtifactType),),
        main_flow_inputs=(
            ArtifactSpec.input(DNA_IMAGE, ImageArtifactType),
            ArtifactSpec.input("PH3", ImageArtifactType),
        ),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
            ArtifactSpec.input(
                CELLS, ObjectLabelsArtifactType, parameter_name="labels"
            ),
        ),
    )
    result = executor(
        _source_bound_image_stack({DNA_IMAGE: dna, "PH3": ph3}),
        cellprofiler_runtime=adapter,
    )
    measurements = _output_measurements(adapter, MEASUREMENTS)

    np.testing.assert_array_equal(result, np.stack((dna, ph3)))
    assert seen == [(3.0, 1), (3.0, 2), (9.0, 1), (9.0, 2)]
    assert _measurement_rows_for_assertion(measurements) == [
        {
            "Intensity_MeanIntensity_DNA": 3.0,
            "object_label": 1,
            "object_name": NUCLEI,
            "source_image_name": DNA_IMAGE,
        },
        {
            "Intensity_MeanIntensity_DNA": 3.0,
            "object_label": 2,
            "object_name": CELLS,
            "source_image_name": DNA_IMAGE,
        },
        {
            "Intensity_MeanIntensity_PH3": 9.0,
            "object_label": 1,
            "object_name": NUCLEI,
            "source_image_name": "PH3",
        },
        {
            "Intensity_MeanIntensity_PH3": 9.0,
            "object_label": 2,
            "object_name": CELLS,
            "source_image_name": "PH3",
        },
    ]
    assert measurements.source_image_name is None


def test_real_object_intensity_batch_preserves_source_and_object_ownership():
    from openhcs.processing.backends.cellprofiler.intensity import (
        measure_object_intensity,
    )

    dna = np.full((4, 5), 3.0, dtype=np.float32)
    ph3 = np.full((4, 5), 9.0, dtype=np.float32)
    nuclei = np.ones((4, 5), dtype=np.int32)
    cells = np.full((4, 5), 2, dtype=np.int32)
    adapter = _source_bound_image_adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                CELLS,
                ObjectLabelsArtifactType,
                plan=_plan(CELLS, ObjectLabelsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        ),
        {DNA_IMAGE: dna, "PH3": ph3},
    )
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=nuclei),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(0, 0),
                source_shape_yx=nuclei.shape,
            ),
        ),
    )
    adapter.add_objects(
        CELLS,
        ObjectLabelSet(
            name=CELLS,
            variant_data=ObjectLabelVariantData(labels=cells),
            domain=ObjectLabelDomain(declared_object_ids=(2,)),
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(0, 0),
                source_shape_yx=cells.shape,
            ),
        ),
    )
    executor = _executor(
        measure_object_intensity,
        adapter,
        (ArtifactSpec.output(MEASUREMENTS, MeasurementsArtifactType),),
        main_flow_inputs=(
            ArtifactSpec.input(DNA_IMAGE, ImageArtifactType),
            ArtifactSpec.input("PH3", ImageArtifactType),
        ),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="labels"
            ),
            ArtifactSpec.input(
                CELLS, ObjectLabelsArtifactType, parameter_name="labels"
            ),
        ),
    )

    executor(
        _source_bound_image_stack({DNA_IMAGE: dna, "PH3": ph3}),
        cellprofiler_runtime=adapter,
    )

    rows = tuple(_output_measurements(adapter, MEASUREMENTS).rows.iter_row_mappings())
    assert {(row["object_name"], row["source_image_name"]) for row in rows} == {
        (NUCLEI, DNA_IMAGE),
        (CELLS, DNA_IMAGE),
        (NUCLEI, "PH3"),
        (CELLS, "PH3"),
    }


def test_cellprofiler_object_only_executor_does_not_iterate_image_stack(
    declaration_owned_cellprofiler_callable,
):
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                CELLS,
                ObjectLabelsArtifactType,
                plan=_plan(CELLS, ObjectLabelsArtifactType),
            ),
            _output_binding(
                "Cytoplasm",
                ObjectLabelsArtifactType,
                plan=_plan("Cytoplasm", ObjectLabelsArtifactType),
            ),
        ),
        plane_projection=RuntimePlaneProjection.stack(),
    )
    nuclei = np.ones((4, 5), dtype=np.int32)
    cells = np.full((4, 5), 2, dtype=np.int32)
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=nuclei),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
    )
    adapter.add_objects(
        CELLS,
        ObjectLabelSet(
            name=CELLS,
            variant_data=ObjectLabelVariantData(labels=cells),
            domain=ObjectLabelDomain(declared_object_ids=(2,)),
        ),
    )
    seen_images = []

    def identify_tertiary_objects(
        image_arg,
        *,
        primary_labels: ObjectLabelValue,
        secondary_labels: ObjectLabelValue,
    ):
        image_pixels = np.asarray(image_payload_data(image_arg))
        primary_pixels = object_label_dense_array(primary_labels)
        secondary_pixels = object_label_dense_array(secondary_labels)
        seen_images.append(image_pixels.shape)
        return ObjectLabelSet(
            name="Cytoplasm",
            variant_data=ObjectLabelVariantData(
                labels=secondary_pixels - primary_pixels
            ),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        )

    identify_tertiary_objects.__processing_contract__ = ProcessingContract.PURE_2D

    executor = _executor(
        declaration_owned_cellprofiler_callable(identify_tertiary_objects),
        adapter,
        (
            ArtifactSpec.output_inheriting_group_scope(
                "Cytoplasm",
                ObjectLabelsArtifactType,
                ArtifactSpec.input(CELLS, ObjectLabelsArtifactType),
            ),
        ),
        main_flow_inputs=(),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                CELLS, ObjectLabelsArtifactType, parameter_name="secondary_labels"
            ),
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="primary_labels"
            ),
        ),
    )
    image = np.zeros((3, 4, 5), dtype=np.float32)
    result = executor(
        image,
        cellprofiler_runtime=adapter,
    )
    cytoplasm = _output_objects(adapter, "Cytoplasm")

    assert seen_images == [(4, 5)]
    np.testing.assert_array_equal(object_label_dense_array(result), cells - nuclei)
    np.testing.assert_array_equal(cytoplasm.labels, cells - nuclei)


def test_cellprofiler_module_executor_records_relationship_and_measurement_outputs(
    declaration_owned_cellprofiler_callable,
):
    adapter, _filemanager = _adapter(
        (
            _output_binding(
                CELLS,
                ObjectLabelsArtifactType,
                plan=_plan(CELLS, ObjectLabelsArtifactType),
            ),
            _output_binding(
                NUCLEI,
                ObjectLabelsArtifactType,
                plan=_plan(NUCLEI, ObjectLabelsArtifactType),
            ),
            _output_binding(
                PARENT_CHILD,
                RelationshipsArtifactType,
                plan=_plan(PARENT_CHILD, RelationshipsArtifactType),
            ),
            _output_binding(
                MEASUREMENTS,
                MeasurementsArtifactType,
                plan=_plan(MEASUREMENTS, MeasurementsArtifactType),
            ),
        ),
        source_bindings=StepSourceBindingsConfig(),
        plane_projection=RuntimePlaneProjection.stack(),
    )
    image = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/src/DNA.tif",),
        ),
    ).payload_with(np.zeros((2, 2), dtype=np.float32), None)
    cells = np.array([[1, 1], [0, 0]], dtype=np.int32)
    nuclei = np.array([[1, 0], [2, 0]], dtype=np.int32)
    adapter.add_objects(
        CELLS,
        ObjectLabelSet(
            name=CELLS,
            variant_data=ObjectLabelVariantData(labels=cells),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
    )
    adapter.add_objects(
        NUCLEI,
        ObjectLabelSet(
            name=NUCLEI,
            variant_data=ObjectLabelVariantData(labels=nuclei),
            domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
        ),
    )

    @declared_processing_contract(ProcessingContract.FLEXIBLE)
    @special_inputs("parent_labels", "child_labels")
    def relate_objects(
        image_arg,
        *,
        parent_labels: ObjectLabelValue,
        child_labels: ObjectLabelValue,
        calculate_distances: RelateObjectsDistanceMethod = (
            RelateObjectsDistanceMethod.NONE
        ),
        calculate_per_parent_means: bool = False,
    ):
        np.testing.assert_array_equal(
            image_payload_data(image_arg), image_payload_data(image)
        )
        np.testing.assert_array_equal(object_label_dense_array(parent_labels), cells)
        np.testing.assert_array_equal(object_label_dense_array(child_labels), nuclei)
        return (
            image_arg,
            DirectedObjectRelationshipPayload(source_ids=(1, 1), target_ids=(1, 2)),
            MeasurementSparseColumnarRows({}, fields=()),
        )

    executor = _executor(
        declaration_owned_cellprofiler_callable(relate_objects),
        adapter,
        (
            ArtifactSpec.output(
                PARENT_CHILD,
                RelationshipsArtifactType,
                relations=(
                    ObjectRelationshipDeclaration(
                        source=ArtifactSpec.input(
                            CELLS, ObjectLabelsArtifactType
                        ).ref(),
                        target=ArtifactSpec.input(
                            NUCLEI, ObjectLabelsArtifactType
                        ).ref(),
                        producer_module_number=1,
                        relationship_type="parent_child",
                        source_role="parent",
                        target_role="child",
                        source_id_field="parent_id",
                        target_id_field="child_id",
                        source_runtime_slice_offset=0,
                        target_runtime_slice_offset=0,
                    ),
                ),
            ),
            ArtifactSpec.output(MEASUREMENTS, MeasurementsArtifactType),
        ),
        runtime_artifact_inputs=(
            ArtifactSpec.input(
                CELLS, ObjectLabelsArtifactType, parameter_name="parent_labels"
            ),
            ArtifactSpec.input(
                NUCLEI, ObjectLabelsArtifactType, parameter_name="child_labels"
            ),
        ),
        main_flow_inputs=(),
    )
    result = executor(image, cellprofiler_runtime=adapter)
    relationship = _output_relationship(adapter, PARENT_CHILD)
    measurements = _output_measurements(adapter, MEASUREMENTS)

    assert result is image
    assert relationship.declaration.source.name == CELLS
    assert relationship.declaration.target.name == NUCLEI
    assert relationship.payload.source_ids == (1, 1)
    assert relationship.payload.target_ids == (1, 2)
    assert relationship.declaration.producer_module_number == 1
    assert measurements.subject.object_name is None
    assert _measurement_rows_for_assertion(measurements) == [
        {
            "object_name": CELLS,
            "object_label": 1,
            "Children_Nuclei_Count": 2,
        },
        {
            "object_name": NUCLEI,
            "object_label": 1,
            "Parent_Cells": 1,
        },
        {
            "object_name": NUCLEI,
            "object_label": 2,
            "Parent_Cells": 1,
        },
    ]
    assert object_measurement_tables_for_test(adapter, CELLS) == ()

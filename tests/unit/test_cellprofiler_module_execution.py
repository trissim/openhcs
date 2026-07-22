from dataclasses import dataclass, replace
import sqlite3
from types import MappingProxyType, SimpleNamespace
from typing import Annotated, cast

import numpy as np
import pytest
import skimage.measure
import skimage.morphology

import openhcs.processing.backends.cellprofiler.secondary as iso
import openhcs.processing.backends.cellprofiler.secondary as ito
from openhcs.constants.constants import (
    AllComponents,
    Backend,
    GroupBy,
    MemoryType,
    VariableComponents,
)
from openhcs.core.aligned_image_payload import (
    AlignedImageSliceContext,
    AlignedImageStack,
    ImageOutputBundle,
    ImagePayloadBundleContext,
    ImagePayloadExecutionMode,
    ImagePayloadSliceProjector,
    aligned_image_stack_kwargs,
    compose_aligned_image_payload,
    pack_aligned_image_outputs,
    payload_slice_count,
    payload_slices_for_alignment,
)
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSidecarRole,
    ArtifactSpec,
    ArtifactSpecRelation,
    ArtifactType,
    GroupLineageSourceRelation,
    ImageArtifactType,
    InputStackBroadcastSourceRelation,
    MeasurementsArtifactType,
    ObjectLineageArtifactType,
    ObjectLabelsArtifactType,
    RelationshipsArtifactType,
    SourceStackLineageSourceRelation,
    SpatialGridArtifactType,
)
from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
from openhcs.core.callable_contract import (
    CallableContract,
    CallableMetadata,
    attach_callable_contract_metadata,
    runtime_image_execution_mode,
)
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
    RuntimeExecutionAxisScope,
)
from openhcs.core.config import DtypeConfig
from openhcs.core.equivalence.keys import RuntimeMeasurementSourcePair
from openhcs.core.function_patterns import InvocationArtifactInputEdgePlan
from openhcs.core.measurement_image_alignment import (
    MeasurementImageLabelAlignmentStrategy,
    MeasurementLabelSourceAlignmentStrategy,
    PreparedMeasurementObjectLabels,
)
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    DataclassMeasurementColumnarRows,
    MeasurementSparseColumnarRows,
)
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    object_label_input_execution_mode,
    object_label_input_execution_mode_from_callable,
    runtime_bound_parameters,
    special_inputs,
)
from openhcs.core.runtime_adapters import RuntimeFunctionInvocationRequest
from openhcs.core.runtime_artifact_queries import (
    MeasurementTableAxisProjection,
    MeasurementTableUnion,
    RuntimeArtifactQueryContext,
)
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.runtime_batch_contracts import SliceIndexRuntimeParameter
from openhcs.core.runtime_image_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
    MaskedImagePayload,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectLabelValue,
    ObjectLabelVariantData,
    object_label_dense_array,
)
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.core.runtime_sparse_labels import SparseIJVLabelRows
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
    measurement_row_mapping,
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
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
    object_label_parent_child_payload,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisProjector,
    RuntimePlaneAxisValueProjection,
    RuntimePlaneProjection,
)
from openhcs.core.runtime_spatial_grid import (
    SpatialGridOrdering,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_slice_projection import (
    RuntimeSliceProjection,
    RuntimeSliceProjectionDeclarationError,
)
from openhcs.core.runtime_spatial_grid import (
    SpatialGrid,
)
from openhcs.core.runtime_stores import (
    RuntimeArtifactBatch,
    RuntimeValueStore,
    StoredRuntimeValue,
)
from openhcs.core.runtime_tabular_values import (
    ColumnarRows,
)
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    ComponentSelector,
    NamedSourceBinding,
    SourceBindingRuntimeContext,
)
from openhcs.core.source_image_provenance import (
    SourceImageProvenancePlanes,
)
from openhcs.core.source_matching import (
    SourceImageSetIdentityPolicy,
)
from openhcs.core.source_metadata import SourceVoxelSpacing
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.core.steps.function_runtime import (
    FunctionOutputContextStrategy,
    PatternGroupData,
    PatternGroupRuntime,
)
from openhcs.core.steps.stream_component_semantics import (
    StreamImagePayloadMetadataProjector,
)
from openhcs.interop.cellprofiler.analyst_export import (
    CPASQLiteRenderer,
    CellProfilerAnalystProjectionBuilder,
    CellProfilerDatabaseExportSettings,
    CellProfilerObjectTableMode,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_measurement_features import (
    ObjectCountFeature,
)
from openhcs.interop.cellprofiler.runtime.adapter import (
    CellProfilerRuntimeAdapter,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
    RuntimeArtifactInputRequest,
    RuntimeArtifactTypeStrategy,
)
from openhcs.interop.cellprofiler.runtime.function_contract_execution import (
    CellProfilerFunctionContractExecutor,
)
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerImageRequest,
    CellProfilerMeasurementImage,
    CellProfilerMeasurementImageDomain,
    CellProfilerSliceAlignedValues,
)
from openhcs.interop.cellprofiler.runtime.main_flow import (
    cellprofiler_main_flow_output,
)
from openhcs.interop.cellprofiler.runtime.measurement_execution_support import (
    ObjectMeasurementOutputRecorder,
    ObjectMeasurementOutputTimings,
    object_measurement_runtime_inputs,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    MeasurementFeatureRecord,
    measurement_table_for_module,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    CellProfilerObjectCoreMeasurementFeature,
    ObjectLocationMeasurementRows,
    measurement_table_rows,
)
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerModuleExecutor,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_execution import (
    CellProfilerObjectMeasurementExecutionPolicy,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_completion import (
    ObjectMeasurementRowCompletionSchema,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    CellProfilerObjectMeasurementRowPolicy,
    CompactObjectMeasurementRowIdentityPolicy,
    DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    CellProfilerMeasurementVector,
)
from openhcs.interop.cellprofiler.runtime.output_record_request import (
    CellProfilerOutputRecordRequest,
)
from openhcs.interop.cellprofiler.runtime.output_recording import (
    CellProfilerOutputRecorder,
)
from openhcs.interop.cellprofiler.runtime.relationship_measurement_rows import (
    DirectParentReferenceFeatureDeclaration,
    DirectParentReferenceMeasurementFeature,
    RelationshipMeasurementRows,
)
from openhcs.processing.backends.cellprofiler.alignment import (
    AlignModule,
    AlignShiftMeasurement,
)
from openhcs.processing.backends.cellprofiler.area_occupied import (
    MeasureImageAreaOccupiedBinaryModule,
)
from openhcs.processing.backends.cellprofiler.classification import (
    ClassificationResult,
    ClassifyObjectsSingleMeasurementModule,
)
from openhcs.processing.backends.cellprofiler.colocalization import (
    MeasureColocalizationModule,
    ObjectColocalizationMeasurements,
)
from openhcs.processing.backends.cellprofiler.color import (
    color_to_gray,
    gray_to_color,
)
from openhcs.processing.backends.cellprofiler.color import (
    color_to_gray as openhcs_color_to_gray,
)
from openhcs.processing.backends.cellprofiler.feature_enhancement import (
    EnhanceMethod,
    NeuriteMethod,
    SpeckleAccuracy,
    enhance_or_suppress_features,
)
from openhcs.processing.backends.cellprofiler.grid import (
    GridDefinition,
    GridShapeRequest,
    NaturalGridShapeStrategy,
    define_grid_automatic,
    identify_objects_in_grid,
)
from openhcs.processing.backends.cellprofiler.illumination import (
    CorrectIlluminationApplyModule,
)
from openhcs.processing.backends.cellprofiler.image_geometry import tile
from openhcs.processing.backends.cellprofiler.intensity import (
    MeasureObjectIntensityModule,
    measure_object_intensity,
)
from openhcs.processing.backends.cellprofiler.intensity_distribution import (
    MeasureObjectIntensityDistributionModule,
)
from openhcs.processing.backends.cellprofiler.measurement_math import (
    CalculateMathInputPolicy,
    CalculateMathModule,
    MathOperation,
    calculate_math,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    MaskObjectsModule,
    closing,
    erode_image,
    mask_objects,
    resize_objects_3d,
)
from openhcs.processing.backends.cellprofiler.neighbors import (
    DistanceMethod,
    NeighborMeasurements,
)
from openhcs.processing.backends.cellprofiler.neighbors import (
    MeasureObjectNeighborsModule,
    measure_object_neighbors,
)
from openhcs.processing.backends.cellprofiler.object_filtering import (
    FilterMethod,
    FilterMode,
    FilterObjectsRemovedObjectSourceRelation,
    FilterObjectsRuntimeInputPlan,
    PerObjectAssignment,
    filter_objects,
)
from openhcs.processing.backends.cellprofiler.object_images import (
    convert_objects_to_image,
)
from openhcs.processing.backends.cellprofiler.primary_objects import (
    _remap_object_label_variant_after_final_relabel,
    identify_primary_objects,
)
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
    RelateObjectsModule,
    RelateObjectsResult,
    relate_objects,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    DistanceMaskedSegmentationStrategy,
    IdentifySecondaryObjectsModule,
    IdentifyTertiaryObjectsModule,
    PropagationSegmentationStrategy,
    SecondarySegmentationRequest,
    _filter_labels,
    _secondary_seed_labels,
)
from openhcs.processing.backends.cellprofiler.shape import (
    MeasureObjectSizeShapeModule,
    ShapeObjectMeasurementRows,
    _surface_area,
    _surface_areas_3d_from_labels,
    measure_object_size_shape,
)
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
)
from openhcs.processing.backends.cellprofiler.texture import (
    MeasureTextureModule,
    measure_texture_objects,
)
from openhcs.processing.backends.cellprofiler.tracking import (
    TrackObjectsModule,
    TrackingImageMeasurement,
    TrackingObjectMeasurement,
    track_objects,
)
from openhcs.processing.backends.cellprofiler.watershed import (
    WatershedMethod,
    WatershedModule,
    WatershedStats,
    watershed_cellprofiler4,
)
from openhcs.processing.backends.cellprofiler.worms import (
    OverlapStyle,
    UntangleWormsModule,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
    Pure2DAuxiliaryOutputAggregator,
    Pure2DInputSlicer,
    Pure2DSliceResultBatch,
)
from tests.unit.cellprofiler_runtime_test_support import (
    cellprofiler_runtime_adapter_for_test,
    cellprofiler_runtime_input_edge_for_test,
    runtime_adapter_request_for_test,
)


def _compiled_callable_contract(
    raw_func,
    *,
    artifact_inputs: tuple[ArtifactSpec, ...] | None = None,
    artifact_outputs: tuple[ArtifactSpec, ...] | None = None,
) -> CallableContract:
    raw_contract = CallableContract.from_callable(raw_func)
    metadata_changes = {}
    if artifact_inputs is not None:
        metadata_changes["artifact_inputs"] = artifact_inputs
    if artifact_outputs is not None:
        metadata_changes["artifact_outputs"] = artifact_outputs
    module_type = CellProfilerModule.for_function_name(raw_contract.function_name)
    if module_type is not None and module_type.uses_cellprofiler_runtime_adapter():
        metadata_changes["runtime_adapter"] = (
            CellProfilerRuntimeAdapter.runtime_adapter_spec()
        )
    return replace(
        raw_contract,
        metadata=replace(raw_contract.metadata, **metadata_changes),
    )


def _module_type_for_contract(
    contract: CallableContract,
) -> type[CellProfilerModule]:
    module_type = CellProfilerModule.for_function_name(contract.function_name)
    if module_type is None:
        raise AssertionError(
            f"No nominal CellProfiler module owns {contract.function_name!r}."
        )
    return module_type


def _module_executor(
    contract: CallableContract,
) -> CellProfilerModuleExecutor:
    raw_func = contract.resolve_canonical_raw_callable()
    executor = CellProfilerModuleExecutor(
        raw_func=raw_func,
        callable_contract=contract,
    )
    module_type = _module_type_for_contract(contract)
    assert executor.raw_func is module_type.require_callable(contract.function_name)
    return executor


def test_default_invocation_keeps_compiled_source_bindings_outside_anchor_group() -> (
    None
):
    image_spec = ArtifactSpec.input("DF_image", ImageArtifactType)
    object_spec = ArtifactSpec.input("Tile_of_grid", ObjectLabelsArtifactType)
    measurement_spec = ArtifactSpec.output(
        "MeasureObjectIntensity_measurements",
        MeasurementsArtifactType,
        relations=(GroupLineageSourceRelation(source=object_spec.ref()),),
    )
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("MeasureObjectIntensity").require_callable(),
        artifact_inputs=(
            *((image_spec,)),
            *((object_spec,)),
        ),
        artifact_outputs=(measurement_spec,),
    )
    executor = _module_executor(contract)
    channel_scope = ComponentGroupScope(
        ("1",),
        component=AllComponents.CHANNEL,
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        artifact_input_edges=(
            _artifact_input_edge_for_test(
                image_spec,
                invocation_scope=channel_scope,
                stored=False,
            ),
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=object_spec.name,
                    path=f"/artifacts/{object_spec.name}",
                    artifact_type=object_spec.artifact_type,
                ),
                invocation_scope=channel_scope,
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(channel_scope,),
                consumer_variable_components=(),
            ),
        ),
        source_bindings=(
            NamedSourceBinding(
                alias="BF_image",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias=image_spec.name,
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
            ),
        ),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=AllComponents.CHANNEL,
            value="1",
        ),
    )
    _activate_runtime_contract(contract, runtime)

    active_inputs = executor.active_input_specs(
        runtime,
    )

    assert active_inputs.specs == (image_spec, object_spec)


def test_active_inputs_restore_repeated_contract_roles_from_unique_storage() -> None:
    objects = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    contract = _compiled_callable_contract(
        _module_type_for_contract(MeasureObjectNeighborsModule).require_callable(),
        artifact_inputs=(objects, objects),
    )
    executor = _module_executor(contract)
    channel_scope = ComponentGroupScope(("1",), component=AllComponents.CHANNEL)
    runtime = _FakeCellProfilerRuntime(
        {},
        artifact_input_edges=(
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=objects.name,
                    path=f"/artifacts/{objects.name}",
                    artifact_type=objects.artifact_type,
                ),
                invocation_scope=channel_scope,
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(channel_scope,),
                consumer_variable_components=(),
            ),
        ),
    )
    _activate_runtime_contract(contract, runtime)

    active_inputs = executor.active_input_specs(
        runtime,
    )

    assert active_inputs.specs == (objects, objects)


def test_runtime_adapter_collapses_equal_duplicate_input_occurrences() -> None:
    cells_spec = ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    cells = ObjectLabelSet(
        name=cells_spec.name,
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[1]], dtype=np.int32)
        ),
    )
    edge = _artifact_input_edge_for_test(cells_spec)
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={cells_spec.name: cells},
        artifact_input_edges=(edge, edge),
    )

    assert runtime.get_objects(cells_spec.name) == cells


def test_active_inputs_exclude_unselected_independent_contract_roles() -> None:
    first = ArtifactSpec.input("OrigStain1", ImageArtifactType)
    second = ArtifactSpec.input("OrigStain2", ImageArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module(
            "CorrectIlluminationCalculate"
        ).require_callable(),
        artifact_inputs=(first, second),
    )
    executor = _module_executor(contract)
    channel_scope = ComponentGroupScope(("1",), component=AllComponents.CHANNEL)
    runtime = _FakeCellProfilerRuntime(
        {},
        artifact_input_edges=(
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name=first.name,
                    path=f"/artifacts/{first.name}",
                    artifact_type=first.artifact_type,
                ),
                invocation_scope=channel_scope,
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(channel_scope,),
                consumer_variable_components=(),
            ),
        ),
    )
    _activate_runtime_contract(contract, runtime)

    assert executor.active_input_specs(runtime).specs == (first,)


def test_output_recording_preserves_complete_callable_return_abi() -> None:
    first = ArtifactSpec.output("Corrected1", ImageArtifactType)
    second = ArtifactSpec.output("Corrected2", ImageArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module(
            "CorrectIlluminationApply"
        ).require_callable(),
        artifact_outputs=(first, second),
    )
    executor = _module_executor(contract)
    projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        axis_size=2,
    )
    first_value = ImagePayloadMetadata(source_image_names=("Corrected1",)).payload_with(
        np.zeros((4, 5), dtype=np.float32),
        None,
    )
    second_value = ImagePayloadMetadata(
        source_image_names=("Corrected2",)
    ).payload_with(
        np.ones((4, 5), dtype=np.float32),
        None,
    )
    stack = pack_aligned_image_outputs(
        (first_value, second_value),
        slice_contexts=(
            AlignedImageSliceContext.main_flow(
                output_key="Corrected1",
                artifact_kind=ImageArtifactType.value,
            ),
            AlignedImageSliceContext.main_flow(
                output_key="Corrected2",
                artifact_kind=ImageArtifactType.value,
            ),
        ),
    )
    image_request = CellProfilerImageRequest(
        payload=stack,
        source_image_name=None,
        image_count=1,
        plane_projection=projection,
    )
    invocation = RuntimeFunctionInvocationRequest(
        image=stack,
        kwargs={},
        source_image_name=None,
        image_count=1,
        plane_projection=projection,
    )

    outputs = CellProfilerOutputRecorder.record_module_outputs(
        callable_contract=executor.callable_contract,
        active_input_edges=(),
        adapter=_FakeCellProfilerRuntime({}),
        returned_values=MappingProxyType(
            {
                first.ref(): first_value,
                second.ref(): second_value,
            }
        ),
        matched_outputs=(),
        invocation=invocation,
        image_request=image_request,
        current_image=stack,
    )

    assert outputs == {
        first.ref(): first_value,
        second.ref(): second_value,
    }


def _output_from_input(
    output_name: str,
    input_name: str,
    *,
    output_type: type[ArtifactType] = ObjectLabelsArtifactType,
    input_type: type[ArtifactType] = ObjectLabelsArtifactType,
) -> ArtifactSpec:
    return ArtifactSpec.output(
        output_name,
        output_type,
        relations=(
            GroupLineageSourceRelation(
                source=ArtifactSpec.input(input_name, input_type).ref()
            ),
        ),
    )


def _measurement_output_for_objects(
    name: str,
    *object_specs: ArtifactSpec,
) -> ArtifactSpec:
    return ArtifactSpec.output(
        name,
        MeasurementsArtifactType,
        relations=tuple(
            ArtifactSpecRelation(source=object_spec.ref())
            for object_spec in object_specs
        ),
    )


def _relationship_output(
    parent: ArtifactSpec,
    child: ArtifactSpec,
) -> ArtifactSpec:
    declaration = ObjectRelationshipDeclaration(
        source=parent.ref(),
        target=child.ref(),
        producer_module_number=1,
        relationship_type="parent_child",
        source_role="parent",
        target_role="child",
        source_id_field="parent_id",
        target_id_field="child_id",
        source_runtime_slice_offset=0,
        target_runtime_slice_offset=0,
    )
    return ArtifactSpec.output(
        declaration.artifact_name(),
        RelationshipsArtifactType,
        relations=(declaration,),
    )


def _artifact_input_edge_for_test(
    spec: ArtifactSpec,
    *,
    invocation_scope: ComponentGroupScope | None = None,
    stored: bool = True,
) -> InvocationArtifactInputEdgePlan:
    scope = invocation_scope or ComponentGroupScope.ungrouped()
    edge = replace(
        cellprofiler_runtime_input_edge_for_test(
            ArtifactInputPlan(
                name=spec.name,
                path=f"/artifacts/{spec.name}",
                artifact_type=spec.artifact_type,
                sidecar_role=spec.sidecar_role,
            ),
            spec=spec,
            invocation_scope=scope,
            producer_selection_scope=ComponentGroupScope.ungrouped(),
            component_scopes=() if scope.is_ungrouped else (scope,),
            consumer_variable_components=(),
        ),
        spec=spec,
    )
    if stored:
        return edge
    return replace(edge, storage_plan=None, projection=None)


def _artifact_output_plan(spec: ArtifactSpec) -> ArtifactOutputPlan:
    return ArtifactOutputPlan(
        name=spec.name,
        path=f"/artifacts/{spec.name}",
        artifact_type=spec.artifact_type,
        sidecar_role=spec.sidecar_role,
        materialization=spec.materialization,
        relations=spec.relations,
    )


def _activate_runtime_contract(
    callable_contract: CallableContract,
    adapter: CellProfilerRuntimeAdapter,
) -> CellProfilerRuntimeAdapter:
    for edge in adapter.request.artifact_inputs.values():
        declared = callable_contract.artifact_inputs.by_ref(edge.spec.ref())
        if declared != edge.spec:
            raise ValueError(
                f"Test runtime input edge {edge.spec!r} is not declared by "
                f"callable {callable_contract.function_name!r}."
            )
    for plan in adapter.request.artifact_outputs.values():
        declared = callable_contract.artifact_outputs.by_ref(plan.ref())
        if declared is None or declared.relations != plan.relations:
            raise ValueError(
                f"Test runtime output plan {plan!r} is not declared exactly by "
                f"callable {callable_contract.function_name!r}."
            )
    source_binding_plan = adapter.request.source_binding_plan.for_artifact_refs(
        callable_contract.artifact_inputs.ref_set()
    )
    adapter.request = replace(
        adapter.request,
        callable_contract=callable_contract,
        source_binding_plan=source_binding_plan,
    )
    return adapter


def _run_module(
    executor: CellProfilerModuleExecutor,
    image,
    *,
    cellprofiler_runtime: CellProfilerRuntimeAdapter,
    **kwargs,
):
    _activate_runtime_contract(executor.callable_contract, cellprofiler_runtime)
    return executor(
        image,
        cellprofiler_runtime=cellprofiler_runtime,
        **kwargs,
    )


def _cellprofiler_output_record_request(
    *,
    callable_contract: CallableContract,
    artifact_input_edges: tuple[InvocationArtifactInputEdgePlan, ...] = (),
    output_plans: tuple[ArtifactOutputPlan, ...],
    **kwargs,
) -> CellProfilerOutputRecordRequest:
    spec = kwargs["spec"]
    source = kwargs.pop("source", None)
    source_image_name = kwargs.pop("source_image_name", None)
    source_aliases = kwargs.pop("source_aliases", ())
    source_image_payload = kwargs.pop("source_image_payload", None)
    if source is None:
        source = CellProfilerMeasurementImage(
            source_image_name=source_image_name,
            source_aliases=source_aliases,
            payload=source_image_payload,
        )
    kwargs.setdefault("current_image", source.payload)
    adapter = kwargs["adapter"]
    if adapter is None:
        adapter = _FakeCellProfilerRuntime({})
        kwargs["adapter"] = adapter
    if isinstance(adapter, _FakeCellProfilerRuntime):
        adapter.install_artifact_input_edges(artifact_input_edges)
        artifact_inputs = adapter.request.artifact_inputs
    else:
        indexed_input_edges = tuple(
            replace(edge, key=replace(edge.key, input_index=input_index))
            for input_index, edge in enumerate(
                (*adapter.request.artifact_inputs.values(), *artifact_input_edges)
            )
        )
        artifact_inputs = {edge.key: edge for edge in indexed_input_edges}
    artifact_outputs = dict(adapter.request.artifact_outputs)
    for plan in output_plans:
        artifact_outputs.setdefault(plan.ref(), plan)
    adapter.request = replace(
        adapter.request,
        artifact_inputs=artifact_inputs,
        artifact_outputs=artifact_outputs,
    )
    _activate_runtime_contract(callable_contract, adapter)
    return CellProfilerOutputRecordRequest(
        callable_contract=callable_contract,
        active_input_edges=tuple(adapter.request.artifact_inputs.values()),
        output_plan=adapter.request.require_artifact_output_plan(spec.ref()),
        source=source,
        **kwargs,
    )


def _record_output(
    request: CellProfilerOutputRecordRequest,
    spec: ArtifactSpec,
    payload: object,
) -> None:
    CellProfilerOutputRecorder.for_artifact_type(spec.artifact_type).record(
        replace(
            request,
            spec=spec,
            output_plan=request.adapter.request.require_artifact_output_plan(
                spec.ref()
            ),
            output_value=payload,
        )
    )


@dataclass(frozen=True, slots=True)
class _SyntheticObjectMeasurement:
    object_label: int
    value: float


@dataclass(frozen=True, slots=True)
class _RuntimeSliceObjectAdapter(RuntimePlaneAxisProjector):
    objects: ObjectLabelSet
    slice_index: int | None
    slice_count: int | None = None
    relationship: ObjectRelationship | None = None

    @property
    def axis_scope(self) -> RuntimeExecutionAxisScope:
        return RuntimeExecutionAxisScope.from_raw(
            "test-axis",
            component=None,
            value=None,
        )

    def get_objects(self, name: str, current_image=None) -> ObjectLabelSet:
        del name, current_image
        return self.objects

    def get_relationship(
        self,
        name: str,
        *,
        current_image=None,
    ) -> ObjectRelationship:
        del name, current_image
        if self.relationship is None:
            raise RuntimeError("Test adapter has no relationship payload.")
        return self.relationship

    def runtime_slice_plane_index(self) -> int | None:
        return self.slice_index

    def runtime_slice_axis_size(self) -> int | None:
        return self.slice_count


@dataclass(frozen=True, slots=True)
class _RuntimeProjectionObjectAdapter(RuntimePlaneAxisProjector):
    objects: ObjectLabelSet
    plane_projection: RuntimePlaneProjection

    def get_objects(self, name: str) -> ObjectLabelSet:
        del name
        return self.objects

    def runtime_slice_plane_index(self) -> int | None:
        return self.plane_projection.runtime_slice_plane_index()

    def runtime_slice_axis_size(self) -> int | None:
        return self.plane_projection.runtime_slice_axis_size()


def _measurement_image_for_labels(
    image,
    labels,
    *,
    label_payload=None,
    plane_projector=None,
    source_aliases=(),
    reference_domain=CellProfilerMeasurementImageDomain.SOURCE_IMAGE,
):
    source = CellProfilerMeasurementImage(
        source_image_name=None,
        source_aliases=source_aliases,
        payload=image,
        reference_domain=reference_domain,
    )
    return MeasurementImageLabelAlignmentStrategy.align(
        source.alignment_request(
            labels=labels,
            label_payload=label_payload,
            plane_projector=plane_projector,
        ).with_source_projected_image()
    )


def test_primary_image_policy_is_bound_to_module_declaration() -> None:
    for module_type in (MaskObjectsModule, TrackObjectsModule):
        assert module_type.primary_image_inputs.__self__ is module_type


def test_special_object_label_input_preserves_runtime_slice_domain() -> None:
    @special_inputs("guiding_labels")
    def consume_guiding_labels(image, guiding_labels: ObjectLabelValue):
        del guiding_labels
        return image

    consume_guiding_labels.__processing_contract__ = ProcessingContract.PURE_2D

    labels = np.array(
        [
            [[0, 1], [0, 0]],
            [[0, 0], [2, 0]],
        ],
        dtype=np.int32,
    )
    objects = ObjectLabelSet(
        name="Guides",
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    guides_spec = ArtifactSpec.input(
        "Guides", ObjectLabelsArtifactType, parameter_name="guiding_labels"
    )
    contract = _compiled_callable_contract(
        consume_guiding_labels,
        artifact_inputs=(guides_spec,),
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            objects={guides_spec.name: objects},
            artifact_input_edges=(
                _artifact_input_edge_for_test(
                    guides_spec,
                ),
            ),
            plane_projection=RuntimePlaneProjection.stack(2),
        ),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    bound = request.bind_parameters()

    assert isinstance(bound["guiding_labels"], ObjectLabelValue)
    np.testing.assert_array_equal(
        object_label_dense_array(
            RuntimeSliceProjection.kwargs_for_slice(
                bound,
                RuntimePlaneAxisValueProjection.from_selected_plane(
                    axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
                ),
            )["guiding_labels"]
        ),
        np.array([[0, 0], [2, 0]], dtype=np.int32),
    )


def test_special_object_label_input_preserves_declared_singleton_label_planes() -> None:
    @special_inputs("guiding_labels")
    def consume_guiding_labels(image, guiding_labels: ObjectLabelValue):
        del guiding_labels
        return image

    consume_guiding_labels.__processing_contract__ = ProcessingContract.PURE_2D

    labels = np.array(
        [
            [[[0, 1], [0, 0]]],
            [[[0, 0], [2, 0]]],
        ],
        dtype=np.int32,
    )
    objects = ObjectLabelSet(
        name="Guides",
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    guides_spec = ArtifactSpec.input(
        "Guides", ObjectLabelsArtifactType, parameter_name="guiding_labels"
    )
    contract = _compiled_callable_contract(
        consume_guiding_labels,
        artifact_inputs=(guides_spec,),
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            objects={guides_spec.name: objects},
            artifact_input_edges=(
                _artifact_input_edge_for_test(
                    guides_spec,
                ),
            ),
            plane_projection=RuntimePlaneProjection.stack(2),
        ),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    bound = request.bind_parameters()

    assert isinstance(bound["guiding_labels"], ObjectLabelValue)
    np.testing.assert_array_equal(
        object_label_dense_array(
            RuntimeSliceProjection.kwargs_for_slice(
                bound,
                RuntimePlaneAxisValueProjection.from_selected_plane(
                    axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=0, axis_size=2
                ),
            )["guiding_labels"]
        ),
        labels[0],
    )
    np.testing.assert_array_equal(
        object_label_dense_array(
            RuntimeSliceProjection.kwargs_for_slice(
                bound,
                RuntimePlaneAxisValueProjection.from_selected_plane(
                    axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
                ),
            )["guiding_labels"]
        ),
        labels[1],
    )


def test_special_object_label_input_preserves_projected_overlap_domains() -> None:
    @special_inputs("guiding_labels")
    def consume_guiding_labels(image, guiding_labels: ObjectLabelValue):
        del guiding_labels
        return image

    consume_guiding_labels.__processing_contract__ = ProcessingContract.PURE_2D

    labels = np.zeros((2, 2, 3, 3), dtype=np.int32)
    labels[0, 0, 0, 0] = 1
    labels[0, 1, 1, 1] = 2
    labels[1, 0, 0, 1] = 3
    labels[1, 1, 1, 2] = 4
    objects = ObjectLabelSet(
        name="Guides",
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (3, 4)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/site-1.tif", "/input/site-2.tif"),
        ),
    )
    guides_spec = ArtifactSpec.input(
        "Guides", ObjectLabelsArtifactType, parameter_name="guiding_labels"
    )
    contract = _compiled_callable_contract(
        consume_guiding_labels,
        artifact_inputs=(guides_spec,),
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            objects={guides_spec.name: objects},
            artifact_input_edges=(
                _artifact_input_edge_for_test(
                    guides_spec,
                ),
            ),
            plane_projection=RuntimePlaneProjection.stack(2),
        ),
        kwargs={},
        current_image=np.zeros((3, 3), dtype=np.float32),
    )

    bound = request.bind_parameters()

    assert isinstance(bound["guiding_labels"], ObjectLabelValue)
    first_slice = RuntimeSliceProjection.kwargs_for_slice(
        bound,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=0, axis_size=2
        ),
    )["guiding_labels"]
    second_slice = RuntimeSliceProjection.kwargs_for_slice(
        bound,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
    )["guiding_labels"]
    assert isinstance(first_slice, ObjectLabelValue)
    assert isinstance(second_slice, ObjectLabelValue)
    np.testing.assert_array_equal(
        object_label_dense_array(first_slice),
        labels[0],
    )
    np.testing.assert_array_equal(
        object_label_dense_array(second_slice),
        labels[1],
    )


def test_special_object_label_input_preserves_nominal_value_for_scalar_parameter() -> (
    None
):
    @special_inputs("guiding_labels")
    def consume_guiding_labels(image, guiding_labels):
        del guiding_labels
        return image

    consume_guiding_labels.__processing_contract__ = ProcessingContract.PURE_2D

    objects = ObjectLabelSet(
        name="Guides",
        variant_data=ObjectLabelVariantData(
            labels=np.array([[0, 1], [0, 0]], dtype=np.int32)
        ),
    )
    guides_spec = ArtifactSpec.input(
        "Guides", ObjectLabelsArtifactType, parameter_name="guiding_labels"
    )
    contract = _compiled_callable_contract(
        consume_guiding_labels,
        artifact_inputs=(guides_spec,),
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            objects={guides_spec.name: objects},
            artifact_input_edges=(
                _artifact_input_edge_for_test(
                    guides_spec,
                ),
            ),
        ),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    bound = request.bind_parameters()

    assert isinstance(bound["guiding_labels"], ObjectLabelValue)
    np.testing.assert_array_equal(
        object_label_dense_array(bound["guiding_labels"]),
        objects.labels,
    )


def test_track_objects_special_input_binds_full_stack_labels_for_pure_3d() -> None:
    labels = np.array(
        [
            [[0, 1], [0, 0]],
            [[0, 0], [2, 0]],
        ],
        dtype=np.int32,
    )
    objects = ObjectLabelSet(
        name="Embryos",
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    embryos_spec = ArtifactSpec.input(
        "Embryos", ObjectLabelsArtifactType, parameter_name="labels"
    )
    contract = _compiled_callable_contract(
        track_objects,
        artifact_inputs=(embryos_spec,),
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            objects={embryos_spec.name: objects},
            artifact_input_edges=(
                _artifact_input_edge_for_test(
                    embryos_spec,
                ),
            ),
            plane_projection=RuntimePlaneProjection(plane_index=1, plane_count=2),
        ),
        kwargs={},
        current_image=np.zeros((2, 2, 2), dtype=np.float32),
    )

    bound = request.bind_parameters()

    assert isinstance(bound["labels"], ObjectLabelValue)
    np.testing.assert_array_equal(object_label_dense_array(bound["labels"]), labels)


def test_watershed_marker_special_input_preserves_declared_image_scale() -> None:
    markers = np.asarray([[1, 0, 2]], dtype=np.int32)
    mask = np.ones(markers.shape, dtype=bool)
    marker_image = ImageMetadataPayload(
        data=markers,
        metadata=ImagePayloadMetadata(intensity_scale=1.0),
    )
    marker_spec = ArtifactSpec.input(
        "Seeds", ImageArtifactType, parameter_name="topology_inputs"
    )
    mask_spec = ArtifactSpec.input(
        "Mask", ImageArtifactType, parameter_name="topology_inputs"
    )
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("Watershed").require_callable(),
        artifact_inputs=(
            marker_spec,
            mask_spec,
        ),
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {
                marker_spec.name: marker_image,
                mask_spec.name: mask,
            },
            callable_contract=contract,
            artifact_input_edges=(
                _artifact_input_edge_for_test(
                    marker_spec,
                ),
                _artifact_input_edge_for_test(
                    mask_spec,
                ),
            ),
        ),
        kwargs={"watershed_method": WatershedMethod.MARKERS},
        current_image=mask,
    )

    bound = WatershedModule.bind_runtime_inputs(request)

    topology_inputs = bound["topology_inputs"]
    np.testing.assert_array_equal(topology_inputs[0], markers)
    assert topology_inputs[0].dtype == np.float32
    np.testing.assert_array_equal(topology_inputs[1], mask.astype(np.float32))

    _, _, labels = watershed_cellprofiler4(
        mask,
        topology_inputs=topology_inputs,
        watershed_method=WatershedMethod.MARKERS,
    )

    assert isinstance(labels, ObjectLabelValue)


def test_runtime_binding_uses_full_compiled_special_input_contract() -> None:
    segmentation_spec = ArtifactSpec.input("Segmentation", ImageArtifactType)
    marker_spec = ArtifactSpec.input(
        "Markers", ImageArtifactType, parameter_name="topology_inputs"
    )
    mask_spec = ArtifactSpec.input(
        "Mask", ImageArtifactType, parameter_name="topology_inputs"
    )
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("Watershed").require_callable(),
        artifact_inputs=(segmentation_spec, marker_spec, mask_spec),
    )
    runtime = _FakeCellProfilerRuntime(
        {
            "Segmentation": np.zeros((2, 2), dtype=np.float32),
            "Markers": np.ones((2, 2), dtype=np.float32),
            "Mask": np.ones((2, 2), dtype=np.float32),
        },
        artifact_input_edges=(
            _artifact_input_edge_for_test(
                marker_spec,
            ),
            _artifact_input_edge_for_test(
                mask_spec,
            ),
        ),
    )
    executor = _module_executor(contract)
    _activate_runtime_contract(executor.callable_contract, runtime)

    bound = executor._runtime_input_kwargs(
        runtime,
        np.zeros((2, 2), dtype=np.float32),
        {"watershed_method": WatershedMethod.MARKERS},
        module_type=WatershedModule,
    )

    assert tuple(bound) == ("topology_inputs",)
    topology_inputs = bound["topology_inputs"]
    np.testing.assert_array_equal(image_payload_data(topology_inputs[0]), 1.0)
    np.testing.assert_array_equal(image_payload_data(topology_inputs[1]), 1.0)


def test_image_artifact_resolution_uses_declared_artifact_alias() -> None:
    spec = ArtifactSpec.input("IllumStain1", ImageArtifactType)
    source = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1.tif",),
        ),
        source_image_names=("OrigStain1",),
    ).payload_with(np.ones((4, 5), dtype=np.float32), None)
    payload = RuntimeArtifactTypeStrategy.for_artifact_type(
        ImageArtifactType
    ).raw_runtime_input_value(
        RuntimeArtifactInputRequest(
            spec=spec,
            value=source,
        )
    )

    metadata = image_payload_metadata(payload)
    assert metadata.source_image_names == ("IllumStain1",)
    assert metadata.source_image_provenance_planes.paths == ("/input/A01_s001_w1.tif",)
    assert (
        RuntimeArtifactTypeStrategy.for_artifact_type(
            ImageArtifactType
        ).source_image_name(RuntimeArtifactInputRequest(spec=spec, value=source))
        is None
    )


def test_main_flow_image_artifact_selects_declared_alias_plane() -> None:
    first_spec = ArtifactSpec.input("Stain1", ImageArtifactType)
    second_spec = ArtifactSpec.input("Stain2", ImageArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("MeasureColocalization").require_callable(),
        artifact_inputs=(first_spec, second_spec),
    )
    current_image = ImagePayloadMetadata(
        source_image_names=("Stain1", "Stain2"),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("stain1.tif", "stain2.tif"),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(
        np.stack(
            (
                np.zeros((4, 5), dtype=np.float32),
                np.ones((4, 5), dtype=np.float32),
            )
        ),
        None,
    )
    adapter = _FakeCellProfilerRuntime(
        {},
        callable_contract=contract,
        artifact_input_edges=(
            replace(
                _artifact_input_edge_for_test(first_spec, stored=False),
                consumes_main_flow=True,
            ),
        ),
    )

    payload = RuntimeArtifactTypeStrategy.for_artifact_type(
        ImageArtifactType
    ).raw_runtime_input_value(
        RuntimeInputBindingRequest(
            adapter=adapter,
            kwargs={},
            current_image=current_image,
        ).artifact_request_for_spec(first_spec)
    )

    np.testing.assert_array_equal(image_payload_data(payload), np.zeros((4, 5)))
    assert image_payload_metadata(payload).source_image_names == ("Stain1",)


def test_single_main_flow_image_artifact_selects_declared_source_binding_plane() -> (
    None
):
    spec = ArtifactSpec.input("OrigStain1", ImageArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module(
            "CorrectIlluminationApply"
        ).require_callable(),
        artifact_inputs=(spec,),
    )
    current_image = ImagePayloadMetadata(
        source_image_names=("OrigStain1", "OrigStain2"),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("stain1.tif", "stain2.tif"),
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    ).payload_with(
        np.stack(
            (
                np.zeros((4, 5), dtype=np.float32),
                np.ones((4, 5), dtype=np.float32),
            )
        ),
        None,
    )

    payload = RuntimeArtifactTypeStrategy.for_artifact_type(
        ImageArtifactType
    ).raw_runtime_input_value(
        RuntimeInputBindingRequest(
            adapter=_FakeCellProfilerRuntime(
                {},
                callable_contract=contract,
                artifact_input_edges=(
                    replace(
                        _artifact_input_edge_for_test(spec, stored=False),
                        consumes_main_flow=True,
                    ),
                ),
            ),
            kwargs={},
            current_image=current_image,
        ).artifact_request_for_spec(spec)
    )

    np.testing.assert_array_equal(image_payload_data(payload), np.zeros((4, 5)))
    assert image_payload_metadata(payload).source_image_names == ("OrigStain1",)


def test_main_flow_image_artifact_projects_named_provenance_plane() -> None:
    first_spec = ArtifactSpec.input("Stain1", ImageArtifactType)
    second_spec = ArtifactSpec.input("Stain2", ImageArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("MeasureColocalization").require_callable(),
        artifact_inputs=(first_spec, second_spec),
    )
    metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("stain1.tif", "stain2.tif"),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    metadata = metadata.with_source_provenance(
        metadata.source_provenance.with_derived_source_image_names(
            ("Stain1", "Stain2")
        ).with_source_image_names(())
    )
    current_image = metadata.payload_with(
        np.stack(
            (
                np.zeros((4, 5), dtype=np.float32),
                np.ones((4, 5), dtype=np.float32),
            )
        ),
        None,
    )

    payload = RuntimeArtifactTypeStrategy.for_artifact_type(
        ImageArtifactType
    ).raw_runtime_input_value(
        RuntimeInputBindingRequest(
            adapter=_FakeCellProfilerRuntime(
                {},
                callable_contract=contract,
                artifact_input_edges=(
                    replace(
                        _artifact_input_edge_for_test(first_spec, stored=False),
                        consumes_main_flow=True,
                    ),
                ),
            ),
            kwargs={},
            current_image=current_image,
        ).artifact_request_for_spec(first_spec)
    )

    np.testing.assert_array_equal(image_payload_data(payload), np.zeros((4, 5)))
    assert image_payload_metadata(payload).source_image_names == ("Stain1",)


def test_main_flow_image_artifact_preserves_declared_runtime_slice_stack() -> None:
    first_spec = ArtifactSpec.input("Stain1", ImageArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("IdentifyPrimaryObjects").require_callable(),
        artifact_inputs=(first_spec,),
    )
    current_image = ImagePayloadMetadata(
        source_image_names=("Stain1", "Stain1"),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("stain1-1.tif", "stain1-2.tif"),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(
        np.stack(
            (
                np.zeros((4, 5), dtype=np.float32),
                np.ones((4, 5), dtype=np.float32),
            )
        ),
        None,
    )

    payload = RuntimeArtifactTypeStrategy.for_artifact_type(
        ImageArtifactType
    ).raw_runtime_input_value(
        RuntimeInputBindingRequest(
            adapter=_FakeCellProfilerRuntime(
                {},
                callable_contract=contract,
                artifact_input_edges=(
                    replace(
                        _artifact_input_edge_for_test(first_spec, stored=False),
                        consumes_main_flow=True,
                    ),
                ),
            ),
            kwargs={},
            current_image=current_image,
        ).artifact_request_for_spec(first_spec)
    )

    np.testing.assert_array_equal(
        image_payload_data(payload), image_payload_data(current_image)
    )
    assert image_payload_metadata(payload).source_image_names == ("Stain1",)


def test_main_flow_projection_binds_singleton_broadcast_artifact_as_2d() -> None:
    original_spec = ArtifactSpec.input("OrigStain1", ImageArtifactType)
    illumination_spec = ArtifactSpec.input(
        "IllumStain1",
        ImageArtifactType,
        relations=(InputStackBroadcastSourceRelation(source=original_spec.ref()),),
        parameter_name="illumination_function",
    )
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module(
            "CorrectIlluminationApply"
        ).require_callable(),
        artifact_inputs=(
            *((original_spec,)),
            *((illumination_spec,)),
        ),
    )
    source_paths = ("stain1.tif", "stain2.tif")
    source_metadata = (
        {"well": "A01", "site": "1", "channel": "1"},
        {"well": "A01", "site": "1", "channel": "2"},
    )
    current_image = ImagePayloadMetadata(
        source_image_names=("OrigStain1", "OrigStain2"),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=source_paths,
            component_metadata=source_metadata,
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)
    illumination = ImagePayloadMetadata(
        source_image_names=("IllumStain1",),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(source_paths[0],),
            component_metadata=(source_metadata[0],),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((1, 4, 5), dtype=np.float32), None)
    runtime = _FakeCellProfilerRuntime(
        {illumination_spec.name: illumination},
        artifact_input_edges=(
            replace(
                _artifact_input_edge_for_test(original_spec, stored=False),
                consumes_main_flow=True,
            ),
            _artifact_input_edge_for_test(
                illumination_spec,
            ),
        ),
    )
    executor = _module_executor(contract)
    _activate_runtime_contract(executor.callable_contract, runtime)

    image_request = executor._image_request(
        current_image,
        runtime,
        module_type=CorrectIlluminationApplyModule,
        active_input_specs=contract.artifact_inputs.specs,
    )
    runtime_kwargs = executor._runtime_input_kwargs(
        runtime,
        current_image,
        {},
        module_type=CorrectIlluminationApplyModule,
        primary_image=image_request.payload,
    )

    assert image_payload_data(image_request.payload).shape == (4, 5)
    assert image_payload_data(runtime_kwargs["illumination_function"]).shape == (4, 5)


def test_repeated_broadcast_inputs_consume_their_declared_source_group_axes() -> None:
    original_specs = tuple(
        ArtifactSpec.input(f"OrigStain{index}", ImageArtifactType)
        for index in (1, 2)
    )
    illumination_specs = tuple(
        ArtifactSpec.input(
            f"IllumStain{index}",
            ImageArtifactType,
            relations=(
                InputStackBroadcastSourceRelation(source=original.ref()),
            ),
            parameter_name="illumination_function",
        )
        for index, original in enumerate(original_specs, start=1)
    )
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module(
            "CorrectIlluminationApply"
        ).require_callable(),
        artifact_inputs=(*original_specs, *illumination_specs),
    )
    source_paths = ("stain1.tif", "stain2.tif")
    source_metadata = tuple(
        {"well": "A01", "site": "1", "channel": str(index)}
        for index in (1, 2)
    )
    current_image = ImagePayloadMetadata(
        source_image_names=tuple(spec.name for spec in original_specs),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=source_paths,
            component_metadata=source_metadata,
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)
    illumination_payloads = tuple(
        ImagePayloadMetadata(
            source_image_names=(spec.name,),
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=(source_paths[index],),
                    component_metadata=(source_metadata[index],),
                )
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(
            np.full((1, 4, 5), index + 1, dtype=np.float32),
            None,
        )
        for index, spec in enumerate(illumination_specs)
    )
    runtime = _FakeCellProfilerRuntime(
        dict(zip((spec.name for spec in illumination_specs), illumination_payloads)),
        artifact_input_edges=(
            *(
                replace(
                    _artifact_input_edge_for_test(spec, stored=False),
                    consumes_main_flow=True,
                )
                for spec in original_specs
            ),
            *(
                _artifact_input_edge_for_test(spec)
                for spec in illumination_specs
            ),
        ),
    )
    executor = _module_executor(contract)
    _activate_runtime_contract(executor.callable_contract, runtime)

    image_request = executor._image_request(
        current_image,
        runtime,
        module_type=CorrectIlluminationApplyModule,
        active_input_specs=contract.artifact_inputs.specs,
    )
    runtime_kwargs = executor._runtime_input_kwargs(
        runtime,
        current_image,
        {},
        module_type=CorrectIlluminationApplyModule,
        primary_image=image_request.payload,
    )

    assert isinstance(image_request.payload, AlignedImageStack)
    illumination_argument = runtime_kwargs["illumination_function"]
    assert isinstance(illumination_argument, AlignedImageStack)
    for index in (0, 1):
        resolved = aligned_image_stack_kwargs(
            {"illumination_function": illumination_argument},
            slice_index=index,
            slice_count=2,
            reference_payload=image_request.payload.slices[index],
        )
        assert image_payload_data(resolved["illumination_function"]).shape == (4, 5)
        np.testing.assert_array_equal(
            image_payload_data(resolved["illumination_function"]),
            index + 1,
        )


def test_single_main_flow_image_preserves_runtime_slice_axis() -> None:
    spec = ArtifactSpec.input("CropRed", ImageArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("GrayToColor").require_callable(),
        artifact_inputs=(spec,),
    )
    current_image = ImagePayloadMetadata(
        source_image_names=("CropRed",),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("crop-red.tif",),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((1, 4, 5), dtype=np.float32), None)

    payload = RuntimeArtifactTypeStrategy.for_artifact_type(
        ImageArtifactType
    ).raw_runtime_input_value(
        RuntimeInputBindingRequest(
            adapter=_FakeCellProfilerRuntime(
                {},
                callable_contract=contract,
                artifact_input_edges=(
                    replace(
                        _artifact_input_edge_for_test(spec, stored=False),
                        consumes_main_flow=True,
                    ),
                ),
            ),
            kwargs={},
            current_image=current_image,
        ).artifact_request_for_spec(spec)
    )

    assert image_payload_data(payload).shape == (1, 4, 5)
    assert image_payload_metadata(payload).plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert image_payload_metadata(payload).source_image_names == ("CropRed",)


def test_object_artifact_source_payload_uses_native_object_provenance() -> None:
    object_spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    objects = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 1], [2, 0]], dtype=np.int32)
        ),
        source_image_name="Stain1",
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1.tif",),
        ),
    )
    request = RuntimeArtifactInputRequest(
        spec=object_spec,
        value=objects,
    )

    payload = RuntimeArtifactTypeStrategy.for_artifact_type(
        ObjectLabelsArtifactType
    ).source_image_payload(request)

    assert payload is objects
    assert image_payload_metadata(payload).source_image_provenance_planes.paths == (
        "/input/A01_s001_w1.tif",
    )


def test_object_input_does_not_replace_preserved_canonical_main_flow_image() -> None:
    object_spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("FilterObjects").require_callable(),
        artifact_inputs=(object_spec,),
    )
    objects = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray((((0, 1), (2, 0)),), dtype=np.int32)
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={object_spec.name: objects},
        artifact_input_edges=(_artifact_input_edge_for_test(object_spec),),
        plane_projection=RuntimePlaneProjection.stack(1),
    )
    executor = _module_executor(contract)
    _activate_runtime_contract(executor.callable_contract, runtime)

    current_image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.full((1, 2, 2), 7, dtype=np.float32))
    image_request = executor._image_request(
        current_image,
        runtime,
        module_type=_module_type_for_contract(contract),
        active_input_specs=contract.artifact_inputs.specs,
    )

    assert image_request.plane_projection == RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        axis_size=1,
    )
    np.testing.assert_array_equal(
        image_payload_data(image_request.payload),
        np.full((1, 2, 2), 7, dtype=np.float32),
    )


def test_object_input_does_not_replace_selected_canonical_main_flow_plane() -> None:
    object_spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("FilterObjects").require_callable(),
        artifact_inputs=(object_spec,),
    )
    objects = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray((((0, 1), (2, 0)),), dtype=np.int32)
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={object_spec.name: objects},
        artifact_input_edges=(_artifact_input_edge_for_test(object_spec),),
        plane_projection=RuntimePlaneProjection.selected(0, 1),
    )
    executor = _module_executor(contract)
    _activate_runtime_contract(executor.callable_contract, runtime)

    current_image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.full((1, 2, 2), 9, dtype=np.float32))
    image_request = executor._image_request(
        current_image,
        runtime,
        module_type=_module_type_for_contract(contract),
        active_input_specs=contract.artifact_inputs.specs,
    )

    assert image_request.plane_projection is None
    assert np.shape(image_payload_data(image_request.payload)) == (2, 2)
    np.testing.assert_array_equal(
        image_payload_data(image_request.payload),
        np.full((2, 2), 9, dtype=np.float32),
    )
    assert image_payload_metadata(image_request.payload).plane_axis is None


def test_singleton_object_measurement_collapses_singleton_plane_context() -> None:
    object_spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("FilterObjects").require_callable(),
        artifact_inputs=(object_spec,),
    )
    objects = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray((((0, 1), (2, 0)),), dtype=np.int32)
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={object_spec.name: objects},
        artifact_input_edges=(_artifact_input_edge_for_test(object_spec),),
        plane_projection=RuntimePlaneProjection.stack(1),
    )
    executor = _module_executor(contract)
    _activate_runtime_contract(executor.callable_contract, runtime)
    measurement_image = executor._object_label_measurement_image(
        object_spec,
        runtime,
        np.zeros((1, 2, 2), dtype=np.float32),
    )

    aligned_image, executable_labels, *_ = object_measurement_runtime_inputs(
        object_label_execution=object_label_input_execution_mode_from_callable(
            executor.raw_func
        ),
        measurement_image=measurement_image,
        object_spec=object_spec,
        label_payload=objects,
        adapter=runtime,
    )

    assert aligned_image.plane_projection is None
    assert np.shape(image_payload_data(aligned_image.payload)) == (2, 2)
    assert RuntimeSliceProjection.preserved_context_for_value(executable_labels) is None


def test_object_input_binding_resolves_exact_compiled_label_artifact() -> None:
    object_spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("FilterObjects").require_callable(),
        artifact_inputs=(object_spec,),
    )
    objects = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 1], [2, 0]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
    )
    current_image = np.zeros((2, 2), dtype=np.float32)
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            objects={object_spec.name: objects},
            artifact_input_edges=(_artifact_input_edge_for_test(object_spec),),
        ),
        kwargs={},
        current_image=current_image,
    )

    assert request.label_payload_for(object_spec) is objects


def test_pure_3d_executor_rejects_runtime_slice_aligned_label_kwargs() -> None:
    def keep_stack(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
        del labels
        return image

    keep_stack.__processing_contract__ = ProcessingContract.PURE_3D
    callable_contract = _compiled_callable_contract(
        keep_stack, artifact_outputs=(ArtifactSpec.output("Image", ImageArtifactType),)
    )

    with pytest.raises(
        ValueError,
        match="keep_stack.*ProcessingContract.PURE_3D.*runtime-slice-aligned kwargs.*labels",
    ):
        CellProfilerFunctionContractExecutor().execute(
            callable_contract,
            keep_stack,
            np.zeros((2, 2, 2), dtype=np.float32),
            {"labels": RuntimeSliceAlignedValues((np.zeros((2, 2), dtype=np.int32),))},
            execution_mode=ImagePayloadExecutionMode.NATURAL,
        )


def test_special_object_label_payload_uses_runtime_plane_selection() -> None:
    labels = np.zeros((2, 4, 4), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 0:2, 0:2] = 2
    objects = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/plate/A01_s001_w3_z001_t001.tif",
                "/plate/A01_s001_w3_z002_t001.tif",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "3", "z_index": "1"},
                {"well": "A01", "site": "1", "channel": "3", "z_index": "2"},
            ),
        ),
    )
    spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("ConvertObjectsToImage").require_callable(),
        artifact_inputs=(spec,),
    )
    adapter = _FakeCellProfilerRuntime(
        {},
        callable_contract=contract,
        objects={spec.name: objects},
        artifact_input_edges=(_artifact_input_edge_for_test(spec),),
        plane_projection=RuntimePlaneProjection(plane_index=1, plane_count=2),
    )
    current_image = np.zeros((4, 4), dtype=np.float32)
    request = RuntimeInputBindingRequest(
        adapter=adapter,
        kwargs={},
        current_image=current_image,
    )

    payload = request.current_plane_label_payload(request.label_payload_for(spec))

    np.testing.assert_array_equal(payload.labels, objects.labels[1])
    assert payload.source_component_metadata == {
        "well": "A01",
        "site": "1",
        "channel": "3",
        "z_index": "2",
    }


def test_measurement_object_label_preparation_projects_runtime_slice_payload() -> None:
    labels = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [2, 0]],
        ],
        dtype=np.int32,
    )
    objects = ObjectLabelSet(
        name="Objects",
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawDNA",
        source_aliases=("rawDNA",),
        payload=np.zeros((2, 2), dtype=np.float32),
    )

    prepared = PreparedMeasurementObjectLabels.from_source(
        measurement_image,
        objects,
        plane_projector=_RuntimeSliceObjectAdapter(
            objects=objects,
            slice_index=1,
            slice_count=2,
        ),
    )

    np.testing.assert_array_equal(prepared.measurement_labels, labels[1])
    assert prepared.completion_payload.domain.declared_object_ids == (2,)


def test_measure_object_neighbors_derives_same_object_topology_from_artifact_identity() -> (
    None
):
    objects = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=np.asarray([[1]], dtype=np.int32)),
    )
    cells_spec = ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    repeated_role_specs = (cells_spec, cells_spec)
    contract = _compiled_callable_contract(
        measure_object_neighbors,
        artifact_inputs=repeated_role_specs,
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            objects={cells_spec.name: objects},
            artifact_input_edges=(_artifact_input_edge_for_test(cells_spec),),
            plane_projection=RuntimePlaneProjection(plane_index=0, plane_count=1),
        ),
        kwargs={},
        current_image=np.zeros((1, 1), dtype=np.float32),
    )

    bound = MeasureObjectNeighborsModule.bind_runtime_inputs(request)

    assert bound["neighbors_are_same_objects"] is True


def test_relateobjects_special_inputs_project_object_labels_to_current_plane() -> None:
    labels = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [2, 0]],
        ],
        dtype=np.int32,
    )
    objects = ObjectLabelSet(
        name="Objects",
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    input_specs = (
        ArtifactSpec.input(
            "Parents", ObjectLabelsArtifactType, parameter_name="parent_labels"
        ),
        ArtifactSpec.input(
            "Children", ObjectLabelsArtifactType, parameter_name="child_labels"
        ),
    )
    contract = _compiled_callable_contract(
        relate_objects,
        artifact_inputs=input_specs,
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            objects={
                spec.name: ObjectLabelSet.from_payload(spec.name, objects)
                for spec in input_specs
            },
            artifact_input_edges=tuple(
                (
                    _artifact_input_edge_for_test(
                        spec,
                    )
                    for spec in input_specs
                )
            ),
            plane_projection=RuntimePlaneProjection(plane_index=1, plane_count=2),
        ),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    bound = RelateObjectsModule.bind_runtime_inputs(request)

    assert isinstance(bound["parent_labels"], ObjectLabelValue)
    assert isinstance(bound["child_labels"], ObjectLabelValue)
    np.testing.assert_array_equal(bound["parent_labels"].labels, labels[1])
    np.testing.assert_array_equal(bound["child_labels"].labels, labels[1])


def test_relateobjects_special_inputs_leave_runtime_slice_index_to_executor() -> None:
    labels = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [2, 0]],
        ],
        dtype=np.int32,
    )
    objects = ObjectLabelSet(
        name="Objects",
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    input_specs = (
        ArtifactSpec.input(
            "Parents", ObjectLabelsArtifactType, parameter_name="parent_labels"
        ),
        ArtifactSpec.input(
            "Children", ObjectLabelsArtifactType, parameter_name="child_labels"
        ),
    )
    contract = _compiled_callable_contract(
        relate_objects,
        artifact_inputs=input_specs,
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            objects={
                spec.name: ObjectLabelSet.from_payload(spec.name, objects)
                for spec in input_specs
            },
            artifact_input_edges=tuple(
                (
                    _artifact_input_edge_for_test(
                        spec,
                    )
                    for spec in input_specs
                )
            ),
            plane_projection=RuntimePlaneProjection(plane_index=1, plane_count=2),
        ),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    bound = RelateObjectsModule.bind_runtime_inputs(request)

    assert "slice_index" not in bound


def test_relateobjects_special_inputs_project_source_binding_labels_to_current_plane() -> (
    None
):
    class SourceAxisObjectAdapter(_FakeCellProfilerRuntime):
        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return 1 if source_aliases == ("Site1", "Site2") else None

    labels = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [2, 0]],
        ],
        dtype=np.int32,
    )
    objects = ObjectLabelSet(
        name="Objects",
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("Site1", "Site2"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    input_specs = (
        ArtifactSpec.input(
            "Parents", ObjectLabelsArtifactType, parameter_name="parent_labels"
        ),
        ArtifactSpec.input(
            "Children", ObjectLabelsArtifactType, parameter_name="child_labels"
        ),
    )
    contract = _compiled_callable_contract(
        relate_objects,
        artifact_inputs=input_specs,
    )
    request = RuntimeInputBindingRequest(
        adapter=SourceAxisObjectAdapter(
            {},
            callable_contract=contract,
            objects={
                spec.name: ObjectLabelSet.from_payload(spec.name, objects)
                for spec in input_specs
            },
            artifact_input_edges=tuple(
                (
                    _artifact_input_edge_for_test(
                        spec,
                    )
                    for spec in input_specs
                )
            ),
        ),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    bound = RelateObjectsModule.bind_runtime_inputs(request)

    assert isinstance(bound["parent_labels"], ObjectLabelValue)
    assert isinstance(bound["child_labels"], ObjectLabelValue)
    np.testing.assert_array_equal(bound["parent_labels"].labels, labels[1])
    np.testing.assert_array_equal(bound["child_labels"].labels, labels[1])
    assert "slice_index" not in bound


def test_relateobjects_runtime_slice_axis_leaves_slice_index_to_executor() -> None:
    class SourceAxisObjectAdapter(_FakeCellProfilerRuntime):
        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return 1 if source_aliases in {("Stain1",), ("Stain2",)} else None

    labels = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [2, 0]],
        ],
        dtype=np.int32,
    )
    objects = ObjectLabelSet(
        name="Objects",
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_names=("Stain1",),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    input_specs = (
        ArtifactSpec.input(
            "Parents", ObjectLabelsArtifactType, parameter_name="parent_labels"
        ),
        ArtifactSpec.input(
            "Children", ObjectLabelsArtifactType, parameter_name="child_labels"
        ),
    )
    contract = _compiled_callable_contract(
        relate_objects,
        artifact_inputs=input_specs,
    )
    request = RuntimeInputBindingRequest(
        adapter=SourceAxisObjectAdapter(
            {},
            callable_contract=contract,
            objects={
                spec.name: ObjectLabelSet.from_payload(spec.name, objects)
                for spec in input_specs
            },
            artifact_input_edges=tuple(
                (
                    _artifact_input_edge_for_test(
                        spec,
                    )
                    for spec in input_specs
                )
            ),
            plane_projection=RuntimePlaneProjection(plane_index=1, plane_count=2),
        ),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    bound = RelateObjectsModule.bind_runtime_inputs(request)

    assert "slice_index" not in bound


def test_relateobjects_scalar_source_name_does_not_create_source_axis() -> None:
    labels = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [2, 0]],
        ],
        dtype=np.int32,
    )
    objects = ObjectLabelSet(
        name="Objects",
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_name="OrigStain1__OrigStain2",
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    input_specs = (
        ArtifactSpec.input(
            "Parents", ObjectLabelsArtifactType, parameter_name="parent_labels"
        ),
        ArtifactSpec.input(
            "Children", ObjectLabelsArtifactType, parameter_name="child_labels"
        ),
    )
    contract = _compiled_callable_contract(
        relate_objects,
        artifact_inputs=input_specs,
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            objects={
                spec.name: ObjectLabelSet.from_payload(spec.name, objects)
                for spec in input_specs
            },
            artifact_input_edges=tuple(
                (
                    _artifact_input_edge_for_test(
                        spec,
                    )
                    for spec in input_specs
                )
            ),
            plane_projection=RuntimePlaneProjection(plane_index=1, plane_count=2),
        ),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    bound = RelateObjectsModule.bind_runtime_inputs(request)

    assert "slice_index" not in bound


def test_track_objects_retained_image_uses_tracked_object_source_payload() -> None:
    current_payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("color0.tif", "color1.tif", "color2.tif"),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "1", "timepoint": "0"},
                {"well": "A01", "site": "1", "channel": "1", "timepoint": "1"},
                {"well": "A01", "site": "1", "channel": "1", "timepoint": "2"},
            ),
        ),
        source_image_names=("OrigColor",),
    ).payload_with(np.zeros((3, 5, 6), dtype=np.float32), None)
    tracked_objects = ObjectLabelSet(
        name="Embryos",
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 5, 6), dtype=np.int32)),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((), (), ()),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("gray0.tif", "gray1.tif", "gray2.tif"),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "1", "timepoint": "0"},
                {"well": "A01", "site": "1", "channel": "1", "timepoint": "1"},
                {"well": "A01", "site": "1", "channel": "1", "timepoint": "2"},
            ),
        ),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={"Embryos": tracked_objects},
        artifact_input_edges=(
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name="Embryos",
                    path="/artifacts/Embryos",
                    artifact_type=ObjectLabelsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        ),
    )
    output = _output_from_input(
        "TrackedCells",
        "Embryos",
        output_type=ImageArtifactType,
    )
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("TrackObjects").require_callable(),
            artifact_inputs=(
                *((ArtifactSpec.input("Embryos", ObjectLabelsArtifactType),)),
                *((ArtifactSpec.input("Embryos", ObjectLabelsArtifactType),)),
            ),
            artifact_outputs=(output,),
        ),
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(item)
            for item in (
                ArtifactSpec.input("Embryos", ObjectLabelsArtifactType),
                ArtifactSpec.input("Embryos", ObjectLabelsArtifactType),
            )
        ),
        output_plans=tuple(_artifact_output_plan(item) for item in ((output,))),
        adapter=runtime,
        spec=output,
        output_value=np.zeros((3, 5, 6), dtype=np.float32),
        source=CellProfilerImageRequest(
            payload=current_payload,
            source_image_name=None,
            image_count=3,
            execution_mode=ImagePayloadExecutionMode.FULL_STACK,
        ),
        current_image=current_payload,
        call_kwargs={},
    )

    source_payload = _module_type_for_contract(
        request.callable_contract
    ).source_payload(request)

    assert isinstance(source_payload, ObjectLabelSet)
    assert source_payload.name == tracked_objects.name
    np.testing.assert_array_equal(source_payload.labels, tracked_objects.labels)
    assert image_payload_metadata(
        source_payload
    ).source_image_provenance_planes.paths == (
        "gray0.tif",
        "gray1.tif",
        "gray2.tif",
    )


def test_nominal_object_label_output_preserves_source_slice_paths() -> None:
    source_image = ImageMetadataPayload(
        data=np.zeros((3, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w1_z001_t001.TIF",
                    "/input/A01_s002_w1_z001_t001.TIF",
                    "/input/A01_s003_w1_z001_t001.TIF",
                ),
                component_metadata=(
                    {"well": "A01", "site": "001", "channel": "D"},
                    {"well": "A01", "site": "002", "channel": "D"},
                    {"well": "A01", "site": "003", "channel": "D"},
                ),
            ),
            source_spatial_domain=SourceSpatialDomain(source_shape_yx=(10, 12)),
        ),
    )
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 4, 5), dtype=np.int32)),
        domain=ObjectLabelDomain(declared_object_count=0),
    ).with_source_image_context(source_image)

    assert isinstance(payload, ObjectLabelPayload)
    assert payload.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.TIF",
        "/input/A01_s002_w1_z001_t001.TIF",
        "/input/A01_s003_w1_z001_t001.TIF",
    )
    assert tuple(
        dict(metadata)
        for metadata in payload.source_image_provenance_planes.component_metadata
        if metadata is not None
    ) == (
        {"well": "A01", "site": "001", "channel": "D"},
        {"well": "A01", "site": "002", "channel": "D"},
        {"well": "A01", "site": "003", "channel": "D"},
    )
    assert payload.source_spatial_shape_yx == (10, 12)


def test_contextual_object_label_output_applies_declared_payload_scope() -> None:
    source_image = ImageMetadataPayload(
        data=np.zeros((3, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w1_z001_t001.TIF",
                    "/input/A01_s001_w1_z002_t001.TIF",
                    "/input/A01_s001_w1_z003_t001.TIF",
                ),
            ),
        ),
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 4, 5), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((), (), ()),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    contextual = labels.with_source_image_context(source_image)
    payload = contextual.with_variants(
        contextual.variant_data,
        domain=contextual.object_label_domain().with_scope(
            ObjectLabelDomainScope.PAYLOAD
        ),
        plane_axis=None,
    )

    assert isinstance(payload, ObjectLabelPayload)
    assert payload.object_label_domain().scope is ObjectLabelDomainScope.PAYLOAD
    assert payload.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.TIF",
        "/input/A01_s001_w1_z002_t001.TIF",
        "/input/A01_s001_w1_z003_t001.TIF",
    )


def test_object_label_output_source_preserves_matching_input_object_plane_context() -> (
    None
):
    source_paths = (
        "/input/A01_s001_w1.tif",
        "/input/A01_s002_w1.tif",
    )
    full_objects = ObjectLabelSet(
        name="Guides",
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 4, 5), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((), ()),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=source_paths,
            component_metadata=({"site": "001"}, {"site": "002"}),
        ),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={"Guides": full_objects},
        artifact_input_edges=(
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name="Guides",
                    path="/artifacts/Guides",
                    artifact_type=ObjectLabelsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        ),
        variable_components=(VariableComponents.CHANNEL,),
    )
    current_image = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=source_paths,
            component_metadata=({"site": "001"}, {"site": "002"}),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.zeros((2, 4, 5), dtype=np.float32), None)
    output = _output_from_input("GridObjects", "Guides")
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module(
                "IdentifyObjectsInGrid"
            ).require_callable(),
            artifact_inputs=(ArtifactSpec.input("Guides", ObjectLabelsArtifactType),),
            artifact_outputs=(output,),
        ),
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(item)
            for item in ((ArtifactSpec.input("Guides", ObjectLabelsArtifactType),))
        ),
        output_plans=tuple(_artifact_output_plan(item) for item in ((output,))),
        adapter=runtime,
        spec=output,
        output_value=np.zeros((2, 4, 5), dtype=np.int32),
        source_image_name=None,
        call_kwargs={},
        current_image=current_image,
    )

    selected = (
        _module_type_for_contract(request.callable_contract)
        .source_context(request)
        .source_payload
    )

    assert selected is full_objects


def test_resize_objects_output_source_policy_uses_input_object_context() -> None:
    image_payload = ImagePayloadMetadata(
        source_path="/input/A01_s001_w1_z001_t001.TIF",
        source_voxel_spacing=SourceVoxelSpacing((2.0, 1.0, 1.0)),
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)
    object_payload = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=np.zeros((4, 5), dtype=np.int32)),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={"Nuclei": object_payload},
        artifact_input_edges=(
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name="ReferenceImage",
                    path="/artifacts/ReferenceImage",
                    artifact_type=ImageArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name="Nuclei",
                    path="/artifacts/Nuclei",
                    artifact_type=ObjectLabelsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        ),
    )

    output = _output_from_input("ResizedNuclei", "Nuclei")
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("ResizeObjects").require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("ReferenceImage", ImageArtifactType),
                ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
            ),
            artifact_outputs=(output,),
        ),
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(item)
            for item in (
                (
                    ArtifactSpec.input("ReferenceImage", ImageArtifactType),
                    ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                )
            )
        ),
        output_plans=tuple(_artifact_output_plan(item) for item in ((output,))),
        adapter=runtime,
        spec=output,
        output_value=np.zeros((4, 5), dtype=np.int32),
        source_image_name="ReferenceImage",
        call_kwargs={},
        source_image_payload=image_payload,
        current_image=image_payload,
    )

    source_context = _module_type_for_contract(
        request.callable_contract
    ).source_context(request)

    assert source_context.source_payload is object_payload
    assert source_context.parent_image_payload is None


def test_watershed_output_source_policy_uses_declared_image_as_parent() -> None:
    assert WatershedModule.module_name == "Watershed"
    image_payload = ImagePayloadMetadata(
        source_path="/input/A01_s001_w1_z001_t001.TIF",
        source_image_names=("DNA",),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1_z001_t001.TIF",),
        ),
        source_voxel_spacing=SourceVoxelSpacing((2.0, 1.0, 1.0)),
    ).payload_with(np.zeros((3, 4, 5), dtype=np.float32), None)
    output = _output_from_input(
        "Nuclei",
        "DNA",
        input_type=ImageArtifactType,
    )
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("Watershed").require_callable(),
            artifact_inputs=(ArtifactSpec.input("DNA", ImageArtifactType),),
            artifact_outputs=(output,),
        ),
        artifact_input_edges=(
            _artifact_input_edge_for_test(ArtifactSpec.input("DNA", ImageArtifactType)),
        ),
        output_plans=tuple(_artifact_output_plan(item) for item in ((output,))),
        adapter=_FakeCellProfilerRuntime({"DNA": image_payload}),
        spec=output,
        output_value=np.zeros((3, 4, 5), dtype=np.int32),
        source_image_name="DNA",
        call_kwargs={},
        source_aliases=("DNA",),
        source_image_payload=image_payload,
        current_image=image_payload,
    )

    policy = _module_type_for_contract(request.callable_contract)
    source_context = policy.source_context(request)

    np.testing.assert_array_equal(
        image_payload_data(source_context.source_payload),
        image_payload_data(image_payload),
    )
    assert source_context.parent_image_payload is source_context.source_payload
    assert image_payload_metadata(
        source_context.source_payload
    ).source_voxel_spacing == SourceVoxelSpacing((2.0, 1.0, 1.0))


def test_object_label_recorder_suppresses_parent_spacing_when_policy_declares_no_parent_image() -> (
    None
):
    image_payload = ImagePayloadMetadata(
        source_path="/input/A01_s001_w1_z001_t001.TIF",
        source_voxel_spacing=SourceVoxelSpacing((2.0, 1.0, 1.0)),
    ).payload_with(np.zeros((3, 4, 5), dtype=np.float32), None)
    object_payload = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 4, 5), dtype=np.int32)),
    )
    recorded: dict[str, object] = {}

    def get_objects(name, *, current_image=None):
        del current_image
        assert name == "Nuclei"
        return object_payload

    def add_objects(name, labels, **kwargs):
        recorded["name"] = name
        recorded["labels"] = labels
        recorded["kwargs"] = kwargs

    runtime = _FakeCellProfilerRuntime(
        {},
        objects={"Nuclei": object_payload},
        artifact_input_edges=(
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name="ReferenceImage",
                    path="/artifacts/ReferenceImage",
                    artifact_type=ImageArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name="Nuclei",
                    path="/artifacts/Nuclei",
                    artifact_type=ObjectLabelsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        ),
    )
    runtime.get_objects = get_objects
    runtime.add_objects = add_objects

    output = _output_from_input("ResizedNuclei", "Nuclei")
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("ResizeObjects").require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("ReferenceImage", ImageArtifactType),
                ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
            ),
            artifact_outputs=(output,),
        ),
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(item)
            for item in (
                (
                    ArtifactSpec.input("ReferenceImage", ImageArtifactType),
                    ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                )
            )
        ),
        output_plans=tuple(_artifact_output_plan(item) for item in ((output,))),
        adapter=runtime,
        spec=output,
        output_value=ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(
                labels=np.zeros((3, 4, 5), dtype=np.int32)
            ),
            domain=ObjectLabelDomain(declared_object_count=0),
        ),
        source_image_name="ReferenceImage",
        call_kwargs={},
        source_image_payload=image_payload,
        current_image=image_payload,
    )

    CellProfilerOutputRecorder.for_artifact_type(ObjectLabelsArtifactType).record(
        request
    )

    assert recorded["name"] == "ResizedNuclei"
    recorded_labels = recorded["labels"]
    assert isinstance(recorded_labels, ObjectLabelValue)
    assert recorded_labels.parent_image_source_voxel_spacing == SourceVoxelSpacing()


def test_contextual_object_label_recorder_fills_missing_parent_spacing_from_declared_parent_image() -> (
    None
):
    image_payload = ImagePayloadMetadata(
        source_path="/input/A01_s001_w2_z001_t001.TIF",
        source_image_names=("Memb",),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w2_z001_t001.TIF",),
        ),
        source_voxel_spacing=SourceVoxelSpacing((2.0, 1.0, 1.0)),
    ).payload_with(np.zeros((3, 4, 5), dtype=np.float32), None)
    output_labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 4, 5), dtype=np.int32))
    )
    recorded: dict[str, object] = {}

    def add_objects(name, labels, **kwargs):
        recorded["name"] = name
        recorded["labels"] = labels
        recorded["kwargs"] = kwargs

    runtime = _FakeCellProfilerRuntime({"Memb": image_payload})
    runtime.add_objects = add_objects

    output = _output_from_input(
        "Cells",
        "Memb",
        input_type=ImageArtifactType,
    )
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module(
                "IdentifySecondaryObjects"
            ).require_callable(),
            artifact_inputs=(ArtifactSpec.input("Memb", ImageArtifactType),),
            artifact_outputs=(output,),
        ),
        artifact_input_edges=(
            _artifact_input_edge_for_test(
                ArtifactSpec.input("Memb", ImageArtifactType)
            ),
        ),
        output_plans=tuple(_artifact_output_plan(item) for item in ((output,))),
        adapter=runtime,
        spec=output,
        output_value=output_labels,
        source_image_name="Memb",
        call_kwargs={},
        source_aliases=("Memb",),
        source_image_payload=image_payload,
        current_image=image_payload,
    )

    CellProfilerOutputRecorder.for_artifact_type(ObjectLabelsArtifactType).record(
        request
    )

    assert recorded["name"] == "Cells"
    labels = recorded["labels"]
    assert isinstance(labels, ObjectLabelValue)
    assert labels.parent_image_source_voxel_spacing == SourceVoxelSpacing(
        (2.0, 1.0, 1.0)
    )


def test_measure_object_size_shape_record_builder_keeps_shape_features_unqualified() -> (
    None
):
    rows = [{"object_label": 1, "Area": 12.0}]
    object_payload = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 3), dtype=np.int32)),
        domain=ObjectLabelDomain(declared_object_count=1),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={"Cells": object_payload},
        artifact_input_edges=(
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name="Cells",
                    path="/artifacts/Cells",
                    artifact_type=ObjectLabelsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        ),
    )
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module(
                "MeasureObjectSizeShape"
            ).require_callable(),
            artifact_inputs=(ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
            artifact_outputs=(
                ArtifactSpec.output(
                    "MeasureObjectSizeShape_1_measurements", MeasurementsArtifactType
                ),
            ),
        ),
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(item)
            for item in ((ArtifactSpec.input("Cells", ObjectLabelsArtifactType),))
        ),
        output_plans=tuple(
            _artifact_output_plan(item)
            for item in (
                (
                    ArtifactSpec.output(
                        "MeasureObjectSizeShape_1_measurements",
                        MeasurementsArtifactType,
                    ),
                )
            )
        ),
        adapter=runtime,
        spec=ArtifactSpec.output(
            "MeasureObjectSizeShape_1_measurements", MeasurementsArtifactType
        ),
        output_value=ShapeObjectMeasurementRows.from_rows(
            rows,
            declared_field_names=("object_label", "Area"),
        ),
        source_image_name="BF_image",
        call_kwargs={},
        source_image_payload=ImagePayloadMetadata(
            source_path="/input/A01_s001_w1.tif"
        ).payload_with(np.zeros((3, 3), dtype=np.float32), None),
        current_image=ImagePayloadMetadata(
            source_path="/input/A01_s001_w1.tif"
        ).payload_with(np.zeros((3, 3), dtype=np.float32), None),
    )

    table = measurement_table_for_module(request)

    assert table.subject == MeasurementSubject(MeasurementScope.OBJECT, "Cells")
    assert table.source_image_name is None


def test_object_output_measurements_derive_count_and_locations_from_output_labels() -> (
    None
):
    labels = ObjectLabelSet(
        name="Cells",
        source_image_name="Mask",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                [
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 2, 2],
                ],
                dtype=np.int32,
            )
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
    )
    object_spec = _output_from_input(
        "Cells",
        "Mask",
        input_type=ImageArtifactType,
    )
    measurement_spec = _measurement_output_for_objects(
        "Watershed_1_measurements",
        object_spec,
    )
    mask_payload = ImagePayloadMetadata(
        source_path="/input/A01_s001_w1_z001_t001.TIF",
        source_image_names=("Mask",),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1_z001_t001.TIF",),
        ),
    ).payload_with(np.ones((3, 3), dtype=np.float32), None)
    runtime = _FakeCellProfilerRuntime({"Mask": mask_payload})
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("Watershed").require_callable(),
            artifact_inputs=(ArtifactSpec.input("Mask", ImageArtifactType),),
            artifact_outputs=(measurement_spec, object_spec),
        ),
        artifact_input_edges=(
            replace(
                _artifact_input_edge_for_test(
                    ArtifactSpec.input("Mask", ImageArtifactType),
                    stored=False,
                ),
                consumes_main_flow=True,
            ),
        ),
        output_plans=tuple(
            _artifact_output_plan(item) for item in ((measurement_spec, object_spec))
        ),
        adapter=runtime,
        spec=measurement_spec,
        output_value=DataclassMeasurementColumnarRows(
            (
                WatershedStats(
                    slice_index=0,
                    object_count=999,
                    mean_area=999.0,
                ),
            ),
            row_type=WatershedStats,
        ),
        source_image_name="Mask",
        source_image_payload=mask_payload,
        current_image=mask_payload,
        call_kwargs={},
    )
    CellProfilerOutputRecorder.for_artifact_type(ObjectLabelsArtifactType).record(
        replace(
            request,
            spec=object_spec,
            output_plan=request.adapter.request.require_artifact_output_plan(
                object_spec.ref()
            ),
            output_value=labels,
        )
    )
    table = measurement_table_for_module(request)

    rows = table.rows.row_mappings()
    count_rows = tuple(row for row in rows if "Count_Cells" in row)
    assert count_rows == ({"slice_index": 0, "Count_Cells": 2},)
    assert type(count_rows[0]["Count_Cells"]) is ObjectCountFeature.measurement_dtype
    assert (
        next(field for field in table.rows.fields if field.name == "Count_Cells").dtype
        is int
    )
    assert all("object_count" not in row and "mean_area" not in row for row in rows)
    assert {
        row["object_label"]
        for row in rows
        if row.get("feature_name") == "Location_Center_X"
    } == {1, 2}
    assert table.subject == MeasurementSubject(MeasurementScope.IMAGE, "Mask")
    assert table.source_image_name == "Mask"

    store = RuntimeValueStore()
    export_table = table.replace_fields(source_path="/input/A01_s001_w1.tif")
    output_plan = ArtifactOutputPlan(
        name=table.name,
        path=f"/memory/{table.name}.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    store.record(
        RuntimeValue.normalize(output_plan, export_table, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    batch = RuntimeArtifactBatch(
        input_specs=(ArtifactSpec.input(table.name, MeasurementsArtifactType),),
        records_by_axis={"A01": store.values()},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )
    settings = CellProfilerDatabaseExportSettings(
        database_type="sqlite",
        sqlite_file="analysis.db",
        experiment_name="Experiment",
        table_prefix="CPA_",
        object_table_mode=CellProfilerObjectTableMode.PER_OBJECT,
        selected_objects=None,
        wants_properties_file=False,
        wants_relationship_tables=False,
    )
    projection = CellProfilerAnalystProjectionBuilder(
        source_binding_plan=CompiledSourceBindingPlan.empty()
    ).build(batch, settings, ())
    assert projection.image_table.rows[0]["Image_Count_Cells"] == 2

    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(CPASQLiteRenderer().render(projection, settings))
        count_column = next(
            row
            for row in connection.execute('PRAGMA table_info("CPA_Per_Image")')
            if row[1] == "Image_Count_Cells"
        )
        assert count_column[2] == "INTEGER"
        assert connection.execute(
            'SELECT "Image_Count_Cells" FROM "CPA_Per_Image"'
        ).fetchone() == (2,)
    finally:
        connection.close()


def test_compiled_measurement_output_preserves_image_and_object_row_ownership() -> None:
    labels = ObjectLabelSet(
        name="Cells",
        source_image_name="Mask",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                [
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 2, 2],
                ],
                dtype=np.int32,
            )
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
    )
    source_spec = ArtifactSpec.input("Mask", ImageArtifactType)
    object_spec = _output_from_input(
        "Cells",
        source_spec.name,
        input_type=source_spec.artifact_type,
    )
    measurement_spec = _measurement_output_for_objects(
        "Watershed_1_measurements",
        object_spec,
    )
    store = RuntimeValueStore()
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=RuntimeExecutionAxisScope("A01_s001"),
        artifact_output_bindings=(
            (
                measurement_spec,
                ArtifactOutputPlan(
                    name=measurement_spec.name,
                    path=f"/memory/{measurement_spec.name}.pkl",
                    artifact_type=measurement_spec.artifact_type,
                    relations=measurement_spec.relations,
                ),
            ),
            (
                object_spec,
                ArtifactOutputPlan(
                    name=object_spec.name,
                    path=f"/memory/{object_spec.name}.pkl",
                    artifact_type=object_spec.artifact_type,
                    relations=object_spec.relations,
                ),
            ),
        ),
        filemanager=_RecordingFileManager(),
    )
    source_image = ImagePayloadMetadata(source_path="/input/Mask.tif").payload_with(
        np.ones((3, 3), dtype=np.float32), None
    )
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("Watershed").require_callable(),
            artifact_inputs=(source_spec,),
            artifact_outputs=(measurement_spec, object_spec),
        ),
        artifact_input_edges=(
            replace(
                _artifact_input_edge_for_test(source_spec, stored=False),
                consumes_main_flow=True,
            ),
        ),
        output_plans=tuple(
            _artifact_output_plan(item) for item in ((measurement_spec, object_spec))
        ),
        adapter=adapter,
        spec=measurement_spec,
        output_value=DataclassMeasurementColumnarRows(
            (
                WatershedStats(
                    slice_index=0,
                    object_count=999,
                    mean_area=999.0,
                ),
            ),
            row_type=WatershedStats,
        ),
        source_image_name="Mask",
        source_image_payload=source_image,
        current_image=source_image,
        call_kwargs={},
    )
    adapter.add_objects(object_spec.name, labels)

    CellProfilerOutputRecorder.for_artifact_type(MeasurementsArtifactType).record(
        request
    )

    records = store.find(
        name=measurement_spec.name,
        artifact_type=MeasurementsArtifactType,
        axis_id="A01_s001",
    )
    tables = tuple(cast(MeasurementTable, record.value.data) for record in records)
    assert len(tables) == 1
    table = tables[0]
    assert table.subject is not None
    assert table.subject.scope is MeasurementScope.IMAGE
    assert table.subject.name == "Mask"
    rows = table.rows.row_mappings()
    image_rows = tuple(row for row in rows if row.get("object_name") is None)
    object_rows = tuple(row for row in rows if row.get("object_name") == "Cells")
    assert [row["Count_Cells"] for row in image_rows] == [2]
    assert {row["feature_name"] for row in object_rows} == {
        "Location_Center_X",
        "Location_Center_Y",
        "Location_Center_Z",
    }


def test_measure_object_size_shape_row_policy_keeps_table_unqualified() -> None:
    policy = MeasureObjectSizeShapeModule.runtime_object_measurement_row_policy()
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="BF_image",
        source_aliases=("BF_image",),
        payload=np.zeros((3, 3), dtype=np.float32),
    )

    assert policy.table_source_image_name((measurement_image,), "BF_image") is None


def test_object_label_output_source_payload_uses_primary_object_input() -> None:
    primary_payload = ObjectLabelSet(
        name="PrimaryObjects",
        variant_data=ObjectLabelVariantData(
            labels=np.array([[0, 1], [0, 0]], dtype=np.int32)
        ),
    )
    auxiliary_payload = ObjectLabelSet(
        name="AuxiliaryObjects",
        variant_data=ObjectLabelVariantData(
            labels=np.array([[0, 0], [2, 0]], dtype=np.int32)
        ),
    )

    object_inputs = (
        ArtifactSpec.input("PrimaryObjects", ObjectLabelsArtifactType),
        ArtifactSpec.input("AuxiliaryObjects", ObjectLabelsArtifactType),
    )
    adapter = _FakeCellProfilerRuntime(
        {},
        objects={
            "PrimaryObjects": primary_payload,
            "AuxiliaryObjects": auxiliary_payload,
        },
        artifact_input_edges=tuple(
            (
                cellprofiler_runtime_input_edge_for_test(
                    ArtifactInputPlan(
                        name=spec.name,
                        path=f"/artifacts/{spec.name}",
                        artifact_type=spec.artifact_type,
                    ),
                    invocation_scope=ComponentGroupScope.ungrouped(),
                    producer_selection_scope=ComponentGroupScope.ungrouped(),
                    component_scopes=(),
                    consumer_variable_components=(),
                )
                for spec in object_inputs
            )
        ),
    )
    output = _output_from_input("Objects", "PrimaryObjects")
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("RelateObjects").require_callable(),
            artifact_inputs=object_inputs,
            artifact_outputs=(output,),
        ),
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(item) for item in (object_inputs)
        ),
        output_plans=tuple(_artifact_output_plan(item) for item in ((output,))),
        adapter=adapter,
        spec=output,
        output_value=np.zeros((2, 2), dtype=np.int32),
        source_image_name=None,
        call_kwargs={},
        source_image_payload=object(),
        current_image=None,
    )

    assert (
        _module_type_for_contract(request.callable_contract)
        .source_context(request)
        .source_payload
        is primary_payload
    )


def test_object_label_output_source_payload_uses_declared_primary_object_input() -> (
    None
):
    current_image = object()
    primary_payload = ObjectLabelSet(
        name="PrimaryObjects",
        variant_data=ObjectLabelVariantData(
            labels=np.array([[0, 1], [0, 0]], dtype=np.int32)
        ),
        source_path="/input/site1.tif",
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/site1.tif",)
        ),
    )

    object_inputs = (ArtifactSpec.input("PrimaryObjects", ObjectLabelsArtifactType),)
    adapter = _FakeCellProfilerRuntime(
        {},
        objects={"PrimaryObjects": primary_payload},
        artifact_input_edges=(
            cellprofiler_runtime_input_edge_for_test(
                ArtifactInputPlan(
                    name="PrimaryObjects",
                    path="/artifacts/PrimaryObjects",
                    artifact_type=ObjectLabelsArtifactType,
                ),
                invocation_scope=ComponentGroupScope.ungrouped(),
                producer_selection_scope=ComponentGroupScope.ungrouped(),
                component_scopes=(),
                consumer_variable_components=(),
            ),
        ),
    )
    source_payload = object()
    output = _output_from_input("Objects", "PrimaryObjects")
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("ResizeObjects").require_callable(),
            artifact_inputs=object_inputs,
            artifact_outputs=(output,),
        ),
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(item) for item in (object_inputs)
        ),
        output_plans=tuple(_artifact_output_plan(item) for item in ((output,))),
        adapter=adapter,
        spec=output,
        output_value=np.zeros((2, 2), dtype=np.int32),
        source_image_name=None,
        call_kwargs={},
        source_image_payload=source_payload,
        current_image=current_image,
    )

    assert (
        _module_type_for_contract(request.callable_contract)
        .source_context(request)
        .source_payload
        is primary_payload
    )


class _RecordingFileManager:
    def __init__(self) -> None:
        self.saved: list[tuple[object, str, str]] = []
        self._existing: set[tuple[str, str]] = set()

    def ensure_directory(self, path: str, backend: str) -> None:
        del path, backend

    def exists(self, path: str, backend: str) -> bool:
        return (path, backend) in self._existing

    def delete(self, path: str, backend: str) -> None:
        self._existing.discard((path, backend))

    def save(self, data: object, path: str, backend: str) -> None:
        self.saved.append((data, path, backend))
        self._existing.add((path, backend))


def test_single_image_replacement_returns_exact_named_bundle() -> None:
    output = ArtifactSpec.output("RGBImage", ImageArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("GrayToColor").require_callable(),
        artifact_outputs=(output,),
    )
    executor = _module_executor(contract)
    output_plan = ArtifactOutputPlan(
        name=output.name,
        path="/memory/RGBImage.pkl",
        artifact_type=output.artifact_type,
    )
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        axis_scope=RuntimeExecutionAxisScope("A01_s001"),
        artifact_output_bindings=((output, output_plan),),
        filemanager=_RecordingFileManager(),
    )
    rgb_image = np.zeros((4, 5, 3), dtype=np.float32)
    adapter.add_image(output.name, rgb_image)

    result = executor._replacement_main_flow_output(
        outputs=((output_plan, output),),
        declared_only_outputs={},
        adapter=adapter,
        current_image=rgb_image,
        invocation_image=rgb_image,
        plane_projection=None,
    )

    assert isinstance(result, ImageOutputBundle)
    assert tuple(context.output_key for context in result.slice_contexts) == (
        output.name,
    )
    np.testing.assert_array_equal(image_payload_data(result.slices[0]), rgb_image)


def test_image_sidecar_is_not_packed_into_replacement_main_flow() -> None:
    output = ArtifactSpec.output("CropGreen", ImageArtifactType)
    sidecar = ArtifactSpec.output(
        "CropGreen__crop_mask",
        ImageArtifactType,
        sidecar_role=ArtifactSidecarRole.CROP_MASK,
    )
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("Crop").require_callable(),
        artifact_outputs=(output, sidecar),
    )
    executor = _module_executor(contract)
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        axis_scope=RuntimeExecutionAxisScope("A01_s001"),
        artifact_output_bindings=tuple(
            (
                spec,
                ArtifactOutputPlan(
                    name=spec.name,
                    path=f"/memory/{spec.name}.pkl",
                    artifact_type=spec.artifact_type,
                    sidecar_role=spec.sidecar_role,
                ),
            )
            for spec in (output, sidecar)
        ),
        filemanager=_RecordingFileManager(),
    )
    cropped_image = np.zeros((4, 5), dtype=np.float32)
    crop_mask = np.ones((4, 5), dtype=bool)
    adapter.add_image(output.name, cropped_image)
    adapter.add_image(sidecar.name, crop_mask)

    result = executor._replacement_main_flow_output(
        outputs=(
            (adapter.request.require_artifact_output_plan(output.ref()), output),
        ),
        declared_only_outputs={},
        adapter=adapter,
        current_image=cropped_image,
        invocation_image=cropped_image,
        plane_projection=None,
    )

    assert isinstance(result, ImageOutputBundle)
    assert tuple(context.output_key for context in result.slice_contexts) == (
        output.name,
    )
    np.testing.assert_array_equal(
        image_payload_data(result.slices[0]),
        cropped_image,
    )


def test_recorded_object_labels_replace_main_flow_without_image_coercion() -> None:
    output = ArtifactSpec.output("SavedCells", ObjectLabelsArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("RelateObjects").require_callable(),
        artifact_outputs=(output,),
    )
    executor = _module_executor(contract)
    output_plan = ArtifactOutputPlan(
        name=output.name,
        path="/memory/SavedCells.pkl",
        artifact_type=output.artifact_type,
    )
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        axis_scope=RuntimeExecutionAxisScope("A01_s001"),
        artifact_output_bindings=((output, output_plan),),
        filemanager=_RecordingFileManager(),
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[0, 1], [1, 0]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
    )
    adapter.add_objects(output.name, labels)

    result = executor._replacement_main_flow_output(
        outputs=((output_plan, output),),
        declared_only_outputs={},
        adapter=adapter,
        current_image=np.zeros((2, 2), dtype=np.float32),
        invocation_image=np.zeros((2, 2), dtype=np.float32),
        plane_projection=None,
    )

    assert isinstance(result, ObjectLabelSet)
    result.validate_artifact_name(output.name)
    np.testing.assert_array_equal(result.labels, labels.labels)


def test_main_flow_strategy_rejects_mixed_artifact_types_in_either_order() -> None:
    image = ArtifactSpec.output("DerivedImage", ImageArtifactType)
    labels = ArtifactSpec.output("SavedCells", ObjectLabelsArtifactType)
    values = (
        (image, np.zeros((2, 2), dtype=np.float32)),
        (
            labels,
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(
                    labels=np.zeros((2, 2), dtype=np.int32)
                )
            ),
        ),
    )

    for outputs in (values, tuple(reversed(values))):
        with pytest.raises(TypeError, match="require one exact artifact type"):
            RuntimeArtifactTypeStrategy.for_main_flow_outputs(outputs)


def test_cellprofiler_adapter_preserves_object_label_source_component_metadata() -> (
    None
):
    store = RuntimeValueStore()
    filemanager = _RecordingFileManager()
    output_spec = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=RuntimeExecutionAxisScope("A01_s001"),
        artifact_output_bindings=(
            (
                output_spec,
                ArtifactOutputPlan(
                    name=output_spec.name,
                    path="/memory/Nuclei.pkl",
                    artifact_type=output_spec.artifact_type,
                ),
            ),
        ),
        filemanager=filemanager,
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[0, 1], [2, 0]], dtype=np.int32)
        ),
        source_path="/input/01_POS002_D.TIF",
        source_component_metadata={
            "well": "01",
            "site": "POS002",
            "channel": "D",
        },
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/01_POS002_D.TIF",),
            component_metadata=({"well": "01", "site": "POS002", "channel": "D"},),
        ),
    )

    record = adapter.add_objects("Nuclei", labels)

    assert isinstance(record.value.data, ObjectLabelSet)
    assert record.value.data.source_path == "/input/01_POS002_D.TIF"
    assert dict(record.value.data.source_component_metadata) == {
        "well": "01",
        "site": "POS002",
        "channel": "D",
    }
    assert record.value.data.source_image_provenance_planes.paths == (
        "/input/01_POS002_D.TIF",
    )
    assert tuple(
        dict(metadata)
        for metadata in record.value.data.source_image_provenance_planes.component_metadata
        if metadata is not None
    ) == ({"well": "01", "site": "POS002", "channel": "D"},)
    assert filemanager.saved[0][0] is record.value.data


def test_default_row_policy_accepts_multi_source_image_row_ownership() -> None:
    rows = (
        {
            "slice_index": 0,
            "source_image_name": "Stain1",
            "feature_name": "x_shift",
            "result_value": 0.0,
        },
        {
            "slice_index": 0,
            "source_image_name": "Stain2",
            "feature_name": "x_shift",
            "result_value": 1.0,
        },
    )

    table = MeasurementTable(
        name="mixed_source_measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            rows,
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("source_image_name", str),
                FieldSpec("feature_name", str),
                FieldSpec("result_value", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
    )

    row_policy = CellProfilerObjectMeasurementRowPolicy()
    row_policy.validate_table_ownership(table)

    assert tuple(table.rows.iter_row_mappings()) == rows
    assert table.source_image_name is None


def test_default_row_policy_preserves_table_source_for_explicit_object_rows() -> None:
    rows = (
        {
            "slice_index": 0,
            "object_name": "Cells",
            "object_label": 1,
            "mean_intensity": 0.5,
        },
    )

    table = MeasurementTable(
        name="cell_intensity_measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            rows,
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("mean_intensity", float),
            ),
        ),
        source_image_name="DNA",
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells", "object_label"),
    )

    row_policy = CellProfilerObjectMeasurementRowPolicy()
    row_policy.validate_table_ownership(table)

    assert tuple(table.rows.iter_row_mappings()) == rows
    assert table.source_image_name == "DNA"


@dataclass(frozen=True, slots=True)
class _SyntheticAxisObjectMeasurement:
    slice_index: int
    object_label: int
    value: float


@dataclass(frozen=True, slots=True)
class _SyntheticTextureObjectMeasurement:
    object_label: int
    scale: int
    direction: int
    gray_levels: int
    angular_second_moment: float


@dataclass(frozen=True, slots=True)
class _SyntheticAreaObjectMeasurement:
    object_label: int
    Area: float


@dataclass(frozen=True, slots=True)
class _ColumnarMeasurementRows(ColumnarRows):
    columns: dict[str, tuple[object, ...]]
    fields: tuple[FieldSpec, ...]
    object_row_identity: MeasurementObjectRowIdentity | None = None

    def __post_init__(self) -> None:
        self.validate_fields()


def _synthetic_object_measurement_function(
    image: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, list[_SyntheticObjectMeasurement]]:
    return image, []


def _synthetic_axis_object_measurement_function(
    image: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, list[_SyntheticAxisObjectMeasurement]]:
    return image, []


def _synthetic_texture_object_measurement_function(
    image: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, list[_SyntheticTextureObjectMeasurement]]:
    return image, []


def _synthetic_area_object_measurement_function(
    image: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, list[_SyntheticAreaObjectMeasurement]]:
    return image, []


def complete_object_measurement_rows(
    rows: ColumnarRows,
    *,
    label_payload,
    object_identity=MeasurementObjectRowIdentity.LABEL_ID,
    row_policy=None,
):
    if row_policy is None:
        row_policy = (
            CompactObjectMeasurementRowIdentityPolicy()
            if object_identity is MeasurementObjectRowIdentity.ROW_ORDINAL
            else CellProfilerObjectMeasurementRowPolicy()
        )
    return row_policy.complete_rows(rows, label_payload=label_payload)


def test_object_measurement_output_recorder_completes_exact_columnar_rows() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 2), dtype=np.int32)),
        domain=ObjectLabelDomain(declared_object_count=2),
    )
    row_policy = CellProfilerObjectMeasurementRowPolicy()
    source_rows = MeasurementSparseColumnarRows.from_rows(
        ({"object_label": 1, "value": 4.0},),
        fields=(
            FieldSpec("object_label", int),
            FieldSpec("value", float),
        ),
        object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
    )
    measurement_output = ArtifactSpec.output(
        "Measurements",
        MeasurementsArtifactType,
    )
    recorder = ObjectMeasurementOutputRecorder(
        callable_contract=_compiled_callable_contract(
            _synthetic_object_measurement_function,
            artifact_outputs=(measurement_output,),
        ),
        measurement_output_plan=ArtifactOutputPlan(
            name=measurement_output.name,
            path="/memory/measurements",
            artifact_type=measurement_output.artifact_type,
        ),
        row_policy=row_policy,
        module_type=MeasureObjectIntensityModule,
        func=_synthetic_object_measurement_function,
        adapter=cast(CellProfilerRuntimeAdapter, object()),
        measurement_images=(),
        object_inputs=(),
        image_measurement_rows=[],
        columnar_rows=[],
        timings=ObjectMeasurementOutputTimings(),
    )

    completed = recorder.completed_measurement_rows(source_rows, payload)

    assert completed.fields == source_rows.fields
    assert completed.object_row_identity is MeasurementObjectRowIdentity.LABEL_ID
    assert completed.covers_declared_object_measurement_domain
    assert completed.row_mappings()[0] == {"object_label": 1, "value": 4.0}
    assert np.isnan(completed.row_mappings()[1]["value"])


def test_payload_scoped_object_measurement_completion_ignores_slice_axis() -> None:
    labels = np.zeros((3, 8, 8), dtype=np.int32)
    for object_id in range(1, 6):
        labels[(object_id - 1) // 2, object_id, object_id] = object_id
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(
            declared_object_count=5,
            scope=ObjectLabelDomainScope.PAYLOAD,
        ),
    )
    rows = MeasurementSparseColumnarRows.from_rows(
        tuple(
            {
                MeasurementRowAxisField.SLICE_INDEX.value: 0,
                MeasurementRowAxisField.OBJECT_LABEL.value: object_id,
                "area": float(object_id),
            }
            for object_id in range(1, 6)
        ),
        fields=(
            FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),
            FieldSpec(MeasurementRowAxisField.OBJECT_LABEL.value, int),
            FieldSpec("area", float),
        ),
        object_row_identity=MeasurementObjectRowIdentity.ROW_ORDINAL,
    )

    completed = complete_object_measurement_rows(
        rows,
        label_payload=label_payload,
        object_identity=MeasurementObjectRowIdentity.ROW_ORDINAL,
        row_policy=CompactObjectMeasurementRowIdentityPolicy(),
    )

    assert [
        measurement_row_mapping(row)[MeasurementRowAxisField.OBJECT_LABEL.value]
        for row in completed
    ] == [1, 2, 3, 4, 5]


class _FakeCellProfilerRuntime(CellProfilerRuntimeAdapter):
    def __init__(
        self,
        images: dict[str, RuntimeArrayData],
        objects: dict[str, ObjectLabelValue] | None = None,
        callable_contract: CallableContract | None = None,
        measurement_tables: dict[str, tuple[MeasurementTable, ...]] | None = None,
        artifact_input_edges: tuple[InvocationArtifactInputEdgePlan, ...] = (),
        artifact_output_bindings: tuple[
            tuple[ArtifactSpec, ArtifactOutputPlan], ...
        ] = (),
        source_bindings: tuple[NamedSourceBinding, ...] = (),
        ordered_pipeline_image_paths: tuple[str, ...] = (),
        variable_components: tuple[VariableComponents, ...] = (),
        plane_projection: RuntimePlaneProjection = RuntimePlaneProjection.stack(),
        axis_scope: RuntimeExecutionAxisScope | None = None,
        source_image_set_identity_policy: SourceImageSetIdentityPolicy = (
            SourceImageSetIdentityPolicy()
        ),
    ) -> None:
        if axis_scope is None:
            axis_scope = RuntimeExecutionAxisScope.from_raw(
                "test-axis",
                component=None,
                value=None,
            )
        source_binding_plan = CompiledSourceBindingPlan(bindings=source_bindings)
        source_binding_context = SourceBindingRuntimeContext(
            step_input_files=ordered_pipeline_image_paths,
            current_step_input_files=ordered_pipeline_image_paths,
            pipeline_input_files=ordered_pipeline_image_paths,
        )
        processing_context = SimpleNamespace(
            microscope_handler=SimpleNamespace(
                parser=SimpleNamespace(semantic_identity=lambda: ()),
            ),
        )
        runtime_value_store = RuntimeValueStore()
        indexed_input_edges = tuple(
            replace(
                edge,
                key=replace(edge.key, input_index=input_index),
            )
            for input_index, edge in enumerate(artifact_input_edges)
        )
        super().__init__(
            request=runtime_adapter_request_for_test(
                runtime_value_store=runtime_value_store,
                callable_contract=callable_contract,
                filemanager=_RecordingFileManager(),
                axis_scope=axis_scope,
                artifact_inputs={edge.key: edge for edge in indexed_input_edges},
                artifact_output_bindings=artifact_output_bindings,
                source_binding_plan=source_binding_plan,
                source_binding_context=source_binding_context,
                microscope_handler=processing_context.microscope_handler,
                variable_components=variable_components,
                plane_projection=plane_projection,
                source_image_set_identity_policy=source_image_set_identity_policy,
            ),
        )
        self.images = images
        self.runtime_objects = dict(objects or {})
        for image_name, image_payload in images.items():
            edges = tuple(
                edge
                for edge in self.request.artifact_inputs.values()
                if edge.storage_plan is not None
                and edge.spec.name == image_name
                and edge.spec.artifact_type is ImageArtifactType
            )
            if not edges:
                continue
            for edge in edges:
                for plan in self._storage_seed_plans(edge):
                    self._store_runtime_artifact(plan, image_payload)
        for object_name, object_labels in (objects or {}).items():
            self._seed_runtime_object(object_name, object_labels)
        self.runtime_measurement_tables = measurement_tables or {}
        self.runtime_measurement_records: dict[
            str,
            list[StoredRuntimeValue],
        ] = {}
        for tables in self.runtime_measurement_tables.values():
            for table in tables:
                self.runtime_measurement_records.setdefault(table.name, []).append(
                    self._store_runtime_artifact(
                        ArtifactOutputPlan(
                            name=table.name,
                            path=f"/artifacts/{table.name}",
                            artifact_type=MeasurementsArtifactType,
                        ),
                        table,
                    )
                )
        self.ordered_pipeline_image_paths = ordered_pipeline_image_paths
        self.measurements: list[MeasurementTable] = []
        self.objects: list[tuple[str, np.ndarray, dict[str, object]]] = []
        self.spatial_grids: dict[str, SpatialGrid] = {}
        self.relationships: list[ObjectRelationship] = []
        self.group_by = GroupBy.NONE

    def install_artifact_input_edges(
        self,
        edges: tuple[InvocationArtifactInputEdgePlan, ...],
    ) -> None:
        """Install exact compiled edges and seed their known fixture payloads."""

        existing = tuple(self.request.artifact_inputs.values())
        unmatched_existing = list(existing)
        added_items = []
        for edge in edges:
            matching_index = next(
                (
                    index
                    for index, existing_edge in enumerate(unmatched_existing)
                    if existing_edge.spec == edge.spec
                ),
                None,
            )
            if matching_index is None:
                added_items.append(edge)
            else:
                unmatched_existing.pop(matching_index)
        added = tuple(added_items)
        indexed = tuple(
            replace(edge, key=replace(edge.key, input_index=input_index))
            for input_index, edge in enumerate((*existing, *added))
        )
        self.request = replace(
            self.request,
            artifact_inputs={edge.key: edge for edge in indexed},
        )
        for edge in added:
            payload = None
            if (
                edge.spec.artifact_type is ImageArtifactType
                and edge.spec.name in self.images
            ):
                payload = self.images[edge.spec.name]
            elif (
                edge.spec.artifact_type is ObjectLabelsArtifactType
                and edge.spec.name in self.runtime_objects
            ):
                payload = self._named_object_labels(
                    edge.spec.name,
                    self.runtime_objects[edge.spec.name],
                )
            if payload is None:
                continue
            for plan in self._storage_seed_plans(edge):
                self._store_runtime_artifact(plan, payload)

    def require_resolvable_source_aliases(self, aliases: tuple[str, ...]) -> None:
        missing = tuple(alias for alias in aliases if alias not in self.images)
        if missing:
            raise AssertionError(f"Unexpected missing image aliases: {missing!r}")

    def has_source_binding(
        self,
        alias: str,
        kind: type[ArtifactType] | None = None,
    ) -> bool:
        binding = self.request.source_binding_plan.binding_for_alias(alias)
        return binding is not None and (kind is None or binding.artifact_kind is kind)

    def runtime_slice_plane_index(self) -> int | None:
        return super().runtime_slice_plane_index()

    def cellprofiler_source_order_path(self, path: str) -> str:
        return path

    def get_image(
        self,
        name: str,
        *,
        group_key: str | None = None,
    ) -> RuntimeArrayData:
        del group_key
        return self.images[name]

    def add_image(
        self,
        name: str,
        data: RuntimeArrayData,
        *,
        materialization_source_metadata: ImagePayloadMetadata | None = None,
    ) -> StoredRuntimeValue:
        self.images[name] = cast(
            RuntimeArrayData,
            ImageArtifactType.normalize_runtime_payload(name, data),
        )
        return super().add_image(
            name,
            data,
            materialization_source_metadata=materialization_source_metadata,
        )

    def get_objects(
        self,
        name: str,
    ) -> ObjectLabelSet:
        record = RuntimeArtifactQueryContext(
            store=self.request.context.runtime_value_store,
            axis_id=self.request.axis_scope.axis_id,
            group_key=self.request.group_key,
            match_group=True,
        ).resolve(
            name=name,
            artifact_type=ObjectLabelsArtifactType,
            purpose="test object-label artifact",
        )
        return cast(ObjectLabelSet, record.value.data)

    def get_objects_across_groups(self, name: str) -> ObjectLabelSet:
        return self.runtime_objects[name]

    def measurement_tables(
        self,
        *,
        group_key: str | None = None,
        match_group: bool = True,
    ) -> tuple[MeasurementTable, ...]:
        del group_key, match_group
        return tuple(
            table
            for tables in self.runtime_measurement_tables.values()
            for table in tables
        )

    def artifact_input_records(
        self,
        name: str,
        artifact_type: type[ArtifactType],
    ) -> tuple[StoredRuntimeValue, ...]:
        if (
            artifact_type is MeasurementsArtifactType
            and name in self.runtime_measurement_records
        ):
            return tuple(self.runtime_measurement_records[name])
        return super().artifact_input_records(name, artifact_type)

    def add_measurements(
        self,
        table: MeasurementTable,
    ) -> None:
        self.measurements.append(table)

    def add_objects(
        self,
        name: str,
        labels: object,
        **kwargs: object,
    ) -> object:
        self.objects.append((name, labels, kwargs))
        return super().add_objects(name, labels, **kwargs)

    def _seed_runtime_object(
        self,
        name: str,
        labels: ObjectLabelValue,
    ) -> None:
        edges = tuple(
            edge
            for edge in self.request.artifact_inputs.values()
            if edge.storage_plan is not None
            and edge.spec.name == name
            and edge.spec.artifact_type is ObjectLabelsArtifactType
        )
        payload = self._named_object_labels(name, labels)
        plans = tuple(plan for edge in edges for plan in self._storage_seed_plans(edge))
        if not plans:
            plans = (
                ArtifactOutputPlan(
                    name=name,
                    path=f"/memory/{name}.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                ),
            )
        for plan in plans:
            self._store_runtime_artifact(plan, payload)

    @staticmethod
    def _storage_seed_plans(
        edge: InvocationArtifactInputEdgePlan,
    ) -> tuple[ArtifactOutputPlan, ...]:
        storage_plan = edge.storage_plan
        if storage_plan is None:
            return ()
        return tuple(
            ArtifactOutputPlan(
                name=storage_plan.name,
                path=storage_plan.path_for_runtime_query(group_key),
                artifact_type=storage_plan.artifact_type,
                group_keys=(group_key,),
                group_component=storage_plan.group_component,
                variable_components=storage_plan.variable_components,
                component_domains=storage_plan.component_domains,
                paths_by_group={
                    group_key: storage_plan.path_for_runtime_query(group_key)
                },
                sidecar_role=storage_plan.sidecar_role,
            )
            for group_key in (storage_plan.group_keys or (None,))
        )

    @staticmethod
    def _named_object_labels(
        name: str,
        labels: ObjectLabelValue,
    ) -> ObjectLabelSet:
        if isinstance(labels, ObjectLabelSet):
            return labels
        if not isinstance(labels, ObjectLabelPayload):
            raise TypeError(
                "Test runtime object inputs require ObjectLabelSet or "
                "ObjectLabelPayload."
            )
        return ObjectLabelSet.from_payload(name, labels)

    def _store_runtime_artifact(
        self,
        plan: ArtifactOutputPlan,
        payload: object,
    ) -> StoredRuntimeValue:
        value = RuntimeValue.normalize(
            plan,
            payload,
            axis_id=self.request.axis_scope.axis_id,
        )
        return self.request.context.runtime_value_store.replace(
            value,
            path=plan.path,
            backend=Backend.MEMORY.value,
        )

    def add_spatial_grid(
        self,
        name: str,
        grid: object,
    ):
        record = super().add_spatial_grid(name, grid)
        self.spatial_grids[name] = cast(
            SpatialGrid | RuntimeSliceAlignedValues,
            record.value.data,
        )
        return record

    def get_spatial_grid(self, name: str) -> SpatialGrid | RuntimeSliceAlignedValues:
        return self.spatial_grids[name]

    def add_relationship(
        self,
        relationship: ObjectRelationship,
        *,
        artifact_type: type[ObjectLineageArtifactType] = RelationshipsArtifactType,
    ):
        record = super().add_relationship(
            relationship,
            artifact_type=artifact_type,
        )
        self.relationships.append(relationship)
        return record


def test_object_label_output_domain_scope_preserves_declared_source_stack() -> None:
    source_spec = ArtifactSpec.input("Threshold", ImageArtifactType)
    output_spec = ArtifactSpec.output_preserving_source_stack_scope(
        "Objects",
        ObjectLabelsArtifactType,
        source_spec,
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("Watershed").require_callable(),
            artifact_inputs=(source_spec,),
            artifact_outputs=(output_spec,),
        ),
    )

    request = _cellprofiler_output_record_request(
        callable_contract=executor.callable_contract,
        artifact_input_edges=(
            replace(
                _artifact_input_edge_for_test(source_spec, stored=False),
                consumes_main_flow=True,
            ),
        ),
        output_plans=tuple(_artifact_output_plan(item) for item in ((output_spec,))),
        spec=output_spec,
        adapter=_FakeCellProfilerRuntime({}),
        output_value=np.ones((3, 4, 5), dtype=np.int32),
        source=CellProfilerMeasurementImage(
            payload=np.ones((3, 4, 5), dtype=np.float32),
            source_image_name="Threshold",
        ),
        current_image=np.ones((3, 4, 5), dtype=np.float32),
        call_kwargs={},
    )

    assert request.object_label_output_domain_scope() is None


def test_object_label_output_domain_scope_uses_declared_group_lineage() -> None:
    source_spec = ArtifactSpec.input("Threshold", ImageArtifactType)
    output_spec = ArtifactSpec.output_inheriting_group_scope(
        "DilatedObjects",
        ObjectLabelsArtifactType,
        source_spec,
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("TrackObjects").require_callable(),
            artifact_inputs=(source_spec,),
            artifact_outputs=(output_spec,),
        ),
    )

    request = _cellprofiler_output_record_request(
        callable_contract=executor.callable_contract,
        artifact_input_edges=(
            replace(
                _artifact_input_edge_for_test(source_spec, stored=False),
                consumes_main_flow=True,
            ),
        ),
        output_plans=tuple(_artifact_output_plan(item) for item in ((output_spec,))),
        spec=output_spec,
        adapter=_FakeCellProfilerRuntime({}),
        output_value=np.ones((3, 4, 5), dtype=np.int32),
        source=CellProfilerMeasurementImage(
            payload=np.ones((3, 4, 5), dtype=np.float32),
            source_image_name="Threshold",
        ),
        current_image=np.ones((3, 4, 5), dtype=np.float32),
        call_kwargs={},
    )

    assert request.object_label_output_domain_scope() is ObjectLabelDomainScope.PAYLOAD


class _CalculateMathObjectOperandAdapter(_FakeCellProfilerRuntime):
    def __init__(self, contract: CallableContract, labels: np.ndarray) -> None:
        object_spec = contract.artifact_inputs.require_by_name_and_artifact_type(
            "Nuclei",
            ObjectLabelsArtifactType,
        )
        measurement_name = "NucleiIntensityMeasurements"
        measurement_spec = contract.artifact_inputs.require_by_name_and_artifact_type(
            measurement_name,
            MeasurementsArtifactType,
        )
        measurement_path = f"/artifacts/{measurement_name}"
        area_measurement_name = "NucleiAreaMeasurements"
        area_measurement_spec = (
            contract.artifact_inputs.require_by_name_and_artifact_type(
                area_measurement_name,
                MeasurementsArtifactType,
            )
        )
        measurements = MeasurementTable(
            name=measurement_name,
            rows=MeasurementSparseColumnarRows.from_rows(
                [
                    {
                        "slice_index": 0,
                        "object_label": 1,
                        "Intensity_MeanIntensity_DNA": 1.0,
                    },
                    {
                        "slice_index": 0,
                        "object_label": 2,
                        "Intensity_MeanIntensity_DNA": 2.0,
                    },
                    {
                        "slice_index": 1,
                        "object_label": 1,
                        "Intensity_MeanIntensity_DNA": 3.0,
                    },
                    {
                        "slice_index": 1,
                        "object_label": 2,
                        "Intensity_MeanIntensity_DNA": 4.0,
                    },
                ],
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_label", int),
                    FieldSpec("Intensity_MeanIntensity_DNA", float),
                ),
            ),
            subject=MeasurementSubject(
                MeasurementScope.OBJECT, "Nuclei", "object_label"
            ),
        )
        super().__init__(
            images={},
            callable_contract=contract,
            objects={
                "Nuclei": ObjectLabelSet(
                    name="Nuclei",
                    variant_data=ObjectLabelVariantData(labels=labels),
                    plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                    domain=ObjectLabelDomain(
                        declared_object_id_domains=((1, 2), (1, 2)),
                        scope=ObjectLabelDomainScope.PLANE,
                    ),
                ),
            },
            artifact_input_edges=(
                _artifact_input_edge_for_test(object_spec),
                _artifact_input_edge_for_test(measurement_spec),
                _artifact_input_edge_for_test(area_measurement_spec),
            ),
            plane_projection=RuntimePlaneProjection.stack(2),
        )
        self._store_runtime_artifact(
            ArtifactOutputPlan(
                name=measurement_name,
                path=measurement_path,
                artifact_type=MeasurementsArtifactType,
            ),
            measurements,
        )
        self._store_runtime_artifact(
            _artifact_output_plan(area_measurement_spec),
            MeasurementTable(
                name=area_measurement_name,
                rows=MeasurementSparseColumnarRows.from_rows(
                    (),
                    fields=(
                        FieldSpec("slice_index", int),
                        FieldSpec("object_label", int),
                        FieldSpec("AreaOccupied_AreaOccupied_Nuclei", float),
                    ),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.OBJECT,
                    "Nuclei",
                    "object_label",
                ),
            ),
        )


def _calculate_math_object_contract() -> CallableContract:
    from openhcs.core.artifacts import (
        ArtifactInputPlan,
        ArtifactSpecCollection,
        GroupLineageSourceRelation,
        MeasurementsArtifactType,
    )
    from openhcs.core.function_patterns import FunctionInvocationKey
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
    from openhcs.core.pipeline.artifact_planning import (
        ArtifactProducer,
        artifact_producers_for_outputs,
    )
    from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting

    module = ModuleBlock(
        name="CalculateMath",
        module_num=7,
        setting_records=[
            ModuleSetting("Name the output measurement", "Ratio"),
            ModuleSetting("Operation", "Divide"),
            ModuleSetting("Select the numerator objects", "Nuclei"),
            ModuleSetting(
                "Select the numerator measurement",
                "Intensity_MeanIntensity_DNA",
            ),
            ModuleSetting("Select the denominator objects", "None"),
            ModuleSetting(
                "Select the denominator measurement",
                "AreaOccupied_AreaOccupied_Nuclei",
            ),
        ],
    )
    nuclei = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    (nuclei_producer,) = artifact_producers_for_outputs(
        (nuclei,),
        groups=(None,),
        invocation_keys=(
            FunctionInvocationKey("identify_primary_objects", "default", 0),
        ),
    )

    def measurement_artifact(
        name: str,
        function_name: str,
    ) -> tuple[ArtifactSpec, ArtifactProducer]:
        spec = ArtifactSpec.output(
            name,
            MeasurementsArtifactType,
            relations=(
                GroupLineageSourceRelation(
                    source=nuclei.for_plan_type(ArtifactInputPlan).ref()
                ),
            ),
        )
        (producer,) = artifact_producers_for_outputs(
            (spec,),
            groups=(None,),
            invocation_keys=(FunctionInvocationKey(function_name, "default", 0),),
        )
        return spec, producer

    intensity_measurements, intensity_producer = measurement_artifact(
        "NucleiIntensityMeasurements",
        "measure_object_intensity",
    )
    area_measurements, area_producer = measurement_artifact(
        "NucleiAreaMeasurements",
        "measure_image_area_occupied",
    )
    available = ArtifactSpecCollection(
        (nuclei, intensity_measurements, area_measurements)
    )
    return CalculateMathModule.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey("calculate_math", "default", 0),
        step_context=ArtifactDeclarationStepContext(
            step_name="CalculateMath",
            step_index=6,
            available_artifacts=available,
            main_flow_artifacts=ArtifactSpecCollection(()),
            available_artifact_producers=(
                nuclei_producer,
                intensity_producer,
                area_producer,
            ),
        ),
    )


def test_calculate_math_object_operands_preserve_label_slice_domain() -> None:
    labels = np.asarray(
        [
            [[0, 1], [2, 0]],
            [[0, 1], [0, 2]],
        ],
        dtype=np.int32,
    )
    contract = _calculate_math_object_contract()
    adapter = _CalculateMathObjectOperandAdapter(contract, labels)
    object_spec = contract.artifact_inputs.by_name_and_artifact_type(
        "Nuclei",
        ObjectLabelsArtifactType,
    )
    assert object_spec is not None
    measurement_spec = contract.artifact_inputs.by_name_and_artifact_type(
        "NucleiIntensityMeasurements",
        MeasurementsArtifactType,
    )
    assert measurement_spec is not None
    request = RuntimeInputBindingRequest(
        selected_object_inputs=(object_spec,),
        adapter=adapter,
        kwargs={
            "operand1_feature": "Intensity_MeanIntensity_DNA",
        },
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    value = CalculateMathInputPolicy().operand_value(
        request,
        feature_kwarg="operand1_feature",
        object_spec=object_spec,
    )

    assert isinstance(value, CellProfilerSliceAlignedValues)
    assert value.slice_count == 2
    np.testing.assert_array_equal(value.value_for_slice(0), [1.0, 2.0])
    np.testing.assert_array_equal(value.value_for_slice(1), [3.0, 4.0])


def test_calculate_math_count_operand_preserves_label_slice_domain() -> None:
    labels = np.asarray(
        [
            [[0, 1], [2, 0]],
            [[0, 1], [0, 0]],
        ],
        dtype=np.int32,
    )
    contract = _calculate_math_object_contract()
    adapter = _CalculateMathObjectOperandAdapter(contract, labels)
    request = RuntimeInputBindingRequest(
        selected_object_inputs=contract.artifact_inputs.of_artifact_type(
            ObjectLabelsArtifactType
        ),
        adapter=adapter,
        kwargs={"operand1_feature": "Count_Nuclei"},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    value = CalculateMathInputPolicy.operand_value(
        request,
        feature_kwarg="operand1_feature",
        object_spec=None,
    )

    assert isinstance(value, RuntimeSliceAlignedValues)
    assert value.slices == (2.0, 2.0)


def test_calculate_math_uses_public_output_name() -> None:
    _image, rows = calculate_math(
        np.zeros((2, 2), dtype=np.float32),
        operand1_value=4.0,
        operand2_value=2.0,
        operation=MathOperation.DIVIDE,
        output_name="Ratio",
    )

    assert tuple(rows.column_values("output_name")) == ("Ratio",)


def test_calculate_math_binds_operand_object_identity_from_artifact_contract() -> None:
    labels = np.asarray(
        [
            [[0, 1], [2, 0]],
            [[0, 1], [0, 2]],
        ],
        dtype=np.int32,
    )
    contract = _calculate_math_object_contract()
    adapter = _CalculateMathObjectOperandAdapter(contract, labels)
    object_spec = contract.artifact_inputs.by_name_and_artifact_type(
        "Nuclei",
        ObjectLabelsArtifactType,
    )
    assert object_spec is not None
    measurement_spec = contract.artifact_inputs.by_name_and_artifact_type(
        "NucleiIntensityMeasurements",
        MeasurementsArtifactType,
    )
    assert measurement_spec is not None
    request = RuntimeInputBindingRequest(
        selected_object_inputs=(object_spec,),
        adapter=adapter,
        kwargs={
            "operand1_feature": "Intensity_MeanIntensity_DNA",
        },
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    operand_specs = CalculateMathInputPolicy.operand_object_specs(request)
    assert operand_specs == (object_spec, None)

    value = CalculateMathInputPolicy().operand_value(
        request,
        feature_kwarg="operand1_feature",
        object_spec=object_spec,
    )

    assert isinstance(value, CellProfilerSliceAlignedValues)
    np.testing.assert_array_equal(value.value_for_slice(0), [1.0, 2.0])
    np.testing.assert_array_equal(value.value_for_slice(1), [3.0, 4.0])
    assert "operand1_object_name" not in request.kwargs


def test_runtime_slice_projection_projects_cellprofiler_slice_aligned_values() -> None:
    values = CellProfilerSliceAlignedValues(
        (
            np.asarray([1.0, 2.0]),
            np.asarray([3.0]),
        )
    )

    np.testing.assert_array_equal(
        RuntimeSliceProjection.value_for_slice(
            values,
            RuntimePlaneAxisValueProjection.from_selected_plane(
                axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
            ),
        ),
        np.asarray([3.0]),
    )


def test_cellprofiler_measurement_vector_preserves_singleton_slice_alignment() -> None:
    value = CellProfilerMeasurementVector(
        (np.asarray([1.0, 2.0]),),
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=1,
        ),
    ).runtime_value

    assert isinstance(value, CellProfilerSliceAlignedValues)
    assert value.slice_count == 1
    np.testing.assert_array_equal(value.value_for_slice(0), [1.0, 2.0])


def test_cellprofiler_measurement_vector_without_plane_domain_is_not_slice_aligned() -> (
    None
):
    value = CellProfilerMeasurementVector((np.asarray([1.0, 2.0]),)).runtime_value

    np.testing.assert_array_equal(value, [1.0, 2.0])


def test_cellprofiler_measurement_vector_consumes_selected_plane() -> None:
    value = CellProfilerMeasurementVector(
        (np.asarray([1.0, 2.0]),),
        plane_projection=RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_index=0,
            axis_size=1,
        ),
    ).runtime_value

    np.testing.assert_array_equal(value, [1.0, 2.0])


def test_single_object_input_policy_preserves_native_label_contract() -> None:
    labels = ObjectLabelSet(
        name="InputObjects",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                (
                    ((0, 1), (0, 0)),
                    ((0, 0), (2, 0)),
                ),
                dtype=np.int32,
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    object_spec = ArtifactSpec.input(
        "InputObjects", ObjectLabelsArtifactType, parameter_name="labels"
    )
    contract = _compiled_callable_contract(
        measure_object_size_shape,
        artifact_inputs=(object_spec,),
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            objects={object_spec.name: labels},
            artifact_input_edges=(
                _artifact_input_edge_for_test(
                    object_spec,
                ),
            ),
            plane_projection=RuntimePlaneProjection.stack(2),
        ),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    bound = MeasureObjectSizeShapeModule.bind_runtime_inputs(request)

    assert bound["labels"] is labels


def test_object_row_binding_projects_the_current_runtime_plane() -> None:
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                (
                    ((0, 1), (0, 0)),
                    ((0, 0), (2, 0)),
                ),
                dtype=np.int32,
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    object_spec = ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    contract = _compiled_callable_contract(
        filter_objects,
        artifact_inputs=(object_spec,),
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            objects={object_spec.name: labels},
            artifact_input_edges=(_artifact_input_edge_for_test(object_spec),),
            plane_projection=RuntimePlaneProjection.selected(1, 2),
        ),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    bound_labels = request.label_argument_for(object_spec, "object_labels")

    np.testing.assert_array_equal(
        object_label_dense_array(bound_labels),
        object_label_dense_array(labels)[1],
    )


@pytest.mark.parametrize(
    ("plane_projection", "expected_shape", "expected_plane_axis", "preserved"),
    (
        pytest.param(
            RuntimePlaneProjection.stack(1),
            (2, 2),
            None,
            False,
            id="implicit-singleton-selection",
        ),
        pytest.param(
            RuntimePlaneProjection.selected(0, 1),
            (2, 2),
            None,
            False,
            id="explicit-selection",
        ),
    ),
)
def test_object_row_binding_honors_authoritative_singleton_runtime_projection(
    plane_projection: RuntimePlaneProjection,
    expected_shape: tuple[int, ...],
    expected_plane_axis: RuntimePlaneAxis | None,
    preserved: bool,
) -> None:
    label_array = np.asarray(
        (((0, 1), (0, 0)),),
        dtype=np.int32,
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_array),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    object_spec = ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    contract = _compiled_callable_contract(
        filter_objects,
        artifact_inputs=(object_spec,),
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            objects={object_spec.name: labels},
            artifact_input_edges=(_artifact_input_edge_for_test(object_spec),),
            plane_projection=plane_projection,
        ),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    bound_labels = request.label_argument_for(object_spec, "object_labels")

    assert isinstance(bound_labels, ObjectLabelValue)
    assert (bound_labels is labels) is preserved
    assert bound_labels.plane_axis is expected_plane_axis
    assert object_label_dense_array(bound_labels).shape == expected_shape
    np.testing.assert_array_equal(
        object_label_dense_array(bound_labels),
        label_array if preserved else label_array[0],
    )


def test_object_row_binding_preserves_nominal_parameter_abi_without_domain() -> None:
    label_array = np.asarray(
        (((0, 1), (0, 0)),),
        dtype=np.int32,
    )
    labels = ObjectLabelSet(
        name="Cells", variant_data=ObjectLabelVariantData(labels=label_array)
    )
    object_spec = ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    contract = _compiled_callable_contract(
        filter_objects,
        artifact_inputs=(object_spec,),
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            objects={object_spec.name: labels},
            artifact_input_edges=(_artifact_input_edge_for_test(object_spec),),
            plane_projection=RuntimePlaneProjection.stack(1),
        ),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    bound_labels = request.label_argument_for(object_spec, "object_labels")

    assert isinstance(bound_labels, ObjectLabelValue)
    np.testing.assert_array_equal(object_label_dense_array(bound_labels), label_array)


def test_adapter_object_record_uses_nominal_full_stack_binding() -> None:
    label_array = np.asarray(
        (((0, 1), (0, 0)),),
        dtype=np.int32,
    )
    store = RuntimeValueStore()
    filemanager = _RecordingFileManager()
    output_plan = ArtifactOutputPlan(
        name="Cells",
        path="/memory/Cells.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    output_spec = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)
    producer = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        axis_scope=RuntimeExecutionAxisScope("A01_s001"),
        artifact_output_bindings=((output_spec, output_plan),),
        filemanager=filemanager,
    )
    producer.add_objects(
        "Cells",
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=label_array),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            domain=ObjectLabelDomain(
                declared_object_id_domains=((1,),),
                scope=ObjectLabelDomainScope.PLANE,
            ),
        ),
    )
    object_spec = ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    input_edge = cellprofiler_runtime_input_edge_for_test(
        ArtifactInputPlan(
            name="Cells",
            path=output_plan.path,
            artifact_type=ObjectLabelsArtifactType,
        ),
        spec=object_spec,
        invocation_scope=ComponentGroupScope.ungrouped(),
        producer_selection_scope=ComponentGroupScope.ungrouped(),
        component_scopes=(),
        consumer_variable_components=(),
    )
    consumer = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            callable_contract=_compiled_callable_contract(
                measure_object_size_shape,
                artifact_inputs=(object_spec,),
            ),
        axis_scope=RuntimeExecutionAxisScope("A01_s001"),
        artifact_inputs={input_edge.key: input_edge},
        filemanager=filemanager,
    )
    request = RuntimeInputBindingRequest(
        adapter=consumer,
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    bound_labels = request.label_argument_for(object_spec, "labels")

    assert isinstance(bound_labels, ObjectLabelValue)
    np.testing.assert_array_equal(object_label_dense_array(bound_labels), label_array)


@pytest.mark.parametrize(
    "slice_by_slice",
    (True, False),
)
def test_image_stack_requirement_does_not_control_scalar_object_input_projection(
    slice_by_slice: bool,
) -> None:
    label_array = np.asarray(
        (((0, 1), (0, 0)),),
        dtype=np.int32,
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_array),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    object_spec = ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    def consume_labels(
        image: np.ndarray,
        labels: ObjectLabelValue,
        *,
        slice_by_slice: bool = False,
    ) -> np.ndarray:
        del labels, slice_by_slice
        return image

    consume_labels.__processing_contract__ = ProcessingContract.FLEXIBLE
    contract = _compiled_callable_contract(
        consume_labels,
        artifact_inputs=(object_spec,),
    )
    adapter = _FakeCellProfilerRuntime(
        {},
        callable_contract=contract,
        objects={object_spec.name: labels},
        artifact_input_edges=(_artifact_input_edge_for_test(object_spec),),
        plane_projection=RuntimePlaneProjection.selected(0, 1),
    )
    request = RuntimeInputBindingRequest(
        adapter=adapter,
        kwargs={"slice_by_slice": slice_by_slice},
        current_image=np.zeros((1, 2, 2), dtype=np.float32),
    )

    bound_labels = request.label_argument_for(object_spec, "labels")

    assert isinstance(bound_labels, ObjectLabelValue)
    assert bound_labels.plane_axis is None
    assert object_label_dense_array(bound_labels).shape == (2, 2)


@pytest.mark.parametrize(
    ("slice_by_slice", "expected_shape", "expected_plane_axis"),
    (
        (True, (2, 2), None),
        (False, (1, 2, 2), RuntimePlaneAxis.RUNTIME_SLICE),
    ),
)
def test_match_image_stack_object_input_follows_declared_image_execution(
    slice_by_slice: bool,
    expected_shape: tuple[int, ...],
    expected_plane_axis: RuntimePlaneAxis | None,
) -> None:
    label_array = np.asarray(
        (((0, 1), (0, 0)),),
        dtype=np.int32,
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_array),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    plane_projection = (
        RuntimePlaneProjection.selected(0, 1)
        if slice_by_slice
        else RuntimePlaneProjection.stack(1)
    )
    object_spec = ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    @object_label_input_execution_mode(ObjectLabelInputExecutionMode.MATCH_IMAGE_STACK)
    def consume_labels(
        image: np.ndarray,
        labels: ObjectLabelValue,
        *,
        slice_by_slice: bool = False,
    ) -> np.ndarray:
        del labels, slice_by_slice
        return image

    consume_labels.__processing_contract__ = ProcessingContract.FLEXIBLE
    contract = _compiled_callable_contract(
        consume_labels,
        artifact_inputs=(object_spec,),
    )
    adapter = _FakeCellProfilerRuntime(
        {},
        callable_contract=contract,
        objects={object_spec.name: labels},
        artifact_input_edges=(_artifact_input_edge_for_test(object_spec),),
        plane_projection=plane_projection,
    )
    request = RuntimeInputBindingRequest(
        adapter=adapter,
        kwargs={"slice_by_slice": slice_by_slice},
        current_image=np.zeros((1, 2, 2), dtype=np.float32),
    )

    bound_labels = request.label_argument_for(object_spec, "labels")

    assert isinstance(bound_labels, ObjectLabelValue)
    assert bound_labels.plane_axis is expected_plane_axis
    assert object_label_dense_array(bound_labels).shape == expected_shape


def test_pure_3d_executor_preserves_nominal_singleton_object_label_stack() -> None:
    label_array = np.asarray(
        (((0, 1), (0, 0)),),
        dtype=np.int32,
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_array),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    seen_label_shape: list[tuple[int, ...]] = []

    def full_stack_identity(
        image: np.ndarray,
        *,
        labels: ObjectLabelSet,
    ) -> np.ndarray:
        seen_label_shape.append(object_label_dense_array(labels).shape)
        return image

    full_stack_identity.__processing_contract__ = ProcessingContract.PURE_3D
    callable_contract = _compiled_callable_contract(
        full_stack_identity,
        artifact_outputs=(ArtifactSpec.output("Image", ImageArtifactType),),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        full_stack_identity,
        np.zeros((1, 2, 2), dtype=np.float32),
        {"labels": labels},
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    assert result.shape == (1, 2, 2)
    assert seen_label_shape == [(1, 2, 2)]


def test_object_row_binding_returns_current_runtime_plane_relationship() -> None:
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                (
                    ((0, 1), (0, 0)),
                    ((0, 0), (2, 0)),
                ),
                dtype=np.int32,
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    declaration = ObjectRelationshipDeclaration.parent_child(
        source=ArtifactSpec.output("Parents", ObjectLabelsArtifactType).ref(),
        target=ArtifactSpec.output("Cells", ObjectLabelsArtifactType).ref(),
        producer_module_number=1,
    )
    relationship = ObjectRelationship(
        name="Parents_Cells_relationships",
        declaration=declaration,
        payload=DirectedObjectRelationshipPayload(
            source_ids=(10, 20, 21),
            target_ids=(1, 2, 3),
            slice_indices=(0, 1, 1),
            slice_count=2,
        ),
    )
    object_spec = ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    relationship_spec = ArtifactSpec.input(
        "Parents_Cells_relationships",
        RelationshipsArtifactType,
    )
    contract = _compiled_callable_contract(
        filter_objects,
        artifact_inputs=(object_spec, relationship_spec),
    )
    adapter = _FakeCellProfilerRuntime(
        {},
        callable_contract=contract,
        objects={object_spec.name: labels},
        artifact_input_edges=(
            _artifact_input_edge_for_test(object_spec),
            _artifact_input_edge_for_test(relationship_spec),
        ),
        plane_projection=RuntimePlaneProjection.selected(1, 2),
    )
    adapter._store_runtime_artifact(
        _artifact_output_plan(relationship_spec),
        relationship,
    )
    request = RuntimeInputBindingRequest(
        selected_object_inputs=(object_spec,),
        adapter=adapter,
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    projected = request.current_plane_relationship_for(relationship_spec)

    assert projected.payload.source_ids == (20, 21)
    assert projected.payload.target_ids == (2, 3)
    assert projected.payload.slice_count == 1


def test_compiled_callable_contract_validates_public_kwargs() -> None:
    kwargs = {
        "unclump_method": "Shape",
        "fill_holes": "After both thresholding and declumping",
        "limit_erase": "Continue",
    }

    validated = CallableContract.from_callable(
        identify_primary_objects
    ).validate_public_kwargs(
        {
            "unclump_method": "Shape",
            "fill_holes": "After both thresholding and declumping",
            "limit_erase": "Continue",
        }
    )

    assert validated == tuple(kwargs.items())


def test_cellprofiler_contract_executor_applies_pure_2d_after_input_resolution():
    calls = []

    def add_one(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image + 1

    add_one.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        add_one, artifact_outputs=(ArtifactSpec.output("Filtered", ImageArtifactType),)
    )
    stack = ImageMetadataPayload(
        np.zeros((2, 4, 5), dtype=np.uint16),
        ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        add_one,
        stack,
        {},
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    assert calls == [(4, 5), (4, 5)]
    assert image_payload_data(result).shape == stack.shape
    np.testing.assert_array_equal(
        image_payload_data(result),
        np.ones_like(stack.data),
    )


def test_cellprofiler_contract_executor_preserves_declared_two_channel_color_plane():
    calls = []

    def split_first_channel(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image[..., 0]

    split_first_channel.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        split_first_channel,
        artifact_outputs=(ArtifactSpec.output("Gray", ImageArtifactType),),
    )
    image = ImageMetadataPayload(
        np.zeros((4, 5, 2), dtype=np.float32),
        ImagePayloadMetadata(source_channel_axis=-1),
    )
    image.data[..., 0] = 7.0

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        split_first_channel,
        image,
        {},
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    assert calls == [(4, 5, 2)]
    np.testing.assert_array_equal(
        image_payload_data(result),
        np.full((4, 5), 7.0),
    )


def test_cellprofiler_contract_executor_slices_declared_two_channel_color_stack():
    calls = []

    def split_first_channel(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image[..., 0]

    split_first_channel.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        split_first_channel,
        artifact_outputs=(ArtifactSpec.output("Gray", ImageArtifactType),),
    )
    stack = ImageMetadataPayload(
        np.zeros((2, 4, 5, 2), dtype=np.float32),
        ImagePayloadMetadata(
            source_channel_axis=-1,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                component_metadata=(
                    {"site": "1"},
                    {"site": "2"},
                ),
            ),
        ),
    )
    stack.data[0, ..., 0] = 3.0
    stack.data[1, ..., 0] = 9.0

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        split_first_channel,
        stack,
        {},
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    assert calls == [(4, 5, 2), (4, 5, 2)]
    np.testing.assert_array_equal(
        image_payload_data(result),
        np.stack(
            (
                np.full((4, 5), 3.0, dtype=np.float32),
                np.full((4, 5), 9.0, dtype=np.float32),
            )
        ),
    )


def test_cellprofiler_contract_executor_preserves_inner_volume_dimensions():
    calls = []

    def add_labels(image: np.ndarray, *, labels: np.ndarray) -> np.ndarray:
        dense_labels = object_label_dense_array(labels)
        calls.append((image.shape, dense_labels.shape, int(dense_labels[0, 0, 0])))
        return image + dense_labels

    add_labels.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        add_labels,
        artifact_inputs=(ArtifactSpec.input("Labels", ObjectLabelsArtifactType),),
        artifact_outputs=(ArtifactSpec.output("Masked", ImageArtifactType),),
    )
    image = ImageMetadataPayload(
        np.zeros((2, 3, 4, 5), dtype=np.float32),
        ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )
    label_data = np.arange(2, dtype=np.int32).reshape((2, 1, 1, 1))
    label_data = np.broadcast_to(label_data, image.shape)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=label_data),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((), (1,)),
        ),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        add_labels,
        image,
        {"labels": labels},
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    assert calls == [((3, 4, 5), (3, 4, 5), index) for index in range(2)]
    assert image_payload_data(result).shape == image.shape
    np.testing.assert_array_equal(image_payload_data(result), label_data)


def test_cellprofiler_contract_executor_rejects_kwarg_only_runtime_axis():
    calls = []

    def keep_labels(
        image: np.ndarray, *, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append((image.shape, labels.shape, int(labels[0, 0, 0])))
        return image, labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        keep_labels,
        artifact_outputs=(
            ArtifactSpec.output("Image", ImageArtifactType),
            ArtifactSpec.output("Labels", ObjectLabelsArtifactType),
        ),
    )
    image = np.zeros((4, 5), dtype=np.float32)
    label_data = np.arange(3, dtype=np.int32).reshape((3, 1, 1, 1))
    label_data = np.broadcast_to(label_data, (3, 2, 4, 5))
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=label_data),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((), (1,), (2,)),
        ),
    )

    with pytest.raises(
        RuntimeSliceProjectionDeclarationError,
        match="Kwargs cannot create image-axis execution semantics",
    ):
        CellProfilerFunctionContractExecutor().execute(
            callable_contract,
            keep_labels,
            image,
            {"labels": labels},
            execution_mode=ImagePayloadExecutionMode.NATURAL,
        )

    assert calls == []


def test_runtime_slice_projection_does_not_count_high_rank_kwargs_by_first_axis():
    labels = np.zeros((3, 2, 4, 5), dtype=np.int32)

    assert RuntimeSliceProjection.slice_count_from_values((labels,)) is None


def test_cellprofiler_contract_executor_projects_declared_runtime_slice_label_kwargs():
    calls = []

    def keep_labels(
        image: np.ndarray, *, labels: ObjectLabelValue
    ) -> tuple[np.ndarray, ObjectLabelValue]:
        dense_labels = object_label_dense_array(labels)
        calls.append((image.shape, dense_labels.shape, int(dense_labels[0, 0])))
        return image, labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        keep_labels,
        artifact_outputs=(
            ArtifactSpec.output("Image", ImageArtifactType),
            ArtifactSpec.output("Labels", ObjectLabelsArtifactType),
        ),
    )
    image = ImageMetadataPayload(
        np.zeros((3, 4, 5), dtype=np.float32),
        ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )
    label_data = np.zeros((3, 4, 5), dtype=np.int32)
    label_data[0, 1:3, 1:3] = 1
    label_data[1, 2:4, 2:4] = 2
    label_data[2, 1:4, 1:4] = 3
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=label_data),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (2,), (3,)),
        ),
    )

    result_image, result_labels = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        keep_labels,
        image,
        {"labels": labels},
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=3,
        ),
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    assert calls == [((4, 5), (4, 5), 0) for _ in range(3)]
    assert image_payload_data(result_image).shape == image.shape
    np.testing.assert_array_equal(object_label_dense_array(result_labels), label_data)


def test_cellprofiler_contract_executor_stacks_singleton_plane_outputs():
    def add_singleton_plane(image: np.ndarray) -> np.ndarray:
        return image[np.newaxis, ...] + 1

    add_singleton_plane.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        add_singleton_plane,
        artifact_outputs=(ArtifactSpec.output("Filtered", ImageArtifactType),),
    )
    stack = ImageMetadataPayload(
        np.zeros((2, 4, 5), dtype=np.uint16),
        ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        add_singleton_plane,
        stack,
        {},
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    assert image_payload_data(result).shape == (2, 1, 4, 5)
    np.testing.assert_array_equal(
        image_payload_data(result),
        np.ones((2, 1, 4, 5), dtype=np.uint16),
    )


def test_cellprofiler_contract_executor_stacks_singleton_color_outputs():
    def add_singleton_color_plane(image: np.ndarray) -> np.ndarray:
        rgb = np.repeat(image[..., np.newaxis], 3, axis=-1)
        return rgb[np.newaxis, ...] + 1

    add_singleton_color_plane.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        add_singleton_color_plane,
        artifact_outputs=(ArtifactSpec.output("Color", ImageArtifactType),),
    )
    stack = ImageMetadataPayload(
        np.zeros((2, 4, 5), dtype=np.uint16),
        ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        add_singleton_color_plane,
        stack,
        {},
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    assert image_payload_data(result).shape == (2, 1, 4, 5, 3)
    np.testing.assert_array_equal(
        image_payload_data(result),
        np.ones((2, 1, 4, 5, 3), dtype=np.uint16),
    )


def test_cellprofiler_contract_executor_stacks_singleton_volume_outputs():
    def add_singleton_volume(image: np.ndarray) -> np.ndarray:
        volume = np.stack((image, image + 1), axis=0)
        return volume[np.newaxis, ...]

    add_singleton_volume.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        add_singleton_volume,
        artifact_outputs=(ArtifactSpec.output("Volume", ImageArtifactType),),
    )
    stack = ImageMetadataPayload(
        np.zeros((3, 4, 5), dtype=np.uint16),
        ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        add_singleton_volume,
        stack,
        {},
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=3,
        ),
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    assert image_payload_data(result).shape == (3, 1, 2, 4, 5)
    np.testing.assert_array_equal(
        image_payload_data(result)[:, 0, 1],
        np.ones((3, 4, 5), dtype=np.uint16),
    )


def test_complete_object_measurement_rows_uses_declared_label_domain() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((4, 4), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_count=3,
        ),
    )

    rows = complete_object_measurement_rows(
        MeasurementSparseColumnarRows.from_rows(
            (),
            fields=(
                FieldSpec("object_label", int),
                FieldSpec("value", float),
            ),
            object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
        ),
        label_payload=payload,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3]
    assert all(np.isnan(row["value"]) for row in rows)


def test_complete_object_measurement_rows_uses_payload_measurement_axis() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((4, 4), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_count=2,
        ),
    )

    rows = complete_object_measurement_rows(
        MeasurementSparseColumnarRows.from_rows(
            (),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("value", float),
            ),
            object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
        ),
        label_payload=payload,
    )

    assert [
        (row["slice_index"], row["object_label"]) for row in rows.iter_row_mappings()
    ] == [(0, 1), (0, 2)]
    assert all(np.isnan(row["value"]) for row in rows.iter_row_mappings())


def test_complete_object_measurement_rows_preserves_sliced_object_label_set_domain() -> (
    None
):
    labels = np.zeros((2, 4, 4), dtype=np.int32)
    labels[0, 1, 1] = 1
    labels[1, 2, 2] = 1
    payload = ObjectLabelSet(
        name="GridObjects",
        variant_data=ObjectLabelVariantData(labels=labels),
        source_image_name="BF_image",
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1, 2, 3), (1, 2, 3)),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    rows = complete_object_measurement_rows(
        MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "object_label": 1, "value": 10.0},
                {"slice_index": 1, "object_label": 1, "value": 20.0},
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("value", float),
            ),
            object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
        ),
        label_payload=payload,
    )

    by_key = {(row["slice_index"], row["object_label"]): row["value"] for row in rows}
    assert tuple(by_key) == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 1),
        (1, 2),
        (1, 3),
    )
    assert by_key[(0, 1)] == 10.0
    assert by_key[(1, 1)] == 20.0
    assert np.isnan(by_key[(0, 2)])
    assert np.isnan(by_key[(0, 3)])
    assert np.isnan(by_key[(1, 2)])
    assert np.isnan(by_key[(1, 3)])


def test_measurement_table_recording_preserves_runtime_slice_rows() -> None:
    runtime = _FakeCellProfilerRuntime({})
    table = MeasurementTable(
        name="AreaShape",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "object_label": 1, "area": 10.0},
                {"slice_index": 1, "object_label": 1, "area": 20.0},
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("area", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )

    runtime.add_measurements(table)

    recorded = runtime.measurements[-1]
    recorded_rows = recorded.rows.row_mappings()
    assert recorded is table
    assert [row["slice_index"] for row in recorded_rows] == [0, 1]
    assert all("image_number" not in row for row in recorded_rows)


def test_measurement_table_derives_exact_schema_from_dataclass_rows() -> None:
    @dataclass(frozen=True, slots=True)
    class StackStats:
        object_count: int
        mean_volume_before: float
        mean_volume_after: float

    table = MeasurementTable(
        name="DilationStats",
        rows=DataclassMeasurementColumnarRows(
            (
                StackStats(
                    object_count=3,
                    mean_volume_before=12.0,
                    mean_volume_after=15.0,
                ),
            ),
            row_type=StackStats,
        ),
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
    )

    assert table.rows.row_mappings() == (
        {
            "object_count": 3,
            "mean_volume_before": 12.0,
            "mean_volume_after": 15.0,
        },
    )
    assert tuple(field.name for field in table.rows.fields) == (
        "object_count",
        "mean_volume_before",
        "mean_volume_after",
    )


def test_measurement_table_preserves_runtime_axis_source_provenance() -> None:
    rows = MeasurementSparseColumnarRows.from_rows(
        (
            {"slice_index": 0, "object_label": 1, "area": 10.0},
            {"slice_index": 1, "object_label": 1, "area": 20.0},
        ),
        fields=(
            FieldSpec("slice_index", int),
            FieldSpec("object_label", int),
            FieldSpec("area", float),
        ),
    )
    source_metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("first.tif", "second.tif")
        )
    )
    table = CellProfilerModule.build_measurement_table(
        name="AreaShape",
        rows=rows,
        object_name="Cells",
        source_image_name=None,
        source_metadata=source_metadata,
    )

    assert [row["slice_index"] for row in table.rows.row_mappings()] == [0, 1]
    assert table.source_image_provenance_planes.paths == (
        "first.tif",
        "second.tif",
    )


def test_measurement_table_projects_source_metadata_from_runtime_slice_rows() -> None:
    source_paths = ("/source/site1.tif", "/source/site2.tif")
    source_metadata = ({"well": "A01", "site": "1"}, {"well": "A01", "site": "2"})
    source_metadata_value = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=source_paths,
            component_metadata=source_metadata,
        ),
    )
    table = CellProfilerModule.build_measurement_table(
        name="AreaShape",
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"slice_index": 1, "object_label": 1, "area": 20.0},),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("area", float),
            ),
        ),
        object_name="Cells",
        source_image_name=None,
        source_metadata=source_metadata_value,
    )

    assert table.rows.row_mappings()[0]["slice_index"] == 1
    assert table.source_path == "/source/site2.tif"
    assert dict(table.source_component_metadata or {}) == {"well": "A01", "site": "2"}
    assert not table.source_image_provenance_planes.has_values


def test_measurement_table_preserves_declared_object_and_source_axes() -> None:
    source_paths = ("/source/site1.tif", "/source/site2.tif")
    cells = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(
            labels=np.zeros((2, 2, 2), dtype=np.uint16)
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=source_paths
        ),
    )
    columns = {
        "slice_index": (0, 1, 0, 1),
        "object_id": (1, 1, 1, 1),
        "value": (10.0, 20.0, 30.0, 40.0),
        MeasurementRowAxisField.OBJECT_NAME.value: (
            "Cells",
            "Cells",
            "Nuclei",
            "Nuclei",
        ),
        MeasurementRowAxisField.SOURCE_IMAGE_NAME.value: (
            "DNA",
            "DNA",
            "DNA",
            "DNA",
        ),
    }
    rows = _ColumnarMeasurementRows(
        columns,
        fields=(
            FieldSpec("slice_index", int),
            FieldSpec("object_id", int),
            FieldSpec("value", float),
            FieldSpec(MeasurementRowAxisField.OBJECT_NAME.value, str),
            FieldSpec(MeasurementRowAxisField.SOURCE_IMAGE_NAME.value, str),
        ),
        object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
    )

    table = CellProfilerModule.build_measurement_table(
        name="ObjectMeasurements",
        rows=rows,
        object_name=None,
        source_image_name="DNA",
        source_metadata=image_payload_metadata(cells),
    )

    assert table.subject == MeasurementSubject(
        MeasurementScope.IMAGE,
        "DNA",
    )
    projected_by_object: dict[str, list[int]] = {}
    for row in table.rows.iter_row_mappings():
        projected_by_object.setdefault(str(row["object_name"]), []).append(
            int(row["slice_index"])
        )
    assert projected_by_object == {
        "Cells": [0, 1],
        "Nuclei": [0, 1],
    }


def test_measure_object_intensity_zero_fills_missing_positive_extent() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[1, 0, 3]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(
            declared_object_count=5,
        ),
    )
    row_policy = MeasureObjectIntensityModule.runtime_object_measurement_row_policy()

    rows = complete_object_measurement_rows(
        MeasurementSparseColumnarRows.from_rows(
            ({"object_label": 1, "value": 7.0},),
            fields=(
                FieldSpec("object_label", int),
                FieldSpec("value", float),
            ),
            object_row_identity=row_policy.object_identity(),
        ),
        label_payload=payload,
        object_identity=row_policy.object_identity(),
        row_policy=row_policy,
    )
    by_label = {row["object_label"]: row for row in rows}

    assert by_label[2]["value"] == 0.0
    assert by_label[3]["value"] == 0.0
    assert by_label[4]["value"] == 0.0
    assert by_label[5]["value"] == 0.0


def test_measure_object_intensity_direct_slice_requires_projected_label_plane() -> None:
    image = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                [
                    [[1, 0], [0, 0]],
                    [[0, 0], [0, 2]],
                ],
                dtype=np.int32,
            )
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    with pytest.raises(ValueError, match="explicit object-ID domain"):
        measure_object_intensity(
            image,
            labels,
            slice_index=1,
        )


def test_measure_object_size_shape_consumes_runtime_projected_label_plane() -> None:
    image = np.zeros((4, 4), dtype=np.float32)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                [
                    [
                        [1, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 2, 2],
                        [0, 0, 2, 2],
                    ],
                ],
                dtype=np.int32,
            )
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    projected_labels = RuntimeSliceProjection.value_for_slice(
        labels,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
    )

    _image, rows = measure_object_size_shape(
        image,
        projected_labels,
        calculate_advanced=False,
        calculate_zernikes=False,
    )

    assert [row["object_label"] for row in rows] == [1, 2]
    assert rows[0]["Area"] == 4.0
    assert np.isnan(rows[1]["Area"])


def test_pure_2d_slice_execution_injects_slice_index_for_declared_callables() -> None:
    seen: list[int] = []

    @runtime_bound_parameters(SliceIndexRuntimeParameter)
    def records_slice_index(image, *, slice_index: int = 0):
        seen.append(slice_index)
        return image

    CellProfilerFunctionContractExecutor(
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=3,
        )
    ).execute_pure_2d_slice(
        CallableContract.from_callable(records_slice_index),
        records_slice_index,
        np.zeros((2, 2), dtype=np.float32),
        {},
        2,
        3,
    )

    assert seen == [2]


def test_complete_object_measurement_rows_uses_declared_slice_local_label_domain() -> (
    None
):
    labels = np.zeros((2, 3, 5), dtype=np.int32)
    labels[0, 0, 0] = 1
    labels[0, 0, 2] = 3
    labels[1, 0, 0] = 1
    labels[1, 0, 1] = 2
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1, 3), (1, 2)),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    row_policy = MeasureObjectIntensityModule.runtime_object_measurement_row_policy()

    rows = complete_object_measurement_rows(
        MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "object_label": 1, "value": 10.0},
                {"slice_index": 0, "object_label": 3, "value": 30.0},
                {"slice_index": 1, "object_label": 1, "value": 100.0},
                {"slice_index": 1, "object_label": 2, "value": 200.0},
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("value", float),
            ),
            object_row_identity=row_policy.object_identity(),
        ),
        label_payload=payload,
        object_identity=row_policy.object_identity(),
        row_policy=row_policy,
    )

    values_by_key = {
        (row["slice_index"], row["object_label"]): row["value"] for row in rows
    }
    assert set(values_by_key) == {
        (0, 1),
        (0, 3),
        (1, 1),
        (1, 2),
    }
    assert values_by_key == {
        (0, 1): 10.0,
        (0, 3): 30.0,
        (1, 1): 100.0,
        (1, 2): 200.0,
    }


def test_complete_object_measurement_rows_orders_sparse_label_domain() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[1, 0, 3]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(
            declared_object_count=5,
        ),
    )

    rows = complete_object_measurement_rows(
        MeasurementSparseColumnarRows.from_rows(
            (
                {"object_label": 3, "value": 30.0},
                {"object_label": 1, "value": 10.0},
            ),
            fields=(
                FieldSpec("object_label", int),
                FieldSpec("value", float),
            ),
            object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
        ),
        label_payload=payload,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[0]["value"] == 10.0
    assert np.isnan(rows[1]["value"])
    assert rows[2]["value"] == 30.0


def test_complete_object_measurement_rows_preserves_measurement_axes() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((4, 4), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_count=2,
        ),
    )

    rows = complete_object_measurement_rows(
        MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "object_label": 1,
                    "scale": 3,
                    "direction": 0,
                    "gray_levels": 256,
                    "angular_second_moment": 0.25,
                },
                {
                    "object_label": 1,
                    "scale": 3,
                    "direction": 1,
                    "gray_levels": 256,
                    "angular_second_moment": 0.5,
                },
            ),
            fields=(
                FieldSpec("object_label", int),
                FieldSpec("scale", int),
                FieldSpec("direction", int),
                FieldSpec("gray_levels", int),
                FieldSpec("angular_second_moment", float),
            ),
            object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
        ),
        label_payload=payload,
    )

    assert {
        (row["object_label"], row["scale"], row["direction"], row["gray_levels"])
        for row in rows
    } == {
        (1, 3, 0, 256),
        (1, 3, 1, 256),
        (2, 3, 0, 256),
        (2, 3, 1, 256),
    }
    missing_rows = [row for row in rows if row["object_label"] == 2]
    assert all(np.isnan(row["angular_second_moment"]) for row in missing_rows)


def test_complete_object_measurement_rows_supports_compact_row_identity() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((4, 4), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_ids=(10, 20, 30, 40, 50),
        ),
    )

    rows = complete_object_measurement_rows(
        MeasurementSparseColumnarRows.from_rows(
            tuple(
                {"object_label": ordinal, "Area": value}
                for ordinal, value in enumerate((10.0, 30.0, 50.0), start=1)
            ),
            fields=(
                FieldSpec("object_label", int),
                FieldSpec("Area", float),
            ),
            object_row_identity=MeasurementObjectRowIdentity.ROW_ORDINAL,
        ),
        label_payload=payload,
        object_identity=MeasurementObjectRowIdentity.ROW_ORDINAL,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["Area"] for row in rows[:3]] == [10.0, 30.0, 50.0]
    assert np.isnan(rows[3].get("Area", np.nan))
    assert np.isnan(rows[4].get("Area", np.nan))


def test_declared_domain_compact_rows_preserve_unmeasured_object_ordinals() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((4, 4), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_ids=(10, 20, 30),
        ),
    )

    row_policy = DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy()
    rows = complete_object_measurement_rows(
        MeasurementSparseColumnarRows.from_rows(
            (
                {"object_label": 10, "Area": 10.0},
                {"object_label": 30, "Area": 30.0},
            ),
            fields=(
                FieldSpec("object_label", int),
                FieldSpec("Area", float),
            ),
            object_row_identity=row_policy.object_identity(),
        ),
        label_payload=payload,
        row_policy=row_policy,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3]
    assert rows[0]["Area"] == 10.0
    assert np.isnan(rows[1].get("Area", np.nan))
    assert rows[2]["Area"] == 30.0


def test_measure_texture_compact_rows_preserve_declared_padding_domain() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((4, 4), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_count=5,
        ),
    )
    row_policy = MeasureTextureModule.runtime_object_measurement_row_policy()

    rows = row_policy.complete_rows(
        MeasurementSparseColumnarRows.from_rows(
            tuple(
                {
                    "object_label": object_label,
                    "slice_index": 0,
                    "scale": 3,
                    "direction": 0,
                    "gray_levels": 256,
                    "angular_second_moment": value,
                }
                for object_label, value in ((10, 0.1), (30, 0.3), (50, 0.5))
            ),
            fields=(
                FieldSpec("object_label", int),
                FieldSpec("slice_index", int),
                FieldSpec("scale", int),
                FieldSpec("direction", int),
                FieldSpec("gray_levels", int),
                FieldSpec("angular_second_moment", float),
            ),
            object_row_identity=row_policy.object_identity(),
        ),
        label_payload=payload,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["angular_second_moment"] for row in rows[:3]] == [0.1, 0.3, 0.5]
    assert all(np.isnan(row.get("angular_second_moment", np.nan)) for row in rows[3:])


def test_measure_texture_row_policy_completes_backend_present_object_domain() -> None:
    image = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape((8, 8))
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[5:7, 5:7] = 3
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_count=3),
    )

    _image, measurements = measure_texture_objects(
        image,
        payload,
        dtype_config=DtypeConfig(),
    )

    assert {
        measurement["object_label"] for measurement in measurements.row_mappings()
    } == {1, 3}

    rows = MeasureTextureModule.runtime_object_measurement_row_policy().complete_rows(
        measurements,
        label_payload=payload,
    )
    direction_zero_rows = [
        row for row in rows if row["direction"] == 0 and row["scale"] == 3
    ]
    assert [row["object_label"] for row in direction_zero_rows] == [1, 2, 3]
    assert np.isfinite(direction_zero_rows[0]["contrast"])
    assert np.isfinite(direction_zero_rows[1]["contrast"])
    assert np.isnan(direction_zero_rows[2]["contrast"])


def test_measure_texture_uses_exact_projected_plane_domain_for_missing_rows() -> None:
    labels = np.zeros((2, 4, 4), dtype=np.int32)
    labels[0, 0, 0] = 5
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_names=("BF_image", "MorphBf"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2, 3, 4, 5), (1,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    projected_payload = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=0, axis_size=2
        ),
    )
    row_policy = MeasureTextureModule.runtime_object_measurement_row_policy()

    rows = row_policy.complete_rows(
        MeasurementSparseColumnarRows.from_rows(
            tuple(
                {
                    "object_label": object_label,
                    "slice_index": 0,
                    "scale": 3,
                    "direction": 0,
                    "gray_levels": 256,
                    "angular_second_moment": value,
                }
                for object_label, value in ((1, 0.1), (2, 0.2), (3, np.nan))
            ),
            fields=(
                FieldSpec("object_label", int),
                FieldSpec("slice_index", int),
                FieldSpec("scale", int),
                FieldSpec("direction", int),
                FieldSpec("gray_levels", int),
                FieldSpec("angular_second_moment", float),
            ),
            object_row_identity=row_policy.object_identity(),
        ),
        label_payload=projected_payload,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["angular_second_moment"] for row in rows[:2]] == [0.1, 0.2]
    assert all(np.isnan(row.get("angular_second_moment", np.nan)) for row in rows[2:])


def test_measure_object_intensity_missing_rows_use_declared_label_extent() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                (
                    [[1, 2, 3]],
                    [[1, 2, 3]],
                ),
                dtype=np.int32,
            )
        ),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1, 2, 3), (1, 2, 3)),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    row_policy = MeasureObjectIntensityModule.runtime_object_measurement_row_policy()
    schema = ObjectMeasurementRowCompletionSchema.for_completion_fields(
        object_id_field=MeasurementRowAxisField.OBJECT_LABEL.value,
        axis_fields=(MeasurementRowAxisField.SLICE_INDEX.value,),
        field_names=(
            MeasurementRowAxisField.SLICE_INDEX.value,
            MeasurementRowAxisField.OBJECT_LABEL.value,
            "integrated_intensity",
        ),
    )

    rows = schema.missing_rows(
        missing_row_keys=((2, (0,)), (3, (1,))),
        label_payload=payload,
        row_policy=row_policy,
    )

    by_axis_object = {(row["slice_index"], row["object_label"]): row for row in rows}
    assert by_axis_object[(0, 2)]["integrated_intensity"] == 0.0
    assert by_axis_object[(1, 3)]["integrated_intensity"] == 0.0


def test_measure_object_intensity_columnar_rows_use_declared_axis_domain() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[1, 2]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
    )
    row_policy = MeasureObjectIntensityModule.runtime_object_measurement_row_policy()

    rows = row_policy.complete_rows(
        _ColumnarMeasurementRows(
            {
                "slice_index": (0, 0, 0),
                "object_label": (1, 2, 3),
                "integrated_intensity": (10.0, 20.0, 0.0),
            },
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("integrated_intensity", float),
            ),
            object_row_identity=row_policy.object_identity(),
        ),
        label_payload=payload,
    )

    assert [row["object_label"] for row in rows] == [1, 2]
    assert [row["integrated_intensity"] for row in rows] == [10.0, 20.0]


def test_measure_object_size_shape_rows_project_measured_sequence_to_cp_ordinals() -> (
    None
):
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((4, 4), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_ids=(10, 20, 30, 40, 50),
        ),
    )
    row_policy = MeasureObjectSizeShapeModule.runtime_object_measurement_row_policy()

    rows = complete_object_measurement_rows(
        MeasurementSparseColumnarRows.from_rows(
            (
                {"object_label": 10, "Area": 10.0, "Center_X": 10.0},
                {"object_label": 30, "Area": 30.0, "Center_X": 30.0},
                {"object_label": 50, "Area": 50.0, "Center_X": 50.0},
            ),
            fields=(
                FieldSpec("object_label", int),
                FieldSpec("Area", float),
                FieldSpec("Center_X", float),
            ),
            object_row_identity=row_policy.object_identity(),
        ),
        label_payload=payload,
        object_identity=row_policy.object_identity(),
        row_policy=row_policy,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[0]["Area"] == 10.0
    assert rows[1]["Area"] == 30.0
    assert rows[2]["Area"] == 50.0
    assert np.isnan(rows[3].get("Area", np.nan))
    assert np.isnan(rows[4].get("Area", np.nan))


def test_measure_object_size_shape_preserves_zero_valued_label_rows() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((4, 4), dtype=np.int32)),
        domain=ObjectLabelDomain(declared_object_count=5),
    )
    row_policy = MeasureObjectSizeShapeModule.runtime_object_measurement_row_policy()

    source_rows = MeasurementSparseColumnarRows.from_rows(
        (
            {
                "object_label": 1,
                "Area": 10.0,
                "Center_X": 10.0,
                "MaxFeretDiameter": 11.0,
            },
            {
                "object_label": 2,
                "Area": 0.0,
                "EulerNumber": 0.0,
                "Center_X": np.nan,
                "Center_Y": np.nan,
                "Center_Z": 0.0,
                "MaxFeretDiameter": np.nan,
            },
            {
                "object_label": 3,
                "Area": 30.0,
                "Center_X": 30.0,
                "MaxFeretDiameter": 33.0,
            },
            {
                "object_label": 4,
                "Area": np.nan,
                "Center_X": np.nan,
                "Center_Y": np.nan,
                "MaxFeretDiameter": 0.0,
            },
        ),
        fields=(
            FieldSpec("object_label", int),
            FieldSpec("Area", float),
            FieldSpec("Center_X", float),
            FieldSpec("MaxFeretDiameter", float),
            FieldSpec("EulerNumber", float),
            FieldSpec("Center_Y", float),
            FieldSpec("Center_Z", float),
            FieldSpec("value", float),
        ),
        object_row_identity=row_policy.object_identity(),
    )
    rows = row_policy.complete_rows(
        source_rows,
        label_payload=payload,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[0]["Center_X"] == 10.0
    assert np.isnan(rows[1]["Center_X"])
    assert rows[2]["Center_X"] == 30.0
    assert np.isnan(rows[3]["Center_X"])
    assert np.isnan(rows[4].get("Center_X", np.nan))
    assert rows[0]["MaxFeretDiameter"] == 11.0
    assert np.isnan(rows[1]["MaxFeretDiameter"])
    assert rows[2]["MaxFeretDiameter"] == 33.0
    assert rows[3]["MaxFeretDiameter"] == 0.0
    assert np.isnan(rows[4].get("MaxFeretDiameter", np.nan))


def test_measure_object_size_shape_preserves_complete_dense_label_domain_rows() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((4, 4), dtype=np.int32)),
        domain=ObjectLabelDomain(declared_object_count=3),
    )
    row_policy = MeasureObjectSizeShapeModule.runtime_object_measurement_row_policy()

    rows = row_policy.complete_rows(
        MeasurementSparseColumnarRows.from_rows(
            (
                {"object_label": 1, "Center_X": 10.0},
                {"object_label": 2, "Center_X": np.nan},
                {"object_label": 3, "Center_X": 30.0},
            ),
            fields=(
                FieldSpec("object_label", int),
                FieldSpec("Center_X", float),
            ),
            object_row_identity=row_policy.object_identity(),
        ),
        label_payload=payload,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3]
    assert rows[0]["Center_X"] == 10.0
    assert np.isnan(rows[1]["Center_X"])
    assert rows[2]["Center_X"] == 30.0


def test_object_measurement_modules_declare_their_cp_row_identity_policies() -> None:
    assert (
        MeasureObjectSizeShapeModule.runtime_object_measurement_row_policy().object_identity()
        is MeasurementObjectRowIdentity.ROW_SEQUENCE
    )
    assert (
        MeasureObjectIntensityDistributionModule.runtime_object_measurement_row_policy().object_identity()
        is MeasurementObjectRowIdentity.ROW_ORDINAL
    )
    assert (
        MeasureObjectIntensityModule.runtime_object_measurement_row_policy().object_identity()
        is MeasurementObjectRowIdentity.LABEL_ID
    )


def test_object_only_measurement_preserves_runtime_slice_carrier_stack() -> None:
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 0:2, 0:2] = 1
    labels[1, 0:3, 0:3] = 1
    object_spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    measurement_spec = ArtifactSpec.output(
        "MeasureObjectSizeShape_1_measurements",
        MeasurementsArtifactType,
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        {
            "Nuclei": ObjectLabelSet(
                name="Nuclei",
                variant_data=ObjectLabelVariantData(labels=labels),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                domain=ObjectLabelDomain(
                    declared_object_id_domains=((1,), (1,)),
                    scope=ObjectLabelDomainScope.PLANE,
                ),
            )
        },
        artifact_input_edges=(_artifact_input_edge_for_test(object_spec),),
        artifact_output_bindings=(
            (measurement_spec, _artifact_output_plan(measurement_spec)),
        ),
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module(
                "MeasureObjectSizeShape"
            ).require_callable(),
            artifact_inputs=(object_spec,),
            artifact_outputs=(measurement_spec,),
        ),
    )

    current_image = np.zeros((2, 5, 5), dtype=np.float32)

    result = _run_module(
        executor,
        current_image,
        cellprofiler_runtime=runtime,
        calculate_advanced=False,
        calculate_zernikes=False,
    )

    assert result is current_image
    assert len(runtime.measurements) == 1
    rows = runtime.measurements[0].rows.row_mappings()
    assert [
        (row["slice_index"], row["object_label"], row["AreaShape_Area"]) for row in rows
    ] == [
        (0, 1, 4.0),
        (1, 1, 9.0),
    ]


def test_resolved_measurement_image_preserves_runtime_slice_projection() -> None:
    image_spec = ArtifactSpec.input("DNA", ImageArtifactType)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("MeasureObjectIntensity").require_callable(),
        artifact_inputs=(image_spec,),
    )
    executor = _module_executor(contract)
    image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE
    ).payload_with(np.zeros((21, 5, 5), dtype=np.float32), None)
    runtime = _FakeCellProfilerRuntime(
        {"DNA": image},
        artifact_input_edges=(_artifact_input_edge_for_test(image_spec),),
        plane_projection=RuntimePlaneProjection.stack(21),
    )
    _activate_runtime_contract(executor.callable_contract, runtime)

    measurement_image = executor._resolved_measurement_image(
        image_spec,
        runtime,
        image,
        ("DNA",),
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    assert measurement_image.plane_projection == (
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=21,
            source_aliases=("DNA",),
        )
    )


def test_object_intensity_measurement_image_batch_preserves_request_labels() -> None:
    from openhcs.core.runtime_batch_contracts import RuntimeBatchInvocationRequest
    from openhcs.processing.backends.cellprofiler._backend import (
        DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    )
    from openhcs.processing.backends.cellprofiler.intensity import (
        measure_object_intensity,
        measure_object_intensity_measurement_image_batch,
    )

    labels_a = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[[1]], [[2]]], dtype=np.int32)
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("ImageA", "ImageB"),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (2,)),
        ),
    )
    labels_b = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[[10]], [[20]]], dtype=np.int32)
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("ImageA", "ImageB"),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((10,), (20,)),
        ),
    )
    requests = (
        RuntimeBatchInvocationRequest(
            source_image_name="ImageA",
            func=measure_object_intensity,
            image=np.asarray([[1.0]], dtype=np.float32),
            kwargs={"labels": labels_a},
            batch_index=0,
            batch_count=2,
            semantic_group_key=(("object_artifact", "Cells"),),
        ),
        RuntimeBatchInvocationRequest(
            source_image_name="ImageB",
            func=measure_object_intensity,
            image=np.asarray([[2.0]], dtype=np.float32),
            kwargs={"labels": labels_b},
            batch_index=1,
            batch_count=2,
            semantic_group_key=(("object_artifact", "Cells"),),
        ),
    )

    def execute_request(_func, request):
        return (
            request.source_image_name,
            int(np.max(object_label_dense_array(request.kwargs["labels"]))),
        )

    outputs = measure_object_intensity_measurement_image_batch(
        measure_object_intensity,
        requests,
        execute_request,
    )

    assert (
        requests[0].kwargs["object_intensity_backend_provider"]
        is DEFAULT_CELLPROFILER_BACKEND_SELECTION
    )
    assert outputs == [("ImageA", 2), ("ImageB", 20)]

    runtime_slice_labels = ObjectLabelSet(
        name="Cytoplasm",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[[1]], [[2]]], dtype=np.int32)
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (2,)),
        ),
    )
    runtime_slice_requests = (
        RuntimeBatchInvocationRequest(
            source_image_name="ImageA",
            func=measure_object_intensity,
            image=np.asarray([[1.0]], dtype=np.float32),
            kwargs={"labels": runtime_slice_labels},
            batch_index=0,
            batch_count=2,
            semantic_group_key=(("object_artifact", "Cytoplasm"),),
        ),
        RuntimeBatchInvocationRequest(
            source_image_name="ImageB",
            func=measure_object_intensity,
            image=np.asarray([[2.0]], dtype=np.float32),
            kwargs={"labels": runtime_slice_labels},
            batch_index=1,
            batch_count=2,
            semantic_group_key=(("object_artifact", "Cytoplasm"),),
        ),
    )

    runtime_slice_outputs = measure_object_intensity_measurement_image_batch(
        measure_object_intensity,
        runtime_slice_requests,
        execute_request,
    )

    assert runtime_slice_outputs == [("ImageA", 2), ("ImageB", 2)]


def test_object_intensity_measurement_image_batch_delegates_natural_prepared_requests() -> (
    None
):
    from openhcs.interop.cellprofiler.runtime.measurement_execution_support import (
        PreparedObjectMeasurementInvocation,
        object_measurement_batch_group_key,
    )
    from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
        ObjectMeasurementInvocation,
    )
    from openhcs.processing.backends.cellprofiler.intensity import (
        measure_object_intensity,
        measure_object_intensity_measurement_image_batch,
    )

    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[[1, 0]], [[2, 3]]], dtype=np.int32)
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2, 3)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    object_spec = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)
    invocation = ObjectMeasurementInvocation(kwargs={})
    semantic_group_key = object_measurement_batch_group_key(
        object_spec=object_spec,
        labels=labels,
    )
    requests = (
        PreparedObjectMeasurementInvocation(
            source_image_name="OrigDNA",
            execution_mode=ImagePayloadExecutionMode.NATURAL,
            func=measure_object_intensity,
            image=np.asarray([[1.0, 2.0]], dtype=np.float32),
            kwargs={"labels": labels},
            batch_index=0,
            batch_count=2,
            semantic_group_key=semantic_group_key,
            measurement_image=CellProfilerMeasurementImage(
                source_image_name="OrigDNA",
                payload=np.asarray([[1.0, 2.0]], dtype=np.float32),
            ),
            object_spec=object_spec,
            invocation=invocation,
            completion_label_payload=labels,
        ),
        PreparedObjectMeasurementInvocation(
            source_image_name="OrigRNA",
            execution_mode=ImagePayloadExecutionMode.NATURAL,
            func=measure_object_intensity,
            image=np.asarray([[3.0, 4.0]], dtype=np.float32),
            kwargs={"labels": labels},
            batch_index=1,
            batch_count=2,
            semantic_group_key=semantic_group_key,
            measurement_image=CellProfilerMeasurementImage(
                source_image_name="OrigRNA",
                payload=np.asarray([[3.0, 4.0]], dtype=np.float32),
            ),
            object_spec=object_spec,
            invocation=invocation,
            completion_label_payload=labels,
        ),
    )
    delegated: list[str | None] = []

    def execute_request(_func, request):
        delegated.append(request.source_image_name)
        return request.source_image_name

    outputs = measure_object_intensity_measurement_image_batch(
        measure_object_intensity,
        requests,
        execute_request,
    )

    assert outputs == ["OrigDNA", "OrigRNA"]
    assert delegated == ["OrigDNA", "OrigRNA"]


def test_concatenated_columnar_rows_preserve_exact_schema() -> None:
    rows = ConcatenatedColumnarRows(
        (
            _ColumnarMeasurementRows(
                {
                    "object_name": ("Cells",),
                    "object_label": (1,),
                    "mean_intensity": (0.5,),
                },
                fields=(
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("mean_intensity", float),
                ),
            ),
            _ColumnarMeasurementRows(
                {
                    "object_name": ("Cells",),
                    "object_label": (2,),
                    "mean_intensity": (0.8,),
                },
                fields=(
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("mean_intensity", float),
                ),
            ),
        )
    )

    assert tuple(field.name for field in rows.fields) == (
        "object_name",
        "object_label",
        "mean_intensity",
    )


def test_measure_object_size_shape_outputs_basic_measurement_rows() -> None:
    image = np.ones((7, 7), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:4, 1:4] = 1

    _image, rows = measure_object_size_shape(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert len(rows) == 1
    assert rows[0]["object_label"] == 1
    assert rows[0]["Area"] == 9.0
    assert rows[0]["Center_X"] == 2.0
    assert rows[0]["Center_Y"] == 2.0


def test_measure_object_size_shape_coordinates_remain_in_local_label_plane() -> None:
    image = np.ones((7, 7), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:4, 1:4] = 1

    _image, rows = measure_object_size_shape(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(251, 501),
                source_shape_yx=(1006, 1000),
            ),
        ),
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert rows[0]["Center_X"] == 2.0
    assert rows[0]["Center_Y"] == 2.0
    assert rows[0]["BoundingBoxMinimum_X"] == 1.0
    assert rows[0]["BoundingBoxMaximum_X"] == 4.0
    assert rows[0]["BoundingBoxMinimum_Y"] == 1.0
    assert rows[0]["BoundingBoxMaximum_Y"] == 4.0


def test_measure_object_size_shape_exports_skimage_perimeter() -> None:
    image = np.ones((9, 9), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    y, x = np.ogrid[-2:3, -2:3]
    labels[2:7, 2:7][x * x + y * y <= 4] = 1

    _image, rows = measure_object_size_shape(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    expected_perimeter = skimage.measure.perimeter(labels == 1, neighborhood=4)
    assert abs(rows[0]["Perimeter"] - expected_perimeter) < 1e-12


def test_measure_object_size_shape_form_factor_uses_exported_perimeter() -> None:
    image = np.ones((9, 9), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    y, x = np.ogrid[-2:3, -2:3]
    labels[2:7, 2:7][x * x + y * y <= 4] = 1

    _image, rows = measure_object_size_shape(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    expected_form_factor = (
        4.0 * np.pi * float(rows[0]["Area"]) / float(rows[0]["Perimeter"]) ** 2
    )
    assert abs(rows[0]["FormFactor"] - expected_form_factor) < 1e-12
    assert abs(rows[0]["Compactness"] - (1.0 / expected_form_factor)) < 1e-12


def test_measure_object_size_shape_orientation_uses_cellprofiler_inertia_tie() -> None:
    image = np.ones((26, 26), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    mask = np.array(
        [
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=bool,
    )
    labels[10:15, 10:15][mask] = 1

    _image, rows = measure_object_size_shape(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        calculate_advanced=True,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert rows[0]["Orientation"] == 45.0


def test_measure_object_size_shape_orientation_is_cpu_dispatch_independent() -> None:
    image = np.ones((9, 9), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[2:7, 2:7] = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ],
        dtype=np.int32,
    )

    _image, rows = measure_object_size_shape(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert rows[0]["Orientation"] == -45.0


def test_measure_object_size_shape_orientation_uses_topmost_point_tie() -> None:
    image = np.ones((20, 20), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    mask = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=bool,
    )
    labels[4:16, 4:16][mask] = 1

    _image, rows = measure_object_size_shape(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        calculate_advanced=True,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert rows[0]["Orientation"] == -45.0


def test_measure_object_size_shape_orientation_uses_positive_inertia_tie() -> None:
    image = np.ones((12, 12), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    mask = np.array(
        [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    labels[1:11, 1:11][mask] = 1

    _image, rows = measure_object_size_shape(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        calculate_advanced=True,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert rows[0]["Orientation"] == 45.0


def test_measure_object_size_shape_zernikes_use_declared_row_ordinal_domain() -> None:
    image = np.ones((12, 12), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[6:10, 6:10] = 3

    _image, rows = measure_object_size_shape(
        image,
        ObjectLabelSet(
            name="Nuclei",
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_count=3),
        ),
        calculate_advanced=False,
        calculate_zernikes=True,
        dtype_config=DtypeConfig(),
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3]
    assert np.isfinite(rows[0]["Zernike_0_0"])
    assert np.isnan(rows[1]["Zernike_0_0"])
    assert np.isfinite(rows[2]["Zernike_0_0"])
    assert rows[1]["Area"] == 16.0
    assert np.isnan(rows[2]["Area"])
    assert rows[1]["MaximumRadius"] > 0.0
    assert rows[2]["MaximumRadius"] == 0.0
    assert rows[1]["MinFeretDiameter"] == 0.0
    assert rows[2]["MinFeretDiameter"] > 0.0


def test_measure_object_size_shape_backend_emits_concrete_cp_index_domain() -> None:
    image = np.ones((12, 12), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[6:10, 6:10] = 3

    _image, rows = measure_object_size_shape(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_count=5),
        ),
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3]
    assert rows[0]["Area"] == 9.0
    assert rows[1]["Area"] == 16.0
    assert np.isnan(rows[2]["Area"])


def test_measure_object_size_shape_uses_explicit_sparse_object_domain() -> None:
    image = np.ones((12, 12), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[6:10, 6:10] = 1000

    _image, rows = measure_object_size_shape(
        image,
        ObjectLabelSet(
            name="Nuclei",
            variant_data=ObjectLabelVariantData(
                labels=SparseIJVLabelRows.from_dense_labels(labels)
            ),
            representation=ObjectLabelRepresentation.SPARSE_IJV,
            domain=ObjectLabelDomain(declared_object_ids=(1, 1000)),
        ),
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert [row["object_label"] for row in rows] == [1, 1000]
    assert rows[1]["Center_X"] == 7.5
    assert rows[1]["Center_Y"] == 7.5


def test_measure_object_size_shape_sparse_coordinates_restore_only_local_patch_offset() -> (
    None
):
    image = np.ones((12, 14), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[4:7, 5:9] = 37

    _image, rows = measure_object_size_shape(
        image,
        ObjectLabelSet(
            name="Nuclei",
            variant_data=ObjectLabelVariantData(
                labels=SparseIJVLabelRows.from_dense_labels(labels)
            ),
            representation=ObjectLabelRepresentation.SPARSE_IJV,
            domain=ObjectLabelDomain(declared_object_ids=(37,)),
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(101, 203),
                source_shape_yx=(200, 300),
            ),
        ),
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert rows[0]["Center_X"] == 6.5
    assert rows[0]["Center_Y"] == 5.0
    assert rows[0]["BoundingBoxMinimum_X"] == 5.0
    assert rows[0]["BoundingBoxMaximum_X"] == 9.0
    assert rows[0]["BoundingBoxMinimum_Y"] == 4.0
    assert rows[0]["BoundingBoxMaximum_Y"] == 7.0


def test_measure_object_size_shape_preserves_sparse_plane_storage() -> None:
    first = np.zeros((8, 8), dtype=np.int32)
    first[1:3, 1:3] = 1
    second = np.zeros((8, 8), dtype=np.int32)
    second[4:7, 4:7] = 2
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=SparseIJVLabelRows.from_slices(
                (
                    SparseIJVLabelRows.from_dense_labels(first),
                    SparseIJVLabelRows.from_dense_labels(second),
                )
            )
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    _image, rows = measure_object_size_shape(
        np.ones((2, 8, 8), dtype=np.float32),
        labels,
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert [(row["slice_index"], row["object_label"], row["Area"]) for row in rows] == [
        (0, 1, 4.0),
        (1, 2, 9.0),
    ]


def test_filterobjects_uses_upstream_form_factor_table_when_available() -> None:
    image = np.ones((9, 9), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    y, x = np.ogrid[-2:3, -2:3]
    labels[2:7, 2:7][x * x + y * y <= 4] = 1
    exported_perimeter = skimage.measure.perimeter(labels == 1, neighborhood=4)
    exported_form_factor = (
        4.0 * np.pi * float(np.count_nonzero(labels)) / exported_perimeter**2
    )

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.LIMITS,
        object_labels=(labels,),
        measurement_features=("AreaShape_FormFactor",),
        measurement_min_values=(0.2,),
        measurement_max_values=(1.0,),
        measurement_use_minimum=(True,),
        measurement_use_maximum=(True,),
        measurement_tables=(
            MeasurementTable(
                name="Shape",
                rows=MeasurementSparseColumnarRows.from_rows(
                    (
                        {
                            "object_label": 1,
                            "FormFactor": exported_form_factor,
                        },
                    ),
                    fields=(
                        FieldSpec("object_label", int),
                        FieldSpec("FormFactor", float),
                    ),
                ),
                subject=MeasurementSubject(MeasurementScope.OBJECT, "Objects"),
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    assert exported_form_factor > 1.0
    _output_image, stats, filtered_labels = result[:3]
    stats = stats.rows[0]
    assert stats.objects_post_filter == 0
    assert object_label_dense_array(filtered_labels).max() == 0


def test_filterobjects_rejects_form_factor_without_measurement_input() -> None:
    image = np.ones((9, 9), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    y, x = np.ogrid[-2:3, -2:3]
    labels[2:7, 2:7][x * x + y * y <= 4] = 1

    with pytest.raises(ValueError, match="no declared measurement-table input"):
        filter_objects(
            image,
            mode=FilterMode.MEASUREMENTS,
            filter_method=FilterMethod.LIMITS,
            object_labels=(labels,),
            measurement_features=("AreaShape_FormFactor",),
            measurement_min_values=(0.2,),
            measurement_max_values=(1.0,),
            measurement_use_minimum=(True,),
            measurement_use_maximum=(True,),
            dtype_config=DtypeConfig(),
        )


def test_cellprofiler_contract_executor_stacks_color_slice_outputs():
    calls = []

    def colorize(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return np.stack((image, image, image), axis=-1)

    colorize.__processing_contract__ = ProcessingContract.PURE_2D
    raw_contract = CallableContract.from_callable(colorize)
    callable_contract = replace(
        raw_contract,
        metadata=replace(
            raw_contract.metadata,
            artifact_outputs=(ArtifactSpec.output("Color", ImageArtifactType),),
        ),
    )
    stack = ImageMetadataPayload(
        np.zeros((2, 4, 5), dtype=np.float32),
        ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        colorize,
        stack,
        {},
        execution_mode=ImagePayloadExecutionMode.NATURAL,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert calls == [(4, 5), (4, 5)]
    assert image_payload_data(result).shape == (2, 4, 5, 3)


def test_gray_to_color_consumes_channel_stack_as_single_rgb_image() -> None:
    image = np.zeros((2, 4, 5), dtype=np.float32)
    image[0] = 0.25
    image[1] = 0.75

    result = gray_to_color(
        image,
        color_scheme="RGB",
        red_channel=-1,
        green_channel=0,
        blue_channel=1,
        rescale_intensity=False,
        dtype_config=DtypeConfig(),
    )

    assert result.shape == (4, 5, 3)
    np.testing.assert_array_equal(result[..., 0], np.zeros((4, 5), dtype=np.float32))
    np.testing.assert_array_equal(
        result[..., 1], np.full((4, 5), 0.25, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        result[..., 2], np.full((4, 5), 0.75, dtype=np.float32)
    )


def test_gray_to_color_inherits_first_declared_input_mask() -> None:
    first_mask = np.ones((4, 5), dtype=bool)
    first_mask[1, 2] = False
    second_mask = np.ones((4, 5), dtype=bool)
    image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/red.tif", "/input/green.tif"),
            component_metadata=(
                {"channel": "1", "source_alias": "Red"},
                {"channel": "2", "source_alias": "Green"},
            ),
        ),
        source_image_names=("Red", "Green"),
    ).payload_with(
        np.zeros((2, 4, 5), dtype=np.float32), np.stack((first_mask, second_mask))
    )

    raw_gray_to_color = CallableContract.from_callable(
        gray_to_color
    ).resolve_canonical_raw_callable()
    result = raw_gray_to_color(
        image,
        color_scheme="RGB",
        red_channel=0,
        green_channel=1,
        blue_channel=-1,
        rescale_intensity=False,
    )

    assert image_payload_data(result).shape == (4, 5, 3)
    np.testing.assert_array_equal(image_payload_mask(result), first_mask)
    metadata = image_payload_metadata(result)
    assert metadata.plane_axis is None
    assert metadata.source_channel_axis == -1
    assert metadata.source_image_provenance_planes.paths == (
        "/input/red.tif",
        "/input/green.tif",
    )


def test_color_to_gray_combines_openhcs_color_stack() -> None:
    image = ImageMetadataPayload(
        np.zeros((2, 4, 5, 3), dtype=np.float32),
        ImagePayloadMetadata(
            source_channel_axis=-1,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    )
    image.data[..., 0] = 2.0
    image.data[..., 1] = 4.0
    image.data[..., 2] = 6.0

    result = color_to_gray(
        image,
        mode="combine",
        image_type="rgb",
        channel_indices=(0, 1, 2),
        contributions=(1.0, 1.0, 2.0),
        dtype_config=DtypeConfig(),
    )

    assert result.shape == (2, 4, 5)
    np.testing.assert_array_equal(result, np.full((2, 4, 5), 4.5, dtype=np.float32))


def test_color_to_gray_rejects_shape_only_color_semantics() -> None:
    image = np.zeros((2, 4, 5, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="source_channel_axis"):
        color_to_gray(
            image,
            mode="combine",
            image_type="rgb",
            channel_indices=(0, 1, 2),
            contributions=(1.0, 1.0, 1.0),
            dtype_config=DtypeConfig(),
        )


def test_color_to_gray_combine_removes_declared_source_channel_axis() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((2, 4, 5, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001.tif",
            source_component_metadata={"site": "1"},
            source_channel_axis=-1,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    )
    image.data[..., 0] = 2.0
    image.data[..., 1] = 4.0
    image.data[..., 2] = 6.0

    result = openhcs_color_to_gray(
        image,
        mode="combine",
        image_type="rgb",
        channel_indices=(0, 1, 2),
        contributions=(1.0, 1.0, 2.0),
        dtype_config=DtypeConfig(),
    )

    assert isinstance(result, ImageMetadataPayload)
    assert result.data.shape == (2, 4, 5)
    metadata = image_payload_metadata(result)
    assert dict(metadata.source_component_metadata) == {"site": "1"}
    assert metadata.source_channel_axis is None
    assert metadata.source_image_provenance_planes.count == 0


def test_color_to_gray_splits_openhcs_color_slice_by_selected_channels() -> None:
    image = ImageMetadataPayload(
        np.zeros((4, 5, 3), dtype=np.float32),
        ImagePayloadMetadata(source_channel_axis=-1),
    )
    image.data[..., 0] = 1.0
    image.data[..., 1] = 2.0
    image.data[..., 2] = 3.0

    outputs = color_to_gray(
        image,
        mode="split",
        image_type="rgb",
        channel_indices=(0, 2),
        dtype_config=DtypeConfig(),
    )
    assert isinstance(outputs, AlignedImageStack)
    red, blue = outputs.slices

    assert red.shape == (4, 5)
    assert blue.shape == (4, 5)
    np.testing.assert_array_equal(red, np.ones((4, 5), dtype=np.float32))
    np.testing.assert_array_equal(blue, np.full((4, 5), 3.0, dtype=np.float32))


def test_color_to_gray_rgb_split_preserves_color_source_metadata() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((4, 5, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_color.tif",
            source_component_metadata={"well": "A01", "site": "1"},
            source_image_names=("OrigColor",),
            source_channel_axis=-1,
        ),
    )
    image.data[..., 1] = 2.0

    green = color_to_gray(
        image,
        mode="split",
        image_type="rgb",
        channel_indices=(1,),
        dtype_config=DtypeConfig(),
    )

    metadata = image_payload_metadata(green)
    assert metadata.source_path == "/input/A01_s001_color.tif"
    assert metadata.source_component_metadata == {
        "well": "A01",
        "site": "1",
    }
    assert metadata.source_image_names == ("OrigColor",)
    assert metadata.source_provenance.source_plane_count == 0
    np.testing.assert_array_equal(green.data, np.full((4, 5), 2.0, dtype=np.float32))


def test_color_to_gray_splits_channel_last_non_rgb_slice() -> None:
    image = ImageMetadataPayload(
        np.zeros((4, 5, 2), dtype=np.float32),
        ImagePayloadMetadata(source_channel_axis=-1),
    )
    image.data[..., 0] = 7.0
    image.data[..., 1] = 11.0

    first_channel = color_to_gray(
        image,
        mode="split",
        image_type="rgb",
        channel_indices=(0,),
        dtype_config=DtypeConfig(),
    )

    assert first_channel.shape == (4, 5)
    np.testing.assert_array_equal(first_channel, np.full((4, 5), 7.0, dtype=np.float32))


def test_color_to_gray_preserves_masked_image_payload() -> None:
    image = np.zeros((3, 4, 3), dtype=np.float32)
    image[..., 0] = 0.75
    mask = np.array(
        (
            (True, False, True, True),
            (True, True, False, True),
            (False, True, True, True),
        )
    )

    red = color_to_gray(
        MaskedImagePayload(
            data=image,
            mask=mask,
            metadata=ImagePayloadMetadata(source_channel_axis=-1),
        ),
        mode="split",
        image_type="rgb",
        channel_indices=(0,),
        dtype_config=DtypeConfig(),
    )

    assert isinstance(red, MaskedImagePayload)
    np.testing.assert_array_equal(red.data, image[..., 0])
    np.testing.assert_array_equal(red.mask, mask)


def test_aligned_payload_treats_hwc_color_as_one_slice() -> None:
    color_slice = np.zeros((4, 5, 3), dtype=np.float32)

    slices = payload_slices_for_alignment(color_slice)

    assert len(slices) == 1
    assert slices[0] is color_slice
    assert payload_slice_count(color_slice) == 1


def test_aligned_payload_slices_runtime_slice_aligned_values() -> None:
    first = np.zeros((4, 5), dtype=np.int32)
    second = np.ones((4, 5), dtype=np.int32)

    slices = payload_slices_for_alignment(RuntimeSliceAlignedValues((first, second)))

    assert slices[0] is first
    assert slices[1] is second


def test_aligned_payload_slices_explicit_masked_image_owner() -> None:
    stack = np.zeros((2, 4, 5), dtype=np.float32)
    mask = np.array(
        (
            np.ones((4, 5), dtype=bool),
            np.zeros((4, 5), dtype=bool),
        )
    )

    aligned = AlignedImageStack(
        tuple(
            MaskedImagePayload(data=stack[index], mask=mask[index])
            for index in range(2)
        )
    )
    slices = payload_slices_for_alignment(aligned)

    assert len(slices) == 2
    assert all(
        isinstance(slice_payload, MaskedImagePayload) for slice_payload in slices
    )
    np.testing.assert_array_equal(slices[0].mask, mask[0])
    np.testing.assert_array_equal(slices[1].mask, mask[1])


def test_aligned_payload_keeps_unowned_source_tagged_grayscale_stack_whole() -> None:
    stack = np.zeros((2, 4, 5), dtype=np.float32)
    payload = ImageMetadataPayload(
        data=stack,
        metadata=ImagePayloadMetadata(
            source_path="/tmp/source.tif",
        ),
    )

    slices = payload_slices_for_alignment(payload)

    assert len(slices) == 1
    assert slices[0] is payload


def test_aligned_payload_keeps_unowned_source_tagged_color_stack_whole() -> None:
    stack = np.zeros((2, 4, 5, 2), dtype=np.float32)
    payload = ImageMetadataPayload(
        data=stack,
        metadata=ImagePayloadMetadata(
            source_path="/tmp/source.tif",
            source_channel_axis=-1,
        ),
    )

    slices = payload_slices_for_alignment(payload)

    assert len(slices) == 1
    assert slices[0] is payload


def test_aligned_payload_keeps_unowned_masked_volume_stack_whole() -> None:
    stack = np.zeros((2, 3, 4, 5), dtype=np.float32)
    mask = np.zeros_like(stack, dtype=bool)
    mask[0] = True

    payload = MaskedImagePayload(data=stack, mask=mask)
    slices = payload_slices_for_alignment(payload)

    assert len(slices) == 1
    assert slices[0] is payload


def test_aligned_payload_keeps_unowned_image_metadata_whole() -> None:
    stack = np.zeros((2, 4, 5), dtype=np.float32)
    payload = ImageMetadataPayload(
        data=stack,
        metadata=ImagePayloadMetadata(
            source_plane_intensity_scales=(65535.0, 255.0),
            source_plane_dtypes=("uint16", "uint8"),
        ),
    )

    slices = payload_slices_for_alignment(payload)

    assert len(slices) == 1
    assert slices[0] is payload


def test_compose_one_image_bundle_stacks_per_image_masks_for_volume_bundle() -> None:
    first = MaskedImagePayload(
        data=np.zeros((3, 4, 5), dtype=np.float32),
        mask=np.ones((3, 4, 5), dtype=bool),
    )
    second = MaskedImagePayload(
        data=np.ones((3, 4, 5), dtype=np.float32),
        mask=np.zeros((3, 4, 5), dtype=bool),
    )

    bundle = ImagePayloadBundleContext.from_payloads((first, second)).compose()

    assert isinstance(bundle, MaskedImagePayload)
    assert bundle.data.shape == (2, 3, 4, 5)
    assert bundle.mask.shape == bundle.data.shape
    np.testing.assert_array_equal(bundle.mask[0], first.mask)
    np.testing.assert_array_equal(bundle.mask[1], second.mask)


def test_cellprofiler_auxiliary_payload_stack_preserves_metadata() -> None:
    first = ImagePayloadMetadata(
        intensity_scale=65535.0, source_dtype="uint16"
    ).payload_with(np.zeros((1, 4, 5), dtype=np.float32), None)
    second = ImagePayloadMetadata(
        intensity_scale=255.0, source_dtype="uint8"
    ).payload_with(np.ones((1, 4, 5), dtype=np.float32), None)

    stacked = Pure2DAuxiliaryOutputAggregator.aggregate(
        [first, second],
        "numpy",
    )

    assert isinstance(stacked, ImageMetadataPayload)
    assert image_payload_data(stacked).shape == (2, 1, 4, 5)
    assert (
        image_payload_metadata(stacked).for_source_plane(0).intensity_scale == 65535.0
    )
    assert image_payload_metadata(stacked).for_source_plane(1).source_dtype == "uint8"


def test_cellprofiler_image_aggregation_uses_nominal_image_payload_type() -> None:
    aggregated = Pure2DAuxiliaryOutputAggregator.aggregate(
        [np.zeros((4, 5), dtype=np.float32)],
        MemoryType.NUMPY.value,
    )

    assert image_payload_data(aggregated).shape == (1, 4, 5)
    assert image_payload_metadata(aggregated).plane_axis is (
        RuntimePlaneAxis.RUNTIME_SLICE
    )


def test_cellprofiler_aligned_main_output_aggregation_transposes_surfaces() -> None:
    outputs = (
        AlignedImageStack(
            (
                np.full((3, 4), 1.0, dtype=np.float32),
                np.full((3, 4), 10.0, dtype=np.float32),
            )
        ),
        AlignedImageStack(
            (
                np.full((3, 4), 2.0, dtype=np.float32),
                np.full((3, 4), 20.0, dtype=np.float32),
            )
        ),
    )

    aggregated = Pure2DAuxiliaryOutputAggregator.aggregate(
        outputs,
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, AlignedImageStack)
    assert len(aggregated.slices) == 2
    np.testing.assert_array_equal(
        image_payload_data(aggregated.slices[0])[:, 0, 0],
        np.asarray((1.0, 2.0)),
    )
    np.testing.assert_array_equal(
        image_payload_data(aggregated.slices[1])[:, 0, 0],
        np.asarray((10.0, 20.0)),
    )


def test_pure_2d_measurement_aggregation_rejects_unwrapped_scalar_records() -> None:
    @dataclass
    class SliceMeasurement(MeasurementFeatureRecord):
        slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
        value: float

    with pytest.raises(TypeError, match="registered nominal aggregator"):
        Pure2DAuxiliaryOutputAggregator.aggregate(
            [SliceMeasurement(0, 1.0), SliceMeasurement(1, 2.0)],
            MemoryType.NUMPY.value,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )


def test_pure_2d_declared_object_label_output_preserves_single_slice_axis() -> None:
    labels = np.asarray(((0, 1), (0, 0)), dtype=np.int32)
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
    )

    aggregated = Pure2DAuxiliaryOutputAggregator.aggregate(
        [label_payload],
        MemoryType.NUMPY.value,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.domain.scope is ObjectLabelDomainScope.PLANE
    np.testing.assert_array_equal(
        object_label_dense_array(aggregated),
        labels[np.newaxis, ...],
    )


def test_full_stack_preserves_nominal_object_label_output() -> None:
    image = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    labels = np.asarray(((0, 1), (0, 0)), dtype=np.int32)
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
    )

    def segment_like(
        stack: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, int], ObjectLabelPayload]:
        assert stack.shape == (1, 2, 2)
        return stack, {"count": 1}, label_payload

    segment_like.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        segment_like,
        artifact_outputs=(
            ArtifactSpec.output("Image", ImageArtifactType),
            ArtifactSpec.output("Measurements", MeasurementsArtifactType),
            ArtifactSpec.output("Labels", ObjectLabelsArtifactType),
        ),
    )

    (
        result_image,
        result_measurements,
        result_labels,
    ) = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        segment_like,
        image,
        {},
        execution_mode=ImagePayloadExecutionMode.FULL_STACK,
    )

    assert result_image is image
    assert result_measurements == {"count": 1}
    assert result_labels is label_payload


def test_full_stack_pure_2d_implicit_main_output_aligns_declared_object_labels() -> (
    None
):
    image = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    labels = np.asarray([[[0, 1], [0, 0]]], dtype=np.int32)
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain.declared(
            scope=ObjectLabelDomainScope.PAYLOAD,
            declared_object_ids=(1,),
        ),
    )

    def watershed_like(
        stack: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, int], ObjectLabelPayload]:
        assert stack.shape == (1, 2, 2)
        return stack, {"count": 1}, label_payload

    watershed_like.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        watershed_like,
        artifact_outputs=(
            ArtifactSpec.output("Image", ImageArtifactType),
            ArtifactSpec.output("Measurements", MeasurementsArtifactType),
            ArtifactSpec.output("Labels", ObjectLabelsArtifactType),
        ),
    )

    (
        result_image,
        result_measurements,
        result_labels,
    ) = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        watershed_like,
        image,
        {},
        execution_mode=ImagePayloadExecutionMode.FULL_STACK,
    )

    assert result_image is image
    assert result_measurements == {"count": 1}
    assert isinstance(result_labels, ObjectLabelPayload)
    assert result_labels.domain.scope is ObjectLabelDomainScope.PAYLOAD
    np.testing.assert_array_equal(object_label_dense_array(result_labels), labels)


def test_full_stack_pure_2d_non_flow_main_keeps_relationship_output_alignment() -> None:
    image = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(((0, 1), (0, 0)), dtype=np.int32)
        ),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
    )
    relationship = DirectedObjectRelationshipPayload(source_ids=(1,), target_ids=(1,))
    parent_spec = ArtifactSpec.input("Parent", ObjectLabelsArtifactType)
    labels_spec = ArtifactSpec.output("Labels", ObjectLabelsArtifactType)
    relationship_spec = ArtifactSpec.output(
        "Relationships",
        RelationshipsArtifactType,
        relations=(
            ObjectRelationshipDeclaration(
                source=parent_spec.ref(),
                target=labels_spec.ref(),
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
    )

    def object_transform_like(
        stack: np.ndarray,
    ) -> tuple[
        np.ndarray,
        dict[str, int],
        DirectedObjectRelationshipPayload,
        ObjectLabelPayload,
    ]:
        assert stack.shape == (1, 2, 2)
        return stack, {"count": 1}, relationship, labels

    object_transform_like.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        object_transform_like,
        artifact_inputs=(parent_spec,),
        artifact_outputs=(
            ArtifactSpec.output("Image", ImageArtifactType),
            ArtifactSpec.output("Measurements", MeasurementsArtifactType),
            relationship_spec,
            labels_spec,
        ),
    )

    (
        result_image,
        result_measurements,
        result_relationship,
        result_labels,
    ) = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        object_transform_like,
        image,
        {},
        execution_mode=ImagePayloadExecutionMode.FULL_STACK,
    )

    assert result_image is image
    assert result_measurements == {"count": 1}
    assert result_relationship is relationship
    assert result_labels is labels
    assert result_labels.domain.scope is ObjectLabelDomainScope.PAYLOAD


def test_pure_2d_aggregation_uses_generic_columnar_row_owner() -> None:
    first = _ColumnarMeasurementRows(
        {
            "slice_index": (0,),
            "object_label": (1,),
            "area": (4.0,),
        },
        fields=(
            FieldSpec("slice_index", int),
            FieldSpec("object_label", int),
            FieldSpec("area", float),
        ),
    )
    second = _ColumnarMeasurementRows(
        {
            "slice_index": (0,),
            "object_label": (1,),
            "area": (9.0,),
        },
        fields=(
            FieldSpec("slice_index", int),
            FieldSpec("object_label", int),
            FieldSpec("area", float),
        ),
    )

    aggregated = Pure2DAuxiliaryOutputAggregator.aggregate(
        [first, second],
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, ConcatenatedColumnarRows)
    assert tuple(aggregated.column_values("slice_index")) == (0, 1)
    assert tuple(aggregated.column_values("area")) == (4.0, 9.0)


def test_pure_2d_aggregation_preserves_concatenated_row_batch_schemas() -> None:
    image_rows = _ColumnarMeasurementRows(
        {"slice_index": (0,), "slope": (0.25,)},
        fields=(FieldSpec("slice_index", int), FieldSpec("slope", float)),
    )
    object_rows = _ColumnarMeasurementRows(
        {"slice_index": (0,), "object_label": (1,), "correlation": (0.5,)},
        fields=(
            FieldSpec("slice_index", int),
            FieldSpec("object_label", int),
            FieldSpec("correlation", float),
        ),
    )

    aggregated = Pure2DAuxiliaryOutputAggregator.aggregate(
        [ConcatenatedColumnarRows((image_rows, object_rows))],
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, ConcatenatedColumnarRows)
    aggregated_image_rows, aggregated_object_rows = aggregated.row_batches
    assert aggregated_image_rows.fields == image_rows.fields
    assert aggregated_object_rows.fields == object_rows.fields


@pytest.mark.parametrize(
    ("selected_site", "expected_value"),
    (("1", 0.4), ("2", 0.8)),
)
def test_illumination_apply_projects_broadcast_input_to_selected_primary_site(
    selected_site: str,
    expected_value: float,
) -> None:
    original_spec = ArtifactSpec.input("OrigHoechst", ImageArtifactType)
    illumination_spec = ArtifactSpec.input(
        "IllumHoechst",
        ImageArtifactType,
        relations=(InputStackBroadcastSourceRelation(source=original_spec.ref()),),
        parameter_name="illumination_function",
    )
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module(
            "CorrectIlluminationApply"
        ).require_callable(),
        artifact_inputs=(original_spec, illumination_spec),
        artifact_outputs=(),
    )
    original_pixels = np.full((3, 4), 0.2, dtype=np.float32)
    original_payload = ImagePayloadMetadata(
        source_path=f"/plate/A01_s00{selected_site}_w5_z001_t001.tif",
        source_component_metadata={
            "well": "A01",
            "site": selected_site,
            "channel": "5",
        },
        source_image_names=("OrigHoechst",),
    ).payload_with(original_pixels, None)
    illumination_payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/prepared/IllumHoechst/A01_s001_w5_z001_t001.npy",
                "/prepared/IllumHoechst/A01_s002_w5_z001_t001.npy",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "5"},
                {"well": "A01", "site": "2", "channel": "5"},
            ),
        ),
        source_image_names=("IllumHoechst",),
    ).payload_with(
        np.stack(
            (
                np.full((3, 4), 0.5, dtype=np.float32),
                np.full((3, 4), 0.25, dtype=np.float32),
            )
        ),
        None,
    )
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigHoechst": original_payload,
            "IllumHoechst": illumination_payload,
        },
        artifact_input_edges=(
            _artifact_input_edge_for_test(
                illumination_spec,
            ),
        ),
        variable_components=(VariableComponents.CHANNEL,),
    )
    executor = _module_executor(contract)

    _activate_runtime_contract(contract, runtime)
    module_type = _module_type_for_contract(contract)
    image_request = CellProfilerImageRequest(
        payload=original_payload,
        source_image_name=original_spec.name,
        source_aliases=(original_spec.name,),
        image_count=1,
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )
    invocation = executor._invocation_request(
        image_request=image_request,
        adapter=runtime,
        current_image=original_payload,
        kwargs={
            "method": "divide",
            "truncate_low": False,
            "truncate_high": False,
        },
        module_type=module_type,
    )
    result = CellProfilerFunctionContractExecutor().execute(
        contract,
        executor.raw_func,
        invocation.image,
        invocation.kwargs,
        execution_mode=invocation.execution_mode,
        plane_projection=invocation.plane_projection,
    )

    assert image_payload_data(result).shape == original_pixels.shape
    np.testing.assert_allclose(image_payload_data(result), expected_value)
    assert image_payload_metadata(result).source_component_metadata == {
        "well": "A01",
        "site": selected_site,
        "channel": "5",
    }


def test_illumination_apply_image_output_uses_original_input_source_payload() -> None:
    source_paths = (
        "/plate/IXMtest_A01_s1_w5.tif",
        "/plate/IXMtest_A01_s2_w5.tif",
    )
    source_metadata = (
        {"Well": "A01", "Site": "1", "ChannelNumber": "5"},
        {"Well": "A01", "Site": "2", "ChannelNumber": "5"},
    )
    orig_mito = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=source_paths, component_metadata=source_metadata
        ),
        source_image_names=("OrigMito",),
    ).payload_with(
        np.stack(
            (
                np.ones((3, 4), dtype=np.float32),
                np.full((3, 4), 2.0, dtype=np.float32),
            )
        ),
        None,
    )
    stale_invocation_payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(source_paths[0], source_paths[0]),
            component_metadata=(source_metadata[0], source_metadata[0]),
        ),
        source_image_names=("OrigHoechst", "OrigMito"),
    ).payload_with(np.zeros((2, 3, 4), dtype=np.float32), None)
    source_spec = ArtifactSpec.input("OrigMito", ImageArtifactType)
    illumination_spec = ArtifactSpec.input("IllumMito", ImageArtifactType)
    output_spec = _output_from_input(
        "Mito",
        source_spec.name,
        output_type=ImageArtifactType,
        input_type=ImageArtifactType,
    )
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigMito": orig_mito,
            "IllumMito": np.ones((3, 4), dtype=np.float32),
        },
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec)
            for spec in (source_spec, illumination_spec)
        ),
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module(
                "CorrectIlluminationApply"
            ).require_callable(),
            artifact_inputs=(source_spec, illumination_spec),
            artifact_outputs=(output_spec,),
        )
    )
    output = np.stack(
        (
            np.full((3, 4), 10.0, dtype=np.float32),
            np.full((3, 4), 20.0, dtype=np.float32),
        )
    )

    CellProfilerOutputRecorder.for_artifact_type(ImageArtifactType).record(
        _cellprofiler_output_record_request(
            callable_contract=executor.callable_contract,
            output_plans=(_artifact_output_plan(output_spec),),
            adapter=runtime,
            spec=output_spec,
            output_value=output,
            source_image_name=None,
            source_aliases=("OrigMito", "IllumMito"),
            source_image_payload=stale_invocation_payload,
            current_image=np.zeros((2, 3, 4), dtype=np.float32),
            call_kwargs={},
        )
    )

    recorded_metadata = image_payload_metadata(runtime.images["Mito"])
    assert recorded_metadata.source_image_provenance_planes.paths == source_paths
    assert (
        recorded_metadata.source_image_provenance_planes.component_metadata
        == source_metadata
    )
    assert recorded_metadata.source_image_names == ("Mito",)


@pytest.mark.parametrize(
    ("output_variable_components", "plane_projection", "expected_plane_axis"),
    (
        ((), None, None),
        (
            (VariableComponents.CHANNEL,),
            RuntimePlaneAxisValueProjection.preserve(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                axis_size=2,
            ),
            RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    ),
)
def test_image_output_projection_uses_exact_invocation_projection(
    output_variable_components: tuple[VariableComponents, ...],
    plane_projection: RuntimePlaneAxisValueProjection | None,
    expected_plane_axis: RuntimePlaneAxis | None,
) -> None:
    source_spec = ArtifactSpec.input("SourceImage", ImageArtifactType)
    output_spec = ArtifactSpec.output(
        "DerivedImage",
        ImageArtifactType,
        relations=(SourceStackLineageSourceRelation(source=source_spec.ref()),),
    )
    output_value = np.stack(
        (
            np.full((4, 5), 7.0, dtype=np.float32),
            np.full((4, 5), 9.0, dtype=np.float32),
        )
    )
    source_paths = (
        "/plate/A01_s001_w1_z001_t001.tif",
        "/plate/A01_s001_w2_z001_t001.tif",
    )
    source_payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_names=(source_spec.name,),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=source_paths,
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "1"},
                {"well": "A01", "site": "1", "channel": "2"},
            ),
        ),
    ).payload_with(np.zeros((2, 4, 5), dtype=np.float32), None)
    runtime = _FakeCellProfilerRuntime(
        {source_spec.name: source_payload},
        artifact_output_bindings=(
            (
                output_spec,
                ArtifactOutputPlan(
                    name=output_spec.name,
                    path=f"/artifacts/{output_spec.name}",
                    artifact_type=output_spec.artifact_type,
                    relations=output_spec.relations,
                    variable_components=output_variable_components,
                ),
            ),
        ),
        variable_components=(VariableComponents.CHANNEL,),
        plane_projection=RuntimePlaneProjection.stack(plane_count=2),
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("MaskImage").require_callable(),
            artifact_inputs=(source_spec,),
            artifact_outputs=(output_spec,),
        ),
    )

    CellProfilerOutputRecorder.for_artifact_type(ImageArtifactType).record(
        _cellprofiler_output_record_request(
            callable_contract=executor.callable_contract,
            artifact_input_edges=(_artifact_input_edge_for_test(source_spec),),
            output_plans=tuple(
                _artifact_output_plan(item) for item in ((output_spec,))
            ),
            adapter=runtime,
            spec=output_spec,
            output_value=output_value,
            source=CellProfilerImageRequest(
                payload=source_payload,
                source_image_name=source_spec.name,
                image_count=2,
                execution_mode=ImagePayloadExecutionMode.FULL_STACK,
                plane_projection=plane_projection,
            ),
            current_image=source_payload,
            call_kwargs={},
        )
    )

    recorded = runtime.images[output_spec.name]
    np.testing.assert_array_equal(image_payload_data(recorded), output_value)
    metadata = image_payload_metadata(recorded)
    assert metadata.plane_axis is expected_plane_axis
    assert metadata.source_image_names == (output_spec.name,)
    if expected_plane_axis is None:
        assert dict(metadata.source_component_metadata or {}) == {
            "well": "A01",
            "site": "1",
        }
        assert metadata.source_image_provenance_planes.count == 0
        assert metadata.source_image_provenance_planes.contributor_count == 2
        assert (
            metadata.source_image_provenance_planes.represented_source_image_names
            == (source_spec.name,)
        )
    else:
        assert metadata.source_image_provenance_planes.paths == source_paths


def test_declared_output_source_uses_projected_object_input_value() -> None:
    object_spec = ArtifactSpec.input(
        "Cells",
        ObjectLabelsArtifactType,
        parameter_name="labels",
    )
    output_spec = ArtifactSpec.output(
        "ColorNeighbors",
        ImageArtifactType,
        relations=(
            SourceStackLineageSourceRelation(source=object_spec.ref()),
        ),
    )
    labels = ObjectLabelSet(
        name=object_spec.name,
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                (((0, 1, 1), (0, 0, 0)),),
                dtype=np.int32,
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    output_plan = replace(
        _artifact_output_plan(output_spec),
        variable_components=(VariableComponents.SITE,),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={object_spec.name: labels},
        artifact_input_edges=(_artifact_input_edge_for_test(object_spec),),
        artifact_output_bindings=((output_spec, output_plan),),
        variable_components=(VariableComponents.SITE,),
        plane_projection=RuntimePlaneProjection.stack(plane_count=1),
    )
    callable_contract = _compiled_callable_contract(
        MeasureObjectNeighborsModule.require_callable(),
        artifact_inputs=(object_spec,),
        artifact_outputs=(output_spec,),
    )
    current_image = ImagePayloadMetadata().payload_with(
        np.zeros((2, 3), dtype=np.float32),
        None,
    )
    output_value = np.zeros((2, 3, 3), dtype=np.float32)
    request = _cellprofiler_output_record_request(
        callable_contract=callable_contract,
        output_plans=(output_plan,),
        adapter=runtime,
        spec=output_spec,
        output_value=output_value,
        source=CellProfilerImageRequest(
            payload=current_image,
            source_image_name=None,
            image_count=1,
            plane_projection=None,
        ),
        current_image=current_image,
        call_kwargs={},
    )

    projected_source = MeasureObjectNeighborsModule.source_payload(request)

    assert isinstance(projected_source, ObjectLabelSet)
    assert projected_source.plane_axis is None
    assert object_label_dense_array(projected_source).shape == (2, 3)

    CellProfilerOutputRecorder.for_artifact_type(ImageArtifactType).record(request)

    recorded = runtime.images[output_spec.name]
    assert image_payload_data(recorded).shape == (2, 3, 3)
    assert image_payload_metadata(recorded).plane_axis is None


def test_image_output_recording_projects_masked_singleton_rgb_payload() -> None:
    source_spec = ArtifactSpec.input("GrayImage", ImageArtifactType)
    output_spec = _output_from_input(
        "ColorImage",
        source_spec.name,
        output_type=ImageArtifactType,
        input_type=ImageArtifactType,
    )
    mask = np.array(
        (
            (True, False, True, True, True),
            (True, True, True, True, True),
            (False, True, True, True, True),
            (True, True, True, False, True),
        ),
        dtype=bool,
    )
    spatial_domain = SourceSpatialDomain(source_shape_yx=(4, 5))
    source_payload = ImagePayloadMetadata(
        source_spatial_domain=spatial_domain,
        source_image_names=(source_spec.name,),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(
        np.zeros((1, 4, 5), dtype=np.float32),
        mask[np.newaxis, ...],
    )
    output_payload = ImagePayloadMetadata(
        source_spatial_domain=spatial_domain,
        source_channel_axis=3,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(
        np.zeros((1, 4, 5, 3), dtype=np.float32),
        mask[np.newaxis, ...],
    )
    runtime = _FakeCellProfilerRuntime(
        {source_spec.name: source_payload},
        plane_projection=RuntimePlaneProjection.stack(plane_count=1),
        variable_components=(VariableComponents.CHANNEL,),
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("GrayToColor").require_callable(),
            artifact_inputs=(source_spec,),
            artifact_outputs=(output_spec,),
        ),
    )

    CellProfilerOutputRecorder.for_artifact_type(ImageArtifactType).record(
        _cellprofiler_output_record_request(
            callable_contract=executor.callable_contract,
            artifact_input_edges=tuple(
                _artifact_input_edge_for_test(spec)
                for spec in executor.callable_contract.artifact_inputs.specs
            ),
            output_plans=tuple(
                _artifact_output_plan(item) for item in ((output_spec,))
            ),
            adapter=runtime,
            spec=output_spec,
            output_value=output_payload,
            source=CellProfilerImageRequest(
                payload=source_payload,
                source_image_name=source_spec.name,
                image_count=1,
                execution_mode=ImagePayloadExecutionMode.FULL_STACK,
            ),
            current_image=source_payload,
            call_kwargs={},
        )
    )

    recorded = runtime.images[output_spec.name]
    assert isinstance(recorded, MaskedImagePayload)
    assert image_payload_data(recorded).shape == (4, 5, 3)
    np.testing.assert_array_equal(image_payload_mask(recorded), mask)
    metadata = image_payload_metadata(recorded)
    assert metadata.plane_axis is None
    assert metadata.normalized_source_channel_axis(recorded) == 2


def test_illumination_apply_image_output_preserves_declared_stack_value() -> None:
    source_path = "/plate/IXMtest_A01_s1_w5.tif"
    source_metadata = {"Well": "A01", "Site": "1", "ChannelNumber": "5"}
    source_payload = ImagePayloadMetadata(
        source_path=source_path,
        source_component_metadata=source_metadata,
        source_image_names=("OrigMito",),
    ).payload_with(np.ones((5, 6), dtype=np.float32), None)
    duplicate_output = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(source_path, source_path),
            component_metadata=(source_metadata, source_metadata),
        ),
        source_image_names=("OrigMito", "OrigSyto"),
    ).payload_with(
        np.stack(
            (
                np.full((5, 6), 10.0, dtype=np.float32),
                np.full((5, 6), 20.0, dtype=np.float32),
            )
        ),
        None,
    )
    source_spec = ArtifactSpec.input("OrigMito", ImageArtifactType)
    illumination_spec = ArtifactSpec.input("IllumMito", ImageArtifactType)
    output_spec = _output_from_input(
        "Mito",
        source_spec.name,
        output_type=ImageArtifactType,
        input_type=ImageArtifactType,
    )
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigMito": source_payload,
            "IllumMito": np.ones((5, 6), dtype=np.float32),
        },
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec)
            for spec in (source_spec, illumination_spec)
        ),
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module(
                "CorrectIlluminationApply"
            ).require_callable(),
            artifact_inputs=(source_spec, illumination_spec),
            artifact_outputs=(output_spec,),
        )
    )

    CellProfilerOutputRecorder.for_artifact_type(ImageArtifactType).record(
        _cellprofiler_output_record_request(
            callable_contract=executor.callable_contract,
            output_plans=(_artifact_output_plan(output_spec),),
            adapter=runtime,
            spec=output_spec,
            output_value=duplicate_output,
            source_image_name=None,
            source_aliases=("OrigMito", "IllumMito"),
            source_image_payload=duplicate_output,
            current_image=np.zeros((5, 6), dtype=np.float32),
            call_kwargs={},
        )
    )

    recorded = runtime.images["Mito"]
    np.testing.assert_allclose(
        image_payload_data(recorded),
        image_payload_data(duplicate_output),
    )
    recorded_metadata = image_payload_metadata(recorded)
    assert recorded_metadata.source_path == source_path
    assert recorded_metadata.source_component_metadata == source_metadata
    assert recorded_metadata.source_image_provenance_planes.paths == (
        source_path,
        source_path,
    )
    assert recorded_metadata.source_image_provenance_planes.component_metadata == (
        source_metadata,
        source_metadata,
    )
    assert recorded_metadata.source_image_names == ("Mito",)


def test_illumination_apply_image_output_preserves_singleton_volume_value() -> None:
    source_path = "/plate/IXMtest_A01_s2_w5.tif"
    source_metadata = {"Well": "A01", "Site": "2", "ChannelNumber": "5"}
    source_payload = ImagePayloadMetadata(
        source_path=source_path,
        source_component_metadata=source_metadata,
        source_image_names=("OrigMito",),
    ).payload_with(np.ones((5, 6), dtype=np.float32), None)
    singleton_output = ImagePayloadMetadata(
        source_path=source_path,
        source_component_metadata=source_metadata,
        source_image_names=("OrigMito",),
    ).payload_with(np.full((1, 5, 6), 30.0, dtype=np.float32), None)
    source_spec = ArtifactSpec.input("OrigMito", ImageArtifactType)
    illumination_spec = ArtifactSpec.input("IllumMito", ImageArtifactType)
    output_spec = _output_from_input(
        "Mito",
        source_spec.name,
        output_type=ImageArtifactType,
        input_type=ImageArtifactType,
    )
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigMito": source_payload,
            "IllumMito": np.ones((5, 6), dtype=np.float32),
        },
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec)
            for spec in (source_spec, illumination_spec)
        ),
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module(
                "CorrectIlluminationApply"
            ).require_callable(),
            artifact_inputs=(source_spec, illumination_spec),
            artifact_outputs=(output_spec,),
        )
    )

    CellProfilerOutputRecorder.for_artifact_type(ImageArtifactType).record(
        _cellprofiler_output_record_request(
            callable_contract=executor.callable_contract,
            output_plans=(_artifact_output_plan(output_spec),),
            adapter=runtime,
            spec=output_spec,
            output_value=singleton_output,
            source_image_name=None,
            source_aliases=("OrigMito", "IllumMito"),
            source_image_payload=singleton_output,
            current_image=np.zeros((5, 6), dtype=np.float32),
            call_kwargs={},
        )
    )

    recorded = runtime.images["Mito"]
    np.testing.assert_allclose(
        image_payload_data(recorded),
        image_payload_data(singleton_output),
    )
    recorded_metadata = image_payload_metadata(recorded)
    assert recorded_metadata.source_path == source_path
    assert recorded_metadata.source_component_metadata == source_metadata
    assert recorded_metadata.source_image_names == ("Mito",)


def test_cellprofiler_contract_executor_slices_aligned_runtime_kwargs():
    calls = []

    def keep_labels(image: np.ndarray, *, labels: ObjectLabelValue):
        calls.append((image.shape, object_label_dense_array(labels).shape))
        return image, labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    raw_contract = CallableContract.from_callable(keep_labels)
    callable_contract = replace(
        raw_contract,
        metadata=replace(
            raw_contract.metadata,
            artifact_outputs=(
                ArtifactSpec.output("Image", ImageArtifactType),
                ArtifactSpec.output("Labels", ObjectLabelsArtifactType),
            ),
        ),
    )
    stack = ImageMetadataPayload(
        np.zeros((2, 4, 5), dtype=np.uint16),
        ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.ones(stack.shape, dtype=np.int32)
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (1,)),
        ),
    )

    result_image, result_labels = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        keep_labels,
        stack,
        {"labels": labels},
        execution_mode=ImagePayloadExecutionMode.NATURAL,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert calls == [((4, 5), (4, 5)), ((4, 5), (4, 5))]
    assert image_payload_data(result_image).shape == stack.shape
    assert object_label_dense_array(result_labels).shape == stack.shape


def test_cellprofiler_contract_executor_aggregates_object_label_payload_auxiliary():
    def keep_payload(image: np.ndarray):
        object_id = int(image[0, 0]) + 1
        labels = np.full(image.shape, object_id, dtype=np.int32)
        return (
            image,
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(
                    labels=labels,
                    unedited_labels=labels + 10,
                    small_removed_labels=labels + 20,
                ),
                domain=ObjectLabelDomain(declared_object_ids=(object_id,)),
            ),
        )

    keep_payload.__processing_contract__ = ProcessingContract.PURE_2D
    raw_contract = CallableContract.from_callable(keep_payload)
    callable_contract = replace(
        raw_contract,
        metadata=replace(
            raw_contract.metadata,
            artifact_outputs=(
                ArtifactSpec.output("Image", ImageArtifactType),
                ArtifactSpec.output("Objects", ObjectLabelsArtifactType),
            ),
        ),
    )
    stack = ImageMetadataPayload(
        np.stack(
            (
                np.zeros((4, 5), dtype=np.uint16),
                np.ones((4, 5), dtype=np.uint16),
            )
        ),
        ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )

    result_image, result_payload = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        keep_payload,
        stack,
        {},
        execution_mode=ImagePayloadExecutionMode.NATURAL,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert image_payload_data(result_image).shape == stack.shape
    assert isinstance(result_payload, ObjectLabelPayload)
    assert result_payload.labels.shape == stack.shape
    np.testing.assert_array_equal(result_payload.labels[0], np.full((4, 5), 1))
    np.testing.assert_array_equal(result_payload.labels[1], np.full((4, 5), 2))
    np.testing.assert_array_equal(
        result_payload.unedited_labels,
        result_payload.labels + 10,
    )
    np.testing.assert_array_equal(
        result_payload.small_removed_labels,
        result_payload.labels + 20,
    )


def test_cellprofiler_contract_executor_preserves_site_metadata_for_object_label_auxiliary():
    def segment(image: object):
        object_id = int(image_payload_metadata(image).source_component_metadata["site"])
        labels = np.full(
            image_payload_data(image).shape,
            object_id,
            dtype=np.int32,
        )
        return (
            image,
            SourceImageObjectLabelBuildRequest(
                image=image,
                labels=labels,
                declared_object_ids=(object_id,),
            ).payload(),
        )

    segment.__processing_contract__ = ProcessingContract.PURE_2D
    raw_contract = CallableContract.from_callable(segment)
    callable_contract = replace(
        raw_contract,
        metadata=replace(
            raw_contract.metadata,
            artifact_outputs=(
                ArtifactSpec.output("Image", ImageArtifactType),
                ArtifactSpec.output("Objects", ObjectLabelsArtifactType),
            ),
        ),
    )
    image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.TIF",
                "/input/A01_s002_w1_z001_t001.TIF",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "1"},
                {"well": "A01", "site": "2", "channel": "1"},
            ),
        ),
    ).payload_with(np.zeros((2, 4, 5), dtype=np.float32), None)

    result_image, result_payload = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        segment,
        image,
        {},
        execution_mode=ImagePayloadExecutionMode.NATURAL,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert image_payload_data(result_image).shape == (2, 4, 5)
    assert isinstance(result_payload, ObjectLabelPayload)
    assert result_payload.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert result_payload.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.TIF",
        "/input/A01_s002_w1_z001_t001.TIF",
    )
    assert tuple(
        dict(item)
        for item in result_payload.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A01", "site": "1", "channel": "1"},
        {"well": "A01", "site": "2", "channel": "1"},
    )
    np.testing.assert_array_equal(result_payload.labels[0], np.full((4, 5), 1))
    np.testing.assert_array_equal(result_payload.labels[1], np.full((4, 5), 2))


def test_aligned_stack_object_label_auxiliary_aggregates_on_runtime_slice_axis():
    def segment(image: object):
        site_index = int(
            image_payload_metadata(image).source_component_metadata["site"]
        )
        labels = np.full(
            image_payload_data(image).shape,
            site_index,
            dtype=np.int32,
        )
        return (
            image,
            SourceImageObjectLabelBuildRequest(
                image=image,
                labels=labels,
            ).label_set(name="Nuclei", source_image_name="OrigBlue"),
        )

    segment.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        segment,
        artifact_outputs=(
            ArtifactSpec.output("Image", ImageArtifactType),
            ArtifactSpec.output("Labels", ObjectLabelsArtifactType),
        ),
    )
    first_site = ImagePayloadMetadata(
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
        },
        source_image_names=("OrigBlue",),
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)
    second_site = ImagePayloadMetadata(
        source_component_metadata={
            "well": "A01",
            "site": "2",
            "channel": "1",
        },
        source_image_names=("OrigBlue",),
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)

    result_image, result_payload = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        segment,
        AlignedImageStack((first_site, second_site)),
        {},
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert image_payload_data(result_image).shape == (2, 4, 5)
    assert isinstance(result_payload, ObjectLabelSet)
    assert result_payload.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert result_payload.source_image_names == ("OrigBlue", "OrigBlue")
    np.testing.assert_array_equal(result_payload.labels[0], np.full((4, 5), 1))
    np.testing.assert_array_equal(result_payload.labels[1], np.full((4, 5), 2))


def test_object_label_measurement_images_bundle_source_metadata() -> None:
    dna_payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1_z001_t001.tif",),
            component_metadata=(
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                    "z_index": "1",
                },
            ),
        ),
        source_image_names=("OrigDNA",),
    ).payload_with(np.full((4, 5), 1, dtype=np.float32), None)
    rna_payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w2_z001_t001.tif",),
            component_metadata=(
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "2",
                    "z_index": "1",
                },
            ),
        ),
        source_image_names=("OrigRNA",),
    ).payload_with(np.full((4, 5), 2, dtype=np.float32), None)

    metadata = CellProfilerMeasurementImage.composed_source_metadata(
        (
            CellProfilerMeasurementImage(
                source_image_name=None,
                payload=dna_payload,
                reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
            ),
            CellProfilerMeasurementImage(
                source_image_name=None,
                payload=rna_payload,
                reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
            ),
        )
    )

    assert metadata is not None
    assert metadata.source_image_provenance_planes.count == 2
    assert tuple(
        plane.source_identity.path
        for plane in metadata.source_image_provenance_planes.planes
    ) == (
        "/input/A01_s001_w1_z001_t001.tif",
        "/input/A01_s001_w2_z001_t001.tif",
    )
    assert dict(metadata.source_component_metadata) == {
        "extension": ".tif",
        "well": "A01",
        "site": "1",
        "z_index": "1",
    }


def test_source_measurement_images_stack_source_metadata() -> None:
    dna_payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1_z001_t001.tif",),
            component_metadata=({"well": "A01", "site": "1", "channel": "1"},),
        ),
        source_image_names=("OrigDNA",),
    ).payload_with(np.full((4, 5), 1, dtype=np.float32), None)
    rna_payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w2_z001_t001.tif",),
            component_metadata=({"well": "A01", "site": "1", "channel": "2"},),
        ),
        source_image_names=("OrigRNA",),
    ).payload_with(np.full((4, 5), 2, dtype=np.float32), None)

    metadata = CellProfilerMeasurementImage.composed_source_metadata(
        (
            CellProfilerMeasurementImage(
                source_image_name="OrigDNA",
                payload=dna_payload,
            ),
            CellProfilerMeasurementImage(
                source_image_name="OrigRNA",
                payload=rna_payload,
            ),
        )
    )

    assert metadata is not None
    assert metadata.source_image_provenance_planes.count == 2


def test_measurement_image_preserves_composed_source_planes_before_collapsing() -> None:
    source_names = ("Mito", "Syto", "Ph_golgi", "Hoechst", "ER")
    payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(
                f"/input/A01_s001_w{channel}.tif"
                for channel in range(1, len(source_names) + 1)
            ),
            component_metadata=tuple(
                {
                    "well": "A01",
                    "site": "1",
                    "channel": str(channel),
                }
                for channel in range(1, len(source_names) + 1)
            ),
        ),
        source_image_names=source_names,
    ).payload_with(
        np.zeros((len(source_names), 4, 5), dtype=np.float32),
        None,
    )

    metadata = CellProfilerMeasurementImage.composed_source_metadata(
        (
            CellProfilerMeasurementImage(
                source_image_name=None,
                source_aliases=source_names,
                payload=payload,
            ),
        )
    )

    assert metadata is not None
    assert metadata.source_image_provenance_planes.count == len(source_names)
    assert metadata.source_image_provenance_planes.component_metadata == tuple(
        {
            "well": "A01",
            "site": "1",
            "channel": str(channel),
        }
        for channel in range(1, len(source_names) + 1)
    )


def test_measurement_images_flatten_aligned_runtime_slices_for_source_metadata() -> (
    None
):
    aligned_sites = AlignedImageStack(
        tuple(
            AlignedImageStack(
                tuple(
                    ImagePayloadMetadata(
                        source_path=(f"/plate/A01_s{site:03d}_w{channel}.tif"),
                        source_component_metadata={
                            "well": "A01",
                            "site": str(site),
                            "channel": str(channel),
                        },
                    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)
                    for channel in (1, 2)
                )
            )
            for site in (1, 2)
        )
    )

    metadata = CellProfilerMeasurementImage.composed_source_metadata(
        (
            CellProfilerMeasurementImage(
                source_image_name=None,
                source_aliases=("CropBlue", "CropGreen"),
                payload=aligned_sites,
            ),
        )
    )

    assert metadata is not None
    assert metadata.source_image_provenance_planes.count == 2
    assert metadata.source_image_provenance_planes.component_metadata == (
        {"well": "A01", "site": "1"},
        {"well": "A01", "site": "2"},
    )


def test_measurement_images_preserve_runtime_slice_axis_across_source_aliases() -> None:
    def source_payload(alias: str, channel: int) -> AlignedImageStack:
        return AlignedImageStack(
            tuple(
                ImagePayloadMetadata(
                    source_path=f"/plate/A01_s{site:03d}_w{channel}.tif",
                    source_component_metadata={
                        "well": "A01",
                        "site": str(site),
                        "channel": str(channel),
                    },
                    source_image_names=(alias,),
                ).payload_with(np.zeros((4, 5), dtype=np.float32), None)
                for site in (1, 2)
            )
        )

    metadata = CellProfilerMeasurementImage.composed_source_metadata(
        (
            CellProfilerMeasurementImage(
                source_image_name="OrigER",
                payload=source_payload("OrigER", 1),
            ),
            CellProfilerMeasurementImage(
                source_image_name="OrigHoechst",
                payload=source_payload("OrigHoechst", 2),
            ),
        )
    )

    assert metadata is not None
    assert metadata.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert metadata.source_image_provenance_planes.component_metadata[:2] == (
        {"well": "A01", "site": "1", "channel": "1"},
        {"well": "A01", "site": "2", "channel": "1"},
    )


def test_measurement_tables_compose_provenance_after_object_rows_clear_source() -> None:
    first_metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1.tif",),
            component_metadata=({"well": "A01", "site": "1"},),
        ),
        source_image_names=("OrigDNA",),
    )
    second_metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s002_w1.tif",),
            component_metadata=({"well": "A01", "site": "2"},),
        ),
        source_image_names=("OrigDNA",),
    )
    tables = (
        MeasurementTable(
            name="Locations",
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"object_name": "Cells", "center_x": 12.0},),
                fields=(
                    FieldSpec("object_name", str),
                    FieldSpec("center_x", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
            measurement_feature_owner=CellProfilerModule,
            source_provenance=first_metadata.source_provenance,
        ),
        MeasurementTable(
            name="Locations",
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"object_name": "Cells", "center_x": 34.0},),
                fields=(
                    FieldSpec("object_name", str),
                    FieldSpec("center_x", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
            measurement_feature_owner=CellProfilerModule,
            source_provenance=second_metadata.source_provenance,
        ),
    )

    combined = MeasurementTableUnion("Locations", tables).as_table()

    assert all(table.source_image_name is None for table in tables)
    assert tuple(
        plane.source_identity.path
        for plane in combined.source_image_provenance_planes.planes
    ) == (
        "/input/A01_s001_w1.tif",
        "/input/A01_s002_w1.tif",
    )
    assert combined.source_image_provenance_planes.component_metadata == (
        {"well": "A01", "site": "1"},
        {"well": "A01", "site": "2"},
    )


def test_image_output_context_preserves_aligned_image_stack_payload():
    dna_payload = ImagePayloadMetadata(source_image_names=("OrigDNA",)).payload_with(
        np.full((4, 5), 1, dtype=np.float32), None
    )
    rna_payload = ImagePayloadMetadata(source_image_names=("OrigRNA",)).payload_with(
        np.full((4, 5), 2, dtype=np.float32), None
    )
    aligned = AlignedImageStack(slices=(dna_payload, rna_payload))

    result = FunctionOutputContextStrategy.for_output_plan(None).contextualize(
        dna_payload,
        aligned,
        None,
        None,
    )

    assert result is aligned


def test_pattern_group_runtime_unstacks_aligned_image_stack_output():
    dna_payload = ImagePayloadMetadata(
        intensity_scale=65535,
        source_image_names=("OrigDNA",),
    ).payload_with(np.full((4, 5), 1, dtype=np.float32), None)
    rna_payload = ImagePayloadMetadata(
        intensity_scale=65535,
        source_image_names=("OrigRNA",),
    ).payload_with(np.full((4, 5), 2, dtype=np.float32), None)
    aligned = AlignedImageStack(slices=(dna_payload, rna_payload))
    loaded = PatternGroupData(
        matching_files=["dna.tif", "rna.tif"],
        main_data_stack=np.zeros((2, 4, 5), dtype=np.float32),
    )
    runtime = object.__new__(PatternGroupRuntime)
    runtime.request = SimpleNamespace(
        execution_plan=SimpleNamespace(
            output_memory_type="numpy",
            device_id=0,
        )
    )

    output = runtime._validate_and_unstack(aligned, loaded)

    assert output == [dna_payload, rna_payload]
    assert image_payload_metadata(
        output.stack_payload
    ).source_plane_intensity_scales == (65535, 65535)


def test_pattern_group_runtime_does_not_invent_nested_axes_from_image_rank():
    first_site = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/plate/A01_s001_w1_z001_t001.tif",
                "/plate/A01_s001_w2_z001_t001.tif",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "1"},
                {"well": "A01", "site": "1", "channel": "2"},
            ),
        ),
    ).payload_with(
        np.stack(
            (
                np.full((4, 5), 1, dtype=np.float32),
                np.full((4, 5), 2, dtype=np.float32),
            )
        ),
        None,
    )
    second_site = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/plate/A01_s002_w1_z001_t001.tif",
                "/plate/A01_s002_w2_z001_t001.tif",
            ),
            component_metadata=(
                {"well": "A01", "site": "2", "channel": "1"},
                {"well": "A01", "site": "2", "channel": "2"},
            ),
        ),
    ).payload_with(
        np.stack(
            (
                np.full((4, 5), 3, dtype=np.float32),
                np.full((4, 5), 4, dtype=np.float32),
            )
        ),
        None,
    )
    aligned = AlignedImageStack(slices=(first_site, second_site))
    loaded = PatternGroupData(
        matching_files=["A01_s001_w2_z001_t001.tif"],
        main_data_stack=np.zeros((4, 5), dtype=np.float32),
    )
    runtime = object.__new__(PatternGroupRuntime)
    runtime.request = SimpleNamespace(
        execution_plan=SimpleNamespace(
            output_memory_type="numpy",
            device_id=0,
        )
    )

    output_slices = runtime._validate_and_unstack(aligned, loaded)

    assert len(output_slices) == 2
    np.testing.assert_array_equal(image_payload_data(output_slices[0]), first_site.data)
    np.testing.assert_array_equal(
        image_payload_data(output_slices[1]), second_site.data
    )
    assert [
        image_payload_metadata(payload).source_image_provenance_planes
        for payload in output_slices
    ] == [
        image_payload_metadata(first_site).source_image_provenance_planes,
        image_payload_metadata(second_site).source_image_provenance_planes,
    ]


def test_pattern_group_runtime_leaves_variable_shape_aligned_outputs_uncached():
    first = np.ones((4, 5), dtype=np.float32)
    second = np.ones((3, 4), dtype=np.float32)
    aligned = AlignedImageStack(slices=(first, second))
    loaded = PatternGroupData(
        matching_files=["first.tif", "second.tif"],
        main_data_stack=np.zeros((2, 4, 5), dtype=np.float32),
    )
    runtime = object.__new__(PatternGroupRuntime)
    runtime.request = SimpleNamespace(
        execution_plan=SimpleNamespace(
            output_memory_type="numpy",
            device_id=0,
        )
    )

    output = runtime._validate_and_unstack(aligned, loaded)

    assert output == [first, second]
    assert output.stack_payload is None


def test_cellprofiler_main_flow_output_preserves_input_source_planes():
    input_image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w3_z001_t001.TIF",
                "/input/A01_s002_w3_z001_t001.TIF",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "3"},
                {"well": "A01", "site": "2", "channel": "3"},
            ),
        ),
    ).payload_with(np.zeros((2, 4, 5), dtype=np.float32), None)
    output_image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)

    result = cellprofiler_main_flow_output(
        input_image,
        output_image,
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    metadata = image_payload_metadata(result)
    assert image_payload_data(result).shape == (2, 4, 5)
    assert metadata.source_image_provenance_planes.paths == (
        "/input/A01_s001_w3_z001_t001.TIF",
        "/input/A01_s002_w3_z001_t001.TIF",
    )
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A01", "site": "1", "channel": "3"},
        {"well": "A01", "site": "2", "channel": "3"},
    )


def test_cellprofiler_main_flow_output_replaces_scalar_output_source_on_stack():
    input_image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w3_z001_t001.TIF",
                "/input/A01_s002_w3_z001_t001.TIF",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "3"},
                {"well": "A01", "site": "2", "channel": "3"},
            ),
        ),
    ).payload_with(np.zeros((2, 4, 5), dtype=np.float32), None)
    output_image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_component_metadata={"well": "A01", "site": "1", "channel": "3"},
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)

    result = cellprofiler_main_flow_output(
        input_image,
        output_image,
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    metadata = image_payload_metadata(result)
    assert metadata.source_image_provenance_planes.paths == (
        "/input/A01_s001_w3_z001_t001.TIF",
        "/input/A01_s002_w3_z001_t001.TIF",
    )
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A01", "site": "1", "channel": "3"},
        {"well": "A01", "site": "2", "channel": "3"},
    )


def test_cellprofiler_contract_executor_projects_batch_relationship_auxiliary():
    from openhcs.core.runtime_batch_contracts import (
        RuntimePure2DSliceBatchRequest,
        pure_2d_batch_executor,
    )

    def batch_relationships(request: RuntimePure2DSliceBatchRequest):
        return [
            (
                image,
                DirectedObjectRelationshipPayload(
                    source_ids=(slice_index + 1,),
                    target_ids=(slice_index + 10,),
                ),
            )
            for slice_index, image in enumerate(request.slices_2d)
        ]

    @pure_2d_batch_executor(batch_relationships)
    def relate(image: np.ndarray):
        return image, DirectedObjectRelationshipPayload(source_ids=(), target_ids=())

    relate.__processing_contract__ = ProcessingContract.PURE_2D
    parent_spec = ArtifactSpec.input("Parents", ObjectLabelsArtifactType)
    child_spec = ArtifactSpec.input("Children", ObjectLabelsArtifactType)
    raw_contract = CallableContract.from_callable(relate)
    callable_contract = replace(
        raw_contract,
        metadata=replace(
            raw_contract.metadata,
            artifact_inputs=(parent_spec, child_spec),
            artifact_outputs=(
                ArtifactSpec.output("Image", ImageArtifactType),
                _relationship_output(parent_spec, child_spec),
            ),
        ),
    )
    stack = ImageMetadataPayload(
        np.zeros((2, 4, 5), dtype=np.float32),
        ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )

    result_image, relationship = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        relate,
        stack,
        {},
        execution_mode=ImagePayloadExecutionMode.NATURAL,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert image_payload_data(result_image).shape == stack.shape
    assert relationship == DirectedObjectRelationshipPayload(
        source_ids=(1, 2),
        target_ids=(10, 11),
        slice_indices=(0, 1),
        slice_count=2,
    )


def test_cellprofiler_contract_executor_aggregates_volume_label_auxiliary():
    def keep_volume_labels(image: np.ndarray):
        labels = (image > 0).astype(np.int32)
        return image, ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        )

    keep_volume_labels.__processing_contract__ = ProcessingContract.PURE_2D
    raw_contract = CallableContract.from_callable(keep_volume_labels)
    callable_contract = replace(
        raw_contract,
        metadata=replace(
            raw_contract.metadata,
            artifact_outputs=(
                ArtifactSpec.output("Image", ImageArtifactType),
                ArtifactSpec.output("Objects", ObjectLabelsArtifactType),
            ),
        ),
    )
    stack = ImageMetadataPayload(
        np.stack(
            (
                np.ones((3, 4, 5), dtype=np.float32),
                np.full((3, 4, 5), 2.0, dtype=np.float32),
            )
        ),
        ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )

    result_image, result_labels = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        keep_volume_labels,
        stack,
        {},
        execution_mode=ImagePayloadExecutionMode.NATURAL,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert image_payload_data(result_image).shape == (2, 3, 4, 5)
    assert isinstance(result_labels, ObjectLabelValue)
    assert object_label_dense_array(result_labels).shape == (2, 3, 4, 5)


def test_cellprofiler_contract_executor_preserves_single_slice_dataclass_auxiliary():
    @dataclass(frozen=True)
    class SliceStats:
        slice_index: int
        threshold_used: float

    def segment(image: np.ndarray, *, slice_index: int = 0, slice_count: int = 1):
        assert slice_count == 1
        return image, SliceStats(slice_index=slice_index, threshold_used=0.25)

    segment.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        segment,
        artifact_outputs=(
            ArtifactSpec.output("Image", ImageArtifactType),
            ArtifactSpec.output("Stats", MeasurementsArtifactType),
        ),
    )
    image = np.ones((4, 5), dtype=np.float32)

    result_image, result_stats = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        segment,
        image,
        {},
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    np.testing.assert_array_equal(result_image, image)
    assert result_stats == SliceStats(slice_index=0, threshold_used=0.25)


def test_cellprofiler_contract_executor_does_not_infer_stack_from_ndarray_kwargs():
    calls = []

    def increment_labels(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape))
        return labels + 1

    increment_labels.__processing_contract__ = ProcessingContract.PURE_2D
    raw_contract = CallableContract.from_callable(increment_labels)
    callable_contract = replace(
        raw_contract,
        metadata=replace(
            raw_contract.metadata,
            artifact_outputs=(ArtifactSpec.output("Image", ImageArtifactType),),
        ),
    )
    image = np.zeros((4, 5), dtype=np.uint16)
    labels = np.stack(
        (
            np.ones((4, 5), dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        increment_labels,
        image,
        {"labels": labels},
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    assert calls == [((4, 5), (2, 4, 5))]
    np.testing.assert_array_equal(result, labels + 1)


def test_mask_objects_uses_object_labels_as_primary_execution_domain() -> None:
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("MaskObjects").require_callable(),
        artifact_inputs=(
            ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
            ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
            ArtifactSpec.input("CarrierImage", ImageArtifactType),
        ),
        artifact_outputs=(),
    )

    module_type = MaskObjectsModule
    assert (
        module_type.primary_image_inputs(
            mask_objects,
            contract.artifact_inputs,
        )
        == ()
    )


def test_pure_2d_batch_lowers_nominal_runtime_output_bundles() -> None:
    first_output = np.ones((2, 2), dtype=np.float32)
    second_output = np.full((2, 2), 2, dtype=np.float32)
    first_relationship = DirectedObjectRelationshipPayload(
        source_ids=(1,), target_ids=(1,)
    )
    second_relationship = DirectedObjectRelationshipPayload(
        source_ids=(2,), target_ids=(2,)
    )
    first_reverse_relationship = DirectedObjectRelationshipPayload(
        source_ids=(1,), target_ids=(1,)
    )
    second_reverse_relationship = DirectedObjectRelationshipPayload(
        source_ids=(2,), target_ids=(2,)
    )
    first_measurements = MeasurementSparseColumnarRows.from_rows((), fields=())
    second_measurements = MeasurementSparseColumnarRows.from_rows((), fields=())

    batch = Pure2DSliceResultBatch.from_results(
        (
            RelateObjectsResult(
                first_output,
                first_relationship,
                first_reverse_relationship,
                first_measurements,
            ),
            RelateObjectsResult(
                second_output,
                second_relationship,
                second_reverse_relationship,
                second_measurements,
            ),
        )
    )

    assert batch.main_outputs == [first_output, second_output]
    assert batch.auxiliary_groups == (
        [first_relationship, second_relationship],
        [first_reverse_relationship, second_reverse_relationship],
        [first_measurements, second_measurements],
    )


def test_cellprofiler_contract_executor_broadcasts_2d_labels_to_image_stack():
    calls = []

    def add_label_values(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape, int(image[0, 0])))
        return image + labels

    add_label_values.__processing_contract__ = ProcessingContract.PURE_2D
    raw_contract = CallableContract.from_callable(add_label_values)
    callable_contract = replace(
        raw_contract,
        metadata=replace(
            raw_contract.metadata,
            artifact_outputs=(ArtifactSpec.output("Image", ImageArtifactType),),
        ),
    )
    image = ImageMetadataPayload(
        np.stack(
            (
                np.full((4, 5), 10, dtype=np.uint16),
                np.full((4, 5), 20, dtype=np.uint16),
            )
        ),
        ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )
    labels = np.ones((4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        add_label_values,
        image,
        {"labels": labels},
        execution_mode=ImagePayloadExecutionMode.NATURAL,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert calls == [((4, 5), (4, 5), 10), ((4, 5), (4, 5), 20)]
    assert image_payload_data(result).shape == image.shape
    np.testing.assert_array_equal(
        image_payload_data(result),
        image.data + labels[np.newaxis, ...],
    )


def test_primary_object_variant_relabel_preserves_accepted_identity_and_rejected_blockers():
    accepted_before = np.array(
        [
            [0, 7, 7, 0],
            [0, 0, 0, 0],
            [0, 3, 3, 0],
        ],
        dtype=np.int32,
    )
    final_labels = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 2, 2, 0],
        ],
        dtype=np.int32,
    )
    unedited_variant = np.array(
        [
            [4, 7, 7, 0],
            [0, 0, 0, 0],
            [0, 3, 3, 0],
        ],
        dtype=np.int32,
    )

    remapped = _remap_object_label_variant_after_final_relabel(
        unedited_variant,
        accepted_before,
        final_labels,
        object_count=2,
    )

    expected = np.array(
        [
            [3, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 2, 2, 0],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(remapped, expected)


def test_secondary_seed_labels_preserve_unedited_ids_and_edge_constraints():
    final_labels = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    unedited_labels = np.array(
        [
            [4, 0, 0, 0],
            [0, 7, 7, 0],
            [0, 3, 3, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    labels_in = _secondary_seed_labels(final_labels, unedited_labels)

    expected = np.array(
        [
            [4, 0, 0, 0],
            [0, 7, 7, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(labels_in, expected)


def test_filter_labels_maps_unedited_secondary_labels_to_accepted_primary_labels():
    primary_labels = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 0, 2],
        ],
        dtype=np.int32,
    )
    secondary_labels = np.array(
        [
            [7, 7, 7, 0],
            [8, 8, 0, 9],
        ],
        dtype=np.int32,
    )

    filtered = _filter_labels(secondary_labels, primary_labels)

    expected = np.array(
        [
            [1, 1, 1, 0],
            [0, 0, 0, 2],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(filtered, expected)


def test_distance_b_limits_expansion_from_accepted_primary_labels(monkeypatch):
    def fake_propagate(
        self: DistanceMaskedSegmentationStrategy,
        request: SecondarySegmentationRequest,
        *,
        regularization: float,
        max_distance: float | None = None,
    ) -> np.ndarray:
        del self, request, regularization, max_distance
        return np.array([[1, 0, 0, 1, 4]], dtype=np.int32)

    monkeypatch.setattr(
        DistanceMaskedSegmentationStrategy,
        "propagate_labels",
        fake_propagate,
    )
    final_labels = np.array([[1, 0, 0, 0, 0]], dtype=np.int32)
    unedited_labels = np.array([[1, 0, 0, 0, 4]], dtype=np.int32)

    segmented = DistanceMaskedSegmentationStrategy().segment(
        SecondarySegmentationRequest(
            inputs=SourceImageObjectLabelBuildRequest(
                image=np.zeros((1, 5), dtype=np.float32),
                labels=final_labels,
                unedited_labels=unedited_labels,
            ),
            thresholded=np.ones((1, 5), dtype=bool),
            distance_to_dilate=2,
            regularization_factor=0.05,
            watershed_backend_provider=None,
        )
    )

    expected = np.array([[1, 0, 0, 1, 0]], dtype=np.int32)
    np.testing.assert_array_equal(segmented, expected)


def test_secondary_propagation_uses_threshold_mask_without_seed_or(monkeypatch):
    captured: dict[str, np.ndarray] = {}

    def fake_propagate(
        self: PropagationSegmentationStrategy,
        request: SecondarySegmentationRequest,
        *,
        regularization: float,
        max_distance: float | None = None,
    ) -> np.ndarray:
        del self, regularization, max_distance
        captured["mask"] = request.thresholded
        return request.seed_labels.copy()

    monkeypatch.setattr(
        PropagationSegmentationStrategy,
        "propagate_labels",
        fake_propagate,
    )
    labels = np.array([[1, 0], [0, 0]], dtype=np.int32)
    thresholded = np.array([[False, False], [False, True]])

    PropagationSegmentationStrategy().segment(
        SecondarySegmentationRequest(
            inputs=SourceImageObjectLabelBuildRequest(
                image=np.zeros((2, 2), dtype=np.float32),
                labels=labels,
                unedited_labels=labels,
            ),
            thresholded=thresholded,
            distance_to_dilate=10,
            regularization_factor=0.05,
            watershed_backend_provider=None,
        )
    )

    np.testing.assert_array_equal(captured["mask"], thresholded)


def test_secondary_propagation_methods_own_numba_default_backend():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    request = SecondarySegmentationRequest(
        inputs=SourceImageObjectLabelBuildRequest(
            image=np.zeros((2, 2), dtype=np.float32),
            labels=np.zeros((2, 2), dtype=np.int32),
            unedited_labels=np.zeros((2, 2), dtype=np.int32),
        ),
        thresholded=np.zeros((2, 2), dtype=bool),
        distance_to_dilate=10,
        regularization_factor=0.05,
        watershed_backend_provider=None,
    )

    assert (
        PropagationSegmentationStrategy().propagation_backend_provider(request)
        is CellProfilerBackendProvider.NUMBA
    )
    assert (
        DistanceMaskedSegmentationStrategy().propagation_backend_provider(request)
        is CellProfilerBackendProvider.NUMBA
    )


def test_identify_secondary_objects_consumes_projected_image_label_and_mask_plane():
    image = np.zeros((5, 5), dtype=np.float32)
    image[1:4, 1:4] = 1.0
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[2, 2] = 1
    payload = ImagePayloadMetadata().payload_with(
        image, np.ones_like(image, dtype=bool)
    )
    primary = SourceImageObjectLabelBuildRequest(
        image=payload,
        labels=labels,
    ).payload()

    _image, _stats, _relationships, secondary = iso.identify_secondary_objects(
        payload,
        primary,
        method=iso.SecondaryMethod.DISTANCE_B,
        distance_to_dilate=1,
        dtype_config=DtypeConfig(),
    )

    secondary_labels = object_label_dense_array(secondary)

    assert secondary_labels.shape == (5, 5)
    assert secondary_labels.max() == 1


def test_identify_secondary_objects_preserves_source_domain_for_relationships():
    image = np.zeros((5, 5), dtype=np.float32)
    image[1:4, 1:4] = 1.0
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[2, 2] = 1
    source_domain = SourceSpatialDomain(
        origin_yx=(2, 3),
        source_shape_yx=(9, 10),
    )
    payload = ImagePayloadMetadata(source_spatial_domain=source_domain).payload_with(
        image, np.ones_like(image, dtype=bool)
    )
    primary = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        source_spatial_domain=source_domain,
    )

    _image, _stats, relationships, secondary = iso.identify_secondary_objects(
        payload,
        primary,
        method=iso.SecondaryMethod.DISTANCE_B,
        distance_to_dilate=1,
        dtype_config=DtypeConfig(),
    )

    assert relationships.source_ids == (1,)
    assert relationships.target_ids == (1,)
    assert secondary.spatial_origin_yx == source_domain.origin_yx
    assert secondary.source_spatial_shape_yx == source_domain.source_shape_yx


def test_secondary_propagation_rejects_undeclared_spatial_domain_mismatch():
    image = np.zeros((6, 7), dtype=np.float32)
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[2, 2] = 1
    mask = np.ones((5, 5), dtype=bool)

    with pytest.raises(ValueError, match="must exactly match"):
        PropagationSegmentationStrategy().propagate_labels(
            SecondarySegmentationRequest(
                inputs=SourceImageObjectLabelBuildRequest(
                    image=image,
                    labels=labels,
                    unedited_labels=labels,
                ),
                thresholded=mask,
                distance_to_dilate=10,
                regularization_factor=1.0,
                watershed_backend_provider=None,
            ),
            regularization=1.0,
        )


def test_parent_child_relationship_aligns_cropped_object_label_payload_to_source_domain():
    parent = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[1, 0], [0, 0]], dtype=np.int32)
        ),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(6, 7),
        ),
    )
    child_labels = np.zeros((6, 7), dtype=np.int32)
    child_labels[2, 3] = 4
    child = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=child_labels),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(0, 0),
            source_shape_yx=(6, 7),
        ),
    )

    relationship = object_label_parent_child_payload(parent, child)

    assert relationship.source_ids == (1,)
    assert relationship.target_ids == (4,)


def test_pure_2d_object_label_payload_aggregation_preserves_source_domain():
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[1, 0], [0, 0]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(6, 7),
        ),
    )

    aggregated = Pure2DAuxiliaryOutputAggregator.aggregate(
        [payload],
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.spatial_origin_yx == (2, 3)
    assert aggregated.source_spatial_shape_yx == (6, 7)


def test_pure_2d_object_label_payload_aggregation_expands_varying_crop_domains():
    first = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.array([[1]], dtype=np.int32)),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(1, 2),
            source_shape_yx=(4, 5),
        ),
        domain=ObjectLabelDomain(
            declared_object_count=3,
        ),
    )
    second = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.array([[2]], dtype=np.int32)),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(4, 5),
        ),
        domain=ObjectLabelDomain(
            declared_object_count=5,
        ),
    )

    aggregated = Pure2DAuxiliaryOutputAggregator.aggregate(
        [first, second],
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.labels.shape == (2, 4, 5)
    assert aggregated.domain.declared_object_id_domains == (
        (1, 2, 3),
        (1, 2, 3, 4, 5),
    )
    assert aggregated.domain.scope is ObjectLabelDomainScope.PLANE
    assert aggregated.spatial_origin_yx is None
    assert aggregated.source_spatial_shape_yx == (4, 5)
    assert aggregated.labels[0, 1, 2] == 1
    assert aggregated.labels[1, 2, 3] == 2


def test_pure_2d_object_label_payload_aggregation_uses_declared_slice_domains():
    first = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[1, 0], [0, 0]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(
            declared_object_ids=(1, 2),
        ),
    )
    second = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[3, 0], [0, 0]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(
            declared_object_ids=(3,),
        ),
    )

    aggregated = Pure2DAuxiliaryOutputAggregator.aggregate(
        [first, second],
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.domain.declared_object_id_domains == ((1, 2), (3,))
    assert aggregated.domain.scope is ObjectLabelDomainScope.PLANE


def test_pure_2d_object_label_payload_slice_projects_plane_domain() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array(
                [
                    [[1, 0], [0, 2]],
                    [[1, 0], [3, 4]],
                ],
                dtype=np.int32,
            )
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (1, 2, 3, 4)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    sliced = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
    )

    assert isinstance(sliced, ObjectLabelPayload)
    assert sliced.labels.shape == (2, 2)
    assert sliced.domain.declared_object_ids == (1, 2, 3, 4)
    assert sliced.domain.declared_object_id_domains == ()
    assert sliced.domain.scope is ObjectLabelDomainScope.PAYLOAD


def test_object_label_endpoint_uses_declared_runtime_slice_axis() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array(
                [
                    [[1, 0], [0, 2]],
                    [[0, 3], [4, 0]],
                ],
                dtype=np.int32,
            )
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (3, 4)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    sliced = RuntimeSliceProjection.object_label_endpoint(
        payload,
        context=RuntimeSliceProjection.context_for_value(
            payload,
            slice_index=1,
            source_description="test object-label endpoint",
        ),
    )

    assert isinstance(sliced, ObjectLabelPayload)
    np.testing.assert_array_equal(
        sliced.labels,
        np.array([[0, 3], [4, 0]], dtype=np.int32),
    )
    assert sliced.domain.declared_object_ids == (3, 4)


def test_object_label_endpoint_keeps_planar_payload_in_slice_scope() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[1, 0], [0, 2]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
    )

    sliced = RuntimeSliceProjection.object_label_endpoint(
        payload,
        context=RuntimeSliceProjection.context_for_value(
            payload,
            slice_index=0,
            source_description="test planar object-label endpoint",
        ),
    )

    assert sliced is payload


def test_object_label_endpoint_rejects_singleton_axis_cardinality_mismatch() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[[1, 0], [0, 2]]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    with pytest.raises(ValueError, match="cardinality mismatch"):
        RuntimeSliceProjection.object_label_endpoint(
            payload,
            context=RuntimePlaneAxisValueProjection.from_selected_plane(
                axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=3
            ),
        )


def test_pure_2d_object_label_set_slice_projects_plane_domain() -> None:
    label_set = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(
            labels=np.array(
                [
                    [[1, 0], [0, 2]],
                    [[1, 0], [3, 4]],
                ],
                dtype=np.int32,
            )
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (1, 2, 3, 4)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    sliced = RuntimeSliceProjection.value_for_slice(
        label_set,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=0, axis_size=2
        ),
    )

    assert isinstance(sliced, ObjectLabelSet)
    assert sliced.labels.shape == (2, 2)
    assert sliced.domain.declared_object_ids == (1, 2)
    assert sliced.domain.declared_object_id_domains == ()
    assert sliced.domain.scope is ObjectLabelDomainScope.PAYLOAD


def test_pure_2d_object_label_payload_slice_projects_grouped_plane_domains() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.zeros((2, 3, 4, 5), dtype=np.int32)
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2, 3), (4, 5, 6)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    sliced = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
    )

    assert isinstance(sliced, ObjectLabelPayload)
    assert sliced.labels.shape == (3, 4, 5)
    assert sliced.domain.declared_object_ids == (4, 5, 6)
    assert sliced.domain.declared_object_id_domains == ()
    assert sliced.domain.scope is ObjectLabelDomainScope.PAYLOAD


def test_grouped_object_label_planes_project_source_provenance_independently() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.zeros((2, 2, 4, 5), dtype=np.int32)
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (3, 4)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/site-1.tif", "/input/site-2.tif"),
            component_metadata=({"site": "1"}, {"site": "2"}),
        ),
    )

    sliced = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
    )

    assert isinstance(sliced, ObjectLabelPayload)
    assert sliced.labels.shape == (2, 4, 5)
    assert sliced.domain.declared_object_ids == (3, 4)
    assert sliced.source_path == "/input/site-2.tif"
    assert dict(sliced.source_component_metadata) == {"site": "2"}


def test_runtime_slice_count_allows_grouped_object_label_planes() -> None:
    parent = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.zeros((2, 3, 4, 5), dtype=np.int32)
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2, 3), (4, 5, 6)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    child = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 4, 5), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2, 3), (4, 5, 6)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    assert RuntimeSliceProjection.slice_count_from_values((parent, child)) == 2


def test_runtime_slice_count_uses_sequence_value_declarations() -> None:
    first = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 4, 5), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    second = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 4, 5), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((3,), (4,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    assert RuntimeSliceProjection.slice_count_from_values(((first, second),)) == 2
    assert (
        RuntimeSliceProjection.slice_count_from_kwargs(
            {"object_labels": (first, second)},
        )
        == 2
    )


def test_runtime_slice_projection_does_not_offset_repeated_scalar_tables() -> None:
    first = MeasurementTable(
        name="MeasureObjectIntensity_7_measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            [{"slice_index": 0, "object_label": 1, "std_intensity": 0.1}],
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("std_intensity", float),
            ),
        ),
        source_image_name="DF_image",
        subject=MeasurementSubject(
            MeasurementScope.OBJECT, "Tile_of_grid", "object_label"
        ),
    )
    second = MeasurementTable(
        name="MeasureObjectIntensity_7_measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            [{"slice_index": 0, "object_label": 1, "std_intensity": 0.2}],
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("std_intensity", float),
            ),
        ),
        source_image_name="DF_image",
        subject=MeasurementSubject(
            MeasurementScope.OBJECT, "Tile_of_grid", "object_label"
        ),
    )

    tables = (first, second)

    with pytest.raises(
        RuntimeSliceProjectionDeclarationError,
        match="source-plane provenance",
    ):
        RuntimeSliceProjection.slice_count_from_values((tables,))
    sliced = RuntimeSliceProjection.kwargs_for_slice(
        {"measurement_tables": tables},
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
    )["measurement_tables"]
    assert tuple(table.rows.row_count() for table in sliced) == (0, 0)


def test_identify_tertiary_aligns_cropped_primary_labels_to_secondary_domain():
    primary = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[1, 0], [0, 0]], dtype=np.int32)
        ),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(6, 7),
        ),
    )
    secondary_array = np.zeros((6, 7), dtype=np.int32)
    secondary_array[2, 3] = 5
    secondary_array[2, 4] = 5
    secondary = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=secondary_array),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(0, 0),
            source_shape_yx=(6, 7),
        ),
    )

    *_, tertiary = ito.identify_tertiary_objects.__wrapped__(
        np.zeros((6, 7), dtype=np.float32),
        primary_labels=primary,
        secondary_labels=secondary,
        shrink_primary=False,
    )

    tertiary_array = object_label_dense_array(tertiary)
    assert tertiary_array.shape == secondary_array.shape
    assert tertiary_array[2, 3] == 0
    assert tertiary_array[2, 4] == 5


def test_identify_tertiary_drops_fully_subtracted_secondary_objects():
    primary = np.zeros((3, 3), dtype=np.int32)
    primary[1, 1] = 1
    secondary = np.zeros((3, 3), dtype=np.int32)
    secondary[1, 1] = 5

    *_, tertiary = ito.identify_tertiary_objects.__wrapped__(
        np.zeros((3, 3), dtype=np.float32),
        primary_labels=ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=primary)
        ),
        secondary_labels=ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=secondary)
        ),
        shrink_primary=False,
    )

    assert np.count_nonzero(object_label_dense_array(tertiary) == 5) == 0


def test_identify_tertiary_preserves_exact_runtime_projected_plane_domains():
    primary = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 3, 3), dtype=np.int32)),
        source_image_names=("rawDNA", "rawActin"),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((), ()),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    secondary = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                [
                    [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 5, 0], [0, 0, 0]],
                ],
                dtype=np.int32,
            )
        ),
        source_image_names=("rawDNA", "rawActin"),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((2,), (5,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    outputs = []
    for slice_index in range(2):
        projection_axis = RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=slice_index, axis_size=2
        )
        *_, output = ito.identify_tertiary_objects.__wrapped__(
            np.zeros((3, 3), dtype=np.float32),
            primary_labels=RuntimeSliceProjection.value_for_slice(
                primary, projection_axis
            ),
            secondary_labels=RuntimeSliceProjection.value_for_slice(
                secondary, projection_axis
            ),
            shrink_primary=False,
        )
        outputs.append(output)

    first, second = outputs
    assert isinstance(first, ObjectLabelPayload)
    assert isinstance(second, ObjectLabelPayload)
    assert first.domain.declared_object_ids == (2,)
    assert second.domain.declared_object_ids == (5,)
    assert first.source_image_names == ("rawDNA",)
    assert second.source_image_names == ("rawActin",)


def test_tertiary_projected_plane_keeps_dense_local_label_domain():
    source = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(
            labels=SparseIJVLabelRows.from_dense_stack(
                np.asarray(
                    (
                        ((0, 2), (0, 0)),
                        ((0, 0), (5, 0)),
                    ),
                    dtype=np.int32,
                )
            )
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(10, 11),
        ),
        source_image_names=("rawDNA", "rawActin"),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((2,), (5,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    labels = np.asarray(((0, 0), (5, 0)), dtype=np.int32)
    projected_source = RuntimeSliceProjection.value_for_slice(
        source,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
    )

    output = ito.TertiaryObjectLabelOutput(
        projected_source,
        labels,
    ).value()

    assert isinstance(output, ObjectLabelSet)
    assert output.representation is ObjectLabelRepresentation.DENSE_LABELS
    assert output.spatial_origin_yx == (2, 3)
    assert output.source_spatial_shape_yx == (10, 11)
    assert output.source_image_names == ("rawActin",)
    np.testing.assert_array_equal(object_label_dense_array(output), labels)


def test_identify_tertiary_single_slice_aligns_payload_domains_before_dense_extraction():
    primary = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[1, 0], [0, 0]], dtype=np.int32)
        ),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(6, 7),
        ),
    )
    secondary_array = np.zeros((6, 7), dtype=np.int32)
    secondary_array[2, 3] = 5
    secondary_array[2, 4] = 5
    secondary = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=secondary_array),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(0, 0),
            source_shape_yx=(6, 7),
        ),
    )

    _, _, _, _, tertiary = ito.identify_tertiary_objects.__wrapped__(
        np.zeros((6, 7), dtype=np.float32),
        primary_labels=primary,
        secondary_labels=secondary,
        shrink_primary=False,
    )

    tertiary_array = object_label_dense_array(tertiary)
    assert tertiary_array.shape == secondary_array.shape
    assert tertiary_array[2, 3] == 0
    assert tertiary_array[2, 4] == 5


def test_identify_tertiary_single_slice_restores_secondary_crop_domain():
    primary = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(((1, 0), (0, 0)), dtype=np.int32)
        ),
        representation=ObjectLabelRepresentation.DENSE_LABELS,
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(6, 7),
        ),
    )
    secondary = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(((5, 5), (0, 0)), dtype=np.int32)
        ),
        representation=ObjectLabelRepresentation.DENSE_LABELS,
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(6, 7),
        ),
    )

    _, _, _, _, tertiary = ito.identify_tertiary_objects.__wrapped__(
        np.zeros((6, 7), dtype=np.float32),
        primary_labels=primary,
        secondary_labels=secondary,
        shrink_primary=False,
    )

    assert isinstance(tertiary, ObjectLabelSet)
    assert (
        object_label_dense_array(tertiary).shape
        == object_label_dense_array(secondary).shape
    )
    assert tertiary.spatial_origin_yx == (2, 3)
    assert tertiary.source_spatial_shape_yx == (6, 7)
    np.testing.assert_array_equal(
        object_label_dense_array(tertiary),
        np.asarray(((0, 5), (0, 0)), dtype=np.int32),
    )


def test_enhance_or_suppress_features_matches_white_tophat_reference():
    image = np.zeros((15, 15), dtype=np.float32)
    image[4, 4] = 1.0
    image[8, 9] = 0.75

    result = enhance_or_suppress_features(
        image,
        radius=4,
        speckle_accuracy=SpeckleAccuracy.SLOW,
        dtype_config=DtypeConfig(),
    )

    expected = skimage.morphology.white_tophat(
        image,
        footprint=skimage.morphology.disk(4),
    ).astype(np.float32)
    np.testing.assert_allclose(image_payload_data(result), expected)


def test_enhance_or_suppress_features_fast_speckles_uses_cellprofiler_disk():
    from scipy import ndimage as ndi

    image = np.zeros((17, 17), dtype=np.float32)
    image[8, 8] = 1.0
    image[8, 13] = 0.5
    footprint = skimage.morphology.disk(5)

    result = enhance_or_suppress_features(
        image,
        radius=5,
        speckle_accuracy=SpeckleAccuracy.FAST,
        dtype_config=DtypeConfig(),
    )

    expected = image - ndi.maximum_filter(
        ndi.minimum_filter(image, footprint=footprint),
        footprint=footprint,
    )
    np.testing.assert_allclose(image_payload_data(result), expected.astype(np.float32))


def test_enhance_or_suppress_features_tubeness_matches_hessian_reference():
    from scipy import ndimage as ndi

    image = np.zeros((21, 21), dtype=np.float32)
    image[5:16, 10] = 1.0
    image[10, 5:16] = 0.5
    smoothing_value = 2.0

    result = enhance_or_suppress_features(
        image,
        enhance_method=EnhanceMethod.NEURITES,
        neurite_method=NeuriteMethod.TUBENESS,
        smoothing_value=smoothing_value,
        dtype_config=DtypeConfig(),
    )

    smoothed = ndi.gaussian_filter(image, smoothing_value)
    hessian = np.zeros((*smoothed.shape, 2, 2), dtype=np.float64)
    hessian[1:-1, :, 0, 0] = (
        smoothed[:-2, :] - (2 * smoothed[1:-1, :]) + smoothed[2:, :]
    )
    hessian[1:-1, 1:-1, 0, 1] = (
        smoothed[2:, 2:] + smoothed[:-2, :-2] - smoothed[2:, :-2] - smoothed[:-2, 2:]
    ) / 4
    hessian[:, 1:-1, 1, 1] = (
        smoothed[:, :-2] - (2 * smoothed[:, 1:-1]) + smoothed[:, 2:]
    )
    a = hessian[:, :, 0, 0]
    b = hessian[:, :, 0, 1]
    c = hessian[:, :, 1, 1]
    linear = -(a + c)
    constant = a * c - b * b
    discriminant = np.maximum(linear * linear - 4 * constant, 0)
    roots = np.empty((*smoothed.shape, 2), dtype=np.float64)
    sqrt_discriminant = np.sqrt(discriminant)
    roots[:, :, 0] = (-linear + sqrt_discriminant) / 2
    roots[:, :, 1] = (-linear - sqrt_discriminant) / 2
    swap = np.abs(roots[:, :, 1]) > np.abs(roots[:, :, 0])
    roots[swap] = roots[swap, ::-1]
    expected = (-roots[..., 0] * (roots[..., 0] < 0) * (smoothing_value**2)).astype(
        np.float32
    )
    np.testing.assert_allclose(
        image_payload_data(result), expected, rtol=1e-6, atol=1e-7
    )


def test_cellprofiler_contract_executor_preserves_multi_image_stack_payload():
    calls = []

    def keep_stack(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image

    keep_stack.__processing_contract__ = ProcessingContract.PURE_2D
    callable_contract = _compiled_callable_contract(
        keep_stack, artifact_outputs=(ArtifactSpec.output("Stack", ImageArtifactType),)
    )
    stack = np.zeros((3, 4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        keep_stack,
        stack,
        {},
        execution_mode=ImagePayloadExecutionMode.FULL_STACK,
    )

    assert calls == [(3, 4, 5)]
    assert result.shape == stack.shape


def test_object_only_measurement_carrier_preserves_color_stack() -> None:
    color_stack = np.zeros((2, 4, 5, 3), dtype=np.float32)
    color_stack[0, :, :, 1] = 7

    carrier = CellProfilerMeasurementImage(
        source_image_name=None,
        source_aliases=(),
        payload=color_stack,
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    assert carrier.payload is color_stack
    assert carrier.reference_domain is CellProfilerMeasurementImageDomain.OBJECT_LABELS


def test_compose_image_payload_rejects_unowned_singleton_broadcast():
    spatial_domain = SourceSpatialDomain(source_shape_yx=(4, 5))
    raw_stack = ImagePayloadMetadata(
        source_spatial_domain=spatial_domain,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(
        np.stack(
            (
                np.full((4, 5), 11, dtype=np.float32),
                np.full((4, 5), 22, dtype=np.float32),
            )
        ),
        None,
    )
    illumination = ImagePayloadMetadata(
        source_spatial_domain=spatial_domain
    ).payload_with(np.full((4, 5), 3, dtype=np.float32), None)

    with pytest.raises(ValueError, match="require an explicit runtime-slice owner"):
        compose_aligned_image_payload(
            "CorrectIlluminationApply",
            (raw_stack, illumination),
        )


def test_compose_image_payload_broadcasts_only_from_declared_stack_owner():
    spatial_domain = SourceSpatialDomain(source_shape_yx=(4, 5))
    raw_stack = ImagePayloadMetadata(
        source_spatial_domain=spatial_domain,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(
        np.stack(
            (
                np.full((4, 5), 11, dtype=np.float32),
                np.full((4, 5), 22, dtype=np.float32),
            )
        ),
        None,
    )
    illumination = ImagePayloadMetadata(
        source_spatial_domain=spatial_domain
    ).payload_with(np.full((4, 5), 3, dtype=np.float32), None)

    composition = compose_aligned_image_payload(
        "CorrectIlluminationApply",
        (raw_stack, illumination),
        stack_broadcast_source_indices=(None, 0),
    )

    assert composition.execution_mode is (
        ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
    )
    assert isinstance(composition.payload, AlignedImageStack)
    assert len(composition.payload.slices) == 2
    first_pair = image_payload_data(composition.payload.slices[0])
    second_pair = image_payload_data(composition.payload.slices[1])
    np.testing.assert_array_equal(first_pair[0], np.full((4, 5), 11))
    np.testing.assert_array_equal(second_pair[0], np.full((4, 5), 22))
    np.testing.assert_array_equal(first_pair[1], np.full((4, 5), 3))
    np.testing.assert_array_equal(second_pair[1], np.full((4, 5), 3))


def test_compose_image_payload_does_not_invent_pairwise_alignment_from_shape():
    first = np.zeros((2, 2, 4, 5), dtype=np.float32)
    second = np.zeros((2, 2, 4, 5), dtype=np.float32)
    first[0, 0] = 11
    first[0, 1] = 99
    first[1, 0] = 98
    first[1, 1] = 22
    second[0, 0] = 3
    second[0, 1] = 97
    second[1, 0] = 96
    second[1, 1] = 7

    composition = compose_aligned_image_payload(
        "MeasureColocalization",
        (first, second),
    )

    assert composition.execution_mode is ImagePayloadExecutionMode.FULL_STACK
    assert image_payload_data(composition.payload).shape == (2, 2, 2, 4, 5)
    np.testing.assert_array_equal(image_payload_data(composition.payload)[0], first)
    np.testing.assert_array_equal(image_payload_data(composition.payload)[1], second)


def test_compose_image_bundle_promotes_grayscale_into_color_bundle():
    spatial_domain = SourceSpatialDomain(source_shape_yx=(4, 5))
    color_data = np.zeros((4, 5, 3), dtype=np.float32)
    color_data[:, :, 0] = 1
    color = ImagePayloadMetadata(
        source_channel_axis=-1,
        source_spatial_domain=spatial_domain,
    ).payload_with(color_data, None)
    grayscale = ImagePayloadMetadata(source_spatial_domain=spatial_domain).payload_with(
        np.full((4, 5), 7, dtype=np.float32), None
    )

    bundle = ImagePayloadBundleContext.from_payloads((color, grayscale)).compose()

    assert bundle.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(bundle[0], color_data)
    np.testing.assert_array_equal(bundle[1, :, :, 0], grayscale)
    np.testing.assert_array_equal(bundle[1, :, :, 1], grayscale)
    np.testing.assert_array_equal(bundle[1, :, :, 2], grayscale)


def test_compose_image_bundle_rejects_shape_selected_singleton_collapse():
    spatial_domain = SourceSpatialDomain(source_shape_yx=(4, 5))
    singleton = ImagePayloadMetadata(
        source_spatial_domain=spatial_domain,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.full((1, 4, 5), 3, dtype=np.float32), None)
    plane = ImagePayloadMetadata(source_spatial_domain=spatial_domain).payload_with(
        np.full((4, 5), 7, dtype=np.float32), None
    )

    with pytest.raises(ValueError, match="projected before composition"):
        ImagePayloadBundleContext.from_payloads((singleton, plane)).compose()


def test_compose_image_bundle_intersects_masks() -> None:
    image_a = np.ones((4, 5), dtype=np.float32)
    image_b = np.full((4, 5), 2, dtype=np.float32)
    mask_a = np.array(
        (
            (True, False, True, True, True),
            (True, True, True, True, True),
            (False, True, True, True, True),
            (True, True, True, False, True),
        )
    )
    mask_b = np.array(
        (
            (True, True, True, False, True),
            (True, True, False, True, True),
            (True, True, True, True, True),
            (True, False, True, True, True),
        )
    )

    bundle = ImagePayloadBundleContext.from_payloads(
        (
            MaskedImagePayload(
                data=image_a,
                mask=mask_a,
                metadata=ImagePayloadMetadata(
                    source_spatial_domain=SourceSpatialDomain(source_shape_yx=(4, 5))
                ),
            ),
            MaskedImagePayload(
                data=image_b,
                mask=mask_b,
                metadata=ImagePayloadMetadata(
                    source_spatial_domain=SourceSpatialDomain(source_shape_yx=(4, 5))
                ),
            ),
        )
    ).compose()

    assert isinstance(bundle, MaskedImagePayload)
    assert bundle.data.shape == (2, 4, 5)
    np.testing.assert_array_equal(bundle.mask, mask_a & mask_b)


def test_compose_image_bundle_aligns_cropped_payload_to_source_domain() -> None:
    full = ImageMetadataPayload(
        data=np.full((4, 5), 2, dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(0, 0),
                source_shape_yx=(4, 5),
            ),
        ),
    )
    cropped = ImageMetadataPayload(
        data=np.full((2, 2), 7, dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(1, 2),
                source_shape_yx=(4, 5),
            ),
        ),
    )

    bundle = ImagePayloadBundleContext.from_payloads((full, cropped)).compose()

    assert isinstance(bundle, ImageMetadataPayload)
    assert bundle.metadata.spatial_origin_yx == (0, 0)
    assert bundle.metadata.source_spatial_shape_yx == (4, 5)
    assert bundle.data.shape == (2, 4, 5)
    np.testing.assert_array_equal(bundle.data[0], full.data)
    expected_cropped = np.zeros((4, 5), dtype=np.float32)
    expected_cropped[1:3, 2:4] = 7
    np.testing.assert_array_equal(bundle.data[1], expected_cropped)


def test_tile_preserves_color_stack_output_shape():
    image = np.zeros((2, 3, 4, 3), dtype=np.float32)
    image[0, :, :, 0] = 1
    image[1, :, :, 1] = 2

    output = tile(image, rows=1, columns=2, dtype_config=DtypeConfig())

    assert output.shape == (3, 8, 3)
    assert image_payload_metadata(output).source_channel_axis == -1
    assert image_payload_metadata(output).source_spatial_domain == SourceSpatialDomain(
        origin_yx=(0, 0),
        source_shape_yx=(3, 8),
    )
    np.testing.assert_array_equal(output[:, :4, 0], np.ones((3, 4)))
    np.testing.assert_array_equal(output[:, 4:, 1], np.full((3, 4), 2))


def test_tile_declares_grayscale_montage_without_channel_axis() -> None:
    image = np.arange(24, dtype=np.float32).reshape((2, 3, 4))

    output = tile(image, rows=1, columns=2, dtype_config=DtypeConfig())

    assert output.shape == (3, 8)
    assert image_payload_metadata(output).source_channel_axis is None
    assert image_payload_metadata(output).source_spatial_domain == SourceSpatialDomain(
        origin_yx=(0, 0),
        source_shape_yx=(3, 8),
    )


def test_tile_aligned_multi_image_stack_tiles_each_runtime_slice() -> None:
    callable_contract = _compiled_callable_contract(
        tile, artifact_outputs=(ArtifactSpec.output("Tiled", ImageArtifactType),)
    )
    aligned_stack = AlignedImageStack(
        slices=(
            np.stack(
                (
                    np.full((3, 4), 1, dtype=np.float32),
                    np.full((3, 4), 2, dtype=np.float32),
                )
            ),
            np.stack(
                (
                    np.full((3, 4), 3, dtype=np.float32),
                    np.full((3, 4), 4, dtype=np.float32),
                )
            ),
        )
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        tile,
        aligned_stack,
        {"rows": 1, "columns": 2},
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert result.shape == (2, 3, 8)
    np.testing.assert_array_equal(result[0, :, :4], np.full((3, 4), 1))
    np.testing.assert_array_equal(result[0, :, 4:], np.full((3, 4), 2))
    np.testing.assert_array_equal(result[1, :, :4], np.full((3, 4), 3))
    np.testing.assert_array_equal(result[1, :, 4:], np.full((3, 4), 4))


def test_cellprofiler_contract_executor_applies_aligned_multi_image_stack():
    calls = []

    def subtract_illumination(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image[0] - image[1]

    subtract_illumination.__processing_contract__ = ProcessingContract.PURE_2D
    raw_contract = CallableContract.from_callable(subtract_illumination)
    callable_contract = replace(
        raw_contract,
        metadata=replace(
            raw_contract.metadata,
            artifact_outputs=(ArtifactSpec.output("Corrected", ImageArtifactType),),
        ),
    )
    aligned_stack = AlignedImageStack(
        slices=(
            np.stack(
                (
                    np.full((4, 5), 11, dtype=np.float32),
                    np.full((4, 5), 3, dtype=np.float32),
                )
            ),
            np.stack(
                (
                    np.full((4, 5), 22, dtype=np.float32),
                    np.full((4, 5), 3, dtype=np.float32),
                )
            ),
        )
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        subtract_illumination,
        aligned_stack,
        {},
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert calls == [(2, 4, 5), (2, 4, 5)]
    assert result.shape == (2, 4, 5)
    np.testing.assert_array_equal(result[0], np.full((4, 5), 8, dtype=np.float32))
    np.testing.assert_array_equal(result[1], np.full((4, 5), 19, dtype=np.float32))


def test_aligned_multi_image_stack_rejects_volumetric_contract() -> None:
    def keep_volume(image: np.ndarray) -> np.ndarray:
        return image

    keep_volume.__processing_contract__ = ProcessingContract.PURE_3D
    raw_contract = CallableContract.from_callable(keep_volume)
    callable_contract = replace(
        raw_contract,
        metadata=replace(
            raw_contract.metadata,
            artifact_outputs=(ArtifactSpec.output("Image", ImageArtifactType),),
        ),
    )
    aligned_stack = AlignedImageStack(
        slices=(
            np.zeros((2, 4, 5), dtype=np.float32),
            np.zeros((2, 4, 5), dtype=np.float32),
        )
    )

    with pytest.raises(ValueError, match="ProcessingContract.PURE_3D"):
        CellProfilerFunctionContractExecutor().execute(
            callable_contract,
            keep_volume,
            aligned_stack,
            {},
            execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
            plane_projection=RuntimePlaneAxisValueProjection.preserve(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                axis_size=2,
            ),
        )


def test_image_measurement_table_retains_exact_declared_current_image() -> None:
    measurement_spec = ArtifactSpec.output(
        "ImageQuality",
        MeasurementsArtifactType,
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("MeasureImageQuality").require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("OrigBlue", ImageArtifactType),
                ArtifactSpec.input("OrigGreen", ImageArtifactType),
            ),
            artifact_outputs=(measurement_spec,),
        )
    )
    request = _cellprofiler_output_record_request(
        callable_contract=executor.callable_contract,
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec, stored=False)
            for spec in executor.callable_contract.artifact_inputs
        ),
        output_plans=tuple(
            _artifact_output_plan(item) for item in ((measurement_spec,))
        ),
        adapter=None,
        spec=measurement_spec,
        output_value=MeasurementSparseColumnarRows.from_rows((), fields=()),
        source_image_name="OrigGreen",
        source_image_payload=np.zeros((4, 5), dtype=np.float32),
        call_kwargs={},
    )

    module_type = _module_type_for_contract(executor.callable_contract)
    rows = MeasurementSparseColumnarRows.from_rows((), fields=())
    source_image_name = module_type.measurement_record_source_image_name(
        request,
        rows,
    )

    assert source_image_name == "OrigGreen"


def test_output_object_threshold_rows_override_relationship_source_retention() -> None:
    assert IdentifySecondaryObjectsModule.clear_source_when_rows_declare_object_name()


def test_colocalization_object_row_policy_projects_source_pair_features() -> None:
    policy = MeasureColocalizationModule.runtime_object_measurement_row_policy()
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="DNA__ER__RNA",
        source_aliases=("DNA", "ER", "RNA"),
        payload=np.zeros((3, 4, 5), dtype=np.float32),
    )

    invocations = policy.invocations(measurement_image, {"do_manders": True})

    assert [invocation.kwargs["channel_1"] for invocation in invocations] == [0, 0, 1]
    assert [invocation.kwargs["channel_2"] for invocation in invocations] == [1, 2, 2]
    assert [invocation.source_pair.first.name for invocation in invocations] == [
        "DNA",
        "DNA",
        "ER",
    ]
    raw_fields = FieldSpec.from_dataclass_type(ObjectColocalizationMeasurements)
    raw_row = {field_spec.name: 0.0 for field_spec in raw_fields}
    raw_row.update(
        slice_index=0,
        object_label=1,
        correlation=0.5,
        overlap=0.6,
        k1=0.1,
        k2=0.2,
        manders_m1=0.7,
        manders_m2=0.8,
        rwc1=0.3,
        rwc2=0.4,
        costes_m1=0.9,
        costes_m2=0.95,
        costes_threshold_1=42.0,
        costes_threshold_2=84.0,
    )
    projected = policy.project_rows(
        MeasurementSparseColumnarRows.from_rows(
            (raw_row,),
            fields=raw_fields,
            object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
        ),
        invocations[0],
    )

    (projected_row,) = projected.row_mappings()
    assert projected_row["Correlation_Correlation_DNA_ER"] == 0.5
    assert projected_row["Correlation_Manders_DNA_ER"] == 0.7
    assert projected_row["Correlation_Manders_ER_DNA"] == 0.8
    assert "Correlation_Slope_DNA_ER" not in projected_row
    with pytest.raises(ValueError, match="exact raw colocalization fields"):
        policy.project_rows(projected, invocations[0])
    assert {
        feature.measurement_row_field_name: feature.source_pair_feature_name(
            invocations[0].source_pair
        )
        for feature in MeasureColocalizationModule.MeasurementFeature
    } == {
        "correlation": "Correlation_Correlation_DNA_ER",
        "slope": "Correlation_Slope_DNA_ER",
        "overlap": "Correlation_Overlap_DNA_ER",
        "k1": "Correlation_K_DNA_ER",
        "k2": "Correlation_K_ER_DNA",
        "manders_m1": "Correlation_Manders_DNA_ER",
        "manders_m2": "Correlation_Manders_ER_DNA",
        "rwc1": "Correlation_RWC_DNA_ER",
        "rwc2": "Correlation_RWC_ER_DNA",
        "costes_m1": "Correlation_Costes_DNA_ER",
        "costes_m2": "Correlation_Costes_ER_DNA",
    }
    assert policy.table_source_image_name((measurement_image,), "DNA__ER__RNA") is None


def test_colocalization_record_builder_derives_source_pair_table_identity() -> None:
    def measure_colocalization(
        image: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        return image, {"correlation": 0.5}

    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module(
                "MeasureColocalization"
            ).require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("DNA", ImageArtifactType),
                ArtifactSpec.input("ER", ImageArtifactType),
            ),
            artifact_outputs=(ArtifactSpec.output("Coloc", MeasurementsArtifactType),),
        )
    )

    table = measurement_table_for_module(
        _cellprofiler_output_record_request(
            callable_contract=executor.callable_contract,
            artifact_input_edges=tuple(
                _artifact_input_edge_for_test(spec)
                for spec in executor.callable_contract.artifact_inputs.specs
            ),
            output_plans=tuple(
                _artifact_output_plan(item)
                for item in ((ArtifactSpec.output("Coloc", MeasurementsArtifactType),))
            ),
            adapter=None,
            spec=ArtifactSpec.output("Coloc", MeasurementsArtifactType),
            output_value=MeasurementSparseColumnarRows.from_rows(
                ({"slice_index": 0, "correlation": 0.5, "manders_m1": 0.7},),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("correlation", float),
                    FieldSpec("manders_m1", float),
                ),
            ),
            call_kwargs={},
            source_aliases=("DNA", "ER"),
        )
    )

    assert table.rows.row_mappings() == (
        {
            "slice_index": 0,
            "correlation": 0.5,
            "manders_m1": 0.7,
        },
    )
    assert (
        table.source_image_name == RuntimeMeasurementSourcePair("DNA", "ER").source_name
    )


def test_measure_object_neighbors_records_object_topology_without_image_source() -> (
    None
):
    def measure_neighbors(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        return image, {"number_of_neighbors": 1.0}

    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module(
                "MeasureObjectNeighbors"
            ).require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
            ),
            artifact_outputs=(
                ArtifactSpec.output("Neighbors", MeasurementsArtifactType),
            ),
        )
    )

    nuclei = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=np.asarray([[1]], dtype=np.int32)),
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
    )
    table = measurement_table_for_module(
        _cellprofiler_output_record_request(
            callable_contract=executor.callable_contract,
            artifact_input_edges=tuple(
                _artifact_input_edge_for_test(spec)
                for spec in executor.callable_contract.artifact_inputs.specs
            ),
            output_plans=tuple(
                _artifact_output_plan(item)
                for item in (
                    (ArtifactSpec.output("Neighbors", MeasurementsArtifactType),)
                )
            ),
            adapter=_FakeCellProfilerRuntime({}, objects={"Nuclei": nuclei}),
            spec=ArtifactSpec.output("Neighbors", MeasurementsArtifactType),
            output_value=DataclassMeasurementColumnarRows(
                (
                    NeighborMeasurements(
                        slice_index=0,
                        object_id=1,
                        scale=4,
                        number_of_neighbors=1,
                        percent_touching=25.0,
                        first_closest_object_number=2,
                        first_closest_distance=3.0,
                        second_closest_object_number=3,
                        second_closest_distance=4.0,
                        angle_between_neighbors=90.0,
                    ),
                ),
                row_type=NeighborMeasurements,
            ),
            source_image_name="OrigBlue",
            call_kwargs={
                "distance_method": DistanceMethod.WITHIN,
                "neighbor_distance": 4,
            },
        )
    )

    rows = table.rows.row_mappings()
    assert table.subject == MeasurementSubject(MeasurementScope.OBJECT, "Nuclei")
    assert table.source_image_name is None
    assert set(rows[0]) == {
        "slice_index",
        "object_id",
        "Neighbors_NumberOfNeighbors_4",
        "Neighbors_PercentTouching_4",
        "Neighbors_FirstClosestObjectNumber_4",
        "Neighbors_FirstClosestDistance_4",
        "Neighbors_SecondClosestObjectNumber_4",
        "Neighbors_SecondClosestDistance_4",
        "Neighbors_AngleBetweenNeighbors_4",
    }


def test_track_objects_record_builder_uses_nominal_image_table_ownership() -> None:
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("TrackObjects").require_callable(),
            artifact_inputs=(ArtifactSpec.input("Embryos", ObjectLabelsArtifactType),),
            artifact_outputs=(
                ArtifactSpec.output("Tracking", MeasurementsArtifactType),
            ),
        )
    )

    embryos = ObjectLabelSet(
        name="Embryos",
        variant_data=ObjectLabelVariantData(labels=np.asarray([[1]], dtype=np.int32)),
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
    )
    object_record = TrackingObjectMeasurement(
        slice_index=0,
        object_label=1,
        scale=50,
        displacement=0.0,
        distance_traveled=0.0,
        final_age=1,
        integrated_distance=0.0,
        label=1,
        lifetime=1,
        linearity=float("nan"),
        parent_image_number=0,
        parent_object_number=0,
        trajectory_x=0.0,
        trajectory_y=0.0,
    )
    image_record = TrackingImageMeasurement(
        slice_index=0,
        scale=50,
        new_object_count=1,
        lost_object_count=0,
        split_object_count=0,
        merged_object_count=0,
    )
    table = measurement_table_for_module(
        _cellprofiler_output_record_request(
            callable_contract=executor.callable_contract,
            artifact_input_edges=tuple(
                _artifact_input_edge_for_test(spec)
                for spec in executor.callable_contract.artifact_inputs.specs
            ),
            output_plans=tuple(
                _artifact_output_plan(item)
                for item in (
                    (ArtifactSpec.output("Tracking", MeasurementsArtifactType),)
                )
            ),
            adapter=_FakeCellProfilerRuntime({}, objects={"Embryos": embryos}),
            spec=ArtifactSpec.output("Tracking", MeasurementsArtifactType),
            output_value=ConcatenatedColumnarRows(
                (
                    DataclassMeasurementColumnarRows(
                        (object_record,),
                        row_type=TrackingObjectMeasurement,
                    ),
                    DataclassMeasurementColumnarRows(
                        (image_record,),
                        row_type=TrackingImageMeasurement,
                    ),
                )
            ),
            source_image_name="OrigColor",
            call_kwargs={"pixel_radius": 50},
        )
    )

    object_feature_names = {
        TrackObjectsModule.measurement_feature_name(field_name, 50)
        for field_name in TrackObjectsModule.measurement_record_field_values(
            object_record
        )
    }
    image_feature_names = {
        TrackObjectsModule.measurement_feature_name(field_name, "Embryos", 50)
        for field_name in TrackObjectsModule.measurement_record_field_values(
            image_record
        )
    }
    mean_feature_names = {
        TrackObjectsModule.mean_measurement_feature_name("Embryos", feature_name)
        for feature_name in object_feature_names
    }
    rows_by_feature = {
        row[MeasurementRowAxisField.FEATURE_NAME.value]: row
        for row in table.rows.iter_row_mappings()
    }

    assert set(rows_by_feature) == (
        object_feature_names | image_feature_names | mean_feature_names
    )
    assert all(
        MeasurementRowAxisField.SCALE.value not in row
        for row in rows_by_feature.values()
    )
    assert all(
        rows_by_feature[feature_name]["object_name"] == "Embryos"
        and "source_image_name" not in rows_by_feature[feature_name]
        for feature_name in object_feature_names
    )
    assert all(
        rows_by_feature[feature_name]["source_image_name"]
        == MeasurementScope.IMAGE.value
        and "object_name" not in rows_by_feature[feature_name]
        for feature_name in image_feature_names | mean_feature_names
    )
    assert table.source_image_name is None
    assert table.subject == MeasurementSubject(
        MeasurementScope.IMAGE,
        MeasurementScope.IMAGE.value,
    )


def test_object_label_output_recorder_uses_output_label_domain() -> None:
    input_labels = np.zeros((5, 5), dtype=np.int32)
    input_labels[1, 1] = 1
    input_labels[2, 2] = 4
    input_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=input_labels),
        domain=ObjectLabelDomain(
            declared_object_count=9,
        ),
    )
    output_labels = input_labels.copy()
    output_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=output_labels),
        domain=ObjectLabelDomain(
            declared_object_ids=tuple(range(1, 5)),
        ),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={
            "InputObjects": ObjectLabelSet.from_payload(
                "InputObjects",
                input_payload,
            )
        },
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module(
                "ExpandOrShrinkObjects"
            ).require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("InputObjects", ObjectLabelsArtifactType),
            ),
            artifact_outputs=(_output_from_input("ExpandedObjects", "InputObjects"),),
        )
    )

    CellProfilerOutputRecorder.for_artifact_type(ObjectLabelsArtifactType).record(
        _cellprofiler_output_record_request(
            callable_contract=executor.callable_contract,
            artifact_input_edges=tuple(
                _artifact_input_edge_for_test(spec)
                for spec in executor.callable_contract.artifact_inputs.specs
            ),
            output_plans=tuple(
                _artifact_output_plan(item)
                for item in ((_output_from_input("ExpandedObjects", "InputObjects"),))
            ),
            adapter=runtime,
            spec=_output_from_input("ExpandedObjects", "InputObjects"),
            output_value=output_payload,
            source_image_name=None,
            call_kwargs={},
        )
    )

    _name, recorded_payload, _kwargs = runtime.objects[0]
    assert isinstance(recorded_payload, ObjectLabelPayload)
    assert recorded_payload.domain.declared_object_count is None
    assert recorded_payload.domain.declared_object_ids == tuple(range(1, 5))
    np.testing.assert_array_equal(recorded_payload.labels, output_labels)


def test_expand_or_shrink_executor_preserves_declared_object_domain() -> None:
    input_labels = np.zeros((7, 7), dtype=np.int32)
    input_labels[3, 3] = 4
    input_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=input_labels),
        domain=ObjectLabelDomain(
            declared_object_count=9,
        ),
    )
    input_spec = ArtifactSpec.input(
        "InputObjects",
        ObjectLabelsArtifactType,
        parameter_name="labels",
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={
            "InputObjects": ObjectLabelSet.from_payload(
                "InputObjects",
                input_payload,
            )
        },
        artifact_input_edges=(_artifact_input_edge_for_test(input_spec),),
    )
    output_spec = _output_from_input("ExpandedObjects", "InputObjects")
    measurement_spec = ArtifactSpec.output(
        "ExpandOrShrinkObjects_measurements",
        MeasurementsArtifactType,
        relations=(ArtifactSpecRelation(source=output_spec.ref()),),
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module(
                "ExpandOrShrinkObjects"
            ).require_callable(),
            artifact_inputs=(input_spec,),
            artifact_outputs=(measurement_spec, output_spec),
        ),
    )
    runtime.request = replace(
        runtime.request,
        artifact_outputs={
            spec.ref(): _artifact_output_plan(spec)
            for spec in executor.callable_contract.artifact_outputs
        },
    )

    _run_module(
        executor,
        np.zeros_like(input_labels, dtype=np.float32),
        cellprofiler_runtime=runtime,
        mode="expand_defined_pixels",
        iterations=1,
        dtype_config=DtypeConfig(),
    )

    _name, recorded_payload, _kwargs = runtime.objects[0]
    assert isinstance(recorded_payload, ObjectLabelSet)
    assert recorded_payload.domain.declared_object_count == 9
    assert recorded_payload.domain.declared_object_ids == ()
    assert int(np.max(recorded_payload.labels)) == 4


def _seed_align_output_provenance(
    request: CellProfilerOutputRecordRequest,
    component_metadata: tuple[dict[str, str], ...],
) -> None:
    image_output_specs = tuple(
        spec
        for spec in request.callable_contract.artifact_outputs
        if spec.artifact_type is ImageArtifactType
        and request.adapter.request.artifact_output_plan(spec.ref()) is not None
    )
    assert len(image_output_specs) == len(component_metadata)
    for spec, metadata in zip(image_output_specs, component_metadata, strict=True):
        request.adapter._store_runtime_artifact(
            request.adapter.request.require_artifact_output_plan(spec.ref()),
            ImagePayloadMetadata(
                source_image_names=(spec.name,),
                source_image_provenance_planes=(
                    SourceImageProvenancePlanes.from_components(
                        component_metadata=(metadata,),
                    )
                ),
            ).payload_with(np.zeros((4, 5), dtype=np.float32)),
        )


def test_align_measurement_builder_records_output_scoped_shifts() -> None:
    def align_function(
        image: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[AlignShiftMeasurement, AlignShiftMeasurement],
    ]:
        return (
            image[0],
            image[1],
            (
                AlignShiftMeasurement(0, 0, 0.0, 0.0),
                AlignShiftMeasurement(0, 1, -1.0, 1.0),
            ),
        )

    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("Align").require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("Stain1Raw", ImageArtifactType),
                ArtifactSpec.input("Stain2Raw", ImageArtifactType),
            ),
            artifact_outputs=(
                ArtifactSpec.output("Stain1", ImageArtifactType),
                ArtifactSpec.output("Stain2", ImageArtifactType),
                ArtifactSpec.output("AlignMeasurements", MeasurementsArtifactType),
            ),
        )
    )

    runtime = _FakeCellProfilerRuntime(
        {},
        source_bindings=(
            NamedSourceBinding(
                alias="Stain1Raw",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="Stain2Raw",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
            ),
        ),
    )
    request = _cellprofiler_output_record_request(
        callable_contract=executor.callable_contract,
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec)
            for spec in executor.callable_contract.artifact_inputs
        ),
        output_plans=tuple(
            _artifact_output_plan(item)
            for item in executor.callable_contract.artifact_outputs
        ),
        adapter=runtime,
        spec=ArtifactSpec.output("AlignMeasurements", MeasurementsArtifactType),
        output_value=DataclassMeasurementColumnarRows(
            (
                AlignShiftMeasurement(0, 0, 0.0, 0.0),
                AlignShiftMeasurement(0, 1, -1.0, 1.0),
            ),
            row_type=AlignShiftMeasurement,
        ),
        source_image_name="Stain1Raw__Stain2Raw",
        call_kwargs={},
    )
    _seed_align_output_provenance(
        request,
        (
            {"well": "A01", "site": "1", "channel": "1"},
            {"well": "A01", "site": "1", "channel": "2"},
        ),
    )
    table = measurement_table_for_module(request)

    assert table.subject == MeasurementSubject(MeasurementScope.IMAGE, "image")
    assert table.source_image_name is None
    assert dict(table.source_component_metadata or {}) == {
        "well": "A01",
        "site": "1",
    }
    assert table.rows.row_mappings() == (
        {
            "slice_index": 0,
            "source_image_name": "Stain1",
            AlignModule.measurement_feature_name("x_shift", "Stain1"): 0,
            AlignModule.measurement_feature_name("y_shift", "Stain1"): 0,
        },
        {
            "slice_index": 0,
            "source_image_name": "Stain2",
            AlignModule.measurement_feature_name("x_shift", "Stain2"): -1,
            AlignModule.measurement_feature_name("y_shift", "Stain2"): 1,
        },
    )


def test_align_measurement_builder_records_additional_output_shifts() -> None:
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("Align").require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("Template", ImageArtifactType),
                ArtifactSpec.input("Red", ImageArtifactType),
                ArtifactSpec.input("Combined", ImageArtifactType),
            ),
            artifact_outputs=(
                ArtifactSpec.output("AlignedTemplate", ImageArtifactType),
                ArtifactSpec.output("AlignedRed", ImageArtifactType),
                ArtifactSpec.output("AlignedCombined", ImageArtifactType),
                ArtifactSpec.output("AlignMeasurements", MeasurementsArtifactType),
            ),
        )
    )

    request = _cellprofiler_output_record_request(
        callable_contract=executor.callable_contract,
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec)
            for spec in executor.callable_contract.artifact_inputs.specs
        ),
        output_plans=tuple(
            _artifact_output_plan(item)
            for item in executor.callable_contract.artifact_outputs
        ),
        adapter=None,
        spec=ArtifactSpec.output("AlignMeasurements", MeasurementsArtifactType),
        output_value=DataclassMeasurementColumnarRows(
            (
                AlignShiftMeasurement(0, 0, 0.0, 0.0),
                AlignShiftMeasurement(0, 1, -2.0, 1.0),
                AlignShiftMeasurement(0, 2, -2.0, 1.0),
            ),
            row_type=AlignShiftMeasurement,
        ),
        source_image_name=None,
        call_kwargs={},
    )
    _seed_align_output_provenance(
        request,
        (
            {"well": "A01", "site": "1"},
            {"well": "A01", "site": "1"},
            {"well": "A01", "site": "1"},
        ),
    )
    table = measurement_table_for_module(request)

    assert table.rows.row_mappings()[-1] == {
        "slice_index": 0,
        "source_image_name": "AlignedCombined",
        AlignModule.measurement_feature_name("x_shift", "AlignedCombined"): -2,
        AlignModule.measurement_feature_name("y_shift", "AlignedCombined"): 1,
    }


def test_classification_rows_include_unclassified_objects() -> None:
    def classify_like(image: np.ndarray) -> tuple[np.ndarray, ClassificationResult]:
        return image, ClassificationResult(
            total_objects=3,
            bin_counts='{"Small": 1, "Large": 1}',
            bin_percentages='{"Small": 33.3333333333, "Large": 33.3333333333}',
            object_classes='{"1": "Small", "3": "Large"}',
        )

    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module(
                "ClassifyObjectsSingleMeasurement"
            ).require_callable(),
            artifact_inputs=(ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
            artifact_outputs=(
                ArtifactSpec.output("ClassifyObjects", MeasurementsArtifactType),
            ),
        )
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={
            "Nuclei": ObjectLabelSet(
                name="Nuclei",
                variant_data=ObjectLabelVariantData(
                    labels=np.array([[[1, 2], [3, 0]]], dtype=np.int32)
                ),
                domain=ObjectLabelDomain(
                    scope=ObjectLabelDomainScope.PLANE,
                    declared_object_id_domains=((1, 2, 3),),
                ),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                source_image_provenance_planes=(
                    SourceImageProvenancePlanes.from_components(
                        paths=("/input/Nuclei.tif",),
                    )
                ),
            )
        },
    )

    rows = measurement_table_for_module(
        _cellprofiler_output_record_request(
            callable_contract=executor.callable_contract,
            artifact_input_edges=tuple(
                _artifact_input_edge_for_test(spec)
                for spec in executor.callable_contract.artifact_inputs.specs
            ),
            output_plans=tuple(
                _artifact_output_plan(item)
                for item in (
                    (ArtifactSpec.output("ClassifyObjects", MeasurementsArtifactType),)
                )
            ),
            adapter=runtime,
            spec=ArtifactSpec.output("ClassifyObjects", MeasurementsArtifactType),
            output_value=ClassificationResult.columnar(
                classify_like(np.zeros((2, 2), dtype=np.float32))[1]
            ),
            source_image_name=None,
            call_kwargs={},
        )
    ).rows.row_mappings()

    object_rows = [row for row in rows if row.get("object_name") == "Nuclei"]
    assert len(object_rows) == 3
    small_feature = ClassifyObjectsSingleMeasurementModule.MeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
        bin_name="Small"
    )
    large_feature = ClassifyObjectsSingleMeasurementModule.MeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
        bin_name="Large"
    )
    assert {
        (row["object_label"], row[small_feature], row[large_feature])
        for row in object_rows
    } == {
        (1, 1, 0),
        (2, 0, 0),
        (3, 0, 1),
    }


def _assert_untangle_measurement_output(
    runtime: _FakeCellProfilerRuntime,
    *,
    primary_rows: list[dict[str, object]],
) -> None:
    assert len(runtime.measurements) == 1
    table = runtime.measurements[0]
    rows = [dict(row) for row in table.rows.iter_row_mappings()]
    assert table.name == "UntangleWorms_3_measurements"
    assert rows[: len(primary_rows)] == primary_rows
    assert rows[len(primary_rows) :] == [
        {
            "Count_OverlappingWorms": 1,
        },
        {
            "object_name": "OverlappingWorms",
            "object_label": 2,
            "slice_index": 0,
            "feature_name": "Location_Center_X",
            "result_value": 2.0,
        },
        {
            "object_name": "OverlappingWorms",
            "object_label": 2,
            "slice_index": 0,
            "feature_name": "Location_Center_Y",
            "result_value": 1.5,
        },
        {
            "Count_NonOverlappingWorms": 1,
        },
        {
            "object_name": "NonOverlappingWorms",
            "object_label": 1,
            "slice_index": 0,
            "feature_name": "Location_Center_X",
            "result_value": 2.0,
        },
        {
            "object_name": "NonOverlappingWorms",
            "object_label": 1,
            "slice_index": 0,
            "feature_name": "Location_Center_Y",
            "result_value": 1.5,
        },
    ]


@pytest.mark.parametrize(
    "measured_object_names",
    (
        ("OverlappingWorms",),
        ("NonOverlappingWorms",),
        ("OverlappingWorms", "NonOverlappingWorms"),
    ),
)
def test_untangle_measurement_object_names_use_compiled_relations(
    measured_object_names: tuple[str, ...],
) -> None:
    overlapping = ArtifactSpec.output(
        "OverlappingWorms",
        ObjectLabelsArtifactType,
    )
    nonoverlapping = ArtifactSpec.output(
        "NonOverlappingWorms",
        ObjectLabelsArtifactType,
    )
    measurements = ArtifactSpec.output(
        "UntangleWorms_measurements",
        MeasurementsArtifactType,
        relations=tuple(
            ArtifactSpecRelation(
                source={
                    overlapping.name: overlapping,
                    nonoverlapping.name: nonoverlapping,
                }[object_name].ref()
            )
            for object_name in measured_object_names
        ),
    )
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("UntangleWorms").require_callable(),
            artifact_inputs=(),
            artifact_outputs=(overlapping, nonoverlapping, measurements),
        ),
        output_plans=tuple(
            _artifact_output_plan(item)
            for item in ((overlapping, nonoverlapping, measurements))
        ),
        adapter=None,
        spec=measurements,
        output_value=MeasurementTable(
            name=measurements.name,
            rows=MeasurementSparseColumnarRows.from_rows((), fields=()),
            subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        ),
        call_kwargs={},
    )

    assert (
        tuple(
            spec.name
            for spec in UntangleWormsModule.measurement_object_output_specs_for_request(
                request
            )
        )
        == measured_object_names
    )
    assert "measurement_object_names" not in vars(UntangleWormsModule)


def test_untangle_object_measurements_do_not_inherit_input_image_identity() -> None:
    source_spec = ArtifactSpec.input("WormObjectsBinary", ImageArtifactType)
    measurement_spec = ArtifactSpec.output(
        "UntangleWorms_measurements",
        MeasurementsArtifactType,
    )
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("UntangleWorms").require_callable(),
            artifact_inputs=(source_spec,),
            artifact_outputs=(
                ArtifactSpec.output("OverlappingWorms", ObjectLabelsArtifactType),
                ArtifactSpec.output("NonOverlappingWorms", ObjectLabelsArtifactType),
                measurement_spec,
            ),
        ),
        artifact_input_edges=(_artifact_input_edge_for_test(source_spec),),
        output_plans=tuple(
            _artifact_output_plan(item)
            for item in (
                (
                    ArtifactSpec.output("OverlappingWorms", ObjectLabelsArtifactType),
                    ArtifactSpec.output(
                        "NonOverlappingWorms", ObjectLabelsArtifactType
                    ),
                    measurement_spec,
                )
            )
        ),
        adapter=None,
        spec=measurement_spec,
        output_value=MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "object_number": 1,
                    "feature_name": "worm_length",
                    "result_value": 12.5,
                },
            ),
            fields=(
                FieldSpec("object_number", int),
                FieldSpec("feature_name", str),
                FieldSpec("result_value", float),
            ),
        ),
        source_image_name="WormObjectsBinary",
        source_image_payload=np.zeros((8, 8), dtype=np.uint8),
        call_kwargs={"overlap_style": OverlapStyle.BOTH},
    )

    source_image_name = UntangleWormsModule.measurement_record_source_image_name(
        request,
        request.output_value,
    )

    assert source_image_name is None


def test_object_only_measurement_carrier_preserves_payload_stack_context() -> None:
    payload = ImagePayloadMetadata(
        source_dtype="float32",
        source_path="/inputs/site-1.tif",
    ).payload_with(np.zeros((4, 6, 6), dtype=np.float32), None)

    carrier = CellProfilerMeasurementImage(
        source_image_name=None,
        source_aliases=(),
        payload=payload,
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    assert carrier.payload is payload
    assert image_payload_data(carrier.payload).shape == (4, 6, 6)
    assert image_payload_metadata(carrier.payload).source_dtype == "float32"
    assert image_payload_metadata(carrier.payload).source_path == "/inputs/site-1.tif"


def test_object_only_measurement_carrier_preserves_aligned_stack() -> None:
    payload = AlignedImageStack(
        (
            ImagePayloadMetadata(source_dtype="float32").payload_with(
                np.zeros((6, 6), dtype=np.float32), None
            ),
            ImagePayloadMetadata(source_dtype="float32").payload_with(
                np.ones((6, 6), dtype=np.float32), None
            ),
        )
    )

    carrier = CellProfilerMeasurementImage(
        source_image_name=None,
        source_aliases=(),
        payload=payload,
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    assert carrier.payload is payload
    assert isinstance(carrier.payload, AlignedImageStack)
    assert len(carrier.payload.slices) == 2


def test_filterobjects_binds_selection_measurement_values_to_label_slices() -> None:
    image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.zeros((2, 6, 6), dtype=np.float32), None)
    children = np.zeros((2, 6, 6), dtype=np.int32)
    children[:, 0:2, 0:2] = 1
    children[:, 3:5, 3:5] = 2
    parents = np.ones_like(children)
    measurements = MeasurementTable(
        name="ChildMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            [
                {"slice_index": 0, "object_label": 1, "AreaShape_Area": 10.0},
                {"slice_index": 0, "object_label": 2, "AreaShape_Area": 20.0},
                {"slice_index": 1, "object_label": 1, "AreaShape_Area": 30.0},
                {"slice_index": 1, "object_label": 2, "AreaShape_Area": 5.0},
            ],
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("AreaShape_Area", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells", "object_label"),
    )
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": image},
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                variant_data=ObjectLabelVariantData(labels=children),
                source_image_provenance_planes=(
                    SourceImageProvenancePlanes.from_components(
                        paths=("/input/site-1.tif", "/input/site-2.tif"),
                        component_metadata=(
                            {"site": "1"},
                            {"site": "2"},
                        ),
                    )
                ),
                domain=ObjectLabelDomain(
                    scope=ObjectLabelDomainScope.PLANE,
                    declared_object_id_domains=((1, 2), (1, 2)),
                ),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
            "Tiles": ObjectLabelSet(
                name="Tiles",
                variant_data=ObjectLabelVariantData(labels=parents),
                source_image_provenance_planes=(
                    SourceImageProvenancePlanes.from_components(
                        paths=("/input/site-1.tif", "/input/site-2.tif"),
                        component_metadata=(
                            {"site": "1"},
                            {"site": "2"},
                        ),
                    )
                ),
                domain=ObjectLabelDomain(
                    scope=ObjectLabelDomainScope.PLANE,
                    declared_object_id_domains=((1,), (1,)),
                ),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
        },
        measurement_tables={"Cells": (measurements,)},
        plane_projection=RuntimePlaneProjection.stack(2),
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(
            frozenset((AllComponents.CHANNEL,))
        ),
    )
    runtime._store_runtime_artifact(
        ArtifactOutputPlan(
            name=measurements.name,
            path=f"/artifacts/{measurements.name}",
            artifact_type=MeasurementsArtifactType,
        ),
        measurements,
    )
    cells_spec = ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    tiles_spec = ArtifactSpec.input("Tiles", ObjectLabelsArtifactType)
    measurement_input_spec = ArtifactSpec.input(
        measurements.name,
        MeasurementsArtifactType,
    )
    runtime.install_artifact_input_edges(
        tuple(
            _artifact_input_edge_for_test(spec)
            for spec in (cells_spec, tiles_spec, measurement_input_spec)
        )
    )
    filtered_cells_spec = ArtifactSpec.output_inheriting_group_scope(
        "FilteredCells",
        ObjectLabelsArtifactType,
        cells_spec,
    )
    filter_relationship_spec = _relationship_output(
        cells_spec,
        filtered_cells_spec,
    )
    filter_measurements_spec = ArtifactSpec.output(
        "FilterObjects_measurements",
        MeasurementsArtifactType,
        relations=tuple(
            ArtifactSpecRelation(source=spec.ref())
            for spec in (filtered_cells_spec, filter_relationship_spec)
        ),
    )
    output_specs = (
        filter_measurements_spec,
        filtered_cells_spec,
        filter_relationship_spec,
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("FilterObjects").require_callable(),
            artifact_inputs=(cells_spec, tiles_spec, measurement_input_spec),
            artifact_outputs=output_specs,
        ),
    )
    runtime.request = replace(
        runtime.request,
        artifact_outputs={
            spec.ref(): _artifact_output_plan(spec)
            for spec in executor.callable_contract.artifact_outputs
        },
    )

    _run_module(
        executor,
        image,
        cellprofiler_runtime=runtime,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
        measurement_features=("AreaShape_Area",),
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        slice_by_slice=True,
    )

    filtered = next(
        value for name, value, _kwargs in runtime.objects if name == "FilteredCells"
    )
    filtered_array = object_label_dense_array(filtered)
    assert filtered_array.shape == (2, 6, 6)
    assert filtered_array[0, 0, 0] == 0
    assert filtered_array[0, 3, 3] == 1
    assert filtered_array[1, 0, 0] == 1
    assert filtered_array[1, 3, 3] == 0


def test_relationship_measurements_preserve_pure_2d_slice_indices() -> None:
    parent_labels = np.zeros((2, 5, 5), dtype=np.int32)
    child_labels = np.zeros((2, 5, 5), dtype=np.int32)
    parent_labels[0, 1:3, 1:3] = 1
    child_labels[0, 1:3, 1:3] = 1
    parent_labels[1, 2:4, 2:4] = 2
    child_labels[1, 2:4, 2:4] = 2
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": np.zeros((2, 5, 5), dtype=np.float32)},
        objects={
            "Parents": ObjectLabelSet(
                name="Parents",
                variant_data=ObjectLabelVariantData(labels=parent_labels),
                domain=ObjectLabelDomain(
                    scope=ObjectLabelDomainScope.PLANE,
                    declared_object_id_domains=((1,), (2,)),
                ),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
            "Children": ObjectLabelSet(
                name="Children",
                variant_data=ObjectLabelVariantData(labels=child_labels),
                domain=ObjectLabelDomain(
                    scope=ObjectLabelDomainScope.PLANE,
                    declared_object_id_domains=((1,), (2,)),
                ),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
        },
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("RelateObjects").require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                ArtifactSpec.input("Children", ObjectLabelsArtifactType),
            ),
            artifact_outputs=(
                _relationship_output(
                    ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                    ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                ),
            ),
        )
    )
    payload = DirectedObjectRelationshipPayload(
        source_ids=(1, 2),
        target_ids=(1, 2),
        slice_indices=(0, 1),
        slice_count=2,
    )
    request = _cellprofiler_output_record_request(
        callable_contract=executor.callable_contract,
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec)
            for spec in executor.callable_contract.artifact_inputs
        ),
        output_plans=tuple(
            _artifact_output_plan(item)
            for item in ((executor.callable_contract.artifact_outputs.specs[0],))
        ),
        adapter=runtime,
        spec=executor.callable_contract.artifact_outputs.specs[0],
        output_value=payload,
        source_image_name=None,
        call_kwargs={},
    )

    _record_output(request, request.spec, payload)
    rows = RelationshipMeasurementRows.for_request(request).rows().row_mappings()
    slice_indices = {int(row["slice_index"]) for row in rows if "slice_index" in row}

    assert slice_indices == {0, 1}
    assert all(int(row["slice_index"]) in {0, 1} for row in rows)


def test_relationship_rows_use_exact_compiled_endpoint_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_ref_reconstruction(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("relationship endpoints must retain exact input edges")

    monkeypatch.setattr(
        RuntimeInputBindingRequest,
        "artifact_request_for_spec",
        reject_ref_reconstruction,
    )
    parent_labels = np.zeros((2, 5, 5), dtype=np.int32)
    child_labels = np.zeros((2, 5, 5), dtype=np.int32)
    parent_labels[0, 1:3, 1:3] = 1
    child_labels[0, 1:3, 1:3] = 1
    parent_labels[1, 2:4, 2:4] = 2
    child_labels[1, 2:4, 2:4] = 2
    parent_spec = ArtifactSpec.input(
        "Parents",
        ObjectLabelsArtifactType,
        parameter_name="parent_labels",
    )
    child_spec = ArtifactSpec.input(
        "Children",
        ObjectLabelsArtifactType,
        parameter_name="child_labels",
    )
    relationship_spec = _relationship_output(parent_spec, child_spec)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("RelateObjects").require_callable(),
        artifact_inputs=(parent_spec, child_spec),
        artifact_outputs=(relationship_spec,),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={
            "Parents": ObjectLabelSet(
                name="Parents",
                variant_data=ObjectLabelVariantData(labels=parent_labels),
                domain=ObjectLabelDomain(
                    scope=ObjectLabelDomainScope.PLANE,
                    declared_object_id_domains=((1,), (2,)),
                ),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
            "Children": ObjectLabelSet(
                name="Children",
                variant_data=ObjectLabelVariantData(labels=child_labels),
                domain=ObjectLabelDomain(
                    scope=ObjectLabelDomainScope.PLANE,
                    declared_object_id_domains=((1,), (2,)),
                ),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
        },
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec) for spec in (parent_spec, child_spec)
        ),
        plane_projection=RuntimePlaneProjection.selected(1, 2),
    )
    payload = DirectedObjectRelationshipPayload(source_ids=(2,), target_ids=(2,))
    request = _cellprofiler_output_record_request(
        callable_contract=contract,
        output_plans=(_artifact_output_plan(relationship_spec),),
        adapter=runtime,
        spec=relationship_spec,
        output_value=payload,
        source_image_name=None,
        call_kwargs={},
    )

    _record_output(request, relationship_spec, payload)
    rows = RelationshipMeasurementRows.for_request(request).rows().row_mappings()

    assert any(
        row.get("object_name") == "Parents"
        and row.get("object_label") == 1
        and row.get("Children_Children_Count") == 1
        for row in rows
    )


def test_output_record_request_rejects_ref_equivalent_endpoint_reconstruction() -> (
    None
):
    endpoint_spec = ArtifactSpec.input(
        "Objects",
        ObjectLabelsArtifactType,
        parameter_name="labels",
    )
    relationship_spec = _relationship_output(endpoint_spec, endpoint_spec)
    contract = CallableContract(
        func=lambda image: image,
        function_name="relationship_edge_probe",
        module_name=__name__,
        metadata=CallableMetadata(
            artifact_inputs=(endpoint_spec,),
            artifact_outputs=(relationship_spec,),
            runtime_adapter=CellProfilerRuntimeAdapter.runtime_adapter_spec(),
        ),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        artifact_input_edges=(
            _artifact_input_edge_for_test(endpoint_spec, stored=False),
        ),
    )
    request = _cellprofiler_output_record_request(
        callable_contract=contract,
        output_plans=(_artifact_output_plan(relationship_spec),),
        adapter=runtime,
        spec=relationship_spec,
        output_value=DirectedObjectRelationshipPayload(source_ids=(), target_ids=()),
        source_image_name=None,
        call_kwargs={},
    )
    ref_equivalent_spec = replace(endpoint_spec)

    assert ref_equivalent_spec == endpoint_spec
    assert ref_equivalent_spec is not endpoint_spec
    with pytest.raises(RuntimeError, match="has no exact compiled input edge"):
        request.exact_input_edge(ref_equivalent_spec)


def test_output_record_request_rejects_ambiguous_exact_endpoint_occurrence() -> None:
    endpoint_spec = ArtifactSpec.input(
        "Objects",
        ObjectLabelsArtifactType,
        parameter_name="labels",
    )
    relationship_spec = _relationship_output(endpoint_spec, endpoint_spec)
    contract = CallableContract(
        func=lambda image: image,
        function_name="relationship_edge_probe",
        module_name=__name__,
        metadata=CallableMetadata(
            artifact_inputs=(endpoint_spec, endpoint_spec),
            artifact_outputs=(relationship_spec,),
            runtime_adapter=CellProfilerRuntimeAdapter.runtime_adapter_spec(),
        ),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        artifact_input_edges=(
            _artifact_input_edge_for_test(endpoint_spec, stored=False),
            _artifact_input_edge_for_test(endpoint_spec, stored=False),
        ),
    )
    request = _cellprofiler_output_record_request(
        callable_contract=contract,
        output_plans=(_artifact_output_plan(relationship_spec),),
        adapter=runtime,
        spec=relationship_spec,
        output_value=DirectedObjectRelationshipPayload(source_ids=(), target_ids=()),
        source_image_name=None,
        call_kwargs={},
    )

    with pytest.raises(RuntimeError, match="multiple exact compiled input edges"):
        request.exact_input_edge(endpoint_spec)


def test_parent_child_relationship_payload_slices_with_pure_2d_kwargs() -> None:
    payload = DirectedObjectRelationshipPayload(
        source_ids=(1, 2, 3, 4),
        target_ids=(10, 20, 30, 40),
        slice_indices=(0, 1, 0, 1),
        slice_count=2,
    )

    sliced = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
    )

    assert sliced == DirectedObjectRelationshipPayload(
        source_ids=(2, 4),
        target_ids=(20, 40),
        slice_count=1,
    )


def test_relationship_rows_project_temporal_endpoints_to_their_declared_slices() -> (
    None
):
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 1, 1] = 1
    labels[0, 2, 2] = 2
    labels[1, 1, 2] = 1
    embryos = ObjectLabelSet(
        name="Embryos",
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1, 2), (1,)),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    embryos_input = ArtifactSpec.input("Embryos", ObjectLabelsArtifactType)
    declaration = ObjectRelationshipDeclaration(
        source=embryos_input.ref(),
        target=embryos_input.ref(),
        producer_module_number=1,
        relationship_type="Parent",
        source_role="parent",
        target_role="child",
        source_id_field="parent_id",
        target_id_field="child_id",
        source_runtime_slice_offset=-1,
        target_runtime_slice_offset=0,
    )
    relationship_spec = ArtifactSpec.output(
        declaration.artifact_name(),
        RelationshipsArtifactType,
        relations=(declaration,),
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("TrackObjects").require_callable(),
            artifact_inputs=(embryos_input,),
            artifact_outputs=(relationship_spec,),
        )
    )
    runtime = _FakeCellProfilerRuntime({}, objects={"Embryos": embryos})
    payload = DirectedObjectRelationshipPayload(
        source_ids=(2,),
        target_ids=(1,),
        slice_indices=(1,),
        slice_count=2,
    )
    request = _cellprofiler_output_record_request(
        callable_contract=executor.callable_contract,
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec)
            for spec in executor.callable_contract.artifact_inputs.specs
        ),
        output_plans=tuple(
            _artifact_output_plan(item) for item in ((relationship_spec,))
        ),
        adapter=runtime,
        spec=relationship_spec,
        output_value=payload,
        source_image_name=None,
        source_image_payload=np.zeros((2, 5, 5), dtype=np.float32),
        call_kwargs={},
    )

    _record_output(request, relationship_spec, payload)
    rows = RelationshipMeasurementRows.for_request(request).rows().row_mappings()

    assert any(
        row.get("slice_index") == 0
        and row.get("object_label") == 2
        and row.get("Children_Embryos_Count") == 1
        for row in rows
    )
    assert any(
        row.get("slice_index") == 1
        and row.get("object_label") == 1
        and row.get("Parent_Embryos") == 2
        for row in rows
    )


def test_object_relationship_slices_with_pure_2d_kwargs() -> None:
    declaration = ObjectRelationshipDeclaration.parent_child(
        source=ArtifactSpec.output("Parents", ObjectLabelsArtifactType).ref(),
        target=ArtifactSpec.output("Children", ObjectLabelsArtifactType).ref(),
        producer_module_number=1,
    )
    relationship = ObjectRelationship(
        name="Parents_Children_relationships",
        declaration=declaration,
        payload=DirectedObjectRelationshipPayload(
            source_ids=(1, 2, 3, 4),
            target_ids=(10, 20, 30, 40),
            slice_indices=(0, 1, 0, 1),
            slice_count=2,
        ),
    )

    sliced = RuntimeSliceProjection.value_for_slice(
        relationship,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
    )

    assert isinstance(sliced, ObjectRelationship)
    assert sliced.payload.source_ids == (2, 4)
    assert sliced.payload.target_ids == (20, 40)
    assert sliced.payload.slice_count == 1


def test_relationship_measurements_reject_singleton_label_domain_broadcast() -> None:
    parent_labels = np.zeros((2, 5, 5), dtype=np.int32)
    child_labels = np.zeros((1, 5, 5), dtype=np.int32)
    parent_labels[:, 1:3, 1:3] = 1
    child_labels[0, 2:4, 2:4] = 1
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": np.zeros((2, 5, 5), dtype=np.float32)},
        objects={
            "Parents": ObjectLabelSet(
                name="Parents",
                variant_data=ObjectLabelVariantData(labels=parent_labels),
                domain=ObjectLabelDomain(
                    scope=ObjectLabelDomainScope.PLANE,
                    declared_object_id_domains=((1,), (1,)),
                ),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
            "Children": ObjectLabelSet(
                name="Children",
                variant_data=ObjectLabelVariantData(labels=child_labels),
                domain=ObjectLabelDomain(
                    scope=ObjectLabelDomainScope.PLANE,
                    declared_object_id_domains=((1,),),
                ),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
        },
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("RelateObjects").require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                ArtifactSpec.input("Children", ObjectLabelsArtifactType),
            ),
            artifact_outputs=(
                _relationship_output(
                    ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                    ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                ),
            ),
        )
    )
    payload = DirectedObjectRelationshipPayload(
        source_ids=(1, 1),
        target_ids=(1, 1),
        slice_indices=(0, 1),
        slice_count=2,
    )
    request = _cellprofiler_output_record_request(
        callable_contract=executor.callable_contract,
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec)
            for spec in executor.callable_contract.artifact_inputs.specs
        ),
        output_plans=tuple(
            _artifact_output_plan(item)
            for item in ((executor.callable_contract.artifact_outputs.specs[0],))
        ),
        adapter=runtime,
        spec=executor.callable_contract.artifact_outputs.specs[0],
        output_value=payload,
        source_image_name=None,
        call_kwargs={},
    )

    _record_output(request, request.spec, payload)
    with pytest.raises(ValueError, match="axis cardinality mismatch"):
        RelationshipMeasurementRows.for_request(request).rows()


def test_relationship_rows_use_declared_relationship_outputs_for_measurement_only_recording() -> (
    None
):
    parent_labels = np.zeros((5, 5), dtype=np.int32)
    child_labels = np.zeros((5, 5), dtype=np.int32)
    parent_labels[1:4, 1:4] = 1
    child_labels[2:3, 2:3] = 1
    input_specs = (
        ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
        ArtifactSpec.input("Children", ObjectLabelsArtifactType),
    )
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": np.zeros((5, 5), dtype=np.float32)},
        objects={
            "Parents": ObjectLabelSet(
                name="Parents",
                variant_data=ObjectLabelVariantData(labels=parent_labels),
                domain=ObjectLabelDomain(declared_object_ids=(1,)),
            ),
            "Children": ObjectLabelSet(
                name="Children",
                variant_data=ObjectLabelVariantData(labels=child_labels),
                domain=ObjectLabelDomain(declared_object_ids=(1,)),
            ),
        },
        artifact_input_edges=tuple(
            (_artifact_input_edge_for_test(spec) for spec in input_specs)
        ),
    )
    relationship_spec = _relationship_output(
        *input_specs,
    )
    measurement_spec = ArtifactSpec.output(
        "RelateObjects_measurements",
        MeasurementsArtifactType,
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("RelateObjects").require_callable(),
            artifact_inputs=input_specs,
            artifact_outputs=(relationship_spec, measurement_spec),
        )
    )
    payload = DirectedObjectRelationshipPayload(source_ids=(1,), target_ids=(1,))
    request = _cellprofiler_output_record_request(
        callable_contract=executor.callable_contract,
        output_plans=tuple(
            _artifact_output_plan(item) for item in ((measurement_spec,))
        ),
        adapter=runtime,
        spec=measurement_spec,
        output_value=MeasurementSparseColumnarRows.from_rows((), fields=()),
        source_image_name=None,
        call_kwargs={},
        declared_only_outputs=MappingProxyType({relationship_spec.ref(): payload}),
    )

    rows = RelationshipMeasurementRows.for_request(request).rows().row_mappings()

    assert any(
        row.get("object_name") == "Parents" and row.get("Children_Children_Count") == 1
        for row in rows
    )
    record_rows = measurement_table_for_module(request).rows.row_mappings()
    assert any(
        row.get("object_name") == "Parents" and row.get("Children_Children_Count") == 1
        for row in record_rows
    )
    assert all(
        "children_with_parents_count" not in row
        and "parent_object_count" not in row
        and "child_object_count" not in row
        for row in record_rows
    )
    CellProfilerOutputRecorder.for_artifact_type(MeasurementsArtifactType).record(
        request
    )
    assert len(runtime.measurements) == 1
    recorded_rows = runtime.measurements[0].rows
    assert any(
        row.get("object_name") == "Parents" and row.get("Children_Children_Count") == 1
        for row in recorded_rows.iter_row_mappings()
    )


def test_object_measurement_table_uses_provenance_without_image_ownership() -> None:
    carrier = np.zeros((2, 5, 5), dtype=np.float32)
    parent_source_paths = (
        "/source/site1_channel1.tif",
        "/source/site2_channel1.tif",
    )
    child_source_paths = (
        "/source/site1_channel2.tif",
        "/source/site2_channel2.tif",
    )
    parents = ObjectLabelSet(
        name="Parents",
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 5, 5), dtype=np.int32)),
        source_image_name="ParentImage",
        source_image_names=("SharedImage",),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=parent_source_paths,
            component_metadata=(
                {"site": "1", "channel": "1"},
                {"site": "2", "channel": "1"},
            ),
        ),
    )
    children = ObjectLabelSet(
        name="Children",
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 5, 5), dtype=np.int32)),
        source_image_name="ChildImage",
        source_image_names=("SharedImage",),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=child_source_paths,
            component_metadata=(
                {"site": "1", "channel": "2"},
                {"site": "2", "channel": "2"},
            ),
        ),
    )
    measurement_spec = ArtifactSpec.output(
        "RelateObjects_measurements",
        MeasurementsArtifactType,
    )
    object_specs = (
        ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
        ArtifactSpec.input("Children", ObjectLabelsArtifactType),
    )
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("RelateObjects").require_callable(),
            artifact_inputs=object_specs,
            artifact_outputs=(measurement_spec,),
        ),
        output_plans=(
            replace(
                _artifact_output_plan(measurement_spec),
                group_component=AllComponents.CHANNEL,
            ),
        ),
        spec=measurement_spec,
        output_value=MeasurementSparseColumnarRows.from_rows((), fields=()),
        source_image_name="Parents",
        source_image_payload=carrier,
        adapter=_FakeCellProfilerRuntime(
            {},
            objects={"Parents": parents, "Children": children},
            artifact_input_edges=tuple(
                (
                    cellprofiler_runtime_input_edge_for_test(
                        ArtifactInputPlan(
                            name=spec.name,
                            path=f"/memory/{spec.name}.pkl",
                            artifact_type=ObjectLabelsArtifactType,
                            group_keys=(channel,),
                            group_component=AllComponents.CHANNEL,
                            paths_by_group={
                                channel: f"/memory/{spec.name}_{channel}.pkl"
                            },
                        ),
                        spec=spec,
                        invocation_scope=ComponentGroupScope.ungrouped(),
                        producer_selection_scope=ComponentGroupScope(
                            (channel,),
                            component=AllComponents.CHANNEL,
                        ),
                        component_scopes=(
                            ComponentGroupScope(
                                (channel,),
                                component=AllComponents.CHANNEL,
                            ),
                        ),
                        consumer_variable_components=(AllComponents.SITE,),
                    )
                    for spec, channel in zip(object_specs, ("1", "2"), strict=True)
                )
            ),
            variable_components=(VariableComponents.SITE,),
            axis_scope=RuntimeExecutionAxisScope.from_raw(
                "test-axis", component=None, value=None
            ),
        ),
        call_kwargs={},
    )

    empty_rows = MeasurementSparseColumnarRows.from_rows((), fields=())
    source_qualified_rows = MeasurementSparseColumnarRows.from_rows(
        (
            {"source_image_name": "Parents"},
            {"source_image_name": "Children"},
        ),
        fields=(FieldSpec("source_image_name", str),),
    )
    provenance_metadata = request.measurement_source_metadata(object_specs)
    source_image_name = RelateObjectsModule.measurement_record_source_image_name(
        request,
        empty_rows,
    )
    source_metadata = RelateObjectsModule.measurement_record_source_metadata(
        request,
        empty_rows,
    )
    source_qualified_image_name = (
        MeasureImageAreaOccupiedBinaryModule.measurement_record_source_image_name(
            request, source_qualified_rows
        )
    )
    source_qualified_metadata = (
        MeasureImageAreaOccupiedBinaryModule.measurement_record_source_metadata(
            request, source_qualified_rows
        )
    )

    assert provenance_metadata.source_image_names == ("SharedImage",)
    assert source_image_name is None
    assert source_metadata.source_image_provenance_planes.paths == parent_source_paths
    assert source_qualified_image_name is None
    assert (
        source_qualified_metadata.source_image_provenance_planes.paths
        == parent_source_paths
    )


def test_object_output_table_uses_provenance_without_image_ownership() -> None:
    source_paths = (
        "/source/site1_channel2.tif",
        "/source/site2_channel2.tif",
    )
    output_labels = ObjectLabelSet(
        name="Cytoplasm",
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 5, 5), dtype=np.int32)),
        source_image_name="OrigGreen",
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=source_paths,
            component_metadata=(
                {"site": "1", "channel": "2"},
                {"site": "2", "channel": "2"},
            ),
        ),
    )
    source_spec = ArtifactSpec.input("OrigGreen", ImageArtifactType)
    object_spec = _output_from_input(
        "Cytoplasm",
        source_spec.name,
        input_type=source_spec.artifact_type,
    )
    measurement_spec = _measurement_output_for_objects(
        "IdentifyTertiaryObjects_measurements",
        object_spec,
    )
    output_plans = tuple(
        _artifact_output_plan(spec) for spec in (measurement_spec, object_spec)
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        artifact_output_bindings=tuple(
            zip((measurement_spec, object_spec), output_plans, strict=True)
        ),
    )
    runtime._store_runtime_artifact(output_plans[1], output_labels)
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module(
                "IdentifyTertiaryObjects"
            ).require_callable(),
            artifact_inputs=(source_spec,),
            artifact_outputs=(measurement_spec, object_spec),
        ),
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(item) for item in ((source_spec,))
        ),
        output_plans=output_plans,
        spec=measurement_spec,
        output_value=MeasurementSparseColumnarRows.from_rows((), fields=()),
        source_image_name=None,
        source_image_payload=np.zeros((2, 5, 5), dtype=np.float32),
        adapter=runtime,
        call_kwargs={},
    )

    rows = MeasurementSparseColumnarRows.from_rows((), fields=())
    source_image_name = (
        IdentifyTertiaryObjectsModule.measurement_record_source_image_name(
            request,
            rows,
        )
    )
    source_metadata = IdentifyTertiaryObjectsModule.measurement_record_source_metadata(
        request,
        rows,
    )

    assert source_image_name is None
    assert source_metadata.source_image_provenance_planes.paths == source_paths


def test_relate_objects_measurement_table_rejects_distinct_image_set_axes() -> None:
    carrier = np.zeros((1, 5, 5), dtype=np.float32)
    parents = ObjectLabelSet(
        name="Parents",
        variant_data=ObjectLabelVariantData(labels=np.zeros((1, 5, 5), dtype=np.int32)),
        source_image_name="ParentImage",
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/source/site1_channel1.tif",),
            component_metadata=({"site": "1", "channel": "1"},),
        ),
    )
    children = ObjectLabelSet(
        name="Children",
        variant_data=ObjectLabelVariantData(labels=np.zeros((1, 5, 5), dtype=np.int32)),
        source_image_name="ChildImage",
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/source/site2_channel2.tif",),
            component_metadata=({"site": "2", "channel": "2"},),
        ),
    )
    measurement_spec = ArtifactSpec.output(
        "RelateObjects_measurements",
        MeasurementsArtifactType,
    )
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("RelateObjects").require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                ArtifactSpec.input("Children", ObjectLabelsArtifactType),
            ),
            artifact_outputs=(measurement_spec,),
        ),
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(item)
            for item in (
                (
                    ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                    ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                )
            )
        ),
        output_plans=tuple(
            _artifact_output_plan(item) for item in ((measurement_spec,))
        ),
        spec=measurement_spec,
        output_value=MeasurementSparseColumnarRows.from_rows((), fields=()),
        source_image_name="Parents",
        source_image_payload=carrier,
        adapter=_FakeCellProfilerRuntime(
            {},
            objects={"Parents": parents, "Children": children},
            artifact_input_edges=tuple(
                (
                    cellprofiler_runtime_input_edge_for_test(
                        ArtifactInputPlan(
                            name=name,
                            path=f"/memory/{name}.pkl",
                            artifact_type=ObjectLabelsArtifactType,
                            group_keys=(channel,),
                            group_component=AllComponents.CHANNEL,
                            paths_by_group={channel: f"/memory/{name}_{channel}.pkl"},
                        ),
                        invocation_scope=ComponentGroupScope.ungrouped(),
                        producer_selection_scope=ComponentGroupScope(
                            (channel,),
                            component=AllComponents.CHANNEL,
                        ),
                        component_scopes=(
                            ComponentGroupScope(
                                (channel,),
                                component=AllComponents.CHANNEL,
                            ),
                        ),
                        consumer_variable_components=(AllComponents.SITE,),
                    )
                    for name, channel in (("Parents", "1"), ("Children", "2"))
                )
            ),
            variable_components=(VariableComponents.SITE,),
            axis_scope=RuntimeExecutionAxisScope.from_raw(
                "test-axis", component=None, value=None
            ),
        ),
        call_kwargs={},
    )

    with pytest.raises(ValueError, match="source image-set axis"):
        RelateObjectsModule.measurement_record_source_metadata(
            request,
            MeasurementSparseColumnarRows.from_rows((), fields=()),
        )


def test_object_lineage_measurement_table_preserves_current_payload_metadata() -> None:
    carrier = np.zeros((2, 5, 5), dtype=np.float32)
    measurement_spec = ArtifactSpec.output(
        "MaskObjects_measurements",
        MeasurementsArtifactType,
    )
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("MaskObjects").require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("Objects", ObjectLabelsArtifactType),
                ArtifactSpec.input("Mask", ObjectLabelsArtifactType),
            ),
            artifact_outputs=(measurement_spec,),
        ),
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(item)
            for item in (
                (
                    ArtifactSpec.input("Objects", ObjectLabelsArtifactType),
                    ArtifactSpec.input("Mask", ObjectLabelsArtifactType),
                )
            )
        ),
        output_plans=tuple(
            _artifact_output_plan(item) for item in ((measurement_spec,))
        ),
        spec=measurement_spec,
        output_value=MeasurementSparseColumnarRows.from_rows((), fields=()),
        source_image_payload=carrier,
        adapter=_FakeCellProfilerRuntime({}),
        call_kwargs={},
    )

    source_metadata = MaskObjectsModule.measurement_record_source_metadata(
        request,
        MeasurementSparseColumnarRows.from_rows((), fields=()),
    )

    assert source_metadata == image_payload_metadata(carrier)


def test_relationship_rows_do_not_slice_payload_scoped_3d_lineage_by_z_plane() -> None:
    parent_labels = np.zeros((3, 5, 5), dtype=np.int32)
    child_labels = np.zeros((3, 5, 5), dtype=np.int32)
    parent_labels[0, 1:3, 1:3] = 1
    parent_labels[1, 2:4, 2:4] = 2
    child_labels[0, 1:3, 1:3] = 1
    child_labels[1, 2:4, 2:4] = 2
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": np.zeros((3, 5, 5), dtype=np.float32)},
        objects={
            "Parents": ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(labels=parent_labels),
                domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
            ),
            "Children": ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(labels=child_labels),
                domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
            ),
        },
    )
    parent_spec = ArtifactSpec.input("Parents", ObjectLabelsArtifactType)
    child_spec = _output_from_input("Children", "Parents")
    relationship_spec = _relationship_output(parent_spec, child_spec)
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("ResizeObjects").require_callable(),
        artifact_inputs=(parent_spec,),
        artifact_outputs=(relationship_spec, child_spec),
    )
    executor = _module_executor(contract)
    payload = DirectedObjectRelationshipPayload(source_ids=(1, 2), target_ids=(1, 2))
    request = _cellprofiler_output_record_request(
        callable_contract=executor.callable_contract,
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec)
            for spec in executor.callable_contract.artifact_inputs.specs
        ),
        output_plans=tuple(
            _artifact_output_plan(item) for item in ((relationship_spec, child_spec))
        ),
        adapter=runtime,
        spec=relationship_spec,
        output_value=payload,
        source_image_name=None,
        call_kwargs={},
    )

    CellProfilerOutputRecorder.for_artifact_type(ObjectLabelsArtifactType).record(
        replace(
            request,
            spec=child_spec,
            output_plan=request.adapter.request.require_artifact_output_plan(
                child_spec.ref()
            ),
            output_value=ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(labels=child_labels),
                domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
            ),
        )
    )
    _record_output(request, request.spec, payload)
    rows = RelationshipMeasurementRows.for_request(request).rows().row_mappings()

    assert all("slice_index" not in row for row in rows)
    parent_rows = [
        row
        for row in rows
        if row.get("object_name") == "Children"
        and DirectParentReferenceFeatureDeclaration.feature_name(
            DirectParentReferenceMeasurementFeature("Parents")
        )
        in row
    ]
    assert [row["Parent_Parents"] for row in parent_rows] == [1, 2]


def test_object_lineage_transform_measurement_record_includes_parent_rows() -> None:
    parent_labels = ObjectLabelSet(
        name="Parents",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[1, 1], [0, 2]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
    )
    child_labels = ObjectLabelSet(
        name="Children",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[1, 1], [0, 2]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
    )
    measurement_spec = ArtifactSpec.output(
        "ResizeObjects_measurements",
        MeasurementsArtifactType,
    )
    child_spec = _output_from_input("Children", "Parents")
    relationship_spec = _relationship_output(
        ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
        child_spec,
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={"Parents": parent_labels, "Children": child_labels},
    )
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("ResizeObjects").require_callable(),
            artifact_inputs=(
                *((ArtifactSpec.input("Parents", ObjectLabelsArtifactType),)),
                *((ArtifactSpec.input("Parents", ObjectLabelsArtifactType),)),
            ),
            artifact_outputs=(measurement_spec, relationship_spec, child_spec),
        ),
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(item)
            for item in (
                ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
            )
        ),
        output_plans=tuple(
            _artifact_output_plan(item)
            for item in ((measurement_spec, relationship_spec, child_spec))
        ),
        adapter=runtime,
        spec=measurement_spec,
        output_value=MeasurementSparseColumnarRows.from_rows((), fields=()),
        source_image_name="Carrier",
        source_image_payload=np.ones((2, 2), dtype=np.float32),
        current_image=np.ones((2, 2), dtype=np.float32),
        call_kwargs={},
    )

    payload = DirectedObjectRelationshipPayload(source_ids=(1, 2), target_ids=(1, 2))
    _record_output(request, child_spec, child_labels)
    _record_output(request, relationship_spec, payload)
    table = measurement_table_for_module(request)
    rows = table.rows.row_mappings()

    assert table.source_image_name is None
    assert [
        row["Parent_Parents"]
        for row in rows
        if row.get("object_name") == "Children" and "Parent_Parents" in row
    ] == [1, 2]


def test_relateobjects_relationship_rows_project_distances_nominally() -> None:
    parent_labels = np.zeros((6, 6), dtype=np.int32)
    child_labels = np.zeros((6, 6), dtype=np.int32)
    parent_labels[1:5, 1:5] = 1
    child_labels[2:4, 2:4] = 1
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": np.zeros((6, 6), dtype=np.float32)},
        objects={
            "Parents": ObjectLabelSet(
                name="Parents",
                variant_data=ObjectLabelVariantData(labels=parent_labels),
                domain=ObjectLabelDomain(declared_object_ids=(1,)),
            ),
            "Children": ObjectLabelSet(
                name="Children",
                variant_data=ObjectLabelVariantData(labels=child_labels),
                domain=ObjectLabelDomain(declared_object_ids=(1,)),
            ),
        },
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("RelateObjects").require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                ArtifactSpec.input("Children", ObjectLabelsArtifactType),
            ),
            artifact_outputs=(
                _relationship_output(
                    ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                    ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                ),
            ),
        )
    )
    payload = DirectedObjectRelationshipPayload(source_ids=(1,), target_ids=(1,))
    request = _cellprofiler_output_record_request(
        callable_contract=executor.callable_contract,
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec)
            for spec in executor.callable_contract.artifact_inputs.specs
        ),
        output_plans=tuple(
            _artifact_output_plan(item)
            for item in ((executor.callable_contract.artifact_outputs.specs[0],))
        ),
        adapter=runtime,
        spec=executor.callable_contract.artifact_outputs.specs[0],
        output_value=MeasurementSparseColumnarRows.from_rows((), fields=()),
        source_image_name=None,
        call_kwargs={},
    )

    _record_output(request, request.spec, payload)
    rows = RelationshipMeasurementRows.for_request(request).rows().row_mappings()

    distance_rows = [
        row
        for row in rows
        if row.get("object_name") == "Children" and "Distance_Centroid_Parents" in row
    ]
    assert len(distance_rows) == 1
    assert distance_rows[0]["Distance_Centroid_Parents"] == 0.0
    assert distance_rows[0]["Distance_Minimum_Parents"] == pytest.approx(np.sqrt(2.5))
    assert type(distance_rows[0]["Distance_Centroid_Parents"]) is float
    assert type(distance_rows[0]["Distance_Minimum_Parents"]) is float


def test_relateobjects_relationship_rows_project_parent_mean_distances() -> None:
    parent_labels = np.zeros((8, 8), dtype=np.int32)
    child_labels = np.zeros((8, 8), dtype=np.int32)
    parent_labels[1:7, 1:7] = 1
    child_labels[2:4, 2:4] = 1
    child_labels[5:7, 5:7] = 2
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": np.zeros((8, 8), dtype=np.float32)},
        objects={
            "Parents": ObjectLabelSet(
                name="Parents",
                variant_data=ObjectLabelVariantData(labels=parent_labels),
                domain=ObjectLabelDomain(declared_object_ids=(1,)),
            ),
            "Children": ObjectLabelSet(
                name="Children",
                variant_data=ObjectLabelVariantData(labels=child_labels),
                domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
            ),
        },
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("RelateObjects").require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                ArtifactSpec.input("Children", ObjectLabelsArtifactType),
            ),
            artifact_outputs=(
                _relationship_output(
                    ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                    ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                ),
            ),
        )
    )
    payload = DirectedObjectRelationshipPayload(source_ids=(1, 1), target_ids=(1, 2))
    request = _cellprofiler_output_record_request(
        callable_contract=executor.callable_contract,
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec)
            for spec in executor.callable_contract.artifact_inputs.specs
        ),
        output_plans=tuple(
            _artifact_output_plan(item)
            for item in ((executor.callable_contract.artifact_outputs.specs[0],))
        ),
        adapter=runtime,
        spec=executor.callable_contract.artifact_outputs.specs[0],
        output_value=MeasurementSparseColumnarRows.from_rows((), fields=()),
        source_image_name=None,
        call_kwargs={"calculate_per_parent_means": True},
    )

    _record_output(request, request.spec, payload)
    rows = RelationshipMeasurementRows.for_request(request).rows().row_mappings()

    mean_rows = [
        row
        for row in rows
        if row.get("object_name") == "Parents"
        and "Mean_Children_Distance_Centroid_Parents" in row
    ]
    assert len(mean_rows) == 1
    assert type(mean_rows[0]["Mean_Children_Distance_Centroid_Parents"]) is float
    assert type(mean_rows[0]["Mean_Children_Distance_Minimum_Parents"]) is float
    assert mean_rows[0]["Mean_Children_Distance_Centroid_Parents"] == pytest.approx(
        np.sqrt(4.5)
    )
    assert "Mean_Children_Distance_Minimum_Parents" in mean_rows[0]


def test_relateobjects_parent_means_align_scoped_child_tables_by_source_plane() -> None:
    source_paths = ("/source/site1.tif", "/source/site2.tif")
    source_metadata = (
        {"well": "A14", "site": "1", "channel": "3"},
        {"well": "A14", "site": "2", "channel": "3"},
    )
    parent_labels = np.zeros((2, 6, 6), dtype=np.int32)
    child_labels = np.zeros((2, 6, 6), dtype=np.int32)
    parent_labels[0, 1:5, 1:5] = 1
    parent_labels[1, 1:3, 1:5] = 1
    parent_labels[1, 3:5, 1:5] = 2
    child_labels[0, 1:3, 1:3] = 1
    child_labels[0, 3:5, 3:5] = 2
    child_labels[0, 0, 0] = 3
    child_labels[1, 1:3, 1:3] = 1
    child_labels[1, 3:5, 3:5] = 2
    object_domain = ObjectLabelDomain(
        scope=ObjectLabelDomainScope.PLANE,
        declared_object_id_domains=((1, 2), (1, 2)),
    )
    colocalization_feature = "Correlation_Correlation_First_Second"
    costes_feature = "Correlation_Costes_Hoechst_Mito"
    scoped_tables = tuple(
        MeasurementTable(
            name="MeasureColocalization_measurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                (
                    *tuple(
                        {
                            "slice_index": 0,
                            "object_name": "Children",
                            "object_label": object_label,
                            colocalization_feature: value,
                            costes_feature: (
                                np.nan
                                if site_index == 0 and object_label == 2
                                else 1.0
                            ),
                        }
                        for object_label, value in enumerate(values, start=1)
                    ),
                    {
                        "slice_index": 0,
                        "object_name": "OtherObjects",
                        "object_label": 1,
                        colocalization_feature: -100.0 - site_index,
                    },
                ),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec(colocalization_feature, float),
                    FieldSpec(costes_feature, float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.IMAGE, "image"),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(source_paths[site_index],),
                component_metadata=(
                    {
                        "well": "A14",
                        "site": str(site_index + 1),
                    },
                ),
            ),
        )
        for site_index, values in enumerate(((0.2, 0.6, 99.0), (0.8, 1.0)))
    )
    unrelated_table = MeasurementTable(
        name="Unrelated_measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "object_name": "Children",
                    "object_label": 1,
                    colocalization_feature: -1.0,
                },
            ),
            fields=(
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec(colocalization_feature, float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Children"),
    )
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": np.zeros((2, 6, 6), dtype=np.float32)},
        objects={
            "Parents": ObjectLabelSet(
                name="Parents",
                variant_data=ObjectLabelVariantData(labels=parent_labels),
                domain=object_domain,
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                source_image_provenance_planes=(
                    SourceImageProvenancePlanes.from_components(
                        paths=source_paths,
                        component_metadata=source_metadata,
                    )
                ),
            ),
            "Children": ObjectLabelSet(
                name="Children",
                variant_data=ObjectLabelVariantData(labels=child_labels),
                domain=object_domain,
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                source_image_provenance_planes=(
                    SourceImageProvenancePlanes.from_components(
                        paths=source_paths,
                        component_metadata=source_metadata,
                    )
                ),
            ),
        },
        measurement_tables={
            "colocalization": scoped_tables,
            "unrelated": (unrelated_table,),
        },
        plane_projection=RuntimePlaneProjection.stack(2),
    )
    child_measurements = ArtifactSpec.output(
        "MeasureColocalization_measurements",
        MeasurementsArtifactType,
        relations=(
            ArtifactSpecRelation(
                ArtifactSpec.output("Children", ObjectLabelsArtifactType).ref()
            ),
        ),
    ).for_plan_type(ArtifactInputPlan)
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("RelateObjects").require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                child_measurements,
            ),
            artifact_outputs=(
                _relationship_output(
                    ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                    ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                ),
            ),
        )
    )
    payload = DirectedObjectRelationshipPayload(
        source_ids=(1, 1, 0, 1, 2),
        target_ids=(1, 2, 3, 1, 2),
        slice_indices=(0, 0, 0, 1, 1),
        slice_count=2,
    )
    request = _cellprofiler_output_record_request(
        callable_contract=executor.callable_contract,
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec)
            for spec in executor.callable_contract.artifact_inputs.specs
        ),
        output_plans=tuple(
            _artifact_output_plan(item)
            for item in ((executor.callable_contract.artifact_outputs.specs[0],))
        ),
        adapter=runtime,
        spec=executor.callable_contract.artifact_outputs.specs[0],
        output_value=MeasurementSparseColumnarRows.from_rows((), fields=()),
        source_image_name=None,
        call_kwargs={"calculate_per_parent_means": True},
    )

    _record_output(request, request.spec, payload)
    projector = RelationshipMeasurementRows.for_request(request)
    _, declaration, _ = projector.output_entries()[0]
    parent_spec = executor.callable_contract.artifact_inputs.by_ref(declaration.source)
    child_spec = executor.callable_contract.artifact_inputs.by_ref(declaration.target)
    assert parent_spec is not None
    assert child_spec is not None
    rows = projector.parent_mean_upstream_measurement_rows(
        parent_spec=parent_spec,
        child_spec=child_spec,
        payload=runtime.relationships[0],
    )

    mean_feature = f"Mean_Children_{colocalization_feature}"
    mean_rows = {
        (row["slice_index"], row["object_label"]): row[mean_feature]
        for row in rows
        if row.get("object_name") == "Parents" and mean_feature in row
    }
    assert mean_rows == pytest.approx({(0, 1): 0.4, (1, 1): 0.8, (1, 2): 1.0})
    assert (0, 0) not in mean_rows

    costes_mean_feature = f"Mean_Children_{costes_feature}"
    costes_mean_rows = {
        (row["slice_index"], row["object_label"]): row[costes_mean_feature]
        for row in rows
        if row.get("object_name") == "Parents" and costes_mean_feature in row
    }
    assert np.isnan(costes_mean_rows.pop((0, 1)))
    assert costes_mean_rows == pytest.approx({(1, 1): 1.0, (1, 2): 1.0})

    core_mean_features = {
        feature: RelateObjectsModule.AggregateMeasurementFeature.MEAN_CHILD.feature_name(
            child_object_name="Children",
            child_feature_name=feature.value,
        )
        for feature in CellProfilerObjectCoreMeasurementFeature
    }
    rows_by_parent = {
        (row["slice_index"], row["object_label"]): row
        for row in rows
        if row.get("object_name") == "Parents"
    }
    assert rows_by_parent[(0, 1)][
        core_mean_features[CellProfilerObjectCoreMeasurementFeature.OBJECT_NUMBER]
    ] == pytest.approx(1.5)
    assert rows_by_parent[(0, 1)][
        core_mean_features[CellProfilerObjectCoreMeasurementFeature.CENTER_X]
    ] == pytest.approx(2.5)
    assert rows_by_parent[(0, 1)][
        core_mean_features[CellProfilerObjectCoreMeasurementFeature.CENTER_Y]
    ] == pytest.approx(2.5)
    assert rows_by_parent[(1, 1)][
        core_mean_features[CellProfilerObjectCoreMeasurementFeature.OBJECT_NUMBER]
    ] == pytest.approx(1.0)
    assert rows_by_parent[(1, 2)][
        core_mean_features[CellProfilerObjectCoreMeasurementFeature.OBJECT_NUMBER]
    ] == pytest.approx(2.0)


def test_relateobjects_relationship_rows_project_distances_from_slice_measurements() -> (
    None
):
    parent_labels = np.zeros((2, 6, 6), dtype=np.int32)
    child_labels = np.zeros((2, 6, 6), dtype=np.int32)
    parent_labels[:, 1:5, 1:5] = 1
    child_labels[:, 2:4, 2:4] = 1
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": np.zeros((2, 6, 6), dtype=np.float32)},
        objects={
            "Parents": ObjectLabelSet(
                name="Parents",
                variant_data=ObjectLabelVariantData(labels=parent_labels),
                domain=ObjectLabelDomain(
                    scope=ObjectLabelDomainScope.PLANE,
                    declared_object_id_domains=((1,), (1,)),
                ),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
            "Children": ObjectLabelSet(
                name="Children",
                variant_data=ObjectLabelVariantData(labels=child_labels),
                domain=ObjectLabelDomain(
                    scope=ObjectLabelDomainScope.PLANE,
                    declared_object_id_domains=((1,), (1,)),
                ),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
        },
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module("RelateObjects").require_callable(),
            artifact_inputs=(
                ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                ArtifactSpec.input("Children", ObjectLabelsArtifactType),
            ),
            artifact_outputs=(
                _relationship_output(
                    ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                    ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                ),
            ),
        )
    )
    payload = DirectedObjectRelationshipPayload(
        source_ids=(1, 1),
        target_ids=(1, 1),
        slice_indices=(0, 1),
        slice_count=2,
    )
    request = _cellprofiler_output_record_request(
        callable_contract=executor.callable_contract,
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(spec)
            for spec in executor.callable_contract.artifact_inputs.specs
        ),
        output_plans=tuple(
            _artifact_output_plan(item)
            for item in ((executor.callable_contract.artifact_outputs.specs[0],))
        ),
        adapter=runtime,
        spec=executor.callable_contract.artifact_outputs.specs[0],
        output_value=MeasurementSparseColumnarRows.from_rows((), fields=()),
        source_image_name=None,
        call_kwargs={},
    )

    _record_output(request, request.spec, payload)
    rows = RelationshipMeasurementRows.for_request(request).rows().row_mappings()

    distance_rows = [
        row
        for row in rows
        if row.get("object_name") == "Children" and "Distance_Centroid_Parents" in row
    ]
    assert {(row["slice_index"], row["object_label"]) for row in distance_rows} == {
        (0, 1),
        (1, 1),
    }


def test_object_relationship_backend_uses_sparse_ijv_contract_nominally() -> None:
    parent_dense = np.zeros((8, 8), dtype=np.int32)
    child_dense = np.zeros((8, 8), dtype=np.int32)
    parent_sparse = ObjectLabelSet(
        name="Parents",
        variant_data=ObjectLabelVariantData(
            labels=SparseIJVLabelRows(
                np.asarray(
                    (
                        (1, 1, 1),
                        (2, 2, 2),
                    ),
                    dtype=np.int32,
                )
            )
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    child_sparse = ObjectLabelSet(
        name="Children",
        variant_data=ObjectLabelVariantData(
            labels=SparseIJVLabelRows(
                np.asarray(
                    (
                        (1, 1, 7),
                        (2, 2, 8),
                        (7, 7, 9),
                    ),
                    dtype=np.int32,
                )
            )
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    parent_dense[1, 1] = 1
    parent_dense[2, 2] = 2
    backend = ObjectRelationshipBackendStrategy.for_memory_type()

    dense_payload = backend.parent_child_payload_from_labels(
        np.zeros((8, 8), dtype=np.int32),
        child_dense,
    )
    sparse_payload = backend.parent_child_payload_from_labels(
        parent_sparse, child_sparse
    )
    mixed_payload = backend.parent_child_payload_from_labels(parent_dense, child_sparse)

    assert dense_payload == DirectedObjectRelationshipPayload(
        source_ids=(),
        target_ids=(),
    )
    assert sparse_payload == DirectedObjectRelationshipPayload(
        source_ids=(1, 2),
        target_ids=(7, 8),
    )
    assert backend.parents_of_from_payload(sparse_payload, 9)[6:9].tolist() == [
        1,
        2,
        0,
    ]
    assert mixed_payload == sparse_payload


def test_define_grid_automatic_uses_integer_lowest_spot_origin() -> None:
    image = np.zeros((20, 20), dtype=np.float32)
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:6, 3:7] = 1
    labels[12:16, 13:17] = 2

    _image, grid = define_grid_automatic.__wrapped__(
        image,
        labels,
        grid_rows=2,
        grid_columns=2,
    )

    assert grid.x_origin == 4.0
    assert grid.y_origin == 3.0
    assert grid.x_spacing == 10.0
    assert grid.y_spacing == 10.0


def test_spatial_grid_output_recorder_accepts_pure_2d_grid_sequence() -> None:
    grid_spec = ArtifactSpec.output("Grid", SpatialGridArtifactType)
    runtime = _FakeCellProfilerRuntime({})
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module("DefineGridManual").require_callable(),
            artifact_inputs=(),
            artifact_outputs=(grid_spec,),
        ),
        output_plans=tuple(_artifact_output_plan(item) for item in ((grid_spec,))),
        adapter=runtime,
        spec=grid_spec,
        output_value=None,
        call_kwargs={},
    )
    grids = [
        SpatialGrid(
            name="grid_info",
            rows=2,
            columns=2,
            x_spacing=8.0,
            y_spacing=8.0,
            x_origin=x_origin,
            y_origin=4.0,
        )
        for x_origin in (1.0, 2.0)
    ]

    _record_output(request, grid_spec, grids)

    recorded = runtime.spatial_grids["Grid"]
    assert isinstance(recorded, RuntimeSliceAlignedValues)
    assert [
        recorded.value_for_slice(slice_index).x_origin for slice_index in range(2)
    ] == [1.0, 2.0]


def test_identify_objects_in_grid_respects_row_primary_ordering() -> None:
    image = np.zeros((6, 9), dtype=np.float32)
    grid = SpatialGrid(
        name="grid_info",
        rows=2,
        columns=3,
        x_spacing=3.0,
        y_spacing=3.0,
        x_origin=1.0,
        y_origin=1.0,
        ordering=SpatialGridOrdering.BY_ROWS,
    )

    _image, _stats, payload = identify_objects_in_grid(
        image,
        topology_inputs=(grid,),
        dtype_config=DtypeConfig(),
    )

    labels = np.asarray(payload.labels)
    assert labels[1, 1] == 1
    assert labels[4, 1] == 2
    assert labels[1, 4] == 3
    assert labels[4, 4] == 4
    assert labels[1, 7] == 5
    assert labels[4, 7] == 6


def test_identify_objects_in_grid_respects_column_primary_ordering() -> None:
    image = np.zeros((6, 9), dtype=np.float32)
    grid = SpatialGrid(
        name="grid_info",
        rows=2,
        columns=3,
        x_spacing=3.0,
        y_spacing=3.0,
        x_origin=1.0,
        y_origin=1.0,
        ordering=SpatialGridOrdering.BY_COLUMNS,
    )

    _image, _stats, payload = identify_objects_in_grid(
        image,
        topology_inputs=(grid,),
        dtype_config=DtypeConfig(),
    )

    labels = np.asarray(payload.labels)
    assert labels[1, 1] == 1
    assert labels[1, 4] == 2
    assert labels[1, 7] == 3
    assert labels[4, 1] == 4
    assert labels[4, 4] == 5
    assert labels[4, 7] == 6


def test_identify_objects_in_grid_fill_boundaries_match_floor_bins() -> None:
    grid = GridDefinition.from_runtime(
        image_shape=(11, 14),
        grid=None,
        grid_rows=3,
        grid_columns=4,
        x_spacing=3.25,
        y_spacing=2.75,
        x_origin=1.2,
        y_origin=1.6,
        ordering=SpatialGridOrdering.BY_ROWS,
    )

    labels = grid.filled_labels()
    row_origin = int(grid.y_location_of_lowest_y_spot - grid.y_spacing / 2)
    col_origin = int(grid.x_location_of_lowest_x_spot - grid.x_spacing / 2)
    expected = np.zeros(labels.shape, dtype=np.int32)
    rows, cols = np.indices(labels.shape)
    row_bins = np.floor((rows - row_origin) / grid.y_spacing).astype(int)
    col_bins = np.floor((cols - col_origin) / grid.x_spacing).astype(int)
    mask = (
        (row_bins >= 0)
        & (row_bins < grid.rows)
        & (col_bins >= 0)
        & (col_bins < grid.columns)
    )
    expected[mask] = grid.spot_table[row_bins[mask], col_bins[mask]]

    np.testing.assert_array_equal(labels, expected)


def test_identify_objects_in_grid_natural_shape_keeps_accepted_guide_shape() -> None:
    image = np.zeros((5, 10), dtype=np.float32)
    guide_labels = np.zeros((5, 10), dtype=np.int32)
    guide_labels[2, 1:6] = 1
    grid = SpatialGrid(
        name="grid_info",
        rows=1,
        columns=2,
        x_spacing=5.0,
        y_spacing=5.0,
        x_origin=2.0,
        y_origin=2.0,
        ordering=SpatialGridOrdering.BY_ROWS,
    )

    _image, _stats, payload = identify_objects_in_grid(
        image,
        topology_inputs=(grid, guide_labels),
        shape_choice="natural_shape_and_location",
        dtype_config=DtypeConfig(),
    )

    labels = np.asarray(payload.labels)
    np.testing.assert_array_equal(labels[2, 1:6], np.asarray([1, 1, 1, 1, 0]))
    assert labels[2, 7] == 0
    assert labels[0, 0] == 0


def test_identify_objects_in_grid_natural_shape_preserves_accepted_grid_ids() -> None:
    image = np.zeros((5, 15), dtype=np.float32)
    guide_labels = np.zeros((5, 15), dtype=np.int32)
    guide_labels[2, 1:3] = 10
    guide_labels[2, 11:13] = 20
    grid = SpatialGrid(
        name="grid_info",
        rows=1,
        columns=3,
        x_spacing=5.0,
        y_spacing=5.0,
        x_origin=2.0,
        y_origin=2.0,
        ordering=SpatialGridOrdering.BY_ROWS,
    )

    _image, _stats, payload = identify_objects_in_grid(
        image,
        topology_inputs=(grid, guide_labels),
        shape_choice="natural_shape_and_location",
        dtype_config=DtypeConfig(),
    )

    labels = np.asarray(payload.labels)
    assert set(np.unique(labels)) == {0, 1, 3}
    np.testing.assert_array_equal(labels[2, 1:3], np.asarray([1, 1]))
    assert labels[2, 7] == 0
    np.testing.assert_array_equal(labels[2, 11:13], np.asarray([3, 3]))
    assert payload.domain.declared_object_count == 3
    assert payload.domain.declared_object_ids == (1, 2, 3)


def test_identify_objects_in_grid_natural_shape_uses_filtered_guides() -> None:
    guide_labels = np.zeros((5, 10), dtype=np.int32)
    guide_labels[2, 1:4] = 7
    filtered_guides = np.zeros_like(guide_labels)
    grid = GridDefinition.from_runtime(
        image_shape=guide_labels.shape,
        grid=None,
        grid_rows=1,
        grid_columns=2,
        x_spacing=5.0,
        y_spacing=5.0,
        x_origin=2.0,
        y_origin=2.0,
        ordering=SpatialGridOrdering.BY_ROWS,
    )

    labels = NaturalGridShapeStrategy().labels(
        GridShapeRequest(
            grid=grid,
            guiding_labels=guide_labels,
            filtered_guides=filtered_guides,
        )
    )
    assert not np.any(labels)

    filtered_guides[2, 1:4] = 7
    labels = NaturalGridShapeStrategy().labels(
        GridShapeRequest(
            grid=grid,
            guiding_labels=guide_labels,
            filtered_guides=filtered_guides,
        )
    )

    np.testing.assert_array_equal(labels[2, 1:4], np.asarray([1, 1, 1]))
    assert 7 not in labels
    assert labels[2, 6] == 0


def test_identify_objects_in_grid_rejects_guides_centered_on_cell_edges() -> None:
    guide_labels = np.zeros((6, 12), dtype=np.int32)
    guide_labels[2, 1:4] = 1
    guide_labels[2, 4:7] = 2
    guide_labels[2, 7:10] = 3
    grid = GridDefinition.from_runtime(
        image_shape=guide_labels.shape,
        grid=None,
        grid_rows=1,
        grid_columns=2,
        x_spacing=6.0,
        y_spacing=6.0,
        x_origin=2.0,
        y_origin=2.0,
        ordering=SpatialGridOrdering.BY_ROWS,
    )

    filtered_guides = grid.filtered_guides(guide_labels)
    labels = grid.labels_from_filtered_guides(filtered_guides)

    np.testing.assert_array_equal(labels[2, 1:4], np.asarray([1, 1, 1]))
    assert not np.any(labels[guide_labels == 2])
    np.testing.assert_array_equal(labels[2, 7:10], np.asarray([2, 2, 2]))


def test_identify_objects_in_grid_location_rows_preserve_empty_grid_slots() -> None:
    image = np.zeros((5, 10), dtype=np.float32)
    guide_labels = np.zeros((5, 10), dtype=np.int32)
    guide_labels[2, 6:8] = 2
    grid = SpatialGrid(
        name="grid_info",
        rows=1,
        columns=2,
        x_spacing=5.0,
        y_spacing=5.0,
        x_origin=2.0,
        y_origin=2.0,
        ordering=SpatialGridOrdering.BY_ROWS,
    )

    _image, stats, payload = identify_objects_in_grid(
        image,
        topology_inputs=(grid, guide_labels),
        shape_choice="natural_shape_and_location",
        dtype_config=DtypeConfig(),
    )
    labels = np.asarray(payload.labels)
    assert not np.any(labels == 1)
    assert np.any(labels == 2)
    runtime = _FakeCellProfilerRuntime(
        {},
        {
            "Guides": ObjectLabelSet(
                name="Guides",
                variant_data=ObjectLabelVariantData(labels=guide_labels),
                domain=ObjectLabelDomain.declared(
                    scope=ObjectLabelDomainScope.PAYLOAD,
                    declared_object_ids=(2,),
                ),
                source_image_provenance_planes=(
                    SourceImageProvenancePlanes.from_components(
                        paths=("/input/site-1.tif",)
                    )
                ),
            )
        },
    )
    object_spec = _output_from_input("GridObjects", "Guides")
    measurement_spec = _measurement_output_for_objects(
        "IdentifyObjectsInGrid_1_measurements",
        object_spec,
    )
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module(
                "IdentifyObjectsInGrid"
            ).require_callable(),
            artifact_inputs=(
                *(
                    (
                        ArtifactSpec.input("Grid", SpatialGridArtifactType),
                        ArtifactSpec.input("Guides", ObjectLabelsArtifactType),
                    )
                ),
                *(
                    (
                        ArtifactSpec.input("Grid", SpatialGridArtifactType),
                        ArtifactSpec.input("Guides", ObjectLabelsArtifactType),
                    )
                ),
            ),
            artifact_outputs=(measurement_spec, object_spec),
        ),
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(item)
            for item in (
                ArtifactSpec.input("Grid", SpatialGridArtifactType),
                ArtifactSpec.input("Guides", ObjectLabelsArtifactType),
                ArtifactSpec.input("Grid", SpatialGridArtifactType),
                ArtifactSpec.input("Guides", ObjectLabelsArtifactType),
            )
        ),
        output_plans=tuple(
            _artifact_output_plan(item) for item in ((measurement_spec, object_spec))
        ),
        adapter=runtime,
        spec=measurement_spec,
        output_value=stats,
        source_image_name=None,
        call_kwargs={
            "topology_inputs": (grid, guide_labels),
            "shape_choice": "natural_shape_and_location",
        },
    )
    _record_output(request, object_spec, payload)
    table = measurement_table_for_module(request)

    by_key = {
        (
            row["object_label"],
            row["feature_name"],
        ): row["result_value"]
        for row in table.rows.iter_row_mappings()
        if row.get("feature_name")
        in {
            CellProfilerObjectCoreMeasurementFeature.CENTER_X.value,
            CellProfilerObjectCoreMeasurementFeature.CENTER_Y.value,
        }
    }
    assert np.isnan(
        by_key[(1, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)]
    )
    assert np.isnan(
        by_key[(1, CellProfilerObjectCoreMeasurementFeature.CENTER_Y.value)]
    )
    assert by_key[(2, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)] == 6.5
    assert by_key[(2, CellProfilerObjectCoreMeasurementFeature.CENTER_Y.value)] == 2.0


def test_identify_objects_in_grid_location_rows_use_slice_aligned_grid() -> None:
    image = np.zeros((5, 15), dtype=np.float32)
    first_guides = np.zeros((5, 15), dtype=np.int32)
    first_guides[2, 1:3] = 2
    first_guides[2, 3:5] = 3
    second_guides = np.zeros((5, 15), dtype=np.int32)
    second_guides[3, 6:8] = 2
    grid = SpatialGrid(
        name="grid_info",
        rows=1,
        columns=3,
        x_spacing=5.0,
        y_spacing=5.0,
        x_origin=2.0,
        y_origin=2.0,
        ordering=SpatialGridOrdering.BY_ROWS,
    )
    outputs = tuple(
        identify_objects_in_grid(
            image,
            topology_inputs=(grid, guides),
            shape_choice="natural_shape_and_location",
            dtype_config=DtypeConfig(),
        )
        for guides in (first_guides, second_guides)
    )
    payload = Pure2DAuxiliaryOutputAggregator.aggregate(
        tuple(output[2] for output in outputs),
        MemoryType.NUMPY.value,
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        {
            "Guides": ObjectLabelSet(
                name="Guides",
                variant_data=ObjectLabelVariantData(
                    labels=np.stack((first_guides, second_guides), axis=0)
                ),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                domain=ObjectLabelDomain(
                    declared_object_id_domains=((2, 3), (2,)),
                    scope=ObjectLabelDomainScope.PLANE,
                ),
                source_image_provenance_planes=(
                    SourceImageProvenancePlanes.from_components(
                        paths=("/input/site-1.tif", "/input/site-2.tif")
                    )
                ),
            )
        },
    )
    object_spec = _output_from_input("GridObjects", "Guides")
    measurement_spec = _measurement_output_for_objects(
        "IdentifyObjectsInGrid_1_measurements",
        object_spec,
    )
    request = _cellprofiler_output_record_request(
        callable_contract=_compiled_callable_contract(
            CellProfilerModule.require_module(
                "IdentifyObjectsInGrid"
            ).require_callable(),
            artifact_inputs=(
                *(
                    (
                        ArtifactSpec.input("Grid", SpatialGridArtifactType),
                        ArtifactSpec.input("Guides", ObjectLabelsArtifactType),
                    )
                ),
                *(
                    (
                        ArtifactSpec.input("Grid", SpatialGridArtifactType),
                        ArtifactSpec.input("Guides", ObjectLabelsArtifactType),
                    )
                ),
            ),
            artifact_outputs=(measurement_spec, object_spec),
        ),
        artifact_input_edges=tuple(
            _artifact_input_edge_for_test(item)
            for item in (
                ArtifactSpec.input("Grid", SpatialGridArtifactType),
                ArtifactSpec.input("Guides", ObjectLabelsArtifactType),
                ArtifactSpec.input("Grid", SpatialGridArtifactType),
                ArtifactSpec.input("Guides", ObjectLabelsArtifactType),
            )
        ),
        output_plans=tuple(
            _artifact_output_plan(item) for item in ((measurement_spec, object_spec))
        ),
        adapter=runtime,
        spec=measurement_spec,
        output_value=RuntimeSliceAlignedValues(tuple(output[1] for output in outputs)),
        source_image_name=None,
        call_kwargs={
            "topology_inputs": (
                RuntimeSliceAlignedValues((grid, grid)),
                np.stack((first_guides, second_guides), axis=0),
            ),
            "shape_choice": "natural_shape_and_location",
        },
    )
    _record_output(request, object_spec, payload)
    table = measurement_table_for_module(request)

    by_key = {
        (
            row["slice_index"],
            row["object_label"],
            row["feature_name"],
        ): row
        for row in table.rows.iter_row_mappings()
        if "object_label" in row
    }
    assert (
        by_key[(0, 1, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)][
            "result_value"
        ]
        == 1.5
    )
    assert (
        by_key[(0, 1, CellProfilerObjectCoreMeasurementFeature.CENTER_Y.value)][
            "result_value"
        ]
        == 2.0
    )
    assert np.isnan(
        by_key[(0, 2, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)][
            "result_value"
        ]
    )
    assert np.isnan(
        by_key[(0, 3, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)][
            "result_value"
        ]
    )
    assert np.isnan(
        by_key[(1, 1, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)][
            "result_value"
        ]
    )
    assert (
        by_key[(1, 2, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)][
            "result_value"
        ]
        == 6.5
    )
    assert (
        by_key[(1, 2, CellProfilerObjectCoreMeasurementFeature.CENTER_Y.value)][
            "result_value"
        ]
        == 3.0
    )


def test_object_location_measurements_preserve_declared_empty_grid_cells() -> None:
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 3:5, 3:5] = 2
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1, 2, 3), (1, 2, 3)),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    rows = ObjectLocationMeasurementRows(
        payload,
        object_name="GridObjects",
    ).rows()

    by_key = {
        (
            row["slice_index"],
            row["object_label"],
            row["feature_name"],
        ): row["result_value"]
        for row in rows
    }
    assert (
        by_key[(0, 1, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)] == 1.5
    )
    assert (
        by_key[(0, 1, CellProfilerObjectCoreMeasurementFeature.CENTER_Y.value)] == 1.5
    )
    assert np.isnan(
        by_key[(0, 2, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)]
    )
    assert np.isnan(
        by_key[(0, 3, CellProfilerObjectCoreMeasurementFeature.CENTER_Y.value)]
    )
    assert np.isnan(
        by_key[(1, 1, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)]
    )
    assert (
        by_key[(1, 2, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)] == 3.5
    )
    assert (
        by_key[(1, 2, CellProfilerObjectCoreMeasurementFeature.CENTER_Y.value)] == 3.5
    )
    assert np.isnan(
        by_key[(1, 3, CellProfilerObjectCoreMeasurementFeature.CENTER_Y.value)]
    )


def test_object_location_measurements_preserve_sparse_overlapping_membership() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=SparseIJVLabelRows(
                np.asarray(
                    (
                        (0, 0, 1),
                        (0, 1, 1),
                        (0, 1, 2),
                        (0, 2, 2),
                    ),
                    dtype=np.int32,
                )
            )
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
    )

    rows = ObjectLocationMeasurementRows(
        payload,
        object_name="OverlappingWorms",
    ).rows()

    by_key = {
        (row["object_label"], row["feature_name"]): row["result_value"] for row in rows
    }
    assert by_key[(1, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)] == 0.5
    assert by_key[(1, CellProfilerObjectCoreMeasurementFeature.CENTER_Y.value)] == 0.0
    assert by_key[(2, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)] == 1.5
    assert by_key[(2, CellProfilerObjectCoreMeasurementFeature.CENTER_Y.value)] == 0.0


def test_object_location_measurements_preserve_sparse_plane_storage() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=SparseIJVLabelRows.from_slices(
                (
                    SparseIJVLabelRows(
                        np.asarray(
                            (
                                (0, 0, 1),
                                (0, 1, 1),
                                (0, 1, 2),
                            ),
                            dtype=np.int32,
                        )
                    ),
                    SparseIJVLabelRows(
                        np.asarray(
                            (
                                (1, 0, 3),
                                (1, 1, 3),
                            ),
                            dtype=np.int32,
                        )
                    ),
                )
            )
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (3,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    rows = ObjectLocationMeasurementRows(
        payload,
        object_name="OverlappingWorms",
    ).rows()

    by_key = {
        (
            row["slice_index"],
            row["object_label"],
            row["feature_name"],
        ): row["result_value"]
        for row in rows
    }
    assert (
        by_key[(0, 1, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)] == 0.5
    )
    assert (
        by_key[(0, 2, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)] == 1.0
    )
    assert (
        by_key[(1, 3, CellProfilerObjectCoreMeasurementFeature.CENTER_X.value)] == 0.5
    )
    assert (
        by_key[(1, 3, CellProfilerObjectCoreMeasurementFeature.CENTER_Y.value)] == 1.0
    )


def test_object_location_measurements_use_payload_domain_for_full_stack_labels() -> (
    None
):
    labels = np.zeros((3, 5, 5), dtype=np.int32)
    labels[1, 1:3, 1:3] = 1
    labels[2, 3:5, 3:5] = 2
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
    )

    rows = ObjectLocationMeasurementRows(
        payload,
        object_name="Nuclei",
        domain_scope=ObjectLabelDomainScope.PAYLOAD,
    ).rows()

    by_key = {
        (
            row["slice_index"],
            row["object_label"],
            row["feature_name"],
        ): row["result_value"]
        for row in rows
    }
    assert {row["slice_index"] for row in rows} == {0}
    assert (
        by_key[(0, 1, CellProfilerObjectCoreMeasurementFeature.CENTER_Z.value)] == 1.0
    )
    assert (
        by_key[(0, 2, CellProfilerObjectCoreMeasurementFeature.CENTER_Z.value)] == 2.0
    )


def test_object_location_measurements_preserve_declared_diagonal_planes() -> None:
    plane = np.array([[1, 0], [0, 2]], dtype=np.int32)
    labels = np.zeros((2, 2, 2, 2), dtype=np.int32)
    labels[0, 0] = plane
    labels[1, 1] = plane
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (1, 2)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    rows = ObjectLocationMeasurementRows(
        payload,
        object_name="GridObjects",
    ).rows()

    assert len(rows) == 12
    by_key = {
        (
            row["slice_index"],
            row["object_label"],
            row["feature_name"],
        ): row["result_value"]
        for row in rows
    }
    assert {row["slice_index"] for row in rows} == {0, 1}
    assert (
        by_key[(0, 1, CellProfilerObjectCoreMeasurementFeature.CENTER_Z.value)] == 0.0
    )
    assert (
        by_key[(1, 1, CellProfilerObjectCoreMeasurementFeature.CENTER_Z.value)] == 1.0
    )


def test_object_location_measurements_preserve_repeated_homogeneous_planes() -> None:
    plane = np.array([[1, 0], [0, 2]], dtype=np.int32)
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.stack((plane, plane), axis=0)),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (1, 2)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    rows = ObjectLocationMeasurementRows(
        payload,
        object_name="GridObjects",
    ).rows()

    assert len(rows) == 12
    assert {row["slice_index"] for row in rows} == {0, 1}
    assert {
        row["result_value"]
        for row in rows
        if row["feature_name"]
        == CellProfilerObjectCoreMeasurementFeature.CENTER_Z.value
    } == {0.0}


def test_sparse_object_label_aggregation_preserves_declared_domain() -> None:
    first = ObjectLabelSet(
        name="GridObjects",
        variant_data=ObjectLabelVariantData(
            labels=SparseIJVLabelRows.from_label_slice(
                np.asarray([[0, 0, 1], [1, 1, 3]], dtype=np.int32)
            )
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=ObjectLabelDomain(
            declared_object_count=4,
        ),
    )
    second = ObjectLabelSet(
        name="GridObjects",
        variant_data=ObjectLabelVariantData(
            labels=SparseIJVLabelRows.from_label_slice(
                np.asarray([[0, 0, 2], [1, 1, 4]], dtype=np.int32)
            )
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=ObjectLabelDomain(
            declared_object_count=4,
        ),
    )

    aggregated = Pure2DAuxiliaryOutputAggregator.aggregate(
        [first, second],
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, ObjectLabelSet)
    assert aggregated.domain.declared_object_id_domains == (
        (1, 2, 3, 4),
        (1, 2, 3, 4),
    )
    assert aggregated.domain.scope is ObjectLabelDomainScope.PLANE
    assert isinstance(aggregated.labels, SparseIJVLabelRows)


def test_measurement_table_collection_rejects_sharded_row_offsets() -> None:
    first = MeasurementTable(
        name="ObjectMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            [{"slice_index": 0, "object_label": 1, "area": 11.0}],
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("area", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Objects"),
    )
    second = MeasurementTable(
        name="ObjectMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            [{"slice_index": 1, "object_label": 1, "area": 13.0}],
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("area", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Objects"),
    )

    with pytest.raises(
        RuntimeSliceProjectionDeclarationError,
        match="source-plane provenance",
    ):
        RuntimeSliceProjection.slice_count_from_values((first, second))


def test_measurement_table_slice_count_rejects_sparse_row_axis_values() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            [
                {"slice_index": 0, "object_label": 1, "area": 11.0},
                {"slice_index": 18, "object_label": 2, "area": 13.0},
            ],
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("area", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Objects"),
    )

    with pytest.raises(
        RuntimeSliceProjectionDeclarationError,
        match="source-plane provenance",
    ):
        RuntimeSliceProjection.measurement_table_slice_count(table)
    with pytest.raises(RuntimeSliceProjectionDeclarationError):
        RuntimeSliceProjection.slice_count_from_values((table,))


def test_measurement_table_slice_count_rejects_columnar_row_inference() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=_ColumnarMeasurementRows(
            {
                "slice_index": (0, 0, 1),
                "object_label": (1, 2, 1),
                "area": (11.0, 12.0, 13.0),
            },
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("area", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Objects"),
    )

    with pytest.raises(
        RuntimeSliceProjectionDeclarationError,
        match="source-plane provenance",
    ):
        RuntimeSliceProjection.measurement_table_slice_count(table)


def test_measurement_table_for_slice_preserves_columnar_rows() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=_ColumnarMeasurementRows(
            {
                "slice_index": (0, 0, 1),
                "feature_name": ("Area", "MeanIntensity", "Area"),
                "result_value": (11.0, 12.0, 13.0),
            },
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("feature_name", str),
                FieldSpec("result_value", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Objects"),
    )

    sliced = MeasurementTableAxisProjection(
        axis=MeasurementRowAxisField.SLICE_INDEX,
        value=1,
        table=table,
    ).apply()

    assert isinstance(sliced.rows, ColumnarRows)
    assert tuple(sliced.rows.columns["slice_index"]) == (1,)
    assert tuple(sliced.rows.columns["result_value"]) == (13.0,)


def test_measurement_table_for_slice_preserves_sparse_exact_schema() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            [
                {"slice_index": 1, "feature_name": "Area", "result_value": 13.0},
                {"slice_index": 1, "MeanIntensity": 17.0},
            ],
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("feature_name", str, required=False),
                FieldSpec("result_value", float, required=False),
                FieldSpec("MeanIntensity", float, required=False),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Objects"),
    )

    sliced = MeasurementTableAxisProjection(
        axis=MeasurementRowAxisField.SLICE_INDEX,
        value=1,
        table=table,
    ).apply()

    assert tuple(field.name for field in sliced.rows.fields) == (
        "slice_index",
        "feature_name",
        "result_value",
        "MeanIntensity",
    )
    assert sliced.rows.row_mappings() == (
        {"slice_index": 1, "feature_name": "Area", "result_value": 13.0},
        {"slice_index": 1, "MeanIntensity": 17.0},
    )


def test_image_payload_pure_2d_slicer_preserves_declared_volume_planes() -> None:
    pairwise = np.zeros((2, 2, 5, 6), dtype=np.float32)
    pairwise[0, 0] = 1.0
    pairwise[0, 1] = 2.0
    pairwise[1, 0] = 3.0
    pairwise[1, 1] = 4.0

    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(pairwise, None)
    slices = Pure2DInputSlicer.strategy_for_value(payload).slice_value(
        payload,
        MemoryType.NUMPY.value,
    )

    assert len(slices) == 2
    np.testing.assert_array_equal(image_payload_data(slices[0]), pairwise[0])
    np.testing.assert_array_equal(image_payload_data(slices[1]), pairwise[1])


def test_numpy_pure_2d_slicer_does_not_invent_a_stack_axis() -> None:
    array = np.zeros((2, 5, 6), dtype=np.float32)
    slicer = Pure2DInputSlicer.strategy_for_value(array)

    assert slicer.is_single_plane_value(array)
    assert slicer.slice_value(array, MemoryType.NUMPY.value) == (array,)


def test_image_payload_pure_2d_slicer_projects_declared_single_volume_mask() -> None:
    data = np.arange(1 * 3 * 4 * 5, dtype=np.float32).reshape(1, 3, 4, 5)
    mask = np.ones((1, 3, 4, 5), dtype=bool)
    mask[0, 1] = False
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(data, mask)

    slices = Pure2DInputSlicer.strategy_for_value(payload).slice_value(
        payload,
        MemoryType.NUMPY.value,
    )

    assert len(slices) == 1
    np.testing.assert_array_equal(image_payload_data(slices[0]), data[0])
    np.testing.assert_array_equal(image_payload_mask(slices[0]), mask[0])


def test_image_payload_pure_2d_slicer_projects_declared_volume_masks() -> None:
    data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    mask = np.ones((2, 3, 4, 5), dtype=bool)
    mask[1, 2] = False
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(data, mask)

    slices = Pure2DInputSlicer.strategy_for_value(payload).slice_value(
        payload,
        MemoryType.NUMPY.value,
    )

    assert len(slices) == 2
    np.testing.assert_array_equal(image_payload_data(slices[1]), data[1])
    np.testing.assert_array_equal(image_payload_mask(slices[1]), mask[1])


def test_image_payload_slice_projector_rejects_out_of_range_mask_axis() -> None:
    data_slice = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    mask = np.ones((2, 4, 5), dtype=bool)
    mask[1] = False

    with pytest.raises(ValueError, match="does not carry the requested"):
        ImagePayloadSliceProjector(
            mask=mask,
            metadata=ImagePayloadMetadata(
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
        ).payload_for_slice(data_slice, 2)


def test_callable_contract_rejects_missing_processing_contract():
    def two_dimensional_only(image: np.ndarray, **kwargs) -> np.ndarray:
        if image.ndim != 2:
            raise RuntimeError("2D only")
        return image

    attach_callable_contract_metadata(
        two_dimensional_only,
        declared_processing_contract="unknown",
    )

    with pytest.raises(TypeError, match="must declare a ProcessingContract"):
        CallableContract.from_callable(
            two_dimensional_only
        ).require_processing_contract()


def test_measurement_image_for_labels_rejects_undeclared_source_stack() -> None:
    image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    labels = np.ones((4, 5), dtype=np.int32)

    with pytest.raises(ValueError, match="incompatible declared domains"):
        _measurement_image_for_labels(image, labels)


def test_measurement_image_for_labels_rejects_undeclared_object_domain_stack() -> None:
    image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    labels = np.ones((4, 5), dtype=np.int32)

    with pytest.raises(ValueError, match="requires an ObjectLabelValue"):
        _measurement_image_for_labels(
            image,
            labels,
            reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
        )


def test_measurement_image_for_labels_rejects_undeclared_object_domain_shape() -> None:
    image = np.ones((10, 12), dtype=np.float32)
    labels = np.ones((4, 5), dtype=np.int32)

    with pytest.raises(ValueError, match="requires an ObjectLabelValue"):
        _measurement_image_for_labels(
            image,
            labels,
            reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
        )


def test_measurement_image_for_labels_rejects_source_domain_shape_mismatch() -> None:
    image = np.ones((10, 12), dtype=np.float32)
    labels = np.ones((4, 5), dtype=np.int32)

    with pytest.raises(ValueError, match="incompatible declared domains"):
        _measurement_image_for_labels(image, labels)


def test_measurement_domain_alignment_projects_declared_source_axis() -> None:
    class Projector(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return None

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"origMemb": 1}.get(source_aliases[0])

        def source_binding_axis_size(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return 2

    image = np.stack(
        (
            np.full((3, 4, 5), 11, dtype=np.float32),
            np.full((3, 4, 5), 22, dtype=np.float32),
        )
    )
    label_planes = np.stack(
        (
            np.full((3, 4, 5), 1, dtype=np.int32),
            np.full((3, 4, 5), 2, dtype=np.int32),
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=label_planes),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (2,)),
        ),
    )
    projector = Projector()
    measurement_image = CellProfilerMeasurementImage(
        source_image_name=None,
        source_aliases=("origMemb",),
        payload=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
            source_image_names=("origDNA", "origMemb"),
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=("/input/origDNA.tif", "/input/origMemb.tif"),
                )
            ),
        ).payload_with(image, None),
    )
    aligned_labels = (
        MeasurementLabelSourceAlignmentStrategy.align_request_labels_to_image_source(
            measurement_image.alignment_request(
                labels=label_planes,
                label_payload=labels,
                plane_projector=projector,
            )
        )
    )

    np.testing.assert_array_equal(
        object_label_dense_array(aligned_labels),
        label_planes[1],
    )
    np.testing.assert_array_equal(
        MeasurementImageLabelAlignmentStrategy.align(
            measurement_image.alignment_request(
                labels=aligned_labels,
                plane_projector=projector,
            )
        ),
        image[1],
    )


def test_measurement_domain_source_axis_projection_preserves_image_payload_context() -> (
    None
):
    class Projector(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return None

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"origMemb": 1}[source_aliases[0]]

        def source_binding_axis_size(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            del source_aliases
            return 2

    data = np.stack(
        (
            np.full((4, 5), 11, dtype=np.float32),
            np.full((4, 5), 22, dtype=np.float32),
        )
    )
    mask = np.stack(
        (
            np.zeros((4, 5), dtype=bool),
            np.ones((4, 5), dtype=bool),
        )
    )
    payload = MaskedImagePayload(
        data=data,
        mask=mask,
        metadata=ImagePayloadMetadata(
            source_image_names=("origDNA", "origMemb"),
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=("/input/origDNA.tif", "/input/origMemb.tif"),
                )
            ),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name=None,
        source_aliases=("origMemb",),
        payload=payload,
    )

    aligned = MeasurementImageLabelAlignmentStrategy.align(
        measurement_image.alignment_request(
            labels=np.ones((4, 5), dtype=np.int32),
            plane_projector=Projector(),
        )
    )

    assert isinstance(aligned, MaskedImagePayload)
    np.testing.assert_array_equal(image_payload_data(aligned), data[1])
    np.testing.assert_array_equal(image_payload_mask(aligned), mask[1])
    metadata = image_payload_metadata(aligned)
    assert metadata.source_path == "/input/origMemb.tif"
    assert metadata.source_provenance.source_plane_count == 0
    assert metadata.source_provenance.represented_source_image_names == ("origMemb",)


def test_measurement_domain_alignment_projects_source_owned_object_labels() -> None:
    class Projector(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return None

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return source_aliases.index("rawGFP")

        def source_binding_axis_size(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            del source_aliases
            return 2

    image = np.stack(
        (
            np.full((4, 5), 11, dtype=np.float32),
            np.full((4, 5), 22, dtype=np.float32),
        )
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        source_image_name="rawGFP",
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        variant_data=ObjectLabelVariantData(
            labels=np.stack(
                (
                    np.full((4, 5), 1, dtype=np.int32),
                    np.full((4, 5), 2, dtype=np.int32),
                )
            )
        ),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (2,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name=None,
        source_aliases=("rawDNA", "rawGFP"),
        payload=image,
    )
    aligned_labels = (
        MeasurementLabelSourceAlignmentStrategy.align_request_labels_to_image_source(
            measurement_image.alignment_request(
                labels=labels,
                label_payload=labels,
                plane_projector=Projector(),
            )
        )
    )

    assert isinstance(aligned_labels, ObjectLabelSet)
    np.testing.assert_array_equal(
        object_label_dense_array(aligned_labels),
        np.full((4, 5), 2, dtype=np.int32),
    )
    assert aligned_labels.source_image_name == "rawGFP"


def test_prepared_measurement_labels_project_runtime_slice_payload_with_dense_labels() -> (
    None
):
    class Projector(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return 1

        def runtime_slice_axis_size(self) -> int | None:
            return 2

    label_planes = np.stack(
        (
            np.full((4, 5), 3, dtype=np.int32),
            np.full((4, 5), 7, dtype=np.int32),
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=label_planes),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((3,), (7,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="BF_image",
        source_aliases=(),
        payload=np.zeros((4, 5), dtype=np.float32),
    )

    prepared = measurement_image.prepare_object_labels(
        labels,
        plane_projector=Projector(),
    )

    assert isinstance(prepared.source_projected_payload, ObjectLabelPayload)
    assert (
        prepared.source_projected_payload.domain.scope is ObjectLabelDomainScope.PAYLOAD
    )
    assert (
        prepared.source_projected_payload.object_label_domain().declared_object_ids
        == (7,)
    )
    np.testing.assert_array_equal(
        object_label_dense_array(prepared.source_projected_payload),
        label_planes[1],
    )
    np.testing.assert_array_equal(
        object_label_dense_array(prepared.source_projected_labels),
        label_planes[1],
    )
    np.testing.assert_array_equal(
        object_label_dense_array(prepared.completion_payload),
        label_planes[1],
    )


def test_prepared_measurement_labels_reject_implicit_image_source_identity_projection() -> (
    None
):
    label_planes = np.stack(
        (
            np.full((4, 5), 3, dtype=np.int32),
            np.full((4, 5), 7, dtype=np.int32),
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=label_planes),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((3,), (7,)),
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=(
                {"site": "1", "channel": "1"},
                {"site": "2", "channel": "1"},
            ),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="BF_image",
        payload=ImageMetadataPayload(
            np.zeros((4, 5), dtype=np.float32),
            metadata=ImagePayloadMetadata(
                source_component_metadata={"site": "2", "channel": "1"},
            ),
        ),
    )

    with pytest.raises(ValueError, match="incompatible declared domains"):
        measurement_image.prepare_object_labels(labels)


def test_prepared_measurement_labels_reject_implicit_image_set_identity_projection() -> (
    None
):
    label_planes = np.stack(
        (
            np.full((4, 5), 3, dtype=np.int32),
            np.full((4, 5), 7, dtype=np.int32),
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=label_planes),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((3,), (7,)),
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=(
                {"site": "1", "channel": "1"},
                {"site": "2", "channel": "1"},
            ),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="Marker_image",
        payload=ImageMetadataPayload(
            np.zeros((4, 5), dtype=np.float32),
            metadata=ImagePayloadMetadata(
                source_component_metadata={"site": "2", "channel": "7"},
            ),
        ),
    )

    with pytest.raises(ValueError, match="incompatible declared domains"):
        measurement_image.prepare_object_labels(labels)


def test_measurement_labels_preserve_stack_for_object_domain_alignment() -> None:
    image = np.ones((1, 4, 5), dtype=np.float32)
    labels = np.arange(2 * 4 * 5, dtype=np.int32).reshape(2, 4, 5)

    measurement_labels = MeasurementLabelSourceAlignmentStrategy.align(image, labels)

    assert measurement_labels.shape == labels.shape
    np.testing.assert_array_equal(measurement_labels, labels)


def test_measurement_label_alignment_preserves_runtime_slice_payload_for_aligned_stack() -> (
    None
):
    first_image = ImagePayloadMetadata(
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(0, 0),
            source_shape_yx=(4, 4),
        ),
    ).payload_with(np.ones((2, 2), dtype=np.float32), None)
    second_image = ImagePayloadMetadata(
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 0),
            source_shape_yx=(4, 4),
        ),
    ).payload_with(np.ones((2, 2), dtype=np.float32), None)
    labels = np.stack(
        (
            np.full((4, 4), 1, dtype=np.int32),
            np.full((4, 4), 2, dtype=np.int32),
        )
    )
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    measurement_labels = MeasurementLabelSourceAlignmentStrategy.align(
        AlignedImageStack((first_image, second_image)),
        labels,
        label_payload=label_payload,
    )

    assert measurement_labels.shape == labels.shape
    np.testing.assert_array_equal(measurement_labels, labels)


def test_measurement_label_alignment_preserves_runtime_slice_payload_for_dense_stack() -> (
    None
):
    image = np.ones((2, 4, 5), dtype=np.float32)
    label_plane = np.full((4, 5), 9, dtype=np.int32)
    labels = np.stack((label_plane, label_plane))
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((9,), (9,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    measurement_labels = MeasurementLabelSourceAlignmentStrategy.align(
        image,
        labels,
        label_payload=label_payload,
    )

    assert measurement_labels.shape == labels.shape
    np.testing.assert_array_equal(measurement_labels, labels)


def test_aligned_stack_kwargs_projects_runtime_slice_labels_to_reference_slice_domain() -> (
    None
):
    reference_slice = ImagePayloadMetadata(
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 0),
            source_shape_yx=(4, 4),
        ),
    ).payload_with(np.ones((2, 2), dtype=np.float32), None)
    labels = np.stack(
        (
            np.full((4, 4), 1, dtype=np.int32),
            np.full((4, 4), 2, dtype=np.int32),
        )
    )
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(4, 4)),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    resolved = aligned_image_stack_kwargs(
        {"labels": label_payload},
        slice_index=1,
        slice_count=2,
        reference_payload=reference_slice,
    )

    assert isinstance(resolved["labels"], ObjectLabelPayload)
    np.testing.assert_array_equal(
        resolved["labels"].labels,
        np.full((2, 2), 2, dtype=np.int32),
    )
    assert resolved["labels"].domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert resolved["labels"].domain.declared_object_ids == (2,)
    assert resolved["labels"].plane_axis is None
    assert resolved["labels"].source_spatial_domain.origin_yx == (2, 0)
    assert resolved["labels"].source_spatial_domain.source_shape_yx == (4, 4)


def test_aligned_stack_kwargs_projects_runtime_slice_labels_without_source_metadata() -> (
    None
):
    labels = np.stack(
        (
            np.full((4, 4), 1, dtype=np.int32),
            np.full((4, 4), 2, dtype=np.int32),
        )
    )
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    resolved = aligned_image_stack_kwargs(
        {"labels": label_payload},
        slice_index=1,
        slice_count=2,
        reference_payload=np.ones((4, 4), dtype=np.float32),
    )

    assert isinstance(resolved["labels"], ObjectLabelPayload)
    np.testing.assert_array_equal(
        resolved["labels"].labels,
        np.full((4, 4), 2, dtype=np.int32),
    )
    assert resolved["labels"].domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert resolved["labels"].domain.declared_object_ids == (2,)
    assert resolved["labels"].plane_axis is None


def test_measurement_labels_do_not_infer_broadcast_from_equal_planes() -> None:
    image = np.ones((2, 4, 5), dtype=np.float32)
    label_plane = np.arange(4 * 5, dtype=np.int32).reshape(4, 5)
    labels = np.stack((label_plane, label_plane))

    measurement_labels = MeasurementLabelSourceAlignmentStrategy.align(image, labels)

    assert measurement_labels.shape == labels.shape
    np.testing.assert_array_equal(measurement_labels, labels)


def _prepared_measurement_labels(
    measurement_image: CellProfilerMeasurementImage,
    labels: ObjectLabelValue,
    *,
    adapter: RuntimePlaneAxisProjector | None = None,
) -> object:
    return PreparedMeasurementObjectLabels.from_source(
        measurement_image,
        labels,
        plane_projector=adapter,
        align_image_to_labels=measurement_image.align_to_labels,
    ).measurement_labels


@dataclass(frozen=True, slots=True)
class _TestRuntimePlaneAxisProjector(RuntimePlaneAxisProjector):
    runtime_slice_index: int | None = None
    runtime_slice_size: int | None = None
    source_binding_index: int | None = None
    source_binding_size: int | None = None

    def runtime_slice_plane_index(self) -> int | None:
        return self.runtime_slice_index

    def runtime_slice_axis_size(self) -> int | None:
        return self.runtime_slice_size

    def source_binding_axis_plane_index(
        self,
        source_aliases: tuple[str, ...],
    ) -> int | None:
        del source_aliases
        return self.source_binding_index

    def source_binding_axis_size(
        self,
        source_aliases: tuple[str, ...],
    ) -> int | None:
        return self.source_binding_size


def test_measurement_labels_select_source_named_plane_from_composed_image() -> None:
    image = np.ones((2, 4, 5), dtype=np.float32)
    dna_labels = np.full((4, 5), 3, dtype=np.int32)
    gfp_labels = np.full((4, 5), 7, dtype=np.int32)
    labels = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=np.stack((dna_labels, gfp_labels))),
        source_image_name="rawDNA",
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((3,), (7,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawDNA__rawGFP",
        source_aliases=("rawDNA", "rawGFP"),
        payload=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
            source_image_names=("rawDNA", "rawGFP"),
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=("/input/rawDNA.tif", "/input/rawGFP.tif"),
                )
            ),
        ).payload_with(image, None),
    )

    measurement_labels = _prepared_measurement_labels(
        measurement_image,
        labels,
        adapter=_TestRuntimePlaneAxisProjector(
            source_binding_index=0,
            source_binding_size=2,
        ),
    )

    assert measurement_labels.shape == dna_labels.shape
    np.testing.assert_array_equal(measurement_labels, dna_labels)


def test_measurement_labels_select_current_source_binding_plane() -> None:
    image = np.ones((3, 4, 5), dtype=np.float32)
    label_planes = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
            np.full((4, 5), 3, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        source_image_name="rawDNA",
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (2,), (3,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawDAPI__rawGFP__rawDNA",
        source_aliases=("rawDAPI", "rawGFP", "rawDNA"),
        payload=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
            source_image_names=("rawDAPI", "rawGFP", "rawDNA"),
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=(
                        "/input/rawDAPI.tif",
                        "/input/rawGFP.tif",
                        "/input/rawDNA.tif",
                    ),
                )
            ),
        ).payload_with(image, None),
    )

    measurement_labels = _prepared_measurement_labels(
        measurement_image,
        labels,
        adapter=_TestRuntimePlaneAxisProjector(
            source_binding_index=2,
            source_binding_size=3,
        ),
    )

    assert measurement_labels.shape == (4, 5)
    np.testing.assert_array_equal(measurement_labels, label_planes[2])


def test_measurement_labels_reject_source_binding_cardinality_mismatch() -> None:
    image = np.ones((2, 4, 5), dtype=np.float32)
    label_planes = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        source_image_name="rawFarRed",
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (2,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawDNA__rawGFP__rawFarRed",
        source_aliases=("rawDNA", "rawGFP", "rawFarRed"),
        payload=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
            source_image_names=("rawDNA", "rawGFP"),
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=("/input/rawDNA.tif", "/input/rawGFP.tif"),
                )
            ),
        ).payload_with(image, None),
    )

    with pytest.raises(
        ValueError,
        match="runtime plane-axis cardinality conflicts with its declared local payload",
    ):
        _prepared_measurement_labels(
            measurement_image,
            labels,
            adapter=_TestRuntimePlaneAxisProjector(
                source_binding_index=0,
                source_binding_size=3,
            ),
        )


def test_measurement_labels_select_measurement_source_binding_over_label_origin() -> (
    None
):
    image = np.ones((2, 4, 5), dtype=np.float32)
    label_planes = np.stack(
        (
            np.full((4, 5), 11, dtype=np.int32),
            np.full((4, 5), 22, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="ExpandedCells",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        source_image_name="rawDNA",
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((11,), (22,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawGFP",
        source_aliases=("rawGFP",),
        payload=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
            source_image_names=("rawDNA", "rawGFP"),
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=("/input/rawDNA.tif", "/input/rawGFP.tif"),
                )
            ),
        ).payload_with(image, None),
    )

    measurement_labels = _prepared_measurement_labels(
        measurement_image,
        labels,
        adapter=_TestRuntimePlaneAxisProjector(
            source_binding_index=1,
            source_binding_size=2,
        ),
    )

    assert measurement_labels.shape == (4, 5)
    np.testing.assert_array_equal(measurement_labels, label_planes[1])


def test_measurement_labels_do_not_slice_site_stack_for_single_source_binding() -> None:
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return None

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"rawGFP": 0}.get(source_aliases[0])

    image = np.ones((3, 4, 5), dtype=np.float32)
    labels = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
            np.full((4, 5), 3, dtype=np.int32),
        )
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawGFP",
        source_aliases=("rawGFP",),
        payload=image,
    )

    del measurement_image
    measurement_labels = MeasurementLabelSourceAlignmentStrategy.align(image, labels)

    assert measurement_labels.shape == labels.shape
    np.testing.assert_array_equal(measurement_labels, labels)


def test_measurement_labels_do_not_project_unowned_label_stack_by_source_alias() -> (
    None
):
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return None

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"rawGFP": 0}.get(source_aliases[0])

    image = np.ones((3, 4, 5), dtype=np.float32)
    label_planes = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
            np.full((4, 5), 3, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (2,), (3,)),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawGFP",
        source_aliases=("rawGFP",),
        payload=image,
    )

    measurement_labels = _prepared_measurement_labels(
        measurement_image,
        labels,
        adapter=Adapter(),
    )

    assert measurement_labels.shape == label_planes.shape
    np.testing.assert_array_equal(measurement_labels, label_planes)


def test_measurement_labels_do_not_project_runtime_slice_label_stack_by_source_alias() -> (
    None
):
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return 1

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"BF_image": 1}.get(source_aliases[0])

    image = np.ones((2, 4, 5), dtype=np.float32)
    label_planes = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        source_image_name="BF_image",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((10,), (20,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="BF_image",
        source_aliases=("BF_image",),
        payload=image,
    )

    measurement_labels = _prepared_measurement_labels(
        measurement_image,
        labels,
        adapter=Adapter(),
    )

    assert measurement_labels.shape == label_planes.shape
    np.testing.assert_array_equal(measurement_labels, label_planes)


def test_measurement_labels_do_not_project_runtime_slice_stack_for_aligned_measurement_image() -> (
    None
):
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return 1

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"CropBlue": 0, "CropGreen": 1}.get(source_aliases[0])

        def source_binding_axis_size(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            del source_aliases
            return 2

    label_planes = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
            np.full((4, 5), 30, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        source_image_name="CropBlue",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((10,), (20,), (30,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="CropBlue__CropGreen",
        source_aliases=("CropBlue", "CropGreen"),
        payload=AlignedImageStack(
            tuple(
                ImagePayloadMetadata(
                    plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
                    source_image_names=("CropBlue", "CropGreen"),
                    source_image_provenance_planes=(
                        SourceImageProvenancePlanes.from_components(
                            paths=(
                                "/input/CropBlue.tif",
                                "/input/CropGreen.tif",
                            ),
                        )
                    ),
                ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)
                for _ in range(3)
            )
        ),
    )

    measurement_labels = _prepared_measurement_labels(
        measurement_image,
        labels,
        adapter=Adapter(),
    )

    assert measurement_labels.shape == label_planes.shape
    np.testing.assert_array_equal(measurement_labels, label_planes)


def test_measurement_labels_project_runtime_slice_stack_for_single_plane_source() -> (
    None
):
    image = np.ones((4, 5), dtype=np.float32)
    label_planes = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        source_image_name="BF_image",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((10,), (20,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="BF_image",
        source_aliases=("BF_image",),
        payload=image,
    )

    measurement_labels = _prepared_measurement_labels(
        measurement_image,
        labels,
        adapter=_TestRuntimePlaneAxisProjector(
            runtime_slice_index=1,
            runtime_slice_size=2,
        ),
    )

    assert measurement_labels.shape == label_planes[1].shape
    np.testing.assert_array_equal(measurement_labels, label_planes[1])


def test_measurement_labels_reject_runtime_stack_without_declared_axis_size() -> None:
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return 2

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return None

    label_planes = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((10,), (20,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="OrigMito",
        source_aliases=("OrigMito",),
        payload=np.ones((4, 5), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="incompatible declared domains"):
        _prepared_measurement_labels(
            measurement_image,
            labels,
            adapter=Adapter(),
        )


def test_measurement_labels_project_runtime_slice_stack_for_object_domain() -> None:
    label_planes = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((10,), (20,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name=None,
        source_aliases=(),
        payload=np.ones((4, 5), dtype=np.float32),
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    measurement_labels = _prepared_measurement_labels(
        measurement_image,
        labels,
        adapter=_TestRuntimePlaneAxisProjector(
            runtime_slice_index=1,
            runtime_slice_size=2,
        ),
    )

    assert measurement_labels.shape == label_planes[1].shape
    np.testing.assert_array_equal(measurement_labels, label_planes[1])


def test_measurement_labels_reject_runtime_slice_stack_without_projection() -> None:
    label_planes = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((10,), (20,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name=None,
        source_aliases=(),
        payload=np.ones((4, 5), dtype=np.float32),
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    with pytest.raises(ValueError, match="Measurement image alignment"):
        PreparedMeasurementObjectLabels.from_source(
            measurement_image,
            labels,
            align_image_to_labels=measurement_image.align_to_labels,
        )


def test_measurement_labels_project_runtime_slice_stack_for_unrelated_source_image() -> (
    None
):
    label_planes = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_name="BF_image",
        source_image_names=("BF_image", "MorphBf"),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((10,), (20,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="DF_image",
        source_aliases=("DF_image",),
        payload=np.ones((4, 5), dtype=np.float32),
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    measurement_labels = _prepared_measurement_labels(
        measurement_image,
        labels,
        adapter=_TestRuntimePlaneAxisProjector(
            runtime_slice_index=1,
            runtime_slice_size=2,
        ),
    )

    assert measurement_labels.shape == label_planes[1].shape
    np.testing.assert_array_equal(measurement_labels, label_planes[1])


def test_measurement_labels_project_source_binding_stack_for_object_domain() -> None:
    label_planes = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("BF_image", "MorphBf"),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((10,), (20,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name=None,
        source_aliases=(),
        payload=np.ones((4, 5), dtype=np.float32),
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    measurement_labels = _prepared_measurement_labels(
        measurement_image,
        labels,
        adapter=_TestRuntimePlaneAxisProjector(
            source_binding_index=0,
            source_binding_size=2,
        ),
    )

    assert measurement_labels.shape == label_planes[0].shape
    np.testing.assert_array_equal(measurement_labels, label_planes[0])


def test_measurement_labels_project_repeated_source_binding_stack_for_object_domain() -> (
    None
):
    label_plane = np.full((4, 5), 10, dtype=np.int32)
    label_planes = np.stack((label_plane, label_plane))
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("BF_image", "MorphBf"),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((10,), (10,)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name=None,
        source_aliases=(),
        payload=np.ones((4, 5), dtype=np.float32),
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    measurement_labels = _prepared_measurement_labels(
        measurement_image,
        labels,
        adapter=_TestRuntimePlaneAxisProjector(
            source_binding_index=0,
            source_binding_size=2,
        ),
    )

    assert measurement_labels.shape == label_plane.shape
    np.testing.assert_array_equal(measurement_labels, label_plane)


def test_measurement_labels_reject_undeclared_nested_runtime_stack_axis() -> None:
    runtime_stack = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
        )
    )
    grouped_stack = np.stack((runtime_stack, runtime_stack))
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=grouped_stack),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_names=("BF_image", "MorphBf"),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((10, 20), (10, 20)),
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name=None,
        source_aliases=(),
        payload=np.ones((4, 5), dtype=np.float32),
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    with pytest.raises(ValueError, match="Measurement image alignment"):
        _prepared_measurement_labels(
            measurement_image,
            labels,
            adapter=_TestRuntimePlaneAxisProjector(
                runtime_slice_index=1,
                runtime_slice_size=2,
            ),
        )


def test_measurement_labels_preserve_payload_scoped_volume_for_source_binding() -> None:
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return None

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"rawDNA": 0}.get(source_aliases[0])

    image = np.ones((3, 4, 5), dtype=np.float32)
    label_volume = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
            np.full((4, 5), 3, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_volume),
        source_image_name="MembFinal",
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PAYLOAD,
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawDNA",
        source_aliases=("rawDNA",),
        payload=image,
    )

    measurement_labels = _prepared_measurement_labels(
        measurement_image,
        labels,
        adapter=Adapter(),
    )

    assert measurement_labels.shape == label_volume.shape
    np.testing.assert_array_equal(measurement_labels, label_volume)


def test_object_label_payload_preserves_source_metadata_for_measurements() -> None:
    image = np.ones((4, 5), dtype=np.float32)
    image_spec = ArtifactSpec.input("rawDNA", ImageArtifactType)
    spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    labels = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=np.ones((4, 5), dtype=np.int32)),
        source_image_name="rawDNA",
    )
    runtime = _FakeCellProfilerRuntime(
        {"rawDNA": image},
        {"Nuclei": labels},
        artifact_input_edges=(_artifact_input_edge_for_test(spec),),
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module(
                "MeasureObjectIntensity"
            ).require_callable(),
            artifact_inputs=(image_spec, spec),
            artifact_outputs=(
                ArtifactSpec.output("Measurements", MeasurementsArtifactType),
            ),
        )
    )
    _activate_runtime_contract(executor.callable_contract, runtime)

    payload = RuntimeArtifactTypeStrategy.for_artifact_type(
        spec.artifact_type
    ).raw_runtime_input_value(
        RuntimeInputBindingRequest(
            adapter=runtime,
            kwargs={},
            current_image=image,
        ).artifact_request_for_spec(spec)
    )

    np.testing.assert_array_equal(payload.labels, labels.labels)
    assert payload.source_image_name == "rawDNA"


def test_runtime_object_label_payload_ignores_measurement_image_as_selector() -> None:
    image = np.ones((4, 5), dtype=np.float32)
    image_spec = ArtifactSpec.input("rawDNA", ImageArtifactType)
    spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    labels = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.stack(
                (
                    np.ones((4, 5), dtype=np.int32),
                    np.full((4, 5), 2, dtype=np.int32),
                )
            )
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=({"well": "A01"}, {"well": "A01"})
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    runtime = _FakeCellProfilerRuntime(
        {"rawDNA": image},
        {"Nuclei": labels},
        artifact_input_edges=(_artifact_input_edge_for_test(spec),),
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module(
                "MeasureObjectIntensity"
            ).require_callable(),
            artifact_inputs=(image_spec, spec),
            artifact_outputs=(
                ArtifactSpec.output("Measurements", MeasurementsArtifactType),
            ),
        )
    )
    _activate_runtime_contract(executor.callable_contract, runtime)

    measurement_image = ImagePayloadMetadata(
        source_component_metadata={"well": "A01"},
    ).payload_with(image, None)
    payload = RuntimeArtifactTypeStrategy.for_artifact_type(
        spec.artifact_type
    ).raw_runtime_input_value(
        RuntimeInputBindingRequest(
            adapter=runtime,
            kwargs={},
            current_image=measurement_image,
        ).artifact_request_for_spec(spec)
    )

    np.testing.assert_array_equal(payload.labels, labels.labels)


def test_full_stack_object_measurement_resolves_complete_label_artifact() -> None:
    image = np.ones((2, 4, 5), dtype=np.float32)
    image_spec = ArtifactSpec.input("rawDNA", ImageArtifactType)
    object_spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    labels = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=np.stack(
                (
                    np.ones((4, 5), dtype=np.int32),
                    np.full((4, 5), 2, dtype=np.int32),
                )
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    runtime = _FakeCellProfilerRuntime(
        {"rawDNA": image},
        {"Nuclei": labels},
        artifact_input_edges=(_artifact_input_edge_for_test(object_spec),),
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module(
                "MeasureObjectIntensity"
            ).require_callable(),
            artifact_inputs=(image_spec, object_spec),
            artifact_outputs=(
                ArtifactSpec.output("Measurements", MeasurementsArtifactType),
            ),
        )
    )
    _activate_runtime_contract(executor.callable_contract, runtime)

    label_payload = RuntimeArtifactTypeStrategy.for_artifact_type(
        object_spec.artifact_type
    ).raw_runtime_input_value(
        RuntimeInputBindingRequest(
            adapter=runtime,
            kwargs={},
            current_image=image,
        ).artifact_request_for_spec(object_spec)
    )
    _, executable_labels, _, _, _, _, _ = object_measurement_runtime_inputs(
        object_label_execution=object_label_input_execution_mode_from_callable(
            executor.raw_func
        ),
        measurement_image=CellProfilerMeasurementImage(
            source_image_name="rawDNA",
            source_aliases=("rawDNA",),
            payload=image,
        ),
        object_spec=object_spec,
        label_payload=label_payload,
        adapter=runtime,
    )

    np.testing.assert_array_equal(
        object_label_dense_array(executable_labels),
        labels.labels,
    )


def test_object_label_payload_for_measurement_image_projects_source_spatial_crop() -> (
    None
):
    module_name = "MeasureObjectIntensity"
    image_name = "CropBlue"
    object_name = "Cytoplasm"
    image = ImagePayloadMetadata(
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(8, 9),
        ),
    ).payload_with(np.ones((3, 4), dtype=np.float32), None)
    rows = SparseIJVLabelRows(
        np.asarray(
            (
                (2, 3, 1),
                (3, 5, 1),
                (4, 6, 2),
                (7, 8, 3),
            ),
            dtype=np.int32,
        )
    )
    labels = ObjectLabelSet(
        name=object_name,
        variant_data=ObjectLabelVariantData(labels=rows),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(0, 0),
            source_shape_yx=(8, 9),
        ),
        source_image_name=image_name,
    )
    image_spec = ArtifactSpec.input(image_name, ImageArtifactType)
    spec = ArtifactSpec.input(object_name, ObjectLabelsArtifactType)
    runtime = _FakeCellProfilerRuntime(
        {image_name: image},
        {object_name: labels},
        artifact_input_edges=(_artifact_input_edge_for_test(spec),),
    )
    executor = _module_executor(
        _compiled_callable_contract(
            CellProfilerModule.require_module(module_name).require_callable(),
            artifact_inputs=(image_spec, spec),
            artifact_outputs=(
                ArtifactSpec.output("Measurements", MeasurementsArtifactType),
            ),
        )
    )
    _activate_runtime_contract(executor.callable_contract, runtime)

    label_payload = RuntimeArtifactTypeStrategy.for_artifact_type(
        spec.artifact_type
    ).raw_runtime_input_value(
        RuntimeInputBindingRequest(
            adapter=runtime,
            kwargs={},
            current_image=image,
        ).artifact_request_for_spec(spec)
    )
    payload = (
        CellProfilerMeasurementImage(
            source_image_name=image_name,
            source_aliases=(image_name,),
            payload=image,
        )
        .prepare_object_labels(
            label_payload,
            plane_projector=runtime,
        )
        .source_projected_payload
    )

    assert payload.representation is ObjectLabelRepresentation.DENSE_LABELS
    assert payload.spatial_origin_yx == (2, 3)
    assert payload.source_spatial_shape_yx == (8, 9)
    np.testing.assert_array_equal(
        object_label_dense_array(payload),
        np.asarray(
            (
                (1, 0, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 2),
            ),
            dtype=np.int32,
        ),
    )


def test_measurement_image_for_labels_uses_object_label_source_spatial_crop() -> None:
    image = ImagePayloadMetadata(
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(0, 0),
            source_shape_yx=(5, 5),
        ),
    ).payload_with(np.arange(25, dtype=np.float32).reshape(5, 5), None)
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((2, 2), dtype=np.int32)),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(1, 1),
            source_shape_yx=(5, 5),
        ),
    )

    aligned = _measurement_image_for_labels(
        image,
        label_payload.labels,
        label_payload=label_payload,
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    np.testing.assert_array_equal(
        np.asarray(aligned),
        np.asarray([[6.0, 7.0], [11.0, 12.0]], dtype=np.float32),
    )


def test_measurement_table_rows_rejects_type_erased_scalar_measurement() -> None:
    row = {"mean_intensity": 1.5}

    with pytest.raises(TypeError, match="schema-bearing ColumnarRows"):
        measurement_table_rows(row)


def test_filterobjects_relabels_additional_object_inputs_by_primary_retention() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    primary = np.zeros((6, 6), dtype=np.int32)
    primary[0:2, 0:2] = 1
    primary[2:5, 2:5] = 2
    cells = np.zeros_like(primary)
    cells[0:2, 0:2] = 10
    cells[2:5, 2:5] = 11

    result = filter_objects(
        image,
        mode=FilterMode.BORDER,
        object_labels=(primary, cells),
        additional_object_count=1,
        dtype_config=DtypeConfig(),
    )

    (
        _output_image,
        stats,
        filtered_primary,
        filtered_cells,
        primary_relationship,
        cells_relationship,
    ) = result
    stats = stats.rows[0]

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert object_label_dense_array(filtered_primary).max() == 1
    assert object_label_dense_array(filtered_primary)[3, 3] == 1
    assert object_label_dense_array(filtered_cells).max() == 1
    assert object_label_dense_array(filtered_cells)[3, 3] == 1
    assert object_label_dense_array(filtered_cells)[0, 0] == 0
    assert primary_relationship.source_ids == (2,)
    assert primary_relationship.target_ids == (1,)
    assert cells_relationship.source_ids == (11,)
    assert cells_relationship.target_ids == (1,)


def test_filterobjects_removed_output_is_exact_complement() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[0:2, 0:2] = 1
    labels[2:5, 2:5] = 2

    result = filter_objects(
        image,
        mode=FilterMode.BORDER,
        object_labels=(labels,),
        emit_removed_objects=True,
        dtype_config=DtypeConfig(),
    )

    _image, stats, retained, removed, retained_relation, removed_relation = result
    assert stats.rows[0].objects_post_filter == 1
    np.testing.assert_array_equal(
        object_label_dense_array(retained),
        np.where(labels == 2, 1, 0),
    )
    np.testing.assert_array_equal(
        object_label_dense_array(removed),
        np.where(labels == 1, 1, 0),
    )
    assert retained_relation.source_ids == (2,)
    assert retained_relation.target_ids == (1,)
    assert removed_relation.source_ids == (1,)
    assert removed_relation.target_ids == (1,)


def test_filterobjects_runtime_plan_uses_compiled_removed_output_relation() -> None:
    source = ArtifactSpec.input("Objects", ObjectLabelsArtifactType)
    retained = ArtifactSpec.output(
        "Retained",
        ObjectLabelsArtifactType,
        relations=(SourceStackLineageSourceRelation(source=source.ref()),),
    )
    removed = ArtifactSpec.output(
        "Removed",
        ObjectLabelsArtifactType,
        relations=(FilterObjectsRemovedObjectSourceRelation(source=source.ref()),),
    )
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("FilterObjects").require_callable(),
        artifact_inputs=(source,),
        artifact_outputs=(retained, removed),
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            artifact_input_edges=(_artifact_input_edge_for_test(source),),
        ),
        kwargs={},
        current_image=None,
    )

    plan = FilterObjectsRuntimeInputPlan.from_request(request)

    assert plan.object_specs == (source,)
    assert removed not in plan.object_specs


def test_filterobjects_removed_output_relation_rejects_non_object_source() -> None:
    with pytest.raises(ValueError, match="requires an object-label source"):
        FilterObjectsRemovedObjectSourceRelation(
            source=ArtifactSpec.input("Image", ImageArtifactType).ref()
        )


def test_filterobjects_runtime_plan_rejects_removed_output_from_other_input() -> None:
    primary = ArtifactSpec.input("Primary", ObjectLabelsArtifactType)
    other = ArtifactSpec.input("Other", ObjectLabelsArtifactType)
    retained = ArtifactSpec.output(
        "Retained",
        ObjectLabelsArtifactType,
        relations=(SourceStackLineageSourceRelation(source=primary.ref()),),
    )
    removed = ArtifactSpec.output(
        "Removed",
        ObjectLabelsArtifactType,
        relations=(FilterObjectsRemovedObjectSourceRelation(source=other.ref()),),
    )
    contract = _compiled_callable_contract(
        CellProfilerModule.require_module("FilterObjects").require_callable(),
        artifact_inputs=(primary, other),
        artifact_outputs=(retained, removed),
    )
    request = RuntimeInputBindingRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            callable_contract=contract,
            artifact_input_edges=(
                _artifact_input_edge_for_test(primary),
                _artifact_input_edge_for_test(other),
            ),
        ),
        kwargs={},
        current_image=None,
    )

    with pytest.raises(ValueError, match="primary filtered input"):
        FilterObjectsRuntimeInputPlan.from_request(request)


def test_filterobjects_preserves_nominal_domains_until_pair_alignment() -> None:
    primary = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((2, 2), dtype=np.int32)),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(1, 1),
            source_shape_yx=(4, 4),
        ),
    )
    secondary_labels = np.zeros((4, 4), dtype=np.int32)
    secondary_labels[1:3, 1:3] = 5
    secondary = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=secondary_labels),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(0, 0),
            source_shape_yx=(4, 4),
        ),
    )

    result = filter_objects(
        np.zeros((4, 4), dtype=np.float32),
        mode=FilterMode.BORDER,
        object_labels=(primary, secondary),
        additional_object_count=1,
        dtype_config=DtypeConfig(),
    )

    filtered_primary, filtered_secondary = result[2:4]
    assert object_label_dense_array(filtered_primary).shape == (2, 2)
    assert object_label_dense_array(filtered_secondary).shape == (4, 4)
    np.testing.assert_array_equal(
        object_label_dense_array(filtered_primary),
        np.ones((2, 2), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        object_label_dense_array(filtered_secondary)[1:3, 1:3],
        np.ones((2, 2), dtype=np.int32),
    )


def test_relateobjects_preserves_nominal_domains_until_pair_alignment() -> None:
    parent = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((2, 2), dtype=np.int32)),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(1, 1),
            source_shape_yx=(4, 4),
        ),
    )
    child_labels = np.zeros((4, 4), dtype=np.int32)
    child_labels[1:3, 1:3] = 1
    child = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=child_labels),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(0, 0),
            source_shape_yx=(4, 4),
        ),
    )

    result = relate_objects(
        np.zeros((4, 4), dtype=np.float32),
        parent_labels=parent,
        child_labels=child,
        dtype_config=DtypeConfig(),
    )
    _output_labels, relationship, reverse_relationship, measurements = result

    assert relationship.source_ids == (1,)
    assert relationship.target_ids == (1,)
    assert reverse_relationship.source_ids == (1,)
    assert reverse_relationship.target_ids == (1,)
    assert isinstance(measurements, ColumnarRows)
    assert measurements.row_count() == 0


def test_filterobjects_uses_named_measurement_feature_rules() -> None:
    image = np.zeros((5, 5), dtype=np.float32)
    primary = np.zeros((5, 5), dtype=np.int32)
    primary[1:3, 1:3] = 1
    primary[3:5, 3:5] = 2
    measurement_rows = MeasurementSparseColumnarRows.from_rows(
        (
            {"object_label": 1, "lower_quartile_intensity": 0.1},
            {"object_label": 2, "lower_quartile_intensity": 0.8},
        ),
        fields=(
            FieldSpec("object_label", int),
            FieldSpec("lower_quartile_intensity", float),
        ),
    )

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.LIMITS,
        object_labels=(primary,),
        measurement_features=("Intensity_LowerQuartileIntensity_DNA",),
        measurement_min_values=(0.2,),
        measurement_max_values=(None,),
        measurement_use_minimum=(True,),
        measurement_use_maximum=(False,),
        measurement_tables=(
            MeasurementTable(
                name="NucleiMeasurements",
                rows=measurement_rows,
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT, "NucleiMeasurements"
                ),
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_primary = result[:3]
    stats = stats.rows[0]

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert object_label_dense_array(filtered_primary)[1, 1] == 0
    assert object_label_dense_array(filtered_primary)[3, 3] == 1


def test_filterobjects_feature_rules_use_bound_measurement_values() -> None:
    image = np.zeros((5, 5), dtype=np.float32)
    primary = np.zeros((5, 5), dtype=np.int32)
    primary[1:3, 1:3] = 1
    primary[3:5, 3:5] = 2

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.LIMITS,
        object_labels=(primary,),
        measurement_values=np.array([0.1, 0.8], dtype=np.float64),
        measurement_features=("Intensity_LowerQuartileIntensity_DNA",),
        measurement_min_values=(0.2,),
        measurement_max_values=(None,),
        measurement_use_minimum=(True,),
        measurement_use_maximum=(False,),
        measurement_tables=(
            MeasurementTable(
                name="UnrelatedMeasurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"object_label": 1, "AreaShape_Area": 4.0},),
                    fields=(
                        FieldSpec("object_label", int),
                        FieldSpec("AreaShape_Area", float),
                    ),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT, "UnrelatedMeasurements"
                ),
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_primary = result[:3]
    stats = stats.rows[0]

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert object_label_dense_array(filtered_primary)[1, 1] == 0
    assert object_label_dense_array(filtered_primary)[3, 3] == 1


def test_filterobjects_bound_measurement_values_follow_object_domain_order() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    primary = np.zeros((6, 6), dtype=np.int32)
    primary[1:3, 1:3] = 3
    primary[3:5, 3:5] = 5

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.LIMITS,
        object_labels=(primary,),
        measurement_values=np.array([0.1, 0.8], dtype=np.float64),
        measurement_features=("Intensity_LowerQuartileIntensity_DNA",),
        measurement_min_values=(0.2,),
        measurement_max_values=(None,),
        measurement_use_minimum=(True,),
        measurement_use_maximum=(False,),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_primary = result[:3]
    stats = stats.rows[0]

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert object_label_dense_array(filtered_primary)[1, 1] == 0
    assert object_label_dense_array(filtered_primary)[3, 3] == 1


def test_filterobjects_binds_measurements_to_sparse_object_label_ids() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    primary = np.zeros((6, 6), dtype=np.int32)
    primary[1:3, 1:3] = 3
    primary[3:5, 3:5] = 5
    measurement_rows = MeasurementSparseColumnarRows.from_rows(
        (
            {"object_label": 3, "lower_quartile_intensity": 0.1},
            {"object_label": 5, "lower_quartile_intensity": 0.8},
        ),
        fields=(
            FieldSpec("object_label", int),
            FieldSpec("lower_quartile_intensity", float),
        ),
    )

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.LIMITS,
        object_labels=(primary,),
        measurement_features=("Intensity_LowerQuartileIntensity_DNA",),
        measurement_min_values=(0.2,),
        measurement_max_values=(None,),
        measurement_use_minimum=(True,),
        measurement_use_maximum=(False,),
        measurement_tables=(
            MeasurementTable(
                name="NucleiMeasurements",
                rows=measurement_rows,
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT, "NucleiMeasurements"
                ),
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_primary = result[:3]
    stats = stats.rows[0]

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert object_label_dense_array(filtered_primary)[1, 1] == 0
    assert object_label_dense_array(filtered_primary)[3, 3] == 1


def test_filterobjects_keeps_maximal_child_per_enclosing_object() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    children = np.zeros((6, 6), dtype=np.int32)
    children[0:2, 0:2] = 1
    children[0:2, 3:5] = 2
    children[3:5, 0:2] = 3
    children[3:5, 3:5] = 4
    parents = np.zeros_like(children)
    parents[0:2, :] = 1
    parents[3:5, :] = 2

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
        object_labels=(children,),
        enclosing_object_labels=parents,
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    [
                        {"object_label": 1, "AreaShape_Area": 10.0},
                        {"object_label": 2, "AreaShape_Area": 20.0},
                        {"object_label": 3, "AreaShape_Area": 40.0},
                        {"object_label": 4, "AreaShape_Area": 30.0},
                    ],
                    fields=(
                        FieldSpec("object_label", int),
                        FieldSpec("AreaShape_Area", float),
                    ),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT, "ChildMeasurements"
                ),
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]
    stats = stats.rows[0]

    assert stats.objects_pre_filter == 4
    assert stats.objects_post_filter == 2
    assert object_label_dense_array(filtered_children)[0, 0] == 0
    assert object_label_dense_array(filtered_children)[0, 3] == 1
    assert object_label_dense_array(filtered_children)[3, 0] == 2
    assert object_label_dense_array(filtered_children)[3, 3] == 0


def test_filterobjects_filters_by_children_count_relationship() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    nuclei = np.zeros((6, 6), dtype=np.int32)
    nuclei[0:2, 0:2] = 1
    nuclei[2:4, 2:4] = 2
    nuclei[4:6, 4:6] = 3
    declaration = ObjectRelationshipDeclaration.parent_child(
        source=ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType).ref(),
        target=ArtifactSpec.output("PH3", ObjectLabelsArtifactType).ref(),
        producer_module_number=1,
    )
    relationship = ObjectRelationship(
        name="Nuclei_PH3_relationships",
        declaration=declaration,
        payload=DirectedObjectRelationshipPayload(
            source_ids=(1, 3), target_ids=(1, 2), slice_indices=(), slice_count=None
        ),
    )

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.LIMITS,
        object_labels=(nuclei,),
        measurement_features=("Children_PH3_Count",),
        measurement_min_values=(1.0,),
        measurement_max_values=(1.0,),
        measurement_use_minimum=(True,),
        measurement_use_maximum=(False,),
        parent_child_relationships=(relationship,),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_nuclei = result[:3]
    stats = stats.rows[0]

    assert stats.objects_pre_filter == 3
    assert stats.objects_post_filter == 2
    assert object_label_dense_array(filtered_nuclei)[0, 0] == 1
    assert object_label_dense_array(filtered_nuclei)[2, 2] == 0
    assert object_label_dense_array(filtered_nuclei)[4, 4] == 2


def test_filterobjects_both_parents_tie_uses_cellprofiler_pixel_order() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    children = np.zeros((6, 6), dtype=np.int32)
    children[0:2, 0:2] = 2
    children[4:6, 4:6] = 1
    parents = np.ones_like(children)

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
        object_labels=(children,),
        enclosing_object_labels=parents,
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    [
                        {"object_label": 1, "AreaShape_Area": 10.0},
                        {"object_label": 2, "AreaShape_Area": 10.0},
                    ],
                    fields=(
                        FieldSpec("object_label", int),
                        FieldSpec("AreaShape_Area", float),
                    ),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT, "ChildMeasurements"
                ),
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]
    stats = stats.rows[0]

    assert stats.objects_post_filter == 1
    assert object_label_dense_array(filtered_children)[0, 0] == 0
    assert object_label_dense_array(filtered_children)[4, 4] == 1


def test_filterobjects_both_parents_minimal_tie_uses_cellprofiler_pixel_order() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    children = np.zeros((6, 6), dtype=np.int32)
    children[0:2, 0:2] = 2
    children[4:6, 4:6] = 1
    parents = np.ones_like(children)

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MINIMAL_PER_OBJECT,
        object_labels=(children,),
        enclosing_object_labels=parents,
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    [
                        {"object_label": 1, "AreaShape_Area": 10.0},
                        {"object_label": 2, "AreaShape_Area": 10.0},
                    ],
                    fields=(
                        FieldSpec("object_label", int),
                        FieldSpec("AreaShape_Area", float),
                    ),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT, "ChildMeasurements"
                ),
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]
    stats = stats.rows[0]

    assert stats.objects_post_filter == 1
    assert object_label_dense_array(filtered_children)[0, 0] == 1
    assert object_label_dense_array(filtered_children)[4, 4] == 0


def test_filterobjects_both_parents_keeps_single_child_for_sparse_parent() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    children = np.zeros((6, 6), dtype=np.int32)
    children[0:2, 0:2] = 1
    children[4:6, 4:6] = 2
    parents = np.zeros_like(children)
    parents[0:2, 0:2] = 1
    parents[4:6, 4:6] = 3

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
        object_labels=(children,),
        enclosing_object_labels=parents,
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    [
                        {"object_label": 1, "AreaShape_Area": 10.0},
                        {"object_label": 2, "AreaShape_Area": 40.0},
                    ],
                    fields=(
                        FieldSpec("object_label", int),
                        FieldSpec("AreaShape_Area", float),
                    ),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT, "ChildMeasurements"
                ),
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]
    stats = stats.rows[0]

    assert stats.objects_post_filter == 2
    assert object_label_dense_array(filtered_children)[0, 0] == 1
    assert object_label_dense_array(filtered_children)[4, 4] == 2


def test_filterobjects_both_parents_uses_all_pixel_overlaps() -> None:
    image = np.zeros((5, 6), dtype=np.float32)
    children = np.zeros((5, 6), dtype=np.int32)
    children[1:3, 0:2] = 1
    children[1:3, 2:5] = 2
    parents = np.zeros_like(children)
    parents[:, 0:3] = 1
    parents[:, 3:6] = 2

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
        object_labels=(children,),
        enclosing_object_labels=parents,
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    [
                        {"object_label": 1, "AreaShape_Area": 10.0},
                        {"object_label": 2, "AreaShape_Area": 20.0},
                    ],
                    fields=(
                        FieldSpec("object_label", int),
                        FieldSpec("AreaShape_Area", float),
                    ),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT, "ChildMeasurements"
                ),
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]
    stats = stats.rows[0]

    assert stats.objects_post_filter == 1
    assert object_label_dense_array(filtered_children)[1, 0] == 0
    assert object_label_dense_array(filtered_children)[1, 3] == 1


def test_filterobjects_most_overlap_uses_label_geometry() -> None:
    image = np.zeros((5, 6), dtype=np.float32)
    children = np.zeros((5, 6), dtype=np.int32)
    children[1:3, 0:2] = 1
    children[1:3, 2:5] = 2
    parents = np.zeros_like(children)
    parents[:, 0:3] = 1
    parents[:, 3:6] = 2

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
        object_labels=(children,),
        enclosing_object_labels=parents,
        per_object_assignment=PerObjectAssignment.PARENT_WITH_MOST_OVERLAP,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    [
                        {"object_label": 1, "AreaShape_Area": 10.0},
                        {"object_label": 2, "AreaShape_Area": 20.0},
                    ],
                    fields=(
                        FieldSpec("object_label", int),
                        FieldSpec("AreaShape_Area", float),
                    ),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT, "ChildMeasurements"
                ),
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]
    stats = stats.rows[0]

    assert stats.objects_post_filter == 2
    assert object_label_dense_array(filtered_children)[1, 0] == 1
    assert object_label_dense_array(filtered_children)[1, 3] == 2


def test_filterobjects_rejects_undeclared_enclosing_stack_projection() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    children = np.zeros((6, 6), dtype=np.int32)
    children[0:2, 0:2] = 1
    children[0:2, 3:5] = 2
    children[3:5, 0:2] = 3
    children[3:5, 3:5] = 4
    parents = np.zeros_like(children)
    parents[0:2, :] = 1
    parents[3:5, :] = 2

    with pytest.raises(ValueError, match="must share a common geometry"):
        filter_objects(
            image,
            mode=FilterMode.MEASUREMENTS,
            filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
            object_labels=(children,),
            enclosing_object_labels=np.stack((parents, parents)),
            per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
            measurement_features=("AreaShape_Area",),
            measurement_tables=(
                MeasurementTable(
                    name="ChildMeasurements",
                    rows=MeasurementSparseColumnarRows.from_rows(
                        [
                            {"object_label": 1, "AreaShape_Area": 10.0},
                            {"object_label": 2, "AreaShape_Area": 20.0},
                            {"object_label": 3, "AreaShape_Area": 40.0},
                            {"object_label": 4, "AreaShape_Area": 30.0},
                        ],
                        fields=(
                            FieldSpec("object_label", int),
                            FieldSpec("AreaShape_Area", float),
                        ),
                    ),
                    subject=MeasurementSubject(
                        MeasurementScope.ARTIFACT, "ChildMeasurements"
                    ),
                ),
            ),
            dtype_config=DtypeConfig(),
        )


def test_image_morphology_backend_rejects_mismatched_footprint_rank() -> None:
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    with pytest.raises(
        ValueError,
        match="Structuring-element rank exceeds the declared spatial domain",
    ):
        erode_image(image, structuring_element=StructuringElement.BALL, size=1)


def test_structuring_element_execution_uses_callable_processing_contract() -> None:
    kwargs = dict(
        CallableContract.from_callable(closing).validate_public_kwargs(
            {
                "structuring_element": StructuringElement.DISK,
                "size": 17,
            }
        )
    )

    assert kwargs["structuring_element"] is StructuringElement.DISK
    contract = CallableContract.from_callable(closing)
    assert contract.require_processing_contract() is ProcessingContract.FLEXIBLE
    assert contract.runtime_image_execution_mode is ImagePayloadExecutionMode.FULL_STACK
    assert closing.__signature__.parameters["slice_by_slice"].default is True


def test_object_measurement_execution_policy_uses_full_stack_for_3d_labels() -> None:
    policy = CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
        ObjectLabelInputExecutionMode.FULL_STACK
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 5, 5), dtype=np.int32)),
    )

    mode = policy.image_execution_mode(
        labels,
        ImagePayloadExecutionMode.NATURAL,
    )

    assert mode is ImagePayloadExecutionMode.FULL_STACK


def test_object_measurement_execution_policy_uses_full_stack_for_source_bound_volume_labels() -> (
    None
):
    policy = CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
        ObjectLabelInputExecutionMode.FULL_STACK
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.zeros((2, 3, 5, 5), dtype=np.int32)
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    mode = policy.image_execution_mode(
        labels,
        ImagePayloadExecutionMode.NATURAL,
    )

    assert mode is ImagePayloadExecutionMode.FULL_STACK


def test_object_measurement_execution_policy_keeps_payload_domain_labels_full_stack() -> (
    None
):
    policy = CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
        ObjectLabelInputExecutionMode.FULL_STACK
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 5, 5), dtype=np.int32)),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
    )

    mode = policy.image_execution_mode(
        labels,
        ImagePayloadExecutionMode.NATURAL,
        runtime_slice_count=3,
    )

    assert mode is ImagePayloadExecutionMode.FULL_STACK


def test_slice_aligned_measurement_preserves_singleton_aligned_image_owner() -> None:
    policy = CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
        ObjectLabelInputExecutionMode.SLICE_ALIGNED
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((5, 5), dtype=np.int32)),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
    )

    mode = policy.image_execution_mode(
        labels,
        ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        runtime_slice_count=1,
    )

    assert mode is ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK


def test_object_measurement_execution_policy_keeps_declared_full_stack_for_plane_labels() -> (
    None
):
    policy = CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
        ObjectLabelInputExecutionMode.FULL_STACK
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 5, 5), dtype=np.int32)),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((), (), ()),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    mode = policy.image_execution_mode(
        labels,
        ImagePayloadExecutionMode.NATURAL,
        runtime_slice_count=3,
    )

    assert mode is ImagePayloadExecutionMode.FULL_STACK


def test_full_stack_object_measurement_executor_preserves_volume_call() -> None:
    calls: list[tuple[int, ...]] = []

    @object_label_input_execution_mode(ObjectLabelInputExecutionMode.FULL_STACK)
    def measure_volume(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
        del labels
        calls.append(tuple(int(axis) for axis in image.shape))
        return image

    measure_volume.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((3, 5, 7), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)
    callable_contract = _compiled_callable_contract(
        measure_volume,
        artifact_outputs=(
            ArtifactSpec.output("Measurements", MeasurementsArtifactType),
        ),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        measure_volume,
        image,
        {"labels": labels},
        execution_mode=ImagePayloadExecutionMode.FULL_STACK,
    )

    assert calls == [(3, 5, 7)]
    np.testing.assert_array_equal(result, image)


def test_full_stack_image_executor_preserves_volume_call() -> None:
    calls: list[tuple[int, ...]] = []

    @runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
    def filter_volume(image: np.ndarray) -> np.ndarray:
        calls.append(tuple(int(axis) for axis in image.shape))
        return image + 1

    filter_volume.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((3, 5, 7), dtype=np.float32)
    callable_contract = _compiled_callable_contract(
        filter_volume,
        artifact_outputs=(ArtifactSpec.output("Filtered", ImageArtifactType),),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        filter_volume,
        image,
        {},
        execution_mode=ImagePayloadExecutionMode.FULL_STACK,
    )

    assert calls == [(3, 5, 7)]
    np.testing.assert_array_equal(result, image + 1)


def test_convert_objects_to_image_contract_preserves_volume_label_payload() -> None:
    labels = np.zeros((3, 5, 7), dtype=np.int32)
    labels[:, 1:4, 2:5] = 1
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
    )

    assert (
        CallableContract.from_callable(
            convert_objects_to_image
        ).require_processing_contract()
        is ProcessingContract.PURE_3D
    )
    callable_contract = _compiled_callable_contract(
        convert_objects_to_image,
        artifact_outputs=(ArtifactSpec.output("Image", ImageArtifactType),),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        convert_objects_to_image,
        np.zeros((5, 7), dtype=np.float32),
        {
            "labels": label_payload,
            "image_mode": "uint16",
        },
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    np.testing.assert_array_equal(image_payload_data(result), labels)


def _convert_objects_source_planes(
    channel: str,
    plane_count: int = 3,
) -> SourceImageProvenancePlanes:
    return SourceImageProvenancePlanes.from_components(
        paths=tuple(
            f"/tmp/source_w{channel}_z{index}.tif"
            for index in range(1, plane_count + 1)
        ),
        component_metadata=tuple(
            {
                "well": "A01",
                "site": "1",
                "channel": channel,
                "z_index": str(index),
            }
            for index in range(1, plane_count + 1)
        ),
    )


def test_convert_objects_to_image_uses_declared_label_source_for_runtime_plane_domain() -> (
    None
):
    labels = np.zeros((3, 5, 7), dtype=np.int32)
    labels[:, 1:4, 2:5] = 1
    label_spatial_domain = SourceSpatialDomain(
        origin_yx=(4, 6),
        source_shape_yx=(5, 7),
    )
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
        source_image_provenance_planes=_convert_objects_source_planes("2"),
        source_spatial_domain=label_spatial_domain,
    )
    primary_image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=_convert_objects_source_planes("0"),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(0, 0),
            source_shape_yx=(3, 4),
        ),
    ).payload_with(np.zeros((3, 3, 4), dtype=np.float32))

    raw_result = convert_objects_to_image(
        primary_image,
        label_payload,
        image_mode="uint16",
    )
    source_spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    output_plan = ArtifactOutputPlan(
        name="NucleiImage",
        path="/memory/NucleiImage.pkl",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.Z_INDEX,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )
    result = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize_from_projector(
        label_payload,
        raw_result,
        output_plan,
        RuntimePlaneProjection.stack(3),
    )

    metadata = image_payload_metadata(result)
    assert metadata.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert metadata.source_provenance == label_payload.source_provenance
    assert metadata.source_spatial_domain == label_spatial_domain
    assert StreamImagePayloadMetadataProjector.item_fields(
        metadata,
        ("well", "site", "channel", "z_index"),
    )["plane_component_values"] == {"z_index": ("1", "2", "3")}


def test_object_label_image_output_rejects_source_plane_count_drift() -> (
    None
):
    labels = np.zeros((3, 5, 7), dtype=np.int32)
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
        source_image_provenance_planes=_convert_objects_source_planes("2"),
    )
    source_spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    output_plan = ArtifactOutputPlan(
        name="NucleiImage",
        path="/memory/NucleiImage.pkl",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.Z_INDEX,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )
    with pytest.raises(
        ValueError,
        match="source-plane provenance must match the declared runtime plane axis",
    ):
        FunctionOutputContextStrategy.for_output_plan(
            output_plan,
        ).contextualize_from_projector(
            label_payload,
            ImagePayloadMetadata().payload_with(
                np.zeros(labels.shape, dtype=np.uint16)
            ),
            output_plan,
            RuntimePlaneProjection.stack(2),
        )


def test_object_label_singleton_volume_output_declares_runtime_plane_axis() -> None:
    labels = np.zeros((1, 5, 7), dtype=np.int32)
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
        source_image_provenance_planes=_convert_objects_source_planes("2", 1),
    )
    raw_result = convert_objects_to_image(
        np.zeros((5, 7), dtype=np.float32),
        label_payload,
        image_mode="uint16",
    )
    source_spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    output_plan = ArtifactOutputPlan(
        name="NucleiImage",
        path="/memory/NucleiImage.pkl",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.Z_INDEX,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )

    result = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize_from_projector(
        label_payload,
        raw_result,
        output_plan,
        RuntimePlaneProjection.stack(1),
    )

    assert image_payload_data(result).shape == (1, 5, 7)
    assert image_payload_metadata(result).plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_object_label_scalar_image_output_does_not_invent_runtime_plane_axis() -> None:
    labels = np.zeros((5, 7), dtype=np.int32)
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
        source_image_provenance_planes=_convert_objects_source_planes("2", 1),
    )
    raw_result = convert_objects_to_image(
        np.zeros((5, 7), dtype=np.float32),
        label_payload,
        image_mode="uint16",
    )
    source_spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    output_plan = ArtifactOutputPlan(
        name="NucleiImage",
        path="/memory/NucleiImage.pkl",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.Z_INDEX,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )

    result = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize_from_projector(
        label_payload,
        raw_result,
        output_plan,
        RuntimePlaneProjection.stack(1),
    )

    assert image_payload_data(result).shape == (5, 7)
    assert image_payload_metadata(result).plane_axis is None


def test_object_measurement_execution_policy_uses_full_stack_for_single_runtime_slice_volume() -> (
    None
):
    policy = CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
        ObjectLabelInputExecutionMode.FULL_STACK
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 5, 5), dtype=np.int32)),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
    )

    mode = policy.image_execution_mode(
        labels,
        ImagePayloadExecutionMode.NATURAL,
        runtime_slice_count=1,
    )

    assert mode is ImagePayloadExecutionMode.FULL_STACK


def test_measure_object_size_shape_payload_runtime_slice_stack_rows_are_per_slice() -> (
    None
):
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 0:2, 0:2] = 1
    labels[1, 0:3, 0:3] = 1
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (1,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    _image, rows = measure_object_size_shape(
        np.zeros((5, 5), dtype=np.float32),
        payload,
        calculate_advanced=False,
        calculate_zernikes=False,
    )

    assert [(row["slice_index"], row["object_label"], row["Area"]) for row in rows] == [
        (0, 1, 4.0),
        (1, 1, 9.0),
    ]


def test_measure_object_size_shape_rejects_projected_plane_with_stack_metadata() -> (
    None
):
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[1:4, 1:4] = 1

    with pytest.raises(
        ValueError,
        match=r"declares 3 plane\(s\), but dense label storage has shape \(5, 5\)",
    ):
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            domain=ObjectLabelDomain(
                declared_object_id_domains=((1,), (1,), (1,)),
                scope=ObjectLabelDomainScope.PLANE,
            ),
        )


def test_measure_object_size_shape_rejects_mismatched_plane_metadata() -> None:
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 0:2, 0:2] = 1
    labels[1, 0:3, 0:3] = 1

    with pytest.raises(
        ValueError,
        match=(r"declares 4 plane\(s\), but dense label storage has shape \(2, 5, 5\)"),
    ):
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            domain=ObjectLabelDomain(
                declared_object_id_domains=((1,), (1,), (1,), (1,)),
                scope=ObjectLabelDomainScope.PLANE,
            ),
        )


def test_measure_object_size_shape_plane_scoped_singleton_uses_2d_schema() -> None:
    labels = np.zeros((1, 5, 5), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    _image, rows = measure_object_size_shape(
        np.zeros((1, 5, 5), dtype=np.float32),
        payload,
        calculate_advanced=False,
        calculate_zernikes=True,
    )

    assert len(rows) == 1
    assert rows[0]["slice_index"] == 0
    assert rows[0]["object_label"] == 1
    assert rows[0]["Area"] == 9.0
    assert "Zernike_0_0" in rows[0]
    assert "Volume" not in rows[0]
    assert "Center_Z" not in rows[0]
    assert tuple(rows[0]) == tuple(field.name for field in rows.fields)


def test_measure_object_size_shape_preserves_declared_repeated_plane_domains() -> None:
    plane = np.zeros((5, 5), dtype=np.int32)
    plane[0:2, 0:2] = 1
    labels = np.zeros((2, 2, 5, 5), dtype=np.int32)
    labels[0, 0] = plane
    labels[1, 1] = plane
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (1,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    _image, rows = measure_object_size_shape(
        np.zeros((5, 5), dtype=np.float32),
        payload,
        calculate_advanced=False,
        calculate_zernikes=False,
    )

    assert [
        (row["slice_index"], row["object_label"], row["Volume"]) for row in rows
    ] == [
        (0, 1, 4.0),
        (1, 1, 4.0),
    ]


def test_measure_object_size_shape_payload_scoped_volume_rows_are_3d() -> None:
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[:, 1:4, 1:4] = 1
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(
            declared_object_ids=(1,),
            scope=ObjectLabelDomainScope.PAYLOAD,
        ),
    )
    label_set = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(
            declared_object_ids=(1,),
            scope=ObjectLabelDomainScope.PAYLOAD,
        ),
    )

    for label_value in (payload, label_set):
        _image, rows = measure_object_size_shape(
            np.zeros((2, 5, 5), dtype=np.float32),
            label_value,
            calculate_advanced=False,
            calculate_zernikes=False,
        )

        assert len(rows) == 1
        assert rows[0]["object_label"] == 1
        assert rows[0]["Volume"] == 18.0
        inertia = skimage.measure.regionprops_table(
            labels,
            properties=("inertia_tensor_eigvals",),
        )
        assert rows[0]["MajorAxisLength"] == pytest.approx(
            4.0 * np.sqrt(inertia["inertia_tensor_eigvals-0"][0])
        )
        assert rows[0]["MinorAxisLength"] == pytest.approx(
            4.0 * np.sqrt(inertia["inertia_tensor_eigvals-2"][0])
        )
        assert "BoundingBoxMinimum_Z" in rows[0]
        assert "Area" not in rows[0]


def test_measure_object_size_shape_plane_scoped_volume_rows_are_3d() -> None:
    labels = np.zeros((2, 3, 5, 5), dtype=np.int32)
    labels[0, :, 0:2, 0:2] = 1
    labels[1, :, 0:3, 0:3] = 1
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (1,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    _image, rows = measure_object_size_shape(
        np.zeros((3, 5, 5), dtype=np.float32),
        payload,
        calculate_advanced=False,
        calculate_zernikes=False,
    )

    assert [
        (row["slice_index"], row["object_label"], row["Volume"]) for row in rows
    ] == [
        (0, 1, 12.0),
        (1, 1, 27.0),
    ]
    assert all("BoundingBoxMinimum_Z" in row for row in rows)
    assert all("Area" not in row for row in rows)


def test_measure_object_size_shape_surface_areas_match_marching_cubes_oracle() -> None:
    labels = np.zeros((4, 5, 6), dtype=np.int32)
    labels[0:3, 1:4, 1:4] = 1
    labels[1:4, 2:5, 3:5] = 2
    label_ids = np.array([1, 2], dtype=np.int32)

    expected = []
    for label_id in label_ids:
        positions = np.argwhere(labels == label_id)
        minimum = positions.min(axis=0)
        maximum = positions.max(axis=0) + 1
        bounds = tuple(
            slice(
                max(int(minimum[axis]) - 1, 0),
                min(int(maximum[axis]) + 1, labels.shape[axis]),
            )
            for axis in range(labels.ndim)
        )
        expected.append(_surface_area(labels[bounds] == label_id))

    np.testing.assert_allclose(
        _surface_areas_3d_from_labels(labels, label_ids),
        np.asarray(expected, dtype=np.float64),
        rtol=1e-6,
        atol=1e-5,
    )


def test_resize_objects_3d_drops_cellprofiler_parent_image_spacing() -> None:
    labels = np.zeros((3, 4, 5), dtype=np.int32)
    labels[:, 1:3, 1:4] = 1
    source = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
        parent_image_source_voxel_spacing=SourceVoxelSpacing((2.0, 1.0, 1.0)),
    )

    _image, _stats, resized, _relationship = resize_objects_3d(
        np.zeros(labels.shape, dtype=np.float32),
        source,
        factor_x=1.0,
        factor_y=1.0,
        factor_z=1.0,
    )

    assert resized.parent_image_source_voxel_spacing == SourceVoxelSpacing()


def test_object_measurement_execution_policy_keeps_declared_full_stack_for_2d_labels() -> (
    None
):
    policy = CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
        ObjectLabelInputExecutionMode.FULL_STACK
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=np.zeros((5, 5), dtype=np.int32)),
    )

    mode = policy.image_execution_mode(
        labels,
        ImagePayloadExecutionMode.NATURAL,
    )

    assert mode is ImagePayloadExecutionMode.FULL_STACK

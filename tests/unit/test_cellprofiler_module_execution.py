from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import skimage.measure
import skimage.morphology

from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadBundleContext,
    ImagePayloadSliceProjector,
    ImagePayloadExecutionMode,
    compose_aligned_image_payload,
    aligned_image_stack_kwargs,
    payload_slice_count,
    payload_slices_for_alignment,
)
from openhcs.core.callable_contract import (
    CallableContract,
    reset_processing_callable_preparation_cache,
    runtime_image_execution_mode,
)
from openhcs.core.pipeline.function_contracts import runtime_bound_parameters
from openhcs.core.pipeline_image_schema import SOURCE_IMAGE_TYPE_METADATA_FIELD
from openhcs.core.runtime_invocation import SliceIndexRuntimeParameter
from openhcs.core.function_patterns import (
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
    FunctionInvocationKey,
)
from openhcs.core.runtime_adapters import RuntimeExecutionAxisScope
from openhcs.core.runtime_slice_projection import (
    RuntimeSliceProjection,
    RuntimeProjectionAxis,
)
from openhcs.core.steps.function_runtime import prepare_compiled_function_group
from openhcs.core.runtime_artifact_queries import (
    MeasurementTableAxisProjection,
)
from openhcs.core.source_matching import SourceAxisMetadataScope
from openhcs.core.source_metadata import SourceVoxelSpacing
from openhcs.core.measurement_row_materialization import (
    columnar_row_values,
)
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingRuntimeContext,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceSelector,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    MeasurementRowAxisState,
    ObjectLabelDomain,
    ObjectLabelRepresentation,
)
from openhcs.core.pipeline.function_contracts import (
    composed_image_payload,
    ObjectLabelMeasurementExecution,
    object_label_measurement_execution,
)
from openhcs.core.measurement_image_alignment import (
    MeasurementImageLabelAlignmentStrategy,
    MeasurementLabelSourceAlignmentStrategy,
    PreparedMeasurementObjectLabels,
)
from openhcs.constants.constants import (
    AllComponents,
    GroupBy,
    MemoryType,
    VariableComponents,
)
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerImageRequest,
    CellProfilerInvocationRequest,
    CellProfilerMeasurementImage,
    CellProfilerSliceAlignedValues,
)
from openhcs.interop.cellprofiler.runtime.processing_contracts import (
    Pure2DSliceCountPolicy,
)
from openhcs.interop.cellprofiler.runtime.main_flow import (
    CELLPROFILER_MEASUREMENT_MAIN_FLOW,
    CELLPROFILER_SIDE_EFFECT_MAIN_FLOW,
    cellprofiler_main_flow_output,
    cellprofiler_recorded_image_main_flow_output,
)
from openhcs.processing.backends.cellprofiler.grid import (
    DefineGridCycleScope,
    DefineGridInvocationOptions,
)
from openhcs.interop.cellprofiler.runtime.execution_mode_policies import (
    CellProfilerInvocationExecutionModePolicy,
)
from openhcs.interop.cellprofiler.runtime.adapter import (
    CellProfilerRuntimeAdapter,
)
from openhcs.interop.cellprofiler.runtime.source_candidates import (
    ParsedSourceCandidate,
    SourceBindingImageSetMatchScope,
    SourceBindingMatchCandidateUniverse,
    SourceCandidateMatcher,
)
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerFunctionContractExecutor,
    CellProfilerFunctionOutputAggregationContract,
    CellProfilerRuntimeCallable,
    CellProfilerPrimaryImageInputPolicy,
    CellProfilerMeasurementImageDomain,
    CellProfilerModuleExecutor,
    CellProfilerModuleRuntimePlan,
    DefaultPrimaryImageInputPolicy,
    ObjectInputBindingRequest,
    ObjectLocationMeasurementRows,
    RelationshipMeasurementRows,
    RuntimeArtifactBindingScope,
    CellProfilerOutputRecordRequest,
    CellProfilerOutputRecorder,
    CellProfilerProcessingContractAuthority,
    CellProfilerPure2DOutputAggregator,
    CellProfilerSpecialInputPolicy,
    CallableInvocationKwargSpec,
    CellProfilerMeasurementImageResolver,
    CurrentImageObjectLabelPlaneAlignment,
    SpecialInputBindingRequest,
    measurement_table_rows,
    OBJECT_ONLY_REFERENCE_IMAGE,
    _unstack_cellprofiler_image_slices,
)
from openhcs.processing.backends.cellprofiler.alignment import AlignModule
from openhcs.processing.backends.cellprofiler.classification import (
    ClassifyObjectsSingleMeasurementModule,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    FieldsFromRowsMeasurementRecordMixin,
    measurement_record_for_module,
)
from openhcs.processing.backends.cellprofiler.library import canonical_module_name
from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
    CellProfilerMeasurementFieldSchema,
    CellProfilerMeasurementMaterializationRequest,
    CellProfilerMeasurementMaterializer,
    CellProfilerMeasurementOutputAxisState,
    CellProfilerMeasurementOutputProjection,
    CellProfilerMeasurementProjectionRequest,
    CellProfilerMeasurementRecord,
    CellProfilerMeasurementSourceContext,
    CellProfilerMeasurementSourceResolver,
    CellProfilerProjectedMeasurementRow,
    AdapterObjectLabelSourceLookup,
)
from openhcs.processing.backends.cellprofiler.measurement_math import (
    CalculateMathInputPolicy,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    CombineObjectsInputPolicy,
    resize_objects_3d,
)
from openhcs.processing.backends.cellprofiler.watershed import WatershedModule
from openhcs.interop.cellprofiler.runtime.relationship_endpoints import (
    RelationshipEndpointContract,
    RelationshipEndpointResolver,
)
from openhcs.interop.cellprofiler.runtime.measurement_execution_support import (
    project_object_label_payload_for_measurement_image,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    ConcatenatedMeasurementColumnarRows,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_execution import (
    CellProfilerObjectMeasurementExecutionDomainPolicy,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    CellProfilerObjectMeasurementRowPolicy,
    CompactMeasuredObjectMeasurementRowPolicy,
    DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy,
    DefaultObjectMeasurementRowPolicy,
)
from openhcs.interop.cellprofiler.runtime.output_contexts import (
    CellProfilerImageOutputContextStrategy,
    CellProfilerImageOutputSourcePayloadPolicy,
    DefaultImageOutputSourcePayloadPolicy,
    CellProfilerObjectLabelOutputContextStrategy,
)
from openhcs.interop.cellprofiler.runtime.output_value_resolution import (
    CellProfilerOutputValueResolutionRequest,
)
from openhcs.interop.cellprofiler.runtime.projection_requirements import (
    CellProfilerRuntimePlaneProjectionCapability,
    CurrentSourceImagePayloadProjectionCapability,
    RuntimeArtifactImageInputProjectionCapability,
    RuntimeArtifactValueProjectionCapability,
    RuntimeSliceKwargProjectionCapability,
)
from openhcs.interop.cellprofiler.runtime.runtime_plane_kwargs import (
    CurrentRuntimePlaneKwargProjection,
    CurrentRuntimePlaneKwargProjectionContract,
)
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
)
from openhcs.processing.backends.cellprofiler.color import (
    color_to_gray as openhcs_color_to_gray,
    gray_to_color,
)
from openhcs.core.source_image_semantics import source_image_payload_role
from benchmark.cellprofiler_library.functions.colortogray import color_to_gray
from benchmark.cellprofiler_library.functions.crop import crop
from benchmark.cellprofiler_library.functions.correctilluminationapply import (
    correct_illumination_apply,
)
from benchmark.cellprofiler_library.functions.align import AlignShiftMeasurement
from benchmark.cellprofiler_library.functions.filterobjects import (
    FilterMethod,
    FilterMode,
    FilterObjectsStats,
    PerObjectAssignment,
    filter_objects,
)
from benchmark.cellprofiler_library.functions.enhanceorsuppressfeatures import (
    EnhanceMethod,
    NeuriteMethod,
    SpeckleAccuracy,
    enhance_or_suppress_features,
)
from benchmark.cellprofiler_library.functions.expandorshrinkobjects import (
    expand_or_shrink_objects,
)
from benchmark.cellprofiler_library.functions.classifyobjects import (
    ClassificationResult,
)
from benchmark.cellprofiler_library.functions.threshold import ThresholdResult
from benchmark.cellprofiler_library.functions.relateobjects import (
    RelateObjectsResult,
    RelationshipMeasurements,
)
from benchmark.cellprofiler_library.functions.identifyprimaryobjects import (
    ExcessObjectHandling,
    FillHolesOption,
    UnclumpMethod,
    identify_primary_objects,
)
from openhcs.processing.backends.cellprofiler.primary_objects import (
    _remap_object_label_variant_after_final_relabel,
)
from benchmark.cellprofiler_library.functions.definegrid import define_grid_automatic
from benchmark.cellprofiler_library.functions.identifyobjectsingrid import (
    identify_objects_in_grid,
    identify_objects_in_grid_with_guides,
)
from openhcs.processing.backends.cellprofiler.grid import (
    GridDefinition,
    GridShapeRequest,
    NaturalGridShapeStrategy,
)
from benchmark.cellprofiler_library.functions.measureobjectsizeshape import (
    measure_object_size_shape,
)
from openhcs.processing.backends.cellprofiler.shape import (
    MeasureObjectSizeShapeModule,
    _surface_area,
    _surface_areas_3d_from_labels,
)
from openhcs.processing.backends.cellprofiler.crop import CropModule
from benchmark.cellprofiler_library.functions.maskimage import mask_image
from benchmark.cellprofiler_library.functions.maskobjects import mask_objects
from benchmark.cellprofiler_library.functions import identifysecondaryobjects as iso
from benchmark.cellprofiler_library.functions import identifytertiaryobjects as ito
from benchmark.cellprofiler_library.functions.identifysecondaryobjects import (
    DistanceMaskedSegmentationStrategy,
    PropagationSegmentationStrategy,
    SecondarySegmentationRequest,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    _filter_labels,
    _secondary_seed_labels,
)
from benchmark.cellprofiler_library.functions.tile import tile
from benchmark.cellprofiler_library.functions.watershed import watershed
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSidecarRole,
    ArtifactSpec,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
    NoMainFlowOutput,
    RelationshipsArtifactType,
    SpatialGridArtifactType,
)
from openhcs.core.callable_contract import attach_callable_contract_metadata
from openhcs.core.config import DtypeConfig
from openhcs.core.module_artifact_contract import (
    DeclaredArtifactOutputPartition,
    ModuleArtifactContract,
    RecordedArtifactOutputPartition,
    RuntimeArtifactInputPartition,
    SourceArtifactInputPartition,
)
from openhcs.core.pipeline.function_contracts import (
    special_input_names_from_callable,
    special_inputs,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    Pure2DSliceResultBatch,
)
from openhcs.core.runtime_invocation import RuntimeSliceAlignedValues
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_semantics import (
    MeasurementObjectRowIdentity,
    ObjectLocationMeasurementFeature,
    ObjectLabelDomainScope,
    ParentChildRelationshipPayload,
    RelationshipSemantics,
    RuntimePlaneAxis,
    RuntimePlaneProjection,
    RuntimePlaneAxisProjector,
    SpatialGridOrdering,
    object_label_parent_child_payload,
)
from openhcs.core.runtime_values import (
    ColumnarRows,
    FieldSpec,
    ImagePayloadMetadata,
    ImageMetadataPayload,
    MaskedImagePayload,
    MeasurementTable,
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectLabelValue,
    ObjectRelationship,
    SingletonObjectLabelStackCollapseStrategy,
    SparseIJVLabelRows,
    SourceImageObjectLabelBuildRequest,
    SpatialGrid,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    RuntimeImagePayloadContext,
    object_label_dense_array,
    SourceImageProvenancePlanes,
)
from openhcs.core.steps.function_runtime import (
    FunctionOutputContextStrategy,
    PatternGroupData,
    PatternGroupRuntime,
)
from openhcs.core.steps.function_output_identity import FunctionOutputIdentityCache
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
    relate_objects,
)
from openhcs.processing.backends.cellprofiler.intensity import measure_object_intensity
from openhcs.processing.backends.cellprofiler.object_images import (
    convert_objects_to_image,
)
from openhcs.processing.materialization import csv_materializer

EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE = RuntimeArtifactBindingScope(
    external_image_names=frozenset(),
    external_object_names=frozenset(),
    runtime_image_names=frozenset(),
)


def _resolved_output_values(
    output_specs: tuple[ArtifactSpec, ...],
    main_output: object,
    artifact_values: tuple[object, ...],
    *,
    func: object = None,
    declared_output_specs: tuple[ArtifactSpec, ...] = (),
) -> dict[str, object]:
    return CellProfilerOutputValueResolutionRequest(
        output_specs=output_specs,
        main_output=main_output,
        artifact_values=artifact_values,
        func=func,
        declared_output_specs=declared_output_specs,
    ).values_by_name()


def _output_record_request_runtime_plan_func(
    contract: ModuleArtifactContract,
    func,
):
    object_inputs = contract.declared_input_collection().of_artifact_type(
        ObjectLabelsArtifactType
    )
    if not object_inputs or special_input_names_from_callable(func):
        return func

    @special_inputs(
        *(f"object_input_{index}" for index, _spec in enumerate(object_inputs))
    )
    def plan_func(image, **_kwargs):
        return image

    return plan_func


def _cellprofiler_output_record_request(
    *,
    executor: CellProfilerModuleExecutor | None = None,
    module_name: str = "TestCellProfilerModule",
    inputs: tuple[ArtifactSpec, ...] = (),
    runtime_artifact_inputs: tuple[ArtifactSpec, ...] = (),
    outputs: tuple[ArtifactSpec, ...] | None = None,
    declared_outputs: tuple[ArtifactSpec, ...] = (),
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
    contract = (
        executor.contract
        if executor is not None
        else ModuleArtifactContract(
            module_name=module_name,
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition, inputs
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition, runtime_artifact_inputs
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition, outputs or (spec,)
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition, declared_outputs
                ),
            ),
        )
    )
    plan_func = _output_record_request_runtime_plan_func(contract, kwargs["func"])
    if plan_func.__module__ == __name__:
        plan_func.__processing_contract__ = ProcessingContract.PURE_2D
    runtime_plan = CellProfilerModuleRuntimePlan.build(
        contract=contract,
        canonical_module_name=canonical_module_name(contract.module_name),
        primary_image_input_policy=CellProfilerPrimaryImageInputPolicy.for_module(
            contract.module_name
        ),
        func=plan_func,
        processing_contract=CellProfilerProcessingContractAuthority.for_callable(
            plan_func
        ),
    )
    return CellProfilerOutputRecordRequest(
        runtime_plan=runtime_plan,
        source=source,
        **kwargs,
    )


@dataclass(frozen=True, slots=True)
class _FakeRuntimeImage:
    data: np.ndarray
    source_image_name: str | None = None


@dataclass(frozen=True, slots=True)
class _SyntheticObjectMeasurement:
    object_label: int
    value: float


@dataclass(frozen=True, slots=True)
class _RuntimeSliceObjectAdapter(RuntimePlaneAxisProjector):
    objects: ObjectLabelSet
    slice_index: int
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

    def runtime_slice_plane_index(self) -> int:
        return self.slice_index

    def runtime_slice_axis_size(self) -> int | None:
        return self.slice_count


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


def _source_candidate(alias: str, site: str) -> ParsedSourceCandidate:
    filename = f"A01_s{site}_{alias}.tif"
    return ParsedSourceCandidate(
        path=filename,
        resolved_path=f"/plate/{filename}",
        filename=filename,
        metadata={
            "Well": "A01",
            "Site": site,
            "Channel": alias,
        },
    )


def test_cellprofiler_executor_warmup_runs_from_compiled_group_preparation(monkeypatch):
    calls = []

    def raw_cellprofiler_module(image):
        return image

    def record_prepare(self, func):
        calls.append((self.contract.module_name, func.__name__))

    monkeypatch.setattr(CellProfilerModuleExecutor, "prepare", record_prepare)
    reset_processing_callable_preparation_cache()

    runtime_callable = CellProfilerRuntimeCallable(
        raw_cellprofiler_module,
        ModuleArtifactContract("WarmupModule"),
        processing_contract=ProcessingContract.FLEXIBLE,
    )
    contract = CallableContract.from_callable(runtime_callable)
    invocation = CompiledFunctionInvocation(
        key=FunctionInvocationKey.from_contract(contract, "default", 0),
        contract=contract,
    )

    prepare_compiled_function_group(
        CompiledFunctionGroup(
            group_key="default",
            invocations=(invocation,),
        )
    )

    assert calls == [("WarmupModule", "raw_cellprofiler_module")]


def test_primary_image_policy_registry_uses_default_for_maskimage() -> None:
    assert isinstance(
        CellProfilerPrimaryImageInputPolicy.for_module("IdentifyPrimaryObjects"),
        DefaultPrimaryImageInputPolicy,
    )
    assert isinstance(
        CellProfilerPrimaryImageInputPolicy.for_module("MaskImage"),
        DefaultPrimaryImageInputPolicy,
    )


def test_source_candidate_matcher_allows_grouped_metadata_values() -> None:
    step_input_candidates = (
        _source_candidate("OrigActin", "1"),
        _source_candidate("OrigActin", "2"),
    )
    target_candidates = (
        _source_candidate("OrigDNA", "1"),
        _source_candidate("OrigDNA", "2"),
        _source_candidate("OrigDNA", "3"),
    )
    full_pipeline_candidates = step_input_candidates + target_candidates
    plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                "OrigActin",
                selector=SourceSelector(inherit_current_scope=False),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
            NamedSourceBinding(
                "OrigDNA",
                selector=SourceSelector(inherit_current_scope=False),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        ),
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField("OrigActin", "Well"),
                        SourceBindingMatchField("OrigDNA", "Well"),
                    )
                ),
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField("OrigActin", "Site"),
                        SourceBindingMatchField("OrigDNA", "Site"),
                    )
                ),
            ),
        ),
    )

    matched = SourceBindingMatchCandidateUniverse(
        step_input_candidates=step_input_candidates,
        target_candidates=target_candidates,
        pipeline_candidates=full_pipeline_candidates,
    ).image_set_candidates(
        "OrigDNA",
        SourceBindingImageSetMatchScope(
            plan=plan.match_plan,
            bindings=plan.bindings,
        ),
    )

    assert tuple(candidate.metadata["Site"] for candidate in matched) == ("1", "2")


def test_source_candidate_matcher_metadata_selector_overrides_semantic_inherited_scope() -> (
    None
):
    candidates = (
        ParsedSourceCandidate(
            path="A01_s001_w1_z001_t001.tif",
            resolved_path="/plate/A01_s001_w1_z001_t001.tif",
            filename="A01_s001_w1_z001_t001.tif",
            metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
                "ChannelNumber": "1",
            },
        ),
        ParsedSourceCandidate(
            path="A01_s001_w3_z001_t001.tif",
            resolved_path="/plate/A01_s001_w3_z001_t001.tif",
            filename="A01_s001_w3_z001_t001.tif",
            metadata={
                "well": "A01",
                "site": "1",
                "channel": "3",
                "ChannelNumber": "3",
            },
        ),
    )
    binding = NamedSourceBinding(
        "OrigSyto",
        selector=SourceSelector(
            metadata=(MetadataSelector("ChannelNumber", "3"),),
        ),
    )

    matched = SourceCandidateMatcher.match_candidates(
        candidates=candidates,
        binding=binding,
        inherit_components={
            "well": "A01",
            "site": "1",
            "channel": "1",
        },
    )

    assert tuple(candidate.path for candidate in matched) == (
        "A01_s001_w3_z001_t001.tif",
    )


def test_special_object_label_input_preserves_runtime_slice_domain() -> None:
    labels = np.array(
        [
            [[0, 1], [0, 0]],
            [[0, 0], [2, 0]],
        ],
        dtype=np.int32,
    )
    objects = ObjectLabelSet(
        name="Guides",
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    request = SpecialInputBindingRequest(
        module_name="IdentifyObjectsInGrid",
        adapter=_RuntimeSliceObjectAdapter(objects=objects, slice_index=1),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
        parameter_names=("guiding_labels",),
        special_input_specs=(ArtifactSpec.input("Guides", ObjectLabelsArtifactType),),
        runtime_inputs=(ArtifactSpec.input("Guides", ObjectLabelsArtifactType),),
    )

    bound = request.bind_positional_parameters()

    assert isinstance(bound["guiding_labels"], np.ndarray)
    np.testing.assert_array_equal(bound["guiding_labels"], labels)
    np.testing.assert_array_equal(
        RuntimeSliceProjection.kwargs_for_slice(
            bound,
            RuntimeProjectionAxis(slice_index=1, extent=2),
        )["guiding_labels"],
        np.array([[0, 0], [2, 0]], dtype=np.int32),
    )


def test_special_object_label_payload_preserves_full_stack_context() -> None:
    class ScopeAwareObjectAdapter(_RuntimeSliceObjectAdapter):
        def __init__(self, objects: ObjectLabelSet) -> None:
            super().__init__(objects=objects, slice_index=1)
            object.__setattr__(self, "current_image_requests", [])

        def get_objects(self, name: str, current_image=None) -> ObjectLabelSet:
            del name
            self.current_image_requests.append(current_image)
            if current_image is None:
                return self.objects
            return ObjectLabelSet(
                name=self.objects.name,
                labels=self.objects.labels[0],
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
                source_image_provenance_planes=SourceImageProvenancePlanes(),
                source_component_metadata={
                    "well": "A01",
                    "site": "1",
                    "channel": "3",
                    "z_index": "1",
                },
            )

    labels = np.zeros((2, 4, 4), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 0:2, 0:2] = 2
    objects = ObjectLabelSet(
        name="Nuclei",
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
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
    adapter = ScopeAwareObjectAdapter(objects)
    spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    request = SpecialInputBindingRequest(
        module_name="ConvertObjectsToImage",
        adapter=adapter,
        kwargs={},
        current_image=np.zeros((4, 4), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
        parameter_names=("labels",),
        special_input_specs=(spec,),
        runtime_inputs=(spec,),
    )

    payload = request.object_label_payload(spec)

    assert payload is objects
    assert adapter.current_image_requests == [None]
    assert payload.source_image_provenance_planes.paths == (
        "/plate/A01_s001_w3_z001_t001.tif",
        "/plate/A01_s001_w3_z002_t001.tif",
    )


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
        labels=labels,
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
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    request = SpecialInputBindingRequest(
        module_name="RelateObjects",
        adapter=_RuntimeSliceObjectAdapter(objects=objects, slice_index=1),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
        parameter_names=("parent_labels", "child_labels"),
        special_input_specs=(
            ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
            ArtifactSpec.input("Children", ObjectLabelsArtifactType),
        ),
        runtime_inputs=(
            ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
            ArtifactSpec.input("Children", ObjectLabelsArtifactType),
        ),
    )

    bound = CellProfilerSpecialInputPolicy.for_module("RelateObjects").bind(request)

    np.testing.assert_array_equal(bound["parent_labels"], labels[1])
    np.testing.assert_array_equal(bound["child_labels"], labels[1])


def test_relateobjects_special_inputs_bind_runtime_slice_index() -> None:
    labels = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [2, 0]],
        ],
        dtype=np.int32,
    )
    objects = ObjectLabelSet(
        name="Objects",
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    request = SpecialInputBindingRequest(
        module_name="RelateObjects",
        func=relate_objects,
        adapter=_RuntimeSliceObjectAdapter(objects=objects, slice_index=1),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
        parameter_names=("parent_labels", "child_labels"),
        special_input_specs=(
            ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
            ArtifactSpec.input("Children", ObjectLabelsArtifactType),
        ),
        runtime_inputs=(
            ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
            ArtifactSpec.input("Children", ObjectLabelsArtifactType),
        ),
    )

    bound = CellProfilerSpecialInputPolicy.for_module("RelateObjects").bind(request)

    assert bound["slice_index"] == 1


def test_relateobjects_special_inputs_project_source_binding_labels_to_current_plane() -> (
    None
):
    class SourceAxisObjectAdapter(_RuntimeSliceObjectAdapter):
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
        labels=labels,
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("Site1", "Site2"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    request = SpecialInputBindingRequest(
        module_name="RelateObjects",
        func=relate_objects,
        adapter=SourceAxisObjectAdapter(objects=objects, slice_index=0),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
        parameter_names=("parent_labels", "child_labels"),
        special_input_specs=(
            ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
            ArtifactSpec.input("Children", ObjectLabelsArtifactType),
        ),
        runtime_inputs=(
            ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
            ArtifactSpec.input("Children", ObjectLabelsArtifactType),
        ),
    )

    bound = CellProfilerSpecialInputPolicy.for_module("RelateObjects").bind(request)

    np.testing.assert_array_equal(bound["parent_labels"], labels[1])
    np.testing.assert_array_equal(bound["child_labels"], labels[1])
    assert "slice_index" not in bound


def test_relateobjects_special_inputs_do_not_bind_source_axis_slice_index() -> None:
    class SourceAxisObjectAdapter(_RuntimeSliceObjectAdapter):
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
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_names=("Stain1",),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    request = SpecialInputBindingRequest(
        module_name="RelateObjects",
        func=relate_objects,
        adapter=SourceAxisObjectAdapter(objects=objects, slice_index=1),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
        parameter_names=("parent_labels", "child_labels"),
        special_input_specs=(
            ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
            ArtifactSpec.input("Children", ObjectLabelsArtifactType),
        ),
        runtime_inputs=(
            ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
            ArtifactSpec.input("Children", ObjectLabelsArtifactType),
        ),
    )

    bound = CellProfilerSpecialInputPolicy.for_module("RelateObjects").bind(request)

    assert "slice_index" not in bound


def test_relateobjects_special_inputs_do_not_bind_composed_source_axis_slice_index() -> (
    None
):
    labels = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [2, 0]],
        ],
        dtype=np.int32,
    )
    objects = ObjectLabelSet(
        name="Objects",
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_name="OrigStain1__OrigStain2",
        source_image_names=("OrigStain1", "OrigStain1"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    request = SpecialInputBindingRequest(
        module_name="RelateObjects",
        func=relate_objects,
        adapter=_RuntimeSliceObjectAdapter(objects=objects, slice_index=1),
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
        parameter_names=("parent_labels", "child_labels"),
        special_input_specs=(
            ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
            ArtifactSpec.input("Children", ObjectLabelsArtifactType),
        ),
        runtime_inputs=(
            ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
            ArtifactSpec.input("Children", ObjectLabelsArtifactType),
        ),
    )

    bound = CellProfilerSpecialInputPolicy.for_module("RelateObjects").bind(request)

    assert "slice_index" not in bound


def test_maskimage_object_label_alignment_uses_image_set_identity() -> None:
    image = RuntimeImagePayloadContext(
        np.zeros((2, 2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("site1_rna.tif", "site2_rna.tif"),
                component_metadata=(
                    {"Well": "A14", "Site": "1", "ChannelNumber": "3"},
                    {"Well": "A14", "Site": "2", "ChannelNumber": "3"},
                ),
            )
        ),
        mask=None,
    ).payload()
    labels = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [2, 0]],
        ],
        dtype=np.int32,
    )
    objects = ObjectLabelSet(
        name="Nuclei",
        labels=labels,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("site1_dna.tif", "site2_dna.tif"),
            component_metadata=(
                {"Well": "A14", "Site": "1", "ChannelNumber": "1"},
                {"Well": "A14", "Site": "2", "ChannelNumber": "1"},
            ),
        ),
    )

    aligned = CurrentImageObjectLabelPlaneAlignment(
        adapter=SimpleNamespace(),
        current_image=image,
        labels=objects,
    ).aligned_dense_value()

    assert isinstance(aligned, RuntimeSliceAlignedValues)
    np.testing.assert_array_equal(aligned.value_for_slice(0), labels[0])
    np.testing.assert_array_equal(aligned.value_for_slice(1), labels[1])


def test_default_image_output_source_uses_unique_primary_image_input() -> None:
    current_payload = RuntimeImagePayloadContext(
        np.zeros((3, 4), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={"channel": "1"},
            source_image_names=("OrigDNA",),
        ),
    ).payload()
    primary_payload = RuntimeImagePayloadContext(
        np.ones((3, 4), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={"channel": "5"},
            source_image_names=("OrigMito",),
        ),
    ).payload()
    runtime = _FakeCellProfilerRuntime({"OrigMito": _FakeRuntimeImage(primary_payload)})
    request = _cellprofiler_output_record_request(
        module_name="ImageAndObjectOutputModule",
        inputs=(
            ArtifactSpec.input("OrigMito", ImageArtifactType),
            ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
        ),
        runtime_artifact_inputs=(
            ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
        ),
        outputs=(ArtifactSpec.output("MaskedMito", ImageArtifactType),),
        adapter=runtime,
        spec=ArtifactSpec.output("MaskedMito", ImageArtifactType),
        output_value=np.ones((3, 4), dtype=np.float32),
        output_values={},
        source=CellProfilerImageRequest(
            payload=current_payload,
            source_image_name="OrigDNA",
            image_count=1,
            execution_mode=ImagePayloadExecutionMode.NATURAL,
        ),
        current_image=current_payload,
        func=lambda image, labels: image,
        call_kwargs={},
    )

    source_payload = CellProfilerImageOutputSourcePayloadPolicy.for_module(
        request.module_name
    ).source_payload(request)

    metadata = image_payload_metadata(source_payload)
    assert metadata.source_component_metadata == {"channel": "5"}
    assert metadata.source_image_names == ("OrigMito",)


def test_stack_image_output_with_scalar_metadata_uses_primary_source_payload() -> None:
    current_payload = RuntimeImagePayloadContext(
        np.zeros((2, 5, 6), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("frame0.tif", "frame1.tif"),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1", "timepoint": "0"},
                    {"well": "A01", "site": "1", "channel": "1", "timepoint": "1"},
                ),
            ),
            source_image_names=("OrigColor",),
        ),
    ).payload()
    output_value = RuntimeImagePayloadContext(
        np.zeros((2, 5, 6), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
            },
        ),
    ).payload()
    runtime = _FakeCellProfilerRuntime(
        {"OrigColor": _FakeRuntimeImage(current_payload)}
    )
    request = _cellprofiler_output_record_request(
        module_name="ImageStackOutputModule",
        inputs=(ArtifactSpec.input("OrigColor", ImageArtifactType),),
        outputs=(ArtifactSpec.output("AdjacentImage", ImageArtifactType),),
        adapter=runtime,
        spec=ArtifactSpec.output("AdjacentImage", ImageArtifactType),
        output_value=output_value,
        output_values={},
        source=CellProfilerImageRequest(
            payload=current_payload,
            source_image_name="OrigColor",
            image_count=2,
            execution_mode=ImagePayloadExecutionMode.FULL_STACK,
        ),
        current_image=current_payload,
        func=lambda image: image,
        call_kwargs={},
    )

    source_payload = CellProfilerImageOutputSourcePayloadPolicy.for_module(
        request.module_name
    ).source_payload(request)

    assert source_payload is current_payload


def test_single_image_output_main_flow_uses_recorded_primary_image_source() -> None:
    current_payload = RuntimeImagePayloadContext(
        np.zeros((3, 4), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={"channel": "1"},
            source_image_names=("OrigDNA",),
        ),
    ).payload()
    mito_payload = RuntimeImagePayloadContext(
        np.arange(12, dtype=np.float32).reshape(3, 4),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={"channel": "5"},
            source_image_names=("OrigMito",),
        ),
    ).payload()
    runtime = _FakeCellProfilerRuntime(
        {"OrigMito": _FakeRuntimeImage(mito_payload, source_image_name="OrigMito")},
        {
            "Cytoplasm": ObjectLabelSet(
                name="Cytoplasm",
                labels=np.ones((3, 4), dtype=np.int32),
                source_image_name="OrigDNA",
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MaskImage",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("OrigMito", ImageArtifactType),
                        ArtifactSpec.input("Cytoplasm", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Cytoplasm", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("MaskedMito", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("MaskedMito", ImageArtifactType),),
                ),
            ),
        )
    )

    result = executor.run(
        mask_image,
        current_payload,
        cellprofiler_runtime=runtime,
        mask_source="objects",
    )

    recorded_metadata = image_payload_metadata(runtime.images["MaskedMito"].data)
    result_metadata = image_payload_metadata(result)
    assert recorded_metadata.source_component_metadata == {"channel": "5"}
    assert result_metadata.source_component_metadata == {"channel": "5"}
    assert result_metadata.source_image_names == ("OrigMito",)


def test_projected_measurement_rows_support_mapping_and_attribute_access() -> None:
    fields, rows = CellProfilerMeasurementOutputProjection(
        fields=(FieldSpec("AreaShape_Area"),),
        rows=({"AreaShape_Area": 7.0, "ObjectName": "Worms"},),
    ).apply()

    (row,) = rows

    assert isinstance(row, CellProfilerProjectedMeasurementRow)
    assert tuple(field.name for field in fields) == ("area_shape_area",)
    assert row["area_shape_area"] == 7.0
    assert row.area_shape_area == 7.0
    assert row.get("object_name") == "Worms"


def test_object_label_output_context_preserves_source_slice_paths() -> None:
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
    labels = np.zeros((3, 4, 5), dtype=np.int32)

    payload = CellProfilerObjectLabelOutputContextStrategy.for_value(
        labels
    ).runtime_object_label_value(labels, source_image)

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
        labels=np.zeros((3, 4, 5), dtype=np.int32),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
    )

    payload = CellProfilerObjectLabelOutputContextStrategy.for_value(
        labels
    ).runtime_object_label_value(
        labels,
        source_image,
        ObjectLabelDomainScope.PAYLOAD,
    )

    assert isinstance(payload, ObjectLabelPayload)
    assert payload.object_label_domain().scope is ObjectLabelDomainScope.PAYLOAD
    assert payload.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.TIF",
        "/input/A01_s001_w1_z002_t001.TIF",
        "/input/A01_s001_w1_z003_t001.TIF",
    )


def test_object_label_output_source_payload_uses_current_image_when_invocation_is_array_lowered() -> (
    None
):
    current_image = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w1_z001_t001.TIF",
            source_component_metadata={
                "well": "A01",
                "site": "001",
                "channel": "1",
            },
        ),
        mask=None,
    ).payload()
    request = _cellprofiler_output_record_request(
        adapter=SimpleNamespace(),
        spec=ArtifactSpec.output("Tile_of_grid", ObjectLabelsArtifactType),
        output_value=np.zeros((4, 5), dtype=np.int32),
        output_values={},
        source_image_name="BF_image",
        func=lambda image: image,
        call_kwargs={},
        source_aliases=("BF_image",),
        source_image_payload=np.zeros((4, 5), dtype=np.float32),
        current_image=current_image,
    )

    selected = request.runtime_plan.object_label_output_source_context_policy.source_context(request).source_payload

    assert selected is current_image


def test_object_label_output_source_payload_prefers_context_explaining_label_planes() -> (
    None
):
    labels = np.zeros((2, 4, 5), dtype=np.int32)
    source_payload = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w1.tif",
            source_component_metadata={"site": "001"},
        ),
        mask=None,
    ).payload()
    current_image = RuntimeImagePayloadContext(
        np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w1.tif",
                    "/input/A01_s002_w1.tif",
                ),
                component_metadata=(
                    {"site": "001"},
                    {"site": "002"},
                ),
            )
        ),
        mask=None,
    ).payload()
    request = _cellprofiler_output_record_request(
        adapter=SimpleNamespace(),
        spec=ArtifactSpec.output("Tile_of_grid", ObjectLabelsArtifactType),
        output_value=labels,
        output_values={},
        source_image_name="BF_image",
        func=lambda image: image,
        call_kwargs={},
        source_aliases=("BF_image",),
        source_image_payload=source_payload,
        current_image=current_image,
    )

    selected = request.runtime_plan.object_label_output_source_context_policy.source_context(request).source_payload

    assert selected is current_image


def test_resize_objects_output_source_policy_uses_input_object_context() -> None:
    image_payload = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w1_z001_t001.TIF",
            source_voxel_spacing=SourceVoxelSpacing((2.0, 1.0, 1.0)),
        ),
        mask=None,
    ).payload()
    object_payload = ObjectLabelSet(
        name="Nuclei",
        labels=np.zeros((4, 5), dtype=np.int32),
    )
    resolved_calls = []

    def get_objects(name, *, current_image=None):
        resolved_calls.append((name, current_image))
        return object_payload

    request = _cellprofiler_output_record_request(
        module_name="ResizeObjects",
        runtime_artifact_inputs=(
            ArtifactSpec.input("ReferenceImage", ImageArtifactType),
            ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
        ),
        adapter=SimpleNamespace(get_objects=get_objects),
        spec=ArtifactSpec.output("ResizedNuclei", ObjectLabelsArtifactType),
        output_value=np.zeros((4, 5), dtype=np.int32),
        output_values={},
        source_image_name="ReferenceImage",
        func=lambda image: image,
        call_kwargs={},
        source_image_payload=image_payload,
        current_image=image_payload,
    )

    source_context = (
        request.runtime_plan.object_label_output_source_context_policy.source_context(
            request
        )
    )

    assert source_context.source_payload is object_payload
    assert source_context.parent_image_payload is None
    assert resolved_calls == [("Nuclei", image_payload)]


def test_watershed_output_source_policy_uses_declared_image_as_parent() -> None:
    assert WatershedModule.module_name == "Watershed"
    image_payload = RuntimeImagePayloadContext(
        np.zeros((3, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w1_z001_t001.TIF",
            source_voxel_spacing=SourceVoxelSpacing((2.0, 1.0, 1.0)),
        ),
        mask=None,
    ).payload()
    request = _cellprofiler_output_record_request(
        module_name="Watershed",
        adapter=SimpleNamespace(),
        spec=ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),
        output_value=np.zeros((3, 4, 5), dtype=np.int32),
        output_values={},
        source_image_name="DNA",
        func=lambda image: image,
        call_kwargs={},
        source_aliases=("DNA",),
        source_image_payload=image_payload,
        current_image=image_payload,
    )

    policy = request.runtime_plan.object_label_output_source_context_policy
    source_context = policy.source_context(request)

    assert source_context.source_payload is image_payload
    assert source_context.parent_image_payload is image_payload


def test_object_label_recorder_suppresses_parent_spacing_when_policy_declares_no_parent_image() -> None:
    image_payload = RuntimeImagePayloadContext(
        np.zeros((3, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w1_z001_t001.TIF",
            source_voxel_spacing=SourceVoxelSpacing((2.0, 1.0, 1.0)),
        ),
        mask=None,
    ).payload()
    object_payload = ObjectLabelSet(
        name="Nuclei",
        labels=np.zeros((3, 4, 5), dtype=np.int32),
    )
    recorded: dict[str, object] = {}

    def get_objects(name, *, current_image=None):
        del current_image
        assert name == "Nuclei"
        return object_payload

    def add_source_image_objects(name, labels, **kwargs):
        recorded["name"] = name
        recorded["labels"] = labels
        recorded["kwargs"] = kwargs

    request = _cellprofiler_output_record_request(
        module_name="ResizeObjects",
        runtime_artifact_inputs=(
            ArtifactSpec.input("ReferenceImage", ImageArtifactType),
            ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
        ),
        adapter=SimpleNamespace(
            add_source_image_objects=add_source_image_objects,
            get_objects=get_objects,
        ),
        spec=ArtifactSpec.output("ResizedNuclei", ObjectLabelsArtifactType),
        output_value=np.zeros((3, 4, 5), dtype=np.int32),
        output_values={},
        source_image_name="ReferenceImage",
        func=lambda image: image,
        call_kwargs={},
        source_image_payload=image_payload,
        current_image=image_payload,
    )

    CellProfilerOutputRecorder.for_artifact_type(ObjectLabelsArtifactType).record(
        request
    )

    assert recorded["name"] == "ResizedNuclei"
    kwargs = recorded["kwargs"]
    assert kwargs["source_image_payload"] is object_payload
    assert kwargs["parent_image_source_voxel_spacing"] == SourceVoxelSpacing()


def test_contextual_object_label_recorder_fills_missing_parent_spacing_from_declared_parent_image() -> None:
    image_payload = RuntimeImagePayloadContext(
        np.zeros((3, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w2_z001_t001.TIF",
            source_voxel_spacing=SourceVoxelSpacing((2.0, 1.0, 1.0)),
        ),
        mask=None,
    ).payload()
    output_labels = ObjectLabelPayload(
        labels=np.zeros((3, 4, 5), dtype=np.int32),
    )
    recorded: dict[str, object] = {}

    def add_objects(name, labels, **kwargs):
        recorded["name"] = name
        recorded["labels"] = labels
        recorded["kwargs"] = kwargs

    request = _cellprofiler_output_record_request(
        module_name="IdentifySecondaryObjects",
        adapter=SimpleNamespace(add_objects=add_objects),
        inputs=(ArtifactSpec.input("Memb", ImageArtifactType),),
        spec=ArtifactSpec.output("Cells", ObjectLabelsArtifactType),
        output_value=output_labels,
        output_values={},
        source_image_name="Memb",
        func=lambda image: image,
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
        labels=ObjectLabelPayload(
            labels=np.zeros((3, 3), dtype=np.int32),
            domain=ObjectLabelDomain(declared_object_count=1),
        ),
    )
    request = _cellprofiler_output_record_request(
        module_name="MeasureObjectSizeShape",
        inputs=(ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
        adapter=SimpleNamespace(
            resolve_source_objects=lambda _name, _current_image: object_payload,
        ),
        spec=ArtifactSpec.output(
            "MeasureObjectSizeShape_1_measurements", MeasurementsArtifactType
        ),
        output_value=rows,
        output_values={},
        source_image_name="BF_image",
        func=measure_object_size_shape,
        call_kwargs={},
        source_image_payload=RuntimeImagePayloadContext(
            np.zeros((3, 3), dtype=np.float32),
            metadata=ImagePayloadMetadata(source_path="/input/A01_s001_w1.tif"),
            mask=None,
        ).payload(),
        current_image=RuntimeImagePayloadContext(
            np.zeros((3, 3), dtype=np.float32),
            metadata=ImagePayloadMetadata(source_path="/input/A01_s001_w1.tif"),
            mask=None,
        ).payload(),
    )

    record = measurement_record_for_module(request)

    assert record.object_name == "Cells"
    assert record.source_context.source_image_name is None


def test_measure_object_size_shape_row_policy_keeps_table_unqualified() -> None:
    policy = CellProfilerObjectMeasurementRowPolicy.for_module("MeasureObjectSizeShape")
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="BF_image",
        source_aliases=("BF_image",),
        payload=np.zeros((3, 3), dtype=np.float32),
    )

    assert policy.table_source_image_name((measurement_image,), "BF_image") is None


def test_object_label_output_source_payload_uses_primary_object_input() -> None:
    primary_payload = object()
    auxiliary_payload = object()

    def get_objects(name, *, current_image=None):
        del current_image
        return {
            "PrimaryObjects": primary_payload,
            "AuxiliaryObjects": auxiliary_payload,
        }[name]

    adapter = SimpleNamespace(get_objects=get_objects)
    object_inputs = (
        ArtifactSpec.input("PrimaryObjects", ObjectLabelsArtifactType),
        ArtifactSpec.input("AuxiliaryObjects", ObjectLabelsArtifactType),
    )
    request = _cellprofiler_output_record_request(
        inputs=object_inputs,
        runtime_artifact_inputs=object_inputs,
        adapter=adapter,
        spec=ArtifactSpec.output("Objects", ObjectLabelsArtifactType),
        output_value=np.zeros((2, 2), dtype=np.int32),
        output_values={},
        source_image_name=None,
        func=lambda image: image,
        call_kwargs={},
        source_image_payload=object(),
        current_image=None,
    )

    assert request.runtime_plan.object_label_output_source_context_policy.source_context(request).source_payload is primary_payload


def test_relationship_derived_object_label_output_uses_child_object_source_payload() -> (
    None
):
    parent_payload = object()
    child_payload = object()
    resolved_calls = []

    def get_objects(name, *, current_image=None):
        resolved_calls.append((name, current_image))
        return {
            "Nuclei": parent_payload,
            "Nucleoli": child_payload,
        }[name]

    object_inputs = (
        ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
        ArtifactSpec.input("Nucleoli", ObjectLabelsArtifactType),
    )
    output_spec = ArtifactSpec.output(
        "NucleoliChildObjects",
        ObjectLabelsArtifactType,
    )
    request = _cellprofiler_output_record_request(
        module_name="RelateObjects",
        inputs=object_inputs,
        runtime_artifact_inputs=object_inputs,
        outputs=(
            output_spec,
            ArtifactSpec.output(
                "Nuclei_Nucleoli_relationships", RelationshipsArtifactType
            ),
            ArtifactSpec.output(
                "Nucleoli_NucleoliChildObjects_relationships",
                RelationshipsArtifactType,
            ),
        ),
        adapter=SimpleNamespace(get_objects=get_objects),
        spec=output_spec,
        output_value=np.zeros((2, 2), dtype=np.int32),
        output_values={},
        source_image_name=None,
        func=lambda image: image,
        call_kwargs={},
        source_image_payload=object(),
        current_image=None,
    )

    assert request.runtime_plan.object_label_output_source_context_policy.source_context(request).source_payload is child_payload
    assert resolved_calls == [("Nucleoli", None)]


def _install_relationship_endpoint_policy_probe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    parent_spec: ArtifactSpec,
    child_spec: ArtifactSpec,
) -> list[tuple[str, str]]:
    resolved_relationships: list[tuple[str, str]] = []
    contract = RelationshipEndpointContract(parent_spec, child_spec)

    def for_request(cls, request):
        del cls

        def endpoint_contract(
            relationship_spec: ArtifactSpec,
        ) -> RelationshipEndpointContract:
            resolved_relationships.append((request.module_name, relationship_spec.name))
            return contract

        return SimpleNamespace(
            endpoint_specs=lambda relationship_spec: (
                endpoint_contract(relationship_spec).parent,
                contract.child,
            ),
            endpoint_contract=endpoint_contract,
            distance_measurements_apply=lambda relationship_spec: False,
        )

    monkeypatch.setattr(
        RelationshipEndpointResolver,
        "for_request",
        classmethod(for_request),
    )
    return resolved_relationships


def test_relateobjects_object_label_source_payload_uses_declared_endpoint_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_parent_payload = object()
    declared_child_payload = object()
    resolved_objects = []
    encoded_parent_spec = ArtifactSpec.input(
        "NameEncodedParent",
        ObjectLabelsArtifactType,
    )
    declared_child_spec = ArtifactSpec.input(
        "DeclaredChild",
        ObjectLabelsArtifactType,
    )
    saved_child_spec = ArtifactSpec.output("SavedChildren", ObjectLabelsArtifactType)
    misleading_relationship_spec = ArtifactSpec.output(
        "NameEncodedParent_NameEncodedChild_relationships",
        RelationshipsArtifactType,
    )
    resolved_relationships = _install_relationship_endpoint_policy_probe(
        monkeypatch,
        parent_spec=declared_child_spec,
        child_spec=saved_child_spec,
    )

    def get_objects(name, *, current_image=None):
        resolved_objects.append((name, current_image))
        return {
            "NameEncodedParent": encoded_parent_payload,
            "DeclaredChild": declared_child_payload,
        }[name]

    request = _cellprofiler_output_record_request(
        module_name="RelateObjects",
        inputs=(encoded_parent_spec, declared_child_spec),
        runtime_artifact_inputs=(encoded_parent_spec, declared_child_spec),
        outputs=(saved_child_spec, misleading_relationship_spec),
        adapter=SimpleNamespace(get_objects=get_objects),
        spec=saved_child_spec,
        output_value=np.zeros((2, 2), dtype=np.int32),
        output_values={},
        source_image_name=None,
        func=lambda image: image,
        call_kwargs={},
        source_image_payload=object(),
        current_image=None,
    )

    assert request.runtime_plan.object_label_output_source_context_policy.source_context(request).source_payload is declared_child_payload
    assert resolved_objects == [("DeclaredChild", None)]
    assert resolved_relationships == [
        ("RelateObjects", "NameEncodedParent_NameEncodedChild_relationships")
    ]


def test_multi_parent_relationship_object_output_uses_declared_primary_input() -> None:
    cells_payload = object()
    nuclei_payload = object()
    resolved_calls = []

    def get_objects(name, *, current_image=None):
        resolved_calls.append((name, current_image))
        return {
            "Cells": cells_payload,
            "Nuclei": nuclei_payload,
        }[name]

    object_inputs = (
        ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
        ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
    )
    output_spec = ArtifactSpec.output("Cytoplasm", ObjectLabelsArtifactType)
    request = _cellprofiler_output_record_request(
        module_name="IdentifyTertiaryObjects",
        inputs=object_inputs,
        runtime_artifact_inputs=object_inputs,
        outputs=(
            output_spec,
            ArtifactSpec.output(
                "Cells_Cytoplasm_relationships", RelationshipsArtifactType
            ),
            ArtifactSpec.output(
                "Nuclei_Cytoplasm_relationships", RelationshipsArtifactType
            ),
        ),
        adapter=SimpleNamespace(get_objects=get_objects),
        spec=output_spec,
        output_value=np.zeros((2, 2), dtype=np.int32),
        output_values={},
        source_image_name=None,
        func=lambda image: image,
        call_kwargs={},
        source_image_payload=object(),
        current_image=None,
    )

    assert request.runtime_plan.object_label_output_source_context_policy.source_context(request).source_payload is cells_payload
    assert resolved_calls == [("Cells", None)]


def test_object_label_output_source_payload_resolves_primary_object_for_current_image() -> (
    None
):
    current_image = object()
    resolved_calls = []
    primary_payload = ObjectLabelSet(
        name="PrimaryObjects",
        labels=np.array([[0, 1], [0, 0]], dtype=np.int32),
        source_path="/input/site1.tif",
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/site1.tif",)
        ),
    )

    def get_objects(name, *, current_image=None):
        resolved_calls.append((name, current_image))
        return primary_payload

    object_inputs = (ArtifactSpec.input("PrimaryObjects", ObjectLabelsArtifactType),)
    request = _cellprofiler_output_record_request(
        inputs=object_inputs,
        runtime_artifact_inputs=object_inputs,
        adapter=SimpleNamespace(get_objects=get_objects),
        spec=ArtifactSpec.output("Objects", ObjectLabelsArtifactType),
        output_value=np.zeros((2, 2), dtype=np.int32),
        output_values={},
        source_image_name=None,
        func=lambda image: image,
        call_kwargs={},
        source_image_payload=object(),
        current_image=current_image,
    )

    assert request.runtime_plan.object_label_output_source_context_policy.source_context(request).source_payload is primary_payload
    assert resolved_calls == [("PrimaryObjects", current_image)]


def test_object_label_output_source_payload_resolves_external_primary_object() -> None:
    current_image = object()
    resolved_calls = []
    primary_payload = ObjectLabelSet(
        name="PrimaryObjects",
        labels=np.array([[0, 1], [0, 0]], dtype=np.int32),
        source_path="/input/site1.tif",
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/site1.tif",)
        ),
    )

    def resolve_source_objects(name, current):
        resolved_calls.append((name, current))
        return primary_payload

    request = _cellprofiler_output_record_request(
        inputs=(ArtifactSpec.input("PrimaryObjects", ObjectLabelsArtifactType),),
        adapter=SimpleNamespace(resolve_source_objects=resolve_source_objects),
        spec=ArtifactSpec.output("Objects", ObjectLabelsArtifactType),
        output_value=np.zeros((2, 2), dtype=np.int32),
        output_values={},
        source_image_name=None,
        func=lambda image: image,
        call_kwargs={},
        source_image_payload=object(),
        current_image=current_image,
    )

    assert request.runtime_plan.object_label_output_source_context_policy.source_context(request).source_payload is primary_payload
    assert resolved_calls == [("PrimaryObjects", current_image)]


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


def test_cellprofiler_adapter_preserves_object_label_source_component_metadata() -> (
    None
):
    store = RuntimeValueStore()
    filemanager = _RecordingFileManager()
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=store,
        axis_scope=RuntimeExecutionAxisScope("A01_s001"),
        artifact_outputs={
            "Nuclei": ArtifactOutputPlan(
                name="Nuclei",
                path="/memory/Nuclei.pkl",
                artifact_type=ObjectLabelsArtifactType,
            )
        },
        filemanager=filemanager,
    )
    labels = ObjectLabelPayload(
        labels=np.array([[0, 1], [2, 0]], dtype=np.int32),
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

    assert isinstance(record.value.data, ObjectLabelPayload)
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
    ) == (
        {"well": "01", "site": "POS002", "channel": "D"},
    )
    assert filemanager.saved[0][0] is record.value.data


def test_source_qualified_image_rows_with_slice_index_use_runtime_axis() -> None:
    adapter = SimpleNamespace(
        cellprofiler_axis_image_number_start=lambda: 7,
        cellprofiler_image_number_for_source_paths=lambda _paths: None,
        cellprofiler_image_number_start_for_source_paths=lambda _paths: 7,
        cellprofiler_source_paths_for_image_name=lambda _name: (),
        can_resolve_source_candidates=False,
    )
    source_payload = RuntimeImagePayloadContext(
        np.zeros((2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(source_path="/plate/A01_s007_w1.tif"),
        mask=None,
    ).payload()
    rows, _mappings = CellProfilerMeasurementProjectionRequest(
        adapter=adapter,
        rows=(
            {
                "slice_index": 0,
                "source_image_name": "Objects1",
                "area_occupied": 10,
            },
            {
                "slice_index": 1,
                "source_image_name": "Objects2",
                "area_occupied": 20,
            },
        ),
        source_context=CellProfilerMeasurementSourceContext(
            source_image_name=None,
            source_image_payload=source_payload,
        ),
        object_name=None,
        need_row_mappings=True,
    ).project_rows()

    assert [row["image_number"] for row in rows] == [7, 8]


def test_default_row_policy_preserves_multi_source_image_row_ownership() -> None:
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

    partitions = DefaultObjectMeasurementRowPolicy().record_partitions(
        CellProfilerMeasurementRecord(
            rows=rows,
            object_name=None,
        )
    )

    assert len(partitions) == 1
    assert partitions[0].rows is rows
    assert partitions[0].object_name is None
    assert partitions[0].source_context.source_image_name is None


def test_default_row_policy_preserves_table_source_for_object_partitions() -> None:
    rows = (
        {
            "slice_index": 0,
            "object_name": "Cells",
            "object_label": 1,
            "mean_intensity": 0.5,
        },
    )

    partitions = DefaultObjectMeasurementRowPolicy().record_partitions(
        CellProfilerMeasurementRecord(
            rows=rows,
            object_name=None,
            source_context=CellProfilerMeasurementSourceContext(
                source_image_name="DNA",
                source_image_payload=np.zeros((1, 2, 2), dtype=np.float32),
            ),
            clear_source_when_rows_declare_object_name=False,
        )
    )

    assert len(partitions) == 1
    assert partitions[0].object_name == "Cells"
    assert partitions[0].source_context.source_image_name == "DNA"
    assert partitions[0].source_context.source_image_payload is not None


@dataclass(frozen=True, slots=True)
class _SyntheticAxisObjectMeasurement:
    image_number: int
    object_label: int
    value: float


@dataclass(frozen=True, slots=True)
class _ColumnarMeasurementRows(ColumnarRows):
    columns: dict[str, tuple[object, ...]]


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


def complete_object_measurement_rows(
    rows,
    *,
    label_payload,
    func,
    object_identity=MeasurementObjectRowIdentity.LABEL_ID,
    row_policy=None,
):
    if row_policy is None:
        row_policy = (
            CompactMeasuredObjectMeasurementRowPolicy()
            if object_identity is MeasurementObjectRowIdentity.ROW_ORDINAL
            else DefaultObjectMeasurementRowPolicy()
        )
    return row_policy.complete_rows(rows, label_payload=label_payload, func=func)


def _recorded_measurements_for_assertion(measurements):
    normalized = []
    for name, rows, kwargs in measurements:
        comparable_kwargs = dict(kwargs)
        if "fields" in comparable_kwargs:
            field_names = tuple(field.name for field in comparable_kwargs["fields"])
            if field_names and "slice_index" not in field_names:
                field_names = ("slice_index", *field_names)
            comparable_kwargs["fields"] = field_names
        normalized_rows = []
        for row in rows:
            normalized_row = dict(row)
            normalized_row.setdefault("slice_index", 0)
            normalized_rows.append(normalized_row)
        normalized.append((name, normalized_rows, comparable_kwargs))
    return normalized


class _FakeCellProfilerRuntime(RuntimePlaneAxisProjector):
    def __init__(
        self,
        images: dict[str, _FakeRuntimeImage],
        objects: dict[str, ObjectLabelSet] | None = None,
        measurement_tables: dict[str, tuple[MeasurementTable, ...]] | None = None,
        image_number_start: int = 1,
        ordered_pipeline_image_paths: tuple[str, ...] = (),
        image_numbers_by_source_path: dict[str, int] | None = None,
        variable_components: tuple[VariableComponents, ...] = (),
    ) -> None:
        self.images = images
        self.runtime_objects = objects or {}
        self.runtime_measurement_tables = measurement_tables or {}
        self.image_number_start = image_number_start
        self.ordered_pipeline_image_paths = ordered_pipeline_image_paths
        self.image_numbers_by_source_path = image_numbers_by_source_path or {}
        self.measurements: list[tuple[str, list[object], dict[str, object]]] = []
        self.objects: list[tuple[str, np.ndarray, dict[str, object]]] = []
        self.spatial_grids: dict[str, SpatialGrid] = {}
        self.relationships: list[tuple[str, dict[str, object]]] = []
        self.axis_scope = RuntimeExecutionAxisScope.from_raw(
            "test-axis",
            component=None,
            value=None,
        )
        self.variable_components = variable_components
        self.group_by = GroupBy.NONE
        self.group_key = None
        self.filename_parser = None
        self.output_identity_cache = FunctionOutputIdentityCache()
        self.processing_context = SimpleNamespace(
            microscope_handler=SimpleNamespace(
                parser=SimpleNamespace(semantic_identity=lambda: ()),
            ),
        )
        self.runtime_value_store = RuntimeValueStore()
        self._measurement_cache = {}
        self.source_binding_plan = CompiledSourceBindingPlan()
        self.source_binding_context = SourceBindingRuntimeContext(
            step_input_files=ordered_pipeline_image_paths,
            current_step_input_files=ordered_pipeline_image_paths,
            pipeline_input_files=ordered_pipeline_image_paths,
        )

    def require_resolvable_source_aliases(self, aliases: tuple[str, ...]) -> None:
        missing = tuple(alias for alias in aliases if alias not in self.images)
        if missing:
            raise AssertionError(f"Unexpected missing image aliases: {missing!r}")

    def cellprofiler_ordered_pipeline_image_paths(self) -> tuple[str, ...]:
        return self.ordered_pipeline_image_paths

    @property
    def can_resolve_source_candidates(self) -> bool:
        return bool(self.ordered_pipeline_image_paths)

    def source_axis_metadata_scope(self) -> SourceAxisMetadataScope:
        return SourceAxisMetadataScope(())

    def runtime_slice_plane_index(self) -> int | None:
        return None

    def source_candidates(
        self,
        file_paths: tuple[str, ...],
    ) -> tuple[ParsedSourceCandidate, ...]:
        return tuple(
            ParsedSourceCandidate(
                path=path,
                resolved_path=path,
                filename=Path(path).name,
                metadata={},
            )
            for path in file_paths
        )

    def cellprofiler_source_order_path(self, path: str) -> str:
        return path

    def cellprofiler_image_number_for_source_paths(
        self,
        source_paths: tuple[str, ...],
    ) -> int | None:
        if not source_paths:
            return None
        first_source_path = self.cellprofiler_source_order_path(source_paths[0])
        if first_source_path in self.image_numbers_by_source_path:
            return self.image_numbers_by_source_path[first_source_path]
        try:
            return self.ordered_pipeline_image_paths.index(first_source_path) + 1
        except ValueError:
            return None

    def cellprofiler_source_path_for_image_number(
        self,
        image_number: int,
    ) -> str | None:
        for source_path, path_image_number in self.image_numbers_by_source_path.items():
            if int(path_image_number) == int(image_number):
                return source_path
        index = int(image_number) - 1
        if 0 <= index < len(self.ordered_pipeline_image_paths):
            return self.ordered_pipeline_image_paths[index]
        return None

    def cellprofiler_axis_image_number_start(self) -> int:
        return self.image_number_start

    def cellprofiler_image_number_start_for_source_paths(
        self,
        source_paths: tuple[str, ...],
    ) -> int:
        image_number = self.cellprofiler_image_number_for_source_paths(source_paths)
        if image_number is not None:
            return image_number
        return self.cellprofiler_axis_image_number_start()

    def cellprofiler_source_paths_for_image_name(
        self,
        image_name: str | None,
    ) -> tuple[str, ...]:
        del image_name
        return ()

    def resolve_source_image(self, alias: str, current_image: object) -> np.ndarray:
        del current_image
        return self.images[alias].data

    def image_payload_for_current_runtime_plane(
        self,
        payload: ObjectLabelValue,
        *,
        current_image: ObjectLabelValue | None,
        projection_capabilities: (
            frozenset[type[CellProfilerRuntimePlaneProjectionCapability]] | None
        ) = None,
    ) -> ObjectLabelValue:
        del current_image, projection_capabilities
        return payload

    def get_image(
        self,
        name: str,
        *,
        current_image: object | None = None,
    ) -> _FakeRuntimeImage:
        del current_image
        return self.images[name]

    def add_image(
        self,
        name: str,
        data: object,
        **kwargs: object,
    ) -> None:
        self.images[name] = _FakeRuntimeImage(
            data,
            source_image_name=(
                str(kwargs["source_image_name"])
                if kwargs.get("source_image_name") is not None
                else None
            ),
        )

    def get_objects(
        self,
        name: str,
        *,
        current_image: object | None = None,
    ) -> ObjectLabelSet:
        del current_image
        return self.runtime_objects[name]

    def get_objects_across_groups(self, name: str) -> ObjectLabelSet:
        return self.runtime_objects[name]

    def measurement_tables(
        self,
        *,
        group_key: str | None = None,
        match_group: bool = True,
        current_image: object | None = None,
    ) -> tuple[MeasurementTable, ...]:
        del group_key, match_group, current_image
        return tuple(
            table
            for tables in self.runtime_measurement_tables.values()
            for table in tables
        )

    def add_measurements(
        self,
        name: str,
        rows: object,
        **kwargs: object,
    ) -> None:
        self.measurements.append((name, measurement_table_rows(rows), kwargs))

    def add_objects(
        self,
        name: str,
        labels: object,
        **kwargs: object,
    ) -> None:
        self.objects.append((name, labels, kwargs))
        self.runtime_objects[name] = (
            labels
            if isinstance(labels, ObjectLabelSet)
            else ObjectLabelSet(name=name, labels=labels)
        )

    def add_source_image_objects(
        self,
        name: str,
        labels: object,
        **kwargs: object,
    ) -> None:
        self.objects.append((name, labels, kwargs))
        self.runtime_objects[name] = SourceImageObjectLabelBuildRequest(
            image=kwargs["source_image_payload"],
            labels=labels,
            domain_scope=(kwargs["domain_scope"] if "domain_scope" in kwargs else None),
        ).label_set(
            name=name,
            source_image_name=(
                str(kwargs["source_image_name"])
                if "source_image_name" in kwargs
                and kwargs["source_image_name"] is not None
                else None
            ),
        )

    def add_spatial_grid(
        self,
        name: str,
        grid: SpatialGrid | RuntimeSliceAlignedValues,
    ) -> None:
        if isinstance(grid, RuntimeSliceAlignedValues):
            self.spatial_grids[name] = RuntimeSliceAlignedValues(
                slices=tuple(value.with_name(name) for value in grid.slices)
            )
            return
        self.spatial_grids[name] = grid.with_name(name)

    def get_spatial_grid(self, name: str) -> SpatialGrid | RuntimeSliceAlignedValues:
        return self.spatial_grids[name]

    def add_relationship(self, name: str, **kwargs: object) -> None:
        self.relationships.append((name, kwargs))


def test_default_image_output_source_policy_maps_output_ordinal_to_primary_input() -> (
    None
):
    red_payload = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_path="/plate/A01_s001_w1_z001_t001.png",
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
                "z_index": "1",
                "timepoint": "1",
                "extension": ".png",
            },
        ),
    ).payload()
    green_payload = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_path="/plate/A01_s001_w2_z001_t001.png",
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "2",
                "z_index": "1",
                "timepoint": "1",
                "extension": ".png",
            },
        ),
    ).payload()
    output_value = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "z_index": "1",
                "timepoint": "1",
                "extension": ".png",
            },
        ),
    ).payload()
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigRed": _FakeRuntimeImage(red_payload),
            "OrigGreen": _FakeRuntimeImage(green_payload),
        }
    )

    request = _cellprofiler_output_record_request(
        module_name="Align",
        inputs=(
            ArtifactSpec.input("OrigRed", ImageArtifactType),
            ArtifactSpec.input("OrigGreen", ImageArtifactType),
        ),
        outputs=(
            ArtifactSpec.output("AlignedRed", ImageArtifactType),
            ArtifactSpec.output("AlignedGreen", ImageArtifactType),
        ),
        spec=ArtifactSpec.output("AlignedGreen", ImageArtifactType),
        adapter=runtime,
        output_value=output_value,
        output_values={},
        source_image_payload=np.stack(
            (
                image_payload_data(red_payload),
                image_payload_data(green_payload),
            )
        ),
        func=lambda image: image,
        call_kwargs={},
        current_image=object(),
    )

    source_payload = DefaultImageOutputSourcePayloadPolicy().source_payload(request)
    assert source_payload is green_payload
    recorded = CellProfilerImageOutputContextStrategy.for_value(
        output_value
    ).runtime_image_value(output_value, source_payload)
    recorded_metadata = image_payload_metadata(recorded)
    assert recorded_metadata.source_path == "/plate/A01_s001_w2_z001_t001.png"
    assert recorded_metadata.source_component_metadata == {
        "well": "A01",
        "site": "1",
        "channel": "2",
        "z_index": "1",
        "timepoint": "1",
        "extension": ".png",
    }


def test_default_image_output_source_policy_uses_invocation_owned_ordinal_primary_input() -> (
    None
):
    red_payload = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_path="/plate/A01_s001_w1_z001_t001.png",
            source_component_metadata={"channel": "1"},
            source_image_names=("OrigRed",),
        ),
    ).payload()
    green_payload = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_path="/plate/A01_s001_w2_z001_t001.png",
            source_component_metadata={"channel": "2"},
            source_image_names=("OrigGreen",),
        ),
    ).payload()
    source_bundle = ImagePayloadBundleContext.from_payloads(
        (red_payload, green_payload)
    ).compose()

    def forbidden_get_image(name, *, current_image=None):
        del name, current_image
        raise AssertionError("Primary source payload should come from invocation.")

    def forbidden_resolve_source_image(name, current_image):
        del name, current_image
        raise AssertionError("Primary source payload should come from invocation.")

    request = _cellprofiler_output_record_request(
        module_name="Align",
        inputs=(
            ArtifactSpec.input("OrigRed", ImageArtifactType),
            ArtifactSpec.input("OrigGreen", ImageArtifactType),
        ),
        outputs=(
            ArtifactSpec.output("AlignedRed", ImageArtifactType),
            ArtifactSpec.output("AlignedGreen", ImageArtifactType),
        ),
        spec=ArtifactSpec.output("AlignedGreen", ImageArtifactType),
        adapter=SimpleNamespace(
            get_image=forbidden_get_image,
            resolve_source_image=forbidden_resolve_source_image,
        ),
        output_value=np.ones((4, 5), dtype=np.float32),
        output_values={},
        source=CellProfilerImageRequest(
            payload=source_bundle,
            source_image_name="OrigRed__OrigGreen",
            source_aliases=("OrigRed", "OrigGreen"),
            image_count=2,
            execution_mode=ImagePayloadExecutionMode.FULL_STACK,
        ),
        current_image=source_bundle,
        func=lambda image: image,
        call_kwargs={},
    )

    source_payload = DefaultImageOutputSourcePayloadPolicy().source_payload(request)
    source_metadata = image_payload_metadata(source_payload)
    assert source_metadata.source_path == "/plate/A01_s001_w2_z001_t001.png"
    assert source_metadata.source_component_metadata == {"channel": "2"}
    assert source_metadata.source_image_names == ("OrigGreen",)


def test_object_label_output_source_uses_unique_primary_image_input() -> None:
    stain1_payload = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_path="/plate/A01_s001_w1_z001_t001.png",
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
                "z_index": "1",
                "timepoint": "1",
            },
            source_image_names=("Stain1",),
        ),
    ).payload()
    stain2_payload = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_path="/plate/A01_s001_w2_z001_t001.png",
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "2",
                "z_index": "1",
                "timepoint": "1",
            },
            source_image_names=("Stain2",),
        ),
    ).payload()
    source_bundle = ImagePayloadBundleContext.from_payloads(
        (stain1_payload, stain2_payload)
    ).compose()
    runtime = _FakeCellProfilerRuntime(
        {
            "Stain1": _FakeRuntimeImage(
                stain1_payload,
                source_image_name="Stain1",
            ),
        }
    )
    request = _cellprofiler_output_record_request(
        module_name="IdentifyPrimaryObjects",
        inputs=(ArtifactSpec.input("Stain1", ImageArtifactType),),
        runtime_artifact_inputs=(ArtifactSpec.input("Stain1", ImageArtifactType),),
        outputs=(ArtifactSpec.output("Objects1", ObjectLabelsArtifactType),),
        spec=ArtifactSpec.output("Objects1", ObjectLabelsArtifactType),
        adapter=runtime,
        output_value=np.ones((4, 5), dtype=np.int32),
        output_values={},
        source=CellProfilerImageRequest(
            payload=source_bundle,
            source_image_name="Stain1__Stain2",
            source_aliases=("Stain1", "Stain2"),
            image_count=2,
            execution_mode=ImagePayloadExecutionMode.FULL_STACK,
        ),
        current_image=source_bundle,
        func=lambda image: image,
        call_kwargs={},
    )

    source_payload = request.runtime_plan.object_label_output_source_context_policy.source_context(request).source_payload
    recorded = CellProfilerObjectLabelOutputContextStrategy.for_value(
        request.output_value
    ).runtime_object_label_value(
        request.output_value,
        source_payload,
        request.object_label_output_domain_scope(),
    )
    recorded_metadata = image_payload_metadata(recorded)

    assert recorded_metadata.source_path == "/plate/A01_s001_w1_z001_t001.png"
    assert recorded_metadata.source_component_metadata == {
        "well": "A01",
        "site": "1",
        "channel": "1",
        "z_index": "1",
        "timepoint": "1",
    }
    assert recorded_metadata.source_image_provenance_planes.paths == ()


def test_object_label_output_source_uses_invocation_owned_unique_primary_image_input() -> (
    None
):
    stain_payload = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_path="/plate/A01_s001_w1_z001_t001.png",
            source_component_metadata={"channel": "1"},
            source_image_names=("Stain1",),
        ),
    ).payload()

    def forbidden_get_image(name, *, current_image=None):
        del name, current_image
        raise AssertionError("Primary source payload should come from invocation.")

    request = _cellprofiler_output_record_request(
        module_name="IdentifyPrimaryObjects",
        inputs=(ArtifactSpec.input("Stain1", ImageArtifactType),),
        runtime_artifact_inputs=(ArtifactSpec.input("Stain1", ImageArtifactType),),
        outputs=(ArtifactSpec.output("Objects1", ObjectLabelsArtifactType),),
        spec=ArtifactSpec.output("Objects1", ObjectLabelsArtifactType),
        adapter=SimpleNamespace(get_image=forbidden_get_image),
        output_value=np.ones((4, 5), dtype=np.int32),
        output_values={},
        source=CellProfilerImageRequest(
            payload=stain_payload,
            source_image_name="Stain1",
            source_aliases=("Stain1",),
            image_count=1,
            execution_mode=ImagePayloadExecutionMode.NATURAL,
        ),
        current_image=stain_payload,
        func=lambda image: image,
        call_kwargs={},
    )

    source_payload = request.runtime_plan.object_label_output_source_context_policy.source_context(request).source_payload
    source_metadata = image_payload_metadata(source_payload)
    assert source_metadata.source_path == "/plate/A01_s001_w1_z001_t001.png"
    assert source_metadata.source_component_metadata == {"channel": "1"}
    assert source_metadata.source_image_names == ("Stain1",)


class _CalculateMathObjectOperandAdapter(_FakeCellProfilerRuntime):
    def __init__(self, labels: np.ndarray) -> None:
        super().__init__(
            images={},
            objects={
                "Nuclei": ObjectLabelSet(name="Nuclei", labels=labels),
            },
            measurement_tables={
                "Nuclei": (
                    MeasurementTable(
                        name="NucleiMeasurements",
                        object_name="Nuclei",
                        object_id_field="object_label",
                        rows=[
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
                    ),
                ),
            },
        )

    def resolve_source_objects(
        self,
        name: str,
        current_image: object,
    ) -> ObjectLabelSet:
        del current_image
        return self.get_objects(name)


def test_calculate_math_object_operands_preserve_label_slice_domain() -> None:
    labels = np.asarray(
        [
            [[0, 1], [2, 0]],
            [[0, 1], [0, 2]],
        ],
        dtype=np.int32,
    )
    adapter = _CalculateMathObjectOperandAdapter(labels)
    request = ObjectInputBindingRequest(
        module_name="CalculateMath",
        func=lambda image: image,
        object_inputs=(ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
        adapter=adapter,
        kwargs={
            "operand1_feature": "Intensity_MeanIntensity_DNA",
            "operand1_object_name": "Nuclei",
        },
        current_image=np.zeros((2, 2), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
    )

    value = CalculateMathInputPolicy().operand_value(
        request,
        feature_kwarg="operand1_feature",
        object_kwarg="operand1_object_name",
    )

    assert isinstance(value, CellProfilerSliceAlignedValues)
    assert value.slice_count == 2
    np.testing.assert_array_equal(value.value_for_slice(0), [1.0, 2.0])
    np.testing.assert_array_equal(value.value_for_slice(1), [3.0, 4.0])


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
            RuntimeProjectionAxis(slice_index=1, extent=2),
        ),
        np.asarray([3.0]),
    )


class _CombineObjectsAdapter:
    def __init__(self, payloads):
        self.payloads = payloads

    def get_objects(self, name, *, current_image):
        del current_image
        return self.payloads[name]


def test_combine_objects_broadcasts_2d_labels_to_runtime_slice_domain() -> None:
    stacked_labels = np.asarray(
        [
            [[0, 1], [0, 0]],
            [[0, 0], [2, 0]],
        ],
        dtype=np.int32,
    )
    plane_labels = np.asarray([[0, 3], [0, 0]], dtype=np.int32)
    adapter = _CombineObjectsAdapter(
        {
            "Primary": ObjectLabelSet(
                name="Primary",
                labels=ObjectLabelPayload(labels=stacked_labels),
            ),
            "Secondary": ObjectLabelSet(
                name="Secondary",
                labels=ObjectLabelPayload(labels=plane_labels),
            ),
        }
    )
    request = ObjectInputBindingRequest(
        module_name="CombineObjects",
        func=lambda image: image,
        object_inputs=(
            ArtifactSpec.input("Primary", ObjectLabelsArtifactType),
            ArtifactSpec.input("Secondary", ObjectLabelsArtifactType),
        ),
        adapter=adapter,
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
    )

    primary, secondary = CombineObjectsInputPolicy().label_pair_payload(request)

    assert primary.shape == (2, 2, 2)
    assert secondary.shape == (2, 2, 2)
    np.testing.assert_array_equal(primary, stacked_labels)
    np.testing.assert_array_equal(secondary[0], plane_labels)
    np.testing.assert_array_equal(secondary[1], plane_labels)

    bound = CombineObjectsInputPolicy().bind(request)
    assert (
        bound["_cellprofiler_execution_mode_override"]
        is ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
    )
    aligned = bound["_cellprofiler_image_override"]
    assert isinstance(aligned, AlignedImageStack)
    assert len(aligned.slices) == 2
    np.testing.assert_array_equal(
        aligned.slices[0],
        np.stack((stacked_labels[0], plane_labels), axis=0),
    )
    np.testing.assert_array_equal(
        aligned.slices[1],
        np.stack((stacked_labels[1], plane_labels), axis=0),
    )


def test_single_object_input_policy_preserves_native_label_contract() -> None:
    labels = ObjectLabelSet(
        name="InputObjects",
        labels=ObjectLabelPayload(
            labels=np.asarray(
                (
                    ((0, 1), (0, 0)),
                    ((0, 0), (2, 0)),
                ),
                dtype=np.int32,
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
            ),
        ),
    )
    adapter = _CombineObjectsAdapter({"InputObjects": labels})
    request = ObjectInputBindingRequest(
        module_name="MeasureObjectSizeShape",
        func=measure_object_size_shape,
        object_inputs=(ArtifactSpec.input("InputObjects", ObjectLabelsArtifactType),),
        adapter=adapter,
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
    )

    bound = MeasureObjectSizeShapeModule().bind(request)

    assert bound["labels"] is labels


def test_object_row_binding_returns_current_runtime_plane_labels() -> None:
    labels = ObjectLabelSet(
        name="Cells",
        labels=ObjectLabelPayload(
            labels=np.asarray(
                (
                    ((0, 1), (0, 0)),
                    ((0, 0), (2, 0)),
                ),
                dtype=np.int32,
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
        ),
    )
    adapter = _RuntimeSliceObjectAdapter(labels, slice_index=1)
    request = ObjectInputBindingRequest(
        module_name="FilterObjects",
        func=filter_objects,
        object_inputs=(ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
        runtime_inputs=(ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
        adapter=adapter,
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
    )

    bound_labels = request.labels_for(
        ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    )

    np.testing.assert_array_equal(
        bound_labels,
        np.asarray(((0, 0), (2, 0)), dtype=np.int32),
    )


def test_object_row_binding_preserves_full_stack_runtime_slice_labels() -> None:
    label_array = np.asarray(
        (((0, 1), (0, 0)),),
        dtype=np.int32,
    )
    labels = ObjectLabelSet(
        name="Cells",
        labels=ObjectLabelPayload(
            labels=label_array,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
        ),
    )
    adapter = _RuntimeSliceObjectAdapter(labels, slice_index=0)
    request = ObjectInputBindingRequest(
        module_name="ResizeObjects",
        func=filter_objects,
        object_inputs=(ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
        runtime_inputs=(ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
        adapter=adapter,
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
        project_object_labels_to_current_plane=False,
    )

    bound_labels = request.labels_for(
        ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    )

    assert isinstance(bound_labels, ObjectLabelSet)
    np.testing.assert_array_equal(object_label_dense_array(bound_labels), label_array)


def test_object_row_binding_preserves_full_stack_dense_rank_without_domain() -> None:
    label_array = np.asarray(
        (((0, 1), (0, 0)),),
        dtype=np.int32,
    )
    labels = ObjectLabelSet(name="Cells", labels=label_array)
    adapter = _RuntimeSliceObjectAdapter(labels, slice_index=0)
    request = ObjectInputBindingRequest(
        module_name="ResizeObjects",
        func=filter_objects,
        object_inputs=(ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
        runtime_inputs=(ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
        adapter=adapter,
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
        project_object_labels_to_current_plane=False,
    )

    bound_labels = request.labels_for(
        ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    )

    np.testing.assert_array_equal(bound_labels, label_array)


def test_adapter_object_record_preserves_single_runtime_slice_stack_for_full_stack_binding() -> None:
    label_array = np.asarray(
        (((0, 1), (0, 0)),),
        dtype=np.int32,
    )
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_scope=RuntimeExecutionAxisScope("A01_s001"),
        artifact_outputs={
            "Cells": ArtifactOutputPlan(
                name="Cells",
                path="/memory/Cells.pkl",
                artifact_type=ObjectLabelsArtifactType,
            )
        },
        filemanager=_RecordingFileManager(),
    )
    adapter.add_objects(
        "Cells",
        ObjectLabelPayload(
            labels=label_array,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
        ),
    )
    request = ObjectInputBindingRequest(
        module_name="ResizeObjects",
        func=resize_objects_3d,
        object_inputs=(ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
        runtime_inputs=(ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
        adapter=adapter,
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
        project_object_labels_to_current_plane=False,
    )

    bound_labels = request.labels_for(
        ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    )

    assert isinstance(bound_labels, ObjectLabelSet)
    np.testing.assert_array_equal(object_label_dense_array(bound_labels), label_array)


def test_pure_3d_executor_preserves_nominal_singleton_object_label_stack() -> None:
    label_array = np.asarray(
        (((0, 1), (0, 0)),),
        dtype=np.int32,
    )
    labels = ObjectLabelSet(
        name="Cells",
        labels=ObjectLabelPayload(
            labels=label_array,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
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

    result = CellProfilerFunctionContractExecutor().execute(
        full_stack_identity,
        np.zeros((1, 2, 2), dtype=np.float32),
        {"labels": labels},
    )

    assert result.shape == (1, 2, 2)
    assert seen_label_shape == [(1, 2, 2)]


def test_object_row_binding_returns_current_runtime_plane_relationship() -> None:
    labels = ObjectLabelSet(
        name="Cells",
        labels=ObjectLabelPayload(
            labels=np.asarray(
                (
                    ((0, 1), (0, 0)),
                    ((0, 0), (2, 0)),
                ),
                dtype=np.int32,
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
        ),
    )
    semantics = RelationshipSemantics.parent_child("Parents", "Cells")
    relationship = ObjectRelationship(
        name="Parents_Cells_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(10, 20, 21),
        target_ids=(1, 2, 3),
        relationship_type=semantics.relationship_type,
        slice_indices=(0, 1, 1),
        slice_count=2,
    )
    adapter = _RuntimeSliceObjectAdapter(
        labels, slice_index=1, relationship=relationship
    )
    request = ObjectInputBindingRequest(
        module_name="FilterObjects",
        func=filter_objects,
        object_inputs=(ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
        runtime_inputs=(
            ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
            ArtifactSpec.input(
                "Parents_Cells_relationships", RelationshipsArtifactType
            ),
        ),
        adapter=adapter,
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
    )

    projected = request.current_plane_relationship_for(
        ArtifactSpec.input("Parents_Cells_relationships", RelationshipsArtifactType)
    )

    assert projected.source_ids == (20, 21)
    assert projected.target_ids == (2, 3)
    assert projected.slice_count == 1


def test_special_inputs_bind_from_declared_role_order_not_runtime_dedup_order() -> None:
    memb_final = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    cell_seeds = np.asarray([[0, 3], [0, 0]], dtype=np.int32)
    intensity_metadata = ImagePayloadMetadata(
        intensity_scale=255,
        source_dtype="uint8",
    )
    adapter = _FakeCellProfilerRuntime(
        {
            "MembFinal": _FakeRuntimeImage(memb_final),
            "cellSeeds": _FakeRuntimeImage(
                ImageMetadataPayload(cell_seeds, intensity_metadata),
            ),
        }
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Watershed",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("MembFinal", ImageArtifactType),
                        ArtifactSpec.input("cellSeeds", ImageArtifactType),
                        ArtifactSpec.input("MembFinal", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("MembFinal", ImageArtifactType),
                        ArtifactSpec.input("cellSeeds", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Cells", ObjectLabelsArtifactType),),
                ),
            ),
        )
    )

    kwargs = executor._runtime_input_kwargs(
        executor.runtime_plan(watershed),
        adapter,
        memb_final,
        {"watershed_method": "markers"},
    )

    np.testing.assert_array_equal(kwargs["markers"], cell_seeds)
    assert kwargs["markers"].dtype == np.int32
    np.testing.assert_array_equal(kwargs["mask"], memb_final)

    def add_image(
        self,
        name: str,
        data: object,
        **kwargs: object,
    ) -> None:
        del kwargs
        self.images[name] = _FakeRuntimeImage(data)


def test_coerce_invocation_kwargs_uses_function_enum_annotations() -> None:
    coerced = CallableInvocationKwargSpec.from_callable(
        identify_primary_objects
    ).coerce_kwargs(
        {
            "unclump_method": "Shape",
            "fill_holes": "After both thresholding and declumping",
            "limit_erase": "Continue",
        },
    )

    assert coerced["unclump_method"] is UnclumpMethod.SHAPE
    assert coerced["fill_holes"] is FillHolesOption.AFTER_BOTH
    assert coerced["limit_erase"] is ExcessObjectHandling.CONTINUE


def test_cellprofiler_contract_executor_applies_pure_2d_after_input_resolution():
    calls = []

    def add_one(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image + 1

    add_one.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((2, 4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(add_one, stack, {})

    assert calls == [(4, 5), (4, 5)]
    assert result.shape == stack.shape
    np.testing.assert_array_equal(result, np.ones_like(stack))


def test_cellprofiler_contract_executor_preserves_declared_two_channel_color_plane():
    calls = []

    def split_first_channel(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image[..., 0]

    split_first_channel.__processing_contract__ = ProcessingContract.PURE_2D
    image = ImageMetadataPayload(
        np.zeros((4, 5, 2), dtype=np.float32),
        ImagePayloadMetadata(
            source_component_metadata={
                SOURCE_IMAGE_TYPE_METADATA_FIELD: "Color image",
            },
        ),
    )
    image.data[..., 0] = 7.0

    result = CellProfilerFunctionContractExecutor().execute(
        split_first_channel,
        image,
        {},
    )

    assert calls == [(4, 5, 2)]
    np.testing.assert_array_equal(
        image_payload_data(result),
        np.full((1, 4, 5), 7.0),
    )


def test_cellprofiler_contract_executor_slices_declared_two_channel_color_stack():
    calls = []

    def split_first_channel(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image[..., 0]

    split_first_channel.__processing_contract__ = ProcessingContract.PURE_2D
    stack = ImageMetadataPayload(
        np.zeros((2, 4, 5, 2), dtype=np.float32),
        ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                component_metadata=(
                    {SOURCE_IMAGE_TYPE_METADATA_FIELD: "Color image", "site": "1"},
                    {SOURCE_IMAGE_TYPE_METADATA_FIELD: "Color image", "site": "2"},
                ),
            ),
        ),
    )
    stack.data[0, ..., 0] = 3.0
    stack.data[1, ..., 0] = 9.0

    result = CellProfilerFunctionContractExecutor().execute(
        split_first_channel,
        stack,
        {},
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


def test_cellprofiler_contract_executor_flattens_volume_stacks_for_pure_2d():
    calls = []

    def add_labels(image: np.ndarray, *, labels: np.ndarray) -> np.ndarray:
        calls.append((image.shape, labels.shape, int(labels[0, 0])))
        return image + labels

    add_labels.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((2, 3, 4, 5), dtype=np.float32)
    labels = np.arange(6, dtype=np.int32).reshape((2, 3, 1, 1))
    labels = np.broadcast_to(labels, image.shape)

    result = CellProfilerFunctionContractExecutor().execute(
        add_labels,
        image,
        {"labels": labels},
    )

    assert calls == [((4, 5), (4, 5), index) for index in range(6)]
    assert result.shape == (6, 4, 5)
    np.testing.assert_array_equal(result, labels.reshape((6, 4, 5)))


def test_cellprofiler_contract_executor_slices_high_rank_labels_by_runtime_axis():
    calls = []

    def keep_labels(
        image: np.ndarray, *, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append((image.shape, labels.shape, int(labels[0, 0, 0])))
        return image, labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((4, 5), dtype=np.float32)
    labels = np.arange(3, dtype=np.int32).reshape((3, 1, 1, 1))
    labels = np.broadcast_to(labels, (3, 2, 4, 5))

    result_image, result_labels = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        image,
        {"labels": labels},
    )

    assert calls == [((4, 5), (2, 4, 5), index) for index in range(3)]
    assert result_image.shape == (3, 4, 5)
    assert result_labels.shape == (3, 2, 4, 5)
    np.testing.assert_array_equal(result_labels, labels)


def test_runtime_slice_projection_counts_high_rank_kwargs_by_first_axis_for_2d_image():
    labels = np.zeros((3, 2, 4, 5), dtype=np.int32)

    assert RuntimeSliceProjection.first_axis_slice_count_from_values((labels,)) == 3


def test_cellprofiler_contract_executor_projects_flat_grouped_label_kwargs():
    calls = []

    def keep_labels(
        image: np.ndarray, *, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append((image.shape, labels.shape, int(labels[0, 0])))
        return image, labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((3, 4, 5), dtype=np.float32)
    labels = np.zeros((2, 3, 4, 5), dtype=np.int32)
    labels[0, :, 1:3, 1:3] = 1
    labels[1, :, 2:4, 2:4] = 2
    flattened_labels = labels.reshape((-1, *labels.shape[-2:]))

    result_image, result_labels = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        image,
        {"labels": flattened_labels},
    )

    expected_labels = np.max(labels, axis=0)
    assert calls == [((4, 5), (4, 5), 0) for _ in range(3)]
    assert result_image.shape == image.shape
    np.testing.assert_array_equal(result_labels, expected_labels)


def test_cellprofiler_contract_executor_preserves_non_flow_main_carrier_outputs():
    calls: list[tuple[int, ...]] = []

    def combine_like(image: np.ndarray):
        slice_index = len(calls)
        calls.append(image.shape)
        main_carrier = np.arange(5, dtype=np.float32) + slice_index
        measurements = [{"slice": slice_index}]
        labels = np.full(image.shape, slice_index + 1, dtype=np.int32)
        return main_carrier, measurements, labels

    combine_like.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((2, 5, 6), dtype=np.float32)

    main_carrier, measurements, labels = CellProfilerFunctionContractExecutor().execute(
        combine_like,
        stack,
        {},
        output_aggregation_contract=(
            CellProfilerFunctionOutputAggregationContract.from_main_flow_replacement(
                False
            )
        ),
    )

    assert calls == [(5, 6), (5, 6)]
    assert isinstance(main_carrier, RuntimeSliceAlignedValues)
    assert main_carrier.slice_count == 2
    np.testing.assert_array_equal(
        main_carrier.value_for_slice(0),
        np.arange(5, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        main_carrier.value_for_slice(1),
        np.arange(5, dtype=np.float32) + 1,
    )
    assert measurements == [
        {"slice": 0, "slice_index": 0},
        {"slice": 1, "slice_index": 1},
    ]
    assert labels.shape == stack.shape
    np.testing.assert_array_equal(labels[0], np.ones((5, 6), dtype=np.int32))
    np.testing.assert_array_equal(labels[1], np.full((5, 6), 2, dtype=np.int32))


def test_cellprofiler_contract_executor_stacks_singleton_plane_outputs():
    def add_singleton_plane(image: np.ndarray) -> np.ndarray:
        return image[np.newaxis, ...] + 1

    add_singleton_plane.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((2, 4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(
        add_singleton_plane,
        stack,
        {},
    )

    assert result.shape == stack.shape
    np.testing.assert_array_equal(result, np.ones_like(stack))


def test_cellprofiler_contract_executor_stacks_singleton_color_outputs():
    def add_singleton_color_plane(image: np.ndarray) -> np.ndarray:
        rgb = np.repeat(image[..., np.newaxis], 3, axis=-1)
        return rgb[np.newaxis, ...] + 1

    add_singleton_color_plane.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((2, 4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(
        add_singleton_color_plane,
        stack,
        {},
    )

    assert result.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(result, np.ones((2, 4, 5, 3), dtype=np.uint16))


def test_cellprofiler_contract_executor_stacks_singleton_volume_outputs():
    def add_singleton_volume(image: np.ndarray) -> np.ndarray:
        volume = np.stack((image, image + 1), axis=0)
        return volume[np.newaxis, ...]

    add_singleton_volume.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((3, 4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(
        add_singleton_volume,
        stack,
        {},
    )

    assert result.shape == (3, 2, 4, 5)
    np.testing.assert_array_equal(result[:, 1], np.ones((3, 4, 5), dtype=np.uint16))


def test_complete_object_measurement_rows_uses_declared_label_domain() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 4), dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_count=3,
        ),
    )

    rows = complete_object_measurement_rows(
        [],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3]
    assert all(np.isnan(row["value"]) for row in rows)


def test_complete_object_measurement_rows_handles_empty_rows_with_axis_fields() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 4), dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_count=2,
        ),
    )

    rows = complete_object_measurement_rows(
        [],
        label_payload=payload,
        func=_synthetic_axis_object_measurement_function,
    )

    assert [row["object_label"] for row in rows] == [1, 2]
    assert all(np.isnan(row["image_number"]) for row in rows)
    assert all(np.isnan(row["value"]) for row in rows)


def test_complete_object_measurement_rows_preserves_sliced_object_label_set_domain() -> (
    None
):
    labels = np.zeros((2, 4, 4), dtype=np.int32)
    labels[0, 1, 1] = 1
    labels[1, 2, 2] = 1
    payload = ObjectLabelSet(
        name="GridObjects",
        labels=labels,
        source_image_name="BF_image",
        domain=ObjectLabelDomain(
            declared_object_count=3,
        ),
    )

    rows = complete_object_measurement_rows(
        [
            {"slice_index": 0, "object_label": 1, "value": 10.0},
            {"slice_index": 1, "object_label": 1, "value": 20.0},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
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


def test_global_image_number_projection_ignores_missing_axis_values() -> None:
    rows = [
        {"slice_index": 0, "object_label": 1, "value": 1.0},
        {"slice_index": np.nan, "object_label": 2, "value": np.nan},
    ]

    projected, projected_mappings = CellProfilerMeasurementProjectionRequest(
        adapter=_FakeCellProfilerRuntime({}),
        rows=rows,
        object_name=None,
        need_row_mappings=True,
    ).project_rows()

    assert projected is projected_mappings
    assert projected[0]["image_number"] == 1
    assert "image_number" not in projected[1]


def test_global_image_number_projection_applies_to_columnar_rows() -> None:
    rows = _ColumnarMeasurementRows(
        {
            MeasurementRowAxisField.IMAGE_NUMBER.value: (1, 2, np.nan),
            "object_label": (1, 2, 3),
            "value": (10.0, 20.0, 30.0),
        }
    )

    projected, projected_mappings = CellProfilerMeasurementProjectionRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            image_number_start=23,
            ordered_pipeline_image_paths=("well-a",),
        ),
        rows=rows,
        object_name=None,
        need_row_mappings=True,
    ).project_rows()

    assert projected is projected_mappings
    assert tuple(
        columnar_row_values(projected, MeasurementRowAxisField.IMAGE_NUMBER.value)
    )[:2] == (23, 24)
    assert np.isnan(
        columnar_row_values(projected, MeasurementRowAxisField.IMAGE_NUMBER.value)[2]
    )


def test_columnar_projection_prefers_slice_index_over_local_image_number() -> None:
    rows = _ColumnarMeasurementRows(
        {
            MeasurementRowAxisField.SLICE_INDEX.value: (0, 1),
            MeasurementRowAxisField.IMAGE_NUMBER.value: (1, 2),
            "object_label": (1, 1),
            "value": (10.0, 20.0),
        }
    )

    projected, _projected_mappings = CellProfilerMeasurementProjectionRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            image_number_start=99,
            image_numbers_by_source_path={
                "/source/site1_ch2.tif": 1,
                "/source/site2_ch2.tif": 2,
            },
        ),
        source_context=CellProfilerMeasurementSourceContext(
            source_image_name=None,
            source_image_payload=RuntimeImagePayloadContext(
                np.zeros((2, 2, 2), dtype=np.float32),
                metadata=ImagePayloadMetadata(
                    source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                        paths=(
                            "/source/site1_ch2.tif",
                            "/source/site2_ch2.tif",
                        )
                    )
                ),
                mask=None,
            ).payload(),
        ),
        rows=rows,
        object_name=None,
        need_row_mappings=True,
    ).project_rows()

    assert tuple(
        columnar_row_values(projected, MeasurementRowAxisField.IMAGE_NUMBER.value)
    ) == (1, 2)


def test_measurement_materializer_preserves_declared_cellprofiler_image_numbers() -> (
    None
):
    runtime = _FakeCellProfilerRuntime(
        {},
        image_number_start=2,
        ordered_pipeline_image_paths=("first.tif", "second.tif"),
    )
    rows = [
        {"image_number": 1, "object_label": 1, "area": 10.0},
        {"image_number": 2, "object_label": 1, "area": 20.0},
    ]

    CellProfilerMeasurementMaterializer.record(
        CellProfilerMeasurementMaterializationRequest.for_rows(
            adapter=runtime,
            name="AreaShape",
            rows=rows,
            object_name="Cells",
            axis_state=MeasurementRowAxisState.IMAGE_NUMBER,
        )
    )

    _name, recorded_rows, _kwargs = runtime.measurements[-1]
    assert [row["image_number"] for row in recorded_rows] == [1, 2]


def test_measurement_materializer_uses_single_image_number_source_path() -> None:
    runtime = _FakeCellProfilerRuntime(
        {},
        ordered_pipeline_image_paths=("first.tif", "second.tif"),
    )

    CellProfilerMeasurementMaterializer.record(
        CellProfilerMeasurementMaterializationRequest.for_rows(
            adapter=runtime,
            name="ImageSetScopedMeasurement",
            rows=[{"image_number": 2, "area": 20.0}],
            object_name=None,
            axis_state=MeasurementRowAxisState.IMAGE_NUMBER,
        )
    )

    _name, recorded_rows, kwargs = runtime.measurements[-1]
    assert recorded_rows[0]["image_number"] == 2
    assert kwargs["source_path"] == "second.tif"


def test_cellprofiler_measurement_output_axis_state_preserves_declared_image_numbers() -> (
    None
):
    rows = [
        {"image_number": 1, "object_label": 1, "area": 10.0},
        {"image_number": 2, "object_label": 1, "area": 20.0},
    ]

    assert (
        CellProfilerMeasurementOutputAxisState.for_rows(rows)
        is MeasurementRowAxisState.IMAGE_NUMBER
    )


def test_cellprofiler_measurement_output_axis_state_projects_runtime_slice_rows() -> (
    None
):
    rows = [
        {"slice_index": 0, "object_label": 1, "area": 10.0},
        {"slice_index": 1, "object_label": 1, "area": 20.0},
    ]

    assert (
        CellProfilerMeasurementOutputAxisState.for_rows(rows)
        is MeasurementRowAxisState.RUNTIME_AXES
    )


def test_cellprofiler_measurement_output_axis_state_projects_slice_rows_with_stale_image_numbers() -> (
    None
):
    rows = _ColumnarMeasurementRows(
        {
            "slice_index": (0, 1),
            "image_number": (1, 1),
            "object_label": (1, 1),
            "area": (10.0, 20.0),
        }
    )

    assert (
        CellProfilerMeasurementOutputAxisState.for_rows(rows)
        is MeasurementRowAxisState.RUNTIME_AXES
    )


def test_object_measurement_projection_uses_object_label_source_paths_for_start() -> (
    None
):
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=np.zeros((2, 2, 2), dtype=np.uint16),
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=("/source/site1.tif", "/source/site2.tif")
                ),
            )
        },
        image_number_start=99,
        ordered_pipeline_image_paths=("/source/site1.tif", "/source/site2.tif"),
    )

    projected, _projected_mappings = CellProfilerMeasurementProjectionRequest(
        adapter=runtime,
        rows=(
            {"slice_index": 0, "object_label": 1, "area": 10.0},
            {"slice_index": 1, "object_label": 1, "area": 20.0},
        ),
        source_context=CellProfilerMeasurementSourceContext(
            source_image_name=None,
            source_image_payload=RuntimeImagePayloadContext(
                np.zeros((2, 2), dtype=np.float32),
                metadata=ImagePayloadMetadata(source_path="/source/site2.tif"),
                mask=None,
            ).payload(),
        ),
        object_name="Cells",
        source_resolver=CellProfilerMeasurementSourceResolver(
            object_source_lookup=AdapterObjectLabelSourceLookup(adapter=runtime),
        ),
    ).project_rows()

    assert [row["image_number"] for row in projected] == [1, 2]


def test_image_measurement_projection_maps_source_qualified_rows_by_source_name() -> (
    None
):
    source_paths = ("/source/blue.tif", "/source/green.tif")
    runtime = _FakeCellProfilerRuntime(
        {},
        image_numbers_by_source_path={
            "/source/blue.tif": 4,
            "/source/green.tif": 5,
        },
    )
    source_payload = RuntimeImagePayloadContext(
        np.zeros((2, 2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_names=("OrigBlue", "OrigGreen"),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=source_paths,
            ),
        ),
        mask=None,
    ).payload()

    projected, _projected_mappings = CellProfilerMeasurementProjectionRequest(
        adapter=runtime,
        rows=(
            {
                "source_image_name": "OrigBlue",
                "feature_name": "ImageQuality_FocusScore_OrigBlue",
                "result_value": 0.1,
            },
            {
                "source_image_name": "OrigGreen",
                "feature_name": "ImageQuality_FocusScore_OrigGreen",
                "result_value": 0.2,
            },
        ),
        source_context=CellProfilerMeasurementSourceContext(
            source_image_payload=source_payload,
        ),
    ).project_rows()

    assert [row["image_number"] for row in projected] == [4, 5]


def test_image_measurement_projection_uses_metadata_without_payload() -> None:
    runtime = _FakeCellProfilerRuntime(
        {},
        image_numbers_by_source_path={
            "/source/tissue-site1.tif": 11,
        },
    )

    projected, _projected_mappings = CellProfilerMeasurementProjectionRequest(
        adapter=runtime,
        rows=(
            {
                "source_image_name": "Tissue",
                "feature_name": "AreaOccupied_AreaOccupied_Tissue",
                "result_value": 42.0,
            },
        ),
        source_context=CellProfilerMeasurementSourceContext(
            source_metadata=ImagePayloadMetadata(
                source_image_names=("Tissue",),
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=("/source/tissue-site1.tif",),
                ),
            ),
        ),
    ).project_rows()

    assert [row["image_number"] for row in projected] == [11]


def test_image_measurement_projection_resolves_source_name_when_payload_paths_are_derived() -> (
    None
):
    class SourceNamedRuntime(_FakeCellProfilerRuntime):
        def cellprofiler_source_paths_for_image_name(
            self,
            image_name: str | None,
        ) -> tuple[str, ...]:
            if image_name == "OrigBlue":
                return ("/source/blue.tif",)
            if image_name == "OrigGreen":
                return ("/source/green.tif",)
            return ()

    runtime = SourceNamedRuntime(
        {},
        image_numbers_by_source_path={
            "/source/blue.tif": 4,
            "/source/green.tif": 5,
        },
    )
    source_payload = RuntimeImagePayloadContext(
        np.zeros((2, 2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_names=("OrigBlue", "OrigGreen"),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/runtime/CropBlue_site1.tif", "/runtime/CropBlue_site2.tif"),
            ),
        ),
        mask=None,
    ).payload()

    projected, _projected_mappings = CellProfilerMeasurementProjectionRequest(
        adapter=runtime,
        rows=(
            {
                "source_image_name": "OrigBlue",
                "feature_name": "Texture_Correlation_OrigBlue",
                "result_value": 0.1,
            },
            {
                "source_image_name": "OrigGreen",
                "feature_name": "Texture_Correlation_OrigGreen",
                "result_value": 0.2,
            },
        ),
        source_context=CellProfilerMeasurementSourceContext(
            source_image_payload=source_payload,
        ),
    ).project_rows()

    assert [row["image_number"] for row in projected] == [4, 5]


def test_image_measurement_projection_maps_unqualified_row_to_single_source() -> None:
    runtime = _FakeCellProfilerRuntime(
        {},
        image_numbers_by_source_path={"/source/blue.tif": 4},
    )
    source_payload = RuntimeImagePayloadContext(
        np.zeros((1, 2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_names=("OrigBlue",),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/source/blue.tif",),
            ),
        ),
        mask=None,
    ).payload()

    projected, _projected_mappings = CellProfilerMeasurementProjectionRequest(
        adapter=runtime,
        rows=(
            {
                "source_image_name": None,
                "feature_name": "Texture_Correlation_OrigBlue",
                "result_value": 0.1,
            },
        ),
        source_context=CellProfilerMeasurementSourceContext(
            source_image_payload=source_payload,
        ),
    ).project_rows()

    assert [row["image_number"] for row in projected] == [4]


def test_measurement_materializer_projects_source_metadata_from_runtime_slice_rows() -> (
    None
):
    source_paths = ("/source/site1.tif", "/source/site2.tif")
    source_metadata = ({"well": "A01", "site": "1"}, {"well": "A01", "site": "2"})
    cells = ObjectLabelSet(
        name="Cells",
        labels=np.zeros((2, 2, 2), dtype=np.uint16),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=source_paths,
            component_metadata=source_metadata,
        ),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={"Cells": cells},
        image_numbers_by_source_path={
            "/source/site1.tif": 1,
            "/source/site2.tif": 2,
        },
    )
    rows = [{"slice_index": 1, "object_label": 1, "area": 20.0}]

    CellProfilerMeasurementMaterializer.record(
        CellProfilerMeasurementMaterializationRequest.for_rows(
            adapter=runtime,
            name="AreaShape",
            rows=rows,
            object_name="Cells",
            source_resolver=CellProfilerMeasurementSourceResolver(
                output_values={"Cells": cells},
            ),
        )
    )

    _name, recorded_rows, kwargs = runtime.measurements[-1]
    assert recorded_rows[0]["image_number"] == 2
    assert kwargs["source_path"] == "/source/site2.tif"
    assert dict(kwargs["source_component_metadata"]) == {"well": "A01", "site": "2"}
    assert "source_image_provenance_planes" not in kwargs


def test_object_measurement_projection_maps_channel_slices_to_same_image_set() -> None:
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=np.zeros((2, 2, 2), dtype=np.uint16),
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=(
                        "/source/site1_ch1.tif",
                        "/source/site1_ch2.tif",
                    )
                ),
            )
        },
        image_number_start=99,
        image_numbers_by_source_path={
            "/source/site1_ch1.tif": 1,
            "/source/site1_ch2.tif": 1,
        },
    )

    projected, _projected_mappings = CellProfilerMeasurementProjectionRequest(
        adapter=runtime,
        rows=(
            {"slice_index": 0, "object_label": 1, "area": 10.0},
            {"slice_index": 1, "object_label": 1, "area": 20.0},
        ),
        object_name="Cells",
        source_resolver=CellProfilerMeasurementSourceResolver(
            object_source_lookup=AdapterObjectLabelSourceLookup(adapter=runtime),
        ),
    ).project_rows()

    assert [row["image_number"] for row in projected] == [1, 2]


def test_global_image_number_projection_uses_source_payload_for_columnar_rows() -> None:
    rows = _ColumnarMeasurementRows(
        {
            MeasurementRowAxisField.IMAGE_NUMBER.value: (1,),
            "object_label": (1,),
            "value": (10.0,),
        }
    )
    source_payload = RuntimeImagePayloadContext(
        np.zeros((1, 1), dtype=np.float32),
        metadata=ImagePayloadMetadata(source_path="well-h12-w1.tif"),
        mask=None,
    ).payload()

    projected, _projected_mappings = CellProfilerMeasurementProjectionRequest(
        adapter=_FakeCellProfilerRuntime(
            {},
            image_number_start=1,
            ordered_pipeline_image_paths=(
                "well-a01-w1.tif",
                "well-h12-w1.tif",
            ),
        ),
        source_context=CellProfilerMeasurementSourceContext(
            source_image_name="rawGFP",
            source_image_payload=source_payload,
        ),
        rows=rows,
        object_name=None,
        need_row_mappings=True,
    ).project_rows()

    assert tuple(
        columnar_row_values(projected, MeasurementRowAxisField.IMAGE_NUMBER.value)
    ) == (2,)


def test_per_object_materializer_partitions_row_owned_columnar_tables_for_projection() -> None:
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=np.zeros((2, 2, 2), dtype=np.uint16),
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=("/source/cells-site1.tif", "/source/cells-site2.tif")
                ),
            ),
            "Nuclei": ObjectLabelSet(
                name="Nuclei",
                labels=np.zeros((2, 2, 2), dtype=np.uint16),
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=("/source/nuclei-site1.tif", "/source/nuclei-site2.tif")
                ),
            ),
        },
        ordered_pipeline_image_paths=(
            "/source/cells-site1.tif",
            "/source/cells-site2.tif",
            "/source/nuclei-site1.tif",
            "/source/nuclei-site2.tif",
        ),
    )
    rows = _ColumnarMeasurementRows(
        {
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
    )

    CellProfilerMeasurementMaterializer.record_table(
        adapter=runtime,
        spec=ArtifactSpec.output("ObjectMeasurements", MeasurementsArtifactType),
        func=_synthetic_object_measurement_function,
        rows=rows,
        object_name=None,
        source_context=CellProfilerMeasurementSourceContext(source_image_name="DNA"),
        measurement_row_policy=DefaultObjectMeasurementRowPolicy(),
    )

    projected_by_object = {
        kwargs["object_name"]: [row["image_number"] for row in recorded_rows]
        for _name, recorded_rows, kwargs in runtime.measurements
    }
    assert projected_by_object == {
        "Cells": [1, 2],
        "Nuclei": [3, 4],
    }


def test_measure_object_intensity_zero_fills_missing_positive_extent() -> None:
    payload = ObjectLabelPayload(
        labels=np.asarray([[1, 0, 3]], dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_count=5,
        ),
    )
    row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
        "MeasureObjectIntensity"
    )

    rows = complete_object_measurement_rows(
        [{"object_label": 1, "value": 7.0}],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
        object_identity=row_policy.object_identity(),
        row_policy=row_policy,
    )
    by_label = {row["object_label"]: row for row in rows}

    assert by_label[2]["value"] == 0.0
    assert by_label[3]["value"] == 0.0
    assert np.isnan(by_label[4]["value"])
    assert np.isnan(by_label[5]["value"])


def test_measure_object_intensity_direct_slice_uses_matching_label_plane() -> None:
    image = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    labels = ObjectLabelPayload(
        labels=np.asarray(
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 2]],
            ],
            dtype=np.int32,
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    _image, rows = measure_object_intensity(
        image,
        labels,
        slice_index=1,
    )
    row_mappings = rows.row_mappings()

    assert [row["object_label"] for row in row_mappings] == [2]
    assert row_mappings[0]["integrated_intensity"] == 4.0


def test_measure_object_size_shape_direct_slice_uses_matching_label_plane() -> None:
    image = np.zeros((4, 4), dtype=np.float32)
    labels = ObjectLabelPayload(
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
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    _image, rows = measure_object_size_shape(
        image,
        labels,
        calculate_advanced=False,
        calculate_zernikes=False,
        slice_index=1,
    )

    assert [(row["object_label"], row["Area"]) for row in rows] == [(2, 4.0)]


def test_pure_2d_slice_execution_injects_slice_index_for_declared_callables() -> None:
    seen: list[int] = []

    @runtime_bound_parameters(SliceIndexRuntimeParameter)
    def records_slice_index(image, *, slice_index: int = 0):
        seen.append(slice_index)
        return image

    CellProfilerFunctionContractExecutor().execute_pure_2d_slice(
        records_slice_index,
        np.zeros((2, 2), dtype=np.float32),
        {},
        2,
        3,
    )

    assert seen == [2]


def test_complete_object_measurement_rows_uses_slice_local_label_domain() -> None:
    labels = np.zeros((2, 3, 5), dtype=np.int32)
    labels[0, 0, 0] = 1
    labels[0, 0, 2] = 3
    labels[1, 0, 0] = 1
    labels[1, 0, 1] = 2
    payload = ObjectLabelPayload(labels=labels)
    row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
        "MeasureObjectIntensity"
    )

    rows = complete_object_measurement_rows(
        [
            {"slice_index": 0, "object_label": 1, "value": 10.0},
            {"slice_index": 0, "object_label": 3, "value": 30.0},
            {"slice_index": 1, "object_label": 1, "value": 100.0},
            {"slice_index": 1, "object_label": 2, "value": 200.0},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
        object_identity=row_policy.object_identity(),
        row_policy=row_policy,
    )

    values_by_key = {
        (row["slice_index"], row["object_label"]): row["value"] for row in rows
    }
    assert values_by_key == {
        (0, 1): 10.0,
        (0, 3): 30.0,
        (1, 1): 100.0,
        (1, 2): 200.0,
    }


def test_complete_object_measurement_rows_orders_sparse_label_domain() -> None:
    payload = ObjectLabelPayload(
        labels=np.asarray([[1, 0, 3]], dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_count=5,
        ),
    )

    rows = complete_object_measurement_rows(
        [
            {"object_label": 3, "value": 30.0},
            {"object_label": 1, "value": 10.0},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[0]["value"] == 10.0
    assert np.isnan(rows[1]["value"])
    assert rows[2]["value"] == 30.0


def test_complete_object_measurement_rows_preserves_measurement_axes() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 4), dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_count=2,
        ),
    )

    rows = complete_object_measurement_rows(
        [
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
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
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
        labels=np.zeros((4, 4), dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_ids=(10, 20, 30, 40, 50),
        ),
    )

    rows = complete_object_measurement_rows(
        [
            {"object_label": 10, "Area": 10.0},
            {"object_label": 30, "Area": 30.0},
            {"object_label": 50, "Area": 50.0},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
        object_identity=MeasurementObjectRowIdentity.ROW_ORDINAL,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["Area"] for row in rows[:3]] == [10.0, 30.0, 50.0]
    assert np.isnan(rows[3]["Area"])
    assert np.isnan(rows[4]["Area"])


def test_declared_domain_compact_rows_preserve_unmeasured_object_ordinals() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 4), dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_ids=(10, 20, 30),
        ),
    )

    rows = complete_object_measurement_rows(
        [
            {"object_label": 10, "Area": 10.0},
            {"object_label": 30, "Area": 30.0},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
        row_policy=DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy(),
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3]
    assert rows[0]["Area"] == 10.0
    assert np.isnan(rows[1]["Area"])
    assert rows[2]["Area"] == 30.0


def test_measure_texture_compact_rows_preserve_declared_padding_domain() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 4), dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_count=5,
        ),
    )
    row_policy = CellProfilerObjectMeasurementRowPolicy.for_module("MeasureTexture")

    rows = row_policy.complete_rows(
        [
            {"object_label": 10, "angular_second_moment": 0.1},
            {"object_label": 30, "angular_second_moment": 0.3},
            {"object_label": 50, "angular_second_moment": 0.5},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["angular_second_moment"] for row in rows[:3]] == [0.1, 0.3, 0.5]
    assert all(np.isnan(row["angular_second_moment"]) for row in rows[3:])


def test_measure_texture_multi_source_plane_domain_zero_fills_missing_extent() -> None:
    labels = np.zeros((2, 2, 4, 4), dtype=np.int32)
    labels[0, 0, 0, 0] = 5
    payload = ObjectLabelPayload(
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_names=("BF_image", "MorphBf"),
        domain=ObjectLabelDomain(
            declared_object_count=5,
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    row_policy = CellProfilerObjectMeasurementRowPolicy.for_module("MeasureTexture")

    rows = row_policy.complete_rows(
        [
            {"object_label": 1, "angular_second_moment": 0.1},
            {"object_label": 2, "angular_second_moment": 0.2},
            {"object_label": 3, "angular_second_moment": np.nan},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["angular_second_moment"] for row in rows[:2]] == [0.1, 0.2]
    assert [row["angular_second_moment"] for row in rows[2:]] == [0.0, 0.0, 0.0]


def test_measure_object_intensity_missing_rows_use_measured_axis_extent() -> None:
    payload = ObjectLabelPayload(
        labels=np.asarray(
            (
                [[1, 2, 3]],
                [[1, 2, 3]],
            ),
            dtype=np.int32,
        ),
        domain=ObjectLabelDomain(declared_object_count=3),
    )
    row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
        "MeasureObjectIntensity"
    )

    rows = row_policy.complete_rows(
        [
            {
                "slice_index": 0,
                "image_number": 1,
                "object_label": 1,
                "integrated_intensity": 10.0,
            },
            {
                "slice_index": 0,
                "image_number": 1,
                "object_label": 3,
                "integrated_intensity": 30.0,
            },
            {
                "slice_index": 1,
                "image_number": 2,
                "object_label": 1,
                "integrated_intensity": 11.0,
            },
            {
                "slice_index": 1,
                "image_number": 2,
                "object_label": 2,
                "integrated_intensity": 22.0,
            },
        ],
        label_payload=payload,
        func=_synthetic_axis_object_measurement_function,
    )

    by_axis_object = {(row["image_number"], row["object_label"]): row for row in rows}
    assert by_axis_object[(1, 2)]["integrated_intensity"] == 0.0
    assert np.isnan(by_axis_object[(2, 3)]["integrated_intensity"])


def test_measure_object_intensity_columnar_rows_use_declared_axis_domain() -> None:
    payload = ObjectLabelPayload(
        labels=np.asarray([[1, 2]], dtype=np.int32),
        domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
    )
    row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
        "MeasureObjectIntensity"
    )

    rows = row_policy.complete_rows(
        _ColumnarMeasurementRows(
            {
                "image_number": (1, 1, 1),
                "object_label": (1, 2, 3),
                "integrated_intensity": (10.0, 20.0, 0.0),
            }
        ),
        label_payload=payload,
        func=_synthetic_axis_object_measurement_function,
    )

    assert [row["object_label"] for row in rows] == [1, 2]
    assert [row["integrated_intensity"] for row in rows] == [10.0, 20.0]


def test_measure_object_size_shape_compact_rows_preserve_emitted_padding() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 4), dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_ids=(10, 20, 30, 40, 50),
        ),
    )
    row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
        "MeasureObjectSizeShape"
    )

    rows = complete_object_measurement_rows(
        [
            {"object_label": 10, "Area": 10.0, "Center_X": 10.0},
            {"object_label": 20},
            {"object_label": 25, "Area": np.nan},
            {"object_label": 30, "Area": 30.0, "Center_X": 30.0},
            {"object_label": 50, "Area": 50.0, "Center_X": 50.0},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
        object_identity=row_policy.object_identity(),
        row_policy=row_policy,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["Area"] for row in rows[:3]] == [10.0, 30.0, 50.0]
    assert np.isnan(rows[3]["Area"])
    assert np.isnan(rows[4]["Area"])


def test_measure_object_size_shape_compacts_zero_valued_padding_rows() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 4), dtype=np.int32),
        domain=ObjectLabelDomain(declared_object_count=5),
    )
    row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
        "MeasureObjectSizeShape"
    )

    rows = row_policy.complete_rows(
        [
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
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[0]["Center_X"] == 10.0
    assert rows[1]["Center_X"] == 30.0
    assert np.isnan(rows[2]["Center_X"])
    assert np.isnan(rows[3]["Center_X"])
    assert np.isnan(rows[4]["Center_X"])
    assert rows[0]["MaxFeretDiameter"] == 11.0
    assert rows[1]["MaxFeretDiameter"] == 33.0
    assert rows[2]["MaxFeretDiameter"] == 0.0
    assert np.isnan(rows[3]["MaxFeretDiameter"])
    assert np.isnan(rows[4]["MaxFeretDiameter"])
    assert {row[MeasurementRowAxisField.OBJECT_ROW_IDENTITY.value] for row in rows} == {
        MeasurementObjectRowIdentity.ROW_ORDINAL.value
    }


def test_measure_object_size_shape_compacts_complete_dense_domain_rows() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 4), dtype=np.int32),
        domain=ObjectLabelDomain(declared_object_count=3),
    )
    row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
        "MeasureObjectSizeShape"
    )

    rows = row_policy.complete_rows(
        [
            {"object_label": 1, "Center_X": 10.0},
            {"object_label": 2, "Center_X": np.nan},
            {"object_label": 3, "Center_X": 30.0},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3]
    assert rows[0]["Center_X"] == 10.0
    assert rows[1]["Center_X"] == 30.0
    assert np.isnan(rows[2]["Center_X"])


def test_measure_object_size_shape_uses_compact_row_identity_policy() -> None:
    assert (
        CellProfilerObjectMeasurementRowPolicy.for_module(
            "MeasureObjectSizeShape"
        ).object_identity()
        is MeasurementObjectRowIdentity.ROW_ORDINAL
    )
    assert (
        CellProfilerObjectMeasurementRowPolicy.for_module(
            "MeasureObjectIntensityDistribution"
        ).object_identity()
        is MeasurementObjectRowIdentity.ROW_ORDINAL
    )
    assert (
        CellProfilerObjectMeasurementRowPolicy.for_module(
            "MeasureObjectIntensity"
        ).object_identity()
        is MeasurementObjectRowIdentity.LABEL_ID
    )


def test_per_image_measurements_use_registered_record_builder() -> None:
    def threshold_like(image):
        return image, ThresholdResult(
            slice_index=0,
            final_threshold=0.3,
            original_threshold=0.2,
            guide_threshold=0.0,
            sigma=1.0,
            weighted_variance=0.4,
            sum_of_entropies=0.5,
        )

    threshold_like.__processing_contract__ = ProcessingContract.PURE_2D
    runtime = _FakeCellProfilerRuntime(
        {"phase": _FakeRuntimeImage(np.ones((4, 4), dtype=np.float32))}
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Threshold",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("phase", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition, ()
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Threshold_5_measurements", MeasurementsArtifactType
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output("phaseThresh", ImageArtifactType),
                        ArtifactSpec.output(
                            "Threshold_5_measurements", MeasurementsArtifactType
                        ),
                    ),
                ),
            ),
        )
    )

    result = executor.run(
        threshold_like,
        np.ones((4, 4), dtype=np.float32),
        cellprofiler_runtime=runtime,
    )

    assert result.shape == (4, 4)
    assert len(runtime.measurements) == 1
    _name, rows, kwargs = runtime.measurements[0]
    assert kwargs["source_image_name"] == "phase"
    assert {row["feature_name"]: row["result_value"] for row in rows} == {
        "FinalThreshold_phaseThresh": 0.3,
        "OrigThreshold_phaseThresh": 0.2,
        "WeightedVariance_phaseThresh": 0.4,
        "SumOfEntropies_phaseThresh": 0.5,
    }


def test_object_only_measurement_preserves_runtime_slice_carrier_stack() -> None:
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 0:2, 0:2] = 1
    labels[1, 0:3, 0:3] = 1
    runtime = _FakeCellProfilerRuntime(
        {},
        {
            "Nuclei": ObjectLabelSet(
                name="Nuclei",
                labels=labels,
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                domain=ObjectLabelDomain(
                    scope=ObjectLabelDomainScope.PLANE,
                ),
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectSizeShape",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "MeasureObjectSizeShape_1_measurements",
                            MeasurementsArtifactType,
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "MeasureObjectSizeShape_1_measurements",
                            MeasurementsArtifactType,
                        ),
                    ),
                ),
            ),
        )
    )

    current_image = np.zeros((2, 5, 5), dtype=np.float32)

    result = executor.run(
        measure_object_size_shape,
        current_image,
        cellprofiler_runtime=runtime,
        calculate_advanced=False,
        calculate_zernikes=False,
    )

    assert result is current_image
    assert len(runtime.measurements) == 1
    _name, rows, _kwargs = runtime.measurements[0]
    assert [(row["slice_index"], row["object_label"], row["area"]) for row in rows] == [
        (0, 1, 4.0),
        (1, 1, 9.0),
    ]


def test_per_object_measurement_reuses_2d_labels_for_each_image_stack_slice() -> None:
    @dataclass(frozen=True)
    class SliceObjectMeasurement:
        slice_index: int
        object_label: int
        value: float

    calls: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []

    def measure(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape, float(image[0, 0])))
        return image, [
            SliceObjectMeasurement(
                slice_index=0, object_label=1, value=float(image[0, 0])
            ),
            SliceObjectMeasurement(
                slice_index=0, object_label=2, value=float(image[0, 1])
            ),
        ]

    measure.__processing_contract__ = ProcessingContract.PURE_2D
    image_stack = np.stack(
        (
            np.asarray([[10.0, 11.0], [0.0, 0.0]], dtype=np.float32),
            np.asarray([[20.0, 21.0], [0.0, 0.0]], dtype=np.float32),
        )
    )
    labels = np.asarray([[1, 2], [0, 0]], dtype=np.int32)
    runtime = _FakeCellProfilerRuntime(
        {"Intensity": _FakeRuntimeImage(image_stack)},
        {
            "Objects": ObjectLabelSet(
                name="Objects",
                labels=labels,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectIntensity",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("Intensity", ImageArtifactType),
                        ArtifactSpec.input("Objects", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("Intensity", ImageArtifactType),
                        ArtifactSpec.input("Objects", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("ObjectIntensity", MeasurementsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("ObjectIntensity", MeasurementsArtifactType),),
                ),
            ),
        )
    )

    result = executor.run(measure, image_stack, cellprofiler_runtime=runtime)

    assert result is image_stack
    assert calls == [((2, 2), (2, 2), 10.0), ((2, 2), (2, 2), 20.0)]
    assert len(runtime.measurements) == 1
    _name, rows, kwargs = runtime.measurements[0]
    assert kwargs["object_name"] == "Objects"
    assert {
        (row["slice_index"], row["object_label"], row["value"]) for row in rows
    } == {
        (0, 1, 10.0),
        (0, 2, 11.0),
        (1, 1, 20.0),
        (1, 2, 21.0),
    }


def test_per_object_measurement_batch_preserves_measurement_image_major_order() -> None:
    from openhcs.core.runtime_batch_contracts import measurement_image_batch_executor

    seen: list[tuple[int, int, str | None, float, int]] = []

    def batch_measurements(func, requests, execute):
        seen.extend(
            (
                request.batch_index,
                request.batch_count,
                request.source_image_name,
                float(np.asarray(request.image)[0, 0]),
                int(np.asarray(request.kwargs["labels"])[0, 0]),
            )
            for request in requests
        )
        return [execute(func, request) for request in requests]

    @measurement_image_batch_executor(batch_measurements)
    def measure(image: np.ndarray, *, labels: np.ndarray):
        return image, [
            _SyntheticObjectMeasurement(
                object_label=int(labels[0, 0]),
                value=float(image[0, 0]),
            )
        ]

    measure.__processing_contract__ = ProcessingContract.PURE_2D

    runtime = _FakeCellProfilerRuntime(
        {
            "IntensityA": _FakeRuntimeImage(
                np.asarray([[10.0]], dtype=np.float32),
                source_image_name="IntensityA",
            ),
            "IntensityB": _FakeRuntimeImage(
                np.asarray([[20.0]], dtype=np.float32),
                source_image_name="IntensityB",
            ),
        },
        {
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=np.asarray([[1]], dtype=np.int32),
            ),
            "Nuclei": ObjectLabelSet(
                name="Nuclei",
                labels=np.asarray([[2]], dtype=np.int32),
            ),
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectIntensity",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("IntensityA", ImageArtifactType),
                        ArtifactSpec.input("IntensityB", ImageArtifactType),
                        ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("IntensityA", ImageArtifactType),
                        ArtifactSpec.input("IntensityB", ImageArtifactType),
                        ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("ObjectIntensity", MeasurementsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("ObjectIntensity", MeasurementsArtifactType),),
                ),
            ),
        )
    )

    executor.run(
        measure,
        np.asarray([[0.0]], dtype=np.float32),
        cellprofiler_runtime=runtime,
    )

    assert seen == [
        (0, 4, "IntensityA", 10.0, 1),
        (1, 4, "IntensityA", 10.0, 2),
        (2, 4, "IntensityB", 20.0, 1),
        (3, 4, "IntensityB", 20.0, 2),
    ]


def test_per_object_measurement_batch_executor_runs_for_single_invocation() -> None:
    from openhcs.core.runtime_batch_contracts import measurement_image_batch_executor

    seen: list[tuple[int, int, str | None]] = []

    def batch_measurements(func, requests, execute):
        seen.extend(
            (
                request.batch_index,
                request.batch_count,
                request.source_image_name,
            )
            for request in requests
        )
        return [
            execute(
                func,
                replace(
                    request,
                    kwargs={**request.kwargs, "batch_context": "declared"},
                ),
            )
            for request in requests
        ]

    @measurement_image_batch_executor(batch_measurements)
    def measure(
        image: np.ndarray,
        *,
        labels: np.ndarray,
        batch_context: str,
    ):
        return image, [
            _SyntheticObjectMeasurement(
                object_label=int(labels[0, 0]),
                value=float(batch_context == "declared"),
            )
        ]

    measure.__processing_contract__ = ProcessingContract.PURE_2D

    runtime = _FakeCellProfilerRuntime(
        {
            "Intensity": _FakeRuntimeImage(
                np.asarray([[10.0]], dtype=np.float32),
                source_image_name="Intensity",
            ),
        },
        {
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=np.asarray([[1]], dtype=np.int32),
            ),
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectIntensity",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("Intensity", ImageArtifactType),
                        ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("Intensity", ImageArtifactType),
                        ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("ObjectIntensity", MeasurementsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("ObjectIntensity", MeasurementsArtifactType),),
                ),
            ),
        )
    )

    executor.run(
        measure,
        np.asarray([[0.0]], dtype=np.float32),
        cellprofiler_runtime=runtime,
    )

    assert seen == [(0, 1, "Intensity")]
    name, rows, kwargs = _recorded_measurements_for_assertion(runtime.measurements)[0]
    assert name == "ObjectIntensity"
    assert kwargs["object_name"] == "Cells"
    assert {"object_label": 1, "value": 1.0, "slice_index": 0} in rows


def test_object_intensity_measurement_image_batch_preserves_request_labels() -> None:
    from openhcs.core.runtime_invocation import RuntimeBatchInvocationRequest
    from openhcs.processing.backends.cellprofiler._backend import (
        DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    )
    from openhcs.processing.backends.cellprofiler.intensity import (
        measure_object_intensity,
        measure_object_intensity_measurement_image_batch,
    )

    labels_a = ObjectLabelSet(
        name="Cells",
        labels=np.asarray([[[1]], [[2]]], dtype=np.int32),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("ImageA", "ImageB"),
    )
    labels_b = ObjectLabelSet(
        name="Cells",
        labels=np.asarray([[[10]], [[20]]], dtype=np.int32),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("ImageA", "ImageB"),
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
        labels=np.asarray([[[1]], [[2]]], dtype=np.int32),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
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
        labels=np.asarray([[[1, 0]], [[2, 3]]], dtype=np.int32),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
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


def test_per_object_measurement_records_declared_empty_measurement_table() -> None:
    def measure(image: np.ndarray, *, labels: np.ndarray):
        return image, []

    measure.__processing_contract__ = ProcessingContract.PURE_2D
    measure.__special_outputs__ = (
        (
            "ObjectSizeShape",
            csv_materializer(fields=("object_label", "area")),
        ),
    )
    image = np.ones((3, 3), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {"Intensity": _FakeRuntimeImage(image)},
        {
            "Objects": ObjectLabelSet(
                name="Objects",
                labels=np.zeros(image.shape, dtype=np.int32),
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectSizeShape",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("Intensity", ImageArtifactType),
                        ArtifactSpec.input("Objects", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("Intensity", ImageArtifactType),
                        ArtifactSpec.input("Objects", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "ObjectSizeShape",
                            MeasurementsArtifactType,
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "ObjectSizeShape",
                            MeasurementsArtifactType,
                        ),
                    ),
                ),
            ),
        )
    )

    result = executor.run(measure, image, cellprofiler_runtime=runtime)

    assert result is image
    assert len(runtime.measurements) == 1
    name, rows, kwargs = runtime.measurements[0]
    assert name == "ObjectSizeShape"
    assert rows == []
    assert kwargs["object_name"] == "Objects"
    assert tuple(field.name for field in kwargs["fields"]) == (
        "object_label",
        "area",
    )


def test_measurement_record_fields_prefers_artifact_materialization_schema() -> None:
    spec = ArtifactSpec.output(
        name="measurements",
        artifact_type=MeasurementsArtifactType,
        materialization=csv_materializer(fields=["object_label", "area"]),
    )

    fields = CellProfilerMeasurementFieldSchema.for_record(
        spec, [], measure_object_size_shape
    )

    assert tuple(field.name for field in fields) == ("object_label", "area")


def test_measure_object_size_shape_declares_schema_on_special_output() -> None:
    spec = ArtifactSpec.output("measurements", MeasurementsArtifactType)

    fields = CellProfilerMeasurementFieldSchema.for_record(
        spec, [], measure_object_size_shape
    )

    assert (
        tuple(field.name for field in fields)
        == MeasureObjectSizeShapeModule.measurement_all_field_names()
    )


def test_measurement_record_fields_infer_concatenated_columnar_rows() -> None:
    rows = ConcatenatedMeasurementColumnarRows(
        (
            _ColumnarMeasurementRows(
                {
                    "object_name": ("Cells",),
                    "object_label": (1,),
                    "mean_intensity": (0.5,),
                }
            ),
            _ColumnarMeasurementRows(
                {
                    "object_name": ("Cells",),
                    "object_label": (2,),
                    "mean_intensity": (0.8,),
                }
            ),
        )
    )

    record = CellProfilerMeasurementRecord(rows=rows)

    assert tuple(field.name for field in record.fields) == (
        "object_name",
        "object_label",
        "mean_intensity",
    )


def test_fields_from_rows_measurement_record_mixin_accepts_row_carriers() -> None:
    @dataclass(frozen=True)
    class _StatsRow:
        slice_index: int
        object_count: int

    class _RowsFieldRecordModule(FieldsFromRowsMeasurementRecordMixin):
        pass

    _rows, fields = _RowsFieldRecordModule.measurement_record_fields(
        SimpleNamespace(),
        [
            _StatsRow(slice_index=0, object_count=3),
            {"location_center_x": 12.0, "location_center_y": 18.0},
        ],
    )

    assert fields is not None
    assert tuple(field.name for field in fields) == (
        "slice_index",
        "object_count",
        "location_center_x",
        "location_center_y",
    )


def test_measure_object_size_shape_outputs_basic_measurement_rows() -> None:
    image = np.ones((7, 7), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:4, 1:4] = 1

    _image, rows = measure_object_size_shape(
        image,
        labels,
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert len(rows) == 1
    assert rows[0]["object_label"] == 1
    assert rows[0]["Area"] == 9.0
    assert rows[0]["Center_X"] == 2.0
    assert rows[0]["Center_Y"] == 2.0


def test_measure_object_size_shape_exports_skimage_perimeter() -> None:
    image = np.ones((9, 9), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    y, x = np.ogrid[-2:3, -2:3]
    labels[2:7, 2:7][x * x + y * y <= 4] = 1

    _image, rows = measure_object_size_shape(
        image,
        labels,
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
        labels,
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
        labels,
        calculate_advanced=True,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert rows[0]["Orientation"] == 45.0


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
        labels,
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
        labels,
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
            labels=labels,
            domain=ObjectLabelDomain(declared_object_count=3),
        ),
        calculate_advanced=False,
        calculate_zernikes=True,
        dtype_config=DtypeConfig(),
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3]
    assert np.isfinite(rows[0]["Zernike_0_0"])
    assert np.isnan(rows[1]["Area"])
    assert np.isnan(rows[1]["Center_X"])
    assert np.isnan(rows[1]["Center_Y"])
    assert np.isfinite(rows[1]["Zernike_0_0"])
    assert np.isnan(rows[2]["Zernike_0_0"])
    assert rows[2]["Area"] == 16.0


def test_measure_object_size_shape_declared_count_uses_measured_dense_extent() -> None:
    image = np.ones((12, 12), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[6:10, 6:10] = 3

    _image, rows = measure_object_size_shape(
        image,
        ObjectLabelPayload(
            labels=labels,
            domain=ObjectLabelDomain(declared_object_count=5),
        ),
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3]
    assert rows[0]["Area"] == 9.0
    assert np.isnan(rows[1]["Area"])
    assert rows[2]["Area"] == 16.0


def test_measure_object_size_shape_uses_present_domain_for_undeclared_dense_labels() -> (
    None
):
    image = np.ones((12, 12), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[6:10, 6:10] = 1000

    _image, rows = measure_object_size_shape(
        image,
        labels,
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert [row["object_label"] for row in rows] == [1, 1000]
    assert rows[1]["Center_X"] == 7.5
    assert rows[1]["Center_Y"] == 7.5


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
                object_name="Objects",
                rows=(
                    {
                        "object_label": 1,
                        "FormFactor": exported_form_factor,
                    },
                ),
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    assert exported_form_factor > 1.0
    _output_image, stats, filtered_labels = result[:3]
    assert stats.objects_post_filter == 0
    assert object_label_dense_array(filtered_labels).max() == 0


def test_filterobjects_derives_form_factor_when_measurement_table_is_absent() -> None:
    image = np.ones((9, 9), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    y, x = np.ogrid[-2:3, -2:3]
    labels[2:7, 2:7][x * x + y * y <= 4] = 1

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
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_labels = result[:3]
    assert stats.objects_post_filter == 1
    assert object_label_dense_array(filtered_labels).max() == 1


def test_cellprofiler_contract_executor_stacks_color_slice_outputs():
    calls = []

    def colorize(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return np.stack((image, image, image), axis=-1)

    colorize.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((2, 4, 5), dtype=np.float32)

    result = CellProfilerFunctionContractExecutor().execute(colorize, stack, {})

    assert calls == [(4, 5), (4, 5)]
    assert result.shape == (2, 4, 5, 3)


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
    np.testing.assert_array_equal(result[..., 1], np.full((4, 5), 0.25, dtype=np.float32))
    np.testing.assert_array_equal(result[..., 2], np.full((4, 5), 0.75, dtype=np.float32))


def test_color_to_gray_combines_openhcs_color_stack() -> None:
    image = np.zeros((2, 4, 5, 3), dtype=np.float32)
    image[..., 0] = 2.0
    image[..., 1] = 4.0
    image[..., 2] = 6.0

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


def test_color_to_gray_combine_retypes_color_source_stack_as_grayscale() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((2, 4, 5, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001.tif",
            source_component_metadata={
                SOURCE_IMAGE_TYPE_METADATA_FIELD: "Color image",
                "site": "1",
            },
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
    source_role = source_image_payload_role(result)
    assert source_role is not None
    assert dict(metadata.source_component_metadata) == {
        SOURCE_IMAGE_TYPE_METADATA_FIELD: source_role.image_type(),
        "site": "1",
    }
    assert metadata.source_image_provenance_planes.count == 0
    assert not source_role.is_channel_last_source_plane(result.data)


def test_color_to_gray_splits_openhcs_color_slice_by_selected_channels() -> None:
    image = np.zeros((4, 5, 3), dtype=np.float32)
    image[..., 0] = 1.0
    image[..., 1] = 2.0
    image[..., 2] = 3.0

    red, blue = color_to_gray(
        image,
        mode="split",
        image_type="rgb",
        channel_indices=(0, 2),
        dtype_config=DtypeConfig(),
    )

    assert red.shape == (4, 5)
    assert blue.shape == (4, 5)
    np.testing.assert_array_equal(red, np.ones((4, 5), dtype=np.float32))
    np.testing.assert_array_equal(blue, np.full((4, 5), 3.0, dtype=np.float32))


def test_color_to_gray_rgb_split_projects_selected_source_plane_metadata() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((4, 5, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w1.tif",
                    "/input/A01_s001_w2.tif",
                    "/input/A01_s001_w3.tif",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "1", "channel": "2"},
                    {"well": "A01", "site": "1", "channel": "3"},
                ),
            ),
            source_image_names=("OrigRed", "OrigGreen", "OrigBlue"),
        ),
    )
    image.data[..., 1] = 2.0

    (green,) = color_to_gray(
        image,
        mode="split",
        image_type="rgb",
        channel_indices=(1,),
        dtype_config=DtypeConfig(),
    )

    metadata = image_payload_metadata(green)
    assert metadata.source_path == "/input/A01_s001_w2.tif"
    assert metadata.source_component_metadata == {
        "well": "A01",
        "site": "1",
        "channel": "2",
    }
    assert metadata.source_image_names == ("OrigGreen",)
    assert metadata.source_provenance.source_plane_count == 0
    np.testing.assert_array_equal(green.data, np.full((4, 5), 2.0, dtype=np.float32))


def test_color_to_gray_splits_channel_last_non_rgb_slice() -> None:
    image = np.zeros((4, 5, 2), dtype=np.float32)
    image[..., 0] = 7.0
    image[..., 1] = 11.0

    (first_channel,) = color_to_gray(
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

    (red,) = color_to_gray(
        MaskedImagePayload(data=image, mask=mask),
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


def test_aligned_payload_slices_masked_image_stacks() -> None:
    stack = np.zeros((2, 4, 5), dtype=np.float32)
    mask = np.array(
        (
            np.ones((4, 5), dtype=bool),
            np.zeros((4, 5), dtype=bool),
        )
    )

    slices = payload_slices_for_alignment(MaskedImagePayload(data=stack, mask=mask))

    assert len(slices) == 2
    assert all(
        isinstance(slice_payload, MaskedImagePayload) for slice_payload in slices
    )
    np.testing.assert_array_equal(slices[0].mask, mask[0])
    np.testing.assert_array_equal(slices[1].mask, mask[1])


def test_aligned_payload_slices_source_tagged_grayscale_stack() -> None:
    stack = np.zeros((2, 4, 5), dtype=np.float32)
    payload = ImageMetadataPayload(
        data=stack,
        metadata=ImagePayloadMetadata(
            source_path="/tmp/source.tif",
            source_component_metadata={
                SOURCE_IMAGE_TYPE_METADATA_FIELD: "Grayscale image",
            },
        ),
    )

    slices = payload_slices_for_alignment(payload)

    assert len(slices) == 2
    assert all(isinstance(slice_payload, ImageMetadataPayload) for slice_payload in slices)
    np.testing.assert_array_equal(slices[0].data, stack[0])
    np.testing.assert_array_equal(slices[1].data, stack[1])


def test_aligned_payload_slices_source_tagged_two_channel_color_stack() -> None:
    stack = np.zeros((2, 4, 5, 2), dtype=np.float32)
    payload = ImageMetadataPayload(
        data=stack,
        metadata=ImagePayloadMetadata(
            source_path="/tmp/source.tif",
            source_component_metadata={
                SOURCE_IMAGE_TYPE_METADATA_FIELD: "Color image",
            },
        ),
    )

    slices = payload_slices_for_alignment(payload)

    assert len(slices) == 2
    assert all(isinstance(slice_payload, ImageMetadataPayload) for slice_payload in slices)
    np.testing.assert_array_equal(slices[0].data, stack[0])
    np.testing.assert_array_equal(slices[1].data, stack[1])


def test_aligned_payload_slices_masked_volume_channel_stacks() -> None:
    stack = np.zeros((2, 3, 4, 5), dtype=np.float32)
    mask = np.zeros_like(stack, dtype=bool)
    mask[0] = True

    slices = payload_slices_for_alignment(MaskedImagePayload(data=stack, mask=mask))

    assert len(slices) == 2
    assert all(
        isinstance(slice_payload, MaskedImagePayload) for slice_payload in slices
    )
    np.testing.assert_array_equal(slices[0].mask, mask[0])
    np.testing.assert_array_equal(slices[1].mask, mask[1])


def test_aligned_payload_slices_preserve_image_metadata() -> None:
    stack = np.zeros((2, 4, 5), dtype=np.float32)
    payload = ImageMetadataPayload(
        data=stack,
        metadata=ImagePayloadMetadata(
            source_plane_intensity_scales=(65535.0, 255.0),
            source_plane_dtypes=("uint16", "uint8"),
        ),
    )

    slices = payload_slices_for_alignment(payload)

    assert len(slices) == 2
    assert slices[0].metadata.intensity_scale == 65535.0
    assert slices[0].metadata.source_dtype == "uint16"
    assert slices[1].metadata.intensity_scale == 255.0
    assert slices[1].metadata.source_dtype == "uint8"


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
    first = RuntimeImagePayloadContext(
        np.zeros((1, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(intensity_scale=65535.0, source_dtype="uint16"),
        mask=None,
    ).payload()
    second = RuntimeImagePayloadContext(
        np.ones((1, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(intensity_scale=255.0, source_dtype="uint8"),
        mask=None,
    ).payload()

    stacked = CellProfilerPure2DOutputAggregator.aggregate(
        [first, second],
        "numpy",
    )

    assert isinstance(stacked, ImageMetadataPayload)
    assert image_payload_data(stacked).shape == (2, 4, 5)
    assert (
        image_payload_metadata(stacked).for_source_plane(0).intensity_scale == 65535.0
    )
    assert image_payload_metadata(stacked).for_source_plane(1).source_dtype == "uint8"


def test_pure_2d_declared_object_label_output_preserves_single_slice_axis() -> None:
    contract = CellProfilerFunctionOutputAggregationContract.from_main_flow_replacement(
        True,
        declared_output_specs=(
            ArtifactSpec.output("Image", ImageArtifactType),
            ArtifactSpec.output("Cells", ObjectLabelsArtifactType),
        ),
    )
    labels = np.asarray(((0, 1), (0, 0)), dtype=np.int32)

    (aggregated,) = contract.aggregate_auxiliary_outputs(
        ([labels],),
        MemoryType.NUMPY.value,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.domain.scope is ObjectLabelDomainScope.PLANE
    np.testing.assert_array_equal(
        object_label_dense_array(aggregated),
        labels[np.newaxis, ...],
    )


def test_full_stack_pure_2d_declared_object_label_output_restores_slice_axis() -> None:
    image = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    labels = np.asarray(((0, 1), (0, 0)), dtype=np.int32)

    def segment_like(stack: np.ndarray) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
        assert stack.shape == (1, 2, 2)
        return stack, {"count": 1}, labels

    segment_like.__processing_contract__ = ProcessingContract.PURE_2D

    result_image, result_measurements, result_labels = (
        CellProfilerFunctionContractExecutor().execute(
            segment_like,
            image,
            {},
            execution_mode=ImagePayloadExecutionMode.FULL_STACK,
            output_aggregation_contract=(
                CellProfilerFunctionOutputAggregationContract.from_main_flow_replacement(
                    True,
                    (
                        ArtifactSpec.output("Image", ImageArtifactType),
                        ArtifactSpec.output("Measurements", MeasurementsArtifactType),
                        ArtifactSpec.output("Labels", ObjectLabelsArtifactType),
                    ),
                )
            ),
        )
    )

    assert result_image is image
    assert result_measurements == {"count": 1}
    assert isinstance(result_labels, ObjectLabelPayload)
    assert result_labels.domain.scope is ObjectLabelDomainScope.PLANE
    np.testing.assert_array_equal(
        object_label_dense_array(result_labels),
        labels[np.newaxis, ...],
    )


def test_full_stack_pure_2d_implicit_main_output_aligns_declared_object_labels() -> None:
    image = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    labels = np.asarray([[[0, 1], [0, 0]]], dtype=np.int32)

    def watershed_like(
        stack: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
        assert stack.shape == (1, 2, 2)
        return stack, {"count": 1}, labels

    watershed_like.__processing_contract__ = ProcessingContract.PURE_2D

    result_image, result_measurements, result_labels = (
        CellProfilerFunctionContractExecutor().execute(
            watershed_like,
            image,
            {},
            execution_mode=ImagePayloadExecutionMode.FULL_STACK,
            output_aggregation_contract=(
                CellProfilerFunctionOutputAggregationContract.from_main_flow_replacement(
                    True,
                    (
                        ArtifactSpec.output("Measurements", MeasurementsArtifactType),
                        ArtifactSpec.output("Labels", ObjectLabelsArtifactType),
                    ),
                )
            ),
        )
    )

    assert result_image is image
    assert result_measurements == {"count": 1}
    assert isinstance(result_labels, ObjectLabelPayload)
    assert result_labels.domain.scope is ObjectLabelDomainScope.PAYLOAD
    np.testing.assert_array_equal(object_label_dense_array(result_labels), labels)


def test_full_stack_pure_2d_non_flow_main_keeps_relationship_output_alignment() -> None:
    image = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    labels = np.asarray(((0, 1), (0, 0)), dtype=np.int32)
    relationship = ParentChildRelationshipPayload(parent_ids=(1,), child_ids=(1,))

    def object_transform_like(
        stack: np.ndarray,
    ) -> tuple[
        np.ndarray,
        dict[str, int],
        ParentChildRelationshipPayload,
        np.ndarray,
    ]:
        assert stack.shape == (1, 2, 2)
        return stack, {"count": 1}, relationship, labels

    object_transform_like.__processing_contract__ = ProcessingContract.PURE_2D

    result_image, result_measurements, result_relationship, result_labels = (
        CellProfilerFunctionContractExecutor().execute(
            object_transform_like,
            image,
            {},
            execution_mode=ImagePayloadExecutionMode.FULL_STACK,
            output_aggregation_contract=(
                CellProfilerFunctionOutputAggregationContract.from_main_flow_replacement(
                    False,
                    (
                        ArtifactSpec.output("Measurements", MeasurementsArtifactType),
                        ArtifactSpec.output(
                            "Parent_Child_relationships",
                            RelationshipsArtifactType,
                        ),
                        ArtifactSpec.output("Labels", ObjectLabelsArtifactType),
                    ),
                )
            ),
        )
    )

    assert result_image is image
    assert result_measurements == {"count": 1}
    assert result_relationship is relationship
    assert isinstance(result_labels, ObjectLabelPayload)
    assert result_labels.domain.scope is ObjectLabelDomainScope.PLANE
    np.testing.assert_array_equal(
        object_label_dense_array(result_labels),
        labels[np.newaxis, ...],
    )


def test_pure_2d_columnar_output_aggregation_stamps_outer_slice_identity() -> None:
    first = _ColumnarMeasurementRows(
        {
            "slice_index": (0,),
            "object_label": (1,),
            "area": (4.0,),
        }
    )
    second = _ColumnarMeasurementRows(
        {
            "slice_index": (0,),
            "object_label": (1,),
            "area": (9.0,),
        }
    )

    aggregated = CellProfilerPure2DOutputAggregator.aggregate(
        [first, second],
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, ColumnarRows)
    assert tuple(aggregated.columns["slice_index"]) == (0, 1)
    assert tuple(aggregated.columns["area"]) == (4.0, 9.0)


def test_module_executor_rewraps_single_image_output_for_openhcs_main_flow() -> None:
    def to_gray(image: np.ndarray) -> np.ndarray:
        return image[..., 0]

    to_gray.__processing_contract__ = ProcessingContract.PURE_2D
    color_slice = np.zeros((4, 5, 3), dtype=np.float32)
    color_stack = color_slice[np.newaxis, ...]
    runtime = _FakeCellProfilerRuntime(
        {"OrigColor": _FakeRuntimeImage(color_slice, source_image_name="OrigColor")}
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="ColorToGray",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("OrigColor", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("OrigGray", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("OrigGray", ImageArtifactType),),
                ),
            ),
        )
    )

    result = executor.run(to_gray, color_stack, cellprofiler_runtime=runtime)

    assert result.shape == (1, 4, 5)
    assert runtime.images["OrigGray"].data.shape == (4, 5)


def test_module_executor_replaces_main_flow_for_declared_image_fan_in_output() -> None:
    def add_channels(image: np.ndarray) -> np.ndarray:
        return image.sum(axis=0)

    add_channels.__processing_contract__ = ProcessingContract.PURE_3D
    channel_stack = np.stack(
        (
            np.full((2, 5, 6), 1.0, dtype=np.float32),
            np.full((2, 5, 6), 2.0, dtype=np.float32),
            np.full((2, 5, 6), 3.0, dtype=np.float32),
        )
    )
    runtime = _FakeCellProfilerRuntime(
        {"Orig": _FakeRuntimeImage(channel_stack, source_image_name="Orig")}
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="ImageMath",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Orig", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Combined", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Combined", ImageArtifactType),),
                ),
            ),
        )
    )

    result = executor.run(add_channels, channel_stack, cellprofiler_runtime=runtime)

    assert result.shape == (1, *channel_stack.shape[1:])
    np.testing.assert_allclose(result[0], 6.0)
    assert runtime.images["Combined"].data.shape == channel_stack.shape[1:]


def test_module_executor_preserves_duplicate_image_roles_for_illumination_apply():
    illumination = np.full((4, 5), 2.0, dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {"IllumGreen": _FakeRuntimeImage(illumination, source_image_name="IllumGreen")}
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="CorrectIlluminationApply",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("IllumGreen", ImageArtifactType),
                        ArtifactSpec.input("IllumGreen", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("IllumGreen", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("CorrGreen", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("CorrGreen", ImageArtifactType),),
                ),
            ),
        )
    )

    result = executor.run(
        correct_illumination_apply,
        illumination,
        cellprofiler_runtime=runtime,
        method="divide",
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_allclose(image_payload_data(result), np.ones((1, 4, 5)))
    np.testing.assert_allclose(
        image_payload_data(runtime.images["CorrGreen"].data),
        np.ones((4, 5)),
    )


def test_illumination_apply_multi_image_outputs_replace_main_flow_with_declared_outputs():
    orig_red = np.full((4, 5), 0.2, dtype=np.float32)
    illum_red = np.full((4, 5), 0.5, dtype=np.float32)
    orig_green = np.full((4, 5), 0.6, dtype=np.float32)
    illum_green = np.full((4, 5), 0.3, dtype=np.float32)
    orig_red_payload = RuntimeImagePayloadContext(
        orig_red,
        mask=None,
        metadata=ImagePayloadMetadata(
            source_path="/plate/A01_s001_w1_z001_t001.tif",
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
            },
            source_image_names=("OrigRed",),
        ),
    ).payload()
    orig_green_payload = RuntimeImagePayloadContext(
        orig_green,
        mask=None,
        metadata=ImagePayloadMetadata(
            source_path="/plate/A01_s001_w2_z001_t001.tif",
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "2",
            },
            source_image_names=("OrigGreen",),
        ),
    ).payload()
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigRed": _FakeRuntimeImage(orig_red_payload, source_image_name="OrigRed"),
            "IllumRed": _FakeRuntimeImage(illum_red, source_image_name="IllumRed"),
            "OrigGreen": _FakeRuntimeImage(
                orig_green_payload,
                source_image_name="OrigGreen",
            ),
            "IllumGreen": _FakeRuntimeImage(
                illum_green,
                source_image_name="IllumGreen",
            ),
        }
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="CorrectIlluminationApply",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("OrigRed", ImageArtifactType),
                        ArtifactSpec.input("IllumRed", ImageArtifactType),
                        ArtifactSpec.input("OrigGreen", ImageArtifactType),
                        ArtifactSpec.input("IllumGreen", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output("CorrectedRed", ImageArtifactType),
                        ArtifactSpec.output("CorrectedGreen", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output("CorrectedRed", ImageArtifactType),
                        ArtifactSpec.output("CorrectedGreen", ImageArtifactType),
                    ),
                ),
            ),
        )
    )

    result = executor.run(
        correct_illumination_apply,
        np.stack((orig_red, orig_green)),
        cellprofiler_runtime=runtime,
        method=("divide", "divide"),
        truncate_low=False,
        truncate_high=False,
    )

    assert isinstance(result, AlignedImageStack)
    assert tuple(context.output_key for context in result.slice_contexts) == (
        "CorrectedRed",
        "CorrectedGreen",
    )
    np.testing.assert_allclose(image_payload_data(result.slices[0]), 0.4)
    np.testing.assert_allclose(image_payload_data(result.slices[1]), 2.0)
    np.testing.assert_allclose(
        image_payload_data(runtime.images["CorrectedRed"].data),
        0.4,
    )
    np.testing.assert_allclose(
        image_payload_data(runtime.images["CorrectedGreen"].data),
        2.0,
    )
    assert image_payload_metadata(result.slices[0]).source_component_metadata == {
        "well": "A01",
        "site": "1",
        "channel": "1",
    }
    assert image_payload_metadata(result.slices[1]).source_component_metadata == {
        "well": "A01",
        "site": "1",
        "channel": "2",
    }
    assert image_payload_metadata(result.slices[0]).source_image_names == ("OrigRed",)
    assert image_payload_metadata(result.slices[1]).source_image_names == ("OrigGreen",)


def test_illumination_apply_image_output_uses_original_input_source_payload() -> None:
    source_paths = (
        "/plate/IXMtest_A01_s1_w5.tif",
        "/plate/IXMtest_A01_s2_w5.tif",
    )
    source_metadata = (
        {"Well": "A01", "Site": "1", "ChannelNumber": "5"},
        {"Well": "A01", "Site": "2", "ChannelNumber": "5"},
    )
    orig_mito = RuntimeImagePayloadContext(
        np.stack(
            (
                np.ones((3, 4), dtype=np.float32),
                np.full((3, 4), 2.0, dtype=np.float32),
            )
        ),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=source_paths, component_metadata=source_metadata
            ),
            source_image_names=("OrigMito",),
        ),
        mask=None,
    ).payload()
    stale_invocation_payload = RuntimeImagePayloadContext(
        np.zeros((2, 3, 4), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(source_paths[0], source_paths[0]),
                component_metadata=(source_metadata[0], source_metadata[0]),
            ),
            source_image_names=("OrigHoechst", "OrigMito"),
        ),
        mask=None,
    ).payload()
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigMito": _FakeRuntimeImage(orig_mito),
            "IllumMito": _FakeRuntimeImage(np.ones((3, 4), dtype=np.float32)),
        }
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="CorrectIlluminationApply",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("OrigMito", ImageArtifactType),
                        ArtifactSpec.input("IllumMito", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Mito", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Mito", ImageArtifactType),),
                ),
            ),
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
            executor=executor,
            adapter=runtime,
            spec=ArtifactSpec.output("Mito", ImageArtifactType),
            output_value=output,
            output_values={"Mito": output},
            source_image_name=None,
            source_aliases=("OrigMito", "IllumMito"),
            source_image_payload=stale_invocation_payload,
            current_image=np.zeros((2, 3, 4), dtype=np.float32),
            func=lambda image: image,
            call_kwargs={},
        )
    )

    recorded_metadata = image_payload_metadata(runtime.images["Mito"].data)
    assert recorded_metadata.source_image_provenance_planes.paths == source_paths
    assert (
        recorded_metadata.source_image_provenance_planes.component_metadata
        == source_metadata
    )
    assert recorded_metadata.source_image_names == ("OrigMito",)


def test_illumination_apply_image_output_collapses_duplicate_source_plane_stack() -> (
    None
):
    source_path = "/plate/IXMtest_A01_s1_w5.tif"
    source_metadata = {"Well": "A01", "Site": "1", "ChannelNumber": "5"}
    source_payload = RuntimeImagePayloadContext(
        np.ones((5, 6), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path=source_path,
            source_component_metadata=source_metadata,
            source_image_names=("OrigMito",),
        ),
        mask=None,
    ).payload()
    duplicate_output = RuntimeImagePayloadContext(
        np.stack(
            (
                np.full((5, 6), 10.0, dtype=np.float32),
                np.full((5, 6), 20.0, dtype=np.float32),
            )
        ),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(source_path, source_path),
                component_metadata=(source_metadata, source_metadata),
            ),
            source_image_names=("OrigMito", "OrigSyto"),
        ),
        mask=None,
    ).payload()
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigMito": _FakeRuntimeImage(source_payload),
            "IllumMito": _FakeRuntimeImage(np.ones((5, 6), dtype=np.float32)),
        }
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="CorrectIlluminationApply",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("OrigMito", ImageArtifactType),
                        ArtifactSpec.input("IllumMito", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Mito", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Mito", ImageArtifactType),),
                ),
            ),
        )
    )

    CellProfilerOutputRecorder.for_artifact_type(ImageArtifactType).record(
        _cellprofiler_output_record_request(
            executor=executor,
            adapter=runtime,
            spec=ArtifactSpec.output("Mito", ImageArtifactType),
            output_value=duplicate_output,
            output_values={"Mito": duplicate_output},
            source_image_name=None,
            source_aliases=("OrigMito", "IllumMito"),
            source_image_payload=duplicate_output,
            current_image=np.zeros((5, 6), dtype=np.float32),
            func=lambda image: image,
            call_kwargs={},
        )
    )

    recorded = runtime.images["Mito"].data
    np.testing.assert_allclose(
        image_payload_data(recorded),
        np.full((5, 6), 10.0, dtype=np.float32),
    )
    recorded_metadata = image_payload_metadata(recorded)
    assert recorded_metadata.source_path == source_path
    assert recorded_metadata.source_component_metadata == source_metadata
    assert recorded_metadata.source_image_provenance_planes.paths == ()
    assert recorded_metadata.source_image_provenance_planes.component_metadata == ()
    assert recorded_metadata.source_image_names == ("OrigMito",)


def test_illumination_apply_image_output_collapses_singleton_contextual_stack() -> None:
    source_path = "/plate/IXMtest_A01_s2_w5.tif"
    source_metadata = {"Well": "A01", "Site": "2", "ChannelNumber": "5"}
    source_payload = RuntimeImagePayloadContext(
        np.ones((5, 6), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path=source_path,
            source_component_metadata=source_metadata,
            source_image_names=("OrigMito",),
        ),
        mask=None,
    ).payload()
    singleton_output = RuntimeImagePayloadContext(
        np.full((1, 5, 6), 30.0, dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path=source_path,
            source_component_metadata=source_metadata,
            source_image_names=("OrigMito",),
        ),
        mask=None,
    ).payload()
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigMito": _FakeRuntimeImage(source_payload),
            "IllumMito": _FakeRuntimeImage(np.ones((5, 6), dtype=np.float32)),
        }
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="CorrectIlluminationApply",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("OrigMito", ImageArtifactType),
                        ArtifactSpec.input("IllumMito", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Mito", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Mito", ImageArtifactType),),
                ),
            ),
        )
    )

    CellProfilerOutputRecorder.for_artifact_type(ImageArtifactType).record(
        _cellprofiler_output_record_request(
            executor=executor,
            adapter=runtime,
            spec=ArtifactSpec.output("Mito", ImageArtifactType),
            output_value=singleton_output,
            output_values={"Mito": singleton_output},
            source_image_name=None,
            source_aliases=("OrigMito", "IllumMito"),
            source_image_payload=singleton_output,
            current_image=np.zeros((5, 6), dtype=np.float32),
            func=lambda image: image,
            call_kwargs={},
        )
    )

    recorded = runtime.images["Mito"].data
    np.testing.assert_allclose(
        image_payload_data(recorded),
        np.full((5, 6), 30.0, dtype=np.float32),
    )
    recorded_metadata = image_payload_metadata(recorded)
    assert recorded_metadata.source_path == source_path
    assert recorded_metadata.source_component_metadata == source_metadata
    assert recorded_metadata.source_image_names == ("OrigMito",)


def test_cellprofiler_contract_executor_slices_aligned_runtime_kwargs():
    calls = []

    def keep_labels(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape))
        return image, labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((2, 4, 5), dtype=np.uint16)
    labels = np.ones_like(stack, dtype=np.int32)

    result_image, result_labels = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        stack,
        {"labels": labels},
    )

    assert calls == [((4, 5), (4, 5)), ((4, 5), (4, 5))]
    assert result_image.shape == stack.shape
    assert result_labels.shape == labels.shape


def test_cellprofiler_contract_executor_aggregates_object_label_payload_auxiliary():
    def keep_payload(image: np.ndarray):
        labels = np.full(image.shape, int(image[0, 0]) + 1, dtype=np.int32)
        return (
            image,
            ObjectLabelPayload(
                labels=labels,
                unedited_labels=labels + 10,
                small_removed_labels=labels + 20,
            ),
        )

    keep_payload.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.stack(
        (
            np.zeros((4, 5), dtype=np.uint16),
            np.ones((4, 5), dtype=np.uint16),
        )
    )

    result_image, result_payload = CellProfilerFunctionContractExecutor().execute(
        keep_payload,
        stack,
        {},
    )

    assert result_image.shape == stack.shape
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
        labels = np.full(
            image_payload_data(image).shape,
            int(image_payload_metadata(image).source_component_metadata["site"]),
            dtype=np.int32,
        )
        return (
            image,
            SourceImageObjectLabelBuildRequest(
                image=image,
                labels=labels,
            ).payload(),
        )

    segment.__processing_contract__ = ProcessingContract.PURE_2D
    image = RuntimeImagePayloadContext(
        np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w1_z001_t001.TIF",
                    "/input/A01_s002_w1_z001_t001.TIF",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "2", "channel": "1"},
                ),
            )
        ),
        mask=None,
    ).payload()

    result_image, result_payload = CellProfilerFunctionContractExecutor().execute(
        segment,
        image,
        {},
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
    first_site = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
            },
            source_image_names=("OrigBlue",),
        ),
    ).payload()
    second_site = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A01",
                "site": "2",
                "channel": "1",
            },
            source_image_names=("OrigBlue",),
        ),
    ).payload()

    result_image, result_payload = CellProfilerFunctionContractExecutor().execute(
        segment,
        AlignedImageStack((first_site, second_site)),
        {},
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
    )

    assert image_payload_data(result_image).shape == (2, 4, 5)
    assert isinstance(result_payload, ObjectLabelSet)
    assert result_payload.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert result_payload.source_image_names == ("OrigBlue", "OrigBlue")
    np.testing.assert_array_equal(result_payload.labels[0], np.full((4, 5), 1))
    np.testing.assert_array_equal(result_payload.labels[1], np.full((4, 5), 2))


def test_measurement_main_flow_keeps_input_without_source_measurement_images():
    input_image = np.zeros((4, 5), dtype=np.float32)
    object_image = CellProfilerMeasurementImage(
        source_image_name=None,
        payload=np.ones((4, 5), dtype=np.uint16),
    )

    result = CELLPROFILER_MEASUREMENT_MAIN_FLOW.output_image(
        input_image=input_image,
        measurement_images=(object_image,),
        variable_components=(),
        parser=None,
        identity_cache=FunctionOutputIdentityCache(),
    )

    assert result is input_image


def test_measurement_main_flow_publishes_single_source_measurement_image():
    input_image = np.zeros((4, 5), dtype=np.float32)
    source_payload = np.ones((4, 5), dtype=np.float32)
    source_image = CellProfilerMeasurementImage(
        source_image_name="OrigDNA",
        payload=source_payload,
    )

    result = CELLPROFILER_MEASUREMENT_MAIN_FLOW.output_image(
        input_image=input_image,
        measurement_images=(source_image,),
        variable_components=(),
        parser=None,
        identity_cache=FunctionOutputIdentityCache(),
    )

    assert result is source_payload


def test_measurement_main_flow_keeps_input_for_single_source_stack_varying_outside_axes():
    input_image = np.zeros((4, 5), dtype=np.float32)
    source_stack = RuntimeImagePayloadContext(
        np.stack(
            (
                np.full((4, 5), 1, dtype=np.float32),
                np.full((4, 5), 2, dtype=np.float32),
            )
        ),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w1_z001_t001.tif",
                    "/input/A01_s001_w2_z001_t001.tif",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "1"},
                    {"well": "A01", "site": "1", "channel": "2"},
                ),
            ),
            source_image_names=("OrigPair",),
        ),
    ).payload()

    result = CELLPROFILER_MEASUREMENT_MAIN_FLOW.output_image(
        input_image=input_image,
        measurement_images=(
            CellProfilerMeasurementImage(
                source_image_name="OrigPair",
                payload=source_stack,
            ),
        ),
        variable_components=(VariableComponents.SITE,),
        parser=None,
        identity_cache=FunctionOutputIdentityCache(),
    )

    assert result is input_image


def test_measurement_main_flow_keeps_input_for_multiple_source_measurement_images():
    input_image = np.zeros((4, 5), dtype=np.float32)
    dna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 1, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
            },
            source_image_names=("OrigDNA",),
        ),
    ).payload()
    rna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 2, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "2",
            },
            source_image_names=("OrigRNA",),
        ),
    ).payload()

    result = CELLPROFILER_MEASUREMENT_MAIN_FLOW.output_image(
        input_image=input_image,
        measurement_images=(
            CellProfilerMeasurementImage(
                source_image_name="OrigDNA",
                payload=dna_payload,
            ),
            CellProfilerMeasurementImage(
                source_image_name="OrigRNA",
                payload=rna_payload,
            ),
        ),
        variable_components=(VariableComponents.CHANNEL,),
        parser=None,
        identity_cache=FunctionOutputIdentityCache(),
    )

    assert result is input_image


def test_measurement_main_flow_keeps_input_for_composed_source_measurement_image():
    input_image = np.zeros((4, 5), dtype=np.float32)
    source_payload = RuntimeImagePayloadContext(
        np.stack(
            (
                np.full((4, 5), 1, dtype=np.float32),
                np.full((4, 5), 2, dtype=np.float32),
            )
        ),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_image_names=("OrigDNA", "OrigRNA"),
        ),
    ).payload()

    result = CELLPROFILER_MEASUREMENT_MAIN_FLOW.output_image(
        input_image=input_image,
        measurement_images=(
            CellProfilerMeasurementImage(
                source_image_name="OrigDNA_OrigRNA",
                source_aliases=("OrigDNA", "OrigRNA"),
                payload=source_payload,
            ),
        ),
        variable_components=(VariableComponents.CHANNEL,),
        parser=None,
        identity_cache=FunctionOutputIdentityCache(),
    )

    assert result is input_image


def test_measurement_main_flow_keeps_input_for_sources_varying_outside_stack_axes():
    input_image = np.zeros((4, 5), dtype=np.float32)
    dna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 1, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
            },
            source_image_names=("OrigDNA",),
        ),
    ).payload()
    rna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 2, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "2",
            },
            source_image_names=("OrigRNA",),
        ),
    ).payload()

    result = CELLPROFILER_MEASUREMENT_MAIN_FLOW.output_image(
        input_image=input_image,
        measurement_images=(
            CellProfilerMeasurementImage(
                source_image_name="OrigDNA",
                payload=dna_payload,
            ),
            CellProfilerMeasurementImage(
                source_image_name="OrigRNA",
                payload=rna_payload,
            ),
        ),
        variable_components=(VariableComponents.SITE,),
        parser=None,
        identity_cache=FunctionOutputIdentityCache(),
    )

    assert result is input_image


def test_object_label_measurement_images_bundle_source_metadata() -> None:
    dna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 1, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
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
        ),
    ).payload()
    rna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 2, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
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
        ),
    ).payload()

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
    dna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 1, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s001_w1_z001_t001.tif",),
                component_metadata=({"well": "A01", "site": "1", "channel": "1"},),
            ),
            source_image_names=("OrigDNA",),
        ),
    ).payload()
    rna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 2, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s001_w2_z001_t001.tif",),
                component_metadata=({"well": "A01", "site": "1", "channel": "2"},),
            ),
            source_image_names=("OrigRNA",),
        ),
    ).payload()

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


def test_measurement_records_compose_source_metadata_after_object_rows_clear_source() -> (
    None
):
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
    records = (
        CellProfilerMeasurementRecord(
            rows=[{"object_name": "Cells", "center_x": 12.0}],
            object_name=None,
            source_context=CellProfilerMeasurementSourceContext(
                source_image_name="OrigDNA",
                source_metadata=first_metadata,
            ),
        ),
        CellProfilerMeasurementRecord(
            rows=[{"object_name": "Cells", "center_x": 34.0}],
            object_name=None,
            source_context=CellProfilerMeasurementSourceContext(
                source_image_name="OrigDNA",
                source_metadata=second_metadata,
            ),
        ),
    )

    metadata = CellProfilerMeasurementRecord.composed_source_metadata(records)

    assert all(record.source_context.source_image_name is None for record in records)
    assert metadata is not None
    assert tuple(
        plane.source_identity.path
        for plane in metadata.source_image_provenance_planes.planes
    ) == (
        "/input/A01_s001_w1.tif",
        "/input/A01_s002_w1.tif",
    )
    assert metadata.source_image_provenance_planes.component_metadata == (
        {"well": "A01", "site": "1"},
        {"well": "A01", "site": "2"},
    )


def test_measurement_main_flow_keeps_input_for_unaddressable_multiple_sources():
    input_image = np.zeros((4, 5), dtype=np.float32)
    dna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 1, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(source_image_names=("OrigDNA",)),
    ).payload()
    rna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 2, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(source_image_names=("OrigRNA",)),
    ).payload()

    result = CELLPROFILER_MEASUREMENT_MAIN_FLOW.output_image(
        input_image=input_image,
        measurement_images=(
            CellProfilerMeasurementImage(
                source_image_name="OrigDNA",
                payload=dna_payload,
            ),
            CellProfilerMeasurementImage(
                source_image_name="OrigRNA",
                payload=rna_payload,
            ),
        ),
        variable_components=(),
        parser=None,
        identity_cache=FunctionOutputIdentityCache(),
    )

    assert result is input_image


def test_measurement_main_flow_keeps_input_for_duplicate_source_identities():
    input_image = np.zeros((4, 5), dtype=np.float32)
    shared_metadata = {
        "well": "A01",
        "site": "1",
        "channel": "3",
    }
    dna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 1, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata=shared_metadata,
            source_image_names=("OrigDNA",),
        ),
    ).payload()
    rna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 2, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata=shared_metadata,
            source_image_names=("OrigRNA",),
        ),
    ).payload()

    result = CELLPROFILER_MEASUREMENT_MAIN_FLOW.output_image(
        input_image=input_image,
        measurement_images=(
            CellProfilerMeasurementImage(
                source_image_name="OrigDNA",
                payload=dna_payload,
            ),
            CellProfilerMeasurementImage(
                source_image_name="OrigRNA",
                payload=rna_payload,
            ),
        ),
        variable_components=(VariableComponents.CHANNEL,),
        parser=None,
        identity_cache=FunctionOutputIdentityCache(),
    )

    assert result is input_image


def test_image_output_context_preserves_aligned_image_stack_payload():
    dna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 1, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(source_image_names=("OrigDNA",)),
    ).payload()
    rna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 2, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(source_image_names=("OrigRNA",)),
    ).payload()
    aligned = AlignedImageStack(slices=(dna_payload, rna_payload))

    result = FunctionOutputContextStrategy.for_output_plan(None).contextualize(
        dna_payload,
        aligned,
        None,
    )

    assert result is aligned


def test_pattern_group_runtime_unstacks_aligned_image_stack_output():
    dna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 1, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(source_image_names=("OrigDNA",)),
    ).payload()
    rna_payload = RuntimeImagePayloadContext(
        np.full((4, 5), 2, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(source_image_names=("OrigRNA",)),
    ).payload()
    aligned = AlignedImageStack(slices=(dna_payload, rna_payload))
    loaded = PatternGroupData(
        matching_files=["dna.tif", "rna.tif"],
        main_data_stack=np.zeros((2, 4, 5), dtype=np.float32),
        source_slice_shapes=((4, 5), (4, 5)),
    )
    runtime = object.__new__(PatternGroupRuntime)

    assert runtime._validate_and_unstack(aligned, loaded) == [dna_payload, rna_payload]


def test_pattern_group_runtime_flattens_aligned_source_plane_bundles():
    first_site = RuntimeImagePayloadContext(
        np.stack(
            (
                np.full((4, 5), 1, dtype=np.float32),
                np.full((4, 5), 2, dtype=np.float32),
            )
        ),
        mask=None,
        metadata=ImagePayloadMetadata(
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
        ),
    ).payload()
    second_site = RuntimeImagePayloadContext(
        np.stack(
            (
                np.full((4, 5), 3, dtype=np.float32),
                np.full((4, 5), 4, dtype=np.float32),
            )
        ),
        mask=None,
        metadata=ImagePayloadMetadata(
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
        ),
    ).payload()
    aligned = AlignedImageStack(slices=(first_site, second_site))
    loaded = PatternGroupData(
        matching_files=["A01_s001_w2_z001_t001.tif"],
        main_data_stack=np.zeros((4, 5), dtype=np.float32),
        source_slice_shapes=((4, 5),),
    )
    runtime = object.__new__(PatternGroupRuntime)

    output_slices = runtime._validate_and_unstack(aligned, loaded)

    assert [
        float(np.asarray(image_payload_data(payload))[0, 0])
        for payload in output_slices
    ] == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    assert [
        dict(image_payload_metadata(payload).source_component_metadata)
        for payload in output_slices
    ] == [
        {"well": "A01", "site": "1", "channel": "1"},
        {"well": "A01", "site": "1", "channel": "2"},
        {"well": "A01", "site": "2", "channel": "1"},
        {"well": "A01", "site": "2", "channel": "2"},
    ]


def test_side_effect_main_flow_publishes_source_bound_image_request():
    current_image = np.zeros((4, 5), dtype=np.float32)
    requested_payload = np.ones((4, 5), dtype=np.float32)

    result = CELLPROFILER_SIDE_EFFECT_MAIN_FLOW.output_image(
        current_image=current_image,
        image_request=CellProfilerImageRequest(
            source_image_name="OrigActin",
            image_count=1,
            payload=requested_payload,
        ),
        variable_components=(),
        parser=None,
        identity_cache=FunctionOutputIdentityCache(),
    )

    assert result is requested_payload


def test_side_effect_main_flow_keeps_current_image_without_source_identity():
    current_image = np.zeros((4, 5), dtype=np.float32)

    result = CELLPROFILER_SIDE_EFFECT_MAIN_FLOW.output_image(
        current_image=current_image,
        image_request=CellProfilerImageRequest(
            source_image_name=None,
            image_count=1,
            payload=np.ones((4, 5), dtype=np.float32),
        ),
        variable_components=(),
        parser=None,
        identity_cache=FunctionOutputIdentityCache(),
    )

    assert result is current_image


def test_side_effect_main_flow_keeps_current_image_for_non_publishable_request():
    current_image = np.zeros((4, 5), dtype=np.float32)

    result = CELLPROFILER_SIDE_EFFECT_MAIN_FLOW.output_image(
        current_image=current_image,
        image_request=CellProfilerImageRequest(
            source_image_name="Nuclei",
            image_count=1,
            payload=np.ones((4, 5), dtype=np.float32),
            publishes_side_effect_main_flow=False,
        ),
        variable_components=(),
        parser=None,
        identity_cache=FunctionOutputIdentityCache(),
    )

    assert result is current_image


def test_side_effect_main_flow_keeps_current_image_for_unaddressable_source_stack():
    current_image = np.zeros((4, 5), dtype=np.float32)
    requested_payload = RuntimeImagePayloadContext(
        np.stack(
            (
                np.full((4, 5), 1, dtype=np.float32),
                np.full((4, 5), 2, dtype=np.float32),
            )
        ),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w3_z001_t001.tif",
                    "/input/A01_s001_w2_z001_t001.tif",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "3"},
                    {"well": "A01", "site": "1", "channel": "2"},
                ),
            ),
            source_image_names=("mCherry", "GFP"),
        ),
    ).payload()

    result = CELLPROFILER_SIDE_EFFECT_MAIN_FLOW.output_image(
        current_image=current_image,
        image_request=CellProfilerImageRequest(
            source_image_name="mCherry_GFP",
            image_count=2,
            payload=requested_payload,
        ),
        variable_components=(VariableComponents.SITE,),
        parser=None,
        identity_cache=FunctionOutputIdentityCache(),
    )

    assert result is current_image


def test_cellprofiler_main_flow_output_preserves_input_source_planes():
    input_image = RuntimeImagePayloadContext(
        np.zeros((2, 4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w3_z001_t001.TIF",
                    "/input/A01_s002_w3_z001_t001.TIF",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "3"},
                    {"well": "A01", "site": "2", "channel": "3"},
                ),
            )
        ),
    ).payload()
    output_image = np.ones((2, 4, 5), dtype=np.float32)

    result = cellprofiler_main_flow_output(input_image, output_image)

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
    input_image = RuntimeImagePayloadContext(
        np.zeros((2, 4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w3_z001_t001.TIF",
                    "/input/A01_s002_w3_z001_t001.TIF",
                ),
                component_metadata=(
                    {"well": "A01", "site": "1", "channel": "3"},
                    {"well": "A01", "site": "2", "channel": "3"},
                ),
            )
        ),
    ).payload()
    output_image = RuntimeImagePayloadContext(
        np.ones((2, 4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={"well": "A01", "site": "1", "channel": "3"},
        ),
    ).payload()

    result = cellprofiler_main_flow_output(input_image, output_image)

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


def test_recorded_image_main_flow_preserves_recorded_source_identity():
    current_image = RuntimeImagePayloadContext(
        np.zeros((1, 4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={"well": "A01", "site": "1", "channel": "1"},
            source_image_names=("OrigDNA",),
        ),
    ).payload()
    invocation_image = RuntimeImagePayloadContext(
        np.zeros((2, 4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
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
            source_image_names=("OrigDNA",),
        ),
    ).payload()
    recorded_image = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(
            source_component_metadata={"channel": "5"},
            source_image_names=("OrigMito",),
        ),
    ).payload()

    result = cellprofiler_recorded_image_main_flow_output(
        current_image=current_image,
        invocation_image=invocation_image,
        recorded_image=recorded_image,
    )

    metadata = image_payload_metadata(result)
    assert image_payload_data(result).shape[-2:] == (4, 5)
    assert metadata.source_component_metadata == {"channel": "5"}
    assert metadata.source_image_names == ("OrigMito",)


def test_cellprofiler_contract_executor_projects_batch_relationship_auxiliary():
    from openhcs.core.runtime_batch_contracts import (
        RuntimePure2DSliceBatchRequest,
        pure_2d_batch_executor,
    )

    def batch_relationships(request: RuntimePure2DSliceBatchRequest):
        return [
            (
                image,
                ParentChildRelationshipPayload(
                    parent_ids=(slice_index + 1,),
                    child_ids=(slice_index + 10,),
                ),
            )
            for slice_index, image in enumerate(request.slices_2d)
        ]

    @pure_2d_batch_executor(batch_relationships)
    def relate(image: np.ndarray):
        return image, ParentChildRelationshipPayload(parent_ids=(), child_ids=())

    relate.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((2, 4, 5), dtype=np.float32)

    result_image, relationship = CellProfilerFunctionContractExecutor().execute(
        relate,
        stack,
        {},
    )

    assert result_image.shape == stack.shape
    assert relationship == ParentChildRelationshipPayload(
        parent_ids=(1, 2),
        child_ids=(10, 11),
        slice_indices=(0, 1),
        slice_count=2,
    )


def test_cellprofiler_contract_executor_aggregates_volume_label_auxiliary():
    def keep_volume_labels(image: np.ndarray):
        labels = (image > 0).astype(np.int32)
        return image, labels

    keep_volume_labels.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.stack(
        (
            np.ones((3, 4, 5), dtype=np.float32),
            np.full((3, 4, 5), 2.0, dtype=np.float32),
        )
    )

    result_image, result_labels = CellProfilerFunctionContractExecutor().execute(
        keep_volume_labels,
        stack,
        {},
    )

    assert result_image.shape == (6, 4, 5)
    assert isinstance(result_labels, np.ndarray)
    assert result_labels.shape == (6, 4, 5)


def test_cellprofiler_contract_executor_preserves_single_slice_dataclass_auxiliary():
    @dataclass(frozen=True)
    class SliceStats:
        slice_index: int
        threshold_used: float

    def segment(image: np.ndarray, *, slice_index: int = 0, slice_count: int = 1):
        assert slice_count == 1
        return image, SliceStats(slice_index=slice_index, threshold_used=0.25)

    segment.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.ones((4, 5), dtype=np.float32)

    result_image, result_stats = CellProfilerFunctionContractExecutor().execute(
        segment,
        image,
        {},
    )

    np.testing.assert_array_equal(result_image, image)
    assert result_stats == SliceStats(slice_index=0, threshold_used=0.25)


def test_cellprofiler_contract_executor_broadcasts_2d_image_to_stacked_kwargs():
    calls = []

    def increment_labels(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape))
        return labels + 1

    increment_labels.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((4, 5), dtype=np.uint16)
    labels = np.stack(
        (
            np.ones((4, 5), dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )

    result = CellProfilerFunctionContractExecutor().execute(
        increment_labels,
        image,
        {"labels": labels},
    )

    assert calls == [((4, 5), (4, 5)), ((4, 5), (4, 5))]
    assert result.shape == labels.shape
    np.testing.assert_array_equal(result, labels + 1)


def test_mask_objects_uses_object_labels_as_primary_execution_domain() -> None:
    contract = ModuleArtifactContract(
        module_name="MaskObjects",
        items=(
            *ModuleArtifactContract.items_for_partition(
                SourceArtifactInputPartition,
                (
                    ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                    ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
                    ArtifactSpec.input("CarrierImage", ImageArtifactType),
                ),
            ),
            *ModuleArtifactContract.items_for_partition(
                RuntimeArtifactInputPartition,
                (
                    ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                    ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
                    ArtifactSpec.input("CarrierImage", ImageArtifactType),
                ),
            ),
            *ModuleArtifactContract.items_for_partition(
                RecordedArtifactOutputPartition, ()
            ),
            *ModuleArtifactContract.items_for_partition(
                DeclaredArtifactOutputPartition, ()
            ),
        ),
    )

    policy = CellProfilerPrimaryImageInputPolicy.for_module("MaskObjects")
    assert (
        policy.primary_image_inputs(
            contract.module_name,
            mask_objects,
            contract.declared_input_specs(),
            special_input_policy=CellProfilerSpecialInputPolicy.for_module(
                contract.module_name
            ),
        )
        == ()
    )


def test_module_executor_slices_aligned_object_labels_for_pure_2d_module():
    calls = []

    def crop_like(image: np.ndarray, *, cropping_labels: np.ndarray) -> np.ndarray:
        calls.append((image.shape, cropping_labels.shape, int(cropping_labels[0, 0])))
        return image + cropping_labels

    crop_like.__processing_contract__ = ProcessingContract.PURE_2D
    image_stack = np.stack(
        (
            np.full((4, 5), 10, dtype=np.float32),
            np.full((4, 5), 20, dtype=np.float32),
        )
    )
    label_stack = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )
    runtime = _FakeCellProfilerRuntime(
        {"InvBlue": _FakeRuntimeImage(image_stack)},
        {
            "NonOverlappingWorms": ObjectLabelSet(
                name="NonOverlappingWorms",
                labels=label_stack,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Crop",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("InvBlue", ImageArtifactType),
                        ArtifactSpec.input(
                            "NonOverlappingWorms", ObjectLabelsArtifactType
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("InvBlue", ImageArtifactType),
                        ArtifactSpec.input(
                            "NonOverlappingWorms", ObjectLabelsArtifactType
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("CropBlue", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("CropBlue", ImageArtifactType),),
                ),
            ),
        )
    )

    result = executor.run(crop_like, image_stack, cellprofiler_runtime=runtime)

    assert calls == [((4, 5), (4, 5), 1), ((4, 5), (4, 5), 2)]
    assert result.shape == image_stack.shape
    np.testing.assert_array_equal(result, image_stack + label_stack)


def test_module_executor_binds_crop_previous_mask_when_image_output_is_pruned() -> None:
    image = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    previous_mask = np.zeros((4, 5), dtype=bool)
    previous_mask[1:3, 1:4] = True
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigRed": _FakeRuntimeImage(image, source_image_name="OrigRed"),
            "CropBlue__crop_mask": _FakeRuntimeImage(
                previous_mask.astype(np.float32),
                source_image_name="OrigBlue",
            ),
        }
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Crop",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("OrigRed", ImageArtifactType),
                        ArtifactSpec.input("CropBlue__crop_mask", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("CropBlue__crop_mask", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Crop_7_measurements", MeasurementsArtifactType
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output("CropRed", ImageArtifactType),
                        ArtifactSpec.output(
                            "CropRed__crop_mask",
                            ImageArtifactType,
                            sidecar_role=ArtifactSidecarRole.CROP_MASK,
                        ),
                        ArtifactSpec.output(
                            "Crop_7_measurements", MeasurementsArtifactType
                        ),
                    ),
                ),
            ),
        )
    )

    result = executor.run(
        crop,
        image,
        cellprofiler_runtime=runtime,
        crop_shape=CropModule.Shape.CROPPING,
        removal_method=CropModule.RemovalMethod.EDGES,
        dtype_config=DtypeConfig(),
    )

    assert result is image
    assert len(runtime.measurements) == 1
    name, rows, kwargs = runtime.measurements[0]
    assert name == "Crop_7_measurements"
    assert kwargs["source_image_name"] == "CropRed"
    assert len(rows) == 1
    assert rows[0]["area_retained"] == 6


def test_output_value_matching_skips_pruned_intermediate_artifacts() -> None:
    image = np.zeros((3, 4), dtype=np.float32)
    stats = {"objects_pre_filter": 3, "objects_post_filter": 2}
    labels = np.ones((3, 4), dtype=np.int32)
    relationship = ParentChildRelationshipPayload(
        parent_ids=np.asarray([1, 2], dtype=np.int32),
        child_ids=np.asarray([1, 2], dtype=np.int32),
        slice_indices=np.asarray([0, 0], dtype=np.int32),
        slice_count=1,
    )

    values = _resolved_output_values(
        (
            ArtifactSpec.output("FilterStats", MeasurementsArtifactType),
            ArtifactSpec.output("FilteredObjects", ObjectLabelsArtifactType),
        ),
        image,
        (stats, labels, relationship),
        func=filter_objects,
    )

    assert values["FilterStats"] is stats
    assert values["FilteredObjects"] is labels

    relationship_values = _resolved_output_values(
        (
            ArtifactSpec.output("FilterStats", MeasurementsArtifactType),
            ArtifactSpec.output("FilteredRelationships", RelationshipsArtifactType),
        ),
        image,
        (stats, labels, relationship),
        func=filter_objects,
    )

    assert relationship_values["FilterStats"] is stats
    assert relationship_values["FilteredRelationships"] is relationship


def test_output_value_matching_uses_declared_sidecar_specs_for_pruned_outputs() -> None:
    cropped_image = np.ones((3, 4), dtype=np.float32)
    crop_mask = np.ones((10, 12), dtype=bool)
    measurements = [{"area_retained": 12}]

    values = _resolved_output_values(
        (
            ArtifactSpec.output("CropGreen", ImageArtifactType),
            ArtifactSpec.output("Crop_6_measurements", MeasurementsArtifactType),
        ),
        cropped_image,
        (crop_mask, measurements),
        declared_output_specs=(
            ArtifactSpec.output("CropGreen", ImageArtifactType),
            ArtifactSpec.output(
                "CropGreen__crop_mask",
                ImageArtifactType,
                sidecar_role=ArtifactSidecarRole.CROP_MASK,
            ),
            ArtifactSpec.output("Crop_6_measurements", MeasurementsArtifactType),
        ),
    )

    assert values["CropGreen"] is cropped_image
    assert values["Crop_6_measurements"] is measurements


def test_pure_2d_batch_lowers_nominal_runtime_output_bundles() -> None:
    first_output = np.ones((2, 2), dtype=np.float32)
    second_output = np.full((2, 2), 2, dtype=np.float32)
    first_relationship = ParentChildRelationshipPayload(parent_ids=(1,), child_ids=(1,))
    second_relationship = ParentChildRelationshipPayload(
        parent_ids=(2,), child_ids=(2,)
    )
    first_measurements = RelationshipMeasurements(
        slice_index=0,
        parent_object_count=1,
        child_object_count=1,
        children_with_parents_count=1,
        mean_children_per_parent=1.0,
        mean_centroid_distance=0.0,
        mean_minimum_distance=0.0,
    )
    second_measurements = RelationshipMeasurements(
        slice_index=1,
        parent_object_count=1,
        child_object_count=1,
        children_with_parents_count=1,
        mean_children_per_parent=1.0,
        mean_centroid_distance=0.0,
        mean_minimum_distance=0.0,
    )

    batch = Pure2DSliceResultBatch.from_results(
        (
            RelateObjectsResult(first_output, first_relationship, first_measurements),
            RelateObjectsResult(
                second_output, second_relationship, second_measurements
            ),
        )
    )

    assert batch.main_outputs == [first_output, second_output]
    assert batch.auxiliary_groups == (
        [first_relationship, second_relationship],
        [first_measurements, second_measurements],
    )


def test_output_value_resolution_preserves_pruned_context_outputs() -> None:
    cropped_image = np.zeros((3, 4), dtype=np.float32)
    crop_mask = np.ones((5, 6), dtype=bool)
    measurements = {"area_retained": 12}
    retained_specs = (
        ArtifactSpec.output("Crop_7_measurements", MeasurementsArtifactType),
    )
    declared_specs = (
        ArtifactSpec.output("CropRed", ImageArtifactType),
        ArtifactSpec.output(
            "CropRed__crop_mask",
            ImageArtifactType,
            sidecar_role=ArtifactSidecarRole.CROP_MASK,
        ),
        ArtifactSpec.output("Crop_7_measurements", MeasurementsArtifactType),
    )

    recorded_values = _resolved_output_values(
        retained_specs,
        cropped_image,
        (crop_mask, measurements),
        func=crop,
        declared_output_specs=declared_specs,
    )
    context_values = _resolved_output_values(
        declared_specs,
        cropped_image,
        (crop_mask, measurements),
        func=crop,
        declared_output_specs=declared_specs,
    )

    assert recorded_values == {"Crop_7_measurements": measurements}
    assert context_values["CropRed"] is cropped_image
    assert context_values["CropRed__crop_mask"] is crop_mask
    assert context_values["Crop_7_measurements"] is measurements


def test_output_value_resolution_preserves_pruned_object_label_context() -> None:
    image = np.zeros((3, 4), dtype=np.float32)
    stats = {"objects_pre_filter": 3, "objects_post_filter": 1}
    labels = ObjectLabelPayload(
        labels=np.ones((3, 4), dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_count=1,
        ),
    )
    relationship = ParentChildRelationshipPayload(parent_ids=(1,), child_ids=(1,))
    retained_specs = (
        ArtifactSpec.output("FilterStats", MeasurementsArtifactType),
        ArtifactSpec.output(
            "Objects1_ColocalizedObjects_relationships",
            RelationshipsArtifactType,
        ),
    )
    declared_specs = (
        ArtifactSpec.output("FilterStats", MeasurementsArtifactType),
        ArtifactSpec.output("ColocalizedObjects", ObjectLabelsArtifactType),
        ArtifactSpec.output(
            "Objects1_ColocalizedObjects_relationships",
            RelationshipsArtifactType,
        ),
    )

    recorded_values = _resolved_output_values(
        retained_specs,
        image,
        (stats, labels, relationship),
        func=filter_objects,
        declared_output_specs=declared_specs,
    )
    context_values = _resolved_output_values(
        declared_specs,
        image,
        (stats, labels, relationship),
        func=filter_objects,
        declared_output_specs=declared_specs,
    )

    assert "ColocalizedObjects" not in recorded_values
    assert context_values["FilterStats"] is stats
    assert context_values["ColocalizedObjects"] is labels
    assert context_values["Objects1_ColocalizedObjects_relationships"] is relationship


def test_relationship_recorder_resolves_pruned_child_endpoint_from_artifact_name() -> (
    None
):
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(np.zeros((3, 4), dtype=np.float32))},
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Objects1", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Objects1", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Objects1_ColocalizedObjects_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Objects1_ColocalizedObjects_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
            ),
        )
    )
    payload = ParentChildRelationshipPayload(parent_ids=(1,), child_ids=(1,))
    image = np.zeros((3, 4), dtype=np.float32)
    image_request = CellProfilerImageRequest(
        source_image_name=None,
        image_count=1,
        payload=image,
    )
    invocation = CellProfilerInvocationRequest(
        source_image_name=None,
        image_count=1,
        image=image,
        kwargs={},
    )

    identity = lambda image: image
    identity.__processing_contract__ = ProcessingContract.PURE_2D
    runtime_plan = executor.runtime_plan(identity)

    CellProfilerOutputRecorder.record_module_outputs(
        runtime_plan=runtime_plan,
        adapter=runtime,
        main_output=image,
        artifact_values=(payload,),
        invocation=invocation,
        image_request=image_request,
        current_image=image,
    )

    name, kwargs = runtime.relationships[0]
    assert name == "Objects1_ColocalizedObjects_relationships"
    assert kwargs["parent_object_name"] == "Objects1"
    assert kwargs["child_object_name"] == "ColocalizedObjects"


def test_relateobjects_relationship_recorder_uses_declared_endpoint_policy_for_pruned_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((3, 4), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime({"Carrier": _FakeRuntimeImage(image)})
    encoded_parent_spec = ArtifactSpec.input(
        "NameEncodedParent",
        ObjectLabelsArtifactType,
    )
    declared_child_spec = ArtifactSpec.input(
        "DeclaredChild",
        ObjectLabelsArtifactType,
    )
    pruned_saved_child_spec = ArtifactSpec.output(
        "SavedChildren",
        ObjectLabelsArtifactType,
    )
    misleading_relationship_spec = ArtifactSpec.output(
        "NameEncodedParent_NameEncodedChild_relationships",
        RelationshipsArtifactType,
    )
    resolved_relationships = _install_relationship_endpoint_policy_probe(
        monkeypatch,
        parent_spec=declared_child_spec,
        child_spec=pruned_saved_child_spec,
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="RelateObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (encoded_parent_spec, declared_child_spec),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (encoded_parent_spec, declared_child_spec),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition, (misleading_relationship_spec,)
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (pruned_saved_child_spec, misleading_relationship_spec),
                ),
            ),
        )
    )
    payload = ParentChildRelationshipPayload(parent_ids=(1,), child_ids=(1,))
    image_request = CellProfilerImageRequest(
        source_image_name=None,
        image_count=1,
        payload=image,
    )
    invocation = CellProfilerInvocationRequest(
        source_image_name=None,
        image_count=1,
        image=image,
        kwargs={},
    )

    CellProfilerOutputRecorder.record_module_outputs(
        runtime_plan=executor.runtime_plan(relate_objects),
        adapter=runtime,
        main_output=image,
        artifact_values=(payload,),
        invocation=invocation,
        image_request=image_request,
        current_image=image,
    )

    name, kwargs = runtime.relationships[0]
    assert name == "NameEncodedParent_NameEncodedChild_relationships"
    assert kwargs["parent_object_name"] == "DeclaredChild"
    assert kwargs["child_object_name"] == "SavedChildren"
    assert resolved_relationships == [
        ("RelateObjects", "NameEncodedParent_NameEncodedChild_relationships")
    ]


def test_cellprofiler_contract_executor_broadcasts_2d_labels_to_image_stack():
    calls = []

    def add_label_values(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape, int(image[0, 0])))
        return image + labels

    add_label_values.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.stack(
        (
            np.full((4, 5), 10, dtype=np.uint16),
            np.full((4, 5), 20, dtype=np.uint16),
        )
    )
    labels = np.ones((4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(
        add_label_values,
        image,
        {"labels": labels},
    )

    assert calls == [((4, 5), (4, 5), 10), ((4, 5), (4, 5), 20)]
    assert result.shape == image.shape
    np.testing.assert_array_equal(result, image + labels[np.newaxis, ...])


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


def test_identify_secondary_objects_collapses_singleton_image_label_and_mask_planes():
    image = np.zeros((1, 5, 5), dtype=np.float32)
    image[0, 1:4, 1:4] = 1.0
    labels = np.zeros((1, 5, 5), dtype=np.int32)
    labels[0, 2, 2] = 1
    payload = RuntimeImagePayloadContext(
        image, mask=np.ones_like(image, dtype=bool), metadata=ImagePayloadMetadata()
    ).payload()

    _image, _stats, _relationships, secondary = iso.identify_secondary_objects(
        payload,
        labels,
        method=iso.SecondaryMethod.DISTANCE_B,
        distance_to_dilate=1,
        dtype_config=DtypeConfig(),
    )

    secondary_labels = object_label_dense_array(secondary)

    assert secondary_labels.shape == (5, 5)
    assert secondary_labels.max() == 1


def test_secondary_propagation_aligns_label_and_mask_planes_to_image_geometry():
    image = np.zeros((6, 7), dtype=np.float32)
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[2, 2] = 1
    mask = np.ones((5, 5), dtype=bool)

    propagated = PropagationSegmentationStrategy().propagate_labels(
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

    assert propagated.shape == image.shape


def test_parent_child_relationship_aligns_cropped_object_label_payload_to_source_domain():
    parent = ObjectLabelPayload(
        labels=np.array([[1, 0], [0, 0]], dtype=np.int32),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(6, 7),
        ),
    )
    child = np.zeros((6, 7), dtype=np.int32)
    child[2, 3] = 4

    relationship = object_label_parent_child_payload(parent, child)

    assert relationship.parent_ids == (1,)
    assert relationship.child_ids == (4,)


def test_pure_2d_object_label_payload_aggregation_preserves_source_domain():
    payload = ObjectLabelPayload(
        labels=np.array([[1, 0], [0, 0]], dtype=np.int32),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(6, 7),
        ),
    )

    aggregated = CellProfilerPure2DOutputAggregator.aggregate(
        [payload],
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.spatial_origin_yx == (2, 3)
    assert aggregated.source_spatial_shape_yx == (6, 7)


def test_pure_2d_object_label_payload_aggregation_expands_varying_crop_domains():
    first = ObjectLabelPayload(
        labels=np.array([[1]], dtype=np.int32),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(1, 2),
            source_shape_yx=(4, 5),
        ),
        domain=ObjectLabelDomain(
            declared_object_count=3,
        ),
    )
    second = ObjectLabelPayload(
        labels=np.array([[2]], dtype=np.int32),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(4, 5),
        ),
        domain=ObjectLabelDomain(
            declared_object_count=5,
        ),
    )

    aggregated = CellProfilerPure2DOutputAggregator.aggregate(
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


def test_pure_2d_object_label_payload_aggregation_derives_missing_plane_domains():
    first = ObjectLabelPayload(
        labels=np.array([[1, 0], [0, 0]], dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_ids=(1, 2),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    second = ObjectLabelPayload(
        labels=np.array([[3, 0], [0, 0]], dtype=np.int32),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    aggregated = CellProfilerPure2DOutputAggregator.aggregate(
        [first, second],
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.domain.declared_object_id_domains == ((1, 2), (3,))
    assert aggregated.domain.scope is ObjectLabelDomainScope.PLANE


def test_pure_2d_object_label_payload_slice_projects_plane_domain() -> None:
    payload = ObjectLabelPayload(
        labels=np.array(
            [
                [[1, 0], [0, 2]],
                [[1, 0], [3, 4]],
            ],
            dtype=np.int32,
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (1, 2, 3, 4)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    sliced = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimeProjectionAxis(slice_index=1, extent=2),
    )

    assert isinstance(sliced, ObjectLabelPayload)
    assert sliced.labels.shape == (2, 2)
    assert sliced.domain.declared_object_ids == (1, 2, 3, 4)
    assert sliced.domain.declared_object_id_domains == ()
    assert sliced.domain.scope is ObjectLabelDomainScope.PLANE


def test_object_label_endpoint_infers_slice_count_from_payload_stack() -> None:
    payload = ObjectLabelPayload(
        labels=np.array(
            [
                [[1, 0], [0, 2]],
                [[0, 3], [4, 0]],
            ],
            dtype=np.int32,
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (3, 4)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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
        labels=np.array([[1, 0], [0, 2]], dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_ids=(1, 2),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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


def test_object_label_endpoint_repeats_singleton_stack_for_declared_runtime_extent() -> (
    None
):
    payload = ObjectLabelPayload(
        labels=np.array([[[1, 0], [0, 2]]], dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    sliced = RuntimeSliceProjection.object_label_endpoint(
        payload,
        context=RuntimeSliceProjection.context_for_value(
            payload,
            slice_index=1,
            slice_count=3,
            source_description="relationship parent labels",
        ),
    )

    assert isinstance(sliced, ObjectLabelPayload)
    np.testing.assert_array_equal(
        sliced.labels,
        np.array([[1, 0], [0, 2]], dtype=np.int32),
    )
    assert sliced.domain.declared_object_ids == (1, 2)


def test_pure_2d_object_label_set_slice_projects_plane_domain() -> None:
    label_set = ObjectLabelSet(
        name="Cells",
        labels=np.array(
            [
                [[1, 0], [0, 2]],
                [[1, 0], [3, 4]],
            ],
            dtype=np.int32,
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (1, 2, 3, 4)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    sliced = RuntimeSliceProjection.value_for_slice(
        label_set,
        RuntimeProjectionAxis(slice_index=0, extent=2),
    )

    assert isinstance(sliced, ObjectLabelSet)
    assert sliced.labels.shape == (2, 2)
    assert sliced.domain.declared_object_ids == (1, 2)
    assert sliced.domain.declared_object_id_domains == ()
    assert sliced.domain.scope is ObjectLabelDomainScope.PLANE


def test_pure_2d_object_label_payload_slice_projects_grouped_plane_domains() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((2, 3, 4, 5), dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,), (3,), (4,), (5,), (6,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    sliced = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimeProjectionAxis(slice_index=1, extent=2),
    )

    assert isinstance(sliced, ObjectLabelPayload)
    assert sliced.labels.shape == (3, 4, 5)
    assert sliced.domain.declared_object_ids == ()
    assert sliced.domain.declared_object_id_domains == ((4,), (5,), (6,))
    assert sliced.domain.scope is ObjectLabelDomainScope.PLANE


def test_runtime_slice_count_allows_grouped_object_label_planes() -> None:
    parent = ObjectLabelPayload(
        labels=np.zeros((2, 3, 4, 5), dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,), (3,), (4,), (5,), (6,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    child = ObjectLabelPayload(
        labels=np.zeros((6, 4, 5), dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,), (3,), (4,), (5,), (6,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    assert RuntimeSliceProjection.slice_count_from_values((parent, child)) == 2


def test_runtime_slice_count_treats_sequence_kwargs_as_operands() -> None:
    first = ObjectLabelPayload(
        labels=np.zeros((2, 4, 5), dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    second = ObjectLabelPayload(
        labels=np.zeros((2, 4, 5), dtype=np.int32),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((3,), (4,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    assert RuntimeSliceProjection.slice_count_from_values(((first, second),)) == 4
    assert (
        RuntimeSliceProjection.slice_count_from_kwargs(
            {"object_labels": (first, second)},
            sequence_kwargs=frozenset({"object_labels"}),
        )
        == 2
    )
    assert (
        Pure2DSliceCountPolicy.slice_count_from_kwargs(
            {"object_labels": (first, second)},
            runtime_slice_sequence_parameter_names=frozenset({"object_labels"}),
        )
        == 2
    )


def test_runtime_slice_projection_offsets_repeated_scalar_measurement_tables() -> None:
    first = MeasurementTable(
        name="MeasureObjectIntensity_7_measurements",
        rows=[{"slice_index": 0, "object_label": 1, "std_intensity": 0.1}],
        object_name="Tile_of_grid",
        object_id_field="object_label",
        source_image_name="DF_image",
    )
    second = MeasurementTable(
        name="MeasureObjectIntensity_7_measurements",
        rows=[{"slice_index": 0, "object_label": 1, "std_intensity": 0.2}],
        object_name="Tile_of_grid",
        object_id_field="object_label",
        source_image_name="DF_image",
    )

    tables = (first, second)

    assert RuntimeSliceProjection.slice_count_from_values((tables,)) == 2
    sliced = RuntimeSliceProjection.kwargs_for_slice(
        {"measurement_tables": tables},
        RuntimeProjectionAxis(slice_index=1, extent=2),
    )["measurement_tables"]
    assert tuple(len(table.rows) for table in sliced) == (0, 1)
    assert list(sliced[1].rows)[0]["std_intensity"] == 0.2


def test_runtime_slice_projection_preserves_image_number_measurement_tables() -> None:
    first = MeasurementTable(
        name="MeasureImageAreaOccupied_21_measurements",
        rows=[
            {"image_number": 1, "area_occupied": 17809.0},
            {"image_number": 2, "area_occupied": 10723.0},
        ],
        source_image_name="ColocalizedRegion",
    )
    second = MeasurementTable(
        name="MeasureImageAreaOccupied_21_measurements",
        rows=[
            {"image_number": 1, "area_occupied": 17809.0},
            {"image_number": 2, "area_occupied": 10723.0},
        ],
        source_image_name="ColocalizedRegion",
    )

    tables = (first, second)
    aligned = (
        RuntimeSliceProjection.measurement_tables_with_repeated_scalar_slice_offsets(
            tables
        )
    )

    assert aligned == tables
    assert RuntimeSliceProjection.slice_count_from_values((aligned,)) is None


def test_identify_tertiary_batch_aligns_cropped_primary_labels_to_secondary_domain():
    from openhcs.core.runtime_batch_contracts import RuntimePure2DSliceBatchRequest

    primary = ObjectLabelPayload(
        labels=np.array([[1, 0], [0, 0]], dtype=np.int32),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(6, 7),
        ),
    )
    secondary = np.zeros((6, 7), dtype=np.int32)
    secondary[2, 3] = 5
    secondary[2, 4] = 5

    results = ito._identify_tertiary_objects_batch(
        RuntimePure2DSliceBatchRequest(
            func=ito.identify_tertiary_objects,
            slices_2d=(np.zeros((6, 7), dtype=np.float32),),
            kwargs={
                "primary_labels": primary,
                "secondary_labels": secondary,
                "shrink_primary": False,
            },
            execute_slice=(
                lambda func, image, kwargs, _slice_index, _slice_count: func(
                    image,
                    **kwargs,
                )
            ),
        )
    )

    tertiary = results[0][-1]
    assert tertiary.shape == secondary.shape
    assert tertiary[2, 3] == 0
    assert tertiary[2, 4] == 5


def test_identify_tertiary_batch_drops_fully_subtracted_secondary_objects():
    from openhcs.core.runtime_batch_contracts import RuntimePure2DSliceBatchRequest

    primary = np.zeros((3, 3), dtype=np.int32)
    primary[1, 1] = 1
    secondary = np.zeros((3, 3), dtype=np.int32)
    secondary[1, 1] = 5

    results = ito._identify_tertiary_objects_batch(
        RuntimePure2DSliceBatchRequest(
            func=ito.identify_tertiary_objects,
            slices_2d=(np.zeros((3, 3), dtype=np.float32),),
            kwargs={
                "primary_labels": primary,
                "secondary_labels": secondary,
                "shrink_primary": False,
            },
            execute_slice=(
                lambda func, image, kwargs, _slice_index, _slice_count: func(
                    image,
                    **kwargs,
                )
            ),
        )
    )

    tertiary = results[0][-1]
    assert np.count_nonzero(tertiary == 5) == 0


def test_identify_tertiary_batch_projects_derived_label_plane_domains():
    from openhcs.core.runtime_batch_contracts import RuntimePure2DSliceBatchRequest

    primary = ObjectLabelPayload(
        labels=np.zeros((2, 3, 3), dtype=np.int32),
        source_image_names=("rawDNA", "rawActin"),
    )
    secondary = ObjectLabelPayload(
        labels=np.asarray(
            [
                [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 5, 0], [0, 0, 0]],
            ],
            dtype=np.int32,
        ),
        source_image_names=("rawDNA", "rawActin"),
    )

    results = ito._identify_tertiary_objects_batch(
        RuntimePure2DSliceBatchRequest(
            func=ito.identify_tertiary_objects,
            slices_2d=(
                np.zeros((3, 3), dtype=np.float32),
                np.zeros((3, 3), dtype=np.float32),
            ),
            kwargs={
                "primary_labels": primary,
                "secondary_labels": secondary,
                "shrink_primary": False,
            },
            execute_slice=(
                lambda func, image, kwargs, _slice_index, _slice_count: func(
                    image,
                    **kwargs,
                )
            ),
        )
    )

    first = results[0][-1]
    second = results[1][-1]
    assert isinstance(first, ObjectLabelPayload)
    assert isinstance(second, ObjectLabelPayload)
    assert first.domain.declared_object_ids == (2,)
    assert second.domain.declared_object_ids == (5,)
    assert first.source_image_names == ("rawDNA",)
    assert second.source_image_names == ("rawActin",)


def test_tertiary_projected_plane_keeps_dense_local_label_domain():
    source = ObjectLabelSet(
        name="Cells",
        labels=SparseIJVLabelRows.from_dense_stack(
            np.asarray(
                (
                    ((0, 2), (0, 0)),
                    ((0, 0), (5, 0)),
                ),
                dtype=np.int32,
            )
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(10, 11),
        ),
        source_image_names=("rawDNA", "rawActin"),
    )
    labels = np.asarray(((0, 0), (5, 0)), dtype=np.int32)

    output = ito.TertiaryObjectLabelOutput(
        source,
        labels,
        source_plane_index=1,
    ).value()

    assert isinstance(output, ObjectLabelSet)
    assert output.representation is ObjectLabelRepresentation.DENSE_LABELS
    assert output.spatial_origin_yx == (2, 3)
    assert output.source_spatial_shape_yx == (10, 11)
    assert output.source_image_names == ("rawActin",)
    np.testing.assert_array_equal(object_label_dense_array(output), labels)


def test_identify_tertiary_single_slice_aligns_payload_domains_before_dense_extraction():
    primary = ObjectLabelPayload(
        labels=np.array([[1, 0], [0, 0]], dtype=np.int32),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(6, 7),
        ),
    )
    secondary = np.zeros((6, 7), dtype=np.int32)
    secondary[2, 3] = 5
    secondary[2, 4] = 5

    _, _, _, _, tertiary = ito.identify_tertiary_objects.__wrapped__(
        np.zeros((6, 7), dtype=np.float32),
        primary_labels=primary,
        secondary_labels=secondary,
        shrink_primary=False,
    )

    assert tertiary.shape == secondary.shape
    assert tertiary[2, 3] == 0
    assert tertiary[2, 4] == 5


def test_identify_tertiary_single_slice_restores_secondary_crop_domain():
    primary = ObjectLabelSet(
        name="Nuclei",
        labels=np.asarray(((1, 0), (0, 0)), dtype=np.int32),
        representation=ObjectLabelRepresentation.DENSE_LABELS,
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(6, 7),
        ),
    )
    secondary = ObjectLabelSet(
        name="Cells",
        labels=np.asarray(((5, 5), (0, 0)), dtype=np.int32),
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
    assert tertiary.shape == secondary.shape
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


def test_cellprofiler_module_executor_normalizes_integer_image_inputs() -> None:
    source_image = "DNA"
    raw = np.full((4, 5), 255, dtype=np.uint8)
    runtime = _FakeCellProfilerRuntime(
        {source_image: _FakeRuntimeImage(raw, source_image_name=source_image)}
    )
    seen: list[np.ndarray] = []

    def capture(image: np.ndarray) -> np.ndarray:
        seen.append(image)
        return image

    capture.__processing_contract__ = ProcessingContract.PURE_2D
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Opening",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input(source_image, ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Normalized", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Normalized", ImageArtifactType),),
                ),
            ),
        )
    )

    result = executor.run(capture, raw, cellprofiler_runtime=runtime)

    assert seen[0].dtype == np.float32
    np.testing.assert_array_equal(seen[0], np.ones_like(raw, dtype=np.float32))
    assert result.dtype == np.float32
    np.testing.assert_array_equal(
        runtime.images["Normalized"].data,
        np.ones_like(raw, dtype=np.float32),
    )


def test_cellprofiler_module_executor_uses_payload_intensity_scale() -> None:
    source_image = "DNA"
    raw = np.array([[0, 4095]], dtype=np.uint16)
    payload = RuntimeImagePayloadContext(
        raw,
        metadata=ImagePayloadMetadata(intensity_scale=4095.0, source_dtype="uint16"),
        mask=None,
    ).payload()
    runtime = _FakeCellProfilerRuntime(
        {source_image: _FakeRuntimeImage(payload, source_image_name=source_image)}
    )
    seen: list[object] = []

    def capture(image: object) -> object:
        seen.append(image)
        return image

    capture.__processing_contract__ = ProcessingContract.PURE_2D
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Opening",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input(source_image, ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Normalized", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Normalized", ImageArtifactType),),
                ),
            ),
        )
    )

    result = executor.run(capture, raw, cellprofiler_runtime=runtime)

    np.testing.assert_allclose(image_payload_data(seen[0]), [[0.0, 1.0]])
    assert image_payload_metadata(seen[0]).intensity_scale == 4095.0
    np.testing.assert_allclose(
        image_payload_data(runtime.images["Normalized"].data),
        [[0.0, 1.0]],
    )
    np.testing.assert_allclose(image_payload_data(result), [[0.0, 1.0]])


def test_cellprofiler_contract_executor_slices_plane_sequence_kwargs():
    calls = []

    def keep_labels(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape, int(labels[0, 0])))
        return labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((4, 5), dtype=np.uint16)
    labels = (
        np.full((4, 5), 1, dtype=np.int32),
        np.full((4, 5), 2, dtype=np.int32),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        image,
        {"labels": labels},
    )

    assert calls == [((4, 5), (4, 5), 1), ((4, 5), (4, 5), 2)]
    assert result.shape == (2, 4, 5)
    np.testing.assert_array_equal(result, np.asarray(labels))


def test_cellprofiler_contract_executor_slices_array_convertible_kwargs():
    class ArrayConvertible:
        def __init__(self, data: np.ndarray) -> None:
            self.shape = data.shape
            self._data = data

        def __array__(self) -> np.ndarray:
            return self._data

    calls = []

    def keep_labels(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape, int(labels[0, 0])))
        return labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((4, 5), dtype=np.uint16)
    labels = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )

    result = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        image,
        {"labels": ArrayConvertible(labels)},
    )

    assert calls == [((4, 5), (4, 5), 1), ((4, 5), (4, 5), 2)]
    assert result.shape == labels.shape
    np.testing.assert_array_equal(result, labels)


def test_cellprofiler_contract_executor_slices_nested_sequence_kwargs():
    calls = []

    def keep_labels(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape, int(labels[0, 0])))
        return labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((2, 2), dtype=np.uint16)
    labels = [
        [[1, 1], [1, 1]],
        [[2, 2], [2, 2]],
    ]

    result = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        image,
        {"labels": labels},
    )

    assert calls == [((2, 2), (2, 2), 1), ((2, 2), (2, 2), 2)]
    np.testing.assert_array_equal(result, np.asarray(labels))


def test_cellprofiler_contract_executor_preserves_multi_image_stack_payload():
    calls = []

    def keep_stack(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image

    keep_stack.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((3, 4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(
        keep_stack,
        stack,
        {},
        force_full_stack=True,
    )

    assert calls == [(3, 4, 5)]
    assert result.shape == stack.shape


def test_correct_illumination_all_scope_module_executor_uses_full_stack():
    calls = []

    def calculate_illumination(image: np.ndarray, *, calculation_scope: str):
        calls.append((image.shape, calculation_scope))
        return image.mean(axis=0).astype(np.float32), []

    calculate_illumination.__processing_contract__ = ProcessingContract.PURE_2D
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="CorrectIlluminationCalculate",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("OrigGreen", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("IllumGreen", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("IllumGreen", ImageArtifactType),),
                ),
            ),
        )
    )
    stack = np.stack(
        (
            np.full((4, 5), 0.25, dtype=np.float32),
            np.full((4, 5), 0.75, dtype=np.float32),
        )
    )
    runtime = _FakeCellProfilerRuntime({"OrigGreen": _FakeRuntimeImage(stack)})

    result = executor.run(
        calculate_illumination,
        stack,
        cellprofiler_runtime=runtime,
        calculation_scope="all_first_cycle",
    )

    assert calls == [((2, 4, 5), "all_first_cycle")]
    np.testing.assert_array_equal(result, stack)
    np.testing.assert_array_equal(
        runtime.images["IllumGreen"].data,
        np.full((4, 5), 0.5, dtype=np.float32),
    )


def test_object_only_reference_image_reduces_color_stacks_to_one_intensity_plane():
    color_stack = np.zeros((2, 4, 5, 3), dtype=np.float32)
    color_stack[0, :, :, 1] = 7

    reference = OBJECT_ONLY_REFERENCE_IMAGE.reference_image(color_stack)

    assert reference.shape == (4, 5)
    np.testing.assert_array_equal(reference, color_stack[0, :, :, 0])


def test_compose_image_payload_aligns_multislice_inputs_with_broadcast():
    raw_stack = np.stack(
        (
            np.full((4, 5), 11, dtype=np.float32),
            np.full((4, 5), 22, dtype=np.float32),
        )
    )
    illumination = np.full((4, 5), 3, dtype=np.float32)

    composition = compose_aligned_image_payload(
        "CorrectIlluminationApply",
        (raw_stack, illumination),
    )

    assert (
        composition.execution_mode
        is ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
    )
    assert isinstance(composition.payload, AlignedImageStack)
    assert len(composition.payload.slices) == 2
    for slice_index, composed_slice in enumerate(composition.payload.slices):
        assert composed_slice.shape == (2, 4, 5)
        np.testing.assert_array_equal(composed_slice[0], raw_stack[slice_index])
        np.testing.assert_array_equal(composed_slice[1], illumination)


def test_compose_image_payload_collapses_pairwise_slice_grids_before_alignment():
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

    assert (
        composition.execution_mode
        is ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
    )
    assert isinstance(composition.payload, AlignedImageStack)
    assert len(composition.payload.slices) == 2
    np.testing.assert_array_equal(
        composition.payload.slices[0],
        np.stack(
            (
                np.full((4, 5), 11, dtype=np.float32),
                np.full((4, 5), 3, dtype=np.float32),
            )
        ),
    )
    np.testing.assert_array_equal(
        composition.payload.slices[1],
        np.stack(
            (
                np.full((4, 5), 22, dtype=np.float32),
                np.full((4, 5), 7, dtype=np.float32),
            )
        ),
    )


def test_compose_image_bundle_promotes_grayscale_into_color_bundle():
    color = np.zeros((4, 5, 3), dtype=np.float32)
    color[:, :, 0] = 1
    grayscale = np.full((4, 5), 7, dtype=np.float32)

    bundle = ImagePayloadBundleContext.from_payloads((color, grayscale)).compose()

    assert bundle.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(bundle[0], color)
    np.testing.assert_array_equal(bundle[1, :, :, 0], grayscale)
    np.testing.assert_array_equal(bundle[1, :, :, 1], grayscale)
    np.testing.assert_array_equal(bundle[1, :, :, 2], grayscale)


def test_compose_image_bundle_collapses_singleton_grayscale_plane_stacks():
    singleton = np.full((1, 4, 5), 3, dtype=np.float32)
    plane = np.full((4, 5), 7, dtype=np.float32)

    bundle = ImagePayloadBundleContext.from_payloads((singleton, plane)).compose()

    assert bundle.shape == (2, 4, 5)
    np.testing.assert_array_equal(bundle[0], singleton[0])
    np.testing.assert_array_equal(bundle[1], plane)


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
            MaskedImagePayload(data=image_a, mask=mask_a),
            MaskedImagePayload(data=image_b, mask=mask_b),
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

    assert output.shape == (1, 3, 8, 3)
    np.testing.assert_array_equal(output[0, :, :4, 0], np.ones((3, 4)))
    np.testing.assert_array_equal(output[0, :, 4:, 1], np.full((3, 4), 2))


def test_cellprofiler_contract_executor_applies_aligned_multi_image_stack():
    calls = []

    def subtract_illumination(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return (image[0] - image[1])[np.newaxis, ...]

    subtract_illumination.__processing_contract__ = ProcessingContract.PURE_2D
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
        subtract_illumination,
        aligned_stack,
        {},
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
    )

    assert calls == [(2, 4, 5), (2, 4, 5)]
    assert result.shape == (2, 4, 5)
    np.testing.assert_array_equal(result[0], np.full((4, 5), 8, dtype=np.float32))
    np.testing.assert_array_equal(result[1], np.full((4, 5), 19, dtype=np.float32))


def test_aligned_multi_image_stack_rejects_volumetric_contract() -> None:
    def keep_volume(image: np.ndarray) -> np.ndarray:
        return image

    keep_volume.__processing_contract__ = ProcessingContract.PURE_3D
    aligned_stack = AlignedImageStack(
        slices=(
            np.zeros((2, 4, 5), dtype=np.float32),
            np.zeros((2, 4, 5), dtype=np.float32),
        )
    )

    with pytest.raises(ValueError, match="ProcessingContract.PURE_3D"):
        CellProfilerFunctionContractExecutor().execute(
            keep_volume,
            aligned_stack,
            {},
            execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        )


def test_aligned_multi_image_stack_slices_runtime_array_kwargs() -> None:
    calls = []

    def keep_labels(
        image: np.ndarray, *, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append((image.shape, labels.shape))
        return image[0], labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
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
                    np.full((4, 5), 7, dtype=np.float32),
                )
            ),
        )
    )
    labels = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )

    result_image, result_labels = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        aligned_stack,
        {"labels": labels},
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
    )

    assert calls == [((2, 4, 5), (4, 5)), ((2, 4, 5), (4, 5))]
    assert result_image.shape == (2, 4, 5)
    assert result_labels.shape == labels.shape
    np.testing.assert_array_equal(result_labels, labels)


def test_module_executor_runs_image_measurements_per_declared_image_without_replacing_main_flow() -> (
    None
):
    calls = []

    def measure_image(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        calls.append(float(image[0, 0]))
        return image, {"mean": float(np.mean(image))}

    measure_image.__processing_contract__ = ProcessingContract.PURE_2D
    current_image = np.zeros((4, 5), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigBlue": _FakeRuntimeImage(
                RuntimeImagePayloadContext(
                    np.ones((4, 5), dtype=np.float32),
                    mask=None,
                    metadata=ImagePayloadMetadata(
                        source_component_metadata={"channel": "1"},
                        source_image_names=("OrigBlue",),
                    ),
                ).payload(),
                source_image_name="OrigBlue",
            ),
            "OrigGreen": _FakeRuntimeImage(
                RuntimeImagePayloadContext(
                    np.full((4, 5), 2, dtype=np.float32),
                    mask=None,
                    metadata=ImagePayloadMetadata(
                        source_component_metadata={"channel": "2"},
                        source_image_names=("OrigGreen",),
                    ),
                ).payload(),
                source_image_name="OrigGreen",
            ),
        },
        variable_components=(VariableComponents.CHANNEL,),
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureImageQuality",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("OrigBlue", ImageArtifactType),
                        ArtifactSpec.input("OrigGreen", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("ImageQuality", MeasurementsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("ImageQuality", MeasurementsArtifactType),),
                ),
            ),
        )
    )

    result = executor.run(
        measure_image,
        current_image,
        cellprofiler_runtime=runtime,
    )

    assert isinstance(result, NoMainFlowOutput)
    assert calls == [1.0, 2.0]
    assert _recorded_measurements_for_assertion(runtime.measurements) == [
        (
            "ImageQuality",
            [
                {"mean": 1.0, "source_image_name": "OrigBlue", "slice_index": 0},
                {"mean": 2.0, "source_image_name": "OrigGreen", "slice_index": 0},
            ],
            {
                "source_image_name": None,
                "fields": ("slice_index", "mean", "source_image_name"),
            },
        )
    ]


def test_module_executor_runs_object_distribution_measurements_per_declared_image_without_replacing_main_flow() -> (
    None
):
    calls = []

    def measure_distribution(
        image: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, float | int]]]:
        calls.append((float(image[0, 0]), labels.copy()))
        return image, [
            {
                "object_label": 1,
                "mean": float(np.mean(image[labels > 0])),
            }
        ]

    measure_distribution.__processing_contract__ = ProcessingContract.PURE_2D
    labels = np.zeros((4, 5), dtype=np.int32)
    labels[1:3, 1:3] = 1
    current_image = np.zeros((4, 5), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigBlue": _FakeRuntimeImage(
                RuntimeImagePayloadContext(
                    np.ones((4, 5), dtype=np.float32),
                    mask=None,
                    metadata=ImagePayloadMetadata(
                        source_component_metadata={"channel": "1"},
                        source_image_names=("OrigBlue",),
                    ),
                ).payload()
            ),
            "OrigGreen": _FakeRuntimeImage(
                RuntimeImagePayloadContext(
                    np.full((4, 5), 2, dtype=np.float32),
                    mask=None,
                    metadata=ImagePayloadMetadata(
                        source_component_metadata={"channel": "2"},
                        source_image_names=("OrigGreen",),
                    ),
                ).payload()
            ),
        },
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=labels,
            )
        },
        variable_components=(VariableComponents.CHANNEL,),
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectIntensityDistribution",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("OrigBlue", ImageArtifactType),
                        ArtifactSpec.input("OrigGreen", ImageArtifactType),
                        ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("MID", MeasurementsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("MID", MeasurementsArtifactType),),
                ),
            ),
        )
    )

    result = executor.run(
        measure_distribution,
        current_image,
        cellprofiler_runtime=runtime,
    )

    assert isinstance(result, NoMainFlowOutput)
    assert [call[0] for call in calls] == [1.0, 2.0]
    for _image_value, bound_labels in calls:
        np.testing.assert_array_equal(bound_labels, labels)
    assert _recorded_measurements_for_assertion(runtime.measurements) == [
        (
            "MID",
            [
                {
                    "object_label": 1,
                    "mean": 1.0,
                    "object_name": "Cells",
                    "source_image_name": "OrigBlue",
                    "slice_index": 0,
                },
                {
                    "object_label": 1,
                    "mean": 2.0,
                    "object_name": "Cells",
                    "source_image_name": "OrigGreen",
                    "slice_index": 0,
                },
            ],
            {
                "object_name": "Cells",
                "source_image_name": None,
                "source_image_provenance_planes": SourceImageProvenancePlanes.from_components(
                    component_metadata=({"channel": "1"}, {"channel": "2"}),
                ),
                "fields": (
                    "slice_index",
                    "object_label",
                    "mean",
                    "object_name",
                    "source_image_name",
                ),
            },
        )
    ]


def test_module_executor_preserves_composed_image_measurements() -> None:
    calls = []

    def measure_pair(
        image: np.ndarray,
        channel_1: int = 0,
        channel_2: int = 1,
    ) -> tuple[np.ndarray, dict[str, float]]:
        calls.append(image.shape)
        return image[channel_1], {
            "correlation": float(np.mean(image[channel_2] - image[channel_1])),
            "delta": float(np.mean(image[channel_2] - image[channel_1])),
        }

    attach_callable_contract_metadata(
        measure_pair,
        declared_processing_contract="flexible",
    )
    composed_image_payload(measure_pair)

    current_image = np.zeros((4, 5), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigBlue": _FakeRuntimeImage(
                RuntimeImagePayloadContext(
                    np.ones((4, 5), dtype=np.float32),
                    mask=None,
                    metadata=ImagePayloadMetadata(
                        source_component_metadata={"channel": "1"},
                        source_image_names=("OrigBlue",),
                    ),
                ).payload(),
                source_image_name="OrigBlue",
            ),
            "OrigGreen": _FakeRuntimeImage(
                RuntimeImagePayloadContext(
                    np.full((4, 5), 3, dtype=np.float32),
                    mask=None,
                    metadata=ImagePayloadMetadata(
                        source_component_metadata={"channel": "2"},
                        source_image_names=("OrigGreen",),
                    ),
                ).payload(),
                source_image_name="OrigGreen",
            ),
        },
        variable_components=(VariableComponents.CHANNEL,),
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureColocalization",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("OrigBlue", ImageArtifactType),
                        ArtifactSpec.input("OrigGreen", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Colocalization", MeasurementsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Colocalization", MeasurementsArtifactType),),
                ),
            ),
        )
    )

    result = executor.run(
        measure_pair,
        current_image,
        cellprofiler_runtime=runtime,
    )

    np.testing.assert_array_equal(
        image_payload_data(result),
        np.stack(
            (
                np.ones((4, 5), dtype=np.float32),
                np.full((4, 5), 3, dtype=np.float32),
            )
        ),
    )
    assert calls == [(2, 4, 5)]
    assert _recorded_measurements_for_assertion(runtime.measurements) == [
        (
            "Colocalization",
            [
                {
                    "correlation": 2.0,
                    "delta": 2.0,
                    "slice_index": 0,
                }
            ],
            {
                "object_name": None,
                "source_image_name": "OrigBlue__OrigGreen",
                "fields": (
                    "slice_index",
                    "correlation",
                    "delta",
                ),
                "source_image_provenance_planes": (
                    SourceImageProvenancePlanes.from_components(
                        component_metadata=(
                            {"channel": "1"},
                            {"channel": "2"},
                        )
                    )
                ),
            },
        )
    ]


def test_colocalization_object_row_policy_projects_source_pair_features() -> None:
    policy = CellProfilerObjectMeasurementRowPolicy.for_module("MeasureColocalization")
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
    projected = policy.project_rows(
        [
            {
                "slice_index": 0,
                "object_label": 1,
                "correlation": 0.5,
                "slope": 0.25,
                "manders_m1": 0.7,
                "manders_m2": 0.8,
                "costes_threshold_1": 42.0,
            }
        ],
        invocations[0],
    )

    assert projected == [
        {
            "slice_index": 0,
            "object_label": 1,
            "Correlation_Correlation_DNA_ER": 0.5,
            "Correlation_Slope_DNA_ER": 0.25,
            "Correlation_Manders_DNA_ER": 0.7,
            "Correlation_Manders_ER_DNA": 0.8,
        }
    ]
    assert policy.table_source_image_name((measurement_image,), "DNA__ER__RNA") is None


def test_colocalization_record_builder_preserves_source_pair_table_identity() -> None:
    def measure_colocalization(
        image: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        return image, {"correlation": 0.5}

    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureColocalization",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("DNA", ImageArtifactType),
                        ArtifactSpec.input("ER", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("DNA", ImageArtifactType),
                        ArtifactSpec.input("ER", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Coloc", MeasurementsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Coloc", MeasurementsArtifactType),),
                ),
            ),
        )
    )

    record = measurement_record_for_module(
        _cellprofiler_output_record_request(
            executor=executor,
            adapter=None,
            spec=ArtifactSpec.output("Coloc", MeasurementsArtifactType),
            output_value={"slice_index": 0, "correlation": 0.5, "manders_m1": 0.7},
            output_values={"Coloc": {"correlation": 0.5}},
            source_image_name="DNA__ER",
            func=measure_colocalization,
            call_kwargs={},
            source_aliases=("DNA", "ER"),
        )
    )

    assert record.rows == [
        {
            "slice_index": 0,
            "correlation": 0.5,
            "manders_m1": 0.7,
        }
    ]
    assert record.source_context.source_image_name == "DNA__ER"


def test_measure_object_neighbors_records_object_topology_without_image_source() -> (
    None
):
    def measure_neighbors(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        return image, {"number_of_neighbors": 1.0}

    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectNeighbors",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Neighbors", MeasurementsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Neighbors", MeasurementsArtifactType),),
                ),
            ),
        )
    )

    record = measurement_record_for_module(
        _cellprofiler_output_record_request(
            executor=executor,
            adapter=None,
            spec=ArtifactSpec.output("Neighbors", MeasurementsArtifactType),
            output_value={"number_of_neighbors": 1.0},
            output_values={"Neighbors": {"number_of_neighbors": 1.0}},
            source_image_name="OrigBlue",
            func=measure_neighbors,
            call_kwargs={},
        )
    )

    assert record.object_name == "Nuclei"
    assert record.source_context.source_image_name is None


def test_track_objects_record_builder_uses_nominal_image_table_ownership() -> None:
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="TrackObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Embryos", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Embryos", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Tracking", MeasurementsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Tracking", MeasurementsArtifactType),),
                ),
            ),
        )
    )

    record = measurement_record_for_module(
        _cellprofiler_output_record_request(
            executor=executor,
            adapter=None,
            spec=ArtifactSpec.output("Tracking", MeasurementsArtifactType),
            output_value=[
                {
                    "image_number": 1,
                    "object_label": 1,
                    "feature_name": "TrackObjects_Label_50",
                    "measurement_value": 1,
                },
                {
                    "image_number": 1,
                    "feature_name": "TrackObjects_NewObjectCount_Embryos_50",
                    "measurement_value": 1,
                },
            ],
            output_values={},
            source_image_name="OrigColor",
            func=_synthetic_object_measurement_function,
            call_kwargs={},
        )
    )

    object_row, image_row = record.rows
    assert object_row["object_name"] == "Embryos"
    assert "source_image_name" not in object_row
    assert image_row["source_image_name"] == "image"
    assert "object_name" not in image_row


def test_object_label_output_recorder_uses_output_label_domain() -> None:
    input_labels = np.zeros((5, 5), dtype=np.int32)
    input_labels[1, 1] = 1
    input_labels[2, 2] = 4
    input_payload = ObjectLabelPayload(
        labels=input_labels,
        domain=ObjectLabelDomain(
            declared_object_count=9,
        ),
    )
    output_labels = input_labels.copy()
    output_payload = ObjectLabelPayload(
        labels=output_labels,
        domain=ObjectLabelDomain(
            declared_object_ids=tuple(range(1, 5)),
        ),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={
            "InputObjects": ObjectLabelSet(
                name="InputObjects",
                labels=input_payload,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="ExpandOrShrinkObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("InputObjects", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("InputObjects", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("ExpandedObjects", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("ExpandedObjects", ObjectLabelsArtifactType),),
                ),
            ),
        )
    )

    CellProfilerOutputRecorder.for_artifact_type(ObjectLabelsArtifactType).record(
        _cellprofiler_output_record_request(
            executor=executor,
            adapter=runtime,
            spec=ArtifactSpec.output("ExpandedObjects", ObjectLabelsArtifactType),
            output_value=output_payload,
            output_values={"ExpandedObjects": output_payload},
            source_image_name=None,
            func=lambda image: image,
            call_kwargs={},
        )
    )

    _name, recorded_payload, _kwargs = runtime.objects[0]
    assert isinstance(recorded_payload, ObjectLabelPayload)
    assert recorded_payload.domain.declared_object_count is None
    assert recorded_payload.domain.declared_object_ids == tuple(range(1, 5))
    np.testing.assert_array_equal(recorded_payload.labels, output_labels)


def test_expand_or_shrink_executor_declares_output_label_extent() -> None:
    input_labels = np.zeros((7, 7), dtype=np.int32)
    input_labels[3, 3] = 4
    input_payload = ObjectLabelPayload(
        labels=input_labels,
        domain=ObjectLabelDomain(
            declared_object_count=9,
        ),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={
            "InputObjects": ObjectLabelSet(
                name="InputObjects",
                labels=input_payload,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="ExpandOrShrinkObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("InputObjects", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("InputObjects", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("ExpandedObjects", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("ExpandedObjects", ObjectLabelsArtifactType),),
                ),
            ),
        )
    )

    executor.run(
        expand_or_shrink_objects,
        np.zeros_like(input_labels, dtype=np.float32),
        cellprofiler_runtime=runtime,
        mode="expand_defined_pixels",
        iterations=1,
        dtype_config=DtypeConfig(),
    )

    _name, recorded_payload, _kwargs = runtime.objects[0]
    assert isinstance(recorded_payload, ObjectLabelSet)
    assert recorded_payload.domain.declared_object_count == 4
    assert recorded_payload.domain.declared_object_ids == ()
    assert int(np.max(recorded_payload.labels)) == 4


def test_align_measurement_builder_records_output_scoped_shifts() -> None:
    def align_function(image: np.ndarray) -> tuple[
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

    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Align",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("Stain1Raw", ImageArtifactType),
                        ArtifactSpec.input("Stain2Raw", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output("Stain1", ImageArtifactType),
                        ArtifactSpec.output("Stain2", ImageArtifactType),
                        ArtifactSpec.output(
                            "AlignMeasurements", MeasurementsArtifactType
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output("Stain1", ImageArtifactType),
                        ArtifactSpec.output("Stain2", ImageArtifactType),
                        ArtifactSpec.output(
                            "AlignMeasurements", MeasurementsArtifactType
                        ),
                    ),
                ),
            ),
        )
    )

    record = measurement_record_for_module(
        _cellprofiler_output_record_request(
            executor=executor,
            adapter=None,
            spec=ArtifactSpec.output("AlignMeasurements", MeasurementsArtifactType),
            output_value=(
                AlignShiftMeasurement(0, 0, 0.0, 0.0),
                AlignShiftMeasurement(0, 1, -1.0, 1.0),
            ),
            output_values={},
            source_image_name="Stain1Raw__Stain2Raw",
            func=align_function,
            call_kwargs={},
        )
    )

    assert record.object_name is None
    assert record.source_context.source_image_name is None
    assert record.rows == [
        {
            "slice_index": 0,
            "source_image_name": "Stain1",
            "feature_name": AlignModule.measurement_feature_name("x_shift"),
            "result_value": 0.0,
        },
        {
            "slice_index": 0,
            "source_image_name": "Stain1",
            "feature_name": AlignModule.measurement_feature_name("y_shift"),
            "result_value": 0.0,
        },
        {
            "slice_index": 0,
            "source_image_name": "Stain2",
            "feature_name": AlignModule.measurement_feature_name("x_shift"),
            "result_value": -1.0,
        },
        {
            "slice_index": 0,
            "source_image_name": "Stain2",
            "feature_name": AlignModule.measurement_feature_name("y_shift"),
            "result_value": 1.0,
        },
    ]


def test_align_measurement_builder_records_additional_output_shifts() -> None:
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Align",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("Template", ImageArtifactType),
                        ArtifactSpec.input("Red", ImageArtifactType),
                        ArtifactSpec.input("Combined", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output("AlignedTemplate", ImageArtifactType),
                        ArtifactSpec.output("AlignedRed", ImageArtifactType),
                        ArtifactSpec.output("AlignedCombined", ImageArtifactType),
                        ArtifactSpec.output(
                            "AlignMeasurements", MeasurementsArtifactType
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output("AlignedTemplate", ImageArtifactType),
                        ArtifactSpec.output("AlignedRed", ImageArtifactType),
                        ArtifactSpec.output("AlignedCombined", ImageArtifactType),
                        ArtifactSpec.output(
                            "AlignMeasurements", MeasurementsArtifactType
                        ),
                    ),
                ),
            ),
        )
    )

    record = measurement_record_for_module(
        _cellprofiler_output_record_request(
            executor=executor,
            adapter=None,
            spec=ArtifactSpec.output("AlignMeasurements", MeasurementsArtifactType),
            output_value=(
                AlignShiftMeasurement(0, 0, 0.0, 0.0),
                AlignShiftMeasurement(0, 1, -2.0, 1.0),
                AlignShiftMeasurement(0, 2, -2.0, 1.0),
            ),
            output_values={},
            source_image_name=None,
            func=lambda image: image,
            call_kwargs={},
        )
    )

    assert record.rows[-2:] == [
        {
            "slice_index": 0,
            "source_image_name": "AlignedCombined",
            "feature_name": AlignModule.measurement_feature_name("x_shift"),
            "result_value": -2.0,
        },
        {
            "slice_index": 0,
            "source_image_name": "AlignedCombined",
            "feature_name": AlignModule.measurement_feature_name("y_shift"),
            "result_value": 1.0,
        },
    ]


def test_align_measurement_builder_uses_declared_outputs_when_images_pruned() -> None:
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Align",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("Plate", ImageArtifactType),
                        ArtifactSpec.input("Well", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "AlignMeasurements", MeasurementsArtifactType
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output("AlignedPlate", ImageArtifactType),
                        ArtifactSpec.output("AlignedWell", ImageArtifactType),
                        ArtifactSpec.output(
                            "AlignMeasurements", MeasurementsArtifactType
                        ),
                    ),
                ),
            ),
        )
    )

    record = measurement_record_for_module(
        _cellprofiler_output_record_request(
            executor=executor,
            adapter=None,
            spec=ArtifactSpec.output("AlignMeasurements", MeasurementsArtifactType),
            output_value=(
                AlignShiftMeasurement(0, 0, 0.0, 0.0),
                AlignShiftMeasurement(0, 1, -3.0, 2.0),
            ),
            output_values={},
            source_image_name=None,
            func=lambda image: image,
            call_kwargs={},
        )
    )

    assert record.rows[-2:] == [
        {
            "slice_index": 0,
            "source_image_name": "AlignedWell",
            "feature_name": AlignModule.measurement_feature_name("x_shift"),
            "result_value": -3.0,
        },
        {
            "slice_index": 0,
            "source_image_name": "AlignedWell",
            "feature_name": AlignModule.measurement_feature_name("y_shift"),
            "result_value": 2.0,
        },
    ]


def test_measure_object_neighbors_binds_small_removed_label_variant() -> None:
    calls = []

    def measure_neighbors(
        image: np.ndarray,
        labels: np.ndarray,
        small_removed_labels: np.ndarray | None = None,
        neighbor_labels: np.ndarray | None = None,
        small_removed_neighbor_labels: np.ndarray | None = None,
        neighbors_are_same_objects: bool = False,
    ) -> tuple[np.ndarray, list[object]]:
        calls.append(
            (
                labels.copy(),
                None if small_removed_labels is None else small_removed_labels.copy(),
                neighbor_labels,
                small_removed_neighbor_labels,
                neighbors_are_same_objects,
            )
        )
        return image, []

    measure_neighbors.__processing_contract__ = ProcessingContract.FLEXIBLE
    final_labels = np.zeros((4, 4), dtype=np.int32)
    final_labels[1, 1] = 1
    small_removed = final_labels.copy()
    small_removed[1, 2] = 2
    fallback = np.zeros((4, 4), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={
            "Nuclei": ObjectLabelSet(
                name="Nuclei",
                labels=final_labels,
                small_removed_labels=small_removed,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectNeighbors",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Neighbors", MeasurementsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Neighbors", MeasurementsArtifactType),),
                ),
            ),
        )
    )

    result = executor.run(
        measure_neighbors,
        fallback,
        cellprofiler_runtime=runtime,
    )

    assert result is fallback
    bound_labels, bound_small_removed, bound_neighbor, bound_small_neighbor, same = (
        calls[0]
    )
    np.testing.assert_array_equal(bound_labels, final_labels)
    np.testing.assert_array_equal(bound_small_removed, small_removed)
    assert bound_neighbor is None
    assert bound_small_neighbor is None
    assert same is True


def test_classification_rows_include_unclassified_objects() -> None:
    def classify_like(image: np.ndarray) -> tuple[np.ndarray, ClassificationResult]:
        return image, ClassificationResult(
            slice_index=0,
            total_objects=3,
            bin_counts='{"Small": 1, "Large": 1}',
            bin_percentages='{"Small": 33.3333333333, "Large": 33.3333333333}',
            object_classes='{"1": "Small", "3": "Large"}',
        )

    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="ClassifyObjectsSingleMeasurement",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("ClassifyObjects", MeasurementsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("ClassifyObjects", MeasurementsArtifactType),),
                ),
            ),
        )
    )

    rows = measurement_record_for_module(
        _cellprofiler_output_record_request(
            executor=executor,
            adapter=None,
            spec=ArtifactSpec.output("ClassifyObjects", MeasurementsArtifactType),
            output_value=classify_like(np.zeros((2, 2), dtype=np.float32))[1],
            output_values={},
            source_image_name=None,
            func=classify_like,
            call_kwargs={},
        )
    ).rows

    object_rows = [row for row in rows if row.get("object_name") == "Nuclei"]
    assert len(object_rows) == 6
    assert {
        (row["object_label"], row["feature_name"], row["result_value"])
        for row in object_rows
    } == {
        (
            1,
            ClassifyObjectsSingleMeasurementModule.MeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
                bin_name="Small"
            ),
            1,
        ),
        (
            1,
            ClassifyObjectsSingleMeasurementModule.MeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
                bin_name="Large"
            ),
            0,
        ),
        (
            2,
            ClassifyObjectsSingleMeasurementModule.MeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
                bin_name="Small"
            ),
            0,
        ),
        (
            2,
            ClassifyObjectsSingleMeasurementModule.MeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
                bin_name="Large"
            ),
            0,
        ),
        (
            3,
            ClassifyObjectsSingleMeasurementModule.MeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
                bin_name="Small"
            ),
            0,
        ),
        (
            3,
            ClassifyObjectsSingleMeasurementModule.MeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
                bin_name="Large"
            ),
            1,
        ),
    }


def test_module_executor_records_multiple_declared_object_outputs() -> None:
    labels_without_overlap = np.ones((4, 5), dtype=np.int32)
    labels_with_overlap = np.full((4, 5), 2, dtype=np.int32)

    def untangle_like(
        image: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
        return image, {"worm_count": 1.0}, labels_with_overlap, labels_without_overlap

    untangle_like.__processing_contract__ = ProcessingContract.PURE_2D
    fallback = np.zeros((4, 5), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {
            "WormBinary": _FakeRuntimeImage(
                fallback,
                source_image_name="WormBinary",
            ),
        }
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="UntangleWorms",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("WormBinary", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "UntangleWorms_3_measurements", MeasurementsArtifactType
                        ),
                        ArtifactSpec.output(
                            "OverlappingWorms", ObjectLabelsArtifactType
                        ),
                        ArtifactSpec.output(
                            "NonOverlappingWorms", ObjectLabelsArtifactType
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "UntangleWorms_3_measurements", MeasurementsArtifactType
                        ),
                        ArtifactSpec.output(
                            "OverlappingWorms", ObjectLabelsArtifactType
                        ),
                        ArtifactSpec.output(
                            "NonOverlappingWorms", ObjectLabelsArtifactType
                        ),
                    ),
                ),
            ),
        )
    )

    result = executor.run(
        untangle_like,
        fallback,
        cellprofiler_runtime=runtime,
    )

    assert result is fallback
    assert _recorded_measurements_for_assertion(runtime.measurements) == [
        (
            "UntangleWorms_3_measurements",
            [{"worm_count": 1.0, "slice_index": 0}],
            {
                "object_name": None,
                "source_image_name": "WormBinary",
                "fields": ("slice_index", "worm_count"),
            },
        )
    ]
    assert [name for name, _labels, _kwargs in runtime.objects] == [
        "OverlappingWorms",
        "NonOverlappingWorms",
    ]
    np.testing.assert_array_equal(runtime.objects[0][1], labels_with_overlap)
    np.testing.assert_array_equal(runtime.objects[1][1], labels_without_overlap)


def test_default_measurement_builder_preserves_row_declared_object_scope() -> None:
    image = np.zeros((4, 5), dtype=np.float32)

    def object_rows(image: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
        return image, [
            {
                "object_name": "Worms",
                "object_number": 1,
                "worm_length": 10.0,
            }
        ]

    object_rows.__processing_contract__ = ProcessingContract.PURE_2D
    runtime = _FakeCellProfilerRuntime(
        {
            "WormBinary": _FakeRuntimeImage(
                image,
                source_image_name="WormBinary",
            ),
        }
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="UntangleWorms",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("WormBinary", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "UntangleWorms_3_measurements", MeasurementsArtifactType
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "UntangleWorms_3_measurements", MeasurementsArtifactType
                        ),
                    ),
                ),
            ),
        )
    )

    executor.run(
        object_rows,
        image,
        cellprofiler_runtime=runtime,
    )

    assert _recorded_measurements_for_assertion(runtime.measurements) == [
        (
            "UntangleWorms_3_measurements",
            [
                {
                    "object_name": "Worms",
                    "object_number": 1,
                    "worm_length": 10.0,
                    "slice_index": 0,
                }
            ],
            {
                "object_name": None,
                "source_image_name": None,
                "fields": (
                    "slice_index",
                    "object_name",
                    "object_number",
                    "worm_length",
                ),
            },
        )
    ]


def test_module_executor_routes_spatial_grid_artifacts() -> None:
    image = np.zeros((20, 20), dtype=np.float32)
    grid = SpatialGrid(
        name="grid_info",
        rows=2,
        columns=2,
        x_spacing=8.0,
        y_spacing=8.0,
        x_origin=4.0,
        y_origin=4.0,
    )
    runtime = _FakeCellProfilerRuntime({"DNA": _FakeRuntimeImage(image)})
    define_executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="DefineGrid",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("DNA", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Grid", SpatialGridArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Grid", SpatialGridArtifactType),),
                ),
            ),
        )
    )
    identify_executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="IdentifyObjectsInGrid",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Grid", SpatialGridArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Grid", SpatialGridArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("GridObjects", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("GridObjects", ObjectLabelsArtifactType),),
                ),
            ),
        )
    )

    def define_grid_like(image: np.ndarray) -> tuple[np.ndarray, SpatialGrid]:
        return image, grid

    define_grid_like.__processing_contract__ = ProcessingContract.PURE_2D

    @special_inputs("grid")
    def identify_grid_like(
        image: np.ndarray,
        grid: SpatialGrid,
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = np.full(image.shape, grid.rows * grid.columns, dtype=np.int32)
        return image, labels

    identify_grid_like.__processing_contract__ = ProcessingContract.PURE_2D

    define_executor.run(define_grid_like, image, cellprofiler_runtime=runtime)
    identify_executor.run(identify_grid_like, image, cellprofiler_runtime=runtime)

    assert runtime.spatial_grids["Grid"].rows == 2
    assert runtime.spatial_grids["Grid"].columns == 2
    assert [name for name, _labels, _kwargs in runtime.objects] == ["GridObjects"]
    np.testing.assert_array_equal(runtime.objects[0][1], np.full(image.shape, 4))


def test_define_grid_manual_once_scope_executes_once_for_stacked_image() -> None:
    image = np.zeros((3, 20, 20), dtype=np.float32)
    grid = SpatialGrid(
        name="grid_info",
        rows=2,
        columns=2,
        x_spacing=8.0,
        y_spacing=8.0,
        x_origin=4.0,
        y_origin=4.0,
    )
    runtime = _FakeCellProfilerRuntime({"DNA": _FakeRuntimeImage(image)})
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="DefineGridManual",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("DNA", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Grid", SpatialGridArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Grid", SpatialGridArtifactType),),
                ),
            ),
        )
    )
    calls = 0

    def define_grid_like(image: np.ndarray) -> tuple[np.ndarray, SpatialGrid]:
        nonlocal calls
        calls += 1
        return image, grid

    attach_callable_contract_metadata(
        define_grid_like,
        declared_processing_contract="pure_2d",
    )

    executor.run(
        define_grid_like,
        image,
        cellprofiler_runtime=runtime,
        invocation_options=DefineGridInvocationOptions(
            cycle_scope=DefineGridCycleScope.ONCE
        ),
    )

    assert calls == 1
    assert runtime.spatial_grids["Grid"].rows == 2
    assert runtime.spatial_grids["Grid"].columns == 2


def test_define_grid_manual_each_cycle_scope_emits_slice_aligned_grids() -> None:
    image = np.stack(
        [
            np.full((20, 20), 1, dtype=np.float32),
            np.full((20, 20), 2, dtype=np.float32),
            np.full((20, 20), 3, dtype=np.float32),
        ],
        axis=0,
    )
    runtime = _FakeCellProfilerRuntime({"DNA": _FakeRuntimeImage(image)})
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="DefineGridManual",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("DNA", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Grid", SpatialGridArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Grid", SpatialGridArtifactType),),
                ),
            ),
        )
    )
    calls = 0

    def define_grid_like(image: np.ndarray) -> tuple[np.ndarray, SpatialGrid]:
        nonlocal calls
        calls += 1
        return image, SpatialGrid(
            name="grid_info",
            rows=2,
            columns=2,
            x_spacing=8.0,
            y_spacing=8.0,
            x_origin=float(image[0, 0]),
            y_origin=4.0,
        )

    attach_callable_contract_metadata(
        define_grid_like,
        declared_processing_contract="pure_2d",
    )

    executor.run(
        define_grid_like,
        image,
        cellprofiler_runtime=runtime,
        invocation_options=DefineGridInvocationOptions(
            cycle_scope=DefineGridCycleScope.EACH_CYCLE
        ),
    )

    assert calls == 3
    grids = runtime.spatial_grids["Grid"]
    assert isinstance(grids, RuntimeSliceAlignedValues)
    assert [grid.x_origin for grid in grids.slices] == [1.0, 2.0, 3.0]


def test_grid_only_module_uses_single_carrier_plane_for_stacked_image() -> None:
    image = np.zeros((3, 20, 20), dtype=np.float32)
    grid = SpatialGrid(
        name="Grid",
        rows=2,
        columns=2,
        x_spacing=8.0,
        y_spacing=8.0,
        x_origin=4.0,
        y_origin=4.0,
    )
    runtime = _FakeCellProfilerRuntime({"DNA": _FakeRuntimeImage(image)})
    runtime.spatial_grids["Grid"] = grid
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="IdentifyObjectsInGrid",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Grid", SpatialGridArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Grid", SpatialGridArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("GridObjects", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("GridObjects", ObjectLabelsArtifactType),),
                ),
            ),
        )
    )

    @special_inputs("grid")
    def identify_grid_like(
        image: np.ndarray,
        grid: SpatialGrid,
    ) -> tuple[np.ndarray, np.ndarray]:
        return image, np.full(image.shape, grid.rows * grid.columns, dtype=np.int32)

    attach_callable_contract_metadata(
        identify_grid_like,
        declared_processing_contract="pure_2d",
    )

    executor.run(identify_grid_like, image, cellprofiler_runtime=runtime)

    assert len(runtime.objects) == 1
    assert runtime.objects[0][1].shape == (20, 20)


def test_grid_input_module_slices_runtime_aligned_grid_for_2d_carrier() -> None:
    image = np.zeros((20, 20), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime({"DNA": _FakeRuntimeImage(image)})
    runtime.spatial_grids["Grid"] = RuntimeSliceAlignedValues(
        slices=(
            SpatialGrid(
                name="Grid",
                rows=2,
                columns=2,
                x_spacing=8.0,
                y_spacing=8.0,
                x_origin=1.0,
                y_origin=4.0,
            ),
            SpatialGrid(
                name="Grid",
                rows=2,
                columns=2,
                x_spacing=8.0,
                y_spacing=8.0,
                x_origin=2.0,
                y_origin=4.0,
            ),
        )
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="IdentifyObjectsInGrid",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Grid", SpatialGridArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Grid", SpatialGridArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("GridObjects", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("GridObjects", ObjectLabelsArtifactType),),
                ),
            ),
        )
    )
    seen_origins: list[float] = []

    @special_inputs("grid")
    def identify_grid_like(
        image: np.ndarray,
        grid: SpatialGrid,
    ) -> tuple[np.ndarray, np.ndarray]:
        seen_origins.append(grid.x_origin)
        return image, np.full(image.shape, grid.x_origin, dtype=np.int32)

    attach_callable_contract_metadata(
        identify_grid_like,
        declared_processing_contract="pure_2d",
    )

    executor.run(identify_grid_like, image, cellprofiler_runtime=runtime)

    assert seen_origins == [1.0, 2.0]
    assert runtime.objects[0][1].shape == (2, 20, 20)


def test_current_runtime_plane_projection_slices_aligned_grid_for_grouped_2d_invocation() -> (
    None
):
    class _PlaneProjector(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return 1

    grids = RuntimeSliceAlignedValues(
        slices=(
            SpatialGrid(
                name="Grid",
                rows=1,
                columns=1,
                x_spacing=1.0,
                y_spacing=1.0,
                x_origin=1.0,
                y_origin=1.0,
            ),
            SpatialGrid(
                name="Grid",
                rows=1,
                columns=1,
                x_spacing=1.0,
                y_spacing=1.0,
                x_origin=2.0,
                y_origin=2.0,
            ),
        )
    )

    projected = CurrentRuntimePlaneKwargProjection(
        image=np.zeros((5, 5), dtype=np.float32),
        kwargs={"grid": grids, "shape_choice": "natural_shape_and_location"},
        plane_projector=_PlaneProjector(),
        project_runtime_slice_kwargs=True,
    ).kwargs_for_invocation()

    assert projected["grid"].x_origin == 2.0
    assert projected["shape_choice"] == "natural_shape_and_location"


def test_current_runtime_plane_projection_broadcasts_singleton_aligned_grid_for_grouped_2d_invocation() -> (
    None
):
    class _PlaneProjector(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return 1

        def runtime_slice_axis_size(self) -> int | None:
            return 2

    grid = SpatialGrid(
        name="Grid",
        rows=1,
        columns=1,
        x_spacing=1.0,
        y_spacing=1.0,
        x_origin=4.0,
        y_origin=4.0,
    )

    projected = CurrentRuntimePlaneKwargProjection(
        image=np.zeros((5, 5), dtype=np.float32),
        kwargs={"grid": RuntimeSliceAlignedValues((grid,))},
        plane_projector=_PlaneProjector(),
        project_runtime_slice_kwargs=True,
    ).kwargs_for_invocation()

    assert projected["grid"] is grid


def test_current_runtime_plane_projection_slices_object_labels_for_grouped_2d_invocation() -> (
    None
):
    class _PlaneProjector(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return 1

    label_planes = np.stack(
        (
            np.full((5, 5), 3, dtype=np.int32),
            np.full((5, 5), 7, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        labels=label_planes,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((3,), (7,)),
        ),
    )

    projected = CurrentRuntimePlaneKwargProjection(
        image=np.zeros((5, 5), dtype=np.float32),
        kwargs={"labels": labels, "shape_choice": "natural_shape_and_location"},
        plane_projector=_PlaneProjector(),
        project_runtime_slice_kwargs=True,
    ).kwargs_for_invocation()

    projected_labels = projected["labels"]
    assert isinstance(projected_labels, ObjectLabelSet)
    assert projected_labels.shape == (5, 5)
    assert projected_labels.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert projected_labels.source_image_names == ()
    np.testing.assert_array_equal(projected_labels.labels, label_planes[1])
    assert projected["shape_choice"] == "natural_shape_and_location"


def test_current_runtime_plane_projection_contract_preserves_payload_scoped_kwargs() -> (
    None
):
    class _PlaneProjector(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return 1

    label_planes = np.stack(
        (
            np.full((5, 5), 3, dtype=np.int32),
            np.full((5, 5), 7, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        labels=label_planes,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
    )
    contract = CurrentRuntimePlaneKwargProjectionContract(
        convert_objects_to_image,
        ImagePayloadExecutionMode.NATURAL,
    )

    projected = CurrentRuntimePlaneKwargProjection(
        image=np.zeros((5, 5), dtype=np.float32),
        kwargs={"labels": labels},
        plane_projector=_PlaneProjector(),
        project_runtime_slice_kwargs=contract.projects_runtime_slice_kwargs(),
    ).kwargs_for_invocation()

    assert projected["labels"] is labels


def test_runtime_plane_projection_contract_projects_source_identity_image_inputs() -> (
    None
):
    def two_dimensional_identity(image):
        return image

    two_dimensional_identity.__processing_contract__ = ProcessingContract.PURE_2D

    contract = CurrentRuntimePlaneKwargProjectionContract(
        convert_objects_to_image,
        ImagePayloadExecutionMode.NATURAL,
    )
    source_stack_contract = CurrentRuntimePlaneKwargProjectionContract(
        two_dimensional_identity,
        ImagePayloadExecutionMode.NATURAL,
    )

    assert not contract.projects_runtime_artifact_image_inputs()
    assert source_stack_contract.projects_runtime_artifact_image_inputs()
    assert source_stack_contract.projects_runtime_slice_kwargs()


def test_runtime_plane_projection_contract_uses_nominal_registered_capabilities() -> (
    None
):
    def two_dimensional_identity(image):
        return image

    @runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
    def full_stack_identity(image):
        return image

    two_dimensional_identity.__processing_contract__ = ProcessingContract.PURE_2D
    full_stack_identity.__processing_contract__ = ProcessingContract.PURE_2D
    registered_capabilities = (
        CellProfilerRuntimePlaneProjectionCapability.registered_capability_types()
    )
    plane_contract = CurrentRuntimePlaneKwargProjectionContract(
        two_dimensional_identity,
        ImagePayloadExecutionMode.NATURAL,
    )
    full_stack_contract = CurrentRuntimePlaneKwargProjectionContract(
        full_stack_identity,
        ImagePayloadExecutionMode.NATURAL,
    )

    assert RuntimeArtifactImageInputProjectionCapability in registered_capabilities
    assert RuntimeSliceKwargProjectionCapability in registered_capabilities
    assert CurrentSourceImagePayloadProjectionCapability in registered_capabilities
    assert plane_contract.requires_projection_capability(
        RuntimeArtifactValueProjectionCapability
    )
    assert plane_contract.projection_capabilities() == frozenset(
        (
            RuntimeArtifactImageInputProjectionCapability,
            RuntimeSliceKwargProjectionCapability,
        )
    )
    assert full_stack_contract.projection_capabilities() == frozenset()


def test_module_image_request_projects_current_image_to_grouped_runtime_plane() -> None:
    def two_dimensional_identity(image):
        return image

    two_dimensional_identity.__processing_contract__ = ProcessingContract.PURE_2D
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="IdentifyPrimaryObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition, ()
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition, ()
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition, ()
                ),
            ),
        )
    )
    adapter = CellProfilerRuntimeAdapter(
        runtime_value_store=RuntimeValueStore(),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=AllComponents.SITE,
            value="2",
        ),
        plane_projection=RuntimePlaneProjection.group(1, 2),
    )
    current_image = RuntimeImagePayloadContext(
        np.stack(
            [
                np.full((5, 6), 11, dtype=np.float32),
                np.full((5, 6), 29, dtype=np.float32),
            ]
        ),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("site1.png", "site2.png"),
                component_metadata=(
                    {AllComponents.SITE.value: "1"},
                    {AllComponents.SITE.value: "2"},
                ),
            )
        ),
        mask=None,
    ).payload()
    plan = executor.runtime_plan(two_dimensional_identity)

    image_request = executor._image_request(plan, current_image, adapter)

    np.testing.assert_array_equal(
        image_payload_data(image_request.payload),
        np.full((5, 6), 29, dtype=np.float32),
    )
    metadata = image_payload_metadata(image_request.payload)
    assert metadata.source_path == "site2.png"
    assert metadata.source_component_metadata == {AllComponents.SITE.value: "2"}


def test_module_executor_preserves_full_stack_runtime_image_input_scope() -> None:
    def volume_identity(image):
        return image

    volume_identity.__processing_contract__ = ProcessingContract.PURE_3D
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Resize",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Monolayer", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Monolayer", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("ResizedMonolayer", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("ResizedMonolayer", ImageArtifactType),),
                ),
            ),
        )
    )
    current_image = RuntimeImagePayloadContext(
        np.zeros((2, 4, 4), dtype=np.float32),
        metadata=ImagePayloadMetadata(),
        mask=None,
    ).payload()
    plan = executor.runtime_plan(volume_identity)

    assert (
        executor._runtime_image_current_image(
            plan,
            SimpleNamespace(),
            ArtifactSpec.input("Monolayer", ImageArtifactType),
            current_image,
        )
        is None
    )
    seen_current_images = []

    def get_image(name, current_image=None):
        seen_current_images.append(current_image)
        return SimpleNamespace(
            data=np.zeros((2, 4, 4), dtype=np.float32),
            source_image_name=f"{name}_source",
        )

    adapter = SimpleNamespace(
        get_image=get_image,
        require_resolvable_source_aliases=lambda aliases: None,
    )
    image_request = executor._image_request(
        plan,
        current_image,
        adapter,
    )
    assert image_request.source_image_name == "Monolayer_source"
    assert seen_current_images == [None]


def test_current_runtime_plane_projection_slices_aligned_object_label_set_for_grouped_2d_invocation() -> (
    None
):
    class _PlaneProjector(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return 1

    first_labels = ObjectLabelSet(
        name="Cells",
        labels=np.full((5, 5), 3, dtype=np.int32),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_ids=(3,),
        ),
    )
    second_labels = ObjectLabelSet(
        name="Cells",
        labels=np.full((5, 5), 7, dtype=np.int32),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_ids=(7,),
        ),
    )
    aligned_labels = RuntimeSliceAlignedValues((first_labels, second_labels))

    projected = CurrentRuntimePlaneKwargProjection(
        image=np.zeros((5, 5), dtype=np.float32),
        kwargs={
            "labels": aligned_labels,
            "measurement_values": RuntimeSliceAlignedValues(
                (np.asarray([1.0]), np.asarray([2.0]))
            ),
        },
        plane_projector=_PlaneProjector(),
        project_runtime_slice_kwargs=True,
    ).kwargs_for_invocation()

    projected_labels = projected["labels"]
    assert projected_labels is second_labels
    np.testing.assert_array_equal(projected_labels.labels, second_labels.labels)
    np.testing.assert_array_equal(projected["measurement_values"], np.asarray([2.0]))


def test_current_runtime_plane_projection_collapses_singleton_object_label_stack() -> (
    None
):
    class _PlaneProjector(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return 1

    label_plane = np.full((5, 5), 3, dtype=np.int32)
    labels = ObjectLabelSet(
        name="Cells",
        labels=label_plane[np.newaxis, :, :],
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((3,),),
        ),
    )

    projected = CurrentRuntimePlaneKwargProjection(
        image=np.zeros((5, 5), dtype=np.float32),
        kwargs={"labels": labels},
        plane_projector=_PlaneProjector(),
        project_runtime_slice_kwargs=True,
    ).kwargs_for_invocation()

    projected_labels = projected["labels"]
    assert isinstance(projected_labels, ObjectLabelSet)
    assert projected_labels.shape == (5, 5)
    assert projected_labels.domain.scope is ObjectLabelDomainScope.PAYLOAD
    np.testing.assert_array_equal(projected_labels.labels, label_plane)


def test_special_input_preserves_object_label_stack_when_group_index_is_out_of_domain() -> (
    None
):
    label_planes = np.stack(
        (
            np.full((5, 5), 3, dtype=np.int32),
            np.full((5, 5), 7, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=label_planes,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((3,), (7,)),
        ),
    )
    spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    request = SpecialInputBindingRequest(
        module_name="MaskImage",
        adapter=_RuntimeSliceObjectAdapter(labels, slice_index=2),
        kwargs={},
        current_image=np.zeros((5, 5), dtype=np.float32),
        binding_scope=EMPTY_RUNTIME_ARTIFACT_BINDING_SCOPE,
        parameter_names=("mask",),
        special_input_specs=(spec,),
        runtime_inputs=(spec,),
    )

    projected = request.current_plane_object_label_runtime_value(spec)

    assert projected.shape == label_planes.shape
    np.testing.assert_array_equal(projected, label_planes)


def test_grid_input_module_unwraps_singleton_runtime_aligned_grid_for_2d_carrier() -> (
    None
):
    image = np.zeros((20, 20), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime({"DNA": _FakeRuntimeImage(image)})
    runtime.spatial_grids["Grid"] = RuntimeSliceAlignedValues(
        slices=(
            SpatialGrid(
                name="Grid",
                rows=2,
                columns=2,
                x_spacing=8.0,
                y_spacing=8.0,
                x_origin=1.0,
                y_origin=4.0,
            ),
        )
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="IdentifyObjectsInGrid",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Grid", SpatialGridArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Grid", SpatialGridArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("GridObjects", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("GridObjects", ObjectLabelsArtifactType),),
                ),
            ),
        )
    )
    seen_origins: list[float] = []

    @special_inputs("grid")
    def identify_grid_like(
        image: np.ndarray,
        grid: SpatialGrid,
    ) -> tuple[np.ndarray, np.ndarray]:
        seen_origins.append(grid.x_origin)
        return image, np.full(image.shape, grid.x_origin, dtype=np.int32)

    attach_callable_contract_metadata(
        identify_grid_like,
        declared_processing_contract="pure_2d",
    )

    executor.run(identify_grid_like, image, cellprofiler_runtime=runtime)

    assert seen_origins == [1.0]
    assert runtime.objects[0][1].shape == (1, 20, 20)


def test_flexible_object_module_slices_tuple_label_stack() -> None:
    image = np.zeros((3, 6, 6), dtype=np.float32)
    labels = np.zeros((3, 6, 6), dtype=np.int32)
    labels[:, 1:4, 1:4] = np.arange(1, 4, dtype=np.int32)[:, None, None]
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(image)},
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=labels,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("FilteredCells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("FilteredCells", ObjectLabelsArtifactType),),
                ),
            ),
        )
    )
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def filter_like(
        image: np.ndarray,
        object_labels: tuple[np.ndarray, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append((tuple(image.shape), tuple(object_labels[0].shape)))
        return image, object_labels[0]

    attach_callable_contract_metadata(
        filter_like,
        declared_processing_contract="flexible",
    )

    executor.run(filter_like, image, cellprofiler_runtime=runtime, slice_by_slice=True)

    assert calls == [((6, 6), (6, 6)), ((6, 6), (6, 6)), ((6, 6), (6, 6))]
    assert runtime.objects[0][1].shape == (3, 6, 6)


def test_object_only_reference_image_collapses_payload_stack() -> None:
    payload = RuntimeImagePayloadContext(
        np.zeros((4, 6, 6), dtype=np.float32),
        metadata=ImagePayloadMetadata(source_dtype="float32"),
        mask=None,
    ).payload()

    reference = OBJECT_ONLY_REFERENCE_IMAGE.reference_image(payload)

    assert reference.shape == (6, 6)


def test_object_only_reference_image_collapses_aligned_stack() -> None:
    payload = AlignedImageStack(
        (
            RuntimeImagePayloadContext(
                np.zeros((6, 6), dtype=np.float32),
                metadata=ImagePayloadMetadata(source_dtype="float32"),
                mask=None,
            ).payload(),
            RuntimeImagePayloadContext(
                np.ones((6, 6), dtype=np.float32),
                metadata=ImagePayloadMetadata(source_dtype="float32"),
                mask=None,
            ).payload(),
        )
    )

    reference = OBJECT_ONLY_REFERENCE_IMAGE.reference_image(payload)

    assert reference.shape == (6, 6)
    assert np.all(reference == 0)


def test_flexible_object_module_slices_measurement_tables_with_label_stack() -> None:
    image = np.zeros((2, 6, 6), dtype=np.float32)
    labels = np.zeros((2, 6, 6), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 3:5, 3:5] = 1
    measurements = MeasurementTable(
        name="CellShape",
        object_name="Cells",
        object_id_field="object_label",
        rows=[
            {"slice_index": 0, "object_label": 1, "Area": 4.0},
            {"slice_index": 1, "object_label": 1, "Area": 9.0},
        ],
    )
    relationship_measurements = MeasurementTable(
        name="RelationshipFacts",
        object_name="Cells",
        object_id_field="object_label",
        rows=[
            {"slice_index": 999, "object_label": 1, "Children_Count": 1},
        ],
    )
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(image)},
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=labels,
            )
        },
        measurement_tables={"Cells": (measurements, relationship_measurements)},
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("FilteredCells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("FilteredCells", ObjectLabelsArtifactType),),
                ),
            ),
        )
    )
    seen_areas: list[tuple[float, ...]] = []

    def filter_like(
        image: np.ndarray,
        object_labels: tuple[np.ndarray, ...] = (),
        measurement_tables: tuple[MeasurementTable, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        del object_labels
        seen_areas.append(
            tuple(
                float(row["Area"]) for table in measurement_tables for row in table.rows
            )
        )
        return image, np.zeros(image.shape, dtype=np.int32)

    attach_callable_contract_metadata(
        filter_like,
        declared_processing_contract="flexible",
    )

    executor.run(filter_like, image, cellprofiler_runtime=runtime, slice_by_slice=True)

    assert seen_areas == [(4.0,), (9.0,)]


def test_filterobjects_binds_selection_measurement_values_to_label_slices() -> None:
    image = np.zeros((2, 6, 6), dtype=np.float32)
    children = np.zeros((2, 6, 6), dtype=np.int32)
    children[:, 0:2, 0:2] = 1
    children[:, 3:5, 3:5] = 2
    parents = np.ones_like(children)
    measurements = MeasurementTable(
        name="ChildMeasurements",
        object_name="Cells",
        object_id_field="object_label",
        rows=[
            {"slice_index": 0, "object_label": 1, "AreaShape_Area": 10.0},
            {"slice_index": 0, "object_label": 2, "AreaShape_Area": 20.0},
            {"slice_index": 1, "object_label": 1, "AreaShape_Area": 30.0},
            {"slice_index": 1, "object_label": 2, "AreaShape_Area": 5.0},
        ],
    )
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(image)},
        objects={
            "Cells": ObjectLabelSet(name="Cells", labels=children),
            "Tiles": ObjectLabelSet(name="Tiles", labels=parents),
        },
        measurement_tables={"Cells": (measurements,)},
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Tiles", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Tiles", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "FilterObjects_measurements", MeasurementsArtifactType
                        ),
                        ArtifactSpec.output("FilteredCells", ObjectLabelsArtifactType),
                        ArtifactSpec.output(
                            "Cells_FilteredCells_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "FilterObjects_measurements", MeasurementsArtifactType
                        ),
                        ArtifactSpec.output("FilteredCells", ObjectLabelsArtifactType),
                        ArtifactSpec.output(
                            "Cells_FilteredCells_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
            ),
        )
    )

    executor.run(
        filter_objects,
        image,
        cellprofiler_runtime=runtime,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
        measurement_features=("AreaShape_Area",),
        enclosing_object_name="Tiles",
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        slice_by_slice=True,
    )

    filtered = next(
        value for name, value, _kwargs in runtime.objects if name == "FilteredCells"
    )
    assert filtered.shape == (2, 6, 6)
    assert filtered[0, 0, 0] == 0
    assert filtered[0, 3, 3] == 1
    assert filtered[1, 0, 0] == 1
    assert filtered[1, 3, 3] == 0


def test_object_declared_measurement_rows_do_not_inherit_carrier_image_number() -> None:
    carrier = RuntimeImagePayloadContext(
        np.zeros((6, 6), dtype=np.float32),
        metadata=ImagePayloadMetadata(source_path="/source/site2.tif"),
        mask=None,
    ).payload()
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(carrier)},
        image_number_start=1,
        image_numbers_by_source_path={"/source/site2.tif": 2},
    )
    record = CellProfilerMeasurementRecord(
        rows=[
            {
                "slice_index": 0,
                "object_name": "Cells",
                "object_label": 1,
                "area": 4.0,
            }
        ],
        object_name=None,
        source_context=CellProfilerMeasurementSourceContext(
            source_image_name=None,
            source_image_payload=carrier,
        ),
    )

    projected_rows, _projected_mappings = CellProfilerMeasurementProjectionRequest(
        adapter=runtime,
        rows=record.rows,
        source_context=record.source_context,
        object_name=record.object_name,
        need_row_mappings=False,
    ).project_rows()

    assert record.source_context.source_image_payload is None
    assert projected_rows == [
        {
            "slice_index": 0,
            "object_name": "Cells",
            "object_label": 1,
            "area": 4.0,
            "image_number": 1,
        }
    ]


def test_mixed_object_measurement_rows_partition_by_declared_owner() -> None:
    carrier = RuntimeImagePayloadContext(
        np.zeros((2, 6, 6), dtype=np.float32),
        metadata=ImagePayloadMetadata(source_path="/source/site2.tif"),
        mask=None,
    ).payload()
    labels = np.zeros((2, 6, 6), dtype=np.int32)
    labels[:, 1:3, 1:3] = 1
    source_paths = ("/source/site1.tif", "/source/site2.tif")
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(carrier)},
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=labels,
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=source_paths
                ),
            )
        },
        image_number_start=1,
        image_numbers_by_source_path={
            "/source/site1.tif": 1,
            "/source/site2.tif": 2,
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "FilterObjects_measurements", MeasurementsArtifactType
                        ),
                        ArtifactSpec.output("FilteredCells", ObjectLabelsArtifactType),
                        ArtifactSpec.output(
                            "Cells_FilteredCells_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "FilterObjects_measurements", MeasurementsArtifactType
                        ),
                        ArtifactSpec.output("FilteredCells", ObjectLabelsArtifactType),
                        ArtifactSpec.output(
                            "Cells_FilteredCells_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
            ),
        )
    )

    def filter_like(
        image: np.ndarray,
        object_labels: tuple[np.ndarray, ...] = (),
    ) -> tuple[
        np.ndarray, FilterObjectsStats, ObjectLabelSet, ParentChildRelationshipPayload
    ]:
        return (
            image,
            FilterObjectsStats(
                slice_index=0,
                objects_pre_filter=1,
                objects_post_filter=1,
                objects_removed=0,
            ),
            ObjectLabelSet(
                name="FilteredCells",
                labels=object_labels[0],
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=source_paths
                ),
            ),
            ParentChildRelationshipPayload(
                parent_ids=(1,),
                child_ids=(1,),
                slice_indices=(0,),
                slice_count=2,
            ),
        )

    attach_callable_contract_metadata(
        filter_like,
        declared_processing_contract="flexible",
    )

    executor.run(
        filter_like, carrier, cellprofiler_runtime=runtime, slice_by_slice=True
    )

    assert len(runtime.measurements) == 3
    assert [
        (kwargs["object_name"], [dict(row)["image_number"] for row in rows])
        for _name, rows, kwargs in runtime.measurements
    ] == [
        (None, [1, 2]),
        ("Cells", [1, 2]),
        ("FilteredCells", [1, 2]),
    ]


def test_flexible_object_module_slices_measurement_tables_with_2d_labels() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:3, 1:3] = 1
    measurements = MeasurementTable(
        name="TileIntensity",
        object_name="Tiles",
        object_id_field="object_label",
        rows=[
            {"slice_index": 0, "object_label": 1, "StdIntensity": 4.0},
            {"slice_index": 1, "object_label": 1, "StdIntensity": 9.0},
        ],
    )
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(image)},
        objects={
            "Tiles": ObjectLabelSet(
                name="Tiles",
                labels=labels,
            )
        },
        measurement_tables={"Tiles": (measurements,)},
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Tiles", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Tiles", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("FilteredTiles", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("FilteredTiles", ObjectLabelsArtifactType),),
                ),
            ),
        )
    )
    seen_areas: list[tuple[float, tuple[int, ...]]] = []

    def filter_like(
        image: np.ndarray,
        object_labels: tuple[np.ndarray, ...] = (),
        measurement_tables: tuple[MeasurementTable, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        area = float(measurement_tables[0].rows[0]["StdIntensity"])
        seen_areas.append((area, tuple(object_labels[0].shape)))
        return image, object_labels[0] * int(area)

    attach_callable_contract_metadata(
        filter_like,
        declared_processing_contract="flexible",
    )

    executor.run(filter_like, image, cellprofiler_runtime=runtime, slice_by_slice=True)

    assert seen_areas == [(4.0, (6, 6)), (9.0, (6, 6))]
    assert runtime.objects[0][1].shape == (2, 6, 6)
    np.testing.assert_array_equal(runtime.objects[0][1][0], labels * 4)
    np.testing.assert_array_equal(runtime.objects[0][1][1], labels * 9)


def test_artifact_measurement_table_does_not_drive_object_only_slicing() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:3, 1:3] = 1
    relationship_facts = MeasurementTable(
        name="RelationshipFacts",
        rows=[
            {"slice_index": index, "object_name": "Cells", "object_label": index}
            for index in range(4)
        ],
    )
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(image)},
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=labels,
            )
        },
        measurement_tables={"Cells": (relationship_facts,)},
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("FilteredCells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("FilteredCells", ObjectLabelsArtifactType),),
                ),
            ),
        )
    )
    calls: list[tuple[int, ...]] = []

    def filter_like(
        image: np.ndarray,
        object_labels: tuple[np.ndarray, ...] = (),
        measurement_tables: tuple[MeasurementTable, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        del measurement_tables
        calls.append(tuple(object_labels[0].shape))
        return image, object_labels[0]

    attach_callable_contract_metadata(
        filter_like,
        declared_processing_contract="flexible",
    )

    executor.run(filter_like, image, cellprofiler_runtime=runtime, slice_by_slice=True)

    assert calls == [(6, 6)]
    assert runtime.objects[0][1].shape == (6, 6)


def test_flexible_object_module_aggregates_sliced_relationship_payloads() -> None:
    image = np.zeros((3, 6, 6), dtype=np.float32)
    labels = np.zeros((3, 6, 6), dtype=np.int32)
    labels[:, 1:4, 1:4] = np.arange(1, 4, dtype=np.int32)[:, None, None]
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(image)},
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=labels,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output("FilteredCells", ObjectLabelsArtifactType),
                        ArtifactSpec.output(
                            "Cells_FilteredCells_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output("FilteredCells", ObjectLabelsArtifactType),
                        ArtifactSpec.output(
                            "Cells_FilteredCells_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
            ),
        )
    )

    def filter_like(
        image: np.ndarray,
        object_labels: tuple[np.ndarray, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray, ParentChildRelationshipPayload]:
        label_id = int(np.max(object_labels[0]))
        return (
            image,
            object_labels[0],
            ParentChildRelationshipPayload(
                parent_ids=(label_id,),
                child_ids=(label_id,),
            ),
        )

    attach_callable_contract_metadata(
        filter_like,
        declared_processing_contract="flexible",
    )

    executor.run(filter_like, image, cellprofiler_runtime=runtime, slice_by_slice=True)

    assert runtime.relationships == [
        (
            "Cells_FilteredCells_relationships",
            {
                "parent_object_name": "Cells",
                "child_object_name": "FilteredCells",
                "parent_ids": (1, 2, 3),
                "child_ids": (1, 2, 3),
                "slice_indices": (0, 1, 2),
                "slice_count": 3,
                "source_path": None,
                "source_component_metadata": None,
                "source_image_provenance_planes": SourceImageProvenancePlanes(),
            },
        )
    ]


def test_flexible_filter_objects_aggregates_measurement_prefixed_relationships() -> (
    None
):
    image = np.zeros((3, 6, 6), dtype=np.float32)
    labels = np.zeros((3, 6, 6), dtype=np.int32)
    labels[:, 2:5, 2:5] = np.arange(1, 4, dtype=np.int32)[:, None, None]
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(image)},
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=labels,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "FilterObjects_measurements", MeasurementsArtifactType
                        ),
                        ArtifactSpec.output("FilteredCells", ObjectLabelsArtifactType),
                        ArtifactSpec.output(
                            "Cells_FilteredCells_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "FilterObjects_measurements", MeasurementsArtifactType
                        ),
                        ArtifactSpec.output("FilteredCells", ObjectLabelsArtifactType),
                        ArtifactSpec.output(
                            "Cells_FilteredCells_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
            ),
        )
    )

    def filter_like(
        image: np.ndarray,
        object_labels: tuple[np.ndarray, ...] = (),
    ) -> tuple[
        np.ndarray,
        FilterObjectsStats,
        np.ndarray,
        ParentChildRelationshipPayload,
    ]:
        label_id = int(np.max(object_labels[0]))
        return (
            image,
            FilterObjectsStats(
                slice_index=0,
                objects_pre_filter=1,
                objects_post_filter=1,
                objects_removed=0,
            ),
            object_labels[0],
            ParentChildRelationshipPayload(
                parent_ids=(label_id,),
                child_ids=(label_id,),
            ),
        )

    attach_callable_contract_metadata(
        filter_like,
        declared_processing_contract="flexible",
    )

    executor.run(filter_like, image, cellprofiler_runtime=runtime, slice_by_slice=True)

    assert runtime.relationships == [
        (
            "Cells_FilteredCells_relationships",
            {
                "parent_object_name": "Cells",
                "child_object_name": "FilteredCells",
                "parent_ids": (1, 2, 3),
                "child_ids": (1, 2, 3),
                "slice_indices": (0, 1, 2),
                "slice_count": 3,
                "source_path": None,
                "source_component_metadata": None,
                "source_image_provenance_planes": SourceImageProvenancePlanes(),
            },
        )
    ]


def test_relationship_measurements_preserve_pure_2d_slice_indices() -> None:
    parent_labels = np.zeros((2, 5, 5), dtype=np.int32)
    child_labels = np.zeros((2, 5, 5), dtype=np.int32)
    parent_labels[0, 1:3, 1:3] = 1
    child_labels[0, 1:3, 1:3] = 1
    parent_labels[1, 2:4, 2:4] = 2
    child_labels[1, 2:4, 2:4] = 2
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(np.zeros((2, 5, 5), dtype=np.float32))},
        objects={
            "Parents": ObjectLabelSet(name="Parents", labels=parent_labels),
            "Children": ObjectLabelSet(name="Children", labels=child_labels),
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="RelateObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Parents_Children_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Parents_Children_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
            ),
        )
    )
    payload = ParentChildRelationshipPayload(
        parent_ids=(1, 2),
        child_ids=(1, 2),
        slice_indices=(0, 1),
        slice_count=2,
    )
    request = _cellprofiler_output_record_request(
        executor=executor,
        adapter=runtime,
        spec=executor.outputs[0],
        output_value=payload,
        output_values={executor.outputs[0].name: payload},
        source_image_name=None,
        func=lambda image: image,
        call_kwargs={},
    )

    rows = RelationshipMeasurementRows.for_request(request).rows()
    slice_indices = {int(row["slice_index"]) for row in rows if "slice_index" in row}

    assert slice_indices == {0, 1}
    assert all(int(row["slice_index"]) in {0, 1} for row in rows)


def test_parent_child_relationship_payload_slices_with_pure_2d_kwargs() -> None:
    payload = ParentChildRelationshipPayload(
        parent_ids=(1, 2, 3, 4),
        child_ids=(10, 20, 30, 40),
        slice_indices=(0, 1, 0, 1),
        slice_count=2,
    )

    sliced = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimeProjectionAxis(slice_index=1, extent=2),
    )

    assert sliced == ParentChildRelationshipPayload(
        parent_ids=(2, 4),
        child_ids=(20, 40),
        slice_count=1,
    )


def test_object_relationship_slices_with_pure_2d_kwargs() -> None:
    semantics = RelationshipSemantics.parent_child("Parents", "Children")
    relationship = ObjectRelationship(
        name="Parents_Children_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 2, 3, 4),
        target_ids=(10, 20, 30, 40),
        relationship_type=semantics.relationship_type,
        slice_indices=(0, 1, 0, 1),
        slice_count=2,
    )

    sliced = RuntimeSliceProjection.value_for_slice(
        relationship,
        RuntimeProjectionAxis(slice_index=1, extent=2),
    )

    assert isinstance(sliced, ObjectRelationship)
    assert sliced.source_ids == (2, 4)
    assert sliced.target_ids == (20, 40)
    assert sliced.slice_count == 1


def test_relationship_measurements_broadcast_singleton_label_counts() -> None:
    parent_labels = np.zeros((2, 5, 5), dtype=np.int32)
    child_labels = np.zeros((1, 5, 5), dtype=np.int32)
    parent_labels[:, 1:3, 1:3] = 1
    child_labels[0, 2:4, 2:4] = 1
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(np.zeros((2, 5, 5), dtype=np.float32))},
        objects={
            "Parents": ObjectLabelSet(name="Parents", labels=parent_labels),
            "Children": ObjectLabelSet(name="Children", labels=child_labels),
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="RelateObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Parents_Children_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Parents_Children_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
            ),
        )
    )
    payload = ParentChildRelationshipPayload(
        parent_ids=(1, 1),
        child_ids=(1, 1),
        slice_indices=(0, 1),
        slice_count=2,
    )
    request = _cellprofiler_output_record_request(
        executor=executor,
        adapter=runtime,
        spec=executor.outputs[0],
        output_value=payload,
        output_values={executor.outputs[0].name: payload},
        source_image_name=None,
        func=lambda image: image,
        call_kwargs={},
    )

    rows = RelationshipMeasurementRows.for_request(request).rows()

    parent_rows = [
        row
        for row in rows
        if row.get("object_name") == "Children"
        and RelationshipMeasurementRows.parent_feature_name("Parents")
        in row
    ]
    assert {(row["slice_index"], row["object_label"]) for row in parent_rows} == {
        (0, 1),
        (1, 1),
    }


def test_relationship_rows_use_declared_relationship_outputs_for_measurement_only_recording() -> (
    None
):
    parent_labels = np.zeros((5, 5), dtype=np.int32)
    child_labels = np.zeros((5, 5), dtype=np.int32)
    parent_labels[1:4, 1:4] = 1
    child_labels[2:3, 2:3] = 1
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(np.zeros((5, 5), dtype=np.float32))},
        objects={
            "Parents": ObjectLabelSet(name="Parents", labels=parent_labels),
            "Children": ObjectLabelSet(name="Children", labels=child_labels),
        },
    )
    relationship_spec = ArtifactSpec.output(
        "Parents_Children_relationships",
        RelationshipsArtifactType,
    )
    measurement_spec = ArtifactSpec.output(
        "RelateObjects_measurements",
        MeasurementsArtifactType,
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="RelateObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition, (measurement_spec,)
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (relationship_spec, measurement_spec),
                ),
            ),
        )
    )
    payload = ParentChildRelationshipPayload(parent_ids=(1,), child_ids=(1,))
    request = _cellprofiler_output_record_request(
        executor=executor,
        adapter=runtime,
        spec=measurement_spec,
        output_value=RelationshipMeasurements(
            slice_index=0,
            parent_object_count=1,
            child_object_count=1,
            children_with_parents_count=1,
            mean_children_per_parent=1.0,
            mean_centroid_distance=0.0,
            mean_minimum_distance=1.0,
        ),
        output_values={relationship_spec.name: payload},
        source_image_name=None,
        func=lambda image: image,
        call_kwargs={},
    )

    rows = RelationshipMeasurementRows.for_request(request).rows()

    assert any(
        row.get("object_name") == "Parents" and row.get("Children_Children_Count") == 1
        for row in rows
    )


def test_relationship_rows_do_not_slice_payload_scoped_3d_lineage_by_z_plane() -> None:
    parent_labels = np.zeros((3, 5, 5), dtype=np.int32)
    child_labels = np.zeros((3, 5, 5), dtype=np.int32)
    parent_labels[0, 1:3, 1:3] = 1
    parent_labels[1, 2:4, 2:4] = 2
    child_labels[0, 1:3, 1:3] = 1
    child_labels[1, 2:4, 2:4] = 2
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(np.zeros((3, 5, 5), dtype=np.float32))},
        objects={
            "Parents": ObjectLabelPayload(
                labels=parent_labels,
                domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
            ),
            "Children": ObjectLabelPayload(
                labels=child_labels,
                domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
            ),
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="ResizeObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("Parents", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Parents", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Parents_Children_relationships", RelationshipsArtifactType
                        ),
                        ArtifactSpec.output("Children", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Parents_Children_relationships", RelationshipsArtifactType
                        ),
                        ArtifactSpec.output("Children", ObjectLabelsArtifactType),
                    ),
                ),
            ),
        )
    )
    payload = ParentChildRelationshipPayload(parent_ids=(1, 2), child_ids=(1, 2))
    request = _cellprofiler_output_record_request(
        executor=executor,
        adapter=runtime,
        spec=executor.outputs[0],
        output_value=payload,
        output_values={
            executor.outputs[0].name: payload,
            "Children": runtime.runtime_objects["Children"],
        },
        source_image_name=None,
        func=lambda image: image,
        call_kwargs={},
    )

    rows = RelationshipMeasurementRows.for_request(request).rows()

    assert all("slice_index" not in row for row in rows)
    parent_rows = [
        row
        for row in rows
        if row.get("object_name") == "Children"
        and RelationshipMeasurementRows.parent_feature_name("Parents")
        in row
    ]
    assert [row["Parent_Parents"] for row in parent_rows] == [1, 2]


def test_relateobjects_relationship_rows_project_distances_nominally() -> None:
    parent_labels = np.zeros((6, 6), dtype=np.int32)
    child_labels = np.zeros((6, 6), dtype=np.int32)
    parent_labels[1:5, 1:5] = 1
    child_labels[2:4, 2:4] = 1
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(np.zeros((6, 6), dtype=np.float32))},
        objects={
            "Parents": ObjectLabelSet(name="Parents", labels=parent_labels),
            "Children": ObjectLabelSet(name="Children", labels=child_labels),
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="RelateObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Parents_Children_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Parents_Children_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
            ),
        )
    )
    payload = ParentChildRelationshipPayload(parent_ids=(1,), child_ids=(1,))
    request = _cellprofiler_output_record_request(
        executor=executor,
        adapter=runtime,
        spec=executor.outputs[0],
        output_value=RelationshipMeasurements(
            slice_index=0,
            parent_object_count=1,
            child_object_count=1,
            children_with_parents_count=1,
            mean_children_per_parent=1.0,
            mean_centroid_distance=0.0,
            mean_minimum_distance=1.0,
        ),
        output_values={executor.outputs[0].name: payload},
        source_image_name=None,
        func=lambda image: image,
        call_kwargs={},
    )

    rows = RelationshipMeasurementRows.for_request(request).rows()

    distance_rows = [
        row
        for row in rows
        if row.get("object_name") == "Children" and "Distance_Centroid_Parents" in row
    ]
    assert len(distance_rows) == 1
    assert distance_rows[0]["Distance_Centroid_Parents"] == pytest.approx(0.0)
    assert distance_rows[0]["Distance_Minimum_Parents"] == pytest.approx(np.sqrt(2.5))


def test_relateobjects_relationship_rows_project_parent_mean_distances() -> None:
    parent_labels = np.zeros((8, 8), dtype=np.int32)
    child_labels = np.zeros((8, 8), dtype=np.int32)
    parent_labels[1:7, 1:7] = 1
    child_labels[2:4, 2:4] = 1
    child_labels[5:7, 5:7] = 2
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(np.zeros((8, 8), dtype=np.float32))},
        objects={
            "Parents": ObjectLabelSet(name="Parents", labels=parent_labels),
            "Children": ObjectLabelSet(name="Children", labels=child_labels),
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="RelateObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Parents_Children_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Parents_Children_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
            ),
        )
    )
    payload = ParentChildRelationshipPayload(parent_ids=(1, 1), child_ids=(1, 2))
    request = _cellprofiler_output_record_request(
        executor=executor,
        adapter=runtime,
        spec=executor.outputs[0],
        output_value=RelationshipMeasurements(
            slice_index=0,
            parent_object_count=1,
            child_object_count=2,
            children_with_parents_count=2,
            mean_children_per_parent=2.0,
            mean_centroid_distance=1.0,
            mean_minimum_distance=1.0,
        ),
        output_values={executor.outputs[0].name: payload},
        source_image_name=None,
        func=lambda image: image,
        call_kwargs={"calculate_per_parent_means": True},
    )

    rows = RelationshipMeasurementRows.for_request(request).rows()

    mean_rows = [
        row
        for row in rows
        if row.get("object_name") == "Parents"
        and "Mean_Children_Distance_Centroid" in row
    ]
    assert len(mean_rows) == 1
    assert mean_rows[0]["Mean_Children_Distance_Centroid"] == pytest.approx(
        np.mean(
            [
                row["Distance_Centroid_Parents"]
                for row in rows
                if row.get("object_name") == "Children"
                and "Distance_Centroid_Parents" in row
            ]
        )
    )
    assert "Mean_Children_Distance_Minimum" in mean_rows[0]


def test_relateobjects_relationship_rows_project_distances_from_slice_measurements() -> (
    None
):
    parent_labels = np.zeros((2, 6, 6), dtype=np.int32)
    child_labels = np.zeros((2, 6, 6), dtype=np.int32)
    parent_labels[:, 1:5, 1:5] = 1
    child_labels[:, 2:4, 2:4] = 1
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(np.zeros((2, 6, 6), dtype=np.float32))},
        objects={
            "Parents": ObjectLabelSet(name="Parents", labels=parent_labels),
            "Children": ObjectLabelSet(name="Children", labels=child_labels),
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="RelateObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("Parents", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Children", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Parents_Children_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (
                        ArtifactSpec.output(
                            "Parents_Children_relationships",
                            RelationshipsArtifactType,
                        ),
                    ),
                ),
            ),
        )
    )
    payload = ParentChildRelationshipPayload(
        parent_ids=(1, 1),
        child_ids=(1, 1),
        slice_indices=(0, 1),
        slice_count=2,
    )
    request = _cellprofiler_output_record_request(
        executor=executor,
        adapter=runtime,
        spec=executor.outputs[0],
        output_value=[
            RelationshipMeasurements(
                slice_index=0,
                parent_object_count=1,
                child_object_count=1,
                children_with_parents_count=1,
                mean_children_per_parent=1.0,
                mean_centroid_distance=0.0,
                mean_minimum_distance=1.0,
            ),
            RelationshipMeasurements(
                slice_index=1,
                parent_object_count=1,
                child_object_count=1,
                children_with_parents_count=1,
                mean_children_per_parent=1.0,
                mean_centroid_distance=0.0,
                mean_minimum_distance=1.0,
            ),
        ],
        output_values={executor.outputs[0].name: payload},
        source_image_name=None,
        func=lambda image: image,
        call_kwargs={},
    )

    rows = RelationshipMeasurementRows.for_request(request).rows()

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
        labels=SparseIJVLabelRows(
            np.asarray(
                (
                    (1, 1, 1),
                    (2, 2, 2),
                ),
                dtype=np.int32,
            )
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    child_sparse = ObjectLabelSet(
        name="Children",
        labels=SparseIJVLabelRows(
            np.asarray(
                (
                    (1, 1, 7),
                    (2, 2, 8),
                    (7, 7, 9),
                ),
                dtype=np.int32,
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

    assert dense_payload == ParentChildRelationshipPayload(
        parent_ids=(),
        child_ids=(),
    )
    assert sparse_payload == ParentChildRelationshipPayload(
        parent_ids=(1, 2),
        child_ids=(7, 8),
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

    assert grid.x_location_of_lowest_x_spot == 4.0
    assert grid.y_location_of_lowest_y_spot == 3.0
    assert grid.x_spacing == 10.0
    assert grid.y_spacing == 10.0


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
        grid=grid,
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
        grid=grid,
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

    _image, _stats, payload = identify_objects_in_grid_with_guides(
        image,
        guide_labels,
        grid=grid,
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

    _image, _stats, payload = identify_objects_in_grid_with_guides(
        image,
        guide_labels,
        grid=grid,
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

    _image, stats, payload = identify_objects_in_grid_with_guides(
        image,
        guide_labels,
        grid=grid,
        shape_choice="natural_shape_and_location",
        dtype_config=DtypeConfig(),
    )
    labels = np.asarray(payload.labels)
    assert not np.any(labels == 1)
    assert np.any(labels == 2)

    record = measurement_record_for_module(
        _cellprofiler_output_record_request(
            module_name="IdentifyObjectsInGrid",
            inputs=(
                ArtifactSpec.input("Grid", SpatialGridArtifactType),
                ArtifactSpec.input("Guides", ObjectLabelsArtifactType),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec.input("Grid", SpatialGridArtifactType),
                ArtifactSpec.input("Guides", ObjectLabelsArtifactType),
            ),
            outputs=(
                ArtifactSpec.output(
                    "IdentifyObjectsInGrid_1_measurements",
                    MeasurementsArtifactType,
                ),
                ArtifactSpec.output("GridObjects", ObjectLabelsArtifactType),
            ),
            adapter=None,
            spec=ArtifactSpec.output(
                "IdentifyObjectsInGrid_1_measurements",
                MeasurementsArtifactType,
            ),
            output_value=stats,
            output_values={"GridObjects": payload},
            source_image_name=None,
            func=identify_objects_in_grid_with_guides,
            call_kwargs={
                "grid": grid,
                "guiding_labels": guide_labels,
                "shape_choice": "natural_shape_and_location",
            },
        )
    )

    by_key = {
        (
            row["object_label"],
            row["feature_name"],
        ): row["result_value"]
        for row in record.rows
        if isinstance(row, dict)
        and row.get("feature_name")
        in {
            ObjectLocationMeasurementFeature.CENTER_X.value,
            ObjectLocationMeasurementFeature.CENTER_Y.value,
        }
    }
    assert np.isnan(by_key[(1, ObjectLocationMeasurementFeature.CENTER_X.value)])
    assert np.isnan(by_key[(1, ObjectLocationMeasurementFeature.CENTER_Y.value)])
    assert by_key[(2, ObjectLocationMeasurementFeature.CENTER_X.value)] == 6.5
    assert by_key[(2, ObjectLocationMeasurementFeature.CENTER_Y.value)] == 2.0


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
        identify_objects_in_grid_with_guides(
            image,
            guides,
            grid=grid,
            shape_choice="natural_shape_and_location",
            dtype_config=DtypeConfig(),
        )
        for guides in (first_guides, second_guides)
    )
    payload = CellProfilerPure2DOutputAggregator.aggregate(
        tuple(output[2] for output in outputs),
        MemoryType.NUMPY.value,
    )

    record = measurement_record_for_module(
        _cellprofiler_output_record_request(
            module_name="IdentifyObjectsInGrid",
            inputs=(
                ArtifactSpec.input("Grid", SpatialGridArtifactType),
                ArtifactSpec.input("Guides", ObjectLabelsArtifactType),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec.input("Grid", SpatialGridArtifactType),
                ArtifactSpec.input("Guides", ObjectLabelsArtifactType),
            ),
            outputs=(
                ArtifactSpec.output(
                    "IdentifyObjectsInGrid_1_measurements",
                    MeasurementsArtifactType,
                ),
                ArtifactSpec.output("GridObjects", ObjectLabelsArtifactType),
            ),
            adapter=None,
            spec=ArtifactSpec.output(
                "IdentifyObjectsInGrid_1_measurements",
                MeasurementsArtifactType,
            ),
            output_value=RuntimeSliceAlignedValues(
                tuple(output[1] for output in outputs)
            ),
            output_values={"GridObjects": payload},
            source_image_name=None,
            func=identify_objects_in_grid_with_guides,
            call_kwargs={
                "grid": RuntimeSliceAlignedValues((grid, grid)),
                "guiding_labels": np.stack((first_guides, second_guides), axis=0),
                "shape_choice": "natural_shape_and_location",
            },
        )
    )

    by_key = {
        (
            row["slice_index"],
            row["object_label"],
            row["feature_name"],
        ): row
        for row in record.rows
        if isinstance(row, dict)
    }
    assert (
        by_key[(0, 1, ObjectLocationMeasurementFeature.CENTER_X.value)]["result_value"]
        == 1.5
    )
    assert (
        by_key[(0, 1, ObjectLocationMeasurementFeature.CENTER_Y.value)]["result_value"]
        == 2.0
    )
    assert np.isnan(
        by_key[(0, 2, ObjectLocationMeasurementFeature.CENTER_X.value)]["result_value"]
    )
    assert np.isnan(
        by_key[(0, 3, ObjectLocationMeasurementFeature.CENTER_X.value)]["result_value"]
    )
    assert np.isnan(
        by_key[(1, 1, ObjectLocationMeasurementFeature.CENTER_X.value)]["result_value"]
    )
    assert (
        by_key[(1, 2, ObjectLocationMeasurementFeature.CENTER_X.value)]["result_value"]
        == 6.5
    )
    assert (
        by_key[(1, 2, ObjectLocationMeasurementFeature.CENTER_Y.value)]["result_value"]
        == 3.0
    )


def test_object_location_measurements_preserve_declared_empty_grid_cells() -> None:
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 3:5, 3:5] = 2
    payload = ObjectLabelPayload(
        labels=labels, domain=ObjectLabelDomain(declared_object_count=3)
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
    assert by_key[(0, 1, ObjectLocationMeasurementFeature.CENTER_X.value)] == 1.5
    assert by_key[(0, 1, ObjectLocationMeasurementFeature.CENTER_Y.value)] == 1.5
    assert np.isnan(by_key[(0, 2, ObjectLocationMeasurementFeature.CENTER_X.value)])
    assert np.isnan(by_key[(0, 3, ObjectLocationMeasurementFeature.CENTER_Y.value)])
    assert np.isnan(by_key[(1, 1, ObjectLocationMeasurementFeature.CENTER_X.value)])
    assert by_key[(1, 2, ObjectLocationMeasurementFeature.CENTER_X.value)] == 3.5
    assert by_key[(1, 2, ObjectLocationMeasurementFeature.CENTER_Y.value)] == 3.5
    assert np.isnan(by_key[(1, 3, ObjectLocationMeasurementFeature.CENTER_Y.value)])


def test_object_location_measurements_use_payload_domain_for_full_stack_labels() -> (
    None
):
    labels = np.zeros((3, 5, 5), dtype=np.int32)
    labels[1, 1:3, 1:3] = 1
    labels[2, 3:5, 3:5] = 2
    payload = ObjectLabelPayload(
        labels=labels,
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
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
    assert by_key[(0, 1, ObjectLocationMeasurementFeature.CENTER_Z.value)] == 1.0
    assert by_key[(0, 2, ObjectLocationMeasurementFeature.CENTER_Z.value)] == 2.0


def test_object_location_measurements_collapse_repeated_diagonal_plane_domain() -> None:
    plane = np.array([[1, 0], [0, 2]], dtype=np.int32)
    labels = np.zeros((2, 2, 2, 2), dtype=np.int32)
    labels[0, 0] = plane
    labels[1, 1] = plane
    payload = ObjectLabelPayload(
        labels=labels,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (1, 2)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    rows = ObjectLocationMeasurementRows(
        payload,
        object_name="GridObjects",
    ).rows()

    assert len(rows) == 4
    by_key = {
        (
            row["slice_index"],
            row["object_label"],
            row["feature_name"],
        ): row["result_value"]
        for row in rows
    }
    assert by_key[(0, 1, ObjectLocationMeasurementFeature.CENTER_X.value)] == 0.0
    assert by_key[(0, 2, ObjectLocationMeasurementFeature.CENTER_X.value)] == 1.0
    assert by_key[(0, 1, ObjectLocationMeasurementFeature.CENTER_Y.value)] == 0.0
    assert by_key[(0, 2, ObjectLocationMeasurementFeature.CENTER_Y.value)] == 1.0


def test_object_location_measurements_collapse_repeated_homogeneous_plane_domain() -> (
    None
):
    plane = np.array([[1, 0], [0, 2]], dtype=np.int32)
    payload = ObjectLabelPayload(
        labels=np.stack((plane, plane), axis=0),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (1, 2)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    rows = ObjectLocationMeasurementRows(
        payload,
        object_name="GridObjects",
    ).rows()

    assert len(rows) == 4
    assert {row["slice_index"] for row in rows} == {0}


def test_sparse_object_label_aggregation_preserves_declared_domain() -> None:
    first = ObjectLabelSet(
        name="GridObjects",
        labels=SparseIJVLabelRows.from_yx_label(
            np.asarray([[0, 0, 1], [1, 1, 3]], dtype=np.int32)
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=ObjectLabelDomain(
            declared_object_count=4,
        ),
    )
    second = ObjectLabelSet(
        name="GridObjects",
        labels=SparseIJVLabelRows.from_yx_label(
            np.asarray([[0, 0, 2], [1, 1, 4]], dtype=np.int32)
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=ObjectLabelDomain(
            declared_object_count=4,
        ),
    )

    aggregated = CellProfilerPure2DOutputAggregator.aggregate(
        [first, second],
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, ObjectLabelSet)
    assert aggregated.domain.declared_object_count == 4
    assert isinstance(aggregated.labels, SparseIJVLabelRows)


def test_measurement_table_collection_slice_count_accepts_sharded_offsets() -> None:
    first = MeasurementTable(
        name="ObjectMeasurements",
        rows=[{"slice_index": 0, "object_label": 1, "area": 11.0}],
        object_name="Objects",
    )
    second = MeasurementTable(
        name="ObjectMeasurements",
        rows=[{"slice_index": 1, "object_label": 1, "area": 13.0}],
        object_name="Objects",
    )

    assert RuntimeSliceProjection.slice_count_from_values((first, second)) == 2


def test_measurement_table_slice_count_ignores_sparse_row_axis_values() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=[
            {"slice_index": 0, "object_label": 1, "area": 11.0},
            {"slice_index": 18, "object_label": 2, "area": 13.0},
        ],
        object_name="Objects",
    )

    assert RuntimeSliceProjection.measurement_table_slice_count(table) is None
    assert RuntimeSliceProjection.slice_count_from_values((table,)) is None


def test_measurement_table_slice_count_accepts_columnar_rows() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=_ColumnarMeasurementRows(
            {
                "slice_index": (0, 0, 1),
                "object_label": (1, 2, 1),
                "area": (11.0, 12.0, 13.0),
            }
        ),
        object_name="Objects",
    )

    assert RuntimeSliceProjection.measurement_table_slice_count(table) == 2


def test_measurement_table_for_slice_preserves_columnar_rows() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=_ColumnarMeasurementRows(
            {
                "slice_index": (0, 0, 1),
                "feature_name": ("Area", "MeanIntensity", "Area"),
                "result_value": (11.0, 12.0, 13.0),
            }
        ),
        fields=(
            FieldSpec("slice_index"),
            FieldSpec("feature_name"),
            FieldSpec("result_value"),
        ),
        object_name="Objects",
    )

    sliced = MeasurementTableAxisProjection(
        axis=MeasurementRowAxisField.SLICE_INDEX,
        value=1,
        table=table,
    ).apply()

    assert isinstance(sliced.rows, ColumnarRows)
    assert tuple(sliced.rows.columns["slice_index"]) == (1,)
    assert tuple(sliced.rows.columns["result_value"]) == (13.0,)


def test_measurement_table_for_slice_normalizes_mixed_sequence_rows() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=[
            {"slice_index": 1, "feature_name": "Area", "result_value": 13.0},
            {"slice_index": 1, "MeanIntensity": 17.0},
        ],
        object_name="Objects",
    )

    sliced = MeasurementTableAxisProjection(
        axis=MeasurementRowAxisField.SLICE_INDEX,
        value=1,
        table=table,
    ).apply()

    assert sliced.fields == ()
    assert all("feature_name" in row for row in sliced.rows)
    assert all("result_value" in row for row in sliced.rows)


def test_unstack_cellprofiler_image_slices_collapses_pairwise_slice_grid() -> None:
    pairwise = np.zeros((2, 2, 5, 6), dtype=np.float32)
    pairwise[0, 0] = 1.0
    pairwise[0, 1] = 2.0
    pairwise[1, 0] = 3.0
    pairwise[1, 1] = 4.0

    slices = _unstack_cellprofiler_image_slices(pairwise, MemoryType.NUMPY.value)

    assert len(slices) == 2
    np.testing.assert_array_equal(image_payload_data(slices[0]), np.full((5, 6), 1.0))
    np.testing.assert_array_equal(image_payload_data(slices[1]), np.full((5, 6), 4.0))


def test_unstack_cellprofiler_image_slices_projects_singleton_volume_stack_mask() -> (
    None
):
    data = np.arange(1 * 3 * 4 * 5, dtype=np.float32).reshape(1, 3, 4, 5)
    mask = np.ones((1, 3, 4, 5), dtype=bool)
    mask[0, 1] = False
    payload = RuntimeImagePayloadContext(
        data, mask=mask, metadata=ImagePayloadMetadata()
    ).payload()

    slices = _unstack_cellprofiler_image_slices(payload, MemoryType.NUMPY.value)

    assert len(slices) == 3
    np.testing.assert_array_equal(image_payload_data(slices[1]), data[0, 1])
    np.testing.assert_array_equal(image_payload_data(slices[2]), data[0, 2])
    np.testing.assert_array_equal(image_payload_mask(slices[1]), mask[0, 1])
    np.testing.assert_array_equal(image_payload_mask(slices[2]), mask[0, 2])


def test_unstack_cellprofiler_image_slices_projects_high_rank_plane_mask() -> None:
    data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    mask = np.ones((2, 3, 4, 5), dtype=bool)
    mask[1, 2] = False
    payload = RuntimeImagePayloadContext(
        data, mask=mask, metadata=ImagePayloadMetadata()
    ).payload()

    slices = _unstack_cellprofiler_image_slices(payload, MemoryType.NUMPY.value)

    assert len(slices) == 6
    np.testing.assert_array_equal(image_payload_data(slices[5]), data[1, 2])
    np.testing.assert_array_equal(image_payload_mask(slices[5]), mask[1, 2])


def test_unstack_cellprofiler_image_slices_projects_source_axis_mask_to_plane() -> None:
    data_slice = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    mask = np.ones((2, 4, 5), dtype=bool)
    mask[1] = False

    payload = ImagePayloadSliceProjector(
        mask=mask,
        metadata=ImagePayloadMetadata(),
    ).payload_for_slice(data_slice, 2)

    np.testing.assert_array_equal(
        image_payload_mask(payload),
        np.zeros((4, 5), dtype=bool),
    )


def test_cellprofiler_contract_executor_rejects_uncoerced_unknown_absorbed_contract():
    def two_dimensional_only(image: np.ndarray, **kwargs) -> np.ndarray:
        if image.ndim != 2:
            raise RuntimeError("2D only")
        return image

    attach_callable_contract_metadata(
        two_dimensional_only,
        declared_processing_contract="unknown",
    )

    with pytest.raises(TypeError, match="no nominal __processing_contract__"):
        CellProfilerProcessingContractAuthority.for_callable(two_dimensional_only)


def test_measurement_image_for_labels_preserves_source_stack_for_2d_labels() -> None:
    image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    labels = np.ones((4, 5), dtype=np.int32)

    measurement_image = _measurement_image_for_labels(image, labels)

    assert measurement_image is image


def test_measurement_image_for_labels_preserves_object_domain_stack_for_2d_labels() -> (
    None
):
    image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    labels = np.ones((4, 5), dtype=np.int32)

    measurement_image = _measurement_image_for_labels(
        image,
        labels,
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    assert measurement_image is image


def test_measurement_image_for_labels_uses_object_domain_reference_shape() -> None:
    image = np.ones((10, 12), dtype=np.float32)
    labels = np.ones((4, 5), dtype=np.int32)

    measurement_image = _measurement_image_for_labels(
        image,
        labels,
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    assert measurement_image.shape == labels.shape
    assert measurement_image.dtype == image.dtype
    np.testing.assert_array_equal(
        measurement_image, np.zeros_like(labels, dtype=image.dtype)
    )


def test_measurement_image_for_labels_keeps_source_domain_shape_mismatch() -> None:
    image = np.ones((10, 12), dtype=np.float32)
    labels = np.ones((4, 5), dtype=np.int32)

    measurement_image = _measurement_image_for_labels(image, labels)

    assert measurement_image is image


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
    labels = np.stack(
        (
            np.full((3, 4, 5), 1, dtype=np.int32),
            np.full((3, 4, 5), 2, dtype=np.int32),
        )
    )
    projector = Projector()
    measurement_image = CellProfilerMeasurementImage(
        source_image_name=None,
        source_aliases=("origMemb",),
        payload=image,
    )
    aligned_labels = (
        MeasurementLabelSourceAlignmentStrategy.align_request_labels_to_image_source(
            measurement_image.alignment_request(
                labels=labels,
                plane_projector=projector,
            )
        )
    )

    np.testing.assert_array_equal(aligned_labels, labels[1])
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
        metadata=ImagePayloadMetadata(source_image_names=("origDNA", "origMemb")),
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
    assert image_payload_metadata(aligned).source_image_names == ("origMemb",)


def test_measurement_images_from_image_request_preserve_execution_mode() -> None:
    first_site = np.stack(
        (
            np.full((4, 5), 11, dtype=np.float32),
            np.full((4, 5), 22, dtype=np.float32),
        )
    )
    second_site = first_site + 100
    image_request = CellProfilerImageRequest(
        source_image_name=None,
        source_aliases=("origDNA", "origMemb"),
        payload=AlignedImageStack((first_site, second_site)),
        image_count=2,
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
    )

    images = CellProfilerMeasurementImageResolver(
        SimpleNamespace()
    ).measurement_images_from_image_request(
        image_request,
        (
            ArtifactSpec.input("origDNA", ImageArtifactType),
            ArtifactSpec.input("origMemb", ImageArtifactType),
        ),
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    assert images is not None
    assert tuple(image.source_image_name for image in images) == ("origDNA", "origMemb")
    assert tuple(image.execution_mode for image in images) == (
        ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
    )
    np.testing.assert_array_equal(
        image_payload_data(images[0].payload),
        np.stack((first_site[0], second_site[0])),
    )
    np.testing.assert_array_equal(
        image_payload_data(images[1].payload),
        np.stack((first_site[1], second_site[1])),
    )


def test_measurement_domain_alignment_projects_source_owned_object_labels() -> None:
    class Projector(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return None

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            del source_aliases
            return None

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
        labels=np.stack(
            (
                np.full((4, 5), 1, dtype=np.int32),
                np.full((4, 5), 2, dtype=np.int32),
            )
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


def test_prepared_measurement_labels_project_runtime_slice_payload_with_dense_labels() -> None:
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
        labels=label_planes,
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
    assert prepared.source_projected_payload.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert prepared.source_projected_payload.object_label_domain().declared_object_ids == (7,)
    np.testing.assert_array_equal(
        object_label_dense_array(prepared.source_projected_payload),
        label_planes[1],
    )
    np.testing.assert_array_equal(prepared.source_projected_labels, label_planes[1])
    np.testing.assert_array_equal(
        object_label_dense_array(prepared.completion_payload),
        label_planes[1],
    )


def test_prepared_measurement_labels_project_payload_plane_by_image_source_identity() -> None:
    label_planes = np.stack(
        (
            np.full((4, 5), 3, dtype=np.int32),
            np.full((4, 5), 7, dtype=np.int32),
        )
    )
    labels = ObjectLabelPayload(
        labels=label_planes,
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

    prepared = measurement_image.prepare_object_labels(labels)

    assert prepared.source_projected_payload.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert prepared.source_projected_payload.object_label_domain().declared_object_ids == (7,)
    np.testing.assert_array_equal(
        object_label_dense_array(prepared.source_projected_payload),
        label_planes[1],
    )
    np.testing.assert_array_equal(prepared.source_projected_labels, label_planes[1])


def test_measurement_labels_collapse_singleton_label_stack() -> None:
    labels = np.ones((1, 4, 5), dtype=np.int32)

    measurement_labels = SingletonObjectLabelStackCollapseStrategy.for_labels(
        labels
    ).collapse(labels)

    assert measurement_labels.shape == (4, 5)
    np.testing.assert_array_equal(measurement_labels, labels[0])


def test_measurement_labels_preserve_stack_for_object_domain_alignment() -> None:
    image = np.ones((1, 4, 5), dtype=np.float32)
    labels = np.arange(2 * 4 * 5, dtype=np.int32).reshape(2, 4, 5)

    measurement_labels = MeasurementLabelSourceAlignmentStrategy.align(image, labels)

    assert measurement_labels.shape == labels.shape
    np.testing.assert_array_equal(measurement_labels, labels)


def test_measurement_label_alignment_preserves_runtime_slice_payload_for_aligned_stack() -> (
    None
):
    first_image = RuntimeImagePayloadContext(
        np.ones((2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(0, 0),
                source_shape_yx=(4, 4),
            ),
        ),
        mask=None,
    ).payload()
    second_image = RuntimeImagePayloadContext(
        np.ones((2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(2, 0),
                source_shape_yx=(4, 4),
            ),
        ),
        mask=None,
    ).payload()
    labels = np.stack(
        (
            np.full((4, 4), 1, dtype=np.int32),
            np.full((4, 4), 2, dtype=np.int32),
        )
    )
    label_payload = ObjectLabelPayload(
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
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
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
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
    reference_slice = RuntimeImagePayloadContext(
        np.ones((2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(2, 0),
                source_shape_yx=(4, 4),
            ),
        ),
        mask=None,
    ).payload()
    labels = np.stack(
        (
            np.full((4, 4), 1, dtype=np.int32),
            np.full((4, 4), 2, dtype=np.int32),
        )
    )
    label_payload = ObjectLabelPayload(
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(4, 4)),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    resolved = aligned_image_stack_kwargs(
        {"labels": label_payload},
        slice_index=1,
        slice_count=2,
        reference_payload=reference_slice,
    )

    np.testing.assert_array_equal(
        resolved["labels"],
        np.full((2, 2), 2, dtype=np.int32),
    )


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
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
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
    assert resolved["labels"].domain.scope is ObjectLabelDomainScope.PLANE
    assert resolved["labels"].plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_measurement_labels_collapse_channel_broadcast_label_stack() -> None:
    image = np.ones((2, 4, 5), dtype=np.float32)
    label_plane = np.arange(4 * 5, dtype=np.int32).reshape(4, 5)
    labels = np.stack((label_plane, label_plane))

    measurement_labels = MeasurementLabelSourceAlignmentStrategy.align(image, labels)

    assert measurement_labels.shape == label_plane.shape
    np.testing.assert_array_equal(measurement_labels, label_plane)


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
        labels=np.stack((dna_labels, gfp_labels)),
        source_image_name="rawDNA",
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawDNA__rawGFP",
        source_aliases=("rawDAPI", "rawGFP", "rawDNA"),
        payload=image,
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
    image = np.ones((2, 4, 5), dtype=np.float32)
    label_planes = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
            np.full((4, 5), 3, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=label_planes,
        source_image_name="rawDNA",
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawDAPI__rawGFP__rawDNA",
        source_aliases=("rawDAPI", "rawGFP", "rawDNA"),
        payload=image,
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


def test_measurement_labels_do_not_source_project_when_alias_count_mismatches_stack() -> (
    None
):
    image = np.ones((2, 4, 5), dtype=np.float32)
    label_planes = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=label_planes,
        source_image_name="rawFarRed",
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawDNA__rawGFP__rawFarRed",
        source_aliases=("rawDNA", "rawGFP", "rawFarRed"),
        payload=image,
    )

    measurement_labels = _prepared_measurement_labels(
        measurement_image,
        labels,
        adapter=_TestRuntimePlaneAxisProjector(
            source_binding_index=0,
            source_binding_size=3,
        ),
    )

    assert measurement_labels.shape == label_planes.shape
    np.testing.assert_array_equal(measurement_labels, label_planes)


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
        labels=label_planes,
        source_image_name="rawDNA",
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawGFP",
        source_aliases=("rawGFP",),
        payload=image,
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
        labels=label_planes,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
        ),
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
        labels=label_planes,
        source_image_name="BF_image",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
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

    label_planes = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
            np.full((4, 5), 30, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=label_planes,
        source_image_name="CropBlue",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="CropBlue__CropGreen",
        source_aliases=("CropBlue", "CropGreen"),
        payload=AlignedImageStack(
            (
                np.ones((2, 4, 5), dtype=np.float32),
                np.ones((2, 4, 5), dtype=np.float32),
                np.ones((2, 4, 5), dtype=np.float32),
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
        labels=label_planes,
        source_image_name="BF_image",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
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


def test_measurement_labels_preserve_runtime_stack_when_group_index_is_out_of_domain() -> (
    None
):
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
        labels=label_planes,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="OrigMito",
        source_aliases=("OrigMito",),
        payload=np.ones((4, 5), dtype=np.float32),
    )

    measurement_labels = _prepared_measurement_labels(
        measurement_image,
        labels,
        adapter=Adapter(),
    )

    assert measurement_labels.shape == label_planes.shape
    np.testing.assert_array_equal(measurement_labels, label_planes)


def test_measurement_labels_project_runtime_slice_stack_for_object_domain() -> None:
    label_planes = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        labels=label_planes,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
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
        labels=label_planes,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_name="BF_image",
        source_image_names=("BF_image", "MorphBf"),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="DF_image",
        source_aliases=("DF_image",),
        payload=np.ones((2, 4, 5), dtype=np.float32),
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
        labels=label_planes,
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("BF_image", "MorphBf"),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
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
        labels=label_planes,
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("BF_image", "MorphBf"),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
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


def test_measurement_labels_project_grouped_runtime_stack_for_object_domain() -> None:
    runtime_stack = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
        )
    )
    grouped_stack = np.stack((runtime_stack, runtime_stack))
    labels = ObjectLabelSet(
        name="Cells",
        labels=grouped_stack,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_names=("BF_image", "MorphBf"),
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
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

    assert measurement_labels.shape == runtime_stack.shape
    np.testing.assert_array_equal(measurement_labels, runtime_stack)


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
        labels=label_volume,
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
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.ones((4, 5), dtype=np.int32),
        source_image_name="rawDNA",
    )
    runtime = _FakeCellProfilerRuntime(
        {"rawDNA": _FakeRuntimeImage(image)},
        {"Nuclei": labels},
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectIntensity",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("rawDNA", ImageArtifactType),
                        ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Measurements", MeasurementsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Measurements", MeasurementsArtifactType),),
                ),
            ),
        )
    )

    payload = CellProfilerMeasurementImageResolver(executor).object_label_payload(
        ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
        runtime,
        image,
    )

    assert payload is labels
    assert payload.source_image_name == "rawDNA"


def test_runtime_object_label_payload_does_not_receive_measurement_image_scope() -> (
    None
):
    class RecordingRuntime(_FakeCellProfilerRuntime):
        recorded_current_image: object = None

        def get_objects(
            self,
            name: str,
            *,
            current_image: object | None = None,
        ) -> ObjectLabelSet:
            self.recorded_current_image = current_image
            return super().get_objects(name, current_image=current_image)

    image = np.ones((4, 5), dtype=np.float32)
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.stack(
            (
                np.ones((4, 5), dtype=np.int32),
                np.full((4, 5), 2, dtype=np.int32),
            )
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=({"well": "A01"}, {"well": "A01"})
        ),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
    )
    runtime = RecordingRuntime(
        {"rawDNA": _FakeRuntimeImage(image)},
        {"Nuclei": labels},
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectIntensity",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("rawDNA", ImageArtifactType),
                        ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Measurements", MeasurementsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Measurements", MeasurementsArtifactType),),
                ),
            ),
        )
    )

    payload = CellProfilerMeasurementImageResolver(executor).object_label_payload(
        ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
        runtime,
        RuntimeImagePayloadContext(
            image,
            mask=None,
            metadata=ImagePayloadMetadata(
                source_component_metadata={"well": "A01"},
            ),
        ).payload(),
    )

    assert payload is labels
    assert runtime.recorded_current_image is None


def test_object_label_payload_for_measurement_image_projects_source_spatial_crop() -> (
    None
):
    module_name = "MeasureObjectIntensity"
    image_name = "CropBlue"
    object_name = "Cytoplasm"
    image = RuntimeImagePayloadContext(
        np.ones((3, 4), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(2, 3),
                source_shape_yx=(8, 9),
            ),
        ),
        mask=None,
    ).payload()
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
        labels=rows,
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(0, 0),
            source_shape_yx=(8, 9),
        ),
        source_image_name=image_name,
    )
    runtime = _FakeCellProfilerRuntime(
        {image_name: _FakeRuntimeImage(image)},
        {object_name: labels},
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name=module_name,
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input(image_name, ImageArtifactType),
                        ArtifactSpec.input(object_name, ObjectLabelsArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input(object_name, ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Measurements", MeasurementsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Measurements", MeasurementsArtifactType),),
                ),
            ),
        )
    )

    resolver = CellProfilerMeasurementImageResolver(executor)
    payload = project_object_label_payload_for_measurement_image(
        CellProfilerMeasurementImage(
            source_image_name=image_name,
            source_aliases=(image_name,),
            payload=image,
        ),
        resolver.object_label_payload(
            ArtifactSpec.input(object_name, ObjectLabelsArtifactType),
            runtime,
            image,
        ),
        adapter=runtime,
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
    image = RuntimeImagePayloadContext(
        np.arange(25, dtype=np.float32).reshape(5, 5),
        metadata=ImagePayloadMetadata(
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(0, 0),
                source_shape_yx=(5, 5),
            ),
        ),
        mask=None,
    ).payload()
    label_payload = ObjectLabelPayload(
        labels=np.ones((2, 2), dtype=np.int32),
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


def test_image_request_source_name_uses_primary_images_not_object_inputs() -> None:
    image = np.ones((4, 5), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {"rawGFP": _FakeRuntimeImage(image)},
        {
            "Nuclei": ObjectLabelSet(
                name="Nuclei",
                labels=np.ones((4, 5), dtype=np.int32),
                source_image_name="rawDNA",
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="IdentifySecondaryObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (
                        ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                        ArtifactSpec.input("rawGFP", ImageArtifactType),
                    ),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("Cells", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("Cells", ObjectLabelsArtifactType),),
                ),
            ),
        )
    )

    passthrough = lambda image: image
    passthrough.__processing_contract__ = ProcessingContract.PURE_2D
    image_request = executor._image_request(
        executor.runtime_plan(passthrough),
        image,
        runtime,
    )

    assert image_request.source_image_name == "rawGFP"


def test_object_only_reference_image_uses_one_stack_plane() -> None:
    image = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)

    reference_image = OBJECT_ONLY_REFERENCE_IMAGE.reference_image(image)

    assert reference_image.shape == (4, 5)
    np.testing.assert_array_equal(reference_image, image[0])


def test_object_only_reference_image_collapses_high_rank_carrier_to_plane() -> None:
    image = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)

    reference_image = OBJECT_ONLY_REFERENCE_IMAGE.reference_image(image)

    assert reference_image.shape == (4, 5)
    np.testing.assert_array_equal(reference_image, image[0, 0])


def test_measurement_table_rows_wrap_scalar_measurement() -> None:
    row = {"mean_intensity": 1.5}

    measurement_rows = measurement_table_rows(row)

    assert measurement_rows == [row]


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
        outline_object_indices=(0, 1),
        dtype_config=DtypeConfig(),
    )

    (
        _output_image,
        stats,
        filtered_primary,
        filtered_cells,
        primary_relationship,
        cells_relationship,
        *_outline_outputs,
    ) = result
    primary_outline, cells_outline = _outline_outputs

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert object_label_dense_array(filtered_primary).max() == 1
    assert filtered_primary[3, 3] == 1
    assert object_label_dense_array(filtered_cells).max() == 1
    assert filtered_cells[3, 3] == 1
    assert filtered_cells[0, 0] == 0
    assert primary_relationship.parent_ids == (2,)
    assert primary_relationship.child_ids == (1,)
    assert cells_relationship.parent_ids == (11,)
    assert cells_relationship.child_ids == (1,)
    assert primary_outline.max() == 1
    assert cells_outline.max() == 1


def test_filterobjects_uses_named_measurement_feature_rules() -> None:
    image = np.zeros((5, 5), dtype=np.float32)
    primary = np.zeros((5, 5), dtype=np.int32)
    primary[1:3, 1:3] = 1
    primary[3:5, 3:5] = 2
    measurement_rows = [
        {"object_label": 1, "lower_quartile_intensity": 0.1},
        {"object_label": 2, "lower_quartile_intensity": 0.8},
    ]

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
            MeasurementTable(name="NucleiMeasurements", rows=measurement_rows),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_primary = result[:3]

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert filtered_primary[1, 1] == 0
    assert filtered_primary[3, 3] == 1


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
                rows=({"object_label": 1, "AreaShape_Area": 4.0},),
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_primary = result[:3]

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert filtered_primary[1, 1] == 0
    assert filtered_primary[3, 3] == 1


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

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert filtered_primary[1, 1] == 0
    assert filtered_primary[3, 3] == 1


def test_filterobjects_binds_measurements_to_sparse_object_label_ids() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    primary = np.zeros((6, 6), dtype=np.int32)
    primary[1:3, 1:3] = 3
    primary[3:5, 3:5] = 5
    measurement_rows = [
        {"object_label": 3, "lower_quartile_intensity": 0.1},
        {"object_label": 5, "lower_quartile_intensity": 0.8},
    ]

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
            MeasurementTable(name="NucleiMeasurements", rows=measurement_rows),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_primary = result[:3]

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert filtered_primary[1, 1] == 0
    assert filtered_primary[3, 3] == 1


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
                rows=[
                    {"object_label": 1, "AreaShape_Area": 10.0},
                    {"object_label": 2, "AreaShape_Area": 20.0},
                    {"object_label": 3, "AreaShape_Area": 40.0},
                    {"object_label": 4, "AreaShape_Area": 30.0},
                ],
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]

    assert stats.objects_pre_filter == 4
    assert stats.objects_post_filter == 2
    assert filtered_children[0, 0] == 0
    assert filtered_children[0, 3] == 1
    assert filtered_children[3, 0] == 2
    assert filtered_children[3, 3] == 0


def test_filterobjects_filters_by_children_count_relationship() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    nuclei = np.zeros((6, 6), dtype=np.int32)
    nuclei[0:2, 0:2] = 1
    nuclei[2:4, 2:4] = 2
    nuclei[4:6, 4:6] = 3
    semantics = RelationshipSemantics.parent_child("Nuclei", "PH3")
    relationship = ObjectRelationship(
        name="Nuclei_PH3_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 3),
        target_ids=(1, 2),
        relationship_type=semantics.relationship_type,
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

    assert stats.objects_pre_filter == 3
    assert stats.objects_post_filter == 2
    assert filtered_nuclei[0, 0] == 1
    assert filtered_nuclei[2, 2] == 0
    assert filtered_nuclei[4, 4] == 2


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
                rows=[
                    {"object_label": 1, "AreaShape_Area": 10.0},
                    {"object_label": 2, "AreaShape_Area": 10.0},
                ],
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]

    assert stats.objects_post_filter == 1
    assert filtered_children[0, 0] == 0
    assert filtered_children[4, 4] == 1


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
                rows=[
                    {"object_label": 1, "AreaShape_Area": 10.0},
                    {"object_label": 2, "AreaShape_Area": 10.0},
                ],
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]

    assert stats.objects_post_filter == 1
    assert filtered_children[0, 0] == 1
    assert filtered_children[4, 4] == 0


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
                rows=[
                    {"object_label": 1, "AreaShape_Area": 10.0},
                    {"object_label": 2, "AreaShape_Area": 40.0},
                ],
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]

    assert stats.objects_post_filter == 2
    assert filtered_children[0, 0] == 1
    assert filtered_children[4, 4] == 2


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
        parent_child_relationship=ParentChildRelationshipPayload(
            parent_ids=(1, 2),
            child_ids=(1, 2),
        ),
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=[
                    {"object_label": 1, "AreaShape_Area": 10.0},
                    {"object_label": 2, "AreaShape_Area": 20.0},
                ],
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]

    assert stats.objects_post_filter == 1
    assert filtered_children[1, 0] == 0
    assert filtered_children[1, 3] == 1


def test_filterobjects_most_overlap_can_use_relationship_payload() -> None:
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
        parent_child_relationship=ParentChildRelationshipPayload(
            parent_ids=(1, 2),
            child_ids=(1, 2),
        ),
        per_object_assignment=PerObjectAssignment.PARENT_WITH_MOST_OVERLAP,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=[
                    {"object_label": 1, "AreaShape_Area": 10.0},
                    {"object_label": 2, "AreaShape_Area": 20.0},
                ],
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]

    assert stats.objects_post_filter == 2
    assert filtered_children[1, 0] == 1
    assert filtered_children[1, 3] == 2


def test_filterobjects_aligns_enclosing_label_stack_to_child_plane() -> None:
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
        enclosing_object_labels=np.stack((parents, parents)),
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=[
                    {"object_label": 1, "AreaShape_Area": 10.0},
                    {"object_label": 2, "AreaShape_Area": 20.0},
                    {"object_label": 3, "AreaShape_Area": 40.0},
                    {"object_label": 4, "AreaShape_Area": 30.0},
                ],
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]

    assert stats.objects_post_filter == 2
    assert filtered_children[0, 3] == 1
    assert filtered_children[3, 0] == 2


def test_structuring_element_execution_policy_uses_full_stack_for_3d_footprint() -> (
    None
):
    policy = CellProfilerInvocationExecutionModePolicy.for_module("Closing")
    image = np.zeros((3, 5, 5), dtype=np.float32)

    mode = policy.execution_mode(
        default=ImagePayloadExecutionMode.NATURAL,
        image=image,
        kwargs={
            "structuring_element": StructuringElement.BALL,
            "size": 1,
        },
    )

    assert mode is ImagePayloadExecutionMode.FULL_STACK


def test_structuring_element_execution_policy_keeps_planewise_for_2d_footprint() -> (
    None
):
    policy = CellProfilerInvocationExecutionModePolicy.for_module("Closing")
    image = np.zeros((3, 5, 5), dtype=np.float32)

    mode = policy.execution_mode(
        default=ImagePayloadExecutionMode.NATURAL,
        image=image,
        kwargs={
            "structuring_element": StructuringElement.DISK,
            "size": 1,
        },
    )

    assert mode is ImagePayloadExecutionMode.NATURAL


def test_structuring_element_execution_policy_uses_object_label_kwargs_rank() -> None:
    policy = CellProfilerInvocationExecutionModePolicy.for_module("ErodeObjects")
    image = np.zeros((5, 5), dtype=np.float32)
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.zeros((3, 5, 5), dtype=np.int32),
    )

    mode = policy.execution_mode(
        default=ImagePayloadExecutionMode.NATURAL,
        image=image,
        kwargs={
            "labels": labels,
            "structuring_element": StructuringElement.BALL,
            "size": 1,
        },
    )

    assert mode is ImagePayloadExecutionMode.FULL_STACK


def test_object_measurement_execution_policy_uses_full_stack_for_3d_labels() -> None:
    policy = CellProfilerObjectMeasurementExecutionDomainPolicy.for_module(
        "MeasureObjectSizeShape"
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.zeros((3, 5, 5), dtype=np.int32),
    )

    mode = policy.execution_mode(
        measure_object_size_shape,
        labels,
        ImagePayloadExecutionMode.NATURAL,
    )

    assert mode is ImagePayloadExecutionMode.FULL_STACK


def test_object_measurement_execution_policy_uses_full_stack_for_source_bound_volume_labels() -> None:
    policy = CellProfilerObjectMeasurementExecutionDomainPolicy.for_module(
        "MeasureObjectIntensity"
    )
    labels = ObjectLabelPayload(
        labels=np.zeros((2, 3, 5, 5), dtype=np.int32),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    mode = policy.execution_mode(
        measure_object_intensity,
        labels,
        ImagePayloadExecutionMode.NATURAL,
    )

    assert mode is ImagePayloadExecutionMode.FULL_STACK


def test_object_measurement_execution_policy_keeps_payload_domain_labels_full_stack() -> None:
    policy = CellProfilerObjectMeasurementExecutionDomainPolicy.for_module(
        "MeasureObjectIntensity"
    )
    labels = ObjectLabelPayload(
        labels=np.zeros((3, 5, 5), dtype=np.int32),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
    )

    mode = policy.execution_mode(
        measure_object_intensity,
        labels,
        ImagePayloadExecutionMode.NATURAL,
        runtime_slice_count=3,
    )

    assert mode is ImagePayloadExecutionMode.FULL_STACK


def test_object_measurement_execution_policy_preserves_plane_domain_runtime_slice_labels() -> None:
    policy = CellProfilerObjectMeasurementExecutionDomainPolicy.for_module(
        "MeasureObjectIntensity"
    )
    labels = ObjectLabelPayload(
        labels=np.zeros((3, 5, 5), dtype=np.int32),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
    )

    mode = policy.execution_mode(
        measure_object_intensity,
        labels,
        ImagePayloadExecutionMode.NATURAL,
        runtime_slice_count=3,
    )

    assert mode is ImagePayloadExecutionMode.NATURAL


def test_full_stack_object_measurement_executor_preserves_volume_call() -> None:
    calls: list[tuple[int, ...]] = []

    @object_label_measurement_execution(ObjectLabelMeasurementExecution.FULL_STACK)
    def measure_volume(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
        del labels
        calls.append(tuple(int(axis) for axis in image.shape))
        return image

    image = np.zeros((3, 5, 7), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)

    result = CellProfilerFunctionContractExecutor().execute(
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

    result = CellProfilerFunctionContractExecutor().execute(
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
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
    )

    assert (
        CellProfilerProcessingContractAuthority.for_callable(convert_objects_to_image)
        is ProcessingContract.PURE_3D
    )

    result = CellProfilerFunctionContractExecutor().execute(
        convert_objects_to_image,
        np.zeros((5, 7), dtype=np.float32),
        {
            "labels": label_payload,
            "image_mode": "uint16",
        },
    )

    np.testing.assert_array_equal(image_payload_data(result), labels)


def test_object_measurement_execution_policy_uses_full_stack_for_single_runtime_slice_volume() -> (
    None
):
    policy = CellProfilerObjectMeasurementExecutionDomainPolicy.for_module(
        "MeasureObjectSizeShape"
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.zeros((3, 5, 5), dtype=np.int32),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    mode = policy.execution_mode(
        measure_object_size_shape,
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
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
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


def test_measure_object_size_shape_projected_plane_metadata_rows_are_single_slice() -> (
    None
):
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[1:4, 1:4] = 1
    payload = ObjectLabelPayload(
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (1,), (1,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    _image, rows = measure_object_size_shape(
        np.zeros((5, 5), dtype=np.float32),
        payload,
        calculate_advanced=False,
        calculate_zernikes=False,
    )

    assert [(row["object_label"], row["Area"]) for row in rows] == [(1, 9.0)]


def test_measure_object_size_shape_mismatched_plane_metadata_uses_label_planes() -> (
    None
):
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 0:2, 0:2] = 1
    labels[1, 0:3, 0:3] = 1
    payload = ObjectLabelPayload(
        labels=labels,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (1,), (1,), (1,)),
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


def test_measure_object_size_shape_collapses_repeated_diagonal_plane_domain() -> None:
    plane = np.zeros((5, 5), dtype=np.int32)
    plane[0:2, 0:2] = 1
    labels = np.zeros((2, 2, 5, 5), dtype=np.int32)
    labels[0, 0] = plane
    labels[1, 1] = plane
    payload = ObjectLabelPayload(
        labels=labels,
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
    ]


def test_measure_object_size_shape_payload_scoped_volume_rows_are_3d() -> None:
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[:, 1:4, 1:4] = 1
    payload = ObjectLabelPayload(
        labels=labels,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PAYLOAD,
        ),
    )
    label_set = ObjectLabelSet(
        name="Cells",
        labels=labels,
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
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
        labels=labels,
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

    assert [(row["slice_index"], row["object_label"], row["Volume"]) for row in rows] == [
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


def test_object_measurement_execution_policy_keeps_2d_labels_natural() -> None:
    policy = CellProfilerObjectMeasurementExecutionDomainPolicy.for_module(
        "MeasureObjectSizeShape"
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.zeros((5, 5), dtype=np.int32),
    )

    mode = policy.execution_mode(
        measure_object_size_shape,
        labels,
        ImagePayloadExecutionMode.NATURAL,
    )

    assert mode is ImagePayloadExecutionMode.NATURAL

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import numpy as np

from openhcs.constants.constants import AllComponents, MEMORY_TYPE_NUMPY
from openhcs.core.artifacts import (
    ArtifactMeasurementSubjectRelation,
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSidecarRole,
    ArtifactSpec,
    ArtifactSpecRef,
    GroupLineageSourceRelation,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
    MetadataArtifactType,
    SpecialArtifactType,
)
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
    RuntimeExecutionAxisScope,
)
from openhcs.core.component_set import ComponentSet
from openhcs.core.function_patterns import (
    CompiledFunctionGroup,
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
    MainFlowInputProjection,
    compile_function_pattern,
)
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_adapters import (
    RuntimeAdapterRequest,
    runtime_adapter,
)
from openhcs.core.memory import stack_runtime_slices, unstack_runtime_slices
from openhcs.core.pipeline.function_contracts import (
    artifact_inputs,
    artifact_outputs,
    composed_image_payload,
    special_outputs,
)
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.core.pipeline.path_planner import PathPlanner, PathPlannerArtifactStage
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    NamedSourceBinding,
    SourceBindingRuntimeContext,
)
from openhcs.core.source_load_plan import SourceLoadPlan
from openhcs.core.source_metadata import (
    SOURCE_PLANE_COUNT_FIELD,
    SOURCE_PLANE_INDEX_FIELD,
)
from openhcs.core.runtime_measurements import MeasurementScope, MeasurementSubject
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis, RuntimePlaneAxisValueProjection, RuntimePlaneProjection
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.steps.function_runtime import (
    ComponentArtifactPlans,
    FunctionCoreExecutor,
    FunctionOutputContextStrategy,
    FunctionRuntimeScope,
    ImageFunctionOutputContextStrategy,
    PatternGroupData,
    PatternGroupRuntime,
)
from openhcs.core.aligned_image_payload import (
    AlignedImageSliceContext,
    AlignedImageStack,
    pack_aligned_image_outputs,
    stack_image_payload_context,
    stack_image_payloads,
    unstack_image_payload_context,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadataCompositionMode,
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
    image_payload_mask,
    with_image_payload_data,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    ObjectLabelSet,
    ObjectLabelVariantData,
    ObjectLabelPayload,
    object_label_dense_array,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
)
from openhcs.core.source_image_provenance import (
    SourceImageProvenancePlanes,
)
from openhcs.processing.backends.assemblers.assemble_stack_cpu import assemble_stack_cpu
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


def passthrough(image):
    return image


def test_special_outputs_is_the_artifact_outputs_public_spelling() -> None:
    assert special_outputs is artifact_outputs

    @special_outputs(" measurements ", (" labels ", None))
    def analyze(image):
        return image

    assert tuple(
        spec.name
        for spec in vars(analyze)[FunctionContractAttribute.artifact_outputs]
    ) == ("measurements", "labels")


class MemoryBackend:
    def __init__(self):
        self._memory_store = {}


class FileManagerStub:
    def __init__(self):
        self.memory = MemoryBackend()
        self.saved = {}
        self.directories = set()

    def _get_backend(self, backend):
        return self.memory

    def ensure_directory(self, path, backend):
        self.directories.add((path, backend))

    def save(self, value, path, backend):
        self.saved[(path, backend)] = value
        self.memory._memory_store[path] = value

    def exists(self, path, backend):
        return path in self.memory._memory_store

    def delete(self, path, backend):
        del self.memory._memory_store[path]
        self.saved.pop((path, backend), None)

    def load(self, path, backend):
        return self.memory._memory_store[path]


class ContextStub:
    def __init__(self):
        self.axis_id = "A01"
        self.filemanager = FileManagerStub()
        self.runtime_value_store = RuntimeValueStore()


@dataclass(frozen=True, slots=True)
class CoreExecutionRequest:
    func_callable: Callable
    main_data_arg: object
    base_kwargs: Mapping[str, object]
    context: ContextStub
    artifact_inputs: Mapping[ArtifactSpecRef, ArtifactInputPlan]
    artifact_outputs: Mapping[ArtifactSpecRef, ArtifactOutputPlan]
    group_key: str = "default"
    execution_group_scope: ComponentGroupScope = ComponentGroupScope.ungrouped()
    processing_contract: ProcessingContract = ProcessingContract.PURE_3D
    runtime_plane_count: int = 1


def _execute_function_core(request: CoreExecutionRequest):
    pattern = (
        request.func_callable
        if request.execution_group_scope.is_ungrouped
        else {request.group_key: [request.func_callable]}
    )
    compiled_pattern = compile_function_pattern(
        pattern,
        request.artifact_inputs,
        request.artifact_outputs,
    )
    compiled_pattern = PathPlannerArtifactStage(
        PathPlanner.__new__(PathPlanner)
    ).compile_invocation_input_edges(
        compiled_pattern,
        artifact_inputs=request.artifact_inputs,
        relation_source_scopes={
            plan.ref(): plan.producer_group_scope()
            for plan in request.artifact_inputs.values()
        },
        execution_group_scope=request.execution_group_scope,
        consumer_variable_components=ComponentSet(),
    )
    compiled_invocation = next(compiled_pattern.iter_invocations())
    declared_contract = compiled_invocation.contract
    contract = replace(
        declared_contract,
        metadata=replace(
            declared_contract.metadata,
            input_memory_type=MEMORY_TYPE_NUMPY,
            output_memory_type=MEMORY_TYPE_NUMPY,
            processing_contract=request.processing_contract,
        ),
    )
    invocation = replace(
        compiled_invocation,
        contract=contract,
        kwargs=tuple(request.base_kwargs.items()),
    )
    execution_plan = SimpleNamespace(
        step_index=0,
        step_scope_id="test::function_step",
        step_name="test",
        axis_id=request.context.axis_id,
        input_memory_type=MEMORY_TYPE_NUMPY,
        device_id=0,
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        source_load_plan=SourceLoadPlan(),
        variable_components=(),
        group_by_value=None,
        execution_group_scope=request.execution_group_scope,
        artifact_inputs=request.artifact_inputs,
        artifact_outputs=request.artifact_outputs,
    )
    component_value = (
        None if request.execution_group_scope.is_ungrouped else request.group_key
    )
    runtime_scope = FunctionRuntimeScope(
        context=request.context,
        execution_plan=execution_plan,
        compiled_group=CompiledFunctionGroup(
            group_key=request.group_key,
            invocations=(invocation,),
        ),
        component_value=component_value,
        artifacts=ComponentArtifactPlans.from_step_component(
            execution_plan,
            component_value,
        ),
        source_binding_context=SourceBindingRuntimeContext.empty(),
        runtime_plane_index=0,
        runtime_plane_count=request.runtime_plane_count,
    )
    group_key = invocation.key.runtime_group_key(runtime_scope.component_value)
    return FunctionCoreExecutor(
        main_data_arg=request.main_data_arg,
        source_memory_type=MEMORY_TYPE_NUMPY,
        runtime_scope=runtime_scope,
        invocation=invocation,
        artifacts=runtime_scope.artifacts.select_for_invocation(
            invocation,
            execution_scope=runtime_scope.execution_plan.execution_group_scope,
            component_key=runtime_scope.component_key,
        ),
        group_key=group_key,
        plane_projection=RuntimePlaneProjection.stack(
            runtime_scope.runtime_plane_count
        ),
    ).execute()


def test_function_core_passes_payload_data_to_array_callable_and_restores_context():
    tiles = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
    payload = ImagePayloadMetadata(
        source_path="/input/A01_s001_w1_z001_t001.tif",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(tiles, None)
    positions = np.array([[0, 0], [4, 0]], dtype=np.float32)

    def assemble(image, **kwargs):
        assert isinstance(image, np.ndarray)
        return assemble_stack_cpu(image, **kwargs)

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=assemble,
            main_data_arg=payload,
            base_kwargs={
                "positions": positions,
                "blend_method": "none",
            },
            context=ContextStub(),
            artifact_inputs={},
            artifact_outputs={},
            processing_contract=ProcessingContract.VOLUMETRIC_TO_SLICE,
        )
    )

    assert image_payload_data(result).shape == (4, 8)
    assert image_payload_data(result).dtype == tiles.dtype
    assert image_payload_metadata(result).source_path == (
        "/input/A01_s001_w1_z001_t001.tif"
    )


def test_function_core_preserves_output_declared_runtime_slice_axis():
    source = np.ones((4, 5), dtype=np.float32)

    def produce_runtime_slice_stack(_image):
        return ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(np.ones((1, 4, 5), dtype=np.float32), None)

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=produce_runtime_slice_stack,
            main_data_arg=source,
            base_kwargs={},
            context=ContextStub(),
            artifact_inputs={},
            artifact_outputs={},
        )
    )

    assert image_payload_data(result).shape == (1, 4, 5)
    assert image_payload_metadata(result).plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_image_output_context_preserves_stack_after_exact_source_projection():
    source_spec = ArtifactSpec.input("OrigGray", ImageArtifactType)
    source = ImagePayloadMetadata(
        source_path="/input/A01_s001_w1_z001_t001.tif",
        source_image_names=(source_spec.name,),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((1, 4, 5), dtype=np.float32), None)
    output = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((1, 4, 5), dtype=np.float32), None)
    output_plan = ArtifactOutputPlan(
        name="CorrGray",
        path="/memory/CorrGray.pkl",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.Z_INDEX,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )

    result = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize(
        source,
        output,
        output_plan,
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=1,
        ),
    )

    assert image_payload_data(result).shape == (1, 4, 5)
    assert image_payload_metadata(result).plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert image_payload_metadata(result).source_image_names == (source_spec.name,)


def test_image_output_context_does_not_infer_axis_for_unmarked_payload():
    source_spec = ArtifactSpec.input("NucleiImage", ImageArtifactType)
    source = ImagePayloadMetadata(
        source_image_names=(source_spec.name,),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w2_z001_t001.tif",
                "/input/A01_s001_w2_z002_t001.tif",
            ),
            component_metadata=(
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "2",
                    "z_index": "1",
                    "timepoint": "1",
                },
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "2",
                    "z_index": "2",
                    "timepoint": "1",
                },
            ),
        ),
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)
    output_plan = ArtifactOutputPlan(
        name="SavedNuclei",
        path="/memory/SavedNuclei.pkl",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.Z_INDEX,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )

    result = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize_from_projector(
        source,
        np.ones((2, 4, 5), dtype=np.uint16),
        output_plan,
        RuntimePlaneProjection.stack(2),
    )

    metadata = image_payload_metadata(result)
    assert metadata.plane_axis is None
    assert metadata.source_image_provenance_planes.paths == (
        "/input/A01_s001_w2_z001_t001.tif",
        "/input/A01_s001_w2_z002_t001.tif",
    )


def test_image_output_context_rejects_declared_output_axis_shape_drift():
    source_spec = ArtifactSpec.input("NucleiImage", ImageArtifactType)
    output = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((1, 4, 5), dtype=np.uint16), None)
    output_plan = ArtifactOutputPlan(
        name="SavedNuclei",
        path="/memory/SavedNuclei.pkl",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.Z_INDEX,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )

    with pytest.raises(
        ValueError,
        match="declared 'runtime_slice' axis of size 2",
    ):
        FunctionOutputContextStrategy.for_output_plan(
            output_plan,
        ).contextualize(
            np.ones((1, 4, 5), dtype=np.float32),
            output,
            output_plan,
            RuntimePlaneAxisValueProjection.preserve(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                axis_size=2,
            ),
        )


def test_image_output_context_preserves_complete_scalar_rgb_identity() -> None:
    source_spec = ArtifactSpec.input("ColorNeighbors", ImageArtifactType)
    source = ImagePayloadMetadata(
        source_path="/input/A01_s001_w1_z001_t001.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
            "z_index": "1",
            "timepoint": "1",
        },
        source_image_names=(source_spec.name,),
        source_channel_axis=2,
    ).payload_with(np.ones((4, 5, 3), dtype=np.float32), None)
    output = with_image_payload_data(
        source,
        np.ones((4, 5, 3), dtype=np.uint8),
    )
    output_plan = ArtifactOutputPlan(
        name="SavedColorNeighbors",
        path="/memory/SavedColorNeighbors.png",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.SITE,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )

    result = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize_from_projector(
        source,
        output,
        output_plan,
        RuntimePlaneProjection.stack(1),
    )

    assert image_payload_data(result).shape == (4, 5, 3)
    metadata = image_payload_metadata(result)
    assert metadata.plane_axis is None
    assert metadata.normalized_source_channel_axis(result) == 2
    assert metadata.source_path == "/input/A01_s001_w1_z001_t001.tif"


def test_image_output_context_preserves_target_axis_across_source_rank_change() -> None:
    source = ImagePayloadMetadata(
        source_path="/input/A01_s001_color.tif",
        source_component_metadata={"well": "A01", "site": "1"},
        source_image_names=("OrigColor",),
        source_channel_axis=3,
    ).payload_with(np.ones((2, 4, 5, 3), dtype=np.float32), None)
    output = ImagePayloadMetadata(
        source_path="/input/A01_s001_color.tif",
        source_channel_axis=4,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((1, 2, 4, 5, 3), dtype=np.float32), None)

    result = FunctionOutputContextStrategy.for_output_plan(None).contextualize(
        source,
        output,
        None,
        None,
    )

    metadata = image_payload_metadata(result)
    assert metadata.normalized_source_channel_axis(result) == 4
    assert metadata.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert dict(metadata.source_component_metadata or {}) == {
        "well": "A01",
        "site": "1",
    }
    assert metadata.source_image_names == ("OrigColor",)


def test_image_output_context_projects_complete_multi_plane_identity() -> None:
    source_spec = ArtifactSpec.input("Volume", ImageArtifactType)
    metadata = ImagePayloadMetadata(
        source_image_names=(source_spec.name,),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.tif",
                "/input/A01_s001_w1_z002_t001.tif",
            ),
            component_metadata=(
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                    "z_index": "1",
                    "timepoint": "1",
                },
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                    "z_index": "2",
                    "timepoint": "1",
                },
            ),
        ),
    )
    source = metadata.payload_with(np.ones((2, 4, 5), dtype=np.float32), None)
    output = metadata.replace_fields(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(
        np.ones((2, 4, 5), dtype=np.uint16),
        None,
    )
    output_plan = ArtifactOutputPlan(
        name="SavedVolume",
        path="/memory/SavedVolume.tif",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.Z_INDEX,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )

    result = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize_from_projector(
        source,
        output,
        output_plan,
        RuntimePlaneProjection.stack(2),
    )

    result_metadata = image_payload_metadata(result)
    assert result_metadata.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert result_metadata.source_image_provenance_planes.count == 2


def test_named_image_outputs_bypass_unrelated_source_axis_projection():
    source = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/red.tif", "/input/green.tif"),
            component_metadata=(
                {"channel": "1", "source_alias": "OrigRed"},
                {"channel": "2", "source_alias": "OrigGreen"},
            ),
        ),
    ).payload_with(np.zeros((2, 4, 5), dtype=np.float32), None)
    outputs = tuple(
        ImagePayloadMetadata(
            source_path=path,
            source_image_names=(name,),
        ).payload_with(np.zeros((4, 5), dtype=np.float32), None)
        for path, name in (
            ("/input/red.tif", "AlignedRed"),
            ("/input/green.tif", "AlignedGreen"),
        )
    )
    bundle = pack_aligned_image_outputs(
        outputs,
        slice_contexts=tuple(
            AlignedImageSliceContext.main_flow(name)
            for name in ("AlignedRed", "AlignedGreen")
        ),
    )

    result = FunctionOutputContextStrategy.for_output_plan(
        None
    ).contextualize_from_projector(
        source,
        bundle,
        None,
        RuntimePlaneProjection.stack(2),
    )

    assert result is bundle


def test_image_output_context_projector_proves_source_ownership_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001.tif",
                "/input/A01_s001_w1_z002.tif",
            ),
            component_metadata=(
                {"well": "A01", "z_index": "1"},
                {"well": "A01", "z_index": "2"},
            ),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.zeros((2, 4, 5), dtype=np.uint16), None)
    output = np.ones((2, 4, 5), dtype=np.uint8)
    ownership_proofs = []
    original = ImageFunctionOutputContextStrategy.output_owns_source_context

    def track_ownership_proof(*args):
        ownership_proofs.append(args)
        return original(*args)

    monkeypatch.setattr(
        ImageFunctionOutputContextStrategy,
        "output_owns_source_context",
        staticmethod(track_ownership_proof),
    )

    result = FunctionOutputContextStrategy.for_output_plan(
        None
    ).contextualize_from_projector(
        source,
        output,
        None,
        RuntimePlaneProjection.stack(2),
    )

    assert len(ownership_proofs) == 1
    np.testing.assert_array_equal(image_payload_data(result), output)
    assert image_payload_metadata(result).source_image_provenance_planes == (
        image_payload_metadata(source).source_image_provenance_planes
    )


def test_image_output_context_uses_base_projector_resolution() -> None:
    assert (
        "contextualize_from_projector"
        not in ImageFunctionOutputContextStrategy.__dict__
    )


def test_composed_function_output_owns_collapsed_source_identity():
    source = ImagePayloadMetadata(
        source_path="/input/A01_s001_w1_z001_t001.tif",
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((1, 4, 5), dtype=np.float32), None)
    output = ImagePayloadMetadata(
        source_path="/input/A01_s001_w1_z001_t001.tif",
        source_channel_axis=-1,
    ).payload_with(np.ones((4, 5, 3), dtype=np.float32), None)

    @composed_image_payload
    def compose_channels(_image):
        return output

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=compose_channels,
            main_data_arg=source,
            base_kwargs={},
            context=ContextStub(),
            artifact_inputs={},
            artifact_outputs={},
        )
    )

    assert image_payload_data(result).shape == (4, 5, 3)
    assert image_payload_metadata(result).plane_axis is None
    assert image_payload_metadata(result).source_channel_axis == -1


def test_crop_mask_sidecar_names_derive_from_core_artifact_role():
    assert ArtifactSidecarRole.CROP_MASK.name_for("CroppedImage") == (
        "CroppedImage__crop_mask"
    )


def test_unstack_payload_context_slices_volume_stack_mask_with_volume_data():
    data = np.ones((1, 3, 4, 5), dtype=np.float32)
    mask = np.ones((1, 3, 4, 5), dtype=bool)
    payload = ImagePayloadMetadata().payload_with(data, mask)

    [slice_payload] = unstack_image_payload_context(
        payload,
        [data[0]],
        default_plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    assert image_payload_mask(slice_payload).shape == data[0].shape


def test_unstack_payload_context_preserves_volumetric_source_slice_identity():
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    mask = np.ones_like(data, dtype=bool)
    mask[1] = False
    plane_metadata = (
        {"well": "A01", "site": 1, "channel": 1, "z_index": 1},
        {"well": "A01", "site": 1, "channel": 1, "z_index": 2},
        {"well": "A01", "site": 1, "channel": 1, "z_index": 3},
    )
    metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1.tif",) * 3,
            component_metadata=plane_metadata,
        )
    )
    payload = metadata.payload_with(data, mask)

    [slice_payload] = unstack_image_payload_context(payload, [data])
    slice_metadata = image_payload_metadata(slice_payload)

    np.testing.assert_array_equal(image_payload_data(slice_payload), data)
    np.testing.assert_array_equal(image_payload_mask(slice_payload), mask)
    assert slice_metadata.source_image_provenance_planes.count == 3
    assert (
        tuple(
            dict(item)
            for item in slice_metadata.source_image_provenance_planes.component_metadata
        )
        == plane_metadata
    )


def test_unstack_payload_context_does_not_expand_scalar_source_without_plane_axis():
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    source_path = "/input/A01_s001_w3_z001_t001.tif"
    metadata = ImagePayloadMetadata(
        source_path=source_path,
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "1",
            SOURCE_PLANE_INDEX_FIELD: "0",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
    )
    payload = metadata.payload_with(data, None)

    [slice_payload] = unstack_image_payload_context(payload, [data])
    slice_metadata = image_payload_metadata(slice_payload)

    assert not slice_metadata.source_image_provenance_planes.has_values
    assert slice_metadata.source_path == source_path
    assert dict(slice_metadata.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "3",
        "z_index": "1",
        SOURCE_PLANE_INDEX_FIELD: "0",
        SOURCE_PLANE_COUNT_FIELD: "3",
    }


def test_unstack_payload_context_preserves_declared_axis_over_default():
    data = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    payload = ImagePayloadMetadata(
        source_image_names=("DNA", "Actin"),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/dna.tif", "/input/actin.tif"),
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    ).payload_with(data, None)

    slices = unstack_image_payload_context(
        payload,
        list(data),
        default_plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    assert [
        image_payload_metadata(slice_payload).source_image_names
        for slice_payload in slices
    ] == [("DNA",), ("Actin",)]


def test_stack_payload_context_projects_single_payload_mask_to_stack_domain():
    data = np.ones((4, 5), dtype=np.float32)
    stack = data[np.newaxis, ...]
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:3] = True
    payload = ImagePayloadMetadata(source_path="/input/A01.tif").payload_with(
        data, mask
    )

    stacked = stack_image_payload_context(
        (payload,),
        stack,
        metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
    )

    assert image_payload_mask(stacked).shape == stack.shape
    np.testing.assert_array_equal(image_payload_mask(stacked), mask[np.newaxis, ...])


def test_stack_image_payloads_uses_payload_memory_and_preserves_masks():
    payloads = tuple(
        ImagePayloadMetadata(source_path=f"/input/{index}.tif").payload_with(
            np.full((4, 5), index, dtype=np.float32),
            np.full((4, 5), bool(index), dtype=bool),
        )
        for index in (0, 1)
    )

    stacked = stack_image_payloads(
        payloads,
        metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
    )

    np.testing.assert_array_equal(
        image_payload_data(stacked),
        np.stack(tuple(image_payload_data(payload) for payload in payloads)),
    )
    np.testing.assert_array_equal(
        image_payload_mask(stacked),
        np.stack(tuple(image_payload_mask(payload) for payload in payloads)),
    )


def test_stack_image_payloads_preserves_declared_singleton_stack_axis():
    payload = np.ones((4, 5), dtype=np.float32)

    stacked = stack_image_payloads(
        (payload,),
        metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
    )

    assert image_payload_data(stacked).shape == (1, 4, 5)
    assert image_payload_metadata(stacked).plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    np.testing.assert_array_equal(image_payload_data(stacked)[0], payload)


def test_stack_payload_context_preserves_single_volumetric_payload_identity():
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    plane_metadata = (
        {"well": "A01", "site": 1, "channel": 1, "z_index": 1},
        {"well": "A01", "site": 1, "channel": 1, "z_index": 2},
        {"well": "A01", "site": 1, "channel": 1, "z_index": 3},
    )
    payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1.tif",) * 3,
            component_metadata=plane_metadata,
        )
    ).payload_with(data, None)

    stacked_payload = stack_image_payload_context(
        (payload,),
        data[np.newaxis, ...],
        metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
    )
    stacked_metadata = image_payload_metadata(stacked_payload)

    assert image_payload_data(stacked_payload).shape == (1, 3, 4, 5)
    assert stacked_metadata.source_image_provenance_planes.count == 3
    assert stacked_metadata.source_image_provenance_planes.contributor_count == 0
    assert (
        tuple(
            dict(item)
            for item in stacked_metadata.source_image_provenance_planes.component_metadata
        )
        == plane_metadata
    )


def test_stack_payload_context_nests_incompatible_singleton_plane_topology():
    data = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    plane_metadata = (
        {"well": "A01", "site": 1, "channel": 1},
        {"well": "A01", "site": 1, "channel": 2},
    )
    payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_w1.tif", "/input/A01_w2.tif"),
            component_metadata=plane_metadata,
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    ).payload_with(data, None)

    stacked_payload = stack_image_payload_context(
        (payload,),
        data[np.newaxis, ...],
        metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
    )
    provenance_planes = image_payload_metadata(
        stacked_payload
    ).source_image_provenance_planes

    assert provenance_planes.count == 1
    assert provenance_planes.contributor_count == 2
    assert tuple(
        dict(contributor.component_metadata)
        for contributor in provenance_planes.contributors
    ) == plane_metadata


def test_stack_payload_context_preserves_singleton_stack_payload_mask_domain():
    data = np.ones((1, 4, 5), dtype=np.float32)
    mask = np.ones_like(data, dtype=bool)
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(data, mask)

    stacked_payload = stack_image_payload_context(
        (payload,),
        data[np.newaxis, ...],
        metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
    )

    np.testing.assert_array_equal(
        image_payload_mask(stacked_payload),
        mask[np.newaxis, ...],
    )


def test_stack_payload_context_composes_single_image_slice_mask_axis():
    data = np.ones((4, 5), dtype=np.float32)
    stack = data[np.newaxis, ...]
    mask = np.ones_like(data, dtype=bool)
    payload = ImagePayloadMetadata().payload_with(data, mask)

    stacked_payload = stack_image_payload_context(
        (payload,),
        stack,
        metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
    )

    np.testing.assert_array_equal(
        image_payload_mask(stacked_payload),
        mask[np.newaxis, ...],
    )


def test_execute_function_core_saves_named_artifacts():
    context = ContextStub()
    measurement_spec = ArtifactSpec.output(
        "measurements",
        MeasurementsArtifactType,
        relations=(ArtifactMeasurementSubjectRelation(),),
    )

    @artifact_outputs(
        measurement_spec,
    )
    def analyze(image):
        return (
            image + 1,
            MeasurementTable(
                name="measurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"count": 2},),
                    fields=(FieldSpec("count", int),),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT,
                ),
            ),
        )

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=analyze,
            main_data_arg=41,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name="measurements",
                    path="/memory/measurements.pkl",
                    artifact_type=MeasurementsArtifactType,
                    relations=measurement_spec.relations,
                ),)},
        )
    )

    assert result == 42
    assert tuple(
        context.filemanager.saved[("/memory/measurements.pkl", "memory")].rows
    ) == ({"count": 2},)
    stored = context.runtime_value_store.find(
        name="measurements",
        axis_id="A01",
    )
    assert len(stored) == 1
    assert tuple(stored[0].value.data.rows) == ({"count": 2},)


def test_trailing_object_labels_do_not_replace_canonical_image_output():
    context = ContextStub()
    source = ImagePayloadMetadata(
        source_path="/input/A01_s1_w1.tif",
    ).payload_with(np.zeros((1, 4, 5), dtype=np.uint16))
    output = np.full((1, 4, 5), 7, dtype=np.uint16)
    measurement_spec = ArtifactSpec.output(
        "cell_counts",
        MeasurementsArtifactType,
        relations=(ArtifactMeasurementSubjectRelation(),),
    )
    labels_spec = ArtifactSpec.output(
        "segmentation_masks",
        ObjectLabelsArtifactType,
    )

    @artifact_outputs(measurement_spec, labels_spec)
    def count(_image):
        return (
            output,
            MeasurementTable(
                name=measurement_spec.name,
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"count": 1},),
                    fields=(FieldSpec("count", int),),
                ),
                subject=MeasurementSubject(MeasurementScope.ARTIFACT),
            ),
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(
                    labels=np.ones((1, 4, 5), dtype=np.int32)
                )
            ),
        )

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=count,
            main_data_arg=source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name=measurement_spec.name,
                    path="/memory/cell-counts.pkl",
                    artifact_type=MeasurementsArtifactType,
                    relations=measurement_spec.relations,
                ), ArtifactOutputPlan(
                    name=labels_spec.name,
                    path="/memory/segmentation-masks.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                ),)},
        )
    )

    np.testing.assert_array_equal(image_payload_data(result), output)
    [stored_labels] = context.runtime_value_store.find(
        name=labels_spec.name,
        axis_id=context.axis_id,
    )
    assert isinstance(stored_labels.value.data, ObjectLabelSet)


def test_execute_function_core_attaches_execution_group_identity_to_artifact():
    context = ContextStub()
    source = ImagePayloadMetadata(
        source_path="/input/A01_s001_w1_z001.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "z_index": "1",
        },
    ).payload_with(np.zeros((3, 4, 5), dtype=np.uint16), None)

    @artifact_outputs(
        ArtifactSpec.output("segmentation_masks", ObjectLabelsArtifactType),
    )
    def segment(image):
        del image
        return ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(
                labels=np.ones((3, 4, 5), dtype=np.int32)
            )
        )

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=segment,
            main_data_arg=source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name="segmentation_masks",
                    path="/memory/A01_w2_segmentation_masks.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    group_component=AllComponents.CHANNEL,
                    group_keys=("2",),
                    paths_by_group={
                        "2": "/memory/A01_w2_segmentation_masks.pkl",
                    },
                ),)},
            group_key="2",
            execution_group_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
        )
    )

    stored = context.runtime_value_store.find(
        name="segmentation_masks",
        axis_id="A01",
        group_key="2",
        match_group=True,
    )
    assert len(stored) == 1
    metadata = image_payload_metadata(stored[0].value.data)
    assert dict(metadata.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "2",
        "z_index": "1",
    }


def test_execute_function_core_attaches_dynamic_execution_group_to_artifact():
    context = ContextStub()
    source = ImagePayloadMetadata(
        source_path="/input/A01_s001_z001.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "z_index": "1",
        },
    ).payload_with(np.zeros((3, 4, 5), dtype=np.uint16), None)

    @artifact_outputs(
        ArtifactSpec.output("segmentation_masks", ObjectLabelsArtifactType),
    )
    def segment(image):
        del image
        return ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(
                labels=np.ones((3, 4, 5), dtype=np.int32)
            )
        )

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=segment,
            main_data_arg=source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name="segmentation_masks",
                    path="/memory/A01_segmentation_masks.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    group_keys=(None,),
                    group_component=AllComponents.CHANNEL,
                    paths_by_group={
                        None: "/memory/A01_segmentation_masks.pkl",
                    },
                ),)},
            group_key="2",
            execution_group_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
        )
    )

    assert (
        "/memory/A01_w2_segmentation_masks.pkl",
        "memory",
    ) in context.filemanager.saved
    stored = context.runtime_value_store.find(
        name="segmentation_masks",
        axis_id="A01",
        group_key="2",
        match_group=True,
    )
    assert len(stored) == 1
    assert stored[0].path == "/memory/A01_w2_segmentation_masks.pkl"
    metadata = image_payload_metadata(stored[0].value.data)
    assert dict(metadata.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "2",
        "z_index": "1",
    }


def test_execute_function_core_rejects_undeclared_tuple_main_output():
    context = ContextStub()
    first = ImagePayloadMetadata(source_image_names=("OrigRed",)).payload_with(
        np.full((4, 5), 1, dtype=np.float32), None
    )
    second = ImagePayloadMetadata(source_image_names=("OrigGreen",)).payload_with(
        np.full((4, 5), 2, dtype=np.float32), None
    )

    def split_outputs(image):
        del image
        return first, second

    with pytest.raises(TypeError, match="Tuple returns require declared"):
        _execute_function_core(
            CoreExecutionRequest(
                func_callable=split_outputs,
                main_data_arg=np.zeros((2, 4, 5), dtype=np.float32),
                base_kwargs={},
                context=context,
                artifact_inputs={},
                artifact_outputs={},
            )
        )


def test_execute_function_core_routes_exact_image_artifact_tuple_to_main_flow():
    context = ContextStub()
    red = ImagePayloadMetadata(source_image_names=("OrigRed",)).payload_with(
        np.full((4, 5), 1, dtype=np.float32), None
    )
    green = ImagePayloadMetadata(source_image_names=("OrigGreen",)).payload_with(
        np.full((4, 5), 2, dtype=np.float32), None
    )

    @artifact_outputs(
        ArtifactSpec.output("Red", ImageArtifactType),
        ArtifactSpec.output("Green", ImageArtifactType),
    )
    def corrected_images(image):
        del image
        main_output = pack_aligned_image_outputs(
            (red, green),
            slice_contexts=(
                AlignedImageSliceContext.main_flow(
                    "Red", artifact_kind=ImageArtifactType.value
                ),
                AlignedImageSliceContext.main_flow(
                    "Green", artifact_kind=ImageArtifactType.value
                ),
            ),
        )
        return main_output

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=corrected_images,
            main_data_arg=np.zeros((2, 4, 5), dtype=np.float32),
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name="Red",
                    path="/memory/red.pkl",
                    artifact_type=ImageArtifactType,
                ), ArtifactOutputPlan(
                    name="Green",
                    path="/memory/green.pkl",
                    artifact_type=ImageArtifactType,
                ),)},
        )
    )

    assert isinstance(result, AlignedImageStack)
    assert tuple(image_payload_data(payload)[0, 0] for payload in result.slices) == (
        1,
        2,
    )
    red_records = context.runtime_value_store.find(name="Red", axis_id="A01")
    green_records = context.runtime_value_store.find(name="Green", axis_id="A01")
    assert len(red_records) == 1
    assert len(green_records) == 1
    assert image_payload_data(red_records[0].value.data)[0, 0] == 1
    assert image_payload_data(green_records[0].value.data)[0, 0] == 2


def test_execute_function_core_keeps_image_sidecar_out_of_main_flow():
    context = ContextStub()
    image = np.full((4, 5), 1, dtype=np.float32)
    mask = np.full((4, 5), 2, dtype=np.float32)

    @artifact_outputs(
        ArtifactSpec.output("CropGreen", ImageArtifactType),
        ArtifactSpec.output(
            "CropGreen__crop_mask",
            ImageArtifactType,
            sidecar_role=ArtifactSidecarRole.CROP_MASK,
        ),
    )
    def crop_outputs(source):
        del source
        return image, mask

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=crop_outputs,
            main_data_arg=np.zeros((4, 5), dtype=np.float32),
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name="CropGreen",
                    path="/memory/crop-green.pkl",
                    artifact_type=ImageArtifactType,
                ), ArtifactOutputPlan(
                    name="CropGreen__crop_mask",
                    path="/memory/crop-green-mask.pkl",
                    artifact_type=ImageArtifactType,
                    sidecar_role=ArtifactSidecarRole.CROP_MASK,
                ),)},
        )
    )

    np.testing.assert_array_equal(image_payload_data(result), image)
    mask_records = context.runtime_value_store.find(
        name="CropGreen__crop_mask",
        axis_id="A01",
    )
    assert len(mask_records) == 1
    np.testing.assert_array_equal(
        image_payload_data(mask_records[0].value.data),
        mask,
    )


def test_execute_function_core_saves_single_image_artifact_output_to_main_flow():
    context = ContextStub()
    source = ImagePayloadMetadata(
        source_path="/input/A01_s1_w1.tif",
        source_component_metadata={"well": "A01", "site": "1", "channel": "1"},
        source_image_names=("OrigGreen",),
    ).payload_with(np.zeros((4, 5), dtype=np.float32))
    output = np.full((4, 5), 7, dtype=np.float32)

    @artifact_outputs(
        ArtifactSpec.output("CorrectedImage", ImageArtifactType),
    )
    def produce_corrected_image(image):
        del image
        return output

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=produce_corrected_image,
            main_data_arg=source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name="CorrectedImage",
                    path="/memory/corrected.pkl",
                    artifact_type=ImageArtifactType,
                ),)},
        )
    )

    np.testing.assert_array_equal(result, output)
    assert image_payload_metadata(result).source_image_names == ("CorrectedImage",)
    assert image_payload_metadata(
        result
    ).source_provenance.represented_source_image_names == (
        "CorrectedImage",
        "OrigGreen",
    )
    stored = context.runtime_value_store.find(name="CorrectedImage", axis_id="A01")
    assert len(stored) == 1
    np.testing.assert_array_equal(image_payload_data(stored[0].value.data), output)
    assert image_payload_metadata(stored[0].value.data).source_image_names == (
        "CorrectedImage",
    )


def test_execute_function_core_names_slice_aligned_image_outputs() -> None:
    context = ContextStub()
    source_slices = tuple(
        ImagePayloadMetadata(
            source_path=f"/input/A01_s{site}_w1.tif",
            source_component_metadata={
                "well": "A01",
                "site": str(site),
                "channel": "1",
            },
            source_image_names=("OrigGreen",),
        ).payload_with(np.full((4, 5), site, dtype=np.float32))
        for site in (1, 2)
    )
    source = RuntimeSliceAlignedValues(source_slices)
    output_slices = tuple(
        np.full((4, 5), site + 10, dtype=np.float32) for site in (1, 2)
    )
    output_spec = ArtifactSpec.output("DerivedImage", ImageArtifactType)

    @artifact_outputs(output_spec)
    def produce_image(image):
        del image
        return RuntimeSliceAlignedValues(output_slices)

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=produce_image,
            main_data_arg=source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name=output_spec.name,
                    path="/memory/derived-image.pkl",
                    artifact_type=ImageArtifactType,
                    variable_components=(AllComponents.SITE,),
                ),)},
            runtime_plane_count=2,
        )
    )

    [stored] = context.runtime_value_store.find(
        name=output_spec.name,
        axis_id=context.axis_id,
    )
    stored_data = stored.value.data
    assert isinstance(stored_data, RuntimeSliceAlignedValues)
    for index, source_payload in enumerate(source_slices):
        output_payload = stored_data.value_for_slice(index)
        metadata = image_payload_metadata(output_payload)
        np.testing.assert_array_equal(
            image_payload_data(output_payload),
            output_slices[index],
        )
        assert metadata.source_path == image_payload_metadata(source_payload).source_path
        assert metadata.source_image_names == (output_spec.name,)
        assert metadata.source_provenance.represented_source_image_names == (
            output_spec.name,
            "OrigGreen",
    )

    assert isinstance(result, RuntimeSliceAlignedValues)
    assert tuple(
        image_payload_metadata(result.value_for_slice(index)).source_image_names
        for index in range(result.slice_count)
    ) == ((output_spec.name,), (output_spec.name,))


def test_execute_function_core_saves_artifact_to_runtime_group_path():
    context = ContextStub()
    measurement_spec = ArtifactSpec.output(
        "measurements",
        MeasurementsArtifactType,
        relations=(ArtifactMeasurementSubjectRelation(),),
    )

    @artifact_outputs(
        measurement_spec,
    )
    def analyze(image):
        return (
            image,
            MeasurementTable(
                name="measurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"site": "2", "count": 3},),
                    fields=(FieldSpec("site", str), FieldSpec("count", int)),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT,
                ),
            ),
        )

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=analyze,
            main_data_arg=np.zeros((2, 2), dtype=np.uint8),
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name="measurements",
                    path="/memory/A01_measurements.pkl",
                    artifact_type=MeasurementsArtifactType,
                    relations=measurement_spec.relations,
                    group_component=AllComponents.SITE,
                    group_keys=("1", "2"),
                    paths_by_group={
                        "1": "/memory/A01_s1_measurements.pkl",
                        "2": "/memory/A01_s2_measurements.pkl",
                    },
                ),)},
            group_key="2",
            execution_group_scope=ComponentGroupScope.from_raw(
                ("2",),
                component=AllComponents.SITE,
            ),
        )
    )

    assert tuple(
        context.filemanager.saved[
            ("/memory/A01_s2_measurements.pkl", "memory")
        ].rows
    ) == ({"site": "2", "count": 3},)
    stored = context.runtime_value_store.find(
        name="measurements",
        axis_id="A01",
        group_key="2",
        match_group=True,
    )
    assert len(stored) == 1
    assert stored[0].path == "/memory/A01_s2_measurements.pkl"


def test_execute_function_core_preserves_main_output_source_metadata():
    context = ContextStub()
    source = ImagePayloadMetadata(
        source_path="/input/01_POS002_D.TIF",
        source_component_metadata={
            "well": "01",
            "site": "POS002",
            "channel": "D",
        },
    ).payload_with(np.zeros((2, 3), dtype=np.uint16), None)

    def threshold(image):
        return np.asarray(image) > 0

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=threshold,
            main_data_arg=source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={},
        )
    )

    metadata = image_payload_metadata(result)
    assert metadata.source_path == "/input/01_POS002_D.TIF"
    assert dict(metadata.source_component_metadata) == {
        "well": "01",
        "site": "POS002",
        "channel": "D",
    }


def test_managed_runtime_adapter_output_preserves_authoritative_source_metadata():
    context = ContextStub()
    ambient_source = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s1_w1.tif",),
            component_metadata=(
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                },
            ),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.zeros((1, 2, 3), dtype=np.uint16), None)
    adapter_output = ImagePayloadMetadata(
        source_path="/input/A01_s1_w3.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "3",
        },
        source_channel_axis=-1,
    ).payload_with(np.ones((2, 3, 3), dtype=np.uint16), None)
    source_input_spec = ArtifactSpec.input("source_image", ImageArtifactType)

    @artifact_inputs(source_input_spec)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    def enhance(image, *, runtime):
        del runtime
        assert image_payload_metadata(image).source_image_provenance_planes == (
            image_payload_metadata(ambient_source).source_image_provenance_planes
        )
        return adapter_output

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=enhance,
            main_data_arg=ambient_source,
            base_kwargs={},
            context=context,
            artifact_inputs={plan.ref(): plan for plan in (ArtifactInputPlan(
                    name=source_input_spec.name,
                    path="/memory/source_image.pkl",
                    artifact_type=source_input_spec.artifact_type,
                ),)},
            artifact_outputs={},
        )
    )

    metadata = image_payload_metadata(result)
    assert image_payload_data(result).shape == (2, 3, 3)
    assert metadata.plane_axis is None
    assert metadata.source_channel_axis == -1
    assert metadata.source_path == "/input/A01_s1_w3.tif"
    assert dict(metadata.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "3",
    }


def test_pattern_group_runtime_stacks_nominal_scalar_rgb_output_as_one_slice():
    scalar_output = ImagePayloadMetadata(
        source_path="/input/A01_s1_w3.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "3",
        },
        source_channel_axis=-1,
    ).payload_with(np.ones((2, 3, 3), dtype=np.uint16), None)

    @artifact_outputs(ArtifactSpec.output("RGBImage", ImageArtifactType))
    def rgb_output(image):
        return image

    output_plan = ArtifactOutputPlan(
        name="RGBImage",
        path="/memory/RGBImage.pkl",
        artifact_type=ImageArtifactType,
    )
    invocation = next(
        compile_function_pattern(
            rgb_output,
            {},
            {plan.ref(): plan for plan in (output_plan,)},
        ).iter_invocations()
    )
    runtime = object.__new__(PatternGroupRuntime)
    runtime.request = SimpleNamespace(
        compiled_group=CompiledFunctionGroup(
            group_key="default",
            invocations=(invocation,),
        ),
        execution_plan=SimpleNamespace(
            output_memory_type=MEMORY_TYPE_NUMPY,
            device_id=0,
            artifact_inputs={},
            execution_group_scope=ComponentGroupScope.ungrouped(),
            artifact_outputs={
                output_plan.ref(): output_plan,
            },
            ),
        component_key=None,
    )

    output = runtime._validate_and_unstack(
        scalar_output,
        PatternGroupData(
            matching_files=["source-1.tif", "source-2.tif"],
            main_data_stack=np.zeros((2, 2, 3), dtype=np.uint16),
        ),
    )

    assert output.slices == (scalar_output,)
    assert output.slice_contexts == (
        AlignedImageSliceContext.main_flow(
            "RGBImage",
            artifact_kind=ImageArtifactType.value,
        ),
    )
    assert image_payload_data(output.stack_payload).shape == (1, 2, 3, 3)
    stack_metadata = image_payload_metadata(output.stack_payload)
    assert stack_metadata.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert stack_metadata.source_channel_axis == 3


def test_pattern_group_runtime_uses_declared_output_slice_cardinality():
    declared_output = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((1, 4, 5), dtype=np.float32), None)

    runtime = object.__new__(PatternGroupRuntime)
    runtime.request = SimpleNamespace(
        compiled_group=compile_function_pattern(passthrough, {}, {}).default_group,
        execution_plan=SimpleNamespace(
            step_index=0,
            step_name="DeclaredOutput",
            output_memory_type=MEMORY_TYPE_NUMPY,
            device_id=0,
            artifact_inputs={},
            execution_group_scope=ComponentGroupScope.ungrouped(),
            artifact_outputs={},
        ),
        component_key=None,
    )

    output = runtime._validate_and_unstack(
        declared_output,
        PatternGroupData(
            matching_files=["source-1.tif", "source-2.tif"],
            main_data_stack=np.zeros((2, 4, 5), dtype=np.float32),
        ),
    )

    assert len(output.slices) == 1
    assert image_payload_data(output.slices[0]).shape == (4, 5)
    assert output.stack_payload is declared_output


def test_pattern_group_runtime_projects_nominal_object_label_stack():
    label_planes = np.asarray(
        (
            ((0, 1, 1), (0, 0, 0)),
            ((0, 0, 0), (2, 2, 0)),
        ),
        dtype=np.int32,
    )
    declared_output = ObjectLabelSet(
        name="SavedChildren",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    runtime = object.__new__(PatternGroupRuntime)
    runtime.request = SimpleNamespace(
        compiled_group=compile_function_pattern(passthrough, {}, {}).default_group,
        execution_plan=SimpleNamespace(
            step_index=0,
            step_name="DeclaredObjectLabels",
            output_memory_type=MEMORY_TYPE_NUMPY,
            device_id=0,
            artifact_inputs={},
            execution_group_scope=ComponentGroupScope.ungrouped(),
            artifact_outputs={},
        ),
        component_key=None,
    )

    output = runtime._validate_and_unstack(
        declared_output,
        PatternGroupData(
            matching_files=["source-anchor.tif"],
            main_data_stack=np.zeros((2, 2, 3), dtype=np.float32),
        ),
    )

    assert output.stack_payload is declared_output
    assert len(output.slices) == 2
    assert all(isinstance(value, ObjectLabelSet) for value in output.slices)
    assert [value.name for value in output.slices] == ["SavedChildren"] * 2
    assert [value.plane_axis for value in output.slices] == [None, None]
    assert [value.domain.scope for value in output.slices] == [
        ObjectLabelDomainScope.PAYLOAD,
        ObjectLabelDomainScope.PAYLOAD,
    ]
    assert [value.domain.declared_object_ids for value in output.slices] == [
        (1,),
        (2,),
    ]
    for output_slice, expected in zip(output.slices, label_planes, strict=True):
        np.testing.assert_array_equal(output_slice.labels, expected)


def test_module_runtime_adapter_records_declared_outputs_and_returns_main_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ContextStub()
    ambient_source = ImagePayloadMetadata(
        source_path="/input/A01_s1_w1.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
        },
    ).payload_with(np.zeros((2, 3), dtype=np.uint16), None)
    main_output = ImagePayloadMetadata(
        source_path="/input/A01_s1_w1.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
        },
    ).payload_with(np.full((2, 3), 7, dtype=np.uint16), None)
    recorded_output = ArtifactOutputPlan(
        name="ModuleImage",
        path="/memory/ModuleImage.pkl",
        artifact_type=ImageArtifactType,
    )

    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
        manages_artifact_outputs=True,
    )
    @artifact_outputs(ArtifactSpec.output("ModuleImage", ImageArtifactType))
    def module_step(image, *, runtime):
        del image, runtime
        context.filemanager.save(main_output, recorded_output.path, "memory")
        return main_output

    contextualization_plans = []
    original_for_output_plan = FunctionOutputContextStrategy.for_output_plan

    def track_contextualization(cls, output_plan):
        del cls
        contextualization_plans.append(output_plan)
        return original_for_output_plan(output_plan)

    monkeypatch.setattr(
        FunctionOutputContextStrategy,
        "for_output_plan",
        classmethod(track_contextualization),
    )

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=module_step,
            main_data_arg=ambient_source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (recorded_output,)},
        )
    )

    assert result is main_output
    assert contextualization_plans == []
    assert context.filemanager.load(recorded_output.path, "memory") is main_output


def test_execute_function_core_records_image_artifact_as_main_flow():
    context = ContextStub()
    source_data = np.arange(6, dtype=np.uint16).reshape(2, 3)
    source = ImagePayloadMetadata(
        source_path="/input/A01_s1_w1.tif",
        source_image_names=("OrigGreen",),
    ).payload_with(source_data)
    illumination_spec = ArtifactSpec.output("Illumination", ImageArtifactType)

    @artifact_outputs(illumination_spec)
    def calculate_illumination(image):
        return image + 1

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=calculate_illumination,
            main_data_arg=source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name=illumination_spec.name,
                    path="/memory/illumination.pkl",
                    artifact_type=ImageArtifactType,
                ),)},
        )
    )

    np.testing.assert_array_equal(image_payload_data(result), source_data + 1)
    stored = context.runtime_value_store.find(
        name=illumination_spec.name,
        axis_id=context.axis_id,
    )
    assert len(stored) == 1
    np.testing.assert_array_equal(
        image_payload_data(stored[0].value.data),
        source_data + 1,
    )
    stored_metadata = image_payload_metadata(stored[0].value.data)
    assert stored_metadata.source_image_names == (illumination_spec.name,)
    assert stored_metadata.source_provenance.represented_source_image_names == (
        illumination_spec.name,
        "OrigGreen",
    )


def test_execute_function_core_preserves_complete_main_output_source_identity_with_object_input():
    context = ContextStub()
    scalar_source = ImagePayloadMetadata(
        source_path="/input/A01_s001_w1_z001.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
            "z_index": "1",
        },
    ).payload_with(np.zeros((4, 5), dtype=np.uint16), None)

    @artifact_outputs(
        ArtifactSpec.output("labels", ObjectLabelsArtifactType),
    )
    def produce_labels(image):
        del image
        return ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(
                labels=np.ones((3, 4, 5), dtype=np.int32)
            )
        )

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=produce_labels,
            main_data_arg=scalar_source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name="labels",
                    path="/memory/labels.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                ),)},
        )
    )

    main_plane_metadata = (
        {"well": "A01", "site": "1", "channel": "1", "z_index": "1"},
        {"well": "A01", "site": "1", "channel": "1", "z_index": "2"},
        {"well": "A01", "site": "1", "channel": "1", "z_index": "3"},
    )
    main_source = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1.tif",) * 3,
            component_metadata=main_plane_metadata,
        )
    ).payload_with(np.ones((3, 4, 5), dtype=np.uint16), None)
    label_input_spec = ArtifactSpec.input(
        "labels",
        ObjectLabelsArtifactType,
        parameter_name="labels",
    )

    @artifact_inputs(label_input_spec)
    def passthrough_pixels(image, labels: ObjectLabelValue):
        del labels
        return np.asarray(image)

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=passthrough_pixels,
            main_data_arg=main_source,
            base_kwargs={},
            context=context,
            artifact_inputs={plan.ref(): plan for plan in (ArtifactInputPlan(
                    name=label_input_spec.name,
                    path="/memory/labels.pkl",
                    artifact_type=label_input_spec.artifact_type,
                ),)},
            artifact_outputs={},
        )
    )

    metadata = image_payload_metadata(result)
    assert (
        tuple(
            dict(item)
            for item in metadata.source_image_provenance_planes.component_metadata
        )
        == main_plane_metadata
    )


def test_execute_function_core_uses_object_input_source_for_image_artifact_output():
    context = ContextStub()
    plane_metadata = (
        {"well": "A01", "site": "1", "channel": "1", "z_index": "1"},
        {"well": "A01", "site": "1", "channel": "1", "z_index": "2"},
        {"well": "A01", "site": "1", "channel": "1", "z_index": "3"},
    )
    label_source = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1.tif",) * 3,
            component_metadata=plane_metadata,
        )
    ).payload_with(np.ones((3, 4, 5), dtype=np.uint16), None)

    @artifact_outputs(
        ArtifactSpec.output("labels", ObjectLabelsArtifactType),
    )
    def produce_labels(image):
        del image
        return ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(
                labels=np.ones((3, 4, 5), dtype=np.int32)
            )
        )

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=produce_labels,
            main_data_arg=label_source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name="labels",
                    path="/memory/labels.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                ),)},
        )
    )

    primary_source = ImagePayloadMetadata(
        source_path="/input/unrelated.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "3",
        },
    ).payload_with(np.zeros((3, 4, 5), dtype=np.uint16), None)

    label_input_spec = ArtifactSpec.input(
        "labels",
        ObjectLabelsArtifactType,
        parameter_name="labels",
    )
    label_image_spec = ArtifactSpec.output("label_image", ImageArtifactType)

    @artifact_inputs(label_input_spec)
    @artifact_outputs(label_image_spec)
    def labels_to_image(image, labels: ObjectLabelValue):
        del image
        return object_label_dense_array(labels, dtype=np.uint16)

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=labels_to_image,
            main_data_arg=primary_source,
            base_kwargs={},
            context=context,
            artifact_inputs={plan.ref(): plan for plan in (ArtifactInputPlan(
                    name="labels",
                    path="/memory/labels.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                ),)},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name="label_image",
                    path="/memory/label_image.pkl",
                    artifact_type=ImageArtifactType,
                    relations=(
                        GroupLineageSourceRelation(label_input_spec.ref()),
                    ),
                ),)},
        )
    )

    stored = context.runtime_value_store.find(name="label_image", axis_id="A01")
    assert len(stored) == 1
    metadata = image_payload_metadata(stored[0].value.data)
    assert (
        tuple(
            dict(item)
            for item in metadata.source_image_provenance_planes.component_metadata
        )
        == plane_metadata
    )


def test_execute_function_core_contextualizes_object_label_artifact():
    context = ContextStub()
    source = ImagePayloadMetadata(
        source_path="/input/01_POS002_D.TIF",
        source_component_metadata={
            "well": "01",
            "site": "POS002",
            "channel": "D",
        },
    ).payload_with(np.zeros((2, 3), dtype=np.uint16), None)

    @artifact_outputs(
        ArtifactSpec.output("nuclei", ObjectLabelsArtifactType),
    )
    def segment(image):
        del image
        return ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(
                labels=np.array([[0, 1, 0], [0, 0, 2]], dtype=np.int32)
            )
        )

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=segment,
            main_data_arg=source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name="nuclei",
                    path="/memory/nuclei.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                ),)},
        )
    )

    stored = context.runtime_value_store.find(name="nuclei", axis_id="A01")
    assert len(stored) == 1
    assert isinstance(stored[0].value.data, ObjectLabelSet)
    assert stored[0].value.data.name == "nuclei"
    assert stored[0].value.data.source_path == "/input/01_POS002_D.TIF"
    assert dict(stored[0].value.data.source_component_metadata) == {
        "well": "01",
        "site": "POS002",
        "channel": "D",
    }


def test_execute_function_core_aggregates_and_names_slice_aligned_object_labels():
    context = ContextStub()
    source_slices = tuple(
        ImagePayloadMetadata(
            source_path=f"/input/A01_s{site}_w1.tif",
            source_component_metadata={
                "well": "A01",
                "site": str(site),
                "channel": "1",
            },
        ).payload_with(np.zeros((4, 5), dtype=np.uint16))
        for site in (1, 2)
    )
    source = RuntimeSliceAlignedValues(source_slices)
    labels = RuntimeSliceAlignedValues(
        (
            np.pad(np.ones((2, 2), dtype=np.int32), ((0, 2), (0, 3))),
            np.pad(
                np.full((2, 2), 2, dtype=np.int32),
                ((2, 0), (3, 0)),
            ),
        )
    )
    output_spec = ArtifactSpec.output("cells", ObjectLabelsArtifactType)

    @artifact_outputs(output_spec)
    def segment(image):
        del image
        return labels

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=segment,
            main_data_arg=source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name=output_spec.name,
                    path="/memory/cells.pkl",
                    artifact_type=ObjectLabelsArtifactType,
                    variable_components=(AllComponents.SITE,),
                ),)},
            runtime_plane_count=2,
        )
    )

    [stored] = context.runtime_value_store.find(
        name=output_spec.name,
        axis_id=context.axis_id,
    )
    label_set = stored.value.data
    assert isinstance(label_set, ObjectLabelSet)
    assert label_set.name == output_spec.name
    assert label_set.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert label_set.declared_plane_count() == 2
    np.testing.assert_array_equal(
        label_set.labels,
        np.stack(
            tuple(
                labels.value_for_slice(index)
                for index in range(labels.slice_count)
            )
        ),
    )
    assert label_set.source_provenance.source_image_provenance_planes.paths == tuple(
        image_payload_metadata(item).source_path for item in source_slices
    )


@dataclass(frozen=True, slots=True)
class _NativeCountRow:
    slice_index: int
    cell_count: int


def test_execute_function_core_wraps_columnar_rows_with_compiled_measurement_identity():
    context = ContextStub()
    measurement_spec = ArtifactSpec.output(
        "cell_counts",
        MeasurementsArtifactType,
        relations=(ArtifactMeasurementSubjectRelation(),),
    )

    @artifact_outputs(measurement_spec)
    def count(image):
        return image, DataclassMeasurementColumnarRows((_NativeCountRow(0, 2),))

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=count,
            main_data_arg=ImagePayloadMetadata(
                source_path="/input/A01_s1_w1.tif"
            ).payload_with(np.zeros((4, 5), dtype=np.uint16)),
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name=measurement_spec.name,
                    path="/memory/cell-counts.pkl",
                    artifact_type=MeasurementsArtifactType,
                    relations=measurement_spec.relations,
                ),)},
        )
    )

    [stored] = context.runtime_value_store.find(
        name=measurement_spec.name,
        axis_id=context.axis_id,
    )
    table = stored.value.data
    assert isinstance(table, MeasurementTable)
    assert table.name == measurement_spec.name
    assert table.subject == MeasurementSubject(MeasurementScope.ARTIFACT)
    assert table.rows.fields == (
        FieldSpec("slice_index", int),
        FieldSpec("cell_count", int),
    )


def test_execute_function_core_rejects_nominal_measurement_subject_mismatch():
    context = ContextStub()
    measurement_spec = ArtifactSpec.output(
        "cell_counts",
        MeasurementsArtifactType,
        relations=(ArtifactMeasurementSubjectRelation(),),
    )

    @artifact_outputs(measurement_spec)
    def count(image):
        return image, MeasurementTable(
            name=measurement_spec.name,
            rows=DataclassMeasurementColumnarRows((_NativeCountRow(0, 2),)),
            subject=MeasurementSubject(MeasurementScope.IMAGE, "input"),
        )

    with pytest.raises(ValueError, match="declares subject"):
        _execute_function_core(
            CoreExecutionRequest(
                func_callable=count,
                main_data_arg=np.zeros((4, 5), dtype=np.uint16),
                base_kwargs={},
                context=context,
                artifact_inputs={},
                artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                        name=measurement_spec.name,
                        path="/memory/cell-counts.pkl",
                        artifact_type=MeasurementsArtifactType,
                        relations=measurement_spec.relations,
                    ),)},
            )
        )


def test_execute_function_core_loads_artifact_input_from_runtime_store_record():
    context = ContextStub()

    @artifact_outputs(
        ArtifactSpec.output("positions", SpecialArtifactType),
    )
    def produce(image):
        return (image, {"x": 1})

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=produce,
            main_data_arg=41,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                    name="positions",
                    path="/memory/positions.pkl",
                ),)},
        )
    )

    context.filemanager.memory._memory_store["/memory/positions.pkl"] = {
        "x": "from-vfs"
    }

    loaded_inputs = []
    positions_input_spec = ArtifactSpec.input(
        "positions",
        SpecialArtifactType,
        parameter_name="positions",
    )

    @artifact_inputs(positions_input_spec)
    def consume(image, positions):
        loaded_inputs.append(positions)
        return image

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=consume,
            main_data_arg=41,
            base_kwargs={},
            context=context,
            artifact_inputs={plan.ref(): plan for plan in (ArtifactInputPlan(
                    name=positions_input_spec.name,
                    path="/memory/positions.pkl",
                    artifact_type=positions_input_spec.artifact_type,
                ),)},
            artifact_outputs={},
        )
    )

    assert result == 41
    assert loaded_inputs == [{"x": 1}]


def test_execute_function_core_requires_store_record_even_when_vfs_payload_exists():
    context = ContextStub()
    context.filemanager.memory._memory_store["/memory/positions.pkl"] = {"x": 1}
    positions_input_spec = ArtifactSpec.input(
        "positions",
        SpecialArtifactType,
        parameter_name="positions",
    )

    @artifact_inputs(positions_input_spec)
    def consume(image, positions):
        return image

    with pytest.raises(RuntimeError, match="Missing RuntimeValueStore record"):
        _execute_function_core(
            CoreExecutionRequest(
                func_callable=consume,
                main_data_arg=41,
                base_kwargs={},
                context=context,
                artifact_inputs={plan.ref(): plan for plan in (ArtifactInputPlan(
                        name=positions_input_spec.name,
                        path="/memory/positions.pkl",
                        artifact_type=positions_input_spec.artifact_type,
                    ),)},
                artifact_outputs={},
            )
        )


def test_execute_function_core_requires_all_declared_artifact_values():
    context = ContextStub()

    @artifact_outputs(
        ArtifactSpec.output("measurements", MeasurementsArtifactType),
    )
    def analyze(image):
        return (image,)

    with pytest.raises(ValueError, match="trailing return count"):
        _execute_function_core(
            CoreExecutionRequest(
                func_callable=analyze,
                main_data_arg=41,
                base_kwargs={},
                context=context,
                artifact_inputs={},
                artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                        name="measurements",
                        path="/memory/measurements.pkl",
                        artifact_type=MeasurementsArtifactType,
                    ),)},
            )
        )


def test_execute_function_core_validates_artifact_kind():
    context = ContextStub()

    @artifact_outputs(
        ArtifactSpec.output("metadata", MetadataArtifactType),
    )
    def analyze(image):
        return (image, ["not", "metadata"])

    with pytest.raises(TypeError, match="expected metadata mapping"):
        _execute_function_core(
            CoreExecutionRequest(
                func_callable=analyze,
                main_data_arg=41,
                base_kwargs={},
                context=context,
                artifact_inputs={},
                artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                        name="metadata",
                        path="/memory/metadata.pkl",
                        artifact_type=MetadataArtifactType,
                    ),)},
            )
        )


def test_execute_function_core_validates_tuple_artifact_kind():
    context = ContextStub()

    @artifact_outputs(
        ArtifactSpec.output("nuclei", ObjectLabelsArtifactType),
    )
    def analyze(image):
        del image
        return {"not": "labels"}

    with pytest.raises(TypeError, match="no registered nominal strategy"):
        _execute_function_core(
            CoreExecutionRequest(
                func_callable=analyze,
                main_data_arg=41,
                base_kwargs={},
                context=context,
                artifact_inputs={},
                artifact_outputs={plan.ref(): plan for plan in (ArtifactOutputPlan(
                        name="nuclei",
                        path="/memory/nuclei.pkl",
                        artifact_type=ObjectLabelsArtifactType,
                    ),)},
            )
        )


def test_function_runtime_stacks_and_unstacks_explicit_runtime_slices():
    slices = [
        np.zeros((4, 5, 3), dtype=np.float32),
        np.ones((4, 5, 3), dtype=np.float32),
    ]

    stack = stack_runtime_slices(
        slices,
        "numpy",
        0,
    )
    unstacked = unstack_runtime_slices(
        stack,
        "numpy",
        0,
        expected_count=2,
    )

    assert stack.shape == (2, 4, 5, 3)
    assert [slice_data.shape for slice_data in unstacked] == [(4, 5, 3), (4, 5, 3)]
    np.testing.assert_array_equal(unstacked[1], slices[1])


def _declared_source_executor(
    spec: ArtifactSpec,
    *,
    source_binding_plan: CompiledSourceBindingPlan = CompiledSourceBindingPlan.empty(),
    main_flow_projection: MainFlowInputProjection = (
        MainFlowInputProjection.DECLARED_SOURCE_IMAGE
    ),
) -> FunctionCoreExecutor:
    @artifact_inputs(spec)
    def declared_source_origin(image):
        return image

    invocation = next(
        compile_function_pattern(
            declared_source_origin,
            {},
            {},
        ).iter_invocations()
    )
    edge = InvocationArtifactInputEdgePlan(
        key=InvocationArtifactInputProjectionKey(
            invocation_key=invocation.key,
            input_index=0,
        ),
        spec=spec,
        storage_plan=None,
        projection=None,
        consumes_main_flow=True,
        main_flow_projection=main_flow_projection,
    )
    invocation = invocation.with_artifact_input_edges((edge,))
    return FunctionCoreExecutor(
        runtime_scope=SimpleNamespace(
            context=ContextStub(),
            source_binding_plan=source_binding_plan,
            source_binding_context=SourceBindingRuntimeContext.empty(),
            axis_scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=None,
                value=None,
            ),
            execution_plan=SimpleNamespace(
                variable_components=(),
                source_load_plan=SourceLoadPlan(),
            ),
        ),
        invocation=invocation,
        artifacts=ComponentArtifactPlans(inputs={edge.key: edge}, outputs={}),
        group_key=None,
        plane_projection=RuntimePlaneProjection.stack(),
        main_data_arg=object(),
        source_memory_type=MEMORY_TYPE_NUMPY,
    )


def test_declared_source_payload_loads_exact_source_bound_ref_with_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = NamedSourceBinding(alias="OrigBlue")
    spec = binding.input_spec()
    source_binding_plan = CompiledSourceBindingPlan(bindings=(binding,))
    executor = _declared_source_executor(
        spec,
        source_binding_plan=source_binding_plan,
    )
    primary = ImagePayloadMetadata(
        source_path="/input/OrigGreen.tif",
        source_image_names=("OrigGreen",),
    ).payload_with(np.zeros((3, 4), dtype=np.float32))
    auxiliary = ImagePayloadMetadata(
        source_path="/input/OrigBlue.tif",
        source_image_names=("OrigBlue",),
    ).payload_with(np.ones((3, 4), dtype=np.float32))
    requested_refs = []

    def source_artifact_payload(
        request: RuntimeAdapterRequest,
        ref,
    ):
        assert request.source_payload is primary
        assert request.source_binding_plan.declares_artifact_ref(ref)
        requested_refs.append(ref)
        return auxiliary

    monkeypatch.setattr(
        RuntimeAdapterRequest,
        "source_artifact_payload",
        source_artifact_payload,
    )

    result = executor.declared_source_payload(
        spec.ref(),
        primary,
        loaded_artifact_payloads={},
    )

    assert result is auxiliary
    assert requested_refs == [spec.ref()]


def test_declared_source_payload_projects_only_exact_main_flow_ref() -> None:
    spec = ArtifactSpec.input("OrigGreen", ImageArtifactType)
    executor = _declared_source_executor(spec)
    primary = ImagePayloadMetadata(
        source_path="/input/OrigGreen.tif",
        source_image_names=("OrigGreen",),
    ).payload_with(np.zeros((3, 4), dtype=np.float32))

    result = executor.declared_source_payload(
        spec.ref(),
        primary,
        loaded_artifact_payloads={},
    )

    assert image_payload_metadata(result).source_image_names == ("OrigGreen",)


def test_declared_source_payload_preserves_compiled_complete_main_flow() -> None:
    spec = ArtifactSpec.input(
        "__openhcs_main_flow_step_3_default_1_tophat",
        ImageArtifactType,
    )
    executor = _declared_source_executor(
        spec,
        main_flow_projection=MainFlowInputProjection.COMPLETE_PAYLOAD,
    )
    primary = ImagePayloadMetadata(
        source_path="/input/axon.tif",
        source_image_names=("Axon",),
    ).payload_with(np.zeros((3, 4), dtype=np.float32))

    result = executor.declared_source_payload(
        spec.ref(),
        primary,
        loaded_artifact_payloads={},
    )

    assert result is primary


def test_declared_source_payload_prefers_exact_loaded_ref_over_main_flow() -> None:
    spec = ArtifactSpec.input("StoredImage", ImageArtifactType)
    executor = _declared_source_executor(spec)
    primary = ImagePayloadMetadata().payload_with(
        np.zeros((3, 4), dtype=np.float32)
    )
    stored = ImagePayloadMetadata(
        source_image_names=("StoredImage",),
    ).payload_with(np.ones((3, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="does not represent declared source image"):
        executor.declared_source_payload(
            spec.ref(),
            primary,
            loaded_artifact_payloads={},
        )

    assert (
        executor.declared_source_payload(
            spec.ref(),
            primary,
            loaded_artifact_payloads={spec.ref(): stored},
        )
        is stored
    )

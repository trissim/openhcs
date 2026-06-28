from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import numpy as np

from openhcs.constants.constants import MEMORY_TYPE_NUMPY
from openhcs.core.artifacts import (
    CROP_MASK_ARTIFACT_SIDECAR,
    ArtifactInputPlan,
    ArtifactKind,
    ArtifactOutputPlan,
    StepResult,
)
from openhcs.core.callable_contract import CallableContract, CallableMetadata
from openhcs.core.function_patterns import (
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
    FunctionInvocationKey,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_adapters import (
    runtime_adapter,
    runtime_adapter_spec_from_callable,
)
from openhcs.core.image_shapes import is_image_stack
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    SourceBindingRuntimeContext,
)
from openhcs.core.source_metadata import (
    SOURCE_PLANE_COUNT_FIELD,
    SOURCE_PLANE_INDEX_FIELD,
)
from openhcs.core.runtime_semantics import RuntimePlaneProjection
from openhcs.core.steps.function_runtime import (
    ComponentArtifactPlans,
    FunctionCoreExecutor,
    FunctionRuntimeScope,
)
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    stack_image_payload_context,
    unstack_image_payload_context,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ObjectLabelPayload,
    SourceImageProvenancePlanes,
    image_payload_data,
    image_payload_metadata,
    RuntimeImagePayloadContext,
    image_payload_mask,
)
from openhcs.processing.backends.assemblers.assemble_stack_cpu import assemble_stack_cpu


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
    artifact_inputs: Mapping[str, ArtifactInputPlan]
    artifact_outputs: Mapping[str, ArtifactOutputPlan]
    group_key: str = "default"


def _execute_function_core(request: CoreExecutionRequest):
    contract = CallableContract(
        func=request.func_callable,
        function_name=request.func_callable.__name__,
        module_name=request.func_callable.__module__,
        metadata=CallableMetadata(
            input_memory_type=MEMORY_TYPE_NUMPY,
            output_memory_type=MEMORY_TYPE_NUMPY,
            runtime_adapter=runtime_adapter_spec_from_callable(request.func_callable),
        ),
    )
    invocation = CompiledFunctionInvocation(
        key=FunctionInvocationKey.from_contract(
            contract,
            request.group_key,
            0,
        ),
        contract=contract,
        kwargs=tuple(request.base_kwargs.items()),
        artifact_input_keys=tuple(request.artifact_inputs),
        artifact_output_keys=tuple(request.artifact_outputs),
    )
    runtime_scope = FunctionRuntimeScope(
        context=request.context,
        execution_plan=SimpleNamespace(
            step_index=0,
            step_scope_id="test::function_step",
            step_name="test",
            axis_id=request.context.axis_id,
            input_memory_type=MEMORY_TYPE_NUMPY,
            device_id=0,
            source_binding_plan=CompiledSourceBindingPlan.empty(),
            variable_components=(),
            group_by_value=None,
            group_projects_runtime_plane=False,
        ),
        compiled_group=CompiledFunctionGroup(
            group_key=request.group_key,
            invocations=(invocation,),
        ),
        artifacts=ComponentArtifactPlans(
            inputs=request.artifact_inputs,
            outputs=request.artifact_outputs,
        ),
        source_binding_context=SourceBindingRuntimeContext.empty(),
        runtime_plane_index=0,
        runtime_plane_count=1,
    )
    group_key = invocation.key.runtime_group_key(runtime_scope.component_value)
    return FunctionCoreExecutor(
        main_data_arg=request.main_data_arg,
        source_memory_type=MEMORY_TYPE_NUMPY,
        runtime_scope=runtime_scope,
        invocation=invocation,
        artifacts=runtime_scope.artifacts.select_for_invocation(invocation),
        group_key=group_key,
        plane_projection=RuntimePlaneProjection.for_execution_group(
            group_key,
            plane_index=None,
            projects_runtime_plane=False,
        ),
    ).execute()


def test_function_core_passes_payload_data_to_array_callable_and_restores_context():
    tiles = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
    payload = RuntimeImagePayloadContext(
        tiles,
        mask=None,
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w1_z001_t001.tif",
        ),
    ).payload()
    positions = np.array([[0, 0], [4, 0]], dtype=np.float32)

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=assemble_stack_cpu,
            main_data_arg=payload,
            base_kwargs={
                "positions": positions,
                "blend_method": "none",
            },
            context=ContextStub(),
            artifact_inputs={},
            artifact_outputs={},
        )
    )

    assert image_payload_data(result).shape == (1, 4, 8)
    assert image_payload_data(result).dtype == tiles.dtype
    assert image_payload_metadata(result).source_path == (
        "/input/A01_s001_w1_z001_t001.tif"
    )


def test_crop_mask_sidecar_names_derive_from_core_artifact_role():
    assert CROP_MASK_ARTIFACT_SIDECAR.name_for("CroppedImage") == (
        "CroppedImage__crop_mask"
    )


def test_unstack_payload_context_slices_volume_stack_mask_with_volume_data():
    data = np.ones((1, 3, 4, 5), dtype=np.float32)
    mask = np.ones((1, 3, 4, 5), dtype=bool)
    payload = RuntimeImagePayloadContext(data, mask=mask, metadata = ImagePayloadMetadata()).payload()

    [slice_payload] = unstack_image_payload_context(payload, [data[0]])

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
    payload = RuntimeImagePayloadContext(
        data,
        mask=mask,
        metadata=metadata,
    ).payload()

    [slice_payload] = unstack_image_payload_context(payload, [data])
    slice_metadata = image_payload_metadata(slice_payload)

    np.testing.assert_array_equal(image_payload_data(slice_payload), data)
    np.testing.assert_array_equal(image_payload_mask(slice_payload), mask)
    assert slice_metadata.source_image_provenance_planes.count == 3
    assert tuple(
        dict(item)
        for item in slice_metadata.source_image_provenance_planes.component_metadata
    ) == plane_metadata


def test_unstack_payload_context_expands_indexed_scalar_source_for_preserved_stack():
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
    payload = RuntimeImagePayloadContext(data, mask=None, metadata=metadata).payload()

    [slice_payload] = unstack_image_payload_context(payload, [data])
    slice_metadata = image_payload_metadata(slice_payload)

    assert slice_metadata.source_image_provenance_planes.paths == (
        source_path,
        source_path,
        source_path,
    )
    assert tuple(
        dict(item)
        for item in slice_metadata.source_image_provenance_planes.component_metadata
    ) == (
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "1",
            SOURCE_PLANE_INDEX_FIELD: "0",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "2",
            SOURCE_PLANE_INDEX_FIELD: "1",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "3",
            SOURCE_PLANE_INDEX_FIELD: "2",
            SOURCE_PLANE_COUNT_FIELD: "3",
        },
    )
    assert dict(slice_metadata.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "3",
        SOURCE_PLANE_COUNT_FIELD: "3",
    }


def test_stack_payload_context_projects_single_payload_mask_to_stack_domain():
    stack = np.ones((1, 4, 5), dtype=np.float32)
    mask = np.zeros((1, 4, 5), dtype=bool)
    mask[:, 1:3, 1:3] = True
    payload = RuntimeImagePayloadContext(
        stack,
        mask=mask,
        metadata=ImagePayloadMetadata(source_path="/input/A01.tif"),
    ).payload()

    stacked = stack_image_payload_context((payload,), stack)

    assert image_payload_mask(stacked).shape == stack.shape
    np.testing.assert_array_equal(image_payload_mask(stacked), mask)


def test_stack_payload_context_preserves_single_volumetric_payload_identity():
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    plane_metadata = (
        {"well": "A01", "site": 1, "channel": 1, "z_index": 1},
        {"well": "A01", "site": 1, "channel": 1, "z_index": 2},
        {"well": "A01", "site": 1, "channel": 1, "z_index": 3},
    )
    payload = RuntimeImagePayloadContext(
        data,
        mask=None,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s001_w1.tif",) * 3,
                component_metadata=plane_metadata,
            )
        ),
    ).payload()

    stacked_payload = stack_image_payload_context((payload,), data[np.newaxis, ...])
    stacked_metadata = image_payload_metadata(stacked_payload)

    assert image_payload_data(stacked_payload).shape == (1, 3, 4, 5)
    assert tuple(
        dict(item)
        for item in stacked_metadata.source_image_provenance_planes.component_metadata
    ) == plane_metadata


def test_stack_payload_context_preserves_singleton_stack_payload_mask_domain():
    data = np.ones((1, 4, 5), dtype=np.float32)
    mask = np.ones_like(data, dtype=bool)
    payload = RuntimeImagePayloadContext(
        data,
        mask=mask,
        metadata=ImagePayloadMetadata(),
    ).payload()

    stacked_payload = stack_image_payload_context((payload,), data)

    np.testing.assert_array_equal(image_payload_mask(stacked_payload), mask)


def test_stack_payload_context_composes_single_image_slice_mask_axis():
    data = np.ones((4, 5), dtype=np.float32)
    stack = data[np.newaxis, ...]
    mask = np.ones_like(data, dtype=bool)
    payload = RuntimeImagePayloadContext(
        data,
        mask=mask,
        metadata=ImagePayloadMetadata(),
    ).payload()

    stacked_payload = stack_image_payload_context((payload,), stack)

    np.testing.assert_array_equal(
        image_payload_mask(stacked_payload),
        mask[np.newaxis, ...],
    )


def test_execute_function_core_saves_named_step_result_artifacts():
    context = ContextStub()

    def analyze(image):
        return StepResult(
            image=image + 1,
            artifacts={"measurements": [{"count": 2}]},
        )

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=analyze,
            main_data_arg=41,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={
                "measurements": ArtifactOutputPlan(
                    name="measurements",
                    path="/memory/measurements.pkl",
                )
            },
        )
    )

    assert result == 42
    assert context.filemanager.saved[
        ("/memory/measurements.pkl", "memory")
    ] == [{"count": 2}]
    stored = context.runtime_value_store.find(
        name="measurements",
        axis_id="A01",
    )
    assert len(stored) == 1
    assert stored[0].value.data == [{"count": 2}]


def test_execute_function_core_preserves_tuple_main_output_without_artifact_plan():
    context = ContextStub()
    first = RuntimeImagePayloadContext(
        np.full((4, 5), 1, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(source_image_names=("OrigRed",)),
    ).payload()
    second = RuntimeImagePayloadContext(
        np.full((4, 5), 2, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(source_image_names=("OrigGreen",)),
    ).payload()

    def split_outputs(image):
        del image
        return first, second

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=split_outputs,
            main_data_arg=np.zeros((2, 4, 5), dtype=np.float32),
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={},
        )
    )

    assert isinstance(result, AlignedImageStack)
    assert tuple(image_payload_data(payload)[0, 0] for payload in result.slices) == (
        1,
        2,
    )
    assert tuple(
        image_payload_metadata(payload).source_image_names
        for payload in result.slices
    ) == (("OrigRed",), ("OrigGreen",))


def test_execute_function_core_routes_exact_image_artifact_tuple_to_main_flow():
    context = ContextStub()
    red = RuntimeImagePayloadContext(
        np.full((4, 5), 1, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(source_image_names=("OrigRed",)),
    ).payload()
    green = RuntimeImagePayloadContext(
        np.full((4, 5), 2, dtype=np.float32),
        mask=None,
        metadata=ImagePayloadMetadata(source_image_names=("OrigGreen",)),
    ).payload()

    def corrected_images(image):
        del image
        return red, green

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=corrected_images,
            main_data_arg=np.zeros((2, 4, 5), dtype=np.float32),
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={
                "Red": ArtifactOutputPlan(
                    name="Red",
                    path="/memory/red.pkl",
                    kind=ArtifactKind.IMAGE,
                ),
                "Green": ArtifactOutputPlan(
                    name="Green",
                    path="/memory/green.pkl",
                    kind=ArtifactKind.IMAGE,
                ),
            },
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


def test_execute_function_core_saves_artifact_to_runtime_group_path():
    context = ContextStub()

    def analyze(image):
        return StepResult(
            image=image,
            artifacts={"measurements": [{"site": "2", "count": 3}]},
        )

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=analyze,
            main_data_arg=41,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={
                "measurements": ArtifactOutputPlan(
                    name="measurements",
                    path="/memory/A01_measurements.pkl",
                    kind=ArtifactKind.MEASUREMENTS,
                    group_keys=("1", "2"),
                    paths_by_group={
                        "1": "/memory/A01_s1_measurements.pkl",
                        "2": "/memory/A01_s2_measurements.pkl",
                    },
                )
            },
            group_key="2",
        )
    )

    assert context.filemanager.saved[
        ("/memory/A01_s2_measurements.pkl", "memory")
    ] == [{"site": "2", "count": 3}]
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
    source = RuntimeImagePayloadContext(
        np.zeros((2, 3), dtype=np.uint16),
        metadata=ImagePayloadMetadata(
            source_path="/input/01_POS002_D.TIF",
            source_component_metadata={
                "well": "01",
                "site": "POS002",
                "channel": "D",
            },
        ),
    mask = None).payload()

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
    ambient_source = RuntimeImagePayloadContext(
        np.zeros((2, 3), dtype=np.uint16),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s1_w1.tif",
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
            },
        ),
        mask=None,
    ).payload()
    adapter_output = RuntimeImagePayloadContext(
        np.ones((2, 3), dtype=np.uint16),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s1_w3.tif",
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "3",
            },
        ),
        mask=None,
    ).payload()

    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    def enhance(image, *, runtime):
        del image, runtime
        return adapter_output

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=enhance,
            main_data_arg=ambient_source,
            base_kwargs={},
            context=context,
            artifact_inputs={
                "source_image": ArtifactInputPlan(
                    name="source_image",
                    path="/memory/source_image.pkl",
                    kind=ArtifactKind.IMAGE,
                )
            },
            artifact_outputs={},
        )
    )

    metadata = image_payload_metadata(result)
    assert metadata.source_path == "/input/A01_s1_w3.tif"
    assert dict(metadata.source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "3",
    }


def test_execute_function_core_preserves_complete_main_output_source_identity_with_object_input():
    context = ContextStub()
    scalar_source = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.uint16),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w1_z001.tif",
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
                "z_index": "1",
            },
        ),
        mask=None,
    ).payload()

    def produce_labels(image):
        del image
        return StepResult(
            image=np.zeros((4, 5), dtype=np.uint16),
            artifacts={
                "labels": np.ones((3, 4, 5), dtype=np.int32),
            },
        )

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=produce_labels,
            main_data_arg=scalar_source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={
                "labels": ArtifactOutputPlan(
                    name="labels",
                    path="/memory/labels.pkl",
                    kind=ArtifactKind.OBJECT_LABELS,
                )
            },
        )
    )

    main_plane_metadata = (
        {"well": "A01", "site": "1", "channel": "1", "z_index": "1"},
        {"well": "A01", "site": "1", "channel": "1", "z_index": "2"},
        {"well": "A01", "site": "1", "channel": "1", "z_index": "3"},
    )
    main_source = RuntimeImagePayloadContext(
        np.ones((3, 4, 5), dtype=np.uint16),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s001_w1.tif",) * 3,
                component_metadata=main_plane_metadata,
            )
        ),
        mask=None,
    ).payload()

    def passthrough_pixels(image, labels):
        del labels
        return np.asarray(image)

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=passthrough_pixels,
            main_data_arg=main_source,
            base_kwargs={},
            context=context,
            artifact_inputs={
                "labels": ArtifactInputPlan(
                    name="labels",
                    path="/memory/labels.pkl",
                    kind=ArtifactKind.OBJECT_LABELS,
                )
            },
            artifact_outputs={},
        )
    )

    metadata = image_payload_metadata(result)
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == main_plane_metadata


def test_execute_function_core_uses_object_input_source_for_image_artifact_output():
    context = ContextStub()
    plane_metadata = (
        {"well": "A01", "site": "1", "channel": "1", "z_index": "1"},
        {"well": "A01", "site": "1", "channel": "1", "z_index": "2"},
        {"well": "A01", "site": "1", "channel": "1", "z_index": "3"},
    )
    label_source = RuntimeImagePayloadContext(
        np.ones((3, 4, 5), dtype=np.uint16),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s001_w1.tif",) * 3,
                component_metadata=plane_metadata,
            )
        ),
        mask=None,
    ).payload()

    def produce_labels(image):
        return StepResult(
            image=image,
            artifacts={"labels": np.ones((3, 4, 5), dtype=np.int32)},
        )

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=produce_labels,
            main_data_arg=label_source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={
                "labels": ArtifactOutputPlan(
                    name="labels",
                    path="/memory/labels.pkl",
                    kind=ArtifactKind.OBJECT_LABELS,
                )
            },
        )
    )

    primary_source = RuntimeImagePayloadContext(
        np.zeros((3, 4, 5), dtype=np.uint16),
        metadata=ImagePayloadMetadata(
            source_path="/input/unrelated.tif",
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "3",
            },
        ),
        mask=None,
    ).payload()

    def labels_to_image(image, labels):
        del image
        return (np.asarray(labels, dtype=np.uint16),)

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=labels_to_image,
            main_data_arg=primary_source,
            base_kwargs={},
            context=context,
            artifact_inputs={
                "labels": ArtifactInputPlan(
                    name="labels",
                    path="/memory/labels.pkl",
                    kind=ArtifactKind.OBJECT_LABELS,
                )
            },
            artifact_outputs={
                "label_image": ArtifactOutputPlan(
                    name="label_image",
                    path="/memory/label_image.pkl",
                    kind=ArtifactKind.IMAGE,
                )
            },
        )
    )

    stored = context.runtime_value_store.find(name="label_image", axis_id="A01")
    assert len(stored) == 1
    metadata = image_payload_metadata(stored[0].value.data)
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == plane_metadata


def test_execute_function_core_contextualizes_bare_object_label_artifact():
    context = ContextStub()
    source = RuntimeImagePayloadContext(
        np.zeros((2, 3), dtype=np.uint16),
        metadata=ImagePayloadMetadata(
            source_path="/input/01_POS002_D.TIF",
            source_component_metadata={
                "well": "01",
                "site": "POS002",
                "channel": "D",
            },
        ),
    mask = None).payload()

    def segment(image):
        return StepResult(
            image=image,
            artifacts={
                "nuclei": np.array([[0, 1, 0], [0, 0, 2]], dtype=np.int32),
            },
        )

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=segment,
            main_data_arg=source,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={
                "nuclei": ArtifactOutputPlan(
                    name="nuclei",
                    path="/memory/nuclei.pkl",
                    kind=ArtifactKind.OBJECT_LABELS,
                )
            },
        )
    )

    stored = context.runtime_value_store.find(name="nuclei", axis_id="A01")
    assert len(stored) == 1
    assert isinstance(stored[0].value.data, ObjectLabelPayload)
    assert stored[0].value.data.source_path == "/input/01_POS002_D.TIF"
    assert dict(stored[0].value.data.source_component_metadata) == {
        "well": "01",
        "site": "POS002",
        "channel": "D",
    }


def test_execute_function_core_loads_artifact_input_from_vfs_via_store_record():
    context = ContextStub()

    def produce(image):
        return StepResult(image=image, artifacts={"positions": {"x": 1}})

    _execute_function_core(
        CoreExecutionRequest(
            func_callable=produce,
            main_data_arg=41,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={
                "positions": ArtifactOutputPlan(
                    name="positions",
                    path="/memory/positions.pkl",
                )
            },
        )
    )

    context.filemanager.memory._memory_store["/memory/positions.pkl"] = {
        "x": "from-vfs"
    }

    loaded_inputs = []

    def consume(image, positions):
        loaded_inputs.append(positions)
        return image

    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=consume,
            main_data_arg=41,
            base_kwargs={},
            context=context,
            artifact_inputs={
                "positions": ArtifactInputPlan(
                    name="positions",
                    path="/memory/positions.pkl",
                )
            },
            artifact_outputs={},
        )
    )

    assert result == 41
    assert loaded_inputs == [{"x": "from-vfs"}]


def test_execute_function_core_refuses_direct_vfs_artifact_input_fallback():
    context = ContextStub()
    context.filemanager.memory._memory_store["/memory/positions.pkl"] = {"x": 1}

    def consume(image, positions):
        return image

    with pytest.raises(RuntimeError, match="Refusing direct VFS fallback"):
        _execute_function_core(
            CoreExecutionRequest(
                func_callable=consume,
                main_data_arg=41,
                base_kwargs={},
                context=context,
                artifact_inputs={
                    "positions": ArtifactInputPlan(
                        name="positions",
                        path="/memory/positions.pkl",
                    )
                },
                artifact_outputs={},
            )
        )


def test_execute_function_core_requires_planned_step_result_artifacts():
    context = ContextStub()

    def analyze(image):
        return StepResult(image=image, artifacts={})

    with pytest.raises(ValueError, match="planned artifact 'measurements'"):
        _execute_function_core(
            CoreExecutionRequest(
                func_callable=analyze,
                main_data_arg=41,
                base_kwargs={},
                context=context,
                artifact_inputs={},
                artifact_outputs={
                    "measurements": ArtifactOutputPlan(
                        name="measurements",
                        path="/memory/measurements.pkl",
                    )
                },
            )
        )


def test_execute_function_core_validates_step_result_artifact_kind():
    context = ContextStub()

    def analyze(image):
        return StepResult(image=image, artifacts={"metadata": ["not", "metadata"]})

    with pytest.raises(TypeError, match="expected metadata mapping"):
        _execute_function_core(
            CoreExecutionRequest(
                func_callable=analyze,
                main_data_arg=41,
                base_kwargs={},
                context=context,
                artifact_inputs={},
                artifact_outputs={
                    "metadata": ArtifactOutputPlan(
                        name="metadata",
                        path="/memory/metadata.pkl",
                        kind=ArtifactKind.METADATA,
                    )
                },
            )
        )


def test_execute_function_core_validates_tuple_artifact_kind():
    context = ContextStub()

    def analyze(image):
        return image, {"not": "labels"}

    with pytest.raises(TypeError, match="Object-label output must be"):
        _execute_function_core(
            CoreExecutionRequest(
                func_callable=analyze,
                main_data_arg=41,
                base_kwargs={},
                context=context,
                artifact_inputs={},
                artifact_outputs={
                    "nuclei": ArtifactOutputPlan(
                        name="nuclei",
                        path="/memory/nuclei.pkl",
                        kind=ArtifactKind.OBJECT_LABELS,
                    )
                },
        )
    )


def test_function_runtime_stacks_and_unstacks_color_image_slices():
    slices = [
        np.zeros((4, 5, 3), dtype=np.float32),
        np.ones((4, 5, 3), dtype=np.float32),
    ]

    stack = ImageStackLayout.for_slices(slices).stack(
        slices=slices,
        memory_type="numpy",
        gpu_id=0,
    )
    unstacked = ImageStackLayout.for_stack(stack).unstack(
        array=stack,
        memory_type="numpy",
        gpu_id=0,
    )

    assert is_image_stack(stack)
    assert stack.shape == (2, 4, 5, 3)
    assert [slice_data.shape for slice_data in unstacked] == [(4, 5, 3), (4, 5, 3)]
    np.testing.assert_array_equal(unstacked[1], slices[1])

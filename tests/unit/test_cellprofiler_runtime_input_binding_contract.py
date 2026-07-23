"""Focused gates for the CellProfiler runtime input-binding boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from openhcs.constants.constants import AllComponents, Backend
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    GroupLineageSourceRelation,
    ImageArtifactType,
    InputStackBroadcastSourceRelation,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
    RuntimeExecutionAxisScope,
)
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.runtime_image_values import ImagePayloadMetadata, image_payload_data
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis, RuntimePlaneProjection
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.core.source_bindings import CompiledSourceBindingPlan
from openhcs.interop.cellprofiler.module_callable_abi import (
    CellProfilerModuleCallableABI,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)
from openhcs.processing.backends.cellprofiler import identify_primary_objects
from openhcs.processing.backends.cellprofiler.watershed import watershed_cellprofiler4

from tests.unit.cellprofiler_runtime_test_support import (
    cellprofiler_runtime_adapter_for_test,
    cellprofiler_runtime_input_edge_for_test,
)


def _contract(func, inputs: tuple[ArtifactSpec, ...] = ()) -> CallableContract:
    contract = CallableContract.from_callable(func)
    return replace(
        contract,
        module_name="RuntimeInputBinding",
        metadata=replace(contract.metadata, artifact_inputs=inputs),
    )


def test_binding_request_requires_contract_on_runtime_adapter_owner() -> None:
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
    )

    with pytest.raises(TypeError, match="requires the compiled CallableContract"):
        RuntimeInputBindingRequest(
            adapter=adapter,
            kwargs={},
            current_image=np.zeros((2, 2), dtype=np.float32),
        )


def test_binding_request_rejects_edges_from_another_callable_contract() -> None:
    def consume(image: np.ndarray) -> np.ndarray:
        return image

    spec = ArtifactSpec.input("Mask", ImageArtifactType)
    edge = cellprofiler_runtime_input_edge_for_test(
        ArtifactInputPlan(
            spec.name,
            "/memory/mask",
            artifact_type=spec.artifact_type,
        ),
        spec=spec,
        invocation_scope=ComponentGroupScope.ungrouped(),
        producer_selection_scope=ComponentGroupScope.ungrouped(),
        component_scopes=(),
        consumer_variable_components=(),
    )
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        callable_contract=_contract(consume),
        artifact_inputs={edge.key: edge},
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
    )

    with pytest.raises(ValueError, match="not declared with sufficient occurrence"):
        RuntimeInputBindingRequest(
            adapter=adapter,
            kwargs={},
            current_image=np.zeros((2, 2), dtype=np.float32),
        )


def test_runtime_adapter_contract_owns_active_input_selection() -> None:
    def consume(image: np.ndarray) -> np.ndarray:
        return image

    first = ArtifactSpec.input("First", ImageArtifactType)
    second = ArtifactSpec.input("Second", ImageArtifactType)
    edge = cellprofiler_runtime_input_edge_for_test(
        ArtifactInputPlan(
            second.name,
            "/memory/second",
            artifact_type=second.artifact_type,
        ),
        spec=second,
        invocation_scope=ComponentGroupScope.ungrouped(),
        producer_selection_scope=ComponentGroupScope.ungrouped(),
        component_scopes=(),
        consumer_variable_components=(),
    )
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        callable_contract=_contract(consume, (first, second)),
        artifact_inputs={edge.key: edge},
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
    )

    request = RuntimeInputBindingRequest(
        adapter=adapter,
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )

    assert request.declared_inputs.specs == (second,)


def test_declared_input_occurrence_owns_callable_parameter_binding() -> None:
    def consume(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del mask
        return image

    spec = ArtifactSpec.input(
        "Mask",
        ImageArtifactType,
        parameter_name="mask",
    )
    path = "/memory/mask"
    input_plan = ArtifactInputPlan(
        "Mask",
        path,
        artifact_type=ImageArtifactType,
    )
    edge = cellprofiler_runtime_input_edge_for_test(
        input_plan,
        spec=spec,
        invocation_scope=ComponentGroupScope.ungrouped(),
        producer_selection_scope=ComponentGroupScope.ungrouped(),
        component_scopes=(),
        consumer_variable_components=(),
    )
    payload = ImagePayloadMetadata().payload_with(
        np.ones((2, 2), dtype=np.float32),
        None,
    )
    output_plan = ArtifactOutputPlan(
        "Mask",
        path,
        artifact_type=ImageArtifactType,
    )
    store = RuntimeValueStore()
    store.replace(
        RuntimeValue.normalize(output_plan, payload, axis_id="A01"),
        path=path,
        backend=Backend.MEMORY.value,
    )
    contract = _contract(consume, (spec,))
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        callable_contract=contract,
        artifact_inputs={edge.key: edge},
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
    )

    bound = RuntimeInputBindingRequest(
        adapter=adapter,
        kwargs={},
        current_image=payload,
    ).bind_parameters()

    assert tuple(bound) == ("mask",)
    np.testing.assert_array_equal(image_payload_data(bound["mask"]), 1.0)


def test_stack_broadcast_input_projects_selected_plane_and_preserves_stack() -> None:
    def consume(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del mask
        return image

    source = ArtifactSpec.input("Membrane", ImageArtifactType)
    mask = ArtifactSpec.input(
        "MonolayerMask",
        ImageArtifactType,
        relations=(InputStackBroadcastSourceRelation(source=source.ref()),),
        parameter_name="mask",
    )
    path = "/memory/monolayer-mask"
    input_plan = ArtifactInputPlan(
        mask.name,
        path,
        artifact_type=ImageArtifactType,
    )
    edge = cellprofiler_runtime_input_edge_for_test(
        input_plan,
        spec=mask,
        invocation_scope=ComponentGroupScope.ungrouped(),
        producer_selection_scope=ComponentGroupScope.ungrouped(),
        component_scopes=(),
        consumer_variable_components=(AllComponents.Z_INDEX,),
    )
    mask_plane_metadata = tuple(
        {
            "well": "A01",
            "site": "1",
            "channel": "2",
            "z_index": str(index + 1),
        }
        for index in range(3)
    )
    mask_provenance = SourceImageProvenancePlanes.from_components(
        paths=tuple(f"/plate/A01_s1_w2_z{index + 1:03}.tif" for index in range(3)),
        component_metadata=mask_plane_metadata,
    )
    mask_metadata = ImagePayloadMetadata(
        source_image_names=(mask.name,),
        source_image_provenance_planes=mask_provenance,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    mask_payload = mask_metadata.payload_with(
        np.stack(
            tuple(np.full((2, 2), index + 1, dtype=np.float32) for index in range(3))
        ),
        None,
    )
    store = RuntimeValueStore()
    store.replace(
        RuntimeValue.normalize(
            ArtifactOutputPlan(mask.name, path, artifact_type=ImageArtifactType),
            mask_payload,
            axis_id="A01",
        ),
        path=path,
        backend=Backend.MEMORY.value,
    )
    contract = _contract(consume, (mask,))
    source_plane_metadata = tuple(
        {
            "well": "A01",
            "site": "1",
            "channel": "0",
            "z_index": str(index + 1),
        }
        for index in range(3)
    )
    source_stack = ImagePayloadMetadata(
        source_image_names=(source.name,),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(
                f"/plate/A01_s1_w0_z{index + 1:03}.tif" for index in range(3)
            ),
            component_metadata=source_plane_metadata,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.zeros((3, 2, 2), dtype=np.float32), None)
    source_binding_plan = CompiledSourceBindingPlan(
        source_stack_components=(
            AllComponents.CHANNEL,
            AllComponents.Z_INDEX,
        )
    )
    stack_adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        callable_contract=contract,
        artifact_inputs={edge.key: edge},
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
        plane_projection=RuntimePlaneProjection.stack(3),
        source_binding_plan=source_binding_plan,
    )

    stack_bound = RuntimeInputBindingRequest(
        adapter=stack_adapter,
        kwargs={},
        current_image=source_stack,
    ).bind_parameters()

    np.testing.assert_array_equal(
        image_payload_data(stack_bound["mask"]), image_payload_data(mask_payload)
    )

    selected_stack_adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        callable_contract=contract,
        artifact_inputs={edge.key: edge},
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
        plane_projection=RuntimePlaneProjection.selected(1, 3),
        source_binding_plan=source_binding_plan,
    )

    selected_stack_bound = RuntimeInputBindingRequest(
        adapter=selected_stack_adapter,
        kwargs={},
        current_image=source_stack,
    ).bind_parameters()

    np.testing.assert_array_equal(
        image_payload_data(selected_stack_bound["mask"]),
        2,
    )

    source_payloads = tuple(
        ImagePayloadMetadata(
            source_path=f"/plate/A01_s1_w0_z{index + 1:03}.tif",
            source_component_metadata=source_plane_metadata[index],
            source_image_names=(source.name,),
        ).payload_with(np.zeros((2, 2), dtype=np.float32), None)
        for index in range(3)
    )

    for index, source_payload in enumerate(source_payloads):
        adapter = cellprofiler_runtime_adapter_for_test(
            runtime_value_store=store,
            callable_contract=contract,
            artifact_inputs={edge.key: edge},
            axis_scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=None,
                value=None,
            ),
            plane_projection=RuntimePlaneProjection.selected(index, 3),
            source_binding_plan=source_binding_plan,
        )

        bound = RuntimeInputBindingRequest(
            adapter=adapter,
            kwargs={},
            current_image=source_payload,
        ).bind_parameters()

        np.testing.assert_array_equal(image_payload_data(bound["mask"]), index + 1)


def test_stack_broadcast_input_collapses_singleton_for_larger_source_stack() -> None:
    def consume(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del mask
        return image

    source = ArtifactSpec.input("MembInvertRemoveHoles", ImageArtifactType)
    mask = ArtifactSpec.input(
        "MonolayerMask",
        ImageArtifactType,
        relations=(InputStackBroadcastSourceRelation(source=source.ref()),),
        parameter_name="mask",
    )
    path = "/memory/monolayer-mask"
    input_plan = ArtifactInputPlan(
        mask.name,
        path,
        artifact_type=ImageArtifactType,
    )
    edge = cellprofiler_runtime_input_edge_for_test(
        input_plan,
        spec=mask,
        invocation_scope=ComponentGroupScope.ungrouped(),
        producer_selection_scope=ComponentGroupScope.ungrouped(),
        component_scopes=(),
        consumer_variable_components=(AllComponents.Z_INDEX,),
    )
    mask_payload = ImagePayloadMetadata(
        source_image_names=(mask.name,),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/plate/A01_s1_w2_z001.tif",),
            component_metadata=(
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "2",
                    "z_index": "1",
                },
            ),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((1, 2, 2), dtype=np.float32), None)
    store = RuntimeValueStore()
    store.replace(
        RuntimeValue.normalize(
            ArtifactOutputPlan(mask.name, path, artifact_type=ImageArtifactType),
            mask_payload,
            axis_id="A01",
        ),
        path=path,
        backend=Backend.MEMORY.value,
    )
    contract = _contract(consume, (mask,))
    source_plane_metadata = tuple(
        {
            "well": "A01",
            "site": "1",
            "channel": "0",
            "z_index": str(index + 1),
        }
        for index in range(3)
    )
    source_stack = ImagePayloadMetadata(
        source_image_names=(source.name,),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(
                f"/plate/A01_s1_w0_z{index + 1:03}.tif" for index in range(3)
            ),
            component_metadata=source_plane_metadata,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.zeros((3, 2, 2), dtype=np.float32), None)
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        callable_contract=contract,
        artifact_inputs={edge.key: edge},
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
        plane_projection=RuntimePlaneProjection.stack(3),
        source_binding_plan=CompiledSourceBindingPlan(
            source_stack_components=(
                AllComponents.CHANNEL,
                AllComponents.Z_INDEX,
            )
        ),
    )

    bound = RuntimeInputBindingRequest(
        adapter=adapter,
        kwargs={},
        current_image=source_stack,
    ).bind_parameters()

    np.testing.assert_array_equal(image_payload_data(bound["mask"]), 1.0)


def _artifact_availability_adapter(
    *specs: ArtifactSpec,
):
    def consume(
        image: np.ndarray,
        labels: object | None = None,
        neighbor_labels: object | None = None,
    ) -> np.ndarray:
        del labels, neighbor_labels
        return image

    input_plan = ArtifactInputPlan(
        "Nuclei",
        "/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    edges = tuple(
        cellprofiler_runtime_input_edge_for_test(
            input_plan,
            spec=spec,
            input_index=input_index,
            invocation_scope=ComponentGroupScope.ungrouped(),
            producer_selection_scope=ComponentGroupScope.ungrouped(),
            component_scopes=(),
            consumer_variable_components=(),
        )
        for input_index, spec in enumerate(specs)
    )
    return cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        callable_contract=_contract(consume, specs),
        artifact_inputs={edge.key: edge for edge in edges},
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
    )


def test_artifact_availability_accepts_one_exact_input_occurrence() -> None:
    labels = ArtifactSpec.input(
        "Nuclei",
        ObjectLabelsArtifactType,
        parameter_name="labels",
    )
    adapter = _artifact_availability_adapter(labels)

    adapter.require_artifact_available(
        name="Nuclei",
        kind=ObjectLabelsArtifactType,
    )

    assert tuple(edge.key.input_index for edge in adapter.request.artifact_inputs.values()) == (
        0,
    )


def test_artifact_availability_preserves_repeated_exact_role_occurrences() -> None:
    measured = ArtifactSpec.input(
        "Nuclei",
        ObjectLabelsArtifactType,
        parameter_name="labels",
    )
    neighbors = ArtifactSpec.input(
        "Nuclei",
        ObjectLabelsArtifactType,
        parameter_name="neighbor_labels",
    )
    adapter = _artifact_availability_adapter(measured, neighbors)

    adapter.require_artifact_available(
        name="Nuclei",
        kind=ObjectLabelsArtifactType,
    )

    assert tuple(edge.key.input_index for edge in adapter.request.artifact_inputs.values()) == (
        0,
        1,
    )
    assert tuple(
        edge.spec.parameter_name for edge in adapter.request.artifact_inputs.values()
    ) == ("labels", "neighbor_labels")


def test_compiled_edges_preserve_postponed_sequence_parameter_type() -> None:
    def consume(
        image: np.ndarray,
        masks: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        del masks
        return image

    first = ArtifactSpec.input(
        "MaskA",
        ImageArtifactType,
        parameter_name="masks",
    )
    second = ArtifactSpec.input(
        "MaskB",
        ImageArtifactType,
        parameter_name="masks",
    )
    store = RuntimeValueStore()
    edges = {}
    for index, spec in enumerate((first, second), start=1):
        path = f"/memory/{spec.name}"
        input_plan = ArtifactInputPlan(
            spec.name,
            path,
            artifact_type=ImageArtifactType,
        )
        edge = cellprofiler_runtime_input_edge_for_test(
            input_plan,
            spec=spec,
            input_index=index - 1,
            invocation_scope=ComponentGroupScope.ungrouped(),
            producer_selection_scope=ComponentGroupScope.ungrouped(),
            component_scopes=(),
            consumer_variable_components=(),
        )
        edges[edge.key] = edge
        output_plan = ArtifactOutputPlan(
            spec.name,
            path,
            artifact_type=ImageArtifactType,
        )
        payload = ImagePayloadMetadata().payload_with(
            np.full((2, 2), index, dtype=np.float32),
            None,
        )
        store.replace(
            RuntimeValue.normalize(output_plan, payload, axis_id="A01"),
            path=path,
            backend=Backend.MEMORY.value,
        )

    contract = _contract(consume, (first, second))
    current_image = ImagePayloadMetadata().payload_with(
        np.zeros((2, 2), dtype=np.float32),
        None,
    )
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        callable_contract=contract,
        artifact_inputs=edges,
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
    )

    bound = RuntimeInputBindingRequest(
        adapter=adapter,
        kwargs={},
        current_image=current_image,
    ).bind_parameters()

    assert isinstance(bound["masks"], tuple)
    assert len(bound["masks"]) == 2


def test_callable_abi_consumes_callable_contract_without_partition_wrapper() -> None:
    def process(image: np.ndarray) -> np.ndarray:
        return image

    source = ArtifactSpec.input("Input", ImageArtifactType)
    output = ArtifactSpec.output(
        "Output",
        ImageArtifactType,
        relations=(GroupLineageSourceRelation(source=source.ref()),),
    )
    contract = _contract(process, (source,))
    contract = replace(
        contract,
        metadata=replace(contract.metadata, artifact_outputs=(output,)),
    )

    CellProfilerModuleCallableABI.validate_callable_artifact_abi(process, contract)


def test_callable_abi_keeps_main_flow_artifacts_in_trailing_slots() -> None:
    source = ArtifactSpec.input("Input", ImageArtifactType)
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)
    labels = ArtifactSpec.output(
        "Objects",
        ObjectLabelsArtifactType,
        relations=(GroupLineageSourceRelation(source=source.ref()),),
    )

    for func in (identify_primary_objects, watershed_cellprofiler4):
        contract = _contract(func, (source,))
        contract = replace(
            contract,
            metadata=replace(
                contract.metadata,
                artifact_outputs=(measurements, labels),
            ),
        )

        CellProfilerModuleCallableABI.validate_callable_artifact_abi(func, contract)


def test_owned_binding_files_do_not_restore_partition_or_special_input_mirrors() -> None:
    project_root = Path(__file__).parents[2]
    source = "\n".join(
        (project_root / relative).read_text(encoding="utf-8")
        for relative in (
            "openhcs/interop/cellprofiler/runtime/artifact_binding.py",
            "openhcs/interop/cellprofiler/module_callable_abi.py",
        )
    )
    for forbidden in (
        "ModuleArtifactContract",
        "ArtifactInputContractPartition",
        "ArtifactInputPartitionStrategy",
        "special_image_inputs",
        "special_input_specs",
        "special_input_parameter",
    ):
        assert forbidden not in source

"""Focused regressions for compiled invocation input source-context ownership."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
    RuntimeExecutionAxisScope,
)
from openhcs.core.component_set import ComponentSet
from openhcs.constants.constants import Backend
from openhcs.core.function_patterns import (
    CompiledFunctionPattern,
    FunctionInvocationKey,
    MainFlowInputProjection,
    compile_function_pattern,
)
from openhcs.core.invocation_artifacts import unnamed_main_flow_artifact_name
from openhcs.core.pipeline.function_contracts import artifact_inputs
from openhcs.core.pipeline.path_planner import PathPlanner, PathPlannerArtifactStage
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
)
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    NamedSourceBinding,
    SourceProjectionRole,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)
from tests.unit.cellprofiler_runtime_test_support import (
    cellprofiler_runtime_adapter_for_test,
)


def _compile_source_context_edges(
    *,
    store_combined_image: bool = False,
) -> tuple[
    CallableContract,
    ArtifactSpec,
    ArtifactSpec,
    CompiledFunctionPattern,
]:
    plate_template = ArtifactSpec.input("PlateTemplate", ImageArtifactType)
    combined_image = ArtifactSpec.input("CombinedImage", ImageArtifactType)

    @artifact_inputs(plate_template, combined_image)
    def align_like(image: np.ndarray) -> np.ndarray:
        return image

    compiled = compile_function_pattern(align_like, {}, {})
    input_plans = {}
    if store_combined_image:
        input_plans[combined_image.ref()] = ArtifactInputPlan(
            combined_image.name,
            "/memory/CombinedImage",
            artifact_type=ImageArtifactType,
        )
    compiled = PathPlannerArtifactStage(
        PathPlanner.__new__(PathPlanner)
    ).compile_invocation_input_edges(
        compiled,
        artifact_inputs=input_plans,
        relation_source_scopes={},
        execution_group_scope=ComponentGroupScope.ungrouped(),
        consumer_variable_components=ComponentSet(),
        main_flow_artifacts=ArtifactSpecCollection((combined_image,)),
    )
    invocation = next(compiled.iter_invocations())
    return (
        replace(invocation.contract, module_name="AlignLike"),
        plate_template,
        combined_image,
        compiled,
    )


def test_compiled_input_edges_own_exact_main_flow_membership() -> None:
    _, plate_template, combined_image, compiled = _compile_source_context_edges()
    edges = next(compiled.iter_invocations()).artifact_input_edges

    assert tuple(edge.spec for edge in edges) == (plate_template, combined_image)
    assert tuple(edge.storage_plan for edge in edges) == (None, None)
    assert tuple(edge.consumes_main_flow for edge in edges) == (False, True)
    assert tuple(edge.main_flow_projection for edge in edges) == (
        None,
        MainFlowInputProjection.COMPLETE_PAYLOAD,
    )


def test_implicit_native_main_flow_consumes_complete_payload_not_source_alias() -> None:
    native_key = FunctionInvocationKey("percentile_normalize", "default", 0)
    cursor = ArtifactSpec.input(
        unnamed_main_flow_artifact_name(0, native_key),
        ImageArtifactType,
    )

    @artifact_inputs(cursor)
    def threshold_like(image: np.ndarray) -> np.ndarray:
        return image

    compiled = compile_function_pattern(threshold_like, {}, {})
    compiled = PathPlannerArtifactStage(
        PathPlanner.__new__(PathPlanner)
    ).compile_invocation_input_edges(
        compiled,
        artifact_inputs={},
        relation_source_scopes={},
        execution_group_scope=ComponentGroupScope.ungrouped(),
        consumer_variable_components=ComponentSet(),
        main_flow_artifacts=ArtifactSpecCollection((cursor,)),
    )
    invocation = next(compiled.iter_invocations())
    edge = invocation.artifact_input_edges[0]
    source_payload = ImagePayloadMetadata(
        source_image_names=("Hoechst",),
    ).payload_with(np.full((2, 3), 11.0, dtype=np.float32), None)
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        callable_contract=replace(invocation.contract, module_name="Threshold"),
        artifact_inputs={edge.key: edge},
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "R04C09",
            component=None,
            value=None,
        ),
    )

    request = RuntimeInputBindingRequest(
        adapter=adapter,
        kwargs={},
        current_image=source_payload,
    ).artifact_request(edge)

    assert edge.consumes_main_flow is True
    assert edge.main_flow_projection is MainFlowInputProjection.COMPLETE_PAYLOAD
    assert request.value is source_payload
    assert image_payload_metadata(request.value).source_image_names == ("Hoechst",)


def test_multiple_main_flow_images_keep_declared_source_projection() -> None:
    first = ArtifactSpec.input("Hoechst", ImageArtifactType)
    second = ArtifactSpec.input("MAP2", ImageArtifactType)

    @artifact_inputs(first, second)
    def consume_both(image: np.ndarray) -> np.ndarray:
        return image

    compiled = PathPlannerArtifactStage(
        PathPlanner.__new__(PathPlanner)
    ).compile_invocation_input_edges(
        compile_function_pattern(consume_both, {}, {}),
        artifact_inputs={},
        relation_source_scopes={},
        execution_group_scope=ComponentGroupScope.ungrouped(),
        consumer_variable_components=ComponentSet(),
        main_flow_artifacts=ArtifactSpecCollection((first, second)),
    )

    edges = next(compiled.iter_invocations()).artifact_input_edges
    assert tuple(edge.main_flow_projection for edge in edges) == (
        MainFlowInputProjection.DECLARED_SOURCE_IMAGE,
        MainFlowInputProjection.DECLARED_SOURCE_IMAGE,
    )


def test_primary_workspace_role_does_not_override_compiled_input_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, plate_template, combined_image, compiled = (
        _compile_source_context_edges()
    )
    edges = next(compiled.iter_invocations()).artifact_input_edges
    edges_by_ref = {edge.spec.ref(): edge for edge in edges}
    source_payload = ImagePayloadMetadata(
        source_image_names=(plate_template.name,),
    ).payload_with(np.full((2, 3), 7.0, dtype=np.float32), None)
    current_payload = ImagePayloadMetadata(
        source_image_names=(combined_image.name,),
    ).payload_with(np.full((2, 3), 11.0, dtype=np.float32), None)
    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                alias=plate_template.name,
                projection_role=SourceProjectionRole.PRIMARY_PLANE,
            ),
        )
    )

    def source_artifact_payload(_request, ref):
        assert ref == plate_template.ref()
        return source_payload

    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        callable_contract=contract,
        artifact_inputs={edge.key: edge for edge in edges},
        source_binding_plan=source_binding_plan,
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
    )
    monkeypatch.setattr(
        type(adapter.request),
        "source_artifact_payload",
        source_artifact_payload,
    )
    request = RuntimeInputBindingRequest(
        adapter=adapter,
        kwargs={},
        current_image=current_payload,
    )

    with pytest.raises(
        ValueError,
        match="does not represent declared source image 'PlateTemplate'",
    ):
        request.artifact_request(
            replace(
                edges_by_ref[plate_template.ref()],
                consumes_main_flow=True,
            )
        )

    source_request = request.artifact_request(edges_by_ref[plate_template.ref()])
    main_flow_request = request.artifact_request(edges_by_ref[combined_image.ref()])

    np.testing.assert_array_equal(image_payload_data(source_request.value), 7.0)
    np.testing.assert_array_equal(image_payload_data(main_flow_request.value), 11.0)
    assert image_payload_metadata(source_request.value).source_image_names == (
        plate_template.name,
    )
    assert image_payload_metadata(main_flow_request.value).source_image_names == (
        combined_image.name,
    )


def test_storage_backed_input_keeps_exact_runtime_authority() -> None:
    contract, _, combined_image, compiled = _compile_source_context_edges(
        store_combined_image=True,
    )
    edges = next(compiled.iter_invocations()).artifact_input_edges
    edge = next(edge for edge in edges if edge.spec == combined_image)

    assert edge.storage_plan is not None
    assert edge.consumes_main_flow is False

    stored_payload = ImagePayloadMetadata(
        source_image_names=(combined_image.name,),
    ).payload_with(np.full((2, 3), 13.0, dtype=np.float32), None)
    current_payload = ImagePayloadMetadata(
        source_image_names=("UnrelatedCurrentImage",),
    ).payload_with(np.full((2, 3), 17.0, dtype=np.float32), None)
    output_plan = ArtifactOutputPlan(
        combined_image.name,
        edge.storage_plan.path,
        artifact_type=ImageArtifactType,
    )
    store = RuntimeValueStore()
    store.replace(
        RuntimeValue.normalize(output_plan, stored_payload, axis_id="A01"),
        path=edge.storage_plan.path,
        backend=Backend.MEMORY.value,
    )
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        callable_contract=contract,
        artifact_inputs={item.key: item for item in edges},
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
    )
    request = RuntimeInputBindingRequest(
        adapter=adapter,
        kwargs={},
        current_image=current_payload,
    )

    runtime_request = request.artifact_request(edge)

    np.testing.assert_array_equal(image_payload_data(runtime_request.value), 13.0)
    assert image_payload_metadata(runtime_request.value).source_image_names == (
        combined_image.name,
    )

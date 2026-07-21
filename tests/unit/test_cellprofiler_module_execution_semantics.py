"""Focused regressions for declaration-owned CellProfiler execution semantics."""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

import openhcs.interop.cellprofiler.runtime.module_execution as module_execution
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    ImageArtifactType,
    MeasurementsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.aligned_image_payload import (
    ImageOutputBundle,
    ImagePayloadExecutionMode,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelVariantData,
    object_label_dense_array,
)
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    object_label_input_execution_mode_from_callable,
)
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerModuleExecutor,
)
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.object_measurement_execution import (
    CellProfilerObjectMeasurementExecutionPolicy,
)
from openhcs.processing.backends.cellprofiler.colocalization import (
    MeasureColocalizationModule,
)
from openhcs.processing.backends.cellprofiler.color import (
    ColorToGrayMode,
    color_to_gray,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    MeasureImageIntensityModule,
)
from tests.unit.cellprofiler_runtime_test_support import (
    cellprofiler_runtime_adapter_for_test,
)


def _image_measurement_executor(
    declared_outputs: tuple[ArtifactSpec, ...],
) -> CellProfilerModuleExecutor:
    image_input = ArtifactSpec.input("Input", ImageArtifactType)
    raw_func = MeasureImageIntensityModule.require_callable()
    raw_contract = CallableContract.from_callable(raw_func)
    return CellProfilerModuleExecutor(
        raw_func=raw_func,
        callable_contract=replace(
            raw_contract,
            metadata=replace(
                raw_contract.metadata,
                artifact_inputs=(image_input,),
                artifact_outputs=declared_outputs,
                runtime_adapter=CellProfilerRuntimeAdapter.runtime_adapter_spec(),
            ),
        ),
    )


def _runtime_with_outputs(
    callable_contract: CallableContract,
    *outputs: ArtifactSpec,
    main_flow_inputs: tuple[ArtifactSpec, ...] = (),
):
    filemanager = Mock()
    filemanager.exists.return_value = False
    invocation_key = FunctionInvocationKey(
        "cellprofiler_module_execution_semantics",
        DEFAULT_GROUP_KEY,
        0,
    )
    input_edges = tuple(
        InvocationArtifactInputEdgePlan(
            key=InvocationArtifactInputProjectionKey(invocation_key, input_index),
            spec=spec,
            storage_plan=None,
            projection=None,
            consumes_main_flow=True,
        )
        for input_index, spec in enumerate(main_flow_inputs)
    )
    return cellprofiler_runtime_adapter_for_test(
        runtime_value_store=RuntimeValueStore(),
        filemanager=filemanager,
        callable_contract=callable_contract,
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "test-axis",
            component=None,
            value=None,
        ),
        artifact_inputs={edge.key: edge for edge in input_edges},
        artifact_outputs={
            output.ref(): ArtifactOutputPlan(
                name=output.name,
                path=f"/tmp/{output.name}.pkl",
                artifact_type=output.artifact_type,
                materialization=output.materialization,
                sidecar_role=output.sidecar_role,
                relations=output.relations,
            )
            for output in outputs
        },
    )


@pytest.mark.parametrize("live_auxiliary", (False, True))
def test_dead_output_liveness_does_not_choose_image_measurement_execution_mode(
    monkeypatch: pytest.MonkeyPatch,
    live_auxiliary: bool,
) -> None:
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)
    auxiliary = ArtifactSpec.output("Auxiliary", ImageArtifactType)
    executor = _image_measurement_executor((measurements, auxiliary))
    runtime = _runtime_with_outputs(
        executor.callable_contract,
        measurements,
        *((auxiliary,) if live_auxiliary else ()),
    )
    observed: list[tuple[str, tuple[ArtifactOutputPlan, ...]]] = []

    def run_per_image(_self, **request):
        observed.append(("per-image", request["active_output_plans"]))
        return request["image"]

    def run_standard(_self, **request):
        observed.append(("standard", request["active_outputs"]))
        return request["image"]

    monkeypatch.setattr(
        CellProfilerModuleExecutor,
        "_run_per_image_measurement",
        run_per_image,
    )
    monkeypatch.setattr(
        CellProfilerModuleExecutor,
        "_run_standard_image",
        run_standard,
    )

    image = np.zeros((3, 4), dtype=np.float32)
    assert executor(image, cellprofiler_runtime=runtime) is image
    assert observed == [
        (
            "per-image",
            tuple(runtime.request.artifact_outputs.values()),
        )
    ]


def test_active_outputs_do_not_restore_declaration_occurrences_from_one_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)
    executor = _image_measurement_executor((measurements, measurements))
    runtime = _runtime_with_outputs(executor.callable_contract, measurements)
    observed: list[tuple[ArtifactOutputPlan, ...]] = []

    def run_per_image(_self, **request):
        observed.append(request["active_output_plans"])
        return request["image"]

    monkeypatch.setattr(
        CellProfilerModuleExecutor,
        "_run_per_image_measurement",
        run_per_image,
    )

    executor(np.zeros((2, 2), dtype=np.float32), cellprofiler_runtime=runtime)

    assert observed == [tuple(runtime.request.artifact_outputs.values())]


def test_active_outputs_consume_compiled_plans_without_declaration_rematching() -> None:
    source = inspect.getsource(CellProfilerModuleExecutor.active_output_plans)

    assert "adapter.request.artifact_outputs.values()" in source
    assert ".select_plans(" not in source
    assert ".select_declared_occurrences(" not in source
    assert ".by_ref(" not in source


def test_main_flow_replacement_uses_declared_artifact_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ArtifactSpec.input("Original", ImageArtifactType)
    output = ArtifactSpec.output(
        "Gray",
        ImageArtifactType,
        relations=(SourceStackLineageSourceRelation(source=source.ref()),),
    )
    raw_contract = CallableContract.from_callable(color_to_gray)
    executor = CellProfilerModuleExecutor(
        raw_func=color_to_gray,
        callable_contract=replace(
            raw_contract,
            metadata=replace(
                raw_contract.metadata,
                artifact_inputs=(source,),
                artifact_outputs=(output,),
                runtime_adapter=CellProfilerRuntimeAdapter.runtime_adapter_spec(),
            ),
        ),
    )
    runtime = _runtime_with_outputs(
        executor.callable_contract,
        output,
        main_flow_inputs=(source,),
    )
    image = ImagePayloadMetadata(
        source_channel_axis=-1,
        source_path="/source/original.tif",
        source_image_names=("Original",),
    ).payload_with(np.zeros((4, 5, 3), dtype=np.float32), None)
    gray_artifact = np.full((4, 5), 7, dtype=np.float32)
    monkeypatch.setattr(
        type(module_execution._CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR),
        "execute",
        lambda _self, *_args, **_kwargs: gray_artifact,
    )

    result = executor(
        image,
        cellprofiler_runtime=runtime,
        mode=ColorToGrayMode.COMBINE,
    )

    assert isinstance(result, ImageOutputBundle)
    assert tuple(context.output_key for context in result.slice_contexts) == ("Gray",)
    np.testing.assert_array_equal(image_payload_data(result.slices[0]), gray_artifact)


def test_composed_image_measurement_callable_retains_standard_execution() -> None:
    raw_func = MeasureColocalizationModule.require_callable()

    assert (
        object_label_input_execution_mode_from_callable(raw_func)
        is ObjectLabelInputExecutionMode.SLICE_ALIGNED
    )
    assert not MeasureColocalizationModule.executes_per_image_measurements(
        raw_func,
        (),
        callable_contract=CallableContract.from_callable(raw_func),
    )


def test_slice_aligned_object_measurement_consumes_declared_singleton_plane() -> None:
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.full((1, 4, 5), 7, dtype=np.int32),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((7,),),
        ),
    )
    policy = CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
        ObjectLabelInputExecutionMode.SLICE_ALIGNED
    )

    projected = policy.semantic_label_payload(labels, labels)

    assert isinstance(projected, ObjectLabelPayload)
    assert object_label_dense_array(projected).shape == (4, 5)
    assert projected.object_label_domain().scope is ObjectLabelDomainScope.PAYLOAD
    assert projected.object_label_domain().declared_object_ids == (7,)
    assert projected.plane_axis is None
    assert (
        policy.image_execution_mode(
            projected,
            ImagePayloadExecutionMode.FULL_STACK,
        )
        is ImagePayloadExecutionMode.FULL_STACK
    )


def test_slice_aligned_object_measurement_rejects_full_stack_label_domain() -> None:
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.ones((2, 4, 5), dtype=np.int32),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,), (1,)),
        ),
    )
    policy = CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
        ObjectLabelInputExecutionMode.SLICE_ALIGNED
    )

    with pytest.raises(
        ValueError,
        match="cannot execute a full-stack image with an unprojected",
    ):
        policy.image_execution_mode(
            policy.semantic_label_payload(labels, labels),
            ImagePayloadExecutionMode.FULL_STACK,
        )


def test_executor_delegates_runtime_kwarg_projection_to_slice_authority() -> None:
    source = inspect.getsource(CellProfilerModuleExecutor._invocation_request)

    assert "RuntimeSliceProjection.kwargs_for_slice(" in source
    assert (
        "RuntimeSliceProjection.value_for_slice(value, runtime_projection)"
        not in source
    )


def test_executor_has_no_local_partition_occurrence_reconstruction() -> None:
    source = inspect.getsource(module_execution)

    assert "_require_partition_specs" not in source
    assert "from collections import Counter" not in source
    assert "ModuleArtifactContract" not in source
    assert ".canonical_output_refs(" not in source
    assert ".main_flow_output_refs(" not in source
    returned_output_source = inspect.getsource(
        CellProfilerModuleExecutor._returned_output_values
    )
    assert "canonical_output_refs" not in returned_output_source
    assert "canonical_output=" not in returned_output_source


def test_output_record_requests_do_not_copy_active_output_specs() -> None:
    tree = ast.parse(inspect.getsource(module_execution))
    request_calls = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CellProfilerOutputRecordRequest"
    )

    assert request_calls
    assert all(
        "active_output_specs"
        not in {keyword.arg for keyword in call.keywords}
        for call in request_calls
    )

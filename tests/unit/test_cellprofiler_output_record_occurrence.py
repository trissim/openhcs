from types import SimpleNamespace

import pytest

from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract, CallableMetadata
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
)
from openhcs.core.runtime_adapters import RuntimeAdapterRequest
from openhcs.interop.cellprofiler.runtime.output_record_request import (
    CellProfilerOutputRecordRequest,
)


def _output_record_request(
    declared_inputs: tuple[ArtifactSpec, ...],
    active_occurrences: tuple[tuple[int, ArtifactSpec], ...],
) -> tuple[
    CellProfilerOutputRecordRequest,
    tuple[InvocationArtifactInputEdgePlan, ...],
]:
    output = ArtifactSpec.output("Result", ObjectLabelsArtifactType)
    contract = CallableContract(
        func=lambda image: image,
        function_name="active_occurrence_probe",
        module_name=__name__,
        metadata=CallableMetadata(
            artifact_inputs=declared_inputs,
            artifact_outputs=(output,),
        ),
    )
    invocation_key = FunctionInvocationKey(
        function_name=contract.function_name,
        group_key="default",
        position=0,
    )
    edges = tuple(
        InvocationArtifactInputEdgePlan(
            key=InvocationArtifactInputProjectionKey(invocation_key, input_index),
            spec=spec,
            storage_plan=None,
            projection=None,
            consumes_main_flow=True,
        )
        for input_index, spec in active_occurrences
    )
    output_plan = ArtifactOutputPlan(
        name=output.name,
        path="/artifacts/result",
        artifact_type=output.artifact_type,
    )
    runtime_request = RuntimeAdapterRequest(
        context=object(),
        callable_contract=contract,
        artifact_inputs={edge.key: edge for edge in edges},
        artifact_outputs={output_plan.ref(): output_plan},
        axis_scope=RuntimeExecutionAxisScope(axis_id="A01"),
    )
    return (
        CellProfilerOutputRecordRequest(
            callable_contract=contract,
            active_input_edges=edges,
            adapter=SimpleNamespace(request=runtime_request),
            spec=output,
            output_plan=output_plan,
            output_value=object(),
            source=SimpleNamespace(),
            call_kwargs={},
            current_image=object(),
        ),
        edges,
    )


def test_output_source_uses_compiled_runtime_occurrence_for_repeated_roles(
    monkeypatch,
) -> None:
    primary = ArtifactSpec.input(
        "Objects",
        ObjectLabelsArtifactType,
        parameter_name="primary_objects",
    )
    neighbor = ArtifactSpec.input(
        "Objects",
        ObjectLabelsArtifactType,
        parameter_name="neighbor_objects",
    )
    output = ArtifactSpec.output_preserving_source_stack_scope(
        "Result",
        ObjectLabelsArtifactType,
        primary,
    )
    contract = CallableContract(
        func=lambda image: image,
        function_name="repeated_role_probe",
        module_name=__name__,
        metadata=CallableMetadata(
            artifact_inputs=(primary, neighbor),
            artifact_outputs=(output,),
        ),
    )
    invocation_key = FunctionInvocationKey(
        function_name=contract.function_name,
        group_key="default",
        position=0,
    )
    edges = tuple(
        InvocationArtifactInputEdgePlan(
            key=InvocationArtifactInputProjectionKey(invocation_key, input_index),
            spec=spec,
            storage_plan=None,
            projection=None,
            consumes_main_flow=True,
        )
        for input_index, spec in enumerate(contract.artifact_inputs)
    )
    output_plan = ArtifactOutputPlan(
        name=output.name,
        path="/artifacts/result",
        artifact_type=output.artifact_type,
        relations=output.relations,
    )
    runtime_request = RuntimeAdapterRequest(
        context=object(),
        callable_contract=contract,
        artifact_inputs={edge.key: edge for edge in edges},
        artifact_outputs={output_plan.ref(): output_plan},
        axis_scope=RuntimeExecutionAxisScope(axis_id="A01"),
    )
    marker = object()

    monkeypatch.setattr(
        CellProfilerOutputRecordRequest,
        "artifact_source_payload",
        lambda _self, _edge: marker,
    )
    request = CellProfilerOutputRecordRequest(
        callable_contract=contract,
        active_input_edges=edges,
        adapter=SimpleNamespace(request=runtime_request),
        spec=output,
        output_plan=output_plan,
        output_value=object(),
        source=SimpleNamespace(),
        call_kwargs={},
        current_image=object(),
    )

    assert request.declared_source_payload() is marker
    assert tuple(runtime_request.artifact_inputs) == tuple(edge.key for edge in edges)


def test_output_record_request_accepts_nonzero_duplicate_ref_occurrence() -> None:
    primary = ArtifactSpec.input(
        "Objects",
        ObjectLabelsArtifactType,
        parameter_name="primary_objects",
    )
    neighbor = ArtifactSpec.input(
        "Objects",
        ObjectLabelsArtifactType,
        parameter_name="neighbor_objects",
    )

    request, edges = _output_record_request(
        (primary, neighbor),
        ((1, neighbor),),
    )

    assert request.exact_input_edge(neighbor) is edges[0]
    with pytest.raises(RuntimeError, match="has no exact compiled input edge"):
        request.exact_input_edge(primary)


def test_output_record_request_accepts_noncontiguous_ordered_subset() -> None:
    declared_inputs = tuple(
        ArtifactSpec.input(name, ObjectLabelsArtifactType)
        for name in ("First", "Inactive", "Third")
    )

    request, edges = _output_record_request(
        declared_inputs,
        ((0, declared_inputs[0]), (2, declared_inputs[2])),
    )

    assert request.active_input_edges == edges
    assert tuple(edge.key.input_index for edge in edges) == (0, 2)


def test_output_record_request_accepts_empty_active_subset() -> None:
    declared = ArtifactSpec.input("Inactive", ObjectLabelsArtifactType)

    request, edges = _output_record_request((declared,), ())

    assert request.active_input_edges == edges == ()


def test_output_record_request_rejects_compacted_parameter_role() -> None:
    primary = ArtifactSpec.input(
        "Objects",
        ObjectLabelsArtifactType,
        parameter_name="primary_objects",
    )
    neighbor = ArtifactSpec.input(
        "Objects",
        ObjectLabelsArtifactType,
        parameter_name="neighbor_objects",
    )

    with pytest.raises(ValueError, match="exact declared occurrence"):
        _output_record_request(
            (primary, neighbor),
            ((0, neighbor),),
        )


@pytest.mark.parametrize("active_indexes", ((1, 1), (2, 0)))
def test_output_record_request_rejects_duplicate_or_reordered_indexes(
    active_indexes: tuple[int, int],
) -> None:
    declared_inputs = tuple(
        ArtifactSpec.input(name, ObjectLabelsArtifactType)
        for name in ("First", "Second", "Third")
    )

    with pytest.raises(ValueError, match="strictly increasing declared occurrence"):
        _output_record_request(
            declared_inputs,
            tuple((index, declared_inputs[index]) for index in active_indexes),
        )


def test_output_record_request_rejects_out_of_range_index() -> None:
    declared = ArtifactSpec.input("Only", ObjectLabelsArtifactType)

    with pytest.raises(ValueError, match=r"declared occurrence range \[0, 1\)"):
        _output_record_request(
            (declared,),
            ((1, declared),),
        )

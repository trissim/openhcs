from dataclasses import dataclass

import pytest

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactKind,
    ArtifactOutputPlan,
    ArtifactSpec,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    compile_function_pattern,
    inject_artifact_input_values,
    inject_kwargs_into_pattern,
    iter_enabled_function_invocations,
    normalize_function_pattern,
    strip_disabled_functions,
)
from openhcs.core.pipeline.function_contracts import (
    artifact_inputs,
    artifact_outputs,
)
from openhcs.core.pipeline.artifact_planning import (
    extract_artifact_declarations,
    normalize_pattern,
)
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.processing.materialization import csv_only


def first(image):
    return image


def second(image):
    return image


def skipped(image):
    return image


def third(image):
    return image


@dataclass(frozen=True, slots=True, kw_only=True)
class ExampleInvocationOptions(RuntimeInvocationOptions):
    mode: str


first.__artifact_outputs__ = {"positions": ArtifactSpec("positions")}
second.__artifact_outputs__ = {"measurements": ArtifactSpec("measurements")}
third.__artifact_outputs__ = {
    "positions": ArtifactSpec("positions"),
    "measurements": ArtifactSpec("measurements"),
}


def test_invocation_keys_preserve_callable_positions_for_list_patterns():
    pattern = [
        first,
        (skipped, {"enabled": False}),
        (second, {"sigma": 1}),
    ]

    keys = [invocation.key for invocation in iter_enabled_function_invocations(pattern)]

    assert keys == [
        FunctionInvocationKey("first", "default", 0),
        FunctionInvocationKey("second", "default", 1),
    ]


def test_invocation_positions_are_renumbered_per_dict_group():
    pattern = {
        "DAPI": [
            first,
            (skipped, {"enabled": False}),
            second,
        ],
        "GFP": [
            third,
        ],
    }

    keys = [
        invocation.key
        for invocation in iter_enabled_function_invocations(pattern)
    ]

    assert keys == [
        FunctionInvocationKey("first", "DAPI", 0),
        FunctionInvocationKey("second", "DAPI", 1),
        FunctionInvocationKey("third", "GFP", 0),
    ]


def test_artifact_planning_normalize_pattern_returns_tuple_api():
    pattern = [first, (skipped, {"enabled": False}), second]

    normalized = [
        (func.__name__, group_key, position)
        for func, group_key, position in normalize_pattern(pattern)
    ]

    assert normalized == [
        ("first", "default", 0),
        ("second", "default", 1),
    ]


def test_artifact_graph_tracks_kind_groups_and_invocation_ownership():
    @artifact_outputs(ArtifactSpec("nuclei", ArtifactKind.OBJECT_LABELS))
    def identify(image):
        return image

    graph = extract_artifact_declarations({"DAPI": identify})

    assert graph.outputs["nuclei"].kind is ArtifactKind.OBJECT_LABELS
    assert graph.output_groups["nuclei"] == {"DAPI"}
    assert graph.producers[0].invocation_keys == (
        FunctionInvocationKey("identify", "DAPI", 0),
    )


def test_artifact_graph_rejects_conflicting_producer_kinds():
    @artifact_outputs(ArtifactSpec("objects", ArtifactKind.OBJECT_LABELS))
    def identify(image):
        return image

    @artifact_outputs(ArtifactSpec("objects", ArtifactKind.MEASUREMENTS))
    def measure(image):
        return image

    with pytest.raises(ValueError, match="Conflicting producer artifact kind"):
        extract_artifact_declarations([identify, measure])


def test_artifact_graph_rejects_local_consumer_producer_kind_mismatch():
    @artifact_outputs(ArtifactSpec("objects", ArtifactKind.OBJECT_LABELS))
    def identify(image):
        return image

    @artifact_inputs(ArtifactSpec("objects", ArtifactKind.MEASUREMENTS))
    def measure(image, objects):
        return image

    with pytest.raises(ValueError, match="produced as object_labels"):
        extract_artifact_declarations([identify, measure])


def test_normalize_function_pattern_is_grouped_source_of_truth():
    normalized = normalize_function_pattern(
        {
            "DAPI": [first, (skipped, {"enabled": False}), (second, {"sigma": 2})],
            "GFP": third,
        }
    )

    assert normalized.is_grouped
    assert [group.group_key for group in normalized.groups] == ["DAPI", "GFP"]
    assert [
        (item.key.function_name, item.key.group_key, item.key.position, item.kwargs)
        for item in normalized.iter_items()
    ] == [
        ("first", "DAPI", 0, ()),
        ("second", "DAPI", 1, (("sigma", 2),)),
        ("third", "GFP", 0, ()),
    ]


def test_callable_contract_is_nominal_source_for_callable_metadata():
    first.input_memory_type = "numpy"
    first.output_memory_type = "cupy"
    first.__artifact_inputs__ = {"positions": ArtifactSpec("positions")}

    contract = CallableContract.from_callable(first)

    assert contract.function_name == "first"
    assert contract.input_memory_type == "numpy"
    assert contract.output_memory_type == "cupy"
    assert contract.artifact_input_names == ("positions",)
    assert contract.artifact_output_names == ("positions",)
    assert contract.select_input_plan_keys(
        {
            "positions": ArtifactInputPlan("positions", "/tmp/positions.pkl"),
            "other": ArtifactInputPlan("other", "/tmp/other.pkl"),
        }
    ) == ("positions",)


def test_compile_function_pattern_builds_invocation_source_of_truth():
    first.input_memory_type = "numpy"
    first.output_memory_type = "numpy"
    second.input_memory_type = "numpy"
    second.output_memory_type = "numpy"
    first.__artifact_inputs__ = {"positions": ArtifactSpec("positions")}

    compiled = compile_function_pattern(
        {
            "DAPI": [
                (first, {"sigma": 1, "__pyqt_reactive_scope_token__": "ui"}),
                second,
            ]
        },
        {"positions": ArtifactInputPlan("positions", "/tmp/positions.pkl")},
        {
            "positions": ArtifactOutputPlan("positions", "/tmp/positions.pkl"),
            "measurements": ArtifactOutputPlan("measurements", "/tmp/measurements.pkl"),
        },
    )

    group = compiled.group_for_component("DAPI")
    assert compiled.is_grouped
    assert group is not None
    assert [invocation.key for invocation in group.invocations] == [
        FunctionInvocationKey("first", "DAPI", 0),
        FunctionInvocationKey("second", "DAPI", 1),
    ]
    assert group.invocations[0].contract.function_name == "first"
    assert group.invocations[0].kwargs == (("sigma", 1),)
    assert group.invocations[0].artifact_input_keys == ("positions",)
    assert group.invocations[0].artifact_output_keys == ("positions",)
    assert group.invocations[1].artifact_output_keys == ("measurements",)


def test_compile_function_pattern_preserves_typed_invocation_options():
    options = ExampleInvocationOptions(mode="once")

    compiled = compile_function_pattern(
        (first, {"sigma": 1}, options),
        {},
        {},
    )

    invocation = compiled.default_group.invocations[0]
    assert invocation.kwargs == (("sigma", 1),)
    assert invocation.invocation_options is options


def test_inject_kwargs_preserves_typed_invocation_options():
    options = ExampleInvocationOptions(mode="once")

    pattern = inject_kwargs_into_pattern(
        (first, {"sigma": 1}, options),
        {"dtype_config": "inherited"},
    )

    assert pattern == (
        first,
        {"dtype_config": "inherited", "sigma": 1},
        options,
    )


def test_compiled_function_pattern_filters_detected_groups_by_compiled_keys():
    compiled = compile_function_pattern({"1": first}, {}, {})

    grouped = compiled.prepare_grouped_patterns(
        {1: ["site1"], "2": ["site2"]},
        default_component="channel",
    )

    assert grouped == {1: ["site1"]}


def test_contract_decorators_declare_artifact_specs():
    materialization = csv_only()

    @artifact_inputs("positions")
    @artifact_outputs("metadata", ("measurements", materialization))
    def analyze(image):
        return image

    assert list(analyze.__artifact_inputs__) == ["positions"]
    assert list(analyze.__artifact_outputs__) == ["metadata", "measurements"]
    assert analyze.__artifact_outputs__["metadata"] == ArtifactSpec("metadata")
    assert analyze.__artifact_outputs__["measurements"].materialization is materialization


def test_artifact_decorators_accept_typed_artifact_specs():
    labels = ArtifactSpec("nuclei", ArtifactKind.OBJECT_LABELS)
    measurements = ArtifactSpec("measurements", ArtifactKind.MEASUREMENTS)

    @artifact_inputs(labels)
    @artifact_outputs(measurements)
    def measure(image, nuclei):
        return image

    assert measure.__artifact_inputs__["nuclei"] == labels
    assert measure.__artifact_outputs__["measurements"] == measurements


def test_strip_disabled_functions_removes_empty_pattern_branches():
    pattern = {
        "DAPI": [first, (skipped, {"enabled": False})],
        "GFP": [(skipped, {"enabled": False})],
    }

    assert strip_disabled_functions(pattern) == {"DAPI": [first]}


def test_inject_kwargs_into_pattern_preserves_user_kwargs_precedence():
    pattern = [first, (second, {"dtype_config": "explicit", "sigma": 2})]

    assert inject_kwargs_into_pattern(pattern, {"dtype_config": "inherited"}) == [
        (first, {"dtype_config": "inherited"}),
        (second, {"dtype_config": "explicit", "sigma": 2}),
    ]


def test_inject_artifact_input_values_only_targets_declared_inputs():
    @artifact_inputs("grid_dimensions")
    def needs_grid(image, grid_dimensions):
        return image

    pattern = [first, (needs_grid, {"sigma": 2})]

    assert inject_artifact_input_values(
        pattern,
        {"grid_dimensions": (3, 4), "unused": "ignored"},
    ) == [
        first,
        (needs_grid, {"grid_dimensions": (3, 4), "sigma": 2}),
    ]

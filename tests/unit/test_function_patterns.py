from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan, ArtifactSpec
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    build_function_invocation_plans,
    function_invocation_key,
    iter_enabled_function_invocations,
)
from openhcs.core.pipeline.function_contracts import (
    artifact_inputs,
    artifact_outputs,
)
from openhcs.core.pipeline.artifact_planning import normalize_pattern
from openhcs.processing.materialization import csv_only


def first(image):
    return image


def second(image):
    return image


def skipped(image):
    return image


def third(image):
    return image


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


def test_function_invocation_key_matches_runtime_identity_contract():
    assert function_invocation_key(first, "DAPI", 2) == FunctionInvocationKey(
        "first",
        "DAPI",
        2,
    )


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


def test_invocation_plans_capture_per_call_artifact_output_ownership():
    artifact_outputs = {
        "positions": ArtifactOutputPlan("positions", "/tmp/positions.pkl"),
        "measurements": ArtifactOutputPlan("measurements", "/tmp/measurements.pkl"),
    }

    plans = build_function_invocation_plans(
        {"DAPI": [first, second], "GFP": third},
        artifact_outputs,
    )

    assert plans[FunctionInvocationKey("first", "DAPI", 0)].artifact_output_keys == (
        "positions",
    )
    assert plans[FunctionInvocationKey("second", "DAPI", 1)].artifact_output_keys == (
        "measurements",
    )
    assert plans[FunctionInvocationKey("third", "GFP", 0)].artifact_output_keys == (
        "positions",
        "measurements",
    )
    assert (
        plans[FunctionInvocationKey("third", "GFP", 0)].select_outputs(
            artifact_outputs
        )
        == artifact_outputs
    )


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

from types import SimpleNamespace

import pytest

from openhcs.constants import GroupBy, VariableComponents
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.function_patterns import compile_function_pattern
from openhcs.core.pipeline.funcstep_contract_validator import (
    FuncStepContractValidator,
)
from openhcs.core.pipeline.function_contracts import artifact_inputs, artifact_outputs
from openhcs.core.runtime_adapters import runtime_adapter


def _function(name="function"):
    def func(image):
        return image

    func.__name__ = name
    func.__module__ = "builtins"
    return func


def _compiled_pattern(func):
    return compile_function_pattern(func, {}, {})


def test_validate_compiled_function_pattern_uses_callable_contract_memory_types():
    func = _function("valid")
    func.input_memory_type = "numpy"
    func.output_memory_type = "cupy"

    assert FuncStepContractValidator.validate_compiled_function_pattern(
        _compiled_pattern(func),
        "step",
    ) == ("numpy", "cupy")


def test_validate_compiled_function_pattern_rejects_missing_contract_memory_types():
    func = _function("missing")

    with pytest.raises(ValueError, match="needs memory type decorator"):
        FuncStepContractValidator.validate_compiled_function_pattern(
            _compiled_pattern(func),
            "step",
        )


def test_validate_compiled_function_pattern_reports_invocation_identity():
    func = _function("invalid")
    func.input_memory_type = "bogus"
    func.output_memory_type = "numpy"

    with pytest.raises(ValueError, match=r"invalid\[default:0\]"):
        FuncStepContractValidator.validate_compiled_function_pattern(
            _compiled_pattern(func),
            "step",
        )


def test_normalized_group_by_resolves_variable_component_conflict_to_none():
    assert (
        FuncStepContractValidator.normalized_group_by(
            GroupBy.CHANNEL,
            (VariableComponents.CHANNEL,),
            "step",
        )
        is GroupBy.NONE
    )


def _runtime_artifact_step_plan(
    *,
    variable_components=(),
    group_by=GroupBy.NONE,
):
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    @artifact_inputs(ArtifactSpec("InputImage", ArtifactKind.IMAGE))
    @artifact_outputs(ArtifactSpec("OutputImage", ArtifactKind.IMAGE))
    def runtime_artifact_step(image, *, runtime):
        return image

    runtime_artifact_step.input_memory_type = "numpy"
    runtime_artifact_step.output_memory_type = "numpy"
    artifact_outputs_by_key = {
        "OutputImage": SimpleNamespace(kind=ArtifactKind.IMAGE),
    }
    step_plan = SimpleNamespace(
        step_name="EnhanceOrSuppressFeatures",
        variable_components=variable_components,
        artifact_outputs=artifact_outputs_by_key,
        compiled_function_pattern=compile_function_pattern(
            runtime_artifact_step,
            {"InputImage": object()},
            artifact_outputs_by_key,
        ),
        group_by=group_by,
    )
    return step_plan


def test_validate_artifact_managed_runtime_scope_rejects_channel_variable_axis():
    step_plan = _runtime_artifact_step_plan(
        variable_components=(VariableComponents.CHANNEL,),
    )

    with pytest.raises(
        ValueError,
        match="cannot expand named runtime artifacts by CHANNEL",
    ):
        FuncStepContractValidator.validate_artifact_managed_runtime_scope(step_plan)


def test_validate_artifact_managed_runtime_scope_rejects_channel_group_by_axis():
    step_plan = _runtime_artifact_step_plan(
        variable_components=(VariableComponents.SITE,),
        group_by=GroupBy.CHANNEL,
    )

    with pytest.raises(
        ValueError,
        match="cannot expand named runtime artifacts by CHANNEL",
    ):
        FuncStepContractValidator.validate_artifact_managed_runtime_scope(step_plan)


def test_validate_artifact_managed_runtime_scope_allows_site_axis():
    step_plan = _runtime_artifact_step_plan(
        variable_components=(VariableComponents.SITE,),
        group_by=GroupBy.NONE,
    )

    FuncStepContractValidator.validate_artifact_managed_runtime_scope(step_plan)

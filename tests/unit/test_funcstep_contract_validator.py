from types import SimpleNamespace

import pytest

from openhcs.constants import GroupBy, VariableComponents
from openhcs.core.artifact_materialization_policy import NO_ARTIFACT_MATERIALIZATION
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan, ArtifactSpec
from openhcs.core.function_patterns import compile_function_pattern
from openhcs.core.pipeline.funcstep_contract_validator import (
    FuncStepContractValidator,
    FunctionStepArtifactContractScope,
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


def _artifact_output(
    name="OutputImage",
    *,
    kind=ArtifactKind.IMAGE,
    materialization=None,
):
    return ArtifactOutputPlan(
        name=name,
        path=f"/tmp/{name}",
        kind=kind,
        materialization=materialization,
    )


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
    artifact_outputs_by_key = {"OutputImage": _artifact_output()}
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
        source_identity_stack_axes=frozenset(),
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


def test_validate_artifact_managed_runtime_scope_allows_source_identity_channel_group_by_axis():
    step_plan = _runtime_artifact_step_plan(
        variable_components=(VariableComponents.SITE,),
        group_by=GroupBy.CHANNEL,
    )
    step_plan.source_identity_stack_axes = frozenset((VariableComponents.CHANNEL.value,))

    FuncStepContractValidator.validate_artifact_managed_runtime_scope(step_plan)


def test_validate_artifact_managed_runtime_scope_allows_site_axis():
    step_plan = _runtime_artifact_step_plan(
        variable_components=(VariableComponents.SITE,),
        group_by=GroupBy.NONE,
    )

    FuncStepContractValidator.validate_artifact_managed_runtime_scope(step_plan)


def test_validate_source_identity_materialization_rejects_multiplane_output():
    step_plan = SimpleNamespace(
        step_name="IdentifyPrimaryObjects",
        variable_components=(
            VariableComponents.SITE,
            VariableComponents.CHANNEL,
        ),
        artifact_outputs={
            "Nuclei": _artifact_output(
                "Nuclei",
                kind=ArtifactKind.OBJECT_LABELS,
            )
        },
        compiled_function_pattern=None,
        group_by=GroupBy.NONE,
        source_identity_stack_axes=frozenset(),
    )

    with pytest.raises(
        ValueError,
        match="materializes source-identity-named artifact output",
    ):
        FuncStepContractValidator.validate_source_identity_materialization_scope(
            step_plan
        )


def test_validate_source_identity_materialization_allows_explicit_opt_out():
    step_plan = SimpleNamespace(
        step_name="IdentifyPrimaryObjects",
        variable_components=(VariableComponents.SITE,),
        artifact_outputs={
            "Nuclei": _artifact_output(
                "Nuclei",
                kind=ArtifactKind.OBJECT_LABELS,
                materialization=NO_ARTIFACT_MATERIALIZATION,
            )
        },
        compiled_function_pattern=None,
        group_by=GroupBy.NONE,
        source_identity_stack_axes=frozenset(),
    )

    FuncStepContractValidator.validate_source_identity_materialization_scope(
        step_plan
    )


def test_validate_source_identity_materialization_allows_scalar_invocation():
    step_plan = SimpleNamespace(
        step_name="IdentifyPrimaryObjects",
        variable_components=(),
        artifact_outputs={
            "Nuclei": _artifact_output(
                "Nuclei",
                kind=ArtifactKind.OBJECT_LABELS,
            )
        },
        compiled_function_pattern=None,
        group_by=GroupBy.SITE,
        source_identity_stack_axes=frozenset(),
    )

    FuncStepContractValidator.validate_source_identity_materialization_scope(
        step_plan
    )


def _artifact_contract_scope(*, site_count):
    func = _function("plain")
    func.input_memory_type = "numpy"
    func.output_memory_type = "numpy"
    return FunctionStepArtifactContractScope(
        step_name="IdentifyPrimaryObjects",
        variable_components=(VariableComponents.SITE,),
        group_by=GroupBy.NONE,
        artifact_outputs={
            "Nuclei": _artifact_output(
                "Nuclei",
                kind=ArtifactKind.OBJECT_LABELS,
            )
        },
        compiled_function_pattern=_compiled_pattern(func),
        variable_component_key_counts={"site": site_count},
    )


def test_validate_artifact_contract_scope_allows_single_key_variable_axis():
    FuncStepContractValidator.validate_artifact_contract_scope(
        _artifact_contract_scope(site_count=1)
    )


def test_validate_artifact_contract_scope_rejects_multi_key_variable_axis():
    with pytest.raises(ValueError, match="component\\(s\\) SITE"):
        FuncStepContractValidator.validate_artifact_contract_scope(
            _artifact_contract_scope(site_count=2)
        )

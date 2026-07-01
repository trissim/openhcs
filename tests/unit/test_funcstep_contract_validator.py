from types import SimpleNamespace

import pytest

from openhcs.constants import GroupBy, VariableComponents
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan, ArtifactSpec
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import compile_function_pattern
from openhcs.core.memory.decorators import numpy
from openhcs.core.module_artifact_contract import (
    ModuleArtifactContract,
    module_artifact_contract,
)
from openhcs.core.pipeline.funcstep_contract_validator import (
    FuncStepContractValidator,
    FunctionStepArtifactContractScope,
)
from openhcs.core.pipeline.function_contracts import (
    allowed_group_by,
    artifact_inputs,
    artifact_outputs,
    require_variable_component_stack,
    required_variable_components,
)
from openhcs.core.config import LazyProcessingConfig
from openhcs.core.runtime_adapters import runtime_adapter
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


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


def test_validate_funcstep_skips_dict_key_lookup_for_groupby_none():
    func = _function("channel_branch")
    func.input_memory_type = "numpy"
    func.output_memory_type = "numpy"
    step = FunctionStep(
        func={"1": func},
        name="dict-none",
        processing_config=LazyProcessingConfig(group_by=GroupBy.NONE),
    )

    def fail_component_lookup(_group_by):
        raise AssertionError("GroupBy.NONE must not query component keys")

    orchestrator = SimpleNamespace(get_component_keys=fail_component_lookup)

    FuncStepContractValidator.validate_funcstep(step, orchestrator=orchestrator)


def test_validate_required_variable_components_allows_declared_axis():
    @required_variable_components(VariableComponents.TIMEPOINT)
    def process(image):
        return image

    contract = CallableContract.from_callable(process)

    FuncStepContractValidator.validate_required_variable_components(
        (VariableComponents.TIMEPOINT,),
        (contract,),
        "track",
    )


def test_validate_required_variable_components_rejects_missing_callable_axis():
    @required_variable_components(VariableComponents.TIMEPOINT)
    def process(image):
        return image

    contract = CallableContract.from_callable(process)

    with pytest.raises(ValueError, match="requires variable_components TIMEPOINT"):
        FuncStepContractValidator.validate_required_variable_components(
            (),
            (contract,),
            "track",
        )


def test_validate_processing_contract_rejects_pure_3d_without_variable_axis():
    @numpy(contract=ProcessingContract.PURE_3D)
    def full_stack(image):
        return image

    with pytest.raises(ValueError, match="PURE_3D stack semantics"):
        FuncStepContractValidator.validate_processing_contract_variable_components(
            (),
            tuple(_compiled_pattern(full_stack).iter_invocations()),
            "full_stack",
        )


def test_validate_processing_contract_allows_pure_3d_with_variable_axis():
    @numpy(contract=ProcessingContract.PURE_3D)
    def full_stack(image):
        return image

    FuncStepContractValidator.validate_processing_contract_variable_components(
        (VariableComponents.Z_INDEX,),
        tuple(_compiled_pattern(full_stack).iter_invocations()),
        "full_stack",
    )


def test_validate_processing_contract_rejects_volumetric_to_slice_without_variable_axis():
    @numpy(contract=ProcessingContract.VOLUMETRIC_TO_SLICE)
    def project_stack(image):
        return image

    with pytest.raises(ValueError, match="VOLUMETRIC_TO_SLICE stack semantics"):
        FuncStepContractValidator.validate_processing_contract_variable_components(
            (),
            tuple(_compiled_pattern(project_stack).iter_invocations()),
            "project",
        )


def test_validate_processing_contract_allows_flexible_slice_by_slice_without_axis():
    @numpy(contract=ProcessingContract.FLEXIBLE)
    def flexible(image, *, slice_by_slice: bool = False):
        return image

    FuncStepContractValidator.validate_processing_contract_variable_components(
        (),
        tuple(_compiled_pattern((flexible, {"slice_by_slice": True})).iter_invocations()),
        "flexible",
    )


def test_validate_processing_contract_rejects_flexible_full_stack_without_axis():
    @numpy(contract=ProcessingContract.FLEXIBLE)
    def flexible(image, *, slice_by_slice: bool = False):
        return image

    with pytest.raises(ValueError, match="FLEXIBLE stack semantics"):
        FuncStepContractValidator.validate_processing_contract_variable_components(
            (),
            tuple(_compiled_pattern((flexible, {"slice_by_slice": False})).iter_invocations()),
            "flexible",
        )


def test_validate_processing_contract_uses_flexible_signature_default():
    @numpy(contract=ProcessingContract.FLEXIBLE)
    def flexible_2d_default(image, *, slice_by_slice: bool = True):
        return image

    FuncStepContractValidator.validate_processing_contract_variable_components(
        (),
        tuple(_compiled_pattern(flexible_2d_default).iter_invocations()),
        "flexible",
    )


def test_validate_declared_stack_requirement_rejects_without_variable_axis():
    @require_variable_component_stack
    @numpy(contract=ProcessingContract.PURE_2D)
    def stacked_callable(image):
        return image

    with pytest.raises(ValueError, match="stack semantics"):
        FuncStepContractValidator.validate_processing_contract_variable_components(
            (),
            tuple(_compiled_pattern(stacked_callable).iter_invocations()),
            "stacked_callable",
        )


def test_validate_allowed_group_by_accepts_declared_group_by():
    @allowed_group_by(GroupBy.NONE)
    def process(image):
        return image

    contract = CallableContract.from_callable(process)

    FuncStepContractValidator.validate_allowed_group_by(
        GroupBy.NONE,
        (contract,),
        "cellprofiler",
    )


def test_validate_allowed_group_by_treats_none_as_groupby_none():
    @allowed_group_by(GroupBy.NONE)
    def process(image):
        return image

    contract = CallableContract.from_callable(process)

    FuncStepContractValidator.validate_allowed_group_by(
        None,
        (contract,),
        "cellprofiler",
    )


def test_validate_allowed_group_by_rejects_forbidden_fanout():
    @allowed_group_by(GroupBy.NONE)
    def process(image):
        return image

    contract = CallableContract.from_callable(process)

    with pytest.raises(ValueError, match="allows group_by NONE; resolved CHANNEL"):
        FuncStepContractValidator.validate_allowed_group_by(
            GroupBy.CHANNEL,
            (contract,),
            "cellprofiler",
        )


def test_validate_required_variable_components_reads_module_contract_axis():
    @module_artifact_contract(
        ModuleArtifactContract(
            "TrackObjects",
            required_variable_components=(VariableComponents.TIMEPOINT,),
        )
    )
    def process(image):
        return image

    contract = CallableContract.from_callable(process)

    with pytest.raises(ValueError, match="TrackObjects"):
        FuncStepContractValidator.validate_required_variable_components(
            (),
            (contract,),
            "TrackObjects",
        )


def test_validate_funcstep_enforces_required_variable_components():
    @required_variable_components(VariableComponents.TIMEPOINT)
    def process(image):
        return image

    process.input_memory_type = "numpy"
    process.output_memory_type = "numpy"
    step = FunctionStep(
        func=process,
        name="TrackObjects",
        processing_config=LazyProcessingConfig(
            variable_components=(),
            group_by=GroupBy.NONE,
        ),
    )

    with pytest.raises(ValueError, match="requires variable_components TIMEPOINT"):
        FuncStepContractValidator.validate_funcstep(step)


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
    )
    return step_plan


def test_validate_artifact_managed_runtime_scope_allows_channel_variable_axis():
    step_plan = _runtime_artifact_step_plan(
        variable_components=(VariableComponents.CHANNEL,),
    )

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


def _artifact_contract_scope():
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
    )


def test_validate_artifact_contract_scope_allows_variable_components():
    FuncStepContractValidator.validate_artifact_contract_scope(
        _artifact_contract_scope()
    )

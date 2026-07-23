from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from openhcs.constants import GroupBy, VariableComponents
from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactInputProjectionPlan,
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecRef,
    ImageArtifactType,
)
from openhcs.core.callable_contract import CallableContract, CallableMetadata
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.function_patterns import (
    CompiledFunctionPattern,
    FunctionPatternSyntax,
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
    NormalizedFunctionItem,
    compile_function_pattern,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    InvocationContractPlan,
    InvocationContractProvider,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.funcstep_contract_validator import (
    FuncStepContractValidator,
)
from openhcs.core.pipeline.function_contracts import (
    allowed_group_by,
    artifact_inputs,
    require_variable_component_stack,
    required_variable_components,
)
from openhcs.core.config import LazyProcessingConfig
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


def _function(name="function"):
    def func(image):
        return image

    func.__name__ = name
    func.__module__ = "builtins"
    return func


def _runtime_sentinel(
    name: str,
    calls: list[str],
) -> Callable[[object], object]:
    def func(image: object) -> object:
        calls.append(name)
        return image

    func.__name__ = name
    func.__module__ = "builtins"
    return func


def _compiled_pattern(func):
    return compile_function_pattern(func, {}, {})


def _compiled_pattern_with_exact_input_edges(
    func: FunctionPatternSyntax,
    input_plans: Mapping[ArtifactSpecRef, ArtifactInputPlan],
) -> CompiledFunctionPattern:
    compiled = compile_function_pattern(func, input_plans, {})
    groups = []
    for group in compiled.groups:
        invocations = []
        for invocation in group.invocations:
            edges = []
            for input_index, spec in enumerate(invocation.contract.artifact_inputs):
                storage_plan = next(
                    plan
                    for plan in input_plans.values()
                    if plan.ref() == spec.ref()
                )
                producer_scope = storage_plan.producer_group_scope()
                invocation_scope = (
                    producer_scope
                    if compiled.is_grouped
                    else ComponentGroupScope.ungrouped()
                )
                edges.append(
                    InvocationArtifactInputEdgePlan(
                        key=InvocationArtifactInputProjectionKey(
                            invocation_key=invocation.key,
                            input_index=input_index,
                        ),
                        spec=spec,
                        storage_plan=storage_plan,
                        projection=ArtifactInputProjectionPlan(
                            invocation_scope=invocation_scope,
                            producer_selection_scope=producer_scope,
                            component_scopes=(
                                () if producer_scope.is_ungrouped else (producer_scope,)
                            ),
                        ),
                    )
                )
            invocations.append(invocation.with_artifact_input_edges(edges))
        groups.append(replace(group, invocations=tuple(invocations)))
    return replace(compiled, groups=tuple(groups))


@dataclass(frozen=True)
class _MetadataTransformProvider(InvocationContractProvider):
    transform: Callable[[CallableMetadata], CallableMetadata]

    def __call__(
        self,
        invocation: NormalizedFunctionItem,
        step_context: ArtifactDeclarationStepContext,
    ) -> InvocationContractPlan:
        del step_context
        return InvocationContractPlan(
            contract=replace(
                invocation.contract,
                metadata=self.transform(invocation.contract.metadata),
            )
        )


def _compiled_semantic_step_plan(
    pattern: FunctionPatternSyntax,
    *,
    provider: InvocationContractProvider,
    variable_components: tuple[VariableComponents, ...] = (),
    group_by: GroupBy = GroupBy.NONE,
) -> CompiledStepPlan:
    return CompiledStepPlan(
        step_index=0,
        step_name="enriched-contract-step",
        step_type="FunctionStep",
        axis_id="A01",
        func=pattern,
        variable_components=variable_components,
        group_by=group_by,
        compiled_function_pattern=compile_function_pattern(
            pattern,
            {},
            {},
            invocation_contract_provider=provider,
        ),
    )


def _artifact_output(
    name="OutputImage",
    *,
    kind=ImageArtifactType,
    materialization=None,
):
    return ArtifactOutputPlan(
        name=name,
        path=f"/tmp/{name}",
        artifact_type=kind,
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


def test_normalized_group_by_resolves_non_grouped_variable_component_conflict():
    assert (
        FuncStepContractValidator.normalized_group_by(
            GroupBy.CHANNEL,
            (VariableComponents.CHANNEL,),
            "step",
            normalize_function_pattern(_function()),
        )
        is GroupBy.NONE
    )


def test_normalized_group_by_rejects_grouped_variable_component_conflict():
    with pytest.raises(
        ValueError,
        match=(
            r"Step 'step' has invalid processing_config: "
            r"group_by=CHANNEL cannot also appear in "
            r"variable_components=\('CHANNEL',\)"
        ),
    ):
        FuncStepContractValidator.normalized_group_by(
            GroupBy.CHANNEL,
            (VariableComponents.CHANNEL,),
            "step",
            normalize_function_pattern({"1": _function()}),
        )


def test_validate_funcstep_rejects_dict_pattern_groupby_none():
    func = _function("channel_branch")
    func.input_memory_type = "numpy"
    func.output_memory_type = "numpy"
    step = FunctionStep(
        func={"1": func},
        name="dict-none",
        processing_config=LazyProcessingConfig(group_by=GroupBy.NONE),
    )

    with pytest.raises(
        ValueError,
        match="Dict pattern requires a concrete group_by component",
    ):
        FuncStepContractValidator.validate_funcstep(step, orchestrator=SimpleNamespace())


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


def test_compiled_group_rejects_enriched_allowed_group_by_before_runtime():
    runtime_calls: list[str] = []
    first = _runtime_sentinel("first", runtime_calls)
    second = _runtime_sentinel("second", runtime_calls)
    provider = _MetadataTransformProvider(
        lambda metadata: replace(
            metadata,
            allowed_group_by=(GroupBy.SITE,),
        )
    )
    step_plan = _compiled_semantic_step_plan(
        {"1": first, "2": second},
        provider=provider,
        group_by=GroupBy.CHANNEL,
    )

    with pytest.raises(ValueError, match="allows group_by SITE; resolved CHANNEL"):
        FuncStepContractValidator.validate_compiled_step_plan(step_plan)

    assert runtime_calls == []


def test_compiled_step_rejects_enriched_stack_requirement_before_runtime():
    runtime_calls: list[str] = []
    process = _runtime_sentinel("full_stack", runtime_calls)
    provider = _MetadataTransformProvider(
        lambda metadata: replace(
            metadata,
            processing_contract=ProcessingContract.PURE_3D,
        )
    )
    step_plan = _compiled_semantic_step_plan(process, provider=provider)

    with pytest.raises(ValueError, match="PURE_3D stack semantics"):
        FuncStepContractValidator.validate_compiled_step_plan(step_plan)

    assert runtime_calls == []


def test_compiled_step_accepts_exact_input_edges_across_scheduler_scope():
    @artifact_inputs(
        ArtifactSpec.input("left", ImageArtifactType),
        ArtifactSpec.input("right", ImageArtifactType),
    )
    @numpy(contract=ProcessingContract.PURE_2D)
    def combine(image):
        return image

    input_plan_values = tuple(
        ArtifactInputPlan(
            name=name,
            path=f"/memory/{name}.pkl",
            artifact_type=ImageArtifactType,
            group_keys=(key,),
            group_component=AllComponents.CHANNEL,
            paths_by_group={
                key: f"/memory/{name}.pkl",
            },
        )
        for name, key in (("left", "1"), ("right", "2"))
    )
    input_plans = {plan.ref(): plan for plan in input_plan_values}
    step_plan = CompiledStepPlan(
        step_index=0,
        step_name="combine",
        step_type="FunctionStep",
        axis_id="A01",
        func=combine,
        variable_components=(),
        group_by=GroupBy.CHANNEL,
        artifact_inputs=input_plans,
        execution_group_scope=ComponentGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.CHANNEL,
        ),
        compiled_function_pattern=_compiled_pattern_with_exact_input_edges(
            combine,
            input_plans,
        ),
    )

    FuncStepContractValidator.validate_compiled_step_plan(step_plan)


def test_compiled_dict_branches_accept_their_exact_input_scopes():
    @artifact_inputs(ArtifactSpec.input("left", ImageArtifactType))
    @numpy(contract=ProcessingContract.PURE_2D)
    def process_left(image):
        return image

    @artifact_inputs(ArtifactSpec.input("right", ImageArtifactType))
    @numpy(contract=ProcessingContract.PURE_2D)
    def process_right(image):
        return image

    pattern = {"1": process_left, "2": process_right}
    input_plan_values = tuple(
        ArtifactInputPlan(
            name=name,
            path=f"/memory/{name}.pkl",
            artifact_type=ImageArtifactType,
            group_keys=(key,),
            group_component=AllComponents.CHANNEL,
            paths_by_group={key: f"/memory/{name}.pkl"},
        )
        for name, key in (("left", "1"), ("right", "2"))
    )
    input_plans = {plan.ref(): plan for plan in input_plan_values}
    step_plan = CompiledStepPlan(
        step_index=0,
        step_name="branch",
        step_type="FunctionStep",
        axis_id="A01",
        func=pattern,
        variable_components=(),
        group_by=GroupBy.CHANNEL,
        artifact_inputs=input_plans,
        execution_group_scope=ComponentGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.CHANNEL,
        ),
        compiled_function_pattern=_compiled_pattern_with_exact_input_edges(
            pattern,
            input_plans,
        ),
    )

    FuncStepContractValidator.validate_compiled_step_plan(step_plan)


def test_compiled_group_allows_distinct_enriched_callables_for_resolved_config():
    runtime_calls: list[str] = []
    first = _runtime_sentinel("first", runtime_calls)
    second = _runtime_sentinel("second", runtime_calls)
    provider = _MetadataTransformProvider(
        lambda metadata: replace(
            metadata,
            allowed_group_by=(GroupBy.CHANNEL,),
            processing_contract=ProcessingContract.PURE_2D,
        )
    )
    step_plan = _compiled_semantic_step_plan(
        {"1": first, "2": second},
        provider=provider,
        group_by=GroupBy.CHANNEL,
    )

    FuncStepContractValidator.validate_compiled_step_plan(step_plan)

    assert runtime_calls == []


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

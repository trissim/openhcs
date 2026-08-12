"""Generic invocation-provider ownership and callable validation tests."""

from __future__ import annotations

from dataclasses import fields, replace
import importlib
from types import MappingProxyType, SimpleNamespace

import pytest

from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    GroupLineageSourceRelation,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    compile_function_pattern,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    CompositeInvocationContractProvider,
    InvocationContractPlan,
    InvocationContractProvider,
    unnamed_main_flow_artifact_name,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.core.runtime_adapters import runtime_adapter
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


def _identity(image, *, sigma: float = 1.0):
    return image


@numpy(contract=ProcessingContract.PURE_3D)
def _native_image(image):
    return image


@artifact_outputs(ArtifactSpec.output("OrigColor", ImageArtifactType))
@numpy(contract=ProcessingContract.PURE_3D)
def _native_named_image(image):
    return image


class _FirstClaimingProvider(InvocationContractProvider):
    def __init__(self, plan: InvocationContractPlan) -> None:
        self.plan = plan

    def __call__(self, invocation, step_context):
        del invocation, step_context
        return self.plan


class _SecondClaimingProvider(InvocationContractProvider):
    def __init__(self, plan: InvocationContractPlan) -> None:
        self.plan = plan

    def __call__(self, invocation, step_context):
        del invocation, step_context
        return self.plan


class _NoClaimProvider(InvocationContractProvider):
    def __call__(self, invocation, step_context):
        del invocation, step_context
        return None


def _normalized_invocation():
    return next(normalize_function_pattern(_identity).iter_items())


def _cellprofiler_invocation_contract(
    module_type,
    invocation,
    step_context: ArtifactDeclarationStepContext,
    *,
    first_module_num: int = 1,
):
    blocks, consumed_names = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=step_context,
    )
    (numbered_blocks,), _next_module_num = module_type.number_step_invocation_blocks(
        (blocks,),
        first_module_num=first_module_num,
    )
    return module_type.invocation_callable_contract(
        invocation=invocation,
        numbered_module_blocks=numbered_blocks,
        consumed_kwarg_names=consumed_names,
        step_context=step_context,
    )


@pytest.mark.parametrize(
    "provider_types",
    (
        (_FirstClaimingProvider, _SecondClaimingProvider),
        (_SecondClaimingProvider, _FirstClaimingProvider),
    ),
)
def test_composite_invocation_provider_rejects_multiple_claims_in_any_order(
    provider_types,
) -> None:
    plan = InvocationContractPlan(CallableContract.from_callable(_identity))
    provider = CompositeInvocationContractProvider(
        tuple(provider_type(plan) for provider_type in provider_types)
    )

    with pytest.raises(
        ValueError, match="FirstClaimingProvider|SecondClaimingProvider"
    ):
        provider(_normalized_invocation(), ArtifactDeclarationStepContext.empty())


def test_composite_invocation_provider_returns_none_for_zero_claims() -> None:
    provider = CompositeInvocationContractProvider((_NoClaimProvider(),))

    assert (
        provider(_normalized_invocation(), ArtifactDeclarationStepContext.empty())
        is None
    )


def test_cellprofiler_invocation_provider_has_exact_immutable_shape() -> None:
    module = importlib.import_module(
        "openhcs.interop.cellprofiler.compile_time_contracts"
    )
    provider_type = vars(module)["CellProfilerInvocationContractProvider"]

    assert issubclass(provider_type, InvocationContractProvider)
    assert tuple(field.name for field in fields(provider_type)) == ("plans",)
    assert provider_type.__dataclass_params__.frozen is True
    assert provider_type.__slots__ == ("plans",)

    key = FunctionInvocationKey("_identity", "default", 0)
    plan = InvocationContractPlan(CallableContract.from_callable(_identity))
    provider = provider_type({(0, key): plan})

    assert isinstance(provider.plans, MappingProxyType)


def test_cellprofiler_declared_module_numbering_preserves_distinct_occurrences() -> None:
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
    from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting

    primary = ModuleBlock(
        name="RelateObjects",
        module_num=91,
        metadata={"module_num": "91", "variable_revision_number": "5"},
        setting_records=[
            ModuleSetting("Select the parent objects", "Nuclei"),
        ],
    )
    companion = ModuleBlock(
        name="RelateObjects",
        module_num=91,
        setting_records=[
            ModuleSetting("Select the child objects", "Cells"),
        ],
    )
    equivalent = (
        replace(
            primary,
            module_num=204,
        ),
        replace(companion, module_num=204),
    )
    different_complete_tuple = (
        primary,
        replace(
            companion,
            setting_records=[
                ModuleSetting("Select the child objects", "Cytoplasm"),
            ],
        ),
    )
    equivalent_different_tuple = tuple(
        replace(block, module_num=block.module_num + 300)
        for block in different_complete_tuple
    )
    different_metadata_tuple = (
        replace(
            primary,
            module_num=504,
            metadata={"module_num": "504", "variable_revision_number": "5"},
        ),
        replace(companion, module_num=504),
    )

    numbered, next_module_num = CellProfilerModule.number_step_invocation_blocks(
        (
            (primary, companion),
            equivalent,
            different_complete_tuple,
            equivalent_different_tuple,
            different_metadata_tuple,
        ),
        first_module_num=7,
    )

    assert tuple(
        tuple(block.module_num for block in blocks) for blocks in numbered
    ) == (
        (7, 8),
        (9, 10),
        (11, 12),
        (13, 14),
        (15, 16),
    )
    assert next_module_num == 17
    assert primary.module_num == 91
    assert "module_number" not in vars(CellProfilerModule)

    next_step, final_module_num = CellProfilerModule.number_step_invocation_blocks(
        ((primary, companion),),
        first_module_num=next_module_num,
    )
    assert tuple(block.module_num for block in next_step[0]) == (17, 18)
    assert final_module_num == 19


def test_grouped_cellprofiler_measurements_keep_numbered_module_identity() -> None:
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
    from openhcs.processing.backends.cellprofiler import identify_primary_objects

    source_specs = ArtifactSpecCollection(
        (
            ArtifactSpec.input("Stain1", ImageArtifactType),
            ArtifactSpec.input("Stain2", ImageArtifactType),
        )
    )
    step_context = ArtifactDeclarationStepContext(
        step_name="IdentifyPrimaryObjects",
        step_index=5,
        available_artifacts=source_specs,
        main_flow_artifacts=source_specs,
    )
    pattern = {
        "1": (
            identify_primary_objects,
            {
                "select_the_input_image": "Stain1",
                "name_the_primary_objects_to_be_identified": "Objects1",
            },
        ),
        "2": (
            identify_primary_objects,
            {
                "select_the_input_image": "Stain2",
                "name_the_primary_objects_to_be_identified": "Objects2",
            },
        ),
    }
    module_type = CellProfilerModule.require_callable_contract_owner(
        next(normalize_function_pattern(identify_primary_objects).iter_items()).contract
    )
    step_invocation_blocks = []
    contracts = []
    numbered_module_nums = []
    for invocation in normalize_function_pattern(pattern).iter_items():
        blocks, consumed_kwarg_names = module_type.module_blocks_for_invocation(
            invocation=invocation,
            step_context=step_context,
        )
        step_invocation_blocks.append(blocks)
        numbered_invocations, _next_module_num = (
            CellProfilerModule.number_step_invocation_blocks(
                tuple(step_invocation_blocks),
                first_module_num=7,
            )
        )
        numbered_blocks = numbered_invocations[-1]
        contract, _consumed_kwarg_names = module_type.invocation_callable_contract(
            invocation=invocation,
            numbered_module_blocks=numbered_blocks,
            consumed_kwarg_names=consumed_kwarg_names,
            step_context=step_context,
        )
        numbered_module_nums.append(numbered_blocks[0].module_num)
        contracts.append(contract)

    measurement_specs = tuple(
        contract.artifact_outputs.for_artifact_type(MeasurementsArtifactType)[0]
        for contract in contracts
    )
    assert tuple(numbered_module_nums) == (7, 8)
    assert tuple(spec.name for spec in measurement_specs) == (
        "IdentifyPrimaryObjects_7_measurements",
        "IdentifyPrimaryObjects_8_measurements",
    )
    assert tuple(spec.relations for spec in measurement_specs) == (
        (
            ArtifactSpecRelation(ArtifactSpec.input("Stain1", ImageArtifactType).ref()),
            GroupLineageSourceRelation(
                ArtifactSpec.input("Stain1", ImageArtifactType).ref()
            ),
            ArtifactSpecRelation(
                ArtifactSpec.output("Objects1", ObjectLabelsArtifactType).ref()
            ),
        ),
        (
            ArtifactSpecRelation(ArtifactSpec.input("Stain2", ImageArtifactType).ref()),
            GroupLineageSourceRelation(
                ArtifactSpec.input("Stain2", ImageArtifactType).ref()
            ),
            ArtifactSpecRelation(
                ArtifactSpec.output("Objects2", ObjectLabelsArtifactType).ref()
            ),
        ),
    )
    ArtifactSpecCollection(
        spec for contract in contracts for spec in contract.artifact_outputs
    ).unique(conflict_context="grouped CellProfiler invocation output")


def test_classification_rules_remain_public_while_declaring_prior_measurements() -> (
    None
):
    from openhcs.core.artifacts import (
        ArtifactInputPlan,
        ArtifactOutputPlan,
        ArtifactSpecCollection,
        ArtifactSpecRelation,
        MeasurementsArtifactType,
        ObjectLabelsArtifactType,
    )
    from openhcs.core.pipeline.artifact_planning import (
        artifact_producers_for_outputs,
    )
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
    from openhcs.processing.backends.cellprofiler import (
        classify_objects_single_measurement,
    )
    from openhcs.processing.backends.cellprofiler.classification import (
        ClassificationBinChoice,
        SingleMeasurementClassificationRule,
    )
    from openhcs.processing.backends.cellprofiler.intensity import (
        MeasureObjectIntensityModule,
    )
    from openhcs.processing.backends.cellprofiler.shape import (
        MeasureObjectSizeShapeModule,
    )

    source = ArtifactSpec.input("SubtractedRed", ImageArtifactType)
    objects = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)
    object_input_ref = objects.for_plan_type(ArtifactInputPlan).ref()
    size_measurements = ArtifactSpec.output(
        "MeasureObjectSizeShape_1_measurements",
        MeasurementsArtifactType,
        measurement_feature_owner=MeasureObjectSizeShapeModule,
        relations=(ArtifactSpecRelation(object_input_ref),),
    )
    intensity_measurements = ArtifactSpec.output(
        "MeasureObjectIntensity_2_measurements",
        MeasurementsArtifactType,
        measurement_feature_owner=MeasureObjectIntensityModule,
        relations=(
            ArtifactSpecRelation(object_input_ref),
            ArtifactSpecRelation(source.ref()),
        ),
    )
    available_artifacts = ArtifactSpecCollection(
        (source, objects, size_measurements, intensity_measurements)
    )
    rules = (
        SingleMeasurementClassificationRule(
            measurement_feature="AreaShape_Area",
            bin_choice=ClassificationBinChoice.CUSTOM,
            custom_thresholds=(0.0, 5.0, 20.0),
            bin_names=("Tiny", "Small", "Large"),
        ),
        SingleMeasurementClassificationRule(
            measurement_feature="Intensity_MeanIntensity_SubtractedRed",
            bin_choice=ClassificationBinChoice.CUSTOM,
            custom_thresholds=(0.05,),
            bin_names=("White", "Red"),
        ),
    )
    pattern = (
        classify_objects_single_measurement,
        {"classification_rules": rules},
    )
    invocation = next(normalize_function_pattern(pattern).iter_items())
    step_context = ArtifactDeclarationStepContext(
        step_name="ClassifyObjects",
        step_index=3,
        available_artifacts=available_artifacts,
        main_flow_artifacts=ArtifactSpecCollection(
            (objects.for_plan_type(ArtifactInputPlan),)
        ),
        available_artifact_producers=(
            *artifact_producers_for_outputs(
                (objects,),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey(
                        "identify_primary_objects",
                        DEFAULT_GROUP_KEY,
                        0,
                    ),
                ),
            ),
            *artifact_producers_for_outputs(
                (size_measurements,),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey(
                        "measure_object_size_shape",
                        DEFAULT_GROUP_KEY,
                        0,
                    ),
                ),
            ),
            *artifact_producers_for_outputs(
                (intensity_measurements,),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey(
                        "measure_object_intensity",
                        DEFAULT_GROUP_KEY,
                        0,
                    ),
                ),
            ),
        ),
    )
    module_type = CellProfilerModule.require_module("ClassifyObjectsSingleMeasurement")
    module_contract, consumed_names = _cellprofiler_invocation_contract(
        module_type,
        invocation,
        step_context,
    )
    contract_plan = InvocationContractPlan(
        contract=module_contract,
        consumed_kwarg_names=consumed_names,
    )
    input_plans = {
        spec.ref(): ArtifactInputPlan(
            name=spec.name,
            path=f"/memory/{spec.name}.pkl",
            artifact_type=spec.artifact_type,
        )
        for spec in module_contract.artifact_inputs.of_artifact_type(
            MeasurementsArtifactType
        )
    }
    output_plans = {
        spec.ref(): ArtifactOutputPlan(
            name=spec.name,
            path=f"/memory/{spec.name}.pkl",
            artifact_type=spec.artifact_type,
        )
        for spec in module_contract.artifact_outputs
    }

    compiled = compile_function_pattern(
        pattern,
        input_plans,
        output_plans,
        invocation_contract_provider=_FirstClaimingProvider(contract_plan),
        step_context=step_context,
    ).default_group.invocations[0]

    assert consumed_names == ()
    assert module_contract.artifact_inputs.names_of_artifact_type(
        MeasurementsArtifactType
    ) == (
        size_measurements.name,
        intensity_measurements.name,
    )
    assert dict(compiled.kwargs)["classification_rules"] == rules
    assert compiled.contract.artifact_inputs.names_of_artifact_type(
        MeasurementsArtifactType
    ) == (
        size_measurements.name,
        intensity_measurements.name,
    )


def test_cellprofiler_invocation_provider_rejects_duplicate_exact_key() -> None:
    from openhcs.interop.cellprofiler.compile_time_contracts import (
        CellProfilerInvocationContractProvider,
    )

    key = FunctionInvocationKey("_identity", "default", 0)
    plan = InvocationContractPlan(CallableContract.from_callable(_identity))

    class DuplicateItems(dict):
        def items(self):
            return (((0, key), plan), ((0, key), plan))

    with pytest.raises(ValueError, match="Duplicate CellProfiler invocation"):
        CellProfilerInvocationContractProvider(DuplicateItems())


def test_cellprofiler_invocation_provider_rejects_noncanonical_callable_before_runtime() -> (
    None
):
    from openhcs.interop.cellprofiler.compile_time_contracts import (
        CellProfilerInvocationContractProvider,
    )
    from openhcs.processing.backends.cellprofiler import color_to_gray

    runtime_calls: list[str] = []

    def counterfeit(image):
        runtime_calls.append("counterfeit")
        return image

    counterfeit.__name__ = color_to_gray.__name__
    counterfeit.__module__ = color_to_gray.__module__
    invocation = next(normalize_function_pattern(counterfeit).iter_items())
    provider = CellProfilerInvocationContractProvider(
        {
            (6, invocation.key): InvocationContractPlan(
                CallableContract.from_callable(color_to_gray)
            )
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            r"step 6.*Counterfeit.*FunctionInvocationKey.*ColorToGrayModule.*"
            r"declaration-owned canonical callable"
        ),
    ):
        provider(
            invocation,
            ArtifactDeclarationStepContext(
                step_name="Counterfeit",
                step_index=6,
            ),
        )

    assert runtime_calls == []


def test_callable_contract_exposes_compile_time_public_kwarg_validation() -> None:
    contract = CallableContract.from_callable(_identity)

    assert callable(contract.validate_public_kwargs)


def test_compile_function_pattern_rejects_unknown_public_kwarg() -> None:
    with pytest.raises(TypeError, match="unexpected|unknown"):
        compile_function_pattern((_identity, {"not_a_parameter": 2}), {}, {})


def test_compile_function_pattern_preserves_validated_kwargs_exactly() -> None:
    compiled = compile_function_pattern((_identity, {"sigma": 2.5}), {}, {})

    assert compiled.default_group.invocations[0].kwargs == (("sigma", 2.5),)


def test_adapter_var_kwargs_cannot_admit_unknown_behavior_kwargs() -> None:
    def permissive_runtime_callable(_resolved, _contract):
        def invoke(*args, **kwargs):
            return args, kwargs

        return invoke

    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        runtime_callable_factory=permissive_runtime_callable,
    )
    def strict(image, *, runtime=None, sigma: float = 1.0):
        return image

    with pytest.raises(TypeError, match="invalid public kwargs"):
        compile_function_pattern((strict, {"adapter_only": True}), {}, {})


def test_invocation_contract_plan_consumes_only_authored_compile_time_kwargs() -> None:
    plan = InvocationContractPlan(
        CallableContract.from_callable(_identity),
        consumed_kwarg_names=("compile_only",),
    )
    compiled = compile_function_pattern(
        (_identity, {"compile_only": "declared"}),
        {},
        {},
        invocation_contract_provider=_FirstClaimingProvider(plan),
        step_context=ArtifactDeclarationStepContext(
            step_name="CompileOnly",
            step_index=4,
        ),
    )

    assert compiled.default_group.invocations[0].kwargs == ()


def test_invocation_contract_plan_rejects_unwritten_consumed_kwarg_with_context() -> (
    None
):
    plan = InvocationContractPlan(
        CallableContract.from_callable(_identity),
        consumed_kwarg_names=("compile_only",),
    )

    with pytest.raises(
        ValueError,
        match=r"step 4.*CompileOnly.*FunctionInvocationKey.*compile_only",
    ):
        compile_function_pattern(
            _identity,
            {},
            {},
            invocation_contract_provider=_FirstClaimingProvider(plan),
            step_context=ArtifactDeclarationStepContext(
                step_name="CompileOnly",
                step_index=4,
            ),
        )


def test_cellprofiler_provider_reconstructs_exact_contract_from_public_step() -> None:
    from openhcs.constants.input_source import InputSource
    from openhcs.core.artifacts import ImageArtifactType
    from openhcs.core.compiled_step_plan import CompiledStepPlan
    from openhcs.core.config import (
        GlobalPipelineConfig,
        LazyProcessingConfig,
        PipelineConfig,
    )
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.pipeline.compilation_session import CompilationSession
    from openhcs.core.pipeline.step_snapshot import StepSnapshot
    from openhcs.core.source_bindings import (
        NamedSourceBinding,
        StepSourceBindingsConfig,
    )
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.interop.cellprofiler.compile_time_contracts import (
        CellProfilerInvocationContractProviderFactory,
    )
    from openhcs.interop.cellprofiler.runtime.module_execution import (
        CellProfilerModuleExecutor,
    )
    from openhcs.processing.backends.cellprofiler import color_to_gray
    from openhcs.processing.backends.cellprofiler.color import ColorToGrayMode

    step = FunctionStep(
        func=(
            color_to_gray,
            {
                "mode": ColorToGrayMode.COMBINE,
                "image_type": "rgb",
                "name_the_output_image": "OrigGray",
            },
        ),
        name="ColorToGray",
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START,
        ),
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(
                NamedSourceBinding(
                    alias="OrigColor",
                    artifact_kind=ImageArtifactType,
                ),
            ),
        ),
    )
    snapshot = StepSnapshot(
        index=0, scope_id="provider-test::functionstep_0", step=step
    )
    session = CompilationSession.from_context(
        context=ProcessingContext(
            step_plans={
                0: CompiledStepPlan(
                    step_index=0,
                    step_name=step.name,
                    step_type=step.__class__.__name__,
                    axis_id="A01",
                )
            },
            axis_id="A01",
        ),
        steps=(step,),
        orchestrator=SimpleNamespace(pipeline_config=PipelineConfig()),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=(snapshot,),
    )
    callable_state = vars(color_to_gray).copy()

    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )
    invocation = next(normalize_function_pattern(step.func).iter_items())
    assert provider is not None
    plan = provider(
        invocation,
        ArtifactDeclarationStepContext(
            step_name=step.name,
            step_index=0,
            source_bindings=step.source_bindings,
            group_by=step.processing_config.group_by,
            input_source=step.processing_config.input_source,
        ),
    )

    assert plan is not None
    contract = plan.contract
    assert plan.consumed_kwarg_names == ("name_the_output_image",)
    assert tuple(spec.name for spec in contract.artifact_inputs) == ("OrigColor",)
    assert tuple(spec.name for spec in contract.artifact_outputs) == ("OrigGray",)
    assert isinstance(
        plan.contract.resolve_runtime_callable(),
        CellProfilerModuleExecutor,
    )
    assert vars(color_to_gray) == callable_state


def test_cellprofiler_provider_advances_native_artifacts_through_generic_graph() -> (
    None
):
    from openhcs.core.compiled_step_plan import CompiledStepPlan
    from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.pipeline.compilation_session import CompilationSession
    from openhcs.core.pipeline.step_snapshot import StepSnapshot
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.interop.cellprofiler.compile_time_contracts import (
        CellProfilerInvocationContractProvider,
        CellProfilerInvocationContractProviderFactory,
    )
    from openhcs.processing.backends.cellprofiler import color_to_gray
    from openhcs.processing.backends.cellprofiler.color import ColorToGrayMode

    native_step = FunctionStep(func=_native_named_image, name="NativeImage")
    cellprofiler_step = FunctionStep(
        func=(
            color_to_gray,
            {
                "mode": ColorToGrayMode.COMBINE,
                "name_the_output_image": "Gray",
            },
        ),
        name="ColorToGray",
    )

    def snapshot(index: int, step: FunctionStep) -> StepSnapshot:
        return StepSnapshot(
            index=index, scope_id=f"mixed-provider::functionstep_{index}", step=step
        )

    steps = (native_step, cellprofiler_step)
    snapshots = tuple(snapshot(index, step) for index, step in enumerate(steps))
    plans = {
        index: CompiledStepPlan(
            step_index=index,
            step_name=step.name,
            step_type=step.__class__.__name__,
            axis_id="A01",
        )
        for index, step in enumerate(steps)
    }
    session = CompilationSession.from_context(
        context=ProcessingContext(step_plans=plans, axis_id="A01"),
        steps=steps,
        orchestrator=SimpleNamespace(pipeline_config=PipelineConfig()),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object(), 1: object()},
        snapshots=snapshots,
    )

    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )

    assert isinstance(provider, CellProfilerInvocationContractProvider)
    assert tuple(provider.plans) == (
        (
            1,
            FunctionInvocationKey(
                function_name="color_to_gray",
                group_key=DEFAULT_GROUP_KEY,
                position=0,
            ),
        ),
    )
    contract = next(iter(provider.plans.values())).contract
    module_contract = contract
    assert tuple(spec.name for spec in module_contract.artifact_inputs) == (
        "OrigColor",
    )
    assert tuple(spec.name for spec in module_contract.artifact_outputs) == ("Gray",)


def test_cellprofiler_provider_leaves_native_same_name_callable_unclaimed() -> None:
    from openhcs.core.compiled_step_plan import CompiledStepPlan
    from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.pipeline.compilation_session import CompilationSession
    from openhcs.core.pipeline.step_snapshot import StepSnapshot
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.interop.cellprofiler.compile_time_contracts import (
        CellProfilerInvocationContractProviderFactory,
    )
    from openhcs.processing.backends.processors.numpy_processor import crop

    step = FunctionStep(func=crop, name="Native NumPy crop")
    snapshot = StepSnapshot(
        index=0,
        scope_id="native-crop-provider::functionstep_0",
        step=step,
    )
    session = CompilationSession.from_context(
        context=ProcessingContext(
            step_plans={
                0: CompiledStepPlan(
                    step_index=0,
                    step_name=step.name,
                    step_type=step.__class__.__name__,
                    axis_id="A01",
                )
            },
            axis_id="A01",
        ),
        steps=(step,),
        orchestrator=SimpleNamespace(pipeline_config=PipelineConfig()),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=(snapshot,),
    )

    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )

    assert provider is None


def test_cellprofiler_invocation_contract_uses_one_complete_step_context() -> None:
    from openhcs.constants.constants import (
        AllComponents,
        GroupBy,
    )
    from openhcs.constants.input_source import InputSource
    from openhcs.core.artifacts import (
        ArtifactSpec,
        ArtifactSpecCollection,
        ImageArtifactType,
    )
    from openhcs.core.source_bindings import (
        ComponentSelector,
        NamedSourceBinding,
        StepSourceBindingsConfig,
    )
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
    from openhcs.processing.backends.cellprofiler import (
        correct_illumination_calculate,
    )

    module_type = CellProfilerModule.require_module("CorrectIlluminationCalculate")
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=tuple(
            NamedSourceBinding(
                alias=f"OrigStain{channel}",
                component_identity=(
                    ComponentSelector(
                        component=AllComponents.CHANNEL,
                        value=str(channel),
                    ),
                ),
            )
            for channel in (1, 2)
        ),
    )
    source_artifacts = ArtifactSpecCollection(
        ArtifactSpec.input(f"OrigStain{channel}", ImageArtifactType)
        for channel in (1, 2)
    )
    step_context = ArtifactDeclarationStepContext(
        step_name="CorrectIlluminationCalculate",
        step_index=3,
        source_bindings=source_bindings,
        group_by=GroupBy.CHANNEL,
        input_source=InputSource.PIPELINE_START,
        available_artifacts=source_artifacts,
        main_flow_artifacts=source_artifacts,
    )
    invocation = next(
        normalize_function_pattern(correct_illumination_calculate).iter_items()
    )

    block_contexts: list[tuple[int, tuple[str, ...]]] = []
    contract_contexts: list[tuple[int, tuple[str, ...]]] = []
    block_descriptor = CellProfilerModule.__dict__["module_blocks_for_invocation"]
    contract_function = CellProfilerModule.callable_contract.__func__

    def capture_blocks(cls, **kwargs):
        context = kwargs["step_context"]
        block_contexts.append(
            (
                id(context),
                tuple(
                    binding.alias
                    for binding in context.source_bindings.binding_declarations
                ),
            )
        )
        return block_descriptor.__func__(cls, **kwargs)

    def capture_contract(cls, **kwargs):
        context = kwargs["step_context"]
        contract_contexts.append(
            (
                id(context),
                tuple(
                    binding.alias
                    for binding in context.source_bindings.binding_declarations
                ),
            )
        )
        return contract_function(cls, **kwargs)

    CellProfilerModule.module_blocks_for_invocation = classmethod(capture_blocks)
    CellProfilerModule.callable_contract = classmethod(capture_contract)
    try:
        contract, consumed_names = _cellprofiler_invocation_contract(
            module_type,
            invocation,
            step_context,
        )
    finally:
        CellProfilerModule.module_blocks_for_invocation = block_descriptor
        del CellProfilerModule.callable_contract

    assert len(block_contexts) == 1
    assert len(contract_contexts) == 2
    assert {
        context_id for context_id, _aliases in (*block_contexts, *contract_contexts)
    } == {id(step_context)}
    assert {
        aliases for _context_id, aliases in (*block_contexts, *contract_contexts)
    } == {("OrigStain1", "OrigStain2")}
    assert contract.artifact_inputs.names_of_artifact_type(ImageArtifactType) == (
        "OrigStain1",
        "OrigStain2",
    )
    assert consumed_names == ()


def test_grouped_contracts_project_prior_main_flow_by_producer_ownership() -> None:
    from openhcs.constants.constants import GroupBy
    from openhcs.core.artifacts import ArtifactInputPlan, ArtifactSpecCollection
    from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
    from openhcs.processing.backends.cellprofiler import identify_primary_objects

    grouped_images = tuple(
        ArtifactSpec.output(name, ImageArtifactType) for name in ("CropOne", "CropTwo")
    )
    producer_keys = tuple(
        FunctionInvocationKey("crop", group_key, 0) for group_key in ("1", "2")
    )
    producers = tuple(
        producer
        for image, key in zip(grouped_images, producer_keys, strict=True)
        for producer in artifact_producers_for_outputs(
            (image,),
            groups=(key.group_key,),
            invocation_keys=(key,),
        )
    )
    step_context = ArtifactDeclarationStepContext(
        step_name="IdentifyPrimaryObjects",
        step_index=1,
        group_by=GroupBy.CHANNEL,
        available_artifacts=ArtifactSpecCollection(grouped_images),
        main_flow_artifacts=ArtifactSpecCollection(
            image.for_plan_type(ArtifactInputPlan) for image in grouped_images
        ),
        available_artifact_producers=producers,
    )
    pattern = {
        group_key: (
            identify_primary_objects,
            {"name_the_primary_objects_to_be_identified": object_name},
        )
        for group_key, object_name in (("1", "NucleiOne"), ("2", "NucleiTwo"))
    }

    input_names_by_group = {
        invocation.key.group_key: tuple(
            spec.name
            for spec in _cellprofiler_invocation_contract(
                CellProfilerModule.require_module("IdentifyPrimaryObjects"),
                invocation,
                step_context,
            )[0].artifact_inputs
        )
        for invocation in normalize_function_pattern(pattern).iter_items()
    }

    assert input_names_by_group == {
        "1": ("CropOne",),
        "2": ("CropTwo",),
    }


def test_cellprofiler_invocation_contract_accepts_main_flow_input_without_binding() -> (
    None
):
    from openhcs.constants.input_source import InputSource
    from openhcs.core.artifacts import (
        ArtifactSpec,
        ArtifactSpecCollection,
        ImageArtifactType,
    )
    from openhcs.core.source_bindings import (
        StepSourceBindingsConfig,
    )
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
    from openhcs.processing.backends.cellprofiler import color_to_gray
    from openhcs.processing.backends.cellprofiler.color import ColorToGrayMode

    invocation = next(
        normalize_function_pattern(
            (
                color_to_gray,
                {
                    "mode": ColorToGrayMode.COMBINE,
                    "select_the_input_image": "Missing",
                    "name_the_output_image": "Gray",
                },
            )
        ).iter_items()
    )
    source_bindings = StepSourceBindingsConfig()
    available_artifacts = ArtifactSpecCollection(
        (ArtifactSpec.input("Missing", ImageArtifactType),)
    )

    step_context = ArtifactDeclarationStepContext(
        step_name="ColorToGray",
        step_index=7,
        source_bindings=source_bindings,
        input_source=InputSource.PIPELINE_START,
        available_artifacts=available_artifacts,
        main_flow_artifacts=available_artifacts,
    )
    contract, consumed_names = _cellprofiler_invocation_contract(
        CellProfilerModule.require_module("ColorToGray"),
        invocation,
        step_context,
        first_module_num=8,
    )

    assert tuple(spec.name for spec in contract.artifact_inputs) == ("Missing",)
    assert consumed_names == ("select_the_input_image", "name_the_output_image")


def test_align_omission_probe_rejects_insufficient_static_source_without_throwing() -> (
    None
):
    from openhcs.constants.input_source import InputSource
    from openhcs.core.artifacts import (
        ArtifactSpec,
        ArtifactSpecCollection,
        ImageArtifactType,
    )
    from openhcs.core.source_bindings import (
        NamedSourceBinding,
        StepSourceBindingsConfig,
    )
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
    from openhcs.processing.backends.cellprofiler import align

    invocation = next(normalize_function_pattern(align).iter_items())
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(NamedSourceBinding(alias="OnlyImage"),),
    )
    source_artifacts = ArtifactSpecCollection(
        (ArtifactSpec.input("OnlyImage", ImageArtifactType),)
    )
    module_type = CellProfilerModule.require_module("Align")
    blocks, consumed_names = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=ArtifactDeclarationStepContext(
            step_name="Align",
            step_index=4,
            source_bindings=source_bindings,
            input_source=InputSource.PIPELINE_START,
            available_artifacts=source_artifacts,
            main_flow_artifacts=source_artifacts,
        ),
    )

    assert blocks == ()
    assert consumed_names == ()


def test_cellprofiler_provider_rejects_under_specified_one_image_align() -> None:
    from openhcs.constants.input_source import InputSource
    from openhcs.core.artifacts import ImageArtifactType
    from openhcs.core.compiled_step_plan import CompiledStepPlan
    from openhcs.core.config import (
        GlobalPipelineConfig,
        LazyProcessingConfig,
        PipelineConfig,
    )
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.pipeline.compilation_session import CompilationSession
    from openhcs.core.pipeline.step_snapshot import StepSnapshot
    from openhcs.core.source_bindings import (
        NamedSourceBinding,
        StepSourceBindingsConfig,
    )
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.interop.cellprofiler.compile_time_contracts import (
        CellProfilerInvocationContractProviderFactory,
    )
    from openhcs.processing.backends.cellprofiler import align

    step = FunctionStep(
        func=align,
        name="Align",
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START,
        ),
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(
                NamedSourceBinding(
                    alias="OnlyImage",
                    artifact_kind=ImageArtifactType,
                ),
            ),
        ),
    )
    snapshot = StepSnapshot(
        index=0,
        scope_id="align-cardinality::functionstep_0",
        step=step,
    )
    session = CompilationSession.from_context(
        context=ProcessingContext(
            step_plans={
                0: CompiledStepPlan(
                    step_index=0,
                    step_name=step.name,
                    step_type=step.__class__.__name__,
                    axis_id="A01",
                )
            },
            axis_id="A01",
        ),
        steps=(step,),
        orchestrator=SimpleNamespace(pipeline_config=PipelineConfig()),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=(snapshot,),
    )

    with pytest.raises(
        ValueError,
        match=r"Align.*cannot reconstruct an exact module block",
    ):
        CellProfilerInvocationContractProviderFactory.provider_for_session(session)


def test_cellprofiler_invocation_contract_allows_source_binding_supersets() -> None:
    from openhcs.constants.input_source import InputSource
    from openhcs.core.artifacts import (
        ArtifactSpec,
        ArtifactSpecCollection,
        ImageArtifactType,
    )
    from openhcs.core.source_bindings import (
        NamedSourceBinding,
        StepSourceBindingsConfig,
    )
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
    from openhcs.processing.backends.cellprofiler import color_to_gray
    from openhcs.processing.backends.cellprofiler.color import ColorToGrayMode

    invocation = next(
        normalize_function_pattern(
            (
                color_to_gray,
                {
                    "mode": ColorToGrayMode.COMBINE,
                    "select_the_input_image": "OrigColor",
                    "name_the_output_image": "Gray",
                },
            )
        ).iter_items()
    )
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(alias="OrigColor"),
            NamedSourceBinding(alias="Unrelated"),
        ),
    )
    available_artifacts = ArtifactSpecCollection(
        ArtifactSpec.input(binding.alias, ImageArtifactType)
        for binding in source_bindings.binding_declarations
    )

    step_context = ArtifactDeclarationStepContext(
        step_name="ColorToGray",
        step_index=8,
        source_bindings=source_bindings,
        input_source=InputSource.PIPELINE_START,
        available_artifacts=available_artifacts,
        main_flow_artifacts=available_artifacts,
    )
    contract, consumed_names = _cellprofiler_invocation_contract(
        CellProfilerModule.require_module("ColorToGray"),
        invocation,
        step_context,
        first_module_num=9,
    )

    assert contract.artifact_inputs.names_of_artifact_type(ImageArtifactType) == (
        "OrigColor",
    )
    assert consumed_names == (
        "select_the_input_image",
        "name_the_output_image",
    )


def test_calculate_math_provider_keeps_object_identity_and_output_name_public() -> None:
    from openhcs.core.artifacts import (
        MeasurementsArtifactType,
        ObjectLabelsArtifactType,
    )
    from openhcs.core.compiled_step_plan import CompiledStepPlan
    from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.pipeline.compilation_session import CompilationSession
    from openhcs.core.pipeline.step_snapshot import StepSnapshot
    from openhcs.core.source_bindings import (
        NamedSourceBinding,
        SourceProjectionRole,
        StepSourceBindingsConfig,
    )
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.interop.cellprofiler.compile_time_contracts import (
        CellProfilerInvocationContractProviderFactory,
    )
    from openhcs.processing.backends.cellprofiler import (
        calculate_math,
        measure_object_size_shape,
    )
    from openhcs.processing.backends.cellprofiler.image_math import ImageMathOperation

    measurement_step = FunctionStep(
        func=(
            measure_object_size_shape,
            {"select_object_sets_to_measure": "Nuclei"},
        ),
        name="MeasureObjectSizeShape",
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(
                NamedSourceBinding(
                    alias="Nuclei",
                    artifact_kind=ObjectLabelsArtifactType,
                    projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                ),
            ),
        ),
    )
    step = FunctionStep(
        func=(
            calculate_math,
            {
                "output_name": "Ratio",
                "operation": ImageMathOperation.DIVIDE,
                "operand1_feature": "AreaShape_Area",
                "operand2_feature": "AreaShape_Area",
                "operand1_object_name": "Nuclei",
                "operand2_object_name": "Nuclei",
            },
        ),
        name="CalculateMath",
        source_bindings=measurement_step.source_bindings,
    )
    snapshots = tuple(
        StepSnapshot(
            index=index,
            scope_id=f"calculate-math-provider::functionstep_{index}",
            step=current_step,
        )
        for index, current_step in enumerate((measurement_step, step))
    )
    session = CompilationSession.from_context(
        context=ProcessingContext(
            step_plans={
                index: CompiledStepPlan(
                    step_index=index,
                    step_name=current_step.name,
                    step_type=current_step.__class__.__name__,
                    axis_id="A01",
                )
                for index, current_step in enumerate((measurement_step, step))
            },
            axis_id="A01",
        ),
        steps=(measurement_step, step),
        orchestrator=SimpleNamespace(pipeline_config=PipelineConfig()),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object(), 1: object()},
        snapshots=snapshots,
    )

    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )
    invocation = next(normalize_function_pattern(step.func).iter_items())
    assert provider is not None
    plan = provider(
        invocation,
        ArtifactDeclarationStepContext(
            step_name=step.name,
            step_index=1,
            source_bindings=step.source_bindings,
            group_by=step.processing_config.group_by,
            input_source=step.processing_config.input_source,
        ),
    )

    assert plan is not None
    contract = plan.contract
    assert plan.consumed_kwarg_names == ()
    assert dict(invocation.kwargs) == {
        "output_name": "Ratio",
        "operation": ImageMathOperation.DIVIDE,
        "operand1_feature": "AreaShape_Area",
        "operand2_feature": "AreaShape_Area",
        "operand1_object_name": "Nuclei",
        "operand2_object_name": "Nuclei",
    }
    assert (
        contract.artifact_inputs.names_of_artifact_type(ObjectLabelsArtifactType) == ()
    )
    assert contract.artifact_inputs.names_of_artifact_type(
        MeasurementsArtifactType
    ) == ("MeasureObjectSizeShape_1_measurements",)
    assert contract.artifact_outputs.names_of_artifact_type(
        MeasurementsArtifactType
    ) == ("CalculateMath_2_measurements",)
    output = contract.artifact_outputs.by_name_and_artifact_type(
        "CalculateMath_2_measurements",
        MeasurementsArtifactType,
    )
    assert output is not None
    assert tuple(relation.source.name for relation in output.relations) == (
        "MeasureObjectSizeShape_1_measurements",
    )


def test_native_unnamed_main_flow_remains_a_canonical_contract_input() -> None:
    from openhcs.constants.input_source import InputSource
    from openhcs.core.compiled_step_plan import CompiledStepPlan
    from openhcs.core.config import (
        GlobalPipelineConfig,
        LazyProcessingConfig,
        PipelineConfig,
    )
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.pipeline.compilation_session import CompilationSession
    from openhcs.core.pipeline.step_snapshot import StepSnapshot
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.interop.cellprofiler.compile_time_contracts import (
        CellProfilerInvocationContractProviderFactory,
    )
    from openhcs.processing.backends.cellprofiler import color_to_gray
    from openhcs.processing.backends.cellprofiler.color import ColorToGrayMode

    def session_for(
        cellprofiler_step: FunctionStep,
    ) -> tuple[CompilationSession, tuple[FunctionStep, ...]]:
        steps = (
            FunctionStep(func=_native_image, name="NativeImage"),
            cellprofiler_step,
        )
        snapshots = tuple(
            StepSnapshot(
                index=index, scope_id=f"native-cursor::functionstep_{index}", step=step
            )
            for index, step in enumerate(steps)
        )
        return (
            CompilationSession.from_context(
                context=ProcessingContext(
                    step_plans={
                        index: CompiledStepPlan(
                            step_index=index,
                            step_name=step.name,
                            step_type=step.__class__.__name__,
                            axis_id="A01",
                        )
                        for index, step in enumerate(steps)
                    },
                    axis_id="A01",
                ),
                steps=steps,
                orchestrator=SimpleNamespace(pipeline_config=PipelineConfig()),
                global_config=GlobalPipelineConfig(),
                step_state_map={index: object() for index in range(len(steps))},
                snapshots=snapshots,
            ),
            steps,
        )

    session, steps = session_for(
        FunctionStep(
            func=(color_to_gray, {"mode": ColorToGrayMode.COMBINE}),
            name="ColorToGray",
        )
    )

    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )
    native_invocation = next(normalize_function_pattern(steps[0].func).iter_items())
    cp_invocation = next(normalize_function_pattern(steps[1].func).iter_items())
    assert provider is not None
    contract = provider.plans[(1, cp_invocation.key)].contract
    cursor_name = unnamed_main_flow_artifact_name(0, native_invocation.key)

    assert tuple(spec.name for spec in contract.artifact_inputs) == (cursor_name,)
    assert contract.artifact_inputs[0].parameter_name is None

    pipeline_start_session, _ = session_for(
        FunctionStep(
            func=(color_to_gray, {"mode": ColorToGrayMode.COMBINE}),
            name="ColorToGray",
            processing_config=LazyProcessingConfig(
                input_source=InputSource.PIPELINE_START
            ),
        )
    )
    with pytest.raises(ValueError, match="cannot reconstruct an exact module block"):
        CellProfilerInvocationContractProviderFactory.provider_for_session(
            pipeline_start_session
        )

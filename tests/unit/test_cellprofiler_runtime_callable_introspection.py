"""CellProfiler runtime callable introspection behavior."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from openhcs.constants.constants import AllComponents, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyProcessingConfig,
    PipelineConfig,
)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.callable_contract import CallableContract
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSidecarRole,
    ImageArtifactType,
)
from openhcs.core.function_patterns import (
    compile_function_pattern,
    normalize_function_pattern,
)
from openhcs.core.function_reference import FunctionReferenceTransportAuthority
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.compilation_session import CompilationSession
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.source_bindings import (
    ComponentSelector,
    NamedSourceBinding,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from objectstate.object_state import ObjectState
from openhcs.interop.cellprofiler.compile_time_contracts import (
    CellProfilerInvocationContractProviderFactory,
)
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerModuleExecutor,
)
from openhcs.processing.backends.cellprofiler import (
    correct_illumination_calculate,
    crop,
    mask_objects,
    save_images,
)


def _snapshot_for_step(index: int, step: FunctionStep) -> StepSnapshot:
    return StepSnapshot(index=index, scope_id=f"test::functionstep_{index}", step=step)


def _compilation_session_for_steps(
    steps: list[FunctionStep],
) -> CompilationSession:
    snapshots = tuple(
        _snapshot_for_step(index, step) for index, step in enumerate(steps)
    )
    return CompilationSession.from_context(
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
    )


def _compiled_output_plans(provider, pattern, step_context):
    """Build the exact planner outputs required by compiled invocation contracts."""

    specs_by_ref = {}
    for invocation in normalize_function_pattern(pattern).iter_items():
        plan = provider(invocation, step_context)
        assert plan is not None
        for spec in plan.contract.artifact_outputs:
            specs_by_ref.setdefault(spec.ref(), spec)
    return {
        spec.ref(): ArtifactOutputPlan(
            spec.name,
            f"/tmp/{index}_{spec.name}",
            artifact_type=spec.artifact_type,
            sidecar_role=spec.sidecar_role,
        )
        for index, spec in enumerate(specs_by_ref.values())
    }


def test_public_cellprofiler_backend_import_registers_compile_time_contract_provider():
    """Plain public CP callables must be enough for compiler-time contract derivation."""
    script = """
from openhcs.core.invocation_artifacts import InvocationContractProviderFactory
from openhcs.processing.backends.cellprofiler import identify_primary_objects
names = sorted(cls.__name__ for cls in InvocationContractProviderFactory.__registry__.values())
assert "CellProfilerInvocationContractProviderFactory" in names, names
assert callable(identify_primary_objects)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )


def test_imported_function_step_values_remain_signature_diffs_in_object_state():
    """Loaded pipeline values must not become ObjectState reset defaults."""
    state = ObjectState(
        FunctionStep(
            name="Loaded CP Step",
            enabled=False,
            debug_pause=True,
            processing_config=LazyProcessingConfig(
                variable_components=[VariableComponents.SITE],
                input_source=InputSource.PIPELINE_START,
            ),
        ),
        scope_id="plate::functionstep_0",
    )

    assert state.parameters["name"] == "Loaded CP Step"
    assert state._signature_defaults["name"] is None
    assert state.parameters["enabled"] is False
    assert state._signature_defaults["enabled"] is True
    assert state.parameters["debug_pause"] is True
    assert state._signature_defaults["debug_pause"] is False
    assert state.parameters["processing_config.variable_components"] == [
        VariableComponents.SITE
    ]
    assert state._signature_defaults["processing_config.variable_components"] is None
    assert (
        state.parameters["processing_config.input_source"] is InputSource.PIPELINE_START
    )
    assert state._signature_defaults["processing_config.input_source"] is None
    assert {
        "name",
        "enabled",
        "debug_pause",
        "processing_config.variable_components",
        "processing_config.input_source",
    } <= state.signature_diff_fields

    state.reset_parameter("enabled")
    state.reset_parameter("processing_config.input_source")

    assert state.parameters["enabled"] is True
    assert state.parameters["processing_config.input_source"] is None


def test_step_invocation_contract_provider_validates_public_cellprofiler_config():
    """CP contracts are derived from public function and source-binding declarations."""
    step = FunctionStep(
        func=(
            mask_objects,
            {"select_the_input_objects": "Objects1"},
        ),
        name="MaskObjects",
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START,
        ),
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(
                NamedSourceBinding(
                    alias="None",
                    artifact_kind=ImageArtifactType,
                ),
            ),
        ),
    )
    session = _compilation_session_for_steps([step])

    with pytest.raises(
        ValueError,
        match="unknown object_labels artifact 'Objects1'",
    ):
        CellProfilerInvocationContractProviderFactory.provider_for_session(session)


def test_public_compile_time_provider_binds_cellprofiler_runtime_from_step_declarations():
    """Public CP FunctionStep declarations must compile to artifact-aware callables."""
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                artifact_kind=ImageArtifactType,
            ),
        ),
    )
    step = FunctionStep(
        func=(
            correct_illumination_calculate,
            {"name_the_output_image": "IllumStain1"},
        ),
        name="CorrectIlluminationCalculate",
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START,
        ),
        source_bindings=source_bindings,
    )
    session = _compilation_session_for_steps([step])
    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )
    assert provider is not None
    step_context = ArtifactDeclarationStepContext(
        step_index=0,
        source_bindings=source_bindings,
        input_source=InputSource.PIPELINE_START,
    )

    compiled = compile_function_pattern(
        step.func,
        {},
        _compiled_output_plans(provider, step.func, step_context),
        invocation_contract_provider=provider,
        step_context=step_context,
    )
    invocation = next(compiled.iter_invocations())

    contract = invocation.contract
    module_type = CellProfilerModule.require_callable_contract_owner(contract)
    assert module_type.require_module_name() == "CorrectIlluminationCalculate"
    assert [spec.name for spec in contract.artifact_inputs] == ["OrigStain1"]
    assert [spec.name for spec in contract.artifact_outputs] == [
        "IllumStain1",
    ]
    assert (
        invocation.contract.require_processing_contract()
        is CallableContract.from_callable(
            correct_illumination_calculate
        ).require_processing_contract()
    )
    assert invocation.kwargs == ()
    with pytest.raises(TypeError, match="invalid public kwargs"):
        invocation.contract.validate_public_kwargs({"unknown": True})
    assert invocation.contract.func is correct_illumination_calculate
    assert invocation.contract.main_flow_outputs.names() == ("IllumStain1",)
    assert not invocation.contract.preserves_input_main_flow()
    assert isinstance(
        invocation.contract.resolve_runtime_callable(),
        CellProfilerModuleExecutor,
    )
    assert invocation.contract.runtime_adapter is not None


def test_adapter_free_cellprofiler_module_keeps_raw_runtime_callable() -> None:
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="ImageToSave",
                artifact_kind=ImageArtifactType,
            ),
        ),
    )
    step = FunctionStep(
        func=(
            save_images,
            {
                "image_to_save": "ImageToSave",
                "materialized_image_artifact_name": "SavedImage",
            },
        ),
        name="SaveImages",
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START,
        ),
        source_bindings=source_bindings,
    )
    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        _compilation_session_for_steps([step])
    )
    assert provider is not None
    invocation = next(normalize_function_pattern(step.func).iter_items())
    plan = provider(
        invocation,
        ArtifactDeclarationStepContext(
            step_index=0,
            source_bindings=source_bindings,
            input_source=InputSource.PIPELINE_START,
        ),
    )

    assert plan is not None
    assert plan.contract.runtime_adapter is None
    assert plan.contract.resolve_runtime_callable() is save_images
    assert (
        plan.contract.artifact_inputs.names()
        == ("ImageToSave",)
    )
    assert plan.contract.main_flow_outputs.names() == ()
    assert plan.contract.canonical_return_output_specs.names() == ()
    assert plan.contract.trailing_return_output_specs.names() == ("SavedImage",)
    assert (
        plan.contract.trailing_return_output_specs[0].sidecar_role
        is ArtifactSidecarRole.MATERIALIZED_IMAGE_COPY
    )


def test_public_compile_time_provider_uses_step_order_for_repeated_modules():
    """Repeated public CP modules derive identity from each step's public kwargs."""
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                artifact_kind=ImageArtifactType,
            ),
        ),
    )
    steps = [
        FunctionStep(
            func=(
                correct_illumination_calculate,
                {
                    "select_the_input_image": "OrigStain1",
                    "name_the_output_image": "FirstIllum",
                },
            ),
            name="CorrectIlluminationCalculate",
            processing_config=LazyProcessingConfig(
                input_source=InputSource.PIPELINE_START,
            ),
            source_bindings=source_bindings,
        ),
        FunctionStep(
            func=(
                correct_illumination_calculate,
                {
                    "select_the_input_image": "OrigStain1",
                    "name_the_output_image": "SecondIllum",
                },
            ),
            name="CorrectIlluminationCalculate",
            processing_config=LazyProcessingConfig(
                input_source=InputSource.PIPELINE_START,
            ),
            source_bindings=source_bindings,
        ),
    ]
    FunctionReferenceTransportAuthority.reference_pipeline_in_place(steps)
    session = _compilation_session_for_steps(steps)
    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )
    assert provider is not None
    step_context = ArtifactDeclarationStepContext(
        step_index=1,
        source_bindings=source_bindings,
        input_source=InputSource.PIPELINE_START,
    )

    compiled = compile_function_pattern(
        steps[1].func,
        {},
        _compiled_output_plans(provider, steps[1].func, step_context),
        invocation_contract_provider=provider,
        step_context=step_context,
    )
    invocation = next(compiled.iter_invocations())

    assert [
        spec.name for spec in invocation.contract.artifact_outputs
    ] == [
        "SecondIllum",
    ]


def test_cellprofiler_compile_time_contract_provider_derives_single_source_input():
    """Source bindings can provide an unambiguous missing CP input-image setting."""
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                artifact_kind=ImageArtifactType,
            ),
        ),
    )
    step = FunctionStep(
        func=(
            correct_illumination_calculate,
            {"name_the_output_image": "IllumStain1"},
        ),
        name="CorrectIlluminationCalculate",
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START,
        ),
        source_bindings=source_bindings,
    )
    session = _compilation_session_for_steps([step])

    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )

    assert provider is not None
    invocation = next(normalize_function_pattern(step.func).iter_items())
    contract = provider.plans[
        (0, invocation.key)
    ].contract
    assert [spec.name for spec in contract.artifact_inputs] == ["OrigStain1"]
    assert [spec.name for spec in contract.artifact_outputs] == [
        "IllumStain1",
    ]


def test_cellprofiler_compile_time_contract_provider_scopes_grouped_source_bindings():
    """Dict-pattern groups derive source inputs from matching component identities."""
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                artifact_kind=ImageArtifactType,
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="OrigStain2",
                artifact_kind=ImageArtifactType,
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
            ),
        ),
    )
    step = FunctionStep(
        func={
            "1": (
                correct_illumination_calculate,
                {"name_the_output_image": "IllumStain1"},
            ),
            "2": (
                correct_illumination_calculate,
                {"name_the_output_image": "IllumStain2"},
            ),
        },
        name="CorrectIlluminationCalculate",
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START,
        ),
        source_bindings=source_bindings,
    )
    session = _compilation_session_for_steps([step])

    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )

    assert provider is not None
    invocations = tuple(normalize_function_pattern(step.func).iter_items())
    first_contract = provider.plans[
        (0, invocations[0].key)
    ].contract
    second_contract = provider.plans[
        (0, invocations[1].key)
    ].contract
    assert [spec.name for spec in first_contract.artifact_inputs] == ["OrigStain1"]
    assert [spec.name for spec in first_contract.artifact_outputs] == [
        "IllumStain1",
    ]
    assert [spec.name for spec in second_contract.artifact_inputs] == ["OrigStain2"]
    assert [spec.name for spec in second_contract.artifact_outputs] == [
        "IllumStain2",
    ]


def test_compile_time_provider_derives_group_contracts_from_public_pattern_after_transport():
    """Grouped CP contracts must come from public kwargs plus source bindings."""
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                artifact_kind=ImageArtifactType,
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="OrigStain2",
                artifact_kind=ImageArtifactType,
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
            ),
        ),
    )
    module = SimpleNamespace(
        pipeline_steps=[
            FunctionStep(
                func={
                    "1": (
                        correct_illumination_calculate,
                        {"name_the_output_image": "IllumStain1"},
                    ),
                    "2": (
                        correct_illumination_calculate,
                        {"name_the_output_image": "IllumStain2"},
                    ),
                },
                name="CorrectIlluminationCalculate",
                processing_config=LazyProcessingConfig(
                    input_source=InputSource.PIPELINE_START,
                ),
                source_bindings=source_bindings,
            )
        ]
    )
    FunctionReferenceTransportAuthority.reference_pipeline_in_place(
        module.pipeline_steps
    )
    session = _compilation_session_for_steps(module.pipeline_steps)
    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )

    assert provider is not None
    step_context = ArtifactDeclarationStepContext(
        step_index=0,
        source_bindings=source_bindings,
        input_source=InputSource.PIPELINE_START,
    )
    compiled = compile_function_pattern(
        module.pipeline_steps[0].func,
        {},
        _compiled_output_plans(
            provider,
            module.pipeline_steps[0].func,
            step_context,
        ),
        invocation_contract_provider=provider,
        step_context=step_context,
    )
    assert {
        invocation.key.group_key: [
            spec.name for spec in invocation.contract.artifact_outputs
        ]
        for invocation in compiled.iter_invocations()
    } == {
        "1": [
            "IllumStain1",
        ],
        "2": [
            "IllumStain2",
        ],
    }


def test_function_pattern_normalization_accepts_dict_patterns():
    """Function-pattern traversal must be generic, not generated-binding specific."""
    assert tuple(
        item.contract.resolve_runtime_callable()
        for item in normalize_function_pattern(
            {"1": (crop, {}), "2": crop}
        ).iter_items()
    ) == (
        crop,
        crop,
    )

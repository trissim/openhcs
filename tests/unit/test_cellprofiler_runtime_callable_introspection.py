"""CellProfiler runtime callable introspection behavior."""

import copy
from inspect import signature
import subprocess
import sys
from types import SimpleNamespace
from typing import get_args

import pytest
from python_introspect import SignatureAnalyzer, UnifiedParameterAnalyzer, is_enableable

from openhcs.constants.constants import AllComponents, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyProcessingConfig,
    PipelineConfig,
    ProcessingConfig,
    StepMaterializationConfig,
)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.artifacts import (
    ArtifactSpec,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    compile_function_pattern,
    normalize_function_pattern,
)
from openhcs.core.function_step_invocation_contracts import (
    FunctionStepInvocationContractBinding,
    FunctionStepInvocationContracts,
)
from openhcs.core.function_reference import FunctionReferenceTransportAuthority
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.module_artifact_contract import (
    DeclaredArtifactOutputPartition,
    ModuleArtifactContract,
    RecordedArtifactOutputPartition,
    RuntimeArtifactInputPartition,
    SourceArtifactInputPartition,
)
from openhcs.core.runtime_adapters import runtime_adapter_spec_from_callable
from openhcs.core.pipeline.compilation_session import CompilationSession
from openhcs.core.pipeline.step_config_universe import (
    StepConfigRoot,
    StepConfigUniverse,
    step_config_declarations,
)
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.source_bindings import (
    ComponentSelector,
    NamedSourceBinding,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.config_framework.object_state import ObjectState
from openhcs.pyqt_gui.widgets.shared.services.cellprofiler_pipeline_rebinding import (
    CellProfilerPipelineRuntimeBindingService,
)
from openhcs.interop.cellprofiler.compile_time_contracts import (
    cellprofiler_module_settings_invocation_contract_provider_for_session,
)
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerGroupedRuntimeCallable,
    CellProfilerProcessingContractAuthority,
    CellProfilerRuntimeCallable,
    cellprofiler_module_callable,
)
from openhcs.processing.backends.cellprofiler import (
    CellProfilerFunctionCatalog,
    correct_illumination_calculate,
    crop,
    identify_tertiary_objects,
    mask_objects,
)
from openhcs.processing.backends.cellprofiler.crop import CropModule
from openhcs.processing.backends.cellprofiler.illumination import (
    FilterSizeMethod,
    IntensityChoice,
    RescaleOption,
    SmoothingMethod,
)
from openhcs.processing.backends.cellprofiler.neighbors import measure_object_neighbors
from pyqt_reactive.services.function_pattern_code_document import (
    FunctionPatternCodeDocumentService,
)


def crop_contract(
    *,
    inputs: tuple[ArtifactSpec, ...] = (),
    outputs: tuple[ArtifactSpec, ...] = (),
) -> ModuleArtifactContract:
    return ModuleArtifactContract(
        module_name="Crop",
        items=(
            *ModuleArtifactContract.items_for_partition(
                SourceArtifactInputPartition, inputs
            ),
            *ModuleArtifactContract.items_for_partition(
                RecordedArtifactOutputPartition, outputs
            ),
            *ModuleArtifactContract.items_for_partition(
                DeclaredArtifactOutputPartition, outputs
            ),
        ),
    )


def _config_universe_for_step(step: FunctionStep) -> StepConfigUniverse:
    roots = []
    declarations = step_config_declarations()
    for config in (
        step.source_bindings,
        ProcessingConfig(),
        StepMaterializationConfig(enabled=False),
    ):
        declaration = next(
            declaration
            for declaration in declarations
            if type(config) is declaration.config_type
        )
        roots.append(StepConfigRoot(declaration=declaration, value=config))
    return StepConfigUniverse(tuple(roots))


def _snapshot_for_step(index: int, step: FunctionStep) -> StepSnapshot:
    return StepSnapshot(
        index=index,
        scope_id=f"test::functionstep_{index}",
        name=step.name,
        step_type=step.__class__.__name__,
        enabled=bool(step.enabled),
        is_function_step=True,
        func=step.func,
        invocation_contracts=step.invocation_contracts,
        configs=_config_universe_for_step(step),
    )


def _compilation_session_for_steps(
    steps: list[FunctionStep],
) -> CompilationSession:
    snapshots = tuple(
        _snapshot_for_step(index, step)
        for index, step in enumerate(steps)
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


def declared_runtime_callable(func, contract):
    metadata = CellProfilerFunctionCatalog.runtime_metadata(func)
    if metadata is None:
        processing_contract = CellProfilerProcessingContractAuthority.for_callable(func)
        declared_processing_contract = processing_contract.name
    else:
        processing_contract = metadata.processing_contract
        declared_processing_contract = metadata.declared_processing_contract
    return cellprofiler_module_callable(
        func,
        contract,
        processing_contract=processing_contract,
        declared_processing_contract=declared_processing_contract,
    )


def test_public_cellprofiler_backend_import_registers_compile_time_contract_provider():
    """Plain public CP callables must be enough for compiler-time contract derivation."""
    script = """
from openhcs.core.invocation_artifacts import InvocationContractProviderFactory
import openhcs.processing.backends.cellprofiler
names = sorted(cls.__name__ for cls in InvocationContractProviderFactory.__registry__.values())
assert "CellProfilerInvocationContractProviderFactory" in names, names
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )


def test_cellprofiler_runtime_callable_hides_runtime_bound_object_inputs():
    """Artifact-bound object inputs must not appear as editable step parameters."""
    runtime_callable = declared_runtime_callable(
        identify_tertiary_objects,
        ModuleArtifactContract(
            module_name="IdentifyTertiaryObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                    ),
                ),
            ),
        ),
    )

    params = UnifiedParameterAnalyzer.analyze(runtime_callable)

    assert "primary_labels" not in params
    assert "secondary_labels" not in params
    assert "shrink_primary" in params


def test_cellprofiler_runtime_callable_rejects_unowned_object_input_policy_early():
    """Runtime object-input semantics must be derived from the declared module."""
    with pytest.raises(NotImplementedError, match="no nominal input binding policy"):
        declared_runtime_callable(
            identify_tertiary_objects,
            ModuleArtifactContract(
                module_name="IdentifyTertiaryObjects_MCP_TEMP",
                items=(
                    *ModuleArtifactContract.items_for_partition(
                        RuntimeArtifactInputPartition,
                        (
                            ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
                            ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                        ),
                    ),
                ),
            ),
        )


def test_function_pattern_object_state_excludes_runtime_bound_inputs():
    """Function-pattern ObjectStates must share callable-owned exclusions."""
    runtime_callable = declared_runtime_callable(
        identify_tertiary_objects,
        ModuleArtifactContract(
            module_name="IdentifyTertiaryObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                    ),
                ),
            ),
        ),
    )

    exclusions = FunctionPatternCodeDocumentService.reserved_parameter_names(
        runtime_callable
    )

    assert exclusions is not None
    assert "image" in exclusions
    assert "primary_labels" in exclusions
    assert "secondary_labels" in exclusions


def test_cellprofiler_runtime_callable_hides_declared_runtime_parameters():
    """Function-owned runtime parameters must not appear as editable settings."""
    runtime_callable = declared_runtime_callable(
        measure_object_neighbors,
        ModuleArtifactContract(
            module_name="MeasureObjectNeighbors",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (
                        ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
                        ArtifactSpec.input("Neighbors", ObjectLabelsArtifactType),
                    ),
                ),
            ),
        ),
    )

    params = UnifiedParameterAnalyzer.analyze(runtime_callable)

    assert "labels" not in params
    assert "neighbor_labels" not in params
    assert "small_removed_labels" not in params
    assert "small_removed_neighbor_labels" not in params
    assert "neighbors_are_same_objects" not in params
    assert "slice_index" not in params
    assert "neighbor_distance" in params


def test_cellprofiler_runtime_callable_analyzes_raw_backend_signature():
    """Runtime adapter parameters must not leak into UI-facing analysis."""
    runtime_callable = declared_runtime_callable(
        crop,
        crop_contract(),
    )

    params = SignatureAnalyzer.analyze(runtime_callable)

    assert "cellprofiler_runtime" not in params
    assert "runtime_invocation_options" not in params
    assert is_enableable(runtime_callable)
    assert params["enabled"].param_type is bool
    assert params["enabled"].default_value is True
    assert "slice_by_slice" not in params
    assert "crop_shape" in params
    assert CropModule.Shape in get_args(params["crop_shape"].param_type)
    assert runtime_callable.__doc__ == crop.__doc__


def test_cellprofiler_runtime_callable_publishes_resolved_enum_annotations():
    """UI-facing runtime callables must not expose stringified enum annotations."""
    runtime_callable = declared_runtime_callable(
        correct_illumination_calculate,
        ModuleArtifactContract(module_name="CorrectIlluminationCalculate"),
    )

    parameters = signature(runtime_callable).parameters

    assert IntensityChoice in get_args(parameters["intensity_choice"].annotation)
    assert RescaleOption in get_args(parameters["rescale_option"].annotation)
    assert SmoothingMethod in get_args(parameters["smoothing_method"].annotation)
    assert FilterSizeMethod in get_args(parameters["filter_size_method"].annotation)


def test_cellprofiler_runtime_callable_rebuilds_with_nominal_equality():
    """ObjectState dirty projection must not depend on wrapper instance identity."""
    contract = crop_contract(
        outputs=(ArtifactSpec.output("CropBlue", ImageArtifactType),)
    )
    runtime_callable = declared_runtime_callable(crop, contract)

    assert copy.deepcopy(runtime_callable) == runtime_callable
    assert hash(copy.deepcopy(runtime_callable)) == hash(runtime_callable)


def test_cellprofiler_runtime_callable_tuple_stays_clean_in_object_state():
    contract = crop_contract(
        outputs=(ArtifactSpec.output("CropBlue", ImageArtifactType),)
    )
    runtime_callable = declared_runtime_callable(crop, contract)
    state = ObjectState(
        FunctionStep(
            func=(runtime_callable, {"crop_shape": "Rectangle"}),
            name=contract.module_name,
        ),
        scope_id="plate::functionstep_0",
    )

    state._live_resolved["func"] = (
        copy.deepcopy(runtime_callable),
        {"crop_shape": "Rectangle"},
    )
    state._saved_resolved["func"] = (
        runtime_callable,
        {"crop_shape": "Rectangle"},
    )

    assert "func" not in state._compute_dirty_fields()


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


def test_step_invocation_contract_provider_does_not_use_step_owned_cellprofiler_contracts():
    """CP contracts must be derived from public declarations, not hidden step state."""
    output = ArtifactSpec.output("ColocalizedRegion", ObjectLabelsArtifactType)
    contract = ModuleArtifactContract(
        module_name="MaskObjects",
        items=(
            *ModuleArtifactContract.items_for_partition(
                SourceArtifactInputPartition,
                (ArtifactSpec.input("None", ImageArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                RuntimeArtifactInputPartition,
                (ArtifactSpec.input("Objects1", ObjectLabelsArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                RecordedArtifactOutputPartition,
                (output,),
            ),
            *ModuleArtifactContract.items_for_partition(
                DeclaredArtifactOutputPartition,
                (output,),
            ),
        ),
    )
    key = FunctionInvocationKey("mask_objects", "default", 0)
    step = FunctionStep(
        func=mask_objects,
        name="MaskObjects",
        invocation_contracts=FunctionStepInvocationContracts(
            (FunctionStepInvocationContractBinding(key, contract),)
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

    with pytest.raises(ValueError, match="Select the input objects"):
        cellprofiler_module_settings_invocation_contract_provider_for_session(
            session
        )


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
        source_bindings=source_bindings,
    )
    session = _compilation_session_for_steps([step])
    provider = cellprofiler_module_settings_invocation_contract_provider_for_session(
        session
    )
    assert provider is not None

    compiled = compile_function_pattern(
        step.func,
        {},
        {},
        invocation_contract_provider=provider,
        step_context=ArtifactDeclarationStepContext(
            step_index=0,
            source_bindings=source_bindings,
        ),
    )
    invocation = next(compiled.iter_invocations())

    assert invocation.contract.module_artifact_contract is not None
    assert invocation.contract.module_artifact_contract.module_name == (
        "CorrectIlluminationCalculate"
    )
    assert [spec.name for spec in invocation.contract.module_artifact_contract.inputs] == [
        "OrigStain1"
    ]
    assert [spec.name for spec in invocation.contract.module_artifact_contract.outputs] == [
        "IllumStain1"
    ]
    assert isinstance(invocation.contract.func, CellProfilerRuntimeCallable)
    assert runtime_adapter_spec_from_callable(invocation.contract.func) is not None


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
                {"name_the_output_image": "FirstIllum"},
            ),
            name="CorrectIlluminationCalculate",
            source_bindings=source_bindings,
        ),
        FunctionStep(
            func=(
                correct_illumination_calculate,
                {"name_the_output_image": "SecondIllum"},
            ),
            name="CorrectIlluminationCalculate",
            source_bindings=source_bindings,
        ),
    ]
    FunctionReferenceTransportAuthority.reference_pipeline_in_place(steps)
    session = _compilation_session_for_steps(steps)
    provider = cellprofiler_module_settings_invocation_contract_provider_for_session(
        session
    )
    assert provider is not None

    compiled = compile_function_pattern(
        steps[1].func,
        {},
        {},
        invocation_contract_provider=provider,
        step_context=ArtifactDeclarationStepContext(
            step_index=1,
            source_bindings=source_bindings,
        ),
    )
    invocation = next(compiled.iter_invocations())

    assert invocation.contract.module_artifact_contract is not None
    assert [spec.name for spec in invocation.contract.module_artifact_contract.outputs] == [
        "SecondIllum"
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
        source_bindings=source_bindings,
    )
    session = _compilation_session_for_steps([step])

    provider = cellprofiler_module_settings_invocation_contract_provider_for_session(
        session
    )

    assert provider is not None
    contract = provider.contracts_by_module_num[1]
    assert [spec.name for spec in contract.inputs] == ["OrigStain1"]
    assert [spec.name for spec in contract.outputs] == ["IllumStain1"]


def test_cellprofiler_compile_time_contract_provider_scopes_grouped_source_bindings():
    """Dict-pattern groups derive source inputs from matching component identities."""
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                artifact_kind=ImageArtifactType,
                component_identity=(
                    ComponentSelector(AllComponents.CHANNEL, "1"),
                ),
            ),
            NamedSourceBinding(
                alias="OrigStain2",
                artifact_kind=ImageArtifactType,
                component_identity=(
                    ComponentSelector(AllComponents.CHANNEL, "2"),
                ),
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
        source_bindings=source_bindings,
    )
    session = _compilation_session_for_steps([step])

    provider = cellprofiler_module_settings_invocation_contract_provider_for_session(
        session
    )

    assert provider is not None
    first_contract = provider.contracts_by_module_num[1]
    second_contract = provider.contracts_by_module_num[2]
    assert [spec.name for spec in first_contract.inputs] == ["OrigStain1"]
    assert [spec.name for spec in first_contract.outputs] == ["IllumStain1"]
    assert [spec.name for spec in second_contract.inputs] == ["OrigStain2"]
    assert [spec.name for spec in second_contract.outputs] == ["IllumStain2"]


def test_compile_time_provider_derives_group_contracts_from_public_pattern_after_transport():
    """Grouped CP contracts must come from public kwargs plus source bindings."""
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                artifact_kind=ImageArtifactType,
                component_identity=(
                    ComponentSelector(AllComponents.CHANNEL, "1"),
                ),
            ),
            NamedSourceBinding(
                alias="OrigStain2",
                artifact_kind=ImageArtifactType,
                component_identity=(
                    ComponentSelector(AllComponents.CHANNEL, "2"),
                ),
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
                source_bindings=source_bindings,
            )
        ]
    )
    FunctionReferenceTransportAuthority.reference_pipeline_in_place(
        module.pipeline_steps
    )
    assert not isinstance(module.pipeline_steps[0].func, CellProfilerGroupedRuntimeCallable)

    session = _compilation_session_for_steps(module.pipeline_steps)
    provider = cellprofiler_module_settings_invocation_contract_provider_for_session(
        session
    )

    assert provider is not None
    compiled = compile_function_pattern(
        module.pipeline_steps[0].func,
        {},
        {},
        invocation_contract_provider=provider,
        step_context=ArtifactDeclarationStepContext(
            step_index=0,
            source_bindings=source_bindings,
        ),
    )
    assert {
        invocation.key.group_key: [
            spec.name
            for spec in invocation.contract.module_artifact_contract.outputs
        ]
        for invocation in compiled.iter_invocations()
    } == {
        "1": ["IllumStain1"],
        "2": ["IllumStain2"],
    }


def test_compile_time_contract_provider_skips_runtime_bound_cellprofiler_callable():
    """Artifact-bound CP callables already carry their compiler contract."""
    output = ArtifactSpec.output("ColocalizedRegion", ObjectLabelsArtifactType)
    runtime_callable = declared_runtime_callable(
        mask_objects,
        ModuleArtifactContract(
            module_name="MaskObjects",
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition,
                    (ArtifactSpec.input("None", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition,
                    (ArtifactSpec.input("Objects1", ObjectLabelsArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (output,),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (output,),
                ),
            ),
        ),
    )
    step = FunctionStep(
        func=runtime_callable,
        name="MaskObjects",
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

    provider = cellprofiler_module_settings_invocation_contract_provider_for_session(
        session
    )

    assert provider is None


def test_runtime_binding_service_does_not_require_import_context_for_public_steps():
    """Code-mode CP steps compile from public declarations, not loaded .cppipe context."""
    step = FunctionStep(func=crop)

    rebound = CellProfilerPipelineRuntimeBindingService.runtime_bound_pipeline_for_plate(
        import_result_provider=None,
        plate_path="/tmp/example#openhcs-cppipe=Example.cppipe",
        pipeline_steps=[step],
    )

    assert rebound == [step]


def test_function_pattern_normalization_accepts_dict_patterns():
    """Function-pattern traversal must be generic, not generated-binding specific."""
    assert tuple(
        item.contract.resolve_runtime_callable()
        for item in normalize_function_pattern({"1": (crop, {}), "2": crop}).iter_items()
    ) == (
        crop,
        crop,
    )

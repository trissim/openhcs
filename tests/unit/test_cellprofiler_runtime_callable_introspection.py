"""CellProfiler runtime callable introspection behavior."""

import copy
from types import ModuleType
from typing import get_args

import pytest
from python_introspect import SignatureAnalyzer, UnifiedParameterAnalyzer, is_enableable

from openhcs.constants.constants import VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.config import LazyProcessingConfig
from openhcs.core.artifacts import (
    ArtifactSpec,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.function_patterns import compile_function_pattern
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.module_artifact_contract import (
    DeclaredArtifactOutputPartition,
    ModuleArtifactContract,
    RecordedArtifactOutputPartition,
    RuntimeArtifactInputPartition,
    SourceArtifactInputPartition,
)
from openhcs.core.runtime_adapters import runtime_adapter_spec_from_callable
from openhcs.core.source_bindings import (
    NamedSourceBinding,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.config_framework.object_state import ObjectState
from openhcs.interop.cellprofiler.runtime.generated_pipeline import (
    CellProfilerGeneratedRuntimeBindingState,
    CellProfilerGeneratedPipelineInvocationContracts,
    bind_generated_pipeline_runtime,
)
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerProcessingContractAuthority,
    CellProfilerRuntimeCallable,
    cellprofiler_module_callable,
)
from openhcs.processing.backends.cellprofiler import (
    CellProfilerFunctionCatalog,
    crop,
    identify_tertiary_objects,
)
from openhcs.processing.backends.cellprofiler.crop import CropModule
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


def test_generated_runtime_binding_rejects_source_binding_contract_drift():
    """Source binding edits must not silently drift from CP artifact inputs."""
    module = ModuleType("test_generated_cp_pipeline")
    module.pipeline_steps = [
        FunctionStep(
            func=crop,
            source_bindings=StepSourceBindingsConfig(
                bindings=(
                    NamedSourceBinding(
                        alias="WrongBlue",
                        artifact_kind=ImageArtifactType,
                    ),
                ),
            ),
        )
    ]

    with pytest.raises(ValueError, match="source bindings drifted"):
        bind_generated_pipeline_runtime(
            module,
            {
                1: crop_contract(
                    inputs=(ArtifactSpec.input("OrigBlue", ImageArtifactType),)
                )
            },
        )


def test_generated_runtime_binding_accepts_matching_source_binding_contract():
    """Matching source bindings can bind to artifact-managed runtime callables."""
    module = ModuleType("test_generated_cp_pipeline")
    module.pipeline_steps = [
        FunctionStep(
            func=crop,
            source_bindings=StepSourceBindingsConfig(
                bindings=(
                    NamedSourceBinding(
                        alias="OrigBlue",
                        artifact_kind=ImageArtifactType,
                    ),
                ),
            ),
        )
    ]

    bind_generated_pipeline_runtime(
        module,
        {1: crop_contract(inputs=(ArtifactSpec.input("OrigBlue", ImageArtifactType),))},
    )

    assert isinstance(module.pipeline_steps[0].func, CellProfilerRuntimeCallable)


def test_generated_contract_provider_binds_cellprofiler_runtime_at_compile_time():
    """Generated CP contracts stay out of FunctionStep source but compile to runtime callables."""
    contract = crop_contract(
        inputs=(ArtifactSpec.input("OrigBlue", ImageArtifactType),),
        outputs=(ArtifactSpec.output("CropBlue", ImageArtifactType),),
    )
    source_bindings = StepSourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="OrigBlue",
                artifact_kind=ImageArtifactType,
            ),
        ),
    )
    provider = CellProfilerGeneratedPipelineInvocationContracts.from_mapping(
        {1: contract}
    ).invocation_contract_provider

    compiled = compile_function_pattern(
        crop,
        {},
        {},
        invocation_contract_provider=provider,
        step_context=ArtifactDeclarationStepContext(
            step_index=0,
            source_bindings=source_bindings,
        ),
    )
    invocation = next(compiled.iter_invocations())

    assert invocation.contract.module_artifact_contract == contract
    assert isinstance(invocation.contract.func, CellProfilerRuntimeCallable)
    assert runtime_adapter_spec_from_callable(invocation.contract.func) is not None


def test_generated_runtime_binding_state_accepts_nested_function_patterns():
    """Generated CP runtime contract checks must traverse list-shaped FunctionStep specs."""
    contract = crop_contract(
        outputs=(ArtifactSpec.output("CropBlue", ImageArtifactType),)
    )
    runtime_callable = declared_runtime_callable(crop, contract)
    state = CellProfilerGeneratedRuntimeBindingState(
        pipeline_steps=[FunctionStep(func=[(runtime_callable, {})])],
        contracts_by_module_num={1: contract},
    )

    assert state.matches_expected_contracts()


def test_generated_runtime_binding_rejects_callable_contract_order_mismatch():
    """Step-order binding must fail loudly when callable and contract diverge."""
    module = ModuleType("test_generated_cp_pipeline_order")
    module.pipeline_steps = [FunctionStep(func=crop)]

    with pytest.raises(ValueError, match="callable does not match"):
        bind_generated_pipeline_runtime(
            module,
            {
                1: ModuleArtifactContract(
                    module_name="IdentifyPrimaryObjects",
                )
            },
        )

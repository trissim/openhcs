"""CellProfiler runtime callable introspection behavior."""

import copy
from types import ModuleType
from typing import get_args

import pytest
from python_introspect import SignatureAnalyzer, is_enableable

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.source_bindings import (
    GroupedSourceBindings,
    NamedSourceBinding,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.config_framework.object_state import ObjectState
from openhcs.interop.cellprofiler.runtime.generated_pipeline import (
    bind_generated_pipeline_runtime,
)
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerRuntimeCallable,
    cellprofiler_module_callable,
)
from openhcs.processing.backends.cellprofiler import crop
from openhcs.processing.backends.cellprofiler.crop import CropShape


def crop_contract(
    *,
    inputs: tuple[ArtifactSpec, ...] = (),
    outputs: tuple[ArtifactSpec, ...] = (),
) -> ModuleArtifactContract:
    return ModuleArtifactContract(
        module_name="Crop",
        inputs=inputs,
        outputs=outputs,
    )


def test_cellprofiler_runtime_callable_analyzes_raw_backend_signature():
    """Runtime adapter parameters must not leak into UI-facing analysis."""
    runtime_callable = cellprofiler_module_callable(
        crop,
        crop_contract(),
    )

    params = SignatureAnalyzer.analyze(runtime_callable)

    assert "cellprofiler_runtime" not in params
    assert "runtime_invocation_options" not in params
    assert is_enableable(runtime_callable)
    assert params["enabled"].param_type is bool
    assert params["enabled"].default_value is True
    assert params["slice_by_slice"].param_type is bool
    assert params["slice_by_slice"].default_value is False
    assert "crop_shape" in params
    assert CropShape in get_args(params["crop_shape"].param_type)
    assert runtime_callable.__doc__ == crop.__doc__


def test_cellprofiler_runtime_callable_rebuilds_with_nominal_equality():
    """ObjectState dirty projection must not depend on wrapper instance identity."""
    contract = crop_contract(outputs=(ArtifactSpec("CropBlue", ArtifactKind.IMAGE),))
    runtime_callable = cellprofiler_module_callable(crop, contract)

    assert copy.deepcopy(runtime_callable) == runtime_callable
    assert hash(copy.deepcopy(runtime_callable)) == hash(runtime_callable)


def test_cellprofiler_runtime_callable_tuple_stays_clean_in_object_state():
    contract = crop_contract(outputs=(ArtifactSpec("CropBlue", ArtifactKind.IMAGE),))
    runtime_callable = cellprofiler_module_callable(crop, contract)
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


def test_generated_runtime_binding_rejects_source_binding_contract_drift():
    """Source binding edits must not silently drift from CP artifact inputs."""
    module = ModuleType("test_generated_cp_pipeline")
    module.pipeline_steps = [
        FunctionStep(
            func=crop,
            source_bindings=StepSourceBindingsConfig(
                groups=(
                    GroupedSourceBindings(
                        bindings=(
                            NamedSourceBinding(
                                alias="WrongBlue",
                                artifact_kind=ArtifactKind.IMAGE,
                            ),
                        ),
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
                    inputs=(ArtifactSpec("OrigBlue", ArtifactKind.IMAGE),)
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
                groups=(
                    GroupedSourceBindings(
                        bindings=(
                            NamedSourceBinding(
                                alias="OrigBlue",
                                artifact_kind=ArtifactKind.IMAGE,
                            ),
                        ),
                    ),
                ),
            ),
        )
    ]

    bind_generated_pipeline_runtime(
        module,
        {
            1: crop_contract(inputs=(ArtifactSpec("OrigBlue", ArtifactKind.IMAGE),))
        },
    )

    assert isinstance(module.pipeline_steps[0].func, CellProfilerRuntimeCallable)


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

from __future__ import annotations

import numpy as np
import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.aligned_image_payload import (
    AlignedImageSliceContext,
    AlignedImageStack,
)
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import (
    CallableContract,
    CallableMetadata,
    FunctionStepExecutionScope,
)
from openhcs.core.runtime_output_matching import RuntimeReturnedOutputMatcher
from openhcs.core.function_patterns import (
    CompiledFunctionInvocation,
    FunctionInvocationKey,
)


def _contract(
    *outputs: ArtifactSpec,
    inputs: tuple[ArtifactSpec, ...] = (),
    execution_scope: FunctionStepExecutionScope = FunctionStepExecutionScope.AXIS,
) -> CallableContract:
    return CallableContract(
        func=lambda value: value,
        function_name="process",
        module_name=None,
        metadata=CallableMetadata(
            artifact_inputs=inputs,
            artifact_outputs=outputs,
            execution_scope=execution_scope,
        ),
    )


def test_runtime_output_matcher_maps_canonical_and_trailing_slots() -> None:
    image = ArtifactSpec.output("Image", ImageArtifactType)
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)

    resolved = RuntimeReturnedOutputMatcher(
        callable_contract=_contract(image, measurements),
        returned_output=("image", "measurements"),
    ).resolve()

    assert resolved == {
        image.ref(): "image",
        measurements.ref(): "measurements",
    }


def test_runtime_output_matcher_uses_exact_multi_canonical_contexts() -> None:
    first = ArtifactSpec.output("First", ImageArtifactType)
    second = ArtifactSpec.output("Second", ImageArtifactType)
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)
    canonical = AlignedImageStack(
        ("second-value", "first-value"),
        (
            AlignedImageSliceContext.main_flow(
                second.name,
                artifact_kind=second.artifact_type.value,
            ),
            AlignedImageSliceContext.main_flow(
                first.name,
                artifact_kind=first.artifact_type.value,
            ),
        ),
    )

    resolved = RuntimeReturnedOutputMatcher(
        callable_contract=_contract(first, second, measurements),
        returned_output=(canonical, "measurements"),
    ).resolve()

    assert resolved == {
        first.ref(): "first-value",
        second.ref(): "second-value",
        measurements.ref(): "measurements",
    }


def test_runtime_output_matcher_binds_selected_plans_after_resolving_complete_abi() -> (
    None
):
    image = ArtifactSpec.output("Image", ImageArtifactType)
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)
    measurement_plan = ArtifactOutputPlan(
        name=measurements.name,
        path="/memory/measurements",
        artifact_type=measurements.artifact_type,
    )

    resolved, matched_outputs = RuntimeReturnedOutputMatcher(
        callable_contract=_contract(image, measurements),
        returned_output=("image", "measurements"),
    ).resolve_plan_values((measurement_plan,))

    assert resolved == {
        image.ref(): "image",
        measurements.ref(): "measurements",
    }
    assert matched_outputs == (
        (measurement_plan, measurements, "measurements"),
    )


def test_runtime_invocation_selects_storage_without_truncating_callable_abi() -> None:
    first_input = ArtifactSpec.input("FirstInput", ImageArtifactType)
    second_input = ArtifactSpec.input("SecondInput", ImageArtifactType)
    first = ArtifactSpec.output("First", ImageArtifactType)
    second = ArtifactSpec.output("Second", ImageArtifactType)
    first_plan = ArtifactOutputPlan(
        name=first.name,
        path="/memory/first",
        artifact_type=first.artifact_type,
    )
    second_plan = ArtifactOutputPlan(
        name=second.name,
        path="/memory/second",
        artifact_type=second.artifact_type,
    )
    compiled = CompiledFunctionInvocation(
        key=FunctionInvocationKey("process", "default", 0),
        contract=_contract(
            first,
            second,
            inputs=(first_input, second_input),
        ),
        artifact_output_plans=(first_plan, second_plan),
    )
    runtime = compiled.for_runtime_outputs(
        output_plans=(second_plan,),
    )
    first_value = np.zeros((4, 5), dtype=np.float32)
    second_value = np.ones((4, 5), dtype=np.float32)
    returned_stack = AlignedImageStack(
        (first_value, second_value),
        (
            AlignedImageSliceContext.main_flow(
                first.name,
                artifact_kind=first.artifact_type.value,
            ),
            AlignedImageSliceContext.main_flow(
                second.name,
                artifact_kind=second.artifact_type.value,
            ),
        ),
    )

    resolved, matched = RuntimeReturnedOutputMatcher(
        callable_contract=runtime.contract,
        returned_output=returned_stack,
    ).resolve_plan_values(runtime.artifact_output_plans)

    assert runtime.contract.canonical_return_output_specs.specs == (first, second)
    assert runtime.contract.artifact_inputs.specs == (first_input, second_input)
    assert resolved == {
        first.ref(): first_value,
        second.ref(): second_value,
    }
    assert matched == ((second_plan, second, second_value),)


@pytest.mark.parametrize(
    ("returned_output", "expected_count"),
    (("canonical", 0), (("canonical", "first", "second"), 2)),
)
def test_runtime_output_matcher_rejects_trailing_slot_count_mismatch(
    returned_output: object,
    expected_count: int,
) -> None:
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)

    with pytest.raises(
        ValueError,
        match=rf"declared trailing output slots: {expected_count} != 1",
    ):
        RuntimeReturnedOutputMatcher(
            callable_contract=_contract(measurements),
            returned_output=returned_output,
        ).resolve()


def test_runtime_output_matcher_rejects_duplicate_abi_specs() -> None:
    objects = ArtifactSpec.output("Objects", ObjectLabelsArtifactType)

    with pytest.raises(ValueError, match="duplicate artifact ref"):
        RuntimeReturnedOutputMatcher(
            callable_contract=_contract(objects, objects),
            returned_output="objects",
        ).resolve()


def test_runtime_output_matcher_rejects_selected_plan_not_in_abi() -> None:
    declared = ArtifactSpec.output("Declared", ImageArtifactType)
    undeclared = ArtifactSpec.output("Undeclared", ImageArtifactType)
    undeclared_plan = ArtifactOutputPlan(
        name=undeclared.name,
        path="/memory/undeclared",
        artifact_type=undeclared.artifact_type,
    )

    with pytest.raises(ValueError, match="plan .* is not declared by the callable ABI"):
        RuntimeReturnedOutputMatcher(
            callable_contract=_contract(declared),
            returned_output="declared",
        ).resolve_plan_values((undeclared_plan,))


@pytest.mark.parametrize(
    "artifact_type",
    (ImageArtifactType, MeasurementsArtifactType),
)
def test_compiled_output_selection_accepts_exact_runtime_group_projection(
    artifact_type,
) -> None:
    output = ArtifactSpec.output("Output", artifact_type)
    compiled_plan = ArtifactOutputPlan(
        name=output.name,
        path="/memory/output",
        artifact_type=output.artifact_type,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        paths_by_group={
            "1": "/memory/output/1",
            "2": "/memory/output/2",
        },
    )
    invocation = CompiledFunctionInvocation(
        key=FunctionInvocationKey("process", "default", 0),
        contract=_contract(output),
        artifact_output_plans=(compiled_plan,),
    )
    runtime_plan = compiled_plan.for_group("1")

    assert invocation.select_outputs({runtime_plan.ref(): runtime_plan}) == {
        runtime_plan.ref(): runtime_plan
    }


def test_compiled_output_selection_rejects_same_ref_non_owner_projection() -> None:
    output = ArtifactSpec.output("Output", ImageArtifactType)
    compiled_plan = ArtifactOutputPlan(
        name=output.name,
        path="/memory/output",
        artifact_type=output.artifact_type,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        paths_by_group={
            "1": "/memory/output/1",
            "2": "/memory/output/2",
        },
    )
    invocation = CompiledFunctionInvocation(
        key=FunctionInvocationKey("process", "default", 0),
        contract=_contract(output),
        artifact_output_plans=(compiled_plan,),
    )
    drifted_plan = ArtifactOutputPlan(
        name=output.name,
        path="/memory/wrong",
        artifact_type=output.artifact_type,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/wrong"},
    )

    with pytest.raises(ValueError, match="differs from its compiled owner"):
        invocation.select_outputs({drifted_plan.ref(): drifted_plan})


def test_runtime_output_matcher_rejects_context_free_multi_canonical_stack() -> None:
    first = ArtifactSpec.output("First", ImageArtifactType)
    second = ArtifactSpec.output("Second", ImageArtifactType)

    with pytest.raises(ValueError, match="require exact AlignedImageStack"):
        RuntimeReturnedOutputMatcher(
            callable_contract=_contract(first, second),
            returned_output=AlignedImageStack(("first", "second")),
        ).resolve()


def test_plate_scope_uses_first_declared_output_as_canonical() -> None:
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)
    image = ArtifactSpec.output("Image", ImageArtifactType)
    contract = _contract(
        measurements,
        image,
        execution_scope=FunctionStepExecutionScope.PLATE,
    )

    assert contract.canonical_return_output_specs.specs == (measurements,)
    assert contract.trailing_return_output_specs.specs == (image,)
    assert RuntimeReturnedOutputMatcher(
        callable_contract=contract,
        returned_output=("measurements", "image"),
    ).resolve() == {
        measurements.ref(): "measurements",
        image.ref(): "image",
    }

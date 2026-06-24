from types import SimpleNamespace

import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.pipeline.compilation_session import CompilationSession
from openhcs.core.pipeline.compiler import PipelineCompiler
from openhcs.core.pipeline.step_snapshot import StepProcessingSnapshot, StepSnapshot
from openhcs.core.source_bindings import EMPTY_SOURCE_BINDINGS
from openhcs.core.steps.function_step import FunctionStep


def _identity(image):
    return image


def _snapshot(
    index: int,
    name: str = "step",
    source_identity_stack_axes=(),
) -> StepSnapshot:
    return StepSnapshot(
        index=index,
        scope_id=f"plate::functionstep_{index}",
        name=name,
        step_type="FunctionStep",
        enabled=True,
        is_function_step=True,
        func=_identity,
        source_bindings=EMPTY_SOURCE_BINDINGS,
        source_identity_stack_axes=source_identity_stack_axes,
        processing=StepProcessingSnapshot(
            variable_components=("site",),
            group_by=None,
            input_source=None,
            config=SimpleNamespace(),
        ),
        materialization_config=SimpleNamespace(enabled=False),
        injectable_values={},
    )


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        axis_id="A01",
        global_config=SimpleNamespace(),
        plate_path=None,
        step_plans={
            0: CompiledStepPlan(
                step_index=0,
                step_name="step",
                step_type="FunctionStep",
                axis_id="A01",
            )
        },
    )


def test_compilation_session_owns_step_snapshot_plan_invariants():
    step = FunctionStep(func=_identity, name="step")
    step_state = object()
    session = CompilationSession.from_context(
        context=_context(),
        steps=[step],
        orchestrator=SimpleNamespace(),
        step_state_map={0: step_state},
        snapshots=(_snapshot(0),),
    )

    assert session.axis_id == "A01"
    assert session.step(0) is step
    assert session.step_state(0) is step_state
    assert session.snapshot(0).name == "step"
    assert session.plan(0).step_name == "step"


def test_compilation_session_rejects_missing_snapshot():
    step = FunctionStep(func=_identity, name="step")

    with pytest.raises(ValueError, match="one StepSnapshot per step"):
        CompilationSession.from_context(
            context=_context(),
            steps=[step],
            orchestrator=SimpleNamespace(),
            step_state_map={0: object()},
            snapshots=(),
        )


def test_compilation_session_rejects_non_contiguous_snapshot_index():
    step = FunctionStep(func=_identity, name="step")

    with pytest.raises(ValueError, match="index mismatch"):
        CompilationSession.from_context(
            context=_context(),
            steps=[step],
            orchestrator=SimpleNamespace(),
            step_state_map={0: object()},
            snapshots=(_snapshot(1),),
        )


def test_compiler_merges_pipeline_and_step_source_identity_stack_axes():
    step = FunctionStep(
        func=_identity,
        name="step",
        source_identity_stack_axes=(AllComponents.CHANNEL,),
    )
    session = CompilationSession.from_context(
        context=_context(),
        steps=[step],
        orchestrator=SimpleNamespace(),
        step_state_map={0: object()},
        snapshots=(
            _snapshot(
                0,
                source_identity_stack_axes=step.source_identity_stack_axes,
            ),
        ),
        source_identity_stack_axes=frozenset({"z_index"}),
    )

    PipelineCompiler._supplement_step_plans(session)

    assert session.plan(0).source_identity_stack_axes == frozenset(
        {"channel", "z_index"}
    )

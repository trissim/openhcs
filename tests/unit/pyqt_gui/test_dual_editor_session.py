"""Regression tests for dual-editor declaration synchronization."""

from __future__ import annotations

from types import SimpleNamespace

from objectstate import ObjectState, ObjectStateRegistry

from openhcs.core.steps.function_step import FunctionStep
from openhcs.pyqt_gui.windows.dual_editor_session import (
    DualEditorFunctionPatternController,
    DualEditorSession,
)


def _original(image):
    return image


def _replacement(image):
    return image


def test_saved_function_pattern_updates_step_object_state() -> None:
    """Saving edited code must update the step declaration used by the pipeline."""

    ObjectStateRegistry.clear()
    editing_step = FunctionStep(func=_original, name="edited step")
    state = ObjectState(editing_step, scope_id="plate::functionstep_0")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    session = DualEditorSession(
        editing_step=editing_step,
        step_editor=SimpleNamespace(state=state),
        func_editor=SimpleNamespace(current_pattern=_replacement),
    )
    changes: list[str] = []
    controller = DualEditorFunctionPatternController(
        session=session,
        detect_changes=lambda: changes.append("changed"),
        invalidate_artifact_plan=lambda: changes.append("invalidated"),
    )

    try:
        controller.handle_change()

        assert session.object_session().to_object(update_delegate=False).func is _replacement
        assert changes == ["changed", "invalidated"]
    finally:
        ObjectStateRegistry.clear()

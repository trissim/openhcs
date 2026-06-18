"""Nominal session helpers for DualEditorWindow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, cast

from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.steps.function_step import FunctionSpec, FunctionStep
from openhcs.config_framework.object_state import ObjectStateRegistry
from objectstate import ObjectState, ObjectStateEditSession

if TYPE_CHECKING:
    from openhcs.pyqt_gui.widgets.step_parameter_editor import StepParameterEditorWidget
    from pyqt_reactive.widgets.function_list_editor import FunctionListEditorWidget

logger = logging.getLogger(__name__)


def _require_function_step(value, *, label: str) -> FunctionStep:
    if not isinstance(value, FunctionStep):
        raise TypeError(f"{label} must be a FunctionStep, got {type(value).__name__}")
    return value


@dataclass(slots=True)
class DualEditorSession:
    """Own step/function-editor synchronization for a dual editor instance."""

    editing_step: FunctionStep
    step_editor: "StepParameterEditorWidget | None" = None
    func_editor: "FunctionListEditorWidget | None" = None

    def object_session(self) -> ObjectStateEditSession[FunctionStep]:
        """Return the generic ObjectState edit boundary for this step window."""

        return ObjectStateEditSession(
            state_provider=lambda: self.step_state,
            fallback_object=self.editing_step,
            expected_type=FunctionStep,
        )

    def current_function_spec(self) -> FunctionSpec:
        """Return the live function spec from ObjectState, falling back to the clone."""
        object_session = self.object_session()
        if object_session.has_parameter("func"):
            return cast(FunctionSpec, object_session.parameter_value("func"))
        return self.editing_step.func

    def current_source_bindings(self):
        """Return live source bindings from ObjectState, falling back to the clone."""
        object_session = self.object_session()
        if object_session.has_parameter("source_bindings"):
            return object_session.parameter_value("source_bindings")
        return self.editing_step.source_bindings

    def current_function_pattern(self) -> FunctionSpec:
        """Return the function editor pattern through the nominal transport authority."""
        if self.func_editor is None:
            return self.normalize_function_spec(self.editing_step.func)
        return self.normalize_function_spec(self.func_editor.current_pattern)

    def normalize_function_spec(self, func_spec: FunctionSpec) -> FunctionSpec:
        """Normalize a function spec through the transport authority."""
        return cast(
            FunctionSpec,
            FunctionStepTransportAuthority.normalize_function_spec(func_spec),
        )

    @property
    def step_state(self) -> ObjectState | None:
        if self.step_editor is None:
            return None
        return self.step_editor.state

    def apply_function_spec_to_state(self, func_spec: FunctionSpec) -> FunctionSpec:
        """Normalize and apply the function spec through the ObjectState boundary."""

        normalized = self.normalize_function_spec(func_spec)
        object_session = self.object_session()
        if object_session.has_parameter("func"):
            object_session.update_parameter("func", normalized)
        else:
            self.editing_step.func = normalized
        return normalized

    def sync_function_editor_from_step(self) -> bool:
        """Refresh the function editor from its authoritative ObjectState context."""
        if self.func_editor is None:
            logger.debug("Function editor does not exist yet; skipping sync")
            return False
        self.func_editor.refresh_from_context()
        return True

    def apply_current_function_pattern(self) -> tuple[bool, FunctionSpec]:
        """Apply the current function editor pattern to step and ObjectState."""
        current_pattern = self.current_function_pattern()
        logger.debug(
            "[FUNC_PATTERN] current_pattern type=%s value=%r",
            type(current_pattern).__name__,
            current_pattern,
        )

        object_session = self.object_session()
        state_func = (
            object_session.parameter_value("func")
            if object_session.has_parameter("func")
            else self.editing_step.func
        )
        if state_func == current_pattern:
            logger.debug("[FUNC_PATTERN] Ignoring no-op function pattern update")
            return False, current_pattern

        with ObjectStateRegistry.atomic("edit func"):
            self.apply_function_spec_to_state(current_pattern)
            state = self.step_state
            if state is not None:
                logger.debug(
                    "[FUNC_PATTERN] ObjectState dirty_fields after update: %s",
                    state.dirty_fields,
                )
        return True, current_pattern

    @staticmethod
    def callable_from_function_spec(func_spec: FunctionSpec) -> Callable | None:
        """Return the first callable represented by a FunctionStep function spec."""
        if callable(func_spec):
            return func_spec
        if isinstance(func_spec, tuple) and func_spec and callable(func_spec[0]):
            return func_spec[0]
        if isinstance(func_spec, list) and func_spec:
            return DualEditorSession.callable_from_function_spec(func_spec[0])
        return None


@dataclass
class DualEditorFunctionPatternController:
    """Qt-slot target for function pattern changes."""

    session: DualEditorSession
    detect_changes: Callable[[], None]
    refresh_artifact_contract_preview: Callable[[FunctionSpec], None]

    def handle_change(self) -> None:
        changed, current_pattern = self.session.apply_current_function_pattern()
        if not changed:
            return
        self.detect_changes()
        self.refresh_artifact_contract_preview(current_pattern)
        logger.debug("Function pattern changed: %r", current_pattern)

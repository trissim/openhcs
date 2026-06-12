"""Nominal session helpers for DualEditorWindow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.steps.function_step import FunctionStep
from openhcs.config_framework.object_state import ObjectStateRegistry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DualEditorSession:
    """Own step/function-editor synchronization for a dual editor instance."""

    editing_step: FunctionStep
    step_editor: Any | None = None
    func_editor: Any | None = None

    def current_function_spec(self) -> Any:
        """Return the live function spec from ObjectState, falling back to the clone."""
        state = self.step_state
        if state is not None and "func" in state.parameters:
            return state.parameters["func"]
        return self.editing_step.func

    def current_source_bindings(self) -> Any:
        """Return live source bindings from ObjectState, falling back to the clone."""
        state = self.step_state
        if state is not None and "source_bindings" in state.parameters:
            return state.parameters["source_bindings"]
        return self.editing_step.source_bindings

    def current_function_pattern(self) -> Any:
        """Return the function editor pattern through the nominal transport authority."""
        if self.func_editor is None:
            return self.normalize_function_spec(self.editing_step.func)
        return self.normalize_function_spec(self.func_editor.current_pattern)

    def normalize_function_spec(self, func_spec: Any) -> Any:
        """Normalize a function spec through the transport authority."""
        return FunctionStepTransportAuthority.normalize_function_spec(func_spec)

    @property
    def step_state(self) -> Any | None:
        if self.step_editor is None:
            return None
        return self.step_editor.state

    def step_state_values(self) -> dict[str, Any]:
        """Return current values from the known step editor state."""
        state = self.step_state
        if state is None:
            return {}
        return state.get_current_values()

    def apply_step_state_to_step(self) -> None:
        """Apply the step editor's nominal state to the editable step object."""
        for param_name, value in self.step_state_values().items():
            object.__setattr__(self.editing_step, param_name, value)
            logger.debug("Applied %s=%r to editing step", param_name, value)

    def sync_function_editor_from_step(self) -> bool:
        """Refresh the function editor from its authoritative ObjectState context."""
        if self.func_editor is None:
            logger.debug("Function editor does not exist yet; skipping sync")
            return False
        self.func_editor.refresh_from_context()
        return True

    def apply_current_function_pattern(self) -> tuple[bool, Any]:
        """Apply the current function editor pattern to step and ObjectState."""
        current_pattern = self.current_function_pattern()
        logger.debug(
            "[FUNC_PATTERN] current_pattern type=%s value=%r",
            type(current_pattern).__name__,
            current_pattern,
        )

        state = self.step_state
        state_func = (
            state.parameters.get("func")
            if state is not None and "func" in state.parameters
            else None
        )
        step_func = self.editing_step.func
        if state_func == current_pattern and step_func == current_pattern:
            logger.debug("[FUNC_PATTERN] Ignoring no-op function pattern update")
            return False, current_pattern

        with ObjectStateRegistry.atomic("edit func"):
            if step_func != current_pattern:
                self.editing_step.func = current_pattern
            if (
                state is not None
                and "func" in state.parameters
                and state_func != current_pattern
            ):
                state.update_parameter("func", current_pattern)
                logger.debug(
                    "Updated ObjectState 'func' parameter for real-time preview"
                )
                logger.debug(
                    "[FUNC_PATTERN] ObjectState dirty_fields after update: %s",
                    state.dirty_fields,
                )
        return True, current_pattern

    @staticmethod
    def callable_from_function_spec(func_spec: Any) -> Callable | None:
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
    refresh_artifact_contract_preview: Callable[[Any], None]

    def handle_change(self) -> None:
        changed, current_pattern = self.session.apply_current_function_pattern()
        if not changed:
            return
        self.detect_changes()
        self.refresh_artifact_contract_preview(current_pattern)
        logger.debug("Function pattern changed: %r", current_pattern)

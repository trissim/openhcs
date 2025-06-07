"""Step parameter editor widget - port of step_parameter_editor.py to Textual."""

import inspect
from typing import Any, Dict
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static, Input, Select
from textual.app import ComposeResult
from textual.reactive import reactive

from openhcs.core.steps.function_step import FunctionStep
from openhcs.constants.constants import VariableComponents, GroupBy


class StepParameterEditorWidget(Container):
    """
    Step parameter editor using Textual forms.

    Ports form generation logic from existing step_parameter_editor.py
    """

    # Reactive properties for automatic updates
    step_name = reactive("")
    group_by = reactive("")
    variable_components = reactive(list)

    def __init__(self, step: FunctionStep):
        super().__init__()
        self.step = step

        # Initialize reactive properties from step
        self.step_name = step.name or ""
        self.group_by = step.group_by or ""
        self.variable_components = step.variable_components or []

    def compose(self) -> ComposeResult:
        """Compose the step parameter form."""
        yield Static("[bold]Step Parameters[/bold]")

        with Vertical():
            # Step name input
            yield Static("Step Name:")
            yield Input(value=self.step_name, id="step_name_input")

            # Group by selection
            yield Static("Group By:")
            group_by_options = [(item.value, item.value) for item in GroupBy]
            yield Select(options=group_by_options, value=self.group_by, id="group_by_select")

            # Variable components selection
            yield Static("Variable Components:")
            for component in VariableComponents:
                is_selected = component in self.variable_components
                yield Input(
                    value="✓" if is_selected else "○",
                    id=f"var_comp_{component.value}",
                    disabled=True
                )

            # Action buttons
            with Horizontal():
                yield Button("Load Step", id="load_step_btn", compact=True)
                yield Button("Save As", id="save_as_btn", compact=True)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        if event.input.id == "step_name_input":
            self.step_name = event.value
            self.step.name = event.value

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "group_by_select":
            self.group_by = event.value
            self.step.group_by = event.value

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "load_step_btn":
            self._load_step()
        elif event.button.id == "save_as_btn":
            self._save_step_as()

    def _load_step(self) -> None:
        """Load step from file."""
        # TODO: Implement file dialog in later sprint
        pass

    def _save_step_as(self) -> None:
        """Save step to file."""
        # TODO: Implement file dialog in later sprint
        pass

"""Step parameter editor widget - port of step_parameter_editor.py to Textual."""

import inspect
import logging # Added logging
from typing import Any, Dict, List, cast
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static, Input, RadioSet, RadioButton
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.message import Message

from openhcs.core.steps.function_step import FunctionStep
from openhcs.constants.constants import VariableComponents, GroupBy

logger = logging.getLogger(__name__) # Added logger

class StepParameterEditorWidget(Container):
    """
    Step parameter editor using Textual forms.
    Ports form generation logic from existing step_parameter_editor.py
    """

    class StepParameterChanged(Message):
        """Message to indicate a parameter has changed."""
        pass

    # Reactive properties for automatic updates
    step_name = reactive("")
    group_by = reactive("")
    selected_variable_component = reactive("")  # Single choice instead of list

    def __init__(self, step: FunctionStep):
        super().__init__()
        self.step = step

        self.step_name = step.name or ""

        # Group by defaults to CHANNEL, with NONE as an option
        if step.group_by and step.group_by in [m.value for m in GroupBy]:
            self.group_by = step.group_by
        else:
            self.group_by = GroupBy.CHANNEL.value  # Default to "channel"
        logger.debug(f"Initialized StepParameterEditorWidget: group_by='{self.group_by}'")

        # Variable components: single choice, defaults to "site"
        if step.variable_components and len(step.variable_components) > 0:
            # Take first component if multiple exist
            first_comp = step.variable_components[0]
            self.selected_variable_component = str(first_comp.value) if hasattr(first_comp, 'value') else str(first_comp)
        else:
            self.selected_variable_component = VariableComponents.SITE.value  # Default to "site"
        logger.debug(f"Initialized StepParameterEditorWidget: selected_variable_component='{self.selected_variable_component}'")


    def compose(self) -> ComposeResult:
        """Compose the step parameter form."""
        yield Static("[bold]Step Parameters[/bold]")

        with Vertical(id="step_param_editor_vertical"):
            yield Static("Step Name:")
            yield Input(value=self.step_name, id="step_name_input")

            yield Static("Group By:")
            with RadioSet(id="group_by_radio"):
                for member in GroupBy:
                    display_name = member.value.capitalize() if member.value else "None"
                    is_selected = member.value == self.group_by
                    yield RadioButton(
                        display_name,
                        value=is_selected,
                        id=f"group_by_radio_{member.value or 'none'}"
                    )

            # Variable components selection (single choice radio buttons)
            yield Static("Variable Component:")
            with RadioSet(id="variable_components_radio"):
                for component_enum_member in VariableComponents:
                    component_value = component_enum_member.value
                    is_selected = component_value == self.selected_variable_component
                    yield RadioButton(
                        component_value.capitalize(),
                        value=is_selected,
                        id=f"var_comp_radio_{component_value}"
                    )

            with Horizontal():
                yield Button("Load Step (N/A)", id="load_step_btn", compact=True, disabled=True)
                yield Button("Save As (N/A)", id="save_as_btn", compact=True, disabled=True)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "step_name_input":
            self.step_name = event.value
            self.step.name = event.value
            self.post_message(self.StepParameterChanged())

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "variable_components_radio":
            # Get the selected radio button value
            if event.pressed and event.pressed.id:
                component_value = event.pressed.id.replace("var_comp_radio_", "")
                self.selected_variable_component = component_value
                # Update step with single component (convert to list for compatibility)
                self.step.variable_components = [component_value]
                logger.debug(f"Variable component changed to: '{component_value}'")
                self.post_message(self.StepParameterChanged())

        elif event.radio_set.id == "group_by_radio":
            # Get the selected radio button value
            if event.pressed and event.pressed.id:
                radio_id = event.pressed.id.replace("group_by_radio_", "")
                # Handle the special case for None (empty string)
                group_by_value = "" if radio_id == "none" else radio_id
                self.group_by = group_by_value
                self.step.group_by = group_by_value
                logger.debug(f"Group by changed to: '{group_by_value}'")
                self.post_message(self.StepParameterChanged())



    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "load_step_btn":
            self._load_step()
        elif event.button.id == "save_as_btn":
            self._save_step_as()

    def _load_step(self) -> None:
        pass

    def _save_step_as(self) -> None:
        pass

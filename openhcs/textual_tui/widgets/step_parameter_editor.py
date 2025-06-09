"""Step parameter editor widget - port of step_parameter_editor.py to Textual."""

import logging
from typing import Any
from pathlib import Path
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Button, Static
from textual.app import ComposeResult
from textual.message import Message

from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.steps.abstract import AbstractStep
from .shared.parameter_form_manager import ParameterFormManager
from .shared.signature_analyzer import SignatureAnalyzer

logger = logging.getLogger(__name__)


class StepParameterEditorWidget(ScrollableContainer):
    """
    Step parameter editor using dynamic form generation.
    Builds forms based on FunctionStep constructor signature.
    """

    class StepParameterChanged(Message):
        """Message to indicate a parameter has changed."""
        pass

    def __init__(self, step: FunctionStep):
        super().__init__()
        self.step = step

        # Create parameter form manager using shared components
        param_info = SignatureAnalyzer.analyze(FunctionStep.__init__)

        # Get current parameter values from step instance
        parameters = {}
        parameter_types = {}
        param_defaults = {}

        for name, info in param_info.items():
            if name in ('func',):  # Skip func parameter
                continue
            current_value = getattr(self.step, name, info.default_value)
            parameters[name] = current_value
            parameter_types[name] = info.param_type
            param_defaults[name] = info.default_value

        self.form_manager = ParameterFormManager(parameters, parameter_types, "step")
        self.param_defaults = param_defaults

    def compose(self) -> ComposeResult:
        """Compose the step parameter form dynamically."""
        yield Static("[bold]Step Parameters[/bold]")

        # Build form using shared form manager
        yield from self._build_step_parameter_form()

        # Action buttons
        with Horizontal():
            yield Button("Load Step (N/A)", id="load_step_btn", compact=True, disabled=True)
            yield Button("Save As (N/A)", id="save_as_btn", compact=True, disabled=True)

    def _build_step_parameter_form(self) -> ComposeResult:
        """Generate form widgets using shared ParameterFormManager."""
        try:
            # Use shared form manager to build form
            yield from self.form_manager.build_form()
        except Exception as e:
            yield Static(f"[red]Error building step parameter form: {e}[/red]")

    def on_input_changed(self, event) -> None:
        """Handle input changes from shared components."""
        if event.input.id.startswith("step_"):
            param_name = event.input.id.split("_", 1)[1]
            if self.form_manager:
                self.form_manager.update_parameter(param_name, event.value)
                final_value = self.form_manager.parameters[param_name]
                self._handle_parameter_change(param_name, final_value)

    def on_checkbox_changed(self, event) -> None:
        """Handle checkbox changes from shared components."""
        if event.checkbox.id.startswith("step_"):
            param_name = event.checkbox.id.split("_", 1)[1]
            if self.form_manager:
                self.form_manager.update_parameter(param_name, event.value)
                final_value = self.form_manager.parameters[param_name]
                self._handle_parameter_change(param_name, final_value)

    def on_radio_set_changed(self, event) -> None:
        """Handle RadioSet changes from shared components."""
        if event.radio_set.id.startswith("step_"):
            param_name = event.radio_set.id.split("_", 1)[1]
            if event.pressed and event.pressed.id:
                enum_value = event.pressed.id[5:]  # Remove "enum_" prefix
                if self.form_manager:
                    self.form_manager.update_parameter(param_name, enum_value)
                    final_value = self.form_manager.parameters[param_name]
                    self._handle_parameter_change(param_name, final_value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "load_step_btn":
            self._load_step()
        elif event.button.id == "save_as_btn":
            self._save_step_as()
        elif event.button.id.startswith("reset_step_"):
            # Individual parameter reset
            param_name = event.button.id.split("_", 2)[2]
            self._reset_parameter(param_name)

    def _handle_parameter_change(self, param_name: str, value: Any) -> None:
        """Update step parameter and emit change message."""
        try:
            # Convert value to appropriate type
            if param_name == 'force_disk_output':
                value = bool(value)
            elif param_name in ('input_dir', 'output_dir') and value:
                value = Path(value)
            
            # Update step attribute
            setattr(self.step, param_name, value)
            logger.debug(f"Updated step parameter {param_name}={value}")
            self.post_message(self.StepParameterChanged())
        except Exception as e:
            logger.error(f"Error updating step parameter {param_name}: {e}")

    def _reset_parameter(self, param_name: str) -> None:
        """Reset a specific parameter to its default value."""
        if not self.form_manager or param_name not in self.param_defaults:
            return

        try:
            # Use form manager to reset parameter
            default_value = self.param_defaults[param_name]
            self.form_manager.reset_parameter(param_name, default_value)

            # Update step instance and notify parent
            self._handle_parameter_change(param_name, default_value)
            logger.debug(f"Reset step parameter {param_name} to default: {default_value}")
        except Exception as e:
            logger.error(f"Error resetting step parameter {param_name}: {e}")

    def _load_step(self) -> None:
        """Load step configuration from file."""
        pass  # TODO: Implement

    def _save_step_as(self) -> None:
        """Save step configuration to file."""
        pass  # TODO: Implement
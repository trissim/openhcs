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

        self.form_manager = ParameterFormManager(parameters, parameter_types, "step", param_info)
        self.param_defaults = param_defaults

    def compose(self) -> ComposeResult:
        """Compose the step parameter form dynamically."""
        yield Static("[bold]Step Parameters[/bold]")

        # Build form using shared form manager
        yield from self._build_step_parameter_form()

        # Action buttons
        with Horizontal():
            yield Button("Load .step", id="load_step_btn", compact=True)
            yield Button("Save .step As", id="save_as_btn", compact=True)

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
            self.run_worker(self._load_step())
        elif event.button.id == "save_as_btn":
            self.run_worker(self._save_step_as())
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

            # Refresh the UI to show the reset value
            self._refresh_form_widgets()

            logger.debug(f"Reset step parameter {param_name} to default: {default_value}")
        except Exception as e:
            logger.error(f"Error resetting step parameter {param_name}: {e}")

    def _refresh_form_widgets(self) -> None:
        """Refresh form widgets to show current parameter values."""
        try:
            current_values = self.form_manager.get_current_values()

            # Update each widget with current value
            for param_name, value in current_values.items():
                widget_id = f"step_{param_name}"
                try:
                    widget = self.query_one(f"#{widget_id}")

                    # Update widget based on type
                    if hasattr(widget, 'value'):
                        # Convert enum to string for display
                        display_value = value.value if hasattr(value, 'value') else value
                        widget.value = display_value
                    elif hasattr(widget, 'pressed'):
                        # RadioSet - find and press the correct radio button
                        if hasattr(value, 'value'):
                            target_id = f"enum_{value.value}"
                            for radio in widget.query("RadioButton"):
                                if radio.id == target_id:
                                    radio.value = True
                                    break
                except Exception as widget_error:
                    logger.debug(f"Could not update widget {widget_id}: {widget_error}")

        except Exception as e:
            logger.warning(f"Failed to refresh form widgets: {e}")

    async def _load_step(self) -> None:
        """Load step configuration from file."""
        from openhcs.textual_tui.windows import open_file_browser_window, BrowserMode
        from openhcs.textual_tui.services.file_browser_service import SelectionMode
        from openhcs.constants.constants import Backend

        def handle_result(result):
            if result and isinstance(result, Path):
                self._load_step_from_file(result)

        # Launch file browser window for .step files
        from openhcs.core.path_cache import get_cached_browser_path, PathCacheKey

        await open_file_browser_window(
            app=self.app,
            file_manager=self.app.filemanager,
            initial_path=get_cached_browser_path(PathCacheKey.STEP_SETTINGS),
            backend=Backend.DISK,
            title="Load Step Settings (.step)",
            mode=BrowserMode.LOAD,
            selection_mode=SelectionMode.FILES_ONLY,
            filter_extensions=['.step'],
            cache_key=PathCacheKey.STEP_SETTINGS,
            on_result_callback=handle_result,
            caller_id="step_parameter_editor"
        )

    async def _save_step_as(self) -> None:
        """Save step configuration to file."""
        from openhcs.textual_tui.windows import open_file_browser_window, BrowserMode
        from openhcs.textual_tui.services.file_browser_service import SelectionMode
        from openhcs.constants.constants import Backend

        def handle_result(result):
            if result and isinstance(result, Path):
                self._save_step_to_file(result)

        # Launch file browser window for saving .step files
        from openhcs.textual_tui.utils.path_cache import get_cached_browser_path, PathCacheKey

        await open_file_browser_window(
            app=self.app,
            file_manager=self.app.filemanager,
            initial_path=get_cached_browser_path(PathCacheKey.STEP_SETTINGS),
            backend=Backend.DISK,
            title="Save Step Settings (.step)",
            mode=BrowserMode.SAVE,
            selection_mode=SelectionMode.FILES_ONLY,
            filter_extensions=['.step'],
            default_filename="step_settings.step",
            cache_key=PathCacheKey.STEP_SETTINGS,
            on_result_callback=handle_result,
            caller_id="step_parameter_editor"
        )

    def _load_step_from_file(self, file_path: Path) -> None:
        """Load step parameters from .step file."""
        import dill as pickle
        try:
            with open(file_path, 'rb') as f:
                step_data = pickle.load(f)

            # Update both the step object and form manager
            for param_name, value in step_data.items():
                if param_name in self.form_manager.parameters:
                    # Update form manager first
                    self.form_manager.update_parameter(param_name, value)
                    # Then update step object and emit change message
                    self._handle_parameter_change(param_name, value)

            # Refresh the UI to show loaded values
            self._refresh_form_widgets()

            logger.debug(f"Loaded {len(step_data)} parameters from {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load step: {e}")

    def _save_step_to_file(self, file_path: Path) -> None:
        """Save step parameters to .step file."""
        import pickle
        try:
            step_data = self.form_manager.get_current_values()
            with open(file_path, 'wb') as f:
                pickle.dump(step_data, f)
        except Exception as e:
            logger.error(f"Failed to save step: {e}")
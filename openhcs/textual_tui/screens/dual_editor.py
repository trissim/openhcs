"""DualEditor modal screen - enhanced port of dual_editor_pane.py to Textual."""

import logging
from typing import Optional, Callable, Union, List, Dict
from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static, TabbedContent, TabPane, Label, Input, Select
from textual.reactive import reactive

from openhcs.core.steps.function_step import FunctionStep
from openhcs.textual_tui.services.pattern_data_manager import PatternDataManager
from openhcs.textual_tui.services.function_registry_service import FunctionRegistryService
# Updated import to get the message class as well
from openhcs.textual_tui.widgets.step_parameter_editor import (
    StepParameterEditorWidget,
    StepParameterEditorWidget)
from openhcs.textual_tui.widgets.function_list_editor import FunctionListEditorWidget

logger = logging.getLogger(__name__)


class DualEditorScreen(ModalScreen):
    """
    Enhanced modal screen for editing steps and functions.

    Ports the complete functionality from dual_editor_pane.py with:
    - Change tracking and validation
    - Tab switching with proper state management
    - Integration with all visual programming services
    - Proper error handling and user feedback
    """

    # Reactive state for change tracking
    has_changes = reactive(False)
    current_tab = reactive("step")

    def __init__(self, step_data: Optional[FunctionStep] = None, is_new: bool = False):
        super().__init__()
        self.step_data = step_data
        self.is_new = is_new

        # Initialize services (reuse existing business logic)
        self.pattern_manager = PatternDataManager()
        self.registry_service = FunctionRegistryService()

        # Create working copy of step data
        if self.step_data:
            self.editing_step = self.pattern_manager.clone_pattern(self.step_data)
            logger.info(f"Editing existing step: {self.editing_step.name}")
        else:
            # Create new step with empty function list (user adds functions manually)
            self.editing_step = FunctionStep(
                func=[],  # Start with empty function list
                name="New Step",
                variable_components=[],
                group_by=""
            )
            logger.info("Creating new step with empty function list")

        # Store original for change detection
        self.original_step = self.pattern_manager.clone_pattern(self.editing_step)

        # Editor widgets (will be created in compose)
        self.step_editor = None
        self.func_editor = None

    def compose(self) -> ComposeResult:
        """Compose the enhanced dual editor modal."""
        with Container(id="dual_editor_dialog"):
            # Dialog title with change indicator
            title_text = "Visual Programming - Edit Step" if not self.is_new else "Visual Programming - New Step"
            yield Static(title_text, id="dialog_title")

            # Status bar for feedback
            yield Static("Ready", id="status_bar")

            with TabbedContent(id="editor_tabs"):
                with TabPane("Step Settings", id="step_tab"):
                    # Create step editor with correct constructor
                    self.step_editor = StepParameterEditorWidget(self.editing_step)
                    yield self.step_editor

                with TabPane("Function Pattern", id="func_tab"):
                    # Create function editor with validated function data
                    func_data = self._validate_function_data(self.editing_step.func)
                    self.func_editor = FunctionListEditorWidget(func_data)
                    yield self.func_editor

            # Action buttons with dynamic state
            with Horizontal(id="dialog_buttons"):
                yield Button("Save", id="save_btn", variant="primary", compact=True, disabled=True)
                yield Button("Cancel", id="cancel_btn", compact=True)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes from child widgets."""
        logger.debug(f"Input changed: {event}")

        # Update change tracking when any input changes
        self._update_change_tracking()
        self._update_status("Modified step parameters")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes from child widgets."""
        logger.debug(f"Select changed: {event}")

        # Update change tracking when any select changes
        self._update_change_tracking()
        self._update_status("Modified step parameters")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses from child widgets and dialog buttons."""
        if event.button.id == "save_btn":
            self._handle_save()
        elif event.button.id == "cancel_btn":
            self._handle_cancel()
        else:
            # Handle other button presses from child widgets
            logger.debug(f"Child widget button pressed: {event.button.id}")
            self._update_change_tracking()
            self._update_status("Modified step configuration")

    def on_step_parameter_editor_widget_step_parameter_changed(
        self, event: StepParameterEditorWidget.StepParameterChanged # Listen for the specific message
    ) -> None:
        """Handle parameter changes from the step editor widget."""
        logger.debug("Received StepParameterChanged from child StepParameterEditorWidget")
        self._update_change_tracking()
        self._update_status("Modified step parameters (via message)")

    def on_function_list_editor_widget_function_pattern_changed(
        self, event: FunctionListEditorWidget.FunctionPatternChanged
    ) -> None:
        """Handle function pattern changes from the function editor widget."""
        logger.debug("Received FunctionPatternChanged from child FunctionListEditorWidget")
        # Direct assignment - backend handles List[(callable, kwargs)] format
        self.editing_step.func = self.func_editor.functions
        self._update_change_tracking()
        self._update_status("Modified function pattern")



    def _update_change_tracking(self) -> None:
        """Update change tracking state."""
        # Compare current editing step with original
        has_changes = not self._steps_equal(self.editing_step, self.original_step)
        self.has_changes = has_changes

        # Update save button state
        try:
            save_btn = self.query_one("#save_btn", Button)
            save_btn.disabled = not has_changes
        except Exception:
            pass

    def _validate_function_data(self, func_data) -> Union[List, callable, None]:
        """Validate and normalize function data for FunctionListEditorWidget."""
        if func_data is None:
            return None
        elif callable(func_data):
            return func_data
        elif isinstance(func_data, (list, dict)):
            return func_data
        else:
            logger.warning(f"Invalid function data type: {type(func_data)}, using None")
            return None

    def _steps_equal(self, step1: FunctionStep, step2: FunctionStep) -> bool:
        """Compare two FunctionSteps for equality."""
        return (
            step1.name == step2.name and
            step1.func == step2.func and
            step1.variable_components == step2.variable_components and
            step1.group_by == step2.group_by
        )

    def _update_status(self, message: str) -> None:
        """Update status bar message."""
        try:
            status_bar = self.query_one("#status_bar", Static)
            status_bar.update(message)
        except Exception:
            pass

    def watch_has_changes(self, has_changes: bool) -> None:
        """React to changes in has_changes state."""
        # Update dialog title to show unsaved changes
        try:
            title = self.query_one("#dialog_title", Static)
            base_title = "Visual Programming - Edit Step" if not self.is_new else "Visual Programming - New Step"
            if has_changes:
                title.update(f"{base_title} *")
            else:
                title.update(base_title)
        except Exception:
            pass



    def _handle_save(self) -> None:
        """Handle save button with validation."""
        # Validate step data
        if not self.editing_step.name or not self.editing_step.name.strip():
            self._update_status("Error: Step name cannot be empty")
            return

        if not self.editing_step.func:
            self._update_status("Error: Function pattern cannot be empty")
            return

        # Save successful
        logger.info(f"Saving step: {self.editing_step.name}")
        self._update_status("Saved successfully")
        self.dismiss(self.editing_step)

    def _handle_cancel(self) -> None:
        """Handle cancel button with change confirmation."""
        if self.has_changes:
            # TODO: Add confirmation dialog for unsaved changes
            logger.info("Cancelling with unsaved changes")
            self._update_status("Cancelled - changes discarded")
        else:
            logger.info("Cancelling without changes")
            self._update_status("Cancelled")

        self.dismiss(None)

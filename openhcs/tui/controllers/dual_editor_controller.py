"""
Dual Editor Controller for Hybrid TUI.

Adapted from TUI2's DualEditorController with:
- Schema-free operation using FunctionStep objects directly
- Integration with hybrid components
- Simplified state management without TUIState dependencies

Manages StepSettingsEditor and FunctionPatternEditor for editing FunctionStep objects.
"""

import asyncio
import copy
import logging
from typing import Any, Callable, Dict, Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, VSplit, Container, Dimension
from prompt_toolkit.widgets import Button, Frame, Label, Box
from ..utils.safe_formatting import SafeLabel

from ..interfaces.component_interfaces import ControllerInterface
from ..components.step_settings_editor import StepSettingsEditor
from ..components.function_pattern_editor import FunctionPatternEditor

logger = logging.getLogger(__name__)

class DualEditorController(ControllerInterface):
    """
    Manages StepSettingsEditor and FunctionPatternEditor for editing FunctionStep objects.

    Provides a dual-pane editor with step settings on one side and function pattern
    editing on the other, with save/cancel functionality.
    """

    def __init__(
        self,
        func_step: Any,  # FunctionStep object
        on_save: Optional[Callable[[Any], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None
    ):
        """
        Initialize the dual editor controller.

        Args:
            func_step: FunctionStep object to edit
            on_save: Callback when changes are saved
            on_cancel: Callback when editing is cancelled
        """
        self.on_save = on_save
        self.on_cancel = on_cancel

        # Step data state
        self.original_func_step = func_step
        self.editing_func_step = copy.deepcopy(func_step)

        # Track changes
        self.has_step_changes = False
        self.has_func_changes = False

        # UI components
        self.step_settings_editor = None
        self.func_pattern_editor = None
        self.save_button = None
        self.cancel_button = None
        self._container = None

        # Initialize components
        self._initialize_components()

    async def initialize_controller(self) -> None:
        """Initialize controller and its managed components."""
        try:
            # Initialize step settings editor with current step data
            step_data = self._extract_step_data(self.editing_func_step)
            await self.step_settings_editor.update_data(step_data)

            # Initialize function pattern editor with current pattern
            func_pattern = getattr(self.editing_func_step, 'func', None)
            await self.func_pattern_editor.update_data(func_pattern)

            logger.info("DualEditorController initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DualEditorController: {e}")
            raise

    async def cleanup_controller(self) -> None:
        """Clean up controller resources."""
        try:
            # Clean up components if they have cleanup methods
            if hasattr(self.step_settings_editor, 'cleanup_async'):
                await self.step_settings_editor.cleanup_async()
            if hasattr(self.func_pattern_editor, 'cleanup_async'):
                await self.func_pattern_editor.cleanup_async()

            logger.info("DualEditorController cleaned up successfully")

        except Exception as e:
            logger.error(f"Failed to cleanup DualEditorController: {e}")

    def get_container(self) -> Container:
        """Get the main container for this controller."""
        return self._container

    def _initialize_components(self):
        """Initialize UI components."""
        # Create step settings editor
        step_data = self._extract_step_data(self.editing_func_step)
        self.step_settings_editor = StepSettingsEditor(
            initial_step_data=step_data,
            change_callback=self._on_step_parameter_changed
        )

        # Create function pattern editor
        func_pattern = getattr(self.editing_func_step, 'func', None)
        self.func_pattern_editor = FunctionPatternEditor(
            initial_pattern=func_pattern,
            change_callback=self._on_func_pattern_changed
        )

        # Create action buttons
        self.save_button = Button(
            "Save Changes",
            handler=lambda: get_app().create_background_task(self._handle_save())
        )
        self.cancel_button = Button(
            "Cancel",
            handler=lambda: get_app().create_background_task(self._handle_cancel())
        )

        # Build layout
        self._build_layout()

    def _build_layout(self):
        """Build the main layout container."""
        # Action buttons toolbar
        buttons_toolbar = VSplit([
            self.save_button,
            Box(width=2),  # Spacer
            self.cancel_button,
            Box(width=Dimension(weight=1)),  # Flexible spacer
            SafeLabel(lambda: self._get_status_text(), style="class:editor-title")
        ], height=1, padding=1)

        # Main editor area - side by side
        main_editor_area = VSplit([
            Frame(
                self.step_settings_editor.container,
                title="Step Settings"
            ),
            Frame(
                self.func_pattern_editor.container,
                title="Function Pattern"
            )
        ], padding=1)

        # Complete layout
        self._container = HSplit([
            main_editor_area,
            buttons_toolbar
        ])

    def _extract_step_data(self, func_step: Any) -> Dict[str, Any]:
        """Extract step data from FunctionStep object."""
        try:
            step_data = {}

            # Extract AbstractStep attributes
            step_attributes = [
                'name', 'variable_components', 'force_disk_output',
                'group_by', 'input_dir', 'output_dir'
            ]

            for attr in step_attributes:
                if hasattr(func_step, attr):
                    step_data[attr] = getattr(func_step, attr)

            return step_data

        except Exception as e:
            logger.error(f"Failed to extract step data: {e}")
            return {}

    def _apply_step_data(self, step_data: Dict[str, Any]):
        """Apply step data changes to the editing FunctionStep."""
        try:
            for attr, value in step_data.items():
                if hasattr(self.editing_func_step, attr):
                    setattr(self.editing_func_step, attr, value)

        except Exception as e:
            logger.error(f"Failed to apply step data: {e}")

    def _on_step_parameter_changed(self, param_name: str, new_value: Any):
        """Handle step parameter changes."""
        try:
            # Update the editing step
            if hasattr(self.editing_func_step, param_name):
                setattr(self.editing_func_step, param_name, new_value)

            # Mark as changed
            self.has_step_changes = True
            self._update_save_button_state()

        except Exception as e:
            logger.error(f"Failed to handle step parameter change: {e}")

    def _on_func_pattern_changed(self, new_pattern: Any):
        """Handle function pattern changes."""
        try:
            # Update the editing step's function pattern
            self.editing_func_step.func = new_pattern

            # Mark as changed
            self.has_func_changes = True
            self._update_save_button_state()

        except Exception as e:
            logger.error(f"Failed to handle function pattern change: {e}")

    def _update_save_button_state(self):
        """Update save button enabled state based on changes."""
        has_changes = self.has_step_changes or self.has_func_changes
        # Note: prompt_toolkit buttons don't have an enabled property
        # We could change the style or text to indicate state
        if has_changes:
            self.save_button.text = "Save Changes *"
        else:
            self.save_button.text = "Save Changes"

    def _get_status_text(self) -> str:
        """Get status text for display."""
        step_name = getattr(self.editing_func_step, 'name', 'Unnamed Step')
        changes_indicator = " *" if (self.has_step_changes or self.has_func_changes) else ""
        return f"Editing: {step_name}{changes_indicator}"

    async def _handle_save(self):
        """Handle save button click."""
        try:
            # Apply current step settings
            step_data = self.step_settings_editor.get_current_value()
            self._apply_step_data(step_data)

            # Apply current function pattern
            func_pattern = self.func_pattern_editor.get_current_value()
            self.editing_func_step.func = func_pattern

            # Copy changes back to original
            for attr in ['name', 'variable_components', 'force_disk_output',
                        'group_by', 'input_dir', 'output_dir', 'func']:
                if hasattr(self.editing_func_step, attr):
                    setattr(self.original_func_step, attr, getattr(self.editing_func_step, attr))

            # Reset change flags
            self.has_step_changes = False
            self.has_func_changes = False
            self._update_save_button_state()

            # Notify callback
            if self.on_save:
                self.on_save(self.original_func_step)

            logger.info("Changes saved successfully")

        except Exception as e:
            logger.error(f"Failed to save changes: {e}")
            from ..utils.dialogs import show_error_dialog
            await show_error_dialog("Save Error", f"Failed to save changes: {e}")

    async def _handle_cancel(self):
        """Handle cancel button click."""
        try:
            # Check for unsaved changes
            if self.has_step_changes or self.has_func_changes:
                from ..utils.dialogs import show_confirmation_dialog

                confirmed = await show_confirmation_dialog(
                    "Unsaved Changes",
                    "You have unsaved changes. Are you sure you want to cancel?"
                )

                if not confirmed:
                    return

            # Reset to original state
            self.editing_func_step = copy.deepcopy(self.original_func_step)
            self.has_step_changes = False
            self.has_func_changes = False
            self._update_save_button_state()

            # Notify callback
            if self.on_cancel:
                self.on_cancel()

            logger.info("Edit cancelled")

        except Exception as e:
            logger.error(f"Failed to cancel edit: {e}")

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self.has_step_changes or self.has_func_changes

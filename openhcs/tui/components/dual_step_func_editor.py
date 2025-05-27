"""
Dual STEP/FUNC Editor for OpenHCS TUI.

Implements the canonical dual editor specified in tui_final.md:
- Replaces plate manager when editing a step
- Toggle between Step settings vs Func pattern editor
- Step settings: All AbstractStep optional params
- Func pattern: Function definitions with kwargs

This is a clean, focused implementation that reduces the entropy
of the original function_pattern_editor.py by following the specification exactly.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, VSplit, Container, Dimension
from prompt_toolkit.widgets import Button, Label, Frame
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl

from ..interfaces.component_interfaces import EditorComponentInterface
from ..state import TUIState

logger = logging.getLogger(__name__)

@dataclass
class StepConfig:
    """Configuration data for a step."""
    name: str = ""
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None
    enabled: bool = True
    description: str = ""

@dataclass
class FuncPattern:
    """Function pattern data."""
    functions: List[Any] = None

    def __post_init__(self):
        if self.functions is None:
            self.functions = []

class DualStepFuncEditor(EditorComponentInterface):
    """
    Dual editor for step configuration and function patterns.

    Implements the canonical dual editor from the TUI specification:
    - Toggle between STEP and FUNC modes
    - STEP mode: Edit AbstractStep parameters
    - FUNC mode: Edit function pattern definitions
    """

    def __init__(self, state: TUIState, step_to_edit: Any = None):
        """
        Initialize the dual editor.

        Args:
            state: TUI state for notifications
            step_to_edit: Step object to edit (optional)
        """
        self.state = state
        self.step_to_edit = step_to_edit

        # Editor state
        self.editing_step_config = True  # Start in STEP mode

        # Data
        self.step_config = StepConfig()
        self.func_pattern = FuncPattern()

        # Extract data from step if provided
        if step_to_edit:
            self._extract_step_data()

        # UI components
        self._container = None
        self._initialize_ui()

    def _extract_step_data(self):
        """Extract data from the step being edited."""
        if not self.step_to_edit:
            return

        try:
            # Extract step configuration
            self.step_config.name = getattr(self.step_to_edit, 'name', '')
            self.step_config.input_dir = getattr(self.step_to_edit, 'input_dir', None)
            self.step_config.output_dir = getattr(self.step_to_edit, 'output_dir', None)
            self.step_config.enabled = getattr(self.step_to_edit, 'enabled', True)
            self.step_config.description = getattr(self.step_to_edit, 'description', '')

            # Extract function pattern if available
            if hasattr(self.step_to_edit, 'function_pattern'):
                pattern = self.step_to_edit.function_pattern
                if isinstance(pattern, list):
                    self.func_pattern.functions = pattern.copy()

        except Exception as e:
            logger.error(f"Failed to extract step data: {e}")

    def _initialize_ui(self):
        """Initialize the UI components."""
        self._container = HSplit([
            self._create_header(),
            self._create_toggle_bar(),
            Window(height=Dimension.exact(1), char='─'),  # Separator
            self._create_content_area()
        ])

    def _create_header(self):
        """Create the editor header."""
        step_name = self.step_config.name or "New Step"
        return Window(
            content=FormattedTextControl([("class:title", f" Editing: {step_name} ")]),
            height=Dimension.exact(1),
            char=' ',
            style="class:title"
        )

    def _create_toggle_bar(self):
        """Create the toggle bar for switching between STEP and FUNC modes."""
        step_button = Button(
            "STEP Settings" if not self.editing_step_config else "[STEP Settings]",
            handler=self._switch_to_step_mode
        )

        func_button = Button(
            "FUNC Pattern" if self.editing_step_config else "[FUNC Pattern]",
            handler=self._switch_to_func_mode
        )

        return VSplit([
            step_button,
            func_button,
            Window(width=Dimension(weight=1)),  # Spacer
            Button("Save", handler=self._save_changes),
            Button("Cancel", handler=self._cancel_editing)
        ], height=Dimension.exact(1))

    def _create_content_area(self):
        """Create the content area that switches between STEP and FUNC editors."""
        if self.editing_step_config:
            return self._create_step_editor()
        else:
            return self._create_func_editor()

    def _create_step_editor(self):
        """Create the step configuration editor."""
        return Frame(
            HSplit([
                # Step name
                VSplit([
                    Label("Name: "),
                    Label(self.step_config.name or "[Enter name]")
                ]),

                # Input directory
                VSplit([
                    Label("Input Dir: "),
                    Label(self.step_config.input_dir or "[Not set]"),
                    Button("Browse", handler=self._browse_input_dir)
                ]),

                # Output directory
                VSplit([
                    Label("Output Dir: "),
                    Label(self.step_config.output_dir or "[Not set]"),
                    Button("Browse", handler=self._browse_output_dir)
                ]),

                # Enabled checkbox
                VSplit([
                    Label("Enabled: "),
                    Label("✓" if self.step_config.enabled else "✗"),
                    Button("Toggle", handler=self._toggle_enabled)
                ]),

                # Description
                VSplit([
                    Label("Description: "),
                    Label(self.step_config.description or "[No description]"),
                    Button("Edit", handler=self._edit_description)
                ])
            ]),
            title="Step Configuration"
        )

    def _create_func_editor(self):
        """Create the function pattern editor."""
        func_count = len(self.func_pattern.functions)

        return Frame(
            HSplit([
                # Function count and controls
                VSplit([
                    Label(f"Functions: {func_count}"),
                    Button("Add", handler=self._add_function),
                    Button("Clear", handler=self._clear_functions)
                ]),

                # Function list (simplified)
                HSplit([
                    Label(f"Function {i+1}: {self._get_function_display(func)}")
                    for i, func in enumerate(self.func_pattern.functions)
                ] if self.func_pattern.functions else [Label("No functions defined")])
            ]),
            title="Function Pattern"
        )

    def _get_function_display(self, func):
        """Get display text for a function."""
        if callable(func):
            return getattr(func, '__name__', str(func))
        elif isinstance(func, tuple) and len(func) >= 1:
            return getattr(func[0], '__name__', str(func[0]))
        else:
            return str(func)

    # Toggle methods
    def _switch_to_step_mode(self):
        """Switch to step configuration mode."""
        if not self.editing_step_config:
            self.editing_step_config = True
            self._refresh_ui()

    def _switch_to_func_mode(self):
        """Switch to function pattern mode."""
        if self.editing_step_config:
            self.editing_step_config = False
            self._refresh_ui()

    def _refresh_ui(self):
        """Refresh the UI after mode switch."""
        # Update toggle bar
        self._container.children[1] = self._create_toggle_bar()

        # Update content area
        self._container.children[3] = self._create_content_area()

        # Invalidate to trigger redraw
        get_app().invalidate()

    # Step editor handlers
    def _browse_input_dir(self):
        """Browse for input directory using FileManager browser."""
        async def browse_async():
            try:
                from ..utils.dialogs import prompt_for_directory_dialog
                from pathlib import Path

                # Current directory or home
                current_dir = Path(self.step_config.input_dir) if self.step_config.input_dir else Path.home()

                # Show directory selection dialog
                selected_path = await prompt_for_directory_dialog(
                    title="Select Input Directory",
                    initial_path=current_dir
                )

                if selected_path:
                    self.step_config.input_dir = str(selected_path)
                    self._refresh_ui()
                    logger.info(f"Input directory set to: {selected_path}")

            except Exception as e:
                logger.error(f"Failed to show input directory browser: {e}")

        # Run async function
        get_app().create_background_task(browse_async())

    def _browse_output_dir(self):
        """Browse for output directory using FileManager browser."""
        async def browse_async():
            try:
                from ..utils.dialogs import prompt_for_directory_dialog
                from pathlib import Path

                # Current directory or home
                current_dir = Path(self.step_config.output_dir) if self.step_config.output_dir else Path.home()

                # Show directory selection dialog
                selected_path = await prompt_for_directory_dialog(
                    title="Select Output Directory",
                    initial_path=current_dir
                )

                if selected_path:
                    self.step_config.output_dir = str(selected_path)
                    self._refresh_ui()
                    logger.info(f"Output directory set to: {selected_path}")

            except Exception as e:
                logger.error(f"Failed to show output directory browser: {e}")

        # Run async function
        get_app().create_background_task(browse_async())

    def _toggle_enabled(self):
        """Toggle step enabled state."""
        self.step_config.enabled = not self.step_config.enabled
        self._refresh_ui()

    def _edit_description(self):
        """Edit step description."""
        from prompt_toolkit.application import get_app
        from prompt_toolkit.widgets import Dialog, Label, Button, TextArea
        from prompt_toolkit.layout import HSplit
        from prompt_toolkit.layout.containers import FloatContainer, Float

        # Create description editor
        desc_input = TextArea(
            text=self.step_config.description,
            multiline=True,
            height=5
        )

        def save_description():
            new_desc = desc_input.text.strip()
            self.step_config.description = new_desc
            self._refresh_ui()
            logger.info(f"Description updated: {new_desc[:50]}...")
            close_dialog()

        def close_dialog():
            get_app().layout = previous_layout
            get_app().invalidate()

        dialog = Dialog(
            title="Edit Step Description",
            body=HSplit([
                Label("Enter step description:"),
                desc_input
            ]),
            buttons=[
                Button("Save", handler=save_description),
                Button("Cancel", handler=close_dialog)
            ],
            width=80,
            modal=True
        )

        # Show dialog
        app = get_app()
        previous_layout = app.layout

        float_container = FloatContainer(
            content=previous_layout.container,
            floats=[
                Float(content=dialog, left=10, top=8)
            ]
        )

        app.layout.container = float_container
        app.invalidate()

    # Function editor handlers
    def _add_function(self):
        """Add a new function to the pattern."""
        from prompt_toolkit.application import get_app
        from prompt_toolkit.widgets import Dialog, Label, Button, RadioList
        from prompt_toolkit.layout import HSplit
        from prompt_toolkit.layout.containers import FloatContainer, Float

        # Available function types
        function_options = [
            ("load_images", "Load Images - Load image files from directory"),
            ("normalize", "Normalize - Normalize image intensities"),
            ("stitch", "Stitch - Stitch images together"),
            ("segment", "Segment - Segment objects in images"),
            ("measure", "Measure - Measure object properties"),
            ("export", "Export - Export results to file")
        ]

        function_selector = RadioList(function_options)

        def add_selected_function():
            selected = function_selector.current_value
            if selected:
                # Create a simple function representation
                func_info = {
                    'type': selected,
                    'name': dict(function_options)[selected].split(' - ')[0],
                    'description': dict(function_options)[selected].split(' - ')[1]
                }
                self.func_pattern.functions.append(func_info)
                self._refresh_ui()
                logger.info(f"Added function: {func_info['name']}")
            close_dialog()

        def close_dialog():
            get_app().layout = previous_layout
            get_app().invalidate()

        dialog = Dialog(
            title="Add Function to Pattern",
            body=HSplit([
                Label("Select function type to add:"),
                function_selector
            ]),
            buttons=[
                Button("Add", handler=add_selected_function),
                Button("Cancel", handler=close_dialog)
            ],
            width=80,
            modal=True
        )

        # Show dialog
        app = get_app()
        previous_layout = app.layout

        float_container = FloatContainer(
            content=previous_layout.container,
            floats=[
                Float(content=dialog, left=10, top=5)
            ]
        )

        app.layout.container = float_container
        app.invalidate()

    def _clear_functions(self):
        """Clear all functions from the pattern."""
        self.func_pattern.functions.clear()
        self._refresh_ui()

    # Save/Cancel handlers
    def _save_changes(self):
        """Save changes and exit editor."""
        try:
            if self.step_to_edit:
                # Apply changes to step object
                self.step_to_edit.name = self.step_config.name
                self.step_to_edit.input_dir = self.step_config.input_dir
                self.step_to_edit.output_dir = self.step_config.output_dir
                self.step_to_edit.enabled = self.step_config.enabled
                self.step_to_edit.description = self.step_config.description

                # Apply function pattern if supported
                if hasattr(self.step_to_edit, 'function_pattern'):
                    self.step_to_edit.function_pattern = self.func_pattern.functions

            # Notify state of completion
            get_app().create_background_task(
                self.state.stop_step_editing()
            )

        except Exception as e:
            logger.error(f"Failed to save step changes: {e}")

    def _cancel_editing(self):
        """Cancel editing and exit without saving."""
        get_app().create_background_task(
            self.state.stop_step_editing()
        )

    # EditorComponentInterface implementation
    @property
    def container(self) -> Container:
        """Return prompt_toolkit container for this component."""
        return self._container

    def update_data(self, data: Any) -> None:
        """Update component with new data."""
        self.step_to_edit = data
        if data:
            self._extract_step_data()
            self._refresh_ui()

    def get_current_value(self) -> Any:
        """Get current edited value."""
        return {
            'step_config': self.step_config,
            'func_pattern': self.func_pattern
        }

    def set_change_callback(self, callback: Callable[[Any], None]) -> None:
        """Set callback for changes."""
        # Not needed for this implementation
        pass

    def reset_to_original(self) -> None:
        """Reset to original values."""
        if self.step_to_edit:
            self._extract_step_data()
            self._refresh_ui()

    def has_changes(self) -> bool:
        """Check if there are unsaved changes."""
        if not self.step_to_edit:
            return False

        # Check if step configuration has changed
        original_name = getattr(self.step_to_edit, 'name', '')
        original_input_dir = getattr(self.step_to_edit, 'input_dir', None)
        original_output_dir = getattr(self.step_to_edit, 'output_dir', None)
        original_enabled = getattr(self.step_to_edit, 'enabled', True)
        original_description = getattr(self.step_to_edit, 'description', '')

        if (self.step_config.name != original_name or
            self.step_config.input_dir != original_input_dir or
            self.step_config.output_dir != original_output_dir or
            self.step_config.enabled != original_enabled or
            self.step_config.description != original_description):
            return True

        # Check if function pattern has changed
        original_functions = getattr(self.step_to_edit, 'function_pattern', [])
        if len(self.func_pattern.functions) != len(original_functions):
            return True

        # Simple comparison - in a real implementation, this would be more sophisticated
        return len(self.func_pattern.functions) > 0

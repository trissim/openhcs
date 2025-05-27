"""
Function Pattern Editor for Hybrid TUI.

Ported from TUI's function_pattern_editor.py with adaptations for:
- Schema-free operation using static analysis
- Component interface compliance
- Async/await patterns
- Clean separation from TUIState dependencies

Key Features:
- Dict key management with dropdown + +/- buttons
- Function registry integration
- Dynamic parameter forms from function signatures
- External vim editor integration
- Load/save .func file operations
- Function list management (add/delete/reorder)
"""

import asyncio
import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, VSplit, ScrollablePane, Container
from prompt_toolkit.widgets import Button, Label, Box, Frame
from prompt_toolkit.formatted_text import HTML

from ..interfaces.component_interfaces import EditorComponentInterface
from ..utils.static_analysis import (
    get_function_signature,
    get_function_registry_by_backend,
    analyze_function_pattern
)
from ..utils.dialogs import show_error_dialog, prompt_for_path_dialog
from ..utils.file_operations import load_func_pattern, save_func_pattern

logger = logging.getLogger(__name__)

def get_function_info(func: Callable) -> Dict[str, Any]:
    """
    Get function information for UI display.

    Args:
        func: Function to analyze

    Returns:
        Dict with name, backend, and other metadata
    """
    if not func:
        return {"name": "None", "backend": "unknown"}

    try:
        # Get function name
        name = getattr(func, '__name__', str(func))

        # Get backend from function attributes
        backend = getattr(func, 'backend', 'unknown')
        if backend == 'unknown':
            # Try to determine from memory type attributes
            if hasattr(func, 'input_memory_type'):
                backend = func.input_memory_type
            elif hasattr(func, 'output_memory_type'):
                backend = func.output_memory_type

        return {
            "name": name,
            "backend": backend,
            "signature": get_function_signature(func)
        }

    except Exception as e:
        logger.warning(f"Failed to get function info for {func}: {e}")
        return {"name": str(func), "backend": "unknown", "error": str(e)}

class FunctionPatternEditor(EditorComponentInterface):
    """
    Function Pattern Editor component for hybrid TUI.

    Provides editing capabilities for function patterns with dict key management,
    function list operations, and external editor integration.
    """

    def __init__(
        self,
        initial_pattern: Union[List, Dict, None] = None,
        change_callback: Optional[Callable[[Any], None]] = None
    ):
        """
        Initialize the Function Pattern Editor.

        Args:
            initial_pattern: Initial function pattern to edit
            change_callback: Callback to notify when pattern changes
        """
        self.change_callback = change_callback

        # Pattern state
        self.original_pattern = self._clone_pattern(initial_pattern or [])
        self.current_pattern = self._clone_pattern(self.original_pattern)

        # UI state
        self.is_dict = isinstance(self.current_pattern, dict)
        self.current_key = None
        if self.is_dict and self.current_pattern:
            self.current_key = list(self.current_pattern.keys())[0]

        # UI components
        self.header = None
        self.key_selector_container = HSplit([])
        self.function_list_container = ScrollablePane(HSplit([]))
        self._container = None

        # Initialize UI
        self._initialize_ui()

    def _initialize_ui(self):
        """Initialize UI components."""
        self.header = self._create_header()
        self._refresh_key_selector()
        self._refresh_function_list()

        # Create main container
        self._container = HSplit([
            self.header,
            self.key_selector_container,
            self.function_list_container
        ])

    @property
    def container(self) -> Container:
        """Return prompt_toolkit container for this component."""
        return self._container

    def update_data(self, data: Any) -> None:
        """Update component with new pattern data."""
        self.current_pattern = self._clone_pattern(data or [])
        self.is_dict = isinstance(self.current_pattern, dict)

        if self.is_dict and self.current_pattern:
            self.current_key = list(self.current_pattern.keys())[0]
        else:
            self.current_key = None

        self._refresh_key_selector()
        self._refresh_function_list()

    def get_current_value(self) -> Any:
        """Get current edited pattern."""
        return self._clone_pattern(self.current_pattern)

    def set_change_callback(self, callback: Callable[[Any], None]) -> None:
        """Set callback for pattern changes."""
        self.change_callback = callback

    def reset_to_original(self) -> None:
        """Reset to original pattern."""
        self.current_pattern = self._clone_pattern(self.original_pattern)
        self.is_dict = isinstance(self.current_pattern, dict)

        if self.is_dict and self.current_pattern:
            self.current_key = list(self.current_pattern.keys())[0]
        else:
            self.current_key = None

        self._refresh_key_selector()
        self._refresh_function_list()
        self._notify_change()

    def has_changes(self) -> bool:
        """Check if component has unsaved changes."""
        return self.current_pattern != self.original_pattern

    def _clone_pattern(self, pattern) -> Union[List, Dict]:
        """Create a deep copy of a function pattern."""
        if isinstance(pattern, dict):
            return {k: self._clone_pattern(v) for k, v in pattern.items()}
        elif isinstance(pattern, list):
            return [self._clone_list_item(item) for item in pattern]
        elif isinstance(pattern, tuple) and len(pattern) == 2 and callable(pattern[0]):
            return (pattern[0], pattern[1].copy() if isinstance(pattern[1], dict) else {})
        else:
            return pattern

    def _clone_list_item(self, item):
        """Clone a single list item (function or tuple)."""
        if isinstance(item, tuple) and len(item) == 2:
            func, kwargs = item
            return (func, kwargs.copy() if isinstance(kwargs, dict) else {})
        else:
            return item

    def _notify_change(self):
        """Notify parent component of pattern change."""
        if self.change_callback:
            try:
                self.change_callback(self.get_current_value())
            except Exception as e:
                logger.error(f"Error in change callback: {e}")

    def _create_header(self):
        """Create header with action buttons."""
        title = Label(HTML("<b>Function Pattern Editor</b>"))

        add_func_button = Button(
            "Add Func",
            handler=lambda: get_app().create_background_task(self._add_function())
        )
        load_func_button = Button(
            "Load .func",
            handler=lambda: get_app().create_background_task(self._load_func_pattern())
        )
        save_as_func_button = Button(
            "Save .func As",
            handler=lambda: get_app().create_background_task(self._save_func_pattern())
        )
        edit_in_vim_button = Button(
            "Edit in Vim",
            handler=lambda: get_app().create_background_task(self._edit_in_vim())
        )

        return VSplit([
            title,
            Box(add_func_button, padding_left=2),
            Box(load_func_button, padding_left=1),
            Box(save_as_func_button, padding_left=1),
            Box(edit_in_vim_button, padding_left=1)
        ], height=1, padding=0)

    def _refresh_key_selector(self):
        """Refresh the key selector UI based on current pattern type."""
        from .grouped_dropdown import GroupedDropdown

        if self.is_dict:
            # Create key dropdown
            key_options = list(self.current_pattern.keys()) if self.current_pattern else []
            if not key_options:
                key_options = ["None"]
                self.current_key = "None"
            elif self.current_key not in key_options:
                self.current_key = key_options[0]

            key_dropdown = GroupedDropdown(
                values=[(key, key) for key in key_options],
                default=self.current_key,
                on_change=self._on_key_selected
            )

            # Key management buttons
            add_key_button = Button(
                "+",
                handler=lambda: get_app().create_background_task(self._add_key())
            )
            remove_key_button = Button(
                "-",
                handler=lambda: get_app().create_background_task(self._remove_key())
            )

            key_management_buttons = VSplit([add_key_button, remove_key_button], padding=1)

            self.key_selector_container.children = [
                VSplit([
                    Label("Pattern Keys: "),
                    key_dropdown.container,
                    key_management_buttons
                ])
            ]
        else:
            # Not a dict - offer conversion
            convert_to_dict_button = Button(
                "Convert to Dict Pattern",
                handler=lambda: get_app().create_background_task(self._convert_list_to_dict_pattern())
            )
            self.key_selector_container.children = [convert_to_dict_button]

    def _refresh_function_list(self):
        """Refresh the function list UI."""
        functions = self._get_current_functions()
        function_items = []

        for i, func_item in enumerate(functions):
            func, kwargs = self._extract_func_and_kwargs(func_item)
            function_items.append(self._create_function_item(i, func, kwargs))

        # Add "Add Function" button
        add_function_button = Button(
            "Add Function",
            handler=lambda: get_app().create_background_task(self._add_function())
        )
        function_items.append(add_function_button)

        self.function_list_container.content = HSplit(function_items)

    def _get_current_functions(self) -> List:
        """Get the current function list based on pattern type and selected key."""
        if self.is_dict:
            if self.current_key and self.current_key in self.current_pattern:
                return self.current_pattern[self.current_key]
            else:
                return []
        else:
            return self.current_pattern if isinstance(self.current_pattern, list) else []

    def _extract_func_and_kwargs(self, func_item):
        """Extract function and kwargs from a function item."""
        if isinstance(func_item, tuple) and len(func_item) == 2:
            return func_item[0], func_item[1]
        elif callable(func_item):
            return func_item, {}
        else:
            return None, {}

    def _create_function_item(self, index: int, func: Callable, kwargs: Dict):
        """Create UI component for a single function item."""
        func_info = get_function_info(func)

        # Function dropdown
        func_dropdown = self._create_function_dropdown(index, func)

        # Function controls
        move_up = Button("↑", handler=lambda: get_app().create_background_task(self._move_function_up(index)))
        move_down = Button("↓", handler=lambda: get_app().create_background_task(self._move_function_down(index)))
        delete_button = Button("Delete", handler=lambda: get_app().create_background_task(self._delete_function(index)))

        # Parameter editor (simplified for now)
        param_editor = self._create_parameter_editor(index, func, kwargs)

        # Combine components
        return Frame(
            HSplit([
                # Function header
                VSplit([
                    func_dropdown,
                    Box(move_up, width=3),
                    Box(move_down, width=3),
                    Box(delete_button, width=8)
                ]),
                # Parameter editor
                param_editor
            ]),
            title=f"Function {index+1}: {func_info['name']} ({func_info['backend']})"
        )

    def _create_function_dropdown(self, index: int, current_func: Callable):
        """Create function selection dropdown."""
        from .grouped_dropdown import GroupedDropdown

        # Get function registry
        registry = get_function_registry_by_backend()

        # Build dropdown options grouped by backend
        options = []
        for backend, funcs in registry.items():
            for func in funcs:
                func_info = get_function_info(func)
                options.append((f"{backend}: {func_info['name']}", func))

        # Find current selection
        current_value = current_func
        for display, func in options:
            if func == current_func:
                current_value = func
                break

        return GroupedDropdown(
            values=options,
            default=current_value,
            on_change=lambda func: get_app().create_background_task(self._on_function_selected(index, func))
        )

    def _create_parameter_editor(self, index: int, func: Callable, kwargs: Dict):
        """Create parameter editor for function."""
        if not func:
            return Label("No function selected")

        try:
            from .parameter_editor import ParameterEditor

            # Create parameter editor with callbacks bound to this function index
            param_editor = ParameterEditor(
                func=func,
                current_kwargs=kwargs,
                on_parameter_change=lambda param_name, new_value: self._update_parameter(index, param_name, new_value),
                on_reset_parameter=lambda param_name: self._reset_parameter(index, param_name),
                on_reset_all_parameters=lambda: self._reset_all_parameters(index)
            )

            return param_editor.container

        except Exception as e:
            logger.error(f"Failed to create parameter editor for {func}: {e}")
            return Label(f"Parameter editor error: {e}")

    def _reset_all_parameters(self, func_index: int):
        """Reset all parameters for function at given index."""
        try:
            functions = self._get_current_functions()
            if 0 <= func_index < len(functions):
                func, _ = self._extract_func_and_kwargs(functions[func_index])

                # Reset to empty kwargs (use function defaults)
                functions[func_index] = (func, {})
                self._update_pattern_functions(functions)
                self._refresh_function_list()

        except Exception as e:
            logger.error(f"Failed to reset all parameters for function {func_index}: {e}")

    # Action methods
    async def _add_function(self):
        """Add a new function to the pattern."""
        try:
            # Get default function from registry
            registry = get_function_registry_by_backend()
            default_func = None

            for backend_funcs in registry.values():
                if backend_funcs:
                    default_func = backend_funcs[0]
                    break

            functions = self._get_current_functions()
            functions.append((default_func, {}) if default_func else (None, {}))

            self._update_pattern_functions(functions)
            self._refresh_function_list()

        except Exception as e:
            logger.error(f"Failed to add function: {e}")
            await show_error_dialog("Error", f"Failed to add function: {e}")

    async def _delete_function(self, index: int):
        """Delete function at given index."""
        try:
            functions = self._get_current_functions()
            if 0 <= index < len(functions):
                functions.pop(index)
                self._update_pattern_functions(functions)
                self._refresh_function_list()

        except Exception as e:
            logger.error(f"Failed to delete function at index {index}: {e}")
            await show_error_dialog("Error", f"Failed to delete function: {e}")

    async def _move_function_up(self, index: int):
        """Move function up in the list."""
        try:
            functions = self._get_current_functions()
            if index > 0 and index < len(functions):
                functions[index], functions[index-1] = functions[index-1], functions[index]
                self._update_pattern_functions(functions)
                self._refresh_function_list()

        except Exception as e:
            logger.error(f"Failed to move function up at index {index}: {e}")

    async def _move_function_down(self, index: int):
        """Move function down in the list."""
        try:
            functions = self._get_current_functions()
            if index >= 0 and index < len(functions) - 1:
                functions[index], functions[index+1] = functions[index+1], functions[index]
                self._update_pattern_functions(functions)
                self._refresh_function_list()

        except Exception as e:
            logger.error(f"Failed to move function down at index {index}: {e}")

    async def _on_function_selected(self, index: int, func: Callable):
        """Handle function selection change."""
        try:
            functions = self._get_current_functions()
            if 0 <= index < len(functions):
                # Preserve existing kwargs if possible
                _, existing_kwargs = self._extract_func_and_kwargs(functions[index])
                functions[index] = (func, existing_kwargs)

                self._update_pattern_functions(functions)
                self._refresh_function_list()

        except Exception as e:
            logger.error(f"Failed to update function at index {index}: {e}")

    def _update_parameter(self, func_index: int, param_name: str, new_value: Any):
        """Update parameter value for function."""
        try:
            functions = self._get_current_functions()
            if 0 <= func_index < len(functions):
                func, kwargs = self._extract_func_and_kwargs(functions[func_index])

                # Update parameter value (already converted by parameter editor)
                kwargs[param_name] = new_value

                functions[func_index] = (func, kwargs)
                self._update_pattern_functions(functions)

        except Exception as e:
            logger.error(f"Failed to update parameter {param_name}: {e}")

    def _reset_parameter(self, func_index: int, param_name: str):
        """Reset parameter to default value."""
        try:
            functions = self._get_current_functions()
            if 0 <= func_index < len(functions):
                func, kwargs = self._extract_func_and_kwargs(functions[func_index])

                # Remove parameter to use default
                if param_name in kwargs:
                    del kwargs[param_name]

                functions[func_index] = (func, kwargs)
                self._update_pattern_functions(functions)
                self._refresh_function_list()

        except Exception as e:
            logger.error(f"Failed to reset parameter {param_name}: {e}")

    def _update_pattern_functions(self, functions: List):
        """Update the pattern with new function list."""
        if self.is_dict:
            if self.current_key:
                self.current_pattern[self.current_key] = functions
        else:
            self.current_pattern = functions

        self._notify_change()

    def _on_key_selected(self, key: str):
        """Handle key selection change."""
        self.current_key = key
        self._refresh_function_list()

    async def _add_key(self):
        """Add new key to dict pattern."""
        try:
            from ..utils.dialogs import show_confirmation_dialog

            # Simple key input (could be enhanced with a proper input dialog)
            new_key = f"key_{len(self.current_pattern) + 1}"

            if new_key not in self.current_pattern:
                self.current_pattern[new_key] = []
                self.current_key = new_key
                self._refresh_key_selector()
                self._refresh_function_list()
                self._notify_change()

        except Exception as e:
            logger.error(f"Failed to add key: {e}")
            await show_error_dialog("Error", f"Failed to add key: {e}")

    async def _remove_key(self):
        """Remove current key from dict pattern."""
        try:
            if self.current_key and self.current_key in self.current_pattern:
                del self.current_pattern[self.current_key]

                # Select new current key
                if self.current_pattern:
                    self.current_key = list(self.current_pattern.keys())[0]
                else:
                    self.current_key = None

                self._refresh_key_selector()
                self._refresh_function_list()
                self._notify_change()

        except Exception as e:
            logger.error(f"Failed to remove key: {e}")
            await show_error_dialog("Error", f"Failed to remove key: {e}")

    async def _convert_list_to_dict_pattern(self):
        """Convert list pattern to dict pattern."""
        try:
            if not self.is_dict:
                # Convert list to dict with "None" key
                dict_pattern = {"None": self.current_pattern}
                self.current_pattern = dict_pattern
                self.is_dict = True
                self.current_key = "None"

                self._refresh_key_selector()
                self._refresh_function_list()
                self._notify_change()

        except Exception as e:
            logger.error(f"Failed to convert to dict pattern: {e}")
            await show_error_dialog("Error", f"Failed to convert pattern: {e}")

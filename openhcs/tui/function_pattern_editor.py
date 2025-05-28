from prompt_toolkit.layout import Container
"""
Function Pattern Editor for OpenHCS TUI.

This module provides a UI component for editing function patterns in OpenHCS,
leveraging static reflection to generate UI components dynamically based on
function registry and signature inspection.

🔒 Clause 234: Pattern Type Conversion Requires Structural Truth
When converting function patterns from List to Dict, use None as the key
to indicate an unnamed structural group, preserving semantic parity.

🔒 Clause 92: Structural Validation First
All external inputs (especially Vim-edited files) are validated before processing
to prevent arbitrary code execution.
"""

import inspect
import pickle # Added for .func load/save
from pathlib import Path # Added for .func load/save
import asyncio # Added for async handlers
import logging # Added for logging in new handlers
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from prompt_toolkit.application import get_app
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.layout import HSplit, ScrollablePane, VSplit, Container
from prompt_toolkit.widgets import (Box, Button, Dialog, Label, TextArea, RadioList as Dropdown)
# Define SafeButton locally to avoid circular imports
class SafeButton(Button):
    """Safe wrapper around Button that handles formatting errors."""

    def __init__(self, text="", handler=None, width=None, **kwargs):
        # Sanitize text before passing to parent
        if text is not None:
            text = str(text).replace('{', '{{').replace('}', '}}').replace(':', ' ')
        super().__init__(text=text, handler=handler, width=width, **kwargs)

    def _get_text_fragments(self):
        """Safe version that handles formatting errors gracefully."""
        try:
            return super()._get_text_fragments()
        except (ValueError, TypeError, AttributeError):
            # Fallback to simple text formatting without centering
            text = str(self.text) if self.text is not None else ""
            safe_text = text.replace('{', '{{').replace('}', '}}')
            return [("class:button", f" {safe_text} ")]

# Import custom components
from openhcs.tui.components import GroupedDropdown, ParameterEditor
from openhcs.tui.services.external_editor_service import ExternalEditorService
from openhcs.tui.utils import show_error_dialog, prompt_for_path_dialog # Import prompt_for_path_dialog

from openhcs.constants.constants import VALID_MEMORY_TYPES
from openhcs.core.pipeline.funcstep_contract_validator import \
    FuncStepContractValidator
# Import from func_registry instead of function_registry to avoid circular imports
from openhcs.processing.func_registry import FUNC_REGISTRY


logger = logging.getLogger(__name__) # Added logger

def get_function_info(func):
    """
    Get metadata for a function from its decorators.

    Leverages the function registry and decorator metadata to extract
    information about the function's backend, memory types, and special parameters.

    Args:
        func: The function to get metadata for

    Returns:
        Dictionary of function metadata

    Raises:
        ValueError: If the function is not callable or lacks required metadata
    """
    if not callable(func):
        raise ValueError(f"Not a callable: {func}")

    # Get basic function metadata
    info = {
        'name': func.__name__,
        'backend': getattr(func, 'backend', 'unknown'),
        'input_memory_type': getattr(func, 'input_memory_type', None),
        'output_memory_type': getattr(func, 'output_memory_type', None),
        'special_inputs': getattr(func, 'special_inputs', []),
        'special_outputs': getattr(func, 'special_outputs', [])
    }

    # Validate memory types
    if info['input_memory_type'] not in VALID_MEMORY_TYPES:
        raise ValueError(f"Invalid input memory type: {info['input_memory_type']}")

    if info['output_memory_type'] not in VALID_MEMORY_TYPES:
        raise ValueError(f"Invalid output memory type: {info['output_memory_type']}")

    return info


class PatternValidationError(Exception):
    """Exception raised when pattern validation fails."""
    pass


class FunctionPatternEditor:
    """Function Pattern Editor leveraging OpenHCS's static reflection."""

    def __init__(self, state: Any, initial_pattern: Union[List, Dict, None] = None, change_callback: Optional[Callable] = None):
        """
        Initialize the Function Pattern Editor.

        Args:
            state: The TUIState instance.
            initial_pattern: The initial function pattern to edit.
            change_callback: Callback to notify when the pattern changes.
        """
        self.state = state
        self.change_callback = change_callback
        self.external_editor_service = ExternalEditorService(state) # Instantiate the service

        # Use and clone the provided pattern
        self.original_pattern = self._clone_pattern(initial_pattern if initial_pattern is not None else [])
        self.current_pattern = self._clone_pattern(self.original_pattern)

        # Determine pattern type
        self.is_dict = isinstance(self.current_pattern, dict)
        self.current_key = None if not self.is_dict else next(iter(self.current_pattern), None)

        # Instantiate ParameterEditor
        # The func_index is implicitly handled by which function's parameters are being shown.
        # The ParameterEditor will need the currently selected function and its kwargs.
        # We'll pass stubs for callbacks for now, to be implemented fully later.

        # Find the initial function and its kwargs to pass to ParameterEditor
        initial_func_for_editor, initial_kwargs_for_editor = self._get_initial_func_for_param_editor()

        self.parameter_editor = ParameterEditor(
            func=initial_func_for_editor,
            current_kwargs=initial_kwargs_for_editor,
            on_parameter_change=self._handle_parameter_change,
            on_reset_parameter=self._handle_reset_parameter,
            on_reset_all_parameters=self._handle_reset_all_parameters
        )

        # Create UI components
        self.header = self._create_header()
        self.key_selector_container = HSplit([])
        self.function_list_container = ScrollablePane(HSplit([])) # This will contain items that include the parameter_editor

        # Initial UI refresh
        self._refresh_key_selector()
        self._refresh_function_list() # This will now build items that use self.parameter_editor

        # Create container
        self._container = HSplit([
            self.header,
            self.key_selector_container,
            self.function_list_container
            # The parameter editor is now part of items within function_list_container
        ])

    @property
    def container(self) -> Container:
        """Return the main container for the FunctionPatternEditor."""
        return self._container # Assuming _container is set in __init__

    def _extract_pattern(self, step) -> Union[List, Dict]:
        """Extract function pattern from a step."""
        if not step:
            return []
        return step.get('func', [])

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
        """Clone a single item from a function list."""
        if isinstance(item, tuple) and len(item) == 2 and callable(item[0]):
            return (item[0], item[1].copy() if isinstance(item[1], dict) else {})
        else:
            return item

    def _get_initial_func_for_param_editor(self) -> Tuple[Optional[Callable], Dict[str, Any]]:
        """Helper to get the first function and its kwargs for initial ParameterEditor setup."""
        functions = self._get_current_functions()
        if functions:
            func, kwargs = self._extract_func_and_kwargs(functions[0])
            return func, kwargs
        return None, {}

    # --- Callback handlers for ParameterEditor ---
    async def _handle_parameter_change(self, param_name: str, new_value_str: str, func_index: int):
        """
        Handles parameter change callback from a ParameterEditor instance.
        Parses the string value and then updates the model.
        """
        parsed_value = self._parse_parameter_value(new_value_str)
        await self._update_function_parameter(param_name, parsed_value, func_index)

    def _parse_parameter_value(self, value_str: str) -> Any:
        """Parse string value to appropriate Python type."""
        if value_str.lower() == 'none':
            return None
        elif value_str.lower() == 'true':
            return True
        elif value_str.lower() == 'false':
            return False
        else:
            return self._parse_numeric_or_string(value_str)

    def _parse_numeric_or_string(self, value_str: str) -> Any:
        """Parse value as numeric type or fallback to string."""
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                return value_str

    async def _update_function_parameter(self, param_name: str, parsed_value: Any, func_index: int):
        """Update a specific parameter for a function."""
        functions = self._get_current_functions()

        if not self._is_valid_function_index(func_index, functions):
            return

        current_func, current_kwargs = self._extract_func_and_kwargs(functions[func_index])
        if current_func is None:
            return

        current_kwargs[param_name] = parsed_value
        functions[func_index] = (current_func, current_kwargs)
        self._update_pattern_functions(functions)
        self._refresh_function_list()

    def _is_valid_function_index(self, func_index: int, functions: list) -> bool:
        """Check if function index is valid."""
        return 0 <= func_index < len(functions)

    async def _handle_reset_parameter(self, param_name: str, func_index: int):
        """Handles reset parameter callback from a ParameterEditor instance."""
        functions = self._get_current_functions()
        if not self._is_valid_function_index(func_index, functions):
            return

        current_func, current_kwargs = self._extract_func_and_kwargs(functions[func_index])
        if not current_func:
            return

        default_value = self._get_parameter_default_value(current_func, param_name)
        self._apply_parameter_reset(current_kwargs, param_name, default_value)

        functions[func_index] = (current_func, current_kwargs)
        self._update_pattern_functions(functions)
        self._refresh_function_list()

    def _get_parameter_default_value(self, func: Callable, param_name: str) -> Any:
        """Get the default value for a parameter from function signature."""
        sig = inspect.signature(func)
        if param_name in sig.parameters:
            return sig.parameters[param_name].default
        return inspect.Parameter.empty

    def _apply_parameter_reset(self, kwargs: Dict, param_name: str, default_value: Any):
        """Apply parameter reset based on default value."""
        if default_value is not inspect.Parameter.empty and default_value is not None:
            kwargs[param_name] = default_value
        elif param_name in kwargs:
            del kwargs[param_name]

    async def _handle_reset_all_parameters(self, func_index: int):
        """Handles reset all parameters callback from a ParameterEditor instance."""
        functions = self._get_current_functions()
        if not self._is_valid_function_index(func_index, functions):
            return

        current_func, _ = self._extract_func_and_kwargs(functions[func_index])
        if not current_func:
            return

        new_kwargs = self._extract_all_default_parameters(current_func)
        functions[func_index] = (current_func, new_kwargs)
        self._update_pattern_functions(functions)
        self._refresh_function_list()

    def _extract_all_default_parameters(self, func: Callable) -> Dict[str, Any]:
        """Extract all parameters with default values from function signature."""
        params_with_defaults = self._get_parameters_with_defaults(func)
        return {p['name']: p['default'] for p in params_with_defaults if p['default'] is not None}

    def _get_parameters_with_defaults(self, func: Callable) -> List[Dict[str, Any]]:
        """Get parameters with their default values from function signature."""
        params_info = []
        sig = inspect.signature(func)

        for name, param in sig.parameters.items():
            if self._should_skip_parameter(name):
                continue

            default_val = param.default if param.default is not inspect.Parameter.empty else None
            params_info.append({'name': name, 'default': default_val})

        return params_info

    def _should_skip_parameter(self, param_name: str) -> bool:
        """Check if parameter should be skipped (self, cls)."""
        return param_name in ('self', 'cls')
    # --- End Callback Handlers ---

    def _create_header(self):
        """Create the editor header with title."""
        # Save/Cancel buttons are removed as this will be a sub-component.
        # DualStepFuncEditorPane will handle the main save/close.
        title = Label(HTML("<b>Function Pattern Editor</b>"))

        add_func_button = SafeButton("Add Func",
            handler=lambda: get_app().create_background_task(self._add_function())
        )
        load_func_button = SafeButton("Load .func",
            handler=lambda: get_app().create_background_task(self._load_func_pattern_from_file_handler())
        )
        save_as_func_button = SafeButton("Save .func As",
            handler=lambda: get_app().create_background_task(self._save_func_pattern_as_file_handler())
        )
        edit_in_vim_button = SafeButton("Edit in Vim",
            handler=lambda: get_app().create_background_task(self._edit_in_vim())
        )

        return VSplit([
            title,
            Box(add_func_button, padding_left=2),
            Box(load_func_button, padding_left=1),
            Box(save_as_func_button, padding_left=1),
            Box(edit_in_vim_button, padding_left=1)
        ], height=1, padding=0) # Ensure it's a single line bar

    def _notify_change(self):
        """Notify the parent component of a change."""
        if self.change_callback:
            self.change_callback()

    def get_pattern(self) -> Union[List, Dict]:
        """Returns the current state of the edited pattern."""
        # Ensure the latest changes from any sub-components are reflected if necessary
        # For instance, if a key was being edited, ensure it's saved to self.current_pattern
        if self.is_dict and self.current_key is not None:
             active_list = self._get_current_functions() # This gets from self.current_pattern based on current_key
             # No, this is not needed, _get_current_functions reads from self.current_pattern
             # self.current_pattern[self.current_key] = active_list
        elif not self.is_dict:
            # self.current_pattern is already the list
            pass
        return self._clone_pattern(self.current_pattern) # Return a clone

    def _refresh_key_selector(self):
        """
        Rebuild the key selector dropdown.

        🔒 Clause 234: Display "Unnamed" for None key to avoid user confusion
        while preserving None in the data model.
        """
        # Get keys for dropdown
        if self.is_dict:
            # Convert None to "Unnamed" for display purposes
            display_keys = []
            for k in self.current_pattern.keys():
                if k is None:
                    display_keys.append((None, "Unnamed"))
                else:
                    display_keys.append((k, str(k)))
        else:
            display_keys = [("None", "None")]

        # Create dropdown for key selection
        key_dropdown = Dropdown(
            options=display_keys,
            default=self.current_key
        )

        # Set handler for key selection
        def on_key_change(key):
            get_app().create_background_task(self._switch_key(key))

        key_dropdown.handler = on_key_change

        # Create key management buttons
        add_key_button = SafeButton("Add Key", # More descriptive
            handler=lambda: get_app().create_background_task(self._add_key())
        )
        remove_key_button = SafeButton("Remove Key", # More descriptive
            handler=lambda: get_app().create_background_task(self._remove_key())
        )
        # Edit in Vim button is now in the header

        # Create container
        key_management_buttons = VSplit([add_key_button, remove_key_button], padding=1)

        if self.is_dict:
            self.key_selector_container.children = [
                VSplit([
                    Label("Pattern Keys: "), # Renamed for clarity
                    key_dropdown,
                    key_management_buttons
                ])
            ]
        else: # Not a dict, no key selector needed, but can offer to convert to dict
            convert_to_dict_button = SafeButton("Convert to Dict Pattern",
                handler=lambda: get_app().create_background_task(self._convert_list_to_dict_pattern())
            )
            self.key_selector_container.children = [convert_to_dict_button]


    def _refresh_function_list(self):
        """Rebuild the function list."""
        functions = self._get_current_functions()
        function_items = self._build_function_list(functions)
        self.function_list_container.content = HSplit(function_items)

    def _build_function_list(self, functions=None):
        """Build the function list items."""
        if functions is None:
            functions = self._get_current_functions()

        function_items = []

        for i, func_item in enumerate(functions):
            func, kwargs = self._extract_func_and_kwargs(func_item)
            function_items.append(self._create_function_item(i, func, kwargs))

        # Add "Add Function" button
        add_function_button = SafeButton("Add Function",
            handler=lambda: get_app().create_background_task(self._add_function())
        )
        function_items.append(add_function_button)

        return function_items

    def _create_function_item(self, index, func, kwargs):
        """Create a UI component for a function item."""
        from prompt_toolkit.widgets import Frame

        # Get function info from registry
        func_info = get_function_info(func) if func else {"name": "None", "backend": ""}

        # Function header with dropdown
        func_dropdown = self._create_function_dropdown(index, func)

        # Function controls
        move_up = SafeButton("↑",
            handler=lambda: get_app().create_background_task(self._move_function_up(index))
        )
        move_down = SafeButton("↓",
            handler=lambda: get_app().create_background_task(self._move_function_down(index))
        )
        delete_button = SafeButton("Delete",
            handler=lambda: get_app().create_background_task(self._delete_function(index))
        )

        # Parameter editor: Instantiate a new ParameterEditor for each item
        # The callbacks will need to be adapted or new ones created that are specific to this 'index'
        item_param_editor = ParameterEditor(
            func=func,
            current_kwargs=kwargs,
            # These handlers in FunctionPatternEditor will use 'index' to update the correct item
            on_parameter_change=lambda p_name, p_val_str, idx=index: get_app().create_background_task(self._handle_parameter_change(p_name, p_val_str, idx)),
            on_reset_parameter=lambda p_name, idx=index: get_app().create_background_task(self._handle_reset_parameter(p_name, idx)),
            on_reset_all_parameters=lambda idx=index: get_app().create_background_task(self._handle_reset_all_parameters(idx))
        )
        param_editor_container = item_param_editor # This is the UI container from ParameterEditor

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
                param_editor_container
            ]),
            title=f"Function {index+1}: {func_info['name']} ({func_info['backend']})"
        )

    def _get_current_functions(self) -> List:
        """
        Get the functions for the current key.

        🔒 Clause 234: Support None as a valid group key for unnamed structural groups.
        """
        if self.is_dict:
            # Handle None key (Clause 234)
            if self.current_key is None and None in self.current_pattern:
                functions = self.current_pattern[None]
            elif self.current_key in self.current_pattern:
                functions = self.current_pattern[self.current_key]
            else:
                functions = []
        else:
            functions = self.current_pattern

        # Ensure it's a list
        if not isinstance(functions, list):
            functions = [functions]

        return functions

    def _extract_func_and_kwargs(self, func_item) -> Tuple[Optional[Callable], Dict]:
        """Extract function and kwargs from a function item."""
        if isinstance(func_item, tuple) and len(func_item) == 2 and callable(func_item[0]):
            return func_item[0], func_item[1]
        elif callable(func_item):
            return func_item, {}
        else:
            return None, {}

    def _create_function_dropdown(self, index, current_func):
        """
        Create a dropdown for selecting functions from the registry.

        Uses FunctionRegistry to get decorated functions grouped by backend.
        Marks backend headers as disabled to prevent selection.

        Args:
            index: The index of the function in the list
            current_func: The currently selected function

        Returns:
            A dropdown component for selecting functions
        """
        from prompt_toolkit.formatted_text import HTML

        # Get functions grouped by backend
        functions_by_backend = self._get_functions_by_backend()

        # Create dropdown options with headers
        dropdown_options = self._create_dropdown_options(functions_by_backend)

        # Create and configure dropdown
        dropdown = self._create_configured_dropdown(dropdown_options, current_func, index)

        return dropdown

    def _get_functions_by_backend(self) -> Dict[str, List[Tuple[Callable, str]]]:
        """Get functions from registry grouped by backend."""
        functions_by_backend = {}

        for backend, funcs in FUNC_REGISTRY.items():
            for func in funcs:
                try:
                    info = get_function_info(func)
                    if backend not in functions_by_backend:
                        functions_by_backend[backend] = []

                    # Include memory type in display name
                    display_name = f"{func.__name__} ({info['input_memory_type']} → {info['output_memory_type']})"
                    functions_by_backend[backend].append((func, display_name))
                except ValueError:
                    # Skip functions without proper metadata
                    continue

        return functions_by_backend

    def _create_dropdown_options(self, functions_by_backend: Dict[str, List[Tuple[Callable, str]]]) -> List[Tuple]:
        """Create dropdown options with disabled headers."""
        from prompt_toolkit.formatted_text import HTML

        dropdown_options = []
        for backend, funcs in sorted(functions_by_backend.items()):
            # Add backend header as disabled option
            dropdown_options.append(('header', HTML(f"<b>--- {backend} ---</b>")))
            # Add functions
            dropdown_options.extend([(func, name) for func, name in funcs])

        return dropdown_options

    def _create_configured_dropdown(self, dropdown_options: List[Tuple], current_func: Callable, index: int) -> GroupedDropdown:
        """Create and configure the dropdown with handler."""
        dropdown = GroupedDropdown(
            options=dropdown_options,
            default=current_func
        )

        # Set handler for selection change
        def on_selection_change(func):
            # Skip header entries
            if func != 'header':
                get_app().create_background_task(self._update_function(index, func))

        dropdown.handler = on_selection_change
        return dropdown

    async def _switch_key(self, key):
        """
        Switch to a different dictionary key.

        🔒 Clause 234: Support None as a valid group key for unnamed structural groups.
        """
        if key == self.current_key:
            return

        # Save current functions for the old key
        if self.is_dict and self.current_key is not None and self.current_key != "None":
            self.current_pattern[self.current_key] = self._get_current_functions()
        elif self.is_dict and self.current_key is None:
            self.current_pattern[None] = self._get_current_functions()

        # Update current key
        self.current_key = key

        # Update is_dict flag
        self.is_dict = key != "None"

        # Refresh UI
        self._refresh_function_list()
        self._notify_change() # Key switch implies pattern structure might change

    async def _convert_list_to_dict_pattern(self):
        """Converts the current list pattern to a dict pattern with a default key."""
        if not self.is_dict:
            self.current_pattern = {None: self.current_pattern} # Clause 234
            self.is_dict = True
            self.current_key = None
            self._refresh_key_selector()
            self._refresh_function_list()
            self._notify_change()

    async def _add_key(self):
        """
        Add a new dictionary key.

        🔒 Clause 234: Pattern Type Conversion Requires Structural Truth
        When converting from list to dict, use None as the key to indicate
        an unnamed structural group, preserving semantic parity.
        """
        # Prompt for key name if adding to existing dict
        if self.is_dict:
            new_key = f"key{len(self.current_pattern) + 1}"
            self.current_pattern[new_key] = []
            await self._switch_key(new_key)
        else:
            # Convert from list to dict using None as key (Clause 234)
            self.current_pattern = {None: self.current_pattern}
            self.is_dict = True
            self.current_key = None

        # Refresh UI
        self._refresh_key_selector()

    async def _remove_key(self):
        """
        Remove the current dictionary key.

        🔒 Clause 234: If only None key remains, convert back to flat list
        to maintain structural truth.
        """
        if not self.is_dict or self.current_key == "None":
            return

        # Remove the key
        if self.current_key in self.current_pattern:
            del self.current_pattern[self.current_key]

        # If no keys left, convert to list
        if not self.current_pattern:
            self.current_pattern = []
            self.is_dict = False
            self.current_key = "None"
        # If only None key remains, convert back to flat list (Clause 234)
        elif list(self.current_pattern.keys()) == [None]:
            self.current_pattern = self.current_pattern[None]
            self.is_dict = False
            self.current_key = "None"
        else:
            # Select another key
            self.current_key = next(iter(self.current_pattern))

        # Refresh UI
        self._refresh_key_selector()
        self._refresh_function_list()

    async def _edit_in_vim(self):
        """
        Edit the pattern in Vim with secure validation.

        🔒 Clause 92: Structural Validation First
        This implementation validates the edited file contains only a valid pattern
        assignment before evaluating it, preventing arbitrary code execution.
        """
        initial_content = f"pattern = {repr(self.current_pattern)}"

        success, new_pattern, error_message = await self.external_editor_service.edit_pattern_in_external_editor(initial_content)

        if success:
            self.current_pattern = new_pattern
            self._refresh_key_selector()
            self._refresh_function_list()
            self._notify_change()
        else:
            # Error message is already shown by the service, or logged by it.
            # The _show_error method in this class might be redundant if the service handles UI feedback.
            pass

    def _show_error(self, message: str):
        """Show an error message to the user using the TUI utility."""
        # Ensure this is run in the app's event loop if called from non-async context
        # or if show_error_dialog itself needs to schedule something.
        # Since show_error_dialog is async, we need to create a background task.
        get_app().create_background_task(
            show_error_dialog(title="Error", message=message, app_state=self.state)
        )

    async def _update_function(self, index, func):
        """
        Update a function in the pattern.

        Args:
            index: The index of the function to update
            func: The new function
        """
        # Skip if func is None (header entry)
        if func is None:
            return

        functions = self._get_current_functions()

        if 0 <= index < len(functions):
            # Extract existing kwargs
            _, kwargs = self._extract_func_and_kwargs(functions[index])

            # Update function with existing kwargs
            functions[index] = (func, kwargs)

            # Update pattern
            self._update_pattern_functions(functions)

            # Refresh function list
            self._refresh_function_list()

    def _update_pattern_functions(self, functions):
        """
        Update the functions in the current pattern.

        🔒 Clause 234: Support None as a valid group key for unnamed structural groups.

        Args:
            functions: The new functions list
        """
        if self.is_dict:
            if self.current_key is None:
                # Update the None key for unnamed structural groups
                self.current_pattern[None] = functions
            elif self.current_key != "None":
                # Update a named key
                self.current_pattern[self.current_key] = functions
        else:
            # Update flat list pattern
            self.current_pattern = functions

    def _create_parameter_editor(self, func, kwargs, func_index):
        """Create a parameter editor for a function."""
        if not func:
            return Label("Select a function to edit parameters")

        # Get function parameters through introspection
        params = self._get_function_parameters(func)

        # Create parameter fields
        param_fields = []

        for param in params:
            name = param['name']
            default = param['default']
            current_value = kwargs.get(name, default)
            required = param['required']
            is_special = param.get('is_special', False)

            # Create parameter field
            param_fields.append(self._create_parameter_field(
                name, default, current_value, required, is_special, func_index
            ))

        # Create "Reset All" button
        reset_all_button = SafeButton("Reset All",
            handler=lambda: get_app().create_background_task(self._reset_all_parameters(func_index))
        )
        param_fields.append(reset_all_button)

        return HSplit(param_fields)

    def _create_parameter_field(self, name, default, current_value, required, is_special, func_index):
        """Create a field for a single parameter."""
        # Create label with required indicator and special marker
        label_text = f"{name}{'*' if required else ''}"
        if is_special:
            label_text = f"{label_text} [S]"
        label = Label(label_text + ": ")

        # Create input field based on parameter type
        input_field = self._create_input_field(name, current_value, func_index)

        # Create reset button
        reset_button = SafeButton("Reset",
            handler=lambda: get_app().create_background_task(self._reset_parameter(name, default, func_index))
        )

        # Combine components
        return VSplit([
            Box(label, width=20),
            input_field,
            Box(reset_button, width=8)
        ])

    async def _move_function_up(self, index):
        """Move a function up in the list."""
        functions = self._get_current_functions()

        if 0 < index < len(functions):
            functions[index], functions[index-1] = functions[index-1], functions[index]
            self._update_pattern_functions(functions) # This calls _notify_change
            self._refresh_function_list()


    async def _move_function_down(self, index):
        """Move a function down in the list."""
        functions = self._get_current_functions()

        if 0 <= index < len(functions) - 1:
            functions[index], functions[index+1] = functions[index+1], functions[index]
            self._update_pattern_functions(functions) # This calls _notify_change
            self._refresh_function_list()


    async def _delete_function(self, index):
        """Delete a function from the pattern."""
        functions = self._get_current_functions()

        if 0 <= index < len(functions):
            del functions[index]
            self._update_pattern_functions(functions) # This calls _notify_change
            self._refresh_function_list()


    async def _add_function(self):
        """Add a new function to the pattern."""
        default_func = None
        # Try to find any function in the registry as a placeholder
        if FUNC_REGISTRY:
            for backend_funcs in FUNC_REGISTRY.values():
                if backend_funcs:
                    default_func = backend_funcs[0]
                    break

        # If no function found, we could add a pure None placeholder,
        # but it's better if the user is forced to select one.
        # For now, adding (None, {}) which the UI should handle.

        functions = self._get_current_functions()
        functions.append((default_func, {})) # Add with a default func if found, or None

        self._update_pattern_functions(functions) # This calls _notify_change
        self._refresh_function_list()

    # _save_pattern and _cancel_editing are removed.
    # The parent component (DualStepFuncEditorPane) will call get_pattern()
    # and handle the save/cancel logic for the entire FunctionStep.

    def _validate_pattern(self) -> bool:
        """
        Validate the current pattern using FuncStepContractValidator.

        Checks:
        1. All functions exist in registry
        2. All functions have memory type declarations
        3. Memory types are consistent
        """
        try:
            # Delegate full validation to FuncStepContractValidator
            FuncStepContractValidator.validate_function_pattern(
                self.current_pattern, "Function Pattern Editor"
            )
            return True
        except ValueError as e: # Catch specific validation errors from the validator
            self._show_error(f"Pattern validation failed: {str(e)}")
            return False
        except Exception as e: # Catch any other unexpected errors during validation
            self._show_error(f"Unexpected error during pattern validation: {str(e)}")
            return False

    async def _load_func_pattern_from_file_handler(self):
        """Handles loading a function pattern from a .func file."""
        file_path_str = await prompt_for_path_dialog(
            title="Load .func Pattern File",
            prompt_message="Enter path to .func pattern file:",
            app_state=self.state # Assuming self.state is accessible and correct
        )

        if not file_path_str:
            logger.info("Load .func pattern operation cancelled.")
            return

        file_path = Path(file_path_str)
        if not file_path.exists() or not file_path.is_file():
            await show_error_dialog("Load Error", f"File not found or is not a file: {file_path}", app_state=self.state)
            return

        try:
            with open(file_path, "rb") as f:
                loaded_pattern = pickle.load(f)

            # Basic validation: should be a list or dict
            if not isinstance(loaded_pattern, (list, dict)):
                await show_error_dialog("Load Error", "File does not contain a valid list or dict pattern.", app_state=self.state)
                return

            self.current_pattern = self._clone_pattern(loaded_pattern) # Use cloned, validated pattern
            self.original_pattern = self._clone_pattern(loaded_pattern) # Update original to prevent immediate "changed" state

            # Update internal state based on loaded pattern type
            self.is_dict = isinstance(self.current_pattern, dict)
            self.current_key = None if not self.is_dict else next(iter(self.current_pattern), None)

            self._refresh_key_selector()
            self._refresh_function_list()
            self._notify_change() # Notify parent of structural change
            logger.info(f"Successfully loaded function pattern from {file_path}")

        except pickle.UnpicklingError as e:
            logger.error(f"Error unpickling .func pattern from {file_path}: {e}", exc_info=True)
            await show_error_dialog("Load Error", f"Error unpickling file: {e}", app_state=self.state)
        except Exception as e:
            logger.error(f"Failed to load .func pattern from {file_path}: {e}", exc_info=True)
            await show_error_dialog("Load Error", f"Failed to load pattern: {e}", app_state=self.state)

    async def _save_func_pattern_as_file_handler(self):
        """Handles saving the current function pattern to a .func file."""
        file_path_str = await prompt_for_path_dialog(
            title="Save .func Pattern As",
            prompt_message="Enter path to save .func pattern file:",
            app_state=self.state,
            initial_value="pattern.func" # Suggest a default filename
        )

        if not file_path_str:
            logger.info("Save .func pattern As operation cancelled.")
            return

        file_path = Path(file_path_str)

        try:
            with open(file_path, "wb") as f:
                pickle.dump(self.current_pattern, f)
            logger.info(f"Successfully saved function pattern to {file_path}")
            # Optionally, show a success message dialog
            # await show_message_dialog("Success", f"Pattern saved to {file_path}", app_state=self.state)
        except pickle.PicklingError as e:
            logger.error(f"Error pickling .func pattern to {file_path}: {e}", exc_info=True)
            await show_error_dialog("Save Error", f"Error pickling pattern: {e}", app_state=self.state)
        except Exception as e:
            logger.error(f"Failed to save .func pattern to {file_path}: {e}", exc_info=True)
            await show_error_dialog("Save Error", f"Failed to save pattern: {e}", app_state=self.state)

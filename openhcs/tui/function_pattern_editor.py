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

import ast
import inspect
import os
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from prompt_toolkit.application import get_app
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.layout import HSplit, ScrollablePane, VSplit
from prompt_toolkit.widgets import (Box, Button, Dialog, Dropdown, Label,
                                    TextArea)

from openhcs.constants.constants import VALID_MEMORY_TYPES
from openhcs.core.pipeline.funcstep_contract_validator import \
    FuncStepContractValidator
# Import from func_registry instead of function_registry to avoid circular imports
from openhcs.processing.func_registry import FUNC_REGISTRY


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


def _validate_pattern_file(content: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Validate that a file contains only a valid pattern assignment.

    Args:
        content: The content of the file to validate

    Returns:
        Tuple of (is_valid, pattern, error_message)
    """
    try:
        # Parse the file
        tree = ast.parse(content)

        # Check that there's only one statement
        if len(tree.body) != 1:
            return False, None, "File must contain exactly one statement (pattern assignment)"

        # Check that the statement is an assignment
        stmt = tree.body[0]
        if not isinstance(stmt, ast.Assign):
            return False, None, "File must contain a pattern assignment"

        # Check that the assignment target is 'pattern'
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name) or stmt.targets[0].id != 'pattern':
            return False, None, "Assignment target must be 'pattern'"

        # Use ast.literal_eval to safely evaluate the pattern
        pattern_str = ast.unparse(stmt.value) if hasattr(ast, 'unparse') else content.strip().split('=', 1)[1].strip()
        try:
            pattern = ast.literal_eval(pattern_str)

            # Validate pattern structure using FuncStepContractValidator
            # This will raise ValueError if the pattern is invalid
            try:
                # Extract functions to validate pattern structure
                # We don't care about the actual functions, just that the structure is valid
                FuncStepContractValidator._extract_functions_from_pattern(
                    pattern, "Function Pattern Editor"
                )
                return True, pattern, None
            except ValueError as e:
                return False, None, f"Invalid pattern structure: {str(e)}"

        except (ValueError, SyntaxError) as e:
            # If literal_eval fails, the pattern contains non-literal expressions
            return False, None, f"Pattern must contain only literals: {str(e)}"

    except SyntaxError as e:
        return False, None, f"Syntax error: {str(e)}"
    except Exception as e:
        return False, None, f"Validation error: {str(e)}"


class GroupedDropdown(Dropdown):
    """Dropdown with support for disabled group headers."""

    def __init__(self, options, default=None):
        # Filter out headers for internal options list
        self.all_options = options
        self.selectable_options = [(value, text) for value, text in options if value != 'header']
        super().__init__(options=self.selectable_options, default=default)

    def _get_text_fragments(self):
        """Override to display all options including headers."""
        result = []
        for i, (value, text) in enumerate(self.all_options):
            if value == 'header':
                # Header style
                result.append(('class:dropdown.header', text))
                result.append(('', '\n'))
            elif value == self.current_value:
                # Selected item
                result.append(('class:dropdown-selected', f"> {text}"))
                result.append(('', '\n'))
            else:
                # Normal item
                result.append(('', f"  {text}"))
                result.append(('', '\n'))
        return result


class FunctionPatternEditor:
    """Function Pattern Editor leveraging OpenHCS's static reflection."""

    def __init__(self, state, step=None):
        """Initialize the Function Pattern Editor."""
        self.state = state
        self.step = step

        # Extract and clone pattern
        self.original_pattern = self._extract_pattern(step) if step else []
        self.current_pattern = self._clone_pattern(self.original_pattern)

        # Determine pattern type
        self.is_dict = isinstance(self.current_pattern, dict)
        self.current_key = None if not self.is_dict else next(iter(self.current_pattern), None)

        # Create UI components
        self.header = self._create_header()
        self.key_selector_container = HSplit([])
        self.function_list_container = ScrollablePane(HSplit([]))

        # Initial UI refresh
        self._refresh_key_selector()
        self._refresh_function_list()

        # Create container
        self.container = HSplit([
            self.header,
            self.key_selector_container,
            self.function_list_container
        ])

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

    def _create_header(self):
        """Create the editor header with title and save/load buttons."""
        title = Label(HTML("<b>Function Pattern Editor</b>"))
        save_button = Button(
            "Save",
            handler=lambda: get_app().create_background_task(self._save_pattern())
        )
        cancel_button = Button(
            "Cancel",
            handler=lambda: get_app().create_background_task(self._cancel_editing())
        )

        return VSplit([
            title,
            Box(save_button, padding=1),
            Box(cancel_button, padding=1)
        ])

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
        add_key_button = Button(
            "+",
            handler=lambda: get_app().create_background_task(self._add_key())
        )
        remove_key_button = Button(
            "-",
            handler=lambda: get_app().create_background_task(self._remove_key())
        )
        edit_in_vim_button = Button(
            "Edit in Vim",
            handler=lambda: get_app().create_background_task(self._edit_in_vim())
        )

        # Create container
        self.key_selector_container.children = [
            VSplit([
                Label("dict_keys: "),
                key_dropdown,
                add_key_button,
                remove_key_button,
                Box(edit_in_vim_button, padding=1)
            ])
        ]

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
        add_function_button = Button(
            "Add Function",
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
        move_up = Button(
            "↑",
            handler=lambda: get_app().create_background_task(self._move_function_up(index))
        )
        move_down = Button(
            "↓",
            handler=lambda: get_app().create_background_task(self._move_function_down(index))
        )
        delete_button = Button(
            "Delete",
            handler=lambda: get_app().create_background_task(self._delete_function(index))
        )

        # Parameter editor
        param_editor = self._create_parameter_editor(func, kwargs, index)

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

        # Get functions from registry grouped by backend
        functions_by_backend = {}
        for backend, funcs in FUNC_REGISTRY.items():
            for func in funcs:
                # Get function info using registry metadata
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

        # Create dropdown options with disabled headers
        dropdown_options = []
        for backend, funcs in sorted(functions_by_backend.items()):
            # Add backend header as disabled option
            dropdown_options.append(('header', HTML(f"<b>--- {backend} ---</b>")))
            # Add functions
            dropdown_options.extend([(func, name) for func, name in funcs])

        # Create dropdown
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
        try:
            # Create temporary file through FileManager
            temp_file_path = self.filemanager.create_temp_file(
                suffix='.py',
                backend=self.backend
            )

            # Write initial content
            self.filemanager.write_file(
                temp_file_path,
                f"pattern = {repr(self.current_pattern)}\n"
                "\n# IMPORTANT: This file must contain ONLY a 'pattern = ...' assignment.\n"
                "# Any other code will be rejected for security reasons.\n"
                "# Valid pattern structures:\n"
                "#  - A callable function\n"
                "#  - A tuple of (callable, kwargs) where kwargs is a dict\n"
                "#  - A list of valid pattern structures\n"
                "#  - A dict mapping keys to valid pattern structures\n",
                backend=self.backend
            )

            # Launch editor
            editor = os.environ.get('EDITOR', 'vim')
            subprocess.run([editor, temp_file_path])

            # Read back the file through FileManager
            content = self.filemanager.read_file(temp_file_path, backend=self.backend)

            # Validate and parse the pattern
            is_valid, pattern, error_message = _validate_pattern_file(content)

            if is_valid and pattern is not None:
                # Update pattern
                self.current_pattern = pattern
                self.is_dict = isinstance(self.current_pattern, dict)

                # 🔒 Clause 234: Handle None key for unnamed structural groups
                if self.is_dict and self.current_pattern:
                    # Check if None is a key in the pattern
                    if None in self.current_pattern and len(self.current_pattern) == 1:
                        # If only None key exists, use it as the current key
                        self.current_key = None
                    else:
                        # Otherwise use the first key
                        self.current_key = next(iter(self.current_pattern))
                else:
                    self.current_key = "None"

                # Refresh UI
                self._refresh_key_selector()
                self._refresh_function_list()
            else:
                # Show error message
                self._show_error(f"Invalid pattern: {error_message}")

        finally:
            # Clean up through FileManager
            if temp_file_path:
                self.filemanager.delete_file(temp_file_path, backend=self.backend)

    def _show_error(self, message: str):
        """Show an error message to the user."""
        from prompt_toolkit.layout.containers import HSplit
        from prompt_toolkit.widgets import Button, Label

        # Create error dialog
        error_dialog = Dialog(
            title="Error",
            body=HSplit([
                Label(message),
            ]),
            buttons=[
                Button("OK", handler=lambda: get_app().exit_dialog())
            ],
            width=80,
            modal=True
        )

        # Show dialog
        get_app().layout.focus(error_dialog)
        get_app().layout.container = error_dialog

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
        reset_all_button = Button(
            "Reset All",
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
        reset_button = Button(
            "Reset",
            handler=lambda: get_app().create_background_task(self._reset_parameter(name, default, func_index))
        )

        # Combine components
        return VSplit([
            Box(label, width=20),
            input_field,
            Box(reset_button, width=8)
        ])

    def _create_input_field(self, name, value, func_index):
        """Create an input field appropriate for the parameter type."""
        # Convert value to string representation
        value_str = str(value) if value is not None else ""

        # Create text area
        text_area = TextArea(
            text=value_str,
            multiline=False,
            height=1
        )

        # Set handler for value change
        def on_text_changed(buffer):
            get_app().create_background_task(self._update_parameter(name, buffer.text, func_index))

        text_area.buffer.on_text_changed += on_text_changed

        return text_area

    def _get_function_parameters(self, func) -> List[Dict]:
        """Get parameters for a function through introspection."""
        params = []

        # Get function signature
        sig = inspect.signature(func)

        # Extract parameters
        for name, param in sig.parameters.items():
            # Skip self/cls
            if name in ('self', 'cls'):
                continue

            # Get default value
            if param.default is not inspect.Parameter.empty:
                default = param.default
            else:
                default = None

            # Get parameter type
            param_type = param.annotation if param.annotation is not inspect.Parameter.empty else None

            # Check if special parameter
            is_special = hasattr(func, 'special_inputs') and name in getattr(func, 'special_inputs', [])

            # Add to parameters
            params.append({
                'name': name,
                'default': default,
                'type': param_type,
                'required': param.default is inspect.Parameter.empty,
                'is_special': is_special
            })

        return params

    async def _update_parameter(self, name, value_str, func_index):
        """Update a parameter value."""
        functions = self._get_current_functions()

        if 0 <= func_index < len(functions):
            # Extract function and kwargs
            func, kwargs = self._extract_func_and_kwargs(functions[func_index])

            # Convert string to appropriate type
            value = self._parse_parameter_value(value_str)

            # Update kwargs
            kwargs[name] = value

            # Update function with new kwargs
            functions[func_index] = (func, kwargs)

            # Update pattern
            self._update_pattern_functions(functions)

    def _parse_parameter_value(self, value_str):
        """Parse a parameter value from string."""
        # Handle special cases
        if value_str.lower() == 'none':
            return None
        elif value_str.lower() == 'true':
            return True
        elif value_str.lower() == 'false':
            return False

        # Try to parse as number
        try:
            # Try int first
            return int(value_str)
        except ValueError:
            try:
                # Then try float
                return float(value_str)
            except ValueError:
                # Fall back to string
                return value_str

    async def _reset_parameter(self, name, default, func_index):
        """Reset a parameter to its default value."""
        functions = self._get_current_functions()

        if 0 <= func_index < len(functions):
            # Extract function and kwargs
            func, kwargs = self._extract_func_and_kwargs(functions[func_index])

            # Reset parameter to default
            if default is not None:
                kwargs[name] = default
            elif name in kwargs:
                del kwargs[name]

            # Update function with new kwargs
            functions[func_index] = (func, kwargs)

            # Update pattern
            self._update_pattern_functions(functions)

            # Refresh function list
            self._refresh_function_list()

    async def _reset_all_parameters(self, func_index):
        """Reset all parameters to their default values."""
        functions = self._get_current_functions()

        if 0 <= func_index < len(functions):
            # Extract function
            func, _ = self._extract_func_and_kwargs(functions[func_index])

            # Create empty kwargs
            kwargs = {}

            # Update function with empty kwargs
            functions[func_index] = (func, kwargs)

            # Update pattern
            self._update_pattern_functions(functions)

            # Refresh function list
            self._refresh_function_list()

    async def _move_function_up(self, index):
        """Move a function up in the list."""
        functions = self._get_current_functions()

        if 0 < index < len(functions):
            functions[index], functions[index-1] = functions[index-1], functions[index]
            self._update_pattern_functions(functions)
            self._refresh_function_list()

    async def _move_function_down(self, index):
        """Move a function down in the list."""
        functions = self._get_current_functions()

        if 0 <= index < len(functions) - 1:
            functions[index], functions[index+1] = functions[index+1], functions[index]
            self._update_pattern_functions(functions)
            self._refresh_function_list()

    async def _delete_function(self, index):
        """Delete a function from the pattern."""
        functions = self._get_current_functions()

        if 0 <= index < len(functions):
            del functions[index]
            self._update_pattern_functions(functions)
            self._refresh_function_list()

    async def _add_function(self):
        """Add a new function to the pattern."""
        # Get a default function from the registry
        default_func = None
        for backend, funcs in FUNC_REGISTRY.items():
            if funcs:
                default_func = funcs[0]
                break

        if not default_func:
            return

        # Add function to pattern
        functions = self._get_current_functions()
        functions.append((default_func, {}))

        # Update pattern
        self._update_pattern_functions(functions)

        # Refresh function list
        self._refresh_function_list()

    async def _save_pattern(self):
        """Save the pattern and exit the editor."""
        # Validate pattern
        if not self._validate_pattern():
            return

        # Update step with new pattern
        if self.step:
            self.step['func'] = self.current_pattern

        # Notify state manager
        self.state.notify('pattern_saved', {
            'step': self.step,
            'pattern': self.current_pattern
        })

    async def _cancel_editing(self):
        """Cancel editing and exit the editor."""
        self.state.notify('editing_cancelled')

    def _validate_pattern(self) -> bool:
        """
        Validate the current pattern using FuncStepContractValidator.

        Checks:
        1. All functions exist in registry
        2. All functions have memory type declarations
        3. Memory types are consistent
        """
        try:
            # Use FuncStepContractValidator to extract and validate functions
            functions = FuncStepContractValidator._extract_functions_from_pattern(
                self.current_pattern, "Function Pattern Editor"
            )

            # Check that all functions have memory type declarations
            for func in functions:
                if not hasattr(func, 'input_memory_type') or not hasattr(func, 'output_memory_type'):
                    self._show_error(f"Function {func.__name__} is missing memory type declarations")
                    return False

                # Validate memory types against known valid types
                if (func.input_memory_type not in VALID_MEMORY_TYPES or
                    func.output_memory_type not in VALID_MEMORY_TYPES):
                    self._show_error(f"Function {func.__name__} has invalid memory types")
                    return False

            return True
        except Exception as e:
            self._show_error(f"Pattern validation failed: {str(e)}")
            return False

# plan_05a_function_pattern_editor_core.md
## Component: Function Pattern Editor Core

### Objective
Implement the core structure of the Function Pattern Editor leveraging OpenHCS's static reflection.

### Plan
1. Leverage `FuncStepContractValidator` for pattern validation and function extraction
2. Use prompt-toolkit's dynamic containers for UI components
3. Implement proper asyncio integration with `get_app().create_background_task()`
4. Provide Vim integration for advanced pattern editing

### Implementation Draft

```python
import os
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

from prompt_toolkit.layout import HSplit, VSplit, ScrollablePane
from prompt_toolkit.widgets import Button, Label, Box, Dropdown
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.application import get_app

from ezstitcher.processing.function_registry import FUNC_REGISTRY, get_function_info
from ezstitcher.core.pipeline.funcstep_contract_validator import FuncStepContractValidator
from ezstitcher.constants.constants import VALID_MEMORY_TYPES


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

        # Create handler for key selection
        def on_key_change(key):
            get_app().create_background_task(self._switch_key(key))

        # Attach handler to dropdown's on_change event
        key_dropdown.on_change = on_key_change

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
        # This will be implemented in plan_05b
        if func:
            # Use get_function_info to extract metadata
            try:
                info = get_function_info(func)
                return Label(f"Function {index+1}: {info['name']} ({info['backend']})")
            except ValueError:
                pass
        return Label(f"Function {index+1}: {func.__name__ if func else 'None'}")

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

        # Update is_dict flag - use special sentinel value "None" to indicate list mode
        # 🔒 Clause 234: Properly distinguish between string "None" sentinel and actual None key
        self.is_dict = key != "None"  # List mode uses "None" string as sentinel
        
        # Ensure we're not conflating None and "None" - explicit check for dictionary mode
        if key is None:
            self.is_dict = True  # None is a valid dictionary key, not list mode

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
        """Edit the pattern in Vim using AST validation."""
        import ast
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w+', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(f"pattern = {repr(self.current_pattern)}\n")

        try:
            # Launch editor
            editor = os.environ.get('EDITOR', 'vim')
            subprocess.run([editor, temp_file_path])

            # Read back the file
            with open(temp_file_path, 'r') as temp_file:
                content = temp_file.read()

            # Parse and validate using AST
            tree = ast.parse(content)
            
            # Validate only contains a single pattern assignment
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
                raise ValueError("File must contain exactly one pattern assignment")
                
            # Extract the pattern value safely
            pattern_node = tree.body[0].value
            pattern = ast.literal_eval(pattern_node)
            
            # Update pattern if valid
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

        finally:
            # Clean up
            os.unlink(temp_file_path)

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
            # Use public validation API
            validator = FuncStepContractValidator()
            validation_result = validator.validate_pattern_structure(
                self.current_pattern,
                context="Function Pattern Editor"
            )

            if not validation_result.valid:
                return False

            # Additional memory type checks
            for func in validation_result.functions:
                if (func.input_memory_type not in VALID_MEMORY_TYPES or
                    func.output_memory_type not in VALID_MEMORY_TYPES):
                    return False

            return True
        except Exception:
            return False

    async def _add_function(self):
        """Add a new function to the pattern."""
        # Get a default function from the registry deterministically
        default_func = None
        # Sort backends and functions for deterministic selection
        sorted_backends = sorted(FUNC_REGISTRY.items(), key=lambda x: x[0])
        for backend, funcs in sorted_backends:
            if funcs:
                # Sort functions by name for determinism
                sorted_funcs = sorted(funcs, key=lambda f: getattr(f, '__name__', str(f)))
                default_func = sorted_funcs[0]
                break

        if not default_func:
            return

        # Add function to pattern with proper cloning
        functions = self._get_current_functions().copy()  # Clone before mutation
        functions.append((default_func, {}))

        # Update pattern through proper method to ensure immutability
        self._update_pattern_functions(functions)

        # Refresh function list
        self._refresh_function_list()

    def _update_pattern_functions(self, functions):
        """
        Update the pattern's functions while maintaining immutability.
        
        Args:
            functions: The new functions list to set
        """
        # Create new container to maintain immutability
        new_functions = functions.copy()
        
        # Update pattern
        if self.is_dict:
            if self.current_key is None:
                # Update the None key for unnamed structural groups
                self.current_pattern = {None: new_functions}
            elif self.current_key != "None":
                # Update a named key
                self.current_pattern = {self.current_key: new_functions}
        else:
            # Update flat list pattern
            self.current_pattern = new_functions
```
        else:
            # Update flat list pattern
            self.current_pattern = new_functions
```

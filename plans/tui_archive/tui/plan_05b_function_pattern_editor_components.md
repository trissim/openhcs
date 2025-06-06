# plan_05b_function_pattern_editor_components.md
## Component: Function Pattern Editor Components

### Objective
Implement the individual components of the Function Pattern Editor leveraging OpenHCS's static introspection.

### Plan
1. Use `FuncStepContractValidator` for function extraction and validation
2. Leverage function registry metadata for dropdown population
3. Use `inspect.signature()` for parameter field generation
4. Implement proper asyncio integration with prompt-toolkit
5. Support None key for unnamed structural groups (Clause 234)
6. Create grouped dropdowns with disabled headers for better UX

### Implementation Draft

```python
def _create_function_item(self, index, func, kwargs):
    """Create a UI component for a function item."""
    from prompt_toolkit.widgets import Frame

    # Get function info from registry
    func_info = get_function_info(func) if func else {"name": "None", "backend": "", "is_valid": True, "validation_errors": []}
    
    # Create title with validation status indicator
    title = f"Function {index+1}: {func_info['name']} ({func_info['backend']})"
    if not func_info.get('is_valid', True):
        # Add validation error indicator to title
        title += " [⚠️ Invalid]"

    # Function header with dropdown
    func_dropdown = self._create_function_dropdown(index, func)

    # Function controls
    # Create closures to capture current index value, not reference
    def create_move_up_handler(idx):
        return lambda: get_app().create_background_task(self._move_function_up(idx))
    
    def create_move_down_handler(idx):
        return lambda: get_app().create_background_task(self._move_function_down(idx))
    
    def create_delete_handler(idx):
        return lambda: get_app().create_background_task(self._delete_function(idx))
    
    move_up = Button(
        "↑",
        handler=create_move_up_handler(index)
    )
    move_down = Button(
        "↓",
        handler=create_move_down_handler(index)
    )
    delete_button = Button(
        "Delete",
        handler=create_delete_handler(index)
    )

    # Parameter editor
    param_editor = self._create_parameter_editor(func, kwargs, index)

    # Create components list
    components = [
        # Function header
        VSplit([
            func_dropdown,
            Box(move_up, width=3),
            Box(move_down, width=3),
            Box(delete_button, width=8)
        ]),
        # Parameter editor
        param_editor
    ]
    
    # Add validation errors if present
    if not func_info.get('is_valid', True) and func_info.get('validation_errors'):
        error_text = "\n".join([f"⚠️ {err}" for err in func_info['validation_errors']])
        error_label = Label(f"Validation Errors:\n{error_text}")
        components.append(Box(error_label, style="bg:#662222 fg:#ffffff"))
    
    # Combine components
    return Frame(
        HSplit(components),
        title=title
    )

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
    from prompt_toolkit.widgets import Dropdown
    from prompt_toolkit.formatted_text import HTML

    # Get functions from registry grouped by backend
    functions_by_backend = {}
    for backend, funcs in FUNC_REGISTRY.items():
        for func in funcs:
            # Get function info using registry metadata
            # Get function info - no try/except needed with new error handling
            info = get_function_info(func)
            
            if backend not in functions_by_backend:
                functions_by_backend[backend] = []
                
            # Include memory type in display name
            # Add validation status indicator for invalid functions
            if info.get('is_valid', True):
                display_name = f"{func.__name__} ({info['input_memory_type']} → {info['output_memory_type']})"
            else:
                display_name = f"{func.__name__} [⚠️]"
                
            functions_by_backend[backend].append((func, display_name))

    # Create dropdown options with disabled headers
    dropdown_options = []
    for backend, funcs in sorted(functions_by_backend.items()):
        # Add backend header as disabled option
        dropdown_options.append(('header', HTML(f"<b>--- {backend} ---</b>")))
        # Add functions
        dropdown_options.extend([(func, name) for func, name in funcs])

    # Create custom dropdown with support for disabled options
    class GroupedDropdown(Dropdown):
        """Dropdown with support for disabled group headers."""

        def __init__(self, options, default=None):
            # Store all options including headers
            self.all_options = options.copy()
            
            # Filter out headers for internal options list
            self.selectable_options = [(value, text) for value, text in options
                                      if not (isinstance(value, str) and value == 'header')]
            
            # Ensure default is in selectable options
            if default is not None and default not in [value for value, _ in self.selectable_options]:
                default = None
                
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

    # Create dropdown
    dropdown = GroupedDropdown(
        options=dropdown_options,
        default=current_func
    )

    # Set handler for selection change
    def on_selection_change(func):
        # Skip header entries - properly check tuple structure
        if not (isinstance(func, str) and func == 'header'):
            get_app().create_background_task(self._update_function(index, func))

    dropdown.on_change = on_selection_change

    return dropdown

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

    # Create "Reset All" button with proper closure
    def create_reset_all_handler(idx):
        return lambda: get_app().create_background_task(self._reset_all_parameters(idx))
        
    reset_all_button = Button(
        "Reset All",
        handler=create_reset_all_handler(func_index)
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

    # Create reset button with proper closure
    def create_reset_handler(param_name, param_default, idx):
        return lambda: get_app().create_background_task(
            self._reset_parameter(param_name, param_default, idx)
        )
        
    reset_button = Button(
        "Reset",
        handler=create_reset_handler(name, default, func_index)
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
    """
    Parse a parameter value from string without type inference.
    
    🔒 Clause 231: Deferred-Default Enforcement
    🔒 Clause 88: No Inferred Capabilities
    
    This method preserves the string representation without attempting to
    infer types. Type conversion happens at function execution time based
    on explicit parameter type annotations.
    
    Args:
        value_str: The string value to parse
        
    Returns:
        The parsed value as a string, preserving the original representation
    """
    # Only handle explicit None, True, False values
    # These are special cases that have well-defined Python literals
    if value_str.lower() == 'none':
        return None
    elif value_str.lower() == 'true':
        return True
    elif value_str.lower() == 'false':
        return False
    
    # For all other values, preserve as string without type inference
    # This avoids violating Clause 88 (No Inferred Capabilities)
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

def _update_pattern_functions(self, functions):
    """
    Update the functions in the current pattern.

    🔒 Clause 234: Support None as a valid group key for unnamed structural groups.
    🔒 Clause 246: Statelessness Mandate - avoid direct mutation of state

    Args:
        functions: The new functions list
        
    Returns:
        None - updates are applied to a new pattern instance
    """
    # Create a deep copy of the current pattern to maintain immutability
    if self.is_dict:
        # Create a new dictionary with the same keys
        new_pattern = {}
        for k, v in self.current_pattern.items():
            if k == self.current_key:
                # Update the current key with the new functions
                new_pattern[k] = functions.copy()
            else:
                # Copy other keys unchanged
                new_pattern[k] = v.copy() if isinstance(v, list) else v
                
        # Special case for None key if it's the current key
        if self.current_key is None:
            new_pattern[None] = functions.copy()
            
        # Replace the current pattern with the new one
        self.current_pattern = new_pattern
    else:
        # For flat list patterns, create a new list
        self.current_pattern = functions.copy()

def get_function_info(func):
    """
    Get metadata for a function from its decorators.

    Leverages the function registry and decorator metadata to extract
    information about the function's backend, memory types, and special parameters.
    
    🔒 Clause 241: Time-Aware Enforcement - provide grace path for UI editing
    
    Args:
        func: The function to get metadata for

    Returns:
        Dictionary of function metadata with optional error information
        The dictionary always contains 'name' and 'is_valid' keys
    """
    from ezstitcher.constants.constants import VALID_MEMORY_TYPES

    # Initialize result with validity flag
    info = {
        'is_valid': True,
        'validation_errors': []
    }
    
    # Check if callable
    if not callable(func):
        return {
            'name': str(func),
            'is_valid': False,
            'validation_errors': [f"Not a callable: {func}"],
            'backend': 'unknown'
        }

    # Get basic function metadata
    info.update({
        'name': func.__name__,
        'backend': getattr(func, 'backend', 'unknown'),
        'input_memory_type': getattr(func, 'input_memory_type', None),
        'output_memory_type': getattr(func, 'output_memory_type', None),
        'special_inputs': getattr(func, 'special_inputs', []),
        'special_outputs': getattr(func, 'special_outputs', [])
    })

    # Validate memory types but don't raise exceptions
    # Instead, collect validation errors and mark as invalid
    if info['input_memory_type'] not in VALID_MEMORY_TYPES:
        info['validation_errors'].append(
            f"Invalid input memory type: {info['input_memory_type']}"
        )
        info['is_valid'] = False

    if info['output_memory_type'] not in VALID_MEMORY_TYPES:
        info['validation_errors'].append(
            f"Invalid output memory type: {info['output_memory_type']}"
        )
        info['is_valid'] = False

    return info
```

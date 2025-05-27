I want to start makign a tui in prompt tool kit now. It has built in vim key bdinings and vim oriented feature set.

reserved horizontal thin bar  at top for menu titles
rseerved horizontal thin bar at bottom for info and loading bar hybrid

there are three column and one row (three zones side by side 1,2 ,3 .

the right (3) most has a static wdith and displays text and button
the contents may switch to a different menu when the func pattern editor is actiated (shown later)

the two other panes are
left (1) most area scrollable selectable with mouse list showing added plates and highlighting them.
middle (2)  shows steps of all pipelines queued for the plate.
they are trated as as single one by just listing all the steps sequentially.
Obtain step name from static introspectoin.

Launch menu:
____________________________________________________________________________________________________________
|_______________1_plate manager______________________|__________2_Step_viewer____________|_____3_Menu______|
|o| ^/v 1: axotomy_n1_03-02-24 | (...)/nvme_usb/    |o| [edit] ^/v 1: pos_gen_pattern    | OpenHCS         |
|!| ^/v 2: axotomy_n2_03-03-24 | (...)/nvme_usb/    |o| [edit] ^/v 2: enhance + assemble | [ add ]         |
|?| ^/v 3: axotomy_n3_03-04-24 | (...)/nvme_usb/    |o| [edit] ^/v 3: trace analysis	 | [ edit ]        |
| |                                                 | |                                  | [ pre-compile ] |
| |                                                 | |                                  | [ compile ]     |
| |                                                 | |                                  | [ run ]         |
| |                                                 | |                                  | [ save ]        |
| |                                                 | |                                  | [ test ]        |
| |                                                 | |                                  | [ settings ]    |
| |                                                 | |                                  |                 |
___________________________________________________ |_r_________________________________ |_________________|
___________________________________________________ _______________________________________________________|

When edit on a step, selected two elements change
- the left pane (1) gets replaced by a another pane: the func pattern editor.
its a a gui for the dict of lists of tuples of func kwarg pattern.

_____________________________________________________________________________________________
| Func Pattern editor  [load] [save]            |
________________________________________________|____________________________________________
| dict_keys: |None|V| +/-  |  [edit in vim] ?   | <- you can have mor than one key, making he list of funcs change.
________________________________________________|____________________________________________
|^| Func 1: |percentile stack normalize|v|      | <- drop down menu generated from using the func register and static analysis of func name definition)
|X| --------------------------------------------|--------------------------------------------
|X|  move  [reset] percentile_low:  0.1 ...     | <- these two kwargs with editabel fields are autogen from the func definition)
|X|   /\   [reset]  percentile_high: 99.9 ...   |
|X|   \/   [add]                                |
|X|        [delete]                             |
|X| ____________________________________________|______________________________________________
|X| Func 2: |n2v2|V|                            |
|X|---------------------------------------------|----------------------------------------------
|X|  	    [reset] random_seed: 42 ...           |<- so are these
|X|   move  [reset] device: cuda ...            |
|X|    /\   [reset] blindspot_prob: 0.05 ...    |
|X|    \/   [reset] max_epochs: 10 ...          |
|X|    	    [reset] batch_size: 4 ...           |
|X|    	    [reset] patch_size: 64 ...          |
|X|    	    [reset] learning_rate: 1e-4 ...     |
|X|    	    [reset] save_model_path: ...        |
|X|    	    [reset all]                         |
|V|    	    [add]                               |
| |    	    [delete]	                          |
| | ____________________________________________|______________________________________________
| | Func 3: |3d_deconv|V|                       |
| |   	    [reset] random_seed:  42 ...        |
| |    move [reset] device: cuda  ...           |
| |     /\  [reset] blindspot_prob: 0.05 ...    |
| | vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv|vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
|_| ____________________________________________|______________________________________________


## Implementation Notes

### Key Components and Requirements

1. **Layout & Panes**
   - Create a three-pane layout with a fixed top menu bar and bottom status bar.
   - Panes 1 (Plate Manager), 2 (Step Viewer), and 3 (Action Menu) must be independently resizable (clamped to ≥20 cols/5 rows) and support both mouse and Vim bindings.

2. **Plate Manager Pane**
   - Populate a scrollable list of *filesystem* paths (no VFS URIs).
   - Emit a selection event into our `ProcessingContext` whenever the user highlights a plate.

3. **Step Viewer Pane**
   - Render the flat list of all `FuncStep` declarations for the selected plate, with a thin visual separator (─) between pipelines.
   - Support keyboard navigation and click selection.

4. **Action Menu Pane**
   - Display buttons for `[add]`, `[pre-compile]`, `[compile]`, `[run]`, `[save]`, `[test]`, and `[settings]`.
   - Wire each to stubbed callbacks in `tui.py` ready for later implementation.

5. **Compile Enforcement & Save Warning**
   - Hook `[pre-compile]`/`[compile]` to the pipeline-compile API.
   - If compile fails, block `[save]` and show an inline banner above the Action Menu.

6. **Bottom Status Bar**
   - Show live logger output (toggleable via `y` in Settings).
   - Clicking the status bar expands a bottom drawer overlay with full log history.

7. **Vim-Edit Integration**
   - Implement a "Edit in Vim" command: dump current FuncStep pattern dict to a temp `.py` in the project workspace, launch `$EDITOR`, then on exit auto-reload if changed.

### Technical Considerations

- Use modular, testable functions (e.g., `build_layout()`, `render_plate_list()`, `compile_pipeline()`, etc.)
- Include docstrings referencing relevant OpenHCS clauses
- Memory types are managed automatically by pre-execution resolving through the planner and validators
- Special dependencies should only appear under kwargs if the function is decorated with special_in or special_out
- The pre-compile and compile buttons trigger validation before execution

### Function Pattern Editor Implementation

The Function Pattern Editor (lines 37-68 in the mockup) is a critical component that replaces the left pane when editing a step. It should:

1. **Dictionary Key Selection**
   - Provide a dropdown for selecting dictionary keys (e.g., `None` in the mockup)
   - `None` represents a simple list of functions rather than a dictionary
   - When adding a key while `None` is selected, convert from list to dict with 1 key and 1 value
   - Adding additional keys creates new empty function lists for each key
   - Include add/remove buttons for managing keys
   - Offer a "edit in vim" option for advanced editing

2. **Function Selection and Management**
   - Display a scrollable, numbered list of functions (e.g., "Func 1:", "Func 2:")
   - For each function, provide:
     - A dropdown menu populated from the function registry in the processing folder
     - Functions must be sorted by their decorators (memory type, backend)
     - Only decorated functions should be available (enforced at pipeline compilation)
     - Move up/down controls for reordering functions
     - Reset, add, and delete buttons

3. **Parameter Configuration**
   - Auto-generate parameter fields based on function signatures using static introspection
   - For each parameter:
     - Display the parameter name and current value
     - Provide appropriate input controls based on parameter type
     - Include a reset button to restore default values from function definitions
   - Support for adding custom parameters
   - Provide a "reset all" option for restoring all parameters to defaults
   - Special parameters (decorated with special_in/special_out) should be listed with the kwargs

4. **Visual Organization**
   - Clear visual separation between functions
   - Collapsible parameter sections for complex functions
   - Scrollable interface for handling functions with many parameters
   - Visual indicators for required vs. optional parameters

5. **Validation and Execution Flow**
   - Pre-compile button initializes the orchestrator for a plate
   - Compile button builds the context for the pipeline and validates it
   - Run button is only enabled if compilation passes
   - Validation should only occur when clicking the compile button
   - The TUI leverages OpenHCS's declarative nature, static reflectivity, and contract purity

6. **Integration Points**
   - Load/save buttons at the top for pattern persistence
   - Dynamic updating of the interface when function selection changes
   - Validation of parameter values against function requirements
   - Preview capability to see the effect of parameter changes

### Development Approach and Best Practices

1. **Modular Architecture**
   - Implement the TUI using a component-based architecture
   - Separate UI components from business logic
   - Use event-driven communication between components
   - Follow the Model-View-Controller pattern where appropriate

2. **Integration with OpenHCS Core**
   - Use the function registry from the processing folder for populating function dropdowns
   - Leverage static introspection of function signatures for parameter fields
   - Connect to the orchestrator for plate initialization and pipeline compilation
   - Respect OpenHCS's declarative principles and contract enforcement

3. **Testing Strategy**
   - Write unit tests for individual components
   - Create integration tests for component interactions
   - Implement end-to-end tests for critical workflows
   - Use mock objects to isolate UI components from backend dependencies

4. **User Experience Considerations**
   - Provide clear visual feedback for actions
   - Implement consistent keyboard shortcuts across the interface
   - Ensure all functionality is accessible via both keyboard and mouse
   - Display helpful error messages when validation fails
   - Include tooltips or help text for complex features

5. **Performance Optimization**
   - Lazy-load components and data when possible
   - Implement efficient rendering for large lists
   - Cache function registry data to avoid repeated introspection
   - Use background threads for long-running operations to keep the UI responsive

6. **Documentation**
   - Document all components with clear docstrings
   - Include references to relevant OpenHCS clauses in documentation
   - Provide usage examples for complex components
   - Create user documentation explaining the TUI workflow

By following these guidelines and implementing the components described above, the TUI will provide a powerful, user-friendly interface to OpenHCS that respects its architectural principles while making it accessible to users.

### Implementation Examples

Here are concrete examples of how to implement key aspects of the TUI by leveraging OpenHCS's existing code:

#### 1. Function Registry Integration

```python
from ezstitcher.processing.function_registry import FUNC_REGISTRY
from prompt_toolkit.widgets import RadioList, Dialog

def build_function_dropdown():
    """Build a dropdown of available functions from the registry."""
    # Get all functions from the registry, grouped by backend
    functions = []
    for backend, funcs in FUNC_REGISTRY.items():
        for func in funcs:
            # Get function info
            name = func.__name__
            doc = func.__doc__ or "No documentation available"
            # Add to list with display name and actual function
            functions.append((func, f"{name} ({backend})"))

    # Create a RadioList for function selection
    return RadioList(values=functions)

def get_function_parameters(func):
    """Extract parameters from a function for the parameter editor."""
    import inspect

    # Get function signature
    sig = inspect.signature(func)

    # Extract parameters and their default values
    parameters = []
    for name, param in sig.parameters.items():
        # Skip self/cls for methods
        if name in ('self', 'cls'):
            continue

        # Get default value if available
        if param.default is not inspect.Parameter.empty:
            default = param.default
        else:
            default = None

        # Get parameter type if available
        param_type = param.annotation if param.annotation is not inspect.Parameter.empty else None

        # Add to parameters list
        parameters.append({
            'name': name,
            'default': default,
            'type': param_type,
            'required': param.default is inspect.Parameter.empty
        })

    return parameters
```

#### 2. Function Pattern Editor

```python
from prompt_toolkit.layout import HSplit, VSplit
from prompt_toolkit.widgets import Button, TextArea, Label, Box
from prompt_toolkit.key_binding import KeyBindings

class FunctionPatternEditor:
    def __init__(self, pattern=None):
        """Initialize the Function Pattern Editor with an optional existing pattern."""
        self.pattern = pattern or []  # Default to empty list
        self.is_dict = isinstance(pattern, dict)
        self.current_key = None if self.is_dict else "None"
        self.functions = []

        # Build the UI components
        self.build_ui()

    def build_ui(self):
        """Build the UI components for the Function Pattern Editor."""
        # Dictionary key selector (if pattern is a dict)
        self.key_dropdown = self._build_key_dropdown()

        # Function list
        self.function_list = self._build_function_list()

        # Parameter editor
        self.parameter_editor = self._build_parameter_editor()

        # Combine components
        self.container = HSplit([
            Label("Function Pattern Editor"),
            VSplit([
                Box(self.key_dropdown, padding=1),
                Button("Add Key", handler=self.add_key),
                Button("Remove Key", handler=self.remove_key)
            ]),
            HSplit([
                self.function_list,
                self.parameter_editor
            ])
        ])

    def _build_key_dropdown(self):
        """Build the dictionary key dropdown."""
        keys = list(self.pattern.keys()) if self.is_dict else ["None"]

        # Create a RadioList for key selection
        from prompt_toolkit.widgets import RadioList
        radio_list = RadioList(
            values=[(k, k) for k in keys],
            default=self.current_key
        )

        # Add handler for key selection
        def on_key_change(key):
            self.current_key = key
            self._update_function_list()

        radio_list.handler = on_key_change
        return radio_list

    def _build_function_list(self):
        """Build the function list for the current key."""
        # Get functions for the current key
        if self.is_dict and self.current_key in self.pattern:
            funcs = self.pattern[self.current_key]
        elif not self.is_dict:
            funcs = self.pattern
        else:
            funcs = []

        # Convert to list if not already
        if not isinstance(funcs, list):
            funcs = [funcs]

        # Store functions
        self.functions = funcs

        # Create function list UI
        from prompt_toolkit.widgets import Frame
        function_frames = []

        for i, func in enumerate(self.functions):
            # Extract function and args
            if isinstance(func, tuple) and len(func) == 2 and callable(func[0]):
                func_obj, kwargs = func
            elif callable(func):
                func_obj, kwargs = func, {}
            else:
                continue

            # Create function frame
            function_frames.append(
                Frame(
                    HSplit([
                        Label(f"Function {i+1}: {func_obj.__name__}"),
                        Button("Edit Parameters", handler=lambda f=func_obj, k=kwargs: self._edit_parameters(f, k)),
                        Button("Move Up", handler=lambda idx=i: self._move_function_up(idx)),
                        Button("Move Down", handler=lambda idx=i: self._move_function_down(idx)),
                        Button("Delete", handler=lambda idx=i: self._delete_function(idx))
                    ]),
                    title=f"Function {i+1}"
                )
            )

        # Add "Add Function" button
        function_frames.append(Button("Add Function", handler=self._add_function))

        return HSplit(function_frames)

    def _build_parameter_editor(self):
        """Build the parameter editor."""
        return HSplit([
            Label("Select a function to edit parameters")
        ])

    def _edit_parameters(self, func, kwargs):
        """Edit parameters for a function."""
        # Get function parameters
        parameters = get_function_parameters(func)

        # Create parameter editor UI
        parameter_widgets = []

        for param in parameters:
            name = param['name']
            default = param['default']
            current_value = kwargs.get(name, default)

            # Create label
            label = Label(f"{name}: ")

            # Create input field
            text_area = TextArea(
                text=str(current_value) if current_value is not None else "",
                multiline=False
            )

            # Create reset button
            reset_button = Button(
                "Reset",
                handler=lambda t=text_area, d=default: t.text(str(d) if d is not None else "")
            )

            # Add to parameter widgets
            parameter_widgets.append(
                HSplit([
                    VSplit([label, text_area, reset_button])
                ])
            )

        # Create save button
        save_button = Button("Save", handler=lambda: self._save_parameters(func, parameters))
        parameter_widgets.append(save_button)

        # Update parameter editor
        self.parameter_editor.children = parameter_widgets
```

#### 3. Leveraging Decorators for Function Information

```python
def get_function_metadata(func):
    """Extract metadata from a decorated function."""
    metadata = {
        'name': func.__name__,
        'doc': func.__doc__ or "No documentation available",
        'module': func.__module__
    }

    # Extract memory type information from decorators
    if hasattr(func, 'input_memory_type'):
        metadata['input_memory_type'] = func.input_memory_type

    if hasattr(func, 'output_memory_type'):
        metadata['output_memory_type'] = func.output_memory_type

    # Extract backend information
    if hasattr(func, 'backend'):
        metadata['backend'] = func.backend

    # Check for special input/output decorators
    if hasattr(func, 'special_inputs'):
        metadata['special_inputs'] = func.special_inputs

    if hasattr(func, 'special_outputs'):
        metadata['special_outputs'] = func.special_outputs

    return metadata

def create_function_info_display(func):
    """Create a display of function information based on its decorators."""
    metadata = get_function_metadata(func)

    # Create display text
    lines = [
        f"Function: {metadata['name']}",
        f"Module: {metadata['module']}",
        f"Backend: {metadata.get('backend', 'Unknown')}",
        f"Input Memory Type: {metadata.get('input_memory_type', 'Unknown')}",
        f"Output Memory Type: {metadata.get('output_memory_type', 'Unknown')}"
    ]

    # Add special inputs/outputs if available
    if 'special_inputs' in metadata:
        lines.append("Special Inputs:")
        for special_input in metadata['special_inputs']:
            lines.append(f"  - {special_input}")

    if 'special_outputs' in metadata:
        lines.append("Special Outputs:")
        for special_output in metadata['special_outputs']:
            lines.append(f"  - {special_output}")

    # Add documentation
    lines.append("\nDocumentation:")
    lines.append(metadata['doc'])

    return "\n".join(lines)
```

These examples demonstrate how to leverage OpenHCS's existing code to implement the TUI efficiently. The function registry, signature inspection, and decorator metadata provide all the information needed to build a dynamic, reflective interface that aligns with OpenHCS's architectural principles.

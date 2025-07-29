Code Generation API
===================

.. module:: openhcs.debug.pickle_to_python

The code generation API provides the core functionality for OpenHCS's bidirectional code/UI interconversion system. This API enables seamless conversion between TUI interface state and executable Python code.

**Key Features**:
- **Three-tier generation**: Function patterns → Pipeline steps → Orchestrator configuration
- **Encapsulation pattern**: Automatic import collection and upward propagation
- **Clean mode support**: Optional filtering of default values for readable output
- **Round-trip integrity**: Perfect fidelity between TUI and code representations

Core Generation Functions
--------------------------

These functions provide the primary code generation capabilities for each tier of the OpenHCS hierarchy.

.. autofunction:: generate_complete_function_pattern_code

   Generate complete Python code for a function pattern with all required imports.
   
   **Usage Example**:
   
   .. code-block:: python
   
       from openhcs.processing.backends.filters.gaussian_filter import gaussian_filter
       from openhcs.debug.pickle_to_python import generate_complete_function_pattern_code
       
       # Create a function pattern
       pattern = gaussian_filter(sigma=2.0, preserve_dtype=True)
       
       # Generate complete code
       code = generate_complete_function_pattern_code(pattern, clean_mode=False)
       print(code)
       # Output:
       # # Edit this function pattern and save to apply changes
       # 
       # # Dynamic imports
       # from openhcs.processing.backends.filters.gaussian_filter import gaussian_filter
       # 
       # pattern = gaussian_filter(sigma=2.0, preserve_dtype=True)

.. autofunction:: generate_complete_pipeline_steps_code

   Generate complete Python code for pipeline steps with encapsulated imports.
   
   **Usage Example**:
   
   .. code-block:: python
   
       from openhcs.core.steps.function_step import FunctionStep
       from openhcs.debug.pickle_to_python import generate_complete_pipeline_steps_code
       
       # Create pipeline steps
       steps = [
           FunctionStep(func=gaussian_filter(sigma=2.0), name="blur"),
           FunctionStep(func=count_cells(method="watershed"), name="count")
       ]
       
       # Generate complete code with all imports
       code = generate_complete_pipeline_steps_code(steps, clean_mode=False)

.. autofunction:: generate_complete_orchestrator_code

   Generate complete Python code for orchestrator configuration with all imports.
   
   **Usage Example**:
   
   .. code-block:: python
   
       from openhcs.core.config import GlobalPipelineConfig
       from openhcs.debug.pickle_to_python import generate_complete_orchestrator_code
       
       # Create orchestrator configuration
       plate_paths = ['/path/to/plate1', '/path/to/plate2']
       pipeline_data = {'/path/to/plate1': steps}
       global_config = GlobalPipelineConfig(num_workers=16)
       
       # Generate complete orchestrator code
       code = generate_complete_orchestrator_code(
           plate_paths, pipeline_data, global_config, clean_mode=False
       )

Import Collection System
------------------------

The import collection system automatically discovers and organizes all required imports from complex data structures.

.. autofunction:: collect_imports_from_data

   Recursively extract function and enum imports from data structures.
   
   **Algorithm**: Performs depth-first traversal of data structures (lists, tuples, dicts) to identify:
   - **Functions**: Objects with ``__module__`` and ``__name__`` attributes
   - **Enums**: Objects that are instances of ``Enum`` class
   - **Module filtering**: Only includes OpenHCS modules (``module.startswith('openhcs')``)
   
   **Returns**: Tuple of (function_imports, enum_imports) as defaultdict(set)

.. autofunction:: format_imports_as_strings

   Convert import dictionaries to formatted import statements.
   
   **Usage Example**:
   
   .. code-block:: python
   
       function_imports = defaultdict(set)
       function_imports['openhcs.processing.backends.filters.gaussian_filter'].add('gaussian_filter')
       
       enum_imports = defaultdict(set)
       enum_imports['openhcs.constants.constants'].add('VariableComponents')
       
       import_lines = format_imports_as_strings(function_imports, enum_imports)
       # Returns:
       # ['from openhcs.constants.constants import VariableComponents',
       #  'from openhcs.processing.backends.filters.gaussian_filter import gaussian_filter']

Pattern Representation
----------------------

These functions handle the conversion of complex data structures to readable Python code representations.

.. autofunction:: generate_readable_function_repr

   Generate readable Python representation of function objects and patterns.
   
   **Supports**:
   - Single functions with parameters
   - Tuple patterns (function, parameters)
   - List patterns [function1, function2, ...]
   - Dict patterns {"channel1": function1, "channel2": function2}
   - Nested combinations of the above

.. autofunction:: generate_clean_dataclass_repr

   Generate clean Python representation of dataclass instances.
   
   **Features**:
   - **Clean mode**: Optionally filter out fields with default values
   - **Recursive**: Handles nested dataclasses automatically
   - **Proper formatting**: Maintains readable indentation and structure
   - **Type safety**: Handles all standard Python types and OpenHCS custom types

Terminal Integration
--------------------

.. module:: openhcs.textual_tui.services.terminal_launcher

The TerminalLauncher service manages external editor integration for bidirectional editing.

.. autoclass:: TerminalLauncher
   :members:

   **Core Workflow**:
   
   1. **File Creation**: Creates temporary file with generated code
   2. **Editor Launch**: Launches user's ``$EDITOR`` with the file
   3. **Completion Detection**: Polls for signal file indicating editor completion
   4. **Callback Execution**: Executes callback with edited content
   5. **Cleanup**: Automatically removes temporary files
   
   **Usage Example**:
   
   .. code-block:: python
   
       from openhcs.textual_tui.services.terminal_launcher import TerminalLauncher
       
       def handle_edited_code(edited_code: str):
           """Callback to handle edited code."""
           try:
               namespace = {}
               exec(edited_code, namespace)
               if 'pattern' in namespace:
                   # Apply the edited pattern
                   apply_pattern(namespace['pattern'])
           except Exception as e:
               show_error(f"Parse error: {e}")
       
       # Launch editor for code editing
       launcher = TerminalLauncher(app)
       await launcher.launch_editor_for_file(
           file_content=generated_code,
           file_extension='.py',
           on_save_callback=handle_edited_code
       )

Widget Integration Pattern
--------------------------

All TUI widgets implement a consistent pattern for code button functionality:

**Standard Code Button Implementation**:

.. code-block:: python

    async def _handle_code_button(self):
        """Standard code button handler."""
        try:
            # 1. Generate complete code with imports
            python_code = generate_complete_*_code(self.data, clean_mode=False)
            
            # 2. Launch editor with callback
            launcher = TerminalLauncher(self.app)
            await launcher.launch_editor_for_file(
                file_content=python_code,
                file_extension='.py',
                on_save_callback=self._handle_edited_code
            )
        except Exception as e:
            self.app.show_error("Code Generation Error", str(e))

**Standard Callback Implementation**:

.. code-block:: python

    def _handle_edited_code(self, edited_code: str):
        """Standard callback for handling edited code."""
        try:
            # Parse edited code
            namespace = {}
            exec(edited_code, namespace)
            
            # Extract expected variables (widget-specific)
            if 'expected_variable' in namespace:
                new_data = namespace['expected_variable']
                self._apply_changes(new_data)
            else:
                self.app.show_error("Parse Error", "Expected variable not found")
                
        except SyntaxError as e:
            self.app.show_error("Syntax Error", f"Invalid Python syntax: {e}")
        except Exception as e:
            self.app.show_error("Edit Error", f"Failed to parse code: {e}")

**Widget-Specific Variables**:

- **FunctionListEditor**: Expects ``pattern = ...`` variable
- **PipelineEditor**: Expects ``pipeline_steps = [...]`` variable  
- **PlateManager**: Expects ``plate_paths``, ``pipeline_data``, ``global_config`` variables

Error Handling
--------------

The code generation system provides comprehensive error handling:

**Syntax Errors**:
- Caught during ``exec()`` execution
- Reported with line numbers and error descriptions
- User can return to editor to fix issues

**Import Errors**:
- Missing modules detected during code execution
- Clear error messages about missing dependencies
- Automatic import collection prevents most import issues

**Type Validation Errors**:
- Incorrect data types detected when applying parsed data
- Validation against expected data structures
- Clear feedback about expected vs. actual types

**State Consistency Errors**:
- TUI state validation after updates
- Rollback to previous state on validation failure
- User notification of consistency issues

Performance Considerations
--------------------------

**Import Collection Performance**:
- **Time Complexity**: O(n) where n is the number of objects in data structure
- **Space Complexity**: O(m) where m is the number of unique imports
- **Optimization**: Set-based deduplication prevents duplicate processing

**Code Generation Performance**:
- **Function Patterns**: <1ms for typical patterns
- **Pipeline Steps**: <10ms for 20+ step pipelines  
- **Orchestrator Config**: <100ms for complex multi-plate configurations

**Memory Usage**:
- **Temporary Files**: Automatically cleaned up after editing
- **Namespace Isolation**: Prevents memory pollution during code execution
- **Callback Management**: Automatic cleanup of callback references

See Also
--------

**Architecture**:

- :doc:`../architecture/code_ui_interconversion` - System architecture and design
- :doc:`../architecture/tui_system` - TUI system integration

**User Guides**:

- :doc:`../user_guide/code_ui_editing` - User guide for bidirectional editing
- :doc:`../user_guide/intermediate_usage` - Advanced TUI usage patterns

**Related APIs**:

- :doc:`tui_system` - TUI system API reference
- :doc:`config` - Configuration system API
- :doc:`function_step` - FunctionStep API for pipeline building

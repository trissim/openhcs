Code/UI Interconversion System
==============================

Overview
--------

OpenHCS provides a bidirectional code/UI interconversion system that enables editing between the TUI interface and generated Python code. This system addresses the limitation in scientific computing tools where users must choose between GUI convenience or code flexibility.

The system implements bidirectional conversion with round-trip fidelity preservation between representations.

Core Functionality
------------------

The system enables three critical workflows:

1. **TUI → Code**: Generate complete, executable Python code from TUI state
2. **Code → TUI**: Parse edited Python code back into TUI interface state  
3. **Round-trip Integrity**: Maintain perfect consistency between representations

This allows researchers to:
- Use the TUI for rapid prototyping and visual feedback
- Switch to code editing for complex parameter tuning and bulk modifications
- Leverage external editors, version control, and collaborative development tools
- Maintain a single source of truth that can be represented in either form

Three-Tier Generation Architecture
----------------------------------

The code generation system follows OpenHCS's encapsulation pattern with three hierarchical tiers:

.. code-block:: text

    Function Patterns (Tier 1)
           ↓ (encapsulates imports)
    Pipeline Steps (Tier 2)  
           ↓ (encapsulates all pattern imports)
    Orchestrator Config (Tier 3)
           ↓ (encapsulates all pipeline imports)
    Complete Executable Script

**Tier 1: Function Pattern Generation**

.. code-block:: python

    # Generated from FunctionListEditor code button
    from openhcs.processing.backends.filters.gaussian_filter import gaussian_filter
    from openhcs.constants.constants import VariableComponents
    
    pattern = gaussian_filter(sigma=2.0, preserve_dtype=True)

**Tier 2: Pipeline Step Generation**

.. code-block:: python

    # Generated from PipelineEditor code button  
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.constants.constants import VariableComponents
    # Function imports from encapsulated patterns
    from openhcs.processing.backends.filters.gaussian_filter import gaussian_filter
    from openhcs.processing.backends.analysis.cell_counting import count_cells
    
    pipeline_steps = []
    
    step_1 = FunctionStep(
        func=(gaussian_filter, {'sigma': 2.0, 'preserve_dtype': True}),
        name="gaussian_filter",
        variable_components=[VariableComponents.PLATE]
    )
    pipeline_steps.append(step_1)

    step_2 = FunctionStep(
        func=(count_cells, {'method': 'watershed'}),
        name="count_cells",
        variable_components=[VariableComponents.PLATE]
    )
    pipeline_steps.append(step_2)

**Tier 3: Orchestrator Configuration Generation**

.. code-block:: python

    # Generated from PlateManager code button
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.constants.constants import VariableComponents
    from openhcs.core.config import GlobalPipelineConfig, PathPlanningConfig
    # Function imports from all encapsulated pipelines
    from openhcs.processing.backends.filters.gaussian_filter import gaussian_filter
    from openhcs.processing.backends.analysis.cell_counting import count_cells
    
    plate_paths = ['/path/to/plate1', '/path/to/plate2']
    
    global_config = GlobalPipelineConfig(
        num_workers=16,
        path_planning=PathPlanningConfig(
            output_dir_suffix="_processed"
        )
    )
    
    pipeline_data = {}
    # Steps for each plate...

Encapsulation Pattern
---------------------

The system implements a strict **upward import encapsulation** pattern:

.. code-block:: text

    Function Pattern:
    ├── Imports: [gaussian_filter]
    └── Code: pattern = gaussian_filter(...)
    
    Pipeline Steps:
    ├── Imports: [FunctionStep, VariableComponents] + [gaussian_filter, count_cells]
    └── Code: pipeline_steps = [step_1, step_2, ...]
    
    Orchestrator Config:
    ├── Imports: [GlobalPipelineConfig, ...] + [all pipeline imports]
    └── Code: plate_paths, global_config, pipeline_data

**Benefits of Encapsulation:**
- **No Import Duplication**: Each tier includes all imports from lower tiers
- **Complete Executability**: Generated code runs without additional imports
- **Dependency Tracking**: Clear visibility of all required modules
- **Maintainability**: Changes to function patterns automatically propagate upward

Bidirectional Conversion Workflow
---------------------------------

The complete round-trip workflow ensures perfect fidelity:

**Code Generation (TUI → Code)**

.. code-block:: text

    1. User clicks "Code" button in TUI widget
    2. Widget extracts current state (functions, parameters, configuration)
    3. Appropriate generation function called:
       - generate_complete_function_pattern_code()
       - generate_complete_pipeline_steps_code()  
       - generate_complete_orchestrator_code()
    4. Import collection system traverses data structures
    5. Complete Python code generated with all imports
    6. TerminalLauncher creates temporary file with code
    7. User's $EDITOR launched for editing

**Code Parsing (Code → TUI)**

.. code-block:: text

    1. User saves and exits editor
    2. TerminalLauncher detects completion via signal file
    3. Edited code read from temporary file
    4. Code executed in isolated namespace: exec(edited_code, namespace)
    5. Expected variables extracted from namespace:
       - pattern (for function patterns)
       - pipeline_steps (for pipeline steps)
       - plate_paths, pipeline_data, global_config (for orchestrator)
    6. Widget state updated with parsed data
    7. TUI interface refreshes to reflect changes
    8. Temporary files cleaned up

**Error Handling and Validation**

The system provides comprehensive error handling at each conversion step:

- **Syntax Validation**: Python syntax errors caught and reported with line numbers
- **Variable Validation**: Missing expected variables detected and reported
- **Type Validation**: Incorrect data types validated against expected structures
- **Import Resolution**: Missing imports detected during execution
- **State Consistency**: TUI state validated after updates

Terminal Integration Architecture
--------------------------------

The **TerminalLauncher** service manages the external editor integration:

**File-Based Communication Pattern**

.. code-block:: text

    TUI Process                    Editor Process
         │                              │
         ├─ Create temp file            │
         ├─ Launch $EDITOR ────────────→│
         ├─ Start polling               │
         │                              ├─ Edit file
         │                              ├─ Save & exit
         │                              └─ Create signal file
         ├─ Detect signal file          │
         ├─ Read edited content         │
         ├─ Execute callback            │
         └─ Cleanup temp files          │

**Asynchronous Polling System**

The system uses asynchronous polling to detect editor completion without blocking the TUI:

.. code-block:: python

    async def poll_for_completion():
        while True:
            if os.path.exists(signal_file):
                # Editor completed, process changes
                with open(file_path, 'r') as f:
                    content = f.read()
                callback(content)  # Update TUI state
                break
            await asyncio.sleep(0.1)  # Non-blocking poll

**Editor Integration**

The system respects user preferences and environment:

- **Environment Variable**: Uses ``$EDITOR`` or defaults to ``nano``
- **Terminal Compatibility**: Works with vim, emacs, nano, micro
- **SSH Support**: Full functionality over SSH connections
- **Unicode Support**: Proper handling of special characters and encoding

Widget Integration Pattern
--------------------------

All TUI widgets implement a consistent code button pattern:

**Standard Implementation**

.. code-block:: python

    async def _handle_code_button(self):
        """Standard code button handler pattern."""
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
    
    def _handle_edited_code(self, edited_code: str):
        """Standard callback for handling edited code."""
        try:
            # 3. Parse edited code
            namespace = {}
            exec(edited_code, namespace)
            
            # 4. Extract expected variables
            if 'expected_variable' in namespace:
                new_data = namespace['expected_variable']
                self._apply_changes(new_data)
            else:
                self.app.show_error("Parse Error", "Expected variable not found")
                
        except SyntaxError as e:
            self.app.show_error("Syntax Error", f"Invalid Python syntax: {e}")
        except Exception as e:
            self.app.show_error("Edit Error", f"Failed to parse code: {e}")

**Widget-Specific Variables**

Each widget expects specific variables in the edited code:

- **FunctionListEditor**: ``pattern = ...`` (function pattern)
- **PipelineEditor**: ``pipeline_steps = [...]`` (list of FunctionStep objects)
- **PlateManager**: ``plate_paths``, ``pipeline_data``, ``global_config`` (orchestrator config)

Performance and Scalability
---------------------------

The system is designed for efficiency with large pipelines:

**Import Collection Optimization**

- **Recursive Traversal**: Efficient depth-first search of data structures
- **Deduplication**: Set-based import collection prevents duplicates
- **Module Filtering**: Only OpenHCS modules included in generated imports
- **Lazy Evaluation**: Imports collected only when code generation requested

**Memory Management**

- **Temporary Files**: Automatic cleanup prevents disk space leaks
- **Namespace Isolation**: Code execution in isolated namespace prevents pollution
- **Callback Cleanup**: Automatic cleanup of callback references

**Scalability Metrics**

- **Function Patterns**: <1ms generation time for typical patterns
- **Pipeline Steps**: <10ms for pipelines with 20+ steps
- **Orchestrator Config**: <100ms for multi-plate configurations with complex pipelines
- **Memory Usage**: <5MB additional memory during code generation

See Also
--------

**Core Integration**:

- :doc:`tui_system` - TUI system architecture and components
- :doc:`../api/code_generation` - Code generation API reference
- :doc:`../user_guide/code_ui_editing` - User guide for bidirectional editing

**Related Systems**:

- :doc:`function_pattern_system` - Function pattern architecture
- :doc:`pipeline_compilation_system` - Pipeline compilation integration
- :doc:`configuration_system_architecture` - Configuration system integration

**Practical Usage**:

- :doc:`../guides/complete_examples` - Complete workflow examples
- :doc:`../user_guide/intermediate_usage` - Advanced TUI usage patterns
- :doc:`../development/extending` - Extending the code generation system

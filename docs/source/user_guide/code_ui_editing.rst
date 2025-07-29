Code/UI Bidirectional Editing
=============================

OpenHCS provides **bidirectional editing** between the TUI interface and Python code. This allows you to seamlessly switch between visual interface editing and code-based editing, leveraging the best of both approaches for bioimage analysis pipeline development.

**Why This Matters**: Most scientific tools provide one-way export from GUI to code. OpenHCS implements true bidirectional conversion, allowing you to edit in either representation while maintaining complete fidelity between them.

Quick Start (5 Minutes)
-----------------------

**1. Open any TUI widget with a "Code" button**

.. code-block:: bash

    # Start OpenHCS TUI
    openhcs-tui
    
    # Navigate to Function List Editor, Pipeline Editor, or Plate Manager
    # Look for the "Code" button in the interface

**2. Click "Code" to generate Python code**

The system automatically generates complete, executable Python code with all necessary imports:

.. code-block:: python

    # Generated from Function Pattern Editor
    from openhcs.processing.backends.filters.gaussian_filter import gaussian_filter
    
    pattern = gaussian_filter(sigma=2.0, preserve_dtype=True)

**3. Edit the code in your preferred editor**

Your ``$EDITOR`` opens automatically (vim, nano, emacs, VS Code, etc.):

.. code-block:: python

    # Edit parameters, add comments, modify the pattern
    pattern = gaussian_filter(sigma=3.5, preserve_dtype=True)  # Increased blur

**4. Save and exit**

The TUI automatically updates to reflect your changes. The interface now shows ``sigma=3.5`` instead of ``sigma=2.0``.

**5. Continue working**

Switch back and forth between TUI and code editing as needed. Each representation stays perfectly synchronized.

Function Pattern Editing
------------------------

**When to Use**: Fine-tuning complex parameters, copying patterns between projects, or when you need syntax highlighting for complex function configurations.

**Complete Workflow Example**:

1. **Start with TUI**: Create a gaussian filter in the Function List Editor
2. **Generate Code**: Click "Code" button to see the pattern as Python code
3. **Edit Parameters**: Modify sigma, add preserve_dtype, adjust other parameters
4. **Apply Changes**: Save and exit editor to update the TUI
5. **Verify Results**: See the updated parameters reflected in the interface

**Real Example - Parameter Tuning**:

.. code-block:: python

    # Original pattern (from TUI)
    from openhcs.processing.backends.filters.gaussian_filter import gaussian_filter
    
    pattern = gaussian_filter(sigma=2.0)
    
    # After editing (your changes)
    pattern = gaussian_filter(
        sigma=3.5,           # Increased blur strength
        preserve_dtype=True, # Maintain original data type
        truncate=4.0         # Adjust truncation for performance
    )

**Benefits**:
- **Syntax Highlighting**: See parameter names and values clearly
- **Copy/Paste**: Easy to duplicate patterns across projects
- **Comments**: Add documentation directly in the pattern
- **Validation**: Python syntax checking catches errors immediately

Pipeline Code Editing
---------------------

**When to Use**: Bulk pipeline modifications, step reordering, adding multiple steps, or when you need to see the entire pipeline structure at once.

**Complete Workflow Example**:

1. **Build Pipeline**: Create a multi-step pipeline in the Pipeline Editor
2. **Generate Code**: Click "Code" button to see all steps as Python code
3. **Bulk Edit**: Add steps, reorder, modify multiple parameters at once
4. **Apply Changes**: Save to update the entire pipeline in the TUI

**Real Example - Adding Cell Counting**:

.. code-block:: python

    # Original pipeline (from TUI)
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.constants.constants import VariableComponents
    from openhcs.processing.backends.filters.gaussian_filter import gaussian_filter
    
    pipeline_steps = []
    
    step_1 = FunctionStep(
        func=gaussian_filter(sigma=2.0),
        name="gaussian_filter",
        variable_components=[VariableComponents.PLATE]
    )
    pipeline_steps.append(step_1)
    
    # After editing - added cell counting step
    from openhcs.processing.backends.analysis.cell_counting import count_cells
    
    # ... existing step_1 ...
    
    step_2 = FunctionStep(
        func=count_cells(method="watershed", min_size=50),
        name="count_cells",
        variable_components=[VariableComponents.PLATE],
        force_disk_output=True  # Save counting results
    )
    pipeline_steps.append(step_2)

**Advanced Pipeline Editing**:

.. code-block:: python

    # Reorder steps by changing append order
    pipeline_steps.append(step_2)  # Count first
    pipeline_steps.append(step_1)  # Then filter
    
    # Conditional steps
    if analysis_type == "detailed":
        step_3 = FunctionStep(func=detailed_analysis(), name="detailed")
        pipeline_steps.append(step_3)
    
    # Bulk parameter changes
    for step in pipeline_steps:
        step.variable_components = [VariableComponents.WELL]  # Change all to well-level

Orchestrator Configuration
-------------------------

**When to Use**: Global configuration changes, multi-plate setup, advanced backend configuration, or when managing complex processing scenarios.

**Complete Workflow Example**:

1. **Setup Plates**: Configure multiple plates in the Plate Manager
2. **Generate Code**: Click "Code" button to see complete orchestrator configuration
3. **Global Changes**: Modify worker counts, backend settings, output configurations
4. **Apply Changes**: Save to update the entire system configuration

**Real Example - Multi-Plate Processing**:

.. code-block:: python

    # Generated orchestrator configuration
    from openhcs.core.config import GlobalPipelineConfig, PathPlanningConfig
    from openhcs.constants.constants import Backend
    
    plate_paths = [
        '/data/experiment1/plate1',
        '/data/experiment1/plate2', 
        '/data/experiment2/plate1'
    ]
    
    global_config = GlobalPipelineConfig(
        num_workers=32,  # Increased for large dataset
        path_planning=PathPlanningConfig(
            output_dir_suffix="_processed_v2",
            global_output_folder="/results/batch_analysis"
        ),
        vfs=VFSConfig(
            default_backend=Backend.ZARR,  # Use ZARR for large datasets
            memory_limit_gb=64
        )
    )
    
    # Pipeline data for each plate
    pipeline_data = {}
    for plate_path in plate_paths:
        pipeline_data[plate_path] = [
            # Same pipeline for all plates
            step_1, step_2, step_3
        ]

**Configuration Scenarios**:

.. code-block:: python

    # High-performance configuration
    global_config = GlobalPipelineConfig(
        num_workers=64,
        use_threading=False,  # Use multiprocessing for CPU-bound tasks
        vfs=VFSConfig(memory_limit_gb=128)
    )
    
    # Memory-constrained configuration  
    global_config = GlobalPipelineConfig(
        num_workers=8,
        vfs=VFSConfig(
            default_backend=Backend.DISK,
            memory_limit_gb=16
        )
    )

Editor Integration
-----------------

**Environment Setup**:

.. code-block:: bash

    # Set your preferred editor
    export EDITOR=vim        # For vim users
    export EDITOR=nano       # For nano users  
    export EDITOR=emacs      # For emacs users
    export EDITOR="code -w"  # For VS Code users (with wait flag)

**Supported Editors**:

- **Terminal Editors**: vim, nano, emacs, micro, helix
- **GUI Editors**: VS Code (``code -w``), Sublime Text, Atom
- **SSH Compatible**: All terminal editors work perfectly over SSH
- **Syntax Highlighting**: Most editors automatically detect ``.py`` files

**SSH Considerations**:

.. code-block:: bash

    # For SSH sessions, use terminal-only editors
    export EDITOR=vim
    # or
    export EDITOR=nano
    
    # Avoid GUI editors over SSH unless X11 forwarding is configured

**File Associations**:

The system creates temporary ``.py`` files, so your editor's Python syntax highlighting will automatically activate, providing:

- **Keyword highlighting**: ``def``, ``class``, ``import``, etc.
- **String highlighting**: Proper string and comment formatting
- **Indentation guides**: Visual indentation assistance
- **Error detection**: Basic syntax error highlighting

Best Practices
--------------

**When to Use Code vs TUI**:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Scenario
     - Use TUI
     - Use Code Editing
   * - Quick parameter changes
     - ✓ Fast visual editing
     - ✗ Overkill for simple changes
   * - Complex parameter tuning
     - ✗ Limited precision
     - ✓ Exact values, comments
   * - Adding single steps
     - ✓ Visual workflow
     - ✗ More overhead
   * - Bulk pipeline changes
     - ✗ Tedious repetition
     - ✓ Efficient bulk editing
   * - Learning the system
     - ✓ Visual feedback
     - ✗ Requires Python knowledge
   * - Sharing configurations
     - ✗ Hard to communicate
     - ✓ Copy/paste code snippets
   * - Version control
     - ✗ No direct integration
     - ✓ Perfect for git workflows

**Version Control Integration**:

.. code-block:: bash

    # Save generated code to version control
    # 1. Generate code from TUI
    # 2. Copy to your project repository
    # 3. Commit with meaningful messages
    
    git add pipeline_config.py
    git commit -m "Add cell counting step to analysis pipeline"
    
    # Share with collaborators
    git push origin feature/cell-counting

**Collaborative Workflows**:

1. **Code Sharing**: Generate code and share via email, Slack, or version control
2. **Configuration Templates**: Create reusable configuration templates
3. **Documentation**: Add comments in code for team understanding
4. **Review Process**: Use code review tools for pipeline validation

Troubleshooting
---------------

**Common Syntax Errors**:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Error
     - Solution
   * - ``SyntaxError: invalid syntax``
     - Check for missing commas, parentheses, or quotes
   * - ``NameError: name 'X' is not defined``
     - Ensure all imports are present at the top
   * - ``IndentationError: expected an indented block``
     - Fix Python indentation (use spaces, not tabs)
   * - ``TypeError: 'X' object is not callable``
     - Check function call syntax: ``func()`` not ``func``

**Import Resolution Issues**:

.. code-block:: python

    # Problem: Missing import
    pattern = gaussian_filter(sigma=2.0)  # NameError
    
    # Solution: Add the import
    from openhcs.processing.backends.filters.gaussian_filter import gaussian_filter
    pattern = gaussian_filter(sigma=2.0)  # Works

**Variable Name Issues**:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Widget
     - Expected Variable
     - Example
   * - Function List Editor
     - ``pattern = ...``
     - ``pattern = gaussian_filter(...)``
   * - Pipeline Editor
     - ``pipeline_steps = [...]``
     - ``pipeline_steps = [step_1, step_2]``
   * - Plate Manager
     - ``plate_paths``, ``pipeline_data``, ``global_config``
     - All three variables required

**Editor Configuration Issues**:

.. code-block:: bash

    # Problem: Editor not found
    # Error: "command not found: vim"
    
    # Solution: Install editor or use available one
    export EDITOR=nano  # Use nano instead
    
    # Problem: Editor doesn't wait
    # VS Code exits immediately, changes not detected
    
    # Solution: Add wait flag
    export EDITOR="code -w"  # Wait for file to close

**State Synchronization Problems**:

If the TUI doesn't update after editing:

1. **Check for syntax errors**: Look for error dialogs in the TUI
2. **Verify variable names**: Ensure you're using the correct variable name
3. **Check imports**: Make sure all required imports are present
4. **Restart if needed**: Close and reopen the TUI widget as a last resort

**Performance Issues**:

For large pipelines or configurations:

- **Use clean mode**: Add ``clean_mode=True`` to reduce generated code size
- **Simplify temporarily**: Edit smaller sections at a time
- **Check memory**: Large configurations may require more system memory

Advanced Usage
--------------

**Custom Function Integration**:

.. code-block:: python

    # Add your own functions to pipelines
    def custom_preprocessing(image_array):
        """Custom preprocessing function."""
        # Your custom logic here
        return processed_array
    
    # Use in pipeline
    step_custom = FunctionStep(
        func=custom_preprocessing,
        name="custom_preprocessing",
        variable_components=[VariableComponents.PLATE]
    )

**Configuration Templates**:

.. code-block:: python

    # Create reusable configuration templates
    HIGH_PERFORMANCE_CONFIG = GlobalPipelineConfig(
        num_workers=64,
        vfs=VFSConfig(memory_limit_gb=128)
    )
    
    MEMORY_EFFICIENT_CONFIG = GlobalPipelineConfig(
        num_workers=8,
        vfs=VFSConfig(default_backend=Backend.DISK)
    )
    
    # Use templates
    global_config = HIGH_PERFORMANCE_CONFIG

**Conditional Processing**:

.. code-block:: python

    # Add conditional logic to pipelines
    if experiment_type == "high_resolution":
        pipeline_steps.append(high_res_step)
    else:
        pipeline_steps.append(standard_step)
    
    # Plate-specific configurations
    for plate_path in plate_paths:
        if "control" in plate_path:
            pipeline_data[plate_path] = control_pipeline
        else:
            pipeline_data[plate_path] = treatment_pipeline

See Also
--------

**Technical Details**:

- :doc:`../architecture/code_ui_interconversion` - System architecture and design
- :doc:`../api/code_generation` - Code generation API reference

**Related Guides**:

- :doc:`intermediate_usage` - Advanced TUI usage patterns
- :doc:`../guides/complete_examples` - Complete workflow examples

**Development**:

- :doc:`../development/extending` - Extending the code generation system
- :doc:`../development/custom_functions` - Adding custom functions to pipelines

TUI System
==========

.. module:: openhcs.textual_tui

OpenHCS provides a production-grade Terminal User Interface (TUI) for interactive pipeline creation, execution, and monitoring. The TUI is built with Textual and supports SSH connectivity for remote server usage.

Key Features
------------

**SSH Compatibility**: Full functionality over SSH connections with optimized rendering
**Real-time Monitoring**: Live pipeline execution progress and resource usage
**Interactive Pipeline Builder**: Visual pipeline creation with drag-and-drop functionality  
**Plate Management**: Browse, select, and configure microscopy plates
**Code Generation**: Export complete, executable Python scripts
**Error Handling**: Comprehensive error reporting and debugging tools

TUI Components
--------------

Main Application
^^^^^^^^^^^^^^^^

The TUI is launched through the main application class:

.. code-block:: python

    from openhcs.textual_tui.app import OpenHCSTUIApp

    # Launch TUI
    app = OpenHCSTUIApp()
    app.run()

    # Or from command line
    # python -m openhcs.textual_tui

Core Windows
^^^^^^^^^^^^

**Plate Manager Window**
    Browse and select microscopy plates, configure processing parameters

**Pipeline Builder Window**  
    Interactive pipeline creation with function selection and parameter tuning

**Execution Monitor Window**
    Real-time pipeline execution monitoring with progress bars and logs

**File Browser Window**
    Navigate file systems with support for different storage backends

Window System
^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.textual_tui.windows.plate_manager_window import PlateManagerWindow
    from openhcs.textual_tui.windows.pipeline_builder_window import PipelineBuilderWindow

    # Windows are managed by the main application
    # Each window provides specific functionality for different workflow stages

SSH Optimization
----------------

The TUI is optimized for SSH usage with several performance enhancements:

**Efficient Rendering**: Minimal screen updates to reduce SSH latency
**Keyboard Navigation**: Full keyboard control for terminal-only environments
**Compact Display**: Information-dense layouts optimized for terminal windows
**Connection Resilience**: Graceful handling of connection interruptions

Usage Patterns
--------------

Interactive Pipeline Creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Launch TUI**: Start the terminal interface
2. **Select Plates**: Browse and choose microscopy data
3. **Build Pipeline**: Add processing steps with visual interface
4. **Configure Parameters**: Set function parameters and execution options
5. **Generate Code**: Export complete Python script
6. **Execute**: Run pipeline with real-time monitoring

.. code-block:: bash

    # Launch TUI
    python -m openhcs.textual_tui

    # Or with specific configuration
    python -m openhcs.textual_tui --config /path/to/config.yaml

Bidirectional Code Editing
^^^^^^^^^^^^^^^^^^^^^^^^^^

The TUI provides **bidirectional editing** between interface and code:

**Code Generation**: Every TUI widget with a "Code" button generates complete, executable Python code with all required imports:

.. code-block:: python

    # Generated from Function Pattern Editor
    from openhcs.processing.backends.filters.gaussian_filter import gaussian_filter

    pattern = gaussian_filter(sigma=2.0, preserve_dtype=True)

**Code Editing**: Click "Code" → Edit in your preferred editor → Save → TUI updates automatically

**Three-Tier System**:

- **Function Patterns**: Individual function configurations with parameters
- **Pipeline Steps**: Complete pipeline definitions with all function imports
- **Orchestrator Config**: Full system configuration with global settings

**Encapsulation Pattern**: Each tier includes all imports from lower tiers, ensuring complete executability.

**Script Generation**: The "Save" button generates complete, executable Python scripts:

.. code-block:: python

    #!/usr/bin/env python3
    """
    OpenHCS Pipeline Script - Generated from TUI
    Generated: 2025-07-28 13:48:48
    """

    import sys
    from pathlib import Path

    # Add OpenHCS to path
    sys.path.insert(0, "/path/to/openhcs")

    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.core.config import GlobalPipelineConfig
    # All function imports from pipeline steps
    from openhcs.processing.backends.filters.gaussian_filter import gaussian_filter
    from openhcs.processing.backends.analysis.cell_counting import count_cells

    def create_pipeline():
        """Create and return the pipeline configuration."""

        # Generated configuration and steps
        plate_paths = ['/path/to/microscopy/data']
        steps = [
            FunctionStep(func=gaussian_filter(sigma=2.0), name="blur"),
            FunctionStep(func=count_cells(method="watershed"), name="count")
        ]
        global_config = GlobalPipelineConfig(num_workers=16)

        return plate_paths, steps, global_config

    def main():
        """Main execution function."""
        plate_paths, steps, global_config = create_pipeline()

        # Process each plate separately
        for plate_path in plate_paths:
            orchestrator = PipelineOrchestrator(
                plate_path=plate_path,
                global_config=global_config
            )
            orchestrator.initialize()
            compiled_contexts = orchestrator.compile_pipelines(steps)
            results = orchestrator.execute_compiled_plate(
                pipeline_definition=steps,
                compiled_contexts=compiled_contexts
            )

    if __name__ == "__main__":
        main()

Remote Server Usage
-------------------

The TUI is designed for remote server deployment:

.. code-block:: bash

    # SSH to remote server
    ssh user@gpu-server.example.com

    # Launch TUI on remote server
    cd /path/to/openhcs
    python -m openhcs.textual_tui

    # TUI runs efficiently over SSH connection
    # Full functionality available remotely

Configuration
-------------

TUI behavior can be configured through YAML files:

.. code-block:: yaml

    # tui_config.yaml
    display:
      compact_mode: true
      refresh_rate: 30
      
    ssh:
      optimize_rendering: true
      keyboard_only: false
      
    defaults:
      num_workers: 8
      output_directory: "/data/results"
      
    file_browser:
      default_backend: "disk"
      show_hidden: false

Services and Components
-----------------------

The TUI uses a service-oriented architecture:

**File Browser Service**: Handle file system navigation across backends
**Pipeline Service**: Manage pipeline creation and validation  
**Execution Service**: Coordinate pipeline execution and monitoring
**Configuration Service**: Handle settings and preferences

Integration with OpenHCS
-------------------------

The TUI integrates seamlessly with OpenHCS core systems:

**Function Registry**: Automatic discovery of available processing functions
**Memory Type System**: Visual indicators for GPU/CPU function compatibility
**VFS System**: Support for disk, memory, and ZARR backends in file browser
**Configuration System**: Direct integration with OpenHCS configuration classes

Keyboard Shortcuts
-------------------

**Global Navigation**:
- ``Ctrl+Q``: Quit application
- ``Tab``: Navigate between widgets
- ``Escape``: Cancel current operation

**Plate Manager**:
- ``Enter``: Select plate
- ``Space``: Toggle plate selection
- ``F5``: Refresh plate list

**Pipeline Builder**:
- ``Ctrl+A``: Add new step
- ``Delete``: Remove selected step
- ``F2``: Edit step parameters

**File Browser**:
- ``Enter``: Navigate into directory
- ``Backspace``: Go up one level
- ``Ctrl+H``: Toggle hidden files

See Also
--------

- :doc:`../architecture/tui_system` - TUI architecture and design
- :doc:`../architecture/code_ui_interconversion` - Code generation architecture
- :doc:`../user_guide/production_examples` - Generated script examples
- :doc:`../architecture/configuration_management_system` - Configuration system

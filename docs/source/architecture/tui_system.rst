TUI System Architecture
=======================

Overview
--------

OpenHCS provides a sophisticated terminal user interface (TUI) built
with the Textual framework - unprecedented for scientific computing
tools. This production-grade interface works anywhere a terminal works,
including remote servers, containers, and SSH connections.

**Note**: This document describes the actual TUI implementation. Some
features are aspirational and marked as “Future Enhancements”.

The Innovation
--------------

**What Makes It Unique**: Most scientific tools are either command-line
only or have basic desktop GUIs. OpenHCS provides a **production-grade
terminal interface** that maintains full functionality in any terminal
environment.

Core Components
---------------

Real-Time Pipeline Editor
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Interactive pipeline creation with live validation:
   ┌─ Pipeline Editor ─────────────────────────────────┐
   │ [Add Step] [Delete] [Edit] [Load] [Save]          │
   │                                                   │
   │ 1. ✓ gaussian_filter (sigma=2.0)                 │
   │ 2. ✓ binary_opening (footprint=disk(3))          │
   │ 3. ⚠ custom_function (missing parameter)         │
   │ 4. ✓ label (connectivity=2)                      │
   │                                                   │
   │ Status: 3/4 steps valid | GPU Memory: 2.1GB      │
   └───────────────────────────────────────────────────┘

**Features**: - **Live validation**: Steps validated as you type -
**Visual feedback**: Color-coded status indicators - **Resource
monitoring**: Real-time GPU memory usage - **Button-based management**:
Add, Delete, Edit, Load, Save operations - **Per-plate pipeline
storage**: Separate pipelines for each plate

Live Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Dynamic configuration with instant validation:
   ┌─ Global Configuration ────────────────────────────┐
   │ Workers: [8] ▲▼     VFS Backend: [memory] ▼       │
   │ GPU Slots: [4] ▲▼   Zarr Compression: [lz4] ▼     │
   │                                                   │
   │ ✓ Configuration valid                             │
   │ ⚠ Warning: High memory usage with 8 workers       │
   └───────────────────────────────────────────────────┘

**Features**: - **Instant validation**: Configuration checked in
real-time - **Smart warnings**: Proactive resource usage alerts -
**Type-safe inputs**: Prevents invalid configuration values -
**Context-sensitive help**: Tooltips and documentation - **Profile
management**: Save/load configuration presets

Integrated Help System
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Context-sensitive help with full type information:
   ┌─ Help: gaussian_filter ───────────────────────────┐
   │ gaussian_filter (sigma: float = 1.0)              │
   │                                                   │
   │ Apply Gaussian blur to image stack.               │
   │                                                   │
   │ Parameters:                                       │
   │ • sigma: float - Standard deviation for blur      │
   │ • mode: str (optional) - Boundary condition       │
   │                                                   │
   │ Memory: numpy → numpy | Contract: SLICE_SAFE      │
   └───────────────────────────────────────────────────┘

**Features**: - **Full type information**: Complete Union types, not
just “Union” - **Parameter separation**: Individual parameters with
descriptions - **Memory contracts**: Shows input/output memory types -
**Processing behavior**: SLICE_SAFE vs CROSS_Z indicators - **Example
usage**: Code snippets and common patterns

Professional Log Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Real-time log viewing with filtering:
   ┌─ System Logs ─────────────────────────────────────┐
   │ [Current Session ▼] [Filter: ERROR ▼] [Tail: ON]  │
   │                                                   │
   │ 12:34:56 INFO  Pipeline compiled successfully     │
   │ 12:34:57 DEBUG GPU memory allocated: 1.2GB        │
   │ 12:34:58 ERROR Step 3 validation failed           │
   │ 12:34:59 INFO  Retrying with CPU fallback         │
   │                                                   │
   │ Lines: 1,247 | Filtered: 23 errors                │
   └───────────────────────────────────────────────────┘

**Features**: - **Multi-file support**: Switch between different log
files - **Real-time tailing**: Live updates as logs are written -
**Advanced filtering**: Filter by level, component, or pattern -
**Session management**: Only shows current session logs - **Search
functionality**: Find specific log entries quickly

Architecture
------------

Textual Framework Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Actual TUI architecture (window-based):
   class OpenHCSTUIApp(App):
       """Main OpenHCS Textual TUI Application."""

       def __init__(self, global_config: Optional[GlobalPipelineConfig] = None):
           super().__init__()
           self.global_config = global_config or get_default_global_config()
           self.storage_registry = storage_registry
           self.filemanager = FileManager(self.storage_registry)

       def compose(self) -> ComposeResult:
           """Compose the main application layout."""
           # Custom window bar for window management
           yield CustomWindowBar(dock="bottom", start_open=True)

           # Status bar for messages
           yield StatusBar()

           # Main content with system monitor background
           yield MainContent(
               filemanager=self.filemanager,
               global_config=self.global_config
           )

Component Architecture
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Actual TUI component architecture:
   TUI Components:
   ├── Core Application (OpenHCSTUIApp)
   ├── Main Layout
   │   ├── CustomWindowBar (window management)
   │   ├── StatusBar (status messages)
   │   └── MainContent (SystemMonitor background)
   ├── Floating Windows (textual-window)
   │   ├── PipelinePlateWindow (PlateManagerWidget + PipelineEditorWidget)
   │   ├── ConfigWindow (configuration editing)
   │   ├── HelpWindow (help system)
   │   ├── DualEditorWindow (function step editing)
   │   └── ErrorDialog (error display)
   ├── Core Widgets
   │   ├── PlateManagerWidget (plate management)
   │   ├── PipelineEditorWidget (pipeline editing)
   │   ├── OpenHCSToolongWidget (log viewing)
   │   ├── FunctionListEditorWidget (function editing)
   │   └── ConfigFormWidget (configuration forms)
   └── Services
       ├── ValidationService (form validation)
       ├── TerminalLauncher (external editor)
       └── GlobalConfigCache (configuration caching)

State Management
~~~~~~~~~~~~~~~~

.. code:: python

   # Actual reactive state implementation:
   class PipelineEditorWidget(ButtonListWidget):
       # Real reactive properties from implementation
       pipeline_steps = reactive([])
       current_plate = reactive("")
       selected_step = reactive("")
       plate_pipelines = reactive({})  # Per-plate pipeline storage

       def watch_pipeline_steps(self, old_steps, new_steps):
           """Automatically called when pipeline_steps changes."""
           logger.debug(f"Pipeline steps changed: {len(new_steps)} steps")
           self._update_button_states()
           self._update_display()

Remote Access Capabilities
--------------------------

SSH-Friendly Design
~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Works perfectly over SSH connections:
   ssh user@remote-server
   cd /path/to/openhcs
   python -m openhcs.textual_tui

   # Full functionality maintained:
   ✅ Interactive editing
   ✅ Real-time updates  
   ✅ Mouse support (when available)
   ✅ Keyboard navigation
   ✅ Copy/paste operations

Web Interface Option
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Optional web interface for browser access:
   python -m openhcs.textual_tui --web

   # Serves TUI in browser:
   🌐 Starting OpenHCS web server...
   🔗 Your TUI will be available at: http://localhost:8000
   📝 Share this URL to give others access to your OpenHCS TUI
   ⚠️  Note: The TUI runs on YOUR machine, others just see it in their browser

Container Compatibility
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Works in Docker containers:
   docker run -it openhcs/openhcs python -m openhcs.textual_tui

   # Kubernetes deployment:
   kubectl run openhcs-tui --image=openhcs/openhcs --stdin --tty \
     --command -- python -m openhcs.textual_tui

Comparison with Other Scientific Tools
--------------------------------------

Traditional Scientific Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----+--------------+--------------+-------------------+------------+
| T   | Interface    | Remote       | Real-time Updates | Help       |
| ool | Type         | Access       |                   | System     |
+=====+==============+==============+===================+============+
| *   | Desktop GUI  | ❌ X11       | ❌ Manual refresh | ⚠️ Basic   |
| *Im |              | forwarding   |                   | tooltips   |
| age |              | only         |                   |            |
| J** |              |              |                   |            |
+-----+--------------+--------------+-------------------+------------+
| *   | Desktop GUI  | ❌ X11       | ❌ Static         | ⚠️         |
| *Ce |              | forwarding   | interface         | Separate   |
| llP |              | only         |                   | doc        |
| rof |              |              |                   | umentation |
| ile |              |              |                   |            |
| r** |              |              |                   |            |
+-----+--------------+--------------+-------------------+------------+
| *   | Desktop GUI  | ❌ X11       | ⚠️ Limited        | ⚠️         |
| *na |              | forwarding   | updates           | Plugin     |
| par |              | required     |                   | -dependent |
| i** |              |              |                   |            |
+-----+--------------+--------------+-------------------+------------+
| **  | Desktop GUI  | ❌ X11       | ❌ Manual refresh | ⚠️         |
| FIJ |              | forwarding   |                   | Wiki-based |
| I** |              | only         |                   | help       |
+-----+--------------+--------------+-------------------+------------+
| **  | **Terminal   | ✅ **SSH     | ✅ **Live         | ✅         |
| Ope | TUI**        | native**     | updates**         | **         |
| nHC |              |              |                   | Integrated |
| S** |              |              |                   | help**     |
+-----+--------------+--------------+-------------------+------------+

Command-Line Tools
~~~~~~~~~~~~~~~~~~

+------+----------------+-----------------+-------------+------------+
| Tool | Interactivity  | Configuration   | Monitoring  | Usability  |
+======+================+=================+=============+============+
| *    | ❌ Batch only  | ⚠️ Config files | ❌ Log      | ⚠️ Expert  |
| *Tra |                |                 | files only  | users      |
| diti |                |                 |             |            |
| onal |                |                 |             |            |
| C    |                |                 |             |            |
| LI** |                |                 |             |            |
+------+----------------+-----------------+-------------+------------+
| *    | ✅             | ✅ **Live       | ✅          | ✅         |
| *Ope | *              | editing**       | **          | **User-    |
| nHCS | *Interactive** |                 | Real-time** | friendly** |
| T    |                |                 |             |            |
| UI** |                |                 |             |            |
+------+----------------+-----------------+-------------+------------+

Performance Characteristics
---------------------------

Resource Usage
~~~~~~~~~~~~~~

.. code:: python

   # Terminal interface characteristics:
   Memory Usage: Lightweight compared to desktop GUIs
   CPU Usage: Low idle, moderate during updates
   Network: Minimal (text-based updates only)
   Latency: Responsive over SSH connections

Scalability
~~~~~~~~~~~

.. code:: python

   # Handles large-scale operations:
   ✅ 100GB+ dataset monitoring
   ✅ Multi-GPU resource tracking
   ✅ Thousands of pipeline steps
   ✅ Real-time log streaming
   ✅ Concurrent user sessions

Future Enhancements
-------------------

Planned Features
~~~~~~~~~~~~~~~~

.. code:: python

   # Roadmap for TUI improvements:
   ├── Advanced Visualizations
   │   ├── ASCII-based image previews
   │   ├── Progress bars with ETA
   │   └── Resource usage graphs
   ├── Collaboration Features
   │   ├── Multi-user editing
   │   ├── Session sharing
   │   └── Real-time collaboration
   ├── Automation Integration
   │   ├── Workflow scheduling
   │   ├── Batch job management
   │   └── CI/CD integration
   └── Mobile Support
       ├── Responsive layouts
       ├── Touch-friendly navigation
       └── Mobile-optimized workflows

Plugin Architecture
~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Extensible widget system:
   class CustomWidget(Widget):
       """User-defined TUI widget."""
       
       def compose(self) -> ComposeResult:
           yield Static("Custom functionality")
       
       def on_mount(self):
           """Register with TUI system."""
           self.app.register_widget(self)

Technical Implementation
------------------------

Event System
~~~~~~~~~~~~

.. code:: python

   # Reactive event handling:
   class PipelineEditor(Widget):
       def on_button_pressed(self, event: Button.Pressed) -> None:
           """Handle button press events."""
           if event.button.id == "add_step":
               self.add_new_step()
           elif event.button.id == "delete_step":
               self.delete_selected_step()
       
       def on_selection_changed(self, event: SelectionList.SelectionChanged) -> None:
           """Handle selection changes."""
           self.selected_step = event.selection
           self.update_step_details()

Validation Integration
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Real-time validation:
   def validate_pipeline_step(self, step_data):
       """Validate step configuration in real-time."""
       try:
           # Use OpenHCS validation services
           result = ValidationService.validate_step(step_data)
           self.update_validation_status(step_data.id, result)
       except Exception as e:
           self.show_validation_error(step_data.id, str(e))

This TUI system represents a paradigm shift in scientific computing
interfaces - providing production-grade functionality in a
terminal-native environment that works anywhere researchers need to
process data.

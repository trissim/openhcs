Declarative Window System
=========================

**Centralized window management with WindowManager integration and declarative specifications.**

*Status: STABLE*
*Module: openhcs.pyqt_gui*

Overview
--------

The Declarative Window System provides a **clean, centralized approach** to managing application windows in OpenHCS PyQt6 GUI. It replaces the ad-hoc floating window dictionary pattern with a declarative specification system integrated with the WindowManager singleton.

**The Problem**: The original implementation used a ``floating_windows: Dict[str, QDialog]`` dictionary in ``main.py`` with hardcoded window creation logic scattered throughout. This created tight coupling between the main window and window implementations, making it difficult to:

- Track window lifecycle consistently
- Ensure singleton-per-scope behavior
- Share window management patterns across the application
- Test window creation in isolation

**The Solution**: A declarative specification pattern where:

1. **WindowSpec** defines window configuration (ID, title, class, initialization behavior)
2. **ManagedWindow** classes encapsulate window setup and signal connections
3. **WindowManager** handles singleton enforcement and lifecycle management
4. **MainWindow** uses declarative specs instead of hardcoded creation logic

Key Architectural Patterns
--------------------------

WindowSpec Declaration Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Hardcoded window creation scattered across main.py.

**Solution**: Centralized declarative specifications in ``_get_window_specs()``.

.. code-block:: python

   def _get_window_specs(self) -> dict[str, WindowSpec]:
       """Return declarative window specifications."""
       return {
           "plate_manager": WindowSpec(
               window_id="plate_manager",
               title="Plate Manager",
               window_class=PlateManagerWindow,
               initialize_on_startup=True,
           ),
           "pipeline_editor": WindowSpec(
               window_id="pipeline_editor",
               title="Pipeline Editor",
               window_class=PipelineEditorWindow,
           ),
           # ... more windows
       }

**Key Points**:

- Window configuration is **data**, not code
- Window ID is the canonical identifier
- Window class encapsulates all setup logic
- ``initialize_on_startup`` for windows needed immediately

ManagedWindow Encapsulation Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Window setup logic mixed with main window concerns.

**Solution**: Each window type has its own ManagedWindow class.

.. code-block:: python

   class PlateManagerWindow(QDialog):
       """Window wrapper for PlateManagerWidget."""

       def __init__(self, main_window, service_adapter):
           super().__init__(main_window)
           self.main_window = main_window
           self.service_adapter = service_adapter
           self.setWindowTitle("Plate Manager")
           self.setModal(False)
           self.resize(600, 400)

           # Setup UI
           from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
           layout = QVBoxLayout(self)
           self.widget = PlateManagerWidget(service_adapter, ...)
           layout.addWidget(self.widget)

           # Setup connections
           self._setup_connections()

       def _setup_connections(self):
           """Connect signals to main window and other windows."""
           self.widget.global_config_changed.connect(...)
           # ... more connections

**Key Points**:

- Window is responsible for its own setup
- Main window only needs to call constructor
- Signal connections encapsulated in the window class
- Easy to test in isolation

WindowManager Integration Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Manual singleton enforcement and window tracking.

**Solution**: Delegate to WindowManager singleton service.

.. code-block:: python

   def show_window(self, window_id: str) -> None:
       """Show window using WindowManager."""
       factory = self._create_window_factory(window_id)
       window = WindowManager.show_or_focus(window_id, factory)
       self._ensure_flash_overlay(window)

   def _create_window_factory(self, window_id: str) -> Callable[[], QDialog]:
       """Create factory function for a window."""
       spec = self.window_specs[window_id]

       def factory() -> QDialog:
           return spec.window_class(self, self.service_adapter)

       return factory

**Key Points**:

- Factory pattern defers creation until needed
- WindowManager enforces singleton-per-scope
- Flash overlay integration for visual feedback
- No manual window tracking needed

Migration from Floating Windows
--------------------------------

Before: Floating Window Dictionary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Before: Hardcoded creation in main.py

   class OpenHCSMainWindow(QMainWindow):
       def __init__(self):
           super().__init__()
           self.floating_windows: Dict[str, QDialog] = {}

       def show_plate_manager(self):
           if "plate_manager" not in self.floating_windows:
               window = QDialog(self)
               window.setWindowTitle("Plate Manager")
               window.setModal(False)
               window.resize(600, 400)

               layout = QVBoxLayout(window)
               from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
               widget = PlateManagerWidget(self.service_adapter, ...)
               layout.addWidget(widget)

               # Setup connections inline (messy!)
               widget.global_config_changed.connect(...)
               widget.progress_started.connect(...)
               # ... many more connections

               self.floating_windows["plate_manager"] = window

           self.floating_windows["plate_manager"].show()
           self.floating_windows["plate_manager"].raise_()
           self.floating_windows["plate_manager"].activateWindow()

**Problems**:

- Window creation logic scattered across methods
- Signal connections mixed with UI setup
- Manual window tracking dictionary
- No consistent singleton enforcement
- Hard to test (requires full main window)

After: Declarative Specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: After: Declarative specs with WindowManager

   class OpenHCSMainWindow(QMainWindow):
       def __init__(self):
           super().__init__()
           self.window_specs = self._get_window_specs()

       def show_window(self, window_id: str) -> None:
           """Show any window using WindowManager."""
           factory = self._create_window_factory(window_id)
           window = WindowManager.show_or_focus(window_id, factory)
           self._ensure_flash_overlay(window)

       def show_plate_manager(self):
           """Show plate manager - delegates to WindowManager."""
           self.show_window("plate_manager")

**Benefits**:

- One method shows any window
- WindowManager handles singleton enforcement
- Window classes handle their own setup
- Easy to add new window types (just add to specs)
- Testable: window classes can be tested without main window

Window Configuration Service
-----------------------------

The ``window_config.py`` module provides the data structures for declarative window specifications.

WindowSpec Dataclass
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass(frozen=True)
   class WindowSpec:
       """
       Declarative specification for an application window.

       Centralizes window configuration (widget, title, initialization behavior).
       """
       window_id: str           # Unique identifier (e.g., "plate_manager")
       title: str               # Window title shown to user
       window_class: Type[ManagedWindow]  # Class to instantiate
       initialize_on_startup: bool = False  # Create on startup (hidden)?

ManagedWindow Base Class
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ManagedWindow(QDialog):
       """Base class for managed application windows."""

       def __init__(self, main_window, service_adapter):
           super().__init__(main_window)
           self.main_window = main_window
           self.service_adapter = service_adapter
           self.setup_ui()
           self.setup_connections()

       def setup_ui(self):
           """Setup window UI. Subclasses implement this."""
           pass

       def setup_connections(self):
           """Setup signal connections. Subclasses implement this."""
           pass

Window Implementations
----------------------

The ``managed_windows.py`` module contains all ManagedWindow implementations.

PlateManagerWindow
~~~~~~~~~~~~~~~~~~~

Wraps ``PlateManagerWidget`` with proper signal connections to main window and pipeline editor.

.. code-block:: python

   class PlateManagerWindow(QDialog):
       def __init__(self, main_window, service_adapter):
           # ... setup code ...
           self._setup_connections()

       def _setup_connections(self):
           # Connect to main window
           self.widget.global_config_changed.connect(
               lambda: self.main_window.on_config_changed(...)
           )

           # Connect progress signals to status bar
           if hasattr(self.main_window, "status_bar"):
               self._setup_progress_signals()

           # Connect to pipeline editor (bidirectional)
           self._connect_to_pipeline_editor()

PipelineEditorWindow
~~~~~~~~~~~~~~~~~~~~

Wraps ``PipelineEditorWidget`` with connections to plate manager.

ImageBrowserWindow
~~~~~~~~~~~~~~~~~~~

Wraps ``ImageBrowserWidget`` with orchestrator updates from plate selection.

LogViewerWindowWrapper
~~~~~~~~~~~~~~~~~~~~~~~

Wraps ``LogViewerWindow`` for log file viewing.

ZMQServerManagerWindow
~~~~~~~~~~~~~~~~~~~~~~~

Wraps ``ZMQServerManagerWidget`` for ZMQ server monitoring.

Integration with BaseFormDialog
--------------------------------

Dialog windows (ConfigWindow, DualEditorWindow, etc.) inherit from ``BaseFormDialog`` from the pyqt-reactive submodule, which provides:

- **Singleton-per-scope behavior**: Only one window per scope_id
- **ObjectState integration**: Automatic save/restore on accept/reject
- **WindowManager registration**: Automatic lifecycle management
- **Change detection**: Optional automatic change detection support

.. code-block:: python

   from pyqt_reactive.widgets.shared import BaseFormDialog

   class ConfigWindow(BaseFormDialog):
       def __init__(self, config_class, initial_config, parent, scope_id):
           super().__init__(parent)
           self.scope_id = scope_id  # Enables singleton behavior
           # ... setup ...

Benefits and Impact
--------------------

Code Reduction
~~~~~~~~~~~~~~

**Eliminated**:

- ~200 lines of hardcoded window creation in main.py
- Floating window dictionary management
- Manual singleton enforcement logic
- Duplicated window setup patterns

**Added**:

- ~50 lines in window_config.py (one-time)
- ~150 lines in managed_windows.py (all windows)
- **Net reduction**: ~100+ lines

Maintainability
~~~~~~~~~~~~~~~~

**Single Source of Truth**: Window specifications are data, not code.

- Add new window: Add one entry to ``_get_window_specs()``
- Modify window: Edit one class in ``managed_windows.py``
- Remove window: Delete one spec and one class

**Testability**: Window classes can be tested independently.

.. code-block:: python

   # Test without full main window
   window = PlateManagerWindow(mock_main_window, mock_service_adapter)
   assert window.windowTitle() == "Plate Manager"
   # Test signal connections...

Consistency
~~~~~~~~~~~~

All windows now follow the same pattern:

1. **ManagedWindow** for simple windows (PlateManager, etc.)
2. **BaseFormDialog** for config dialogs (from pyqt-reactive)
3. **WindowManager** for singleton enforcement (all windows)

Extensibility
~~~~~~~~~~~~~~

**Adding a new window type**:

1. Create ``MyWindow(QDialog)`` class in ``managed_windows.py``
2. Add spec to ``_get_window_specs()`` in ``main.py``
3. Done! WindowManager handles the rest.

Related Patterns
-----------------

**See Also**:

- :doc:`window_manager_usage` - WindowManager API reference
- :doc:`service-layer-architecture` - Service layer patterns
- :doc:`ui_services_architecture` - UI service patterns

**Integration Points**:

- **WindowManager** (pyqt-reactive): Singleton window registry
- **BaseFormDialog** (pyqt-reactive): Base class for config dialogs
- **ObjectState**: State management for config dialogs
- **ServiceAdapter**: Provides services to windows

Summary
-------

The Declarative Window System demonstrates how moving from imperative window creation to declarative specifications improves:

- **Maintainability**: Centralized configuration, clear separation of concerns
- **Testability**: Window classes testable in isolation
- **Consistency**: All windows follow the same patterns
- **Extensibility**: New windows require minimal boilerplate

By leveraging the WindowManager singleton and BaseFormDialog from pyqt-reactive, OpenHCS achieves robust window management with minimal custom code.

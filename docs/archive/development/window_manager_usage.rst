Window Manager Usage Guide
=========================

Overview
--------

``WindowManager`` provides singleton window management with navigation support for inheritance tracking.

Key features:

- One window per ``scope_id`` (prevents duplicates).
- Auto-cleanup on close (no manual unregistration).
- Navigation API for focusing and scrolling to fields.
- Fail-loud on stale references.

ServiceRegistry Integration
-------------------------

Window handlers use ``ServiceRegistry`` for widget access instead of manual traversal:

.. code-block:: python

    from pyqt_reactive.services import ServiceRegistry
    from my_widgets import PlateManagerWidget

    def create_plate_config_window(scope_id: str, object_state=None):
        # Get plate manager from ServiceRegistry
        plate_manager = ServiceRegistry.get(PlateManagerWidget)
        if not plate_manager:
            return None

        window = ConfigWindow(
            config_class=PipelineConfig,
            current_config=orchestrator.pipeline_config,
            scope_id=scope_id,
        )
        window.show()
        return window

Basic Usage
-----------

Show or focus a window via a factory callback:

.. code-block:: python

   from openhcs.pyqt_gui.services.window_manager import WindowManager

   # Define factory function that creates the window
   def create_config_window():
       return ConfigWindow(
           config_class=PipelineConfig,
           initial_config=current_config,
           parent=self,
           on_save_callback=self._on_config_saved,
           scope_id="plate1",
       )

   # Show window (creates new or focuses existing)
   window = WindowManager.show_or_focus("plate1", create_config_window)

Before/after behavior:

.. code-block:: python
   :caption: Before WindowManager (duplicates)

   config_window = ConfigWindow(...)
   config_window.show()

.. code-block:: python
   :caption: After WindowManager (singleton per scope)

   WindowManager.show_or_focus(scope_id, lambda: ConfigWindow(...))

Migration Examples
------------------

PlateManager: Edit Config Button
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :caption: Before

   def action_edit_config(self):
       """Edit configuration for selected plate."""
       config_window = ConfigWindow(
           config_class=PipelineConfig,
           initial_config=self._get_current_config(),
           parent=self,
           on_save_callback=self._on_config_saved,
           scope_id=str(self.selected_plate_path),
       )
       config_window.show()

.. code-block:: python
   :caption: After

   def action_edit_config(self):
       """Edit configuration for selected plate."""
       from openhcs.pyqt_gui.services.window_manager import WindowManager

       scope_id = str(self.selected_plate_path)

       def create_window():
           return ConfigWindow(
               config_class=PipelineConfig,
               initial_config=self._get_current_config(),
               parent=self,
               on_save_callback=self._on_config_saved,
               scope_id=scope_id,
           )

       WindowManager.show_or_focus(scope_id, create_window)

PipelineEditor: Edit Step Button
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :caption: Before

   def action_edit(self):
       """Edit selected step."""
       step_window = DualEditorWindow(
           editing_step=selected_step,
           parent=self,
           on_save_callback=self._on_step_saved,
           scope_id=f"{self.scope_id}::step_{index}",
       )
       step_window.show()

.. code-block:: python
   :caption: After

   def action_edit(self):
       """Edit selected step."""
       from openhcs.pyqt_gui.services.window_manager import WindowManager

       scope_id = f"{self.scope_id}::step_{index}"

       def create_window():
           return DualEditorWindow(
               editing_step=selected_step,
               parent=self,
               on_save_callback=self._on_step_saved,
               scope_id=scope_id,
           )

       WindowManager.show_or_focus(scope_id, create_window)

Navigation API (Future: Inheritance Tracking)
---------------------------------------------

Basic navigation calls:

.. code-block:: python

   # Focus window without navigation
   WindowManager.focus_and_navigate(scope_id="plate1")

   # Focus window and navigate to item
   WindowManager.focus_and_navigate(
       scope_id="plate1",
       item_id="step_3",  # Select step 3 in list
   )

   # Focus window and navigate to field
   WindowManager.focus_and_navigate(
       scope_id="plate1",
       field_path="well_filter_config.well_filter",  # Scroll to this field
   )

Implementing navigation in windows (optional methods):

.. code-block:: python

   class ConfigWindow(BaseFormDialog):
       """Config window with navigation support."""

       def select_and_scroll_to_field(self, field_path: str):
           """Navigate to field in tree/form (called by WindowManager).

           Args:
               field_path: Dotted path like "well_filter_config.well_filter"
           """
           # 1. Find tree item matching field_path
           tree_item = self.tree_helper.find_item_by_path(self.tree_widget, field_path)
           if not tree_item:
               return

           # 2. Expand parent nodes
           parent = tree_item.parent()
           while parent:
               parent.setExpanded(True)
               parent = parent.parent()

           # 3. Select and scroll to item
           self.tree_widget.setCurrentItem(tree_item)
           self.tree_widget.scrollToItem(tree_item)

           # 4. Flash to highlight
           self.queue_flash(field_path)


   class DualEditorWindow(BaseFormDialog):
       """Step editor with navigation support."""

       def select_and_scroll_to_item(self, item_id: str):
           """Navigate to tab/section for item (called by WindowManager).

           Args:
               item_id: Item identifier (e.g., tab index, field name)
           """
           # Switch to tab containing the item
           tab_index = self._get_tab_for_item(item_id)
           if tab_index >= 0:
               self.tab_widget.setCurrentIndex(tab_index)

           # Scroll to widget
           widget = self._get_widget_for_item(item_id)
           if widget:
               widget.setFocus()

Future: Inheritance Tracking
----------------------------

Show the source window and scroll to it when the user clicks an inherited field:

.. code-block:: python

   class InheritanceTreeWidget(QTreeWidget):
       """Tree widget showing field inheritance."""

       def on_field_clicked(self, field_path: str, source_scope: str):
           """User clicked field - show source window and scroll to it.

           Args:
               field_path: Field that was clicked
               source_scope: Scope where field is defined (not inherited)
           """
           from openhcs.pyqt_gui.services.window_manager import WindowManager

           # Try to focus existing window and navigate
           if WindowManager.focus_and_navigate(source_scope, field_path=field_path):
               return  # Window exists and navigated

           # Window not open - create it and navigate
           def create_window():
               return ConfigWindow(
                   config_class=self._get_config_class(source_scope),
                   initial_config=self._get_config(source_scope),
                   scope_id=source_scope,
               )

           window = WindowManager.show_or_focus(source_scope, create_window)

           # Navigate after window is shown (give Qt time to render)
           QTimer.singleShot(100, lambda: window.select_and_scroll_to_field(field_path))

Utility Methods
---------------

.. code-block:: python

   # Check if window is open
   if WindowManager.is_open("plate1"):
       print("Config window already open for plate1")

   # Get all open window scopes
   open_scopes = WindowManager.get_open_scopes()
   print(f"Open windows: {open_scopes}")

   # Programmatically close window
   WindowManager.close_window("plate1")

Architecture Notes
------------------

Auto-cleanup
^^^^^^^^^^^^

Windows are automatically unregistered when closed (no manual cleanup needed):

.. code-block:: python

   # WindowManager hooks into closeEvent
   window.closeEvent = lambda event: (
       unregister_from_registry(),
       call_original_closeEvent(event),
   )

Fail-Loud on Stale References
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a window is deleted but still in the registry, a ``RuntimeError`` is raised and the stale reference is cleaned up:

.. code-block:: python

   try:
       window.isVisible()  # Test if still valid
   except RuntimeError:
       # Window deleted - clean up stale reference
       del WindowManager._scoped_windows[scope_id]

Duck Typing for Navigation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigation methods are optional—windows implement them if supported:

.. code-block:: python

   # WindowManager checks if method exists before calling
   if hasattr(window, "select_and_scroll_to_field"):
       window.select_and_scroll_to_field(field_path)

Benefits
--------

1. Prevents duplicate windows: only one config window per plate.
2. Better UX: focusing brings existing window to front.
3. Auto-cleanup: no memory leaks from forgotten references.
4. Extensible: navigation API ready for inheritance tracking.
5. Fail-loud: catches deleted windows early.
6. Fits OpenHCS patterns: similar to ``ObjectStateRegistry`` for states.

Declarative Window Specifications
---------------------------------

For simple windows that don't require the full BaseFormDialog machinery, use the **WindowSpec** pattern with **ManagedWindow** classes.

WindowSpec Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Define window specifications declaratively in your main window:

.. code-block:: python

   from openhcs.pyqt_gui.services.window_config import WindowSpec
   
   class OpenHCSMainWindow(QMainWindow):
       def __init__(self):
           super().__init__()
           self.window_specs = self._get_window_specs()
       
       def _get_window_specs(self) -> dict[str, WindowSpec]:
           """Return declarative window specifications."""
           from openhcs.pyqt_gui.windows.managed_windows import (
               PlateManagerWindow,
               PipelineEditorWindow,
               ImageBrowserWindow,
           )
           
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
               "image_browser": WindowSpec(
                   window_id="image_browser",
                   title="Image Browser",
                   window_class=ImageBrowserWindow,
               ),
           }

ManagedWindow Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a ManagedWindow class for each window type:

.. code-block:: python

   # In openhcs/pyqt_gui/windows/managed_windows.py
   
   class PlateManagerWindow(QDialog):
       """Window wrapper for PlateManagerWidget."""
       
       def __init__(self, main_window, service_adapter):
           super().__init__(main_window)
           self.main_window = main_window
           self.service_adapter = service_adapter
           self.setWindowTitle("Plate Manager")
           self.setModal(False)
           self.resize(600, 400)
           
           # Create and add widget
           from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
           layout = QVBoxLayout(self)
           self.widget = PlateManagerWidget(
               self.service_adapter,
               self.service_adapter.get_current_color_scheme(),
           )
           layout.addWidget(self.widget)
           
           # Setup signal connections
           self._setup_connections()
       
       def _setup_connections(self):
           """Connect signals to main window and other windows."""
           # Connect to main window
           self.widget.global_config_changed.connect(
               lambda: self.main_window.on_config_changed(
                   self.service_adapter.get_global_config()
               )
           )
           
           # Connect progress signals to status bar
           if hasattr(self.main_window, "status_bar"):
               self._setup_progress_signals()
           
           # Connect to other windows via WindowManager
           self._connect_to_pipeline_editor()

Showing Windows with WindowManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use WindowManager with a factory function:

.. code-block:: python

   from pyqt_reactive.services.window_manager import WindowManager
   
   def show_window(self, window_id: str) -> None:
       """Show window using WindowManager."""
       factory = self._create_window_factory(window_id)
       window = WindowManager.show_or_focus(window_id, factory)
       
       # Optional: Initialize hidden for startup windows
       spec = self.window_specs[window_id]
       if spec.initialize_on_startup and window_id == "log_viewer":
           window.hide()
       
       # Ensure flash overlay is ready
       self._ensure_flash_overlay(window)
   
   def _create_window_factory(self, window_id: str) -> Callable[[], QDialog]:
       """Create factory function for a window."""
       spec = self.window_specs[window_id]
       
       def factory() -> QDialog:
           return spec.window_class(self, self.service_adapter)
       
       return factory

Window-to-Window Communication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ManagedWindows can communicate via WindowManager:

.. code-block:: python

   def _connect_to_pipeline_editor(self):
       """Connect plate manager to pipeline editor."""
       from pyqt_reactive.services.window_manager import WindowManager
       
       # Get pipeline window if it exists
       pipeline_window = WindowManager._scoped_windows.get("pipeline_editor")
       if pipeline_window:
           # Access the widget and connect signals
           pipeline_widget = pipeline_window.widget
           self.widget.plate_selected.connect(pipeline_widget.set_current_plate)
           self.widget.orchestrator_config_changed.connect(
               pipeline_widget.on_orchestrator_config_changed
           )

When to Use Each Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~

**Use WindowSpec + ManagedWindow for:**

- Simple container windows (Plate Manager, Image Browser, etc.)
- Windows that don't need ObjectState/form management
- Windows that wrap existing widgets
- Quick prototyping

**Use BaseFormDialog for:**

- Configuration dialogs (ConfigWindow, DualEditorWindow)
- Forms with ParameterFormManager
- Windows that need ObjectState integration
- Complex multi-tab dialogs

Comparison
~~~~~~~~~~~

.. list-table:: Window Management Patterns
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - WindowSpec + ManagedWindow
     - BaseFormDialog
   * - Complexity
     - Low
     - High
   * - ObjectState
     - Manual
     - Automatic
   * - Form Management
     - No
     - Built-in
   * - Singleton Enforcement
     - Via WindowManager
     - Via WindowManager
   * - Save/Restore
     - Manual
     - Automatic
   * - Best For
     - Simple containers
     - Complex config dialogs

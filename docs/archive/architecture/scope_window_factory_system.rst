Scope Window Factory System
===========================

Overview
--------

The scope window factory system provides a pattern-based handler mechanism for creating windows based on ``scope_id`` values. This system replaces monolithic factory classes with a flexible registration pattern that allows adding new window types without modifying core code.

**Module**: ``pyqt_reactive.services.scope_window_factory``

Core Components
---------------

ScopeWindowRegistry
~~~~~~~~~~~~~~~~~~

Central registry that maps regex patterns to handler functions.

.. code-block:: python

    from pyqt_reactive.services.scope_window_factory import ScopeWindowRegistry

    # Register handlers at application startup
    ScopeWindowRegistry.register_handler(
        pattern=r"^/path/to/plate$",
        handler=create_plate_config_window
    )

Scope ID Pattern Matching
~~~~~~~~~~~~~~~~~~~~~~~~

Window creation is triggered by matching ``scope_id`` strings against registered patterns:

- **Global config**: ``""`` (empty string)
- **Plate configs**: ``/path/to/plate`` (no ``::`` separator)
- **Plate list root**: ``__plates__`` (special root state)
- **Step editors**: ``/path/to/plate::step_N`` (``::step_N`` suffix)
- **Function scopes**: ``/path/to/plate::step_N::func_M`` (additional ``::func_M``)

Registration Order
~~~~~~~~~~~~~~~~~

Patterns are evaluated in registration order. More specific patterns should be registered first:

.. code-block:: python

    def register_openhcs_window_handlers():
        # Order matters - more specific patterns first

        # Step/function editors (match ::step_N or ::step_N::func_M)
        ScopeWindowRegistry.register_handler(
            pattern=r"^.*::step_\d+(::func_\d+)?$",
            handler=_create_step_editor_window
        )

        # Plate configs (match /path - no :: separator)
        ScopeWindowRegistry.register_handler(
            pattern=r"^/[^:]*$",
            handler=_create_plate_config_window
        )

        # Plates root list
        ScopeWindowRegistry.register_handler(
            pattern=r"^__plates__$",
            handler=_create_plates_root_window
        )

        # Global config (empty string)
        ScopeWindowRegistry.register_handler(
            pattern=r"^$",
            handler=_create_global_config_window
        )

Handler Functions
----------------

Handler functions create and show windows for a given ``scope_id`` and optional ``object_state``.

**Signature**:

.. code-block:: python

    def handler(scope_id: str, object_state=None) -> Optional[QWidget]:
        """Create and show a window for the given scope_id.

        Args:
            scope_id: The scope identifier (pattern-matched)
            object_state: Optional ObjectState instance (for time-travel)

        Returns:
            QWidget: The created window, or None if no window should be created
        """

**Example Handler**:

.. code-block:: python

    from pyqt_reactive.services.scope_window_factory import ScopeWindowRegistry

    def _create_global_config_window(scope_id: str, object_state=None) -> Optional[QWidget]:
        """Create GlobalPipelineConfig editor window."""
        from openhcs.pyqt_gui.windows.config_window import ConfigWindow
        from openhcs.core.config import GlobalPipelineConfig
        from openhcs.config_framework.global_config import (
            get_current_global_config,
            set_global_config_for_editing,
        )

        current_config = (
            get_current_global_config(GlobalPipelineConfig) or GlobalPipelineConfig()
        )

        def handle_save(new_config):
            set_global_config_for_editing(GlobalPipelineConfig, new_config)

        window = ConfigWindow(
            config_class=GlobalPipelineConfig,
            current_config=current_config,
            on_save_callback=handle_save,
            scope_id=scope_id,
        )
        window.show()
        window.raise_()
        window.activateWindow()
        return window

Window Creation
---------------

Via WindowFactory
~~~~~~~~~~~~~~~~~

Use the generic ``WindowFactory`` class to create windows:

.. code-block:: python

    from pyqt_reactive.services import WindowFactory

    # Create window for a scope
    window = WindowFactory.create_window_for_scope(scope_id, object_state)

Via WindowManager (with Time-Travel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For windows that should be managed as singletons:

.. code-block:: python

    from openhcs.pyqt_gui.services.window_manager import WindowManager

    def show_config_window():
        def factory():
            return ConfigWindow(...)

        window = WindowManager.show_or_focus(scope_id, factory)

Time-Travel Integration
-----------------------

The scope window factory integrates with time-travel through the ``object_state`` parameter:

1. **Time-Travel Reopens Windows**: When time-traveling to a dirty state, the system:
   - Calls ``WindowFactory.create_window_for_scope(scope_id, object_state)``
   - Handler receives the dirty ``object_state`` for proper reconstruction
   - Window is shown and focused

2. **ObjectState for Context**: Handlers use ``object_state`` to reconstruct UI state:

.. code-block:: python

    def _create_step_editor_window(scope_id: str, object_state=None):
        """Create step editor with time-travel support."""
        # Get step from object_state (if provided)
        if object_state:
            step = object_state.object_instance
        else:
            # Find step by token (provenance navigation)
            step = _find_step_by_token(plate_path, step_token)

        window = DualEditorWindow(
            step_data=step,
            orchestrator=orchestrator,
            scope_id=scope_id,
        )
        window.show()
        return window

Registration in OpenHCS
------------------------

Handlers are registered during application initialization in ``main.py``:

.. code-block:: python

    from openhcs.pyqt_gui.services.window_handlers import (
        register_openhcs_window_handlers,
    )

    class OpenHCSMainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            # ... other initialization ...

            # Register OpenHCS window handlers
            register_openhcs_window_handlers()

Handler Implementation Guide
--------------------------

Plate Config Handler
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def _create_plate_config_window(scope_id: str, object_state=None):
        from openhcs.pyqt_gui.windows.config_window import ConfigWindow
        from openhcs.core.config import PipelineConfig
        from pyqt_reactive.services import ServiceRegistry
        from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget

        # Get plate manager from ServiceRegistry
        plate_manager = ServiceRegistry.get(PlateManagerWidget)
        if not plate_manager:
            return None

        orchestrator = ObjectStateRegistry.get_object(scope_id)
        if not orchestrator:
            return None

        window = ConfigWindow(
            config_class=PipelineConfig,
            current_config=orchestrator.pipeline_config,
            on_save_callback=None,  # ObjectState handles save
            scope_id=scope_id,
        )
        window.show()
        window.raise_()
        window.activateWindow()
        return window

Step Editor Handler
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def _create_step_editor_window(scope_id: str, object_state=None):
        from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow

        parts = scope_id.split("::")
        if len(parts) < 2:
            return None

        plate_path = parts[0]
        step_token = parts[1]
        is_function_scope = len(parts) >= 3

        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if not orchestrator:
            return None

        # Get step from object_state (time-travel) or find by token
        step = None
        if object_state:
            step_state = ObjectStateRegistry.get_by_scope(step_scope_id)
            step = step_state.object_instance if step_state else None
        else:
            step = _find_step_by_token(plate_manager, plate_path, step_token)

        if not step:
            return None

        window = DualEditorWindow(
            step_data=step,
            is_new=False,
            on_save_callback=None,
            orchestrator=orchestrator,
            parent=None,
        )

        if is_function_scope and window.tab_widget:
            window.tab_widget.setCurrentIndex(1)

        window.show()
        window.raise_()
        window.activateWindow()
        return window

Special Case: No Window
~~~~~~~~~~~~~~~~~~~~~~~

Some scopes (like ``__plates__``) represent state without a window:

.. code-block:: python

    def _create_plates_root_window(scope_id: str, object_state=None):
        """Root plate list state - no window to create."""
        logger.debug(f"[WINDOW_FACTORY] Skipping window creation for __plates__ scope")
        return None

Best Practices
--------------

Pattern Specificity
~~~~~~~~~~~~~~~~~~~

- **Specific first**: Register patterns with ``::`` separators before generic paths
- **Global last**: Register ``^$`` (empty string) handler last
- **Test order**: Verify patterns match expected scopes in correct order

Error Handling
~~~~~~~~~~~~~~

- Return ``None`` when window cannot be created
- Log warnings for missing dependencies
- Validate scope_id format before processing

Window Display
~~~~~~~~~~~~~~

Always call these three methods after creating a window:

.. code-block:: python

    window.show()
    window.raise_()
    window.activateWindow()

Scope ID Design
~~~~~~~~~~~~~~~~

- Use filesystem-like paths: ``/path/to/plate``
- Use ``::`` for hierarchy: ``plate::step_N::func_M``
- Keep identifiers stable (don't change scope IDs for existing data)

Integration Points
------------------

WindowManager Integration
~~~~~~~~~~~~~~~~~~~~~~~~

``WindowManager`` calls ``WindowFactory.create_window_for_scope()`` internally:

.. code-block:: python

    # WindowManager uses scope window factory for time-travel
    window = WindowFactory.create_window_for_scope(scope_id, state)

Provenance Navigation
~~~~~~~~~~~~~~~~~~~~~~

Scope IDs appear in provenance tracking. Clicking a provenance entry:
1. Calls ``WindowManager.focus_and_navigate(scope_id)``
2. If window doesn't exist, calls ``WindowFactory.create_window_for_scope()``
3. Handler creates window and navigates to the field

See Also
--------

- :doc:`/architecture/service_registry_integration` - ServiceRegistry pattern
- :doc:`/development/window_manager_usage` - Window management with singletons
- :doc:`/architecture/time_travel_system` - Time-travel architecture

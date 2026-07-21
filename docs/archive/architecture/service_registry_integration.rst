Service Registry Integration
==========================

Overview
--------

The ServiceRegistry provides centralized service and widget management, eliminating circular dependencies and simplifying component discovery. Widgets and services register themselves at creation time, making them available throughout the application without manual tracking.

**Module**: ``pyqt_reactive.services.service_registry``

Core Concepts
-------------

Service Registration
~~~~~~~~~~~~~~~~~~

Services register themselves via the ``AutoRegisterServiceMixin``:

.. code-block:: python

    from pyqt_reactive.services import ServiceRegistry, AutoRegisterServiceMixin

    class PlateManagerWidget(QWidget, AutoRegisterServiceMixin):
        """Plate manager widget auto-registers with ServiceRegistry."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # AutoRegisterServiceMixin registers this instance automatically

Service Resolution
~~~~~~~~~~~~~~~~~~~

Services are retrieved by type:

.. code-block:: python

    from pyqt_reactive.services import ServiceRegistry
    from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget

    # Get plate manager widget
    plate_manager = ServiceRegistry.get(PlateManagerWidget)

    if plate_manager:
        # Use the widget
        plate_manager.refresh_plate_list()

AutoRegisterServiceMixin
------------------------

Mixin for automatic service registration.

**Usage**:

.. code-block:: python

    class MyWidget(QWidget, AutoRegisterServiceMixin):
        """Widget that auto-registers with ServiceRegistry."""

        def __init__(self):
            super().__init__()
            # ServiceRegistry.set(MyWidget, self) called automatically

**Behind the scenes**:

.. code-block:: python

    class AutoRegisterServiceMixin:
        """Mixin to auto-register widgets with ServiceRegistry."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            ServiceRegistry.set(type(self), self)

ServiceRegistry API
-------------------

Register Services
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyqt_reactive.services import ServiceRegistry

    # Register a service instance
    ServiceRegistry.set(MyService, my_service_instance)

    # Register with explicit service key
    ServiceRegistry.set("my_service_key", my_service_instance)

Retrieve Services
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get by type (preferred)
    plate_manager = ServiceRegistry.get(PlateManagerWidget)

    # Get by key
    service = ServiceRegistry.get("my_service_key")

    # Get with default fallback
    manager = ServiceRegistry.get(PlateManagerWidget, None)

Check Registration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Check if service exists
    if ServiceRegistry.has(PlateManagerWidget):
        # Use it
        pass

    # Check with key
    if ServiceRegistry.has("my_service"):
        pass

Clear Services
~~~~~~~~~~~~~~

.. code-block:: python

    # Clear a specific service
    ServiceRegistry.clear(MyService)

    # Clear all services (use with caution)
    ServiceRegistry.clear_all()

Common Use Cases
----------------

Widget-to-Widget Communication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously required ``floating_windows`` dictionary or ``QApplication`` traversal:

.. code-block:: python

    # Before: Traversal through all windows
    for widget in QApplication.topLevelWidgets():
        if hasattr(widget, 'floating_windows'):
            plate_dialog = widget.floating_windows.get("plate_manager")
            if plate_dialog:
                plate_manager = plate_dialog.findChild(PlateManagerWidget)
                break

Now use ServiceRegistry:

.. code-block:: python

    # After: Direct lookup
    from pyqt_reactive.services import ServiceRegistry
    from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget

    plate_manager = ServiceRegistry.get(PlateManagerWidget)
    if plate_manager:
        # Connect signals
        self.plate_selected.connect(plate_manager.set_current_plate)

Window Handler Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handlers access widgets from ServiceRegistry:

.. code-block:: python

    def _create_plate_config_window(scope_id: str, object_state=None):
        from openhcs.pyqt_gui.windows.config_window import ConfigWindow
        from pyqt_reactive.services import ServiceRegistry
        from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget

        # Get plate manager from ServiceRegistry
        plate_manager = ServiceRegistry.get(PlateManagerWidget)
        if not plate_manager:
            logger.warning("Could not find PlateManager for plate config window")
            return None

        orchestrator = ObjectStateRegistry.get_object(scope_id)
        if not orchestrator:
            return None

        window = ConfigWindow(
            config_class=PipelineConfig,
            current_config=orchestrator.pipeline_config,
            scope_id=scope_id,
        )
        window.show()
        return window

Pipeline-to-Plate Manager Connection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Connect pipeline editor to plate manager via ServiceRegistry:

.. code-block:: python

    def _connect_pipeline_to_plate_manager(self, pipeline_widget):
        from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
        from pyqt_reactive.services import ServiceRegistry

        # Get plate manager from ServiceRegistry
        plate_manager_widget = ServiceRegistry.get(PlateManagerWidget)

        if plate_manager_widget:
            # Connect plate selection signal to pipeline editor
            plate_manager_widget.plate_selected.connect(
                pipeline_widget.set_current_plate
            )

            # Set current plate if already selected
            if plate_manager_widget.selected_plate_path:
                pipeline_widget.set_current_plate(
                    plate_manager_widget.selected_plate_path
                )

            logger.debug("Connected pipeline editor to plate manager")
        else:
            logger.warning("Could not find plate manager widget to connect")

Service Lifecycle
-----------------

Registration Timing
~~~~~~~~~~~~~~~~~~

Services are registered at widget creation time:

.. code-block:: python

    # In main.py
    self.plate_manager_widget = PlateManagerWidget(...)
    # PlateManagerWidget.__init__ calls ServiceRegistry.set(PlateManagerWidget, self)

    # Plate manager is now available immediately
    other_widget.connect_to_plate_manager()

Unregistration
~~~~~~~~~~~~~~

Widgets are unregistered automatically when destroyed (if subclassing QObject):

.. code-block:: python

    # ServiceRegistry hooks into object destruction
    def on_object_destroyed(self):
        service_type = self._registered_service_type
        ServiceRegistry.clear(service_type)

Singleton Pattern
~~~~~~~~~~~~~~~~~~

ServiceRegistry enforces one instance per service type:

.. code-block:: python

    # First registration
    service1 = MyService()
    ServiceRegistry.set(MyService, service1)

    # Second registration overwrites first
    service2 = MyService()
    ServiceRegistry.set(MyService, service2)  # Replaces service1

    # Only service2 is available
    retrieved = ServiceRegistry.get(MyService)
    assert retrieved is service2  # True

Thread Safety
-------------

ServiceRegistry is **not thread-safe** by default. All service operations should occur on the main GUI thread:

.. code-block:: python

    # CORRECT: Main thread
    def main_thread_function():
        service = ServiceRegistry.get(MyService)
        service.do_something()

    # INCORRECT: Background thread
    def background_thread_function():
        # This can cause race conditions
        service = ServiceRegistry.get(MyService)
        service.do_something()

If thread-safe access is needed, implement a wrapper:

.. code-block:: python

    from threading import Lock

    class ThreadSafeServiceRegistry:
        def __init__(self):
            self._services = {}
            self._lock = Lock()

        def get(self, service_type, default=None):
            with self._lock:
                return self._services.get(service_type, default)

        def set(self, service_type, instance):
            with self._lock:
                self._services[service_type] = instance

Best Practices
--------------

Use Type-Based Keys
~~~~~~~~~~~~~~~~~~~

Prefer class types over string keys:

.. code-block:: python

    # PREFERRED: Type-based
    ServiceRegistry.set(PlateManagerWidget, widget)
    manager = ServiceRegistry.get(PlateManagerWidget)

    # AVOID: String-based (unless necessary)
    ServiceRegistry.set("plate_manager", widget)
    manager = ServiceRegistry.get("plate_manager")

Check Before Use
~~~~~~~~~~~~~~~~

Always check if service exists:

.. code-block:: python

    manager = ServiceRegistry.get(PlateManagerWidget)
    if manager:
        manager.refresh()
    else:
        logger.warning("PlateManager not available")

Auto-Register Widgets
~~~~~~~~~~~~~~~~~~~~~

Use ``AutoRegisterServiceMixin`` for widgets:

.. code-block:: python

    # DO: Auto-register
    class PlateManagerWidget(QWidget, AutoRegisterServiceMixin):
        def __init__(self):
            super().__init__()
            # Registered automatically

    # AVOID: Manual registration
    class PlateManagerWidget(QWidget):
        def __init__(self):
            super().__init__()
            ServiceRegistry.set(PlateManagerWidget, self)  # Boilerplate

Avoid Circular Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

ServiceRegistry breaks dependency chains:

.. code-block:: python

    # BEFORE: Circular dependency
    # PipelineEditor needs PlateManager
    # PlateManager needs PipelineEditor
    # Both import each other → circular

    # AFTER: ServiceRegistry breaks the cycle
    # PipelineEditor imports PlateManagerWidget (type only)
    # PlateManagerWidget imports PipelineEditorWidget (type only)
    # Both resolve via ServiceRegistry.get() at runtime

Service Keys Design
~~~~~~~~~~~~~~~~~~

Design service keys with clear semantics:

.. code-block:: python

    # Use concrete widget/service types
    ServiceRegistry.set(PlateManagerWidget, widget)

    # For multiple instances, use distinct service classes
    class MainPlateManagerWidget(PlateManagerWidget): pass
    class SecondaryPlateManagerWidget(PlateManagerWidget): pass

    ServiceRegistry.set(MainPlateManagerWidget, widget1)
    ServiceRegistry.set(SecondaryPlateManagerWidget, widget2)

Migration from floating_windows
------------------------------

Before
~~~~~~~

.. code-block:: python

    # Main window tracks widgets manually
    class OpenHCSMainWindow(QMainWindow):
        def __init__(self):
            self.floating_windows = {}

        def create_plate_manager(self):
            window = PlateManagerWindow()
            self.floating_windows["plate_manager"] = window

        def get_plate_manager(self):
            if "plate_manager" in self.floating_windows:
                window = self.floating_windows["plate_manager"]
                return window.findChild(PlateManagerWidget)
            return None

After
~~~~~~

.. code-block:: python

    # Widgets auto-register
    class PlateManagerWidget(QWidget, AutoRegisterServiceMixin):
        pass

    # Any code can access widgets
    from pyqt_reactive.services import ServiceRegistry

    plate_manager = ServiceRegistry.get(PlateManagerWidget)

Integration Points
------------------

Window Handlers
~~~~~~~~~~~~~~

Window handlers use ServiceRegistry to access widgets:

.. code-block:: python

    def _create_step_editor_window(scope_id: str, object_state=None):
        plate_manager = ServiceRegistry.get(PlateManagerWidget)
        orchestrator = ObjectStateRegistry.get_object(plate_path)

        window = DualEditorWindow(
            step_data=step,
            orchestrator=orchestrator,
        )
        return window

Main Window
~~~~~~~~~~~~

Main window removes widget tracking:

.. code-block:: python

    class OpenHCSMainWindow(QMainWindow):
        def __init__(self):
            # Before: self.floating_windows = {}
            # After: No tracking needed

            self.plate_manager_widget = PlateManagerWidget()
            # Auto-registered, accessible everywhere

See Also
--------

- :doc:`/architecture/scope_window_factory_system` - Handler-based window creation
- :doc:`/development/window_manager_usage` - Window management patterns
- :doc:`/architecture/ui_services_architecture` - Service layer architecture

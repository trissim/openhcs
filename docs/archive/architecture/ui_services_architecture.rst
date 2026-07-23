UI Services Architecture
========================

Consolidated service layer for ParameterFormManager operations.

The Problem: Monolithic Form Manager
-------------------------------------

The ParameterFormManager class originally contained all logic for parameter forms in a single 2600+ line class: widget finding, value collection, signal management, parameter operations, form initialization, and styling. This monolithic design made the code hard to test, extend, and maintain. Changes to one concern (e.g., widget styling) required understanding the entire class.

The Solution: Service-Oriented Architecture
---------------------------------------------

The UI services extract specialized responsibilities into focused service classes, each handling one concern. Services are grouped by related functionality (widget operations, value collection, signal management, parameter operations, form initialization) while maintaining clean interfaces. This enables testing individual services in isolation and makes the codebase easier to understand and extend.

Overview
--------

The UI services provide a clean separation of concerns for the ParameterFormManager.
Originally implemented as 17+ separate service files, these have been consolidated
into 5 cohesive services plus 2 base classes, reducing complexity while maintaining
all functionality.

Service Consolidation
---------------------

The services were restructured following the principle of grouping related functionality:

.. list-table:: Service Consolidation
   :header-rows: 1
   :widths: 30 40 30

   * - New Service
     - Merged From
     - Responsibility
   * - ``WidgetService``
     - WidgetFinder, WidgetStyling, WidgetUpdate
     - Widget finding, styling, and value updates
   * - ``ValueCollectionService``
     - NestedValueCollection, DataclassReconstruction, DataclassUnpacker
     - Value collection and dataclass operations
   * - ``SignalService``
     - SignalBlocking, SignalConnection, CrossWindowRegistration
     - Signal management and cross-window updates
   * - ``ParameterOpsService``
     - ParameterReset, PlaceholderRefresh
     - Parameter operations and placeholder management
   * - ``FormInitService``
     - InitializationServices, InitializationStepFactory, FormBuildOrchestrator, InitialRefreshStrategy
     - Form initialization and widget building

Standalone services kept as-is:

- ``EnabledFieldStylingService`` - Specific concern for enabled/disabled field styling
- ``FlagContextManager`` - Clean context manager for manager flags
- ``FieldChangeDispatcher`` - Unified event-driven field change handling (see :doc:`field_change_dispatcher`)
- ``ParameterServiceABC``, ``EnumDispatchService`` - Base classes for type-safe dispatch

WidgetService
-------------

Consolidated service for widget finding, styling, and value updates.

.. code-block:: python

    from openhcs.pyqt_gui.widgets.shared.services.widget_service import WidgetService

    # Find widgets
    checkbox = WidgetService.find_optional_checkbox(manager, param_name)
    widget = WidgetService.get_widget_safe(manager, param_name)

    # Style widgets
    WidgetService.make_readonly(widget, color_scheme)
    WidgetService.apply_dimming(widget, opacity=0.5)

    # Update widget values (instance method)
    service = WidgetService()
    service.update_widget_value(widget, value, param_name, manager=manager)

Key methods:

- ``find_optional_checkbox(manager, param_name)`` - Find optional checkbox for a parameter
- ``find_nested_checkbox(manager, param_name)`` - Find checkbox in nested manager
- ``find_group_box(container, group_box_type)`` - Find group box within container
- ``get_widget_safe(manager, param_name)`` - Safely get widget from manager
- ``make_readonly(widget, color_scheme)`` - Make widget read-only without greying
- ``update_widget_value(widget, value, param_name, ...)`` - Update widget with signal blocking

ValueCollectionService
----------------------

Handles value collection from nested managers and dataclass operations.

It works hand-in-hand with ``ObjectState``:

- PFM uses ObjectState as the single source of truth for parameters and nested states.
- Live context collection walks ObjectStateRegistry (not PFMs) and uses ``get_user_modified_values``/overlays.
- Cancel/save flows rely on ObjectState baselines (mark_saved/restore_saved).

.. code-block:: python

    from openhcs.pyqt_gui.widgets.shared.services.value_collection_service import ValueCollectionService

    service = ValueCollectionService()

    # Collect nested value with type-safe dispatch
    value = service.collect_nested_value(manager, param_name, nested_manager)

    # Unpack dataclass fields to instance attributes
    ValueCollectionService.unpack_to_self(target, source, prefix="config_")

Uses discriminated union dispatch based on ``ParameterInfo`` types:

- ``OptionalDataclassInfo`` - Optional[Dataclass] parameters
- ``DirectDataclassInfo`` - Direct dataclass parameters
- ``GenericInfo`` - Generic/primitive parameters

SignalService
-------------

Manages Qt signal blocking, connection, and cross-window registration.

.. code-block:: python

    from openhcs.pyqt_gui.widgets.shared.services.signal_service import SignalService

    # Block signals (context manager)
    with SignalService.block_signals(checkbox):
        checkbox.setChecked(True)

    # Block multiple widgets
    with SignalService.block_signals(widget1, widget2):
        widget1.setValue(1)
        widget2.setValue(2)

    # Connect all signals for a manager
    SignalService.connect_all_signals(manager)

    # Cross-window registration
    with SignalService.cross_window_registration(manager):
        dialog.exec()

ParameterOpsService
-------------------

Handles parameter reset and placeholder refresh operations.

.. code-block:: python

    from openhcs.pyqt_gui.widgets.shared.services.parameter_ops_service import ParameterOpsService

    service = ParameterOpsService()

    # Reset parameter with type-safe dispatch
    service.reset_parameter(manager, param_name)

    # Refresh placeholders with live context from other windows
    service.refresh_with_live_context(manager)

    # Refresh all placeholders in a form
    service.refresh_all_placeholders(manager)

Uses discriminated union dispatch for reset operations:

- ``_reset_OptionalDataclassInfo`` - Reset Optional[Dataclass] with checkbox sync
- ``_reset_DirectDataclassInfo`` - Reset direct dataclass via nested manager
- ``_reset_GenericInfo`` - Reset generic field with context-aware value

FormInitService
---------------

Orchestrates form initialization with metaprogrammed services.

.. code-block:: python

    from openhcs.pyqt_gui.widgets.shared.services.form_init_service import (
        FormBuildOrchestrator,
        InitialRefreshStrategy,
        ParameterExtractionService,
        ConfigBuilderService,
        ServiceFactoryService
    )

    # Extract parameters using metaprogrammed service
    extracted = ParameterExtractionService.build(object_instance, exclude_params, initial_values)

    # Build config using metaprogrammed service
    form_config = ConfigBuilderService.build(field_id, extracted, context_obj, color_scheme, parent_manager, service)

    # Create all services using metaprogrammed factory
    services = ServiceFactoryService.build()

    # Build widgets with unified async/sync path
    orchestrator = FormBuildOrchestrator()
    orchestrator.build_widgets(manager, content_layout, param_infos, use_async=True)

    # Execute initial refresh strategy
    InitialRefreshStrategy.execute(manager)

Key components:

- ``ParameterExtractionService`` - Extracts parameters from object instance
- ``ConfigBuilderService`` - Builds ParameterFormConfig with derived values
- ``ServiceFactoryService`` - Auto-instantiates all manager services
- ``FormBuildOrchestrator`` - Handles async/sync widget creation
- ``InitialRefreshStrategy`` - Enum-driven initial placeholder refresh

Type-Safe Dispatch Pattern
--------------------------

Services use discriminated union dispatch via ``ParameterServiceABC``:

.. code-block:: python

    class ParameterOpsService(ParameterServiceABC):
        def _get_handler_prefix(self) -> str:
            return '_reset_'

        def _reset_OptionalDataclassInfo(self, info, manager) -> None:
            # Handle Optional[Dataclass] reset
            ...

        def _reset_DirectDataclassInfo(self, info, manager) -> None:
            # Handle direct Dataclass reset
            ...

        def _reset_GenericInfo(self, info, manager) -> None:
            # Handle generic field reset
            ...

This pattern eliminates if/elif type-checking smell with polymorphic dispatch
based on the concrete ``ParameterInfo`` type.

Architecture Benefits
---------------------

The consolidated architecture provides:

- **Reduced File Count**: 17 files → 9 files (including base classes)
- **Cohesive Grouping**: Related functionality in single services
- **Consistent Patterns**: All services use same ABC-based dispatch
- **Clear Responsibilities**: Each service has well-defined scope
- **Easier Discovery**: Developers find functionality in fewer places
- **Maintainability**: Changes localized to single service files

See Also
--------

- :doc:`field_change_dispatcher` - Unified event-driven field change handling
- :doc:`service-layer-architecture` - Framework-agnostic service layer patterns
- :doc:`parameter_form_service_architecture` - ParameterFormService architecture
- :doc:`parameter_form_lifecycle` - Form lifecycle management

Parameter Form Service Architecture
====================================

**Service-oriented refactoring of parameter form management with context layer builders and auto-registration.**

*Status: IN DEVELOPMENT (partially functional)*
*Module: openhcs.pyqt_gui.widgets.shared*

Overview
--------

The parameter form system has been refactored from a monolithic 2653-line class into a service-oriented architecture with clear separation of concerns. The main branch's ``ParameterFormManager`` contained all logic in one class, making it difficult to test, extend, and maintain.

The refactored architecture extracts specialized responsibilities into service classes:

- **Context Layer Builders**: Auto-registered builders for constructing context stacks
- **Placeholder Refresh Service**: Manages placeholder text updates with live context
- **Parameter Reset Service**: Type-safe parameter reset with discriminated union dispatch
- **Widget Update Service**: Handles widget value updates
- **Enabled Field Styling Service**: Manages enabled field styling
- **Signal Connection Service**: Coordinates signal connections

This creates a cleaner, more testable architecture while preserving all functionality from the main branch.

Architecture Comparison
-----------------------

Main Branch (Monolithic)
~~~~~~~~~~~~~~~~~~~~~~~~

The main branch implementation is fully functional but poorly factored:

.. code-block:: text

    ParameterFormManager (2653 lines)
    ├── Widget Creation (500+ lines)
    ├── Context Building (200+ lines)
    ├── Placeholder Refresh (100+ lines)
    ├── Reset Logic (200+ lines)
    ├── Widget Updates (200+ lines)
    ├── Enabled Styling (200+ lines)
    ├── Cross-Window Updates (300+ lines)
    └── Nested Manager Handling (200+ lines)

Current Branch (Service-Oriented)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The refactored implementation separates concerns:

.. code-block:: text

    ParameterFormManager (1200 lines - orchestration only)
    └── Delegates to Services:
        ├── ContextLayerBuilders (auto-registered via metaclass)
        ├── PlaceholderRefreshService
        ├── ParameterResetService
        ├── WidgetUpdateService
        ├── EnabledFieldStylingService
        └── SignalConnectionService

Context Layer Builder System
-----------------------------

The context layer builder system replaces the monolithic ``_build_context_stack()`` method with auto-registered builder classes.

Context Layer Types
~~~~~~~~~~~~~~~~~~~

:py:class:`~openhcs.pyqt_gui.widgets.shared.context_layer_builders.ContextLayerType` defines the layer order:

.. code-block:: python

    class ContextLayerType(Enum):
        """Context layer types in application order."""
        GLOBAL_STATIC_DEFAULTS = "global_static_defaults"  # Fresh GlobalPipelineConfig()
        GLOBAL_LIVE_VALUES = "global_live_values"          # Live GlobalPipelineConfig
        PARENT_CONTEXT = "parent_context"                  # Parent context(s)
        PARENT_OVERLAY = "parent_overlay"                  # Parent's user-modified values
        SIBLING_CONTEXTS = "sibling_contexts"              # Sibling nested manager values
        CURRENT_OVERLAY = "current_overlay"                # Current form values

Layers are applied in enum definition order, with later layers overriding earlier ones.

Builder Pattern
~~~~~~~~~~~~~~~

Each layer type has a dedicated builder class:

.. code-block:: python

    class ContextLayerBuilder(ABC):
        """Base class for context layer builders."""

        _layer_type: ContextLayerType = None  # Set by subclass

        @abstractmethod
        def can_build(self, manager: 'ParameterFormManager', **kwargs) -> bool:
            """Return True if this builder can build a layer for the given manager."""
            pass

        @abstractmethod
        def build(self, manager: 'ParameterFormManager', **kwargs) -> Union[ContextLayer, List[ContextLayer]]:
            """Build and return context layer(s)."""
            pass

Auto-Registration Metaclass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Builders are automatically registered via :py:class:`~openhcs.pyqt_gui.widgets.shared.context_layer_builders.ContextLayerBuilderMeta`:

.. code-block:: python

    class ContextLayerBuilderMeta(type):
        """Metaclass that auto-registers context layer builders."""

        def __new__(mcs, name, bases, namespace):
            cls = super().__new__(mcs, name, bases, namespace)

            # Auto-register if _layer_type is defined
            if hasattr(cls, '_layer_type') and cls._layer_type is not None:
                CONTEXT_LAYER_BUILDERS[cls._layer_type] = cls()

            return cls

This eliminates manual registration boilerplate - just define ``_layer_type`` and the builder is automatically registered.

Sibling Inheritance System
---------------------------

The :py:class:`~openhcs.pyqt_gui.widgets.shared.context_layer_builders.SiblingContextsBuilder` enables nested managers to inherit from each other.

Problem
~~~~~~~

When ``PipelineConfig`` contains both ``well_filter_config: WellFilterConfig`` and ``path_planning_config: PathPlanningConfig``, and ``PathPlanningConfig`` inherits from ``WellFilterConfig``, the ``path_planning_config.well_filter`` field should inherit from ``well_filter_config.well_filter``.

The main branch achieved this by including parent's user-modified values in the context stack. The refactored branch makes this explicit with a dedicated ``SIBLING_CONTEXTS`` layer.

Solution
~~~~~~~~

:py:class:`~openhcs.pyqt_gui.widgets.shared.context_layer_builders.SiblingContextsBuilder` collects values from all sibling nested managers:

.. code-block:: python

    class SiblingContextsBuilder(ContextLayerBuilder):
        """Builder for SIBLING_CONTEXTS layer(s)."""

        _layer_type = ContextLayerType.SIBLING_CONTEXTS

        def can_build(self, manager, live_context=None, **kwargs) -> bool:
            # Only apply for nested managers with live_context
            return manager._parent_manager is not None and live_context is not None

        def build(self, manager, live_context=None, **kwargs) -> List[ContextLayer]:
            layers = []

            # Iterate through all types in live_context
            for ctx_type, ctx_values in live_context.items():
                # Skip self, parent, and global config
                if self._should_skip_type(manager, ctx_type):
                    continue

                # Convert dict to instance and add to layers
                if isinstance(ctx_values, dict):
                    sibling_instance = ctx_type(**ctx_values)
                    layers.append(ContextLayer(
                        layer_type=self._layer_type,
                        instance=sibling_instance
                    ))

            return layers

This enables ``path_planning_config.well_filter`` to see ``well_filter_config.well_filter`` during placeholder resolution.

Placeholder Refresh Service
----------------------------

:py:class:`~openhcs.pyqt_gui.widgets.shared.services.placeholder_refresh_service.PlaceholderRefreshService` manages placeholder text updates with live context.

Key Features
~~~~~~~~~~~~

1. **Live context collection** from other open windows
2. **Sibling value collection** for nested manager inheritance
3. **User-modified vs all values** - controls which values are included in overlay
4. **Recursive nested manager refresh** - propagates updates to all nested forms

User-Modified vs All Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The service supports two modes for building the overlay:

.. code-block:: python

    def refresh_with_live_context(self, manager, live_context=None,
                                  use_user_modified_only: bool = False):
        """Refresh placeholders with live context.

        Args:
            use_user_modified_only: If True, only include user-modified values in overlay.
                                     If False, include all current values.
        """
        # Build overlay based on mode
        current_values = (manager.get_user_modified_values()
                         if use_user_modified_only
                         else manager.get_current_values())

**When to use each mode:**

- ``use_user_modified_only=True``: During reset, so reset fields don't override sibling values
- ``use_user_modified_only=False``: During normal refresh, so edited fields propagate to other fields

This enables correct sibling inheritance after reset.

Parameter Reset Service
-----------------------

:py:class:`~openhcs.pyqt_gui.widgets.shared.services.parameter_reset_service.ParameterResetService` handles parameter reset with type-safe discriminated union dispatch.

Discriminated Union Dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of type-checking smells like:

.. code-block:: python

    if ParameterTypeUtils.is_optional_dataclass(param_type):
        # ... 30 lines
    elif is_dataclass(param_type):
        # ... 15 lines
    else:
        # ... 40 lines

The service uses polymorphic dispatch:

.. code-block:: python

    class ParameterResetService(ParameterServiceABC):
        """Service for resetting parameters with type-safe dispatch."""

        def reset_parameter(self, manager, param_name: str):
            """Reset parameter using type-safe dispatch."""
            info = manager.form_structure.get_parameter_info(param_name)
            self.dispatch(info, manager)  # Auto-dispatches to correct handler

        def _reset_OptionalDataclassInfo(self, info: OptionalDataclassInfo, manager):
            """Reset optional dataclass field."""
            # Type checker knows info is OptionalDataclassInfo!
            ...

        def _reset_DataclassInfo(self, info: DataclassInfo, manager):
            """Reset dataclass field."""
            # Type checker knows info is DataclassInfo!
            ...

        def _reset_GenericInfo(self, info: GenericInfo, manager):
            """Reset generic field."""
            # Type checker knows info is GenericInfo!
            ...

Handlers are auto-discovered based on naming convention: ``_reset_{ParameterInfoClassName}``.

User-Set Fields Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~

The service tracks which fields have been explicitly set by the user:

.. code-block:: python

    def _update_reset_tracking(self, manager, param_name: str, reset_value: Any):
        """Update reset field tracking for lazy behavior."""
        if reset_value is None:
            # Track as reset field
            manager.reset_fields.add(param_name)
            # CRITICAL: Remove from user-set fields when resetting to None
            manager._user_set_fields.discard(param_name)
        else:
            # Remove from reset tracking
            manager.reset_fields.discard(param_name)

This ensures :py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager.get_user_modified_values` correctly excludes reset fields.

Execution Flow Examples
-----------------------

Understanding the complete execution flow helps debug issues.

User Edits a Field
~~~~~~~~~~~~~~~~~~~

1. User types in widget → widget emits signal
2. ``_emit_parameter_change()`` called with new value
3. Field added to ``_user_set_fields`` (marks as user-edited)
4. ``parameter_changed`` signal emitted
5. ``_on_parameter_changed()`` called (signal handler)
6. ``refresh_with_live_context(use_user_modified_only=False)`` called
7. ``get_current_values()`` includes the edited field
8. Edited field added to ``live_context[type]``
9. Sibling values collected from other nested managers
10. Context stack built with all layers
11. Placeholders refreshed for all fields
12. Nested managers refreshed recursively

**Result:** Other fields see the edited value in their placeholders immediately.

User Resets a Field
~~~~~~~~~~~~~~~~~~~~

1. User clicks reset button
2. ``reset_parameter(param_name)`` called
3. ``ParameterResetService.reset_parameter()`` dispatches to handler
4. Handler resets value to None (for lazy configs)
5. Field removed from ``_user_set_fields`` (marks as not user-edited)
6. Field added to ``reset_fields`` (marks as reset)
7. Widget updated to show None
8. ``refresh_with_live_context(use_user_modified_only=True)`` called
9. ``get_user_modified_values()`` excludes the reset field
10. Reset field NOT added to ``live_context[type]``
11. Sibling values collected (includes sibling's value for this field)
12. Context stack built with sibling layer
13. Placeholder resolved from sibling value
14. Nested managers refreshed recursively

**Result:** Reset field inherits from sibling config correctly.

Opening a New Window
~~~~~~~~~~~~~~~~~~~~

1. New dialog created with ``ParameterFormManager``
2. Manager registers in ``_active_form_managers`` (class-level registry)
3. ``InitialRefreshStrategy.execute()`` called
4. Strategy determines refresh mode (global config, pipeline config, etc.)
5. ``refresh_with_live_context()`` called
6. ``collect_live_context_from_other_windows()`` collects from all other managers
7. Live context includes values from all open windows
8. Context stack built with live values
9. Placeholders show live values from other windows
10. User sees current state immediately

**Result:** New window shows live values from other open windows.

Live Context Structure
----------------------

Understanding the live context dict structure is critical for debugging placeholder issues.

Live Context Dict Format
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``live_context`` dict maps **types** to their **current values**:

.. code-block:: python

    live_context = {
        GlobalPipelineConfig: {'well_filter': 'test', 'path_planning': {...}},
        PipelineConfig: {'well_filter': 'test2', 'path_planning_config': {...}},
        WellFilterConfig: {'well_filter': 'test3'},
        PathPlanningConfig: {'well_filter': None, 'other_field': 'value'},
    }

**Key points:**

- Keys are **types** (classes), not instances
- Values are **dicts** of field names to values
- Same type can appear multiple times (base type + lazy type)
- Nested dataclasses are stored as fully reconstructed instances in ``get_user_modified_values()``

Collection Process
~~~~~~~~~~~~~~~~~~

1. **Root manager** calls ``collect_live_context_from_other_windows()``
2. Iterates through ``_active_form_managers`` (class-level registry)
3. For each manager, calls ``get_user_modified_values()`` (only user-edited fields)
4. Maps values by type: ``live_context[type(manager.object_instance)] = values``
5. Also maps by base type and lazy type for flexible matching

Sibling Value Collection
~~~~~~~~~~~~~~~~~~~~~~~~~

For nested managers, sibling values are added to live context:

.. code-block:: python

    # In refresh_with_live_context()
    if manager._parent_manager is not None:
        for sibling_name, sibling_manager in manager._parent_manager.nested_managers.items():
            if sibling_manager is manager:
                continue  # Skip self

            sibling_values = sibling_manager.get_current_values()
            sibling_type = type(sibling_manager.object_instance)
            live_context[sibling_type] = sibling_values

**Critical:** Sibling collection uses ``get_current_values()`` (all values), not ``get_user_modified_values()`` (only edited values).

Context Stack Application Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Layers are applied in this order (later layers override earlier ones):

1. **GLOBAL_STATIC_DEFAULTS** - Fresh ``GlobalPipelineConfig()`` (only for root global config editing)
2. **GLOBAL_LIVE_VALUES** - Live ``GlobalPipelineConfig`` from other windows
3. **PARENT_CONTEXT** - Parent context(s) with live values merged in
4. **PARENT_OVERLAY** - Parent's user-modified values (filtered to exclude current nested config)
5. **SIBLING_CONTEXTS** - Sibling nested manager values (enables sibling inheritance)
6. **CURRENT_OVERLAY** - Current form values (always applied last)

Debugging Placeholder Issues
-----------------------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue: Placeholder shows wrong value after reset**

Check:

1. Is ``use_user_modified_only=True`` passed to ``refresh_with_live_context()``?
2. Is the reset field removed from ``_user_set_fields``?
3. Does ``get_user_modified_values()`` exclude the reset field?
4. Is sibling value collection working (check logs for "Added sibling")?

**Issue: Cross-field updates don't work**

Check:

1. Is ``use_user_modified_only=False`` (default) for normal refresh?
2. Is the edited field added to ``_user_set_fields`` in ``_emit_parameter_change()``?
3. Is ``refresh_with_live_context()`` called after parameter change?
4. Are nested managers being refreshed recursively?

**Issue: Sibling inheritance not working**

Check:

1. Is ``SiblingContextsBuilder`` registered in ``CONTEXT_LAYER_BUILDERS``?
2. Does ``can_build()`` return True (nested manager + live_context exists)?
3. Are sibling values being collected (check logs for "Added sibling")?
4. Is ``SIBLING_CONTEXTS`` layer being applied before ``CURRENT_OVERLAY``?

Logging and Debugging
~~~~~~~~~~~~~~~~~~~~~~

Enable debug logging to see context stack construction:

.. code-block:: python

    import logging
    logging.getLogger('openhcs.pyqt_gui.widgets.shared').setLevel(logging.DEBUG)

Key log messages:

- ``🔍 REFRESH: {field_id} refreshing with live context`` - Refresh started
- ``🔍 COLLECT_CONTEXT: Collecting from {field_id}`` - Collecting from other manager
- ``🔍 REFRESH: Added sibling {name} values`` - Sibling values collected
- ``🔍 SIBLING_BUILD: Building for {field_id}`` - Sibling layer being built
- ``[PLACEHOLDER] {field_id}.{param_name}: resolved text='{text}'`` - Placeholder resolved

User-Set Fields Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~

**Critical for debugging reset issues:**

- ``_user_set_fields`` is a ``set()`` that tracks which fields have been explicitly edited by the user
- Starts **empty** (not populated during initialization)
- Populated in ``_emit_parameter_change()`` when user edits a widget
- Cleared in ``_update_reset_tracking()`` when field is reset to None
- Used by ``get_user_modified_values()`` to distinguish user edits from inherited values

**Common bug:** If ``_user_set_fields`` is populated during initialization, inherited values will be treated as user edits, breaking sibling inheritance.

Migration Status
----------------

Current Status
~~~~~~~~~~~~~~

✅ **Implemented and Working:**

- Context layer builder system with auto-registration
- Sibling inheritance via ``SiblingContextsBuilder``
- Placeholder refresh service with ``use_user_modified_only`` parameter
- Parameter reset service with discriminated union dispatch
- User-set fields tracking (starts empty, populated on user edits)

⚠️ **Partially Working:**

- Cross-field updates work when editing fields
- Reset button correctly inherits from sibling configs
- Placeholders resolve from global pipeline config after reset

❌ **Known Issues:**

- Some edge cases may not be fully tested
- Performance optimizations from main branch not all ported (async widget creation, batched refreshes)

Missing from Main Branch
~~~~~~~~~~~~~~~~~~~~~~~~

Features that exist in main branch but not yet ported:

1. **Async widget creation** - Progressive rendering for large forms
2. **Batched placeholder refreshes** - ``reset_all_parameters()`` does single refresh at end
3. **Parent overlay filtering** - Verify ``exclude_params`` access is correct

See Also
--------

- :doc:`service-layer-architecture` - General service layer patterns
- :doc:`parameter_form_lifecycle` - Form lifecycle management (describes main branch)
- :doc:`context_system` - Thread-local context management
- :py:class:`~openhcs.pyqt_gui.widgets.shared.context_layer_builders.ContextLayerBuilder` - Base builder class
- :py:class:`~openhcs.pyqt_gui.widgets.shared.services.placeholder_refresh_service.PlaceholderRefreshService` - Placeholder refresh service
- :py:class:`~openhcs.pyqt_gui.widgets.shared.services.parameter_reset_service.ParameterResetService` - Parameter reset service

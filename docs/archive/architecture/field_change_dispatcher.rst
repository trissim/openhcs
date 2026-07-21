Field Change Dispatcher Architecture
=====================================

Unified event-driven architecture for parameter form field changes.

Overview
--------

The ``FieldChangeDispatcher`` centralizes all field change handling in parameter forms,
replacing scattered callback connections with a single event-driven dispatch point.
This architecture eliminates "callback spaghetti" and provides consistent behavior
for sibling inheritance, cross-window updates, and nested form propagation.

Problem Statement
-----------------

Prior to the dispatcher, field changes were handled through multiple overlapping paths:

1. ``_emit_parameter_change()`` - Local signal emission
2. ``_on_nested_parameter_changed()`` - Parent notification for nested changes
3. ``_emit_cross_window_change()`` - Cross-window context updates
4. Various signal connections in ``SignalService``

This caused several bugs:

- **First keystroke missed**: Sibling placeholders didn't update on first input
  because parent chain wasn't marked modified before sibling refresh
- **Reset broke inheritance**: Individual field reset cleared sibling-inherited
  placeholders because it bypassed proper context building
- **Non-dataclass roots excluded**: ``FunctionStep`` and other non-dataclass roots
  couldn't participate in sibling inheritance

Solution: Event-Driven Dispatch
-------------------------------

All field changes now flow through a single ``FieldChangeEvent``:

.. code-block:: python

    @dataclass
    class FieldChangeEvent:
        field_name: str                        # Leaf field name
        value: Any                             # New value
        source_manager: ParameterFormManager   # Where change originated
        is_reset: bool = False                 # True if reset operation

The ``FieldChangeDispatcher`` (singleton, stateless) handles all events:

.. code-block:: python

    from openhcs.pyqt_gui.widgets.shared.services.field_change_dispatcher import (
        FieldChangeDispatcher, FieldChangeEvent
    )

    # Widget change handler
    def on_widget_change(param_name, value, manager):
        converted_value = manager._convert_widget_value(value, param_name)
        event = FieldChangeEvent(param_name, converted_value, manager)
        FieldChangeDispatcher.instance().dispatch(event)

    # Reset operation
    event = FieldChangeEvent(param_name, reset_value, manager, is_reset=True)
    FieldChangeDispatcher.instance().dispatch(event)

Dispatch Flow
-------------

When ``dispatch(event)`` is called:

1. **Update Source Data Model**:
   - ``source.parameters[field_name] = value``
   - Add/remove from ``_user_set_fields`` based on ``is_reset``

2. **Mark Parent Chain Modified**:
   - Walk up ``_parent_manager`` chain
   - Update each parent's ``parameters`` with collected nested value
   - Add nested field name to parent's ``_user_set_fields``
   - This ensures ``root.get_user_modified_values()`` includes the new value

3. **Refresh Sibling Placeholders**:
   - Find siblings via ``parent.nested_managers``
   - For each sibling with same field name, call ``refresh_single_placeholder()``
   - Skip if field is in sibling's ``_user_set_fields`` (user-set value preserved)

4. **Apply Enabled Styling**:
   - If ``field_name == 'enabled'``, apply visual styling

5. **Emit Local Signal**:
   - ``source.parameter_changed.emit(field_name, value)``

6. **Emit Cross-Window Signal**:
   - Build full path: ``"Root.nested.field_name"``
   - Update thread-local global config if editing global config
   - ``root.context_value_changed.emit(full_path, value, ...)``

Sibling Inheritance via Root Form
---------------------------------

The dispatcher enables sibling inheritance through the ``build_context_stack()``
function in ``context_manager.py``:

.. code-block:: python

    # Find root manager (walk up parent chain)
    root_manager = manager
    while root_manager._parent_manager is not None:
        root_manager = root_manager._parent_manager

    # Get root's values (contains all sibling configs)
    root_values = root_manager.get_user_modified_values()

    # Build context stack with root form values
    stack = build_context_stack(
        context_obj=manager.context_obj,
        overlay=manager.parameters,
        root_form_values=root_values,
        root_form_type=root_manager.dataclass_type,
        ...
    )

For non-dataclass roots (e.g., ``FunctionStep``), the stack builder wraps values
in a ``SimpleNamespace`` to maintain a unified code path:

.. code-block:: python

    if root_form_type and is_dataclass(root_form_type):
        root_instance = root_form_type(**root_form_values)
    else:
        # Non-dataclass root - wrap in SimpleNamespace
        root_instance = SimpleNamespace(**root_form_values)

    stack.enter_context(config_context(root_instance))

This allows ``FunctionStep`` parameters (like ``step_well_filter_config``) to
participate in sibling inheritance just like dataclass-based configurations.

Integration Points
------------------

**Widget Creation** (``widget_creation_config.py``):

.. code-block:: python

    def on_widget_change(pname, value, mgr=manager):
        converted_value = mgr._convert_widget_value(value, pname)
        event = FieldChangeEvent(pname, converted_value, mgr)
        FieldChangeDispatcher.instance().dispatch(event)

    PyQt6WidgetEnhancer.connect_change_signal(widget, param_name, on_widget_change)

**Parameter Updates** (``parameter_form_manager.py``):

.. code-block:: python

    def update_parameter(self, param_name: str, value: Any) -> None:
        # ... convert value, update widget ...
        event = FieldChangeEvent(param_name, converted_value, self)
        FieldChangeDispatcher.instance().dispatch(event)

**Reset Operations** (``parameter_ops_service.py``):

.. code-block:: python

    def _reset_GenericInfo(self, info, manager) -> None:
        # ... update parameters, tracking ...
        if reset_value is None:
            self.refresh_single_placeholder(manager, param_name)

Benefits
--------

- **Single Entry Point**: All changes flow through one dispatcher
- **Consistent Ordering**: Parent marking always before sibling refresh
- **Reentrancy Safe**: Guard prevents recursive dispatch
- **Debug Friendly**: Centralized logging with ``DEBUG_DISPATCHER`` flag
- **Framework Agnostic**: Core logic in ``build_context_stack()`` works with any UI

See Also
--------

- :doc:`ui_services_architecture` - UI service layer overview
- :doc:`parameter_form_lifecycle` - Form lifecycle management
- :doc:`context_system` - Configuration context and inheritance

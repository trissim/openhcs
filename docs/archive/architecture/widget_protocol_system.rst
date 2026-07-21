Widget Protocol System
======================

**ABC-based widget contracts replacing duck typing with fail-loud type checking.**

*Module: openhcs.ui.shared*

The Problem: Duck Typing in UI Code
-----------------------------------

Before this system, the OpenHCS UI layer relied heavily on duck typing to interact with
widgets. Code would check ``hasattr(widget, 'get_value')`` or try to call methods and
catch exceptions. This created several problems:

1. **Silent failures** - If a widget didn't have a method, the code would silently skip it
   or use a fallback, masking bugs that should have been caught during development.

2. **Scattered dispatch tables** - Each module maintained its own ``WIDGET_UPDATE_DISPATCH``
   and ``WIDGET_GET_DISPATCH`` dictionaries mapping widget types to handler functions.
   These tables were duplicated, inconsistent, and hard to maintain.

3. **Inconsistent Qt APIs** - Qt widgets have inconsistent APIs: ``QLineEdit.text()`` vs
   ``QSpinBox.value()`` vs ``QComboBox.currentData()``. Each place that read widget values
   had to know about these differences.

4. **No discoverability** - There was no central registry of what widgets existed or what
   capabilities they had. Finding all widgets that support placeholders required grepping
   the codebase.

The Solution: ABC-Based Contracts
---------------------------------

The Widget Protocol System solves these problems by defining explicit Abstract Base Class
(ABC) contracts. Instead of asking "does this widget have a get_value method?", we ask
"is this widget a ValueGettable?". This is a fundamental shift from structural typing
(duck typing) to nominal typing (explicit inheritance).

The key insight is that widget capabilities are composable. A text field can get and set
values, show placeholders, and emit change signals. A checkbox can get and set values and
emit signals, but doesn't need placeholders. By defining each capability as a separate ABC,
widgets can mix and match exactly the capabilities they need.

This follows established OpenHCS patterns:

- **StorageBackendMeta** - Metaclass auto-registration for storage backends
- **MemoryTypeConverter** - Adapter pattern for normalizing inconsistent memory APIs

Design Philosophy
~~~~~~~~~~~~~~~~~

- **Explicit inheritance over duck typing** - Widgets declare capabilities via ABC inheritance
- **Fail-loud over fail-silent** - Missing implementations raise ``TypeError`` immediately
- **Discoverable over scattered** - All capabilities tracked in a central registry
- **Multiple inheritance for composable capabilities** - Mix and match ABCs as needed

Architecture
------------

The system consists of 6 modules that work together:

.. list-table:: Widget Protocol Modules
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Purpose
   * - ``widget_protocols.py``
     - ABC definitions (ValueGettable, ValueSettable, PlaceholderCapable, etc.)
   * - ``widget_registry.py``
     - WidgetMeta metaclass for auto-registration
   * - ``widget_adapters.py``
     - Qt widget adapters implementing ABCs
   * - ``widget_dispatcher.py``
     - ABC-based dispatch with explicit isinstance checks
   * - ``widget_operations.py``
     - Centralized operations API
   * - ``widget_factory.py``
     - Type-based widget creation

Widget ABCs
-----------

The foundation of the system is six Abstract Base Classes, each representing a single
widget capability. These ABCs are intentionally minimal—each defines exactly one
responsibility, allowing widgets to compose capabilities through multiple inheritance.

Think of these like interfaces in Java or protocols in Swift. A widget that inherits
from ``ValueGettable`` is making a contract: "I promise to implement ``get_value()``".
If the widget fails to implement the method, Python raises ``TypeError`` at class
definition time, not at runtime when you try to use it.

.. code-block:: python

    from openhcs.ui.shared.widget_protocols import (
        ValueGettable,      # get_value() -> Any
        ValueSettable,      # set_value(value: Any) -> None
        PlaceholderCapable, # set_placeholder(text: str) -> None
        RangeConfigurable,  # configure_range(min, max) -> None
        EnumSelectable,     # set_enum_options(enum_type) / get_selected_enum()
        ChangeSignalEmitter # connect_change_signal() / disconnect_change_signal()
    )

Here's what the simplest ABC looks like. The ``@abstractmethod`` decorator ensures
that any concrete class must implement this method:

.. code-block:: python

    class ValueGettable(ABC):
        """ABC for widgets that can return a value."""

        @abstractmethod
        def get_value(self) -> Any:
            """Get the current value from the widget."""
            pass

Metaclass Auto-Registration
---------------------------

One of the pain points with widget systems is keeping a registry of available widgets
in sync with the actual widget classes. Add a new widget class, forget to register it,
and you get mysterious "widget not found" errors at runtime.

The ``WidgetMeta`` metaclass solves this by automatically registering widgets when
their classes are defined. This mirrors the ``StorageBackendMeta`` pattern used
elsewhere in OpenHCS. When Python processes a class definition with this metaclass,
it automatically adds the class to the global registry:

.. code-block:: python

    from openhcs.ui.shared.widget_registry import WidgetMeta, WIDGET_IMPLEMENTATIONS

    class LineEditAdapter(QLineEdit, ValueGettable, ValueSettable,
                          metaclass=WidgetMeta):
        _widget_id = "line_edit"

        def get_value(self) -> Any:
            return self.text()

        def set_value(self, value: Any) -> None:
            self.setText(str(value) if value else "")

    # Auto-registered:
    assert WIDGET_IMPLEMENTATIONS["line_edit"] is LineEditAdapter

Registry functions:

- ``get_widget_class(widget_id)`` - Get class by ID
- ``get_widget_capabilities(widget_class)`` - Get ABCs a class implements
- ``list_widgets_with_capability(abc)`` - Find all widgets implementing an ABC

Qt Widget Adapters
------------------

Here's where theory meets practice. Qt widgets have notoriously inconsistent APIs.
To get a value, you call ``text()`` on a QLineEdit, ``value()`` on a QSpinBox, and
``currentData()`` on a QComboBox. Setting values is similarly inconsistent. And
placeholders? QLineEdit has ``setPlaceholderText()``, but QSpinBox uses a completely
different mechanism called "special value text" that only shows when the value equals
the minimum.

The adapter layer normalizes these inconsistencies. Each adapter wraps a Qt widget
and implements the appropriate ABCs, translating the uniform interface to Qt-specific
calls:

.. code-block:: python

    # The problem - Qt inconsistency:
    line_edit.text()           # vs spinbox.value()
    line_edit.setText(v)       # vs spinbox.setValue(v)
    line_edit.setPlaceholderText(t)  # vs spinbox.setSpecialValueText(t)

    # The solution - ABC-normalized interface:
    adapter.get_value()         # Uniform for all widgets
    adapter.set_value(v)        # Uniform for all widgets
    adapter.set_placeholder(t)  # Uniform for all widgets

The adapter implementations handle edge cases that would otherwise be scattered
throughout the codebase. For example, ``SpinBoxAdapter`` treats the minimum value
with special value text as "None", allowing spinboxes to represent optional integers.

Available adapters:

- ``LineEditAdapter`` - QLineEdit wrapper
- ``SpinBoxAdapter`` - QSpinBox wrapper (handles None via special value text)
- ``DoubleSpinBoxAdapter`` - QDoubleSpinBox wrapper
- ``ComboBoxAdapter`` - QComboBox wrapper with enum support
- ``CheckBoxAdapter`` - QCheckBox wrapper
- ``GroupBoxCheckboxAdapter`` - QGroupBox with checkbox title

ABC-Based Dispatch
------------------

With ABCs and adapters in place, we need a dispatch layer that routes operations
to the right methods. The key difference from duck typing is that we use ``isinstance``
checks against ABCs rather than ``hasattr`` checks for methods.

This might seem like a minor distinction, but it fundamentally changes error handling.
With duck typing, missing a method might silently fall through to a default case.
With ABC dispatch, attempting an operation on a widget that doesn't support it
raises an immediate, descriptive ``TypeError``:

.. code-block:: python

    from openhcs.ui.shared.widget_dispatcher import WidgetDispatcher

    # BEFORE (duck typing):
    if hasattr(widget, 'get_value'):
        value = widget.get_value()

    # AFTER (ABC-based, fails loud):
    value = WidgetDispatcher.get_value(widget)  # TypeError if not ValueGettable

Error message on failure:

.. code-block:: text

    TypeError: Widget QLabel does not implement ValueGettable ABC.
    Add ValueGettable to widget's base classes and implement get_value() method.

Centralized Operations
----------------------

While ``WidgetDispatcher`` handles the low-level dispatch, ``WidgetOperations`` provides
the API that most code should use. It wraps the dispatcher with additional conveniences
like finding all value-capable widgets in a container and "try-style" operations for
optional capabilities.

The distinction between fail-loud and try-style operations is important. Use fail-loud
operations when the widget *must* support the capability (a bug if it doesn't). Use
try-style when the capability is genuinely optional (e.g., setting placeholders on
widgets that may or may not support them):

.. code-block:: python

    from openhcs.ui.shared.widget_operations import WidgetOperations

    ops = WidgetOperations()

    # Fail-loud operations (raise TypeError if ABC not implemented)
    value = ops.get_value(widget)
    ops.set_value(widget, 42)
    ops.set_placeholder(widget, "Pipeline default: 100")
    ops.configure_range(widget, 0, 100)
    ops.connect_change_signal(widget, on_change_callback)

    # Try-style operations (return False if unsupported)
    if ops.try_set_placeholder(widget, text):
        print("Placeholder set")
    else:
        print("Widget doesn't support placeholders")

    # Find all value-capable widgets in a container
    value_widgets = ops.get_all_value_widgets(form_container)

Widget Factory
--------------

The final piece is the factory that creates widgets based on Python types. When building
forms from dataclass definitions, we need to create appropriate widgets for each field
type. The factory maps Python types to widget constructors:

.. code-block:: python

    from openhcs.ui.shared.widget_factory import WidgetFactory

    factory = WidgetFactory()

    # Type-based creation
    widget = factory.create_widget_for_type(str)   # -> LineEditAdapter
    widget = factory.create_widget_for_type(int)   # -> SpinBoxAdapter
    widget = factory.create_widget_for_type(bool)  # -> CheckBoxAdapter
    widget = factory.create_widget_for_type(MyEnum)  # -> ComboBoxAdapter with enum

The factory handles type resolution including ``Optional[T]`` unwrapping, enum detection,
and ``List[Enum]`` for multi-select widgets.

See Also
--------

- :doc:`field_change_dispatcher` - Event-driven field change handling
- :doc:`ui_services_architecture` - Service layer using these protocols
- :doc:`abstract_manager_widget` - ABC that uses widget protocols
- :doc:`parametric_widget_creation` - Widget creation configuration

Parametric Widget Creation
==========================

**Dataclass-based configuration for widget creation strategies.**

*Module: openhcs.pyqt_gui.widgets.shared.widget_creation_config*

The Widget Creation Problem
---------------------------

OpenHCS configuration is deeply nested. A pipeline configuration contains plate configurations,
which contain step configurations, which contain processing parameters—many of which are
themselves nested dataclasses. Some of these nested types are optional (``Optional[DataclassType]``),
requiring checkbox-gated visibility.

Building forms for this structure requires answering many questions for each field:

- Is this a simple value (string, int, enum) or a nested structure?
- Should it have a label? A reset button?
- Does it need a container layout, and if so, horizontal or vertical?
- Is it optional? Does it need a checkbox to toggle None/Instance?

Before this system, these decisions were scattered across if/elif chains with duplicated
logic. Adding a new widget type (like multi-select enums) required modifying multiple
code paths that had grown organically and inconsistently.

The Solution: Configuration Objects
-----------------------------------

The parametric widget creation system consolidates all these decisions into configuration
dataclasses. Each widget creation type (REGULAR, NESTED, OPTIONAL_NESTED) has a corresponding
``WidgetCreationConfig`` that declares:

- What kind of container to create
- What kind of main widget to create
- Whether labels, reset buttons, or checkboxes are needed
- Handler functions for optional features

This follows the same pattern as ``openhcs/core/memory/framework_config.py`` for memory
type handling—replace conditionals with configuration lookup.

Widget Creation Types
---------------------

There are three fundamentally different ways to render a configuration field, distinguished
by nesting and optionality:

.. code-block:: python

    from openhcs.pyqt_gui.widgets.shared.widget_creation_config import WidgetCreationType

    class WidgetCreationType(Enum):
        REGULAR = "regular"          # Simple widgets (int, str, bool, enum)
        NESTED = "nested"            # Nested dataclass forms
        OPTIONAL_NESTED = "optional_nested"  # Optional[Dataclass] with checkbox

Each type has a corresponding ``WidgetCreationConfig`` dataclass:

.. code-block:: python

    @dataclass
    class WidgetCreationConfig:
        layout_type: str                    # "horizontal" or "vertical"
        is_nested: bool                     # Whether creates sub-form
        create_container: WidgetOperationHandler  # Container widget factory
        setup_layout: Optional[WidgetOperationHandler]  # Layout configuration
        create_main_widget: WidgetOperationHandler  # Main widget factory
        needs_label: bool                   # Show field label
        needs_reset_button: bool            # Show reset button
        needs_unwrap_type: bool             # Unwrap Optional[T] to T
        is_optional: bool = False           # Has None/Instance toggle
        needs_checkbox: bool = False        # Show optional checkbox
        create_title_widget: Optional[OptionalTitleHandler] = None
        connect_checkbox_logic: Optional[CheckboxLogicHandler] = None

ParameterFormManager ABC
------------------------

The widget creation system needs a consistent interface to the form manager. This ABC
defines that interface, inspired by React's component model. Like React components,
form managers maintain state (``parameters``, ``nested_managers``, ``widgets``) and
provide methods to mutate that state (``update_parameter``, ``reset_parameter``).

The React analogy is deliberate: parameter forms are essentially hierarchical
UI components with controlled inputs. The ``create_widget`` method is analogous to
React's ``render()``—it produces UI elements based on current state. The
``_apply_to_nested_managers`` method enables recursive traversal, similar to React's
component tree:

.. code-block:: python

    from openhcs.pyqt_gui.widgets.shared.widget_creation_types import ParameterFormManager

    class ParameterFormManager(ABC):
        """React-quality reactive form manager interface."""

        # State (like React component state)
        parameters: Dict[str, Any]
        nested_managers: Dict[str, Any]
        widgets: Dict[str, Any]

        # State mutations (like setState)
        @abstractmethod
        def update_parameter(self, param_name: str, value: Any) -> None: ...

        @abstractmethod
        def reset_parameter(self, param_name: str) -> None: ...

        # Widget creation (like render)
        @abstractmethod
        def create_widget(self, param_name: str, param_type: Type,
                         current_value: Any, widget_id: str) -> Any: ...

        # Component tree traversal
        @abstractmethod
        def _apply_to_nested_managers(self, callback: Callable) -> None: ...

Type Definitions
----------------

The handler functions have complex signatures because they need access to everything
the form manager knows about the field being rendered. TypedDicts and type aliases
provide documentation and type checking for these signatures:

Handler type aliases for type safety:

.. code-block:: python

    # Main widget operation handler
    WidgetOperationHandler = Callable[
        [ParameterFormManager, ParameterInfo, DisplayInfo, FieldIds,
         Any, Optional[Type], ...],
        Any
    ]

    # Optional title widget handler
    OptionalTitleHandler = Callable[
        [ParameterFormManager, ParameterInfo, DisplayInfo, FieldIds,
         Any, Optional[Type]],
        Dict[str, Any]
    ]

    # Checkbox toggle logic handler
    CheckboxLogicHandler = Callable[
        [ParameterFormManager, ParameterInfo, Any, Any, Any, Any, Any, Type],
        None
    ]

Helper TypedDicts:

.. code-block:: python

    class DisplayInfo(TypedDict, total=False):
        field_label: str
        checkbox_label: str
        description: str

    class FieldIds(TypedDict, total=False):
        widget_id: str
        optional_checkbox_id: str

Configuration Registry
----------------------

The ``_WIDGET_CREATION_CONFIG`` dict maps types to configurations:

.. code-block:: python

    _WIDGET_CREATION_CONFIG = {
        WidgetCreationType.REGULAR: WidgetCreationConfig(
            layout_type="horizontal",
            is_nested=False,
            create_container=_create_regular_container,
            setup_layout=None,
            create_main_widget=_create_regular_widget,
            needs_label=True,
            needs_reset_button=True,
            needs_unwrap_type=False,
        ),
        WidgetCreationType.NESTED: WidgetCreationConfig(
            layout_type="vertical",
            is_nested=True,
            create_container=_create_nested_container,
            setup_layout=_setup_nested_layout,
            create_main_widget=_create_nested_form,
            needs_label=False,
            needs_reset_button=False,
            needs_unwrap_type=True,
        ),
        WidgetCreationType.OPTIONAL_NESTED: WidgetCreationConfig(
            layout_type="vertical",
            is_nested=True,
            is_optional=True,
            needs_checkbox=True,
            create_container=_create_optional_container,
            setup_layout=_setup_nested_layout,
            create_main_widget=_create_nested_form,
            create_title_widget=_create_optional_title_widget,
            connect_checkbox_logic=_connect_optional_checkbox_logic,
            needs_label=False,
            needs_reset_button=False,
            needs_unwrap_type=True,
        ),
    }

Handler Functions
-----------------

Handlers encapsulate widget creation logic:

.. code-block:: python

    def _create_nested_form(manager, param_info, display_info, field_ids,
                           current_value, unwrapped_type, **kwargs) -> Any:
        """Create nested form and store in manager.nested_managers."""
        nested_manager = manager._create_nested_form_inline(
            param_info.name, unwrapped_type, current_value
        )
        manager.nested_managers[param_info.name] = nested_manager
        return nested_manager.build_form()

    def _create_optional_title_widget(manager, param_info, display_info,
                                      field_ids, current_value, unwrapped_type):
        """Create checkbox + title + reset button for optional dataclass."""
        # Returns (title_widget, checkbox) tuple
        ...

Usage Pattern
-------------

The ``ParameterFormManager`` uses the config for type-dispatched widget creation:

.. code-block:: python

    def _create_parameter_widget(self, param_info, param_type, current_value):
        # Determine widget type
        widget_type = self._classify_widget_type(param_type)

        # Get configuration
        config = _WIDGET_CREATION_CONFIG[widget_type]

        # Execute creation pipeline
        container = config.create_container(self, param_info, ...)
        if config.setup_layout:
            config.setup_layout(self, param_info, container, ...)
        widget = config.create_main_widget(self, param_info, ...)

        if config.needs_label:
            self._add_label(container, param_info)
        if config.needs_reset_button:
            self._add_reset_button(container, param_info)

See Also
--------

- :doc:`widget_protocol_system` - ABC contracts for widget operations
- :doc:`field_change_dispatcher` - Dispatches changes from created widgets
- :doc:`parameter_form_service_architecture` - Service architecture using these configs
- :doc:`abstract_manager_widget` - ABC that orchestrates widget creation

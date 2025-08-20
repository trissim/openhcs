UI Patterns
===========

UI patterns and architectural approaches for OpenHCS framework-agnostic interfaces.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The OpenHCS UI system uses key patterns for framework independence:

- **Functional Dispatch**: Type-based dispatch tables instead of if/elif chains
- **Service Layer**: Framework-agnostic business logic extraction
- **Utility Classes**: Shared components for cross-framework compatibility

Functional Dispatch Pattern
---------------------------

The functional dispatch pattern solves a common problem in UI development: handling different widget types with different operations. Instead of writing long chains of if/elif statements that check widget types, you create a lookup table that maps types to functions.

This pattern emerged during the UI refactor when we noticed the same type-checking logic repeated across both PyQt6 and Textual implementations. By centralizing this logic into dispatch tables, we eliminated code duplication and made the system more extensible.

Type-Based Dispatch
~~~~~~~~~~~~~~~~~~~

The core idea is simple: create a dictionary where keys are types and values are functions that know how to handle those types. This eliminates the need to manually check types in your code.

.. code-block:: python

    # DO: Type-based dispatch
    WIDGET_STRATEGIES: Dict[Type, Callable] = {
        QCheckBox: lambda w: w.isChecked(),
        QComboBox: lambda w: w.itemData(w.currentIndex()),
        QSpinBox: lambda w: w.value(),
        QLineEdit: lambda w: w.text(),
    }

    def get_widget_value(widget: Any) -> Any:
        strategy = WIDGET_STRATEGIES.get(type(widget))
        return strategy(widget) if strategy else None

Attribute-Based Dispatch
~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you need to dispatch based on what methods a widget has rather than its exact type. This is useful when multiple widget types share the same interface but have different class hierarchies.

.. code-block:: python

    # DO: Attribute dispatch
    SIGNAL_CONNECTIONS = {
        'textChanged': lambda w, cb: w.textChanged.connect(cb),
        'stateChanged': lambda w, cb: w.stateChanged.connect(cb),
    }

Anti-Pattern: If/Elif Chains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before the refactor, our codebase was full of repetitive type-checking logic. Every time we needed to handle different widget types, we'd write the same if/elif pattern. This became a maintenance nightmare when adding new widget types or changing existing behavior.

.. code-block:: python

    # DON'T: Verbose conditionals
    if isinstance(widget, QComboBox):
        return widget.itemData(widget.currentIndex())
    elif hasattr(widget, 'isChecked'):
        return widget.isChecked()
    # ... many more conditions

**Why This Matters:** When you have 15+ widget types and 5+ different operations, if/elif chains become unmanageable. Adding a new widget type means finding and updating every chain. With dispatch tables, you just add one entry to the dictionary.

**Performance Benefit:** Dictionary lookup is O(1) while if/elif chains are O(n). With many widget types, this difference becomes noticeable.

Service Layer Pattern
---------------------

The service layer pattern addresses a fundamental problem in UI development: business logic gets mixed with presentation code. When you have multiple UI frameworks (like PyQt6 and Textual), this mixing leads to duplicated logic and maintenance headaches.

During the refactor, we discovered that 80% of the parameter form logic was identical between frameworks - only the widget creation differed. The service layer pattern extracts this shared logic into framework-agnostic classes.

Framework-Agnostic Services
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Separate business logic into dedicated service classes:

.. code-block:: python

    # DO: Service layer for business logic
    class ParameterFormService:
        def analyze_parameters(self, parameters: Dict[str, Any],
                              parameter_types: Dict[str, Type]) -> FormStructure:
            # Business logic separated from UI
            structure = FormStructure()
            for name, param_type in parameter_types.items():
                info = self._analyze_parameter(name, param_type, parameters.get(name))
                structure.parameters.append(info)
            return structure

Service Integration
~~~~~~~~~~~~~~~~~~~

UI frameworks consume services without business logic:

.. code-block:: python

    # PyQt6 Implementation
    class PyQt6FormManager:
        def __init__(self):
            self.service = ParameterFormService()

        def build_form(self, params, types):
            structure = self.service.analyze_parameters(params, types)
            for param_info in structure.parameters:
                widget = self._create_widget(param_info)
                self.layout.addWidget(widget)

    # Textual Implementation
    class TextualFormManager:
        def __init__(self):
            self.service = ParameterFormService()  # Same service

        def compose(self, params, types):
            structure = self.service.analyze_parameters(params, types)
            for param_info in structure.parameters:
                yield self._create_textual_widget(param_info)

Anti-Pattern: Mixed Concerns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # DON'T: Business logic in UI
    class BadFormManager:
        def build_form(self, params, types):
            for name, param_type in types.items():
                # Analysis logic mixed with UI
                if dataclasses.is_dataclass(param_type):
                    fields = dataclasses.fields(param_type)
                    # More logic...
                widget = QLineEdit()  # UI creation mixed in

Benefits: Framework independence, testability, maintainability, reusability.

Utility Classes Overview
------------------------

The refactor created eight utility classes that encapsulate common patterns. These aren't just code organization - they solve specific problems that kept recurring across the codebase.

**The Pattern:** Instead of scattering related functionality across multiple files, we grouped related operations into focused utility classes. Each class has a single responsibility and can be used by both UI frameworks.

Core Classes
~~~~~~~~~~~~

**EnumDisplayFormatter**
  Centralized enum formatting for consistent display.

  - Methods: ``get_display_text()``, ``get_placeholder_text()``
  - Support: PyQt6 + Textual
  - Usage: Replace scattered enum formatting logic

**FieldPathDetector** (``openhcs/core/field_path_detection.py``)
  Automatic field path detection for dataclass introspection.

  - Methods: ``find_field_path_for_type()``
  - Support: Framework-agnostic
  - Usage: Dynamic field path resolution

**ParameterFormService**
  Framework-agnostic business logic for parameter forms.

  - Methods: ``analyze_parameters()``, ``get_parameter_display_info()``
  - Support: PyQt6 + Textual
  - Usage: Shared service layer

**ParameterTypeUtils**
  Type introspection utilities for parameter analysis.

  - Methods: ``is_optional_dataclass()``, ``get_optional_inner_type()``
  - Support: Framework-agnostic
  - Usage: Type analysis for widget creation

Supporting Classes
~~~~~~~~~~~~~~~~~~

**ParameterFormBase**
  Abstract base class and shared configuration.

  - Components: ``ParameterFormConfig``, ``ParameterFormManagerBase``
  - Support: PyQt6 + Textual
  - Usage: Base class for form implementations

**ParameterNameFormatter**
  Consistent parameter name formatting.

  - Methods: ``to_display_name()``, ``to_field_label()``
  - Support: PyQt6 + Textual
  - Usage: Consistent parameter labeling

**FieldIdGenerator**
  Unique field ID generation.

  - Methods: ``generate_field_id()``, ``generate_widget_id()``
  - Support: PyQt6 + Textual
  - Usage: Collision-free identification

**ParameterFormConstants**
  Centralized constants eliminating magic strings.

  - Categories: UI text, widget naming, framework constants
  - Support: PyQt6 + Textual
  - Usage: Single source of truth for hardcoded values

Quick Reference
---------------

Practical do/don't examples for common UI implementation scenarios.

Widget Creation
~~~~~~~~~~~~~~~

.. code-block:: python

    # DO: Dispatch tables for widget creation
    WIDGET_FACTORIES = {
        bool: lambda: QCheckBox(),
        int: lambda: NoScrollSpinBox(),
        str: lambda: QLineEdit(),
        Path: lambda: EnhancedPathWidget(),
    }

    def create_widget(param_type: Type) -> QWidget:
        factory = WIDGET_FACTORIES.get(param_type)
        return factory() if factory else QLineEdit()

    # DON'T: Verbose if/elif chains
    def create_widget_bad(param_type: Type) -> QWidget:
        if param_type == bool:
            return QCheckBox()
        elif param_type == int:
            return NoScrollSpinBox()
        # ... many more conditions

Enum Handling
~~~~~~~~~~~~~

.. code-block:: python

    # DO: Use EnumDisplayFormatter
    from openhcs.ui.shared.enum_display_formatter import EnumDisplayFormatter

    def populate_combo(combo: QComboBox, enum_class: Type[Enum]):
        for enum_value in enum_class:
            text = EnumDisplayFormatter.get_display_text(enum_value)
            combo.addItem(text, enum_value)

    # DON'T: Hardcode enum formatting
    def populate_combo_bad(combo: QComboBox, enum_class: Type[Enum]):
        for enum_value in enum_class:
            text = enum_value.name.upper()  # Hardcoded
            combo.addItem(text, enum_value)

Constants Usage
~~~~~~~~~~~~~~~

.. code-block:: python

    # DO: Use centralized constants
    from openhcs.ui.shared.parameter_form_constants import CONSTANTS

    def setup_widget(widget: QWidget):
        widget.setProperty(CONSTANTS.WIDGET_TYPE_PROPERTY,
                          CONSTANTS.PARAMETER_WIDGET_TYPE)

    # DON'T: Magic strings
    def setup_widget_bad(widget: QWidget):
        widget.setProperty("widget_type", "parameter_widget")

Key Principles
~~~~~~~~~~~~~~

1. Use dispatch tables instead of if/elif chains
2. Extract business logic into service classes
3. Centralize formatting using utility classes
4. Eliminate magic strings using constants
5. Generate IDs systematically

When to Apply These Patterns
----------------------------

**Use Functional Dispatch When:**
- You have 3+ different types that need different handling
- You find yourself writing the same if/elif pattern repeatedly
- You need to add new types frequently
- Performance matters (dispatch is O(1) vs O(n) for if/elif)

**Use Service Layer When:**
- You have multiple UI frameworks or might add more
- Business logic is mixed with presentation code
- You're duplicating logic across different parts of the system
- You want to unit test business logic without UI dependencies

**Use Utility Classes When:**
- You have related functions scattered across multiple files
- The same formatting/conversion logic appears in multiple places
- You need consistent behavior across different frameworks
- You want to eliminate magic strings and hardcoded values

**Signs You Need These Patterns:**
- Copy-pasting code between UI implementations
- Bugs that require fixes in multiple places
- Difficulty testing business logic
- Long if/elif chains for type checking
- Magic strings scattered throughout the codebase


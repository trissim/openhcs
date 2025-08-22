UI Utilities Migration Guide
============================

Migration from class-based to functional utilities, achieving 94% code reduction while maintaining full functionality and establishing functional programming as the preferred pattern.

Overview
--------

The UI refactor consolidated four separate utility modules (975 lines) into a single functional module (57 lines), representing one of the most dramatic code reductions in the project.

**The Challenge:** OpenHCS had scattered utility classes across multiple modules, each with over-engineered abstractions that violated the project's simplicity principles. These utilities were duplicated across PyQt6 and Textual frameworks with inconsistent interfaces.

**The Solution:** A single `ui_utils.py` module with 8 pure functional utilities that work across all UI frameworks. The migration eliminated class hierarchies, reduced complexity, and established functional programming as the preferred pattern.

**Impact:** 94% code reduction, unified cross-framework utilities, and a template for future utility development.

Migration from Class-Based Utilities
-------------------------------------

The migration transformed complex class-based abstractions into simple functional utilities.

Before: Scattered Class-Based Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original approach used separate modules with class-based abstractions:

.. code-block:: python

    # debug_config.py (321 lines) - Over-engineered debugging
    class DebugConfigManager:
        def __init__(self, config_type, debug_level="INFO"):
            self.config_type = config_type
            self.debug_level = debug_level
            self.logger = self._setup_logger()
        
        def _setup_logger(self):
            # 50+ lines of logger configuration
            pass
        
        def log_parameter_change(self, param_name, old_value, new_value):
            # Complex logging logic with formatting
            pass
        
        def log_form_creation(self, form_type, parameters):
            # Detailed form creation logging
            pass
        
        # ... 15+ more methods

.. code-block:: python

    # enum_display_formatter.py (170 lines) - Class-based enum formatting
    class EnumDisplayFormatter:
        def __init__(self, enum_type, display_style="UPPER"):
            self.enum_type = enum_type
            self.display_style = display_style
            self._cache = {}
        
        def get_display_text(self, enum_value):
            if enum_value in self._cache:
                return self._cache[enum_value]
            # Complex formatting logic
            pass
        
        def get_placeholder_text(self, enum_value, prefix="Default"):
            # Placeholder generation with caching
            pass
        
        # ... 8+ more methods

.. code-block:: python

    # parameter_name_formatter.py (276 lines) - Complex name formatting
    class ParameterNameFormatter:
        def __init__(self, naming_convention="snake_case"):
            self.naming_convention = naming_convention
            self.formatters = self._build_formatters()
        
        def _build_formatters(self):
            # Complex formatter registry
            pass
        
        def format_display_name(self, param_name):
            # Multi-step formatting pipeline
            pass
        
        def format_checkbox_label(self, param_name):
            # Checkbox-specific formatting
            pass
        
        # ... 12+ more methods

.. code-block:: python

    # field_id_generator.py (208 lines) - ID generation with validation
    class FieldIdGenerator:
        def __init__(self, prefix="", separator="_"):
            self.prefix = prefix
            self.separator = separator
            self.generated_ids = set()
        
        def generate_field_id(self, parent, param):
            # Complex ID generation with collision detection
            pass
        
        def validate_id(self, field_id):
            # ID validation logic
            pass
        
        # ... 10+ more methods

After: Unified Functional Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The new approach uses simple functional utilities in a single module:

.. code-block:: python

    # ui_utils.py (57 lines) - Simple functional utilities
    
    def format_param_name(name: str) -> str:
        """Convert snake_case to Title Case: 'param_name' -> 'Param Name'"""
        return name.replace('_', ' ').title()
    
    def format_checkbox_label(name: str) -> str:
        """Create checkbox label: 'param_name' -> 'Enable Param Name'"""
        return f"Enable {format_param_name(name)}"
    
    def format_field_label(name: str) -> str:
        """Create field label: 'param_name' -> 'Param Name:'"""
        return f"{format_param_name(name)}:"
    
    def format_field_id(parent: str, param: str) -> str:
        """Generate field ID: 'parent', 'param' -> 'parent_param'"""
        return f"{parent}_{param}"
    
    def format_reset_button_id(widget_id: str) -> str:
        """Generate reset button ID: 'widget_id' -> 'reset_widget_id'"""
        return f"reset_{widget_id}"
    
    def format_enum_display(enum_value: Enum) -> str:
        """Get enum display text: Enum.VALUE -> 'VALUE'"""
        return enum_value.name.upper()
    
    def log_debug(message: str, level: str = "DEBUG") -> None:
        """Simple debug logging: message -> logger output"""
        logging.getLogger("openhcs.ui").log(getattr(logging, level), message)
    
    def get_widget_value(widget: Any) -> Any:
        """Get widget value using framework-agnostic approach."""
        if hasattr(widget, 'value'):
            return widget.value
        elif hasattr(widget, 'text'):
            return widget.text()
        elif hasattr(widget, 'isChecked'):
            return widget.isChecked()
        return None

**Key Transformation Principles:**

1. **Eliminate State**: No instance variables or caching
2. **Pure Functions**: Same input always produces same output
3. **Single Responsibility**: Each function does one thing well
4. **Framework Agnostic**: Works with PyQt6, Textual, and future frameworks
5. **Fail-Loud**: No defensive programming or silent failures

Cross-Framework Utility Usage
------------------------------

The functional utilities work seamlessly across different UI frameworks.

PyQt6 Usage Patterns
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # PyQt6 parameter form usage
    from openhcs.ui.shared.ui_utils import (
        format_param_name, format_field_id, format_checkbox_label
    )
    
    class PyQtParameterForm:
        def create_parameter_widget(self, param_name: str, parent_id: str):
            # Use functional utilities
            display_name = format_param_name(param_name)
            field_id = format_field_id(parent_id, param_name)
            
            # Create PyQt6 widgets
            label = QLabel(f"{display_name}:")
            widget = QLineEdit()
            widget.setObjectName(field_id)
            
            return label, widget
        
        def create_checkbox_widget(self, param_name: str):
            checkbox_text = format_checkbox_label(param_name)
            checkbox = QCheckBox(checkbox_text)
            return checkbox

Textual TUI Usage Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Textual TUI usage - same utilities
    from openhcs.ui.shared.ui_utils import (
        format_param_name, format_field_id, get_widget_value
    )
    
    class TextualParameterForm:
        def create_parameter_widget(self, param_name: str, parent_id: str):
            # Same functional utilities work here
            display_name = format_param_name(param_name)
            field_id = format_field_id(parent_id, param_name)
            
            # Create Textual widgets
            label = Static(f"{display_name}:")
            widget = Input(id=field_id)
            
            return label, widget
        
        def get_form_values(self, widgets):
            # Framework-agnostic value extraction
            values = {}
            for widget in widgets:
                value = get_widget_value(widget)
                values[widget.id] = value
            return values

Universal Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The utilities are designed to work with any UI framework:

.. code-block:: python

    # Framework-agnostic usage
    def create_form_for_any_framework(parameters, framework_type):
        """Create form using functional utilities - works with any framework."""
        
        widgets = []
        for param_name in parameters:
            # Universal formatting
            display_name = format_param_name(param_name)
            field_id = format_field_id("form", param_name)
            
            if framework_type == "pyqt6":
                widget = create_pyqt6_widget(display_name, field_id)
            elif framework_type == "textual":
                widget = create_textual_widget(display_name, field_id)
            elif framework_type == "future_framework":
                widget = create_future_widget(display_name, field_id)
            
            widgets.append(widget)
        
        return widgets

Functional Programming Adoption
--------------------------------

The migration established functional programming as the preferred pattern for UI utilities.

Functional Programming Benefits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Functional approach benefits:
    
    # 1. Composability - functions can be easily combined
    def create_full_label(param_name: str) -> str:
        return format_field_label(format_param_name(param_name))
    
    # 2. Testability - pure functions are easy to test
    def test_format_param_name():
        assert format_param_name("test_param") == "Test Param"
        assert format_param_name("another_test") == "Another Test"
    
    # 3. Predictability - same input always produces same output
    result1 = format_checkbox_label("enable_feature")
    result2 = format_checkbox_label("enable_feature")
    assert result1 == result2  # Always true
    
    # 4. No side effects - functions don't modify global state
    original_name = "test_param"
    formatted = format_param_name(original_name)
    assert original_name == "test_param"  # Unchanged

Functional Patterns in UI Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Functional mapping for widget creation
    def create_widgets_functional(parameters):
        """Create widgets using functional mapping."""
        return [
            create_widget(param_name, format_param_name(param_name))
            for param_name in parameters
        ]
    
    # Functional filtering for widget validation
    def get_valid_widgets(widgets):
        """Filter widgets using functional approach."""
        return [
            widget for widget in widgets
            if get_widget_value(widget) is not None
        ]
    
    # Functional reduction for form values
    def collect_form_values(widgets):
        """Collect form values using functional reduction."""
        from functools import reduce
        
        def add_widget_value(acc, widget):
            value = get_widget_value(widget)
            if value is not None:
                acc[widget.id] = value
            return acc
        
        return reduce(add_widget_value, widgets, {})

Code Consolidation Strategies
-----------------------------

The migration used systematic strategies to achieve maximum code reduction while preserving functionality.

Elimination of Over-Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original utilities suffered from classic over-engineering patterns:

.. code-block:: python

    # Before: Over-engineered with unnecessary abstractions
    class ParameterNameFormatter:
        def __init__(self, naming_convention="snake_case"):
            self.naming_convention = naming_convention
            self.formatters = {
                'snake_case': self._format_snake_case,
                'camel_case': self._format_camel_case,
                'pascal_case': self._format_pascal_case
            }
            self.cache = {}
            self.validation_rules = self._build_validation_rules()

        def _build_validation_rules(self):
            # 30+ lines of validation rule construction
            pass

        def format_display_name(self, param_name):
            if param_name in self.cache:
                return self.cache[param_name]

            # Validate input
            if not self._validate_parameter_name(param_name):
                raise ValueError(f"Invalid parameter name: {param_name}")

            # Apply formatting strategy
            formatter = self.formatters[self.naming_convention]
            result = formatter(param_name)

            # Cache result
            self.cache[param_name] = result
            return result

.. code-block:: python

    # After: Simple functional approach
    def format_param_name(name: str) -> str:
        """Convert snake_case to Title Case: 'param_name' -> 'Param Name'"""
        return name.replace('_', ' ').title()

**Consolidation Principles:**

1. **Eliminate Caching**: Simple operations don't need caching overhead
2. **Remove Validation**: Fail-loud principle - let errors surface naturally
3. **Single Format**: OpenHCS uses snake_case consistently, no need for multiple formats
4. **No State**: Pure functions eliminate instance variables and complexity

Pattern Consolidation
~~~~~~~~~~~~~~~~~~~~~

Multiple similar patterns were consolidated into single implementations:

.. code-block:: python

    # Before: Separate methods for each label type
    class LabelFormatter:
        def format_field_label(self, name):
            return f"{self._format_base_name(name)}:"

        def format_checkbox_label(self, name):
            return f"Enable {self._format_base_name(name)}"

        def format_button_label(self, name):
            return f"{self._format_base_name(name)} Action"

        def format_group_label(self, name):
            return f"[{self._format_base_name(name)}]"

        def _format_base_name(self, name):
            # Complex base formatting logic
            pass

.. code-block:: python

    # After: Composable functional utilities
    def format_param_name(name: str) -> str:
        """Base formatting function."""
        return name.replace('_', ' ').title()

    def format_field_label(name: str) -> str:
        """Compose with base formatter."""
        return f"{format_param_name(name)}:"

    def format_checkbox_label(name: str) -> str:
        """Compose with base formatter."""
        return f"Enable {format_param_name(name)}"

Dependency Elimination
~~~~~~~~~~~~~~~~~~~~~~

The migration eliminated external dependencies and complex imports:

.. code-block:: python

    # Before: Complex dependency tree
    from openhcs.ui.shared.debug_config import DebugConfigManager
    from openhcs.ui.shared.enum_display_formatter import EnumDisplayFormatter
    from openhcs.ui.shared.parameter_name_formatter import ParameterNameFormatter
    from openhcs.ui.shared.field_id_generator import FieldIdGenerator
    from openhcs.ui.shared.validation_engine import ValidationEngine
    from openhcs.ui.shared.caching_manager import CachingManager

    class ComplexParameterForm:
        def __init__(self):
            self.debug_manager = DebugConfigManager("parameter_form")
            self.enum_formatter = EnumDisplayFormatter()
            self.name_formatter = ParameterNameFormatter()
            self.id_generator = FieldIdGenerator()
            self.validator = ValidationEngine()
            self.cache = CachingManager()

.. code-block:: python

    # After: Single import, no dependencies
    from openhcs.ui.shared.ui_utils import (
        format_param_name, format_field_id, format_checkbox_label,
        format_enum_display, log_debug
    )

    class SimpleParameterForm:
        def create_widget(self, param_name):
            # Direct functional calls, no object management
            display_name = format_param_name(param_name)
            field_id = format_field_id("form", param_name)
            log_debug(f"Created widget: {field_id}")

Migration Examples
------------------

These examples show specific before/after transformations from the migration.

Debug Configuration Migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Before: 321 lines of debug configuration
    class DebugConfigManager:
        def __init__(self, config_type, debug_level="INFO"):
            self.config_type = config_type
            self.debug_level = debug_level
            self.logger = self._setup_logger()
            self.formatters = self._setup_formatters()
            self.handlers = self._setup_handlers()

        def log_parameter_change(self, param_name, old_value, new_value):
            message = self.formatters['parameter_change'].format(
                param=param_name, old=old_value, new=new_value
            )
            self.logger.log(self._get_log_level(), message)

        # ... 15+ more methods

.. code-block:: python

    # After: Single function
    def log_debug(message: str, level: str = "DEBUG") -> None:
        """Simple debug logging: message -> logger output"""
        logging.getLogger("openhcs.ui").log(getattr(logging, level), message)

    # Usage:
    log_debug(f"Parameter changed: {param_name} {old_value} -> {new_value}")

Enum Display Migration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Before: 170 lines of enum formatting
    class EnumDisplayFormatter:
        def __init__(self, enum_type, display_style="UPPER"):
            self.enum_type = enum_type
            self.display_style = display_style
            self._cache = {}
            self._reverse_cache = {}

        def get_display_text(self, enum_value):
            if enum_value in self._cache:
                return self._cache[enum_value]

            if self.display_style == "UPPER":
                result = enum_value.name.upper()
            elif self.display_style == "TITLE":
                result = enum_value.name.title()
            # ... more formatting options

            self._cache[enum_value] = result
            return result

.. code-block:: python

    # After: Single function
    def format_enum_display(enum_value: Enum) -> str:
        """Get enum display text: Enum.VALUE -> 'VALUE'"""
        return enum_value.name.upper()

    # Usage:
    display_text = format_enum_display(MyEnum.SOME_VALUE)

Field ID Generation Migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Before: 208 lines of ID generation
    class FieldIdGenerator:
        def __init__(self, prefix="", separator="_"):
            self.prefix = prefix
            self.separator = separator
            self.generated_ids = set()
            self.collision_handlers = {}

        def generate_field_id(self, parent, param):
            base_id = f"{self.prefix}{parent}{self.separator}{param}"

            if base_id in self.generated_ids:
                return self._handle_collision(base_id)

            self.generated_ids.add(base_id)
            return base_id

        def _handle_collision(self, base_id):
            # Complex collision resolution logic
            pass

.. code-block:: python

    # After: Single function
    def format_field_id(parent: str, param: str) -> str:
        """Generate field ID: 'parent', 'param' -> 'parent_param'"""
        return f"{parent}_{param}"

    # Usage:
    field_id = format_field_id("config", "output_dir")

Benefits
--------

- **94% Code Reduction**: 975 lines → 57 lines while maintaining full functionality
- **Cross-Framework Compatibility**: Same utilities work with PyQt6, Textual, and future frameworks
- **Functional Programming**: Establishes pure functional patterns as preferred approach
- **Zero Dependencies**: No external dependencies or complex abstractions
- **Easy Testing**: Pure functions are trivial to test and validate
- **Composability**: Functions can be easily combined for complex operations
- **Maintainability**: Simple functions are easy to understand and modify
- **Performance**: No object instantiation or method lookup overhead
- **Template for Future Development**: Establishes patterns for new utility development
- **Elimination of Over-Engineering**: Removes unnecessary abstractions and complexity

See Also
--------

- :doc:`ui-patterns` - UI patterns that use functional utilities and dispatch
- :doc:`integration-testing` - Testing framework that validates utility functionality
- :doc:`../architecture/service-layer-architecture` - Service layer patterns that complement functional utilities
- :doc:`../architecture/step-editor-generalization` - Step editors that use functional utilities

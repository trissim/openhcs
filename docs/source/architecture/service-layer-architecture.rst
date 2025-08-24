Service Layer Architecture
==========================

Framework-agnostic business logic extraction enabling cross-framework compatibility.

Overview
--------

The service layer architecture emerged from a practical problem: OpenHCS supports both PyQt6 and Textual UIs, but the business logic for parameter forms was duplicated between them. Every time we fixed a bug or added a feature, we had to implement it twice.

UI frameworks naturally encourage mixing presentation logic with business logic. PyQt6 widgets know how to display themselves, so it's tempting to put parameter analysis logic directly in the widget creation code. But this creates tight coupling and makes code reuse impossible.

The solution extracts all business logic into framework-agnostic service classes. UI frameworks become thin presentation layers that consume services. The same service can power both PyQt6 and Textual implementations.

This pattern eliminated duplicated code and reduced parameter form bugs since fixes only need to be made once.

Core Service Pattern
--------------------

The core pattern is simple: create classes that contain only business logic, with no dependencies on UI frameworks. These services take data as input and return data as output, making them easy to test and reuse.

Framework-Agnostic Services
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Services focus purely on the "what" and "how" of business logic, leaving the "where to display it" to UI frameworks.

.. code-block:: python

    class ParameterFormService:
        """Framework-agnostic service for parameter form business logic."""
        
        def __init__(self, debug_config: Optional[DebugConfig] = None):
            self.debugger = get_debugger(debug_config)
            self._field_id_generator = FieldIdGenerator()
            self._type_utils = ParameterTypeUtils()
            self._name_formatter = ParameterNameFormatter()
        
        def analyze_parameters(self, parameters: Dict[str, Any], 
                              parameter_types: Dict[str, Type],
                              field_id: str) -> FormStructure:
            """Analyze parameters and create framework-agnostic form structure."""
            # Business logic separated from UI framework
            structure = FormStructure(field_id=field_id)
            
            for param_name, param_type in parameter_types.items():
                param_info = self._analyze_single_parameter(
                    param_name, param_type, parameters.get(param_name)
                )
                structure.parameters.append(param_info)
            
            return structure

Service Integration Pattern
---------------------------

PyQt6 Integration
~~~~~~~~~~~~~~~~~

.. code-block:: python

    class PyQt6ParameterFormManager:
        """PyQt6-specific UI implementation using shared service."""
        
        def __init__(self):
            self.service = ParameterFormService()  # Inject service
            self.widget_strategies = PyQt6WidgetStrategies()
        
        def build_form(self, parameters: Dict[str, Any], 
                      parameter_types: Dict[str, Type], 
                      field_id: str):
            # Use service for business logic
            structure = self.service.analyze_parameters(parameters, parameter_types, field_id)
            
            # UI framework handles only presentation
            for param_info in structure.parameters:
                widget = self._create_widget(param_info)
                self.layout.addWidget(widget)

Textual Integration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class TextualParameterFormManager:
        """Textual-specific UI implementation using same service."""
        
        def __init__(self):
            self.service = ParameterFormService()  # Same service
            self.widget_strategies = TextualWidgetStrategies()
        
        def compose(self, parameters: Dict[str, Any], 
                   parameter_types: Dict[str, Type], 
                   field_id: str):
            # Identical business logic via service
            structure = self.service.analyze_parameters(parameters, parameter_types, field_id)
            
            # Different UI framework, same logic
            for param_info in structure.parameters:
                widget = self._create_textual_widget(param_info)
                yield widget

Shared Infrastructure
---------------------

Utility Class Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

Services coordinate multiple utility classes:

.. code-block:: python

    class ParameterFormService:
        def __init__(self):
            # Compose utility classes for shared functionality
            self._type_utils = ParameterTypeUtils()
            self._name_formatter = ParameterNameFormatter()
            self._field_id_generator = FieldIdGenerator()
            self._enum_formatter = EnumDisplayFormatter()
        
        def get_parameter_display_info(self, param_name: str, param_type: Type) -> ParameterDisplayInfo:
            """Coordinate utilities for parameter analysis."""
            return ParameterDisplayInfo(
                name=param_name,
                display_name=self._name_formatter.to_display_name(param_name),
                is_optional=self._type_utils.is_optional_dataclass(param_type),
                field_id=self._field_id_generator.generate_field_id(param_name),
                enum_options=self._enum_formatter.get_enum_options(param_type) if self._is_enum(param_type) else None
            )

Context-Aware Behavior
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def get_reset_values(self, dataclass_type: Type, current_config: Any,
                        is_global_config_editing: Optional[bool] = None) -> Dict[str, Any]:
        """Context-driven reset behavior."""
        
        # Auto-detect editing mode if not explicitly provided
        if is_global_config_editing is None:
            is_global_config_editing = not LazyDefaultPlaceholderService.has_lazy_resolution(dataclass_type)
        
        if is_global_config_editing:
            # Global config editing: Use actual default values
            return self._get_static_defaults(dataclass_type)
        else:
            # Lazy config editing: Use None to show placeholder text
            return {field.name: None for field in dataclasses.fields(dataclass_type)}

Cross-Framework Compatibility
-----------------------------

Shared Data Structures
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @dataclass
    class FormStructure:
        """Framework-agnostic form structure."""
        field_id: str
        parameters: List[ParameterInfo]
        nested_forms: Dict[str, 'FormStructure'] = field(default_factory=dict)
    
    @dataclass  
    class ParameterInfo:
        """Framework-agnostic parameter information."""
        name: str
        display_name: str
        param_type: Type
        is_optional: bool
        default_value: Any
        field_id: str

Strategy Pattern Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Framework-specific strategies consume service output
    class PyQt6WidgetStrategies:
        def create_widget_for_parameter(self, param_info: ParameterInfo) -> QWidget:
            # Use service-provided parameter info for widget creation
            if param_info.param_type == bool:
                widget = QCheckBox()
                widget.setChecked(param_info.default_value)
            elif param_info.is_optional:
                widget = self._create_optional_widget(param_info)
            return widget

Benefits
--------

- **Framework Independence**: Same business logic works across PyQt6 and Textual
- **Code Reuse**: Eliminates duplication between UI implementations
- **Testability**: Business logic can be unit tested without UI dependencies
- **Maintainability**: Changes to logic don't require UI modifications
- **Separation of Concerns**: Clear boundary between business logic and presentation
- **Context Awareness**: Services adapt behavior based on usage context

See Also
--------

- :doc:`configuration_system_architecture` - Configuration system architecture used by services
- :doc:`step-editor-generalization` - Step editors that use service layer patterns
- :doc:`../development/ui-patterns` - UI patterns that leverage service layer architecture

Unified Parameter Analyzer Migration Guide
==========================================

Overview
--------

The ``UnifiedParameterAnalyzer`` provides a single, consistent interface
for analyzing parameters from any source in OpenHCS TUI. This replaces
the fragmented approach of using different analyzers for different
parameter sources.

Before vs After
---------------

Before (Fragmented Approach)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Different analyzers for different sources
   from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer
   from openhcs.textual_tui.services.config_reflection_service import FieldIntrospector

   # Function analysis
   param_info = SignatureAnalyzer.analyze(my_function)

   # Dataclass analysis  
   field_specs = FieldIntrospector.analyze_dataclass(MyConfig, instance)

   # Different return types, different interfaces

After (Unified Approach)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Single analyzer for all sources
   from openhcs.textual_tui.widgets.shared.unified_parameter_analyzer import UnifiedParameterAnalyzer

   # Works for functions, dataclasses, instances, etc.
   param_info = UnifiedParameterAnalyzer.analyze(my_function)
   param_info = UnifiedParameterAnalyzer.analyze(MyConfig)
   param_info = UnifiedParameterAnalyzer.analyze(config_instance)

   # Consistent return type: Dict[str, UnifiedParameterInfo]

Usage Examples
--------------

Analyzing Functions
~~~~~~~~~~~~~~~~~~~

.. code:: python

   def my_function(param1: str, param2: int = 42) -> str:
       """Example function.
       
       Args:
           param1: Required string parameter
           param2: Optional integer with default
       """
       return f"{param1}_{param2}"

   # Analyze function
   param_info = UnifiedParameterAnalyzer.analyze(my_function)

   # Access parameter information
   for name, info in param_info.items():
       print(f"{name}: {info.param_type}, required={info.is_required}")
       print(f"  Description: {info.description}")
       print(f"  Default: {info.default_value}")

Analyzing Dataclasses
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   @dataclasses.dataclass
   class MyConfig:
       """Configuration dataclass.
       
       Args:
           setting1: Primary setting
           setting2: Secondary setting with default
       """
       setting1: str
       setting2: int = 100

   # Analyze dataclass type
   param_info = UnifiedParameterAnalyzer.analyze(MyConfig)

   # Analyze dataclass instance
   instance = MyConfig(setting1="test", setting2=200)
   param_info = UnifiedParameterAnalyzer.analyze(instance)

Nested Dataclass Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   @dataclasses.dataclass
   class NestedConfig:
       """Nested configuration.
       
       Args:
           nested_field: A nested field
           main_config: Main configuration object
       """
       nested_field: str = "default"
       main_config: MyConfig = dataclasses.field(default_factory=MyConfig)

   # Analyze with nested support
   param_info = UnifiedParameterAnalyzer.analyze_nested(NestedConfig)

   # Check for nested dataclasses
   for name, info in param_info.items():
       if info.source_type.endswith("_nested"):
           print(f"{name} contains nested dataclass: {info.param_type}")

Migration Steps
---------------

Step 1: Update Imports
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # OLD
   from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer
   from openhcs.textual_tui.services.config_reflection_service import FieldIntrospector

   # NEW
   from openhcs.textual_tui.widgets.shared.unified_parameter_analyzer import UnifiedParameterAnalyzer

Step 2: Replace Analysis Calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # OLD - Function analysis
   param_info = SignatureAnalyzer.analyze(func)

   # NEW - Unified analysis
   param_info = UnifiedParameterAnalyzer.analyze(func)

   # OLD - Dataclass analysis
   field_specs = FieldIntrospector.analyze_dataclass(ConfigClass, instance)

   # NEW - Unified analysis
   param_info = UnifiedParameterAnalyzer.analyze(ConfigClass)
   # or
   param_info = UnifiedParameterAnalyzer.analyze(instance)

Step 3: Update Parameter Access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # OLD - Different access patterns
   for name, param_info in signature_params.items():
       description = param_info.description
       param_type = param_info.param_type

   for field_spec in field_specs:
       description = field_spec.label  # No docstring info
       param_type = field_spec.actual_type

   # NEW - Consistent access pattern
   for name, param_info in unified_params.items():
       description = param_info.description  # Always available
       param_type = param_info.param_type
       is_required = param_info.is_required
       default_value = param_info.default_value
       source_type = param_info.source_type

Step 4: Update Form Manager Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # OLD - Inconsistent constructor calls
   form_manager = ParameterFormManager(params, types, id)  # Missing param_info
   form_manager = ParameterFormManager(params, types, id, param_info)  # With param_info

   # NEW - Always pass param_info
   param_info = UnifiedParameterAnalyzer.analyze(target)
   parameters = {name: info.default_value for name, info in param_info.items()}
   parameter_types = {name: info.param_type for name, info in param_info.items()}

   form_manager = ParameterFormManager(parameters, parameter_types, id, param_info)

Backward Compatibility
----------------------

The unified analyzer provides backward compatibility aliases:

.. code:: python

   # These work for existing code during migration
   from openhcs.textual_tui.widgets.shared.unified_parameter_analyzer import (
       ParameterAnalyzer,  # Alias for UnifiedParameterAnalyzer
       analyze_parameters  # Alias for UnifiedParameterAnalyzer.analyze
   )

   # Existing code continues to work
   param_info = ParameterAnalyzer.analyze(target)
   param_info = analyze_parameters(target)

Benefits of Migration
---------------------

Consistency
~~~~~~~~~~~

-  Single interface for all parameter sources
-  Consistent return types and access patterns
-  Unified help functionality across all forms

Maintainability
~~~~~~~~~~~~~~~

-  Single codebase to maintain instead of multiple analyzers
-  Consistent behavior across different parameter types
-  Easier to add new features (they work everywhere)

Developer Experience
~~~~~~~~~~~~~~~~~~~~

-  One pattern to learn instead of multiple
-  Clear migration path with backward compatibility
-  Comprehensive documentation and examples

User Experience
~~~~~~~~~~~~~~~

-  Consistent help functionality across all parameter forms
-  No more missing help buttons in some forms
-  Uniform behavior across the application

Common Migration Issues
-----------------------

Issue 1: Missing Parameter Info
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # PROBLEM: Old code doesn't pass param_info
   form_manager = ParameterFormManager(params, types, id)

   # SOLUTION: Always analyze and pass param_info
   param_info = UnifiedParameterAnalyzer.analyze(target)
   form_manager = ParameterFormManager(params, types, id, param_info)

Issue 2: Different Return Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # PROBLEM: FieldIntrospector returns FieldSpec objects
   field_specs = FieldIntrospector.analyze_dataclass(ConfigClass, instance)

   # SOLUTION: Use unified analyzer, convert if needed
   param_info = UnifiedParameterAnalyzer.analyze(ConfigClass)
   # param_info is Dict[str, UnifiedParameterInfo]

Issue 3: Nested Dataclass Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # PROBLEM: Nested forms don't get parameter info
   nested_form = ParameterFormManager(nested_params, nested_types, nested_id)

   # SOLUTION: Analyze nested dataclass and pass param_info
   nested_param_info = UnifiedParameterAnalyzer.analyze(nested_dataclass_type)
   nested_form = ParameterFormManager(nested_params, nested_types, nested_id, nested_param_info)

Testing
-------

Run the unified analyzer tests to ensure everything works:

.. code:: bash

   # Test the unified parameter analyzer implementation
   python -m pytest tests/textual_tui/ -k "parameter" -v

   # Or test the entire TUI system
   python -m pytest tests/textual_tui/ -v

Next Steps
----------

1. Migrate existing code to use ``UnifiedParameterAnalyzer``
2. Remove ``FieldIntrospector`` once all usage is migrated
3. Update ``SignatureAnalyzer`` to use unified interface internally
4. Add comprehensive tests for all parameter sources
5. Update documentation and examples

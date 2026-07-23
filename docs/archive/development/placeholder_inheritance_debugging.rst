Placeholder Inheritance Debugging Guide
=======================================

**Systematic debugging approaches for the simplified OpenHCS placeholder inheritance system.**

*Status: STABLE - Simplified Implementation*
*Module: openhcs.pyqt_gui.widgets.shared.parameter_form_manager*

Overview
--------

The simplified placeholder inheritance system enables configuration fields to inherit values from parent configurations through contextvars-based lazy dataclass resolution. When users click reset buttons on inherited fields, the system uses explicit context management to allow proper inheritance resolution. This guide provides systematic debugging approaches for inheritance chain failures.

The system now uses explicit contextvars-based context management, eliminating complex context discovery and frame injection mechanisms while maintaining full inheritance functionality.

Understanding the inheritance flow is essential for debugging: child fields with ``None`` values trigger lazy resolution, which checks thread-local context for parent field values, then generates placeholder text showing inherited values like "Pipeline default: {inherited_value}".

System Architecture
-------------------

Core Components
~~~~~~~~~~~~~~~

**Form Managers**: UI components that manage individual configuration sections. Each form manager has a ``field_id`` that identifies its configuration section and manages widgets for that section's parameters.

**Context Building**: Process of creating configuration context for inheritance resolution by collecting current values from all form managers and combining widget state with parameter values.

**Exclusion Logic**: Mechanism to exclude specific fields during reset operations, implemented in the widget reading loop of :py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager._build_context_from_current_form_values`.

**Lazy Resolution**: On-demand value resolution that checks thread-local context for parent field values and falls back to defaults when inheritance chains are available.

Field Naming Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~

The system handles two distinct field naming patterns that require different context building approaches:

**Root Config Form Managers** use ``field_id`` values that match the dataclass type name (e.g., ``GlobalPipelineConfig``, ``PipelineConfig``). These represent the entire root configuration object, not a nested field within it. Widget names follow the pattern ``GlobalPipelineConfig.num_workers``.

**Nested Form Managers** use ``field_id`` values that match actual dataclass field names (e.g., ``well_filter_config``, ``zarr_config``). These correspond directly to context fields with the same names. Widget names follow the pattern ``well_filter_config.well_filter``.

The critical fixes eliminate artificial field naming:

1. **Nested Form Managers**: Removed ``nested_`` prefix hack, now use actual field names directly
2. **Root Config Form Managers**: Changed from artificial ``config`` field_id to dataclass type name

Root vs Nested Detection Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system uses generic detection logic to distinguish root configs from nested configs:

.. code-block:: python

   def _build_context_from_current_form_values(self, exclude_field=None):
       current_context = get_current_global_config(GlobalPipelineConfig)

       # Generic root vs nested detection
       if hasattr(current_context, self.field_id):
           # Nested config: field_id exists as actual field in context
           # Examples: "well_filter_config", "zarr_config"
           current_dataclass_instance = getattr(current_context, self.field_id)
       else:
           # Root config: field_id doesn't exist as field in context
           # Examples: "GlobalPipelineConfig", "PipelineConfig"
           current_dataclass_instance = current_context

This pattern works generically for any dataclass hierarchy without hardcoding specific class names.

Context Building Process
------------------------

Normal Context Building Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager._build_context_from_current_form_values` orchestrates context building by iterating through all form managers to collect current values. For each form manager, it reads widget values and combines them with parameter values, then builds context with all current form state for lazy dataclass resolution.

The method first checks ``hasattr(current_context, context_field_name)`` to verify the form's dataclass exists in context. If found, it enters the widget reading loop where current widget values are collected and combined with existing parameter values.

Context Building During Reset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reset operations call context building with an ``exclude_field`` parameter to exclude the target field from widget value reading. The exclusion logic in the widget reading loop checks ``exclude_field and param_name == exclude_field`` and skips reading the widget value for excluded fields.

This allows the context to be built without the field being reset, enabling lazy resolution to inherit from parent configurations instead of using the current (soon-to-be-reset) value.

The reset flow: user clicks reset → :py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager.reset_parameter` calls context building with exclusion → context built without target field → lazy resolution inherits from parent → placeholder text updated with inherited value.

Debugging Patterns
------------------

Essential Debug Output
~~~~~~~~~~~~~~~~~~~~~~

Add these debug prints to trace the inheritance system:

.. code-block:: python

   # In _build_context_from_current_form_values()
   print(f"🔍 CONTEXT BUILD DEBUG: {self.field_id} building context with exclude_field='{exclude_field}'")

   # Root vs nested detection debugging
   if hasattr(current_context, self.field_id):
       print(f"🔍 CONTEXT DEBUG: NESTED CONFIG - field_id='{self.field_id}' found in context")
   else:
       print(f"🔍 CONTEXT DEBUG: ROOT CONFIG - field_id='{self.field_id}' using current_context directly")

   # In widget reading loop
   print(f"🔍 EXCLUSION DEBUG: Checking param_name='{param_name}' vs exclude_field='{exclude_field}'")
   if exclude_field and param_name == exclude_field:
       print(f"🔍 WIDGET DEBUG: {self.field_id}.{param_name} EXCLUDED from context (reset)")

   # In lazy resolution code
   print(f"🔍 LAZY RESOLUTION DEBUG: Resolving {dataclass_name}.{field_name}")
   print(f"🔍 LAZY RESOLUTION DEBUG: Context has {parent_field} = '{parent_value}'")

Common Debugging Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Root Config Context Building Failure**:

.. code-block:: python

   # Symptom: Non-nested fields (num_workers, materialization_results_path) don't reset placeholders
   # Cause: Using artificial field_id="config" instead of dataclass type name

   # WRONG:
   form_manager = ParameterFormManager.from_dataclass_instance(
       field_id="config"  # ❌ GlobalPipelineConfig has no "config" field
   )

   # CORRECT:
   form_manager = ParameterFormManager.from_dataclass_instance(
       field_id=type(current_config).__name__  # ✅ "GlobalPipelineConfig"
   )

**Nested Config Field Path Issues**:

.. code-block:: python

   # Symptom: Nested fields don't update placeholders after sibling changes
   # Cause: Using artificial "nested_" prefix instead of actual field names

   # WRONG:
   nested_manager = ParameterFormManager(..., field_id="nested_well_filter_config")

   # CORRECT:
   field_path = FieldPathDetector.find_field_path_for_type(parent_type, nested_type)
   nested_manager = ParameterFormManager(..., field_id=field_path)  # "well_filter_config"

Successful Operation Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Successful Root Config Context Building**:

.. code-block:: text

   🔍 CONTEXT DEBUG: ROOT CONFIG - field_id='GlobalPipelineConfig' using current_context directly
   🔍 CONTEXT DEBUG: current_dataclass_instance=GlobalPipelineConfig(...)

**Successful Nested Config Context Building**:

.. code-block:: text

   🔍 CONTEXT DEBUG: NESTED CONFIG - field_id='well_filter_config' found in context
   🔍 CONTEXT DEBUG: current_dataclass_instance=WellFilterConfig(...)

**Successful Exclusion**:

.. code-block:: text

   🔍 EXCLUSION DEBUG: Checking param_name='well_filter' vs exclude_field='well_filter'
   🔍 WIDGET DEBUG: nested_step_materialization_config.well_filter EXCLUDED from context (reset)

**Successful Inheritance**:

.. code-block:: text

   🔍 LAZY RESOLUTION DEBUG: Resolving StepMaterializationConfig.well_filter
   🔍 LAZY RESOLUTION DEBUG: Context has step_well_filter_config.well_filter = '789'
   🔍 OVERRIDE CHECK: StepMaterializationConfig.well_filter default='None' has_override=False
   🔍 OVERRIDE CHECK: StepWellFilterConfig.well_filter default='1' has_override=True

Common Failure Patterns
-----------------------

Context Building Failures
~~~~~~~~~~~~~~~~~~~~~~~~~

**Field Naming Mismatch**:

.. code-block:: text

   🔍 CONTEXT DEBUG: form_field_name='nested_well_filter_config', context_field_name='nested_well_filter_config', hasattr=False

This indicates the field naming fix is not applied. The ``context_field_name`` should strip the ``nested_`` prefix to match the actual context field name.

**Investigation Steps**: 1. Verify the prefix stripping logic is applied 2. Check if ``field_id`` follows expected naming patterns 3. Confirm context contains the expected field names

Exclusion Logic Failures
~~~~~~~~~~~~~~~~~~~~~~~~

**Exclusion Not Working**:

.. code-block:: text

   🔍 WIDGET DEBUG: nested_step_materialization_config.well_filter widget value = 'some_value'

If widget values are being read for the excluded field, exclusion is not working.

**Investigation Steps**: 1. Verify context building reaches the widget reading loop 2. Check if ``exclude_field`` parameter is passed correctly 3. Confirm ``param_name`` matches ``exclude_field`` exactly

Inheritance Resolution Failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Wrong Inheritance Chain**:

.. code-block:: text

   🔍 LAZY RESOLUTION DEBUG: Context has wrong_parent_field = 'unexpected_value'

This indicates context contains wrong parent values or inheritance logic errors.

**Investigation Steps**: 1. Trace lazy resolution debug output to verify inheritance path 2. Check if context building collected correct parent values 3. Verify inheritance decorators are properly configured

Testing and Validation
----------------------

Automated Testing
~~~~~~~~~~~~~~~~~

Use :py:func:`tests.pyqt_gui.functional.test_reset_placeholder_simplified.TestResetPlaceholderInheritance.test_comprehensive_inheritance_chains` to validate inheritance chains. This test verifies multiple inheritance levels, reset functionality, and placeholder text accuracy through UI automation.

Manual Testing Protocol
~~~~~~~~~~~~~~~~~~~~~~~

1. **Set Parent Field**: Set parent field to concrete value (e.g., ``step_well_filter_config.well_filter = "789"``)
2. **Verify Child Inheritance**: Verify child field shows inherited placeholder (e.g., ``step_materialization_config.well_filter`` shows "Pipeline default: 789")
3. **Test Reset Functionality**: Reset child field and verify placeholder updates correctly
4. **Test Parent Reset**: Reset parent field and verify child field updates to new inherited value
5. **Validate Chain Propagation**: Test multiple levels of inheritance to ensure chains propagate correctly

Known Issues and Limitations
----------------------------

Non-Nested Field Reset Bug
~~~~~~~~~~~~~~~~~~~~~~~~~~

Non-nested fields don't reset placeholder values properly when a config is saved and reopened. When reopened, resetting a concrete non-nested field causes the placeholder to show the concrete value instead of the inherited value.

This appears related to how the configuration cache system interacts with reset functionality for non-nested fields, where concrete values become part of cached context and reset logic may not properly exclude them.

Field Path Validation
~~~~~~~~~~~~~~~~~~~~~

**Validation Checks**:

.. code-block:: python

   def validate_field_path_mapping():
       """Ensure all form field_ids map correctly to context fields"""
       from openhcs.core.config import GlobalPipelineConfig
       import dataclasses

       # Get all dataclass fields from GlobalPipelineConfig
       context_fields = {f.name for f in dataclasses.fields(GlobalPipelineConfig)
                        if dataclasses.is_dataclass(f.type)}

       # Verify form managers use these exact field names (no artificial prefixes)
       assert "well_filter_config" in context_fields
       assert "nested_well_filter_config" not in context_fields  # Should not exist

       return True

**Root Config Validation**:

.. code-block:: python

   def validate_root_config_field_id(form_manager, config_instance):
       """Ensure root config form managers use dataclass type name as field_id"""
       expected_field_id = type(config_instance).__name__
       actual_field_id = form_manager.field_id

       assert actual_field_id == expected_field_id, f"Root config field_id should be '{expected_field_id}', got '{actual_field_id}'"

       # Verify this field_id doesn't exist as a field in the config
       assert not hasattr(config_instance, actual_field_id), f"field_id '{actual_field_id}' should not exist as field in {type(config_instance).__name__}"

Architecture Improvements Implemented
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The field path system redesign has eliminated the fragile ``nested_`` string prefix dependencies:

**✅ Completed Improvements**:

- **Eliminated artificial field naming**: No more ``nested_`` prefixes or ``config`` field_ids
- **Direct field path mapping**: Form managers use actual dataclass field names
- **Root config detection**: Generic ``hasattr()`` logic works for any dataclass hierarchy
- **Context building alignment**: Field paths match dataclass structure exactly
- **Visual programming compliance**: UI field names directly reflect code structure

This comprehensive debugging approach helps identify whether issues are in context building, exclusion logic, inheritance resolution, or field path mapping.

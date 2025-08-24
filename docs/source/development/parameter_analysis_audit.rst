Parameter Analysis Architecture Audit
=====================================

Overview
--------

This document audits the current parameter analysis systems in OpenHCS
TUI and identifies architectural inconsistencies that need to be
addressed.

Current Systems
---------------

1. SignatureAnalyzer (``openhcs/textual_tui/widgets/shared/signature_analyzer.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Extract parameter information from functions and
dataclasses **Features**: - ✅ Docstring parsing with multiline support
- ✅ AST-based parsing with regex fallback - ✅ Handles both functions
and dataclasses - ✅ Returns ``ParameterInfo`` objects with descriptions
- ✅ Supports Google, NumPy, and Sphinx docstring formats

**Usage Patterns**:

.. code:: python

   # Function forms (working correctly)
   param_info = SignatureAnalyzer.analyze(self.func)
   form_manager = ParameterFormManager(params, types, id, param_info)

   # Config forms (recently fixed)
   param_info = SignatureAnalyzer.analyze(GlobalPipelineConfig)
   form_manager = ParameterFormManager(params, types, id, param_info)

   # Nested dataclasses (within ParameterFormManager)
   nested_param_info = SignatureAnalyzer.analyze(param_type)

2. FieldIntrospector (``openhcs/textual_tui/services/config_reflection_service.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Analyze dataclass fields for form generation **Features**:
- ❌ No docstring parsing - ✅ Field metadata extraction - ✅ Nested
dataclass support - ✅ Returns ``FieldSpec`` objects - ❌ No help
functionality

**Usage Patterns**:

.. code:: python

   # Config dialogs and screens
   field_specs = FieldIntrospector.analyze_dataclass(config_class, instance)
   config_form = ConfigFormWidget(field_specs)

3. ParameterFormManager (``openhcs/textual_tui/widgets/shared/parameter_form_manager.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Generate parameter forms from analyzed data **Constructor
Signatures**:

.. code:: python

   # Correct usage (with parameter_info)
   ParameterFormManager(parameters, parameter_types, field_id, parameter_info)

   # Incorrect usage (missing parameter_info)
   ParameterFormManager(parameters, parameter_types, field_id)

Current Usage Analysis
----------------------

✅ Correct Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **FunctionPaneWidget**
   (``openhcs/textual_tui/widgets/function_pane.py:42-46``)

   .. code:: python

      param_info = SignatureAnalyzer.analyze(self.func)
      self.form_manager = ParameterFormManager(parameters, parameter_types, f"func_{index}", param_info)

2. **ConfigFormWidget**
   (``openhcs/textual_tui/widgets/config_form.py:49-54``)

   .. code:: python

      param_info = SignatureAnalyzer.analyze(dataclass_type)
      self.form_manager = ParameterFormManager(parameters, parameter_types, "config", param_info)

❌ Inconsistent Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **StepParameterEditor**
   (``openhcs/textual_tui/widgets/step_parameter_editor.py:34-49``)

   .. code:: python

      # INCONSISTENCY: Analyzes parameters but doesn't pass param_info to form manager!
      param_info = SignatureAnalyzer.analyze(FunctionStep.__init__)  # ✅ Gets parameter info

      # ... processes param_info to build parameters dict ...

      # ❌ But doesn't pass param_info to ParameterFormManager!
      self.form_manager = ParameterFormManager(parameters, parameter_types, "step")
      # Should be: ParameterFormManager(parameters, parameter_types, "step", param_info)

2. **Nested Dataclass Forms**
   (``openhcs/textual_tui/widgets/shared/parameter_form_manager.py:57-60``)

   .. code:: python

      # ❌ Nested forms created without parameter_info!
      nested_form_manager = ParameterFormManager(
          nested_parameters,
          nested_parameter_types,
          nested_field_id
      )
      # Should analyze nested dataclass: SignatureAnalyzer.analyze(param_type)

3. **ConfigFormScreen**
   (``openhcs/textual_tui/screens/config_form.py:88-114``)

   .. code:: python

      # Duplicate form creation logic - bypasses ParameterFormManager entirely!
      # Creates widgets manually instead of using shared form manager

Architectural Issues Identified
-------------------------------

1. **Dual Analysis Systems**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``SignatureAnalyzer`` and ``FieldIntrospector`` do overlapping work
-  ``FieldIntrospector`` lacks docstring parsing capabilities
-  Different return types: ``ParameterInfo`` vs ``FieldSpec``

2. **Inconsistent Constructor Usage**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Some components pass ``parameter_info``, others don’t
-  Missing help functionality in components without ``parameter_info``

3. **Duplicate Form Creation Logic**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``ConfigFormScreen`` creates widgets manually
-  ``ParameterFormManager`` creates widgets systematically
-  Two different approaches for the same goal

4. **Missing Help in Nested Forms**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Nested dataclass forms don’t receive ``parameter_info``
-  No help buttons for nested parameters

5. **Architectural Drift**
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Function forms evolved with ``SignatureAnalyzer``
-  Config forms evolved with ``FieldIntrospector``
-  Recent fixes created hybrid approaches

Impact Assessment
-----------------

Current State
~~~~~~~~~~~~~

-  **Function parameters**: ✅ Full help functionality (param_info
   passed correctly)
-  **Config parameters**: ✅ Help functionality (recently fixed -
   param_info now passed)
-  **Step parameters**: ❌ No help functionality (param_info analyzed
   but not passed)
-  **Nested parameters**: ❌ No help functionality (param_info never
   analyzed for nested forms)
-  **Manual config forms**: ❌ No help functionality (bypasses
   ParameterFormManager entirely)

User Experience Impact
~~~~~~~~~~~~~~~~~~~~~~

-  **Inconsistent help availability**: Function params have (?) help,
   step params don’t
-  **Confusing UX**: Similar-looking parameter forms behave differently
-  **Missing documentation**: Users can’t get help for step
   configuration
-  **Architectural confusion**: Developers don’t know which pattern to
   follow

Technical Debt
~~~~~~~~~~~~~~

-  **Code duplication**: Multiple form creation patterns for same goal
-  **Maintenance burden**: Changes need to be made in multiple places
-  **Testing complexity**: Different code paths for similar
   functionality
-  **Onboarding difficulty**: New developers confused by inconsistent
   patterns

Recommendations
---------------

1. **Eliminate FieldIntrospector** - Replace with SignatureAnalyzer
2. **Standardize ParameterFormManager usage** - Always pass
   parameter_info
3. **Fix nested dataclass forms** - Propagate parameter_info to nested
   forms
4. **Remove duplicate form creation** - Use ParameterFormManager
   everywhere
5. **Create unified interface** - Single entry point for all parameter
   analysis

Summary
-------

Critical Issues Found
~~~~~~~~~~~~~~~~~~~~~

1. **StepParameterEditor inconsistency**: Analyzes parameters but
   doesn’t pass param_info to form manager
2. **Nested dataclass forms**: No parameter analysis, missing help
   functionality
3. **Manual config forms**: Bypass ParameterFormManager, duplicate form
   creation logic
4. **Architectural drift**: Two different analysis systems
   (SignatureAnalyzer vs FieldIntrospector)

Quick Wins (Low Risk, High Impact)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Fix StepParameterEditor to pass param_info (1-line change)
2. Fix nested dataclass forms to analyze and pass param_info
3. Remove manual config form creation, use ParameterFormManager

Major Refactoring (Higher Risk, High Impact)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Eliminate FieldIntrospector duplication
2. Create unified parameter analysis interface
3. Comprehensive testing of unified system

Refactoring Progress
--------------------

✅ Completed Tasks
~~~~~~~~~~~~~~~~~~

1. ✅ **Audit Parameter Analysis Architecture** (COMPLETE)

   -  Documented all inconsistencies and architectural drift
   -  Identified specific files and line numbers requiring changes

2. ✅ **Create Unified Parameter Analysis Interface** (COMPLETE)

   -  Created ``UnifiedParameterAnalyzer`` with consistent interface
   -  Added comprehensive tests and migration guide
   -  Provided backward compatibility aliases

3. ✅ **Refactor SignatureAnalyzer to be Universal** (COMPLETE)

   -  Extended SignatureAnalyzer to handle functions, dataclasses, and
      instances
   -  Added ``_analyze_dataclass_instance`` method
   -  Maintained backward compatibility

4. ✅ **Standardize ParameterFormManager Constructor** (COMPLETE)

   -  Fixed StepParameterEditor to pass parameter_info
   -  Fixed nested dataclass forms to pass parameter_info
   -  Fixed testing methods to pass parameter_info

5. ✅ **Fix Nested Dataclass Parameter Info Propagation** (COMPLETE)

   -  Nested forms now receive parameter_info for help functionality
   -  All nested parameters now have help buttons

6. ✅ **Consolidate Form Creation Patterns** (COMPLETE)

   -  Migrated ConfigFormScreen to use ParameterFormManager
   -  Removed duplicate manual form creation logic
   -  Added proper event handling for ParameterFormManager

7. ✅ **Eliminate FieldIntrospector Duplication** (COMPLETE)

   -  Updated ConfigDialogScreen to use unified system
   -  Updated ConfigWindow to use unified system
   -  Updated ConfigFormWidget to use SignatureAnalyzer
   -  Added ``from_dataclass`` class method for backward compatibility

🔄 Remaining Tasks
~~~~~~~~~~~~~~~~~~

8. Create Parameter Analysis Tests
9. Update Documentation and Examples

🎯 **MAJOR ACHIEVEMENT: UNIFIED PARAMETER ANALYSIS**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All parameter forms in OpenHCS TUI now use the same unified analysis
system: - **Function parameters**: ✅ Help buttons with docstring
descriptions - **Config parameters**: ✅ Help buttons with docstring
descriptions - **Step parameters**: ✅ Help buttons with docstring
descriptions - **Nested parameters**: ✅ Help buttons with docstring
descriptions

Implementation Status
---------------------

✅ Files Successfully Updated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``openhcs/textual_tui/widgets/step_parameter_editor.py`` (line 49) -
   ✅ FIXED: Now passes param_info to ParameterFormManager
-  ``openhcs/textual_tui/widgets/shared/parameter_form_manager.py``
   (nested form creation) - ✅ FIXED: Nested forms now analyze and pass
   param_info
-  ``openhcs/textual_tui/widgets/config_form.py`` (manual form creation)
   - ✅ FIXED: Now uses ParameterFormManager with SignatureAnalyzer
-  ``openhcs/textual_tui/services/config_reflection_service.py`` - ✅
   MAINTAINED: FieldIntrospector kept for backward compatibility
   alongside unified system

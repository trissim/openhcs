Dict Pattern Special Outputs: Architectural Case Study
======================================================

Problem Statement
-----------------

OpenHCS needed to support special outputs (cross-step communication and
materialization) from dict patterns, but the original special I/O system
was designed around single functions per step. This created a
fundamental architectural tension between component-specific processing
and step-to-step communication.

Background Context
------------------

Original Special I/O Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Purpose**: Cross-step communication (positions generation →
   assembly) and analysis materialization
-  **Assumption**: Single function per step with simple key matching
-  **Architecture**: Declarative compilation with runtime execution
   filtering

Dict Pattern Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Use Case**: Component-specific processing
   (``{'DAPI': analyze_nuclei, 'GFP': analyze_proteins}``)
-  **Benefit**: Eliminates need for separate channel isolation steps
-  **Challenge**: Multiple functions per step, each potentially
   producing special outputs

The Architectural Tension
-------------------------

Cross-Step Communication Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Position generation (dict pattern)
   {'DAPI': ashlar_compute_positions}  # Produces: DAPI_positions

   # Assembly step (single pattern)  
   assemble_images  # Expects: positions

   # PROBLEM: DAPI_positions ≠ positions (linking fails)

Execution Filtering Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Step plan (compiled, namespaced)
   step_special_outputs_plan = {'DAPI_cell_counts': {...}}

   # Function attributes (original, not namespaced)
   func_special_outputs = {'cell_counts'}

   # Current filtering logic FAILS
   outputs_plan_for_this_call = {
       key: value for key, value in step_special_outputs_plan.items()
       if key in func_special_outputs  # 'DAPI_cell_counts' not in {'cell_counts'}
   }

Analysis Framework
------------------

Forest-Level Thinking Principles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Architectural Immunity**: No solutions that create technical debt
2. **Compilation Model Integrity**: Compiled plans are single source of
   truth
3. **Fail-Loud Philosophy**: Clear errors over silent failures
4. **Minimal Complexity**: Simplest solution that maintains full
   functionality

Compiler-Inspired Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~

Drawing from compiler design patterns for symbol resolution and scoping:
- **Scope Promotion**: Single-key dict patterns promote to global scope
- **Namespacing**: Multi-key dict patterns maintain component-specific
scope - **Collision Detection**: Compiler validates unique output keys

Solution Architecture
---------------------

1. Full Namespacing System
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pattern**: ``dict_key_chain_position_original_key``

.. code:: python

   {
       'DAPI': [preprocess_func, count_cells_func],  # Chain
       'GFP': analyze_proteins_func                  # Single
   }

   # Produces:
   # DAPI_0_preprocessing_metadata
   # DAPI_1_cell_counts  
   # GFP_0_protein_levels

2. Scope Promotion Rules
~~~~~~~~~~~~~~~~~~~~~~~~

**Single-key dict patterns auto-promote to global scope:**

.. code:: python

   {'DAPI': ashlar_compute_positions}  # DAPI_0_positions → positions

**Multi-key dict patterns maintain namespaced scope:**

.. code:: python

   {'DAPI': analyze, 'GFP': analyze}  # DAPI_0_results, GFP_0_results (no promotion)

3. Execution Filtering Fix
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use execution context to filter step plan:**

.. code:: python

   def get_special_outputs_for_function(step_special_outputs_plan, dict_key, chain_position, func_special_outputs):
       if dict_key is None:
           return step_special_outputs_plan  # Single pattern
       
       prefix = f"{dict_key}_{chain_position}_"
       return {
           key: value for key, value in step_special_outputs_plan.items()
           if key.startswith(prefix) and key[len(prefix):] in func_special_outputs
       }

Implementation Strategy
-----------------------

Phase 1: Execution Filtering Fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Update ``_execute_chain_core()`` to pass dict_key and chain_position
-  Implement function-specific filtering logic
-  Maintain backward compatibility with single patterns

Phase 2: Scope Promotion Rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Add promotion logic to path planner for single-key dict patterns
-  Implement collision detection for promoted outputs
-  Update validation error messages

Phase 3: Compiler Validation Enhancement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Add collision detection for duplicate special output keys
-  Enhance error messages with clear resolution guidance
-  Validate promotion rules during compilation

Validation Requirements
-----------------------

Collision Detection
~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Should fail compilation:
   step1 = FunctionStep(func={'DAPI': ashlar_func})    # Promotes to 'positions'
   step2 = FunctionStep(func={'Calcein': ashlar_func}) # Also promotes to 'positions'
   step3 = FunctionStep(func=assemble_func)            # Expects 'positions' - which one?

Unmatched Input Detection
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Should fail compilation:
   step1 = FunctionStep(func={'DAPI': analyze_func})   # Produces 'DAPI_0_cell_counts'
   step2 = FunctionStep(func=process_func)             # Expects 'positions' - not found

Alternative Approaches Considered
---------------------------------

Separate Decorators
~~~~~~~~~~~~~~~~~~~

**Rejected**: Would require ``@special_outputs`` (cross-step) and
``@materialize_outputs`` (analysis) **Reason**: Increases complexity
without solving the core namespacing issue

Manual Aliasing
~~~~~~~~~~~~~~~

**Rejected**: ``special_output_aliases={'DAPI_positions': 'positions'}``
**Reason**: Adds syntax complexity and potential for user error

Scope Resolution Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rejected**: ``@special_inputs("DAPI::positions")`` **Reason**:
Unfamiliar syntax that defeats the purpose of implicit compilation

Key Insights
------------

Architectural Lessons
~~~~~~~~~~~~~~~~~~~~~

1. **Mixed Concerns**: The original special I/O system mixed cross-step
   communication with analysis materialization
2. **Execution Context**: The step plan alone is insufficient -
   execution needs function-specific context
3. **Namespacing Granularity**: Full namespacing (dict_key +
   chain_position) provides complete precision

Design Principles Validated
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Compiled Plan as Truth**: The step plan should be authoritative,
   but execution needs proper filtering
2. **Fail-Loud Validation**: Compiler should catch all linking and
   collision issues
3. **User Experience**: Simple patterns should “just work” without extra
   syntax

Future Extensibility
--------------------

Manual Override Capability
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Potential future feature
   step = FunctionStep(func={'DAPI': ashlar_func})
   step.special_outputs_mapping = {'positions': 'dapi_specific_positions'}  # Override promotion

Advanced Namespacing
~~~~~~~~~~~~~~~~~~~~

The full namespacing system provides foundation for future complex
patterns while maintaining backward compatibility.

Implementation Results (2025-01-19)
-----------------------------------

**Funcplan System: Beyond the Original Architecture**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The actual implementation exceeded the original architectural plan by
eliminating runtime filtering entirely:

**Original Plan**: Complex runtime filtering with execution context

.. code:: python

   def get_special_outputs_for_function(step_special_outputs_plan, dict_key, chain_position, func_special_outputs):
       # Complex filtering logic...

**Implemented Solution**: Explicit compilation-time mapping (funcplan
system)

.. code:: python

   # Compilation: Generate explicit execution mapping
   funcplan = {"ashlar_compute_tile_positions_gpu_default_0": ["positions"]}

   # Execution: Simple dictionary lookup
   execution_key = f"{func_name}_{dict_key}_{chain_position}"
   outputs_to_save = funcplan.get(execution_key, [])

**Key Architectural Insights Discovered**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Pattern Structure vs Execution Grouping**: The original analysis
   missed the distinction between:

   -  **Pattern type** (dict vs list) → determines function identity
   -  **Execution grouping** (channel iteration) → determines execution
      mechanics

2. **Materialization Function Coupling**: Dict patterns required special
   handling for materialization function extraction from all functions
   in the pattern.

3. **Directory Creation Responsibility**: Materialization functions must
   ensure their target directories exist before saving files.

**Implementation Status**
~~~~~~~~~~~~~~~~~~~~~~~~~

✅ **Funcplan Generation**: Compilation creates explicit execution
mappings ✅ **Funcplan Lookup**: Execution uses simple dictionary lookup
✅ **Pattern Type Handling**: List patterns use “default”, dict patterns
use actual keys ✅ **Materialization Functions**: Proper extraction from
dict patterns ✅ **Directory Creation**: All materialization functions
ensure target directories exist

**Architectural Benefits Achieved**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Zero Runtime Complexity**: All logic moved to compilation phase
2. **Explicit Execution Mapping**: No guessing or filtering required
3. **Forest-Level Thinking**: Preserved clean separation between
   compilation and execution
4. **Fail-Loud Behavior**: Missing funcplan entries cause clear errors

Conclusion
----------

This case study demonstrates how architectural tensions can be resolved
through compiler-inspired design patterns while maintaining system
integrity. The final implementation went beyond the original plan by
eliminating runtime complexity entirely through explicit
compilation-time mapping.

The key insight: **Move all complexity to compilation, keep execution
simple**. The funcplan system provides explicit execution mapping that
eliminates the need for runtime filtering while maintaining full
functionality.

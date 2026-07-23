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

   # Generates namespaced keys:
   # DAPI_0_cell_counts (from count_cells_func at position 0 in DAPI chain)
   # DAPI_1_protein_data (from analyze_proteins_func at position 1 in DAPI chain)
   # GFP_0_analysis_results (from analyze_proteins_func at position 0 in GFP chain)

**Benefits**:
- Unique keys across all dict components
- Preserves original function metadata
- Enables cross-step communication with explicit namespacing

2. Scope Promotion for Single-Key Dicts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rule**: Single-key dict patterns auto-promote to global scope

.. code:: python

   # Single-key dict pattern
   {'positions': ashlar_compute_positions}

   # Auto-promotes to global scope:
   # positions (not positions_0_positions)

   # Consuming step can use simple key:
   @special_inputs("positions")
   def assemble_images(image_stack, positions):
       return stitch(image_stack, positions)

**Benefits**:
- Maintains backward compatibility
- Simplifies common single-component use cases
- Preserves existing cross-step communication patterns

3. Collision Detection System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Validation**: Compiler ensures unique output keys across all patterns

.. code:: python

   # INVALID: Collision detected at compile time
   {
       'DAPI': count_cells_func,     # Produces: cell_counts
       'GFP': count_cells_func       # Produces: cell_counts (COLLISION!)
   }

   # Compiler error: "Duplicate special output key 'cell_counts' detected"

**Benefits**:
- Fail-loud behavior prevents runtime errors
- Clear error messages guide developers
- Maintains compilation model integrity

4. Execution Filtering Enhancement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Reverse-lookup mapping from original keys to namespaced keys

.. code:: python

   # Compilation generates reverse mapping
   original_to_namespaced = {
       'cell_counts': 'DAPI_0_cell_counts',
       'protein_data': 'GFP_0_protein_data'
   }

   # Enhanced filtering logic
   def filter_special_outputs_for_function(step_plan, func_special_outputs):
       filtered_plan = {}
       for original_key in func_special_outputs:
           if original_key in original_to_namespaced:
               namespaced_key = original_to_namespaced[original_key]
               if namespaced_key in step_plan:
                   filtered_plan[namespaced_key] = step_plan[namespaced_key]
       return filtered_plan

**Benefits**:
- Functions use original, clean metadata
- Compilation handles namespacing complexity
- Runtime execution remains efficient

Implementation Strategy
----------------------

Phase 1: Compilation Enhancement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Extend compiler to generate namespaced keys and reverse mappings

**Changes**:
1. Modify special output discovery to generate namespaced keys
2. Create reverse-lookup mapping during compilation
3. Add collision detection validation
4. Implement scope promotion rules for single-key dicts

Phase 2: Runtime Execution Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Update execution filtering to use reverse mappings

**Changes**:
1. Enhance ``filter_special_outputs_for_function`` with reverse lookup
2. Update special output saving to use namespaced keys
3. Maintain backward compatibility for existing pipelines

Phase 3: Cross-Step Communication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Enable cross-step communication with namespaced keys

**Changes**:
1. Update special input resolution to handle namespaced keys
2. Add developer guidance for cross-step communication patterns
3. Provide clear error messages for missing dependencies

Architectural Benefits
---------------------

Maintains Compilation Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Single source of truth: compiled plans contain all execution information
- No runtime discovery or dynamic key generation
- Predictable behavior across all execution contexts

Preserves Function Purity
~~~~~~~~~~~~~~~~~~~~~~~~~

- Functions declare clean, semantic output keys
- No function-level awareness of namespacing complexity
- Testable functions with clear interfaces

Enables Component-Specific Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Dict patterns work seamlessly with special I/O
- Component-specific analysis with cross-step communication
- Eliminates need for separate channel isolation steps

Fail-Loud Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~

- Collision detection at compile time
- Clear error messages for missing dependencies
- No silent failures or unexpected behavior

Future Extensions
----------------

Advanced Namespacing Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hierarchical Namespacing**: Support nested dict patterns with multi-level namespacing

.. code:: python

   {
       'DAPI': {
           'nuclei': count_nuclei_func,
           'background': measure_background_func
       },
       'GFP': analyze_proteins_func
   }

   # Generates: DAPI_nuclei_0_cell_counts, DAPI_background_0_intensity, GFP_0_protein_data

**Conditional Outputs**: Support conditional special outputs based on component properties

Cross-Pattern Communication
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pattern-to-Pattern Dependencies**: Enable communication between different dict pattern components

.. code:: python

   # Step 1: Generate component-specific data
   {'DAPI': generate_nuclei_masks, 'GFP': generate_protein_masks}

   # Step 2: Cross-component analysis
   {'analysis': correlate_nuclei_proteins}  # Uses both DAPI and GFP outputs

This case study demonstrates how compiler design principles can solve complex architectural challenges while maintaining system integrity and developer experience.

Pipeline Compilation Debugging Guide
====================================

Overview
--------

This guide provides systematic approaches to debugging pipeline
compilation issues in OpenHCS. The 5-phase compilation system is
designed to fail fast and provide clear error messages, but
understanding the compilation flow is essential for effective debugging.

Common Error Patterns
---------------------

Phase 1 Errors: Path Planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``"Context step_plans must be initialized before path planning"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Root Cause**: The ``step_plans`` dictionary in ProcessingContext is
not properly initialized.

**Investigation Steps**: 1. Check if ``context.step_plans`` exists and
is a dict 2. Verify ``context.well_id`` is set 3. Verify
``context.input_dir`` is set 4. Check if
``PipelineCompiler.initialize_step_plans_for_context()`` was called

**Common Causes**: - ProcessingContext created without proper
initialization - Missing well_id or input_dir in context - Orchestrator
not calling initialization methods in correct order

``"No patterns for well {well_id} in {input_dir}"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Root Cause**: Microscope handler cannot find image files for the
specified well.

**Investigation Steps**: 1. Check if input directory exists and contains
images 2. Verify well_id format matches filename patterns 3. Check
microscope handler is correctly detecting file patterns 4. Verify file
extensions match ``DEFAULT_IMAGE_EXTENSIONS``

Phase 2 Errors: Zarr Store Declaration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Zarr Configuration Issues
^^^^^^^^^^^^^^^^^^^^^^^^^

**Root Cause**: Problems with zarr store setup for steps requiring zarr
backend.

Phase 3 Errors: Materialization Planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``"step_plans is empty in context for materialization planning"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Root Cause**: Phase 1 (path planning) did not populate step_plans.

**Investigation Steps**: 1. Verify Phase 1 completed successfully 2.
Check if pipeline_definition is empty 3. Verify all steps have valid
UIDs

Backend Selection Issues
^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Steps assigned incorrect read/write backends

**Investigation Steps**: 1. Check step’s step position in pipeline
(first/last have special rules) 2. Check ``force_disk_output`` flag 3.
Verify VFS configuration in global_config

Phase 4 Errors: Memory Contract Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``"Function '{func}' does not have explicit memory type declarations"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Root Cause**: Function missing memory type decorators.

**Investigation Steps**: 1. Check if function has ``@numpy``,
``@torch``, etc. decorators 2. Verify decorator is from correct module
(``openhcs.core.memory.decorators``) 3. Check if function has
``input_memory_type`` and ``output_memory_type`` attributes

``"Functions in step '{step}' have inconsistent memory types"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Root Cause**: Multiple functions in same step have different memory
types.

**Investigation Steps**: 1. Identify all functions in the step’s func
pattern 2. Check memory type declarations for each function 3. Ensure
all functions use same input/output memory types

Phase 5 Errors: GPU Resource Assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU Memory Type Validation Failures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Root Cause**: GPU memory types specified but no GPU available.

**Investigation Steps**: 1. Check if CUDA/GPU libraries are available 2.
Verify GPU device count 3. Check memory type requirements vs available
resources

Debugging Workflow
------------------

1. Identify Compilation Phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Look at the error message to determine which phase failed: - Path
planning: Mentions directories, patterns, or special I/O - Zarr store
declaration: Mentions zarr configuration or store setup -
Materialization: Mentions backends or disk requirements - Memory
validation: Mentions memory types or function contracts - GPU
assignment: Mentions GPU devices or memory types

2. Check Prerequisites
~~~~~~~~~~~~~~~~~~~~~~

For each phase, verify prerequisites are met:

**Phase 1 Prerequisites**: - ProcessingContext properly initialized -
well_id and input_dir set - Microscope handler configured - Input
directory contains valid image files

**Phase 2 Prerequisites**: - Phase 1 completed successfully - step_plans
populated with basic structure - VFS configuration available

**Phase 3 Prerequisites**: - All functions have memory type decorators -
Function patterns are valid - No conflicting memory types within steps

**Phase 4 Prerequisites**: - Memory types validated - GPU libraries
available (if GPU types used)

3. Inspect Intermediate State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add debugging code to inspect state between phases:

.. code:: python

   # After Phase 1
   print("Step plans after path planning:")
   for step_id, plan in context.step_plans.items():
       print(f"  {step_id}: {plan}")

   # After Phase 2  
   print("Backends after materialization planning:")
   for step_id, plan in context.step_plans.items():
       print(f"  {step_id}: read={plan.get('read_backend')}, write={plan.get('write_backend')}")

   # After Phase 3
   print("Memory types after validation:")
   for step_id, plan in context.step_plans.items():
       print(f"  {step_id}: in={plan.get('input_memory_type')}, out={plan.get('output_memory_type')}")

4. Validate Function Decorators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check that all functions have proper decorators:

.. code:: python

   def check_function_decorators(func):
       """Debug helper to check function decorators."""
       print(f"Function: {func.__name__}")
       print(f"  input_memory_type: {getattr(func, 'input_memory_type', 'MISSING')}")
       print(f"  output_memory_type: {getattr(func, 'output_memory_type', 'MISSING')}")
       print(f"  special_inputs: {getattr(func, '__special_inputs__', 'None')}")
       print(f"  special_outputs: {getattr(func, '__special_outputs__', 'None')}")
       print(f"  chain_breaker: {getattr(func, '__chain_breaker__', 'False')}")

5. Test Individual Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test each component in isolation:

.. code:: python

   # Test path planner
   try:
       PipelinePathPlanner.prepare_pipeline_paths(context, steps)
       print("Path planning: SUCCESS")
   except Exception as e:
       print(f"Path planning: FAILED - {e}")

   # Test materialization planner
   try:
       MaterializationFlagPlanner.prepare_pipeline_flags(context, steps)
       print("Materialization planning: SUCCESS")
   except Exception as e:
       print(f"Materialization planning: FAILED - {e}")

Common Solutions
----------------

Missing Memory Type Decorators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Functions not decorated with memory types

**Solution**: Add appropriate decorators:

.. code:: python

   from openhcs.processing.function_registry import torch

   @torch
   def my_function(image_stack):
       return processed_stack

Inconsistent Memory Types
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Functions in same step have different memory types

**Solution**: Ensure all functions use same memory types:

.. code:: python

   # All functions in this step must use torch
   @torch
   def func1(data): return result1

   @torch  
   def func2(data): return result2

Missing Special I/O Declarations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Functions use special inputs/outputs without declaring them

**Solution**: Add special I/O decorators:

.. code:: python

   from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs

   @special_outputs("positions", "metadata")
   def generate_data(image):
       return processed_image, positions, metadata

   @special_inputs("positions")
   def use_data(image, positions):
       return result

Context Initialization Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: ProcessingContext not properly initialized

**Solution**: Ensure orchestrator creates context correctly:

.. code:: python

   context = ProcessingContext(
       global_config=self.global_config,
       well_id=well_id,
       filemanager=self.filemanager
   )
   context.orchestrator = self
   context.microscope_handler = self.microscope_handler
   context.input_dir = self.input_dir

Prevention Strategies
---------------------

1. **Use Type Hints**: Add type hints to function signatures
2. **Validate Early**: Check function decorators at import time
3. **Test Compilation**: Write tests that compile pipelines without
   executing
4. **Document Contracts**: Clearly document special I/O requirements
5. **Use Linting**: Implement AST-based validation for decorator
   requirements

Advanced Debugging
------------------

Enable Debug Logging
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import logging
   logging.getLogger('openhcs.core.pipeline').setLevel(logging.DEBUG)

Inspect Step Plans Schema
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def validate_step_plan(step_id, plan):
       """Validate step plan has required fields."""
       required_fields = [
           'step_name', 'step_type', 'well_id',
           'input_dir', 'output_dir', 
           'read_backend', 'write_backend'
       ]
       
       missing = [field for field in required_fields if field not in plan]
       if missing:
           print(f"Step {step_id} missing fields: {missing}")
       else:
           print(f"Step {step_id}: OK")

Memory Type Compatibility Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a matrix to verify memory type compatibility across pipeline
steps:

.. code:: python

   def check_memory_compatibility(step_plans):
       """Check memory type compatibility between adjacent steps."""
       steps = sorted(step_plans.items(), key=lambda x: x[1].get('pipeline_position', 0))
       
       for i in range(len(steps) - 1):
           current_step = steps[i]
           next_step = steps[i + 1]
           
           current_output = current_step[1].get('output_memory_type')
           next_input = next_step[1].get('input_memory_type')
           
           if current_output != next_input:
               print(f"Memory type mismatch: {current_step[0]} outputs {current_output}, "
                     f"{next_step[0]} expects {next_input}")

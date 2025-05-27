# Special I/O Decorator Consolidation Plan

## Overview

This plan outlines the specific changes required to consolidate the special I/O decorators in the OpenHCS codebase. We will standardize on plural decorator names (`special_outputs` and `special_inputs`) and support multiple named outputs/inputs while maintaining the existing architectural principles.

## Current Architecture

The current architecture for special I/O handling has several issues:

1. **Multiple decorator sets** causing confusion:
   - Function-level: `special_output` and `special_input` in `function_contracts.py`
   - Step-level: `special_out` and `special_in` in `step_contracts.py`

2. **Import errors** in processor files:
   - Files try to import `special_out` from `function_contracts.py` where it doesn't exist

3. **Inconsistent naming** between singular and abbreviated forms:
   - `special_output` vs `special_out`
   - `special_input` vs `special_in`

4. **Limited support** for multiple outputs/inputs in a single decorator call

5. **Inconsistent return values** in functions with special outputs:
   - Some functions return only the special output without the required 3D array as first return value
   - This violates the contract that the first return value must always be a 3D array

6. **Incorrect chain_breaker usage**:
   - Some code passes arguments to the chain_breaker decorator
   - The chain_breaker decorator should take no arguments and simply mark a function as a chain breaker

## Required Changes

### 1. Function Contracts Module

**File**: `openhcs/core/pipeline/function_contracts.py`

**Changes**:
1. Add new `special_outputs(*output_names)` decorator (lines 25-44)
2. Add new `special_inputs(*input_names)` decorator (lines 47-67)
3. Remove old `special_output(key)` and `special_input(key, required)` decorators

**Implementation**:
```python
def special_outputs(*output_names) -> Callable[[F], F]:
    """
    Decorator that marks a function as producing special outputs.

    Args:
        *output_names: Names of the additional return values (excluding the first)
                      that can be consumed by other functions

    Example:
        @special_outputs("positions", "metadata")
        def process_image(image):
            # First return value is always the processed image (3D array)
            # Second return value is positions
            # Third return value is metadata
            return processed_image, positions, metadata
    """
    def decorator(func: F) -> F:
        func.__special_outputs__ = set(output_names)
        return func
    return decorator

def special_inputs(*input_names) -> Callable[[F], F]:
    """
    Decorator that marks a function as requiring special inputs.

    Args:
        *input_names: Names of the additional input parameters (excluding the first)
                     that must be produced by other functions

    Example:
        @special_inputs("positions", "metadata")
        def stitch_images(image_stack, positions, metadata):
            # First parameter is always the input image (3D array)
            # Additional parameters are special inputs from other functions
            return stitched_image
    """
    def decorator(func: F) -> F:
        func.__special_inputs__ = {name: True for name in input_names}
        return func
    return decorator

def chain_breaker(func: F) -> F:
    """
    Decorator that marks a function as a chain breaker.

    Chain breakers are functions that explicitly break the automatic chaining
    of functions in a pipeline. They are used when a function needs to operate
    independently of the normal pipeline flow.

    The path planner will force any step following a chain breaker step
    (a step with a single function in the pattern) to have its input be the
    same as the input of the step at the beginning of the pipeline.

    This decorator takes no arguments - its presence alone is sufficient to
    mark a function as a chain breaker.

    Returns:
        The decorated function with chain breaker attribute set
    """
    func.__chain_breaker__ = True
    return func
```

### 2. Path Planner

**File**: `openhcs/core/pipeline/path_planner.py`

**Changes**:
1. Update extraction of special outputs (line 87):
   ```python
   s_outputs_keys = getattr(core_callable, '__special_outputs__', set())
   ```
   No change needed as it already expects a set.

2. Update extraction of special inputs (line 88):
   ```python
   s_inputs_info = getattr(core_callable, '__special_inputs__', {})
   ```
   No change needed as it already expects a dictionary.

3. Update fallback mechanism for non-FunctionSteps (lines 91-106):
   - No changes needed as it already handles different types of raw_s_outputs and raw_s_inputs.

4. Update special output path generation (lines 178-193):
   - No changes needed as it already iterates over s_outputs_keys.

5. Update special input validation and linking (lines 196-211):
   - No changes needed as it already iterates over s_inputs_info.keys().

### 3. Function Step Contract Validator

**File**: `openhcs/core/pipeline/funcstep_contract_validator.py`

**Changes**:
1. Update how it checks for special I/O attributes (around line 101):
   ```python
   if hasattr(f_callable, '__special_inputs__') or \
      hasattr(f_callable, '__special_outputs__') or \
      hasattr(f_callable, '__chain_breaker__'):
       # Ensure the step has a single callable pattern
       # ...
   ```
   No change needed as it already checks for the correct attributes.

### 4. Affected Processor Files

#### 4.1 Ashlar Processor CuPy

**File**: `openhcs/processing/backends/pos_gen/ashlar_processor_cupy.py`

**Changes**:
1. Update import (line 20):
   ```python
   # From
   from openhcs.core.pipeline.function_contracts import special_out, chain_breaker

   # To
   from openhcs.core.pipeline.function_contracts import special_outputs, chain_breaker
   ```

2. Update decorator usage and function signature (line 151):
   ```python
   # From
   @chain_breaker(SpecialKey.POSITION_ARRAY)
   @special_out(SpecialKey.POSITION_ARRAY)
   def gpu_ashlar_align_cupy(tiles, num_rows, num_cols, **kwargs):
       # ...
       return offsets  # or affine_mats

   # To
   @chain_breaker
   @special_outputs("positions")
   def gpu_ashlar_align_cupy(tiles, num_rows, num_cols, **kwargs):
       # ...
       return processed_tiles, positions  # Return processed tiles and positions
   ```

3. Update function implementation to return both processed tiles and positions:
   - Modify the function to return both the processed tiles and the positions
   - **CRITICAL**: Ensure the first return value is ALWAYS the processed 3D array (tiles)
   - If the function currently only returns positions, modify it to return the input tiles as the first return value
   - Rename the second return value from "offsets" to "positions" for clarity
   - Example implementation:
     ```python
     def gpu_ashlar_align_cupy(tiles, num_rows, num_cols, **kwargs):
         # Original implementation
         offsets = compute_offsets(tiles, num_rows, num_cols, **kwargs)

         # Modified to return both tiles and positions
         return tiles, offsets  # First return value is the input tiles, second is the positions
     ```

#### 4.2 MIST Processor CuPy

**File**: `openhcs/processing/backends/pos_gen/mist_processor_cupy.py`

**Changes**:
1. Update import (line 21):
   ```python
   # From
   from openhcs.core.pipeline.function_contracts import special_out

   # To
   from openhcs.core.pipeline.function_contracts import special_outputs
   ```

2. Update decorator usage and function signature (line 205):
   ```python
   # From
   @special_out(SpecialKey.POSITION_ARRAY)
   def mist_compute_tile_positions(image_stack, num_rows, num_cols, **kwargs):
       # ...
       return positions

   # To
   @special_outputs("positions")
   def mist_compute_tile_positions(image_stack, num_rows, num_cols, **kwargs):
       # ...
       return processed_stack, positions  # Return processed stack and positions
   ```

3. Update function implementation to return both processed stack and positions:
   - Modify the function to return both the processed image stack and the positions
   - **CRITICAL**: Ensure the first return value is ALWAYS the processed 3D array (image_stack)
   - If the function currently only returns positions, modify it to return the input image_stack as the first return value
   - Keep the second return value as "positions" for clarity
   - Example implementation:
     ```python
     def mist_compute_tile_positions(image_stack, num_rows, num_cols, **kwargs):
         # Original implementation
         positions = compute_positions(image_stack, num_rows, num_cols, **kwargs)

         # Modified to return both image_stack and positions
         return image_stack, positions  # First return value is the input image_stack, second is the positions
     ```

#### 4.3 Assembler Files (Consumers of Position Data)

**Files**:
- `openhcs/processing/backends/assemblers/assemble_stack_cupy.py`
- `openhcs/processing/backends/assemblers/assemble_stack_cpu.py`

**Changes for assemble_stack_cupy.py**:
1. Update import:
   ```python
   # From
   from openhcs.core.pipeline.function_contracts import special_input

   # To
   from openhcs.core.pipeline.function_contracts import special_inputs
   ```

2. Update decorator usage and function signature:
   ```python
   # From
   @special_input(SpecialKey.POSITION_ARRAY)
   def assemble_stack_cupy(image_stack, position_array, **kwargs):
       # ...

   # To
   @special_inputs("positions")
   def assemble_stack_cupy(image_stack, positions, **kwargs):
       # ...
   ```

3. Update function implementation:
   - Rename parameter from "position_array" to "positions" for clarity
   - Update any references to this parameter within the function

**Changes for assemble_stack_cpu.py**:
1. Update import:
   ```python
   # From
   from openhcs.core.pipeline.function_contracts import special_input

   # To
   from openhcs.core.pipeline.function_contracts import special_inputs
   ```

2. Update decorator usage and function signature:
   ```python
   # From
   @special_input(SpecialKey.POSITION_ARRAY)
   def assemble_stack_cpu(image_stack, position_array, **kwargs):
       # ...

   # To
   @special_inputs("positions")
   def assemble_stack_cpu(image_stack, positions, **kwargs):
       # ...
   ```

3. Update function implementation:
   - Rename parameter from "position_array" to "positions" for clarity
   - Update any references to this parameter within the function

#### 4.4 Other Files

**Search Command**:
```bash
grep -r "from openhcs.core.pipeline.function_contracts import special_" --include="*.py" openhcs/
```

Update all found files similarly, ensuring:
1. Consistent naming between producers and consumers
2. First parameter and return value are always 3D arrays
3. Parameter names in `special_inputs` match output names in `special_outputs`

### 5. Step Contracts Module

**File**: `openhcs/core/steps/step_contracts.py`

**Changes**:
1. Mark `special_out` and `special_in` as deprecated (lines 42-84):
   ```python
   def special_out(key: str) -> Callable[[S], S]:
       """
       DEPRECATED: Use special_outputs from function_contracts instead.
       """
       import warnings
       warnings.warn(
           "special_out is deprecated. Use special_outputs from function_contracts instead.",
           DeprecationWarning, stacklevel=2
       )
       # Rest of the function unchanged
   ```

   ```python
   def special_in(key: str, required: bool = True) -> Callable[[S], S]:
       """
       DEPRECATED: Use special_inputs from function_contracts instead.
       """
       import warnings
       warnings.warn(
           "special_in is deprecated. Use special_inputs from function_contracts instead.",
           DeprecationWarning, stacklevel=2
       )
       # Rest of the function unchanged
   ```

2. Search for direct usage:
   ```bash
   grep -r "@special_in\|@special_out" --include="*.py" openhcs/
   ```
   Update all found usages to use the new decorators.

### 6. Pipeline Compiler

**File**: `openhcs/core/pipeline/compiler.py`

**Changes**:
1. No changes needed for initialization of special I/O dictionaries (lines 130-131):
   ```python
   current_plan.setdefault("special_inputs", OrderedDict())
   current_plan.setdefault("special_outputs", OrderedDict())
   ```
   These lines already use the correct plural names.

## Implementation Sequence

### Phase 1: Add New Decorators (File 1)
1. Edit `openhcs/core/pipeline/function_contracts.py`
2. Add new `special_outputs(*output_names)` decorator
3. Add new `special_inputs(*input_names)` decorator
4. Update `chain_breaker` decorator to not take any arguments
5. Keep old decorators temporarily for backward compatibility

### Phase 2: Update Processor Files (Files 4.1, 4.2, 4.3)
1. Edit `openhcs/processing/backends/pos_gen/ashlar_processor_cupy.py`
2. Edit `openhcs/processing/backends/pos_gen/mist_processor_cupy.py`
3. Find and edit all other files using the old decorators

### Phase 3: Deprecate Step Decorators (File 5)
1. Edit `openhcs/core/steps/step_contracts.py`
2. Mark `special_out` and `special_in` as deprecated
3. Find and update any direct usage of these decorators

### Phase 4: Remove Old Decorators (File 1 again)
1. Edit `openhcs/core/pipeline/function_contracts.py`
2. Remove old `special_output(key)` and `special_input(key, required)` decorators

### Phase 5: Testing
1. Run unit tests for the pipeline compiler
2. Run integration tests for the full pipeline
3. Verify special I/O detection and linking works correctly

## Contract Assumptions

1. **First parameter and return value**: The first parameter and first return value are ALWAYS 3D arrays/tensors (by contract)
2. **Multiple return values**: Functions with `special_outputs` MUST return at least two values:
   - First return value: The processed 3D array (or the input array if no processing is done)
   - Second and subsequent return values: The special outputs specified in the decorator
3. **Decorator scope**: The decorators only specify the additional inputs/outputs beyond the standard first parameter/return value
4. **Parameter name matching**: The parameter names in `special_inputs` must exactly match the output names in `special_outputs` for proper linking
5. **Chain breaker behavior**: The `chain_breaker` decorator takes no arguments and forces the next step to use the pipeline's initial input

## Verification Steps

After each phase, verify:

1. **Imports work**: No import errors in processor files
   ```bash
   # Run Python with the -c flag to check imports
   python -c "from openhcs.processing.backends.pos_gen.ashlar_processor_cupy import gpu_ashlar_align_cupy"
   python -c "from openhcs.processing.backends.pos_gen.mist_processor_cupy import mist_compute_tile_positions"
   ```

2. **Decorators work**: Functions with decorators have correct attributes
   ```python
   # Check in Python interpreter
   from openhcs.processing.backends.pos_gen.ashlar_processor_cupy import gpu_ashlar_align_cupy
   print(hasattr(gpu_ashlar_align_cupy, '__special_outputs__'))  # Should print True
   print(gpu_ashlar_align_cupy.__special_outputs__)  # Should print {'positions'}
   ```

3. **Path planning works**: Special I/O paths are correctly generated
   ```bash
   # Run a test pipeline that uses the special I/O functions
   python -m openhcs.tests.test_pipeline_path_planner
   ```

4. **Pipeline executes**: Full pipeline runs without errors
   ```bash
   # Run a test pipeline that includes both producer and consumer functions
   python -m openhcs.tests.test_pipeline_execution
   ```

5. **Return values match**: Ensure the functions return the expected values
   ```python
   # Check in Python interpreter
   from openhcs.processing.backends.pos_gen.ashlar_processor_cupy import gpu_ashlar_align_cupy
   import numpy as np
   # Create test data
   test_tiles = np.random.rand(4, 100, 100)  # 4 tiles of 100x100
   # Call function
   processed_tiles, positions = gpu_ashlar_align_cupy(test_tiles, 2, 2)
   # Check return types
   print(type(processed_tiles), processed_tiles.shape)  # Should be ndarray with same shape as input
   print(type(positions), positions.shape)  # Should be ndarray with shape (4, 2)
   ```

## Expected Outcome

1. **Standardized naming convention** (plural form)
2. **No import errors** in processor files
3. **Support for multiple named outputs/inputs** in a single decorator call
4. **More explicit contract** between producers and consumers
5. **Simplified codebase** with a single set of decorators
6. **Consistent return values** with all special output functions returning at least two values:
   - First return value always being a 3D array
   - Second and subsequent return values being the special outputs
7. **Correct chain_breaker usage** with no arguments passed to the decorator

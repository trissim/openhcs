# Code Changes Made - OpenHCS Debug Session

## Summary of Fixes Applied

### 1. Backend vs MemoryType Confusion (FIXED)
**Problem**: Mixing `MemoryType.MEMORY.value` and `Backend.MEMORY.value`
**Solution**: Use correct constants for each context

```python
# BEFORE (incorrect)
context.filemanager.save(data, path, MemoryType.MEMORY.value)

# AFTER (correct)  
context.filemanager.save(data, path, Backend.MEMORY.value)
```

**Rule**: 
- FileManager operations: `Backend.MEMORY.value`
- Stack utils: `MemoryType` for array conversion
- Compiler: `MemoryType` for memory contracts

### 2. Path Planner Nested Outputs Bug (FIXED)
**Problem**: `workspace_outputs_outputs_outputs_outputs` directories
**Root Cause**: Each step was adding `_outputs` suffix to previous step's output

**Files Modified**: `openhcs/core/pipeline/path_planner.py`

```python
# BEFORE: Always add suffix
step_output_dir = step_input_dir.with_name(f"{step_input_dir.name}{current_suffix}")

# AFTER: Only first step gets suffix, subsequent steps work in place
if i == 0:
    # First step gets output suffix
    step_output_dir = step_input_dir.with_name(f"{step_input_dir.name}{current_suffix}")
else:
    # Subsequent steps work in place
    step_output_dir = step_input_dir
```

### 3. Memory Collision Prevention (FIXED)
**Problem**: Multiple wells saving special outputs with same filename
**Root Cause**: Parallel threads (A01, D02) both saving `positions.pkl`

**Files Modified**: `openhcs/core/steps/function_step.py`

```python
# Added well_id parameter to functions
def _execute_function_core(..., well_id: str) -> Any:

# Added well_id prefix to special output filenames
prefixed_filename = f"{well_id}_{vfs_path_obj.name}"
# Result: A01_positions.pkl, D02_positions.pkl
```

### 4. Image-Filename Mismatch Handling (FIXED)
**Problem**: Flattening operations return fewer images than inputs
**Solution**: Handle mismatched counts gracefully

```python
# Added before image saving loop
num_outputs = len(output_slices)
num_inputs = len(matching_files)

if num_outputs < num_inputs:
    logger.debug(f"Function returned {num_outputs} images from {num_inputs} inputs - likely flattening operation")
elif num_outputs > num_inputs:
    logger.warning(f"Function returned more images ({num_outputs}) than inputs ({num_inputs}) - unexpected")
```

## Detailed Code Changes

### openhcs/core/steps/function_step.py

#### Function Signatures Updated
```python
# Line 30-38: Added well_id parameter
def _execute_function_core(
    func_callable: Callable,
    main_data_arg: Any,
    base_kwargs: Dict[str, Any],
    context: 'ProcessingContext',
    special_inputs_plan: Dict[str, str],
    special_outputs_plan: TypingOrderedDict[str, str],
    well_id: str  # ADDED
) -> Any:

# Line 114-121: Added well_id parameter  
def _execute_chain_core(
    initial_data_stack: Any,
    func_chain: List[Union[Callable, Tuple[Callable, Dict]]],
    context: 'ProcessingContext',
    step_special_inputs_plan: Dict[str, str],
    step_special_outputs_plan: TypingOrderedDict[str, str],
    well_id: str  # ADDED
) -> Any:
```

#### Special Output Saving (Line 95-105)
```python
# Add well_id prefix to filename for memory backend to avoid thread collisions
from pathlib import Path
vfs_path_obj = Path(vfs_path)
prefixed_filename = f"{well_id}_{vfs_path_obj.name}"
prefixed_vfs_path = str(vfs_path_obj.parent / prefixed_filename)

logger.debug(f"Saving special output '{output_key}' to VFS path '{prefixed_vfs_path}' (memory backend)")
# Ensure directory exists for memory backend
parent_dir = str(Path(prefixed_vfs_path).parent)
context.filemanager.ensure_directory(parent_dir, Backend.MEMORY.value)
context.filemanager.save(value_to_save, prefixed_vfs_path, Backend.MEMORY.value)
```

#### Function Call Updates
```python
# Line 137-145: Pass well_id to _execute_function_core
current_stack = _execute_function_core(
    func_callable=actual_callable,
    main_data_arg=current_stack,
    base_kwargs=base_kwargs_for_item,
    context=context,
    special_inputs_plan=step_special_inputs_plan,
    special_outputs_plan=outputs_plan_for_this_call,
    well_id=well_id  # ADDED
)

# Line 204-213: Pass well_id in _process_single_pattern_group
if isinstance(executable_func_or_chain, list):
    processed_stack = _execute_chain_core(
        main_data_stack, executable_func_or_chain, context,
        special_inputs_map, special_outputs_map, well_id  # ADDED
    )
elif callable(executable_func_or_chain):
    processed_stack = _execute_function_core(
        executable_func_or_chain, main_data_stack, final_base_kwargs, context,
        special_inputs_map, special_outputs_map, well_id  # ADDED
    )
```

### openhcs/core/pipeline/path_planner.py

#### Output Directory Logic (Line 154-194)
```python
# For first step (i == 0), create output directory with suffix
# For subsequent steps (i > 0), work in place (use same directory as input)
if i == 0:
    # Use same directory as input with appropriate suffix based on step name
    step_name_lower = step_name.lower()
    current_suffix = path_config.output_dir_suffix # Default
    if "position" in step_name_lower:
        current_suffix = path_config.positions_dir_suffix
    elif "stitch" in step_name_lower:
        current_suffix = path_config.stitched_dir_suffix

    # For first step, use workspace directory name instead of input directory name
    if hasattr(context, 'workspace_path') and context.workspace_path:
        workspace_path = Path(context.workspace_path)
        step_output_dir = workspace_path.with_name(f"{workspace_path.name}{current_suffix}")
    else:
        step_output_dir = step_input_dir.with_name(f"{step_input_dir.name}{current_suffix}")
else:
    # Subsequent steps work in place - use same directory as input
    step_output_dir = step_input_dir
```

## Status of Each Fix

✅ **Backend/MemoryType**: WORKING - No more type errors
✅ **Path planner**: WORKING - No more nested directories  
✅ **Memory collision**: WORKING - Well-prefixed filenames
✅ **Image mismatch**: IMPLEMENTED - Graceful handling added

## Remaining Issue

❌ **Channel grouping**: `create_composite` with `variable_components=['channel']` not working as expected
- Both w1 and w2 reaching Ashlar instead of being composited
- Need to investigate pattern discovery and grouping logic
- Reference EZStitcher docs for intended behavior

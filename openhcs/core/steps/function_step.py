"""
FunctionStep implementation for pattern-based processing.

This module contains the FunctionStep class. During execution, FunctionStep instances
are stateless regarding their configuration. All operational parameters, including
the function(s) to execute, special input/output keys, their VFS paths, and memory types,
are retrieved from this step's entry in `context.step_plans`.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, OrderedDict as TypingOrderedDict

from openhcs.constants.constants import (DEFAULT_IMAGE_EXTENSION,
                                             DEFAULT_IMAGE_EXTENSIONS,
                                             DEFAULT_SITE_PADDING, Backend,
                                             MemoryType, VariableComponents, GroupBy)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.abstract import AbstractStep, get_step_id
from openhcs.formats.func_arg_prep import prepare_patterns_and_functions
from openhcs.core.memory.stack_utils import stack_slices, unstack_slices
from openhcs.core.memory.gpu_cleanup import cleanup_memory_by_type

logger = logging.getLogger(__name__)

# Environment variable to disable universal GPU defragmentation
DISABLE_GPU_DEFRAG = os.getenv('OPENHCS_DISABLE_GPU_DEFRAG', 'false').lower() == 'true'

def _targeted_gpu_cleanup_after_function(func_name: str, input_memory_type: str, device_id: Optional[int] = None, failed: bool = False) -> None:
    """Targeted GPU memory cleanup for specific frameworks used by the function."""
    if DISABLE_GPU_DEFRAG:
        return

    try:
        from openhcs.core.memory.gpu_cleanup import cleanup_gpu_memory_by_framework
        
        # Clean up the input memory type framework (what the function received)
        cleanup_gpu_memory_by_framework(input_memory_type, device_id)
        
        status = "FAILED" if failed else "SUCCESS"
        logger.debug(f"🔧 TARGETED CLEANUP ({status}): {func_name} completed, cleaned frameworks: {input_memory_type}")

    except ImportError:
        pass  # GPU frameworks not available
    except Exception as e:
        logger.warning(f"🔧 TARGETED CLEANUP: Failed for {func_name}: {e}")

def _is_3d(array: Any) -> bool:
    """Check if an array is 3D."""
    return hasattr(array, 'ndim') and array.ndim == 3

def _execute_function_core(
    func_callable: Callable,
    main_data_arg: Any,
    base_kwargs: Dict[str, Any],
    context: 'ProcessingContext',
    special_inputs_plan: Dict[str, str],  # {'arg_name_for_func': 'special_path_value'}
    special_outputs_plan: TypingOrderedDict[str, str], # {'output_key': 'special_path_value'}, order matters
    well_id: str, # Add well_id parameter
    input_memory_type: str,
    device_id: int
) -> Any: # Returns the main processed data stack
    """
    Executes a single callable, handling its special I/O.
    - Loads special inputs from VFS paths in `special_inputs_plan`.
    - Calls `func_callable(main_data_arg, **all_kwargs)`.
    - If `special_outputs_plan` is non-empty, expects func to return (main_out, sp_val1, sp_val2,...).
    - Saves special outputs positionally to VFS paths in `special_outputs_plan`.
    - Returns the main processed data stack.
    """
    final_kwargs = base_kwargs.copy()

    if special_inputs_plan:
        for arg_name, path_info in special_inputs_plan.items():
            # Extract path string from the path info dictionary
            # Current format: {"path": "/path/to/file.pkl", "source_step_id": "step_123"}
            if isinstance(path_info, dict) and 'path' in path_info:
                special_path_value = path_info['path']
            else:
                special_path_value = path_info  # Fallback if it's already a string

            # Add well_id prefix to filename for memory backend to match special output logic
            from pathlib import Path
            special_path_obj = Path(special_path_value)
            prefixed_filename = f"{well_id}_{special_path_obj.name}"
            prefixed_special_path = str(special_path_obj.parent / prefixed_filename)

            logger.debug(f"Loading special input '{arg_name}' from path '{prefixed_special_path}' (memory backend)")
            try:
                final_kwargs[arg_name] = context.filemanager.load(prefixed_special_path, Backend.MEMORY.value)
            except Exception as e:
                logger.error(f"Failed to load special input '{arg_name}' from '{prefixed_special_path}': {e}", exc_info=True)
                raise

    # Auto-inject context if function signature expects it
    import inspect
    sig = inspect.signature(func_callable)
    if 'context' in sig.parameters:
        final_kwargs['context'] = context

    # 🔍 DEBUG: Log input dimensions
    input_shape = getattr(main_data_arg, 'shape', 'no shape attr')
    input_type = type(main_data_arg).__name__
    logger.info(f"🔍 FUNCTION INPUT: {func_callable.__name__} - shape: {input_shape}, type: {input_type}")



    try:
        raw_function_output = func_callable(main_data_arg, **final_kwargs)

        # 🔍 DEBUG: Log output dimensions
        output_shape = getattr(raw_function_output, 'shape', 'no shape attr')
        output_type = type(raw_function_output).__name__
        logger.info(f"🔍 FUNCTION OUTPUT: {func_callable.__name__} - shape: {output_shape}, type: {output_type}")

        # 🔧 TARGETED GPU CLEANUP: Clean memory for specific framework after function
        step_id = next(iter(context.step_plans.keys()))
        _targeted_gpu_cleanup_after_function(func_callable.__name__, input_memory_type, device_id)

    except Exception as e:
        # 🔧 TARGETED GPU CLEANUP: Clean memory even on failure
        step_id = next(iter(context.step_plans.keys()))
        _targeted_gpu_cleanup_after_function(func_callable.__name__, input_memory_type, device_id, failed=True)
        raise e

    main_output_data = raw_function_output
    
    if special_outputs_plan: 
        num_special_outputs = len(special_outputs_plan)
        if not isinstance(raw_function_output, tuple) or len(raw_function_output) != (1 + num_special_outputs):
            raise ValueError(
                f"Function '{getattr(func_callable, '__name__', 'unknown')}' was expected to return a tuple of "
                f"{1 + num_special_outputs} values (main_output + {num_special_outputs} special) "
                f"based on 'special_outputs' in step plan, but returned {len(raw_function_output) if isinstance(raw_function_output, tuple) else type(raw_function_output)} values."
            )
        main_output_data = raw_function_output[0]
        returned_special_values_tuple = raw_function_output[1:]

        # Iterate through special_outputs_plan (which must be ordered by compiler)
        # and match with positionally returned special values.
        for i, (output_key, vfs_path_info) in enumerate(special_outputs_plan.items()):
            if i < len(returned_special_values_tuple):
                value_to_save = returned_special_values_tuple[i]
                # Extract path string from the path info dictionary
                # Current format: {"path": "/path/to/file.pkl"}
                if isinstance(vfs_path_info, dict) and 'path' in vfs_path_info:
                    vfs_path = vfs_path_info['path']
                else:
                    vfs_path = vfs_path_info  # Fallback if it's already a string
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
            else:
                # This indicates a mismatch that should ideally be caught by schema/validation
                logger.error(f"Mismatch: {num_special_outputs} special outputs planned, but fewer values returned by function for key '{output_key}'.")
                # Or, if partial returns are allowed, this might be a warning. For now, error.
                raise ValueError(f"Function did not return enough values for all planned special outputs. Missing value for '{output_key}'.")
    
    return main_output_data

def _execute_chain_core(
    initial_data_stack: Any,
    func_chain: List[Union[Callable, Tuple[Callable, Dict]]],
    context: 'ProcessingContext',
    step_special_inputs_plan: Dict[str, str],
    step_special_outputs_plan: TypingOrderedDict[str, str],
    well_id: str,  # Add well_id parameter
    device_id: int,
    input_memory_type: str,
) -> Any:
    current_stack = initial_data_stack
    for i, func_item in enumerate(func_chain):
        actual_callable: Callable
        base_kwargs_for_item: Dict[str, Any] = {}
        is_last_in_chain = (i == len(func_chain) - 1)

        if isinstance(func_item, tuple) and len(func_item) == 2 and callable(func_item[0]):
            actual_callable, base_kwargs_for_item = func_item
        elif callable(func_item):
            actual_callable = func_item
        else:
            raise TypeError(f"Invalid item in function chain: {func_item}.")
        
        outputs_plan_for_this_call = step_special_outputs_plan if is_last_in_chain else {}
        
        current_stack = _execute_function_core(
            func_callable=actual_callable,
            main_data_arg=current_stack,
            base_kwargs=base_kwargs_for_item,
            context=context,
            special_inputs_plan=step_special_inputs_plan,
            special_outputs_plan=outputs_plan_for_this_call,
            well_id=well_id,
            device_id=device_id,
            input_memory_type=input_memory_type,
        )
    return current_stack

def _process_single_pattern_group(
    context: 'ProcessingContext',
    pattern_group_info: Any, 
    executable_func_or_chain: Any, 
    base_func_args: Dict[str, Any], 
    step_input_dir: Path,
    step_output_dir: Path,
    well_id: str,
    component_value: str, 
    read_backend: str,
    write_backend: str,
    input_memory_type_from_plan: str, # Explicitly from plan
    output_memory_type_from_plan: str, # Explicitly from plan
    device_id: Optional[int],
    same_directory: bool,
    force_disk_output_flag: bool,
    special_inputs_map: Dict[str, str],
    special_outputs_map: TypingOrderedDict[str, str],
    variable_components: Optional[List[str]] = None
) -> None:
    start_time = time.time()
    pattern_repr = str(pattern_group_info)[:100]
    print(f"🔥 PATTERN: Processing {pattern_repr} for well {well_id}")

    try:
        if not context.microscope_handler:
             raise RuntimeError("MicroscopeHandler not available in context.")

        matching_files = context.microscope_handler.path_list_from_pattern(
            str(step_input_dir), pattern_group_info, context.filemanager, read_backend,
            [vc.value for vc in variable_components] if variable_components else None
        )

        if not matching_files:
            logger.warning(f"No matching files for pattern group {pattern_repr} in {step_input_dir}")
            return

        print(f"🔥 PATTERN: Found {len(matching_files)} files: {[Path(f).name for f in matching_files]}")

        # Sort files to ensure consistent ordering (especially important for z-stacks)
        matching_files.sort()
        print(f"🔥 PATTERN: Sorted files: {[Path(f).name for f in matching_files]}")

        raw_slices = []
        for file_path_suffix in matching_files:
            full_file_path = step_input_dir / file_path_suffix
            try:
                image = context.filemanager.load(str(full_file_path), read_backend)
                if image is not None:
                    raw_slices.append(image)
            except Exception as e:
                logger.error(f"Error loading image {full_file_path}: {e}", exc_info=True)
        
        if not raw_slices:
            logger.warning(f"No valid images loaded for pattern group {pattern_repr} in {step_input_dir}")
            return

        # 🔍 DEBUG: Log stacking operation
        logger.info(f"🔍 STACKING: {len(raw_slices)} slices → memory_type: {input_memory_type_from_plan}")
        if raw_slices:
            slice_shapes = [getattr(s, 'shape', 'no shape') for s in raw_slices[:3]]  # First 3 shapes
            logger.info(f"🔍 STACKING: Sample slice shapes: {slice_shapes}")



        main_data_stack = stack_slices(
            slices=raw_slices, memory_type=input_memory_type_from_plan, gpu_id=device_id
        )

        # � CPU CLEANUP: Clear source arrays after GPU conversion to prevent memory doubling
        from openhcs.constants.constants import MemoryType, Backend
        if (input_memory_type_from_plan != MemoryType.NUMPY.value and
            read_backend == Backend.DISK.value):
            del raw_slices
            import gc
            gc.collect()
            logger.debug(f"🔥 CPU CLEANUP: Cleared source arrays after conversion from disk to {input_memory_type_from_plan}")

        # �🔍 DEBUG: Log stacked result
        stack_shape = getattr(main_data_stack, 'shape', 'no shape')
        stack_type = type(main_data_stack).__name__
        logger.info(f"🔍 STACKED RESULT: shape: {stack_shape}, type: {stack_type}")
        
        final_base_kwargs = base_func_args.copy()
        
        if isinstance(executable_func_or_chain, list):
            processed_stack = _execute_chain_core(
                main_data_stack, executable_func_or_chain, context,
                special_inputs_map, special_outputs_map, well_id,
                device_id, input_memory_type_from_plan
            )
        elif callable(executable_func_or_chain):
            processed_stack = _execute_function_core(
                executable_func_or_chain, main_data_stack, final_base_kwargs, context,
                special_inputs_map, special_outputs_map, well_id, input_memory_type_from_plan, device_id
            )
        else:
            raise TypeError(f"Invalid executable_func_or_chain: {type(executable_func_or_chain)}")

        # 🔍 DEBUG: Check what shape the function actually returned
        input_shape = getattr(main_data_stack, 'shape', 'unknown')
        output_shape = getattr(processed_stack, 'shape', 'unknown')
        processed_type = type(processed_stack).__name__
        logger.info(f"🔍 PROCESSING RESULT: input: {input_shape} → output: {output_shape}, type: {processed_type}")

        if not _is_3d(processed_stack):
             raise ValueError(f"Main processing must result in a 3D array, got {getattr(processed_stack, 'shape', 'unknown')}")

        # 🔍 DEBUG: Log unstacking operation
        logger.info(f"🔍 UNSTACKING: shape: {output_shape} → memory_type: {output_memory_type_from_plan}")



        output_slices = unstack_slices(
            array=processed_stack, memory_type=output_memory_type_from_plan, gpu_id=device_id, validate_slices=True
        )

        # 🔍 DEBUG: Log unstacked result
        if output_slices:
            unstacked_shapes = [getattr(s, 'shape', 'no shape') for s in output_slices[:3]]  # First 3 shapes
            logger.info(f"🔍 UNSTACKED RESULT: {len(output_slices)} slices, sample shapes: {unstacked_shapes}")

        # Handle cases where function returns fewer images than inputs (e.g., z-stack flattening, channel compositing)
        # In such cases, we save only the returned images using the first N input filenames
        num_outputs = len(output_slices)
        num_inputs = len(matching_files)

        if num_outputs < num_inputs:
            logger.debug(f"Function returned {num_outputs} images from {num_inputs} inputs - likely flattening operation")
        elif num_outputs > num_inputs:
            logger.warning(f"Function returned more images ({num_outputs}) than inputs ({num_inputs}) - unexpected")

        # Save the output images using the first N input filenames
        for i, img_slice in enumerate(output_slices):
            try:
                # FAIL FAST: No fallback filenames - if we have more outputs than inputs, something is wrong
                if i >= len(matching_files):
                    raise ValueError(
                        f"Function returned {num_outputs} output slices but only {num_inputs} input files available. "
                        f"Cannot generate filename for output slice {i}. This indicates a bug in the function or "
                        f"unstacking logic - functions should return same or fewer images than inputs."
                    )

                input_filename = matching_files[i]
                output_filename = Path(input_filename).name

                output_path = Path(step_output_dir) / output_filename

                # Always ensure we can write to the output path (delete if exists)
                if context.filemanager.exists(str(output_path), write_backend):
                    context.filemanager.delete(str(output_path), write_backend)

                # Ensure directory exists for the backend we're about to save to
                context.filemanager.ensure_directory(str(step_output_dir), write_backend)
                context.filemanager.save(img_slice, str(output_path), write_backend)

                if force_disk_output_flag and write_backend != Backend.DISK.value:
                    logger.info(f"Force disk output: saving additional copy to disk: {output_path}")
                    # Ensure directory exists for disk backend too
                    context.filemanager.ensure_directory(str(step_output_dir), Backend.DISK.value)
                    context.filemanager.save(img_slice, str(output_path), Backend.DISK.value)
            except Exception as e:
                logger.error(f"Error saving output slice {i} for pattern {pattern_repr}: {e}", exc_info=True)

        # 🔥 CLEANUP: If function returned fewer images than inputs, delete the unused input files
        # This prevents unused channel files from remaining in memory after compositing
        if num_outputs < num_inputs:
            for j in range(num_outputs, num_inputs):
                unused_input_filename = matching_files[j]
                unused_input_path = Path(step_input_dir) / unused_input_filename
                if context.filemanager.exists(str(unused_input_path), write_backend):
                    context.filemanager.delete(str(unused_input_path), write_backend)
                    print(f"🔥 CLEANUP: Deleted unused input file: {unused_input_filename}")

        # 🔍 VRAM TRACKING: Log memory before cleanup
        try:
            from openhcs.core.memory.gpu_cleanup import log_gpu_memory_usage
            log_gpu_memory_usage(f"pattern group {pattern_repr} (before cleanup)")
        except Exception:
            pass

        # 🔥 GPU CLEANUP: Clear GPU memory after processing pattern group
        # Clean both input and output memory types to ensure comprehensive cleanup
        cleanup_memory_by_type(input_memory_type_from_plan, device_id)
        cleanup_memory_by_type(output_memory_type_from_plan, device_id)

        # Force garbage collection to release any remaining references
        import gc
        gc.collect()

        logger.debug(f"🔥 GPU CLEANUP: Cleared {input_memory_type_from_plan} and {output_memory_type_from_plan} GPU memory after pattern group")

        # 🔍 VRAM TRACKING: Log memory after cleanup
        try:
            from openhcs.core.memory.gpu_cleanup import log_gpu_memory_usage
            log_gpu_memory_usage(f"pattern group {pattern_repr} (after cleanup)")
        except Exception:
            pass

        logger.debug(f"Finished pattern group {pattern_repr} in {(time.time() - start_time):.2f}s.")
    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        logger.error(f"Error processing pattern group {pattern_repr}: {e}", exc_info=True)
        logger.error(f"Full traceback for pattern group {pattern_repr}:\n{full_traceback}")
        raise ValueError(f"Failed to process pattern group {pattern_repr}: {e}") from e

class FunctionStep(AbstractStep):
    @property
    def requires_disk_input(self) -> bool: return False 
    @property
    def requires_disk_output(self) -> bool: return False

    def __init__(
        self,
        func: Union[Callable, Tuple[Callable, Dict], List[Union[Callable, Tuple[Callable, Dict]]]],
        *, name: Optional[str] = None, variable_components: List[VariableComponents] = [VariableComponents.SITE],
        group_by: GroupBy = GroupBy.CHANNEL, force_disk_output: bool = False,
        input_dir: Optional[Union[str, Path]] = None, output_dir: Optional[Union[str, Path]] = None
    ):
        actual_func_for_name = func
        if isinstance(func, tuple): actual_func_for_name = func[0]
        elif isinstance(func, list) and func:
             first_item = func[0]
             if isinstance(first_item, tuple): actual_func_for_name = first_item[0]
             elif callable(first_item): actual_func_for_name = first_item
        
        super().__init__(
            name=name or getattr(actual_func_for_name, '__name__', 'FunctionStep'),
            variable_components=variable_components, group_by=group_by,
            force_disk_output=force_disk_output,
            input_dir=input_dir, output_dir=output_dir
        )
        self.func = func # This is used by prepare_patterns_and_functions at runtime

    def process(self, context: 'ProcessingContext') -> None:
        # Generate step_id from object reference (elegant stateless approach)
        step_id = get_step_id(self)
        step_plan = context.step_plans[step_id]

        # Get step name for logging
        step_name = step_plan['step_name']

        try:
            well_id = step_plan['well_id']
            step_input_dir = Path(step_plan['input_dir'])
            step_output_dir = Path(step_plan['output_dir'])
            variable_components = step_plan['variable_components']
            group_by = step_plan['group_by']
            
            # special_inputs/outputs are dicts: {'key': 'vfs_path_value'}
            special_inputs = step_plan['special_inputs']
            special_outputs = step_plan['special_outputs'] # Should be OrderedDict if order matters

            force_disk_output = step_plan['force_disk_output']
            read_backend = step_plan['read_backend']
            write_backend = step_plan['write_backend']
            input_mem_type = step_plan['input_memory_type']
            output_mem_type = step_plan['output_memory_type']

            # Only access gpu_id if the step requires GPU (has GPU memory types)
            from openhcs.constants.constants import VALID_GPU_MEMORY_TYPES
            requires_gpu = (input_mem_type in VALID_GPU_MEMORY_TYPES or
                           output_mem_type in VALID_GPU_MEMORY_TYPES)

            if requires_gpu:
                device_id = step_plan['gpu_id']
                logger.debug(f"🔥 DEBUG: Step {step_id} gpu_id from plan: {device_id}, input_mem: {input_mem_type}, output_mem: {output_mem_type}")
            else:
                device_id = None  # CPU-only step
                logger.debug(f"🔥 DEBUG: Step {step_id} is CPU-only, input_mem: {input_mem_type}, output_mem: {output_mem_type}")

            logger.debug(f"🔥 DEBUG: Step {step_id} read_backend: {read_backend}, write_backend: {write_backend}")

            if not all([well_id, step_input_dir, step_output_dir]):
                raise ValueError(f"Plan missing essential keys for step {step_id}")

            same_dir = str(step_input_dir) == str(step_output_dir)
            logger.info(f"Step {step_id} ({step_name}) I/O: read='{read_backend}', write='{write_backend}'.")
            logger.info(f"Step {step_id} ({step_name}) Paths: input_dir='{step_input_dir}', output_dir='{step_output_dir}', same_dir={same_dir}")

            # 🔍 VRAM TRACKING: Log memory at step start
            try:
                from openhcs.core.memory.gpu_cleanup import log_gpu_memory_usage
                log_gpu_memory_usage(f"step {step_name} start")
            except Exception:
                pass

            if not context.microscope_handler:
                raise RuntimeError(f"MicroscopeHandler not in context for step {step_id}")

            # Ensure variable_components is never None - use default if missing
            if variable_components is None:
                variable_components = [VariableComponents.SITE]  # Default fallback
                logger.warning(f"Step {step_id} ({step_name}) had None variable_components, using default [SITE]")

            patterns_by_well = context.microscope_handler.auto_detect_patterns(
                str(step_input_dir),           # folder_path
                context.filemanager,           # filemanager
                read_backend,                  # backend
                well_filter=[well_id],         # well_filter
                extensions=DEFAULT_IMAGE_EXTENSIONS,  # extensions
                group_by=group_by.value if group_by else None,             # group_by
                variable_components=[vc.value for vc in variable_components]  # variable_components (guaranteed not None)
            )

            # 🔥 STEP EXECUTION DEBUG
            print(f"🔥 STEP: '{step_name}' processing well {well_id}")
            print(f"🔥 STEP: group_by={group_by}, variable_components={variable_components}")

            if well_id in patterns_by_well:
                if isinstance(patterns_by_well[well_id], dict):
                    # Grouped patterns (when group_by is set)
                    for comp_val, pattern_list in patterns_by_well[well_id].items():
                        print(f"🔥 STEP: Component '{comp_val}' has {len(pattern_list)} patterns: {pattern_list}")
                else:
                    # Ungrouped patterns (when group_by is None)
                    print(f"🔥 STEP: Found {len(patterns_by_well[well_id])} ungrouped patterns: {patterns_by_well[well_id]}")



            if well_id not in patterns_by_well or not patterns_by_well[well_id]:
                # DEBUG: Check what's actually available
                logger.error(f"🔥 PATTERN DEBUG: Failed to find patterns for well {well_id}")
                logger.error(f"🔥 PATTERN DEBUG: Looking in directory: {step_input_dir}")
                logger.error(f"🔥 PATTERN DEBUG: Read backend: {read_backend}")
                logger.error(f"🔥 PATTERN DEBUG: All detected wells: {list(patterns_by_well.keys())}")
                logger.error(f"🔥 PATTERN DEBUG: Total patterns found: {sum(len(patterns) for patterns in patterns_by_well.values())}")

                # Check memory backend state if using memory
                if read_backend == "memory":
                    from openhcs.io.base import storage_registry
                    memory_backend = storage_registry["memory"]
                    logger.error(f"🔥 PATTERN DEBUG: Memory backend has {len(memory_backend._memory_store)} entries")

                    # Show memory keys that contain the well ID
                    well_keys = [key for key in memory_backend._memory_store.keys() if well_id in key]
                    logger.error(f"🔥 PATTERN DEBUG: Memory keys containing '{well_id}': {well_keys}")

                    # Show all memory keys for context
                    all_keys = list(memory_backend._memory_store.keys())[:10]  # First 10 keys
                    logger.error(f"🔥 PATTERN DEBUG: First 10 memory keys: {all_keys}")

                    # Check what the normalized directory path looks like
                    normalized_dir = memory_backend._normalize(step_input_dir)
                    logger.error(f"🔥 PATTERN DEBUG: Normalized directory path: '{normalized_dir}'")

                raise ValueError(f"No patterns for well {well_id} in {step_input_dir} for step {step_id}.")
            
            # Get func from step plan (stored by FuncStepContractValidator during compilation)
            func_from_plan = step_plan.get('func')
            if func_from_plan is None:
                raise ValueError(f"Step plan missing 'func' for step: {step_plan.get('step_name', 'Unknown')} (ID: {step_id})")

            grouped_patterns, comp_to_funcs, comp_to_base_args = prepare_patterns_and_functions(
                patterns_by_well[well_id], func_from_plan, component=group_by.value if group_by else None
            )

            for comp_val, current_pattern_list in grouped_patterns.items():
                exec_func_or_chain = comp_to_funcs[comp_val]
                base_kwargs = comp_to_base_args[comp_val]
                for pattern_item in current_pattern_list:
                    _process_single_pattern_group(
                        context, pattern_item, exec_func_or_chain, base_kwargs,
                        step_input_dir, step_output_dir, well_id, comp_val,
                        read_backend, write_backend, input_mem_type, output_mem_type,
                        device_id, same_dir, force_disk_output,
                        special_inputs, special_outputs, # Pass the maps from step_plan
                        variable_components
                    )
            logger.info(f"FunctionStep {step_id} ({step_name}) completed for well {well_id}.")

            # 📄 OPENHCS METADATA: Create metadata file automatically after step completion
            self._create_openhcs_metadata_automatically(context, step_plan)

            # 🔍 VRAM TRACKING: Log memory before final cleanup
            try:
                from openhcs.core.memory.gpu_cleanup import log_gpu_memory_usage
                log_gpu_memory_usage(f"step {step_name} completion (before final cleanup)")
            except Exception:
                pass

            # 🔥 FINAL GPU CLEANUP: Comprehensive cleanup after step completion
            cleanup_memory_by_type(input_mem_type, device_id)
            cleanup_memory_by_type(output_mem_type, device_id)

            # Force garbage collection to ensure all references are released
            import gc
            gc.collect()

            logger.debug(f"🔥 FINAL GPU CLEANUP: Comprehensive cleanup after step {step_name} completion")

            # 🔍 VRAM TRACKING: Log memory after final cleanup
            try:
                from openhcs.core.memory.gpu_cleanup import log_gpu_memory_usage
                log_gpu_memory_usage(f"step {step_name} completion (after final cleanup)")
            except Exception:
                pass

        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            logger.error(f"Error in FunctionStep {step_id} ({step_name}): {e}", exc_info=True)
            logger.error(f"Full traceback for FunctionStep {step_id} ({step_name}):\n{full_traceback}")

            # 🔥 ERROR GPU CLEANUP: Clean up even on error to prevent memory leaks
            try:
                cleanup_memory_by_type(input_mem_type, device_id)
                cleanup_memory_by_type(output_mem_type, device_id)
                import gc
                gc.collect()
                logger.debug(f"🔥 ERROR GPU CLEANUP: Cleaned up GPU memory after error in step {step_name}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup GPU memory after error: {cleanup_error}")

            raise

    def _extract_component_metadata(self, context: 'ProcessingContext', group_by: GroupBy) -> Optional[Dict[str, str]]:
        """
        Extract component metadata from context cache safely.

        Args:
            context: ProcessingContext containing metadata_cache
            group_by: GroupBy enum specifying which component to extract

        Returns:
            Dictionary mapping component keys to display names, or None if not available
        """
        try:
            if hasattr(context, 'metadata_cache') and context.metadata_cache:
                return context.metadata_cache.get(group_by, None)
            else:
                logger.debug(f"No metadata_cache available in context for {group_by.value}")
                return None
        except Exception as e:
            logger.debug(f"Error extracting {group_by.value} metadata from cache: {e}")
            return None

    def _create_openhcs_metadata_automatically(
        self,
        context: 'ProcessingContext',
        step_plan: Dict[str, Any]
    ) -> None:
        """
        Automatically create OpenHCS metadata file after step execution.

        This is a pure function that creates OpenHCS metadata using only context state.
        Always creates metadata - no flags needed.

        Args:
            context: ProcessingContext containing microscope_handler and other state
            step_plan: Step plan dictionary containing output_dir, write_backend, etc.
        """
        # Check if this well is responsible for metadata creation
        if not step_plan.get('create_openhcs_metadata', False):
            logger.debug(f"Well {step_plan.get('well_id', 'unknown')} skipping metadata creation (not responsible)")
            return

        logger.debug(f"Well {step_plan.get('well_id', 'unknown')} creating metadata (responsible well)")

        try:
            # Extract required information from context and step_plan
            step_output_dir = Path(step_plan['output_dir'])
            write_backend = step_plan['write_backend']

            # Check if we have microscope handler for metadata extraction
            if not context.microscope_handler:
                logger.debug("No microscope_handler in context - skipping OpenHCS metadata creation")
                return

            # Get source microscope information
            source_parser_name = context.microscope_handler.parser.__class__.__name__

            # Extract metadata from source microscope handler
            try:
                grid_dimensions = context.microscope_handler.metadata_handler.get_grid_dimensions(context.input_dir)
                pixel_size = context.microscope_handler.metadata_handler.get_pixel_size(context.input_dir)
            except Exception as e:
                logger.debug(f"Could not extract grid_dimensions/pixel_size from source: {e}")
                grid_dimensions = [1, 1]  # Default fallback
                pixel_size = 1.0  # Default fallback

            # Get list of image files in output directory
            try:
                image_files = []
                if context.filemanager.exists(str(step_output_dir), write_backend):
                    # List files in output directory
                    files = context.filemanager.list_files(str(step_output_dir), write_backend)
                    # Filter for image files (common extensions)
                    image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
                    image_files = [f for f in files if Path(f).suffix.lower() in image_extensions]
                    logger.debug(f"Found {len(image_files)} image files in {step_output_dir}")
            except Exception as e:
                logger.debug(f"Could not list image files in output directory: {e}")
                image_files = []

            # Create metadata structure
            metadata = {
                "microscope_handler_name": "openhcsdata",
                "source_filename_parser_name": source_parser_name,
                "grid_dimensions": list(grid_dimensions) if hasattr(grid_dimensions, '__iter__') else [1, 1],
                "pixel_size": float(pixel_size) if pixel_size is not None else 1.0,
                "image_files": image_files,
                "channels": self._extract_component_metadata(context, GroupBy.CHANNEL),
                "wells": self._extract_component_metadata(context, GroupBy.WELL),
                "sites": self._extract_component_metadata(context, GroupBy.SITE),
                "z_indexes": self._extract_component_metadata(context, GroupBy.Z_INDEX)
            }

            # Save metadata file using same backend as step output
            from openhcs.microscopes.openhcs import OpenHCSMetadataHandler
            metadata_path = step_output_dir / OpenHCSMetadataHandler.METADATA_FILENAME

            # Always ensure we can write to the metadata path (delete if exists) - same pattern as images
            if context.filemanager.exists(str(metadata_path), write_backend):
                context.filemanager.delete(str(metadata_path), write_backend)

            # Ensure output directory exists
            context.filemanager.ensure_directory(str(step_output_dir), write_backend)

            # Create JSON content (not TSV) - OpenHCS handler expects JSON format
            import json
            json_content = json.dumps(metadata, indent=2)
            context.filemanager.save(json_content, str(metadata_path), write_backend)
            logger.debug(f"Created OpenHCS metadata file ({write_backend}): {metadata_path}")

        except Exception as e:
            # Graceful degradation - log error but don't fail the step
            logger.warning(f"Failed to create OpenHCS metadata file: {e}")
            logger.debug(f"OpenHCS metadata creation error details:", exc_info=True)
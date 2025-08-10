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
import gc
import json
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, OrderedDict as TypingOrderedDict

from openhcs.constants.constants import (DEFAULT_IMAGE_EXTENSION,
                                             DEFAULT_IMAGE_EXTENSIONS,
                                             DEFAULT_SITE_PADDING, Backend,
                                             MemoryType, VariableComponents, GroupBy)
from openhcs.constants.input_source import InputSource
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.abstract import AbstractStep, get_step_id
from openhcs.formats.func_arg_prep import prepare_patterns_and_functions
from openhcs.core.memory.stack_utils import stack_slices, unstack_slices
from openhcs.microscopes.openhcs import OpenHCSMetadataHandler

logger = logging.getLogger(__name__)





def get_all_image_paths(input_dir, backend, well_id, filemanager, microscope_handler):
    """
    Get all image file paths for a specific well from a directory.

    Args:
        input_dir: Directory to search for images
        well_id: Well identifier to filter files
        backend: Backend to use for file listing
        filemanager: FileManager instance
        microscope_handler: Microscope handler with parser for filename parsing

    Returns:
        List of full file paths for the well
    """
    # List all image files in directory
    all_image_files = filemanager.list_image_files(str(input_dir), backend)

    # Filter by well using parser (FIXED: was using naive string matching)
    well_files = []
    parser = microscope_handler.parser

    for f in all_image_files:
        filename = os.path.basename(str(f))
        metadata = parser.parse_filename(filename)
        if metadata and metadata.get('well') == well_id:
            well_files.append(str(f))

    # Remove duplicates and sort
    sorted_files = sorted(list(set(well_files)))

    # Prepare full file paths
    full_file_paths = [str(input_dir / Path(f).name) for f in sorted_files]

    logger.debug(f"Found {len(all_image_files)} total files, {len(full_file_paths)} for well {well_id}")

    return full_file_paths


def create_image_path_getter(well_id, filemanager, microscope_handler):
    """
    Create a specialized image path getter function using runtime context.

    Args:
        well_id: Well identifier
        filemanager: FileManager instance
        microscope_handler: Microscope handler with parser for filename parsing

    Returns:
        Function that takes (input_dir, backend) and returns image paths for the well
    """
    def get_paths_for_well(input_dir, backend):
        return get_all_image_paths(
            input_dir=input_dir,
            well_id=well_id,
            backend=backend,
            filemanager=filemanager,
            microscope_handler=microscope_handler
        )
    return get_paths_for_well

# Environment variable to disable universal GPU defragmentation
DISABLE_GPU_DEFRAG = os.getenv('OPENHCS_DISABLE_GPU_DEFRAG', 'false').lower() == 'true'

def _bulk_preload_step_images(
    step_input_dir: Path,
    step_output_dir: Path,
    well_id: str,
    read_backend: str,
    patterns_by_well: Dict[str, Any],
    filemanager: 'FileManager',
    microscope_handler: 'MicroscopeHandler',
    zarr_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Pre-load all images for this step from source backend into memory backend.

    This reduces I/O overhead by doing a single bulk read operation
    instead of loading images per pattern group.

    Note: External conditional logic ensures this is only called for non-memory backends.
    """
    import time
    start_time = time.time()

    logger.debug(f"🔄 BULK PRELOAD: Loading images from {read_backend} to memory for well {well_id}")

    # Get all files for this well from patterns
    all_files = []
    # Create specialized path getter for this well
    get_paths_for_well = create_image_path_getter(well_id, filemanager, microscope_handler)

    # Get all image paths for this well
    full_file_paths = get_paths_for_well(step_input_dir, read_backend)

    if not full_file_paths:
        raise RuntimeError(f"🔄 BULK PRELOAD: No files found for well {well_id} in {step_input_dir} with backend {read_backend}")

    # Load from source backend with conditional zarr_config
    if read_backend == Backend.ZARR.value:
        raw_images = filemanager.load_batch(full_file_paths, read_backend, zarr_config=zarr_config)
    else:
        raw_images = filemanager.load_batch(full_file_paths, read_backend)

    # Ensure directory exists in memory backend before saving
    filemanager.ensure_directory(str(step_input_dir), Backend.MEMORY.value)

    # Save to memory backend using OUTPUT paths
   # memory_paths = [str(step_output_dir / Path(fp).name) for fp in full_file_paths]
    for file_path in full_file_paths:
        if filemanager.exists(file_path, Backend.MEMORY.value):
            filemanager.delete(file_path, Backend.MEMORY.value)
            logger.debug(f"🔄 BULK PRELOAD: Deleted existing file {file_path} before bulk preload")

    filemanager.save_batch(raw_images, full_file_paths, Backend.MEMORY.value)
    logger.debug(f"🔄 BULK PRELOAD: Saving {file_path} to memory")

    # Clean up source references - keep only memory backend references
    del raw_images

    load_time = time.time() - start_time
    logger.debug(f"🔄 BULK PRELOAD: Completed in {load_time:.2f}s - {len(full_file_paths)} images now in memory")

def _bulk_writeout_step_images(
    step_output_dir: Path,
    write_backend: str,
    well_id: str,
    zarr_config: Optional[Dict[str, Any]],
    filemanager: 'FileManager',
    microscope_handler: Optional[Any] = None
) -> None:
    """
    Write all processed images from memory to final backend (disk/zarr).

    This reduces I/O overhead by doing a single bulk write operation
    instead of writing images per pattern group.

    Note: External conditional logic ensures this is only called for non-memory backends.
    """
    import time
    start_time = time.time()

    logger.debug(f"🔄 BULK WRITEOUT: Writing images from memory to {write_backend} for well {well_id}")

    # Create specialized path getter and get memory paths for this well
    get_paths_for_well = create_image_path_getter(well_id, filemanager, microscope_handler)
    memory_file_paths = get_paths_for_well(step_output_dir, Backend.MEMORY.value)

    if not memory_file_paths:
        raise RuntimeError(f"🔄 BULK WRITEOUT: No image files found for well {well_id} in memory directory {step_output_dir}")

    # Convert relative memory paths back to absolute paths for target backend
    # Memory backend stores relative paths, but target backend needs absolute paths
#    file_paths = 
#    for memory_path in memory_file_paths:
#        # Get just the filename and construct proper target path
#        filename = Path(memory_path).name
#        target_path = step_output_dir / filename
#        file_paths.append(str(target_path))

    file_paths = memory_file_paths
    logger.debug(f"🔄 BULK WRITEOUT: Found {len(file_paths)} image files in memory to write")

    # Load all data from memory backend
    memory_data = filemanager.load_batch(file_paths, Backend.MEMORY.value)

    # Ensure output directory exists before bulk write
    filemanager.ensure_directory(str(step_output_dir), Backend.DISK.value)

    # Bulk write to target backend with conditional zarr_config
    if write_backend == Backend.ZARR.value:
        # Calculate zarr dimensions from file paths
        if microscope_handler is not None:
            n_channels, n_z, n_fields = _calculate_zarr_dimensions(file_paths, microscope_handler)
            # Parse well to get row and column for zarr structure
            row, col = microscope_handler.parser.extract_row_column(well_id)
            filemanager.save_batch(memory_data, file_paths, write_backend,
                                 chunk_name=well_id, zarr_config=zarr_config,
                                 n_channels=n_channels, n_z=n_z, n_fields=n_fields,
                                 row=row, col=col)
        else:
            # Fallback without dimensions if microscope_handler not available
            filemanager.save_batch(memory_data, file_paths, write_backend, chunk_name=well_id, zarr_config=zarr_config)
    else:
        filemanager.save_batch(memory_data, file_paths, write_backend)

    write_time = time.time() - start_time
    logger.debug(f"🔄 BULK WRITEOUT: Completed in {write_time:.2f}s - {len(memory_data)} images written to {write_backend}")

def _calculate_zarr_dimensions(file_paths: List[Union[str, Path]], microscope_handler) -> tuple[int, int, int]:
    """
    Calculate zarr dimensions (n_channels, n_z, n_fields) from file paths using microscope parser.

    Args:
        file_paths: List of file paths to analyze
        microscope_handler: Microscope handler with filename parser

    Returns:
        Tuple of (n_channels, n_z, n_fields)
    """
    parsed_files = []
    for file_path in file_paths:
        filename = Path(file_path).name
        metadata = microscope_handler.parser.parse_filename(filename)
        parsed_files.append(metadata)

    # Count unique values for each dimension from actual files
    n_channels = len(set(f.get('channel') for f in parsed_files if f.get('channel') is not None))
    n_z = len(set(f.get('z_index') for f in parsed_files if f.get('z_index') is not None))
    n_fields = len(set(f.get('site') for f in parsed_files if f.get('site') is not None))

    # Ensure at least 1 for each dimension (handle cases where metadata is missing)
    n_channels = max(1, n_channels)
    n_z = max(1, n_z)
    n_fields = max(1, n_fields)

    return n_channels, n_z, n_fields



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
        logger.info(f"�� SPECIAL_INPUTS_DEBUG : special_inputs_plan = {special_inputs_plan}")
        for arg_name, path_info in special_inputs_plan.items():
            logger.info(f"🔍 SPECIAL_INPUTS_DEBUG: Processing arg_name='{arg_name}', path_info={path_info} (type: {type(path_info)})")


            # Extract path string from the path info dictionary
            # Current format: {"path": "/path/to/file.pkl", "source_step_id": "step_123"}
            if isinstance(path_info, dict) and 'path' in path_info:
                special_path_value = path_info['path']
                logger.info(f"🔍 SPECIAL_INPUTS_DEBUG: Extracted path from dict: '{special_path_value}' (type: {type(special_path_value)})")
            else:
                special_path_value = path_info  # Fallback if it's already a string
                logger.info(f"🔍 SPECIAL_INPUTS_DEBUG: Using path_info directly: '{special_path_value}' (type: {type(special_path_value)})")

            logger.info(f"Loading special input '{arg_name}' from path '{special_path_value}' (memory backend)")
            try:
                final_kwargs[arg_name] = context.filemanager.load(special_path_value, Backend.MEMORY.value)
            except Exception as e:
                logger.error(f"Failed to load special input '{arg_name}' from '{special_path_value}': {e}", exc_info=True)
                raise

    # Auto-inject context if function signature expects it
    import inspect
    sig = inspect.signature(func_callable)
    if 'context' in sig.parameters:
        final_kwargs['context'] = context

    # 🔍 DEBUG: Log input dimensions
    input_shape = getattr(main_data_arg, 'shape', 'no shape attr')
    input_type = type(main_data_arg).__name__
    logger.debug(f"🔍 FUNCTION INPUT: {func_callable.__name__} - shape: {input_shape}, type: {input_type}")

    # ⚡ INFO: Terse function execution log for user feedback
    logger.info(f"⚡ Executing: {func_callable.__name__}")

    # 🔍 DEBUG: Log function attributes before execution
    logger.debug(f"🔍 FUNCTION ATTRS: {func_callable.__name__} - special_outputs: {getattr(func_callable, '__special_outputs__', 'None')}")
    logger.debug(f"🔍 FUNCTION ATTRS: {func_callable.__name__} - input_memory_type: {getattr(func_callable, 'input_memory_type', 'None')}")
    logger.debug(f"🔍 FUNCTION ATTRS: {func_callable.__name__} - output_memory_type: {getattr(func_callable, 'output_memory_type', 'None')}")

    raw_function_output = func_callable(main_data_arg, **final_kwargs)

    # 🔍 DEBUG: Log output dimensions and type details
    output_shape = getattr(raw_function_output, 'shape', 'no shape attr')
    output_type = type(raw_function_output).__name__
    logger.debug(f"🔍 FUNCTION OUTPUT: {func_callable.__name__} - shape: {output_shape}, type: {output_type}")

    # 🔍 DEBUG: If it's a tuple, log details about each element
    if isinstance(raw_function_output, tuple):
        logger.debug(f"🔍 FUNCTION OUTPUT: {func_callable.__name__} - tuple length: {len(raw_function_output)}")
        for i, element in enumerate(raw_function_output):
            elem_shape = getattr(element, 'shape', 'no shape attr')
            elem_type = type(element).__name__
            logger.debug(f"🔍 FUNCTION OUTPUT: {func_callable.__name__} - element[{i}]: shape={elem_shape}, type={elem_type}")
    else:
        logger.debug(f"🔍 FUNCTION OUTPUT: {func_callable.__name__} - not a tuple, single return value")

    main_output_data = raw_function_output

    # 🔍 DEBUG: Log special output plan status
    logger.debug(f"🔍 SPECIAL OUTPUT PLAN: {special_outputs_plan}")
    logger.debug(f"🔍 SPECIAL OUTPUT PLAN: Is empty? {not special_outputs_plan}")
    logger.debug(f"🔍 SPECIAL OUTPUT PLAN: Length: {len(special_outputs_plan) if special_outputs_plan else 0}")

    # Only log special outputs if there are any (avoid spamming empty dict logs)
    if special_outputs_plan:
        logger.debug(f"🔍 SPECIAL OUTPUT: {special_outputs_plan}")
    if special_outputs_plan:
        num_special_outputs = len(special_outputs_plan)
        logger.debug(f"🔍 SPECIAL OUTPUT PROCESSING: Expected {num_special_outputs} special outputs")
        logger.debug(f"🔍 SPECIAL OUTPUT PROCESSING: Function returned type: {type(raw_function_output)}")
        logger.debug(f"🔍 SPECIAL OUTPUT PROCESSING: Function returned tuple length: {len(raw_function_output) if isinstance(raw_function_output, tuple) else 'not tuple'}")

        if not isinstance(raw_function_output, tuple) or len(raw_function_output) != (1 + num_special_outputs):
            logger.error(f"🔍 SPECIAL OUTPUT ERROR: Function '{getattr(func_callable, '__name__', 'unknown')}' special output mismatch")
            logger.error(f"🔍 SPECIAL OUTPUT ERROR: Expected tuple of {1 + num_special_outputs} values")
            logger.error(f"🔍 SPECIAL OUTPUT ERROR: Got {type(raw_function_output)} with {len(raw_function_output) if isinstance(raw_function_output, tuple) else 'N/A'} values")
            raise ValueError(
                f"Function '{getattr(func_callable, '__name__', 'unknown')}' was expected to return a tuple of "
                f"{1 + num_special_outputs} values (main_output + {num_special_outputs} special) "
                f"based on 'special_outputs' in step plan, but returned {len(raw_function_output) if isinstance(raw_function_output, tuple) else type(raw_function_output)} values."
            )
        main_output_data = raw_function_output[0]
        returned_special_values_tuple = raw_function_output[1:]

        # 🔍 DEBUG: Log what we extracted
        logger.debug(f"🔍 SPECIAL OUTPUT PROCESSING: Extracted main_output_data type: {type(main_output_data)}")
        logger.debug(f"🔍 SPECIAL OUTPUT PROCESSING: Extracted main_output_data shape: {getattr(main_output_data, 'shape', 'no shape')}")
        logger.debug(f"🔍 SPECIAL OUTPUT PROCESSING: Extracted {len(returned_special_values_tuple)} special values")

        # Iterate through special_outputs_plan (which must be ordered by compiler)
        # and match with positionally returned special values.
        for i, (output_key, vfs_path_info) in enumerate(special_outputs_plan.items()):
            logger.info(f"Saving special output '{output_key}' to VFS path '{vfs_path_info}' (memory backend)")
            if i < len(returned_special_values_tuple):
                value_to_save = returned_special_values_tuple[i]
                # Extract path string from the path info dictionary
                # Current format: {"path": "/path/to/file.pkl"}
                if isinstance(vfs_path_info, dict) and 'path' in vfs_path_info:
                    vfs_path = vfs_path_info['path']
                else:
                    vfs_path = vfs_path_info  # Fallback if it's already a string
               # # Add well_id prefix to filename for memory backend to avoid thread collisions
               # from pathlib import Path
               # vfs_path_obj = Path(vfs_path)
               # prefixed_filename = f"{well_id}_{vfs_path_obj.name}"
               # prefixed_vfs_path = str(vfs_path_obj.parent / prefixed_filename)

                logger.info(f"🔍 SPECIAL_SAVE: Saving '{output_key}' to '{vfs_path}' (memory backend)")
                # Ensure directory exists for memory backend
                parent_dir = str(Path(vfs_path).parent)
                context.filemanager.ensure_directory(parent_dir, Backend.MEMORY.value)
                context.filemanager.save(value_to_save, vfs_path, Backend.MEMORY.value)
                logger.info(f"🔍 SPECIAL_SAVE: Successfully saved '{output_key}' to memory")
                logger.info(f"🔍 SPECIAL_SAVE: Successfully saved '{output_key}' to memory")
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
    step_id: str,  # Add step_id for funcplan lookup
    dict_key: str = "default"  # Add dict_key for funcplan lookup
) -> Any:
    current_stack = initial_data_stack
    current_memory_type = input_memory_type  # Track memory type from frozen context

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

        # Convert to function's input memory type (noop if same)
        from openhcs.core.memory.converters import convert_memory
        current_stack = convert_memory(
            data=current_stack,
            source_type=current_memory_type,
            target_type=actual_callable.input_memory_type,
            gpu_id=device_id,
            allow_cpu_roundtrip=False
        )

        # Use funcplan to determine which outputs this function should save
        funcplan = context.step_plans[step_id].get("funcplan", {})
        func_name = getattr(actual_callable, '__name__', 'unknown')

        # Construct execution key: function_name_dict_key_chain_position
        execution_key = f"{func_name}_{dict_key}_{i}"

        if execution_key in funcplan:
            # Get outputs this specific function should save
            outputs_to_save = funcplan[execution_key]
            outputs_plan_for_this_call = {
                key: step_special_outputs_plan[key]
                for key in outputs_to_save
                if key in step_special_outputs_plan
            }
            logger.info(f"🔍 FUNCPLAN: {execution_key} -> {outputs_to_save}")
            logger.info(f"🔍 FUNCPLAN: outputs_plan_for_this_call = {outputs_plan_for_this_call}")
        else:
            # Fallback: no funcplan entry, save nothing
            outputs_plan_for_this_call = {}
            logger.info(f"🔍 FUNCPLAN: No entry for {execution_key}, saving nothing")

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

        # Update current memory type from frozen context
        current_memory_type = actual_callable.output_memory_type

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
    zarr_config: Optional[Dict[str, Any]],
    variable_components: Optional[List[str]] = None,
    step_id: Optional[str] = None  # Add step_id for funcplan lookup
) -> None:
    start_time = time.time()
    pattern_repr = str(pattern_group_info)[:100]
    logger.debug(f"🔥 PATTERN: Processing {pattern_repr} for well {well_id}")

    try:
        if not context.microscope_handler:
             raise RuntimeError("MicroscopeHandler not available in context.")

        matching_files = context.microscope_handler.path_list_from_pattern(
            str(step_input_dir), pattern_group_info, context.filemanager, Backend.MEMORY.value,
            [vc.value for vc in variable_components] if variable_components else None
        )

        if not matching_files:
            raise ValueError(
                f"No matching files found for pattern group {pattern_repr} in {step_input_dir}. "
                f"This indicates either: (1) no image files exist in the directory, "
                f"(2) files don't match the pattern, or (3) pattern parsing failed. "
                f"Check that input files exist and match the expected naming convention."
            )

        logger.debug(f"🔥 PATTERN: Found {len(matching_files)} files: {[Path(f).name for f in matching_files]}")

        # Sort files to ensure consistent ordering (especially important for z-stacks)
        matching_files.sort()
        logger.debug(f"🔥 PATTERN: Sorted files: {[Path(f).name for f in matching_files]}")

        full_file_paths = [str(step_input_dir / f) for f in matching_files]
        raw_slices = context.filemanager.load_batch(full_file_paths, Backend.MEMORY.value)
        
        if not raw_slices:
            raise ValueError(
                f"No valid images loaded for pattern group {pattern_repr} in {step_input_dir}. "
                f"Found {len(matching_files)} matching files but failed to load any valid images. "
                f"This indicates corrupted image files, unsupported formats, or I/O errors. "
                f"Check file integrity and format compatibility."
            )

        # 🔍 DEBUG: Log stacking operation
        logger.debug(f"🔍 STACKING: {len(raw_slices)} slices → memory_type: {input_memory_type_from_plan}")
        if raw_slices:
            slice_shapes = [getattr(s, 'shape', 'no shape') for s in raw_slices[:3]]  # First 3 shapes
            logger.debug(f"🔍 STACKING: Sample slice shapes: {slice_shapes}")

        main_data_stack = stack_slices(
            slices=raw_slices, memory_type=input_memory_type_from_plan, gpu_id=device_id
        )

        # 🔍 DEBUG: Log stacked result
        stack_shape = getattr(main_data_stack, 'shape', 'no shape')
        stack_type = type(main_data_stack).__name__
        logger.debug(f"🔍 STACKED RESULT: shape: {stack_shape}, type: {stack_type}")
        
        logger.info(f"🔍 special_outputs_map: {special_outputs_map}")
        
        final_base_kwargs = base_func_args.copy()
        
        # Get step function from step plan
        step_func = context.step_plans[step_id]["func"]

        if isinstance(step_func, dict):
            dict_key_for_funcplan = component_value  # Use actual dict key for dict patterns
        else:
            dict_key_for_funcplan = "default"  # Use default for list/single patterns

        if isinstance(executable_func_or_chain, list):
            processed_stack = _execute_chain_core(
                main_data_stack, executable_func_or_chain, context,
                special_inputs_map, special_outputs_map, well_id,
                device_id, input_memory_type_from_plan, step_id, dict_key_for_funcplan
            )
        elif callable(executable_func_or_chain):
            # For single functions, we don't need chain execution, but we still need the right dict_key
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
        logger.debug(f"🔍 PROCESSING RESULT: input: {input_shape} → output: {output_shape}, type: {processed_type}")

        # 🔍 DEBUG: Additional validation logging
        logger.debug(f"🔍 VALIDATION: processed_stack type: {type(processed_stack)}")
        logger.debug(f"🔍 VALIDATION: processed_stack has shape attr: {hasattr(processed_stack, 'shape')}")
        logger.debug(f"🔍 VALIDATION: processed_stack has ndim attr: {hasattr(processed_stack, 'ndim')}")
        if hasattr(processed_stack, 'ndim'):
            logger.debug(f"🔍 VALIDATION: processed_stack ndim: {processed_stack.ndim}")
        if hasattr(processed_stack, 'shape'):
            logger.debug(f"🔍 VALIDATION: processed_stack shape: {processed_stack.shape}")

        if not _is_3d(processed_stack):
            logger.error(f"🔍 VALIDATION ERROR: processed_stack is not 3D")
            logger.error(f"🔍 VALIDATION ERROR: Type: {type(processed_stack)}")
            logger.error(f"🔍 VALIDATION ERROR: Shape: {getattr(processed_stack, 'shape', 'no shape attr')}")
            logger.error(f"🔍 VALIDATION ERROR: Has ndim: {hasattr(processed_stack, 'ndim')}")
            if hasattr(processed_stack, 'ndim'):
                logger.error(f"🔍 VALIDATION ERROR: ndim value: {processed_stack.ndim}")
            raise ValueError(f"Main processing must result in a 3D array, got {getattr(processed_stack, 'shape', 'unknown')}")

        # 🔍 DEBUG: Log unstacking operation
        logger.debug(f"🔍 UNSTACKING: shape: {output_shape} → memory_type: {output_memory_type_from_plan}")



        output_slices = unstack_slices(
            array=processed_stack, memory_type=output_memory_type_from_plan, gpu_id=device_id, validate_slices=True
        )

        # 🔍 DEBUG: Log unstacked result
        if output_slices:
            unstacked_shapes = [getattr(s, 'shape', 'no shape') for s in output_slices[:3]]  # First 3 shapes
            logger.debug(f"🔍 UNSTACKED RESULT: {len(output_slices)} slices, sample shapes: {unstacked_shapes}")

        # Handle cases where function returns fewer images than inputs (e.g., z-stack flattening, channel compositing)
        # In such cases, we save only the returned images using the first N input filenames
        num_outputs = len(output_slices)
        num_inputs = len(matching_files)

        if num_outputs < num_inputs:
            logger.debug(f"Function returned {num_outputs} images from {num_inputs} inputs - likely flattening operation")
        elif num_outputs > num_inputs:
            logger.warning(f"Function returned more images ({num_outputs}) than inputs ({num_inputs}) - unexpected")

        # Save the output images using batch operations
        try:
            # Prepare batch data
            output_data = []
            output_paths_batch = []

            for i, img_slice in enumerate(output_slices):
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
                if context.filemanager.exists(str(output_path), Backend.MEMORY.value):
                    context.filemanager.delete(str(output_path), Backend.MEMORY.value)

                output_data.append(img_slice)
                output_paths_batch.append(str(output_path))

            # Ensure directory exists
            context.filemanager.ensure_directory(str(step_output_dir), Backend.MEMORY.value)

                          # Only pass zarr_config to zarr backend - fail loud for invalid parameters
                    #if write_backend == Backend.ZARR.value:
          # Batch save
           # context.filemanager.save_batch(output_data, output_paths_batch, write_backend, zarr_config=zarr_config)
           #         else:
            context.filemanager.save_batch(output_data, output_paths_batch, Backend.MEMORY.value)

            # Force disk output if needed
            if force_disk_output_flag and write_backend != Backend.DISK.value:
                logger.info(f"Force disk output: saving additional copy to disk at {step_output_dir}")
                context.filemanager.ensure_directory(str(step_output_dir), Backend.DISK.value)
                # Disk backend doesn't need zarr_config - fail loud for invalid parameters
                context.filemanager.save_batch(output_data, output_paths_batch, Backend.DISK.value)

        except Exception as e:
            logger.error(f"Error saving batch of output slices for pattern {pattern_repr}: {e}", exc_info=True)

        # 🔥 CLEANUP: If function returned fewer images than inputs, delete the unused input files
        # This prevents unused channel files from remaining in memory after compositing
        if num_outputs < num_inputs:
            for j in range(num_outputs, num_inputs):
                unused_input_filename = matching_files[j]
                unused_input_path = Path(step_input_dir) / unused_input_filename
                if context.filemanager.exists(str(unused_input_path), Backend.MEMORY.value):
                    context.filemanager.delete(str(unused_input_path), Backend.MEMORY.value)
                    logger.debug(f"🔥 CLEANUP: Deleted unused input file: {unused_input_filename}")



        logger.debug(f"Finished pattern group {pattern_repr} in {(time.time() - start_time):.2f}s.")
    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        logger.error(f"Error processing pattern group {pattern_repr}: {e}", exc_info=True)
        logger.error(f"Full traceback for pattern group {pattern_repr}:\n{full_traceback}")
        raise ValueError(f"Failed to process pattern group {pattern_repr}: {e}") from e

class FunctionStep(AbstractStep):

    def __init__(
        self,
        func: Union[Callable, Tuple[Callable, Dict], List[Union[Callable, Tuple[Callable, Dict]]]],
        *, name: Optional[str] = None, variable_components: List[VariableComponents] = [VariableComponents.SITE],
        group_by: GroupBy = GroupBy.CHANNEL, force_disk_output: bool = False,
        input_dir: Optional[Union[str, Path]] = None, output_dir: Optional[Union[str, Path]] = None,
        input_source: InputSource = InputSource.PREVIOUS_STEP
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
            input_dir=input_dir, output_dir=output_dir,
            input_source=input_source
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
            func_from_plan = step_plan['func']
            
            # special_inputs/outputs are dicts: {'key': 'vfs_path_value'}
            special_inputs = step_plan['special_inputs']
            special_outputs = step_plan['special_outputs'] # Should be OrderedDict if order matters

            force_disk_output = step_plan['force_disk_output']
            read_backend = step_plan['read_backend']
            write_backend = step_plan['write_backend']
            input_mem_type = step_plan['input_memory_type']
            output_mem_type = step_plan['output_memory_type']
            microscope_handler = context.microscope_handler
            filemanager = context.filemanager

            # Create path getter for this well
            get_paths_for_well = create_image_path_getter(well_id, filemanager, microscope_handler)

            # Get patterns first for bulk preload
            patterns_by_well = microscope_handler.auto_detect_patterns(
                str(step_input_dir),           # folder_path
                filemanager,           # filemanager
                read_backend,                  # backend
                well_filter=[well_id],         # well_filter
                extensions=DEFAULT_IMAGE_EXTENSIONS,  # extensions
                group_by=group_by.value if group_by else None,             # group_by
                variable_components=[vc.value for vc in variable_components] if variable_components else None  # variable_components
            )            


            # Only access gpu_id if the step requires GPU (has GPU memory types)
            from openhcs.constants.constants import VALID_GPU_MEMORY_TYPES
            requires_gpu = (input_mem_type in VALID_GPU_MEMORY_TYPES or
                           output_mem_type in VALID_GPU_MEMORY_TYPES)

                        # Ensure variable_components is never None - use default if missing
            if variable_components is None:
                variable_components = [VariableComponents.SITE]  # Default fallback
                logger.warning(f"Step {step_id} ({step_name}) had None variable_components, using default [SITE]")
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

            # 🔄 MATERIALIZATION READ: Bulk preload if not reading from memory
            if read_backend != Backend.MEMORY.value:
                _bulk_preload_step_images(step_input_dir, step_output_dir, well_id, read_backend,
                                        patterns_by_well,filemanager, microscope_handler, step_plan["zarr_config"])

            # 🔄 ZARR CONVERSION: Convert loaded memory data to zarr if needed
            convert_to_zarr_path = step_plan.get('convert_to_zarr')
            if convert_to_zarr_path:
                logger.info(f"Converting loaded data to zarr: {convert_to_zarr_path}")
                zarr_config = step_plan.get('zarr_config', context.global_config.zarr)

                # Get memory paths and data, then create zarr paths pointing to plate root
                memory_paths = get_paths_for_well(step_input_dir, Backend.MEMORY.value)
                memory_data = filemanager.load_batch(memory_paths, Backend.MEMORY.value)

                # Create zarr paths by joining convert_to_zarr_path with just the filename
                # This creates paths like /plate/images.zarr/image001.tiff
                # The zarr backend will use the filename as the key within the store
                zarr_paths = []
                for memory_path in memory_paths:
                    filename = Path(memory_path).name
                    zarr_path = Path(convert_to_zarr_path) / filename
                    zarr_paths.append(str(zarr_path))

                # Parse actual filenames to determine dimensions
                # Calculate zarr dimensions from zarr paths (which contain the filenames)
                n_channels, n_z, n_fields = _calculate_zarr_dimensions(zarr_paths, context.microscope_handler)
                # Parse well to get row and column for zarr structure
                row, col = context.microscope_handler.parser.extract_row_column(well_id)

                filemanager.save_batch(memory_data, zarr_paths, Backend.ZARR.value,
                                     chunk_name=well_id, zarr_config=zarr_config,
                                     n_channels=n_channels, n_z=n_z, n_fields=n_fields,
                                     row=row, col=col)

                # 📄 OPENHCS METADATA: Create metadata for zarr conversion (in plate directory)
                # convert_to_zarr_path points to the zarr store (e.g., /plate/images.zarr)
                # but metadata should be in the plate directory (e.g., /plate)
                plate_dir = context.zarr_conversion_path
                self._create_openhcs_metadata_for_materialization(context, plate_dir, Backend.ZARR.value)

            # 🔍 VRAM TRACKING: Log memory at step start
            try:
                from openhcs.core.memory.gpu_cleanup import log_gpu_memory_usage
                log_gpu_memory_usage(f"step {step_name} start")
            except ImportError:
                pass  # GPU cleanup not available



                log_gpu_memory_usage(f"step {step_name} start")
            except Exception:
                pass

            logger.info(f"🔥 STEP: Starting processing for '{step_name}' well {well_id} (group_by={group_by.name}, variable_components={[vc.name for vc in variable_components]})")

            if well_id not in patterns_by_well:
                raise ValueError(
                    f"No patterns detected for well '{well_id}' in step '{step_name}' (ID: {step_id}). "
                    f"This indicates either: (1) no image files found for this well, "
                    f"(2) image files don't match the expected naming pattern, or "
                    f"(3) pattern detection failed. Check input directory: {step_input_dir}"
                )

            if isinstance(patterns_by_well[well_id], dict):
                # Grouped patterns (when group_by is set)
                for comp_val, pattern_list in patterns_by_well[well_id].items():
                    logger.debug(f"🔥 STEP: Component '{comp_val}' has {len(pattern_list)} patterns: {pattern_list}")
            else:
                # Ungrouped patterns (when group_by is None)
                logger.debug(f"🔥 STEP: Found {len(patterns_by_well[well_id])} ungrouped patterns: {patterns_by_well[well_id]}")

            if func_from_plan is None:
                raise ValueError(f"Step plan missing 'func' for step: {step_plan.get('step_name', 'Unknown')} (ID: {step_id})")

            grouped_patterns, comp_to_funcs, comp_to_base_args = prepare_patterns_and_functions(
                patterns_by_well[well_id], func_from_plan, component=group_by.value if group_by else None
            )

            logger.info(f"🔍 DICT_PATTERN: grouped_patterns keys: {list(grouped_patterns.keys())}")
            logger.info(f"🔍 DICT_PATTERN: comp_to_funcs keys: {list(comp_to_funcs.keys())}")
            logger.info(f"🔍 DICT_PATTERN: func_from_plan type: {type(func_from_plan)}")
            if isinstance(func_from_plan, dict):
                logger.info(f"🔍 DICT_PATTERN: func_from_plan keys: {list(func_from_plan.keys())}")

            for comp_val, current_pattern_list in grouped_patterns.items():
                logger.info(f"🔍 DICT_PATTERN: Processing component '{comp_val}' with {len(current_pattern_list)} patterns")
                exec_func_or_chain = comp_to_funcs[comp_val]
                base_kwargs = comp_to_base_args[comp_val]
                logger.info(f"🔍 DICT_PATTERN: Component '{comp_val}' exec_func_or_chain: {exec_func_or_chain}")
                for pattern_item in current_pattern_list:
                    _process_single_pattern_group(
                        context, pattern_item, exec_func_or_chain, base_kwargs,
                        step_input_dir, step_output_dir, well_id, comp_val,
                        read_backend, write_backend, input_mem_type, output_mem_type,
                        device_id, same_dir, force_disk_output,
                        special_inputs, special_outputs, # Pass the maps from step_plan
                        step_plan["zarr_config"],
                        variable_components, step_id  # Pass step_id for funcplan lookup
                    )
            logger.info(f"🔥 STEP: Completed processing for '{step_name}' well {well_id}.")
            
            # 📄 MATERIALIZATION WRITE: Only if not writing to memory
            if write_backend != Backend.MEMORY.value:
                memory_paths = get_paths_for_well(step_output_dir, Backend.MEMORY.value)
                memory_data = filemanager.load_batch(memory_paths, Backend.MEMORY.value)
                # Calculate zarr dimensions (ignored by non-zarr backends)
                n_channels, n_z, n_fields = _calculate_zarr_dimensions(memory_paths, context.microscope_handler)
                row, col = context.microscope_handler.parser.extract_row_column(well_id)
                filemanager.ensure_directory(step_output_dir, write_backend)
                filemanager.save_batch(memory_data, memory_paths, write_backend,
                                     chunk_name=well_id, zarr_config=step_plan["zarr_config"],
                                     n_channels=n_channels, n_z=n_z, n_fields=n_fields,
                                     row=row, col=col)
            
            logger.info(f"FunctionStep {step_id} ({step_name}) completed for well {well_id}.")

            # 📄 OPENHCS METADATA: Create metadata file automatically after step completion
            self._create_openhcs_metadata_for_materialization(context, step_plan['output_dir'], step_plan['write_backend'])

            # 🔬 SPECIAL DATA MATERIALIZATION
            special_outputs = step_plan.get('special_outputs', {})
            logger.debug(f"🔍 MATERIALIZATION: special_outputs from step_plan: {special_outputs}")
            logger.debug(f"🔍 MATERIALIZATION: special_outputs is empty? {not special_outputs}")
            if special_outputs:
                logger.info(f"🔬 MATERIALIZATION: Starting materialization for {len(special_outputs)} special outputs")
                self._materialize_special_outputs(filemanager, step_plan, special_outputs)
                logger.info(f"🔬 MATERIALIZATION: Completed materialization")
            else:
                logger.debug(f"🔍 MATERIALIZATION: No special outputs to materialize")



        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            logger.error(f"Error in FunctionStep {step_id} ({step_name}): {e}", exc_info=True)
            logger.error(f"Full traceback for FunctionStep {step_id} ({step_name}):\n{full_traceback}")



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

    def _create_openhcs_metadata_for_materialization(
        self,
        context: 'ProcessingContext',
        output_dir: str,
        write_backend: str
    ) -> None:
        """
        Create OpenHCS metadata file for materialization writes.

        Args:
            context: ProcessingContext containing microscope_handler and other state
            output_dir: Output directory path where metadata should be written
            write_backend: Backend being used for the write (disk/zarr)
        """
        # Check if this is a materialization write (disk/zarr) - memory writes don't need metadata
        if write_backend == Backend.MEMORY.value:
            logger.debug(f"Skipping metadata creation (memory write)")
            return

        logger.debug(f"Creating metadata for materialization write: {write_backend} -> {output_dir}")

        try:
            # Extract required information
            step_output_dir = Path(output_dir)

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
                    # Filter for image files (common extensions) and convert to strings
                    image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
                    image_files = [str(f) for f in files if Path(f).suffix.lower() in image_extensions]
                    logger.debug(f"Found {len(image_files)} image files in {step_output_dir}")
            except Exception as e:
                logger.debug(f"Could not list image files in output directory: {e}")
                image_files = []

            # Detect available backends based on actual output files
            available_backends = self._detect_available_backends(step_output_dir)

            # Create metadata structure
            metadata = {
                "microscope_handler_name": context.microscope_handler.microscope_type,
                "source_filename_parser_name": source_parser_name,
                "grid_dimensions": list(grid_dimensions) if hasattr(grid_dimensions, '__iter__') else [1, 1],
                "pixel_size": float(pixel_size) if pixel_size is not None else 1.0,
                "image_files": image_files,
                "channels": self._extract_component_metadata(context, GroupBy.CHANNEL),
                "wells": self._extract_component_metadata(context, GroupBy.WELL),
                "sites": self._extract_component_metadata(context, GroupBy.SITE),
                "z_indexes": self._extract_component_metadata(context, GroupBy.Z_INDEX),
                "available_backends": available_backends
            }

            # Save metadata file using disk backend (JSON files always on disk)
            metadata_path = Path(context.output_plate_root) / OpenHCSMetadataHandler.METADATA_FILENAME

            # Always ensure we can write to the metadata path (delete if exists)
            if context.filemanager.exists(str(metadata_path), Backend.DISK.value):
                context.filemanager.delete(str(metadata_path), Backend.DISK.value)

            # Ensure output plate root directory exists on disk
            context.filemanager.ensure_directory(str(context.output_plate_root), Backend.DISK.value)

            # Create JSON content - OpenHCS handler expects JSON format
            json_content = json.dumps(metadata, indent=2)
            context.filemanager.save(json_content, str(metadata_path), Backend.DISK.value)
            logger.debug(f"Created OpenHCS metadata file at output plate root (disk): {metadata_path}")

        except Exception as e:
            # Graceful degradation - log error but don't fail the step
            logger.warning(f"Failed to create OpenHCS metadata file: {e}")
            logger.debug(f"OpenHCS metadata creation error details:", exc_info=True)

    def _detect_available_backends(self, output_dir: Path) -> Dict[str, bool]:
        """Detect which storage backends are actually available based on output files."""

        backends = {Backend.ZARR.value: False, Backend.DISK.value: False}

        # Check for zarr stores
        if list(output_dir.glob("*.zarr")):
            backends[Backend.ZARR.value] = True

        # Check for image files
        for ext in DEFAULT_IMAGE_EXTENSIONS:
            if list(output_dir.glob(f"*{ext}")):
                backends[Backend.DISK.value] = True
                break

        logger.debug(f"Backend detection result: {backends}")
        return backends

    def _materialize_special_outputs(self, filemanager, step_plan, special_outputs):
        """Load special data from memory and call materialization functions."""
        logger.debug(f"🔍 MATERIALIZE_METHOD: Processing {len(special_outputs)} special outputs")

        for output_key, output_info in special_outputs.items():
            logger.debug(f"🔍 MATERIALIZE_METHOD: Processing output_key: {output_key}")
            logger.debug(f"🔍 MATERIALIZE_METHOD: output_info: {output_info}")

            mat_func = output_info.get('materialization_function')
            logger.debug(f"🔍 MATERIALIZE_METHOD: materialization_function: {mat_func}")

            if mat_func:
                path = output_info['path']
                logger.info(f"🔬 MATERIALIZING: {output_key} from {path}")

                try:
                    filemanager.ensure_directory(Path(path).parent, Backend.MEMORY.value)
                    special_data = filemanager.load(path, Backend.MEMORY.value)
                    logger.debug(f"🔍 MATERIALIZE_METHOD: Loaded special data type: {type(special_data)}")

                    result_path = mat_func(special_data, path, filemanager)
                    logger.info(f"🔬 MATERIALIZED: {output_key} → {result_path}")

                except Exception as e:
                    logger.error(f"🔬 MATERIALIZATION ERROR: Failed to materialize {output_key}: {e}")
                    raise
            else:
                logger.warning(f"🔬 MATERIALIZATION: No materialization function for {output_key}, skipping")




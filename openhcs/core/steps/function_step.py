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
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    OrderedDict as TypingOrderedDict,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    pass


from openhcs.constants.constants import (
    DEFAULT_IMAGE_EXTENSIONS,
    Backend,
    VariableComponents,
)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.progress import emit, ProgressPhase, ProgressStatus
from openhcs.formats.func_arg_prep import prepare_patterns_and_functions
from openhcs.core.memory import stack_slices, unstack_slices

# OpenHCS imports moved to local imports to avoid circular dependencies


logger = logging.getLogger(__name__)


def _generate_materialized_paths(
    memory_paths: List[str], step_output_dir: Path, materialized_output_dir: Path
) -> List[str]:
    """Generate materialized file paths by replacing step output directory."""
    materialized_paths = []
    for memory_path in memory_paths:
        relative_path = Path(memory_path).relative_to(step_output_dir)
        materialized_path = materialized_output_dir / relative_path
        materialized_paths.append(str(materialized_path))
    return materialized_paths


def _filter_special_outputs_for_function(
    outputs_to_save: List[str], special_outputs_map: Dict
) -> Dict:
    """Filter special outputs for a specific function call.

    Args:
        outputs_to_save: List of output keys this function should save
        special_outputs_map: Map of all special outputs for the step

    Returns:
        Filtered map for this function
    """
    result = {}
    for key in outputs_to_save:
        if key in special_outputs_map:
            result[key] = special_outputs_map[key]

    return result


def _select_special_plan_for_component(
    plan_by_group: Optional[Dict], component_key: Optional[str], default_plan: Dict
) -> Dict:
    """Select precompiled special I/O plan for a component."""
    if not plan_by_group:
        return default_plan

    if component_key in plan_by_group:
        return plan_by_group[component_key]
    if None in plan_by_group:
        return plan_by_group[None]
    return default_plan


def _filter_patterns_by_component(
    patterns: Union[List, Dict], component: str, target_value: str, microscope_handler
) -> Union[List, Dict]:
    """Filter patterns to only include those matching a specific component value.

    Pattern strings encode fixed component values (e.g., 'A01_s{iii}_w1_z003_t001.tif' has z=003).
    This function extracts those values by temporarily replacing placeholders with dummy values
    to parse the pattern, following the same convention used in PatternDiscoveryEngine.

    Args:
        patterns: List of patterns or dict of grouped patterns
        component: Component name to filter by (e.g., 'z_index', 'channel')
        target_value: Target component value (e.g., '3' for z-slice 3)
        microscope_handler: MicroscopeHandler for parsing patterns

    Returns:
        Filtered patterns in the same format as input
    """
    from openhcs.formats.pattern.pattern_discovery import PatternDiscoveryEngine

    def filter_pattern_list(pattern_list: List) -> List:
        """Filter a list of patterns by component value."""
        filtered = []
        for pattern in pattern_list:
            # Replace placeholder with dummy value to make pattern parseable
            # This follows the same convention as PatternDiscoveryEngine
            pattern_template = str(pattern).replace(
                PatternDiscoveryEngine.PLACEHOLDER_PATTERN, "001"
            )
            metadata = microscope_handler.parser.parse_filename(pattern_template)

            if metadata and str(metadata.get(component)) == str(target_value):
                filtered.append(pattern)

        return filtered

    # If patterns is already grouped (dict), filter within each group
    if isinstance(patterns, dict):
        filtered = {}
        for group_key, pattern_list in patterns.items():
            filtered_list = filter_pattern_list(pattern_list)
            if filtered_list:
                filtered[group_key] = filtered_list
        return filtered
    else:
        # Patterns is a flat list
        return filter_pattern_list(patterns)


def _save_materialized_data(
    filemanager,
    memory_data: List,
    materialized_paths: List[str],
    materialized_backend: str,
    step_plan: Dict,
    context,
    axis_id: str,
) -> None:
    """Save data to materialized location using appropriate backend."""

    # Build kwargs with parser metadata (all backends receive it)
    save_kwargs = {
        "parser_name": context.microscope_handler.parser.__class__.__name__,
        "microscope_type": context.microscope_handler.microscope_type,
    }

    if materialized_backend == Backend.ZARR.value:
        n_channels, n_z, n_fields = _calculate_zarr_dimensions(
            materialized_paths, context.microscope_handler
        )
        row, col = context.microscope_handler.parser.extract_component_coordinates(
            axis_id
        )
        save_kwargs.update(
            {
                "chunk_name": axis_id,
                "zarr_config": step_plan.get("zarr_config"),
                "n_channels": n_channels,
                "n_z": n_z,
                "n_fields": n_fields,
                "row": row,
                "col": col,
            }
        )

    filemanager.save_batch(
        memory_data, materialized_paths, materialized_backend, **save_kwargs
    )


def get_all_image_paths(input_dir, backend, axis_id, filemanager, microscope_handler):
    """
    Get all image file paths for a specific well from a directory.

    Args:
        input_dir: Directory to search for images
        axis_id: Well identifier to filter files
        backend: Backend to use for file listing
        filemanager: FileManager instance
        microscope_handler: Microscope handler with parser for filename parsing

    Returns:
        List of full file paths for the well
    """
    # List all image files in directory
    all_image_files = filemanager.list_image_files(str(input_dir), backend)

    # Filter by well using parser (FIXED: was using naive string matching)
    axis_files = []
    parser = microscope_handler.parser

    for f in all_image_files:
        filename = os.path.basename(str(f))
        metadata = parser.parse_filename(filename)
        # Use dynamic multiprocessing axis instead of hardcoded 'well'
        from openhcs.constants import MULTIPROCESSING_AXIS

        axis_key = MULTIPROCESSING_AXIS.value
        if metadata and metadata.get(axis_key) == axis_id:
            axis_files.append(str(f))

    # Remove duplicates and sort
    sorted_files = sorted(list(set(axis_files)))

    # Prepare full file paths
    input_dir_path = Path(input_dir)
    full_file_paths = [str(input_dir_path / Path(f).name) for f in sorted_files]

    logger.debug(
        f"Found {len(all_image_files)} total files, {len(full_file_paths)} for axis {axis_id}"
    )

    return full_file_paths


def create_image_path_getter(axis_id, filemanager, microscope_handler):
    """
    Create a specialized image path getter function using runtime context.

    Args:
        axis_id: Well identifier
        filemanager: FileManager instance
        microscope_handler: Microscope handler with parser for filename parsing

    Returns:
        Function that takes (input_dir, backend) and returns image paths for the well
    """

    def get_paths_for_axis(input_dir, backend):
        return get_all_image_paths(
            input_dir=input_dir,
            axis_id=axis_id,
            backend=backend,
            filemanager=filemanager,
            microscope_handler=microscope_handler,
        )

    return get_paths_for_axis


# Environment variable to disable universal GPU defragmentation
DISABLE_GPU_DEFRAG = os.getenv("OPENHCS_DISABLE_GPU_DEFRAG", "false").lower() == "true"


def _bulk_preload_step_images(
    step_input_dir: Path,
    step_output_dir: Path,
    axis_id: str,
    read_backend: str,
    patterns_by_well: Dict[str, Any],
    filemanager: "FileManager",
    microscope_handler: "MicroscopeHandler",
    zarr_config: Optional[Dict[str, Any]] = None,
    patterns_to_preload: Optional[List[str]] = None,
    variable_components: Optional[List[str]] = None,
) -> None:
    """
    Pre-load images for this step from source backend into memory backend.

    This reduces I/O overhead by doing a single bulk read operation
    instead of loading images per pattern group.

    Args:
        patterns_to_preload: Optional list of specific patterns to preload (for sequential mode).
        variable_components: Required when patterns_to_preload is provided, for pattern expansion.

    Note: External conditional logic ensures this is only called for non-memory backends.
    """
    import time

    start_time = time.time()

    # Get file paths based on mode
    if patterns_to_preload is not None:
        # Sequential mode: expand patterns to files
        all_files = [
            f
            for p in patterns_to_preload
            for f in microscope_handler.path_list_from_pattern(
                str(step_input_dir), p, filemanager, read_backend, variable_components
            )
        ]
        # Ensure full paths (prepend directory if needed)
        full_file_paths = [
            str(step_input_dir / f) if not Path(f).is_absolute() else f
            for f in set(all_files)
        ]
    else:
        # Normal mode: get all files for well
        get_paths_for_axis = create_image_path_getter(
            axis_id, filemanager, microscope_handler
        )
        full_file_paths = get_paths_for_axis(step_input_dir, read_backend)

    if not full_file_paths:
        raise RuntimeError(
            f"🔄 BULK PRELOAD: No files found for well {axis_id} in {step_input_dir} with backend {read_backend}"
        )

    # Load from source backend with conditional zarr_config
    if read_backend == Backend.ZARR.value:
        raw_images = filemanager.load_batch(
            full_file_paths, read_backend, zarr_config=zarr_config
        )
    else:
        raw_images = filemanager.load_batch(full_file_paths, read_backend)

    # Ensure directory exists in memory backend before saving
    filemanager.ensure_directory(str(step_input_dir), Backend.MEMORY.value)

    # Save to memory backend using OUTPUT paths
    # memory_paths = [str(step_output_dir / Path(fp).name) for fp in full_file_paths]
    for file_path in full_file_paths:
        if filemanager.exists(file_path, Backend.MEMORY.value):
            filemanager.delete(file_path, Backend.MEMORY.value)

    filemanager.save_batch(raw_images, full_file_paths, Backend.MEMORY.value)

    # Clean up source references - keep only memory backend references
    del raw_images

    load_time = time.time() - start_time


def _bulk_writeout_step_images(
    step_output_dir: Path,
    write_backend: str,
    axis_id: str,
    zarr_config: Optional[Dict[str, Any]],
    filemanager: "FileManager",
    microscope_handler: Optional[Any] = None,
) -> None:
    """
    Write all processed images from memory to final backend (disk/zarr).

    This reduces I/O overhead by doing a single bulk write operation
    instead of writing images per pattern group.

    Note: External conditional logic ensures this is only called for non-memory backends.
    """
    import time

    start_time = time.time()

    # Create specialized path getter and get memory paths for this well
    get_paths_for_axis = create_image_path_getter(
        axis_id, filemanager, microscope_handler
    )
    memory_file_paths = get_paths_for_axis(step_output_dir, Backend.MEMORY.value)

    if not memory_file_paths:
        raise RuntimeError(
            f"🔄 BULK WRITEOUT: No image files found for well {axis_id} in memory directory {step_output_dir}"
        )

    # Convert relative memory paths back to absolute paths for target backend
    # Memory backend stores relative paths, but target backend needs absolute paths
    #    file_paths =
    #    for memory_path in memory_file_paths:
    #        # Get just the filename and construct proper target path
    #        filename = Path(memory_path).name
    #        target_path = step_output_dir / filename
    #        file_paths.append(str(target_path))

    file_paths = memory_file_paths

    # Load all data from memory backend
    memory_data = filemanager.load_batch(file_paths, Backend.MEMORY.value)

    # Ensure output directory exists before bulk write
    filemanager.ensure_directory(str(step_output_dir), Backend.DISK.value)

    # Bulk write to target backend with conditional zarr_config
    if write_backend == Backend.ZARR.value:
        # Calculate zarr dimensions from file paths
        if microscope_handler is not None:
            n_channels, n_z, n_fields = _calculate_zarr_dimensions(
                file_paths, microscope_handler
            )
            # Parse well to get row and column for zarr structure
            row, col = microscope_handler.parser.extract_component_coordinates(axis_id)
            filemanager.save_batch(
                memory_data,
                file_paths,
                write_backend,
                chunk_name=axis_id,
                zarr_config=zarr_config,
                n_channels=n_channels,
                n_z=n_z,
                n_fields=n_fields,
                row=row,
                col=col,
            )
        else:
            # Fallback without dimensions if microscope_handler not available
            filemanager.save_batch(
                memory_data,
                file_paths,
                write_backend,
                chunk_name=axis_id,
                zarr_config=zarr_config,
            )
    else:
        filemanager.save_batch(memory_data, file_paths, write_backend)

    write_time = time.time() - start_time


def _calculate_zarr_dimensions(
    file_paths: List[Union[str, Path]], microscope_handler
) -> tuple[int, int, int]:
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
    n_channels = len(
        set(f.get("channel") for f in parsed_files if f.get("channel") is not None)
    )
    n_z = len(
        set(f.get("z_index") for f in parsed_files if f.get("z_index") is not None)
    )
    n_fields = len(
        set(f.get("site") for f in parsed_files if f.get("site") is not None)
    )

    # Ensure at least 1 for each dimension (handle cases where metadata is missing)
    n_channels = max(1, n_channels)
    n_z = max(1, n_z)
    n_fields = max(1, n_fields)

    return n_channels, n_z, n_fields


def _is_3d(array: Any) -> bool:
    """Check if an array is 3D."""
    return hasattr(array, "ndim") and array.ndim == 3


def _execute_function_core(
    func_callable: Callable,
    main_data_arg: Any,
    base_kwargs: Dict[str, Any],
    context: "ProcessingContext",
    special_inputs_plan: Dict[str, str],  # {'arg_name_for_func': 'special_path_value'}
    special_outputs_plan: TypingOrderedDict[
        str, str
    ],  # {'output_key': 'special_path_value'}, order matters
    axis_id: str,  # Add axis_id parameter
    input_memory_type: str,
    device_id: int,
    source_paths: Sequence[str] = (),
) -> Any:  # Returns the main processed data stack
    """
    Executes a single callable, handling its special I/O.
    - Loads special inputs from VFS paths in `special_inputs_plan`.
    - Calls `func_callable(main_data_arg, **all_kwargs)`.
    - If `special_outputs_plan` is non-empty, expects func to return (main_out, sp_val1, sp_val2,...).
    - Saves special outputs positionally to VFS paths in `special_outputs_plan`.
    - Returns the main processed data stack.
    """
    final_kwargs = base_kwargs.copy()

    # Log dtype_config in kwargs
    if special_inputs_plan:
        logger.info(
            f"�� SPECIAL_INPUTS_DEBUG : special_inputs_plan = {special_inputs_plan}"
        )
        for arg_name, path_info in special_inputs_plan.items():
            # Extract path string from the path info dictionary
            # Current format: {"path": "/path/to/file.pkl", "source_step_id": "step_123"}
            if isinstance(path_info, dict) and "path" in path_info:
                special_path_value = path_info["path"]
            else:
                special_path_value = path_info  # Fallback if it's already a string

            logger.info(
                f"Loading special input '{arg_name}' from path '{special_path_value}' (memory backend)"
            )
            try:
                final_kwargs[arg_name] = context.filemanager.load(
                    special_path_value, Backend.MEMORY.value
                )
            except Exception as e:
                logger.error(
                    f"Failed to load special input '{arg_name}' from '{special_path_value}': {e}",
                    exc_info=True,
                )
                raise

    # Auto-inject context if function signature expects it
    import inspect

    sig = inspect.signature(func_callable)
    if "context" in sig.parameters:
        final_kwargs["context"] = context

    # 🔍 DEBUG: Log input dimensions
    input_shape = getattr(main_data_arg, "shape", "no shape attr")
    input_type = type(main_data_arg).__name__

    # ⚡ INFO: Terse function execution log for user feedback
    logger.info(f"⚡ Executing: {func_callable.__name__}")

    # 🔍 DEBUG: Log function attributes before execution

    raw_function_output = func_callable(main_data_arg, **final_kwargs)

    # Check if function returned a tuple (indicates @special_outputs decorator)
    # Functions with @special_outputs ALWAYS return tuples: (main_output, special_output_1, ...)
    # We ALWAYS extract the main output from the tuple, regardless of funcplan
    # The funcplan only controls which special outputs to SAVE, not which to extract
    if isinstance(raw_function_output, tuple):
        # Function returned a tuple - extract main output (first element)
        main_output_data = raw_function_output[0]
        returned_special_values_tuple = raw_function_output[1:]

        # Only SAVE special outputs if they're in the funcplan
        # (funcplan controls what to save, not what to extract)
        if special_outputs_plan:
            # Iterate through special_outputs_plan (which must be ordered by compiler)
            # and match with positionally returned special values.
            for i, (output_key, vfs_path_info) in enumerate(
                special_outputs_plan.items()
            ):
                logger.info(
                    f"Saving special output '{output_key}' to VFS path '{vfs_path_info}' (memory backend)"
                )
                if i < len(returned_special_values_tuple):
                    value_to_save = returned_special_values_tuple[i]
                    from openhcs.processing.materialization.payloads import (
                        MaterializationPayload,
                    )

                    if isinstance(value_to_save, MaterializationPayload):
                        value_to_save = value_to_save.bind_source_paths(source_paths)
                    # Extract path string from the path info dictionary
                    # Current format: {"path": "/path/to/file.pkl"}
                    if isinstance(vfs_path_info, dict) and "path" in vfs_path_info:
                        vfs_path = vfs_path_info["path"]
                    else:
                        vfs_path = vfs_path_info  # Fallback if it's already a string

                    # DEBUG: List what's currently in VFS before saving
                    from polystore.base import (
                        storage_registry as global_storage_registry,
                    )

                    global_memory_backend = global_storage_registry[
                        Backend.MEMORY.value
                    ]
                    global_existing_keys = list(
                        global_memory_backend._memory_store.keys()
                    )

                    # Check filemanager's memory backend
                    filemanager_memory_backend = context.filemanager._get_backend(
                        Backend.MEMORY.value
                    )
                    filemanager_existing_keys = list(
                        filemanager_memory_backend._memory_store.keys()
                    )

                    if vfs_path in filemanager_existing_keys:
                        logger.warning(
                            f"🔍 VFS_DEBUG: WARNING - '{vfs_path}' ALREADY EXISTS in FILEMANAGER memory backend!"
                        )
                        existing_value = context.filemanager.load(
                            vfs_path, Backend.MEMORY.value
                        )
                        if isinstance(existing_value, MaterializationPayload):
                            value_to_save = existing_value.merge(value_to_save)
                        elif isinstance(existing_value, list) and isinstance(
                            value_to_save, list
                        ):
                            value_to_save = existing_value + value_to_save
                        else:
                            raise TypeError(
                                f"Special output '{output_key}' was emitted more than once "
                                f"with non-mergeable values {type(existing_value).__name__} "
                                f"and {type(value_to_save).__name__}"
                            )
                        context.filemanager.delete(vfs_path, Backend.MEMORY.value)

                    # Ensure directory exists for memory backend
                    parent_dir = str(Path(vfs_path).parent)
                    context.filemanager.ensure_directory(
                        parent_dir, Backend.MEMORY.value
                    )
                    context.filemanager.save(
                        value_to_save, vfs_path, Backend.MEMORY.value
                    )
                else:
                    # This indicates a mismatch that should ideally be caught by schema/validation
                    logger.error(
                        f"Mismatch: special_outputs_plan wants to save '{output_key}', but function only returned {len(returned_special_values_tuple)} special values."
                    )
                    raise ValueError(
                        f"Function did not return enough values for all planned special outputs. Missing value for '{output_key}'."
                    )
    else:
        # Function did not return a tuple - use output directly
        main_output_data = raw_function_output

    return main_output_data


def _execute_chain_core(
    initial_data_stack: Any,
    func_chain: List[Union[Callable, Tuple[Callable, Dict]]],
    context: "ProcessingContext",
    step_special_inputs_plan: Dict[str, str],
    step_special_outputs_plan: TypingOrderedDict[str, str],
    axis_id: str,  # Add axis_id parameter
    device_id: int,
    input_memory_type: str,
    step_index: int,  # Add step_index for funcplan lookup
    source_paths: Sequence[str],
    dict_key: str = "default",  # Add dict_key for funcplan lookup
) -> Any:
    current_stack = initial_data_stack
    current_memory_type = input_memory_type  # Track memory type from frozen context

    for i, func_item in enumerate(func_chain):
        actual_callable: Callable
        base_kwargs_for_item: Dict[str, Any] = {}
        is_last_in_chain = i == len(func_chain) - 1

        # Resolve FunctionReference objects to actual functions in worker process
        from openhcs.core.pipeline.compiler import FunctionReference

        if isinstance(func_item, FunctionReference):
            actual_callable = func_item.resolve()
        elif isinstance(func_item, tuple) and len(func_item) == 2:
            func_or_ref, kwargs = func_item
            if isinstance(func_or_ref, FunctionReference):
                actual_callable = func_or_ref.resolve()
            elif callable(func_or_ref):
                actual_callable = func_or_ref
            else:
                raise TypeError(f"Invalid function in tuple: {func_or_ref}")
            base_kwargs_for_item = kwargs
            # Strip UI-only metadata keys (never passed to runtime callables)
            if (
                isinstance(base_kwargs_for_item, dict)
                and "__pyqt_reactive_scope_token__" in base_kwargs_for_item
            ):
                base_kwargs_for_item = {
                    k: v
                    for k, v in base_kwargs_for_item.items()
                    if k != "__pyqt_reactive_scope_token__"
                }
        elif callable(func_item):
            actual_callable = func_item
        else:
            raise TypeError(f"Invalid item in function chain: {func_item}.")

        # Convert to function's input memory type (noop if same)
        from openhcs.core.memory import convert_memory

        current_stack = convert_memory(
            data=current_stack,
            source_type=current_memory_type,
            target_type=actual_callable.input_memory_type,
            gpu_id=device_id,
        )

        # Use funcplan to determine which outputs this function should save
        funcplan = context.step_plans[step_index].get("funcplan", {})
        func_name = getattr(actual_callable, "__name__", "unknown")

        # Construct execution key: function_name_dict_key_chain_position
        execution_key = f"{func_name}_{dict_key}_{i}"

        if execution_key in funcplan:
            outputs_to_save = funcplan[execution_key]
            outputs_plan_for_this_call = _filter_special_outputs_for_function(
                outputs_to_save, step_special_outputs_plan
            )
        else:
            # Fallback: no funcplan entry, save nothing
            outputs_plan_for_this_call = {}

        current_stack = _execute_function_core(
            func_callable=actual_callable,
            main_data_arg=current_stack,
            base_kwargs=base_kwargs_for_item,
            context=context,
            special_inputs_plan=step_special_inputs_plan,
            special_outputs_plan=outputs_plan_for_this_call,
            axis_id=axis_id,
            device_id=device_id,
            input_memory_type=input_memory_type,
            source_paths=source_paths,
        )

        # Update current memory type from frozen context
        current_memory_type = actual_callable.output_memory_type

    return current_stack


def _process_single_pattern_group(
    context: "ProcessingContext",
    pattern_group_info: Any,
    executable_func_or_chain: Any,
    base_func_args: Dict[str, Any],
    step_input_dir: Path,
    step_output_dir: Path,
    axis_id: str,
    component_value: str,
    read_backend: str,
    write_backend: str,
    input_memory_type_from_plan: str,  # Explicitly from plan
    output_memory_type_from_plan: str,  # Explicitly from plan
    device_id: Optional[int],
    same_directory: bool,
    special_inputs_map: Dict[str, str],
    special_outputs_map: TypingOrderedDict[str, str],
    zarr_config: Optional[Dict[str, Any]],
    variable_components: Optional[List[str]] = None,
    step_index: Optional[int] = None,  # Add step_index for funcplan lookup
) -> None:
    start_time = time.time()
    pattern_repr = str(pattern_group_info)[:100]
    logger.debug(f"🔥 PATTERN: Processing {pattern_repr} for well {axis_id}")

    try:
        if not context.microscope_handler:
            raise RuntimeError("MicroscopeHandler not available in context.")

        matching_files = context.microscope_handler.path_list_from_pattern(
            str(step_input_dir),
            pattern_group_info,
            context.filemanager,
            Backend.MEMORY.value,
            [vc.value for vc in variable_components] if variable_components else None,
        )

        if not matching_files:
            raise ValueError(
                f"No matching files found for pattern group {pattern_repr} in {step_input_dir}. "
                f"This indicates either: (1) no image files exist in the directory, "
                f"(2) files don't match the pattern, or (3) pattern parsing failed. "
                f"Check that input files exist and match the expected naming convention."
            )

        logger.debug(
            f"🔥 PATTERN: Found {len(matching_files)} files: {[Path(f).name for f in matching_files]}"
        )

        # Sort files to ensure consistent ordering (especially important for z-stacks)
        matching_files.sort()
        logger.debug(
            f"🔥 PATTERN: Sorted files: {[Path(f).name for f in matching_files]}"
        )

        full_file_paths = [str(step_input_dir / f) for f in matching_files]
        aligned_source_paths = [str(step_output_dir / f) for f in matching_files]
        raw_slices = context.filemanager.load_batch(
            full_file_paths, Backend.MEMORY.value
        )

        if not raw_slices:
            raise ValueError(
                f"No valid images loaded for pattern group {pattern_repr} in {step_input_dir}. "
                f"Found {len(matching_files)} matching files but failed to load any valid images. "
                f"This indicates corrupted image files, unsupported formats, or I/O errors. "
                f"Check file integrity and format compatibility."
            )

        # 🔍 DEBUG: Log stacking operation
        if raw_slices:
            slice_shapes = [
                getattr(s, "shape", "no shape") for s in raw_slices[:3]
            ]  # First 3 shapes

        main_data_stack = stack_slices(
            slices=raw_slices, memory_type=input_memory_type_from_plan, gpu_id=device_id
        )

        # 🔍 DEBUG: Log stacked result
        stack_shape = getattr(main_data_stack, "shape", "no shape")
        stack_type = type(main_data_stack).__name__

        final_base_kwargs = base_func_args.copy()

        component_key = None if component_value is None else str(component_value)
        # Get step function from step plan
        step_func = context.step_plans[step_index]["func"]

        # DEBUG: Log component_key and available groups
        logger.info(
            f"🔍 COMPONENT_KEY: component_value={component_value}, component_key={component_key}"
        )
        special_outputs_by_group = context.step_plans[step_index].get(
            "special_outputs_by_group"
        )
        if special_outputs_by_group:
            logger.info(f"🔍 AVAILABLE_GROUPS: {list(special_outputs_by_group.keys())}")
        else:
            logger.info(f"🔍 NO special_outputs_by_group in step plan")

        if isinstance(step_func, dict):
            dict_key_for_funcplan = (
                component_key  # Use actual dict key for dict patterns
            )
        else:
            dict_key_for_funcplan = "default"  # Use default for list/single patterns
        special_inputs_for_component = _select_special_plan_for_component(
            context.step_plans[step_index].get("special_inputs_by_group"),
            component_key,
            special_inputs_map,
        )
        special_outputs_for_component = _select_special_plan_for_component(
            context.step_plans[step_index].get("special_outputs_by_group"),
            component_key,
            special_outputs_map,
        )

        # DEBUG: Log selected special outputs
        logger.info(f"🔍 SELECTED_OUTPUTS: {special_outputs_for_component}")

        # Resolve FunctionReference if needed
        from openhcs.core.pipeline.compiler import FunctionReference

        if isinstance(executable_func_or_chain, FunctionReference):
            executable_func_or_chain = executable_func_or_chain.resolve()
        elif (
            isinstance(executable_func_or_chain, tuple)
            and len(executable_func_or_chain) == 2
        ):
            func_or_ref, kwargs = executable_func_or_chain
            if isinstance(func_or_ref, FunctionReference):
                executable_func_or_chain = (func_or_ref.resolve(), kwargs)

        if isinstance(executable_func_or_chain, list):
            processed_stack = _execute_chain_core(
                main_data_stack,
                executable_func_or_chain,
                context,
                special_inputs_for_component,
                special_outputs_for_component,
                axis_id,
                device_id,
                input_memory_type_from_plan,
                step_index,
                aligned_source_paths,
                dict_key_for_funcplan,
            )
        elif callable(executable_func_or_chain) or (
            isinstance(executable_func_or_chain, tuple)
            and len(executable_func_or_chain) == 2
        ):
            # Handle both direct callable and (callable, kwargs) tuple
            if isinstance(executable_func_or_chain, tuple):
                actual_func, _ = executable_func_or_chain
            else:
                actual_func = executable_func_or_chain

            # For single functions, apply funcplan filtering like in chain execution
            funcplan = context.step_plans[step_index].get("funcplan", {})
            func_name = getattr(actual_func, "__name__", "unknown")
            execution_key = f"{func_name}_{dict_key_for_funcplan}_0"  # Position 0 for single functions

            if execution_key in funcplan:
                outputs_to_save = funcplan[execution_key]
                filtered_special_outputs_map = _filter_special_outputs_for_function(
                    outputs_to_save, special_outputs_for_component
                )
            else:
                # Fallback: no funcplan entry, save nothing
                filtered_special_outputs_map = {}

            processed_stack = _execute_function_core(
                executable_func_or_chain,
                main_data_stack,
                final_base_kwargs,
                context,
                special_inputs_for_component,
                filtered_special_outputs_map,
                axis_id,
                input_memory_type_from_plan,
                device_id,
                aligned_source_paths,
            )
        else:
            raise TypeError(
                f"Invalid executable_func_or_chain: {type(executable_func_or_chain)}"
            )

        # 🔍 DEBUG: Check what shape the function actually returned
        input_shape = getattr(main_data_stack, "shape", "unknown")
        output_shape = getattr(processed_stack, "shape", "unknown")
        processed_type = type(processed_stack).__name__

        # 🔍 DEBUG: Additional validation logging

        if not _is_3d(processed_stack):
            logger.error("🔍 VALIDATION ERROR: processed_stack is not 3D")
            logger.error(f"🔍 VALIDATION ERROR: Type: {type(processed_stack)}")
            logger.error(
                f"🔍 VALIDATION ERROR: Shape: {getattr(processed_stack, 'shape', 'no shape attr')}"
            )
            logger.error(
                f"🔍 VALIDATION ERROR: Has ndim: {hasattr(processed_stack, 'ndim')}"
            )
            if hasattr(processed_stack, "ndim"):
                logger.error(f"🔍 VALIDATION ERROR: ndim value: {processed_stack.ndim}")
            raise ValueError(
                f"Main processing must result in a 3D array, got {getattr(processed_stack, 'shape', 'unknown')}"
            )

        # 🔍 DEBUG: Log unstacking operation

        output_slices = unstack_slices(
            array=processed_stack,
            memory_type=output_memory_type_from_plan,
            gpu_id=device_id,
            validate_slices=True,
        )

        # 🔍 DEBUG: Log unstacked result
        if output_slices:
            unstacked_shapes = [
                getattr(s, "shape", "no shape") for s in output_slices[:3]
            ]  # First 3 shapes
            # Log values of first slice
            if len(output_slices) > 0 and hasattr(output_slices[0], "min"):
                first_slice = output_slices[0]

        # Handle cases where function returns fewer images than inputs (e.g., z-stack flattening, channel compositing)
        # In such cases, we save only the returned images using the first N input filenames
        num_outputs = len(output_slices)
        num_inputs = len(matching_files)

        if num_outputs < num_inputs:
            logger.debug(
                f"Function returned {num_outputs} images from {num_inputs} inputs - likely flattening operation"
            )
        elif num_outputs > num_inputs:
            logger.warning(
                f"Function returned more images ({num_outputs}) than inputs ({num_inputs}) - unexpected"
            )

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
            context.filemanager.ensure_directory(
                str(step_output_dir), Backend.MEMORY.value
            )

            # Batch save
            context.filemanager.save_batch(
                output_data, output_paths_batch, Backend.MEMORY.value
            )

        except Exception as e:
            logger.error(
                f"Error saving batch of output slices for pattern {pattern_repr}: {e}",
                exc_info=True,
            )

        # 🔥 CLEANUP: If function returned fewer images than inputs, delete the unused input files
        # This prevents unused channel files from remaining in memory after compositing
        if num_outputs < num_inputs:
            for j in range(num_outputs, num_inputs):
                unused_input_filename = matching_files[j]
                unused_input_path = Path(step_input_dir) / unused_input_filename
                if context.filemanager.exists(
                    str(unused_input_path), Backend.MEMORY.value
                ):
                    context.filemanager.delete(
                        str(unused_input_path), Backend.MEMORY.value
                    )
                    logger.debug(
                        f"🔥 CLEANUP: Deleted unused input file: {unused_input_filename}"
                    )

        logger.debug(
            f"Finished pattern group {pattern_repr} in {(time.time() - start_time):.2f}s."
        )
    except Exception as e:
        import traceback

        full_traceback = traceback.format_exc()
        logger.error(
            f"Error processing pattern group {pattern_repr}: {e}", exc_info=True
        )
        logger.error(
            f"Full traceback for pattern group {pattern_repr}:\n{full_traceback}"
        )
        raise ValueError(f"Failed to process pattern group {pattern_repr}: {e}") from e


class FunctionStep(AbstractStep):
    # Fields with dedicated editors - hidden from regular ParameterFormManager
    # but included in ObjectState for tracking and preview
    _ui_special_fields = ("func",)

    def __init__(
        self,
        func: Union[
            Callable,
            Tuple[Callable, Dict],
            List[Union[Callable, Tuple[Callable, Dict]]],
        ] = [],
        **kwargs,
    ):
        # Generate default name from function if not provided
        if "name" not in kwargs or kwargs["name"] is None:
            actual_func_for_name = func
            if isinstance(func, tuple):
                actual_func_for_name = func[0]
            elif isinstance(func, list) and func:
                first_item = func[0]
                if isinstance(first_item, tuple):
                    actual_func_for_name = first_item[0]
                elif callable(first_item):
                    actual_func_for_name = first_item
            kwargs["name"] = getattr(actual_func_for_name, "__name__", "FunctionStep")

        super().__init__(**kwargs)
        self.func = func  # This is used by prepare_patterns_and_functions at runtime

    def process(self, context: "ProcessingContext", step_index: int) -> None:
        # Access step plan by index (step_plans keyed by index, not step_id)
        step_plan = context.step_plans[step_index]

        # Get step name for logging
        step_name = step_plan["step_name"]

        try:
            axis_id = step_plan["axis_id"]
            step_input_dir = Path(step_plan["input_dir"])
            step_output_dir = Path(step_plan["output_dir"])
            variable_components = step_plan["variable_components"]
            group_by = step_plan["group_by"]
            func_from_plan = step_plan["func"]

            # special_inputs/outputs are dicts: {'key': 'vfs_path_value'}
            special_inputs = step_plan["special_inputs"]
            special_outputs = step_plan[
                "special_outputs"
            ]  # Should be OrderedDict if order matters

            read_backend = step_plan["read_backend"]
            write_backend = step_plan["write_backend"]
            input_mem_type = step_plan["input_memory_type"]
            output_mem_type = step_plan["output_memory_type"]
            microscope_handler = context.microscope_handler
            filemanager = context.filemanager

            # Create path getter for this well
            get_paths_for_axis = create_image_path_getter(
                axis_id, filemanager, microscope_handler
            )

            # Store path getter in step_plan for streaming access
            step_plan["get_paths_for_axis"] = get_paths_for_axis

            # Get patterns first for bulk preload
            # Use dynamic filter parameter based on current multiprocessing axis
            from openhcs.constants import MULTIPROCESSING_AXIS

            axis_name = MULTIPROCESSING_AXIS.value
            filter_kwargs = {f"{axis_name}_filter": [axis_id]}

            patterns_by_well = microscope_handler.auto_detect_patterns(
                str(step_input_dir),  # folder_path
                filemanager,  # filemanager
                read_backend,  # backend
                extensions=DEFAULT_IMAGE_EXTENSIONS,  # extensions
                group_by=group_by,  # Pass GroupBy enum directly
                variable_components=(
                    [vc.value for vc in variable_components]
                    if variable_components
                    else []
                ),  # variable_components for placeholder logic
                **filter_kwargs,  # Dynamic filter parameter
            )

            # Debug: Log discovered patterns
            if axis_id not in patterns_by_well:
                logger.warning(
                    f"🔍 PATTERN DISCOVERY: No patterns found for well {axis_id}!"
                )

            # Only access gpu_id if the step requires GPU (has GPU memory types)
            from openhcs.constants.constants import VALID_GPU_MEMORY_TYPES

            requires_gpu = (
                input_mem_type in VALID_GPU_MEMORY_TYPES
                or output_mem_type in VALID_GPU_MEMORY_TYPES
            )

            # Ensure variable_components is never None - use default if missing
            if variable_components is None:
                variable_components = [VariableComponents.SITE]  # Default fallback
                logger.warning(
                    f"Step {step_index} ({step_name}) had None variable_components, using default [SITE]"
                )
            if requires_gpu:
                device_id = step_plan["gpu_id"]
                logger.debug(
                    f"🔥 DEBUG: Step {step_index} gpu_id from plan: {device_id}, input_mem: {input_mem_type}, output_mem: {output_mem_type}"
                )
            else:
                device_id = None  # CPU-only step
                logger.debug(
                    f"🔥 DEBUG: Step {step_index} is CPU-only, input_mem: {input_mem_type}, output_mem: {output_mem_type}"
                )

            logger.debug(
                f"🔥 DEBUG: Step {step_index} read_backend: {read_backend}, write_backend: {write_backend}"
            )

            if not all([axis_id, step_input_dir, step_output_dir]):
                raise ValueError(f"Plan missing essential keys for step {step_index}")

            same_dir = str(step_input_dir) == str(step_output_dir)
            logger.info(
                f"Step {step_index} ({step_name}) I/O: read='{read_backend}', write='{write_backend}'."
            )
            logger.info(
                f"Step {step_index} ({step_name}) Paths: input_dir='{step_input_dir}', output_dir='{step_output_dir}', same_dir={same_dir}"
            )

            # Import psutil for memory logging
            import psutil
            import os

            # 🔄 INPUT CONVERSION: Convert loaded input data to zarr if configured
            if "input_conversion_dir" in step_plan:
                input_conversion_dir = step_plan["input_conversion_dir"]
                input_conversion_backend = step_plan["input_conversion_backend"]

                logger.info(f"Converting input data to zarr: {input_conversion_dir}")

                # Get paths from input data using the original read backend (e.g., disk)
                # NOT from memory - the data hasn't been converted yet!
                source_paths = get_paths_for_axis(step_input_dir, read_backend)
                memory_data = filemanager.load_batch(source_paths, read_backend)

                # Generate conversion paths (input_dir → conversion_dir)
                conversion_paths = _generate_materialized_paths(
                    source_paths, Path(step_input_dir), Path(input_conversion_dir)
                )

                # Parse actual filenames to determine dimensions
                # Calculate zarr dimensions from conversion paths (which contain the filenames)
                n_channels, n_z, n_fields = _calculate_zarr_dimensions(
                    conversion_paths, context.microscope_handler
                )
                # Parse well to get row and column for zarr structure
                row, col = (
                    context.microscope_handler.parser.extract_component_coordinates(
                        axis_id
                    )
                )

                # Save using existing materialized data infrastructure
                _save_materialized_data(
                    filemanager,
                    memory_data,
                    conversion_paths,
                    input_conversion_backend,
                    step_plan,
                    context,
                    axis_id,
                )

                logger.info(
                    f"🔬 Converted {len(conversion_paths)} input files to {input_conversion_dir}"
                )

                # Update metadata after conversion
                conversion_dir = Path(step_plan["input_conversion_dir"])
                zarr_subdir = (
                    conversion_dir.name
                    if step_plan["input_conversion_uses_virtual_workspace"]
                    else None
                )
                _update_metadata_for_zarr_conversion(
                    conversion_dir.parent,
                    step_plan["input_conversion_original_subdir"],
                    zarr_subdir,
                    context,
                )

            logger.info(
                f"🔥 STEP: Starting processing for '{step_name}' well {axis_id} (group_by={group_by.name if group_by else None}, variable_components={[vc.name for vc in variable_components] if variable_components else []})"
            )

            if axis_id not in patterns_by_well:
                raise ValueError(
                    f"No patterns detected for well '{axis_id}' in step '{step_name}' (index: {step_index}). "
                    f"This indicates either: (1) no image files found for this well, "
                    f"(2) image files don't match the expected naming pattern, or "
                    f"(3) pattern detection failed. Check input directory: {step_input_dir}"
                )

            if isinstance(patterns_by_well[axis_id], dict):
                # Grouped patterns (when group_by is set)
                for comp_val, pattern_list in patterns_by_well[axis_id].items():
                    logger.debug(
                        f"🔥 STEP: Component '{comp_val}' has {len(pattern_list)} patterns: {pattern_list}"
                    )
            else:
                # Ungrouped patterns (when group_by is None)
                logger.debug(
                    f"🔥 STEP: Found {len(patterns_by_well[axis_id])} ungrouped patterns: {patterns_by_well[axis_id]}"
                )

            if func_from_plan is None:
                raise ValueError(
                    f"Step plan missing 'func' for step: {step_plan.get('step_name', 'Unknown')} (index: {step_index})"
                )

            # 🔄 SEQUENTIAL PROCESSING: Filter patterns BEFORE grouping by group_by component
            # This ensures sequential filtering works independently of group_by
            if context.current_sequential_combination:
                seq_config = context.global_config.sequential_processing_config
                seq_component = seq_config.sequential_components[0].value
                target_value = context.current_sequential_combination[0]

                # Filter patterns by sequential component
                patterns_by_well[axis_id] = _filter_patterns_by_component(
                    patterns_by_well[axis_id],
                    seq_component,
                    target_value,
                    microscope_handler,
                )

                filtered_count = (
                    len(patterns_by_well[axis_id])
                    if isinstance(patterns_by_well[axis_id], list)
                    else sum(len(v) for v in patterns_by_well[axis_id].values())
                )

            # Now group patterns by group_by component (if set)
            grouped_patterns, comp_to_funcs, comp_to_base_args = (
                prepare_patterns_and_functions(
                    patterns_by_well[axis_id],
                    func_from_plan,
                    component=group_by.value if group_by else None,
                )
            )

            total_steps = len(context.step_plans)
            total_groups = 0
            for pattern_list in grouped_patterns.values():
                total_groups += len(pattern_list)
            completed_groups = 0
            if total_groups == 0:
                raise ValueError(
                    f"No pattern groups found for step {step_index} ({step_name}) in well {axis_id}"
                )

            # Sequential filtering now happens BEFORE prepare_patterns_and_functions() above
            # This ensures it works correctly when sequential_components != group_by

            # Non-sequential processing: process all patterns for all component values
            process = psutil.Process(os.getpid())

            # Preload files ONCE for all filtered patterns (before processing loop)
            if read_backend != Backend.MEMORY.value:
                mem_before_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"📊 MEMORY: Before preload: {mem_before_mb:.1f} MB RSS")

                # If sequential mode, only preload filtered patterns
                if context.current_sequential_combination:
                    # Collect all patterns from filtered grouped_patterns
                    patterns_to_preload = []
                    for comp_val, pattern_list in grouped_patterns.items():
                        patterns_to_preload.extend(pattern_list)

                    logger.info(
                        f"� SEQUENTIAL: Preloading {len(patterns_to_preload)} filtered patterns"
                    )
                    _bulk_preload_step_images(
                        step_input_dir,
                        step_output_dir,
                        axis_id,
                        read_backend,
                        patterns_by_well,
                        filemanager,
                        microscope_handler,
                        step_plan["zarr_config"],
                        patterns_to_preload=patterns_to_preload,
                        variable_components=(
                            [vc.value for vc in variable_components]
                            if variable_components
                            else []
                        ),
                    )
                else:
                    # Non-sequential: preload all patterns
                    _bulk_preload_step_images(
                        step_input_dir,
                        step_output_dir,
                        axis_id,
                        read_backend,
                        patterns_by_well,
                        filemanager,
                        microscope_handler,
                        step_plan["zarr_config"],
                    )

                mem_after_mb = process.memory_info().rss / 1024 / 1024
                logger.info(
                    f"📊 MEMORY: After preload: {mem_after_mb:.1f} MB RSS (+{mem_after_mb - mem_before_mb:.1f} MB)"
                )

            # Process each component value
            for comp_val, current_pattern_list in grouped_patterns.items():
                exec_func_or_chain = comp_to_funcs[comp_val]
                base_kwargs = comp_to_base_args[comp_val]

                # Process all patterns for this component value
                for pattern_item in current_pattern_list:
                    _process_single_pattern_group(
                        context,
                        pattern_item,
                        exec_func_or_chain,
                        base_kwargs,
                        step_input_dir,
                        step_output_dir,
                        axis_id,
                        comp_val,
                        read_backend,
                        write_backend,
                        input_mem_type,
                        output_mem_type,
                        device_id,
                        same_dir,
                        special_inputs,
                        special_outputs,
                        step_plan["zarr_config"],
                        variable_components,
                        step_index,
                    )
                    completed_groups += 1
                    emit(
                        execution_id=context.execution_id,
                        plate_id=context.plate_id,
                        axis_id=axis_id,
                        step_name=step_name,
                        phase=ProgressPhase.PATTERN_GROUP,
                        status=ProgressStatus.RUNNING,
                        completed=completed_groups,
                        total=total_groups,
                        percent=(completed_groups / total_groups) * 100.0,
                        component=str(comp_val),
                        pattern=str(pattern_item),
                        worker_slot=context.worker_slot,
                        owned_wells=context.owned_wells,
                    )

            logger.info(
                f"🔥 STEP: Completed processing for '{step_name}' well {axis_id}."
            )

            # 📄 MATERIALIZATION WRITE: Only if not writing to memory
            if write_backend != Backend.MEMORY.value:
                memory_paths = get_paths_for_axis(step_output_dir, Backend.MEMORY.value)
                memory_data = filemanager.load_batch(memory_paths, Backend.MEMORY.value)
                # Calculate zarr dimensions (ignored by non-zarr backends)
                n_channels, n_z, n_fields = _calculate_zarr_dimensions(
                    memory_paths, context.microscope_handler
                )
                row, col = (
                    context.microscope_handler.parser.extract_component_coordinates(
                        axis_id
                    )
                )
                filemanager.ensure_directory(step_output_dir, write_backend)

                # Build save kwargs with parser metadata for all backends
                save_kwargs = {
                    "chunk_name": axis_id,
                    "zarr_config": step_plan["zarr_config"],
                    "n_channels": n_channels,
                    "n_z": n_z,
                    "n_fields": n_fields,
                    "row": row,
                    "col": col,
                    "parser_name": context.microscope_handler.parser.__class__.__name__,
                    "microscope_type": context.microscope_handler.microscope_type,
                }

                filemanager.save_batch(
                    memory_data, memory_paths, write_backend, **save_kwargs
                )

            # 📄 PER-STEP MATERIALIZATION: Additional materialized output if configured
            if "materialized_output_dir" in step_plan:
                materialized_output_dir = step_plan["materialized_output_dir"]
                materialized_backend = step_plan["materialized_backend"]

                memory_paths = get_paths_for_axis(step_output_dir, Backend.MEMORY.value)
                memory_data = filemanager.load_batch(memory_paths, Backend.MEMORY.value)
                materialized_paths = _generate_materialized_paths(
                    memory_paths, step_output_dir, Path(materialized_output_dir)
                )

                filemanager.ensure_directory(
                    materialized_output_dir, materialized_backend
                )
                _save_materialized_data(
                    filemanager,
                    memory_data,
                    materialized_paths,
                    materialized_backend,
                    step_plan,
                    context,
                    axis_id,
                )

                logger.info(
                    f"🔬 Materialized {len(materialized_paths)} files to {materialized_output_dir}"
                )

            # 📄 STREAMING: Execute all configured streaming backends
            from openhcs.core.config import StreamingConfig

            streaming_configs_found = []
            for key, config_instance in step_plan.items():
                if isinstance(config_instance, StreamingConfig):
                    streaming_configs_found.append((key, config_instance))

            for key, config_instance in streaming_configs_found:
                # Get paths at runtime like materialization does
                step_output_dir = step_plan["output_dir"]
                get_paths_for_axis = step_plan[
                    "get_paths_for_axis"
                ]  # Get the path getter from step_plan

                # Get memory paths (where data actually is)
                memory_paths = get_paths_for_axis(step_output_dir, Backend.MEMORY.value)

                # For materialized steps, use materialized paths for streaming (for correct source extraction)
                # but load from memory paths (where data actually is)
                if "materialized_output_dir" in step_plan:
                    materialized_output_dir = step_plan["materialized_output_dir"]
                    streaming_paths = _generate_materialized_paths(
                        memory_paths, step_output_dir, Path(materialized_output_dir)
                    )
                else:
                    streaming_paths = memory_paths

                # Load from memory (where data actually is)
                streaming_data = filemanager.load_batch(
                    memory_paths, Backend.MEMORY.value
                )
                kwargs = config_instance.get_streaming_kwargs(
                    context
                )  # Pass context for microscope handler access

                # Add pre-built source value for layer/window naming
                # During pipeline execution: source = step_name
                kwargs["source"] = step_name

                # Execute streaming - use streaming_paths (materialized paths) for metadata extraction
                filemanager.save_batch(
                    streaming_data,
                    streaming_paths,
                    config_instance.backend.value,
                    **kwargs,
                )

                # Add small delay between image and ROI streaming to prevent race conditions
                import time

                time.sleep(0.1)

            logger.info(
                f"FunctionStep {step_index} ({step_name}) completed for well {axis_id}."
            )

            # 📄 OPENHCS METADATA: Create metadata file automatically after step completion
            # Track which backend was actually used for writing files
            actual_write_backend = step_plan["write_backend"]

            # Only create OpenHCS metadata for disk/zarr backends, not OMERO
            # OMERO has its own metadata system and doesn't use openhcs_metadata.json
            if actual_write_backend not in [
                Backend.OMERO_LOCAL.value,
                Backend.MEMORY.value,
            ]:
                from openhcs.microscopes.openhcs import OpenHCSMetadataGenerator

                metadata_generator = OpenHCSMetadataGenerator(context.filemanager)

                # Main step output metadata
                is_pipeline_output = actual_write_backend != Backend.MEMORY.value
                metadata_generator.create_metadata(
                    context,
                    step_plan["output_dir"],
                    actual_write_backend,
                    is_main=is_pipeline_output,
                    plate_root=step_plan["output_plate_root"],
                    sub_dir=step_plan["sub_dir"],
                    results_dir=step_plan.get(
                        "analysis_results_dir"
                    ),  # Pass pre-calculated results directory
                )

            # 📄 MATERIALIZED METADATA: Create metadata for materialized directory if it exists
            # This must be OUTSIDE the main write_backend check because materializations
            # can happen even when the main step writes to memory
            if "materialized_output_dir" in step_plan:
                materialized_backend = step_plan["materialized_backend"]
                # Only create metadata if materialized backend is also disk/zarr
                if materialized_backend not in [
                    Backend.OMERO_LOCAL.value,
                    Backend.MEMORY.value,
                ]:
                    from openhcs.microscopes.openhcs import OpenHCSMetadataGenerator

                    metadata_generator = OpenHCSMetadataGenerator(context.filemanager)
                    metadata_generator.create_metadata(
                        context,
                        step_plan["materialized_output_dir"],
                        materialized_backend,
                        is_main=False,
                        plate_root=step_plan["materialized_plate_root"],
                        sub_dir=step_plan["materialized_sub_dir"],
                        results_dir=step_plan.get(
                            "materialized_analysis_results_dir"
                        ),  # Pass pre-calculated materialized results directory
                    )

            #  SPECIAL DATA MATERIALIZATION
            special_outputs = step_plan.get("special_outputs", {})
            if special_outputs:
                logger.info(
                    f"🔬 MATERIALIZATION: Starting materialization for {len(special_outputs)} special outputs"
                )
                # Special outputs ALWAYS use the main materialization backend (disk/zarr),
                # not the step's write backend (which may be memory for intermediate steps).
                # This ensures analysis results are always persisted.
                # Note: _materialize_special_outputs will replace zarr with disk automatically
                from openhcs.core.pipeline.materialization_flag_planner import (
                    MaterializationFlagPlanner,
                )

                vfs_config = context.get_vfs_config()
                materialization_backend = (
                    MaterializationFlagPlanner._resolve_materialization_backend(
                        context, vfs_config
                    )
                )
                self._materialize_special_outputs(
                    filemanager,
                    step_plan,
                    special_outputs,
                    materialization_backend,
                    context,
                )
                logger.info("🔬 MATERIALIZATION: Completed materialization")

        except Exception as e:
            import traceback

            full_traceback = traceback.format_exc()
            logger.error(
                f"Error in FunctionStep {step_index} ({step_name}): {e}", exc_info=True
            )
            logger.error(
                f"Full traceback for FunctionStep {step_index} ({step_name}):\n{full_traceback}"
            )

            raise

    def _extract_component_metadata(
        self, context: "ProcessingContext", component: "VariableComponents"
    ) -> Optional[Dict[str, str]]:
        """
        Extract component metadata from context cache safely.

        Args:
            context: ProcessingContext containing metadata_cache
            component: VariableComponents enum specifying which component to extract

        Returns:
            Dictionary mapping component keys to display names, or None if not available
        """
        try:
            if hasattr(context, "metadata_cache") and context.metadata_cache:
                return context.metadata_cache.get(component, None)
            else:
                logger.debug(
                    f"No metadata_cache available in context for {component.value}"
                )
                return None
        except Exception as e:
            logger.debug(f"Error extracting {component.value} metadata from cache: {e}")
            return None

    def _create_openhcs_metadata_for_materialization(
        self, context: "ProcessingContext", output_dir: str, write_backend: str
    ) -> None:
        """
        Create OpenHCS metadata file for materialization writes.

        Args:
            context: ProcessingContext containing microscope_handler and other state
            output_dir: Output directory path where metadata should be written
            write_backend: Backend being used for the write (disk/zarr)
        """
        # Only create OpenHCS metadata for disk/zarr backends
        # OMERO has its own metadata system, memory doesn't need metadata
        if write_backend in [Backend.MEMORY.value, Backend.OMERO_LOCAL.value]:
            logger.debug(f"Skipping metadata creation (backend={write_backend})")
            return

        logger.debug(
            f"Creating metadata for materialization write: {write_backend} -> {output_dir}"
        )

        try:
            # Extract required information
            step_output_dir = Path(output_dir)

            # Check if we have microscope handler for metadata extraction
            if not context.microscope_handler:
                logger.debug(
                    "No microscope_handler in context - skipping OpenHCS metadata creation"
                )
                return

            # Get source microscope information
            source_parser_name = context.microscope_handler.parser.__class__.__name__

            # Extract metadata from source microscope handler
            try:
                grid_dimensions = (
                    context.microscope_handler.metadata_handler.get_grid_dimensions(
                        context.input_dir
                    )
                )
                pixel_size = context.microscope_handler.metadata_handler.get_pixel_size(
                    context.input_dir
                )
            except Exception as e:
                logger.debug(
                    f"Could not extract grid_dimensions/pixel_size from source: {e}"
                )
                grid_dimensions = [1, 1]  # Default fallback
                pixel_size = 1.0  # Default fallback

            # Get list of image files in output directory
            try:
                image_files = []
                if context.filemanager.exists(str(step_output_dir), write_backend):
                    # List files in output directory
                    files = context.filemanager.list_files(
                        str(step_output_dir), write_backend
                    )
                    # Filter for image files (common extensions) and convert to strings
                    image_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
                    image_files = [
                        str(f)
                        for f in files
                        if Path(f).suffix.lower() in image_extensions
                    ]
                    logger.debug(
                        f"Found {len(image_files)} image files in {step_output_dir}"
                    )
            except Exception as e:
                logger.debug(f"Could not list image files in output directory: {e}")
                image_files = []

            # Detect available backends based on actual output files
            available_backends = self._detect_available_backends(step_output_dir)

            # Create metadata structure
            metadata = {
                "microscope_handler_name": context.microscope_handler.microscope_type,
                "source_filename_parser_name": source_parser_name,
                "grid_dimensions": (
                    list(grid_dimensions)
                    if hasattr(grid_dimensions, "__iter__")
                    else [1, 1]
                ),
                "pixel_size": float(pixel_size) if pixel_size is not None else 1.0,
                "image_files": image_files,
                "channels": self._extract_component_metadata(
                    context, VariableComponents.CHANNEL
                ),
                "wells": self._extract_component_metadata(
                    context, VariableComponents.WELL
                ),
                "sites": self._extract_component_metadata(
                    context, VariableComponents.SITE
                ),
                "z_indexes": self._extract_component_metadata(
                    context, VariableComponents.Z_INDEX
                ),
                "timepoints": self._extract_component_metadata(
                    context, VariableComponents.TIMEPOINT
                ),
                "available_backends": available_backends,
            }

            # Save metadata file using disk backend (JSON files always on disk)
            from openhcs.microscopes.openhcs import OpenHCSMetadataHandler

            metadata_path = step_output_dir / OpenHCSMetadataHandler.METADATA_FILENAME

            # Always ensure we can write to the metadata path (delete if exists)
            if context.filemanager.exists(str(metadata_path), Backend.DISK.value):
                context.filemanager.delete(str(metadata_path), Backend.DISK.value)

            # Ensure output directory exists on disk
            context.filemanager.ensure_directory(
                str(step_output_dir), Backend.DISK.value
            )

            # Create JSON content - OpenHCS handler expects JSON format
            import json

            json_content = json.dumps(metadata, indent=2)
            context.filemanager.save(
                json_content, str(metadata_path), Backend.DISK.value
            )
            logger.debug(f"Created OpenHCS metadata file (disk): {metadata_path}")

        except Exception as e:
            # Graceful degradation - log error but don't fail the step
            logger.warning(f"Failed to create OpenHCS metadata file: {e}")
            logger.debug("OpenHCS metadata creation error details:", exc_info=True)

    def _detect_available_backends(self, output_dir: Path) -> Dict[str, bool]:
        """Detect which storage backends are actually available based on output files."""

        backends = {Backend.ZARR.value: False, Backend.DISK.value: False}

        # Check for zarr stores - look for .zarray or .zgroup files (zarr metadata)
        # Zarr stores don't need .zarr extension - any directory with zarr metadata is a store
        if list(output_dir.glob("**/.zarray")) or list(output_dir.glob("**/.zgroup")):
            backends[Backend.ZARR.value] = True

        # Check for image files
        for ext in DEFAULT_IMAGE_EXTENSIONS:
            if list(output_dir.glob(f"*{ext}")):
                backends[Backend.DISK.value] = True
                break

        logger.debug(f"Backend detection result: {backends}")
        return backends

    def _build_analysis_filename(
        self,
        output_key: str,
        step_index: int,
        step_plan: Dict,
        dict_key: Optional[str] = None,
        context=None,
    ) -> str:
        """Build analysis result filename from first image path template.

        Uses first image filename as template to preserve all metadata components.
        Falls back to well ID only if no images available.

        Args:
            output_key: Special output key (e.g., 'rois', 'cell_counts')
            step_index: Pipeline step index
            step_plan: Step plan dictionary
            dict_key: Optional channel/component key for dict pattern functions
            context: Processing context (for accessing microscope handler)
        """
        memory_paths = step_plan["get_paths_for_axis"](
            step_plan["output_dir"], Backend.MEMORY.value
        )

        if not memory_paths:
            return f"{step_plan['axis_id']}_{output_key}_step{step_index}.roi.zip"

        # Filter paths by channel if dict_key provided (for dict pattern functions)
        if dict_key and context:
            # Use microscope handler to parse filenames and filter by channel
            microscope_handler = context.microscope_handler
            parser = microscope_handler.parser

            filtered_paths = []
            for path in memory_paths:
                filename = Path(path).name
                metadata = parser.parse_filename(filename)
                if metadata and str(metadata.get("channel")) == str(dict_key):
                    filtered_paths.append(path)

            if filtered_paths:
                memory_paths = filtered_paths

        # Use first image as template: "A01_s001_w1_z001_t001.tif" -> "A01_s001_w1_z001_t001_rois_step7.roi.zip"
        base_filename = Path(memory_paths[0]).stem
        return f"{base_filename}_{output_key}_step{step_index}.roi.zip"

    def _materialize_special_outputs(
        self, filemanager, step_plan, special_outputs, backend, context
    ):
        """Materialize special outputs (ROIs, cell counts) to disk and streaming backends."""
        # Collect backends: main + streaming
        from openhcs.core.config import StreamingConfig

        backends = [backend]
        backend_kwargs = {backend: {}}

        for config in step_plan.values():
            if isinstance(config, StreamingConfig):
                backends.append(config.backend.value)
                backend_kwargs[config.backend.value] = config.get_streaming_kwargs(
                    context
                )

        # Get analysis directory (pre-calculated by compiler)
        has_step_mat = "materialized_output_dir" in step_plan
        analysis_output_dir = Path(
            step_plan[
                (
                    "materialized_analysis_results_dir"
                    if has_step_mat
                    else "analysis_results_dir"
                )
            ]
        )
        images_dir = str(
            step_plan["materialized_output_dir" if has_step_mat else "output_dir"]
        )

        # Add images_dir and source to all backend kwargs
        step_name = step_plan.get("step_name", "unknown_step")
        for kwargs in backend_kwargs.values():
            kwargs["images_dir"] = images_dir
            kwargs["source"] = (
                step_name  # Pre-built source value for layer/window naming
            )

        filemanager._materialization_context = {"images_dir": images_dir}

        # Get dict pattern info
        def _resolve_materializer_inputs(mat_spec, *, dict_key):
            options = getattr(mat_spec, "options", {}) or {}
            inputs_spec = options.get("inputs") or {}
            if not inputs_spec:
                return {}
            if not isinstance(inputs_spec, dict):
                raise ValueError(
                    f"MaterializationSpec.options['inputs'] must be a dict, got {type(inputs_spec)}"
                )

            resolved: Dict[str, Any] = {}

            for input_name, input_desc in inputs_spec.items():
                if not isinstance(input_desc, dict):
                    raise ValueError(
                        f"Materialization input '{input_name}' must be a dict, got {type(input_desc)}"
                    )

                kind = input_desc.get("kind")
                if kind != "image_slices":
                    raise ValueError(
                        f"Unsupported materialization input kind for '{input_name}': {kind}. "
                        "Supported kinds: 'image_slices'."
                    )

                source = input_desc.get("source")
                if source == "step_input":
                    source_dir = step_plan["input_dir"]
                    source_backend = step_plan.get("read_backend", Backend.MEMORY.value)
                elif source == "step_output":
                    source_dir = step_plan["output_dir"]
                    source_backend = Backend.MEMORY.value
                else:
                    raise ValueError(
                        f"Unsupported materialization input source for '{input_name}': {source}. "
                        "Supported sources: 'step_input', 'step_output'."
                    )

                get_paths_for_axis = step_plan.get("get_paths_for_axis")
                if get_paths_for_axis is None:
                    raise ValueError(
                        "Step plan missing get_paths_for_axis (cannot resolve materializer inputs)"
                    )

                paths = get_paths_for_axis(source_dir, source_backend)

                # Optional grouping filter (only when this materialization invocation is group-specific)
                if dict_key is not None:
                    group_by_key = input_desc.get("group_by")
                    if group_by_key is None:
                        group_by = step_plan.get("group_by")
                        if (
                            group_by is not None
                            and getattr(group_by, "value", None) is not None
                        ):
                            group_by_key = str(group_by.value)

                    if group_by_key is None:
                        raise ValueError(
                            f"Cannot resolve materialization input '{input_name}' for group '{dict_key}': "
                            "no group_by specified in the input spec and step_plan['group_by'] is NONE."
                        )
                    if context is None:
                        raise ValueError(
                            f"Cannot resolve materialization input '{input_name}' for group '{dict_key}': "
                            "context is required for filename parsing."
                        )

                    parser = context.microscope_handler.parser
                    filtered_paths = []
                    for p in paths:
                        filename = Path(p).name
                        metadata = parser.parse_filename(filename)
                        if metadata and str(metadata.get(group_by_key)) == str(
                            dict_key
                        ):
                            filtered_paths.append(p)
                    paths = filtered_paths

                if not paths:
                    raise ValueError(
                        f"Materialization input '{input_name}' resolved to 0 paths "
                        f"(source={source}, dir={source_dir}, backend={source_backend}, group={dict_key})."
                    )

                resolved[input_name] = filemanager.load_batch(paths, source_backend)

            return resolved

        # Materialize each special output
        for output_key, output_info in special_outputs.items():
            mat_spec = output_info.get("materialization_spec")
            if not mat_spec:
                continue

            memory_path = output_info["path"]
            step_index = step_plan["pipeline_position"]

            # For dict patterns, materialize only the channels that produced this output
            channels_to_process = output_info.get("group_keys") or [None]
            paths_by_group = output_info.get("paths_by_group") or {}

            for dict_key in channels_to_process:
                # Build channel-specific memory path if needed
                if dict_key is not None:
                    if dict_key in paths_by_group:
                        channel_path = paths_by_group[dict_key]
                    elif None in paths_by_group:
                        channel_path = paths_by_group[None]
                    else:
                        from openhcs.core.pipeline.path_planner import (
                            PipelinePathPlanner,
                        )

                        channel_path = PipelinePathPlanner.build_dict_pattern_path(
                            memory_path, dict_key
                        )
                else:
                    channel_path = paths_by_group.get(None, memory_path)

                if not filemanager.exists(channel_path, Backend.MEMORY.value):
                    logger.info(
                        f"Skipping special output '{output_key}' for group '{dict_key}' - no data saved at {channel_path}"
                    )
                    continue

                # Load data
                filemanager.ensure_directory(
                    Path(channel_path).parent, Backend.MEMORY.value
                )
                data = filemanager.load(channel_path, Backend.MEMORY.value)

                # Build analysis filename and path (pass dict_key for channel-specific naming)
                filename = self._build_analysis_filename(
                    output_key, step_index, step_plan, dict_key, context
                )
                analysis_path = analysis_output_dir / filename

                # Materialize to all backends
                from openhcs.processing.materialization import materialize

                extra_inputs = _resolve_materializer_inputs(mat_spec, dict_key=dict_key)
                source_paths = sorted(
                    step_plan["get_paths_for_axis"](
                        step_plan["output_dir"], Backend.MEMORY.value
                    )
                )
                materialize(
                    mat_spec,
                    data,
                    str(analysis_path),
                    filemanager,
                    backends,
                    backend_kwargs,
                    context=context,
                    extra_inputs=extra_inputs,
                    source_paths=source_paths,
                    output_key=output_key,
                    step_index=step_index,
                )


def _update_metadata_for_zarr_conversion(
    plate_root: Path,
    original_subdir: str,
    zarr_subdir: str | None,
    context: "ProcessingContext",
) -> None:
    """Update metadata after zarr conversion.

    If zarr_subdir is None: add zarr to original_subdir's available_backends
    If zarr_subdir is set: create complete metadata for zarr subdirectory, set original main=false
    """
    from polystore.metadata_writer import get_metadata_path, AtomicMetadataWriter
    from openhcs.microscopes.openhcs import OpenHCSMetadataGenerator

    if zarr_subdir:
        # Create complete metadata for zarr subdirectory (skip if already complete)
        zarr_dir = plate_root / zarr_subdir
        metadata_generator = OpenHCSMetadataGenerator(context.filemanager)
        metadata_generator.create_metadata(
            context,
            str(zarr_dir),
            "zarr",  # Zarr subdirectory uses zarr backend
            is_main=True,
            plate_root=str(plate_root),
            sub_dir=zarr_subdir,
            skip_if_complete=True,
        )

        # Set original subdirectory to main=false
        metadata_path = get_metadata_path(plate_root)
        writer = AtomicMetadataWriter()
        writer.merge_subdirectory_metadata(
            metadata_path, {original_subdir: {"main": False}}
        )
        logger.info(
            f"Ensured complete metadata for {zarr_subdir}, set {original_subdir} main=false"
        )
    else:
        # Shared subdirectory - add zarr to available_backends
        metadata_path = get_metadata_path(plate_root)
        writer = AtomicMetadataWriter()
        writer.merge_subdirectory_metadata(
            metadata_path, {original_subdir: {"available_backends": {"zarr": True}}}
        )
        logger.info(f"Updated metadata: {original_subdir} now has zarr backend")

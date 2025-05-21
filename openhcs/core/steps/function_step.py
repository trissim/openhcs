"""
FunctionStep implementation for pattern-based processing.

This module contains the FunctionStep class and its helper functions for
pattern-based file selection and function dispatching.

The FunctionStep is the canonical, schema-bound, stateless functional step
that transforms image arrays. It is the foundation for all specialized
steps in the OpenHCS pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from openhcs.constants.constants import (DEFAULT_IMAGE_EXTENSION,
                                            DEFAULT_IMAGE_EXTENSIONS,
                                            DEFAULT_SITE_PADDING, Backend,
                                            MemoryType)
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.step_result import StepResult
from openhcs.formats.func_arg_prep import prepare_patterns_and_functions

logger = logging.getLogger(__name__)


class FunctionStep(AbstractStep):
    """
    Base class for function steps with memory type awareness.

    Function steps are the canonical, schema-bound, stateless functional steps
    that transform image arrays. They are the foundation for all specialized
    steps in the OpenHCS pipeline.

    This class accepts a function that takes a VirtualImageArray3D and returns
    a VirtualImageArray3D, and applies it to the input image array.

    Unlike disk-mandated steps (ImageAssemblyStep, PositionGenerationStep),
    FunctionStep is memory-native by default (requires_disk_output=False) but
    supports optional filesystem persistence via force_disk_output.

    Note: input_memory_type, output_memory_type, and well_id are not constructor parameters.
    These values are assigned during planning and are available in the step plan.
    """

    @property
    def requires_disk_input(self) -> bool:
        return False

    @property
    def requires_disk_output(self) -> bool:
        return False

    def __init__(
        self,
        func: Callable,
        *,
        name: Optional[str] = None,
        variable_components: Optional[List[str]] = ['site'],
        group_by: str = "channel",
        force_disk_output: bool = False
    ):
        """
        Initialize a FuncStep using a single declarative function.

        Args:
            func: A callable that defines the transformation logic
            name: Optional name for the step (defaults to function name)
            variable_components: Components that vary in filename parsing
            group_by: Component to group by during batching (default: 'channel')
            force_disk_output: Whether to force output to disk regardless of backend
        """
        super().__init__(
            name=name or getattr(func, '__name__', 'FunctionStep'),
            variable_components=variable_components,
            group_by=group_by,
            force_disk_output=force_disk_output
        )

        self.func = func
        self.special_inputs = getattr(func, "__special_inputs__", {})
        self.special_outputs = getattr(func, "__special_outputs__", set())
        self.chain_breaker = getattr(func, "__chain_breaker__", False)

    def process(self, context: 'ProcessingContext') -> StepResult:
        """
        Process the step with pattern-based file selection and function dispatching.

        This implementation uses standalone helper functions to handle different responsibilities:
        1. validate_step_plan: Validate step plan and extract key values
        2. detect_patterns: Detect patterns based on variable_components
        3. map_functions_to_patterns: Map functions to patterns based on group_by
        4. process_pattern: Process each pattern with its corresponding function
        5. save_results: Save the processed results

        Args:
            context: The processing context

        Returns:
            StepResult with metadata about processed files

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        try:
            # Get step_id from context
            # Note: context.current_step_id is set by the Pipeline before calling the step's process method
            # It contains the unique identifier of the current step being executed
            step_id = context.current_step_id

            # Validate step plan and extract key values - using helper function defined in the same file
            # Call the standalone function directly by name, not through self
            plan_values = validate_step_plan(context, step_id)

            # Extract values from plan_values
            well_id = plan_values['well_id']
            input_dir = plan_values['input_dir']
            output_dir = plan_values['output_dir']
            variable_components = plan_values['variable_components']
            group_by = plan_values['group_by']
            processing_funcs = plan_values['func']
            read_backend = plan_values['read_backend']  # Using read_backend to match materialization flag planner
            write_backend = plan_values['write_backend']  # Using write_backend to match materialization flag planner
            force_disk_output = plan_values['force_disk_output']  # Get force_disk_output from plan_values

            # Note: read_backend and write_backend are set by the materialization flag planner
            # based on requires_disk_input, requires_disk_output, and force_disk_output
            input_memory_type = plan_values['input_memory_type']
            output_memory_type = plan_values['output_memory_type']
            device_id = plan_values['device_id']

            # Check if input_dir and output_dir are the same
            same_directory = str(input_dir) == str(output_dir)
            if same_directory:
                logger.warning(
                    "Input directory and output directory are the same: %s. "
                    "Will delete existing processed files before saving new ones.",
                    input_dir
                )

            # Log backend information
            logger.info(
                f"Using read_backend={read_backend}, write_backend={write_backend}, "
                f"force_disk_output={force_disk_output}"
            )

            # Log key values for debugging
            logger.debug(
                "Processing step %s with well_id=%s, input_dir=%s, output_dir=%s, "
                "variable_components=%s, group_by=%s",
                step_id, well_id, input_dir, output_dir, variable_components, group_by
            )

            # Detect patterns based on variable_components - direct call
            patterns_by_well = context.microscope_handler.auto_detect_patterns(
                folder_path=input_dir,
                well_filter=[well_id],
                extensions=DEFAULT_IMAGE_EXTENSIONS,  # Use constant for extensions
                group_by=group_by,
                variable_components=variable_components,
                backend=read_backend  # Using read_backend to match materialization flag planner
            )

            # Validate that patterns are found for the specified well_id
            if well_id not in patterns_by_well:
                raise ValueError(
                    f"Clause 65 Violation: No patterns found for well {well_id} in {input_dir}. "
                    f"Available wells: {list(patterns_by_well.keys())}"
                )

            patterns = patterns_by_well[well_id]
            if not patterns:
                raise ValueError(f"Clause 65 Violation: No patterns found for well {well_id} in {input_dir}")

            # Map functions to patterns based on group_by - direct call
            grouped_patterns, component_to_funcs, component_to_args = prepare_patterns_and_functions(
                patterns, processing_funcs, component=group_by
            )

            # Get filemanager from context as required by clause STEPS_CONTEXT_FILEMANAGER
            filemanager = context.filemanager

            # Process each pattern with its corresponding function and immediately save results
            # Store only metadata, not the actual processed arrays
            results = {}

            # Function calling sequence:
            # 1. Iterate through each component value (e.g., channel "1", "2")
            # 2. For each component, get the list of patterns and corresponding function
            # 3. Process each pattern independently with process_and_save_pattern
            # 4. Collect metadata results in a list under the component value
            for component_value, component_patterns in grouped_patterns.items():
                component_func = component_to_funcs[component_value]
                component_args = component_to_args[component_value]

                # Process each pattern and immediately save results
                component_results = []
                for pattern in component_patterns:
                    # Process and save in one operation to avoid memory accumulation
                    # Call the standalone function directly by name, not through self
                    result_metadata = process_and_save_pattern(
                        context, pattern, component_func, component_args,
                        input_dir, output_dir, well_id, component_value,
                        read_backend, write_backend,  # Using read_backend and write_backend to match materialization flag planner
                        input_memory_type, output_memory_type,
                        device_id, same_directory, force_disk_output
                    )

                    # Store only the metadata, not the actual processed array
                    component_results.append(result_metadata)

                # Store component results (metadata only, not actual arrays)
                # This creates a nested structure: results[component_value] = [metadata1, metadata2, ...]
                results[component_value] = component_results

            # Return StepResult with metadata for traceability
            # Note: This metadata is used by the Pipeline for logging and debugging
            # No downstream components rely on specific metadata fields
            return StepResult(
                metadata={
                    # Metadata fields - all for debugging and logging only
                    "processed_components": list(results.keys()),  # List of component values processed
                    "results": results,  # Contains only metadata, not actual arrays
                    "well_id": well_id,
                    "input_memory_type": input_memory_type,
                    "output_memory_type": output_memory_type,
                    "read_backend": read_backend,
                    "write_backend": write_backend,
                    "same_directory_operation": same_directory,
                    "force_disk_output": force_disk_output
                }
            )

        except Exception as e:
            # Wrap all exceptions with clear context
            logger.error("Error in FunctionStep.process: %s", e)
            raise ValueError(f"Error in FunctionStep.process: {e}") from e


# Helper functions defined outside the class to maintain statelessness (Clause 246)

def validate_step_plan(context: 'ProcessingContext', step_id: str) -> Dict[str, Any]:
    """
    Validate that the step plan contains all required fields and extract key values.

    Args:
        context: The processing context
        step_id: The ID of the current step

    Returns:
        Dictionary containing extracted values from the step plan

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    # Check that step_id exists in context.step_plans
    if step_id not in context.step_plans:
        raise ValueError(f"Clause 65 Violation: Step plan not found for step_id {step_id}")

    step_plan = context.step_plans[step_id]

    # Validate all required fields are present
    required_fields = [
        'well_id', 'input_dir', 'output_dir',
        'input_memory_type', 'output_memory_type',
        'read_backend', 'write_backend',  # Using read_backend and write_backend to match materialization flag planner
        'variable_components', 'group_by', 'func'
    ]

    for field in required_fields:
        if field not in step_plan:
            raise ValueError(f"Clause 65 Violation: '{field}' is required in step plan")

    # Check for force_disk_output (optional field with default False)
    force_disk_output = step_plan.get('force_disk_output', False)

    # Extract values from step_plan
    result = {
        'well_id': step_plan['well_id'],
        'input_dir': step_plan['input_dir'],
        'output_dir': step_plan['output_dir'],
        'variable_components': step_plan['variable_components'],
        'group_by': step_plan['group_by'],
        'func': step_plan['func'],
        'read_backend': step_plan['read_backend'],  # Using read_backend to match materialization flag planner
        'write_backend': step_plan['write_backend'],  # Using write_backend to match materialization flag planner
        'input_memory_type': step_plan['input_memory_type'],
        'output_memory_type': step_plan['output_memory_type'],
        'force_disk_output': force_disk_output,  # Include force_disk_output in result
    }

    # Handle gpu_id requirements for GPU memory types
    gpu_memory_types = [MemoryType.CUPY.value, MemoryType.TORCH.value, MemoryType.TENSORFLOW.value, MemoryType.JAX.value]
    if result['input_memory_type'] in gpu_memory_types or result['output_memory_type'] in gpu_memory_types:
        if 'gpu_id' not in step_plan:
            raise ValueError(
                f"Clause 65 Violation: 'gpu_id' is required in step plan for GPU memory types: "
                f"{result['input_memory_type']} or {result['output_memory_type']}"
            )
        result['gpu_id'] = step_plan['gpu_id']
    else:
        # For CPU memory types, gpu_id is not required but might be present
        result['gpu_id'] = None
        if 'gpu_id' in step_plan:
            result['gpu_id'] = step_plan['gpu_id']

    # Backward compatibility: also set device_id to the same value as gpu_id
    result['device_id'] = result['gpu_id']

    return result


def process_and_save_pattern(
    context: 'ProcessingContext',
    pattern: Any,
    component_func: Any,
    component_args: Dict[str, Any],
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    well_id: str,
    component_value: str,
    read_backend: str,
    write_backend: str,
    input_memory_type: str,
    output_memory_type: str,
    device_id: Optional[int],
    same_directory: bool,
    force_disk_output: bool
) -> Dict[str, Any]:
    """
    Process a single pattern with its corresponding function and immediately save the results.

    Args:
        context: The processing context
        pattern: The pattern to process
        component_func: The function to apply
        component_args: Arguments for the function
        input_dir: The input directory
        output_dir: The output directory
        well_id: The well ID
        component_value: The component value (e.g., channel)
        read_backend: Backend to use for input file operations
        write_backend: Backend to use for output file operations
        input_memory_type: Memory type for input data
        output_memory_type: Memory type for output data
        device_id: Device ID for GPU operations
        same_directory: Whether input and output directories are the same
        force_disk_output: Whether to force disk output regardless of write_backend

    Returns:
        Metadata about the processed and saved results (not the actual arrays)

    Raises:
        ValueError: If processing or saving the pattern fails
    """
    start_time = time.time()

    try:
        # Find matching files for the pattern
        matching_files = context.microscope_handler.path_list_from_pattern(
            directory=input_dir,
            pattern=pattern,
            backend=read_backend  # Using read_backend to match materialization flag planner
        )

        if not matching_files:
            raise ValueError(
                f"No matching files found for pattern {pattern} in {input_dir}"
            )

        # Load raw slices using FileManager
        raw_slices = []
        for file_path in matching_files:
            try:
                # FileManager handles path conversion internally
                image = context.filemanager.load_image(
                    str(Path(input_dir) / file_path),
                    read_backend  # Using read_backend to match materialization flag planner
                )
                if image is not None:
                    raw_slices.append(image)
            except Exception as e:
                # Log error but continue with other files
                logger.error(
                    "Error loading image %s: %s",
                    file_path, e
                )

        if not raw_slices:
            raise ValueError(
                f"No valid images loaded for pattern {pattern} in {input_dir}"
            )

        # Stack slices into a 3D array
        stack = stack_slices(
            slices=raw_slices,
            memory_type=input_memory_type,
            gpu_id=device_id,  # Using device_id consistently throughout the codebase
            allow_single_slice=False  # Enforce multiple slices
        )

        # Apply the function with appropriate arguments
        if isinstance(component_func, list):
            # Apply a list of functions in sequence
            processed_stack = stack
            for func_item in component_func:
                if isinstance(func_item, tuple) and len(func_item) == 2 and callable(func_item[0]):
                    # It's a (function, kwargs) tuple
                    func, kwargs = func_item
                    processed_stack = func(processed_stack, **kwargs)
                else:
                    # It's just a function
                    processed_stack = func_item(processed_stack)
        else:
            # Apply a single function with its arguments
            processed_stack = component_func(stack, **component_args)

        # Validate that the result is a 3D array
        if not _is_3d(processed_stack):
            raise ValueError(
                f"Clause 278 Violation: Function must return a 3D array, "
                f"got shape {getattr(processed_stack, 'shape', 'unknown')}. "
                f"All FuncStep.apply() implementations must return a 3D array of shape [Z, Y, X]."
            )

        # Get original shape and dtype for metadata
        original_shape = processed_stack.shape
        original_dtype = str(processed_stack.dtype)

        # Unstack the 3D array into 2D slices
        output_slices = unstack_slices(
            array=processed_stack,
            memory_type=output_memory_type,
            gpu_id=device_id,  # Using device_id consistently throughout the codebase
            validate_slices=True  # Validate that slices are 2D
        )

        # Save processed slices
        output_paths = []
        for i, img_slice in enumerate(output_slices):
            try:
                # Construct output filename
                output_filename = context.microscope_handler.parser.construct_filename(
                    well=well_id,
                    site=i+1,  # 1-based site index
                    channel=component_value,
                    extension=DEFAULT_IMAGE_EXTENSION,
                    site_padding=DEFAULT_SITE_PADDING
                )

                # Create full output path
                output_path = Path(output_dir) / output_filename
                output_paths.append(str(output_path))

                # Delete existing file if input and output directories are the same
                if same_directory and context.filemanager.exists(
                    str(output_path),
                    write_backend  # Using write_backend to match materialization flag planner
                ):
                    logger.info("Deleting existing file before saving: %s", output_path)
                    context.filemanager.delete_file(
                        str(output_path),
                        write_backend  # Using write_backend to match materialization flag planner
                    )

                # Save using FileManager
                context.filemanager.save_image(
                    img_slice,
                    str(output_path),
                    write_backend  # Using write_backend to match materialization flag planner
                )

                # Handle force_disk_output - save an additional copy to disk if needed
                if force_disk_output and write_backend != Backend.DISK.value:
                    logger.info("Force disk output enabled, saving additional copy to disk: %s", output_path)
                    # Save using FileManager with 'disk' backend
                    context.filemanager.save_image(
                        path=str(output_path),
                        image=img_slice,
                        backend=Backend.DISK.value  # Always use 'disk' backend for forced disk output
                    )
            except Exception as e:
                # Log error but continue with other slices
                logger.error(
                    "Error saving image %s: %s",
                    output_filename, e
                )

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Return metadata about the processed and saved results
        return {
            # Required fields (used by downstream components)
            "output_paths": output_paths,  # List of saved file paths - only required field

            # Optional fields (for debugging and logging only)
            "shape": original_shape,       # Original 3D array shape
            "dtype": original_dtype,       # Original data type
            "pattern": pattern,            # Original pattern
            "processing_time_ms": processing_time_ms  # Processing time in milliseconds
        }
        # Note: Downstream components do not rely on any metadata fields except output_paths.
        # The metadata structure is primarily for debugging and traceability.

    except Exception as e:
        # Wrap exceptions with clear context
        raise ValueError(f"Error processing and saving pattern {pattern}: {e}") from e
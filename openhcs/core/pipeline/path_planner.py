"""
Pipeline path planning module for OpenHCS.

This module provides the PipelinePathPlanner class, which is responsible for
determining input and output paths for each step in a pipeline in a single pass.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from openhcs.constants.constants import READ_BACKEND, WRITE_BACKEND
from openhcs.core.context.processing_context import ProcessingContext # ADDED
from openhcs.core.pipeline.pipeline_utils import get_core_callable
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep


logger = logging.getLogger(__name__)

# Metadata resolver registry for extensible metadata injection
METADATA_RESOLVERS: Dict[str, Dict[str, Any]] = {
    "grid_dimensions": {
        "resolver": lambda context: context.microscope_handler.get_grid_dimensions(context.input_dir),
        "description": "Grid dimensions (num_rows, num_cols) for position generation functions"
    },
    # Future extensions can be added here:
    # "pixel_size": {
    #     "resolver": lambda context: context.microscope_handler.get_pixel_size(context.input_dir),
    #     "description": "Pixel size in micrometers"
    # },
}

def resolve_metadata(key: str, context: ProcessingContext) -> Any:
    """
    Resolve metadata using registered resolvers.

    Args:
        key: The metadata key to resolve
        context: The processing context containing microscope handler

    Returns:
        The resolved metadata value

    Raises:
        ValueError: If no resolver is registered for the key
    """
    if key not in METADATA_RESOLVERS:
        raise ValueError(f"No metadata resolver registered for key '{key}'. Available keys: {list(METADATA_RESOLVERS.keys())}")

    resolver_func = METADATA_RESOLVERS[key]["resolver"]
    try:
        return resolver_func(context)
    except Exception as e:
        raise ValueError(f"Failed to resolve metadata for key '{key}': {e}") from e

def register_metadata_resolver(key: str, resolver_func: Callable[[ProcessingContext], Any], description: str) -> None:
    """
    Register a new metadata resolver.

    Args:
        key: The metadata key
        resolver_func: Function that takes ProcessingContext and returns the metadata value
        description: Human-readable description of what this metadata provides
    """
    METADATA_RESOLVERS[key] = {
        "resolver": resolver_func,
        "description": description
    }
    logger.debug(f"Registered metadata resolver for key '{key}': {description}")

def inject_metadata_into_pattern(func_pattern: Any, metadata_key: str, metadata_value: Any) -> Any:
    """
    Inject metadata into a function pattern by modifying or creating kwargs.

    Args:
        func_pattern: The original function pattern (callable, tuple, list, or dict)
        metadata_key: The parameter name to inject
        metadata_value: The value to inject

    Returns:
        Modified function pattern with metadata injected
    """
    # Case 1: Direct callable -> convert to (callable, {metadata_key: metadata_value})
    if callable(func_pattern) and not isinstance(func_pattern, type):
        return (func_pattern, {metadata_key: metadata_value})

    # Case 2: (callable, kwargs) tuple -> update kwargs
    elif isinstance(func_pattern, tuple) and len(func_pattern) == 2 and callable(func_pattern[0]):
        func, existing_kwargs = func_pattern
        updated_kwargs = existing_kwargs.copy()
        updated_kwargs.update({metadata_key: metadata_value})
        return (func, updated_kwargs)

    # Case 3: Single-item list -> inject into the single item and return as list
    elif isinstance(func_pattern, list) and len(func_pattern) == 1:
        single_item = func_pattern[0]
        # Recursively inject into the single item
        modified_item = inject_metadata_into_pattern(single_item, metadata_key, metadata_value)
        return [modified_item]

    # Case 4: Multi-item lists or dict patterns -> not supported for metadata injection
    # These complex patterns should not be used with metadata-requiring functions
    else:
        raise ValueError(f"Cannot inject metadata into complex function pattern: {type(func_pattern)}. "
                        f"Functions requiring metadata should use simple patterns (callable, (callable, kwargs), or single-item lists).")

# FIRST_STEP_OUTPUT_SUFFIX removed

class PlanError(ValueError):
    """Error raised when pipeline planning fails."""
    pass

class PipelinePathPlanner:
    """Plans and prepares execution paths for pipeline steps."""

    # Removed resolve_special_path static method

    @staticmethod
    def prepare_pipeline_paths(
        context: ProcessingContext, # CHANGED: context is now the primary input
        pipeline_definition: List[AbstractStep]
        # step_plans, well_id, initial_pipeline_input_dir are now derived from context
    ) -> Dict[str, Dict[str, Any]]: # Return type is still the modified step_plans from context
        """
        Prepare path information in a single pass through the pipeline.
        Modifies context.step_plans in place.
    
        Args:
            context: The ProcessingContext, containing step_plans, well_id, input_dir, and config.
            pipeline_definition: List of AbstractStep instances.
            
        Returns:
            The modified step_plans dictionary (from context.step_plans).
        """
        path_config = context.get_path_planning_config()
        step_plans = context.step_plans # Work on the context's step_plans
        well_id = context.well_id

        # ALWAYS use plate_path for path planning calculations to ensure consistent naming
        # Store the real input_dir for first step override at the end
        real_input_dir = context.input_dir
        if context.zarr_conversion_path:
            # For zarr conversion, use zarr conversion path for calculations
            initial_pipeline_input_dir = Path(context.zarr_conversion_path)
        elif hasattr(context, 'plate_path') and context.plate_path:
            # Use plate_path for all calculations to ensure consistent output naming
            initial_pipeline_input_dir = Path(context.plate_path)
        else:
            # Fallback to input_dir if plate_path not available
            initial_pipeline_input_dir = context.input_dir

        if not step_plans: # Should be initialized by PipelineCompiler before this call
            raise ValueError("Context step_plans must be initialized before path planning.")
        if not initial_pipeline_input_dir:
            raise ValueError("Context input_dir must be set before path planning.")

        steps = pipeline_definition

        # Modify step_plans in place
    
        # Track available special outputs by key for validation
        declared_outputs = {}
    
        # Single pass through steps
        for i, step in enumerate(steps):
            step_id = step.step_id
            step_name = step.name

            # --- Determine contract sources ---
            s_outputs_keys: Set[str] = set()
            s_inputs_info: Dict[str, bool] = {}

            if isinstance(step, FunctionStep):
                core_callable = get_core_callable(step.func)
                if core_callable:
                    s_outputs_keys = getattr(core_callable, '__special_outputs__', set())
                    s_inputs_info = getattr(core_callable, '__special_inputs__', {})
            else: # For non-FunctionSteps, assume contracts are direct attributes if they exist
                raw_s_outputs = getattr(step, 'special_outputs', set())
                if isinstance(raw_s_outputs, str):
                    s_outputs_keys = {raw_s_outputs}
                elif isinstance(raw_s_outputs, list):
                    s_outputs_keys = set(raw_s_outputs)
                elif isinstance(raw_s_outputs, set):
                    s_outputs_keys = raw_s_outputs
                
                raw_s_inputs = getattr(step, 'special_inputs', {})
                if isinstance(raw_s_inputs, str):
                    s_inputs_info = {raw_s_inputs: True}
                elif isinstance(raw_s_inputs, list):
                    s_inputs_info = {k: True for k in raw_s_inputs}
                elif isinstance(raw_s_inputs, dict):
                    s_inputs_info = raw_s_inputs
                
                is_cb = getattr(step, 'chain_breaker', False)

            # --- Process input directory ---
            if i == 0: # First step
                if step_id in step_plans and "input_dir" in step_plans[step_id]:
                    step_input_dir = Path(step_plans[step_id]["input_dir"])
                elif hasattr(step, "input_dir") and step.input_dir is not None:
                    step_input_dir = Path(step.input_dir) # User override on step object
                else:
                    step_input_dir = initial_pipeline_input_dir # Fallback to pipeline-level input dir
            else: # Subsequent steps (i > 0)
                if step_id in step_plans and "input_dir" in step_plans[step_id]:
                    step_input_dir = Path(step_plans[step_id]["input_dir"])
                elif hasattr(step, "input_dir") and step.input_dir is not None:
                    # Keep input from step kwargs/attributes for subsequent steps too
                    step_input_dir = Path(step.input_dir)
                else:
                    # Default: Use previous step's output
                    prev_step = steps[i-1]
                    prev_step_id = prev_step.step_id
                    if prev_step_id in step_plans and "output_dir" in step_plans[prev_step_id]:
                        step_input_dir = Path(step_plans[prev_step_id]["output_dir"])
                    else:
                        # This should ideally not be reached if previous steps always have output_dir
                        raise ValueError(f"Previous step {prev_step.name} (ID: {prev_step_id}) has no output_dir in step_plans.")

            # --- Chain breaker logic ---
            chain_breaker_read_backend = None  # Track if this step should use disk backend
            prev_is_chain_breaker = False  # Initialize for all steps
            if i > 0:  # Not the first step
                # Check if the PREVIOUS step is a chain breaker
                prev_step = steps[i-1]
                prev_is_chain_breaker = False
                if isinstance(prev_step, FunctionStep):
                    func_to_check = prev_step.func

                    if isinstance(func_to_check, (tuple, list)) and func_to_check:
                        func_to_check = func_to_check[0]

                    # If func_to_check is a tuple (function, params), extract just the function
                    if isinstance(func_to_check, tuple) and len(func_to_check) >= 1:
                        func_to_check = func_to_check[0]

                    if callable(func_to_check):
                        prev_is_chain_breaker = getattr(func_to_check, '__chain_breaker__', False)
                        if prev_is_chain_breaker:
                            logger.info(f"🔗 CHAINBREAKER: Detected chainbreaker function '{func_to_check.__name__}' in step '{prev_step.name}'")

                # If previous step is chain breaker, use first step's input dir and same backend as first step
                if prev_is_chain_breaker:
                    # Get first step's input_dir - check step_plans first, then calculate it
                    first_step = steps[0]
                    first_step_id = first_step.step_id
                    first_step_input_dir = None

                    # Try to get from step_plans (if first step was already processed)
                    if first_step_id in step_plans and "input_dir" in step_plans[first_step_id]:
                        first_step_input_dir = step_plans[first_step_id]["input_dir"]
                    # Otherwise, calculate it the same way we do for first step
                    elif hasattr(first_step, "input_dir") and first_step.input_dir is not None:
                        first_step_input_dir = str(first_step.input_dir)
                    else:
                        first_step_input_dir = str(initial_pipeline_input_dir)

                    if first_step_input_dir:
                        original_step_input_dir = step_input_dir
                        step_input_dir = Path(first_step_input_dir)

                        # Use same backend as first step instead of hardcoded 'disk'
                        if first_step_id in step_plans and READ_BACKEND in step_plans[first_step_id]:
                            chain_breaker_read_backend = step_plans[first_step_id][READ_BACKEND]
                            logger.info(f"🔗 CHAINBREAKER: Step '{step_name}' will use same backend as first step: '{chain_breaker_read_backend}'")
                        else:
                            from openhcs.constants.constants import Backend
                            chain_breaker_read_backend = Backend.DISK.value
                            logger.info(f"🔗 CHAINBREAKER: Step '{step_name}' using fallback disk backend (first step backend not yet determined)")

                        logger.info(f"🔗 CHAINBREAKER: Step '{step_name}' redirected from '{original_step_input_dir}' to first step input '{first_step_input_dir}'")
                    else:
                        logger.warning(f"Step '{step_name}' follows chain breaker '{prev_step.name}' but could not determine first step input_dir")
                
            # --- Process output directory ---
            # Check if step_plans already has this step with output_dir
            if step_id in step_plans and "output_dir" in step_plans[step_id]:
                step_output_dir = Path(step_plans[step_id]["output_dir"])
            elif hasattr(step, "output_dir") and step.output_dir is not None:
                # Keep output from step kwargs
                step_output_dir = Path(step.output_dir)
            elif i < len(steps) - 1:
                next_step = steps[i+1]
                next_step_id = next_step.step_id
                if next_step_id in step_plans and "input_dir" in step_plans[next_step_id]:
                    # Use next step's input from step_plans
                    step_output_dir = Path(step_plans[next_step_id]["input_dir"])
                elif hasattr(next_step, "input_dir") and next_step.input_dir is not None:
                    # Use next step's input from step attribute
                    step_output_dir = Path(next_step.input_dir)
                else:
                    # For first step (i == 0) OR steps following chainbreakers, create output directory with suffix
                    # For other subsequent steps (i > 0), work in place (use same directory as input)
                    if i == 0 or prev_is_chain_breaker:
                        # Create output directory with suffix
                        current_suffix = path_config.output_dir_suffix
                        step_output_dir = step_input_dir.with_name(f"{step_input_dir.name}{current_suffix}")
                    else:
                        # Subsequent steps work in place - use same directory as input
                        step_output_dir = step_input_dir
            else:
                # Last step: Work in place - use same directory as input
                step_output_dir = step_input_dir
                
            # --- Rule: First step and chainbreaker followers use global output logic ---
            if (i == 0 or prev_is_chain_breaker):
                # For the first step and chain breakers, apply global output folder logic
                # Always use plate_path.name for consistent output naming
                if hasattr(context, 'plate_path') and context.plate_path:
                    plate_path = Path(context.plate_path)
                    # Check if global output folder is configured
                    global_output_folder = path_config.global_output_folder
                    if global_output_folder:
                        # Use global output folder: {global_folder}/{plate_name}{suffix}
                        global_folder = Path(global_output_folder)
                        step_output_dir = global_folder / f"{plate_path.name}{path_config.output_dir_suffix}"
                    else:
                        # Use plate parent directory: {plate_parent}/{plate_name}{suffix}
                        step_output_dir = plate_path.with_name(f"{plate_path.name}{path_config.output_dir_suffix}")
                else:
                    # Fallback to input directory name if plate_path not available
                    step_output_dir = step_input_dir.with_name(f"{step_input_dir.name}{path_config.output_dir_suffix}")

            # --- Process special I/O ---
            special_outputs = {}
            special_inputs = {}

            # Process special outputs
            if s_outputs_keys: # Use the keys derived from core_callable or step attribute
                for key in sorted(list(s_outputs_keys)): # Iterate over sorted keys
                    # Use key directly - no unnecessary sanitization!
                    output_path = Path(step_output_dir) / f"{key}.pkl"
                    special_outputs[key] = {"path": str(output_path)}
                    # Register this output for future steps
                    declared_outputs[key] = {
                        "step_id": step_id,
                        "position": i,
                        "path": str(output_path)
                    }
                
            # Process special inputs
            metadata_injected_steps = {}  # Track steps that need metadata injection
            if s_inputs_info: # Use the info derived from core_callable or step attribute
                for key in sorted(list(s_inputs_info.keys())): # Iterate over sorted keys
                    # Check if special input exists from earlier step
                    if key in declared_outputs:
                        # Normal step-to-step special input linking
                        producer = declared_outputs[key]
                        # Validate producer comes before consumer
                        if producer["position"] >= i:
                            producer_step_name = steps[producer["position"]].name # Ensure 'steps' is the pipeline_definition list
                            raise PlanError(f"Step '{step_name}' cannot consume special input '{key}' from later step '{producer_step_name}'")

                        special_inputs[key] = {
                            "path": producer["path"],
                            "source_step_id": producer["step_id"]
                        }
                    elif key in s_outputs_keys:
                        # Current step produces this special input itself - self-fulfilling
                        # This will be handled when special outputs are processed
                        # For now, we'll create a placeholder that will be updated
                        output_path = Path(step_output_dir) / f"{key}.pkl"
                        special_inputs[key] = {
                            "path": str(output_path),
                            "source_step_id": step_id  # Self-reference
                        }
                    elif key in METADATA_RESOLVERS:
                        # Metadata special input - resolve and inject into function pattern
                        try:
                            metadata_value = resolve_metadata(key, context)
                            logger.debug(f"Resolved metadata '{key}' = {metadata_value} for step '{step_name}'")

                            # Store metadata for injection into function pattern
                            # This will be handled by FuncStepContractValidator
                            metadata_injected_steps[key] = metadata_value

                        except Exception as e:
                            raise PlanError(f"Step '{step_name}' requires metadata '{key}', but resolution failed: {e}")
                    else:
                        # No producer step and no metadata resolver
                        available_metadata = list(METADATA_RESOLVERS.keys())
                        raise PlanError(f"Step '{step_name}' requires special input '{key}', but no upstream step produces it "
                                      f"and no metadata resolver is available. Available metadata keys: {available_metadata}")

            # Store metadata injection info for FuncStepContractValidator
            if metadata_injected_steps and isinstance(step, FunctionStep):
                # We need to modify the function pattern to inject metadata
                # This will be stored in step_plans and picked up by FuncStepContractValidator
                original_func = step.func
                modified_func = original_func

                # Inject each metadata value into the function pattern
                for metadata_key, metadata_value in metadata_injected_steps.items():
                    modified_func = inject_metadata_into_pattern(modified_func, metadata_key, metadata_value)
                    logger.debug(f"Injected metadata '{metadata_key}' into function pattern for step '{step_name}'")

                # Store the modified function pattern - FuncStepContractValidator will pick this up
                step.func = modified_func



            # Update step plan with path info
            step_plans[step_id].update({
                "input_dir": str(step_input_dir),
                "output_dir": str(step_output_dir),
                "pipeline_position": i,
                "follows_chain_breaker": prev_is_chain_breaker,  # Flag for zarr conversion logic
                "special_inputs": special_inputs,
                "special_outputs": special_outputs,
            })

            # Apply chain breaker read backend if needed
            if chain_breaker_read_backend is not None:
                step_plans[step_id][READ_BACKEND] = chain_breaker_read_backend

            # --- Ensure directories exist using appropriate backends ---
            # Get the write backend for this step's output directory
            if step_id in step_plans and WRITE_BACKEND in step_plans[step_id]:
                output_backend = step_plans[step_id][WRITE_BACKEND]
                context.filemanager.ensure_directory(step_output_dir, output_backend)
                logger.debug(f"Created output directory {step_output_dir} using backend {output_backend}")

            # Get the read backend for this step's input directory (if not first step)
            if i > 0 and step_id in step_plans and READ_BACKEND in step_plans[step_id]:
                input_backend = step_plans[step_id][READ_BACKEND]
                context.filemanager.ensure_directory(step_input_dir, input_backend)
                logger.debug(f"Created input directory {step_input_dir} using backend {input_backend}")
            elif i == 0:
                # First step always uses disk backend for input
                context.filemanager.ensure_directory(step_input_dir, "disk")
                logger.debug(f"Created first step input directory {step_input_dir} using disk backend")

        # --- Final path connectivity validation after all steps are processed ---
        for i, step in enumerate(steps):
            if i == 0:
                continue  # Skip first step

            curr_step_id = step.step_id
            prev_step_id = steps[i-1].step_id
            curr_step_name = step.name
            prev_step_name = steps[i-1].name

            curr_step_input_dir = step_plans[curr_step_id]["input_dir"]
            prev_step_output_dir = step_plans[prev_step_id]["output_dir"]

            # Check if the PREVIOUS step is a chain breaker
            prev_step = steps[i-1]
            prev_is_chain_breaker_flag_from_plan = False
            if isinstance(prev_step, FunctionStep):
                func_to_check = prev_step.func

                if isinstance(func_to_check, (tuple, list)) and func_to_check:
                    func_to_check = func_to_check[0]

                # If func_to_check is a tuple (function, params), extract just the function
                if isinstance(func_to_check, tuple) and len(func_to_check) >= 1:
                    func_to_check = func_to_check[0]

                if callable(func_to_check):
                    prev_is_chain_breaker_flag_from_plan = getattr(func_to_check, '__chain_breaker__', False)

            # Check path connectivity unless the previous step is a chain breaker
            if not prev_is_chain_breaker_flag_from_plan and curr_step_input_dir != prev_step_output_dir:
                # Check if connected through special I/O
                has_special_connection = False
                for _, input_info in step_plans[curr_step_id].get("special_inputs", {}).items(): # key variable renamed to _
                    if input_info["source_step_id"] == prev_step_id:
                        has_special_connection = True
                        break

                if not has_special_connection:
                    raise PlanError(f"Path discontinuity: {prev_step_name} output ({prev_step_output_dir}) doesn't connect to {curr_step_name} input ({curr_step_input_dir})") # Added paths to error

        # === ZARR CONVERSION FIRST STEP OVERRIDE ===
        # If zarr conversion is happening, override first step to read from original location
        if context.zarr_conversion_path and steps:
            first_step_id = steps[0].step_id
            step_plans[first_step_id]['input_dir'] = context.original_input_dir
            step_plans[first_step_id]['convert_to_zarr'] = context.zarr_conversion_path
            logger.info(f"Zarr conversion: first step reads from {context.original_input_dir}, converts to {context.zarr_conversion_path}")

        # === FIRST STEP INPUT OVERRIDE ===
        # If we used plate_path for calculations but real input is different, override first step
        elif hasattr(context, 'plate_path') and context.plate_path and real_input_dir and steps:
            plate_path_str = str(context.plate_path)
            real_input_str = str(real_input_dir)
            if plate_path_str != real_input_str:
                first_step_id = steps[0].step_id
                step_plans[first_step_id]['input_dir'] = real_input_str
                logger.info(f"Path planning: used plate_path ({plate_path_str}) for calculations, overriding first step to read from real input ({real_input_str})")

        return step_plans
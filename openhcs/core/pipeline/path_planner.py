"""
Pipeline path planning module for OpenHCS.

This module provides the PipelinePathPlanner class, which is responsible for
determining input and output paths for each step in a pipeline in a single pass.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from openhcs.constants.constants import READ_BACKEND
from openhcs.core.context.processing_context import ProcessingContext # ADDED
from openhcs.core.pipeline.pipeline_utils import get_core_callable, to_snake_case
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep


logger = logging.getLogger(__name__)

# Metadata resolver registry for extensible metadata injection
METADATA_RESOLVERS: Dict[str, Dict[str, Any]] = {
    "grid_dimensions": {
        "resolver": lambda context: context.microscope_handler.get_grid_dimensions(context.input_dir),
        "description": "Grid dimensions (num_cols, num_rows) for position generation functions"
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
        initial_pipeline_input_dir = context.input_dir # Assuming context.input_dir is the equivalent

        if not step_plans: # Should be initialized by PipelineCompiler before this call
            raise ValueError("Context step_plans must be initialized before path planning.")
        if not initial_pipeline_input_dir:
            raise ValueError("Context input_dir must be set before path planning.")

        steps = pipeline_definition

        # Modify step_plans in place (step_paths is an alias to context.step_plans)
        step_paths = step_plans
    
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
                if step_id in step_paths and "input_dir" in step_paths[step_id]:
                    step_input_dir = Path(step_paths[step_id]["input_dir"])
                elif hasattr(step, "input_dir") and step.input_dir is not None:
                    step_input_dir = Path(step.input_dir) # User override on step object
                else:
                    step_input_dir = initial_pipeline_input_dir # Fallback to pipeline-level input dir
            else: # Subsequent steps (i > 0)
                if step_id in step_paths and "input_dir" in step_paths[step_id]:
                    step_input_dir = Path(step_paths[step_id]["input_dir"])
                elif hasattr(step, "input_dir") and step.input_dir is not None:
                    # Keep input from step kwargs/attributes for subsequent steps too
                    step_input_dir = Path(step.input_dir)
                else:
                    # Default: Use previous step's output
                    prev_step = steps[i-1]
                    prev_step_id = prev_step.step_id
                    if prev_step_id in step_paths and "output_dir" in step_paths[prev_step_id]:
                        step_input_dir = Path(step_paths[prev_step_id]["output_dir"])
                    else:
                        # This should ideally not be reached if previous steps always have output_dir
                        raise ValueError(f"Previous step {prev_step.name} (ID: {prev_step_id}) has no output_dir in step_plans.")

            # --- Chain breaker logic ---
            chain_breaker_read_backend = None  # Track if this step should use disk backend
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

                # If previous step is chain breaker, use first step's input dir and set disk backend
                if prev_is_chain_breaker:
                    # Get first step's input_dir - check step_paths first, then calculate it
                    first_step = steps[0]
                    first_step_id = first_step.step_id
                    first_step_input_dir = None

                    # Try to get from step_paths (if first step was already processed)
                    if first_step_id in step_paths and "input_dir" in step_paths[first_step_id]:
                        first_step_input_dir = step_paths[first_step_id]["input_dir"]
                    # Otherwise, calculate it the same way we do for first step
                    elif hasattr(first_step, "input_dir") and first_step.input_dir is not None:
                        first_step_input_dir = str(first_step.input_dir)
                    else:
                        first_step_input_dir = str(initial_pipeline_input_dir)

                    if first_step_input_dir:
                        original_step_input_dir = step_input_dir
                        step_input_dir = Path(first_step_input_dir)
                        chain_breaker_read_backend = 'disk'  # Store for later application
                        logger.info(f"🔗 CHAINBREAKER: Step '{step_name}' redirected from '{original_step_input_dir}' to first step input '{first_step_input_dir}' and will use disk backend")
                    else:
                        logger.warning(f"Step '{step_name}' follows chain breaker '{prev_step.name}' but could not determine first step input_dir")
                
            # --- Process output directory ---
            # Check if step_paths already has this step with output_dir
            if step_id in step_paths and "output_dir" in step_paths[step_id]:
                step_output_dir = Path(step_paths[step_id]["output_dir"])
            elif hasattr(step, "output_dir") and step.output_dir is not None:
                # Keep output from step kwargs
                step_output_dir = Path(step.output_dir)
            elif i < len(steps) - 1:
                next_step = steps[i+1]
                next_step_id = next_step.step_id
                if next_step_id in step_paths and "input_dir" in step_paths[next_step_id]:
                    # Use next step's input from step_plans
                    step_output_dir = Path(step_paths[next_step_id]["input_dir"])
                elif hasattr(next_step, "input_dir") and next_step.input_dir is not None:
                    # Use next step's input from step attribute
                    step_output_dir = Path(next_step.input_dir)
                else:
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
                            # Check if global output folder is configured
                            global_output_folder = path_config.global_output_folder
                            if global_output_folder:
                                # Use global output folder: {global_folder}/{workspace_name}{suffix}
                                global_folder = Path(global_output_folder)
                                step_output_dir = global_folder / f"{workspace_path.name}{current_suffix}"
                            else:
                                # Use current behavior: same parent as workspace
                                step_output_dir = workspace_path.with_name(f"{workspace_path.name}{current_suffix}")
                        else:
                            step_output_dir = step_input_dir.with_name(f"{step_input_dir.name}{current_suffix}")
                    else:
                        # Subsequent steps work in place - use same directory as input
                        step_output_dir = step_input_dir
            else:
                # Last step: Always create output directory with suffix (final results)
                step_name_lower = step_name.lower()
                current_suffix = path_config.output_dir_suffix # Default
                if "position" in step_name_lower:
                    current_suffix = path_config.positions_dir_suffix
                elif "stitch" in step_name_lower:
                    current_suffix = path_config.stitched_dir_suffix

                # For last step, use workspace directory name instead of input directory name
                if hasattr(context, 'workspace_path') and context.workspace_path:
                    workspace_path = Path(context.workspace_path)
                    # Check if global output folder is configured
                    global_output_folder = path_config.global_output_folder
                    if global_output_folder:
                        # Use global output folder: {global_folder}/{workspace_name}{suffix}
                        global_folder = Path(global_output_folder)
                        step_output_dir = global_folder / f"{workspace_path.name}{current_suffix}"
                    else:
                        # Use current behavior: same parent as workspace
                        step_output_dir = workspace_path.with_name(f"{workspace_path.name}{current_suffix}")
                else:
                    step_output_dir = step_input_dir.with_name(f"{step_input_dir.name}{current_suffix}")
                
            # --- Rule: First step must have different input and output ---
            if i == 0 and step_output_dir == step_input_dir:
                # For the first step, always use the general output_dir_suffix if it needs differentiation
                # Use workspace directory name instead of input directory name
                if hasattr(context, 'workspace_path') and context.workspace_path:
                    workspace_path = Path(context.workspace_path)
                    # Check if global output folder is configured
                    global_output_folder = path_config.global_output_folder
                    if global_output_folder:
                        # Use global output folder: {global_folder}/{workspace_name}{suffix}
                        global_folder = Path(global_output_folder)
                        step_output_dir = global_folder / f"{workspace_path.name}{path_config.output_dir_suffix}"
                    else:
                        # Use current behavior: same parent as workspace
                        step_output_dir = workspace_path.with_name(f"{workspace_path.name}{path_config.output_dir_suffix}")
                else:
                    step_output_dir = step_input_dir.with_name(f"{step_input_dir.name}{path_config.output_dir_suffix}")

            # --- Process special I/O ---
            special_outputs = {}
            special_inputs = {}

            # Process special outputs
            if s_outputs_keys: # Use the keys derived from core_callable or step attribute
                for key in sorted(list(s_outputs_keys)): # Iterate over sorted keys
                    snake_case_key = to_snake_case(key)
                    # Path generation updated to [output_dir]/[snake_case_key].pkl
                    output_path = Path(step_output_dir) / f"{snake_case_key}.pkl"
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
                        snake_case_key = to_snake_case(key)
                        output_path = Path(step_output_dir) / f"{snake_case_key}.pkl"
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



            # Create step path info
            step_paths[step_id] = {
                "input_dir": str(step_input_dir),
                "output_dir": str(step_output_dir),
                "pipeline_position": i,
                "special_inputs": special_inputs,
                "special_outputs": special_outputs,
            }

            # Apply chain breaker read backend if needed
            if chain_breaker_read_backend is not None:
                step_paths[step_id][READ_BACKEND] = chain_breaker_read_backend

        # --- Final path connectivity validation after all steps are processed ---
        for i, step in enumerate(steps):
            if i == 0:
                continue  # Skip first step

            curr_step_id = step.step_id
            prev_step_id = steps[i-1].step_id
            curr_step_name = step.name
            prev_step_name = steps[i-1].name

            curr_step_input_dir = step_paths[curr_step_id]["input_dir"]
            prev_step_output_dir = step_paths[prev_step_id]["output_dir"]

            # Check if the PREVIOUS step is a chain breaker
            prev_step = steps[i-1]
            prev_is_chain_breaker_flag_from_plan = False
            if isinstance(prev_step, FunctionStep):
                func_to_check = prev_step.func
                if isinstance(func_to_check, (tuple, list)) and func_to_check:
                    func_to_check = func_to_check[0]
                if callable(func_to_check):
                    prev_is_chain_breaker_flag_from_plan = getattr(func_to_check, '__chain_breaker__', False)

            # Check path connectivity unless the previous step is a chain breaker
            if not prev_is_chain_breaker_flag_from_plan and curr_step_input_dir != prev_step_output_dir:
                # Check if connected through special I/O
                has_special_connection = False
                for _, input_info in step_paths[curr_step_id].get("special_inputs", {}).items(): # key variable renamed to _
                    if input_info["source_step_id"] == prev_step_id:
                        has_special_connection = True
                        break

                if not has_special_connection:
                    raise PlanError(f"Path discontinuity: {prev_step_name} output ({prev_step_output_dir}) doesn't connect to {curr_step_name} input ({curr_step_input_dir})") # Added paths to error

        return step_paths
"""
Pipeline path planning module for OpenHCS.

This module provides the PipelinePathPlanner class, which is responsible for
determining input and output paths for each step in a pipeline in a single pass.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from openhcs.constants.constants import READ_BACKEND, WRITE_BACKEND, Backend
from openhcs.constants.input_source import InputSource
from openhcs.core.config import MaterializationBackend
from openhcs.core.context.processing_context import ProcessingContext # ADDED
from openhcs.core.pipeline.pipeline_utils import get_core_callable
from openhcs.core.pipeline.funcstep_contract_validator import FuncStepContractValidator
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

        # DEBUG: Log initial context values
        logger.info(f"🚀 PATH PLANNER INIT - Context values:")
        logger.info(f"  📂 context.input_dir: {repr(context.input_dir)}")
        logger.info(f"  📂 context.plate_path: {repr(getattr(context, 'plate_path', 'NOT_SET'))}")
        logger.info(f"  📂 context.zarr_conversion_path: {repr(getattr(context, 'zarr_conversion_path', 'NOT_SET'))}")

        if context.zarr_conversion_path:
            # For zarr conversion, use zarr conversion path for calculations
            initial_pipeline_input_dir = Path(context.zarr_conversion_path)
            logger.info(f"  🔄 Using zarr_conversion_path: {repr(initial_pipeline_input_dir)}")
        else:
            # Use actual image directory provided by microscope handler
            initial_pipeline_input_dir = Path(context.input_dir)
            logger.info(f"  🎯 Using input_dir: {repr(initial_pipeline_input_dir)}")

        # NOTE: sub_dir and .zarr are for OUTPUT paths only, not input paths
        # Microscope handler provides the correct input directory

        if not step_plans: # Should be initialized by PipelineCompiler before this call
            raise ValueError("Context step_plans must be initialized before path planning.")
        if not initial_pipeline_input_dir:
            raise ValueError("Context input_dir must be set before path planning.")

        steps = pipeline_definition

        # Transform dict patterns with special outputs before processing (only once)
        logger.info(f"🔍 PATH_PLANNER_CALL: Starting path planning for {len(pipeline_definition)} steps")
        for step in pipeline_definition:
            if isinstance(step, FunctionStep):
                logger.info(f"🔍 STEP_CHECK: Step {step.name} is FunctionStep, func type: {type(step.func)}")
                logger.info(f"🔍 STEP_CHECK: Step {step.name} func value: {step.func}")
                logger.info(f"🔍 STEP_CHECK: Step {step.name} is dict? {isinstance(step.func, dict)}")

            if isinstance(step, FunctionStep) and isinstance(step.func, dict):
                # Dict patterns no longer need function transformation
                # Functions keep their original __special_outputs__
                logger.info(f"🔍 DICT_PATTERN: Processing dict pattern for step {step.name} (no transformation needed)")

        # Modify step_plans in place

        # Track available special outputs by key for validation
        declared_outputs = {}

        # First pass: determine all step output directories
        step_output_dirs = {}

        # Single pass through steps
        for i, step in enumerate(steps):
            step_id = step.step_id
            step_name = step.name

            # --- Determine contract sources ---
            s_outputs_keys: Set[str] = set()
            s_inputs_info: Dict[str, bool] = {}

            if isinstance(step, FunctionStep):
                # For dict patterns, collect special outputs from ALL functions, not just the first
                if isinstance(step.func, dict):
                    all_functions = FuncStepContractValidator._extract_functions_from_pattern(step.func, step.name)
                    s_outputs_keys = set()
                    s_inputs_info = {}
                    # Also collect materialization functions from all functions in dict pattern
                    materialization_functions = {}
                    for func in all_functions:
                        s_outputs_keys.update(getattr(func, '__special_outputs__', set()))
                        s_inputs_info.update(getattr(func, '__special_inputs__', {}))
                        materialization_functions.update(getattr(func, '__materialization_functions__', {}))
                else:
                    # Non-dict pattern - use original logic
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
                elif step.input_dir is not None:
                    step_input_dir = Path(step.input_dir) # User override on step object
                else:
                    step_input_dir = initial_pipeline_input_dir # Fallback to pipeline-level input dir
            else: # Subsequent steps (i > 0)
                if step_id in step_plans and "input_dir" in step_plans[step_id]:
                    step_input_dir = Path(step_plans[step_id]["input_dir"])
                elif step.input_dir is not None:
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

            # --- InputSource strategy resolution ---
            input_source = getattr(step, 'input_source', InputSource.PREVIOUS_STEP)
            pipeline_start_read_backend = None  # Track if this step should use disk backend

            logger.info(f"🔍 INPUT_SOURCE: Step '{step_name}' using strategy: {input_source.value}")

            if input_source == InputSource.PIPELINE_START:
                # Step reads from original pipeline input directory
                original_step_input_dir = step_input_dir
                step_input_dir = Path(initial_pipeline_input_dir)

                # Set VFS backend consistency for pipeline start strategy
                # Use materialization backend from config instead of hardcoded 'disk'
                vfs_config = context.get_vfs_config()
                pipeline_start_read_backend = vfs_config.materialization_backend.value

                logger.info(f"🔍 INPUT_SOURCE: Step '{step_name}' redirected from '{original_step_input_dir}' to pipeline start '{initial_pipeline_input_dir}'")
            elif input_source == InputSource.PREVIOUS_STEP:
                # Standard chaining logic - step_input_dir already set correctly above
                logger.info(f"🔍 INPUT_SOURCE: Step '{step_name}' using previous step output: {step_input_dir}")
            else:
                logger.warning(f"🔍 INPUT_SOURCE: Unknown input source strategy '{input_source}' for step '{step_name}', defaulting to PREVIOUS_STEP")
                
            # --- Process output directory ---
            # Check if step_plans already has this step with output_dir
            if step_id in step_plans and "output_dir" in step_plans[step_id]:
                step_output_dir = Path(step_plans[step_id]["output_dir"])
            elif step.output_dir is not None:
                # Keep output from step kwargs
                step_output_dir = Path(step.output_dir)
            elif i < len(steps) - 1:
                next_step = steps[i+1]
                next_step_id = next_step.step_id
                if next_step_id in step_plans and "input_dir" in step_plans[next_step_id]:
                    # Use next step's input from step_plans
                    step_output_dir = Path(step_plans[next_step_id]["input_dir"])
                elif next_step.input_dir is not None:
                    # Use next step's input from step attribute
                    step_output_dir = Path(next_step.input_dir)
                else:
                    # For first step (i == 0) OR steps using PIPELINE_START, create output directory with suffix
                    # For other subsequent steps (i > 0), work in place (use same directory as input)
                    if i == 0 or input_source == InputSource.PIPELINE_START:
                        # Create output directory with suffix
                        current_suffix = path_config.output_dir_suffix
                        step_output_dir = step_input_dir.with_name(f"{step_input_dir.name}{current_suffix}")
                    else:
                        # Subsequent steps work in place - use same directory as input
                        step_output_dir = step_input_dir
            else:
                # Last step: Work in place - use same directory as input
                step_output_dir = step_input_dir
                
            # --- Rule: First step and pipeline start steps use global output logic ---
            if (i == 0 or input_source == InputSource.PIPELINE_START):
                # For the first step and chain breakers, apply global output folder logic
                # Always use plate_path.name for consistent output naming
                if hasattr(context, 'plate_path') and context.plate_path:
                    plate_path = Path(context.plate_path)

                    # DEBUG: Log detailed path construction info
                    logger.info(f"🔍 PATH PLANNER DEBUG - Step {i} ({step_id}):")
                    logger.info(f"  📁 Raw plate_path: {repr(context.plate_path)}")
                    logger.info(f"  📁 Path object: {repr(plate_path)}")
                    logger.info(f"  📁 plate_path.name: {repr(plate_path.name)}")
                    logger.info(f"  📁 plate_path.name (bytes): {plate_path.name.encode('unicode_escape')}")
                    logger.info(f"  📁 output_dir_suffix: {repr(path_config.output_dir_suffix)}")

                    # Check if global output folder is configured
                    global_output_folder = path_config.global_output_folder
                    logger.info(f"  🌍 global_output_folder (raw): {repr(global_output_folder)}")

                    # Clean global output folder path - strip whitespace and newlines
                    if global_output_folder:
                        global_output_folder = global_output_folder.strip()
                        logger.info(f"  🧹 global_output_folder (cleaned): {repr(global_output_folder)}")

                    # Build base output name
                    output_name = f"{plate_path.name}{path_config.output_dir_suffix}"
                    output_path = Path(output_name)

                    # Apply sub_dir if configured
                    if path_config.sub_dir:
                        output_path = output_path / path_config.sub_dir
                        logger.info(f"  📁 Applied sub_dir: {repr(output_path)}")

                    # Add .zarr to the final component if using zarr backend
                    vfs_config = context.get_vfs_config()
                    if vfs_config.materialization_backend == MaterializationBackend.ZARR:
                        output_path = output_path.with_suffix('.zarr')
                        logger.info(f"  🗃️  Added .zarr suffix: {repr(output_path)}")

                    if global_output_folder:
                        # Use global output folder
                        global_folder = Path(global_output_folder)
                        step_output_dir = global_folder / output_path
                        logger.info(f"  ✅ Final output_dir (global): {repr(step_output_dir)}")
                    else:
                        # Use plate parent directory
                        step_output_dir = plate_path.parent / output_path
                        logger.info(f"  ✅ Final output_dir (local): {repr(step_output_dir)}")
                else:
                    # Fallback to input directory name if plate_path not available
                    logger.info(f"🔍 PATH PLANNER DEBUG - Step {i} ({step_id}) - FALLBACK:")
                    logger.info(f"  📁 No plate_path, using step_input_dir: {repr(step_input_dir)}")
                    logger.info(f"  📁 step_input_dir.name: {repr(step_input_dir.name)}")
                    constructed_name = f"{step_input_dir.name}{path_config.output_dir_suffix}"
                    logger.info(f"  🔧 Constructed name: {repr(constructed_name)}")
                    step_output_dir = step_input_dir.with_name(constructed_name)
                    logger.info(f"  ✅ Final output_dir (fallback): {repr(step_output_dir)}")

            # Store the output directory for this step
            step_output_dirs[step_id] = step_output_dir

            # --- Process special I/O ---
            special_outputs = {}
            special_inputs = {}

            # Process special outputs
            if s_outputs_keys: # Use the keys derived from core_callable or step attribute
                # Determine final output directory (last step's output directory)
                final_output_dir = None
                if len(steps) > 0:
                    last_step_id = steps[-1].step_id
                    if last_step_id in step_output_dirs:
                        final_output_dir = step_output_dirs[last_step_id]
                    elif i == len(steps) - 1:  # This is the last step
                        final_output_dir = step_output_dir

                # Get materialization results path from config
                results_base_path = PipelinePathPlanner._resolve_materialization_results_path(path_config, context, final_output_dir)

                # Extract materialization functions from decorator (if FunctionStep)
                # For dict patterns, materialization_functions was already collected above
                # For non-dict patterns, extract from core_callable
                if isinstance(step, FunctionStep):
                    if not isinstance(step.func, dict):  # Non-dict pattern
                        materialization_functions = {}
                        if core_callable:
                            materialization_functions = getattr(core_callable, '__materialization_functions__', {})
                    # For dict patterns, materialization_functions was already set above

                for key in sorted(list(s_outputs_keys)): # Iterate over sorted keys
                    # Build path using materialization results config
                    filename = f"{well_id}_{key}.pkl"
                    output_path = Path(results_base_path) / filename

                    # Get materialization function for this key
                    mat_func = materialization_functions.get(key)

                    special_outputs[key] = {
                        "path": str(output_path),
                        "materialization_function": mat_func
                    }
                    # Register this output for future steps
                    declared_outputs[key] = {
                        "step_id": step_id,
                        "position": i,
                        "path": str(output_path)
                    }

            # Apply scope promotion rules for dict patterns
            if isinstance(step, FunctionStep) and isinstance(step.func, dict):
                special_outputs, declared_outputs = _apply_scope_promotion_rules(
                    step.func, special_outputs, declared_outputs, step_id, i
                )

            # Generate funcplan for execution
            funcplan = _generate_funcplan(step, special_outputs)
                
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
                "input_source": input_source.value,  # Store input source strategy for debugging
                "special_inputs": special_inputs,
                "special_outputs": special_outputs,
                "funcplan": funcplan,
            })

            # Apply pipeline start read backend if needed
            if pipeline_start_read_backend is not None:
                step_plans[step_id][READ_BACKEND] = pipeline_start_read_backend

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
                # First step always uses disk backend for input (literal directory creation)
                context.filemanager.ensure_directory(step_input_dir, Backend.DISK.value)
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

            # Check if the CURRENT step uses PIPELINE_START input source
            curr_step = steps[i]
            curr_step_input_source = getattr(curr_step, 'input_source', InputSource.PREVIOUS_STEP)

            # Check path connectivity unless the current step uses PIPELINE_START
            if curr_step_input_source != InputSource.PIPELINE_START and curr_step_input_dir != prev_step_output_dir:
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
            # Create zarr store inside the original plate directory
            path_config = context.get_path_planning_config()
            zarr_store_path = Path(context.zarr_conversion_path) / f"{path_config.sub_dir}.zarr"
            step_plans[first_step_id]['convert_to_zarr'] = str(zarr_store_path)
            logger.info(f"Zarr conversion: first step reads from {context.original_input_dir}, converts to {zarr_store_path}")

        # === FIRST STEP INPUT OVERRIDE ===
        # No longer needed - we now use actual input_dir from the start

        # === SET OUTPUT PLATE ROOT IN CONTEXT ===
        # Determine output plate root from first step's output directory
        if steps and step_output_dirs:
            first_step_id = steps[0].step_id
            if first_step_id in step_output_dirs:
                first_step_output = step_output_dirs[first_step_id]
                context.output_plate_root = PipelinePathPlanner.resolve_output_plate_root(first_step_output, path_config)

        return step_plans

    @staticmethod
    def _resolve_materialization_results_path(path_config, context, final_output_dir=None):
        """Resolve materialization results path from config."""
        results_path = path_config.materialization_results_path

        if not Path(results_path).is_absolute():
            # Use final output directory as base instead of plate_path
            if final_output_dir:
                base_folder = Path(final_output_dir)
            else:
                # Fallback to plate_path if final_output_dir not available
                base_folder = Path(context.plate_path)
            return str(base_folder / results_path)
        else:
            return results_path

    @staticmethod
    def resolve_output_plate_root(step_output_dir: Union[str, Path], path_config) -> Path:
        """
        Resolve output plate root directory from step output directory.

        Args:
            step_output_dir: Step's output directory
            path_config: PathPlanningConfig with sub_dir

        Returns:
            Output plate root directory
        """
        step_output_path = Path(step_output_dir)

        if not path_config.sub_dir:
            return step_output_path

        # Remove sub_dir component: if path ends with sub_dir(.zarr), return parent
        if step_output_path.name in (path_config.sub_dir, f"{path_config.sub_dir}.zarr"):
            return step_output_path.parent

        return step_output_path











def _has_special_outputs(func_or_tuple):
    """
    Check if a function or tuple contains a function with special outputs.

    Follows the pattern from get_core_callable() for extracting functions from patterns.
    """
    if isinstance(func_or_tuple, tuple) and len(func_or_tuple) >= 1:
        # Check the function part of (function, kwargs) tuple
        func = func_or_tuple[0]
        return callable(func) and not isinstance(func, type) and hasattr(func, '__special_outputs__')
    elif callable(func_or_tuple) and not isinstance(func_or_tuple, type):
        return hasattr(func_or_tuple, '__special_outputs__')
    else:
        return False


def _apply_scope_promotion_rules(dict_pattern, special_outputs, declared_outputs, step_id, step_position):
    """
    Apply scope promotion rules for dict pattern special outputs.

    Rules:
    - Single-key dict patterns: Promote to global scope (DAPI_0_positions → positions)
    - Multi-key dict patterns: Keep namespaced (DAPI_0_positions, GFP_0_positions)

    Args:
        dict_pattern: The dict pattern from the step
        special_outputs: Current special outputs dict
        declared_outputs: Global declared outputs dict
        step_id: Current step ID
        step_position: Current step position

    Returns:
        tuple: (updated_special_outputs, updated_declared_outputs)
    """
    import copy

    # Only apply promotion for single-key dict patterns
    if len(dict_pattern) != 1:
        logger.debug(f"🔍 SCOPE_PROMOTION: Multi-key dict pattern ({len(dict_pattern)} keys), keeping namespaced outputs")
        return special_outputs, declared_outputs

    # Get the single dict key
    dict_key = list(dict_pattern.keys())[0]
    logger.debug(f"🔍 SCOPE_PROMOTION: Single-key dict pattern with key '{dict_key}', applying promotion rules")

    # Create copies to avoid modifying originals
    promoted_special_outputs = copy.deepcopy(special_outputs)
    promoted_declared_outputs = copy.deepcopy(declared_outputs)

    # Find namespaced outputs that should be promoted
    outputs_to_promote = []
    for output_key in list(special_outputs.keys()):
        # Check if this is a namespaced output from our dict key
        if output_key.startswith(f"{dict_key}_0_"):  # Single functions have chain position 0
            original_key = output_key[len(f"{dict_key}_0_"):]  # Extract original key
            outputs_to_promote.append((output_key, original_key))

    # Apply promotions
    for namespaced_key, promoted_key in outputs_to_promote:
        logger.debug(f"🔍 SCOPE_PROMOTION: Promoting {namespaced_key} → {promoted_key}")

        # Check for collisions with existing promoted outputs
        if promoted_key in promoted_declared_outputs:
            existing_step = promoted_declared_outputs[promoted_key]["step_id"]
            raise PlanError(
                f"Scope promotion collision: Step '{step_id}' wants to promote '{namespaced_key}' → '{promoted_key}', "
                f"but step '{existing_step}' already produces '{promoted_key}'. "
                f"Use explicit special output naming to resolve this conflict."
            )

        # Add promoted output to special_outputs
        promoted_special_outputs[promoted_key] = special_outputs[namespaced_key]

        # Add promoted output to declared_outputs
        promoted_declared_outputs[promoted_key] = {
            "step_id": step_id,
            "position": step_position,
            "path": special_outputs[namespaced_key]["path"]
        }

        # Keep the namespaced version as well for materialization
        # (materialization system can handle both)

    logger.debug(f"🔍 SCOPE_PROMOTION: Promoted {len(outputs_to_promote)} outputs for single-key dict pattern")
    return promoted_special_outputs, promoted_declared_outputs


def _generate_funcplan(step, special_outputs):
    """
    Generate funcplan mapping for execution.

    Maps function execution contexts to their outputs_to_save.

    Args:
        step: The step being processed
        special_outputs: Dict of special outputs for this step

    Returns:
        Dict mapping execution_key -> outputs_to_save list
    """
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.core.pipeline.pipeline_utils import get_core_callable

    funcplan = {}

    if not isinstance(step, FunctionStep):
        return funcplan

    if not special_outputs:
        return funcplan

    # Extract all functions from the pattern
    all_functions = []

    if isinstance(step.func, dict):
        # Dict pattern: {'DAPI': func, 'GFP': [func1, func2]}
        for dict_key, func_or_list in step.func.items():
            if isinstance(func_or_list, list):
                # Chain in dict pattern
                for chain_position, func_item in enumerate(func_or_list):
                    func_callable = get_core_callable(func_item)
                    if func_callable and hasattr(func_callable, '__special_outputs__'):
                        execution_key = f"{func_callable.__name__}_{dict_key}_{chain_position}"
                        func_outputs = func_callable.__special_outputs__
                        # Find which step outputs this function should save
                        outputs_to_save = [key for key in special_outputs.keys() if key in func_outputs]
                        if outputs_to_save:
                            funcplan[execution_key] = outputs_to_save
                            logger.debug(f"🔍 FUNCPLAN: {execution_key} -> {outputs_to_save}")
            else:
                # Single function in dict pattern
                func_callable = get_core_callable(func_or_list)
                if func_callable and hasattr(func_callable, '__special_outputs__'):
                    execution_key = f"{func_callable.__name__}_{dict_key}_0"
                    func_outputs = func_callable.__special_outputs__
                    # Find which step outputs this function should save
                    outputs_to_save = [key for key in special_outputs.keys() if key in func_outputs]
                    if outputs_to_save:
                        funcplan[execution_key] = outputs_to_save
                        logger.debug(f"🔍 FUNCPLAN: {execution_key} -> {outputs_to_save}")

    elif isinstance(step.func, list):
        # Chain pattern: [func1, func2]
        for chain_position, func_item in enumerate(step.func):
            func_callable = get_core_callable(func_item)
            if func_callable and hasattr(func_callable, '__special_outputs__'):
                execution_key = f"{func_callable.__name__}_default_{chain_position}"
                func_outputs = func_callable.__special_outputs__
                # Find which step outputs this function should save
                outputs_to_save = [key for key in special_outputs.keys() if key in func_outputs]
                if outputs_to_save:
                    funcplan[execution_key] = outputs_to_save
                    logger.debug(f"🔍 FUNCPLAN: {execution_key} -> {outputs_to_save}")

    else:
        # Single function pattern
        func_callable = get_core_callable(step.func)
        if func_callable and hasattr(func_callable, '__special_outputs__'):
            execution_key = f"{func_callable.__name__}_default_0"
            func_outputs = func_callable.__special_outputs__
            # Find which step outputs this function should save
            outputs_to_save = [key for key in special_outputs.keys() if key in func_outputs]
            if outputs_to_save:
                funcplan[execution_key] = outputs_to_save
                logger.debug(f"🔍 FUNCPLAN: {execution_key} -> {outputs_to_save}")

    logger.info(f"🔍 FUNCPLAN: Generated funcplan with {len(funcplan)} entries for step {step.name}")
    return funcplan
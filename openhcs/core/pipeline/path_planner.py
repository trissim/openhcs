"""
Pipeline path planning module for OpenHCS.

This module provides the PipelinePathPlanner class, which is responsible for
determining input and output paths for each step in a pipeline in a single pass.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

# DEFAULT_OUT_DIR_SUFFIX removed
from openhcs.core.context.processing_context import ProcessingContext # ADDED
from openhcs.core.pipeline.pipeline_utils import get_core_callable, to_snake_case
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep


logger = logging.getLogger(__name__)

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
        first_step_input: Optional[Path] = None # Added type hint
    
        # Get base input dir from first step if available
        if steps and len(steps) > 0:
            first_step_instance = steps[0]
            if first_step_instance.uid in step_paths and "input_dir" in step_paths[first_step_instance.uid]:
                 first_step_input = Path(step_paths[first_step_instance.uid]["input_dir"])
    
        # Single pass through steps
        for i, step in enumerate(steps):
            step_id = step.uid
            step_name = step.name

            # --- Determine contract sources ---
            s_outputs_keys: Set[str] = set()
            s_inputs_info: Dict[str, bool] = {}
            is_cb: bool = False

            if isinstance(step, FunctionStep):
                core_callable = get_core_callable(step.func)
                if core_callable:
                    s_outputs_keys = getattr(core_callable, '__special_outputs__', set())
                    s_inputs_info = getattr(core_callable, '__special_inputs__', {})
                    is_cb = getattr(core_callable, '__chain_breaker__', False)
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
                        prev_step_id = prev_step.uid
                        if prev_step_id in step_paths and "output_dir" in step_paths[prev_step_id]:
                            step_input_dir = Path(step_paths[prev_step_id]["output_dir"])
                        else:
                            # This should ideally not be reached if previous steps always have output_dir
                            raise ValueError(f"Previous step {prev_step.name} (ID: {prev_step_id}) has no output_dir in step_plans.")
                
                # Save first step's actual input_dir for potential use by chain breakers
                if i == 0:
                    first_step_input = step_input_dir
                
                # --- Process output directory ---
                # Check if step_paths already has this step with output_dir
                if step_id in step_paths and "output_dir" in step_paths[step_id]:
                    step_output_dir = Path(step_paths[step_id]["output_dir"])
                elif hasattr(step, "output_dir") and step.output_dir is not None:
                    # Keep output from step kwargs
                    step_output_dir = Path(step.output_dir)
                elif i < len(steps) - 1:
                    next_step = steps[i+1]
                    next_step_id = next_step.uid
                    if next_step_id in step_paths and "input_dir" in step_paths[next_step_id]:
                        # Use next step's input from step_plans
                        step_output_dir = Path(step_paths[next_step_id]["input_dir"])
                    elif hasattr(next_step, "input_dir") and next_step.input_dir is not None:
                        # Use next step's input from step attribute
                        step_output_dir = Path(next_step.input_dir)
                    else:
                        # Use same directory as input with appropriate suffix based on step name
                        step_name_lower = step_name.lower()
                        current_suffix = path_config.output_dir_suffix # Default
                        if "position" in step_name_lower:
                            current_suffix = path_config.positions_dir_suffix
                        elif "stitch" in step_name_lower:
                            current_suffix = path_config.stitched_dir_suffix
                        step_output_dir = step_input_dir.with_name(f"{step_input_dir.name}{current_suffix}")
                else:
                    # Last step uses input directory with appropriate suffix
                    step_name_lower = step_name.lower()
                    current_suffix = path_config.output_dir_suffix # Default
                    if "position" in step_name_lower:
                        current_suffix = path_config.positions_dir_suffix
                    elif "stitch" in step_name_lower:
                        current_suffix = path_config.stitched_dir_suffix
                    step_output_dir = step_input_dir.with_name(f"{step_input_dir.name}{current_suffix}")
                
                # --- Rule: First step must have different input and output ---
                if i == 0 and step_output_dir == step_input_dir:
                    # For the first step, always use the general output_dir_suffix if it needs differentiation
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
                if s_inputs_info: # Use the info derived from core_callable or step attribute
                    for key in sorted(list(s_inputs_info.keys())): # Iterate over sorted keys
                        # Validate special input exists from earlier step
                        if key not in declared_outputs:
                            raise PlanError(f"Step '{step_name}' requires special input '{key}', but no upstream step produces it")
                        
                        producer = declared_outputs[key]
                        # Validate producer comes before consumer
                        if producer["position"] >= i:
                            producer_step_name = steps[producer["position"]].name # Ensure 'steps' is the pipeline_definition list
                            raise PlanError(f"Step '{step_name}' cannot consume special input '{key}' from later step '{producer_step_name}'")
                        
                        special_inputs[key] = {
                            "path": producer["path"],
                            "source_step_id": producer["step_id"]
                        }
                
                # --- Process chain breaker ---
                # Use the 'is_cb' flag derived earlier
                if is_cb and i > 0:
                    # If this is a chain breaker and not the first step,
                    # the next step's input should be the first step's input
                    if i < len(steps) - 1:
                        # Find the next step
                        # next_step = steps[i+1] # Not used directly here
                        # Set this step's output to match first step's input
                        # This logic ensures the next step's input_dir will be first_step_input
                        # if it normally chains from this step's output_dir.
                        if first_step_input is not None: # Ensure first_step_input was set
                            step_output_dir = first_step_input
                        else:
                            # This case should ideally not happen if pipeline has >1 steps and first_step_input is always captured
                            logger.warning(f"Chain breaker step {step_name} encountered but first_step_input is not available.")


                # Create step path info
                step_paths[step_id] = {
                    "input_dir": str(step_input_dir),
                    "output_dir": str(step_output_dir),
                    "pipeline_position": i,
                    "special_inputs": special_inputs,
                    "special_outputs": special_outputs,
                    "chainbreaker": is_cb # Store the chainbreaker status
                }
            
            # --- Final path connectivity validation in the same pass ---
            for i, step in enumerate(steps):
                if i == 0:
                    continue  # Skip first step
                    
                curr_step_id = step.uid
                prev_step_id = steps[i-1].uid
                curr_step_name = step.name
                prev_step_name = steps[i-1].name
                
                curr_step_input_dir = step_paths[curr_step_id]["input_dir"]
                prev_step_output_dir = step_paths[prev_step_id]["output_dir"]
                
                # Check if this step is a chain breaker using the stored flag
                # is_chain_breaker_flag_from_plan = hasattr(step, "chain_breaker") and step.chain_breaker # Old way
                is_chain_breaker_flag_from_plan = step_paths[curr_step_id].get("chainbreaker", False) # New way

                # Check path connectivity unless it's a chain breaker
                if not is_chain_breaker_flag_from_plan and curr_step_input_dir != prev_step_output_dir:
                    # Check if connected through special I/O
                    has_special_connection = False
                    for _, input_info in step_paths[curr_step_id].get("special_inputs", {}).items(): # key variable renamed to _
                        if input_info["source_step_id"] == prev_step_id:
                            has_special_connection = True
                            break
                            
                    if not has_special_connection:
                        raise PlanError(f"Path discontinuity: {prev_step_name} output ({prev_step_output_dir}) doesn't connect to {curr_step_name} input ({curr_step_input_dir})") # Added paths to error
            
            return step_paths
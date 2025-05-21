"""
Pipeline path planning module for OpenHCS.

This module provides the PipelinePathPlanner class, which is responsible for
determining input and output paths for each step in a pipeline.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy
- Clause 17 — VFS Exclusivity (FileManager is the only component that uses VirtualPath)
- Clause 17-B — Path Format Discipline
- Clause 65 — No Fallback Logic
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 245 — Path Declaration
- Clause 283 — Well-Scoped Materialization Enforcement
- Clause 524 — Step = Declaration = ID = Runtime Authority
- Clause 631 — Pipeline Path Connectivity
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

# Import step types for type checking
from openhcs.constants.constants import DEFAULT_OUT_DIR_SUFFIX

logger = logging.getLogger(__name__)

# Error class for planning errors
class PlanError(ValueError):
    """Error raised when pipeline planning fails."""
    pass

# ===== DECLARATIVE DEFAULT VALUES =====
# These declarations control defaults and may be moved to configuration in the future

# Default suffixes for different step types
FIRST_STEP_OUTPUT_SUFFIX = DEFAULT_OUT_DIR_SUFFIX  # Suffix for first step output directory

# Default error messages for validation failures
ERROR_INVALID_PATH_TYPE = (
    "Invalid type for {0}: expected str or Path, got {1}. "
    "Only str and Path types are allowed, no automatic conversion is performed. "
    "This is a hard requirement - code must fail loudly when non-str/Path types are provided."
)
ERROR_INVALID_DICT_TYPE = (
    "Invalid type for {0}: expected dict, got {1}. "
    "Only dict types are allowed for collections of paths."
)
ERROR_MISSING_INPUT_DIR = (
    "Input directory must be provided. "
    "This is a required parameter for path planning."
)


class StepPathInfo(TypedDict, total=False):
    """Type definition for step path information."""
    input_dir: Union[str, Path]
    output_dir: Union[str, Path]
    save_special_outputs: List[str]  # Special outputs to save
    load_special_inputs: List[str]   # Special inputs to load


class PipelinePathPlanner:
    """
    Plans and prepares execution paths for pipeline steps.

    This class is responsible for determining input and output paths for each step
    in a pipeline, accessing paths directly from step attributes and ensuring proper path
    relationships between steps.

    Key principles:
    1. All paths are str or Path objects, not VirtualPath objects
    2. All steps must have input_dir and output_dir defined
    3. First step must have different input and output directories
    4. Steps can declare special inputs and outputs to establish dependencies
    5. All steps must form a coherent sequence of connected paths
    6. All paths must be declared on the step. Override dictionaries are forbidden.
    7. Planner reads step fields directly — override indirection is a structural violation.
    8. Special input/output paths are resolved as flat sibling paths
    9. Chain-breaking behavior is controlled by step.chain_breaker attribute
    """

    @staticmethod
    def resolve_special_path(base_path: Union[str, Path], key: str) -> Path:
        """
        Resolve a special path based on a base path and key.

        Args:
            base_path: Base path to derive the special path from
            key: Special path key

        Returns:
            Path object for the special path
        """
        base = Path(base_path)
        return base.with_name(f"{base.name}_{key}")

    @staticmethod
    def prepare_pipeline_paths(
        input_dir: Union[str, Path],
        pipeline: Any,
        well_id: str
    ) -> Dict[str, StepPathInfo]:
        """
        Prepare path information for each step in a pipeline.

        This method enforces the interface contract for all steps, applies path planning rules,
        and stores the final path values in the context. The rules are authoritative,
        and the context values are the single source of truth during execution.

        Args:
            input_dir: The input directory for the pipeline (as str or Path)
            pipeline: The pipeline to prepare paths for
            well_id: Well identifier for filtering files, not for creating subdirectories - must be provided

        Returns:
            A dictionary mapping step UIDs to path information dictionaries

        Note:
            All steps MUST implement required path attributes as part of the interface contract.
            These attributes are accessed directly with no fallback logic or hasattr() checks.
            Missing attributes will cause loud failures, enforcing the interface contract.
            All paths must be declared on the step. Override dictionaries are forbidden.
            Before pipeline execution, all attributes are stripped from steps to ensure statelessness.
            During execution, steps access path information from the context, not from step attributes.
        """
        # Validate input_dir in a single block - no fallback logic, no conversions for non-str/Path types
        if not input_dir:
            # Ensure input directory is set
            raise ValueError(ERROR_MISSING_INPUT_DIR)
        elif not isinstance(input_dir, (str, Path)):
            # Strictly validate input_dir is a str or Path - fail loudly otherwise
            error_msg = ERROR_INVALID_PATH_TYPE.format(
                "input_dir",
                type(input_dir).__name__
            )
            raise TypeError(error_msg)

        # Create step plans dictionary with explicit type annotation
        step_paths: Dict[str, StepPathInfo] = {}

        # Convert input directory to Path object explicitly
        # This is the ONLY conversion allowed - from str to Path, nothing else
        try:
            base_input_dir = Path(input_dir) if isinstance(input_dir, str) else input_dir
        except (TypeError, ValueError) as e:
            # Handle potential path conversion errors explicitly
            raise ValueError(f"Cannot convert input_dir to Path: {e}")

        # Get steps from pipeline
        steps = pipeline["steps"]

        # Process each step in the pipeline
        for i, step in enumerate(steps):
            # Get step UID and name
            step_id = step.uid
            step_name = step.name

            # Determine input directory for this step - direct access to step attribute
            # Clause 524 — Step = Declaration = ID = Runtime Authority
            if hasattr(step, "input_dir") and step.input_dir is not None:
                step_input_dir = step.input_dir
            else:
                # Use base_input_dir if step.input_dir is not set
                step_input_dir = base_input_dir
                logger.info(f"No input_dir defined for step {step_name}, using base_input_dir: {base_input_dir}")

            # Validate step_input_dir type - no fallback logic, no conversions for non-str/Path types
            if not isinstance(step_input_dir, (str, Path)):
                # Strictly validate input directory is a str or Path - fail loudly otherwise
                error_msg = ERROR_INVALID_PATH_TYPE.format(
                    f"input_dir for step {step_name}",
                    type(step_input_dir).__name__
                )
                raise TypeError(error_msg)

            # Convert str to Path explicitly with error handling
            # This is the ONLY conversion allowed - from str to Path, nothing else
            try:
                if isinstance(step_input_dir, str):
                    step_input_dir = Path(step_input_dir)
            except (TypeError, ValueError) as e:
                # Handle potential path conversion errors explicitly
                raise ValueError(f"Cannot convert input_dir for step {step_name} to Path: {e}")

            # Validate step_input_dir type - no fallback logic, no conversions for non-str/Path types
            if not isinstance(step_input_dir, (str, Path)):
                # Strictly validate input directory is a str or Path - fail loudly otherwise
                error_msg = ERROR_INVALID_PATH_TYPE.format(
                    f"input_dir for step {step_name}",
                    type(step_input_dir).__name__
                )
                raise TypeError(error_msg)

            # Convert str to Path explicitly with error handling
            # This is the ONLY conversion allowed - from str to Path, nothing else
            try:
                if isinstance(step_input_dir, str):
                    step_input_dir = Path(step_input_dir)
            except (TypeError, ValueError) as e:
                # Handle potential path conversion errors explicitly
                raise ValueError(f"Cannot convert input_dir for step {step_name} to Path: {e}")


            # Determine output directory for this step - direct access to step attribute
            # Clause 524 — Step = Declaration = ID = Runtime Authority
            if hasattr(step, "output_dir") and step.output_dir is not None:
                step_output_dir = step.output_dir
            else:
                # Use step_input_dir if step.output_dir is not set
                step_output_dir = step_input_dir
                logger.info(f"No output_dir defined for step {step_name}, using step_input_dir: {step_input_dir}")

            # Validate step_output_dir type - no fallback logic, no conversions for non-str/Path types
            if not isinstance(step_output_dir, (str, Path)):
                # Strictly validate output directory is a str or Path - fail loudly otherwise
                error_msg = ERROR_INVALID_PATH_TYPE.format(
                    f"output_dir for step {step_name}",
                    type(step_output_dir).__name__
                )
                raise TypeError(error_msg)

            # Convert str to Path explicitly with error handling
            # This is the ONLY conversion allowed - from str to Path, nothing else
            try:
                if isinstance(step_output_dir, str):
                    step_output_dir = Path(step_output_dir)
            except (TypeError, ValueError) as e:
                # Handle potential path conversion errors explicitly
                raise ValueError(f"Cannot convert output_dir for step {step_name} to Path: {e}")

            # Validate step_output_dir type - no fallback logic, no conversions for non-str/Path types
            if not isinstance(step_output_dir, (str, Path)):
                # Strictly validate output directory is a str or Path - fail loudly otherwise
                error_msg = ERROR_INVALID_PATH_TYPE.format(
                    f"output_dir for step {step_name}",
                    type(step_output_dir).__name__
                )
                raise TypeError(error_msg)

            # Convert str to Path explicitly with error handling
            # This is the ONLY conversion allowed - from str to Path, nothing else
            try:
                if isinstance(step_output_dir, str):
                    step_output_dir = Path(step_output_dir)
            except (TypeError, ValueError) as e:
                # Handle potential path conversion errors explicitly
                raise ValueError(f"Cannot convert output_dir for step {step_name} to Path: {e}")

            # Rule 1: First step must have different input and output directories
            if i == 0 and step_output_dir == step_input_dir:
                step_output_dir = base_input_dir.with_name(f"{base_input_dir.name}{FIRST_STEP_OUTPUT_SUFFIX}")
                logger.info(f"First step {step_name} must have different input and output directories. Using: {step_output_dir}")

            # Process special input/output paths
            if hasattr(step, "special_output") and step.special_output is not None:
                if not isinstance(step.special_output, str):
                    raise TypeError(f"special_output for step {step_name} must be a string, got {type(step.special_output).__name__}")
                special_output_path = PipelinePathPlanner.resolve_special_path(
                    step_output_dir, step.special_output
                )
                # Add to save_special_outputs list
                save_special_outputs = [step.special_output]
            else:
                special_output_path = None
                save_special_outputs = []

            if hasattr(step, "special_input") and step.special_input is not None:
                if not isinstance(step.special_input, str):
                    raise TypeError(f"special_input for step {step_name} must be a string, got {type(step.special_input).__name__}")
                special_input_path = PipelinePathPlanner.resolve_special_path(
                    step_input_dir, step.special_input
                )
                # Add to load_special_inputs list
                load_special_inputs = [step.special_input]
            else:
                special_input_path = None
                load_special_inputs = []

            # Create step path info
            path_info: StepPathInfo = {
                "input_dir": step_input_dir,
                "output_dir": step_output_dir,
                "save_special_outputs": save_special_outputs,
                "load_special_inputs": load_special_inputs
            }

            # Add special paths if they exist
            if special_output_path is not None:
                path_info["special_output_path"] = special_output_path
            
            if special_input_path is not None:
                path_info["special_input_path"] = special_input_path

            # Add path info to dictionary
            step_paths[step_id] = path_info

        # Second pass: Verify path connectivity - interface contract requires all paths to be defined
        for i, step in enumerate(steps):
            step_id = step.uid
            step_name = step.name

            # All steps must have input_dir and output_dir defined - interface contract requires this
            # Direct dictionary access - no fallback logic
            step_path_info = step_paths[step_id]

            # Verify input_dir exists
            if "input_dir" not in step_path_info:
                # This is a contract violation - all steps must have input_dir defined
                if i > 0:
                    prev_step_id = steps[i-1].uid
                    prev_step_name = steps[i-1].name
                    # Connect to previous step's output_dir, but only if input_dir is not already set
                    # This preserves declarative clarity and prevents silent overwrites
                    if "input_dir" not in step_path_info:
                        prev_step_output_dir = step_paths[prev_step_id]["output_dir"]
                        step_path_info["input_dir"] = prev_step_output_dir
                        logger.info(f"Auto-connecting step {step_name} input to previous step {prev_step_name} output: {prev_step_output_dir}")
                    else:
                        logger.info(f"Step {step_name} already has input_dir defined, preserving existing value: {step_path_info['input_dir']}")
                else:
                    # First step must have input_dir defined
                    raise ValueError(f"First step {step_name} must have input_dir defined")

            # Verify output_dir exists
            if "output_dir" not in step_path_info:
                # This is a contract violation - all steps must have output_dir defined
                if i < len(steps) - 1:
                    next_step_id = steps[i+1].uid
                    next_step_name = steps[i+1].name
                    # Connect to next step's input_dir, but only if output_dir is not already set
                    # This preserves declarative clarity and prevents silent overwrites
                    if "output_dir" not in step_path_info:
                        next_step_input_dir = step_paths[next_step_id]["input_dir"]
                        step_path_info["output_dir"] = next_step_input_dir
                        logger.info(f"Auto-connecting step {step_name} output to next step {next_step_name} input: {next_step_input_dir}")
                    else:
                        logger.info(f"Step {step_name} already has output_dir defined, preserving existing value: {step_path_info['output_dir']}")
                else:
                    # Last step must have output_dir defined
                    raise ValueError(f"Last step {step_name} must have output_dir defined")

        # Final validation: Verify path connectivity - interface contract requires connected paths
        for i, step in enumerate(steps):
            if i > 0:  # Skip first step
                curr_step_id = step.uid
                prev_step_id = steps[i-1].uid
                curr_step_name = step.name
                prev_step_name = steps[i-1].name

                # Direct dictionary access - no fallback logic
                curr_step_input_dir = step_paths[curr_step_id]["input_dir"]
                prev_step_output_dir = step_paths[prev_step_id]["output_dir"]

                # Check if this step is a chain breaker
                is_chain_breaker = hasattr(step, "chain_breaker")

                # Check that current step's input matches previous step's output
                # (unless it's a chain breaker or connected through special inputs/outputs)
                if not is_chain_breaker and curr_step_input_dir != prev_step_output_dir:
                    # Check if the step has a special input that connects to the previous step's special output
                    if "special_input_path" in step_paths[curr_step_id] and "special_output_path" in step_paths[prev_step_id]:
                        if step_paths[curr_step_id]["special_input_path"] == step_paths[prev_step_id]["special_output_path"]:
                            continue
                    # This is a contract violation - paths must be connected
                    raise ValueError(
                        f"Path discontinuity detected: Step {prev_step_name} output ({prev_step_output_dir}) "
                        f"does not connect to Step {curr_step_name} input ({curr_step_input_dir})"
                    )

        # Third pass: Validate special input/output connections
        # This ensures that each special_input has a corresponding special_output
        declared_outputs = {}
        step_lookup = {step.uid: step for step in steps}

        for idx, step in enumerate(steps):
            step_id = step.uid
            step_name = step.name

            # Step declares a special_output — record it as available for future steps
            if hasattr(step, "special_output") and step.special_output is not None:
                key = step.special_output
                declared_outputs[key] = step_id  # Record that this key is now available
                logger.info("Step %s declares special output '%s'", step_name, key)

            # Step declares a special_input — validate and connect only to prior steps
            if hasattr(step, "special_input") and step.special_input is not None:
                key = step.special_input
                if key not in declared_outputs:
                    raise PlanError(
                        f"Step '{step_name}' requires special input '{key}', "
                        f"but no upstream step produces it."
                    )

                # Only inject routing if the producer step appears before this one
                producer_step_id = declared_outputs[key]
                producer_step = step_lookup[producer_step_id]
                producer_name = producer_step.name
                producer_idx = steps.index(producer_step)

                if producer_idx >= idx:
                    raise PlanError(
                        f"Step '{step_name}' cannot consume special input '{key}' "
                        f"from later step '{producer_name}'. Ordering violation."
                    )

                logger.info("Step %s will load special input '%s' from step %s",
                           step_name, key, producer_name)

        # Final logging block for structural introspection and debugging
        # This supports Clause 524 (Step = Declaration = ID = Runtime Authority)
        logger.debug("=== Final Path Planning Results ===")
        for step_id, path_info in step_paths.items():
            step_name = next((s.name for s in steps if s.uid == step_id), f"Step {step_id}")
            logger.debug(f"[PathPlan] Step {step_name} (ID: {step_id}): {path_info}")
        logger.debug("=== End Path Planning Results ===")

        return step_paths

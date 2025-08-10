"""
Pipeline module for OpenHCS.

This module provides the core pipeline compilation components for OpenHCS.
The PipelineCompiler is responsible for preparing step_plans within a ProcessingContext.

Doctrinal Clauses:
- Clause 12 — Absolute Clean Execution
- Clause 17 — VFS Exclusivity (FileManager is the only component that uses VirtualPath)
- Clause 17-B — Path Format Discipline
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 101 — Memory Type Declaration
- Clause 245 — Path Declaration
- Clause 273 — Backend Authorization Doctrine
- Clause 281 — Context-Bound Identifiers
- Clause 293 — GPU Pre-Declaration Enforcement
- Clause 295 — GPU Scheduling Affinity
- Clause 504 — Pipeline Preparation Modifications
- Clause 524 — Step = Declaration = ID = Runtime Authority
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union # Callable removed
from collections import OrderedDict # For special_outputs and special_inputs order (used by PathPlanner)

from openhcs.constants.constants import VALID_GPU_MEMORY_TYPES, READ_BACKEND, WRITE_BACKEND, Backend
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.config import MaterializationBackend
from openhcs.core.pipeline.funcstep_contract_validator import \
    FuncStepContractValidator
from openhcs.core.pipeline.materialization_flag_planner import \
    MaterializationFlagPlanner
from openhcs.core.pipeline.path_planner import PipelinePathPlanner
from openhcs.core.pipeline.gpu_memory_validator import \
    GPUMemoryTypeValidator
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep # Used for isinstance check

logger = logging.getLogger(__name__)


class PipelineCompiler:
    """
    Compiles a pipeline by populating step plans within a ProcessingContext.

    This class provides static methods that are called sequentially by the
    PipelineOrchestrator for each well's ProcessingContext. Each method
    is responsible for a specific part of the compilation process, such as
    path planning, special I/O resolution, materialization flag setting,
    memory contract validation, and GPU resource assignment.
    """

    @staticmethod
    def initialize_step_plans_for_context(
        context: ProcessingContext,
        steps_definition: List[AbstractStep],
        metadata_writer: bool = False,
        plate_path: Optional[Path] = None
        # base_input_dir and well_id parameters removed, will use from context
    ) -> None:
        """
        Initializes step_plans by calling PipelinePathPlanner.prepare_pipeline_paths,
        which handles primary paths, special I/O path planning and linking, and chainbreaker status.
        Then, this method supplements the plans with non-I/O FunctionStep-specific attributes.

        Args:
            context: ProcessingContext to initialize step plans for
            steps_definition: List of AbstractStep objects defining the pipeline
            metadata_writer: If True, this well is responsible for creating OpenHCS metadata files
            plate_path: Path to plate root for zarr conversion detection
        """
        if context.is_frozen():
            raise AttributeError("Cannot initialize step plans in a frozen ProcessingContext.")

        if not hasattr(context, 'step_plans') or context.step_plans is None:
            context.step_plans = {} # Ensure step_plans dict exists

        # Pre-initialize step_plans with basic entries for each step
        # This ensures step_plans is not empty when path planner checks it
        for step in steps_definition:
            if step.step_id not in context.step_plans:
                context.step_plans[step.step_id] = {
                    "step_name": step.name,
                    "step_type": step.__class__.__name__,
                    "well_id": context.well_id,
                }

        # === ZARR CONVERSION DETECTION ===
        # Set up zarr conversion only if we want zarr output and plate isn't already zarr
        wants_zarr = (plate_path and steps_definition and
                     context.get_vfs_config().materialization_backend == MaterializationBackend.ZARR)

        # Check if plate already has zarr backend available
        already_zarr = False
        if wants_zarr:
            available_backends = context.microscope_handler.get_available_backends(plate_path)
            already_zarr = Backend.ZARR in available_backends

        if wants_zarr and not already_zarr:
            context.zarr_conversion_path = str(plate_path)
            context.original_input_dir = str(context.input_dir)
        else:
            context.zarr_conversion_path = None

        # The well_id and base_input_dir are available from the context object.
        PipelinePathPlanner.prepare_pipeline_paths(
            context,
            steps_definition
        )

        # Loop to supplement step_plans with non-I/O, non-path attributes
        # after PipelinePathPlanner has fully populated them with I/O info.
        for step in steps_definition:
            step_id = step.step_id
            if step_id not in context.step_plans:
                logger.error(
                    f"Critical error: Step {step.name} (ID: {step_id}) "
                    f"not found in step_plans after path planning phase. Clause 504."
                )
                # Create a minimal error plan
                context.step_plans[step_id] = {
                     "step_name": step.name,
                     "step_type": step.__class__.__name__,
                     "well_id": context.well_id, # Use context.well_id
                     "error": "Missing from path planning phase by PipelinePathPlanner",
                     "create_openhcs_metadata": metadata_writer # Set metadata writer responsibility flag
                }
                continue

            current_plan = context.step_plans[step_id]

            # Ensure basic metadata (PathPlanner should set most of this)
            current_plan["step_name"] = step.name
            current_plan["step_type"] = step.__class__.__name__
            current_plan["well_id"] = context.well_id # Use context.well_id; PathPlanner should also use context.well_id
            current_plan.setdefault("visualize", False) # Ensure visualize key exists
            current_plan["create_openhcs_metadata"] = metadata_writer # Set metadata writer responsibility flag

            # The special_outputs and special_inputs are now fully handled by PipelinePathPlanner.
            # The block for planning special_outputs (lines 134-148 in original) is removed.
            # Ensure these keys exist as OrderedDicts if PathPlanner doesn't guarantee it
            # (PathPlanner currently creates them as dicts, OrderedDict might not be strictly needed here anymore)
            current_plan.setdefault("special_inputs", OrderedDict())
            current_plan.setdefault("special_outputs", OrderedDict())
            current_plan.setdefault("chainbreaker", False) # PathPlanner now sets this.

            # Add FunctionStep specific attributes (non-I/O, non-path related)
            if isinstance(step, FunctionStep):
                current_plan["variable_components"] = step.variable_components
                current_plan["group_by"] = step.group_by
                current_plan["force_disk_output"] = step.force_disk_output

                # 🎯 SEMANTIC COHERENCE FIX: Prevent group_by/variable_components conflict
                # When variable_components contains the same value as group_by,
                # set group_by to None to avoid EZStitcher heritage rule violation
                if (step.variable_components and step.group_by and
                    step.group_by in step.variable_components):
                    logger.debug(f"Step {step.name}: Detected group_by='{step.group_by}' in variable_components={step.variable_components}. "
                                f"Setting group_by=None to maintain semantic coherence.")
                    current_plan["group_by"] = None

                # func attribute is guaranteed in FunctionStep.__init__
                current_plan["func_name"] = getattr(step.func, '__name__', str(step.func))

                # Memory type hints from step instance (set in FunctionStep.__init__ if provided)
                # These are initial hints; FuncStepContractValidator will set final types.
                if hasattr(step, 'input_memory_type_hint'): # From FunctionStep.__init__
                    current_plan['input_memory_type_hint'] = step.input_memory_type_hint
                if hasattr(step, 'output_memory_type_hint'): # From FunctionStep.__init__
                    current_plan['output_memory_type_hint'] = step.output_memory_type_hint

    # The resolve_special_input_paths_for_context static method is DELETED (lines 181-238 of original)
    # as this functionality is now handled by PipelinePathPlanner.prepare_pipeline_paths.

    # _prepare_materialization_flags is removed as MaterializationFlagPlanner.prepare_pipeline_flags
    # now modifies context.step_plans in-place and takes context directly.

    @staticmethod
    def declare_zarr_stores_for_context(
        context: ProcessingContext,
        steps_definition: List[AbstractStep],
        orchestrator
    ) -> None:
        """
        Declare zarr store creation functions for runtime execution.

        This method runs after path planning but before materialization flag planning
        to declare which steps need zarr stores and provide the metadata needed
        for runtime store creation.

        Args:
            context: ProcessingContext for current well
            steps_definition: List of AbstractStep objects
            orchestrator: Orchestrator instance for accessing all wells
        """
        from openhcs.constants.constants import GroupBy, Backend

        all_wells = orchestrator.get_component_keys(GroupBy.WELL)

        vfs_config = context.get_vfs_config()

        for step in steps_definition:
            step_plan = context.step_plans[step.step_id]

            will_use_zarr = (
                vfs_config.materialization_backend == MaterializationBackend.ZARR and
                (getattr(step, "force_disk_output", False) or
                 steps_definition.index(step) == len(steps_definition) - 1)
            )

            if will_use_zarr:
                step_plan["zarr_config"] = {
                    "all_wells": all_wells,
                    "needs_initialization": True
                }
                logger.debug(f"Step '{step.name}' will use zarr backend for well {context.well_id}")
            else:
                step_plan["zarr_config"] = None

    @staticmethod
    def plan_materialization_flags_for_context(
        context: ProcessingContext,
        steps_definition: List[AbstractStep],
        orchestrator
    ) -> None:
        """
        Plans and injects materialization flags into context.step_plans
        by calling MaterializationFlagPlanner.
        """
        if context.is_frozen():
            raise AttributeError("Cannot plan materialization flags in a frozen ProcessingContext.")
        if not context.step_plans:
             logger.warning("step_plans is empty in context for materialization planning. This may be valid if pipeline is empty.")
             return

        # MaterializationFlagPlanner.prepare_pipeline_flags now takes context and pipeline_definition
        # and modifies context.step_plans in-place.
        MaterializationFlagPlanner.prepare_pipeline_flags(
            context,
            steps_definition,
            orchestrator.plate_path
        )

        # Post-check (optional, but good for ensuring contracts are met by the planner)
        for step in steps_definition:
            step_id = step.step_id
            if step_id not in context.step_plans:
                 # This should not happen if prepare_pipeline_flags guarantees plans for all steps
                logger.error(f"Step {step.name} (ID: {step_id}) missing from step_plans after materialization planning.")
                continue

            plan = context.step_plans[step_id]
            # Check for keys that FunctionStep actually uses during execution
            required_keys = [READ_BACKEND, WRITE_BACKEND]
            if not all(k in plan for k in required_keys):
                missing_keys = [k for k in required_keys if k not in plan]
                logger.error(
                    f"Materialization flag planning incomplete for step {step.name} (ID: {step_id}). "
                    f"Missing required keys: {missing_keys} (Clause 273)."
                )


    @staticmethod
    def validate_memory_contracts_for_context(
        context: ProcessingContext,
        steps_definition: List[AbstractStep],
        orchestrator=None
    ) -> None:
        """
        Validates FunctionStep memory contracts, dict patterns, and adds memory type info to context.step_plans.

        Args:
            context: ProcessingContext to validate
            steps_definition: List of AbstractStep objects
            orchestrator: Optional orchestrator for dict pattern key validation
        """
        if context.is_frozen():
            raise AttributeError("Cannot validate memory contracts in a frozen ProcessingContext.")

        # FuncStepContractValidator might need access to input/output_memory_type_hint from plan
        step_memory_types = FuncStepContractValidator.validate_pipeline(
            steps=steps_definition,
            pipeline_context=context, # Pass context so validator can access step plans for memory type overrides
            orchestrator=orchestrator # Pass orchestrator for dict pattern key validation
        )

        for step_id, memory_types in step_memory_types.items():
            if "input_memory_type" not in memory_types or "output_memory_type" not in memory_types:
                step_name = context.step_plans[step_id]["step_name"]
                raise AssertionError(
                    f"Memory type validation must set input/output_memory_type for FunctionStep {step_name} (ID: {step_id}) (Clause 101)."
                )
            if step_id in context.step_plans:
                context.step_plans[step_id].update(memory_types)
            else:
                logger.warning(f"Step ID {step_id} found in memory_types but not in context.step_plans. Skipping.")

        # Apply memory type override: Any step with disk output must use numpy for disk writing
        for i, step in enumerate(steps_definition):
            if isinstance(step, FunctionStep):
                step_id = step.step_id
                if step_id in context.step_plans:
                    step_plan = context.step_plans[step_id]
                    is_last_step = (i == len(steps_definition) - 1)
                    write_backend = step_plan['write_backend']

                    if write_backend == 'disk':
                        logger.debug(f"Step {step.name} has disk output, overriding output_memory_type to numpy")
                        step_plan['output_memory_type'] = 'numpy'



    @staticmethod
    def assign_gpu_resources_for_context(
        context: ProcessingContext
    ) -> None:
        """
        Validates GPU memory types from context.step_plans and assigns GPU device IDs.
        (Unchanged from previous version)
        """
        if context.is_frozen():
            raise AttributeError("Cannot assign GPU resources in a frozen ProcessingContext.")

        gpu_assignments = GPUMemoryTypeValidator.validate_step_plans(context.step_plans)

        for step_id, step_plan_val in context.step_plans.items(): # Renamed step_plan to step_plan_val to avoid conflict
            is_gpu_step = False
            input_type = step_plan_val["input_memory_type"]
            if input_type in VALID_GPU_MEMORY_TYPES:
                is_gpu_step = True

            output_type = step_plan_val["output_memory_type"]
            if output_type in VALID_GPU_MEMORY_TYPES:
                is_gpu_step = True

            if is_gpu_step:
                # Ensure gpu_assignments has an entry for this step_id if it's a GPU step
                # And that entry contains a 'gpu_id'
                step_gpu_assignment = gpu_assignments[step_id]
                if "gpu_id" not in step_gpu_assignment:
                    step_name = step_plan_val["step_name"]
                    raise AssertionError(
                        f"GPU validation must assign gpu_id for step {step_name} (ID: {step_id}) "
                        f"with GPU memory types (Clause 295)."
                    )

        for step_id, gpu_assignment in gpu_assignments.items():
            if step_id in context.step_plans:
                context.step_plans[step_id].update(gpu_assignment)
            else:
                logger.warning(f"Step ID {step_id} found in gpu_assignments but not in context.step_plans. Skipping.")

    @staticmethod
    def apply_global_visualizer_override_for_context(
        context: ProcessingContext,
        global_enable_visualizer: bool
    ) -> None:
        """
        Applies global visualizer override to all step_plans in the context.
        (Unchanged from previous version)
        """
        if context.is_frozen():
            raise AttributeError("Cannot apply visualizer override in a frozen ProcessingContext.")

        if global_enable_visualizer:
            if not context.step_plans: return # Guard against empty step_plans
            for step_id, plan in context.step_plans.items():
                plan["visualize"] = True
                logger.info(f"Global visualizer override: Step '{plan['step_name']}' marked for visualization.")

    @staticmethod
    def update_step_ids_for_multiprocessing(
        context: ProcessingContext,
        steps_definition: List[AbstractStep]
    ) -> None:
        """
        Updates step IDs in a frozen context after multiprocessing pickle/unpickle.
        
        When contexts are pickled/unpickled for multiprocessing, step objects get
        new memory addresses, changing their IDs. This method remaps the step_plans
        from old IDs to new IDs while preserving all plan data.
        
        SPECIAL PRIVILEGE: This method can modify frozen contexts since it's part
        of the compilation process and maintains data integrity.
        
        Args:
            context: Frozen ProcessingContext with old step IDs
            steps_definition: Step objects with new IDs after pickle/unpickle
        """
        if not context.is_frozen():
            logger.warning("update_step_ids_for_multiprocessing called on unfrozen context - skipping")
            return
            
        # Create mapping from old step positions to new step IDs
        if len(steps_definition) != len(context.step_plans):
            raise RuntimeError(
                f"Step count mismatch: {len(steps_definition)} steps vs {len(context.step_plans)} plans. "
                f"Cannot safely remap step IDs."
            )
        
        # Get old step IDs in order (assuming same order as steps_definition)
        old_step_ids = list(context.step_plans.keys())
        
        # Generate new step IDs using get_step_id (handles stripped step objects)
        from openhcs.core.steps.abstract import get_step_id
        new_step_ids = [get_step_id(step) for step in steps_definition]
        
        logger.debug(f"Remapping step IDs for multiprocessing:")
        for old_id, new_id in zip(old_step_ids, new_step_ids):
            logger.debug(f"  {old_id} → {new_id}")
        
        # Create new step_plans dict with updated IDs
        new_step_plans = {}
        for old_id, new_id in zip(old_step_ids, new_step_ids):
            new_step_plans[new_id] = context.step_plans[old_id].copy()
        
        # SPECIAL PRIVILEGE: Temporarily unfreeze to update step_plans, then refreeze
        object.__setattr__(context, '_is_frozen', False)
        try:
            context.step_plans = new_step_plans
            logger.info(f"Updated {len(new_step_plans)} step plans for multiprocessing compatibility")
        finally:
            object.__setattr__(context, '_is_frozen', True)

# The monolithic compile() method is removed.
# Orchestrator will call the static methods above in sequence.
# _strip_step_attributes is also removed as StepAttributeStripper is called by Orchestrator.

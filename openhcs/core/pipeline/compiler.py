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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union # Callable removed
from collections import OrderedDict # For special_outputs and special_inputs order (used by PathPlanner)

from openhcs.constants.constants import VALID_GPU_MEMORY_TYPES
from openhcs.core.context.processing_context import ProcessingContext
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
        steps_definition: List[AbstractStep]
        # base_input_dir and well_id parameters removed, will use from context
    ) -> None:
        """
        Initializes step_plans by calling PipelinePathPlanner.prepare_pipeline_paths,
        which handles primary paths, special I/O path planning and linking, and chainbreaker status.
        Then, this method supplements the plans with non-I/O FunctionStep-specific attributes.
        """
        if context.is_frozen():
            raise AttributeError("Cannot initialize step plans in a frozen ProcessingContext.")

        if not hasattr(context, 'step_plans') or context.step_plans is None:
            context.step_plans = {} # Ensure step_plans dict exists
        # The well_id and base_input_dir are available from the context object.
        PipelinePathPlanner.prepare_pipeline_paths(
            context,
            steps_definition
        )

        # Loop to supplement step_plans with non-I/O, non-path attributes
        # after PipelinePathPlanner has fully populated them with I/O info.
        for step in steps_definition:
            step_id = step.uid
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
                     "error": "Missing from path planning phase by PipelinePathPlanner"
                }
                continue

            current_plan = context.step_plans[step_id]

            # Ensure basic metadata (PathPlanner should set most of this)
            current_plan["step_name"] = step.name
            current_plan["step_type"] = step.__class__.__name__
            current_plan["well_id"] = context.well_id # Use context.well_id; PathPlanner should also use context.well_id
            current_plan.setdefault("visualize", False) # Ensure visualize key exists

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
                if hasattr(step, 'func'): # func attribute is set in FunctionStep.__init__
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
    def plan_materialization_flags_for_context(
        context: ProcessingContext,
        steps_definition: List[AbstractStep]
        # well_id parameter removed, will use from context if needed by internal logic (currently not)
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
            steps_definition
        )

        # Post-check (optional, but good for ensuring contracts are met by the planner)
        for step in steps_definition:
            step_id = step.uid
            if step_id not in context.step_plans:
                 # This should not happen if prepare_pipeline_flags guarantees plans for all steps
                logger.error(f"Step {step.name} (ID: {step_id}) missing from step_plans after materialization planning.")
                continue

            plan = context.step_plans[step_id]
            if not all(k in plan for k in ["requires_disk_input", "requires_disk_output", "read_backend", "write_backend"]):
                logger.error(
                    f"Materialization flag planning incomplete for step {step.name} (ID: {step_id}). "
                    f"Missing one or more required flags (Clause 273)."
                )


    @staticmethod
    def validate_memory_contracts_for_context(
        context: ProcessingContext,
        steps_definition: List[AbstractStep]
    ) -> None:
        """
        Validates FunctionStep memory contracts and adds memory type info to context.step_plans.
        (Unchanged from previous version, but relies on step_plans having hints if available)
        """
        if context.is_frozen():
            raise AttributeError("Cannot validate memory contracts in a frozen ProcessingContext.")

        # FuncStepContractValidator might need access to input/output_memory_type_hint from plan
        step_memory_types = FuncStepContractValidator.validate_pipeline(
            steps=steps_definition,
            step_plans=context.step_plans # Pass step_plans for hints
        )

        for step_id, memory_types in step_memory_types.items():
            if "input_memory_type" not in memory_types or "output_memory_type" not in memory_types:
                step_name = context.step_plans.get(step_id, {}).get("step_name", step_id)
                raise AssertionError(
                    f"Memory type validation must set input/output_memory_type for FunctionStep {step_name} (ID: {step_id}) (Clause 101)."
                )
            if step_id in context.step_plans:
                context.step_plans[step_id].update(memory_types)
            else:
                logger.warning(f"Step ID {step_id} found in memory_types but not in context.step_plans. Skipping.")


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
            input_type = step_plan_val.get("input_memory_type")
            if input_type in VALID_GPU_MEMORY_TYPES:
                is_gpu_step = True

            output_type = step_plan_val.get("output_memory_type")
            if output_type in VALID_GPU_MEMORY_TYPES:
                is_gpu_step = True

            if is_gpu_step:
                # Ensure gpu_assignments has an entry for this step_id if it's a GPU step
                # And that entry contains a 'gpu_id'
                step_gpu_assignment = gpu_assignments.get(step_id, {})
                if "gpu_id" not in step_gpu_assignment:
                    step_name = step_plan_val.get("step_name", step_id)
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
                logger.info(f"Global visualizer override: Step '{plan.get('step_name', step_id)}' marked for visualization.")

# The monolithic compile() method is removed.
# Orchestrator will call the static methods above in sequence.
# _strip_step_attributes is also removed as StepAttributeStripper is called by Orchestrator.

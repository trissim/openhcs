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

import inspect
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union # Callable removed
from collections import OrderedDict # For special_outputs and special_inputs order (used by PathPlanner)

from openhcs.constants.constants import VALID_GPU_MEMORY_TYPES, READ_BACKEND, WRITE_BACKEND, Backend
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.config import MaterializationBackend, PathPlanningConfig
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


def _normalize_step_attributes(pipeline_definition: List[AbstractStep]) -> None:
    """Backwards compatibility: Set missing step attributes to constructor defaults."""
    sig = inspect.signature(AbstractStep.__init__)
    defaults = {name: param.default for name, param in sig.parameters.items()
                if name != 'self' and param.default != inspect.Parameter.empty}

    for step in pipeline_definition:
        for attr_name, default_value in defaults.items():
            if not hasattr(step, attr_name):
                setattr(step, attr_name, default_value)


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
        orchestrator,
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
            orchestrator: Orchestrator instance for well filter resolution
            metadata_writer: If True, this well is responsible for creating OpenHCS metadata files
            plate_path: Path to plate root for zarr conversion detection
        """
        if context.is_frozen():
            raise AttributeError("Cannot initialize step plans in a frozen ProcessingContext.")

        if not hasattr(context, 'step_plans') or context.step_plans is None:
            context.step_plans = {} # Ensure step_plans dict exists

        # === THREAD-LOCAL CONTEXT SETUP ===
        # Set thread-local context for lazy resolution during compilation
        # Use orchestrator's current effective config instead of potentially stale context.global_config
        from openhcs.core.config import set_current_global_config, GlobalPipelineConfig
        effective_config = orchestrator.get_effective_config()
        set_current_global_config(GlobalPipelineConfig, effective_config)
        logger.debug("🔧 THREAD-LOCAL: Set thread-local context for lazy resolution using orchestrator effective config")

        # === BACKWARDS COMPATIBILITY PREPROCESSING ===
        # Ensure all steps have complete attribute sets based on AbstractStep constructor
        # This must happen before any other compilation logic to eliminate defensive programming
        logger.debug("🔧 BACKWARDS COMPATIBILITY: Normalizing step attributes...")
        _normalize_step_attributes(steps_definition)

        # === WELL FILTER RESOLUTION ===
        # Resolve well filters for steps with materialization configs
        # This must happen after normalization to ensure materialization_config exists
        logger.debug("🎯 WELL FILTER RESOLUTION: Resolving step well filters...")
        _resolve_step_well_filters(steps_definition, context, orchestrator)

        # Pre-initialize step_plans with basic entries for each step
        # Use step index as key instead of step_id for multiprocessing compatibility
        for step_index, step in enumerate(steps_definition):
            if step_index not in context.step_plans:
                context.step_plans[step_index] = {
                    "step_name": step.name,
                    "step_type": step.__class__.__name__,
                    "well_id": context.well_id,
                }

        # === INPUT CONVERSION DETECTION ===
        # Check if first step needs zarr conversion
        if steps_definition and plate_path:
            first_step = steps_definition[0]
            vfs_config = context.get_vfs_config()

            # Only convert if default materialization backend is ZARR
            wants_zarr_conversion = (
                vfs_config.materialization_backend == MaterializationBackend.ZARR
            )

            if wants_zarr_conversion:
                # Check if input plate is already zarr format
                available_backends = context.microscope_handler.get_available_backends(plate_path)
                already_zarr = Backend.ZARR in available_backends

                if not already_zarr:
                    # Inject input conversion config using existing PathPlanningConfig pattern
                    path_config = context.get_path_planning_config()
                    conversion_config = PathPlanningConfig(
                        output_dir_suffix="",  # No suffix - write to plate root
                        global_output_folder=plate_path.parent,  # Parent of plate
                        sub_dir=path_config.sub_dir  # Use same sub_dir (e.g., "images")
                    )
                    context.step_plans[0]["input_conversion_config"] = conversion_config
                    logger.debug(f"Input conversion to zarr enabled for first step: {first_step.name}")

        # The well_id and base_input_dir are available from the context object.
        PipelinePathPlanner.prepare_pipeline_paths(
            context,
            steps_definition
        )

        # Loop to supplement step_plans with non-I/O, non-path attributes
        # after PipelinePathPlanner has fully populated them with I/O info.
        for step_index, step in enumerate(steps_definition):
            if step_index not in context.step_plans:
                logger.error(
                    f"Critical error: Step {step.name} (index: {step_index}) "
                    f"not found in step_plans after path planning phase. Clause 504."
                )
                # Create a minimal error plan
                context.step_plans[step_index] = {
                     "step_name": step.name,
                     "step_type": step.__class__.__name__,
                     "well_id": context.well_id, # Use context.well_id
                     "error": "Missing from path planning phase by PipelinePathPlanner",
                     "create_openhcs_metadata": metadata_writer # Set metadata writer responsibility flag
                }
                continue

            current_plan = context.step_plans[step_index]

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

            # Add step-specific attributes (non-I/O, non-path related)
            current_plan["variable_components"] = step.variable_components
            current_plan["group_by"] = step.group_by

            # Store materialization_config if present
            if step.materialization_config is not None:
                current_plan["materialization_config"] = step.materialization_config

            # Add FunctionStep specific attributes
            if isinstance(step, FunctionStep):

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

        for step_index, step in enumerate(steps_definition):
            step_plan = context.step_plans[step_index]

            will_use_zarr = (
                vfs_config.materialization_backend == MaterializationBackend.ZARR and
                step_index == len(steps_definition) - 1
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
        for step_index, step in enumerate(steps_definition):
            if step_index not in context.step_plans:
                 # This should not happen if prepare_pipeline_flags guarantees plans for all steps
                logger.error(f"Step {step.name} (index: {step_index}) missing from step_plans after materialization planning.")
                continue

            plan = context.step_plans[step_index]
            # Check for keys that FunctionStep actually uses during execution
            required_keys = [READ_BACKEND, WRITE_BACKEND]
            if not all(k in plan for k in required_keys):
                missing_keys = [k for k in required_keys if k not in plan]
                logger.error(
                    f"Materialization flag planning incomplete for step {step.name} (index: {step_index}). "
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

        for step_index, memory_types in step_memory_types.items():
            if "input_memory_type" not in memory_types or "output_memory_type" not in memory_types:
                step_name = context.step_plans[step_index]["step_name"]
                raise AssertionError(
                    f"Memory type validation must set input/output_memory_type for FunctionStep {step_name} (index: {step_index}) (Clause 101)."
                )
            if step_index in context.step_plans:
                context.step_plans[step_index].update(memory_types)
            else:
                logger.warning(f"Step index {step_index} found in memory_types but not in context.step_plans. Skipping.")

        # Apply memory type override: Any step with disk output must use numpy for disk writing
        for step_index, step in enumerate(steps_definition):
            if isinstance(step, FunctionStep):
                if step_index in context.step_plans:
                    step_plan = context.step_plans[step_index]
                    is_last_step = (step_index == len(steps_definition) - 1)
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

        for step_index, step_plan_val in context.step_plans.items(): # Renamed step_plan to step_plan_val to avoid conflict
            is_gpu_step = False
            input_type = step_plan_val["input_memory_type"]
            if input_type in VALID_GPU_MEMORY_TYPES:
                is_gpu_step = True

            output_type = step_plan_val["output_memory_type"]
            if output_type in VALID_GPU_MEMORY_TYPES:
                is_gpu_step = True

            if is_gpu_step:
                # Ensure gpu_assignments has an entry for this step_index if it's a GPU step
                # And that entry contains a 'gpu_id'
                step_gpu_assignment = gpu_assignments[step_index]
                if "gpu_id" not in step_gpu_assignment:
                    step_name = step_plan_val["step_name"]
                    raise AssertionError(
                        f"GPU validation must assign gpu_id for step {step_name} (index: {step_index}) "
                        f"with GPU memory types (Clause 295)."
                    )

        for step_index, gpu_assignment in gpu_assignments.items():
            if step_index in context.step_plans:
                context.step_plans[step_index].update(gpu_assignment)
            else:
                logger.warning(f"Step index {step_index} found in gpu_assignments but not in context.step_plans. Skipping.")

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
            for step_index, plan in context.step_plans.items():
                plan["visualize"] = True
                logger.info(f"Global visualizer override: Step '{plan['step_name']}' marked for visualization.")

    @staticmethod
    def resolve_lazy_dataclasses_for_context(context: ProcessingContext, orchestrator) -> None:
        """
        Resolve all lazy dataclass instances in step plans to their base configurations.

        This method should be called after all compilation phases but before context
        freezing to ensure step plans are safe for pickling in multiprocessing contexts.

        Args:
            context: ProcessingContext to process
            orchestrator: PipelineOrchestrator to get current effective config from
        """
        from openhcs.core.lazy_config import resolve_lazy_configurations_for_serialization
        from openhcs.core.config import set_current_global_config, GlobalPipelineConfig

        # Use orchestrator's current effective config as authoritative source
        # This ensures compilation resolves the same values as UI placeholders
        effective_config = orchestrator.get_effective_config()
        set_current_global_config(GlobalPipelineConfig, effective_config)

        # Use the shared recursive resolution function to handle nested structures
        for step_index, step_plan in context.step_plans.items():
            context.step_plans[step_index] = resolve_lazy_configurations_for_serialization(step_plan)

    @staticmethod
    def compile_pipelines(
        orchestrator,
        pipeline_definition: List[AbstractStep],
        well_filter: Optional[List[str]] = None,
        enable_visualizer_override: bool = False
    ) -> Dict[str, ProcessingContext]:
        """
        Compile-all phase: Prepares frozen ProcessingContexts for each well.

        This method iterates through the specified wells, creates a ProcessingContext
        for each, and invokes the various phases of the PipelineCompiler to populate
        the context's step_plans. After all compilation phases for a well are complete,
        its context is frozen. Finally, attributes are stripped from the pipeline_definition,
        making the step objects stateless for the execution phase.

        Args:
            orchestrator: The PipelineOrchestrator instance to use for compilation
            pipeline_definition: The list of AbstractStep objects defining the pipeline.
            well_filter: Optional list of well IDs to process. If None, processes all found wells.
            enable_visualizer_override: If True, all steps in all compiled contexts
                                        will have their 'visualize' flag set to True.

        Returns:
            A dictionary mapping well IDs to their compiled and frozen ProcessingContexts.
            The input `pipeline_definition` list (of step objects) is modified in-place
            to become stateless.
        """
        from openhcs.constants.constants import GroupBy, OrchestratorState
        from openhcs.core.pipeline.step_attribute_stripper import StepAttributeStripper

        if not orchestrator.is_initialized():
            raise RuntimeError("PipelineOrchestrator must be explicitly initialized before calling compile_pipelines().")

        if not pipeline_definition:
            raise ValueError("A valid pipeline definition (List[AbstractStep]) must be provided.")

        try:
            compiled_contexts: Dict[str, ProcessingContext] = {}
            wells_to_process = orchestrator.get_component_keys(GroupBy.WELL, well_filter)

            if not wells_to_process:
                logger.warning("No wells found to process based on filter.")
                return {}

            logger.info(f"Starting compilation for wells: {', '.join(wells_to_process)}")

            # Determine responsible well for metadata creation (lexicographically first)
            responsible_well = sorted(wells_to_process)[0] if wells_to_process else None
            logger.debug(f"Designated responsible well for metadata creation: {responsible_well}")

            for well_id in wells_to_process:
                logger.debug(f"Compiling for well: {well_id}")
                context = orchestrator.create_context(well_id)

                # Determine if this well is responsible for metadata creation
                is_responsible = (well_id == responsible_well)
                logger.debug(f"Well {well_id} metadata responsibility: {is_responsible}")

                PipelineCompiler.initialize_step_plans_for_context(context, pipeline_definition, orchestrator, metadata_writer=is_responsible, plate_path=orchestrator.plate_path)
                PipelineCompiler.declare_zarr_stores_for_context(context, pipeline_definition, orchestrator)
                PipelineCompiler.plan_materialization_flags_for_context(context, pipeline_definition, orchestrator)
                PipelineCompiler.validate_memory_contracts_for_context(context, pipeline_definition, orchestrator)
                PipelineCompiler.assign_gpu_resources_for_context(context)

                if enable_visualizer_override:
                    PipelineCompiler.apply_global_visualizer_override_for_context(context, True)

                # Resolve all lazy dataclasses before freezing to ensure multiprocessing compatibility
                PipelineCompiler.resolve_lazy_dataclasses_for_context(context, orchestrator)

                context.freeze()
                compiled_contexts[well_id] = context
                logger.debug(f"Compilation finished for well: {well_id}")

            # After processing all wells, strip attributes and finalize
            logger.info("Stripping attributes from pipeline definition steps.")
            StepAttributeStripper.strip_step_attributes(pipeline_definition, {})

            # Log path planning summary once per plate
            if compiled_contexts:
                first_context = next(iter(compiled_contexts.values()))
                logger.info(f"📁 PATH PLANNING SUMMARY:")
                logger.info(f"   Main pipeline output: {first_context.output_plate_root}")

                # Check for materialization steps in first context
                materialization_steps = []
                for step_id, plan in first_context.step_plans.items():
                    if 'materialized_output_dir' in plan:
                        step_name = plan.get('step_name', f'step_{step_id}')
                        mat_path = plan['materialized_output_dir']
                        materialization_steps.append((step_name, mat_path))

                for step_name, mat_path in materialization_steps:
                    logger.info(f"   Materialization {step_name}: {mat_path}")

            orchestrator._state = OrchestratorState.COMPILED
            logger.info(f"🏁 COMPILATION COMPLETE: {len(compiled_contexts)} wells compiled successfully")
            return compiled_contexts
        except Exception as e:
            orchestrator._state = OrchestratorState.COMPILE_FAILED
            logger.error(f"Failed to compile pipelines: {e}")
            raise



# The monolithic compile() method is removed.
# Orchestrator will call the static methods above in sequence.
# _strip_step_attributes is also removed as StepAttributeStripper is called by Orchestrator.


def _resolve_step_well_filters(steps_definition: List[AbstractStep], context, orchestrator):
    """
    Resolve well filters for steps with materialization configs.

    This function handles step-level well filtering by resolving patterns like
    "row:A", ["A01", "B02"], or max counts against the available wells for the plate.

    Args:
        steps_definition: List of pipeline steps
        context: Processing context for the current well
        orchestrator: Orchestrator instance with access to available wells
    """
    from openhcs.core.utils import WellFilterProcessor

    # Get available wells from orchestrator using correct method
    from openhcs.constants.constants import GroupBy
    available_wells = orchestrator.get_component_keys(GroupBy.WELL)
    if not available_wells:
        logger.warning("No available wells found for well filter resolution")
        return

    # Initialize step_well_filters in context if not present
    if not hasattr(context, 'step_well_filters'):
        context.step_well_filters = {}

    # Process each step that has materialization config with well filter
    for step_index, step in enumerate(steps_definition):
        if (hasattr(step, 'materialization_config') and
            step.materialization_config and
            step.materialization_config.well_filter is not None):

            try:
                # Resolve the well filter pattern to concrete well IDs
                resolved_wells = WellFilterProcessor.resolve_compilation_filter(
                    step.materialization_config.well_filter,
                    available_wells
                )

                # Store resolved wells in context for path planner
                # Use structure expected by path planner
                context.step_well_filters[step_index] = {  # Use step_index instead of step.step_id
                    'resolved_wells': sorted(resolved_wells),
                    'filter_mode': step.materialization_config.well_filter_mode,
                    'original_filter': step.materialization_config.well_filter
                }

                logger.debug(f"Step '{step.name}' well filter '{step.materialization_config.well_filter}' "
                           f"resolved to {len(resolved_wells)} wells: {sorted(resolved_wells)}")

            except Exception as e:
                logger.error(f"Failed to resolve well filter for step '{step.name}': {e}")
                raise ValueError(f"Invalid well filter '{step.materialization_config.well_filter}' "
                               f"for step '{step.name}': {e}")

    logger.debug(f"Well filter resolution complete. {len(context.step_well_filters)} steps have well filters.")

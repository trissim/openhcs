"""
Pipeline module for OpenHCS.

This module provides the core pipeline components for OpenHCS:
- PipelineExecutor: Executes a compiled pipeline

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
- Clause 311 — All declarative schemas are deprecated
- Clause 504 — Pipeline Preparation Modifications
- Clause 524 — Step = Declaration = ID = Runtime Authority
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

from openhcs.constants.constants import VALID_GPU_MEMORY_TYPES
from openhcs.core.pipeline.funcstep_contract_validator import \
    FuncStepContractValidator
from openhcs.core.pipeline.materialization_flag_planner import \
    MaterializationFlagPlanner
from openhcs.core.pipeline.path_planner import PipelinePathPlanner
# Import directly from the module to avoid circular dependency
# from openhcs.core.pipeline.gpu_memory_validator import GPUMemoryTypeValidator
from openhcs.core.pipeline.step_attribute_stripper import \
    StepAttributeStripper
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep

logger = logging.getLogger(__name__)


class PipelineCompiler:
    """
    Compiles a pipeline into an executable form.

    This class is responsible for:
    1. Planning paths for each step using direct step field access
    2. Creating step plans
    3. Validating memory type contracts
    4. Assigning GPU device IDs
    5. Stripping step attributes

    # Clause 12 — Absolute Clean Execution
    # Clause 17-B — Path Format Discipline
    # Clause 66 — Immutability After Construction
    # Clause 88 — No Inferred Capabilities
    # Clause 101 — Memory Type Declaration
    # Clause 245 — Path Declaration
    # Clause 273 — Backend Authorization Doctrine
    # Clause 281 — Context-Bound Identifiers
    # Clause 293 — GPU Pre-Declaration Enforcement
    # Clause 295 — GPU Scheduling Affinity
    # Clause 504 — Pipeline Preparation Modifications
    # Clause 524 — Step = Declaration = ID = Runtime Authority
    """


    @staticmethod
    def _initialize_pipeline_context() -> Dict[str, bool]:
        """
        Step 1: Initialize the pipeline context to track planner execution.

        Returns:
            Dictionary with planner execution flags
        """
        return {
            "path_planner_done": False,
            "materialization_planner_done": False,
            "memory_contract_validator_done": False,
            "gpu_memory_validator_done": False,
            "attribute_stripper_done": False
        }


    @staticmethod
    def _prepare_step_paths(
        steps: List[AbstractStep],
        input_dir: Union[str, Path],
        well_id: str,
        pipeline_context: Dict[str, bool]
    ) -> Dict[str, Dict[str, Union[str, Path]]]:
        """
        Step 3: Prepare paths for all steps using the path planner.

        Args:
            steps: List of steps to prepare paths for
            input_dir: Input directory for the pipeline
            well_id: Well ID for the pipeline
            path_overrides: Dictionary of path overrides
            pipeline_context: Pipeline context to track planner execution

        Returns:
            Dictionary mapping step UIDs to path information

        Raises:
            AssertionError: If path planning fails for any step
        """
        # Prepare paths using path planner
        step_paths = PipelinePathPlanner.prepare_pipeline_paths(
            input_dir=input_dir,  # Plain path, not VirtualPath
            pipeline={"steps": steps},
            well_id=well_id
        )

        # Verify that path planning has run successfully
        for step in steps:
            step_id = step.uid
            if step_id not in step_paths:
                raise AssertionError(
                    f"Path planning must be completed for all steps. "
                    f"Missing path information for step {step.name} (ID: {step_id}) (Clause 504)."
                )
            if "input_dir" not in step_paths[step_id]:
                raise AssertionError(
                    f"Path planning must resolve input_dir for all steps. "
                    f"Missing input_dir for step {step.name} (ID: {step_id}) (Clause 504)."
                )
            if "output_dir" not in step_paths[step_id]:
                raise AssertionError(
                    f"Path planning must resolve output_dir for all steps. "
                    f"Missing output_dir for step {step.name} (ID: {step_id}) (Clause 504)."
                )

        # Mark path planner as done
        pipeline_context["path_planner_done"] = True

        return step_paths

    @staticmethod
    def _prepare_materialization_flags(
        steps: List[AbstractStep],
        well_id: str,
        pipeline_context: Dict[str, bool]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Step 4: Prepare materialization flags for all steps using the flag planner.

        Args:
            steps: List of steps to prepare flags for
            well_id: Well ID for the pipeline
            pipeline_context: Pipeline context to track planner execution

        Returns:
            Dictionary mapping step UIDs to materialization flags

        Raises:
            AssertionError: If materialization planning fails for any step
        """
        # Prepare materialization flags using flag planner
        step_flags = MaterializationFlagPlanner.prepare_pipeline_flags(
            steps=steps,
            well_id=well_id
        )

        # Verify that materialization planning has run successfully
        for step in steps:
            step_id = step.uid
            # Check if step has flags
            if step_id not in step_flags:
                raise AssertionError(
                    f"Materialization flag planning must be completed for all steps. "
                    f"Missing flag information for step {step.name} (ID: {step_id}) "
                    f"(Clause 504)."
                )

            # Check required flags
            flags = step_flags[step_id]
            if "requires_disk_input" not in flags:
                raise AssertionError(
                    f"Materialization flag planning must set requires_disk_input. "
                    f"Missing for step {step.name} (ID: {step_id}) (Clause 273)."
                )
            if "requires_disk_output" not in flags:
                raise AssertionError(
                    f"Materialization flag planning must set requires_disk_output. "
                    f"Missing for step {step.name} (ID: {step_id}) (Clause 273)."
                )
            if "read_backend" not in flags:
                raise AssertionError(
                    f"Materialization flag planning must set read_backend. "
                    f"Missing for step {step.name} (ID: {step_id}) (Clause 273)."
                )
            if "write_backend" not in flags:
                raise AssertionError(
                    f"Materialization flag planning must set write_backend. "
                    f"Missing for step {step.name} (ID: {step_id}) (Clause 273)."
                )

        # Mark materialization planner as done
        pipeline_context["materialization_planner_done"] = True

        return step_flags



    @staticmethod
    def _create_initial_step_plans(
        steps: List[AbstractStep],
        step_paths: Dict[str, Dict[str, Union[str, Path]]],
        step_flags: Dict[str, Dict[str, Any]],
        well_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Step 5: Combine paths and flags into initial step plans.

        Note: Special contracts are now handled by the path planner and included in step_paths.

        Args:
            steps: List of steps to create plans for
            step_paths: Dictionary mapping step UIDs to path information
            step_flags: Dictionary mapping step UIDs to materialization flags
            well_id: Well ID for the pipeline

        Returns:
            Dictionary mapping step UIDs to initial step plans
        """
        step_plans = {}
        for step in steps:
            step_id = step.uid
            step_paths_entry = step_paths.get(step_id)
            step_flags_entry = step_flags.get(step_id)

            if step_paths_entry:
                # Use step.uid as key only, don't duplicate in the dictionary body
                step_plans[step_id] = {
                    "step_name": step.name,
                    "step_type": step.__class__.__name__,
                    "input_dir": step_paths_entry.get("input_dir"),
                    "output_dir": step_paths_entry.get("output_dir"),
                    "well_id": well_id,
                    "visualize": False  # Clause 273: Planner declares visual intent via this flag
                }

                # Add positions_folder if present
                if "positions_folder" in step_paths_entry:
                    step_plans[step_id]["positions_folder"] = step_paths_entry["positions_folder"]

                # Add materialization flags if present
                if step_flags_entry:
                    step_plans[step_id].update(step_flags_entry)

                # Add special contract information from step_paths
                save_outputs = step_paths_entry.get("save_special_outputs", [])
                if save_outputs:
                    step_plans[step_id]["save_special_outputs"] = save_outputs

                load_inputs = step_paths_entry.get("load_special_inputs", [])
                if load_inputs:
                    step_plans[step_id]["load_special_inputs"] = load_inputs

                # Extract step-specific attributes before they are stripped
                # This ensures that all necessary attributes are preserved in the step plans
                if isinstance(step, FunctionStep):
                    # For FunctionStep, preserve variable_components, group_by, and func
                    step_plans[step_id]["variable_components"] = step.variable_components
                    step_plans[step_id]["group_by"] = step.group_by
                    step_plans[step_id]["func"] = step.func

        return step_plans

    @staticmethod
    def _validate_memory_contracts(
        steps: List[AbstractStep],
        pipeline_context: Dict[str, bool]
    ) -> Dict[str, Dict[str, str]]:
        """
        Step 7: Validate FunctionStep memory contracts and get memory types.

        Args:
            steps: List of steps to validate
            pipeline_context: Pipeline context to track planner execution

        Returns:
            Dictionary mapping step UIDs to memory type information

        Raises:
            AssertionError: If memory type validation fails
        """
        # Validate FunctionStep memory contracts and get memory types
        step_memory_types = FuncStepContractValidator.validate_pipeline(
            steps=steps,
            pipeline_context=pipeline_context  # Pass the context to enforce planner order
        )

        # Verify that memory type validation has run successfully
        for step_id, memory_types in step_memory_types.items():
            if "input_memory_type" not in memory_types:
                raise AssertionError(
                    f"Memory type validation must set input_memory_type for all FunctionSteps. "
                    f"Missing input_memory_type for step {step_id} (Clause 101)."
                )
            if "output_memory_type" not in memory_types:
                raise AssertionError(
                    f"Memory type validation must set output_memory_type for all FunctionSteps. "
                    f"Missing output_memory_type for step {step_id} (Clause 101)."
                )

        # Mark memory contract validator as done
        pipeline_context["memory_contract_validator_done"] = True

        return step_memory_types

    @staticmethod
    def _add_memory_types_to_step_plans(
        step_plans: Dict[str, Dict[str, Any]],
        step_memory_types: Dict[str, Dict[str, str]]
    ) -> None:
        """
        Step 8: Add memory types to step plans.

        Args:
            step_plans: Dictionary mapping step UIDs to step plans
            step_memory_types: Dictionary mapping step UIDs to memory type information
        """
        for step_id, memory_types in step_memory_types.items():
            if step_id in step_plans:
                step_plans[step_id].update(memory_types)

    @staticmethod
    def _validate_gpu_memory_types(
        step_plans: Dict[str, Dict[str, Any]],
        pipeline_context: Dict[str, bool]
    ) -> Dict[str, Dict[str, int]]:
        """
        Step 9: Validate GPU memory types and assign GPU device IDs.

        Args:
            step_plans: Dictionary mapping step UIDs to step plans
            pipeline_context: Pipeline context to track planner execution

        Returns:
            Dictionary mapping step UIDs to GPU assignment information

        Raises:
            AssertionError: If GPU validation fails
        """
        # Import here to avoid circular dependency
        from openhcs.core.pipeline.gpu_memory_validator import \
            GPUMemoryTypeValidator

        # Validate GPU memory types and assign GPU device IDs
        gpu_assignments = GPUMemoryTypeValidator.validate_step_plans(step_plans)

        # Verify that GPU validation has run successfully
        for step_id, step_plan in step_plans.items():
            # Check GPU input memory types
            input_type = step_plan.get("input_memory_type")
            if input_type in VALID_GPU_MEMORY_TYPES:
                if "gpu_id" not in gpu_assignments.get(step_id, {}):
                    step_name = step_plan.get("step_name", step_id)
                    raise AssertionError(
                        f"GPU validation must assign gpu_id for steps with GPU memory types. "
                        f"Missing gpu_id for step {step_name} with GPU input type {input_type} "
                        f"(Clause 295)."
                    )

            # Check GPU output memory types
            output_type = step_plan.get("output_memory_type")
            if output_type in VALID_GPU_MEMORY_TYPES:
                if "gpu_id" not in gpu_assignments.get(step_id, {}):
                    step_name = step_plan.get("step_name", step_id)
                    raise AssertionError(
                        f"GPU validation must assign gpu_id for steps with GPU memory types. "
                        f"Missing gpu_id for step {step_name} with GPU output type {output_type} "
                        f"(Clause 295)."
                    )

        # Mark GPU memory validator as done
        pipeline_context["gpu_memory_validator_done"] = True

        return gpu_assignments

    @staticmethod
    def _add_gpu_device_ids_to_step_plans(
        step_plans: Dict[str, Dict[str, Any]],
        gpu_assignments: Dict[str, Dict[str, int]]
    ) -> None:
        """
        Step 10: Add GPU device IDs to step plans.

        Args:
            step_plans: Dictionary mapping step UIDs to step plans
            gpu_assignments: Dictionary mapping step UIDs to GPU assignment information
        """
        for step_id, gpu_assignment in gpu_assignments.items():
            if step_id in step_plans:
                step_plans[step_id].update(gpu_assignment)

    @staticmethod
    def _strip_step_attributes(
        steps: List[AbstractStep],
        step_plans: Dict[str, Dict[str, Any]],
        pipeline_context: Dict[str, bool]
    ) -> None:
        """
        Step 11: Strip all attributes from steps.

        Args:
            steps: List of steps to strip attributes from
            step_plans: Dictionary mapping step UIDs to step plans
            pipeline_context: Pipeline context to track planner execution

        Raises:
            AssertionError: If steps have no attributes to strip
        """
        # Verify that steps still have attributes to strip (first run check)
        for step in steps:
            remaining_attrs = set(vars(step).keys())
            if not remaining_attrs:
                raise AssertionError(
                    f"Step attribute stripping must only run once. "
                    f"Step {step.__class__.__name__} has no attributes "
                    f"(ID: {step.uid}) (Clause 66)."
                )

        # Now strip all attributes
        StepAttributeStripper.strip_step_attributes(steps, step_plans)

        # Mark attribute stripper as done
        pipeline_context["attribute_stripper_done"] = True

    @staticmethod
    def _apply_global_visualizer_override(
        step_plans: Dict[str, Dict[str, Any]],
        global_enable_visualizer: bool
    ) -> None:
        """
        Step 12: Apply global visualizer override if enabled.

        Args:
            step_plans: Dictionary mapping step UIDs to step plans
            global_enable_visualizer: Whether to enable visualization for all steps
        """
        if global_enable_visualizer:
            for step_id, plan in step_plans.items():
                plan["visualize"] = True
                msg = "Global visualizer override: Step '%s' marked for visualization."
                logger.info(msg, step_id)


    @staticmethod
    def compile(
        steps: List[AbstractStep],
        input_dir: Union[str, Path],
        well_id: str,
        global_enable_visualizer: bool = False
    ) -> Dict[str, Dict[str, Union[str, Path]]]:
        """
        Compile a pipeline into step plans with OS filesystem paths.

        Args:
            steps: List of steps to compile
            input_dir: Input directory for the pipeline (as a str or Path object)
            well_id: Well ID for the pipeline
            global_enable_visualizer: Whether to enable visualization for all steps

        Returns:
            Dictionary mapping step UIDs to step plans with all paths as OS filesystem paths

        Doctrinal Clauses:
            - Clause 17 — VFS Exclusivity (FileManager is the only component that uses VirtualPath)
            - Clause 17-B — Path Format Discipline
            - Clause 66 — Immutability After Construction
            - Clause 88 — No Inferred Capabilities
            - Clause 245 — Path Declaration
            - Clause 251 — Special Output Contract
            - Clause 293 — GPU Pre-Declaration Enforcement
            - Clause 295 — GPU Scheduling Affinity
            - Clause 524 — Step = Declaration = ID = Runtime Authority
        """
        # IMPORTANT: Order of planners is critical for doctrinal compliance

        # 1. Initialize pipeline context
        pipeline_context = PipelineCompiler._initialize_pipeline_context()

        # 3. Prepare paths using path planner
        step_paths = PipelineCompiler._prepare_step_paths(
            steps=steps,
            input_dir=input_dir,
            well_id=well_id,
            pipeline_context=pipeline_context
        )

        # 4. Prepare materialization flags
        step_flags = PipelineCompiler._prepare_materialization_flags(
            steps=steps,
            well_id=well_id,
            pipeline_context=pipeline_context
        )

        # 5. Combine paths and flags into initial step plans
        # Note: Special contracts are now handled by the path planner
        step_plans = PipelineCompiler._create_initial_step_plans(
            steps=steps,
            step_paths=step_paths,
            step_flags=step_flags,
            step_contracts={},  # Empty dict since special contracts are in step_paths
            well_id=well_id
        )

        # 7. Validate FunctionStep memory contracts and get memory types
        step_memory_types = PipelineCompiler._validate_memory_contracts(
            steps=steps,
            pipeline_context=pipeline_context
        )

        # 8. Add memory types to step plans
        PipelineCompiler._add_memory_types_to_step_plans(
            step_plans=step_plans,
            step_memory_types=step_memory_types
        )

        # 9. Validate GPU memory types and assign GPU device IDs
        gpu_assignments = PipelineCompiler._validate_gpu_memory_types(
            step_plans=step_plans,
            pipeline_context=pipeline_context
        )

        # 10. Add GPU device IDs to step plans
        PipelineCompiler._add_gpu_device_ids_to_step_plans(
            step_plans=step_plans,
            gpu_assignments=gpu_assignments
        )

        # 11. Strip all attributes from steps
        PipelineCompiler._strip_step_attributes(
            steps=steps,
            step_plans=step_plans,
            pipeline_context=pipeline_context
        )

        # 12. Apply global visualizer override if enabled
        PipelineCompiler._apply_global_visualizer_override(
            step_plans=step_plans,
            global_enable_visualizer=global_enable_visualizer
        )

        return step_plans

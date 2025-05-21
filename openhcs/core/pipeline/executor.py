"""
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
- Clause 297 — Immutable Result Enforcement
- Clause 311 — All declarative schemas are deprecated
- Clause 504 — Pipeline Preparation Modifications
- Clause 524 — Step = Declaration = ID = Runtime Authority
"""

import concurrent.futures
import logging
from typing import Any, List, Optional, Union

from openhcs.constants.constants import (DEFAULT_NUM_WORKERS,
                                            VALID_GPU_MEMORY_TYPES)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.step_result import StepResult

logger = logging.getLogger(__name__)




class PipelineExecutor:
    """
    Executes a compiled pipeline.

    This class is responsible for:
    1. Executing each step in the pipeline
    2. Handling step results
    3. Updating the context
    4. Managing parallel execution of multiple pipelines

    This executor is completely stateless and does not perform any GPU management
    or resource allocation. All GPU IDs must be assigned during planning/compilation,
    not during execution.

    # Clause 66 — Immutability After Construction
    # Clause 88 — No Inferred Capabilities
    # Clause 281 — Context-Bound Identifiers
    # Clause 293 — GPU Pre-Declaration Enforcement
    # Clause 295 — GPU Scheduling Affinity
    # Clause 297 — Immutable Result Enforcement
    # Clause 524 — Step = Declaration = ID = Runtime Authority
    """

    @staticmethod
    def execute(
        steps: Union[List[AbstractStep], List[List[AbstractStep]]],
        context: Union[ProcessingContext, List[ProcessingContext]],
        max_workers: int = DEFAULT_NUM_WORKERS,
        visualizer: Optional[Any] = None
    ) -> Union[ProcessingContext, List[ProcessingContext]]:
        """
        Execute one or more pipelines with the given context(s).

        This fully unified method handles both input normalization and execution
        in a single place, eliminating the need for separate execution methods.

        Args:
            steps: Single pipeline or list of pipelines to execute
            context: Single context or list of contexts
            max_workers: Maximum number of worker threads (default: DEFAULT_NUM_WORKERS)
            visualizer: Optional NapariStreamVisualizer instance for real-time visualization

        Returns:
            Updated context or list of updated contexts (matching input type)

        Raises:
            ValueError: If inputs are invalid or inconsistent
            TypeError: If a step returns a non-StepResult value

        # Clause 297 — Immutable Result Enforcement
        # Clause 524 — Step = Declaration = ID = Runtime Authority
        """
        # Determine if we're dealing with a single pipeline or multiple pipelines
        is_single_pipeline = not isinstance(steps[0], list)
        is_single_context = not isinstance(context, list)

        # Normalize inputs to lists for unified processing
        pipeline_list = [steps] if is_single_pipeline else steps
        context_list = [context] if is_single_context else context

        # Validate input combinations
        if is_single_pipeline and not is_single_context:
            raise ValueError("Single pipeline with multiple contexts is not supported")

        if not is_single_pipeline and is_single_context:
            raise ValueError("Multiple pipelines with single context is not supported")

        if len(pipeline_list) != len(context_list):
            raise ValueError(
                f"Number of pipelines ({len(pipeline_list)}) "
                f"must match number of contexts ({len(context_list)})"
            )

        # Define the execution logic inline
        def execute_pipeline(pipeline, ctx):
            # Verify GPU assignments if needed
            PipelineExecutor._verify_gpu_assignments(pipeline, ctx)

            logger.info("Executing pipeline with %d steps for well %s", len(pipeline), ctx.well_id)

            # Process each step in sequence
            for i, step in enumerate(pipeline):
                step_id = step.uid
                logger.info("Executing step %d/%d: %s (ID: %s)",
                           i+1, len(pipeline), step.name, step_id)

                # Validate step plan exists
                if step_id not in ctx.step_plans:
                    raise ValueError(f"Step plan not found for step: {step.name} (ID: {step_id})")

                # Execute the step
                step_result = step.process(ctx)

                # Enforce strict StepResult return type (Clause 297)
                if not isinstance(step_result, StepResult):
                    step_type = type(step_result).__name__
                    raise TypeError(
                        f"Step '{step.name}' returned {step_type} instead of StepResult. "
                        f"All steps must return a StepResult instance "
                        f"(Clause 297 — Immutable Result Enforcement)."
                    )

                # Visualize step result if needed
                if visualizer:
                    PipelineExecutor._visualize_step_result(step_id, step_result, ctx, visualizer)

                # Update context with step result
                ctx.update_from_step_result(step_result)

            logger.info("Pipeline execution completed for well %s", ctx.well_id)
            return ctx

        # For a single pipeline, execute directly without thread overhead
        if len(pipeline_list) == 1:
            result = execute_pipeline(pipeline_list[0], context_list[0])
            # Return single context or list based on input type
            return result if is_single_context else [result]

        # For sequential execution (max_workers=1), use list comprehension
        if max_workers == 1:
            results = [
                execute_pipeline(pipeline, ctx)
                for pipeline, ctx in zip(pipeline_list, context_list)
            ]
            return results

        # For parallel execution, use ThreadPoolExecutor
        # Use at most as many workers as pipelines
        effective_workers = min(max_workers, len(pipeline_list))
        logger.info(
            "Executing %d pipelines using %d worker threads",
            len(pipeline_list),
            effective_workers
        )

        # Results container
        results = [None] * len(pipeline_list)

        # Create a thread pool with the appropriate number of workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            # Submit all pipeline execution tasks
            future_to_index = {}

            for i, (pipeline, ctx) in enumerate(zip(pipeline_list, context_list)):
                logger.info("Submitting pipeline %d to thread pool", i)
                future = executor.submit(execute_pipeline, pipeline, ctx)
                future_to_index[future] = i

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                    logger.info("Pipeline %d completed successfully", idx)
                except Exception as e:
                    logger.error("Pipeline %d failed: %s", idx, e)
                    # Re-raise the exception to fail loudly
                    error_msg = "Pipeline {} failed: {}".format(idx, str(e))
                    raise RuntimeError(error_msg) from e

        return results

    @staticmethod
    def _visualize_step_result(
        step_id: str,
        step_result: StepResult,
        context: ProcessingContext,
        visualizer: Any
    ) -> None:
        """
        Push step result to the visualizer if the step is flagged for visualization.

        Args:
            step_id: The step ID
            step_result: The step result
            context: The processing context
            visualizer: The NapariStreamVisualizer instance

        # Clause 368 — Visualization Must Be Observer-Only
        """
        if not visualizer:
            return

        step_plan = context.step_plans.get(step_id, {})
        if not step_plan.get('visualize', False):
            return

        if not hasattr(step_result, 'data') or step_result.data is None:
            return

        try:
            logger.debug("Step '%s' flagged for visualization. Pushing tensor.", step_id)
            visualizer.push_tensor(
                step_id=step_id,
                tensor=step_result.data,
                well_id=context.well_id
            )
        except Exception as e:
            logger.error("Error during visualization push for step '%s': %s",
                        step_id, str(e), exc_info=True)

    @staticmethod
    def _verify_gpu_assignments(
        steps: List[AbstractStep],  # pylint: disable=unused-argument
        context: ProcessingContext
    ) -> None:
        """
        Verify that all GPU steps have pre-assigned GPU IDs.

        This method checks that all steps requiring GPU memory have
        pre-assigned GPU IDs in their step plans. It does not assign
        GPU IDs or perform any GPU management.

        Args:
            steps: List of steps to execute
            context: Processing context

        Raises:
            AssertionError: If any GPU step doesn't have a pre-assigned GPU ID

        # Clause 293 — GPU Pre-Declaration Enforcement
        # Clause 295 — GPU Scheduling Affinity
        """
        # Check if any step requires GPU
        requires_gpu = False
        for step_id, step_plan in context.step_plans.items():
            input_memory_type = step_plan.get('input_memory_type')
            output_memory_type = step_plan.get('output_memory_type')

            if (input_memory_type in VALID_GPU_MEMORY_TYPES or
                output_memory_type in VALID_GPU_MEMORY_TYPES):
                requires_gpu = True
                break

        # If any step requires GPU, verify GPU ID is assigned
        if requires_gpu:
            if not hasattr(context, 'gpu_id') or context.gpu_id is None:
                raise AssertionError(
                    f"Clause 295 Violation: Context for well {context.well_id} "
                    f"has no pre-assigned GPU ID. "
                    "GPU IDs must be assigned during planning phase, not at runtime."
                )

            # Verify that all GPU steps have pre-assigned GPU IDs in their step plans
            for step_id, step_plan in context.step_plans.items():
                input_memory_type = step_plan.get('input_memory_type')
                output_memory_type = step_plan.get('output_memory_type')

                if ((input_memory_type in VALID_GPU_MEMORY_TYPES or
                     output_memory_type in VALID_GPU_MEMORY_TYPES) and
                    'gpu_id' not in step_plan):
                    # This should never happen if planning was done correctly
                    raise AssertionError(
                        f"Clause 295 Violation: Step {step_id} for well {context.well_id} "
                        f"requires GPU memory ({input_memory_type}/{output_memory_type}) "
                        f"but has no pre-assigned GPU ID in its step plan. "
                        f"GPU IDs must be assigned during planning phase, not at runtime."
                    )

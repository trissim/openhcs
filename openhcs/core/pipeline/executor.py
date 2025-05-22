"""
PipelineExecutor for OpenHCS.

NOTE: As of the refactoring to a two-phase (compile-then-execute) model
in PipelineOrchestrator, the responsibilities of this PipelineExecutor
have been largely absorbed by PipelineOrchestrator. This class is
currently a placeholder and may be removed in future versions once all
dependencies are updated.

The PipelineOrchestrator now directly handles:
- Sequential execution of steps for a single well (_execute_single_well).
- Parallel execution of multiple wells (execute_compiled_plate).
- VFS-based visualization calls.
- Management of immutable ProcessingContexts.
"""

import logging
from typing import Any, List, Optional, Union

# Imports kept for historical context or if any minor utility remains,
# but core execution logic is deprecated.
from openhcs.constants.constants import DEFAULT_NUM_WORKERS
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.abstract import AbstractStep
# StepResult is no longer used by the core execution flow.
# from openhcs.core.steps.step_result import StepResult 

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """
    Executes a compiled pipeline. (DEPRECATED)

    The core execution logic has been moved to PipelineOrchestrator.
    This class is a placeholder and its methods should no longer be directly used
    for pipeline execution if using the new PipelineOrchestrator interface.
    """

    @staticmethod
    def execute(
        steps: Union[List[AbstractStep], List[List[AbstractStep]]],
        context: Union[ProcessingContext, List[ProcessingContext]],
        max_workers: int = DEFAULT_NUM_WORKERS, # pylint: disable=unused-argument
        visualizer: Optional[Any] = None # pylint: disable=unused-argument
    ) -> Union[ProcessingContext, List[ProcessingContext]]:
        """
        DEPRECATED: Executes one or more pipelines.
        Please use PipelineOrchestrator.compile_pipelines() followed by
        PipelineOrchestrator.execute_compiled_plate().
        """
        logger.warning("PipelineExecutor.execute() is deprecated. Use PipelineOrchestrator methods.")
        # Original logic removed. Raise an error or return minimal response.
        if isinstance(context, list):
            if not context:
                 return [] # Return empty list if input is empty list
            # For multiple contexts, it's unclear what to return without execution
            # For now, returning them unchanged to avoid breaking type signatures badly.
            return context 
        return context # Return single context unchanged

    # _visualize_step_result and _verify_gpu_assignments are no longer relevant
    # as StepResult is removed and GPU validation is part of compilation.

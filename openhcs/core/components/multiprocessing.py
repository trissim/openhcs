"""
Generic multiprocessing coordinator for configurable axis iteration.

This module provides a generic replacement for the hardcoded Well-based
multiprocessing logic, allowing any component to serve as the multiprocessing axis.
"""

import logging
from typing import Generic, TypeVar, Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass

from .framework import ComponentConfiguration

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Enum)


@dataclass
class Task(Generic[T]):
    """Represents a single multiprocessing task."""
    axis_value: str
    context: Any  # ProcessingContext - avoiding circular import


class MultiprocessingCoordinator(Generic[T]):
    """
    Generic coordinator for multiprocessing along any component axis.
    
    This class replaces the hardcoded Well-based multiprocessing logic with
    a configurable system that can use any component as the multiprocessing axis.
    """
    
    def __init__(self, config: ComponentConfiguration[T]):
        """
        Initialize the coordinator with a component configuration.
        
        Args:
            config: ComponentConfiguration specifying the multiprocessing axis
        """
        self.config = config
        self.axis = config.multiprocessing_axis
        logger.debug(f"MultiprocessingCoordinator initialized with axis: {self.axis.value}")
    
    def create_tasks(
        self,
        orchestrator,
        pipeline_definition: List[Any],
        axis_filter: Optional[List[str]] = None
    ) -> Dict[str, Task[T]]:
        """
        Create tasks for each value of the multiprocessing axis.

        This method replaces the hardcoded well iteration logic with generic
        component iteration based on the configured multiprocessing axis.

        Args:
            orchestrator: PipelineOrchestrator instance
            pipeline_definition: List of pipeline steps
            axis_filter: Optional filter for axis values

        Returns:
            Dictionary mapping axis values to Task objects
        """
        # Get axis values from orchestrator using the multiprocessing axis component directly
        # The orchestrator should accept VariableComponents enum directly
        axis_values = orchestrator.get_component_keys(self.axis, axis_filter)
        
        if not axis_values:
            logger.warning(f"No {self.axis.value} values found for multiprocessing")
            return {}
        
        logger.info(f"Creating tasks for {len(axis_values)} {self.axis.value} values: {axis_values}")
        
        # Create tasks
        tasks = {}
        for axis_value in axis_values:
            context = orchestrator.create_context(axis_value)
            tasks[axis_value] = Task(axis_value=axis_value, context=context)
            logger.debug(f"Created task for {self.axis.value}: {axis_value}")
        
        return tasks
    
    def execute_tasks(
        self,
        tasks: Dict[str, Task[T]],
        pipeline_definition: List[Any],
        executor,
        processor_func: Callable
    ) -> Dict[str, Any]:
        """
        Execute tasks using the provided executor and processor function.
        
        This method provides a generic interface for task execution that can
        work with any multiprocessing axis.
        
        Args:
            tasks: Dictionary of tasks to execute
            pipeline_definition: List of pipeline steps
            executor: Executor instance (ThreadPoolExecutor or ProcessPoolExecutor)
            processor_func: Function to process each task
            
        Returns:
            Dictionary mapping axis values to execution results
        """
        if not tasks:
            logger.warning("No tasks to execute")
            return {}
        
        logger.info(f"Executing {len(tasks)} tasks for {self.axis.value} axis")
        
        # Submit tasks to executor
        future_to_axis_value = {}
        for axis_value, task in tasks.items():
            future = executor.submit(processor_func, pipeline_definition, task.context)
            future_to_axis_value[future] = axis_value
            logger.debug(f"Submitted task for {self.axis.value}: {axis_value}")
        
        # Collect results
        results = {}
        import concurrent.futures
        for future in concurrent.futures.as_completed(future_to_axis_value):
            axis_value = future_to_axis_value[future]
            try:
                result = future.result()
                results[axis_value] = result
                logger.debug(f"Task completed for {self.axis.value}: {axis_value}")
            except Exception as e:
                logger.error(f"Task failed for {self.axis.value} {axis_value}: {e}")
                results[axis_value] = {"status": "error", "error": str(e)}
        
        logger.info(f"Completed execution of {len(results)} tasks")
        return results

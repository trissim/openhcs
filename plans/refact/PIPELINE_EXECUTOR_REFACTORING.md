# PipelineExecutor Refactoring

## Overview

This document outlines the changes required to refactor the PipelineExecutor class to support the new stateless execution model with strict separation between compilation and execution phases, and VFS-based cross-step communication.

## 1. Current Implementation

The current PipelineExecutor (`openhcs/core/pipeline/executor.py`):

1. Enforces StepResult return type from steps
2. Updates the ProcessingContext with each StepResult
3. Allows context modification during execution

Key problematic code:

```python
# From executor.py
step_result = step.process(ctx)

# Enforce strict StepResult return type (Clause 297)
if not isinstance(step_result, StepResult):
    step_type = type(step_result).__name__
    raise TypeError(
        f"Step '{step.name}' returned {step_type} instead of StepResult. "
        f"All steps must return a StepResult instance "
        f"(Clause 297 — Immutable Result Enforcement)."
    )

# Update context with step result
ctx.update_from_step_result(step_result)
```

## 2. Required Changes

1. Remove StepResult enforcement and context updating
2. Ensure PipelineExecutor only reads from the immutable context
3. Change error handling to catch exceptions rather than checking return values
4. Update doctrines to reflect the new execution model

## 3. Implementation Approach

```python
class PipelineExecutor:
    """
    Executes a compiled pipeline.
    
    This class is responsible for:
    1. Executing each step in the pipeline according to their step plans
    2. Ensuring steps don't modify the context
    3. Maintaining pipeline order for proper cross-step communication
    4. Managing parallel execution of multiple pipelines
    
    This executor is completely stateless and does not perform any GPU management
    or resource allocation. All GPU IDs must be assigned during planning/compilation,
    not during execution.
    
    🔒 Clause 12 — Absolute Clean Execution
    🔒 Clause 66 — Immutability After Construction  
    🔒 Clause 17 — VFS Exclusivity
    🔒 Clause 246 — Statelessness Mandate
    """
    
    def __init__(self, filemanager: Optional['FileManager'] = None):
        """
        Initialize the executor.
        
        Args:
            filemanager: Optional FileManager instance to provide to steps if needed
        """
        self.filemanager = filemanager
    
    def execute(
        self,
        pipeline: List[AbstractStep],
        context: 'ProcessingContext',
        visualizer: Optional[Any] = None
    ) -> 'ProcessingContext':
        """
        Execute a compiled pipeline with the given context.
        
        Args:
            pipeline: List of steps to execute
            context: Immutable ProcessingContext from compilation
            visualizer: Optional visualizer for intermediate results
            
        Returns:
            The unmodified context after execution
            
        Raises:
            TypeError: If the context is not immutable (frozen)
            Various exceptions from individual steps
        """
        # Verify that context is frozen (immutable)
        if not context.is_frozen():
            raise TypeError(
                "ProcessingContext must be frozen before execution. "
                "Use PipelineCompiler.compile to create a properly frozen context."
            )
        
        # Verify GPU assignments if needed
        self._verify_gpu_assignments(pipeline, context)
        
        logger.info("Executing pipeline with %d steps for well %s", len(pipeline), context.well_id)
        
        # Process each step in strict pipeline order
        # This order is critical for cross-step special key communication
        for i, step in enumerate(pipeline):
            step_id = step.uid
            logger.info("Executing step %d/%d: %s (ID: %s)",
                        i+1, len(pipeline), step.name, step_id)
            
            # Validate step plan exists
            if step_id not in context.step_plans:
                raise ValueError(f"Step plan not found for step: {step.name} (ID: {step_id})")
            
            # Verify pipeline position for strict ordering
            step_plan = context.step_plans[step_id]
            if step_plan.get("pipeline_position", i) != i:
                raise ValueError(
                    f"Step {step.name} is at position {i} but its step plan "
                    f"indicates position {step_plan.get('pipeline_position')}. "
                    f"Pipeline order is critical for cross-step special key communication."
                )
            
            try:
                # Execute the step (returns None)
                result = step.process(context)
                
                # Verify that step returns None, not StepResult
                if result is not None:
                    raise TypeError(
                        f"Step '{step.name}' returned {type(result).__name__} instead of None. "
                        f"All steps must return None (Clause 246 — Statelessness Mandate)."
                    )
                    
                # Visualize step output if needed
                if visualizer:
                    self._visualize_step_output(step_id, context, visualizer)
                    
            except Exception as e:
                logger.error("Error executing step %s: %s", step.name, e)
                raise
        
        logger.info("Pipeline execution completed for well %s", context.well_id)
        return context
```

## 4. Pipeline Order Enforcement

In the new design, pipeline order is critical because:
1. Special output keys from earlier steps must be available to special input keys in later steps
2. The VFS paths and data must be established before they can be consumed
3. The step plan linking established during compilation depends on this order

The executor must enforce that steps are executed in the same order as they were compiled.

## 5. Visualization Refactoring

The current visualization approach relies on StepResult. This needs to be modified:

```python
def _visualize_step_output(self, step_id: str, context: 'ProcessingContext', visualizer: Any) -> None:
    """
    Visualize step output using the provided visualizer.
    
    Args:
        step_id: ID of the step whose output to visualize
        context: The processing context
        visualizer: The visualizer to use
    """
    step_plan = context.step_plans.get(step_id, {})
    if not step_plan.get('visualize', False):
        return
        
    # Get output paths from step plan
    output_dir = step_plan.get('output_dir')
    if not output_dir:
        return
        
    # Let visualizer handle the output directory directly
    visualizer.visualize(output_dir, context=context, step_id=step_id)
```

## 6. Parallel Execution

The parallel execution methods also need to be updated to handle the strict pipeline ordering requirement:

```python
def execute_parallel(
    self,
    pipeline_list: List[List[AbstractStep]],
    context_list: List['ProcessingContext'],
    num_workers: Optional[int] = None,
    visualizer: Optional[Any] = None
) -> List['ProcessingContext']:
    """
    Execute multiple pipelines in parallel.
    
    Important: Each pipeline-context pair is independent and executed in isolation.
    Cross-pipeline special key linking is not supported.
    
    Args:
        pipeline_list: List of pipelines to execute
        context_list: List of contexts from compilation (must be frozen)
        num_workers: Number of worker threads to use
        visualizer: Optional visualizer for intermediate results
        
    Returns:
        List of contexts after execution
    """
    # Validation code
    # ...
    
    # Define the execution logic inline
    def execute_pipeline(pipeline, ctx):
        # Execute single pipeline, letting exceptions propagate
        return self.execute(pipeline, ctx, visualizer)
    
    # Use ThreadPoolExecutor for parallel execution
    # ...
    
    return results
```

## 7. Integration Points

1. **PipelineOrchestrator:** Ensure it calls compile first, then execute separately
2. **Step Implementations:** Update all steps to return None, not StepResult
3. **Visualization:** Update to work with context and output paths directly

## 8. Doctrinal Enforcement

- **Clause 12 — Absolute Clean Execution:** No context modifications during execution
- **Clause 66 — Immutability After Construction:** Context must be frozen before execution
- **Clause 17 — VFS Exclusivity:** All inter-step communication via VFS
- **Clause 246 — Statelessness Mandate:** Steps have no state after compilation
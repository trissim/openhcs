# ProcessingContext Immutability Implementation

## Overview

This document defines the implementation approach for enforcing immutability in the ProcessingContext after compilation. This is a critical part of the refactoring effort to ensure strict separation between compilation and execution phases and support cross-step special key communication.

## 1. Current Implementation

The current ProcessingContext (`openhcs/core/context/processing_context.py`) has:

1. A mutable interface through `update_from_step_result(step_result)`
2. No mechanism to prevent modification after compilation
3. Direct attribute setting through `setattr` in `__init__`

```python
def update_from_step_result(self, step_result: 'StepResult') -> None:
    """Update context from a step result."""
    # Handle output_path if present
    if step_result.output_path is not None:
        self.outputs['output_path'] = step_result.output_path

    # Update outputs
    for key, value in step_result.results.items():
        self.outputs[key] = value

    # Apply context updates
    for key, value in step_result.context_updates.items():
        setattr(self, key, value)
```

## 2. Required Changes

1. Add immutability flag to prevent modifications after compilation
2. Remove `update_from_step_result` method (as StepResult is being eliminated)
3. Add validation checks to prevent attribute modification after freezing
4. Ensure the filemanager and step_plans are properly initialized

## 3. Implementation Approach

```python
class ProcessingContext:
    """
    Maintains state during pipeline execution.
    
    The ProcessingContext holds the immutable step plans created during compilation
    and provides access to the FileManager for steps during execution. After
    compilation, the context becomes immutable to ensure strict separation between
    compilation and execution phases.
    
    🔒 Clause 66 — Immutability After Construction
    🔒 Clause 281 — Context-Bound Identifiers
    🔒 Clause 12 — Absolute Clean Execution
    """
    
    def __init__(
        self,
        step_plans: Optional[Dict[str, Dict[str, Any]]] = None,
        well_id: Optional[str] = None,
        filemanager: Optional['FileManager'] = None,
        **kwargs
    ):
        """
        Initialize the processing context.
        
        Args:
            step_plans: Dictionary mapping step IDs to execution plans
            well_id: Identifier of the well being processed
            filemanager: FileManager instance for VFS operations
            **kwargs: Additional context attributes
        """
        self.step_plans = step_plans or {}
        self.well_id = well_id
        self.filemanager = filemanager
        self._is_frozen = False  # Immutability flag
        
        # Add any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Override setattr to enforce immutability after freezing.
        
        Args:
            name: Attribute name
            value: Attribute value
            
        Raises:
            AttributeError: If the context is frozen and attempting to modify attributes
        """
        if getattr(self, '_is_frozen', False) and name != '_is_frozen':
            raise AttributeError(
                f"Cannot modify frozen ProcessingContext attribute '{name}'. "
                f"Context is immutable after compilation (Clause 66)."
            )
        super().__setattr__(name, value)
    
    def freeze(self) -> None:
        """
        Freeze the context to prevent further modifications.
        
        This method should be called by the PipelineCompiler after compilation is complete.
        
        🔒 Clause 66 — Immutability After Construction
        """
        # Validate that required attributes are present
        if not self.step_plans:
            raise ValueError("Cannot freeze ProcessingContext with empty step_plans")
            
        if not self.filemanager:
            raise ValueError("Cannot freeze ProcessingContext without a filemanager")
            
        if not self.well_id:
            raise ValueError("Cannot freeze ProcessingContext without a well_id")
        
        self._is_frozen = True
    
    def is_frozen(self) -> bool:
        """
        Check if the context is frozen.
        
        Returns:
            True if the context is frozen, False otherwise
        """
        return self._is_frozen
    
    def validate_step_id(self, step_id: str) -> None:
        """
        Validate that a step ID exists in the step plans.
        
        Args:
            step_id: Step ID to validate
            
        Raises:
            ValueError: If the step ID does not exist in step_plans
        """
        if step_id not in self.step_plans:
            raise ValueError(f"Step ID '{step_id}' not found in step plans")
    
    def get_step_plan(self, step_id: str) -> Dict[str, Any]:
        """
        Get the step plan for a step.
        
        Args:
            step_id: Step ID
            
        Returns:
            Step plan for the step
            
        Raises:
            ValueError: If the step ID does not exist in step_plans
        """
        self.validate_step_id(step_id)
        return self.step_plans[step_id]
        
    def inject_plan(self, step_id: str, plan: Dict[str, Any]) -> None:
        """
        Inject a step plan into the context.
        
        This method is the canonical way to add step plans to the context.
        All step configuration must be injected into the context using this method.
        Cannot be called if the context is frozen.
        
        Args:
            step_id: The unique identifier of the step
            plan: The step execution plan
            
        Raises:
            AttributeError: If the context is frozen
            
        🔒 Clause 281 — Context-Bound Identifiers
        🔒 Clause 524 — Step = Declaration = ID = Runtime Authority
        """
        if self._is_frozen:
            raise AttributeError(
                f"Cannot inject plan for step '{step_id}'. "
                f"ProcessingContext is immutable after compilation (Clause 66)."
            )
        self.step_plans[step_id] = plan
```

## 4. Integration with the Pipeline Compiler and Executor

### Compiler Integration

The PipelineCompiler must freeze the ProcessingContext after compilation:

```python
def compile_pipeline(steps, input_dir, well_id, filemanager):
    # Perform compilation steps
    # ...
    
    # Create context with plans
    context = ProcessingContext(
        step_plans=step_plans,
        well_id=well_id,
        filemanager=filemanager
    )
    
    # Freeze the context to prevent further modifications
    context.freeze()
    
    return context
```

### Executor Integration

The PipelineExecutor must verify that the context is frozen:

```python
def execute(self, pipeline, context):
    # Verify context is frozen
    if not context.is_frozen():
        raise ValueError(
            "Cannot execute pipeline with non-frozen ProcessingContext. "
            "Context must be compiled and frozen before execution."
        )
    
    # Execute steps
    # ...
```

## 5. Migration Strategy

1. Update ProcessingContext with immutability support
2. Add `freeze()` call to PipelineCompiler
3. Remove all code that modifies the context during execution
4. Ensure all test fixtures create proper immutable contexts
5. Update all step implementations to work with immutable context

## 6. Doctrinal Enforcement

- **Clause 66 — Immutability After Construction:** Enforced by the freezing mechanism
- **Clause 281 — Context-Bound Identifiers:** Maintained by using step_plans for all execution parameters
- **Clause 12 — Absolute Clean Execution:** Supported by preventing modifications during execution
- **Clause 17 — VFS Exclusivity:** All inter-step communication through FileManager
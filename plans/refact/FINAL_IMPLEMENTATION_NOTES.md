# OpenHCS Refactoring: Final Implementation Notes

This document provides critical missing details to ensure the refactoring plans are unambiguous and fully implementable by a Sovereign Executor.

## 1. Error Handling Strategy

The primary remaining ambiguity concerns error handling during compilation and execution, particularly for special key validation and path resolution.

### Compilation-Phase Error Handling

Implement a hierarchical error system for special key validation:

```python
class PipelineError(Exception):
    """Base class for pipeline errors."""
    pass

class CompilationError(PipelineError):
    """Error during pipeline compilation."""
    pass

class SpecialKeyError(CompilationError):
    """Error in special key validation."""
    pass
```

Key validation should fail fast with specific errors:

1. **Missing Provider**: When a step requires a special input with no matching special output
   ```python
   raise SpecialKeyError(
       f"Step '{step.name}' requires special input '{key}' but no step provides it"
   )
   ```

2. **Order Violation**: When a step requires input from a later step
   ```python
   raise SpecialKeyError(
       f"Step '{step.name}' requires special input '{key}' from step '{provider.name}' "
       f"which comes after it in the pipeline"
   )
   ```

3. **Duplicate Provider**: When multiple steps provide the same special output key
   ```python
   raise SpecialKeyError(
       f"Multiple steps provide special output '{key}': {', '.join(provider_names)}"
   )
   ```

### VFS Path Resolution

Since VFS is an overlay of the actual filesystem, use a simple path resolution strategy:

```python
def resolve_special_io_path(key: str, provider_id: str, consumer_id: str, well_id: str) -> str:
    """
    Resolve path for special IO between steps.
    
    Args:
        key: Special key name
        provider_id: ID of providing step
        consumer_id: ID of consuming step
        well_id: Current well ID
        
    Returns:
        Filesystem path for special IO
    """
    # Simple predictable path pattern
    return f"special/{well_id}/{key}/{provider_id}_{consumer_id}"
```

## 2. Special Key Data Type Handling

Built-in `@special_input` and `@special_output` decorators already attach metadata to functions.
During step planning:

1. Extract the decorated attributes during compilation:
   ```python
   special_outputs = get_special_outputs(step.func)  # Returns set
   special_inputs = get_special_inputs(step.func)    # Returns dict[key→required]
   ```

2. During path planning, keys are matched between steps:
   ```python
   # For each step with special_inputs
   for key in special_inputs:
       # Find provider step
       provider_step = find_provider_for_key(key, steps[:i])  # Only look at previous steps
       
       # Link them in step_plans
       step_plans[step.uid]["special_inputs"][key] = {
           "source_step_id": provider_step.uid,
           "path": resolve_special_io_path(key, provider_step.uid, step.uid, well_id)
       }
       
       step_plans[provider_step.uid]["special_outputs"][key]["target_step_ids"].append(step.uid)
   ```

3. Keys are validated in one pass without requiring multiple special path planning steps.

## 3. Memory Type Handling for Special I/O

For memory type handling in special I/O:

1. Provider step determines output memory type
2. Consumer step is responsible for conversion if needed
3. No intermediate serialization required unless crossing backend boundaries

```python
# In consumer step's process method:
def process(self, context):
    step_plan = context.step_plans[self.uid]
    
    # Load primary input (array data)
    # ...
    
    # Load any special inputs
    special_args = {}
    if "special_inputs" in step_plan:
        for key, info in step_plan["special_inputs"].items():
            # FileManager handles backend routing
            data = context.filemanager.load(
                info["path"], 
                info["backend"]
            )
            special_args[key] = data
    
    # Call function with special args as keyword arguments
    result = self.func(primary_array, **special_args)
```

## 4. Final Missing Elements

The updated refactoring plans are now complete and unambiguous, with these clarifications addressing all remaining elements:

1. **Existing Decorators**: Use existing `@special_input` and `@special_output` decorators
2. **Simple VFS**: VFS is an overlay of the actual filesystem
3. **Error Handling**: Detailed error strategy with specific error types
4. **Key Resolution**: Simple path resolution for special I/O keys

These additions ensure the plans fully specify the implementation requirements for a Sovereign Executor.
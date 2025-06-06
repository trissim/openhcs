# Special I/O System: Cross-Step Communication

## Overview

The Special I/O system enables data exchange between pipeline steps outside the primary input/output directories. It uses a declarative decorator system combined with VFS path resolution to create directed data flow connections between steps.

## Architecture Components

### Decorator System

Functions declare their special I/O requirements using decorators:

```python
from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs

@special_outputs("positions", "metadata")
def generate_positions(image_stack):
    """Function that produces special outputs."""
    positions = calculate_positions(image_stack)
    metadata = extract_metadata(image_stack)
    
    # Return: (main_output, special_output_1, special_output_2, ...)
    return processed_image, positions, metadata

@special_inputs("positions", "metadata")
def stitch_images(image_stack, positions, metadata):
    """Function that consumes special inputs."""
    # positions and metadata are automatically loaded from VFS
    return stitch(image_stack, positions, metadata)
```

### Decorator Implementation

```python
def special_outputs(*output_names: str) -> Callable[[F], F]:
    """Mark function as producing special outputs."""
    def decorator(func: F) -> F:
        func.__special_outputs__ = set(output_names)
        return func
    return decorator

def special_inputs(*input_names: str) -> Callable[[F], F]:
    """Mark function as consuming special inputs."""
    def decorator(func: F) -> F:
        # Store as dict with True values for compatibility
        func.__special_inputs__ = {name: True for name in input_names}
        return func
    return decorator
```

## Compilation-Time Path Resolution

### Phase 1: Special Output Registration

During path planning, the compiler extracts special outputs and creates VFS paths:

```python
# In PipelinePathPlanner.prepare_pipeline_paths()
def process_special_outputs(step, step_output_dir, declared_outputs):
    """Process special outputs for a step."""
    
    # Extract special outputs from function decorators
    s_outputs_keys = getattr(step.func, '__special_outputs__', set())
    
    special_outputs = {}
    for key in sorted(list(s_outputs_keys)):
        # Convert to snake_case for file naming
        snake_case_key = to_snake_case(key)
        
        # Generate VFS path: [output_dir]/[snake_case_key].pkl
        output_path = Path(step_output_dir) / f"{snake_case_key}.pkl"
        special_outputs[key] = {"path": str(output_path)}
        
        # Register this output globally for linking
        declared_outputs[key] = {
            "step_id": step.uid,
            "position": step_position,
            "path": str(output_path)
        }
    
    return special_outputs
```

### Phase 2: Special Input Linking

The compiler links special inputs to previously declared outputs:

```python
def process_special_inputs(step, step_position, declared_outputs):
    """Link special inputs to their source outputs."""
    
    # Extract special inputs from function decorators
    s_inputs_dict = getattr(step.func, '__special_inputs__', {})
    
    special_inputs = {}
    for key in s_inputs_dict.keys():
        # Find the source step that produces this output
        if key not in declared_outputs:
            raise ValueError(f"Special input '{key}' not found in any previous step")
        
        source_info = declared_outputs[key]
        source_step_position = source_info["position"]
        
        # Validate dependency order (inputs must come from earlier steps)
        if source_step_position >= step_position:
            raise ValueError(
                f"Special input '{key}' in step {step_position} "
                f"depends on output from step {source_step_position}. "
                "Dependencies must be from earlier steps."
            )
        
        # Link to source path
        special_inputs[key] = {"path": source_info["path"]}
    
    return special_inputs
```

### Path Generation Strategy

Special I/O paths follow a standardized pattern:

```python
def generate_special_io_path(step_output_dir, key):
    """Generate standardized VFS path for special I/O."""
    
    # Convert key to snake_case for filesystem compatibility
    snake_case_key = to_snake_case(key)
    
    # Path format: [step_output_dir]/[snake_case_key].pkl
    return str(Path(step_output_dir) / f"{snake_case_key}.pkl")

# Examples:
# Key "positions" → "positions.pkl"
# Key "cellMetadata" → "cell_metadata.pkl"
# Key "stitchingParams" → "stitching_params.pkl"
```

## Runtime Execution

### Special Output Handling

During function execution, special outputs are saved to VFS:

```python
def _execute_function_core(func_callable, main_data_arg, base_kwargs, 
                          context, special_inputs_plan, special_outputs_plan):
    """Execute function with special I/O handling."""
    
    # 1. Load special inputs from VFS
    final_kwargs = base_kwargs.copy()
    for arg_name, special_path in special_inputs_plan.items():
        logger.debug(f"Loading special input '{arg_name}' from '{special_path}'")
        special_data = context.filemanager.load(special_path, "memory")
        final_kwargs[arg_name] = special_data
    
    # 2. Execute function
    raw_function_output = func_callable(main_data_arg, **final_kwargs)
    
    # 3. Handle special outputs
    if special_outputs_plan:
        # Function returns (main_output, special_output_1, special_output_2, ...)
        if isinstance(raw_function_output, tuple):
            main_output = raw_function_output[0]
            special_values = raw_function_output[1:]
        else:
            raise ValueError("Function with special outputs must return tuple")
        
        # Save special outputs positionally
        for i, (output_key, vfs_path) in enumerate(special_outputs_plan.items()):
            if i < len(special_values):
                value_to_save = special_values[i]
                logger.debug(f"Saving special output '{output_key}' to '{vfs_path}'")
                context.filemanager.save(value_to_save, vfs_path, "memory")
            else:
                raise ValueError(f"Missing special output value for key '{output_key}'")
        
        return main_output
    else:
        return raw_function_output
```

### Step Plan Integration

Special I/O information is stored in step plans:

```python
# Example step plan with special I/O
step_plan = {
    "step_name": "Position Generation",
    "step_id": "step_001",
    "input_dir": "/workspace/A01/input",
    "output_dir": "/workspace/A01/step1_out",
    
    # Special outputs produced by this step
    "special_outputs": {
        "positions": {"path": "/workspace/A01/step1_out/positions.pkl"},
        "metadata": {"path": "/workspace/A01/step1_out/metadata.pkl"}
    },
    
    # Special inputs consumed by this step (empty for first step)
    "special_inputs": {},
    
    # Other configuration...
}

# Later step that consumes the outputs
step_plan_2 = {
    "step_name": "Image Stitching",
    "step_id": "step_002",
    "input_dir": "/workspace/A01/step1_out",
    "output_dir": "/workspace/A01/step2_out",
    
    # Special inputs linked to previous step's outputs
    "special_inputs": {
        "positions": {"path": "/workspace/A01/step1_out/positions.pkl"},
        "metadata": {"path": "/workspace/A01/step1_out/metadata.pkl"}
    },
    
    # No special outputs
    "special_outputs": {},
}
```

## Data Flow Validation

### Dependency Graph Construction

The compiler builds a dependency graph to validate special I/O connections:

```python
def validate_special_io_dependencies(steps):
    """Validate special I/O dependency graph."""
    
    # Build dependency graph
    dependency_graph = {}
    declared_outputs = {}
    
    for i, step in enumerate(steps):
        step_id = step.uid
        dependency_graph[step_id] = {"depends_on": [], "provides": []}
        
        # Register outputs
        special_outputs = getattr(step.func, '__special_outputs__', set())
        for output_key in special_outputs:
            if output_key in declared_outputs:
                raise ValueError(f"Duplicate special output key: {output_key}")
            declared_outputs[output_key] = {"step_id": step_id, "position": i}
            dependency_graph[step_id]["provides"].append(output_key)
        
        # Register dependencies
        special_inputs = getattr(step.func, '__special_inputs__', {})
        for input_key in special_inputs.keys():
            if input_key not in declared_outputs:
                raise ValueError(f"Unresolved special input: {input_key}")
            
            source_step = declared_outputs[input_key]["step_id"]
            dependency_graph[step_id]["depends_on"].append(source_step)
    
    # Check for cycles
    if has_cycles(dependency_graph):
        raise ValueError("Circular dependencies detected in special I/O")
    
    return dependency_graph

def has_cycles(graph):
    """Check for cycles in dependency graph using DFS."""
    visited = set()
    rec_stack = set()
    
    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph[node]["depends_on"]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    
    return False
```

### Order Validation

```python
def validate_execution_order(steps):
    """Ensure special inputs come from earlier steps."""
    
    declared_outputs = {}
    
    for i, step in enumerate(steps):
        # Check inputs reference earlier steps
        special_inputs = getattr(step.func, '__special_inputs__', {})
        for input_key in special_inputs.keys():
            if input_key not in declared_outputs:
                raise ValueError(f"Special input '{input_key}' not declared by any previous step")
            
            source_position = declared_outputs[input_key]["position"]
            if source_position >= i:
                raise ValueError(
                    f"Special input '{input_key}' in step {i} "
                    f"references output from step {source_position}. "
                    "Dependencies must be from earlier steps."
                )
        
        # Register outputs for future steps
        special_outputs = getattr(step.func, '__special_outputs__', set())
        for output_key in special_outputs:
            declared_outputs[output_key] = {"position": i, "step_id": step.uid}
```

## VFS Integration

### Backend Selection

Special I/O typically uses memory backend for performance:

```python
def plan_special_io_backends(step_plans):
    """Plan backends for special I/O data."""
    
    for step_id, step_plan in step_plans.items():
        # Special I/O usually uses memory backend
        for output_key, output_info in step_plan.get("special_outputs", {}).items():
            output_info["backend"] = "memory"
        
        for input_key, input_info in step_plan.get("special_inputs", {}).items():
            input_info["backend"] = "memory"
```

### Serialization Handling

The VFS automatically handles serialization for special I/O data:

```python
# Memory backend stores Python objects directly
context.filemanager.save(positions_array, "/vfs/positions.pkl", "memory")
# → Stored as Python object in memory

# Disk backend would serialize to pickle format
context.filemanager.save(positions_array, "/workspace/positions.pkl", "disk")
# → Serialized to .pkl file on disk
```

## Error Handling

### Common Validation Errors

```python
class SpecialIOError(Exception):
    """Base class for special I/O errors."""
    pass

class UnresolvedSpecialInputError(SpecialIOError):
    """Special input has no corresponding output."""
    pass

class CircularDependencyError(SpecialIOError):
    """Circular dependency detected in special I/O."""
    pass

class SpecialOutputMismatchError(SpecialIOError):
    """Function output count doesn't match declared special outputs."""
    pass
```

### Runtime Error Handling

```python
def safe_special_io_execution(func, special_outputs_plan, *args, **kwargs):
    """Execute function with safe special I/O handling."""
    
    try:
        result = func(*args, **kwargs)
        
        if special_outputs_plan:
            if not isinstance(result, tuple):
                raise SpecialOutputMismatchError(
                    f"Function {func.__name__} declared special outputs "
                    f"but returned {type(result)}, expected tuple"
                )
            
            expected_count = len(special_outputs_plan) + 1  # +1 for main output
            actual_count = len(result)
            
            if actual_count != expected_count:
                raise SpecialOutputMismatchError(
                    f"Function {func.__name__} returned {actual_count} values "
                    f"but declared {expected_count} (1 main + {len(special_outputs_plan)} special)"
                )
        
        return result
        
    except Exception as e:
        logger.error(f"Special I/O execution failed for {func.__name__}: {e}")
        raise
```

## Performance Considerations

### Memory Management

```python
def optimize_special_io_memory(step_plans):
    """Optimize memory usage for special I/O."""
    
    # Identify data lifetime
    data_lifetime = {}
    for step_id, step_plan in step_plans.items():
        for output_key in step_plan.get("special_outputs", {}):
            data_lifetime[output_key] = {"created": step_id, "last_used": None}
    
    for step_id, step_plan in step_plans.items():
        for input_key in step_plan.get("special_inputs", {}):
            if input_key in data_lifetime:
                data_lifetime[input_key]["last_used"] = step_id
    
    # Plan cleanup points
    cleanup_points = {}
    for key, lifetime in data_lifetime.items():
        if lifetime["last_used"]:
            cleanup_step = lifetime["last_used"]
            if cleanup_step not in cleanup_points:
                cleanup_points[cleanup_step] = []
            cleanup_points[cleanup_step].append(key)
    
    return cleanup_points
```

### Caching Strategy

```python
class SpecialIOCache:
    """Cache for special I/O data to avoid redundant loading."""
    
    def __init__(self):
        self._cache = {}
        self._access_count = {}
    
    def get(self, path, filemanager):
        """Get data with caching."""
        if path not in self._cache:
            self._cache[path] = filemanager.load(path, "memory")
            self._access_count[path] = 0
        
        self._access_count[path] += 1
        return self._cache[path]
    
    def cleanup_unused(self, threshold=1):
        """Remove data accessed less than threshold times."""
        to_remove = [
            path for path, count in self._access_count.items()
            if count < threshold
        ]
        
        for path in to_remove:
            del self._cache[path]
            del self._access_count[path]
```

## Future Enhancements

### Planned Features

1. **Optional Special Inputs**: Support for optional special inputs with default values
2. **Typed Special I/O**: Type hints and validation for special I/O data
3. **Streaming Special I/O**: Support for large special I/O data that doesn't fit in memory
4. **Special I/O Versioning**: Version tracking for special I/O data compatibility
5. **Cross-Pipeline Special I/O**: Share special I/O data between different pipeline runs

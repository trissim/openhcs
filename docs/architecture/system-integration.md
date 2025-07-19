# System Integration: VFS, Memory Types, and Compilation

## Overview

This document describes how the three core systems of OpenHCS work together to provide seamless data processing: the Virtual File System (VFS), the Memory Type System, and the Pipeline Compilation System.

**Note**: This document describes the actual system integration implementation. Some optimization strategies are planned for future development.

## Integration Architecture

### The Four-Layer Architecture Stack

```
┌─────────────────────────────────────────────────────────────┐
│                 Configuration Management                    │
│  • Hierarchical config flow (Global → Context → Steps)    │
│  • Immutable dataclasses with pull-based access           │
│  • Live configuration updates and validation              │
│  • Backend selection and path planning configuration      │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Compilation                     │
│  • Function pattern resolution                             │
│  • Memory type extraction from decorators                  │
│  • Step plan generation and validation                     │
│  • Resource allocation (GPU, backends)                     │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Memory Type System                       │
│  • Cross-library array conversion (numpy ↔ torch ↔ cupy)  │
│  • GPU device management and placement                     │
│  • Stack/unstack operations with type conversion           │
│  • Strict validation and error handling                    │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                 Virtual File System (VFS)                  │
│  • Backend abstraction (disk, memory, zarr)               │
│  • Format-specific serialization/deserialization          │
│  • Location transparency and path virtualization           │
│  • Type-aware storage operations                          │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Coordination

The three systems coordinate to provide end-to-end data processing:

1. **Compilation Phase**: Determines what conversions are needed
2. **Execution Phase**: Orchestrates conversions and processing
3. **Storage Phase**: Handles persistence and retrieval

## Compilation-Time Integration

### Memory Type Planning

During compilation, the system extracts memory type requirements and plans conversions:

```python
# Phase 1: Extract memory types from function decorators
def analyze_function_memory_requirements(func_pattern):
    """Extract memory types from function patterns."""
    if isinstance(func_pattern, dict):
        # Component-specific: different memory types per component
        return {
            component: {
                'input_type': getattr(func, 'input_memory_type'),
                'output_type': getattr(func, 'output_memory_type')
            }
            for component, func in func_pattern.items()
        }
    elif isinstance(func_pattern, list):
        # Sequential: validate consistency across chain
        memory_types = []
        for func in func_pattern:
            input_type = getattr(func, 'input_memory_type')
            output_type = getattr(func, 'output_memory_type')
            memory_types.append((input_type, output_type))
        
        # Validate chain compatibility
        for i in range(len(memory_types) - 1):
            if memory_types[i][1] != memory_types[i+1][0]:
                raise MemoryTypeError(f"Incompatible memory types in chain")
        
        return memory_types[0][0], memory_types[-1][1]  # First input, last output
    else:
        # Single function
        return getattr(func_pattern, 'input_memory_type'), getattr(func_pattern, 'output_memory_type')
```

### Backend Selection Coordination

The compiler coordinates memory types with VFS backend selection:

```python
def plan_backend_selection(step_position, memory_types, data_size):
    """Coordinate backend selection with memory type requirements."""
    
    # First step: must read from disk
    if step_position == 0:
        read_backend = "disk"
    else:
        # Intermediate steps can use memory backend
        read_backend = "memory"
    
    # Last step: must write to disk  
    if step_position == last_position:
        write_backend = "disk"
    else:
        # GPU memory types benefit from memory backend
        if memory_types['output_type'] in GPU_MEMORY_TYPES:
            write_backend = "memory"  # Keep on GPU
        else:
            write_backend = "memory"  # Avoid disk I/O
    
    return read_backend, write_backend
```

### Step Plan Generation

The compiler generates comprehensive step plans that coordinate all systems:

```python
step_plan = {
    # Basic metadata
    "step_name": "GPU Image Processing",
    "step_type": "FunctionStep",
    "well_id": "A01",
    
    # VFS configuration
    "input_dir": "/workspace/A01/input",
    "output_dir": "/workspace/A01/step1_out",
    "read_backend": "disk",      # From backend planner
    "write_backend": "memory",   # From backend planner
    
    # Memory type configuration  
    "input_memory_type": "torch",   # From function decorator
    "output_memory_type": "torch",  # From function decorator
    "gpu_id": 0,                    # From GPU resource planner
    
    # Function pattern configuration
    "func_pattern": gpu_processing_func,
    "variable_components": ["site"],
    "group_by": "channel",
    
    # Special I/O configuration
    "special_inputs": {
        "positions": {
            "path": "/vfs/positions.pkl",
            "backend": "memory"
        }
    },
    "special_outputs": {
        "metadata": {
            "path": "/vfs/metadata.pkl", 
            "backend": "memory"
        }
    }
}
```

## Runtime Integration

### Complete Execution Flow

During execution, the three systems work together seamlessly:

```python
def process(self, context: ProcessingContext):
    """Complete execution flow showing system integration (FunctionStep.process)."""

    step_plan = context.step_plans[self.step_id]

    # 1. VFS: Load images from storage
    raw_slices = []
    for file_path in matching_files:
        # VFS handles format-specific deserialization
        image = context.filemanager.load_image(
            file_path,
            step_plan['read_backend']
        )
        raw_slices.append(image)  # Usually numpy arrays from TIFF

    # 2. Memory System: Stack with type conversion
    image_stack = stack_slices(
        slices=raw_slices,
        memory_type=step_plan['input_memory_type'],  # torch
        gpu_id=step_plan.get('gpu_id')               # 0 or None
    )
    # Result: torch.Tensor on GPU 0
    
    # 3. Load special inputs (if any)
    special_kwargs = {}
    for input_name, input_config in step_plan['special_inputs'].items():
        # VFS loads from specified backend
        special_data = context.filemanager.load(
            input_config['path'],
            input_config['backend']
        )
        special_kwargs[input_name] = special_data
    
    # 4. Execute function in native memory type
    # Function pattern resolution handled by prepare_patterns_and_functions
    result_stack = self._execute_function_core(image_stack, step_plan, context, **special_kwargs)
    # Function operates entirely in torch on GPU
    
    # 5. Handle special outputs (if any)
    if hasattr(result_stack, '__len__') and len(result_stack) > 1:
        main_result = result_stack[0]
        special_outputs = result_stack[1:]
        
        for i, (output_name, output_config) in enumerate(step_plan['special_outputs'].items()):
            # VFS saves to specified backend
            context.filemanager.save(
                special_outputs[i],
                output_config['path'],
                output_config['backend']
            )
    else:
        main_result = result_stack
    
    # 6. Memory System: Unstack with type conversion
    output_slices = unstack_slices(
        array=main_result,
        memory_type=step_plan['output_memory_type'],  # torch
        gpu_id=step_plan['gpu_id']                    # 0
    )
    # Result: List of torch.Tensor on GPU 0
    
    # 7. VFS: Save images to storage
    for i, slice_2d in enumerate(output_slices):
        output_path = step_plan['output_dir'] / f"processed_{i:03d}.tif"
        
        # VFS handles memory type conversion for disk storage
        context.filemanager.save_image(
            slice_2d,                        # torch.Tensor
            output_path,
            step_plan['write_backend']       # memory or disk
        )
        # If disk: automatically converts torch → numpy → TIFF
        # If memory: stores torch.Tensor directly
```

### Automatic Conversion Points

The systems automatically handle conversions at key integration points:

#### VFS ↔ Memory Type Integration

```python
# VFS automatically detects and converts memory types for disk storage
def save_image_with_conversion(data, path, backend):
    if backend == "disk":
        # Convert any memory type to numpy for TIFF
        if isinstance(data, torch.Tensor):
            numpy_data = data.cpu().numpy()
        elif hasattr(data, 'get'):  # CuPy
            numpy_data = data.get()
        elif hasattr(data, 'device_get'):  # JAX
            numpy_data = jax.device_get(data)
        else:
            numpy_data = data  # Already numpy
        
        # VFS handles TIFF serialization
        tifffile.imwrite(path, numpy_data)
    
    elif backend == "memory":
        # Store in original memory type
        memory_store[path] = data
```

#### Memory Type ↔ Compilation Integration

```python
# Compilation system coordinates memory types across steps
def plan_inter_step_conversions(pipeline_steps):
    """Plan memory type conversions between pipeline steps."""
    conversions = []
    
    for i in range(len(pipeline_steps) - 1):
        current_step = pipeline_steps[i]
        next_step = pipeline_steps[i + 1]
        
        current_output = current_step.output_memory_type
        next_input = next_step.input_memory_type
        
        if current_output != next_input:
            # Plan conversion in the intermediate storage
            conversion = {
                'from_type': current_output,
                'to_type': next_input,
                'location': 'intermediate_storage',
                'method': 'automatic'
            }
            conversions.append(conversion)
    
    return conversions
```

## Performance Optimization

### Conversion Minimization

The integrated system minimizes unnecessary conversions:

```python
# Optimal: Keep data in same memory type across steps
pipeline = [
    FunctionStep(func=torch_preprocess),   # torch → torch
    FunctionStep(func=torch_process),      # torch → torch  
    FunctionStep(func=torch_postprocess)   # torch → torch
]
# Result: No memory type conversions, data stays on GPU

# Suboptimal: Mixed memory types cause conversions
pipeline = [
    FunctionStep(func=numpy_preprocess),   # numpy → numpy
    FunctionStep(func=torch_process),      # torch → torch (conversion!)
    FunctionStep(func=numpy_postprocess)   # numpy → numpy (conversion!)
]
# Result: 2 conversions (numpy→torch, torch→numpy)
```

### Backend Selection Strategy

```python
def optimize_backend_selection(step_plans):
    """Optimize backend selection for performance."""
    
    for i, step_plan in enumerate(step_plans):
        # GPU memory types benefit from memory backend
        if step_plan['input_memory_type'] in GPU_MEMORY_TYPES:
            if i > 0:  # Not first step
                step_plan['read_backend'] = 'memory'
        
        if step_plan['output_memory_type'] in GPU_MEMORY_TYPES:
            if i < len(step_plans) - 1:  # Not last step
                step_plan['write_backend'] = 'memory'
        
        # Large data benefits from streaming
        if step_plan['estimated_data_size'] > LARGE_DATA_THRESHOLD:
            step_plan['streaming_enabled'] = True
```

## Error Handling and Validation

### Cross-System Validation

The integrated system validates compatibility across all layers:

```python
def validate_system_integration(step_plans):
    """Validate integration across VFS, memory types, and compilation."""
    
    for step_plan in step_plans:
        # Validate memory type compatibility
        if step_plan['input_memory_type'] not in SUPPORTED_MEMORY_TYPES:
            raise ValidationError(f"Unsupported input memory type: {step_plan['input_memory_type']}")
        
        # Validate backend compatibility
        if step_plan['read_backend'] not in SUPPORTED_BACKENDS:
            raise ValidationError(f"Unsupported read backend: {step_plan['read_backend']}")
        
        # Validate GPU requirements
        if step_plan['input_memory_type'] in GPU_MEMORY_TYPES:
            if step_plan['gpu_id'] is None:
                raise ValidationError("GPU memory type requires gpu_id")
        
        # Validate special I/O paths
        for special_input in step_plan.get('special_inputs', {}).values():
            if not validate_vfs_path(special_input['path'], special_input['backend']):
                raise ValidationError(f"Invalid special input path: {special_input['path']}")
```

### Error Recovery

```python
def handle_conversion_errors(data, source_type, target_type, allow_fallback=True):
    """Handle memory type conversion errors with fallback strategies."""
    
    try:
        # Attempt direct conversion
        return convert_memory_type(data, source_type, target_type)
    
    except MemoryConversionError as e:
        if allow_fallback:
            # Fallback to CPU roundtrip
            logger.warning(f"Direct conversion failed, using CPU fallback: {e}")
            cpu_data = convert_to_cpu(data, source_type)
            return convert_from_cpu(cpu_data, target_type)
        else:
            raise
    
    except Exception as e:
        # Log detailed error information
        logger.error(f"Conversion failed: {source_type} → {target_type}")
        logger.error(f"Data shape: {getattr(data, 'shape', 'unknown')}")
        logger.error(f"Data type: {type(data)}")
        raise MemoryConversionError(f"Failed to convert {source_type} to {target_type}") from e
```

## Current Implementation Status

### Implemented Features
- ✅ Four-layer architecture stack with Configuration, Compilation, Memory Types, and VFS
- ✅ Comprehensive step plan generation with backend and memory type coordination
- ✅ MaterializationFlagPlanner for intelligent backend selection
- ✅ Memory type extraction and validation during compilation
- ✅ VFS integration with FileManager and multiple storage backends
- ✅ Runtime execution flow with stack/unstack operations and type conversion
- ✅ Special I/O integration using VFS memory backend for cross-step communication
- ✅ Cross-system validation and error handling

### Future Enhancements

1. **Automatic Memory Type Selection**: Based on data size and available resources
2. **Streaming Processing**: Handle datasets larger than memory across all systems
3. **Performance Optimization**: Intelligent backend selection and conversion minimization
4. **Distributed Processing**: Coordinate memory types across multiple nodes
5. **Advanced Error Recovery**: Fallback strategies for conversion failures
6. **Memory Pool Management**: Efficient GPU memory reuse across steps
7. **Resource Prediction**: Predict memory and storage requirements before execution

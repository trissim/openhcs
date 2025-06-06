# Pipeline Compilation System Architecture

## Overview

OpenHCS implements a **declarative, compile-time pipeline system** that treats configuration as a first-class compilation target. This architecture separates pipeline definition from execution, enabling compile-time validation, resource optimization, and reproducible execution.

## Core Philosophy

The system is designed around three fundamental principles:

1. **Declaration Phase**: Functions declare their contracts via decorators
2. **Compilation Phase**: Multi-pass compiler builds execution plans  
3. **Execution Phase**: Stateless execution against immutable contexts

This approach is analogous to a programming language compiler, but for data processing pipelines.

## Multi-Pass Compiler Architecture

The pipeline compiler operates in four sequential phases, each building upon the previous:

### Phase 1: Path Planning (`PipelinePathPlanner`)

**Purpose**: Establishes the data flow topology of the pipeline

- **Input**: Step definitions + ProcessingContext (well_id, input_dir)
- **Output**: Basic `step_plans` with input/output directories and special I/O paths
- **Responsibilities**:
  - Determines input/output directories for each step
  - Creates VFS paths for special I/O (cross-step communication)
  - Links special outputs from one step to special inputs of another
  - Handles chain breaker logic

**Key Error**: `"Context step_plans must be initialized before path planning"`
- Indicates this phase failed to properly initialize the step_plans structure

### Phase 2: Materialization Planning (`MaterializationFlagPlanner`)

**Purpose**: Decides where data lives (VFS backend strategy)

- **Input**: `step_plans` from Phase 1
- **Output**: Backend selection (disk vs memory) for each step
- **Strategy**:
  - First step: Always reads from disk (input images)
  - Last step: Always writes to disk (final outputs)
  - Middle steps: Can use memory backend for speed
  - FunctionSteps: Can use intermediate backends
  - Non-FunctionSteps: Must use persistent backends

### Phase 3: Memory Contract Validation (`FuncStepContractValidator`)

**Purpose**: Ensures memory type compatibility across the pipeline

- **Input**: `step_plans` + function decorators
- **Output**: Memory type validation and injection into step_plans
- **Validation**:
  - All functions must have explicit memory type declarations
  - Functions in the same step must have consistent memory types
  - Memory types must be valid (numpy, cupy, torch, tensorflow, jax)

### Phase 4: GPU Resource Assignment (`GPUMemoryTypeValidator`)

**Purpose**: Resource allocation for GPU-accelerated steps

- **Input**: `step_plans` with memory types
- **Output**: GPU device assignments
- **Logic**:
  - Identifies steps requiring GPU memory types
  - Assigns available GPU devices
  - Validates GPU resource availability

## Function Pattern System

### The Sacred Four Patterns

OpenHCS supports four fundamental function execution patterns that provide unified handling of different processing strategies:

#### 1. Single Function Pattern
```python
FunctionStep(func=my_function)
```
- **Use Case**: Apply same function to all data
- **Execution**: `my_function(image_stack)` for each pattern group

#### 2. Parameterized Function Pattern
```python
FunctionStep(func=(my_function, {'param': value}))
```
- **Use Case**: Apply function with specific parameters
- **Execution**: `my_function(image_stack, param=value)` for each pattern group

#### 3. Sequential Function Chain
```python
FunctionStep(func=[func1, func2, func3])
```
- **Use Case**: Apply multiple functions in sequence
- **Execution**: `func3(func2(func1(image_stack)))` for each pattern group

#### 4. Component-Specific Functions
```python
FunctionStep(func={'channel_1': func_dapi, 'channel_2': func_gfp}, group_by='channel')
```
- **Use Case**: Different processing per component (channel, site, etc.)
- **Execution**: `func_dapi` for channel_1 data, `func_gfp` for channel_2 data

### Pattern Resolution Flow

1. **Pattern Detection**: `microscope_handler.auto_detect_patterns()` finds image files matching well/component criteria
2. **Pattern Grouping**: `prepare_patterns_and_functions()` groups patterns by component and resolves func patterns
3. **Execution**: For each pattern group: load images → stack → process → unstack → save

## Decorator System

### Memory Type Decorators

Functions declare their memory interface using decorators:

```python
@torch(input_type="torch", output_type="torch")
def my_function(image_stack):
    return processed_stack

@numpy  # Shorthand for numpy input/output
def another_function(data):
    return result
```

**Supported Memory Types**: `numpy`, `cupy`, `torch`, `tensorflow`, `jax`

**Benefits**:
- No runtime overhead - pure metadata
- Enables compile-time memory type checking
- Supports automatic memory type conversion planning

### Special I/O Decorators

Functions declare cross-step dependencies:

```python
@special_outputs("positions", "metadata")
def generate_positions(image_stack):
    return processed_stack, positions, metadata

@special_inputs("positions")
def stitch_images(image_stack, positions):
    return stitched_stack
```

**Compiler Behavior**:
- Automatically links outputs to inputs
- Creates VFS paths for intermediate data
- Validates dependency chains at compile time

### Chain Breaker Decorator

```python
@chain_breaker
def independent_function(image_stack):
    return result
```

Forces the next step to read from the pipeline's original input directory rather than the previous step's output.

## Virtual File System (VFS)

### Abstraction Layer

The VFS provides a unified interface for all storage operations:

```python
# Same API regardless of backend
filemanager.save(data, "path/to/data", "memory")
filemanager.save(data, "path/to/data", "disk")
data = filemanager.load("path/to/data", "memory")
```

### Backend Types

- **Memory Backend**: Fast intermediate data (numpy arrays, tensors)
- **Disk Backend**: Persistent data (images, final outputs)
- **Zarr Backend**: Chunked array storage (future)

### Location Transparency

Data can be moved between backends without changing application code. The materialization planner decides optimal storage locations based on:
- Step position in pipeline
- Step type (FunctionStep vs others)
- Resource constraints
- Performance requirements

## ProcessingContext Lifecycle

### 1. Creation
```python
context = ProcessingContext(
    global_config=config,
    well_id="A01",
    filemanager=filemanager
)
```

### 2. Population (Compilation)
```python
# Phase 1: Path planning
PipelinePathPlanner.prepare_pipeline_paths(context, steps)

# Phase 2: Materialization planning  
MaterializationFlagPlanner.prepare_pipeline_flags(context, steps)

# Phase 3: Memory contract validation
memory_types = FuncStepContractValidator.validate_pipeline(steps)
# Inject memory types into context.step_plans

# Phase 4: GPU resource assignment
GPUMemoryTypeValidator.validate_step_plans(context.step_plans)
```

### 3. Freezing
```python
context.freeze()  # Makes context immutable
```

### 4. Execution
```python
for step in steps:
    step.process(context)  # Read-only access to frozen context
```

## Step Plans Structure

Each step gets a comprehensive execution plan:

```python
context.step_plans[step_id] = {
    # Basic metadata
    "step_name": "Z-Stack Flattening",
    "step_type": "FunctionStep", 
    "well_id": "A01",
    
    # I/O configuration
    "input_dir": "/path/to/input",
    "output_dir": "/path/to/output",
    "read_backend": "disk",
    "write_backend": "memory",
    
    # Memory configuration
    "input_memory_type": "numpy",
    "output_memory_type": "torch",
    "gpu_id": 0,
    
    # Special I/O
    "special_inputs": {
        "positions": {"path": "/vfs/positions.pkl", "backend": "memory"}
    },
    "special_outputs": {
        "metadata": {"path": "/vfs/metadata.pkl", "backend": "memory"}
    },
    
    # Flags
    "requires_disk_input": True,
    "requires_disk_output": False,
    "force_disk_output": False,
    "visualize": False
}
```

## Execution Model

### Stateless Steps

After compilation, step objects become pure templates:
- All configuration lives in `context.step_plans[step_id]`
- Same step definition reused across wells with different configs
- Functional programming approach to pipeline execution

### VFS-Based Data Flow

- No direct data passing between steps
- All data flows through VFS paths specified in step_plans
- Location transparency: data can be in memory or on disk
- Automatic serialization/deserialization based on backend

## Benefits of This Architecture

1. **Compile-Time Safety**: Catch errors before expensive execution
2. **Resource Optimization**: Global view enables smart resource allocation  
3. **Reproducibility**: Immutable contexts ensure consistent results
4. **Scalability**: Stateless execution enables easy parallelization
5. **Debuggability**: Can inspect and modify plans before execution
6. **Flexibility**: VFS abstraction allows different storage strategies
7. **Performance**: Memory-aware planning optimizes data movement

## Error Handling

The system is designed to **fail fast** during compilation rather than during execution:

- Missing memory type declarations → Compilation error
- Incompatible memory types → Compilation error  
- Missing special input dependencies → Compilation error
- Invalid step plan structure → Compilation error

This approach prevents expensive pipeline failures after processing has begun.

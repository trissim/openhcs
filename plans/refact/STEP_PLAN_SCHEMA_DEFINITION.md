# Step Plan Schema Definition

## Overview

This document defines the centralized schema for step_plans in OpenHCS. It formalizes the structure that was previously distributed across multiple files and provides a definitive reference for all components that interact with step_plans.

## 1. Step Plan Schema

```python
{
    # Core Identifiers
    "step_id": str,                  # Unique identifier (REQUIRED)
    "step_name": str,                # Human-readable name (REQUIRED)
    "step_type": str,                # Type of step (e.g., 'function', 'assembly') (REQUIRED)
    
    # Primary I/O Configuration
    "input_dir": str,                # Primary input directory path (REQUIRED)
    "output_dir": str,               # Primary output directory path (REQUIRED)
    "read_backend": str,             # Backend for reading ('disk', 'memory', 'zarr') (REQUIRED)
    "write_backend": str,            # Backend for writing ('disk', 'memory', 'zarr') (REQUIRED)
    "force_disk_output": bool,       # Whether to save an additional copy to disk (OPTIONAL, default: False)
    "same_directory": bool,          # Whether input and output directories are the same (COMPUTED)
    
    # Memory Configuration
    "input_memory_type": str,        # Memory type for input ('numpy', 'cupy', etc.) (REQUIRED)
    "output_memory_type": str,       # Memory type for output ('numpy', 'cupy', etc.) (REQUIRED)
    "gpu_id": int,                   # GPU device ID (-1 for CPU) (REQUIRED for GPU memory types)
    
    # Special I/O Configuration (Pipeline Cross-Step Communication)
    "special_inputs": {              # Dictionary of special inputs (OPTIONAL)
        "KEY1": {                    # Each key must match a special_output key from a previous step
            "path": str,             # VFS path for this special input (REQUIRED)
            "backend": str,          # Backend for this special input (REQUIRED)
            "materialize": bool,     # Whether to materialize this input (OPTIONAL, default: False)
            "source_step_id": str,   # ID of the step producing this special input (REQUIRED)
        },
        # Additional special inputs...
    },
    
    "special_outputs": {             # Dictionary of special outputs (OPTIONAL)
        "KEY2": {                    # Each key must match a special_input key in a future step
            "path": str,             # VFS path for this special output (REQUIRED)
            "backend": str,          # Backend for this special output (REQUIRED)
            "materialize": bool,     # Whether to materialize this output (OPTIONAL, default: False)
            "target_step_ids": [str],# IDs of steps consuming this special output (REQUIRED)
        },
        # Additional special outputs...
    },
    
    # Function Step Specific Fields
    "func": Any,                     # Function pattern (REQUIRED for function steps)
    "variable_components": List[str], # Variable components (REQUIRED for function steps)
    "group_by": Any,                 # Group by specification (REQUIRED for function steps)
    "component_values": List[Any],   # Component values to process (OPTIONAL)
    
    # Visualization
    "visualize": bool,               # Whether to visualize step output (OPTIONAL, default: False)
    
    # Execution Order Tracking
    "pipeline_position": int,        # Position in the pipeline (COMPUTED)
}
```

## 2. Special Key Linking Mechanism

Special inputs and outputs create directed connections between steps in the pipeline:

1. A step with `@special_output(KEY)` produces data that is stored in the VFS
2. A later step in the pipeline with `@special_input(KEY)` consumes that data from the VFS
3. The matching key creates a directed connection between these steps
4. The compiler validates that all special inputs have corresponding special outputs from earlier steps

This directed graph of connections allows data to flow between steps outside the primary input/output directories.

## 3. Decorator & Attribute Flow

The information flow for special inputs/outputs is:

1. Decorators (`@special_input`, `@special_output`) are applied to functions
2. Functions are provided to steps during initialization
3. Steps extract attributes from these functions and set them on themselves
4. During compilation, these attributes are inspected to build the step plans
5. After compilation, attributes are stripped from steps

## 4. Step Plan Validation

A centralized validation function shall enforce that:

1. All required fields are present and of the correct type
2. GPU memory types have a valid GPU ID
3. Backend names are valid
4. Memory types are valid
5. Special input/output key validation:
   - Each special input key matches a special output key from an earlier step
   - Each special output key matches a special input key from a later step
   - No orphaned special inputs or outputs
   - No cyclic dependencies

## 5. Implementation Plan

1. Create a new module `openhcs/core/pipeline/step_plan_schema.py` with:
   - Schema definition (as a dictionary or using TypedDict)
   - Validation function
   - Helper function for creating default step plans

2. Update `PipelineCompiler` to use this centralized schema:
   - Extract special input/output attributes from steps
   - Build special_inputs and special_outputs dictionaries
   - Validate key matching between steps
   - Assign consistent VFS paths for matching keys

3. Remove redundant validation from `function_step.py`

4. Update all components that interact with step_plans to reference this schema

## 6. Function Argument Handling

For functions with special inputs:

```python
@special_input(KEY1)
def process_with_input(primary_array, special_input_data=None, **kwargs):
    # process using special input
    # special_input_data is loaded from VFS by FunctionStep
    pass
```

For functions with special outputs:

```python
@special_output(KEY2)
def process_with_output(primary_array, **kwargs):
    # process and create special output
    special_output_data = create_output()
    # Return the special output, which FunctionStep will write to VFS
    return special_output_data
```

## 7. Doctrinal Enforcement

The step plan schema enforces several key doctrines:

- **Clause 245 — Declarative Enforcement**: Schema provides a single source of truth
- **Clause 66 — Immutability After Construction**: Step plans are immutable after compilation
- **Clause 17 — VFS Exclusivity**: All paths are managed through the VFS
- **Clause 500 — File Decomposition**: Schema is centralized in a dedicated file
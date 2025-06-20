# Detailed Rewrite Plan: architecture_overview.rst

## Overview
**Target**: Complete rewrite of `docs/source/concepts/architecture_overview.rst`  
**Goal**: Document the actual OpenHCS architecture based on production codebase  
**Approach**: Replace outdated EZStitcher concepts with current GPU-native, two-phase architecture

## Current Problems (From Fact-Check)
- ❌ **Project name**: "EZStitcher" → should be "OpenHCS"
- ❌ **Specialized step classes**: Documented but deprecated
- ❌ **Single-phase execution**: Documented but replaced by two-phase system
- ❌ **Missing GPU architecture**: Memory type system not documented
- ❌ **Missing VFS**: Multi-backend storage system not documented
- ✅ **Core concepts**: Pipeline → Step hierarchy preserved

## Production Codebase Architecture (Source of Truth)

### Verified Core Components
From `openhcs/core/orchestrator/orchestrator.py`:
- **PipelineOrchestrator**: Two-phase execution (compile → execute)
- **compile_pipelines()**: Creates frozen ProcessingContexts
- **execute_compiled_plate()**: Stateless execution with parallel processing

From `openhcs/core/pipeline/__init__.py`:
- **Pipeline**: Inherits from `list`, behaves as `List[AbstractStep]`
- **Metadata support**: name, description, creation timestamp

From `openhcs/core/steps/function_step.py`:
- **FunctionStep**: Function-based approach with four patterns
- **Memory type integration**: Automatic GPU optimization

From `openhcs/core/memory/decorators.py`:
- **Memory type decorators**: @cupy, @torch, @tensorflow, @jax, @pyclesperanto
- **Thread-local GPU management**: Automatic CUDA stream handling

From `openhcs/io/base.py`:
- **VFS system**: Disk, Memory, Zarr backends
- **Storage registry**: Global singleton with backend switching

## Detailed Section-by-Section Rewrite Plan

### Section 1: Title and Introduction
**Current (Lines 1-15)**:
```rst
Architecture Overview
===================

This document provides an overview of the EZStitcher architecture and its core components.
```

**Rewrite To**:
```rst
Architecture Overview
===================

OpenHCS is a GPU-native, high-content screening platform built on a compositional architecture that separates compilation from execution. This design enables powerful features like automatic GPU optimization, memory type management, and parallel processing while maintaining clean, declarative interfaces.

**Key Architectural Principles**:

- **Two-Phase Execution**: Compile-then-execute model for robust error handling
- **GPU-Native Design**: Memory type system with automatic optimization
- **Function Composition**: Declarative function patterns for complex workflows
- **Storage Abstraction**: Virtual File System supporting multiple backends
- **Stateless Execution**: Immutable contexts enable safe parallel processing

**Core Components**:

- **PipelineOrchestrator**: Coordinates compilation and execution
- **Pipeline**: List-based container for processing steps
- **FunctionStep**: Function pattern system for flexible processing
- **Memory Type System**: Automatic GPU acceleration and conversion
- **VFS**: Multi-backend storage (disk, memory, zarr)
```

### Section 2: Two-Phase Execution Architecture
**Current (Lines 17-45)**: Single-phase execution model

**Rewrite To**:
```rst
Two-Phase Execution Architecture
------------------------------

OpenHCS uses a sophisticated two-phase execution model that separates compilation from execution, enabling robust error handling and parallel processing.

Compilation Phase
~~~~~~~~~~~~~~~~

The compilation phase analyzes the pipeline and creates frozen, immutable execution contexts:

.. code-block:: python

    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.pipeline import Pipeline
    from openhcs.core.steps.function_step import FunctionStep

    # Create orchestrator and pipeline
    orchestrator = PipelineOrchestrator(plate_path=Path("/path/to/plate"))
    orchestrator.initialize()

    pipeline = Pipeline(steps=[
        FunctionStep(func=my_processing_function, name="Process Images")
    ], name="My Pipeline")

    # Compilation phase: Create frozen contexts
    compiled_contexts = orchestrator.compile_pipelines(
        pipeline_definition=pipeline,  # Pipeline IS a list of steps
        well_filter=["A01", "B01"]     # Process specific wells
    )

**Compilation Steps**:

1. **Path Planning**: Resolve file patterns and inject metadata
2. **Memory Contract Validation**: Verify function memory types
3. **GPU Resource Assignment**: Allocate GPU devices to steps
4. **Context Freezing**: Create immutable execution contexts

Execution Phase
~~~~~~~~~~~~~~

The execution phase runs the stateless pipeline against compiled contexts:

.. code-block:: python

    # Execution phase: Run with frozen contexts
    results = orchestrator.execute_compiled_plate(
        pipeline_definition=pipeline,  # Same pipeline, now stateless
        compiled_contexts=compiled_contexts,
        max_workers=4  # Parallel execution
    )

    # Check results
    for well_id, result in results.items():
        if result.get('success'):
            print(f"Well {well_id}: Processing completed")
        else:
            print(f"Well {well_id}: Error - {result.get('error')}")

**Benefits of Two-Phase Architecture**:

- **Early Error Detection**: Compilation catches errors before execution
- **Parallel Safety**: Frozen contexts enable safe concurrent execution
- **Resource Optimization**: GPU allocation planned during compilation
- **Reproducibility**: Immutable contexts ensure consistent results
```

**Context Engine Sources**:
- openhcs/core/orchestrator/orchestrator.py lines 255-281 (compile_pipelines)
- openhcs/core/orchestrator/orchestrator.py lines 403-409 (execute_compiled_plate)
- openhcs/core/pipeline/compiler.py lines 43-52 (compilation phases)

### Section 3: Function Pattern System
**Current (Lines 47-85)**: Class-based step system

**Rewrite To**:
```rst
Function Pattern System
----------------------

OpenHCS uses a powerful function pattern system that enables declarative composition of complex processing workflows.

The Four Function Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Single Function Pattern**

Apply the same function to all data:

.. code-block:: python

    from openhcs.core.memory.decorators import cupy
    from openhcs.processing.backends.processors.cupy_processor import stack_percentile_normalize

    @cupy  # GPU-accelerated
    def normalize_images(image_stack):
        return stack_percentile_normalize(image_stack)

    step = FunctionStep(func=normalize_images, name="Normalize Images")

**2. Parameterized Function Pattern**

Function with custom parameters:

.. code-block:: python

    step = FunctionStep(
        func=(normalize_images, {'percentile': 99}),  # Function + parameters
        name="Custom Normalization"
    )

**3. Sequential Function Pattern**

Chain multiple functions together:

.. code-block:: python

    step = FunctionStep(
        func=[
            denoise_images,                              # First: Denoise
            (enhance_contrast, {'clip_limit': 0.03}),   # Then: Enhance
            normalize_images                             # Finally: Normalize
        ],
        name="Multi-Step Enhancement"
    )

**4. Component-Specific Function Pattern**

Different functions for different components:

.. code-block:: python

    from openhcs.constants.constants import GroupBy

    step = FunctionStep(
        func={
            'DAPI': process_nuclei,      # DAPI-specific processing
            'GFP': process_proteins,     # GFP-specific processing
            'BF': process_brightfield    # Brightfield-specific processing
        },
        group_by=GroupBy.CHANNEL,  # Process by channel
        name="Channel-Specific Processing"
    )

Variable Components and Grouping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Control how data is grouped and processed:

.. code-block:: python

    from openhcs.constants.constants import VariableComponents

    # Process Z-stacks: group by (well, site, channel), vary z_index
    step = FunctionStep(
        func=create_max_projection,
        variable_components=[VariableComponents.Z_INDEX],
        name="Z-Stack Flattening"
    )

    # Process channels: group by (well, site), vary channel
    step = FunctionStep(
        func=create_composite,
        variable_components=[VariableComponents.CHANNEL],
        name="Channel Compositing"
    )

    # Process sites: group by (well), vary site
    step = FunctionStep(
        func=generate_positions,
        variable_components=[VariableComponents.SITE],
        name="Position Generation"
    )
```

**Context Engine Sources**:
- openhcs/core/steps/function_step.py lines 415-434 (FunctionStep constructor)
- docs/architecture/function-patterns.md (all four patterns verified)
- openhcs/constants/constants.py (VariableComponents, GroupBy enums)

### Section 4: GPU-Native Memory Type System
**Current**: Missing entirely

**Add New Section**:
```rst
GPU-Native Memory Type System
----------------------------

OpenHCS features a sophisticated memory type system that provides automatic GPU acceleration with zero-copy conversions between frameworks.

Memory Type Decorators
~~~~~~~~~~~~~~~~~~~~~

Functions declare their memory interface using decorators:

.. code-block:: python

    from openhcs.core.memory.decorators import cupy, torch, tensorflow, jax

    @cupy  # CuPy GPU arrays
    def cupy_processing(image_stack):
        import cupy as cp
        return cp.max(image_stack, axis=0, keepdims=True)

    @torch  # PyTorch GPU tensors
    def torch_processing(image_tensor):
        import torch
        return torch.max(image_tensor, dim=0, keepdim=True)[0]

    @tensorflow  # TensorFlow GPU tensors
    def tf_processing(image_tensor):
        import tensorflow as tf
        return tf.reduce_max(image_tensor, axis=0, keepdims=True)

    @jax  # JAX GPU arrays
    def jax_processing(image_array):
        import jax.numpy as jnp
        return jnp.max(image_array, axis=0, keepdims=True)

Automatic Memory Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~

The system automatically converts between memory types using optimal methods:

.. code-block:: python

    # Mixed memory types in same pipeline
    pipeline = Pipeline(steps=[
        FunctionStep(func=cupy_processing, name="CuPy Step"),      # CuPy GPU
        FunctionStep(func=torch_processing, name="PyTorch Step"),  # PyTorch GPU
        FunctionStep(func=tf_processing, name="TensorFlow Step")   # TensorFlow GPU
    ])

    # System automatically handles conversions:
    # NumPy → CuPy (GPU upload)
    # CuPy → PyTorch (DLPack zero-copy)
    # PyTorch → TensorFlow (DLPack zero-copy)

**Conversion Methods**:

- **DLPack**: Zero-copy GPU-to-GPU conversion (preferred)
- **CUDA Array Interface**: Direct GPU memory sharing
- **CPU Roundtrip**: Fallback when direct conversion unavailable

GPU Resource Management
~~~~~~~~~~~~~~~~~~~~~~

Automatic GPU assignment and thread-local stream management:

.. code-block:: python

    # GPU assignment happens during compilation
    compiled_contexts = orchestrator.compile_pipelines(pipeline)

    # Each step gets assigned optimal GPU device
    # Thread-local CUDA streams for parallel execution
    # Automatic memory cleanup and defragmentation

**Features**:

- **Thread-local streams**: Each thread gets persistent CUDA streams
- **Device affinity**: Steps assigned to optimal GPU devices
- **Memory cleanup**: Automatic GPU memory management
- **Fallback handling**: CPU execution when GPU memory insufficient
```

**Context Engine Sources**:
- openhcs/core/memory/decorators.py lines 121-133 (memory_types decorator)
- openhcs/core/memory/decorators.py lines 234-248 (cupy decorator)
- openhcs/core/memory/conversion_functions.py lines 189-216 (DLPack conversion)
- openhcs/core/pipeline/gpu_memory_validator.py lines 120-165 (GPU assignment)

## Implementation Checklist

### Content Verification (Using Production Codebase)
- [x] **Two-phase execution**: Verified from orchestrator.py implementation
- [x] **Function patterns**: Verified from function_step.py and function-patterns.md
- [x] **Memory decorators**: Verified from memory/decorators.py
- [x] **GPU assignment**: Verified from gpu_memory_validator.py
- [x] **VFS system**: Verified from io/base.py storage backends
- [x] **Pipeline as list**: Verified from pipeline/__init__.py

### Cross-Reference Updates Needed
- [ ] Update: All user guides to reference new architecture
- [ ] Create: `docs/source/concepts/memory_optimization.rst`
- [ ] Create: `docs/source/concepts/gpu_acceleration.rst`

### Estimated Effort
**Writing time**: 16-20 hours  
**Testing time**: 8-10 hours (verify all architectural examples work)  
**Total**: 24-30 hours

### Success Criteria
1. **Accurate architecture**: Reflects actual production implementation
2. **GPU system documented**: Memory type system and automatic optimization
3. **Two-phase execution**: Clear explanation of compile-then-execute model
4. **Function patterns**: All four patterns with working examples
5. **VFS integration**: Multi-backend storage system documented

This plan provides comprehensive documentation of the actual OpenHCS architecture based entirely on production codebase verification, ensuring accuracy and completeness.

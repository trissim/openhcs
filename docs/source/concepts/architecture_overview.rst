====================
Architecture Overview
====================

Pipeline Architecture
--------------------

OpenHCS is built around a modern, GPU-accelerated pipeline architecture designed for high-content screening and large-scale bioimage analysis. The architecture consists of four main layers:

.. note::
   OpenHCS is primarily used through the TUI (Terminal User Interface) which generates production-ready scripts.
   See :doc:`../user_guide/basic_usage` for the complete TUI workflow.

1. **PipelineOrchestrator**: Multi-well execution engine with GPU resource management
2. **FunctionStep**: Processing operations with automatic memory type conversion
3. **Function Registry**: 574+ GPU-accelerated processing functions
4. **VFS System**: Virtual file system with multiple storage backends

Key components:

* :doc:`PipelineOrchestrator <pipeline_orchestrator>` - Multi-well execution engine
* :doc:`FunctionStep <step>` - GPU-accelerated processing operations
* :doc:`Function Handling <function_handling>` - Flexible function patterns
* :doc:`Processing Context <processing_context>` - Execution state management

**Modern Architecture Design**:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────┐
    │                PipelineOrchestrator                     │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
    │  │    Well 1   │  │    Well 2   │  │    Well N   │     │
    │  │             │  │             │  │             │     │
    │  │ FunctionStep│  │ FunctionStep│  │ FunctionStep│     │
    │  │      ↓      │  │      ↓      │  │      ↓      │     │
    │  │ FunctionStep│  │ FunctionStep│  │ FunctionStep│     │
    │  │      ↓      │  │      ↓      │  │      ↓      │     │
    │  │ FunctionStep│  │ FunctionStep│  │ FunctionStep│     │
    │  └─────────────┘  └─────────────┘  └─────────────┘     │
    └─────────────────────────────────────────────────────────┘
                              │
                    ┌─────────────────────┐
                    │    VFS Backends     │
                    │  Memory │ Disk │ZARR│
                    └─────────────────────┘

**Execution Model**:
- **Parallel Wells**: Each well processed independently across worker threads
- **Sequential Steps**: Within each well, steps execute in sequence
- **GPU Coordination**: Automatic GPU resource management and memory optimization
- **VFS Integration**: All I/O operations go through the Virtual File System

Core Components
---------------

**Execution Engine:**

* :doc:`**PipelineOrchestrator** <pipeline_orchestrator>`: Multi-well execution engine with GPU resource management and parallel processing
* :doc:`**FunctionStep** <step>`: GPU-accelerated processing operations with automatic memory type conversion
* :doc:`**Processing Context** <processing_context>`: Execution state management and configuration

**Function System:**

* **Function Registry**: 574+ GPU-accelerated functions across multiple computational backends
* :doc:`**Function Patterns** <function_handling>`: Single functions, chains, and component-specific processing
* **Memory Type System**: Automatic conversion between NumPy, CuPy, PyTorch, JAX, pyclesperanto

**Storage and I/O:**

* **VFS System**: Virtual file system with memory, disk, and ZARR backends
* **FileManager**: Unified file operations with automatic backend selection
* :doc:`**Storage Adapters** <storage_adapter>`: Backend-specific storage implementations

**Configuration and Management:**

* **GlobalPipelineConfig**: System-wide configuration for workers, GPU, and storage
* **Microscope Detection**: Automatic microscope type detection and handling
* **Resource Management**: GPU allocation, memory optimization, and cleanup

**Modern Architecture Principles**:

1. **GPU-First Design**: All processing functions support GPU acceleration
2. **Memory Type Agnostic**: Automatic conversion between computational backends
3. **Parallel Execution**: Multi-well processing with configurable worker threads
4. **VFS Abstraction**: All I/O operations go through the virtual file system
5. **Compilation System**: 4-phase pipeline compilation for optimization and validation

Key Component Relationships
------------------------

The relationship between the main components is hierarchical:

- :doc:`**PipelineOrchestrator** <pipeline_orchestrator>`: Coordinates execution across wells and provides plate-specific services
- :doc:`**Pipeline** <pipeline>`: Contains and manages a sequence of Steps
- :doc:`**Step** <step>`: Performs specific processing operations

Workflow Composition and Modularity
-----------------------------

OpenHCS's architecture is designed around a modular, composable API that allows for flexible workflow creation:

**Component Roles**

- :doc:`**Pipeline** <pipeline>`: Serves as a container for a sequence of steps, managing their execution order and data flow. Pipelines can be composed, reused, and shared across different projects.

- :doc:`**Step** <step>`: Represents a single processing operation with well-defined inputs and outputs. Steps are highly configurable through parameters like `variable_components` and `group_by`, allowing for flexible function handling patterns.

**Step Types**: OpenHCS provides various step types for common tasks:
  - **PositionGenerationStep**: Analyzes images to generate position files describing how tiles fit together
  - **ImageStitchingStep**: Assembles processed images into a single stitched image using position files
  - **ZFlatStep**: Handles Z-stack flattening with pre-configured projection methods
  - **FocusStep**: Performs focus-based Z-stack processing using focus detection algorithms
  - **CompositeStep**: Creates composite images from multiple channels with configurable weights

These step types can be seamlessly mixed in the same pipeline, allowing you to combine image processing, Z-stack handling, channel compositing, position generation, and image assembly in a single workflow.

**Workflow Composition**

This modular design allows you to:

1. **Mix and match processing steps**: Combine regular Steps with specialized PositionGenerationStep and ImageStitchingStep in a single pipeline, creating complete workflows from image processing to stitching.
2. **Create end-to-end workflows**: Build pipelines that take raw microscopy images all the way through processing, position generation, and final stitched image assembly.
3. **Reuse common workflows**: Create standard pipelines for common tasks and reuse them across projects.
4. **Customize processing per channel**: Apply different processing to different channels using function dictionaries.
5. **Handle complex data structures**: Process Z-stacks, multi-channel images, and tiled images with consistent patterns.
6. **Scale from simple to complex**: Start with basic workflows and gradually add complexity as needed.

Typical Processing Flow
--------------------

A typical workflow built from scratch:

.. code-block:: python

    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import ZFlatStep, NormStep, CompositeStep, PositionGenerationStep, ImageStitchingStep

    # Setup orchestrator
    orchestrator = PipelineOrchestrator(plate_path="path/to/plate")

    # Position generation pipeline
    pos_pipe = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            ZFlatStep(),                # Flatten Z-stacks
            NormStep(),                 # Normalize to enhance contrast
            CompositeStep(),            # Create composite from channels
            PositionGenerationStep()    # Generate positions
        ],
        name="Position Generation"
    )

    # Assembly pipeline
    asm_pipe = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            NormStep(),                 # Normalize to enhance contrast
            ImageStitchingStep()        # Stitch images
        ],
        name="Assembly"
    )

    # Run pipelines
    orchestrator.run(pipelines=[pos_pipe, asm_pipe])

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
* :doc:`**Storage Backends** <storage_backends>`: Backend-specific storage implementations

**Configuration and Management:**

* **GlobalPipelineConfig**: System-wide configuration for workers, GPU, and storage
* **Microscope Detection**: Automatic microscope type detection and handling
* **Resource Management**: GPU allocation, memory optimization, and cleanup

**Modern Architecture Principles**:

1. **GPU-First Design**: All processing functions support GPU acceleration
2. **Memory Type Agnostic**: Automatic conversion between computational backends
3. **Parallel Execution**: Multi-well processing with configurable worker threads
4. **VFS Abstraction**: All I/O operations go through the virtual file system
5. **Compilation System**: 5-phase pipeline compilation for optimization and validation

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

**Step Types**: OpenHCS provides flexible mechanisms for common tasks:
   - For image processing, Z-stack flattening, focus detection, and channel compositing, these operations are typically handled using :py:class:`~openhcs.core.steps.function_step.FunctionStep` combined with appropriate processing functions and `variable_components`.
   - Complex workflows like position generation and image stitching are core functionalities, often achieved through specialized processing backends orchestrated within the pipeline.

This modular design allows you to combine diverse operations, creating comprehensive workflows from raw image input to final stitched image assembly.

**Workflow Composition**

This modular design allows you to:

1. **Mix and match processing steps**: Combine regular Steps with specialized PositionGenerationStep and ImageStitchingStep in a single pipeline, creating complete workflows from image processing to stitching.
2. **Create end-to-end workflows**: Build pipelines that take raw microscopy images all the way through processing, position generation, and final stitched image assembly.
3. **Reuse common workflows**: Create standard pipelines for common tasks and reuse them across projects.
4. **Customize processing per channel**: Apply different processing to different channels using function dictionaries.
5. **Handle complex data structures**: Process Z-stacks, multi-channel images, and tiled images with consistent patterns.
6. **Scale from simple to complex**: Start with basic workflows and gradually add complexity as needed.

Typical Processing Flow (Conceptual Example)
------------------------------------------

A typical workflow in OpenHCS involves defining a sequence of processing steps within a Pipeline, and orchestrating their execution across wells. This example demonstrates a conceptual flow.

.. code-block:: python

    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.pipeline.pipeline import Pipeline
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy
    from openhcs.constants.constants import VariableComponents

    # Assume a plate_path is set up and contains raw image data
    plate_path = "/path/to/your/raw/data"

    # 1. Initialize the PipelineOrchestrator for your dataset
    orchestrator = PipelineOrchestrator(
        input_directory=plate_path, # Path to raw microscopy data
        output_directory="/path/to/output", # Where processed data will be saved
        config_path="/path/to/your/config.toml" # Optional: path to a TOML configuration file
    )

    # 2. Define your processing steps using FunctionStep
    # Example: A step to normalize images
    normalize_step = FunctionStep(
        func=stack_percentile_normalize,
        name="ImageNormalization",
        variable_components=[VariableComponents.SITE] # Process each site in parallel
    )

    # Example: A step for basic image assembly (e.g., Z-projection)
    # This uses a function from a backend to combine Z-slices
    z_project_step = FunctionStep(
        func=assemble_stack_cupy, # Assuming this performs a Z-projection
        name="ZStackProjection",
        variable_components=[VariableComponents.WELL] # Process each well's Z-stack
    )

    # 3. Create a Pipeline, adding your defined steps
    my_pipeline = Pipeline(
        steps=[normalize_step, z_project_step],
        name="MyProcessingPipeline"
    )

    # 4. Compile and Run the pipeline(s)
    # The compilation phase integrates step definitions with the ProcessingContext.
    # The orchestrator then executes these compiled pipelines.
    orchestrator.run(pipelines=[my_pipeline])

    print(f"Pipeline '{my_pipeline.name}' execution initiated.")
    print("Check the output directory for results.")

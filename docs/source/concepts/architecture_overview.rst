====================
Architecture Overview
====================

Pipeline Architecture
--------------------

EZStitcher is built around a flexible pipeline architecture that allows you to create custom image processing workflows. The architecture consists of three main components:

.. note::
   The EZ module provides a simplified interface that wraps this architecture.
   See :doc:`../user_guide/basic_usage` for details.

1. **PipelineOrchestrator**: Coordinates the execution of pipelines across wells
2. **Pipeline**: A sequence of processing steps
3. **Step**: A single processing operation

Key components:

* :doc:`PipelineOrchestrator <pipeline_orchestrator>`
* :doc:`Pipeline <pipeline>`
* :doc:`Step <step>`

This hierarchical design allows complex workflows to be built from simple, reusable components:

.. code-block:: text

    ┌─────────────────────────────────────────┐
    │            PipelineOrchestrator         │
    │                                         │
    │  ┌─────────┐    ┌─────────┐             │
    │  │ Pipeline│    │ Pipeline│    ...      │
    │  │         │    │         │             │
    │  │ ┌─────┐ │    │ ┌─────┐ │             │
    │  │ │Step │ │    │ │Step │ │             │
    │  │ └─────┘ │    │ └─────┘ │             │
    │  │ ┌─────┐ │    │ ┌─────┐ │             │
    │  │ │Step │ │    │ │Step │ │             │
    │  │ └─────┘ │    │ └─────┘ │             │
    │  │   ...   │    │   ...   │             │
    │  └─────────┘    └─────────┘             │
    └─────────────────────────────────────────┘

When you run a pipeline, data flows through the steps in sequence. Each step processes the images and passes the results to the next step through a shared context object.

Core Components
--------------

**Pipeline Management:**

* :doc:`**PipelineOrchestrator** <pipeline_orchestrator>`: Coordinates the entire workflow and manages plate-specific operations
* :doc:`**Pipeline** <pipeline>`: A sequence of processing steps that are executed in order
* **ProcessingContext**: Maintains state during pipeline execution

**Pipeline Factories:**

* :doc:`Pipeline factories <pipeline_factory>` provide a convenient way to create common pipeline configurations

**Step Components:**

* :doc:`**Step** <step>`: A single processing operation that can be applied to images
* **Pre-defined Steps**: Provides optimized implementations for common operations (ZFlatStep, CompositeStep, etc.)

**Image Processing:**

* **ImageProcessor**: Provides static image processing functions
* **FocusAnalyzer**: Provides static focus detection methods for Z-stacks
* **Stitcher**: Performs image stitching

**Infrastructure:**

* **MicroscopeHandler**: Handles microscope-specific functionality
* **FileSystemManager**: Handles file system operations and image loading
* **Config**: Manages configuration settings for various components

These components work together to process microscopy images in a flexible and extensible way. The organization follows the typical workflow:

1. Pipeline setup and management
2. Step definition and execution
3. Image processing operations
4. Supporting infrastructure

Key Component Relationships
------------------------

The relationship between the main components is hierarchical:

- :doc:`**PipelineOrchestrator** <pipeline_orchestrator>`: Coordinates execution across wells and provides plate-specific services
- :doc:`**Pipeline** <pipeline>`: Contains and manages a sequence of Steps
- :doc:`**Step** <step>`: Performs specific processing operations

Workflow Composition and Modularity
-----------------------------

EZStitcher's architecture is designed around a modular, composable API that allows for flexible workflow creation:

**Component Roles**

- :doc:`**Pipeline** <pipeline>`: Serves as a container for a sequence of steps, managing their execution order and data flow. Pipelines can be composed, reused, and shared across different projects.

- :doc:`**Step** <step>`: Represents a single processing operation with well-defined inputs and outputs. Steps are highly configurable through parameters like `variable_components` and `group_by`, allowing for flexible function handling patterns.

**Step Types**: EZStitcher provides various step types for common tasks:
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

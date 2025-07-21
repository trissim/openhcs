.. _pipeline-orchestrator:

===================
PipelineOrchestrator
===================

Role and Responsibilities
------------------------

The PipelineOrchestrator is the central execution engine of OpenHCS that coordinates bioimage analysis workflows across entire microscopy plates.
For an overview of the complete architecture, see :doc:`architecture_overview`.

The ``PipelineOrchestrator`` manages the complete lifecycle of pipeline execution: from compilation and validation to multi-well parallel processing. It abstracts plate-specific details and provides a unified interface for executing complex bioimage analysis workflows.

**Real-World Usage** (from TUI-generated scripts):

.. code-block:: python

   from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
   from openhcs.core.config import GlobalPipelineConfig

   # Create orchestrator for a microscopy plate
   orchestrator = PipelineOrchestrator(plate_path, global_config=global_config)

   # Three-phase execution workflow:

   # 1. Initialize orchestrator
   orchestrator.initialize()

   # 2. Compile pipelines (validation, optimization, GPU assignment)
   compiled_contexts = orchestrator.compile_pipelines(steps)

   # 3. Execute compiled pipelines with parallel processing
   results = orchestrator.execute_compiled_plate(
       pipeline_definition=steps,
       compiled_contexts=compiled_contexts,
       max_workers=global_config.num_workers
   )

.. figure:: ../_static/orchestrator_pipeline_relationship.png
   :alt: Orchestrator and Pipeline Relationship
   :width: 80%
   :align: center

   The relationship between PipelineOrchestrator and Pipeline components.

Key responsibilities:

* **Pipeline Compilation**:
  - 5-phase compilation system: step plan initialization → ZARR store declaration → materialization → memory validation → GPU assignment
  - Memory type validation and GPU resource allocation
  - VFS backend selection and optimization

* **Multi-Well Execution**:
  - Parallel processing across wells with configurable worker threads
  - GPU resource coordination and memory management
  - Error handling and process cleanup

* **Configuration Management**:
  - Global pipeline configuration (workers, GPU settings, VFS backends)
  - ZARR storage configuration for large datasets
  - Microscope-specific handling and pattern detection

**Complete Configuration Example** (from TUI-generated scripts):

.. code-block:: python

   from openhcs.core.config import (
       GlobalPipelineConfig, PathPlanningConfig, VFSConfig, ZarrConfig,
       MaterializationBackend, ZarrCompressor, ZarrChunkStrategy
   )
   from openhcs.constants.constants import Backend, Microscope

   global_config = GlobalPipelineConfig(
       num_workers=5,
       path_planning=PathPlanningConfig(
           output_dir_suffix="_stitched",
           global_output_folder="/path/to/outputs/",
           materialization_results_path="results"
       ),
       vfs=VFSConfig(
           intermediate_backend=Backend.MEMORY,
           materialization_backend=MaterializationBackend.ZARR
       ),
       zarr=ZarrConfig(
           store_name="images.zarr",
           compressor=ZarrCompressor.ZSTD,
           compression_level=1,
           shuffle=True,
           chunk_strategy=ZarrChunkStrategy.SINGLE,
           ome_zarr_metadata=True,
           write_plate_metadata=True
       ),
       microscope=Microscope.AUTO,
       use_threading=None
   )

The orchestrator abstracts the complexity of multi-well parallel processing, GPU resource management, and VFS backend coordination, allowing pipeline steps to focus purely on image processing logic.

.. note::
   While :doc:`pipeline` defines *what* processing to perform, the orchestrator controls *how* and *where* that processing is applied across a plate with automatic optimization.

Orchestrator Lifecycle
-----------------------

The PipelineOrchestrator follows a strict three-phase lifecycle that ensures proper resource management and error handling:

**Phase 1: Initialization**
  - Microscope detection and handler setup
  - VFS backend configuration
  - GPU resource discovery and allocation

**Phase 2: Compilation**
  - Pipeline validation and optimization
  - Memory type contract verification
  - Path planning and materialization setup
  - GPU assignment and resource scheduling

**Phase 3: Execution**
  - Multi-well parallel processing
  - Real-time GPU memory management
  - Error handling and cleanup
  - Results aggregation and storage

Plate-Specific Services
----------------------

The orchestrator provides several plate-specific services that abstract away the details of different plate formats:

1. **Workspace and Original Data Protection**:

   The orchestrator creates a workspace to protect original data:

   .. code-block:: python

       # Create an orchestrator with a plate path
       orchestrator = PipelineOrchestrator(
           config=config,
           plate_path="path/to/plate"  # Original plate path
       )

       # Access the workspace path (contains symlinks to original images)
       workspace_path = orchestrator.workspace_path

   **Used by**: Pipelines and steps use this workspace path as their input directory, ensuring that original data is protected from modification.

2. **Microscope Handler**: Understands the specific plate format and how to parse filenames

   .. code-block:: python

       # The microscope handler knows how to interpret filenames for the specific plate type
       microscope_handler = orchestrator.microscope_handler

       # Parse a filename to extract components (channel, z-index, site, etc.)
       components = microscope_handler.parser.parse_filename("image_c1_z3_s2.tif")

       # Generate patterns for finding images
       patterns = microscope_handler.auto_detect_patterns(input_dir)

   **Used by**: The `get_stitcher()` method uses the microscope handler's parser to configure the stitcher. The `stitch_images()` and `generate_positions()` methods use it to understand the plate format and parse filenames.

3. **Position Generation**: Generates position files for stitching

   .. code-block:: python

       # Generate positions for a specific well
       positions_file, _ = orchestrator.generate_positions(
           well="A01",
           input_dir=input_dir,
           positions_dir=positions_dir
       )

   **Used by**: The `PositionGenerationStep` calls this method to generate position files for stitching. Internally, this method uses the microscope handler and a stitcher instance obtained via `get_stitcher()`.

4. **Image Stitching**: Stitches images using position files

   .. code-block:: python

       # Stitch images for a specific well
       orchestrator.stitch_images(
           well="A01",
           input_dir=input_dir,
           output_dir=output_dir,
           positions_file=positions_file
       )

   **Used by**: The `ImageStitchingStep` calls this method to stitch images. Internally, this method uses the microscope handler and a stitcher instance obtained via `get_stitcher()`.

5. **Thread-Safe Stitcher Creation**:

   The `get_stitcher()` method creates a new `Stitcher` instance configured for the plate:

   .. code-block:: python

       # Get a thread-safe stitcher instance
       stitcher = orchestrator.get_stitcher()

   **Used by**: The `stitch_images()` and `generate_positions()` methods call this internally to get a thread-safe stitcher instance. Steps don't need to call this directly.

.. _orchestrator-running-pipelines:

Running Pipelines
----------------

For detailed API documentation, see :doc:`../api/pipeline_orchestrator`.

The orchestrator can run one or more pipelines:

.. code-block:: python

    # Run a single pipeline
    orchestrator.run(pipelines=[pipeline])

    # Run multiple pipelines in sequence
    orchestrator.run(pipelines=[pipeline1, pipeline2, pipeline3])

When multiple pipelines are provided, they are executed in sequence for each well. If ``num_workers`` is greater than 1, multiple wells are processed in parallel.

.. figure:: ../_static/pipeline_execution_flow.png
   :alt: Pipeline Execution Flow
   :width: 80%
   :align: center

   Pipeline execution flow with multiple wells and pipelines.

The execution flow is:

1. For each well in the plate (or well_filter if specified):
   a. Create a thread to process the well (if multithreading is enabled)
   b. For each pipeline in the pipelines list:
      i. Run the pipeline on the well
      ii. Wait for completion before starting the next pipeline

This approach ensures that:
- Multiple wells can be processed concurrently (controlled by ``num_workers``)
- Pipelines are executed in sequence for each well
- Each pipeline has access to the results of previous pipelines

For more information on how pipelines are executed, see :ref:`pipeline-running`.

.. _orchestrator-pipeline-relationship:

Orchestrator-Pipeline Relationship
-------------------------------

The relationship between the PipelineOrchestrator and Pipeline is a key aspect of EZStitcher's architecture:

.. figure:: ../_static/orchestrator_pipeline_steps.png
   :alt: Orchestrator, Pipeline, and Steps Relationship
   :width: 80%
   :align: center

   The hierarchical relationship between Orchestrator, Pipeline, and Steps.

**Responsibilities:**

* **PipelineOrchestrator**: Manages plate-level operations and multithreaded execution
* **Pipeline**: Manages a sequence of processing steps and their execution
* **Step**: Performs a specific processing operation on images

**Communication Flow:**

1. The orchestrator provides plate-specific services to pipelines
2. Pipelines use these services to execute their steps
3. Steps access the orchestrator through the pipeline's context

**Key Interactions:**

* The orchestrator creates a ProcessingContext for each pipeline
* The context includes a reference to the orchestrator
* Steps can access the orchestrator through this context reference
* Specialized steps (like PositionGenerationStep) use orchestrator services

For more information on pipelines and their structure, see :ref:`pipeline-concept`.

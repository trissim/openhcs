.. _pipeline-orchestrator:

===================
PipelineOrchestrator
===================

Role and Responsibilities
------------------------

The PipelineOrchestrator is a key component of the EZStitcher architecture.
For an overview of the complete architecture, see :doc:`architecture_overview`.

The ``PipelineOrchestrator`` is the top-level component that manages all plate-specific operations and coordinates the execution of pipelines. It serves as an abstraction layer between the plate-specific details and the pipeline steps.

.. figure:: ../_static/orchestrator_pipeline_relationship.png
   :alt: Orchestrator and Pipeline Relationship
   :width: 80%
   :align: center

   The relationship between PipelineOrchestrator and Pipeline components.

Key responsibilities:

* **Plate Management**:
  - Plate and well detection
  - Microscope handler initialization (specific to each plate type)
  - Image locator configuration

* **Workspace Initialization**:
  - Creates a workspace by mirroring the plate folder path structure
  - Creates symlinks to the original images in this workspace
  - Ensures that modifications happen on workspace copies, not original data
  - Provides this workspace as the input for pipelines

For detailed information about directory structure, see :doc:`directory_structure`.

* **Pipeline Execution**:
  - Multithreaded execution across wells
  - Error handling and logging

* **Specialized Services**:
  - Provides configured `Stitcher` instances suitable for the plate
  - Manages position generation specific to the plate format
  - Abstracts plate-specific operations that depend on the microscope handler

The orchestrator acts as a "plate manager" that knows how to handle the specific details of different plate formats, allowing the pipeline steps to focus on their image processing tasks without needing to know about the underlying plate structure.

.. note::
   While a :ref:`pipeline <pipeline-concept>` defines *what* processing to perform, the orchestrator controls *how* and *where* that processing is applied across a plate.

Creating an Orchestrator
-----------------------

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration
    config = PipelineConfig(
        num_workers=2  # Use 2 worker threads
    )

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path="path/to/plate"
    )

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

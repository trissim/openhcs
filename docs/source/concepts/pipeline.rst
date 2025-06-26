.. _pipeline-concept:

=======
Pipeline
=======

.. _pipeline-overview:

Overview
-------

The Pipeline is a key component of the EZStitcher architecture.
For an overview of the complete architecture, see :doc:`architecture_overview`.

A ``Pipeline`` is a sequence of processing steps that are executed in order. It provides:

* Step management (adding, removing, reordering)
* Context passing between steps
* Input/output directory management
* Automatic directory resolution between steps

.. _pipeline-creation:

Creating a Pipeline
-----------------

The recommended way to create a pipeline is to provide all steps at once during initialization:

.. code-block:: python

    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create a pipeline with all steps at once (recommended approach)
    pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,    # Pipeline input directory
        output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched", # Pipeline output directory
        steps=[
            Step(
                func=(IP.create_projection, {'method': 'max_projection'}),
                variable_components=['z_index'],
                input_dir=orchestrator.workspace_path
            ),

            Step(
                func=IP.stack_percentile_normalize
            ),

            PositionGenerationStep()
        ],
        name="My Processing Pipeline"
    )

Alternatively, you can add steps one by one using the ``add_step()`` method:

.. code-block:: python

    # Create an empty pipeline
    pipeline = Pipeline(name="My Processing Pipeline")

    # Add steps one by one
    pipeline.add_step(Step(name="Z-Stack Flattening",
                          func=(IP.create_projection, {'method': 'max_projection'}),
                          variable_components=['z_index'],
                          input_dir=orchestrator.workspace_path))

    pipeline.add_step(Step(name="Image Enhancement",
                          func=IP.stack_percentile_normalize))

    pipeline.add_step(PositionGenerationStep(name="Generate Positions"))

The first approach (providing all steps at once) is recommended for most use cases as it's more concise and easier to understand. The second approach (adding steps one by one) is useful for dynamic scenarios where steps need to be added conditionally or configured based on the output of previous steps.

.. _pipeline-parameters:

Pipeline Parameters
----------------

For detailed API documentation, see :doc:`../api/pipeline`.

A ``Pipeline`` accepts the following parameters:

* **name**: A human-readable name for the pipeline (optional but recommended for logging)
* **steps**: A list of Step objects to execute in sequence
* **input_dir**: The directory containing input images (typically ``orchestrator.workspace_path``)
* **output_dir**: The directory where final output will be saved
* **well_filter**: List of wells to process (optional, can be overridden by the orchestrator)

Each parameter plays an important role:

* **name** helps identify the pipeline in logs and debugging output
* **steps** defines the sequence of operations to perform
* **input_dir** establishes the initial input directory for the pipeline
* **output_dir** establishes the final output directory, typically used by the last step in the pipeline
* **well_filter** allows for selective processing of specific wells

.. _pipeline-running:

Running a Pipeline
----------------

A pipeline can be run directly, but it's typically run through the orchestrator:

.. code-block:: python

    # Run through the orchestrator (recommended)
    success = orchestrator.run(pipelines=[pipeline])

    if success:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline failed. Check logs for details.")

    # Run directly (advanced usage)
    results = pipeline.run(
        input_dir="path/to/input",
        output_dir="path/to/output",
        well_filter=["A01", "B02"],
        orchestrator=orchestrator  # Required for microscope handler access
    )

Running through the orchestrator is recommended because it:

1. Handles multithreaded execution across wells
2. Provides plate-specific services to the pipeline
3. Manages error handling and logging
4. Ensures proper directory resolution

For detailed information on how the orchestrator runs pipelines, see :ref:`orchestrator-running-pipelines`.

.. _pipeline-context:

Pipeline Context
--------------

When a pipeline runs, it creates a ``ProcessingContext`` that is passed from step to step. This context holds:

* Input/output directories
* Well filter
* Configuration
* Results from previous steps
* Reference to the orchestrator

This allows steps to communicate and build on each other's results. The context is created at the beginning of pipeline execution and updated by each step as it runs.

.. figure:: ../_static/pipeline_context_flow.png
   :alt: Pipeline Context Flow
   :width: 80%
   :align: center

   The flow of context between steps in a pipeline.

The context serves as a communication mechanism between:

1. The orchestrator and the pipeline
2. The pipeline and its steps
3. Different steps within the pipeline

For example, steps like ``PositionGenerationStep`` use the orchestrator reference in the context to access plate-specific services. For more information on the relationship between the orchestrator and pipeline, see :ref:`orchestrator-pipeline-relationship`.

.. _pipeline-multithreaded:

Multithreaded Processing
---------------------

Pipelines can be run in a multithreaded environment through the orchestrator:

.. code-block:: python

    # Create configuration with custom directory suffixes
    config = PipelineConfig(
        out_dir_suffix="_output",           # For regular steps
        positions_dir_suffix="_pos",        # For position generation
        stitched_dir_suffix="_stitched"     # For stitching
    )

    # Create orchestrator with multithreading
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path=plate_path
    )

    # Run the pipeline with multithreading
    # Each well will be processed in a separate thread
    orchestrator.run(pipelines=[pipeline])

The number of worker threads determines how many wells can be processed concurrently. This can significantly improve performance when processing multiple wells.

.. important::
   Multithreading happens at the well level, not the step level. Each well is processed in a separate thread, but steps within a pipeline are executed sequentially for each well.

Key points about multithreaded processing:

1. The orchestrator creates a thread pool with ``num_workers`` threads
2. Each well is assigned to a thread from the pool
3. All pipelines for a well are executed in the same thread
4. Steps within a pipeline are executed sequentially

This approach provides good performance while avoiding race conditions and ensuring that steps have access to the results of previous steps.

For more information on how the orchestrator manages multithreaded execution, see :ref:`orchestrator-running-pipelines`.

.. _pipeline-directory-resolution:

Directory Resolution
------------------

EZStitcher automatically resolves directories for steps in a pipeline, minimizing the need for manual directory management.

Pipelines manage input and output directories for steps.
For detailed information about directory structure, see :doc:`directory_structure`.

.. _pipeline-saving-loading:

Saving and Loading Pipelines
-------------------------

While EZStitcher doesn't have built-in functions for saving and loading pipelines, you can easily save your pipeline configurations as Python scripts:

.. code-block:: python

    # save_pipeline.py
    def create_basic_pipeline(plate_path, num_workers=1):
        """Create a basic processing pipeline."""
        # Create configuration
        config = PipelineConfig(
            num_workers=num_workers
        )

        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            config=config,
            plate_path=plate_path
        )

        # Create pipeline
        pipeline = Pipeline(
            input_dir=orchestrator.workspace_path,
            output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched",
            steps=[
                # Pipeline steps...
            ],
            name="Basic Processing Pipeline"
        )

        return orchestrator, pipeline

This approach allows you to:
* Parameterize your pipelines
* Reuse pipeline configurations across projects
* Version control your pipeline configurations

.. _pipeline-best-practices:

Best Practices
------------

For comprehensive best practices on using pipelines effectively, see :ref:`best-practices-pipeline` in the :doc:`../user_guide/best_practices` guide.

.. _pipeline-factory-integration:

Pipeline Factory Integration
-------------------------

While you can create pipelines manually as shown in this document, EZStitcher also provides the :doc:`pipeline_factory` for creating pre-configured pipelines for common workflows:

.. code-block:: python

    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    # Create orchestrator
    orchestrator = PipelineOrchestrator(plate_path=plate_path)

    # Create a factory with default settings
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True  # Apply normalization (default)
    )

    # Create the pipelines
    pipelines = factory.create_pipelines()

    # Run the pipelines
    orchestrator.run(pipelines=pipelines)

The ``AutoPipelineFactory`` creates two pipelines:

1. **Position Generation Pipeline**: Creates position files for stitching
2. **Image Assembly Pipeline**: Stitches images using the position files

For more information on pipeline factories, see :doc:`pipeline_factory`.

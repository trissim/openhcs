.. _pipeline-concept:

========
Pipeline
========

.. _pipeline-overview:

Overview
--------

A Pipeline in OpenHCS is a sequence of processing steps that are executed in order on microscopy data. Pipelines provide the foundation for creating reproducible bioimage analysis workflows.

For detailed technical information about how pipelines work internally, see :doc:`../architecture/pipeline_compilation_system`.

A ``Pipeline`` provides:

* Sequential step execution with automatic data flow
* Flexible function pattern support (single functions, chains, channel-specific processing)
* Automatic memory type conversion between NumPy, CuPy, PyTorch, etc.
* GPU resource management and optimization
* Integration with OpenHCS's compilation system

.. _pipeline-creation:

Creating a Pipeline
-------------------

In OpenHCS, pipelines are created as lists of ``FunctionStep`` objects and executed by the ``PipelineOrchestrator``:

**Real Pipeline Example** (from TUI-generated scripts):

.. code-block:: python

    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.core.config import GlobalPipelineConfig
    from openhcs.constants.constants import VariableComponents
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.processing.backends.processors.cupy_processor import tophat, create_composite
    from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy

    # Define pipeline steps
    steps = []

    # Step 1: Preprocessing chain
    steps.append(FunctionStep(
        func=[
            (stack_percentile_normalize, {
                'low_percentile': 1.0,
                'high_percentile': 99.0,
                'target_max': 65535.0
            }),
            (tophat, {
                'selem_radius': 50,
                'downsample_factor': 4
            })
        ],
        name="preprocess",
        variable_components=[VariableComponents.SITE],
        force_disk_output=False
    ))

    # Step 2: Create composite
    steps.append(FunctionStep(
        func=[(create_composite, {})],
        name="composite",
        variable_components=[VariableComponents.CHANNEL],
        force_disk_output=False
    ))

    # Step 3: Assembly
    steps.append(FunctionStep(
        func=[(assemble_stack_cupy, {
            'blend_method': 'fixed',
            'fixed_margin_ratio': 0.1
        })],
        name="assemble",
        variable_components=[VariableComponents.SITE],
        force_disk_output=False
    ))

**Pipeline Execution** (three-phase workflow):

.. code-block:: python

    # Create orchestrator for the plate
    orchestrator = PipelineOrchestrator(plate_path, global_config=global_config)

    # Phase 1: Initialize
    orchestrator.initialize()

    # Phase 2: Compile pipeline (validation, optimization, GPU assignment)
    compiled_contexts = orchestrator.compile_pipelines(steps)

    # Phase 3: Execute with parallel processing
    results = orchestrator.execute_compiled_plate(
        pipeline_definition=steps,
        compiled_contexts=compiled_contexts,
        max_workers=global_config.num_workers
    )

**Pipeline Structure**:

- **Steps are executed sequentially** within each well
- **Wells are processed in parallel** across multiple worker threads
- **Each step processes all images** before moving to the next step
- **Memory backend** is used for intermediate results (unless force_disk_output=True)
- **GPU resources** are automatically managed and assigned during compilation

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

See Also
--------

**Next Steps**:

- :doc:`step` - Understanding individual pipeline steps
- :doc:`function_handling` - Function patterns and processing options
- :doc:`pipeline_orchestrator` - Running pipelines with the orchestrator

**Technical Details**:

- :doc:`../architecture/pipeline_compilation_system` - How pipelines are compiled and optimized
- :doc:`../architecture/function_pattern_system` - Advanced function pattern system
- :doc:`../architecture/memory_type_system` - Memory type handling in pipelines

**API Reference**:

- :doc:`../api/orchestrator` - PipelineOrchestrator API documentation
- :doc:`../api/function_step` - FunctionStep API documentation
- :doc:`../api/config` - Pipeline configuration options

**Practical Guides**:

- :doc:`../guides/pipeline_compilation_workflow` - Complete pipeline creation workflow
- :doc:`../user_guide/intermediate_usage` - Building custom pipelines
- :doc:`../user_guide/best_practices` - Pipeline development best practices

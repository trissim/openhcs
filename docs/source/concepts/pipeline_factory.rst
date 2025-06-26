.. _pipeline-factory-concept:

===============
Pipeline Factory
===============

.. _pipeline-factory-overview:

Overview
--------

Pipeline factories are used internally by the EZ module to create pipelines with sensible defaults.
They are not typically used directly by end users, who should prefer the EZ module or custom pipelines.

The ``AutoPipelineFactory`` is a unified factory class that creates pre-configured pipelines for all common stitching workflows. It simplifies pipeline creation by automatically configuring the appropriate steps based on input parameters, with no need to differentiate between different types of pipelines.

.. note::
   The EZ module provides a simplified interface that wraps the `AutoPipelineFactory`. 
   For details, see :doc:`../user_guide/basic_usage`.

.. code-block:: python

    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    # Create a factory with custom configuration
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True,
        flatten_z=True,  # Flatten Z-stacks in the assembly pipeline
        z_method="max",  # Use maximum intensity projection
        channel_weights=[0.7, 0.3, 0]  # Use only first two channels for reference image
    )

    # Create the pipelines
    pipelines = factory.create_pipelines()

    # Run the pipelines
    orchestrator.run(pipelines=pipelines)

The ``AutoPipelineFactory`` handles all types of stitching workflows with a single implementation:

- 2D single-channel stitching
- 2D multi-channel stitching
- Z-stack per plane stitching
- Z-stack projection stitching

This unified approach simplifies the API and makes it easier to create pipelines for common use cases.

.. _pipeline-factory-structure:

Pipeline Structure
-----------------

The ``AutoPipelineFactory`` creates two pipelines with a consistent structure:

1. **Position Generation Pipeline**
   
   Creates position files for stitching

   * **Steps:**
     * Flatten Z (if ``flatten_z=True``)
     * Normalize (if ``normalize=True``)
     * Create composite (always)
     * Generate positions (always)
   
   * **Purpose:** Process images and generate position files for stitching

2. **Image Assembly Pipeline**
   
   Stitches images using the position files

   * **Steps:**
     * Normalize (if ``normalize=True``)
     * Flatten Z (if ``flatten_z=True``)
     * Stitch images (always)
   
   * **Purpose:** Process and stitch images using the position files

This structure is consistent regardless of data type (single/multi-channel, single/multi-Z), with parameters controlling step behavior rather than pipeline structure.

.. _pipeline-factory-parameters:

Parameters
---------

For detailed API documentation, see :doc:`../api/pipeline_factory`.

The ``AutoPipelineFactory`` accepts the following parameters:

- ``input_dir``: Input directory containing images
- ``output_dir``: Output directory for stitched images (optional)
- ``normalize``: Whether to include normalization (default: True)
- ``normalization_params``: Parameters for normalization (optional)
- ``well_filter``: Wells to process (optional)
- ``flatten_z``: Whether to flatten Z-stacks in the assembly pipeline (default: False)
- ``z_method``: Z-stack processing method (default: "max")
  - Projection methods: "max", "mean", "median", etc.
  - Focus detection methods: "combined", "laplacian", "tenengrad", "normalized_variance", "fft"
- ``channel_weights``: Weights for channel compositing in the reference image (optional)

Important behaviors to note:

- Z-stacks are always flattened for position generation regardless of the ``flatten_z`` setting
- Channel compositing is always performed for position generation
- If ``channel_weights`` is None, weights are distributed evenly across all channels

.. _pipeline-factory-steps:

Step Types
---------

The ``AutoPipelineFactory`` uses various step types from the steps module:

- ``ZFlatStep``: For Z-stack flattening using projection methods (used in both pipelines when appropriate)
- ``FocusStep``: For Z-stack processing using focus detection methods (used when z_method is a focus method)
- ``CompositeStep``: For channel compositing (always used in position generation)
- ``PositionGenerationStep``: For generating position files
- ``ImageStitchingStep``: For stitching images

These steps simplify the pipeline creation process by encapsulating common operations with appropriate defaults.

.. _pipeline-factory-examples:

Examples
-------

Basic Single-Channel Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True
    )
    pipelines = factory.create_pipelines()

Multi-Channel Pipeline with Custom Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        channel_weights=[0.7, 0.3, 0]  # Use only first two channels for reference image
    )
    pipelines = factory.create_pipelines()

Z-Stack Pipeline with Projection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        flatten_z=True,  # Flatten Z-stacks in the assembly pipeline
        z_method="max"   # Use maximum intensity projection
    )
    pipelines = factory.create_pipelines()

Z-Stack Pipeline with Focus Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        flatten_z=True,  # Flatten Z-stacks in the assembly pipeline
        z_method="combined"   # Use combined focus metric
    )
    pipelines = factory.create_pipelines()

Pipeline with Custom Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True,
        normalization_params={'low_percentile': 0.5, 'high_percentile': 99.5}
    )
    pipelines = factory.create_pipelines()

.. _pipeline-factory-vs-custom:

Factory Pipelines vs. Custom Pipelines
-----------------------------------

EZStitcher offers two main approaches for creating stitching pipelines:

1. **Using AutoPipelineFactory**: For standard workflows with configurable parameters
2. **Building custom pipelines**: For maximum flexibility and control

While both approaches are valid, they serve different purposes and should be used in different scenarios:

**When to Use AutoPipelineFactory:**
- For standard stitching workflows without custom processing steps
- When the built-in parameters (normalize, flatten_z, z_method, etc.) are sufficient
- For quick prototyping and getting started quickly
- When you want to leverage pre-configured, optimized pipelines

**When to Create Custom Pipelines:**
- When you need custom processing steps beyond what AutoPipelineFactory provides
- When you need precise control over pipeline structure
- When you need to implement specialized workflows
- When you want maximum readability and maintainability for complex pipelines

.. important::
   While it is technically possible to modify pipelines created by AutoPipelineFactory after creation,
   this approach is generally not recommended. Creating custom pipelines from scratch is usually more
   readable, maintainable, and less error-prone for any workflow that requires customization beyond
   what AutoPipelineFactory parameters provide.

For custom workflows, create pipelines from scratch instead of modifying factory pipelines:

.. code-block:: python

    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, ZFlatStep, CompositeStep, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create a custom pipeline with steps
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Normalize images
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 2: Custom processing (example)
            Step(
                name="Custom Enhancement",
                func=custom_enhance
            ),

            # Step 3: Create composite for position generation
            CompositeStep(weights=[0.7, 0.3, 0]),

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Custom Position Generation Pipeline"
    )

    # Create assembly pipeline
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Normalize images
            NormStep(),

            # Step 2: Stitch images
            ImageStitchingStep()
        ],
        name="Custom Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

This approach provides several benefits:

1. **Readability**: The pipeline structure is explicit and easy to understand
2. **Maintainability**: Changes can be made directly to the pipeline definition
3. **Flexibility**: Complete control over each step and its parameters
4. **Robustness**: No risk of unexpected behavior from modifying factory pipelines

.. seealso::
   - :doc:`pipeline` for more information about pipelines
   - :doc:`step` for more information about steps
   - :doc:`../user_guide/basic_usage` for beginner examples
   - :doc:`../user_guide/intermediate_usage` for intermediate examples
   - :doc:`../development/extending` for information about extending pipeline factories

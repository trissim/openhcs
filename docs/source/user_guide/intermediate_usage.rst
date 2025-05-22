=================
Intermediate Usage
=================

This section shows how to reimplement the EZ module functionality using pipelines and steps, providing a bridge between the simplified EZ module and the advanced usage of EZStitcher.

**What You'll Learn:**

1. How the EZ module works under the hood
2. How to create custom pipelines with steps
3. How to reimplement EZ module functionality with more control

**Learning Path:**

1. If you are new to EZStitcher, start with the :doc:`basic_usage` guide (beginner level)
2. After completing this intermediate guide, see :doc:`advanced_usage` for advanced techniques

EZStitcher automatically chains *input_dir* / *output_dir* between steps.
See :doc:`../concepts/directory_structure` for details on how directories are managed.

--------------------------------------------------------------------
Understanding the EZ Module Under the Hood
--------------------------------------------------------------------

The EZ module provides a simplified interface, but behind the scenes, it creates pipelines and steps. When you call ``stitch_plate()``, it creates pipelines similar to this:

.. code-block:: python

   from ezstitcher import stitch_plate

   # This simple call...
   stitch_plate("path/to/plate")

   # ...creates pipelines and steps similar to this:
   # 1. Position Generation Pipeline with:
   #    - ZFlatStep (if Z-stacks are detected)
   #    - NormStep (for normalization)
   #    - CompositeStep (for channel compositing)
   #    - PositionGenerationStep
   #
   # 2. Assembly Pipeline with:
   #    - NormStep (for normalization)
   #    - ImageStitchingStep

Each specialized step serves a specific purpose in the image processing pipeline:

* ``ZFlatStep``: Converts 3D Z-stacks into 2D images using various projection methods
    - ``method="max"``: Maximum intensity projection (brightest pixel)
    - ``method="mean"``: Average intensity projection
    - ``method="focus"``: Focus-based projection for better detail

* ``NormStep``: Normalizes image intensities for consistent visualization
    - Applies percentile-based normalization (default: 1-99 percentile)
    - Helps balance brightness across different images

* ``CompositeStep``: Combines multiple channels into a single reference image
    - Accepts weights to control channel contributions, equal weighting by default
    - Example: ``weights=[0.7, 0.3, 0]`` uses 70% channel 1, 30% channel 2, 0% channel 3

* ``PositionGenerationStep``: Analyzes images to determine how tiles fit together
    - Detects overlapping regions between adjacent tiles
    - Generates position information for stitching

* ``ImageStitchingStep``: Combines all tiles into final stitched image
    - Uses positions from PositionGenerationStep
    - Handles blending between overlapping regions

By understanding this structure, you can create custom pipelines that provide more control while still leveraging the power of EZStitcher's steps.

--------------------------------------------------------------------
Reimplementing EZ Module Functionality
--------------------------------------------------------------------

Here's how to reimplement the basic EZ module functionality using pipelines and steps:

.. code-block:: python

   from pathlib import Path
   from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
   from ezstitcher.core.pipeline import Pipeline
   from ezstitcher.core.steps import NormStep, ZFlatStep, CompositeStep, PositionGenerationStep, ImageStitchingStep

   plate_path = Path("~/data/PlateA").expanduser()
   orchestrator = PipelineOrchestrator(plate_path)

   # Position generation pipeline
   pos_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           ZFlatStep(),  # Z-stack flattening
           NormStep(),  # Normalization
           CompositeStep(),  # Channel compositing
           PositionGenerationStep(),  # Position generation
       ],
       name="Position Generation",
   )

   # Assembly pipeline
   asm_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           NormStep(),  # Normalization
           ImageStitchingStep(),  # Image stitching
       ],
       name="Assembly",
   )

   orchestrator.run(pipelines=[pos_pipe, asm_pipe])

This approach gives you more control over the processing steps while still using the pre-defined steps that provide a clean interface for common operations.

--------------------------------------------------------------------
Simple Examples of Custom Pipelines
--------------------------------------------------------------------

**Z-stack processing:**

Here's how to process Z-stacks with custom pipelines:

.. code-block:: python

   from pathlib import Path
   from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
   from ezstitcher.core.pipeline import Pipeline
   from ezstitcher.core.steps import NormStep, ZFlatStep, FocusStep, CompositeStep, PositionGenerationStep, ImageStitchingStep

   plate_path = Path("~/data/PlateA").expanduser()
   orchestrator = PipelineOrchestrator(plate_path)

   # Position generation pipeline with Z-stack flattening
   pos_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           ZFlatStep(),  # Z-stack flattening
           NormStep(),  # Normalization
           CompositeStep(),  # Channel compositing
           PositionGenerationStep(),  # Position generation
       ],
       name="Position Generation",
   )

   # Assembly pipeline
   asm_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           NormStep(),  # Normalization
           #This is the only difference from the previous example
           FocusStep(focus_options={'metric': 'combined'}),  # Focus-based Z processing
           ImageStitchingStep(),  # Image stitching
       ],
       name="Assembly",
   )

   orchestrator.run(pipelines=[pos_pipe, asm_pipe])

**Customizing step parameters:**

You can customize the behavior of steps by passing parameters:

.. code-block:: python

   # Customize Z-flattening method
   ZFlatStep(method="mean")  # Use mean projection instead of max projection

   # Customize focus metrics
   FocusStep(focus_options={'metric': 'combined'})  # Use combined focus metric
   FocusStep(focus_options={'metric': 'laplacian'})  # Use Laplacian focus metric

   # Customize normalization
   NormStep(percentile=95)  # Use 95th percentile for normalization

   # Customize channel compositing
   CompositeStep(weights=[0.7, 0.3, 0])  # Custom weights for RGB channels

--------------------------------------------------------------------
When to Move to Advanced Usage
--------------------------------------------------------------------

Consider moving to the advanced usage level when:

* You need to implement custom processing functions
* You want to understand the implementation details of steps
* You need to extend EZStitcher with new functionality
* You want to create your own custom steps

The advanced usage level provides deeper insights into how EZStitcher works and how to extend it for your specific needs.

Next up: :doc:`advanced_usage`.


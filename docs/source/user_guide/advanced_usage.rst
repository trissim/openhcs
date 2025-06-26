==============
Advanced Usage
==============

This page shows **three advanced skills** for users who need to go beyond pre-defined steps:

1. Write *custom processing functions* and wire them into pipelines using the base Step class
2. Enable **multithreaded** execution for large plates
3. Implement advanced functional patterns for complex workflows

**Learning Path:**

1. If you are new to EZStitcher, start with the :doc:`basic_usage` guide (beginner level)
2. Next, learn about custom pipelines with steps in :doc:`intermediate_usage` (intermediate level)
3. Now you're ready for this advanced usage guide with the base Step class
4. For integration with other tools, see :doc:`integration`

---------------------------------------------------------------------
Understanding Pre-defined Steps
---------------------------------------------------------------------

Pre-defined steps are simply wrapped versions of the base Step class with pre-configured parameters.
For example, when you use ``NormStep()``, you're actually using this under the hood:

.. code-block:: python

   # NormStep is equivalent to:
   Step(
       func=(IP.stack_percentile_normalize, {
           'low_percentile': 0.1,
           'high_percentile': 99.9
       }),
       name="Percentile Normalization"
   )

Similarly, ``ZFlatStep`` wraps ``IP.create_projection`` with ``variable_components=['z_index']``,
and ``CompositeStep`` wraps ``IP.create_composite`` with ``variable_components=['channel']``.

You can create your own custom steps by following the same pattern. For more details, see:
- :doc:`../concepts/step` for step configuration
- :doc:`../concepts/function_handling` for function patterns
- :doc:`../api/steps` for API reference

---------------------------------------------------------------------
1. Creating custom processing functions
---------------------------------------------------------------------

Custom functions receive **a list of NumPy arrays** (images) and must return the *same‑length* list.
For details on function patterns, see :doc:`../concepts/function_handling`.

.. code-block:: python

   import numpy as np
   from skimage import filters

   def custom_enhance(images, sigma=1.0, contrast=1.5):
       """Gaussian blur + contrast stretch."""
       out = []
       for im in images:
           blurred = filters.gaussian(im, sigma=sigma)
           mean    = blurred.mean()
           out.append(np.clip(mean + contrast * (blurred - mean), 0, 1))
       return out

   # Use in a Step with any of the function patterns:
   step = Step(func=custom_enhance)  # Basic usage
   step = Step(func=(custom_enhance, {'sigma': 2.0, 'contrast': 1.8}))  # With arguments

---------------------------------------------------------------------
2. Building an advanced custom pipeline
---------------------------------------------------------------------

Below we denoise, normalise, enhance and then stitch — all with **two concise pipelines**.

.. code-block:: python

   from pathlib import Path

   from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
   from ezstitcher.core.pipeline           import Pipeline
   from ezstitcher.core.steps              import Step, NormStep, PositionGenerationStep, ImageStitchingStep, ZFlatStep, CompositeStep
   from ezstitcher.core.image_processor    import ImageProcessor as IP

   # ---------- orchestrator ----------------------------------------
   plate_path   = Path("~/data/PlateA").expanduser()
   orchestrator = PipelineOrchestrator(plate_path)

   # ---------- helper functions -----------------------------------
   def denoise(images, strength=0.5):
       from skimage.restoration import denoise_nl_means
       return [denoise_nl_means(im, h=strength) for im in images]

   # ---------- position pipeline ----------------------------------
   pos_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           ZFlatStep(method="max"),  # Z-stack flattening
           Step(func=(denoise, {"strength": 0.4})),  # Custom denoising
           NormStep(),  # Normalization (replaces Step(func=IP.stack_percentile_normalize))
           CompositeStep(),  # Channel compositing
           PositionGenerationStep(),  # Position generation
       ],
       name="Position Generation",
   )
   positions_dir = pos_pipe.steps[-1].output_dir

   # ---------- assembly pipeline ----------------------------------
   asm_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       output_dir=Path("out/stitched"),
       steps=[
           Step(func=(denoise, {"strength": 0.4})),  # Custom denoising
           NormStep(),  # Normalization (replaces Step(func=IP.stack_percentile_normalize))
           ImageStitchingStep(positions_dir=positions_dir),  # Image stitching
       ],
       name="Assembly",
   )

   orchestrator.run(pipelines=[pos_pipe, asm_pipe])

---------------------------------------------------------------------
3. Channel‑aware processing with ``group_by='channel'``
---------------------------------------------------------------------

.. code-block:: python

   def process_dapi(images):
       return IP.stack_percentile_normalize([IP.tophat(im, size=15) for im in images])

   def process_gfp(images):
       return IP.stack_percentile_normalize([IP.sharpen(im, sigma=1.0, amount=1.5) for im in images])

   channel_step = Step(func={"1": process_dapi, "2": process_gfp}, group_by="channel")

.. important::
   The interplay between ``group_by`` and ``variable_components`` controls **how your function loops**. 
   See :doc:`../concepts/step` and :doc:`../concepts/function_handling` for detailed explanations.

---------------------------------------------------------------------
4. Conditional processing based on context
---------------------------------------------------------------------

The *context* dict is passed to every Step when ``pass_context=True``.

.. code-block:: python

   def conditional(images, context):
       if context["well"] == "A01":
           return process_control(images)
       return process_treatment(images)

   cond_step = Step(func=conditional, pass_context=True)

---------------------------------------------------------------------
5. Multithreading for large plates
---------------------------------------------------------------------

.. code-block:: python

   from ezstitcher.core.config import PipelineConfig

   cfg = PipelineConfig(num_workers=4)  # use 4 threads
   orchestrator = PipelineOrchestrator(plate_path, config=cfg)
   orchestrator.run(pipelines=[pos_pipe, asm_pipe])

Threads are allocated **per well**; inside a well, steps run sequentially.
Adjust `num_workers` to avoid memory exhaustion.

---------------------------------------------------------------------
6. Adding a new microscope handler
---------------------------------------------------------------------

Implement :class:`~ezstitcher.core.microscope_handler.BaseMicroscopeHandler` and register it via ``register_handler``.
See :doc:`../development/extending` for the full walkthrough.

---------------------------------------------------------------------
Choosing the right approach
--------------------------

* **EZ module** → Quick wins with minimal code for standard plates
* **Custom pipelines** → Full control for specialized workflows and research prototypes

For more information on the three-tier approach and when to use each approach, see the :ref:`three-tier-approach` section in the introduction.


Next steps
~~~~~~~~~~

* Read the :doc:`integration` guide for BaSiCPy and N2V2 (Careamics) integration examples
* Review :doc:`../concepts/best_practices` for pipeline organization and optimization tips
* Explore :doc:`../concepts/architecture_overview` to understand core concepts in greater detail



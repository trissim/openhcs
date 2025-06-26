Steps
=====

.. module:: ezstitcher.core.steps

This module contains the Step class and all step implementations for the EZStitcher pipeline architecture,
including the base Step class and various step types like ZFlatStep, FocusStep, CompositeStep, PositionGenerationStep, and ImageStitchingStep.

For conceptual explanation, see :doc:`../concepts/step`.
For information about function handling in steps, see :doc:`../concepts/function_handling`.
For information about directory structure, see :doc:`../concepts/directory_structure`.

Step
----

.. py:class:: Step(*, func, variable_components=None, group_by=None, input_dir=None, output_dir=None, well_filter=None, name=None)

   A processing step in a pipeline.

   For detailed information about step parameters and their usage, see :ref:`step-parameters` in the :doc:`../concepts/step` documentation.

   For information about variable components, see :ref:`variable-components` in the :doc:`../concepts/step` documentation.

   For information about the group_by parameter, see :ref:`group-by` in the :doc:`../concepts/step` documentation.

   For best practices when using steps, see :doc:`../user_guide/best_practices` documentation.

   :param func: The processing function(s) to apply. Can be a single callable, a tuple of (function, kwargs), a list of functions or function tuples, or a dictionary mapping component values to functions or function tuples.
   :type func: callable, tuple, list, or dict
   :param variable_components: Components that vary across files (e.g., 'z_index', 'channel')
   :type variable_components: list
   :param group_by: How to group files for processing (e.g., 'channel', 'site')
   :type group_by: str
   :param input_dir: The input directory
   :type input_dir: str or Path
   :param output_dir: The output directory
   :type output_dir: str or Path
   :param well_filter: Wells to process
   :type well_filter: list
   :param name: Human-readable name for the step
   :type name: str

   .. py:method:: process(context)

      Process the step with the given context.

      :param context: The processing context
      :type context: :class:`~ezstitcher.core.pipeline.ProcessingContext`
      :return: The updated processing context
      :rtype: :class:`~ezstitcher.core.pipeline.ProcessingContext`

PositionGenerationStep
---------------------

.. py:class:: PositionGenerationStep(*, name="Position Generation", input_dir=None, output_dir=None)

   A specialized Step for generating positions.

   This step takes processed reference images and generates position files for stitching.
   It stores the positions file in the context for later use by ImageStitchingStep.

   :param name: Name of the step (optional)
   :type name: str
   :param input_dir: Input directory (optional)
   :type input_dir: str or Path
   :param output_dir: Output directory for positions files (optional)
   :type output_dir: str or Path

   .. py:method:: process(context)

      Generate positions for stitching and store them in the context.

      :param context: The processing context
      :type context: :class:`~ezstitcher.core.pipeline.ProcessingContext`
      :return: The updated processing context
      :rtype: :class:`~ezstitcher.core.pipeline.ProcessingContext`

ImageStitchingStep
----------------

.. py:class:: ImageStitchingStep(*, name="Image Stitching", input_dir=None, positions_dir=None, output_dir=None)

   A specialized Step for stitching images using position files.

   This step stitches images using position files. It works with the PositionGenerationStep
   to create complete stitched images from individual tiles.

   :param name: Name of the step (optional)
   :type name: str
   :param input_dir: Input directory containing images to stitch (optional)
   :type input_dir: str or Path
   :param positions_dir: Directory containing position files (optional, can be provided in context)
   :type positions_dir: str or Path
   :param output_dir: Output directory for stitched images (optional)
   :type output_dir: str or Path

   .. py:method:: process(context)

      Stitch images using the positions file from the context.

      This step:
      1. Locates the positions file for the current well
      2. Loads images according to the positions file
      3. Stitches the images together
      4. Saves the stitched image to the output directory

      :param context: The processing context
      :type context: :class:`~ezstitcher.core.pipeline.ProcessingContext`
      :return: The updated processing context
      :rtype: :class:`~ezstitcher.core.pipeline.ProcessingContext`

ZFlatStep
--------

.. py:class:: ZFlatStep(*, method="max", input_dir=None, output_dir=None, well_filter=None)

   Specialized step for Z-stack flattening.

   This step performs Z-stack flattening using the specified method.
   It pre-configures variable_components=['z_index'] and group_by=None.

   :param method: Projection method. Options: "max", "mean", "median", "min", "std", "sum"
   :type method: str
   :param input_dir: Input directory
   :type input_dir: str or Path, optional
   :param output_dir: Output directory
   :type output_dir: str or Path, optional
   :param well_filter: Wells to process
   :type well_filter: list, optional

FocusStep
--------

.. py:class:: FocusStep(*, focus_options=None, input_dir=None, output_dir=None, well_filter=None)

   Specialized step for focus-based Z-stack processing.

   This step finds the best focus plane in a Z-stack using FocusAnalyzer.
   It pre-configures variable_components=['z_index'] and group_by=None.

   :param focus_options: Dictionary of focus analyzer options:
                        - metric: Focus metric. Options: "combined", "normalized_variance",
                                 "laplacian", "tenengrad", "fft" or a dictionary of weights (default: "combined")
   :type focus_options: dict, optional
   :param input_dir: Input directory
   :type input_dir: str or Path, optional
   :param output_dir: Output directory
   :type output_dir: str or Path, optional
   :param well_filter: Wells to process
   :type well_filter: list, optional

CompositeStep
-----------

.. py:class:: CompositeStep(*, weights=None, input_dir=None, output_dir=None, well_filter=None)

   Specialized step for creating composite images from multiple channels.

   This step creates composite images from multiple channels with specified weights.
   It pre-configures variable_components=['channel'] and group_by=None.

   :param weights: List of weights for each channel. If None, equal weights are used.
   :type weights: list, optional
   :param input_dir: Input directory
   :type input_dir: str or Path, optional
   :param output_dir: Output directory
   :type output_dir: str or Path, optional
   :param well_filter: Wells to process
   :type well_filter: list, optional
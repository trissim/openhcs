EZ Module
=========

.. module:: ezstitcher.ez

This module provides a simplified interface for stitching microscopy images with minimal code.

EZStitcher Class
--------------

.. py:class:: EZStitcher(input_path, output_path=None, normalize=True, flatten_z=None, z_method="max", channel_weights=None, well_filter=None)

   Simplified interface for microscopy image stitching.

   This class provides an easy-to-use interface for common stitching workflows,
   hiding the complexity of pipelines and orchestrators.

   :param input_path: Path to the plate folder
   :type input_path: str or Path
   :param output_path: Path for output (default: input_path + "_stitched")
   :type output_path: str or Path, optional
   :param normalize: Whether to apply normalization
   :type normalize: bool, default=True
   :param flatten_z: Whether to flatten Z-stacks (auto-detected if None)
   :type flatten_z: bool or None, optional
   :param z_method: Method for Z-flattening ("max", "mean", "focus", etc.)
   :type z_method: str, default="max"
   :param channel_weights: Weights for channel compositing (auto-detected if None)
   :type channel_weights: list of float or None, optional
   :param well_filter: List of wells to process (processes all if None)
   :type well_filter: list of str or None, optional

   .. py:method:: set_options(**kwargs)

      Update configuration options.

      :param kwargs: Configuration options to update
      :return: self for method chaining

   .. py:method:: stitch()

      Run the complete stitching process with current settings.

      :return: Path to the output directory
      :rtype: Path

stitch_plate Function
------------------

.. py:function:: stitch_plate(input_path, output_path=None, **kwargs)

   One-liner function to stitch a plate of microscopy images.

   :param input_path: Path to the plate folder
   :type input_path: str or Path
   :param output_path: Path for output (default: input_path + "_stitched")
   :type output_path: str or Path, optional
   :param kwargs: Additional options passed to EZStitcher
   :return: Path to the stitched output
   :rtype: Path

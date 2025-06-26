Configuration
=============

.. module:: ezstitcher.core.config

This module contains configuration classes for ezstitcher.

PipelineConfig
--------------

.. py:class:: PipelineConfig

   Configuration for the pipeline orchestrator.

   .. py:attribute:: out_dir_suffix
      :type: str
      :value: "_out"

      Suffix for processing steps output directories.

   .. py:attribute:: positions_dir_suffix
      :type: str
      :value: "_positions"

      Suffix for position generation step output directories.

   .. py:attribute:: stitched_dir_suffix
      :type: str
      :value: "_stitched"

      Suffix for stitching step output directories.

   .. py:attribute:: num_workers
      :type: int
      :value: 1

      Number of worker threads for parallel processing.

   .. py:attribute:: well_filter
      :type: list or None
      :value: None

      Optional list of wells to process.

   .. py:attribute:: stitcher
      :type: StitcherConfig
      :value: StitcherConfig()

      Configuration for the Stitcher class.



StitcherConfig
--------------

.. py:class:: StitcherConfig

   Configuration for the Stitcher class.

   .. py:attribute:: tile_overlap
      :type: float
      :value: 10.0

      Percentage overlap between tiles.

   .. py:attribute:: max_shift
      :type: int
      :value: 50

      Maximum allowed shift in pixels.

   .. py:attribute:: margin_ratio
      :type: float
      :value: 0.1

      Ratio of image size to use as margin for blending.

   .. py:attribute:: pixel_size
      :type: float
      :value: 1.0

      Pixel size in micrometers.





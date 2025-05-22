PipelineOrchestrator
==================

.. module:: ezstitcher.core.pipeline_orchestrator

The PipelineOrchestrator is the central coordinator that manages the execution of multiple pipelines across wells.

For conceptual explanation, see :doc:`../concepts/pipeline_orchestrator`.
For information about directory structure, see :doc:`../concepts/directory_structure`.

.. py:class:: PipelineOrchestrator(plate_path=None, workspace_path=None, config=None, fs_manager=None, image_preprocessor=None, focus_analyzer=None)

   The central coordinator that manages the execution of multiple pipelines across wells.

   For detailed information about how the orchestrator runs pipelines, see :ref:`orchestrator-running-pipelines` in the :doc:`../concepts/pipeline_orchestrator` documentation.

   For information about the relationship between the orchestrator and pipeline, see :ref:`orchestrator-pipeline-relationship` in the :doc:`../concepts/pipeline_orchestrator` documentation.

   :param plate_path: Path to the plate folder (optional, can be provided later in run())
   :type plate_path: str or Path
   :param workspace_path: Path to the workspace folder (optional, defaults to plate_path.parent/plate_path.name_workspace)
   :type workspace_path: str or Path
   :param config: Configuration for the pipeline orchestrator
   :type config: :class:`~ezstitcher.core.config.PipelineConfig`
   :param fs_manager: File system manager (optional, a new instance will be created if not provided)
   :type fs_manager: :class:`~ezstitcher.core.file_system_manager.FileSystemManager`
   :param image_processor: Image processor (optional, a new instance will be created if not provided)
   :type image_processor: :class:`~ezstitcher.core.image_processor.ImageProcessor`
   :param focus_analyzer: Focus analyzer (optional, a new instance will be created if not provided)
   :type focus_analyzer: :class:`~ezstitcher.core.focus_analyzer.FocusAnalyzer`

   .. py:method:: run(plate_path=None, pipelines=None)

      Run the pipeline orchestrator with the specified pipelines.

      :param plate_path: Path to the plate folder (optional if provided in __init__)
      :type plate_path: str or Path
      :param pipelines: List of pipelines to run
      :type pipelines: list of :class:`~ezstitcher.core.pipeline.Pipeline`
      :return: True if successful, False otherwise
      :rtype: bool

   .. py:method:: process_well(well, pipelines)

      Process a single well with the specified pipelines.

      :param well: Well identifier
      :type well: str
      :param pipelines: List of pipelines to run
      :type pipelines: list of :class:`~ezstitcher.core.pipeline.Pipeline`
      :return: True if successful, False otherwise
      :rtype: bool

   .. note::

      The ``setup_directories()`` method has been removed. Directory paths are now automatically resolved between steps.
      See :doc:`../concepts/directory_structure` for details on how EZStitcher manages directories.

   .. py:method:: detect_plate_structure(plate_path)

      Detect the plate structure and available wells.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path

   .. py:method:: generate_positions(well, input_dir, positions_dir)

      Generate stitching positions for a well.

      :param well: Well identifier
      :type well: str
      :param input_dir: Input directory containing reference images
      :type input_dir: str or Path
      :param positions_dir: Output directory for positions files
      :type positions_dir: str or Path
      :return: Tuple of (positions_dir, reference_pattern)
      :rtype: tuple

   .. py:method:: stitch_images(well, input_dir, output_dir, positions_path)

      Stitch images for a well.

      :param well: Well identifier
      :type well: str
      :param input_dir: Input directory containing processed images
      :type input_dir: str or Path
      :param output_dir: Output directory for stitched images
      :type output_dir: str or Path
      :param positions_path: Path to positions file
      :type positions_path: str or Path

Related Classes
--------------

For documentation on related classes, see:

- :doc:`pipeline` - Documentation for the Pipeline class and ProcessingContext
- :doc:`steps` - Documentation for the Step class and its specialized subclasses

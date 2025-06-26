Pipeline
=======

.. module:: ezstitcher.core.pipeline

This module contains the Pipeline class and related components for the EZStitcher pipeline architecture.

For conceptual explanation, see :doc:`../concepts/pipeline`.
For information about directory structure, see :doc:`../concepts/directory_structure`.

Pipeline
-------

.. py:class:: Pipeline(steps=None, name=None)

   A sequence of processing steps that are executed in order.

   For detailed information on pipeline construction, including best practices and different approaches,
   see :doc:`../concepts/pipeline`.

   :param steps: Initial list of steps
   :type steps: list of :class:`~ezstitcher.core.steps.Step`
   :param name: Human-readable name for the pipeline
   :type name: str

   .. py:method:: add_step(step)

      Add a step to the pipeline. See :doc:`../concepts/pipeline` for best practices on when to use this method
      versus providing all steps during initialization.

      :param step: The step to add
      :type step: :class:`~ezstitcher.core.steps.Step`
      :return: Self, for method chaining
      :rtype: :class:`Pipeline`

   .. py:method:: run(input_dir=None, output_dir=None, well_filter=None, microscope_handler=None, orchestrator=None, positions_file=None)

      Execute the pipeline.

      This method can either:

      1. Take individual parameters and create a ProcessingContext internally, or
      2. Take a pre-configured ProcessingContext object (when called from PipelineOrchestrator)

      The orchestrator parameter is required as it provides access to the microscope handler and other components.

      :param input_dir: Optional input directory override
      :type input_dir: str or Path
      :param output_dir: Optional output directory override
      :type output_dir: str or Path
      :param well_filter: Optional well filter override
      :type well_filter: list
      :param microscope_handler: Optional microscope handler override
      :type microscope_handler: :class:`~ezstitcher.core.microscope_interfaces.MicroscopeHandler`
      :param orchestrator: PipelineOrchestrator instance (required)
      :type orchestrator: :class:`~ezstitcher.core.processing_pipeline.PipelineOrchestrator`
      :param positions_file: Optional positions file to use for stitching
      :type positions_file: str or Path
      :return: The results of the pipeline execution
      :rtype: dict

   .. py:attribute:: input_dir

      Get or set the input directory for the pipeline.

      :type: Path or None

   .. py:attribute:: output_dir

      Get or set the output directory for the pipeline.

      :type: Path or None

ProcessingContext
---------------

.. py:class:: ProcessingContext(input_dir=None, output_dir=None, well_filter=None, config=None, **kwargs)

   Maintains state during pipeline execution.

   The ProcessingContext holds input/output directories, well filter, configuration,
   and results during pipeline execution. It serves as a communication mechanism
   between steps in a pipeline, allowing each step to access and modify shared state.

   For detailed information about how the context is used for communication between steps,
   see :ref:`pipeline-context` in the :doc:`../concepts/pipeline` documentation.

   :param input_dir: The input directory
   :type input_dir: str or Path
   :param output_dir: The output directory
   :type output_dir: str or Path
   :param well_filter: Wells to process
   :type well_filter: list
   :param config: Configuration parameters
   :type config: dict
   :param **kwargs: Additional context attributes that will be added to the context

   .. py:attribute:: input_dir

      The input directory for processing.

      :type: Path or None

   .. py:attribute:: output_dir

      The output directory for processing results.

      :type: Path or None

   .. py:attribute:: well_filter

      List of wells to process.

      :type: list or None

   .. py:attribute:: config

      Configuration parameters.

      :type: dict

   .. py:attribute:: results

      Processing results.

      :type: dict

Step Classes
-----------

For documentation on step classes like ``ImageStitchingStep``, see :doc:`steps`.

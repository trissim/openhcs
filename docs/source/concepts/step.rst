====
Step
====

Overview
--------

A ``FunctionStep`` is the primary building block for OpenHCS bioimage analysis pipelines. Each step represents a single processing operation that can be applied to microscopy images with automatic GPU acceleration, memory type conversion, and parallel execution.

For an overview of the complete architecture, see :doc:`architecture_overview`.

OpenHCS uses ``FunctionStep`` as the unified step type that supports:

1. **Function Patterns**: Single functions, function chains, and component-specific processing
2. **Memory Type Integration**: Automatic conversion between NumPy, CuPy, PyTorch, JAX, pyclesperanto
3. **GPU Acceleration**: Automatic GPU resource management and optimization
4. **Variable Components**: Processing by site, channel, or other microscopy dimensions

**Real-World Examples** (from TUI-generated scripts):

.. code-block:: python

   from openhcs.core.steps.function_step import FunctionStep
   from openhcs.constants.constants import VariableComponents
   from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
   from openhcs.processing.backends.processors.cupy_processor import tophat

   # Single function step
   normalize_step = FunctionStep(
       func=[(stack_percentile_normalize, {
           'low_percentile': 1.0,
           'high_percentile': 99.0,
           'target_max': 65535.0
       })],
       name="normalize",
       variable_components=[VariableComponents.SITE]
   )

   # Function chain step (multiple operations in sequence)
   preprocess_step = FunctionStep(
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
       variable_components=[VariableComponents.SITE]
   )

FunctionStep Architecture
-------------------------

OpenHCS FunctionSteps follow a functional, stateless architecture with automatic resource management:

**Core Principles**:

- **Stateless Processing**: Steps are pure functions that don't maintain state between executions
- **Memory Type Agnostic**: Functions work with any memory type (NumPy, CuPy, PyTorch, JAX, pyclesperanto)
- **GPU Resource Management**: Automatic GPU allocation and memory optimization
- **VFS Integration**: All I/O operations go through the Virtual File System

**Processing Flow**:

1. **Input Loading**: Images loaded as 3D arrays (ZYX format) from VFS backends
2. **Memory Type Conversion**: Automatic conversion to function's required memory type
3. **Function Execution**: Processing with GPU acceleration when available
4. **Output Conversion**: Results converted back to storage format
5. **VFS Storage**: Results saved through VFS with backend optimization

**Function Pattern Examples** (from TUI-generated scripts):

.. code-block:: python

   # Single function with parameters
   step_1 = FunctionStep(
       func=[(create_composite, {})],
       name="composite",
       variable_components=[VariableComponents.CHANNEL],
       force_disk_output=False
   )

   # Function chain (multiple operations in sequence)
   step_2 = FunctionStep(
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
   )

   # Dictionary pattern (component-specific processing)
   step_3 = FunctionStep(
       func={
           '1': [(count_cells_single_channel, {
               'min_sigma': 1.0,
               'max_sigma': 10.0,
               'detection_method': DetectionMethod.WATERSHED
           })],
           '2': [(skan_axon_skeletonize_and_analyze, {
               'voxel_spacing': (1.0, 1.0, 1.0),
               'min_branch_length': 10.0
           })]
       },
       name="analysis",
       variable_components=[VariableComponents.SITE],
       force_disk_output=False
   )

Variable Components
-------------------

Variable components define how OpenHCS processes images across different microscopy dimensions:

.. code-block:: python

   from openhcs.constants.constants import VariableComponents

   # Process each site independently (most common)
   variable_components=[VariableComponents.SITE]

   # Process each channel independently
   variable_components=[VariableComponents.CHANNEL]

   # Process each timepoint independently
   variable_components=[VariableComponents.TIME]

**Processing Behavior**:

- **SITE**: Each microscopy site processed separately (parallel processing across sites)
- **CHANNEL**: Each channel processed separately (useful for channel-specific operations)
- **TIME**: Each timepoint processed separately (for time-series analysis)

The orchestrator automatically handles parallel execution across the specified variable components.

.. _step-parameters:

Step Parameters
-------------

For detailed API documentation, see :doc:`../api/steps`.

* ``name``: Human-readable name for the step
* ``func``: The processing function(s) to apply (see :doc:`function_handling`)
* ``variable_components``: Components that vary across files (e.g., 'z_index', 'channel')
* ``group_by``: How to group files for processing (e.g., 'channel', 'site')
* ``input_dir``: The input directory (optional, can inherit from pipeline)
* ``output_dir``: The output directory (optional, can inherit from pipeline)
* ``well_filter``: Wells to process (optional, can inherit from pipeline)

For practical examples of how to use these parameters in different scenarios, see:

* :doc:`../user_guide/basic_usage` - Basic examples of step parameters
* :doc:`../user_guide/intermediate_usage` - Examples of variable_components and group_by
* :doc:`../user_guide/advanced_usage` - Advanced examples of func parameter
* :doc:`../user_guide/best_practices` - Best practices for step parameters

Processing Arguments
------------------

Processing arguments are passed directly with the function using the tuple pattern ``(func, kwargs)``. For detailed information about function patterns and usage, see :doc:`function_handling`.

.. code-block:: python

    # Pass arguments to a function
    step = Step(
        func=(IP.create_projection, {'method': 'max_projection'}),
        name="Z-Stack Flattening",
        variable_components=['z_index'],
        input_dir=orchestrator.workspace_path
    )

This pattern can be used with:

* Single functions (:ref:`function-single`, :ref:`function-with-arguments`)
* Lists of functions (:ref:`function-lists`, :ref:`function-lists-with-arguments`)
* Dictionaries of functions (:ref:`function-dictionaries`, :ref:`function-dictionary-tuples`)
* Mixed function types (:ref:`function-mixed-types`)

.. note::
   Always use the tuple pattern ``(func, kwargs)`` to pass arguments to processing functions.
   This is the recommended approach for all function arguments.

Step Initialization Best Practices
--------------------------------

When initializing steps, it's important to follow best practices for directory specification.

Steps can specify input and output directories.
For detailed information about directory structure, see :doc:`directory_structure`.

.. _variable-components:

Variable Components
-----------------

The ``variable_components`` parameter specifies which components will be grouped together for processing. It determines how images are organized into stacks before being passed to the processing function.

**Key concept**: Images that share the same values for all components *except* the variable component will be grouped together into a stack.

In most cases, you don't need to set this explicitly as it defaults to ``['site']``, but there are specific cases where you should change it.

For practical examples of how to use variable_components in different scenarios, see:

* :doc:`../user_guide/intermediate_usage` - Examples for Z-stack processing and channel compositing
* :doc:`../user_guide/advanced_usage` - Advanced examples with custom functions

.. code-block:: python

    # IMPORTANT: For Z-stack flattening, use ZFlatStep instead of raw Step with variable_components
    # This is the recommended approach for Z-stack flattening
    from ezstitcher.core.steps import ZFlatStep

    # Maximum intensity projection (default)
    step = ZFlatStep()  # Uses max_projection by default

    # With specific projection method
    step = ZFlatStep(method="mean")  # Uses mean_projection

    # IMPORTANT: For channel compositing, use CompositeStep instead of raw Step with variable_components
    # This is the recommended approach for channel compositing
    from ezstitcher.core.steps import CompositeStep

    # Without weights (equal weighting for all channels)
    step = CompositeStep()  # Equal weights for all channels

    # With custom weights (70% channel 1, 30% channel 2)
    step = CompositeStep(weights=[0.7, 0.3])  # Custom channel weights

    # For most other operations, the default 'site' is appropriate
    # This groups images with the same channel, z_index, etc. but different site values
    # The function will receive a stack of images with varying site values
    step = Step(
        func=stack(IP.sharpen),
        name="Enhance Images"
        # variable_components defaults to ['site']
    )

.. _group-by:

Group By
-------

The ``group_by`` parameter is specifically designed for use with function dictionaries in ``Step``. It specifies what component the keys in your function dictionary correspond to.

For practical examples of how to use group_by in different scenarios, see:

* :doc:`../user_guide/intermediate_usage` - Examples for channel-specific processing
* :doc:`../user_guide/advanced_usage` - Advanced examples with dictionaries of functions

.. code-block:: python

    # When using a dictionary of channel-specific functions
    step = Step(
        func={"1": process_dapi, "2": process_calcein},
        name="Channel-Specific Processing",
        # variable_components defaults to ['site']
        group_by='channel'  # Keys "1" and "2" correspond to channel values
    )

**Key concept**: The ``group_by`` parameter tells the Step what the keys in the function dictionary represent.

In this example:
- ``group_by='channel'`` means the keys in the function dictionary ("1" and "2") correspond to channel values
- Images with channel="1" will be processed by ``process_dapi``
- Images with channel="2" will be processed by ``process_calcein``

**Parameter Relationships and Constraints**:

1. ``group_by`` is **only needed when using a dictionary of functions**. It's not needed for single functions or lists of functions.

2. ``group_by`` should **NEVER be the same as** ``variable_components``:

   This is a critical rule that must be followed to avoid logical errors. When ``variable_components=['channel']``, it means we're processing each channel separately. When ``group_by='channel'``, it means we're grouping functions by channel. If these were the same, it would create a logical contradiction in how the images are processed.

   .. code-block:: python

       # CORRECT: variable_components and group_by are different
       step = Step(
           func={"1": process_dapi, "2": process_calcein},
           name="Channel-Specific Processing",
           variable_components=['site'],  # Process each site separately
           group_by='channel'  # Keys "1" and "2" correspond to channel values
       )

       # INCORRECT: variable_components and group_by are the same
       # This will lead to logical errors and should never be done
       step = Step(
           func={"1": process_dapi, "2": process_calcein},
           name="Incorrect Setup",
           variable_components=['channel'],  # Process each channel separately
           group_by='channel'  # Keys "1" and "2" correspond to channel values
       )

3. ``group_by`` is typically only set when ``variable_components`` is left at its default value of ``['site']``:

   .. code-block:: python

       # Typical pattern: variable_components defaults to ['site'], group_by is set to 'channel'
       step = Step(
           func={"1": process_dapi, "2": process_calcein},
           name="Channel-Specific Processing",
           # variable_components defaults to ['site']
           group_by='channel'  # Keys "1" and "2" correspond to channel values
       )

4. ``input_dir`` must be specified for the first step in a pipeline, typically using ``orchestrator.workspace_path``.

5. ``output_dir`` is optional and will be automatically determined if not specified.

6. ``well_filter`` is optional and will inherit from the pipeline's context if not specified.

.. _step-best-practices:

StepResult
---------

The StepResult class is the canonical interface for step outputs in EZStitcher. It provides
a clear structure for step results, separating normal processing results from context updates
and storage operations.

.. code-block:: python

    from ezstitcher.core.step_result import StepResult
    from ezstitcher.io.virtual_path import PhysicalPath

    # Create a StepResult with an output path and context updates
    result = StepResult(
        output_path=PhysicalPath("/path/to/output"),
        context_update={"key": "value"},
        metadata={"execution_time": 1.23},
        results={"image": image_array},
        storage_operations=[("storage_key", data)]
    )

    # Or use the factory method
    result = StepResult.create(
        output_path=PhysicalPath("/path/to/output"),
        context_update={"key": "value"}
    )

    # StepResult is immutable, so use methods to create new instances
    result2 = result.add_result("new_key", "new_value")
    result3 = result2.update_context("context_key", "context_value")
    result4 = result3.store("storage_key", data)
    result5 = result4.add_metadata("metadata_key", "metadata_value")
    result6 = result5.with_output_path(PhysicalPath("/new/path"))

    # Merge two StepResults
    merged = result.merge(result2)

Storage Adapter Usage
-----------------

Steps automatically use the ``StorageAdapter`` when available. The ``Step._save_images`` method checks for a
StorageAdapter in the context and uses it if available, falling back to FileManager only when necessary.

When a StorageAdapter is available (storage_mode is "memory" or "zarr"), processed images are stored using the
StorageAdapter instead of being written directly to disk. This provides several benefits:

1. **Performance**: Memory storage can be faster than disk I/O for intermediate results
2. **Persistence**: Zarr storage provides immediate persistence to disk
3. **Flexibility**: Different storage backends can be used without changing step code

For detailed information about storage adapters, see :doc:`storage_adapter`.

.. code-block:: python

    # Create an orchestrator with ZARR storage configuration
    from openhcs.core.config import VFSConfig, ZarrConfig
    from openhcs.constants.constants import MaterializationBackend

    global_config = GlobalPipelineConfig(
        vfs=VFSConfig(materialization_backend=MaterializationBackend.ZARR),
        zarr=ZarrConfig()
    )

    orchestrator = PipelineOrchestrator(
        plate_path="path/to/plate",
        global_config=global_config
    )

    # Create a pipeline with steps
    pipeline = Pipeline(
        steps=[
            Step(name="Image Enhancement", func=IP.stack_percentile_normalize),
            # ... more steps
        ]
    )

    # Run the pipeline
    # Steps will automatically use the StorageAdapter
    orchestrator.run(pipelines=[pipeline])

Best Practices
------------

For comprehensive best practices on using steps effectively, see :ref:`best-practices-steps` in the :doc:`../user_guide/best_practices` guide.

For information on when to use specialized steps, see :ref:`best-practices-steps` in the :doc:`../user_guide/best_practices` guide.

For channel-specific processing with different functions per channel, using a raw ``Step`` with a dictionary
of functions and ``group_by='channel'`` is the appropriate approach.

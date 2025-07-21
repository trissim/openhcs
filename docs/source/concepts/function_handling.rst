.. _function-handling:

=================
Function Handling
=================

Function handling is a key aspect of OpenHCS pipeline configuration. OpenHCS supports flexible function patterns that allow you to compose complex bioimage analysis workflows.

For detailed technical information about the function pattern system, see :doc:`../architecture/function_pattern_system`.
For step configuration details, see :doc:`step`.

The ``FunctionStep`` class supports several patterns for processing functions, providing flexibility in how images are processed. This page provides a concise overview of the available patterns.

.. _function-patterns-overview:

Function Patterns Overview
--------------------------

The ``func`` parameter of the ``FunctionStep`` class can accept several types of values:

1. **Single Function**: A callable that processes 3D image arrays. If no arguments are needed, the function can be provided directly without a tuple (e.g., ``func=my_function``).
2. **Function with Arguments**: A tuple of ``(function, kwargs)`` where kwargs is a dictionary of arguments. If no arguments are needed, an empty dictionary can be provided (e.g., ``func=(my_function, {})``), or simply the function itself (as in "Single Function").
3. **List of Functions**: A sequence of functions applied one after another (function chains)
4. **Dictionary of Functions**: A mapping from component values to functions, used with ``variable_components``

**Real-World Examples** (from TUI-generated scripts):

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.constants.constants import VariableComponents
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.processing.backends.processors.cupy_processor import tophat, create_composite
    from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel

    # 1. Single function with parameters
    step = FunctionStep(
        func=[create_composite],
        name="composite",
        variable_components=[VariableComponents.CHANNEL],
        force_disk_output=False
    )

    # 2. Function with complex parameters
    step = FunctionStep(
        func=[(stack_percentile_normalize, {
            'low_percentile': 1.0,
            'high_percentile': 99.0,
            'target_max': 65535.0
        })],
        name="normalize",
        variable_components=[VariableComponents.SITE],
        force_disk_output=False
    )

    # 3. Function chain (list of functions applied in sequence)
    step = FunctionStep(
        func=[
            stack_percentile_normalize, {
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

    # 4. Dictionary pattern (component-specific processing)
    step = FunctionStep(
        func={
            '1': [(count_cells_single_channel, {
                'min_sigma': 1.0,
                'max_sigma': 10.0,
                'detection_method': DetectionMethod.WATERSHED
            })],
            '2': [(skan_axon_skeletonize_and_analyze, {
                'voxel_spacing': (1.0, 1.0, 1.0),
                'min_branch_length': 10.0,
                'analysis_dimension': AnalysisDimension.TWO_D
            })]
        },
        name="channel_specific_analysis",
        variable_components=[VariableComponents.SITE],
        force_disk_output=False
    )

.. _function-when-to-use:

When to Use Each Pattern
------------------------

**Single Function Pattern**:
- Simple operations with minimal parameters
- Creating composite images, basic transformations
- When you need one function applied uniformly

**Function Chain Pattern**:
- Multi-step preprocessing workflows
- When operations must be applied in sequence
- Common pattern: normalize → filter → enhance

**Dictionary Pattern**:
- Channel-specific or component-specific processing
- Different analysis methods for different channels
- When processing logic varies by microscopy component

**Best Practices**:

    # Use meaningful step names for debugging and monitoring, especially for FunctionSteps.
    # Example:
    # step = FunctionStep(
    #    func=[(my_processing_function, {})],
    #    name="descriptive_step_name",
    #    variable_components=[VariableComponents.SITE],
    #    force_disk_output=False # Use memory backend for intermediate steps
    # )

**When to use each function pattern:**

1. **Single Function**: Use for simple operations that don't require arguments
2. **Function with Arguments**: Use when you need to customize function behavior with parameters
3. **List of Functions**: Use when you need to apply multiple processing steps in sequence
4. **Dictionary of Functions**: Use for component-specific processing (e.g., different functions for different channels)

**Key Guidelines:**

- For Z-stack flattening, implement with a :py:class:`~openhcs.core.steps.function_step.FunctionStep` using a suitable function that operates on the 'z_index' component.
- For channel compositing, implement with a :py:class:`~openhcs.core.steps.function_step.FunctionStep` using a suitable function that operates on the 'channel' component.
- For focus detection, implement with a :py:class:`~openhcs.core.steps.function_step.FunctionStep` using a suitable function that performs focus detection.
- For channel-specific processing, use a dictionary of functions with ``group_by='channel'``
- For custom processing chains, use lists of functions

For detailed information about pre-defined steps, see :ref:`variable-components` in :doc:`step`.

.. _function-stack-utility:

Memory Type Integration
-----------------------

OpenHCS automatically handles memory type conversion between different computational backends:

.. code-block:: python

    # Functions can use different memory types - OpenHCS handles conversion
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize  # PyTorch
    from openhcs.processing.backends.processors.cupy_processor import tophat  # CuPy
    from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel  # NumPy

    # Chain functions with different memory types - automatic conversion
    step = FunctionStep(
        func=[
            stack_percentile_normalize,  # PyTorch function
            tophat                       # CuPy function
        ],
        name="mixed_backend_processing",
        variable_components=[VariableComponents.SITE]
    )

**Automatic Conversion**: OpenHCS automatically converts between NumPy, CuPy, PyTorch, JAX, and pyclesperanto arrays based on function requirements.

.. _function-advanced-patterns:

Advanced Patterns
-----------------

**Complex Dictionary Patterns**:

.. code-block:: python

    # Multi-function chains per component
    step = FunctionStep(
        func={
            '1': [
                (normalize_function, {}),
                (analysis_function_1, {})
            ],
            '2': [
                (normalize_function, {}),
                (analysis_function_2, {})
            ]
        },
        name="complex_component_processing",
        variable_components=[VariableComponents.SITE]
    )

**GPU Resource Management**: OpenHCS automatically manages GPU memory and device assignment for optimal performance.

.. _function-best-practices:

Best Practices from TUI-Generated Scripts
-----------------------------------------

- **Use descriptive step names** for pipeline debugging and monitoring
- **Set force_disk_output=False** for intermediate steps to use memory backend
- **Use appropriate variable_components** (SITE for parallel processing, CHANNEL for channel-specific operations)
- **Chain related operations** in single steps to minimize I/O overhead
- **Use dictionary patterns** when different components need different processing logic

For comprehensive best practices, see :doc:`../user_guide/best_practices`.

See Also
--------

**Technical Deep Dive**:

- :doc:`../architecture/function_pattern_system` - Complete technical documentation of function patterns
- :doc:`../architecture/memory_type_system` - Memory type decorators and automatic conversion

**API Reference**:

- :doc:`../api/function_step` - FunctionStep class documentation
- :doc:`../api/processing_backends` - Available processing functions

**Integration Guides**:

- :doc:`../guides/memory_type_integration` - Memory type system integration
- :doc:`../guides/pipeline_compilation_workflow` - How function patterns are compiled

Function Step
=============

.. module:: openhcs.core.steps.function_step

The FunctionStep class is the core building block for OpenHCS pipelines. It wraps functions with metadata and execution context, enabling GPU-accelerated bioimage analysis workflows.

FunctionStep Class
------------------

.. autoclass:: FunctionStep
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Key Features
------------

**Memory Type Contracts**: FunctionStep automatically handles conversion between NumPy, CuPy, PyTorch, TensorFlow, and JAX arrays based on function decorators.

**Variable Components**: Support for processing across different dimensions (sites, channels, z-stacks) with intelligent batching.

**Special I/O**: Cross-step communication for analysis results like cell counts and measurements.

**GPU Optimization**: Automatic GPU resource allocation and memory management.

Usage Examples
--------------

Basic Function Step
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.processing.backends.processors.torch_processor import gaussian_filter_torch

    step = FunctionStep(
        func=gaussian_filter_torch,
        sigma=2.0,
        name="gaussian_blur"
    )

Variable Components
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.constants.constants import VariableComponents

    # Process each channel separately
    step = FunctionStep(
        func=normalize_channel,
        variable_components=[VariableComponents.CHANNEL],
        percentile_low=1.0,
        percentile_high=99.0
    )

Dict Patterns
^^^^^^^^^^^^^

.. code-block:: python

    # Different functions for different channels
    step = FunctionStep(
        func={
            '1': count_cells_dapi,      # DAPI channel
            '2': trace_neurites_gfp     # GFP channel
        },
        variable_components=[VariableComponents.CHANNEL]
    )

Special Outputs
^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.core.pipeline.function_contracts import special_outputs

    @special_outputs("cell_counts", "measurements")
    def analyze_cells(image):
        # Analysis logic here
        return processed_image, cell_count_data, measurement_data

    step = FunctionStep(func=analyze_cells)

See Also
--------

- :doc:`../architecture/function_pattern_system` - Detailed function pattern documentation
- :doc:`../architecture/memory_type_system` - Memory type conversion system
- :doc:`../architecture/special_io_system` - Special I/O system documentation

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
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize

    step = FunctionStep(
        func=stack_percentile_normalize,
        low_percentile=1.0,
        high_percentile=99.0,
        name="normalize"
    )

Variable Components
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.constants.constants import VariableComponents

    # Process each site separately
    step = FunctionStep(
        func=stack_percentile_normalize,
        variable_components=[VariableComponents.SITE],
        low_percentile=1.0,
        high_percentile=99.0
    )

Function Chains
^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.processing.backends.processors.cupy_processor import tophat

    # Chain multiple functions together
    step = FunctionStep(
        func=[
            (stack_percentile_normalize, {'low_percentile': 1.0, 'high_percentile': 99.0}),
            (tophat, {'selem_radius': 50})
        ],
        name="preprocess"
    )

Dict Patterns
^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel
    from openhcs.processing.backends.analysis.skan_axon_analysis import skan_axon_skeletonize_and_analyze

    # Different functions for different channels
    step = FunctionStep(
        func={
            '1': count_cells_single_channel,      # DAPI channel
            '2': skan_axon_skeletonize_and_analyze # GFP channel
        },
        variable_components=[VariableComponents.CHANNEL]
    )

Parameters
----------

func : callable, tuple, list, or dict
    The function(s) to execute. Can be:

    - Single function: ``my_function``
    - Function with parameters: ``(my_function, {'param': value})``
    - Function chain: ``[(func1, params1), (func2, params2)]``
    - Dict pattern: ``{'1': func1, '2': func2}`` for different channels/components

name : str, optional
    Human-readable name for the step. Defaults to function name.

variable_components : list of VariableComponents, optional
    Dimensions to process separately. Default: ``[VariableComponents.SITE]``

    - ``VariableComponents.SITE``: Process each imaging site separately
    - ``VariableComponents.CHANNEL``: Process each channel separately
    - ``VariableComponents.Z``: Process each z-slice separately

group_by : GroupBy, optional
    How to group processing. Default: ``GroupBy.CHANNEL``

force_disk_output : bool, optional
    Force output to disk backend. Default: ``False``

input_dir : str or Path, optional
    Override input directory for this step.

output_dir : str or Path, optional
    Override output directory for this step.

See Also
--------

- :doc:`../architecture/function_pattern_system` - Detailed function pattern documentation
- :doc:`../architecture/memory_type_system` - Memory type conversion system
- :doc:`../architecture/special_io_system` - Special I/O system documentation
- :doc:`../user_guide/production_examples` - Real-world usage examples

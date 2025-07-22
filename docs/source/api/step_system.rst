Step System
===========

.. module:: openhcs.core.steps

OpenHCS implements a **hierarchical step system** for building GPU-accelerated bioimage analysis pipelines. The system consists of an abstract base class and concrete implementations optimized for different processing patterns.

Step Architecture
-----------------

OpenHCS uses a **two-tier step architecture**:

**AbstractStep** (Base Class)
    Defines the core step interface and lifecycle management

    - **Stateful During Compilation**: Holds configuration during pipeline definition
    - **Stateless During Execution**: Configuration stripped after compilation for performance
    - **Context-Based Execution**: All runtime data comes from frozen ProcessingContext

**FunctionStep** (Primary Implementation)
    Wraps functions with execution metadata for pattern-based processing

    - **Function Wrapping**: Converts any callable into a pipeline step
    - **Pattern Support**: Single functions, chains, and dictionaries
    - **GPU Integration**: Automatic memory type conversion and GPU scheduling
    - **Special I/O**: Cross-step communication through special inputs/outputs

AbstractStep
------------

.. autoclass:: AbstractStep
   :members:
   :undoc-members:
   :show-inheritance:

The base class for all OpenHCS pipeline steps. Provides core functionality for:

- **Step Identification**: Unique ID generation for stateless execution
- **Lifecycle Management**: Compilation and execution phase separation
- **Context Integration**: Interface with ProcessingContext and VFS
- **Validation**: Abstract methods for step-specific validation

**Core Parameters**:

:param name: Human-readable step identifier
:type name: str, optional
:param variable_components: Components that vary across processing patterns
:type variable_components: List[VariableComponents], optional
:param force_disk_output: Force filesystem output regardless of backend
:type force_disk_output: bool, optional
:param group_by: Grouping strategy for file processing
:type group_by: GroupBy, optional
:param input_dir: Input directory hint for path planning
:type input_dir: str or Path, optional
:param output_dir: Output directory hint for path planning
:type output_dir: str or Path, optional

FunctionStep
------------

.. autoclass:: FunctionStep
   :members:
   :undoc-members:
   :show-inheritance:

The primary step implementation for OpenHCS pipelines. FunctionStep wraps any callable (function, method, or lambda) with execution metadata, enabling GPU-accelerated processing with automatic memory management.

**Core Parameters**:

:param func: The processing function(s) to execute
:type func: Union[Callable, Tuple[Callable, Dict], List[...], Dict[str, ...]]
:param name: Human-readable step name (defaults to function name)
:type name: str, optional
:param variable_components: Components that vary across processing patterns
:type variable_components: List[VariableComponents], default=[VariableComponents.SITE]
:param group_by: File grouping strategy for processing
:type group_by: GroupBy, default=GroupBy.CHANNEL
:param force_disk_output: Force filesystem output for this step
:type force_disk_output: bool, default=False
:param input_dir: Input directory hint for path planning
:type input_dir: str or Path, optional
:param output_dir: Output directory hint for path planning
:type output_dir: str or Path, optional

Function Patterns
------------------

FunctionStep supports **four distinct function patterns** for different processing needs:

**Single Function Pattern**

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize

    # Simple function call
    step = FunctionStep(
        func=stack_percentile_normalize,
        name="normalize"
    )

**Function with Parameters Pattern**

.. code-block:: python

    # Function with specific parameters
    step = FunctionStep(
        func=[(stack_percentile_normalize, {
            'low_percentile': 1.0,
            'high_percentile': 99.0,
            'target_max': 65535.0
        })],
        name="normalize"
    )

**Chain Pattern (List)**

.. code-block:: python

    from openhcs.processing.backends.processors.cupy_processor import tophat, create_composite

    # Sequential function execution
    step = FunctionStep(
        func=[
            (tophat, {'selem_radius': 50}),
            create_composite
        ],
        name="preprocess_and_composite"
    )

**Dictionary Pattern**

.. code-block:: python

    from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel
    from openhcs.processing.backends.analysis.skan_axon_analysis import skan_axon_skeletonize_and_analyze

    # Different functions for different channels
    step = FunctionStep(
        func={
            '1': count_cells_single_channel,      # DAPI channel
            '2': skan_axon_skeletonize_and_analyze # GFP channel
        },
        variable_components=[VariableComponents.CHANNEL],
        group_by=GroupBy.CHANNEL
    )

Variable Components and Grouping
--------------------------------

**Variable Components** define which file components vary during processing:

.. code-block:: python

    from openhcs.constants.constants import VariableComponents, GroupBy

    # Process each site separately
    step = FunctionStep(
        func=stack_percentile_normalize,
        variable_components=[VariableComponents.SITE],
        group_by=GroupBy.CHANNEL
    )

    # Process each channel separately
    step = FunctionStep(
        func=create_composite,
        variable_components=[VariableComponents.CHANNEL],
        group_by=GroupBy.CHANNEL
    )

Production Pipeline Example
---------------------------

Complete example from gold standard script showing real-world step usage:

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.constants.constants import VariableComponents

    # Import processing functions
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.processing.backends.processors.cupy_processor import tophat, create_composite
    from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel
    from openhcs.processing.backends.analysis.skan_axon_analysis import skan_axon_skeletonize_and_analyze
    from openhcs.processing.backends.pos_gen.ashlar_main_gpu import ashlar_compute_tile_positions_gpu
    from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy

    # Complete neurite analysis pipeline
    pipeline_steps = [
        # Step 1: Normalize images (PyTorch GPU)
        FunctionStep(
            func=[(stack_percentile_normalize, {
                'low_percentile': 1.0,
                'high_percentile': 99.0,
                'target_max': 65535.0
            })],
            name="normalize",
            variable_components=[VariableComponents.SITE],
            force_disk_output=False
        ),

        # Step 2: Morphological preprocessing (CuPy GPU)
        FunctionStep(
            func=[(tophat, {'selem_radius': 50})],
            name="preprocess",
            variable_components=[VariableComponents.SITE],
            force_disk_output=False
        ),

        # Step 3: Create composite images (CuPy GPU)
        FunctionStep(
            func=[create_composite],
            name="composite",
            variable_components=[VariableComponents.CHANNEL],
            force_disk_output=False
        ),

        # Step 4: Cell counting analysis
        FunctionStep(
            func=count_cells_single_channel,
            name="cell_count",
            variable_components=[VariableComponents.SITE],
            force_disk_output=False
        ),

        # Step 5: Neurite tracing analysis
        FunctionStep(
            func=skan_axon_skeletonize_and_analyze,
            name="neurite_trace",
            variable_components=[VariableComponents.SITE],
            force_disk_output=False
        ),

        # Step 6: GPU stitching positions
        FunctionStep(
            func=[(ashlar_compute_tile_positions_gpu, {
                'overlap_ratio': 0.1,
                'max_shift': 15.0,
                'stitch_alpha': 0.2
            })],
            name="positions",
            variable_components=[VariableComponents.CHANNEL],
            force_disk_output=True
        ),

        # Step 7: GPU image assembly
        FunctionStep(
            func=[(assemble_stack_cupy, {
                'blend_method': "fixed",
                'fixed_margin_ratio': 0.1
            })],
            name="assemble",
            variable_components=[VariableComponents.CHANNEL],
            force_disk_output=True
        )
    ]

Step Lifecycle
--------------

OpenHCS steps follow a **three-phase lifecycle**:

**Phase 1: Definition** (Stateful)
    Steps are created with configuration attributes

    .. code-block:: python

        step = FunctionStep(func=my_function, name="my_step")
        # step.name, step.func, etc. are accessible

**Phase 2: Compilation** (Attribute Stripping)
    Configuration is extracted and steps become stateless shells

    .. code-block:: python

        # After compilation by PipelineOrchestrator
        # step.name, step.func, etc. are stripped for performance

**Phase 3: Execution** (Stateless)
    Steps operate purely from ProcessingContext

    .. code-block:: python

        def process(self, context):
            step_id = get_step_id(self)
            step_plan = context.step_plans[step_id]
            # All configuration comes from step_plan

GPU Integration
---------------

FunctionStep provides **automatic GPU integration**:

**Memory Type Conversion**:
- Automatic conversion between NumPy, CuPy, PyTorch, JAX, TensorFlow
- Functions decorated with memory type decorators handle conversion
- Input/output arrays automatically converted to appropriate types

**GPU Scheduling**:
- Automatic GPU device assignment based on availability
- Memory management prevents CUDA out-of-memory errors
- Parallel execution across multiple GPUs when available

**Backend Selection**:
- Functions can specify preferred computational backend
- Automatic fallback to CPU when GPU unavailable
- Performance optimization based on data size and complexity

See Also
--------

- :doc:`function_step` - Detailed FunctionStep API reference
- :doc:`../concepts/step` - Step concepts and patterns
- :doc:`../concepts/function_handling` - Function pattern details
- :doc:`../architecture/memory_type_system` - GPU memory management
- :doc:`../architecture/function_pattern_system` - Pattern system architecture
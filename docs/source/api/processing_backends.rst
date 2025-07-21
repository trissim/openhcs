Processing Backends
==================

.. module:: openhcs.processing.backends

OpenHCS processing backends provide GPU-accelerated bioimage analysis functions organized by computational backend and functionality. All functions use memory type decorators for automatic GPU memory conversion.

Backend Organization
--------------------

Processing backends are organized into functional categories:

**Processors** (``openhcs.processing.backends.processors``)
    Basic image processing operations: filtering, normalization, morphology

**Analysis** (``openhcs.processing.backends.analysis``) 
    Bioimage analysis: cell counting, neurite tracing, segmentation

**Assemblers** (``openhcs.processing.backends.assemblers``)
    Image assembly: stitching, tile positioning, blending

**Position Generation** (``openhcs.processing.backends.pos_gen``)
    Tile position calculation: Ashlar, MIST algorithms

**Enhancement** (``openhcs.processing.backends.enhance``)
    Image enhancement: denoising, deconvolution, flatfield correction

Memory Type Decorators
----------------------

All processing functions use memory type decorators for automatic conversion:

.. code-block:: python

    from openhcs.core.memory.decorators import numpy, cupy, torch, pyclesperanto

    @numpy
    def process_cpu(image): 
        # NumPy implementation
        pass

    @cupy  
    def process_gpu_cupy(image):
        # CuPy GPU implementation
        pass

    @torch
    def process_gpu_torch(image):
        # PyTorch GPU implementation  
        pass

    @pyclesperanto
    def process_gpu_cle(image):
        # pyclesperanto GPU implementation
        pass

Key Processing Functions
------------------------

Image Processors
^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.processing.backends.processors.cupy_processor import tophat, create_composite

    # PyTorch normalization
    step = FunctionStep(
        func=stack_percentile_normalize,
        low_percentile=1.0,
        high_percentile=99.0,
        target_max=65535.0
    )

    # CuPy morphological processing
    step = FunctionStep(
        func=tophat,
        selem_radius=50,
        downsample_factor=4
    )

Analysis Functions
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel
    from openhcs.processing.backends.analysis.skan_axon_analysis import skan_axon_skeletonize_and_analyze

    # Cell counting with special outputs
    step = FunctionStep(
        func=count_cells_single_channel,
        detection_method=DetectionMethod.BLOB_LOG,
        min_sigma=1.0,
        max_sigma=10.0,
        threshold=0.1
    )

    # Neurite tracing analysis
    step = FunctionStep(
        func=skan_axon_skeletonize_and_analyze,
        analysis_dimension=AnalysisDimension.TWO_D,
        min_branch_length=10.0,
        summarize=True
    )

Assembly Functions
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy
    from openhcs.processing.backends.pos_gen.ashlar_main_gpu import ashlar_compute_tile_positions_gpu

    # GPU tile position calculation
    step = FunctionStep(
        func=ashlar_compute_tile_positions_gpu,
        overlap_ratio=0.1,
        max_shift=15.0,
        stitch_alpha=0.2
    )

    # GPU image assembly
    step = FunctionStep(
        func=assemble_stack_cupy,
        blend_method="fixed",
        fixed_margin_ratio=0.1
    )

Function Registry Integration
-----------------------------

All processing functions are automatically discovered and registered:

.. code-block:: python

    from openhcs.processing import (
        FUNC_REGISTRY,
        get_functions_by_memory_type,
        get_function_by_name
    )

    # Get all CuPy functions
    cupy_functions = get_functions_by_memory_type('cupy')

    # Get specific function info
    func_info = get_function_by_name('tophat')

Special I/O Integration
-----------------------

Analysis functions support special I/O for cross-step communication:

.. code-block:: python

    from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs

    @special_outputs("cell_counts", "measurements")
    @numpy
    def analyze_cells(image):
        # Analysis produces special outputs
        return processed_image, cell_data, measurements

    @special_inputs("positions")
    @cupy
    def assemble_images(tiles, positions):
        # Assembly uses positions from previous step
        return assembled_image

Available Backends
------------------

**NumPy Backend**: CPU processing with NumPy arrays
**CuPy Backend**: GPU processing with CUDA acceleration  
**PyTorch Backend**: GPU processing with PyTorch tensors
**TensorFlow Backend**: GPU processing with TensorFlow tensors
**JAX Backend**: GPU processing with JAX arrays
**pyclesperanto Backend**: GPU processing with OpenCL acceleration

See Also
--------

- :doc:`../architecture/function_registry_system` - Function discovery and registration
- :doc:`../architecture/memory_type_system` - Automatic memory conversion
- :doc:`../architecture/special_io_system` - Cross-step communication
- :doc:`function_step` - Using functions in pipelines

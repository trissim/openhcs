Processing Backends
==================

.. module:: openhcs.processing.backends

OpenHCS provides **574+ processing functions** organized into functional categories with automatic GPU acceleration and memory type conversion. Functions come from three sources: **native OpenHCS implementations**, **auto-registered external libraries**, and **specialized research algorithms**.

Function Organization - The Forest View
----------------------------------------

OpenHCS organizes processing functions into **5 main functional categories**, each containing multiple computational backends:

**🔧 Processors** - Basic Image Processing
    Core image processing operations implemented across 6 backends (NumPy, CuPy, PyTorch, TensorFlow, JAX, pyclesperanto)

    - **Filtering & Normalization**: Gaussian filters, percentile normalization, contrast enhancement
    - **Morphological Operations**: Top-hat filtering, opening/closing, erosion/dilation
    - **Composite Creation**: Multi-channel image composition and blending
    - **Stack Operations**: 3D stack processing, slice-wise operations

**🔬 Analysis** - Bioimage Analysis
    Specialized analysis algorithms for microscopy data with research-grade implementations

    - **Cell Counting**: Blob detection, watershed segmentation, machine learning approaches
    - **Neurite Tracing**: HMM-based tracing, skeleton analysis, morphometry
    - **Segmentation**: Instance segmentation, semantic segmentation, 3D analysis
    - **Feature Extraction**: Shape analysis, intensity measurements, spatial statistics

**🧩 Assemblers** - Image Assembly
    High-performance image stitching and assembly with GPU acceleration

    - **Tile Assembly**: Subpixel positioning, blending algorithms, overlap handling
    - **Stack Assembly**: 3D volume reconstruction, multi-channel assembly
    - **GPU Optimization**: CuPy-accelerated assembly for large datasets

**📍 Position Generation** - Stitching Algorithms
    State-of-the-art position calculation algorithms for microscopy stitching

    - **Ashlar Algorithm**: GPU-accelerated edge alignment, phase correlation
    - **MIST Algorithm**: Feature-based registration, robust outlier handling
    - **Custom Algorithms**: Research-specific positioning methods

**✨ Enhancement** - Advanced Image Enhancement
    Cutting-edge enhancement algorithms for microscopy image quality improvement

    - **Flatfield Correction**: BaSiC algorithm implementation (NumPy/CuPy)
    - **Denoising**: N2V2 self-supervised denoising (PyTorch)
    - **Deconvolution**: Self-supervised 2D/3D deconvolution algorithms
    - **Restoration**: Advanced restoration techniques for microscopy artifacts

Three-Tier Function System
--------------------------

OpenHCS functions come from **three distinct sources**, each with different characteristics:

**Tier 1: Native OpenHCS Functions**
    Hand-crafted implementations optimized for microscopy workflows

    - **Direct Import**: ``from openhcs.processing.backends.processors.cupy_processor import tophat``
    - **Explicit Backends**: Separate implementations for each computational backend
    - **Research Optimized**: Designed specifically for high-content screening workflows
    - **Examples**: ``stack_percentile_normalize``, ``create_composite``, ``count_cells_single_channel``

**Tier 2: Auto-Registered External Libraries**
    Popular libraries automatically integrated with OpenHCS memory system

    - **pyclesperanto**: 200+ GPU-accelerated OpenCL functions auto-registered
    - **scikit-image**: 300+ CPU functions with automatic 3D behavior analysis
    - **Automatic Decoration**: Functions gain OpenHCS memory type attributes on import
    - **Contract Classification**: Functions classified as SLICE_SAFE, CROSS_Z, or DIM_CHANGE

**Tier 3: Specialized Research Algorithms**
    Cutting-edge algorithms from research publications and collaborations

    - **HMM Neurite Tracing**: Advanced neurite tracing using Hidden Markov Models
    - **Ashlar GPU Stitching**: Complete GPU port of Ashlar stitching algorithm
    - **N2V2 Denoising**: Self-supervised denoising with Noise2Void v2
    - **BaSiC Flatfield**: Flatfield correction using BaSiC algorithm

Function Discovery and Registration
-----------------------------------

OpenHCS uses a **two-phase automatic registration system**:

**Phase 1: Native Function Scanning**
    Recursively scans ``openhcs.processing.backends`` for decorated functions

    .. code-block:: python

        # Functions are automatically discovered by their decorators
        @cupy_func
        def tophat(image, selem_radius=10):
            # Implementation automatically registered
            pass

**Phase 2: External Library Integration**
    Auto-registers external libraries with contract analysis

    .. code-block:: python

        # pyclesperanto functions automatically decorated on import
        import pyclesperanto as cle
        # 200+ functions now available as OpenHCS functions

        # scikit-image functions analyzed and registered
        from skimage import filters
        # Compatible functions automatically decorated

Memory Type System Integration
------------------------------

All functions integrate with OpenHCS memory type system for automatic GPU acceleration:

.. code-block:: python

    from openhcs.core.memory.decorators import numpy, cupy, torch, pyclesperanto

    @numpy
    def process_cpu(image):
        # NumPy implementation - automatic CPU processing
        pass

    @cupy
    def process_gpu_cupy(image):
        # CuPy GPU implementation - automatic GPU memory management
        pass

    @torch
    def process_gpu_torch(image):
        # PyTorch GPU implementation - automatic tensor conversion
        pass

    @pyclesperanto
    def process_gpu_opencl(image):
        # pyclesperanto OpenCL implementation - automatic OpenCL memory
        pass

    @pyclesperanto
    def process_gpu_cle(image):
        # pyclesperanto GPU implementation
        pass

Forest-Level Usage Patterns
---------------------------

**Complete Neurite Analysis Pipeline** (Gold Standard Example)

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.constants.constants import VariableComponents

    # Tier 1: Native OpenHCS Functions
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.processing.backends.processors.cupy_processor import tophat, create_composite
    from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel

    # Tier 3: Specialized Research Algorithms
    from openhcs.processing.backends.analysis.skan_axon_analysis import skan_axon_skeletonize_and_analyze
    from openhcs.processing.backends.pos_gen.ashlar_main_gpu import ashlar_compute_tile_positions_gpu
    from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy

    # Complete processing pipeline using all three tiers
    pipeline_steps = [
        # Basic Processing (Tier 1)
        FunctionStep(
            func=[(stack_percentile_normalize, {
                'low_percentile': 1.0, 'high_percentile': 99.0, 'target_max': 65535.0
            })],
            name="normalize", variable_components=[VariableComponents.SITE]
        ),

        # Morphological Processing (Tier 1)
        FunctionStep(
            func=[(tophat, {'selem_radius': 50})],
            name="preprocess", variable_components=[VariableComponents.SITE]
        ),

        # Composite Creation (Tier 1)
        FunctionStep(
            func=[create_composite],
            name="composite", variable_components=[VariableComponents.CHANNEL]
        ),

        # Cell Analysis (Tier 1)
        FunctionStep(
            func=count_cells_single_channel,
            name="cell_count", variable_components=[VariableComponents.SITE]
        ),

        # Advanced Neurite Tracing (Tier 3)
        FunctionStep(
            func=skan_axon_skeletonize_and_analyze,
            name="neurite_trace", variable_components=[VariableComponents.SITE]
        )
    ]

**Auto-Registered External Functions** (Tier 2)

.. code-block:: python

    # pyclesperanto functions (200+ available)
    import pyclesperanto as cle

    # Functions automatically become OpenHCS-compatible
    step = FunctionStep(
        func=cle.gaussian_blur,  # Auto-registered pyclesperanto function
        sigma_x=2.0, sigma_y=2.0
    )

    # scikit-image functions (300+ available)
    from skimage import filters

    step = FunctionStep(
        func=filters.gaussian,  # Auto-registered scikit-image function
        sigma=1.5
    )

**Function Categories by Computational Backend**

.. code-block:: python

    # Processors: 6 backends × ~20 functions = 120+ functions
    from openhcs.processing.backends.processors import (
        numpy_processor,     # CPU processing
        cupy_processor,      # CUDA GPU processing
        torch_processor,     # PyTorch GPU processing
        tensorflow_processor, # TensorFlow GPU processing
        jax_processor,       # JAX GPU processing
        pyclesperanto_processor  # OpenCL GPU processing
    )

    # Analysis: Specialized algorithms (~50 functions)
    from openhcs.processing.backends.analysis import (
        cell_counting_cpu,           # Cell detection algorithms
        skan_axon_analysis,         # Neurite tracing
        hmm_axon,                   # HMM-based tracing
        pyclesperanto_registry      # 200+ auto-registered functions
    )

    # Enhancement: Advanced algorithms (~20 functions)
    from openhcs.processing.backends.enhance import (
        basic_processor_numpy,       # BaSiC flatfield correction
        basic_processor_cupy,        # GPU BaSiC implementation
        n2v2_processor_torch,        # N2V2 denoising
        self_supervised_3d_deconvolution  # 3D deconvolution
    )

    # Position Generation: Stitching algorithms (~10 functions)
    from openhcs.processing.backends.pos_gen import (
        ashlar_main_gpu,            # GPU Ashlar algorithm
        mist_processor_cupy,        # MIST algorithm
        ashlar_processor_cupy       # Alternative Ashlar implementation
    )

    # Assemblers: Image assembly (~5 functions)
    from openhcs.processing.backends.assemblers import (
        assemble_stack_cupy,        # GPU assembly
        assemble_stack_cpu          # CPU assembly
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

Function Registry Statistics
---------------------------

OpenHCS automatically discovers and registers **574+ functions** across all categories:

**Native OpenHCS Functions**: ~150 functions
    - Processors: 120+ functions (6 backends × 20 functions)
    - Analysis: 20+ specialized algorithms
    - Enhancement: 10+ advanced algorithms
    - Position Generation: 5+ stitching algorithms
    - Assemblers: 5+ assembly functions

**Auto-Registered External Libraries**: ~400+ functions
    - pyclesperanto: 200+ GPU-accelerated OpenCL functions
    - scikit-image: 200+ CPU functions with 3D analysis

**Specialized Research Algorithms**: ~24 functions
    - HMM neurite tracing algorithms
    - Advanced stitching implementations
    - Self-supervised enhancement methods
    - Custom bioimage analysis tools

Special I/O Integration
-----------------------

Advanced functions support cross-step communication through special I/O:

.. code-block:: python

    from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs

    @special_outputs("cell_counts", "measurements")
    @numpy
    def analyze_cells(image):
        # Analysis produces special outputs for downstream steps
        return processed_image, cell_data, measurements

    @special_inputs("positions")
    @cupy
    def assemble_images(tiles, positions):
        # Assembly uses positions from previous step
        return assembled_image

Computational Backend Summary
-----------------------------

**CPU Backends**:
- **NumPy**: Universal CPU processing, maximum compatibility
- **scikit-image**: 200+ auto-registered analysis functions

**GPU Backends**:
- **CuPy**: CUDA GPU acceleration, excellent NumPy compatibility
- **PyTorch**: Deep learning integration, tensor operations
- **TensorFlow**: Machine learning workflows, distributed processing
- **JAX**: High-performance computing, automatic differentiation
- **pyclesperanto**: OpenCL acceleration, 200+ auto-registered functions

Function Discovery
------------------

Functions are automatically discovered through:

1. **Module Scanning**: Recursive import of processing backends
2. **Decorator Detection**: Functions with memory type decorators
3. **External Registration**: Auto-registration of compatible libraries
4. **Contract Analysis**: Automatic 3D behavior classification

See Also
--------

- :doc:`../architecture/function_registry_system` - Complete registry system documentation
- :doc:`../architecture/memory_type_system` - Automatic memory conversion details
- :doc:`../architecture/special_io_system` - Cross-step communication patterns
- :doc:`function_step` - Using functions in pipelines
- :doc:`../concepts/function_handling` - Function pattern concepts

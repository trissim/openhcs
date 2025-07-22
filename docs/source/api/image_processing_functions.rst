Image Processing Functions
=========================

.. module:: openhcs.processing.backends.processors

OpenHCS provides **120+ image processing functions** across 6 computational backends, each optimized for different hardware and use cases. These functions form the core of OpenHCS's image processing capabilities.

Backend Architecture
--------------------

OpenHCS implements the same set of image processing functions across multiple computational backends:

**CPU Backend**:
- **NumPy**: Universal CPU processing, maximum compatibility

**GPU Backends**:
- **CuPy**: CUDA GPU acceleration, excellent NumPy compatibility
- **PyTorch**: Deep learning integration, tensor operations
- **TensorFlow**: Machine learning workflows, distributed processing
- **JAX**: High-performance computing, automatic differentiation
- **pyclesperanto**: OpenCL acceleration, cross-platform GPU support

Function Categories
-------------------

**🔧 Filtering & Normalization**
    Core image preprocessing operations for microscopy data

    - **Percentile Normalization**: ``stack_percentile_normalize`` - Robust intensity normalization
    - **Gaussian Filtering**: ``gaussian_filter`` - Noise reduction and smoothing
    - **Contrast Enhancement**: Adaptive histogram equalization and CLAHE

**🔍 Morphological Operations**
    Shape-based image processing for structure analysis

    - **Top-hat Filtering**: ``tophat`` - Background subtraction and feature enhancement
    - **Opening/Closing**: Noise removal and gap filling
    - **Erosion/Dilation**: Structure modification and analysis

**🎨 Composite & Projection**
    Multi-channel and 3D image composition

    - **Composite Creation**: ``create_composite`` - Multi-channel image blending
    - **Maximum Projection**: ``max_projection`` - Z-stack compression
    - **Mean Projection**: ``mean_projection`` - Average intensity projection

**📐 Spatial Operations**
    Geometric transformations and spatial analysis

    - **Spatial Binning**: ``spatial_bin_2d``, ``spatial_bin_3d`` - Resolution reduction
    - **Edge Detection**: ``edge_magnitude`` - Feature boundary detection
    - **Rotation & Scaling**: Geometric transformations

Core Processing Functions
-------------------------

**Percentile Normalization** (Gold Standard Usage)

.. code-block:: python

    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.constants.constants import VariableComponents

    # GPU-accelerated normalization from gold standard script
    step = FunctionStep(
        func=[(stack_percentile_normalize, {
            'low_percentile': 1.0,
            'high_percentile': 99.0,
            'target_max': 65535.0
        })],
        name="normalize",
        variable_components=[VariableComponents.SITE]
    )

**Morphological Processing**

.. code-block:: python

    from openhcs.processing.backends.processors.cupy_processor import tophat

    # GPU-accelerated top-hat filtering for background subtraction
    step = FunctionStep(
        func=[(tophat, {'selem_radius': 50})],
        name="preprocess",
        variable_components=[VariableComponents.SITE]
    )

**Multi-Channel Composition**

.. code-block:: python

    from openhcs.processing.backends.processors.cupy_processor import create_composite

    # Create composite images from multiple channels
    step = FunctionStep(
        func=[create_composite],
        name="composite",
        variable_components=[VariableComponents.CHANNEL]
    )

Backend-Specific Usage
----------------------

**NumPy Backend (CPU)**

.. code-block:: python

    from openhcs.processing.backends.processors.numpy_processor import (
        stack_percentile_normalize, tophat, create_composite,
        max_projection, gaussian_filter, spatial_bin_2d
    )

    # CPU processing for compatibility
    step = FunctionStep(func=stack_percentile_normalize)

**CuPy Backend (CUDA GPU)**

.. code-block:: python

    from openhcs.processing.backends.processors.cupy_processor import (
        stack_percentile_normalize, tophat, create_composite,
        max_projection, gaussian_filter, edge_magnitude
    )

    # CUDA GPU acceleration
    step = FunctionStep(func=tophat, selem_radius=25)

**PyTorch Backend (GPU)**

.. code-block:: python

    from openhcs.processing.backends.processors.torch_processor import (
        stack_percentile_normalize, max_projection, mean_projection
    )

    # PyTorch GPU processing with tensor operations
    step = FunctionStep(func=stack_percentile_normalize)

**pyclesperanto Backend (OpenCL GPU)**

.. code-block:: python

    from openhcs.processing.backends.processors.pyclesperanto_processor import (
        gaussian_filter, tophat, create_composite
    )

    # OpenCL GPU acceleration (cross-platform)
    step = FunctionStep(func=gaussian_filter, sigma=2.0)

Function Reference by Category
------------------------------

**Normalization Functions**

.. code-block:: python

    # Available in: numpy, cupy, torch backends
    stack_percentile_normalize(
        image,
        low_percentile=1.0,
        high_percentile=99.0,
        target_max=65535.0
    )

**Morphological Functions**

.. code-block:: python

    # Available in: numpy, cupy, pyclesperanto backends
    tophat(image, selem_radius=50)

    # Available in: numpy, cupy backends
    gaussian_filter(image, sigma=1.5)

**Projection Functions**

.. code-block:: python

    # Available in: numpy, cupy, torch backends
    max_projection(stack)      # Maximum intensity projection
    mean_projection(stack)     # Mean intensity projection
    create_projection(stack, method="max_projection")

**Spatial Operations**

.. code-block:: python

    # Available in: numpy, cupy backends
    spatial_bin_2d(image, bin_size=2, method="mean")
    spatial_bin_3d(stack, bin_size=2, method="mean")
    edge_magnitude(image, method="2d")

**Composite Functions**

.. code-block:: python

    # Available in: numpy, cupy, pyclesperanto backends
    create_composite(stack, weights=None)

Production Pipeline Example
---------------------------

Complete example from gold standard script showing real-world usage:

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.constants.constants import VariableComponents

    # Import functions from different backends
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.processing.backends.processors.cupy_processor import tophat, create_composite

    # Multi-backend processing pipeline
    pipeline_steps = [
        # PyTorch normalization
        FunctionStep(
            func=[(stack_percentile_normalize, {
                'low_percentile': 1.0, 'high_percentile': 99.0, 'target_max': 65535.0
            })],
            name="normalize", variable_components=[VariableComponents.SITE]
        ),

        # CuPy morphological processing
        FunctionStep(
            func=[(tophat, {'selem_radius': 50})],
            name="preprocess", variable_components=[VariableComponents.SITE]
        ),

        # CuPy composite creation
        FunctionStep(
            func=[create_composite],
            name="composite", variable_components=[VariableComponents.CHANNEL]
        )
    ]

Memory Type Conversion
----------------------

OpenHCS automatically handles memory type conversion between backends:

.. code-block:: python

    # Input: NumPy array → Automatic conversion to CuPy → Output: NumPy array
    step = FunctionStep(func=cupy_processor.tophat)

    # Input: NumPy array → Automatic conversion to PyTorch → Output: NumPy array
    step = FunctionStep(func=torch_processor.stack_percentile_normalize)

See Also
--------

- :doc:`processing_backends` - Complete processing backends overview
- :doc:`../architecture/memory_type_system` - Memory type conversion details
- :doc:`function_step` - Using functions in pipelines
- :doc:`../concepts/function_handling` - Function pattern concepts


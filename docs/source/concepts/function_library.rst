Function Library
================

OpenHCS provides 574+ processing functions across multiple computational backends, all unified under a consistent 3D array interface. Understanding this function library is essential for building effective analysis pipelines.

The 3D Array Contract
---------------------

All OpenHCS functions follow a fundamental contract: they accept 3D arrays as input and return 3D arrays as output. This consistency enables seamless function composition and automatic memory management.

.. code-block:: python

   # All functions follow this pattern:
   # input_3d_array (Z, Y, X) → output_3d_array (Z, Y, X)
   
   def example_function(image_stack):
       """
       Args:
           image_stack: 3D array with shape (Z, Y, X)
       Returns:
           processed_stack: 3D array with shape (Z, Y, X)
       """
       return processed_stack

**Array Dimensions**:

- **Z**: Number of images in the stack (channels, Z-planes, or timepoints)
- **Y**: Image height (rows)
- **X**: Image width (columns)

**Why 3D**: Even single 2D images are represented as 3D arrays with Z=1. This consistent interface allows functions to work with single images, multi-channel data, Z-stacks, and time series without modification.

Automatic 2D Function Wrapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many image processing libraries provide 2D functions. OpenHCS automatically wraps these to work with 3D data:

.. code-block:: python

   # Original 2D function from scikit-image
   from skimage.filters import gaussian
   
   # OpenHCS automatically wraps it to process each Z-slice
   @numpy  # Memory type decorator
   def gaussian_filter_3d(image_stack, sigma=1.0):
       # Automatically applies gaussian() to each slice in the stack
       return stack_of_processed_slices

**How it works**: OpenHCS detects 2D functions and automatically applies them to each slice in the Z-dimension, then restacks the results into a 3D output.

Available Function Categories
----------------------------

Image Processing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Filtering and Enhancement**: ~150 functions

.. code-block:: python

   from openhcs.processing.backends.processors.cupy_processor import (
       gaussian_filter, tophat, enhance_contrast, edge_magnitude
   )
   
   # Noise reduction and enhancement
   step = FunctionStep(func=(gaussian_filter, {'sigma': 2.0}))
   step = FunctionStep(func=(tophat, {'selem_radius': 25}))

**Morphological Operations**: ~80 functions

.. code-block:: python

   from openhcs.processing.backends.processors.cupy_processor import (
       binary_opening, binary_closing, binary_erosion, binary_dilation
   )
   
   # Shape-based processing
   step = FunctionStep(func=(binary_opening, {'footprint_radius': 3}))

**Segmentation and Thresholding**: ~60 functions

.. code-block:: python

   from openhcs.processing.backends.processors.cupy_processor import (
       threshold_otsu, watershed, label_connected_components
   )
   
   # Object detection and segmentation
   step = FunctionStep(func=(threshold_otsu, {'binary': True}))

Analysis Functions
~~~~~~~~~~~~~~~~~

**Cell Counting and Detection**: ~40 functions

.. code-block:: python

   from openhcs.processing.backends.analysis.cell_counting_cpu import (
       count_cells_single_channel, DetectionMethod
   )
   
   # Automated cell counting
   step = FunctionStep(
       func=(count_cells_single_channel, {
           'detection_method': DetectionMethod.WATERSHED,
           'min_sigma': 1.0,
           'max_sigma': 10.0
       })
   )

**Neurite and Structure Analysis**: ~30 functions

.. code-block:: python

   from openhcs.processing.backends.analysis.skan_axon_analysis import (
       skan_axon_skeletonize_and_analyze, AnalysisDimension
   )
   
   # Neurite tracing and measurement
   step = FunctionStep(
       func=(skan_axon_skeletonize_and_analyze, {
           'analysis_dimension': AnalysisDimension.TWO_D,
           'min_branch_length': 10.0
       })
   )

**Feature Measurement**: ~50 functions

.. code-block:: python

   from openhcs.processing.backends.analysis.feature_extraction import (
       measure_intensity_features, measure_morphology_features
   )
   
   # Quantitative measurements
   step = FunctionStep(func=(measure_intensity_features, {}))

Assembly Functions
~~~~~~~~~~~~~~~~~

**Image Stitching**: ~25 functions

.. code-block:: python

   from openhcs.processing.backends.assemblers.assemble_stack_cupy import (
       assemble_stack_cupy
   )
   
   # Combine multiple images into larger field of view
   step = FunctionStep(func=(assemble_stack_cupy, {}))

**Projection and Compositing**: ~35 functions

.. code-block:: python

   from openhcs.processing.backends.processors.cupy_processor import (
       max_projection, mean_projection, create_composite
   )
   
   # Combine Z-stacks or create multi-channel composites
   step = FunctionStep(func=(max_projection, {}))

Memory Type System
------------------

Functions are organized by computational backend, each optimized for different hardware:

NumPy Backend (CPU)
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openhcs.processing.backends.processors.numpy_processor import (
       gaussian_filter, tophat, threshold_otsu
   )
   
   # CPU processing - compatible with all systems
   step = FunctionStep(func=(gaussian_filter, {'sigma': 2.0}))

**When to use**: Compatibility with all systems, small datasets, functions not available on GPU.

CuPy Backend (CUDA GPU)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openhcs.processing.backends.processors.cupy_processor import (
       gaussian_filter, tophat, threshold_otsu
   )
   
   # CUDA GPU acceleration - 10-100x faster for large images
   step = FunctionStep(func=(gaussian_filter, {'sigma': 2.0}))

**When to use**: NVIDIA GPUs, large datasets, performance-critical processing.

PyTorch Backend (GPU)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openhcs.processing.backends.processors.torch_processor import (
       stack_percentile_normalize, max_projection
   )
   
   # PyTorch GPU processing with automatic memory management
   step = FunctionStep(func=(stack_percentile_normalize, {}))

**When to use**: Deep learning integration, advanced tensor operations, automatic differentiation.

pyclesperanto Backend (OpenCL GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openhcs.processing.backends.processors.pyclesperanto_processor import (
       gaussian_filter, tophat, create_composite
   )
   
   # OpenCL GPU acceleration - works with AMD, Intel, NVIDIA GPUs
   step = FunctionStep(func=(gaussian_filter, {'sigma': 2.0}))

**When to use**: Non-NVIDIA GPUs, cross-platform GPU acceleration.

Automatic Memory Type Conversion
--------------------------------

OpenHCS automatically converts between memory types when chaining functions from different backends:

.. code-block:: python

   # Chain functions from different backends - automatic conversion
   step = FunctionStep(
       func=[
           (gaussian_filter, {}),           # CuPy (GPU)
           (stack_percentile_normalize, {}), # PyTorch (GPU)
           (count_cells_single_channel, {})  # NumPy (CPU)
       ],
       name="mixed_backend_chain"
   )

**How it works**: OpenHCS detects memory type requirements and automatically converts data between NumPy arrays, CuPy arrays, PyTorch tensors, and pyclesperanto arrays as needed.

**Performance optimization**: Conversions are minimized by grouping operations by memory type when possible.

Function Discovery and Selection
--------------------------------

Finding Available Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openhcs.processing.func_registry import get_functions_by_memory_type
   
   # List all available CuPy functions
   cupy_functions = get_functions_by_memory_type('cupy')
   print(f"Available CuPy functions: {len(cupy_functions)}")

**Function naming**: Functions are organized by backend and functionality:
- ``processors/``: Basic image processing
- ``analysis/``: Quantitative analysis  
- ``assemblers/``: Image assembly and stitching
- ``enhancers/``: Advanced enhancement algorithms

Choosing the Right Backend
~~~~~~~~~~~~~~~~~~~~~~~~~

**Performance considerations**:
- **GPU backends**: 10-100x faster for large images
- **CPU backends**: Better for small images or when GPU memory is limited
- **Memory usage**: GPU backends require sufficient GPU memory

**Compatibility considerations**:
- **NumPy**: Works on all systems
- **CuPy**: Requires NVIDIA GPU with CUDA
- **PyTorch**: Requires GPU with PyTorch installation
- **pyclesperanto**: Requires OpenCL-compatible GPU

Function Parameters and Configuration
------------------------------------

All function parameters can be specified in the FunctionStep:

.. code-block:: python

   # Parameters passed directly to the function
   step = FunctionStep(
       func=(gaussian_filter, {
           'sigma': 2.0,              # Function parameter
           'truncate': 4.0            # Function parameter
       }),
       name="blur"             # Step parameter
   )

**Parameter types**:
- **Function parameters**: Passed to the processing function
- **Step parameters**: Control OpenHCS behavior (name, variable_components, etc.)

The function library provides a comprehensive toolkit for bioimage analysis while maintaining consistency and performance across different computational backends. The 3D array contract and automatic memory management enable complex analysis workflows without manual data type coordination.

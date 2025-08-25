Function Library
================

OpenHCS provides seamless integration with major Python image processing libraries, all unified under a consistent 3D array interface. Understanding this function library is essential for building effective analysis pipelines.

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

Supported Python Libraries
--------------------------

OpenHCS provides native integration with major Python image processing libraries through its unified registry system:

Scikit-Image Integration
~~~~~~~~~~~~~~~~~~~~~~~

**Modules**: filters, morphology, segmentation, feature, measure, transform, restoration, exposure

.. code-block:: python

   from skimage import filters, morphology, segmentation

   # All scikit-image functions work seamlessly in OpenHCS
   step = FunctionStep(func=(filters.gaussian, {'sigma': 2.0}))
   step = FunctionStep(func=(morphology.opening, {'footprint': morphology.disk(3)}))
   step = FunctionStep(func=(segmentation.watershed, {}))

**Benefits**: CPU-compatible, extensive documentation, mature algorithms

CuCIM Integration (GPU-Accelerated Scikit-Image)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Modules**: filters, morphology, measure, segmentation, feature, restoration, transform, exposure

.. code-block:: python

   from cucim import skimage as cusk

   # GPU-accelerated versions of scikit-image functions
   step = FunctionStep(func=(cusk.filters.gaussian, {'sigma': 2.0}))
   step = FunctionStep(func=(cusk.morphology.opening, {}))
   step = FunctionStep(func=(cusk.segmentation.watershed, {}))

**Benefits**: 10-100x faster than CPU, identical API to scikit-image

Pyclesperanto Integration (OpenCL GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Coverage**: Comprehensive GPU image processing library

.. code-block:: python

   import pyclesperanto as cle

   # OpenCL GPU acceleration (works with AMD, Intel, NVIDIA)
   step = FunctionStep(func=(cle.gaussian_blur, {'sigma_x': 2.0, 'sigma_y': 2.0}))
   step = FunctionStep(func=(cle.opening_box, {'radius_x': 3, 'radius_y': 3}))
   step = FunctionStep(func=(cle.watershed, {}))

**Benefits**: Cross-platform GPU support, optimized for image processing

OpenHCS-Specific Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

**Specialized Analysis**: Cell counting, neurite tracing, feature extraction

.. code-block:: python

   from openhcs.processing.backends.analysis.cell_counting_cpu import (
       count_cells_single_channel, DetectionMethod
   )
   from openhcs.processing.backends.analysis.skan_axon_analysis import (
       skan_axon_skeletonize_and_analyze, AnalysisDimension
   )

   # Specialized analysis functions built for HCS workflows
   step = FunctionStep(
       func=(count_cells_single_channel, {
           'detection_method': DetectionMethod.WATERSHED,
           'min_sigma': 1.0,
           'max_sigma': 10.0
       })
   )

**Image Assembly**: Stitching, projection, compositing

.. code-block:: python

   from openhcs.processing.backends.assemblers.assemble_stack_cupy import (
       assemble_stack_cupy
   )
   from openhcs.processing.backends.processors.torch_processor import (
       max_projection, stack_percentile_normalize
   )

   # HCS-optimized assembly and processing
   step = FunctionStep(func=(assemble_stack_cupy, {}))
   step = FunctionStep(func=(stack_percentile_normalize, {}))

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

The function library provides seamless access to the Python image processing ecosystem while maintaining consistency and performance across different computational backends. The 3D array contract and automatic memory management enable complex analysis workflows without manual data type coordination.

Key Benefits of Library Integration
----------------------------------

**Unified Interface**: All functions follow the same 3D array contract regardless of underlying library

**Automatic Memory Management**: OpenHCS handles conversions between NumPy, CuPy, PyTorch, and pyclesperanto arrays

**Performance Optimization**: GPU-accelerated versions automatically used when available

**Ecosystem Leverage**: Access to the full Python image processing ecosystem without vendor lock-in

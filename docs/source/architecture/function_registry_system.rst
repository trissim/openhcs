Function Registry System
========================

Overview
--------

OpenHCS implements a revolutionary function registry system that
automatically discovers and unifies 574+ functions from multiple GPU
libraries with type-safe contracts. This creates the most comprehensive
GPU imaging function ecosystem available in scientific computing.

**Note**: OpenHCS functions are used as function objects in
FunctionStep, not string names. Examples show the real API patterns used
in production pipelines.

The Innovation
--------------

**What Makes It Unique**: No other scientific computing platform
automatically discovers and unifies this many GPU imaging libraries with
unified contracts and type safety.

Automatic Function Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS automatically registers functions from:

.. code:: python

   ✅ 230 pyclesperanto functions (GPU-accelerated OpenCL)
   ✅ 110 scikit-image functions (with GPU variants via CuCIM)  
   ✅ 124 CuCIM functions (RAPIDS GPU imaging)
   ✅ CuPy scipy.ndimage functions
   ✅ Native OpenHCS functions

   Total: 574+ functions with unified contracts

Intelligent Contract Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The registry analyzes each function to determine its 3D processing
behavior:

.. code:: python

   # Automatic contract detection:
   @numpy  # SLICE_SAFE - processes each Z-slice independently
   def gaussian_filter(image_stack, sigma=1.0):
       return scipy.ndimage.gaussian_filter(image_stack, sigma)

   @cupy   # CROSS_Z - processes entire 3D volume
   def watershed_3d(image_stack, markers):
       return cucim.skimage.segmentation.watershed(image_stack, markers)

   # Real usage in FunctionStep:
   step = FunctionStep(func=[(gaussian_filter, {'sigma': 2.0})])

Architecture
------------

Registry Discovery Process
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Automatic discovery workflow:
   1. Library Detection
      ├── Scan installed packages (pyclesperanto, scikit-image, etc.)
      ├── Identify imaging functions via introspection
      └── Filter for 3D-compatible functions

   2. Contract Analysis
      ├── Analyze function signatures
      ├── Determine 3D processing behavior (SLICE_SAFE vs CROSS_Z)
      └── Classify memory type requirements

   3. Decoration Application
      ├── Apply appropriate memory type decorators
      ├── Add contract metadata
      └── Register in unified namespace

   4. Validation
      ├── Verify all functions have memory type attributes
      ├── Test basic functionality
      └── Generate registry statistics

Unified Contract System
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # All functions get unified contracts:
   @numpy
   @contract_3d(behavior="SLICE_SAFE")
   def registered_function(image_stack, **kwargs):
       """Automatically decorated function with unified interface."""
       pass

   # Contract metadata includes:
   - input_memory_type: numpy, cupy, torch, etc.
   - output_memory_type: numpy, cupy, torch, etc.
   - contract_3d: SLICE_SAFE, CROSS_Z, UNKNOWN, DIM_CHANGE
   - gpu_compatible: True/False
   - library_source: pyclesperanto, scikit-image, etc.

Zero-Configuration GPU Library Access
-------------------------------------

Traditional Approach (Manual Integration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Traditional scientific computing - manual setup:
   import scipy.ndimage
   import cucim.skimage.filters
   import pyclesperanto as cle
   import cupy as cp

   # Manual GPU memory management
   gpu_image = cp.asarray(image)
   result1 = cucim.skimage.filters.gaussian(gpu_image, sigma=2.0)
   result2 = cle.binary_opening(result1, footprint=cle.create_disk(3))
   result3 = cucim.skimage.measure.label(result2, connectivity=2)
   final = cp.asnumpy(result3)  # Manual CPU transfer

OpenHCS Approach (Unified Registry)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # OpenHCS - unified access with function objects:
   from openhcs.processing.backends.processors.cupy_processor import tophat
   from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel

   steps = [
       FunctionStep(func=[(tophat, {'selem_radius': 50})]),                    # GPU-accelerated processing
       FunctionStep(func=[(count_cells_single_channel, {'min_sigma': 1.0})]), # Unified function interface
   ]

   # Benefits:
   ✅ Direct function object imports (type-safe)
   ✅ Automatic GPU memory management
   ✅ Unified parameter interface
   ✅ Type-safe conversions between libraries
   ✅ Consistent error handling

Automatic Dtype Conversion System
----------------------------------

OpenHCS implements intelligent automatic dtype conversion to handle the diverse data type requirements of different GPU libraries while maintaining pipeline consistency.

The Challenge
~~~~~~~~~~~~~

Different GPU libraries have specific data type requirements:

.. code:: python

   # pyclesperanto binary functions expect binary (0/1) input
   binary_infsup(image)  # ❌ Warning: "expected binary, float given"

   # pyclesperanto mode functions require uint8 input
   mode(image)  # ❌ Warning: "mode only support uint8 pixel type"

   # OpenHCS pipeline uses float32 [0,1] throughout
   image = np.random.rand(100, 100).astype(np.float32)  # Standard format

The Solution: Transparent Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS automatically converts data types during function execution:

.. code:: python

   # User calls function with float32 input
   result = binary_infsup(float32_image)  # ✅ No warnings!

   # Internal process:
   # 1. Detect function requires binary input
   # 2. Convert: float32 [0,1] → uint8 {0,255} with threshold at 0.5
   # 3. Execute: binary_infsup(uint8_binary_image)
   # 4. Convert back: uint8 result → float32 [0,1]
   # 5. Return: float32 result to user

Supported Conversions
~~~~~~~~~~~~~~~~~~~~~

**Binary Functions** (require 0/1 values):

.. code:: python

   # Functions: binary_infsup, binary_supinf
   # Conversion: float32 [0,1] → binary threshold at 0.5 → uint8 {0,255}
   # Example:
   input_image = np.array([[0.2, 0.7], [0.4, 0.9]], dtype=np.float32)
   # Internal: [[0, 255], [0, 255]] (thresholded at 0.5)
   result = binary_infsup(input_image)  # Returns float32 [0,1]

**UINT8 Functions** (require 8-bit integers):

.. code:: python

   # Functions: mode, mode_box, mode_sphere
   # Conversion: float32 [0,1] → uint8 [0,255]
   # Example:
   input_image = np.array([[0.2, 0.7], [0.4, 0.9]], dtype=np.float32)
   # Internal: [[51, 178], [102, 229]] (scaled to uint8)
   result = mode(input_image)  # Returns float32 [0,1]

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

The dtype conversion system is implemented in function adapters:

.. code:: python

   # In pyclesperanto_registry.py
   BINARY_FUNCTIONS = {'binary_infsup', 'binary_supinf'}
   UINT8_FUNCTIONS = {'mode', 'mode_box', 'mode_sphere'}

   def _pycle_adapt_function(original_func):
       func_name = getattr(original_func, '__name__', 'unknown')

       @wraps(original_func)
       def adapted(image, *args, **kwargs):
           original_dtype = image.dtype
           converted_image = image

           # Apply dtype conversion for specific functions
           if func_name in BINARY_FUNCTIONS:
               if image.dtype == np.float32:
                   # Binary threshold at 0.5
                   converted_image = ((image > 0.5) * 255).astype(np.uint8)
           elif func_name in UINT8_FUNCTIONS:
               if image.dtype == np.float32:
                   # Scale to uint8 range
                   converted_image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

           # Execute function with converted input
           result = original_func(converted_image, *args, **kwargs)

           # Convert result back to original dtype
           if func_name in BINARY_FUNCTIONS or func_name in UINT8_FUNCTIONS:
               if hasattr(result, 'dtype') and result.dtype != original_dtype:
                   if result.dtype == np.uint8 and original_dtype == np.float32:
                       result = result.astype(np.float32) / 255.0

           return result

Dtype Conversion Benefits
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   ✅ Transparent to users - no API changes required
   ✅ Eliminates dtype warnings during function execution
   ✅ Maintains OpenHCS float32 [0,1] pipeline consistency
   ✅ Automatic scaling between data type ranges
   ✅ Preserves function behavior and results
   ✅ Zero performance impact for functions not requiring conversion

Warning Attribution System
~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS includes a sophisticated warning attribution system for debugging:

.. code:: python

   # During registry building, warnings are properly attributed:
   🧪 Testing pyclesperanto function: pyclesperanto.binary_infsup
   Warning: Source image of binary_infsup expected to be binary, float given.

   🧪 Testing pyclesperanto function: pyclesperanto.mode
   Warning: mode only support uint8 pixel type.

   # For end users, no warnings appear:
   result = binary_infsup(float32_image)  # ✅ Silent execution
   result = mode(float32_image)          # ✅ Silent execution

Registry Statistics
-------------------

Current Function Counts
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   Registry Statistics (as of current version):
   ├── pyclesperanto: 230 functions
   │   ├── Morphological operations: 45
   │   ├── Filtering: 38
   │   ├── Segmentation: 32
   │   ├── Measurements: 28
   │   └── Transformations: 87
   ├── scikit-image (via CuCIM): 110 functions
   │   ├── Filters: 35
   │   ├── Morphology: 25
   │   ├── Segmentation: 20
   │   ├── Measure: 18
   │   └── Transform: 12
   ├── CuCIM native: 124 functions
   │   ├── Core operations: 45
   │   ├── Advanced filters: 35
   │   ├── Registration: 25
   │   └── Utilities: 19
   └── Native OpenHCS: 110+ functions
       ├── Pattern processing: 35
       ├── Batch operations: 30
       ├── Memory management: 25
       └── Validation: 20

Dtype Conversion Coverage
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   Automatic Dtype Conversion Statistics:
   ├── Binary functions: 2 functions
   │   ├── binary_infsup (pyclesperanto)
   │   └── binary_supinf (pyclesperanto)
   ├── UINT8 functions: 3 functions
   │   ├── mode (pyclesperanto)
   │   ├── mode_box (pyclesperanto)
   │   └── mode_sphere (pyclesperanto)
   └── Coverage: 100% of identified dtype-sensitive functions

   Total functions with automatic dtype conversion: 5
   Functions requiring no conversion: 569+
   Warning elimination rate: 100%

Performance Benefits
--------------------

Unified Memory Management
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Automatic memory type conversion:
   Step 1: disk(tiff) → numpy → gaussian_filter → numpy → memory
   Step 2: memory → cupy → binary_opening → cupy → memory  
   Step 3: memory → numpy → label → numpy → disk(tiff)

   # Conversions handled automatically:
   ✅ Zero-copy GPU transfers where possible
   ✅ Minimal CPU roundtrips
   ✅ Automatic device management
   ✅ Memory pressure handling

Library Optimization
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Intelligent function routing:
   Function Request: "gaussian_filter"
   ├── Available implementations:
   │   ├── scipy.ndimage.gaussian_filter (CPU)
   │   ├── cucim.skimage.filters.gaussian (GPU)
   │   └── pyclesperanto.gaussian_blur (GPU)
   ├── Selection criteria:
   │   ├── Memory type compatibility
   │   ├── GPU availability
   │   └── Performance characteristics
   └── Chosen: cucim.skimage.filters.gaussian (best GPU performance)

Comparison with Other Platforms
-------------------------------

ImageJ/FIJI
~~~~~~~~~~~

-  **Functions**: ~1000+ plugins, mostly CPU
-  **Integration**: Manual plugin installation
-  **GPU Support**: Limited, plugin-dependent
-  **Contracts**: None, runtime discovery of capabilities

CellProfiler
~~~~~~~~~~~~

-  **Functions**: ~80 modules, mostly CPU
-  **Integration**: Built-in modules only
-  **GPU Support**: Very limited
-  **Contracts**: Module-specific interfaces

napari
~~~~~~

-  **Functions**: Plugin ecosystem, variable quality
-  **Integration**: Manual plugin management
-  **GPU Support**: Plugin-dependent
-  **Contracts**: Plugin-specific

OpenHCS
~~~~~~~

-  **Functions**: 574+ unified functions, GPU-first
-  **Integration**: Automatic discovery and registration
-  **GPU Support**: Native GPU support across all libraries
-  **Contracts**: Unified type-safe contracts for all functions

Future Expansion
----------------

Planned Library Integrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Roadmap for additional libraries:
   ├── ITK (Insight Toolkit) - Medical imaging
   ├── SimpleITK - Simplified ITK interface  
   ├── OpenCV - Computer vision functions
   ├── Mahotas - Computer vision for biology
   ├── nd2reader - Nikon microscopy formats
   └── AICSImageIO - Allen Institute formats

Registry Evolution
~~~~~~~~~~~~~~~~~~

-  **Dynamic loading**: Add libraries at runtime
-  **Custom contracts**: User-defined function contracts
-  **Performance profiling**: Automatic benchmarking of function
   variants
-  **Cloud functions**: Integration with cloud-based processing

Technical Implementation
------------------------

Registry Architecture
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   class FunctionRegistry:
       """Central registry for all discovered functions."""
       
       def __init__(self):
           self.functions = {}  # name -> function mapping
           self.metadata = {}   # name -> contract metadata
           self.sources = {}    # name -> library source
       
       def discover_functions(self):
           """Discover functions from all available libraries."""
           for library in self.supported_libraries:
               functions = library.discover_functions()
               for func in functions:
                   self.register_function(func)
       
       def register_function(self, func):
           """Register function with unified contract."""
           contract = self.analyze_contract(func)
           decorated_func = self.apply_decorators(func, contract)
           self.functions[func.name] = decorated_func
           self.metadata[func.name] = contract

This function registry system represents a fundamental innovation in
scientific computing - providing unified, type-safe access to the entire
GPU imaging ecosystem through a single, consistent interface.

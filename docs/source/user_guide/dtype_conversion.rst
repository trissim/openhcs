Automatic Dtype Conversion
===========================

OpenHCS provides transparent automatic data type conversion to handle the diverse requirements of different GPU libraries while maintaining pipeline consistency.

Overview
--------

Different GPU libraries have specific data type requirements that can conflict with OpenHCS's standardized float32 [0,1] pipeline. OpenHCS automatically handles these conversions behind the scenes, eliminating warnings and ensuring correct function behavior.

Why Dtype Conversion is Needed
-------------------------------

GPU libraries often have strict data type requirements:

.. code:: python

   # pyclesperanto binary functions expect binary (0/1) input
   import pyclesperanto as cle
   image = np.random.rand(100, 100).astype(np.float32)  # [0,1] range
   result = cle.binary_infsup(image)  # ❌ Warning: "expected binary, float given"
   
   # pyclesperanto mode functions require uint8 input
   result = cle.mode(image)  # ❌ Warning: "mode only support uint8 pixel type"

Without automatic conversion, users would need to manually handle these conversions:

.. code:: python

   # Manual conversion (what you DON'T need to do in OpenHCS)
   binary_image = ((image > 0.5) * 255).astype(np.uint8)  # Manual binary conversion
   result = cle.binary_infsup(binary_image)
   result = result.astype(np.float32) / 255.0  # Manual conversion back

How It Works
------------

OpenHCS automatically detects function requirements and applies appropriate conversions:

.. code:: python

   from openhcs.processing.func_registry import get_function_by_name
   
   # Get registered functions (these have automatic dtype conversion)
   binary_infsup = get_function_by_name('binary_infsup', 'pyclesperanto')
   mode = get_function_by_name('mode', 'pyclesperanto')
   
   # Use with standard OpenHCS float32 input
   image = np.random.rand(100, 100).astype(np.float32)  # [0,1] range
   
   # These work without warnings or manual conversion
   result1 = binary_infsup(image)  # ✅ Automatic binary conversion
   result2 = mode(image)          # ✅ Automatic uint8 conversion
   
   # Results are automatically converted back to float32 [0,1]
   print(result1.dtype)  # float32
   print(result2.dtype)  # float32

Supported Conversions
---------------------

Binary Functions
~~~~~~~~~~~~~~~~

Functions that require binary (0/1) input:

.. code:: python

   # Affected functions:
   - binary_infsup
   - binary_supinf
   
   # Conversion process:
   # 1. Input: float32 [0,1]
   # 2. Threshold: values > 0.5 become 1, others become 0
   # 3. Scale: binary {0,1} → uint8 {0,255}
   # 4. Execute function with uint8 binary input
   # 5. Convert result back to float32 [0,1]

Example:

.. code:: python

   input_image = np.array([[0.2, 0.7], [0.4, 0.9]], dtype=np.float32)
   
   # Internal conversion:
   # [[0.2, 0.7], [0.4, 0.9]] → threshold at 0.5 → [[0, 1], [0, 1]]
   # [[0, 1], [0, 1]] → scale to uint8 → [[0, 255], [0, 255]]
   
   result = binary_infsup(input_image)
   # Result is converted back to float32 [0,1] range

UINT8 Functions
~~~~~~~~~~~~~~~

Functions that require 8-bit unsigned integer input:

.. code:: python

   # Affected functions:
   - mode
   - mode_box  
   - mode_sphere
   
   # Conversion process:
   # 1. Input: float32 [0,1]
   # 2. Clip: ensure values are in [0,1] range
   # 3. Scale: [0,1] → [0,255] and convert to uint8
   # 4. Execute function with uint8 input
   # 5. Convert result back to float32 [0,1]

Example:

.. code:: python

   input_image = np.array([[0.2, 0.7], [0.4, 0.9]], dtype=np.float32)
   
   # Internal conversion:
   # [[0.2, 0.7], [0.4, 0.9]] → scale to uint8 → [[51, 178], [102, 229]]
   
   result = mode(input_image)
   # Result is converted back to float32 [0,1] range

Pipeline Integration
--------------------

Dtype conversion integrates seamlessly with OpenHCS pipelines:

.. code:: python

   from openhcs.processing.func_registry import get_function_by_name
   from openhcs.processing.step import FunctionStep
   
   # Get functions with automatic dtype conversion
   gaussian = get_function_by_name('gaussian_filter', 'skimage')
   binary_op = get_function_by_name('binary_infsup', 'pyclesperanto')
   mode_filter = get_function_by_name('mode', 'pyclesperanto')
   
   # Create pipeline steps
   steps = [
       FunctionStep(gaussian, sigma=1.0),      # float32 → float32
       FunctionStep(binary_op),                # float32 → auto convert → float32
       FunctionStep(mode_filter),              # float32 → auto convert → float32
   ]
   
   # All conversions happen automatically
   # Pipeline maintains float32 [0,1] consistency throughout

Best Practices
--------------

Input Data Preparation
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # ✅ Good: Use standard OpenHCS format
   image = load_image().astype(np.float32)
   if image.max() > 1.0:
       image = image / image.max()  # Normalize to [0,1]
   
   # ❌ Avoid: Manual dtype conversion for specific functions
   # OpenHCS handles this automatically

Function Selection
~~~~~~~~~~~~~~~~~~

.. code:: python

   # ✅ Good: Use registered functions for automatic conversion
   from openhcs.processing.func_registry import get_function_by_name
   binary_infsup = get_function_by_name('binary_infsup', 'pyclesperanto')
   
   # ❌ Avoid: Direct library calls (no automatic conversion)
   import pyclesperanto as cle
   result = cle.binary_infsup(image)  # May show warnings

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Dtype conversion has minimal performance impact:
   ✅ Only applied to functions that need it (5 out of 574+ functions)
   ✅ Conversion operations are fast (simple scaling/thresholding)
   ✅ No impact on functions that don't require conversion
   ✅ GPU memory transfers remain optimized

Troubleshooting
---------------

If you encounter dtype-related issues:

.. code:: python

   # Check if function has automatic conversion
   from openhcs.processing.func_registry import get_function_by_name
   func = get_function_by_name('function_name', 'library_name')
   
   if func is None:
       print("Function not found in registry")
   else:
       print("Function has automatic dtype conversion")
   
   # Verify input data format
   print(f"Input dtype: {image.dtype}")
   print(f"Input range: [{image.min():.3f}, {image.max():.3f}]")
   
   # Expected: dtype=float32, range=[0.0, 1.0]

For functions not yet covered by automatic conversion, you can request support by filing an issue with the specific function name and library.

Memory Type System and Stack Utils
==================================

Overview
--------

OpenHCS implements a memory type system that enables conversion between
different array libraries (NumPy, PyTorch, CuPy, TensorFlow, JAX,
pyclesperanto) while maintaining dimensional constraints and GPU device
management.

**Note**: All code examples reflect the actual OpenHCS implementation
and are verified against the current codebase.

Core Principles
---------------

Clause 278: Mandatory 3D Output Enforcement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All functions must return a 3D array of shape ``[Z, Y, X]``, even when
operating on a single 2D slice. This prevents silent shape coercion and
enforces explicit intent throughout the pipeline.

Memory Type Discipline
~~~~~~~~~~~~~~~~~~~~~~

-  **Explicit Declaration**: All functions must declare input/output
   memory types via decorators
-  **Automatic Conversion**: Stack utils handle conversion between
   memory types
-  **GPU Discipline**: Explicit GPU device management and validation
-  **Strict Validation**: Fail fast on invalid inputs rather than silent
   coercion

Stack Utils Architecture
------------------------

``stack_slices()``: 2D → 3D Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converts a list of 2D images into a single 3D array with specified
memory type:

.. literalinclude:: ../../../openhcs/core/memory/stack_utils.py
   :language: python
   :lines: 137-150
   :caption: stack_slices function signature and validation

**Input Requirements**:

- ``slices``: List of 2D arrays (any supported memory type)
- ``memory_type``: Target memory type (``numpy``, ``cupy``, ``torch``, ``tensorflow``, ``jax``, ``pyclesperanto``)
- ``gpu_id``: GPU device ID (required, validated for GPU memory types)

**Output Guarantees**:

- Always returns 3D array of shape ``[Z, Y, X]``
- All slices converted to target memory type
- GPU placement enforced for GPU memory types

**Validation**:

- All input slices must be 2D
- Empty slice list raises error
- GPU device ID validated for GPU memory types

``unstack_slices()``: 3D → 2D Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Splits a 3D array into a list of 2D slices with specified memory type:

.. literalinclude:: ../../../openhcs/core/memory/stack_utils.py
   :language: python
   :lines: 337-350
   :caption: unstack_slices function signature and validation

**Input Requirements**:

- ``array``: 3D array (any supported memory type)
- ``memory_type``: Target memory type for output slices
- ``gpu_id``: GPU device ID (required)
- ``validate_slices``: Optional validation that output slices are 2D

**Output Guarantees**:

- Returns list of 2D slices in target memory type
- Preserves spatial dimensions (Y, X)
- GPU placement enforced for GPU memory types

Memory Conversion System
------------------------

Core Conversion Function
~~~~~~~~~~~~~~~~~~~~~~~~

The memory conversion system provides direct conversion between any supported memory types:

.. literalinclude:: ../../../openhcs/core/memory/converters.py
   :language: python
   :lines: 61-87
   :caption: Core memory conversion function

**Supported Conversions**:

- **NumPy** ↔ **CuPy**: Direct GPU/CPU transfer
- **PyTorch** ↔ **CuPy**: GPU tensor sharing
- **TensorFlow** ↔ **JAX**: Cross-framework GPU arrays
- **pyclesperanto** ↔ **All**: OpenCL GPU arrays

**Conversion Strategy**:

1. **Direct Conversion**: When libraries share memory layout
2. **CPU Roundtrip**: When direct GPU-to-GPU conversion unavailable
3. **Fail-Loud**: Clear errors for unsupported conversions

Memory Type Detection
~~~~~~~~~~~~~~~~~~~~~

Automatic detection of array memory types:

.. literalinclude:: ../../../openhcs/core/memory/utils.py
   :language: python
   :lines: 80-120
   :caption: Memory type detection logic

**Detection Strategy**:

- **Type-based**: Uses ``type(data).__module__`` patterns
- **Attribute-based**: Checks for library-specific attributes
- **Fail-loud**: Raises error for unknown types

GPU Device Management
---------------------

Device Validation
~~~~~~~~~~~~~~~~~

Strict GPU device validation for GPU memory types:

.. literalinclude:: ../../../openhcs/core/memory/stack_utils.py
   :language: python
   :lines: 80-95
   :caption: GPU device validation

**Validation Rules**:

- GPU memory types require valid ``gpu_id``
- CPU memory types ignore ``gpu_id``
- Invalid device IDs raise immediate errors

Device Movement
~~~~~~~~~~~~~~

Moving arrays between GPU devices:

.. literalinclude:: ../../../openhcs/core/memory/utils.py
   :language: python
   :lines: 280-320
   :caption: GPU device movement utilities

**Movement Strategy**:

- **PyTorch**: ``.to(device)`` method
- **CuPy**: Context manager with device switching
- **TensorFlow**: ``tf.device()`` context
- **JAX**: ``jax.device_put()`` with explicit device

Integration with FunctionStep
-----------------------------

Pipeline Integration
~~~~~~~~~~~~~~~~~~~

Stack utils integrate seamlessly with FunctionStep execution:

.. code-block:: python

   # FunctionStep automatically handles memory type conversion
   def process_images(context, pattern_group, memory_type, gpu_id):
       # 1. Load 2D slices from files
       slices_2d = [load_image(path) for path in pattern_group]

       # 2. Stack to 3D with target memory type
       stack_3d = stack_slices(slices_2d, memory_type, gpu_id)

       # 3. Process with target memory type
       processed_3d = processing_function(stack_3d)

       # 4. Unstack back to 2D for saving
       output_slices = unstack_slices(processed_3d, "numpy", gpu_id)

       return output_slices

**Memory Type Flow**:

1. **Input**: Files loaded as NumPy arrays
2. **Processing**: Converted to target memory type (torch, cupy, etc.)
3. **Output**: Converted back to NumPy for file saving

Performance Characteristics
---------------------------

Conversion Performance
~~~~~~~~~~~~~~~~~~~~~

**Fast Conversions** (shared memory):

- NumPy ↔ CuPy (GPU memory sharing)
- PyTorch ↔ CuPy (GPU tensor sharing)
- TensorFlow ↔ JAX (similar GPU layouts)

**Slow Conversions** (CPU roundtrip):

- PyTorch ↔ TensorFlow (different GPU memory models)
- pyclesperanto ↔ JAX (OpenCL ↔ CUDA)

**Optimization Strategy**:

- Minimize conversions within processing chains
- Use consistent memory types for related operations
- Leverage GPU memory sharing when possible

Memory Usage Patterns
~~~~~~~~~~~~~~~~~~~~~

**Efficient Patterns**:

.. code-block:: python

   # Good: Consistent memory type throughout chain
   stack_3d = stack_slices(slices, "torch", gpu_id)
   processed = torch_function_1(stack_3d)
   processed = torch_function_2(processed)
   output = unstack_slices(processed, "numpy", gpu_id)

**Inefficient Patterns**:

.. code-block:: python

   # Bad: Multiple conversions
   stack_3d = stack_slices(slices, "torch", gpu_id)
   cupy_result = convert_memory(stack_3d, "torch", "cupy", gpu_id)
   torch_result = convert_memory(cupy_result, "cupy", "torch", gpu_id)

Error Handling and Validation
-----------------------------

Strict Validation Philosophy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS follows fail-loud principles for memory operations:

**Dimensional Validation**:

- Input slices must be exactly 2D
- Output arrays must be exactly 3D
- No silent dimension coercion

**Memory Type Validation**:

- Unknown memory types raise immediate errors
- GPU operations require valid device IDs
- Conversion failures provide detailed error messages

**Example Error Messages**:

.. code-block:: text

   MemoryConversionError: Failed to convert torch tensor to cupy array
   Reason: GPU device 2 not available (only 0-1 available)
   Source: torch.cuda.FloatTensor on device 0
   Target: cupy array on device 2

Future Extensions
----------------

Planned Enhancements
~~~~~~~~~~~~~~~~~~~

**Memory Pressure Detection**:

- Automatic fallback to CPU when GPU memory exhausted
- Smart memory type selection based on available resources

**Lazy Conversion**:

- Defer conversions until actually needed
- Chain multiple operations before converting

**Memory Pool Management**:

- Reuse allocated arrays to reduce allocation overhead
- GPU memory pool optimization for large datasets

This memory type system ensures type safety, performance, and maintainability across OpenHCS's multi-backend processing pipeline.

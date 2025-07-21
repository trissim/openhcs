Memory Type System and Stack Utils
==================================

Overview
--------

OpenHCS implements a sophisticated memory type system that enables
seamless conversion between different array libraries (NumPy, PyTorch,
CuPy, TensorFlow, JAX, pyclesperanto) while maintaining strict
dimensional constraints and GPU device discipline.

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

.. code:: python

   # Actual stack_slices signature from openhcs/core/memory/stack_utils.py
   stack_3d = stack_slices(
       slices=[img1_2d, img2_2d, img3_2d],  # List of 2D arrays (any memory type)
       memory_type="torch",                  # Target memory type
       gpu_id=0                             # GPU device ID
   )
   # Returns: torch.Tensor of shape [3, Y, X] on GPU 0

**Input Requirements**:
- ``slices``: List of 2D arrays (any supported memory type)
- ``memory_type``: Target memory type (``numpy``, ``cupy``, ``torch``, ``tensorflow``, ``jax``, ``pyclesperanto``)
- ``gpu_id``: GPU device ID (required, validated for GPU memory types)

**Output Guarantees**: - Always returns 3D array of shape ``[Z, Y, X]``
- All slices converted to target memory type - GPU placement enforced
for GPU memory types

**Validation**: - All input slices must be 2D - Empty slice list raises
error - Single slice requires explicit ``allow_single_slice=True`` - GPU
device ID validated for GPU memory types

``unstack_slices()``: 3D → 2D Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Splits a 3D array into a list of 2D slices with specified memory type:

.. code:: python

   slices_2d = unstack_slices(
       array=stack_3d,        # 3D array (any memory type)
       memory_type="numpy",   # Target memory type for output slices
       gpu_id=0,             # GPU device ID
       validate_slices=True  # Validate output slices are 2D
   )

**Input Requirements**: - ``array``: 3D array (any supported memory
type) - ``memory_type``: Target memory type for output slices -
``gpu_id``: GPU device ID (required) - ``validate_slices``: Whether to
validate output slices are 2D

**Output Guarantees**: - Returns list of 2D arrays - All slices
converted to target memory type - GPU placement enforced for GPU memory
types

**Validation**: - Input array must be 3D - Output slices validated as 2D
(if ``validate_slices=True``) - GPU device ID validated for GPU memory
types

Memory Type System
------------------

Supported Memory Types
~~~~~~~~~~~~~~~~~~~~~~

+------------+--------+------------+----------+---------------------+
| Memory     | L      | GPU        | Use      | Serialization       |
| Type       | ibrary | Support    | Cases    | Format              |
+============+========+============+==========+=====================+
| ``numpy``  | NumPy  | No         | CPU      | ``.npy``, ``.tiff`` |
|            |        |            | pro      |                     |
|            |        |            | cessing, |                     |
|            |        |            | I/O      |                     |
|            |        |            | op       |                     |
|            |        |            | erations |                     |
+------------+--------+------------+----------+---------------------+
| ``cupy``   | CuPy   | Yes        | GPU-acc  | ``.cupy`` (custom)  |
|            |        |            | elerated |                     |
|            |        |            | Nu       |                     |
|            |        |            | mPy-like |                     |
|            |        |            | op       |                     |
|            |        |            | erations |                     |
+------------+--------+------------+----------+---------------------+
| ``torch``  | P      | Yes        | Deep     | ``.pt``, ``.pth``   |
|            | yTorch |            | l        |                     |
|            |        |            | earning, |                     |
|            |        |            | neural   |                     |
|            |        |            | networks |                     |
+------------+--------+------------+----------+---------------------+
| ``te       | Tens   | Yes        | Machine  | ``.tf`` (serialized |
| nsorflow`` | orFlow |            | l        | tensor)             |
|            |        |            | earning, |                     |
|            |        |            | Te       |                     |
|            |        |            | nsorFlow |                     |
|            |        |            | models   |                     |
+------------+--------+------------+----------+---------------------+
| ``jax``    | JAX    | Yes        | High-per | ``.npy`` (via       |
|            |        |            | formance | device_get)         |
|            |        |            | co       |                     |
|            |        |            | mputing, |                     |
|            |        |            | research |                     |
+------------+--------+------------+----------+---------------------+
| ``pycle    | p      | Yes        | GPU-acc  | ``.cle`` (custom)   |
| speranto`` | yclesp |            | elerated |                     |
|            | eranto |            | image    |                     |
|            |        |            | pr       |                     |
|            |        |            | ocessing |                     |
+------------+--------+------------+----------+---------------------+

Array Conversion Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The memory type system implements a sophisticated conversion
architecture that coordinates three distinct layers:

Layer 1: VFS Serialization (Bytes ↔ Arrays)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **Purpose**: Convert between raw bytes (disk storage) and array
   objects (memory)
-  **Location**: FileManager backends (DiskStorageBackend,
   MemoryStorageBackend)
-  **Responsibility**: Format-specific serialization (TIFF, NPY, PT
   files)

Layer 2: Memory Type Conversion (Array ↔ Array)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **Purpose**: Convert between different array libraries while
   preserving data
-  **Location**: MemoryWrapper and conversion_functions.py
-  **Responsibility**: Cross-library conversion (numpy ↔ torch ↔ cupy)

Layer 3: Stack Operations (2D ↔ 3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **Purpose**: Dimensional transformation with memory type coordination
-  **Location**: stack_utils.py
-  **Responsibility**: Stacking/unstacking with automatic type
   conversion

Memory Type Detection
~~~~~~~~~~~~~~~~~~~~~

The system automatically detects memory types of input data:

.. code:: python

   def _detect_memory_type(data: Any) -> str:
       """Detect memory type with strict validation."""
       if isinstance(data, MemoryWrapper):
           return data.memory_type
       elif isinstance(data, np.ndarray):
           return "numpy"
       elif isinstance(data, torch.Tensor):
           return "torch"
       # ... other types
       else:
           raise ValueError(f"Could not detect memory type of {type(data)}")

**Strict Validation**: Fails loudly if memory type cannot be detected,
preventing silent errors.

Memory Type Conversion
~~~~~~~~~~~~~~~~~~~~~~

Conversion uses the ``MemoryWrapper`` class for consistent behavior:

.. code:: python

   # Convert slice to target memory type
   wrapped = MemoryWrapper(slice_data, memory_type=detected_type, gpu_id=gpu_id)

   if target_type == "numpy":
       converted = wrapped.to_numpy()
   elif target_type == "torch":
       converted = wrapped.to_torch(allow_cpu_roundtrip=False)
   elif target_type == "cupy":
       converted = wrapped.to_cupy(allow_cpu_roundtrip=False)
   # ... other types

**GPU Discipline**: - GPU memory types require valid ``gpu_id >= 0`` -
No automatic CPU roundtrips for GPU types - Explicit device placement
validation

Complete Conversion Flow
------------------------

End-to-End Data Journey
~~~~~~~~~~~~~~~~~~~~~~~

The complete data transformation follows this path through the system:

::

   Disk Storage (TIFF/NPY files)
       ↓ VFS Layer 1: Deserialization
   Raw Arrays (usually numpy)
       ↓ Stack Utils Layer 3: Stacking + Type Conversion
   3D Array (target memory type)
       ↓ Function Execution
   3D Result Array (function's output memory type)
       ↓ Stack Utils Layer 3: Unstacking + Type Conversion
   2D Arrays (target memory type)
       ↓ VFS Layer 1: Serialization
   Disk Storage (TIFF/NPY files)

Detailed Conversion Steps
~~~~~~~~~~~~~~~~~~~~~~~~~

Step 1: VFS Deserialization (Disk → Arrays)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # FileManager.load_image() calls DiskStorageBackend
   def load_image(self, file_path, backend):
       if backend == "disk":
           # Format-specific deserialization
           if file_path.endswith('.tiff'):
               return tifffile.imread(file_path)  # → numpy array
           elif file_path.endswith('.npy'):
               return np.load(file_path)  # → numpy array
           elif file_path.endswith('.pt'):
               return torch.load(file_path)  # → torch tensor
       elif backend == "memory":
           # Direct object retrieval (no conversion)
           return memory_store[file_path]

**Key Points**: - Disk backend always deserializes to specific array
types based on file format - Memory backend stores objects directly (no
serialization) - TIFF files always become numpy arrays - Format
determines initial memory type

Step 2: Stack Utils Conversion (2D → 3D + Memory Type)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # stack_slices() in FunctionStep execution
   def stack_slices(slices, memory_type, gpu_id):
       converted_slices = []
       for slice_2d in slices:
           # Detect current memory type
           current_type = _detect_memory_type(slice_2d)

           # Wrap in MemoryWrapper for conversion
           wrapped = MemoryWrapper(slice_2d, current_type, gpu_id)

           # Convert to target memory type
           if memory_type == "torch":
               converted = wrapped.to_torch(allow_cpu_roundtrip=False)
           elif memory_type == "numpy":
               converted = wrapped.to_numpy()
           # ... other types

           converted_slices.append(converted.data)  # Extract raw array

       # Stack using target library's stack function
       if memory_type == "torch":
           return torch.stack(converted_slices)
       elif memory_type == "numpy":
           return np.stack(converted_slices)

**Key Points**: - Each 2D slice converted individually to target memory
type - MemoryWrapper handles cross-library conversion - GPU device
placement enforced during conversion - Final stacking uses target
library’s native stack function

Step 3: MemoryWrapper Conversion (Array → Array)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # MemoryWrapper.to_torch() example
   def to_torch(self, allow_cpu_roundtrip=False):
       if self._memory_type == "numpy":
           # NumPy → PyTorch conversion
           tensor = torch.from_numpy(self._data)
           if self._gpu_id is not None:
               tensor = tensor.to(f"cuda:{self._gpu_id}")
           return MemoryWrapper(tensor, "torch", self._gpu_id)

       elif self._memory_type == "cupy":
           # CuPy → PyTorch via CUDA Array Interface
           if _supports_cuda_array_interface(self._data):
               tensor = torch.as_tensor(self._data, device=f"cuda:{self._gpu_id}")
               return MemoryWrapper(tensor, "torch", self._gpu_id)
           else:
               # Fallback to CPU roundtrip if allowed
               if allow_cpu_roundtrip:
                   numpy_data = self._data.get()  # CuPy → NumPy
                   tensor = torch.from_numpy(numpy_data).to(f"cuda:{self._gpu_id}")
                   return MemoryWrapper(tensor, "torch", self._gpu_id)
               else:
                   raise MemoryConversionError("CUDA Array Interface not supported")

**Key Points**: - Direct GPU-to-GPU conversion when possible (CUDA Array
Interface, DLPack) - CPU roundtrip as fallback (if explicitly allowed) -
Device placement preserved during conversion - Strict error handling
prevents silent failures

Step 4: Function Execution (Native Memory Type)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Function operates in its declared memory type
   @torch(input_type="torch", output_type="torch")
   def my_gpu_function(image_stack):
       # Receives torch.Tensor on GPU
       # All operations use PyTorch GPU functions
       result = torch.nn.functional.conv3d(image_stack, kernel)
       return result  # Returns torch.Tensor on GPU

**Key Points**: - Function receives data in its declared input memory
type - All operations use native library functions - No conversion
overhead during function execution - Output memory type determined by
function decorator

Step 5: Reverse Conversion (3D → 2D + Memory Type)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # unstack_slices() after function execution
   def unstack_slices(array_3d, memory_type, gpu_id):
       # Convert 3D array to target memory type first
       current_type = _detect_memory_type(array_3d)
       wrapped = MemoryWrapper(array_3d, current_type, gpu_id)

       if memory_type == "numpy":
           converted_3d = wrapped.to_numpy()
       # ... other conversions

       # Unstack to 2D slices
       slices_2d = [converted_3d.data[i] for i in range(converted_3d.data.shape[0])]
       return slices_2d

Step 6: VFS Serialization (Arrays → Disk)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # FileManager.save_image() calls DiskStorageBackend
   def save_image(self, data, file_path, backend):
       if backend == "disk":
           # Convert to numpy for TIFF output (most common)
           if isinstance(data, torch.Tensor):
               numpy_data = data.cpu().numpy()
           elif hasattr(data, 'get'):  # CuPy
               numpy_data = data.get()
           else:
               numpy_data = data

           # Format-specific serialization
           if file_path.endswith('.tiff'):
               tifffile.imwrite(file_path, numpy_data)
           elif file_path.endswith('.npy'):
               np.save(file_path, numpy_data)
       elif backend == "memory":
           # Store object directly
           memory_store[file_path] = data

**Key Points**: - Disk storage usually requires conversion to numpy (for
TIFF) - Memory storage preserves original memory type - Format
determines serialization method - GPU arrays moved to CPU for disk
storage

Integration with FunctionStep
-----------------------------

Complete Execution Flow
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def _process_single_pattern_group():
       # 1. Load images from VFS (usually numpy arrays from disk)
       raw_slices = []
       for file_path in matching_files:
           image = context.filemanager.load_image(file_path, read_backend)
           raw_slices.append(image)  # 2D images
       
       # 2. Stack into 3D with function's input memory type
       image_stack = stack_slices(
           slices=raw_slices,
           memory_type=input_memory_type_from_plan,  # From function decorator
           gpu_id=device_id
       )
       
       # 3. Execute function(s) - operates in native memory type
       result_stack = execute_function_pattern(
           func_pattern=executable_func_or_chain,
           image_stack=image_stack,
           **base_kwargs
       )
       
       # 4. Unstack to 2D slices with output memory type
       output_slices = unstack_slices(
           array=result_stack,
           memory_type=output_memory_type_from_plan,  # From function decorator
           gpu_id=device_id
       )
       
       # 5. Save slices to VFS (usually converted back to numpy for disk)
       for i, slice_2d in enumerate(output_slices):
           output_path = step_output_dir / f"output_{i}.tif"
           context.filemanager.save_image(slice_2d, output_path, write_backend)

Memory Type Flow in Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Input**: Images loaded as numpy arrays (from disk)
2. **Stack Conversion**: Convert to function’s input memory type
3. **Processing**: Function operates in its native memory type
4. **Unstack Conversion**: Convert to function’s output memory type
5. **Output**: Usually converted back to numpy for disk storage

Compilation Integration
~~~~~~~~~~~~~~~~~~~~~~~

The pipeline compiler coordinates memory types throughout the system:

Phase 1: Memory Type Extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # During compilation, extract memory types from function decorators
   def extract_memory_types(func):
       input_type = getattr(func, 'input_memory_type', None)
       output_type = getattr(func, 'output_memory_type', None)

       if input_type is None or output_type is None:
           raise ValueError(f"Function {func.__name__} missing memory type decorators")

       return input_type, output_type

   # For function patterns, validate consistency
   def validate_pattern_memory_types(func_pattern):
       if isinstance(func_pattern, list):
           # Sequential pattern - all functions must have same types
           types = [extract_memory_types(f) for f in func_pattern]
           if not all(t == types[0] for t in types):
               raise ValueError("Sequential functions must have consistent memory types")
       elif isinstance(func_pattern, dict):
           # Component-specific pattern - extract per component
           return {comp: extract_memory_types(func) for comp, func in func_pattern.items()}

Phase 2: Step Plan Population
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Compiler injects memory types into step plans
   step_plan = {
       "step_name": "GPU Processing",
       "input_memory_type": "torch",    # From function decorator
       "output_memory_type": "torch",   # From function decorator
       "gpu_id": 0,                     # Assigned by GPU resource planner
       "read_backend": "disk",          # From materialization planner
       "write_backend": "memory",       # From materialization planner
       # ... other configuration
   }

Phase 3: Runtime Coordination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # FunctionStep.process() uses step plan for conversions
   def process(self, context):
       step_plan = context.get_step_plan(self.step_id)

       # Extract memory type configuration
       input_memory_type = step_plan['input_memory_type']
       output_memory_type = step_plan['output_memory_type']
       gpu_id = step_plan['gpu_id']

       # Load and stack with input memory type
       image_stack = stack_slices(
           slices=raw_slices,
           memory_type=input_memory_type,
           gpu_id=gpu_id
       )

       # Execute function (operates in native memory type)
       result_stack = func(image_stack)

       # Unstack with output memory type
       output_slices = unstack_slices(
           array=result_stack,
           memory_type=output_memory_type,
           gpu_id=gpu_id
       )

Function Decorator Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   @torch(input_type="torch", output_type="torch")
   def my_gpu_function(image_stack):
       # Receives torch tensor on GPU
       # Returns torch tensor on GPU
       return processed_stack

   @numpy
   def my_cpu_function(image_stack):
       # Receives numpy array
       # Returns numpy array
       return processed_stack

**Compiler Integration**: - Memory types extracted from function
decorators during compilation - Injected into step plans as
``input_memory_type`` and ``output_memory_type`` - Used by stack utils
for automatic conversion

Cross-Step Memory Type Coordination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Pipeline with mixed memory types
   pipeline = [
       FunctionStep(func=cpu_preprocessing),     # numpy → numpy
       FunctionStep(func=gpu_processing),        # torch → torch
       FunctionStep(func=cpu_postprocessing)     # numpy → numpy
   ]

   # Compiler generates step plans with automatic conversions:
   # Step 1: disk(tiff) → numpy → numpy → memory
   # Step 2: memory → torch → torch → memory
   # Step 3: memory → numpy → numpy → disk(tiff)

**Automatic Conversion Points**: - Between steps with different memory
types - When reading from disk (usually numpy) - When writing to disk
(usually numpy) - During special I/O operations

Error Handling and Validation
-----------------------------

Common Errors
~~~~~~~~~~~~~

Dimensional Validation Errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   ValueError: Slice at index 0 is not a 2D array. All slices must be 2D.
   ValueError: Array must be 3D, got shape (512, 512)

Memory Type Errors
^^^^^^^^^^^^^^^^^^

.. code:: python

   ValueError: Could not detect memory type of <class 'list'>
   ValueError: Unsupported memory type: invalid_type

GPU Device Errors
^^^^^^^^^^^^^^^^^

.. code:: python

   ValueError: Invalid GPU device ID: -1. Must be a non-negative integer.
   MemoryConversionError: Failed to move tensor to device 5: device not available

Real-World Examples
~~~~~~~~~~~~~~~~~~~

**Actual OpenHCS Functions** (from current codebase):

.. code:: python

   # NumPy CPU processing (from openhcs/processing/backends/processors/numpy_processor.py)
   @numpy_func
   def max_projection(stack: np.ndarray) -> np.ndarray:
       """Create a maximum intensity projection from a Z-stack."""
       _validate_3d_array(stack)
       projection_2d = np.max(stack, axis=0)
       return projection_2d.reshape(1, projection_2d.shape[0], projection_2d.shape[1])

   # CuPy GPU processing (from openhcs/processing/backends/processors/cupy_processor.py)
   @cupy_func
   def tophat(image: "cp.ndarray", selem_radius: int = 50) -> "cp.ndarray":
       """Apply morphological top-hat filter using CuPy GPU acceleration."""
       # GPU-accelerated morphological operations
       return processed_image

   # PyTorch GPU processing (from openhcs/processing/backends/processors/torch_processor.py)
   @torch_func
   def stack_percentile_normalize(stack: "torch.Tensor",
                                  low_percentile: float = 1.0,
                                  high_percentile: float = 99.0) -> "torch.Tensor":
       """Normalize image stack using percentile-based scaling."""
       # PyTorch GPU tensor operations
       return normalized_stack

Best Practices
~~~~~~~~~~~~~~

Function Development
^^^^^^^^^^^^^^^^^^^^

-  Always use memory type decorators
-  Test functions with different memory types
-  Validate input/output shapes explicitly
-  Handle GPU device availability gracefully

Pipeline Design
^^^^^^^^^^^^^^^

-  Minimize memory type conversions
-  Use GPU types only when beneficial
-  Consider memory usage for large datasets
-  Plan GPU resource allocation carefully

Debugging
^^^^^^^^^

-  Check function decorators are properly applied
-  Validate memory types in step plans
-  Monitor GPU memory usage
-  Use strict validation during development

Performance Considerations
--------------------------

Memory Type Selection Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def select_memory_type(data_size, has_gpu, processing_type):
       """Intelligent memory type selection."""
       if processing_type == "deep_learning":
           return "torch" if has_gpu else "numpy"
       elif processing_type == "array_operations" and has_gpu:
           return "cupy"
       elif processing_type == "machine_learning":
           return "tensorflow" if has_gpu else "numpy"
       else:
           return "numpy"  # Safe default

Conversion Optimization
~~~~~~~~~~~~~~~~~~~~~~~

-  **Minimize Conversions**: Keep data in same memory type when possible
-  **Batch Operations**: Group operations by memory type
-  **GPU Memory Management**: Monitor GPU memory usage
-  **Lazy Conversion**: Convert only when necessary

Memory Usage Patterns
~~~~~~~~~~~~~~~~~~~~~

-  **Small Data**: Use numpy for simplicity
-  **Large Data + GPU**: Use cupy/torch for performance
-  **Deep Learning**: Use torch/tensorflow
-  **Research/HPC**: Consider JAX for advanced optimizations

Future Enhancements
-------------------

Planned Features
~~~~~~~~~~~~~~~~

-  **Automatic Memory Type Selection**: Based on data size and available
   resources
-  **Memory Pool Management**: Efficient GPU memory reuse
-  **Distributed Memory Types**: Support for multi-GPU and multi-node
   processing
-  **Memory Type Profiling**: Performance analysis and optimization
   recommendations

Integration Improvements
~~~~~~~~~~~~~~~~~~~~~~~~

-  **Lazy Loading**: Load data only when needed in target memory type
-  **Streaming Processing**: Handle datasets larger than memory
-  **Automatic Batching**: Split large arrays for memory-constrained
   processing
-  **Memory Type Caching**: Cache converted data for reuse

See Also
--------

**Core Integration**:

- :doc:`function_pattern_system` - Function patterns and memory type integration
- :doc:`function_registry_system` - Function discovery with memory type contracts
- :doc:`pipeline_compilation_system` - Memory type validation during compilation

**Practical Usage**:

- :doc:`../api/processing_backends` - Processing functions with memory type decorators
- :doc:`../guides/memory_type_integration` - Complete memory type integration guide
- :doc:`../api/function_step` - FunctionStep memory type handling

**Advanced Topics**:

- :doc:`gpu_resource_management` - GPU device management and allocation
- :doc:`concurrency_model` - Multi-processing with GPU memory types
- :doc:`compilation_system_detailed` - Memory contract validation details

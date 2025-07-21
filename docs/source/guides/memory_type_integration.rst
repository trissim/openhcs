Memory Type System Integration
==============================

OpenHCS provides a sophisticated memory type system that enables seamless conversion between different array libraries while maintaining strict dimensional constraints and GPU device discipline.

Core Concepts
-------------

**Memory Type Decorators**: Functions declare their memory interface using decorators
**Automatic Conversion**: OpenHCS automatically converts between memory types during pipeline execution
**GPU Device Discipline**: Strict device placement and validation for GPU operations
**3D Enforcement**: All functions must return 3D arrays of shape [Z, Y, X]

Available Memory Types
----------------------

OpenHCS supports six memory types with automatic conversion:

.. list-table::
   :header-rows: 1

   * - Memory Type
     - Library
     - GPU Support
     - Use Cases
   * - ``numpy``
     - NumPy
     - No
     - CPU processing, I/O operations
   * - ``cupy``
     - CuPy
     - Yes
     - GPU-accelerated NumPy-like operations
   * - ``torch``
     - PyTorch
     - Yes
     - Deep learning, neural networks
   * - ``tensorflow``
     - TensorFlow
     - Yes
     - Machine learning, TensorFlow models
   * - ``jax``
     - JAX
     - Yes
     - High-performance computing, research
   * - ``pyclesperanto``
     - pyclesperanto
     - Yes
     - GPU-accelerated image processing

Memory Type Decorators
----------------------

Functions declare their memory interface using decorators from ``openhcs.core.memory.decorators``:

Basic Usage
^^^^^^^^^^^

.. code-block:: python

    from openhcs.core.memory.decorators import numpy, cupy, torch, jax, pyclesperanto

    @numpy
    def process_cpu(image_stack):
        """CPU processing with NumPy arrays."""
        import numpy as np
        return np.median(image_stack, axis=0, keepdims=True)

    @cupy
    def process_gpu_cupy(image_stack):
        """GPU processing with CuPy arrays."""
        import cupy as cp
        return cp.median(image_stack, axis=0, keepdims=True)

    @torch
    def process_gpu_torch(image_stack):
        """GPU processing with PyTorch tensors."""
        import torch
        return torch.median(image_stack, dim=0, keepdim=True)[0]

Advanced Memory Type Specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.core.memory.decorators import memory_types

    # Mixed input/output types
    @memory_types(input_type="numpy", output_type="torch")
    def neural_network_inference(image_stack):
        """Convert NumPy input to PyTorch for GPU inference."""
        import torch
        # Function receives NumPy array, returns PyTorch tensor
        model = torch.load('model.pt')
        return model(image_stack)

    # Explicit type specification
    @torch(input_type="torch", output_type="torch", oom_recovery=True)
    def memory_intensive_operation(image_stack):
        """GPU operation with automatic OOM recovery."""
        # Automatic GPU memory management
        return torch.nn.functional.conv3d(image_stack, kernel)

Automatic Memory Conversion
----------------------------

OpenHCS automatically handles memory type conversion during pipeline execution:

Pipeline Example
^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.processing.backends.processors.cupy_processor import tophat
    from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel

    # Mixed memory types in single pipeline
    steps = [
        # Step 1: PyTorch GPU normalization
        FunctionStep(
            func=stack_percentile_normalize,  # @torch decorated
            low_percentile=1.0,
            high_percentile=99.0,
            name="normalize"
        ),
        
        # Step 2: CuPy GPU morphology
        FunctionStep(
            func=tophat,  # @cupy decorated
            selem_radius=50,
            name="tophat"
        ),
        
        # Step 3: NumPy CPU analysis
        FunctionStep(
            func=count_cells_single_channel,  # @numpy decorated
            detection_method="blob_log",
            name="count_cells"
        )
    ]

    # OpenHCS automatically converts:
    # Input (any type) → torch → cupy → numpy → output

Conversion Flow
^^^^^^^^^^^^^^^

1. **Input Detection**: OpenHCS detects the memory type of input data
2. **Target Conversion**: Converts to the function's declared input type
3. **Function Execution**: Function operates in its native memory type
4. **Output Conversion**: Converts output to next function's input type

.. code-block:: python

    # Automatic conversion example
    numpy_array = load_image()  # NumPy array from disk
    
    # Step 1: numpy → torch conversion (automatic)
    torch_result = torch_function(numpy_array)
    
    # Step 2: torch → cupy conversion (automatic)  
    cupy_result = cupy_function(torch_result)
    
    # Step 3: cupy → numpy conversion (automatic)
    final_result = numpy_function(cupy_result)

GPU Device Management
---------------------

OpenHCS provides automatic GPU device management with thread-local CUDA streams:

Thread-Local GPU Streams
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    @cupy(oom_recovery=True)
    def gpu_intensive_cupy(image_stack):
        """Each thread gets its own CUDA stream."""
        import cupy as cp
        # Automatic thread-local stream management
        # OOM recovery automatically enabled
        return cp.median(image_stack, axis=0, keepdims=True)

    @torch(oom_recovery=True)
    def gpu_intensive_torch(image_stack):
        """PyTorch with automatic OOM recovery."""
        import torch
        # Automatic CUDA stream management
        # Memory cleanup on OOM
        return torch.median(image_stack, dim=0, keepdim=True)[0]

Zero-Copy GPU Conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenHCS uses advanced GPU interoperability for efficient conversions:

.. code-block:: python

    # Zero-copy conversions when possible:
    # CuPy ↔ PyTorch: CUDA Array Interface
    # PyTorch ↔ JAX: DLPack protocol
    # TensorFlow ↔ JAX: DLPack protocol
    
    @cupy
    def cupy_function(data):
        return data * 2
    
    @torch  
    def torch_function(data):
        return data + 1
    
    # Conversion uses CUDA Array Interface (zero-copy)
    pipeline = [cupy_function, torch_function]

Memory Type Validation
----------------------

OpenHCS enforces strict memory type validation:

Validation Examples
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.core.memory.wrapper import MemoryWrapper
    from openhcs.constants.constants import MemoryType

    # Strict memory type detection
    def validate_memory_type(data):
        """Detect and validate memory type."""
        detected_type = MemoryWrapper.detect_memory_type(data)
        
        if detected_type == MemoryType.UNKNOWN:
            raise ValueError("Unknown memory type - cannot process")
        
        return detected_type

    # GPU device validation
    @torch(input_type="torch", output_type="torch")
    def gpu_only_function(image_stack):
        """Function requires GPU tensor."""
        if not image_stack.is_cuda:
            raise ValueError("Function requires CUDA tensor")
        return image_stack.median(dim=0, keepdim=True)[0]

Error Handling and Recovery
---------------------------

OpenHCS provides comprehensive error handling for memory operations:

OOM Recovery
^^^^^^^^^^^^

.. code-block:: python

    @torch(oom_recovery=True)
    def memory_intensive_function(large_image_stack):
        """Automatic GPU memory recovery on OOM."""
        try:
            # Large GPU operation
            result = torch.nn.functional.conv3d(large_image_stack, large_kernel)
            return result
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Automatic memory cleanup and retry
                torch.cuda.empty_cache()
                # Function automatically retried with smaller batch
                pass
            raise

Conversion Error Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.core.memory.conversion_functions import MemoryConversionError

    try:
        # Attempt GPU-to-GPU conversion
        converted_data = convert_cupy_to_torch(cupy_array, allow_cpu_roundtrip=False)
    except MemoryConversionError as e:
        # Handle conversion failure
        print(f"Conversion failed: {e.reason}")
        # Fallback to CPU roundtrip if allowed
        converted_data = convert_cupy_to_torch(cupy_array, allow_cpu_roundtrip=True)

Best Practices
--------------

**Function Design**:
- Always use memory type decorators
- Return 3D arrays [Z, Y, X] even for 2D operations
- Enable OOM recovery for GPU functions

**Pipeline Design**:
- Group functions by memory type when possible
- Use GPU types for compute-intensive operations
- Use NumPy for I/O and simple operations

**Performance Optimization**:
- Minimize memory type conversions
- Use zero-copy conversions when available
- Enable thread-local GPU streams for parallelization

See Also
--------

- :doc:`../architecture/memory_type_system` - Detailed memory type architecture
- :doc:`../api/processing_backends` - Processing function examples
- :doc:`../architecture/function_registry_system` - Function discovery and registration

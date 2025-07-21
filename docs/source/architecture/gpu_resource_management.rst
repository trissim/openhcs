GPU Resource Management System
==============================

Overview
--------

OpenHCS implements a GPU resource management system that coordinates GPU
device allocation during pipeline compilation. The system provides GPU
detection, registry initialization, and compilation-time GPU assignment
to ensure consistent GPU usage across pipeline steps.

**Note**: This document describes the actual GPU management
implementation. Runtime load balancing and slot acquisition features are
planned for future development.

Architecture Components
-----------------------

GPU Registry Singleton
~~~~~~~~~~~~~~~~~~~~~~

The core of the system is a thread-safe global GPU registry:

.. code:: python

   # Global GPU registry structure
   GPU_REGISTRY: Dict[int, Dict[str, int]] = {
       0: {"max_pipelines": 2, "active": 0},  # GPU 0 can handle 2 concurrent pipelines
       1: {"max_pipelines": 2, "active": 0},  # GPU 1 can handle 2 concurrent pipelines
       # ... more GPUs
   }

   # Thread safety
   _registry_lock = threading.Lock()
   _registry_initialized = False

Registry Initialization
~~~~~~~~~~~~~~~~~~~~~~~

The registry is initialized once during application startup:

.. code:: python

   def setup_global_gpu_registry(global_config: Optional[GlobalPipelineConfig] = None) -> None:
       """Initialize GPU registry using global configuration."""

       config_to_use = global_config or get_default_global_config()
       initialize_gpu_registry(configured_num_workers=config_to_use.num_workers)

   def initialize_gpu_registry(configured_num_workers: int) -> None:
       """Initialize GPU registry based on available hardware."""

       global GPU_REGISTRY, _registry_initialized

       with _registry_lock:
           if _registry_initialized:
               raise RuntimeError("GPU registry already initialized")

           # 1. Detect available GPUs
           available_gpus = _detect_available_gpus()
           logger.info(f"Detected GPUs: {available_gpus}")

           if not available_gpus:
               logger.warning("No GPUs detected. GPU memory types will not be available.")
               _registry_initialized = True
               GPU_REGISTRY.clear()
               return

           # 2. Calculate max concurrent pipelines per GPU
           max_cpu_threads = os.cpu_count() or configured_num_workers
           pipelines_per_gpu = max(1, math.ceil(max_cpu_threads / len(available_gpus)))

           # 3. Initialize registry (simplified structure)
           GPU_REGISTRY.clear()
           for gpu_id in available_gpus:
               GPU_REGISTRY[gpu_id] = {"max_pipelines": pipelines_per_gpu}

           _registry_initialized = True
           logger.info(f"GPU registry initialized: {GPU_REGISTRY}")

GPU Detection
~~~~~~~~~~~~~

Multi-library GPU detection with fallback strategy:

.. code:: python

   def _detect_available_gpus() -> List[int]:
       """Detect available GPUs across multiple libraries."""

       available_gpus = set()

       # Check PyTorch GPUs
       torch_gpu = check_torch_gpu_available()
       if torch_gpu is not None:
           available_gpus.add(torch_gpu)

       # Check CuPy GPUs
       cupy_gpu = check_cupy_gpu_available()
       if cupy_gpu is not None:
           available_gpus.add(cupy_gpu)

       # Check TensorFlow GPUs
       tf_gpu = check_tf_gpu_available()
       if tf_gpu is not None:
           available_gpus.add(tf_gpu)

       # Check JAX GPUs
       jax_gpu = check_jax_gpu_available()
       if jax_gpu is not None:
           available_gpus.add(jax_gpu)

       return sorted(list(available_gpus))

   def check_torch_gpu_available() -> Optional[int]:
       """Check PyTorch GPU availability."""
       try:
           import torch
           if torch.cuda.is_available():
               return torch.cuda.current_device()
       except Exception:
           pass
       return None

   def check_cupy_gpu_available() -> Optional[int]:
       """Check CuPy GPU availability."""
       try:
           import cupy
           if cupy.cuda.is_available():
               return cupy.cuda.get_device_id()
       except Exception:
           pass
       return None

GPU Allocation Strategy
-----------------------

Compilation-Time Assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU devices are assigned during pipeline compilation, not execution:

.. code:: python

   class GPUMemoryTypeValidator:
       """Validates GPU memory types and assigns GPU devices."""

       @staticmethod
       def validate_step_plans(step_plans: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
           """Validate GPU memory types and assign GPU IDs."""

           # 1. Check if any step requires GPU
           requires_gpu = any(
               step_plan.get('input_memory_type') in VALID_GPU_MEMORY_TYPES or
               step_plan.get('output_memory_type') in VALID_GPU_MEMORY_TYPES
               for step_plan in step_plans.values()
           )

           if not requires_gpu:
               return {}  # No GPU assignment needed

           # 2. Get GPU registry status
           gpu_registry = get_gpu_registry_status()
           if not gpu_registry:
               raise ValueError(
                   "🔥 COMPILATION FAILED: No GPUs available in registry but pipeline contains GPU-decorated functions!"
               )

           # 3. Assign first available GPU (simplified assignment)
           # All steps in pipeline use same GPU for affinity
           gpu_id = list(gpu_registry.keys())[0]

           # 4. Assign GPU to all GPU-requiring steps
           for step_id, step_plan in step_plans.items():
               input_type = step_plan.get('input_memory_type')
               output_type = step_plan.get('output_memory_type')

               if (input_type in VALID_GPU_MEMORY_TYPES or
                   output_type in VALID_GPU_MEMORY_TYPES):

                   step_plan['gpu_id'] = gpu_id
                   logger.debug(
                       "Step %s assigned gpu_id %s for memory types: %s/%s",
                       step_id, gpu_id, input_type, output_type
                   )

           return {}  # No additional assignments needed

GPU Affinity Strategy
~~~~~~~~~~~~~~~~~~~~~

All steps in a pipeline use the same GPU for optimal performance:

.. code:: python

   # GPU affinity is automatically enforced during compilation
   # All GPU-requiring steps in a pipeline receive the same gpu_id
   # This ensures optimal memory locality and reduces GPU context switching

Registry Status Access
----------------------

GPU Registry Status
~~~~~~~~~~~~~~~~~~~

.. code:: python

   def get_gpu_registry_status() -> Dict[int, Dict[str, int]]:
       """Get current GPU registry status."""

       with _registry_lock:
           if not _registry_initialized:
               return {}

           # Return deep copy to prevent external modification
           return {
               gpu_id: info.copy()
               for gpu_id, info in GPU_REGISTRY.items()
           }

   def is_gpu_registry_initialized() -> bool:
       """Check if the GPU registry has been initialized."""

       with _registry_lock:
           return _registry_initialized

Memory Type Integration
-----------------------

GPU Memory Type Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The system validates GPU memory types against available hardware:

.. code:: python

   # GPU memory types that require GPU devices
   VALID_GPU_MEMORY_TYPES = {"cupy", "torch", "tensorflow", "jax", "pyclesperanto"}

   # Validation is performed during compilation by GPUMemoryTypeValidator
   # Library-specific validation ensures GPU compatibility before execution

Current Implementation Status
-----------------------------

Implemented Features
~~~~~~~~~~~~~~~~~~~~

-  ✅ GPU registry initialization and detection
-  ✅ Compilation-time GPU assignment
-  ✅ GPU affinity enforcement (same GPU per pipeline)
-  ✅ Multi-library GPU detection (PyTorch, CuPy, TensorFlow, JAX)
-  ✅ Thread-safe registry access

Future Enhancements
~~~~~~~~~~~~~~~~~~~

1. **Runtime GPU Slot Management**: Dynamic GPU slot acquisition/release
   during execution
2. **Load Balancing**: Intelligent GPU assignment based on current
   utilization
3. **GPU Memory Monitoring**: Real-time memory usage tracking and
   optimization
4. **Error Handling**: GPU failure detection and recovery mechanisms
5. **Multi-Node GPU Management**: Coordinate GPUs across multiple
   machines
6. **Performance Profiling**: Detailed GPU performance metrics and
   recommendations

# GPU Resource Management System

## Overview

OpenHCS implements a sophisticated GPU resource management system that coordinates GPU device allocation across multiple wells and pipeline steps. The system ensures optimal GPU utilization while preventing resource conflicts and memory exhaustion.

## Architecture Components

### GPU Registry Singleton

The core of the system is a thread-safe global GPU registry:

```python
# Global GPU registry structure
GPU_REGISTRY: Dict[int, Dict[str, int]] = {
    0: {"max_pipelines": 2, "active": 0},  # GPU 0 can handle 2 concurrent pipelines
    1: {"max_pipelines": 2, "active": 0},  # GPU 1 can handle 2 concurrent pipelines
    # ... more GPUs
}

# Thread safety
_registry_lock = threading.Lock()
_registry_initialized = False
```

### Registry Initialization

The registry is initialized once during application startup:

```python
def initialize_gpu_registry(configured_num_workers: int) -> None:
    """Initialize GPU registry based on available hardware."""
    
    global GPU_REGISTRY, _registry_initialized
    
    with _registry_lock:
        if _registry_initialized:
            raise RuntimeError("GPU registry already initialized")
        
        # 1. Detect available GPUs
        available_gpus = detect_available_gpus()
        logger.info(f"Detected GPUs: {available_gpus}")
        
        if not available_gpus:
            logger.warning("No GPUs detected. GPU memory types will not be available.")
            _registry_initialized = True
            return
        
        # 2. Calculate pipelines per GPU
        cpu_count = os.cpu_count() or configured_num_workers
        pipelines_per_gpu = max(1, cpu_count // len(available_gpus))
        
        # 3. Initialize registry
        GPU_REGISTRY.clear()
        for gpu_id in available_gpus:
            GPU_REGISTRY[gpu_id] = {
                "max_pipelines": pipelines_per_gpu,
                "active": 0
            }
        
        _registry_initialized = True
        logger.info(f"GPU registry initialized: {GPU_REGISTRY}")
```

### GPU Detection

Multi-library GPU detection with fallback strategy:

```python
def detect_available_gpus() -> List[int]:
    """Detect available GPUs across multiple libraries."""
    
    available_gpus = set()
    
    # Check PyTorch GPUs
    try:
        torch_gpus = check_torch_gpu_available()
        if torch_gpus:
            available_gpus.update(torch_gpus)
    except Exception as e:
        logger.debug("PyTorch GPU detection failed: %s", e)
    
    # Check CuPy GPUs
    try:
        cupy_gpus = check_cupy_gpu_available()
        if cupy_gpus:
            available_gpus.update(cupy_gpus)
    except Exception as e:
        logger.debug("CuPy GPU detection failed: %s", e)
    
    # Check TensorFlow GPUs
    try:
        tf_gpus = check_tensorflow_gpu_available()
        if tf_gpus:
            available_gpus.update(tf_gpus)
    except Exception as e:
        logger.debug("TensorFlow GPU detection failed: %s", e)
    
    # Check JAX GPUs
    try:
        jax_gpu = check_jax_gpu_available()
        if jax_gpu is not None:
            available_gpus.add(jax_gpu)
    except Exception as e:
        logger.debug("JAX GPU detection failed: %s", e)
    
    return sorted(list(available_gpus))

def check_torch_gpu_available() -> List[int]:
    """Check PyTorch GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except ImportError:
        pass
    return []

def check_cupy_gpu_available() -> List[int]:
    """Check CuPy GPU availability."""
    try:
        import cupy
        return list(range(cupy.cuda.runtime.getDeviceCount()))
    except (ImportError, Exception):
        pass
    return []
```

## GPU Allocation Strategy

### Compilation-Time Assignment

GPU devices are assigned during pipeline compilation, not execution:

```python
class GPUMemoryTypeValidator:
    """Validates GPU memory types and assigns GPU devices."""
    
    @staticmethod
    def validate_step_plans(step_plans: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Validate GPU memory types and assign GPU IDs."""
        
        # 1. Check if any step requires GPU
        requires_gpu = False
        for step_id, step_plan in step_plans.items():
            input_type = step_plan.get('input_memory_type')
            output_type = step_plan.get('output_memory_type')
            
            if (input_type in VALID_GPU_MEMORY_TYPES or 
                output_type in VALID_GPU_MEMORY_TYPES):
                requires_gpu = True
                break
        
        if not requires_gpu:
            return {}  # No GPU assignment needed
        
        # 2. Get GPU registry status
        gpu_registry = get_gpu_registry_status()
        if not gpu_registry:
            raise ValueError(
                "Clause 293 Violation: No GPUs available for assignment. "
                "Cannot validate GPU memory types."
            )
        
        # 3. Find least loaded GPU
        least_loaded_gpu = min(
            gpu_registry.items(),
            key=lambda x: x[1]['active'] / x[1]['max_pipelines']
        )[0]
        
        # 4. Assign GPU to all GPU-requiring steps
        gpu_assignments = {}
        for step_id, step_plan in step_plans.items():
            input_type = step_plan.get('input_memory_type')
            output_type = step_plan.get('output_memory_type')
            
            if (input_type in VALID_GPU_MEMORY_TYPES or 
                output_type in VALID_GPU_MEMORY_TYPES):
                
                step_plan['gpu_id'] = least_loaded_gpu
                gpu_assignments[step_id] = {"gpu_id": least_loaded_gpu}
                
                logger.debug(
                    "Step %s assigned gpu_id %s for memory types: %s/%s",
                    step_id, least_loaded_gpu, input_type, output_type
                )
        
        return gpu_assignments
```

### GPU Affinity Strategy

All steps in a pipeline use the same GPU for optimal performance:

```python
def assign_gpu_affinity(step_plans):
    """Ensure all GPU steps in pipeline use same GPU."""
    
    assigned_gpu = None
    
    # Find first GPU-requiring step
    for step_id, step_plan in step_plans.items():
        if 'gpu_id' in step_plan:
            assigned_gpu = step_plan['gpu_id']
            break
    
    if assigned_gpu is not None:
        # Assign same GPU to all GPU-requiring steps
        for step_id, step_plan in step_plans.items():
            input_type = step_plan.get('input_memory_type')
            output_type = step_plan.get('output_memory_type')
            
            if (input_type in VALID_GPU_MEMORY_TYPES or 
                output_type in VALID_GPU_MEMORY_TYPES):
                step_plan['gpu_id'] = assigned_gpu
```

## Runtime GPU Management

### GPU Slot Acquisition

During parallel execution, threads acquire GPU slots:

```python
def acquire_gpu_slot() -> Optional[int]:
    """Acquire a GPU slot for pipeline execution."""
    
    with _registry_lock:
        if not _registry_initialized:
            raise RuntimeError(
                "Clause 295 Violation: GPU registry not initialized. "
                "Must call initialize_gpu_registry() first."
            )
        
        # Find first GPU with available slots
        for gpu_id, info in GPU_REGISTRY.items():
            if info["active"] < info["max_pipelines"]:
                # Increment active count
                info["active"] += 1
                logger.debug(
                    "Acquired GPU %s. Active pipelines: %s/%s",
                    gpu_id, info["active"], info["max_pipelines"]
                )
                return gpu_id
        
        # No slots available
        logger.warning("No GPU slots available. All GPUs are at maximum capacity.")
        return None

def release_gpu_slot(gpu_id: int) -> None:
    """Release a GPU slot after pipeline completion."""
    
    with _registry_lock:
        if not _registry_initialized:
            raise RuntimeError("GPU registry not initialized")
        
        if gpu_id not in GPU_REGISTRY:
            raise ValueError(f"Invalid GPU ID: {gpu_id}")
        
        if GPU_REGISTRY[gpu_id]["active"] <= 0:
            raise ValueError(f"GPU {gpu_id} has no active pipelines to release")
        
        # Decrement active count
        GPU_REGISTRY[gpu_id]["active"] -= 1
        logger.debug(
            "Released GPU %s. Active pipelines: %s/%s",
            gpu_id, GPU_REGISTRY[gpu_id]["active"], GPU_REGISTRY[gpu_id]["max_pipelines"]
        )
```

### Execution Integration

GPU slots are managed during pipeline execution:

```python
def _execute_single_well(self, pipeline_definition, context, visualizer):
    """Execute pipeline for single well with GPU management."""
    
    # Check if pipeline requires GPU
    requires_gpu = any(
        step_plan.get('gpu_id') is not None 
        for step_plan in context.step_plans.values()
    )
    
    acquired_gpu = None
    
    try:
        if requires_gpu:
            # Acquire GPU slot
            acquired_gpu = acquire_gpu_slot()
            if acquired_gpu is None:
                raise RuntimeError("No GPU slots available for execution")
            
            logger.debug(f"Well {context.well_id} acquired GPU {acquired_gpu}")
        
        # Execute pipeline steps
        for step in pipeline_definition:
            step.process(context)
        
        return {"status": "success", "well_id": context.well_id}
        
    except Exception as e:
        logger.error(f"Pipeline execution failed for well {context.well_id}: {e}")
        return {"status": "error", "well_id": context.well_id, "error": str(e)}
        
    finally:
        # Always release GPU slot
        if acquired_gpu is not None:
            release_gpu_slot(acquired_gpu)
            logger.debug(f"Well {context.well_id} released GPU {acquired_gpu}")
```

## Memory Type Integration

### GPU Memory Type Validation

The system validates GPU memory types against available hardware:

```python
# GPU memory types that require GPU devices
VALID_GPU_MEMORY_TYPES = {"cupy", "torch", "tensorflow", "jax"}

def validate_gpu_memory_type(memory_type: str, gpu_id: int):
    """Validate GPU memory type against hardware."""
    
    if memory_type not in VALID_GPU_MEMORY_TYPES:
        return True  # CPU memory types don't need validation
    
    # Check if GPU is available
    if gpu_id is None:
        raise ValueError(f"GPU memory type '{memory_type}' requires gpu_id")
    
    # Validate GPU device exists
    gpu_registry = get_gpu_registry_status()
    if gpu_id not in gpu_registry:
        raise ValueError(f"GPU device {gpu_id} not available")
    
    # Library-specific validation
    if memory_type == "torch":
        import torch
        if not torch.cuda.is_available():
            raise ValueError("PyTorch CUDA not available")
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(f"PyTorch GPU {gpu_id} not available")
    
    elif memory_type == "cupy":
        import cupy
        if gpu_id >= cupy.cuda.runtime.getDeviceCount():
            raise ValueError(f"CuPy GPU {gpu_id} not available")
    
    # ... other library validations
    
    return True
```

### Device Placement Coordination

GPU device placement is coordinated with memory type conversion:

```python
def convert_with_device_placement(data, target_memory_type, gpu_id):
    """Convert data with explicit GPU device placement."""
    
    if target_memory_type == "torch":
        import torch
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
            if gpu_id is not None:
                tensor = tensor.to(f"cuda:{gpu_id}")
            return tensor
    
    elif target_memory_type == "cupy":
        import cupy
        if gpu_id is not None:
            with cupy.cuda.Device(gpu_id):
                return cupy.asarray(data)
        else:
            return cupy.asarray(data)
    
    # ... other conversions
```

## Performance Optimization

### Load Balancing

The system implements intelligent load balancing across GPUs:

```python
def get_optimal_gpu_assignment(step_plans_list):
    """Get optimal GPU assignment for multiple pipelines."""
    
    gpu_registry = get_gpu_registry_status()
    if not gpu_registry:
        return {}
    
    # Calculate load per GPU
    gpu_loads = {
        gpu_id: info['active'] / info['max_pipelines']
        for gpu_id, info in gpu_registry.items()
    }
    
    # Sort GPUs by load (least loaded first)
    sorted_gpus = sorted(gpu_loads.items(), key=lambda x: x[1])
    
    # Assign pipelines to least loaded GPUs
    assignments = {}
    gpu_index = 0
    
    for i, step_plans in enumerate(step_plans_list):
        requires_gpu = any(
            step_plan.get('input_memory_type') in VALID_GPU_MEMORY_TYPES or
            step_plan.get('output_memory_type') in VALID_GPU_MEMORY_TYPES
            for step_plan in step_plans.values()
        )
        
        if requires_gpu:
            gpu_id = sorted_gpus[gpu_index % len(sorted_gpus)][0]
            assignments[i] = gpu_id
            gpu_index += 1
    
    return assignments
```

### Memory Management

```python
def monitor_gpu_memory_usage():
    """Monitor GPU memory usage across all devices."""
    
    memory_stats = {}
    
    for gpu_id in GPU_REGISTRY.keys():
        try:
            # PyTorch memory stats
            import torch
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                torch.cuda.set_device(gpu_id)
                allocated = torch.cuda.memory_allocated(gpu_id)
                cached = torch.cuda.memory_reserved(gpu_id)
                
                memory_stats[gpu_id] = {
                    "allocated_mb": allocated / 1024 / 1024,
                    "cached_mb": cached / 1024 / 1024,
                    "library": "torch"
                }
        
        except Exception as e:
            logger.debug(f"Failed to get memory stats for GPU {gpu_id}: {e}")
    
    return memory_stats
```

## Error Handling and Recovery

### GPU Failure Handling

```python
def handle_gpu_failure(gpu_id: int, error: Exception):
    """Handle GPU failure during execution."""
    
    logger.error(f"GPU {gpu_id} failure: {error}")
    
    with _registry_lock:
        if gpu_id in GPU_REGISTRY:
            # Mark GPU as unavailable
            GPU_REGISTRY[gpu_id]["max_pipelines"] = 0
            
            # Release any active slots
            active_count = GPU_REGISTRY[gpu_id]["active"]
            GPU_REGISTRY[gpu_id]["active"] = 0
            
            logger.warning(
                f"GPU {gpu_id} marked as unavailable. "
                f"Released {active_count} active slots."
            )

def recover_gpu_if_available(gpu_id: int):
    """Attempt to recover a failed GPU."""
    
    try:
        # Test GPU availability
        if test_gpu_functionality(gpu_id):
            with _registry_lock:
                if gpu_id in GPU_REGISTRY:
                    # Restore GPU capacity
                    original_capacity = calculate_gpu_capacity(gpu_id)
                    GPU_REGISTRY[gpu_id]["max_pipelines"] = original_capacity
                    
                    logger.info(f"GPU {gpu_id} recovered and restored to service")
                    return True
    
    except Exception as e:
        logger.debug(f"GPU {gpu_id} recovery failed: {e}")
    
    return False
```

### Resource Exhaustion Handling

```python
def handle_gpu_memory_exhaustion(gpu_id: int):
    """Handle GPU memory exhaustion."""
    
    logger.warning(f"GPU {gpu_id} memory exhaustion detected")
    
    # Attempt memory cleanup
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            logger.info(f"Cleared GPU {gpu_id} cache")
    
    except Exception as e:
        logger.debug(f"Failed to clear GPU {gpu_id} cache: {e}")
    
    # Temporarily reduce capacity
    with _registry_lock:
        if gpu_id in GPU_REGISTRY:
            current_max = GPU_REGISTRY[gpu_id]["max_pipelines"]
            if current_max > 1:
                GPU_REGISTRY[gpu_id]["max_pipelines"] = current_max - 1
                logger.warning(
                    f"Reduced GPU {gpu_id} capacity from {current_max} to {current_max - 1}"
                )
```

## Configuration and Monitoring

### Registry Status Monitoring

```python
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

def get_gpu_utilization_stats():
    """Get GPU utilization statistics."""
    
    registry = get_gpu_registry_status()
    
    stats = {
        "total_gpus": len(registry),
        "total_capacity": sum(info["max_pipelines"] for info in registry.values()),
        "total_active": sum(info["active"] for info in registry.values()),
        "utilization_percent": 0.0,
        "per_gpu": {}
    }
    
    if stats["total_capacity"] > 0:
        stats["utilization_percent"] = (stats["total_active"] / stats["total_capacity"]) * 100
    
    for gpu_id, info in registry.items():
        utilization = (info["active"] / info["max_pipelines"]) * 100 if info["max_pipelines"] > 0 else 0
        stats["per_gpu"][gpu_id] = {
            "active": info["active"],
            "capacity": info["max_pipelines"],
            "utilization_percent": utilization
        }
    
    return stats
```

## Future Enhancements

### Planned Features

1. **Dynamic GPU Scaling**: Automatically adjust GPU capacity based on workload
2. **GPU Memory Pooling**: Shared memory pools across pipeline instances
3. **Multi-Node GPU Management**: Coordinate GPUs across multiple machines
4. **GPU Performance Profiling**: Detailed performance metrics and optimization recommendations
5. **Intelligent GPU Selection**: ML-based GPU assignment based on workload characteristics

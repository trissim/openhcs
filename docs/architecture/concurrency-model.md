# OpenHCS Concurrency Model

## Overview

OpenHCS implements a sophisticated **well-level parallelism** model with **strict thread isolation** and **immutable compilation artifacts**. This design provides excellent performance while maintaining thread safety through architectural constraints rather than complex locking mechanisms.

## **🎯 Core Concurrency Philosophy**

### **Well-Level Parallelism**
- **Parallel Unit**: Each well is processed independently in its own thread
- **Sequential Within Well**: All steps for a well execute sequentially in the same thread
- **No Cross-Well Dependencies**: Wells never share data or coordinate during execution

### **Immutable Compilation Artifacts**
- **Compile Once, Execute Many**: Step plans compiled once, then immutable during execution
- **Frozen Contexts**: ProcessingContext frozen after compilation, preventing state corruption
- **Stateless Steps**: Step objects stripped of mutable state after compilation

## **🏗️ Concurrency Architecture**

### **Two-Phase Execution Model**

```python
# Phase 1: Compilation (Single-threaded)
compiled_contexts = orchestrator.compile_pipelines(pipeline_definition, wells)
# Result: Immutable ProcessingContext for each well

# Phase 2: Execution (Multi-threaded)
results = orchestrator.execute_compiled_plate(pipeline_definition, compiled_contexts)
# Result: Parallel execution across wells
```

### **Thread Pool Execution**

```python
def execute_compiled_plate(self, pipeline_definition, compiled_contexts, max_workers=None):
    """Execute with configurable parallelism."""
    
    actual_max_workers = max_workers or self.global_config.num_workers
    
    if actual_max_workers > 1 and len(compiled_contexts) > 1:
        # Parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
            future_to_well_id = {
                executor.submit(self._execute_single_well, pipeline_definition, context, visualizer): well_id
                for well_id, context in compiled_contexts.items()
            }
            
            for future in concurrent.futures.as_completed(future_to_well_id):
                well_id = future_to_well_id[future]
                try:
                    result = future.result()
                    execution_results[well_id] = result
                except Exception as exc:
                    logger.error(f"Well {well_id} exception: {exc}", exc_info=True)
                    execution_results[well_id] = {"status": "error", "error": str(exc)}
    else:
        # Sequential execution
        for well_id, context in compiled_contexts.items():
            execution_results[well_id] = self._execute_single_well(pipeline_definition, context, visualizer)
```

## **🔒 Thread Safety Mechanisms**

### **1. Immutable State Design**

#### **Frozen ProcessingContext**
```python
class ProcessingContext:
    """Context with immutability enforcement."""
    
    def freeze(self):
        """Make context immutable after compilation."""
        self._is_frozen = True
    
    def __setattr__(self, name, value):
        """Prevent modification of frozen context."""
        if getattr(self, '_is_frozen', False) and name != '_is_frozen':
            raise AttributeError(f"Cannot modify '{name}' of frozen ProcessingContext")
        super().__setattr__(name, value)
```

**Thread Safety Guarantee**: Frozen contexts cannot be modified, eliminating race conditions.

#### **Stateless Step Objects**
```python
# After compilation, step objects are stripped of mutable state
def strip_step_attributes(pipeline_definition):
    """Remove mutable attributes from step objects."""
    for step in pipeline_definition:
        # Remove compilation-time attributes
        delattr(step, 'input_dir')
        delattr(step, 'output_dir') 
        delattr(step, 'variable_components')
        # Step becomes stateless, safe for concurrent access
```

**Thread Safety Guarantee**: Stateless objects can be safely shared across threads.

### **2. Thread-Local Resource Isolation**

#### **FileManager Per Thread**
```python
class FileManager:
    """FileManager with strict thread isolation."""
    
    def __init__(self, registry):
        # Thread Safety:
        #   Each FileManager instance must be scoped to a single execution context.
        #   Do NOT share FileManager instances across pipelines or threads.
        #   For isolation, create a dedicated registry for each FileManager.
        self.registry = registry
        self._backend_cache = {}  # Per-instance backend cache
```

**Thread Safety Guarantee**: Each thread gets its own FileManager instance with isolated backend cache.

#### **Backend Instance Isolation**
```python
def _get_backend(self, backend_name):
    """Get backend with per-FileManager caching."""
    # Thread Safety:
    #   This method is thread-safe for a single FileManager instance.
    #   Backend instances are NOT shared across FileManager instances.
    if backend_name not in self._backend_cache:
        backend_class = self.registry[backend_name]
        self._backend_cache[backend_name] = backend_class()  # New instance per FileManager
    
    return self._backend_cache[backend_name]
```

**Thread Safety Guarantee**: Backend instances are never shared between threads.

### **3. Global Resource Coordination**

#### **Thread-Safe GPU Registry**
```python
# Global GPU registry with thread-safe access
GPU_REGISTRY: Dict[int, Dict[str, int]] = {}
_registry_lock = threading.Lock()

def acquire_gpu_slot() -> Optional[int]:
    """Thread-safe GPU slot acquisition."""
    with _registry_lock:
        # Find first available GPU
        for gpu_id, info in GPU_REGISTRY.items():
            if info["active"] < info["max_pipelines"]:
                info["active"] += 1  # Atomic increment
                return gpu_id
        return None

def release_gpu_slot(gpu_id: int):
    """Thread-safe GPU slot release."""
    with _registry_lock:
        GPU_REGISTRY[gpu_id]["active"] -= 1  # Atomic decrement
```

**Thread Safety Guarantee**: GPU resource allocation is atomic and consistent across threads.

#### **Memory Backend Isolation**
```python
class MemoryStorageBackend(StorageBackend):
    """Memory backend with per-instance storage."""
    
    def __init__(self):
        self._memory_store = {}  # Per-instance memory store
        self._prefixes = set()   # Per-instance namespace tracking
```

**Thread Safety Guarantee**: Each thread gets its own memory backend instance with isolated storage.

## **🔄 Execution Flow Thread Safety**

### **Single Well Execution**
```python
def _execute_single_well(self, pipeline_definition, context, visualizer):
    """Execute pipeline for single well - thread-safe by design."""
    
    # 1. Context is frozen (immutable)
    assert context.is_frozen()
    
    # 2. Each thread has its own FileManager
    filemanager = context.filemanager  # Thread-local instance
    
    # 3. GPU slot acquisition (thread-safe)
    gpu_id = acquire_gpu_slot() if requires_gpu else None
    
    try:
        # 4. Sequential step execution within thread
        for step in pipeline_definition:
            step.process(context)  # Step is stateless, context is immutable
        
        return {"status": "success", "well_id": context.well_id}
        
    finally:
        # 5. GPU slot release (thread-safe)
        if gpu_id is not None:
            release_gpu_slot(gpu_id)
```

### **FunctionStep Thread Safety**
```python
def process(self, context):
    """FunctionStep execution - thread-safe by design."""
    
    # 1. Read immutable step plan
    step_plan = context.step_plans[self.step_id]  # Immutable after compilation
    
    # 2. Use thread-local FileManager
    filemanager = context.filemanager  # Thread-local instance
    
    # 3. Load data using isolated backends
    for file_path in matching_files:
        image = filemanager.load_image(file_path, read_backend)  # Isolated backend
        raw_slices.append(image)
    
    # 4. Process data (pure computation)
    result = func(image_stack)  # Function operates on local data
    
    # 5. Save data using isolated backends
    for i, slice_2d in enumerate(output_slices):
        filemanager.save_image(slice_2d, output_path, write_backend)  # Isolated backend
```

## **🎯 Concurrency Guarantees**

### **What is Thread-Safe:**

#### **✅ Immutable Data Structures**
- **Frozen ProcessingContext**: Cannot be modified after compilation
- **Step Plans**: Immutable dictionaries with execution configuration
- **Stateless Steps**: No mutable state after attribute stripping

#### **✅ Thread-Local Resources**
- **FileManager Instances**: One per thread, never shared
- **Backend Instances**: Isolated per FileManager
- **Memory Storage**: Separate memory store per backend instance

#### **✅ Atomic Global Operations**
- **GPU Registry Access**: Protected by locks for atomic updates
- **Configuration Access**: Immutable configuration objects

### **What Requires Coordination:**

#### **🔒 GPU Resource Management**
- **Slot Acquisition/Release**: Thread-safe with locks
- **Registry Status**: Atomic reads with locks
- **Capacity Management**: Thread-safe updates

#### **🔒 Global Configuration Updates**
- **Live Config Changes**: Coordinated through orchestrator
- **Registry Initialization**: One-time setup with thread-safe checks

## **⚡ Performance Characteristics**

### **Scalability Model**
```python
# Optimal parallelism calculation
max_workers = min(
    num_wells,                    # Don't create more threads than wells
    global_config.num_workers,    # Respect configured limit
    available_gpu_slots           # Don't exceed GPU capacity
)
```

### **Resource Utilization**
- **CPU Cores**: One thread per core (configurable)
- **GPU Devices**: Multiple pipelines per GPU (based on memory capacity)
- **Memory**: Isolated per thread, no sharing overhead
- **I/O**: Parallel disk access across threads

### **Contention Points**
- **GPU Registry**: Minimal contention (fast lock operations)
- **Disk I/O**: Natural parallelism across different directories
- **Memory Allocation**: Thread-local, no contention

## **🚀 Advanced Concurrency Features**

### **Exception Isolation**
```python
# Exceptions in one well don't affect others
for future in concurrent.futures.as_completed(future_to_well_id):
    well_id = future_to_well_id[future]
    try:
        result = future.result()
        execution_results[well_id] = result
    except Exception as exc:
        # Exception isolated to this well
        logger.error(f"Well {well_id} exception: {exc}", exc_info=True)
        execution_results[well_id] = {"status": "error", "error": str(exc)}
        # Other wells continue processing
```

### **Resource Cleanup**
```python
def _execute_single_well(self, pipeline_definition, context, visualizer):
    """Guaranteed resource cleanup per thread."""
    acquired_gpu = None
    
    try:
        if requires_gpu:
            acquired_gpu = acquire_gpu_slot()
        
        # Execute pipeline...
        
    finally:
        # Always cleanup, even on exception
        if acquired_gpu is not None:
            release_gpu_slot(acquired_gpu)
```

### **Graceful Degradation**
```python
# Automatic fallback to sequential execution
if actual_max_workers <= 1 or len(compiled_contexts) <= 1:
    logger.info("Executing wells sequentially")
    for well_id, context in compiled_contexts.items():
        execution_results[well_id] = self._execute_single_well(pipeline_definition, context, visualizer)
```

## **🎯 Why This Model is Brilliant**

### **1. Eliminates Complex Locking**
- **Immutable State**: No need to lock shared data structures
- **Thread Isolation**: No shared mutable resources between threads
- **Minimal Coordination**: Only GPU registry requires locking

### **2. Excellent Error Isolation**
- **Well-Level Failures**: One well failure doesn't affect others
- **Resource Cleanup**: Guaranteed cleanup per thread
- **Exception Propagation**: Clear error reporting per well

### **3. Predictable Performance**
- **Linear Scaling**: Performance scales with number of cores/GPUs
- **No Lock Contention**: Minimal synchronization overhead
- **Resource Efficiency**: Optimal utilization of available hardware

### **4. Simple Mental Model**
- **Easy to Reason About**: Each well is independent
- **Debugging Friendly**: Clear thread boundaries and isolated state
- **Maintainable**: No complex synchronization logic

## **🔮 Future Enhancements**

### **Planned Concurrency Features**
1. **Work Stealing**: Dynamic load balancing between threads
2. **Pipeline Parallelism**: Parallel execution of steps within a well
3. **Distributed Processing**: Multi-node execution coordination
4. **Adaptive Threading**: Dynamic thread pool sizing based on workload
5. **Memory Pool Management**: Shared memory pools for large datasets

This concurrency model represents **production-grade parallel processing architecture** that achieves excellent performance while maintaining simplicity and thread safety through careful design rather than complex synchronization.

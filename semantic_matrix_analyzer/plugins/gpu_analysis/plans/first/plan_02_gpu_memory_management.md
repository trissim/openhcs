# Plan 02: GPU Memory Management

## Objective

Implement a comprehensive GPU memory management system that keeps all data in GPU VRAM whenever possible, minimizing CPU-GPU transfers and maximizing performance for the Semantic Matrix Analyzer.

## Background

The OpenHCS IO system provides several storage backends:
- `DiskStorageBackend`: Stores data on disk
- `MemoryStorageBackend`: Stores data in CPU memory
- `ZarrStorageBackend`: Stores data in Zarr format (chunked arrays)

However, none of these backends are optimized for GPU memory. For GPU-accelerated semantic analysis, we need to keep data in GPU memory as much as possible to avoid the overhead of CPU-GPU transfers.

## Current State

The SMA codebase likely loads data from disk or CPU memory into GPU memory for processing, then transfers results back to CPU memory. This creates unnecessary overhead, especially for large datasets or when performing multiple operations on the same data.

## Implementation Plan

### 1. Create GPU Storage Backend

Create a new `GPUStorageBackend` class that implements the `StorageBackend` interface:

```python
class GPUStorageBackend(StorageBackend):
    """
    GPU storage backend implementation.
    
    This class provides a concrete implementation of the storage backend interfaces
    for GPU memory. It stores data directly in GPU memory using PyTorch tensors.
    """
    
    def __init__(self, device="cuda"):
        """Initialize the GPU storage backend."""
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self._gpu_store = {}  # Dict[str, torch.Tensor]
        self._prefixes = set()  # Declared directory-like namespaces
    
    def _normalize(self, path: Union[str, Path]) -> str:
        """Normalize a path to a string key."""
        return Path(path).as_posix()
    
    def load(self, file_path: Union[str, Path], **kwargs) -> Any:
        """Load data from GPU memory."""
        key = self._normalize(file_path)
        
        if key not in self._gpu_store:
            raise FileNotFoundError(f"GPU key '{key}' not found.")
        
        return self._gpu_store[key]
    
    def save(self, data: Any, output_path: Union[str, Path], **kwargs) -> None:
        """Save data to GPU memory."""
        key = self._normalize(output_path)
        
        # Convert data to tensor if it's not already
        if not isinstance(data, torch.Tensor):
            data = self._to_tensor(data)
        
        # Move tensor to GPU
        data = data.to(self.device)
        
        # Store tensor
        self._gpu_store[key] = data
    
    def _to_tensor(self, data: Any) -> torch.Tensor:
        """Convert data to a PyTorch tensor."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device)
        elif isinstance(data, list):
            return torch.tensor(data, device=self.device)
        elif isinstance(data, dict):
            return {k: self._to_tensor(v) for k, v in data.items()}
        else:
            return torch.tensor(data, device=self.device)
    
    # Implement other required methods...
```

### 2. Create GPU Memory Manager

Create a `GPUMemoryManager` class to manage GPU memory allocation and deallocation:

```python
class GPUMemoryManager:
    """
    GPU memory manager.
    
    This class manages GPU memory allocation and deallocation, ensuring that
    data is kept in GPU memory as much as possible while avoiding out-of-memory errors.
    """
    
    def __init__(self, device="cuda", max_memory_fraction=0.9):
        """Initialize the GPU memory manager."""
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.max_memory_fraction = max_memory_fraction
        self._cache = {}  # Dict[str, torch.Tensor]
        self._usage = {}  # Dict[str, int]  # Usage count for each tensor
        
    def allocate(self, key: str, data: Any) -> torch.Tensor:
        """Allocate GPU memory for data."""
        # Convert data to tensor if it's not already
        if not isinstance(data, torch.Tensor):
            data = self._to_tensor(data)
        
        # Move tensor to GPU
        data = data.to(self.device)
        
        # Store tensor in cache
        self._cache[key] = data
        self._usage[key] = 1
        
        return data
    
    def get(self, key: str) -> torch.Tensor:
        """Get data from GPU memory."""
        if key not in self._cache:
            raise KeyError(f"Key '{key}' not found in GPU memory.")
        
        # Increment usage count
        self._usage[key] += 1
        
        return self._cache[key]
    
    def release(self, key: str) -> None:
        """Release GPU memory for data."""
        if key not in self._cache:
            return
        
        # Decrement usage count
        self._usage[key] -= 1
        
        # If usage count is 0, remove from cache
        if self._usage[key] <= 0:
            del self._cache[key]
            del self._usage[key]
    
    def clear(self) -> None:
        """Clear all GPU memory."""
        self._cache.clear()
        self._usage.clear()
        
        # Force CUDA to release memory
        torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage statistics."""
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_reserved = torch.cuda.max_memory_reserved(self.device)
        
        return {
            "allocated": allocated / (1024 ** 3),  # GB
            "reserved": reserved / (1024 ** 3),  # GB
            "max_reserved": max_reserved / (1024 ** 3),  # GB
        }
    
    def _to_tensor(self, data: Any) -> torch.Tensor:
        """Convert data to a PyTorch tensor."""
        # Implementation details...
```

### 3. Create GPU Tensor Cache

Create a `GPUTensorCache` class to cache frequently used tensors:

```python
class GPUTensorCache:
    """
    GPU tensor cache.
    
    This class caches frequently used tensors in GPU memory to avoid
    repeated CPU-GPU transfers.
    """
    
    def __init__(self, device="cuda", max_size=100):
        """Initialize the GPU tensor cache."""
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.max_size = max_size
        self._cache = {}  # Dict[str, torch.Tensor]
        self._lru = []  # List[str]  # Least recently used keys
        
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get a tensor from the cache."""
        if key not in self._cache:
            return None
        
        # Update LRU
        self._lru.remove(key)
        self._lru.append(key)
        
        return self._cache[key]
    
    def put(self, key: str, tensor: torch.Tensor) -> None:
        """Put a tensor in the cache."""
        # If cache is full, remove least recently used tensor
        if len(self._cache) >= self.max_size:
            lru_key = self._lru.pop(0)
            del self._cache[lru_key]
        
        # Add tensor to cache
        self._cache[key] = tensor
        self._lru.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._lru.clear()
```

### 4. Create GPU AST Manager

Create a `GPUASTManager` class to manage AST tensors in GPU memory:

```python
class GPUASTManager:
    """
    GPU AST manager.
    
    This class manages AST tensors in GPU memory, ensuring that
    ASTs are kept in GPU memory as much as possible.
    """
    
    def __init__(self, device="cuda"):
        """Initialize the GPU AST manager."""
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self._ast_cache = {}  # Dict[str, Dict[str, torch.Tensor]]
        
    def tensorize(self, file_path: Union[str, Path], ast_node: Any) -> Dict[str, torch.Tensor]:
        """Tensorize an AST and store it in GPU memory."""
        key = str(file_path)
        
        # Check if AST is already tensorized
        if key in self._ast_cache:
            return self._ast_cache[key]
        
        # Tensorize AST
        from gpu_analysis.ast_tensor import ASTTensorizer
        tensorizer = ASTTensorizer(device=self.device)
        ast_tensors = tensorizer.tensorize(ast_node)
        
        # Store tensorized AST
        self._ast_cache[key] = ast_tensors
        
        return ast_tensors
    
    def get(self, file_path: Union[str, Path]) -> Optional[Dict[str, torch.Tensor]]:
        """Get a tensorized AST from GPU memory."""
        key = str(file_path)
        return self._ast_cache.get(key)
    
    def clear(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Clear AST tensors from GPU memory."""
        if file_path is None:
            self._ast_cache.clear()
        else:
            key = str(file_path)
            if key in self._ast_cache:
                del self._ast_cache[key]
```

### 5. Integrate with SMA Configuration System

Extend SMA's configuration system to support GPU memory management:

```python
def extend_config_manager(config_manager):
    """Extend SMA's configuration manager with GPU memory management options."""
    # Add GPU memory management options
    config_manager.set_default("gpu.memory.max_fraction", 0.9)
    config_manager.set_default("gpu.memory.cache_size", 100)
    config_manager.set_default("gpu.memory.keep_in_vram", True)
    
    # Add GPU backend options
    config_manager.set_default("gpu.backend.enabled", True)
    config_manager.set_default("gpu.backend.device", "cuda")
    
    # Add GPU AST options
    config_manager.set_default("gpu.ast.cache_enabled", True)
    config_manager.set_default("gpu.ast.max_cache_size", 50)
```

## Integration with SMA

### 1. Register GPU Storage Backend

Register the GPU storage backend with SMA's storage registry:

```python
def register_gpu_backend():
    """Register GPU storage backend with SMA's storage registry."""
    from openhcs.constants.constants import Backend
    from openhcs.io.base import storage_registry
    
    # Add GPU backend to registry
    registry = storage_registry()
    registry[Backend.GPU.value] = GPUStorageBackend
```

### 2. Create GPU Memory Context Manager

Create a context manager for GPU memory operations:

```python
class GPUMemoryContext:
    """
    Context manager for GPU memory operations.
    
    This class provides a context manager for GPU memory operations,
    ensuring that data is kept in GPU memory within the context and
    properly released when the context exits.
    """
    
    def __init__(self, manager: GPUMemoryManager):
        """Initialize the GPU memory context."""
        self.manager = manager
        self.keys = []
        
    def __enter__(self):
        """Enter the context."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        # Release all allocated memory
        for key in self.keys:
            self.manager.release(key)
        
    def allocate(self, key: str, data: Any) -> torch.Tensor:
        """Allocate GPU memory for data within the context."""
        tensor = self.manager.allocate(key, data)
        self.keys.append(key)
        return tensor
```

### 3. Create GPU-Aware File Manager

Create a GPU-aware file manager that integrates with SMA's file management system:

```python
class GPUFileManager:
    """
    GPU-aware file manager.
    
    This class provides a GPU-aware file manager that integrates with
    SMA's file management system, ensuring that files are loaded into
    GPU memory when needed and kept there as long as possible.
    """
    
    def __init__(self, device="cuda"):
        """Initialize the GPU file manager."""
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.memory_manager = GPUMemoryManager(device=device)
        self.tensor_cache = GPUTensorCache(device=device)
        self.ast_manager = GPUASTManager(device=device)
        
    def load_file(self, file_path: Union[str, Path]) -> Any:
        """Load a file into GPU memory."""
        # Implementation details...
        
    def save_file(self, data: Any, file_path: Union[str, Path]) -> None:
        """Save data from GPU memory to a file."""
        # Implementation details...
        
    def load_ast(self, file_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
        """Load an AST into GPU memory."""
        # Implementation details...
```

## Performance Considerations

1. **Memory Pressure**: Monitor GPU memory usage and implement strategies to handle out-of-memory situations:
   - LRU eviction for cached tensors
   - Automatic fallback to CPU memory when GPU memory is exhausted
   - Prioritize keeping frequently accessed data in GPU memory

2. **Data Transfer Minimization**: Minimize CPU-GPU transfers by:
   - Keeping intermediate results in GPU memory
   - Batching operations to amortize transfer costs
   - Using pinned memory for faster transfers when necessary

3. **Tensor Sharing**: Share tensors between operations to avoid redundant allocations:
   - Use reference counting to track tensor usage
   - Only release memory when all references are gone
   - Use in-place operations where possible

4. **Memory Fragmentation**: Mitigate GPU memory fragmentation by:
   - Allocating tensors of similar sizes together
   - Reusing tensor storage when possible
   - Periodically compacting memory by reallocating tensors

## Testing Strategy

1. **Memory Usage Tests**: Verify that GPU memory is used efficiently:
   - Monitor memory usage during operations
   - Check for memory leaks
   - Verify that memory is properly released

2. **Performance Tests**: Measure the performance improvement from keeping data in GPU memory:
   - Compare with CPU-only implementation
   - Measure time spent on data transfers
   - Identify bottlenecks

3. **Stress Tests**: Test behavior under memory pressure:
   - Process large datasets that exceed GPU memory
   - Verify graceful fallback to CPU
   - Measure performance degradation

## Success Criteria

1. Data remains in GPU memory throughout the analysis pipeline, with minimal CPU-GPU transfers.

2. Performance is significantly improved compared to the CPU-only implementation, especially for large datasets.

3. The implementation gracefully handles out-of-memory situations, falling back to CPU when necessary.

4. The GPU memory management system integrates seamlessly with SMA's existing storage backends.

## References

1. OpenHCS IO system: `/home/ts/code/projects/brain/openhcs/io/`

2. PyTorch CUDA documentation: https://pytorch.org/docs/stable/cuda.html

3. Voetter, R. F. (2020-2021). "Parallel Lexing, Parsing and Semantic Analysis on the GPU."

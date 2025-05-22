# Plan 05: Batch Processing and Configuration Integration

## Objective

Implement batch processing capabilities and integrate with SMA's configuration system to optimize GPU utilization and provide a seamless user experience.

## Background

The research paper highlights that batch processing significantly improves GPU utilization by amortizing kernel launch overhead and enabling better parallelism. Additionally, proper configuration integration is essential for allowing users to tune performance parameters and control GPU memory usage.

## Current State

The SMA codebase likely processes files individually, which is suboptimal for GPU acceleration. Additionally, we need to integrate our GPU-accelerated module with SMA's configuration system to allow users to control GPU-specific parameters.

## Implementation Plan

### 1. Implement Batch Processing Manager

Create a `BatchProcessingManager` class to handle batch processing of multiple files:

```python
class BatchProcessingManager:
    """
    Batch processing manager for GPU-accelerated analysis.
    
    This class manages batch processing of multiple files to optimize GPU utilization.
    """
    
    def __init__(self, device="cuda", batch_size=10, memory_manager=None):
        """Initialize the batch processing manager."""
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.batch_size = batch_size
        self.memory_manager = memory_manager or GPUMemoryManager(device=self.device)
        
        # Initialize analyzers
        self.semantic_analyzer = SemanticAnalyzer(
            device=self.device, memory_manager=self.memory_manager)
        self.pattern_matcher = GPUPatternMatcherRegistry(
            device=self.device)
        
    def process_batch(self, file_paths, analysis_types=None):
        """
        Process a batch of files using GPU acceleration.
        
        Args:
            file_paths: List of file paths to process
            analysis_types: Types of analysis to perform (if None, perform all)
            
        Returns:
            Dictionary mapping file paths to analysis results
        """
        # Split files into batches
        batches = self._split_into_batches(file_paths)
        
        # Process each batch
        results = {}
        for batch in batches:
            batch_results = self._process_single_batch(batch, analysis_types)
            results.update(batch_results)
        
        return results
    
    def _split_into_batches(self, file_paths):
        """Split file paths into batches of appropriate size."""
        return [file_paths[i:i+self.batch_size] 
                for i in range(0, len(file_paths), self.batch_size)]
    
    def _process_single_batch(self, file_paths, analysis_types):
        """Process a single batch of files."""
        # Load all files in the batch
        files = {}
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                files[file_path] = f.read()
        
        # Tensorize all ASTs in the batch
        ast_tensors = {}
        for file_path, content in files.items():
            ast_tensors[file_path] = self._tensorize_ast(content)
        
        # Perform analyses on the batch
        results = {}
        
        # Semantic analysis
        if "semantic" in analysis_types or analysis_types is None:
            for file_path, content in files.items():
                results[file_path] = self.semantic_analyzer.analyze(
                    content, file_path, analysis_types)
        
        # Pattern matching
        if "pattern" in analysis_types or analysis_types is None:
            pattern_results = self._batch_pattern_matching(
                files, ast_tensors)
            for file_path, matches in pattern_results.items():
                if file_path not in results:
                    results[file_path] = {}
                results[file_path]["pattern_matches"] = matches
        
        return results
    
    def _tensorize_ast(self, code):
        """Tensorize AST and keep it in GPU memory."""
        # Generate a unique key for this AST
        key = f"ast_{hash(code)}"
        
        # Check if AST is already tensorized and in GPU memory
        cached_ast = self.memory_manager.get(key)
        if cached_ast is not None:
            return cached_ast
        
        # Tensorize AST
        from gpu_analysis.ast_tensor import ASTTensorizer
        tensorizer = ASTTensorizer(device=self.device)
        ast_tensors = tensorizer.tensorize(code)
        
        # Cache AST in GPU memory
        self.memory_manager.allocate(key, ast_tensors)
        
        return ast_tensors
    
    def _batch_pattern_matching(self, files, ast_tensors):
        """Perform pattern matching on a batch of files."""
        # Group patterns by type
        pattern_groups = self._group_patterns_by_type()
        
        # Process each pattern type in batch
        results = {file_path: [] for file_path in files}
        
        for pattern_type, patterns in pattern_groups.items():
            # Get the appropriate matcher
            matcher = self.pattern_matcher.get_matcher(pattern_type)
            if matcher is None:
                continue
            
            # Process each file with all patterns of this type
            for file_path, content in files.items():
                for pattern in patterns:
                    matches = matcher.match_pattern(
                        pattern, file_path, content, ast_tensors[file_path])
                    results[file_path].extend(matches)
        
        return results
    
    def _group_patterns_by_type(self):
        """Group patterns by type for batch processing."""
        # Implementation details...
```

### 2. Implement Dynamic Batch Size Adjustment

Implement dynamic batch size adjustment based on GPU memory usage:

```python
class DynamicBatchSizeManager:
    """
    Dynamic batch size manager.
    
    This class dynamically adjusts the batch size based on GPU memory usage.
    """
    
    def __init__(self, device="cuda", initial_batch_size=10, 
                 min_batch_size=1, max_batch_size=100,
                 target_memory_usage=0.8):
        """Initialize the dynamic batch size manager."""
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_usage = target_memory_usage
        
    def get_batch_size(self):
        """Get the current batch size."""
        return self.batch_size
    
    def update_batch_size(self, memory_usage):
        """
        Update the batch size based on GPU memory usage.
        
        Args:
            memory_usage: Current GPU memory usage as a fraction (0.0 to 1.0)
            
        Returns:
            New batch size
        """
        if memory_usage > self.target_memory_usage * 1.1:
            # Memory usage is too high, decrease batch size
            self.batch_size = max(self.min_batch_size, 
                                 int(self.batch_size * 0.8))
        elif memory_usage < self.target_memory_usage * 0.9:
            # Memory usage is low, increase batch size
            self.batch_size = min(self.max_batch_size, 
                                 int(self.batch_size * 1.2))
        
        return self.batch_size
```

### 3. Integrate with SMA Configuration System

Create a `GPUConfigManager` class to integrate with SMA's configuration system:

```python
class GPUConfigManager:
    """
    GPU configuration manager.
    
    This class manages GPU-specific configuration options and integrates
    with SMA's configuration system.
    """
    
    def __init__(self, config=None):
        """Initialize the GPU configuration manager."""
        self.config = config or {}
        self._set_defaults()
        
    def _set_defaults(self):
        """Set default values for GPU configuration options."""
        # Device configuration
        self.config.setdefault("gpu.device", "cuda")
        self.config.setdefault("gpu.enabled", True)
        
        # Memory configuration
        self.config.setdefault("gpu.memory.max_fraction", 0.9)
        self.config.setdefault("gpu.memory.cache_size", 100)
        self.config.setdefault("gpu.memory.keep_in_vram", True)
        
        # Batch processing configuration
        self.config.setdefault("gpu.batch.enabled", True)
        self.config.setdefault("gpu.batch.initial_size", 10)
        self.config.setdefault("gpu.batch.min_size", 1)
        self.config.setdefault("gpu.batch.max_size", 100)
        self.config.setdefault("gpu.batch.dynamic_sizing", True)
        
        # Performance configuration
        self.config.setdefault("gpu.performance.dynamic_dispatch", True)
        self.config.setdefault("gpu.performance.cpu_threshold", 1000)  # AST nodes
        
    def get(self, key, default=None):
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set a configuration value."""
        self.config[key] = value
        
    def update(self, config_dict):
        """Update configuration with values from a dictionary."""
        self.config.update(config_dict)
        
    def create_memory_manager(self):
        """Create a GPU memory manager based on configuration."""
        device = self.get("gpu.device")
        max_fraction = self.get("gpu.memory.max_fraction")
        cache_size = self.get("gpu.memory.cache_size")
        
        return GPUMemoryManager(
            device=device,
            max_memory_fraction=max_fraction,
            cache_size=cache_size
        )
    
    def create_batch_size_manager(self):
        """Create a dynamic batch size manager based on configuration."""
        device = self.get("gpu.device")
        initial_size = self.get("gpu.batch.initial_size")
        min_size = self.get("gpu.batch.min_size")
        max_size = self.get("gpu.batch.max_size")
        target_memory_usage = self.get("gpu.memory.max_fraction") * 0.9
        
        return DynamicBatchSizeManager(
            device=device,
            initial_batch_size=initial_size,
            min_batch_size=min_size,
            max_batch_size=max_size,
            target_memory_usage=target_memory_usage
        )
```

### 4. Implement Dynamic Dispatch

Implement dynamic dispatch between CPU and GPU based on input size:

```python
class DynamicDispatcher:
    """
    Dynamic dispatcher between CPU and GPU.
    
    This class dynamically dispatches analysis tasks to either CPU or GPU
    based on input size and configuration.
    """
    
    def __init__(self, config_manager):
        """Initialize the dynamic dispatcher."""
        self.config_manager = config_manager
        self.cpu_threshold = config_manager.get("gpu.performance.cpu_threshold")
        self.dynamic_dispatch = config_manager.get("gpu.performance.dynamic_dispatch")
        
        # Initialize CPU and GPU analyzers
        self.gpu_analyzer = GPUSemanticAnalysisManager(
            device=config_manager.get("gpu.device"),
            config=config_manager.config
        )
        self.cpu_analyzer = CPUSemanticAnalysisManager(
            config=config_manager.config
        )
        
    def dispatch(self, code, file_path=None, analysis_types=None):
        """
        Dispatch analysis to either CPU or GPU based on input size.
        
        Args:
            code: Python code to analyze
            file_path: Optional path to the file
            analysis_types: Types of analysis to perform (if None, perform all)
            
        Returns:
            Dictionary of analysis results
        """
        if not self.dynamic_dispatch:
            # Always use GPU if dynamic dispatch is disabled
            return self.gpu_analyzer.analyze(code, file_path, analysis_types)
        
        # Estimate input size
        input_size = self._estimate_input_size(code)
        
        # Dispatch based on input size
        if input_size < self.cpu_threshold:
            # Use CPU for small inputs
            return self.cpu_analyzer.analyze(code, file_path, analysis_types)
        else:
            # Use GPU for large inputs
            return self.gpu_analyzer.analyze(code, file_path, analysis_types)
    
    def _estimate_input_size(self, code):
        """Estimate the size of the input code."""
        # Simple heuristic: count the number of AST nodes
        import ast
        try:
            tree = ast.parse(code)
            return sum(1 for _ in ast.walk(tree))
        except:
            # If parsing fails, use code length as a fallback
            return len(code)
```

### 5. Create Main Integration Module

Create a main integration module to tie everything together:

```python
class GPUAcceleratedAnalysis:
    """
    Main integration module for GPU-accelerated analysis.
    
    This class provides a unified interface for GPU-accelerated analysis
    that integrates with SMA's existing systems.
    """
    
    def __init__(self, sma_config=None):
        """Initialize the GPU-accelerated analysis module."""
        # Create GPU configuration manager
        self.config_manager = GPUConfigManager(config=sma_config)
        
        # Create dynamic dispatcher
        self.dispatcher = DynamicDispatcher(self.config_manager)
        
        # Create batch processing manager
        self.batch_manager = BatchProcessingManager(
            device=self.config_manager.get("gpu.device"),
            batch_size=self.config_manager.get("gpu.batch.initial_size"),
            memory_manager=self.config_manager.create_memory_manager()
        )
        
        # Create dynamic batch size manager
        self.batch_size_manager = self.config_manager.create_batch_size_manager()
        
    def analyze(self, code, file_path=None, analysis_types=None):
        """
        Analyze a single file using GPU acceleration.
        
        Args:
            code: Python code to analyze
            file_path: Optional path to the file
            analysis_types: Types of analysis to perform (if None, perform all)
            
        Returns:
            Dictionary of analysis results
        """
        return self.dispatcher.dispatch(code, file_path, analysis_types)
    
    def analyze_batch(self, file_paths, analysis_types=None):
        """
        Analyze a batch of files using GPU acceleration.
        
        Args:
            file_paths: List of file paths to analyze
            analysis_types: Types of analysis to perform (if None, perform all)
            
        Returns:
            Dictionary mapping file paths to analysis results
        """
        # Update batch size based on current memory usage
        if self.config_manager.get("gpu.batch.dynamic_sizing"):
            memory_usage = self._get_memory_usage()
            new_batch_size = self.batch_size_manager.update_batch_size(memory_usage)
            self.batch_manager.batch_size = new_batch_size
        
        # Process batch
        return self.batch_manager.process_batch(file_paths, analysis_types)
    
    def _get_memory_usage(self):
        """Get current GPU memory usage as a fraction."""
        import torch
        allocated = torch.cuda.memory_allocated(self.config_manager.get("gpu.device"))
        reserved = torch.cuda.memory_reserved(self.config_manager.get("gpu.device"))
        
        if reserved == 0:
            return 0.0
        
        return allocated / reserved
```

## Integration with SMA

### 1. Register with SMA's Plugin System

Create a function to register our GPU-accelerated module with SMA's plugin system:

```python
def register_gpu_module(sma_config):
    """
    Register GPU-accelerated module with SMA's plugin system.
    
    Args:
        sma_config: SMA configuration object
        
    Returns:
        GPUAcceleratedAnalysis instance
    """
    # Create GPU-accelerated analysis module
    gpu_module = GPUAcceleratedAnalysis(sma_config=sma_config)
    
    # Register with SMA's plugin system
    # Implementation details depend on SMA's plugin architecture
    
    return gpu_module
```

### 2. Create Configuration Schema

Create a configuration schema for SMA's configuration system:

```python
def get_gpu_config_schema():
    """
    Get GPU configuration schema for SMA's configuration system.
    
    Returns:
        Dictionary defining the configuration schema
    """
    return {
        "gpu.device": {
            "type": "string",
            "default": "cuda",
            "description": "GPU device to use for acceleration"
        },
        "gpu.enabled": {
            "type": "boolean",
            "default": True,
            "description": "Whether to enable GPU acceleration"
        },
        "gpu.memory.max_fraction": {
            "type": "float",
            "default": 0.9,
            "min": 0.1,
            "max": 1.0,
            "description": "Maximum fraction of GPU memory to use"
        },
        "gpu.memory.cache_size": {
            "type": "integer",
            "default": 100,
            "min": 1,
            "description": "Maximum number of items to cache in GPU memory"
        },
        "gpu.memory.keep_in_vram": {
            "type": "boolean",
            "default": True,
            "description": "Whether to keep data in GPU memory between operations"
        },
        "gpu.batch.enabled": {
            "type": "boolean",
            "default": True,
            "description": "Whether to enable batch processing"
        },
        "gpu.batch.initial_size": {
            "type": "integer",
            "default": 10,
            "min": 1,
            "description": "Initial batch size for processing"
        },
        "gpu.batch.min_size": {
            "type": "integer",
            "default": 1,
            "min": 1,
            "description": "Minimum batch size for processing"
        },
        "gpu.batch.max_size": {
            "type": "integer",
            "default": 100,
            "min": 1,
            "description": "Maximum batch size for processing"
        },
        "gpu.batch.dynamic_sizing": {
            "type": "boolean",
            "default": True,
            "description": "Whether to dynamically adjust batch size based on memory usage"
        },
        "gpu.performance.dynamic_dispatch": {
            "type": "boolean",
            "default": True,
            "description": "Whether to dynamically dispatch between CPU and GPU based on input size"
        },
        "gpu.performance.cpu_threshold": {
            "type": "integer",
            "default": 1000,
            "min": 1,
            "description": "Input size threshold for using CPU instead of GPU"
        }
    }
```

## Performance Considerations

1. **Batch Processing**: Process multiple files simultaneously to amortize kernel launch overhead and improve GPU utilization.

2. **Dynamic Batch Sizing**: Adjust batch size dynamically based on GPU memory usage to maximize throughput while avoiding out-of-memory errors.

3. **Dynamic Dispatch**: Use CPU for small inputs and GPU for large inputs to minimize overhead for small tasks.

4. **Configuration Tuning**: Allow users to tune performance parameters through the configuration system to optimize for their specific hardware and workloads.

## Testing Strategy

1. **Batch Processing Tests**: Verify that batch processing produces the same results as processing files individually.

2. **Configuration Tests**: Verify that configuration options correctly affect the behavior of the GPU-accelerated module.

3. **Performance Tests**: Measure the performance improvement from batch processing and dynamic dispatch.

4. **Integration Tests**: Verify that the GPU-accelerated module integrates seamlessly with SMA's existing systems.

## Success Criteria

1. Batch processing is correctly implemented and improves performance for multiple files.

2. The GPU-accelerated module integrates seamlessly with SMA's configuration system.

3. Dynamic dispatch correctly routes tasks to CPU or GPU based on input size.

4. Users can tune performance parameters through the configuration system.

## References

1. Voetter, R. F. (2020-2021). "Parallel Lexing, Parsing and Semantic Analysis on the GPU."

2. SMA codebase: `semantic_matrix_analyzer/config/`

3. GPU analysis module: `gpu_analysis/`

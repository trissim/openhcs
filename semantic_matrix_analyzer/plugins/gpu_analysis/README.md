# GPU-Accelerated Analysis Module

This module provides GPU-accelerated analysis functionality for the Semantic Matrix Analyzer (SMA). It implements the same interfaces as SMA's existing analysis modules, but uses GPU acceleration for improved performance.

## Features

- **GPU-Friendly AST Representation**: Optimized for parallel processing on GPUs
- **GPU-Accelerated Pattern Matching**: Fast pattern matching using parallel algorithms
- **GPU-Accelerated Semantic Analysis**: Two-phase approach for efficient semantic analysis
- **Batch Processing**: Process multiple files simultaneously for better GPU utilization
- **Dynamic Configuration**: Tune GPU-specific parameters through a unified interface

## Architecture

The module follows a clean plugin architecture with proper separation of concerns:

- **Core Components**:
  - `ast_tensor.py`: GPU-friendly AST representation
  - `ast_adapter.py`: Adapters between SMA's AST and GPU-friendly format
  - `pattern_matcher.py`: GPU-accelerated pattern matching
  - `analyzers/semantic_analyzer.py`: GPU-accelerated semantic analysis

- **Integration Components**:
  - `plugin.py`: Plugin interface for SMA integration
  - `batch_processor.py`: Batch processing for better GPU utilization
  - `config_manager.py`: Configuration management

## Usage

### Basic Usage

```python
from gpu_analysis.plugin import GPUAnalysisPlugin

# Create plugin
plugin = GPUAnalysisPlugin()

# Analyze code
result = plugin.analyze_code("def hello(): print('Hello, world!')")

# Analyze file
result = plugin.analyze_file("/path/to/file.py")

# Analyze multiple files in batch
results = plugin.analyze_batch(["/path/to/file1.py", "/path/to/file2.py"])
```

### Pattern Matchers

The module provides GPU-accelerated pattern matchers for different types of patterns:

```python
from gpu_analysis.pattern_matcher import (
    GPUPatternMatcherRegistry, PatternType, PatternMatch
)

# Create pattern matcher registry
matcher_registry = GPUPatternMatcherRegistry(device="cuda")

# Create patterns
string_pattern = Pattern(pattern_type=PatternType.STRING, pattern="def hello")
regex_pattern = Pattern(pattern_type=PatternType.REGEX, pattern=r"def\s+(\w+)\s*\(")
ast_pattern = Pattern(pattern_type=PatternType.AST, pattern=("FunctionDef", None))

# Match patterns
matches = matcher_registry.match_patterns(
    [string_pattern, regex_pattern, ast_pattern],
    file_path,
    code,
    ast_node
)
```

### Semantic Analyzers

The module provides GPU-accelerated analyzers for different types of analysis:

```python
from gpu_analysis.analyzers.semantic_analyzer import SemanticAnalyzer

# Initialize analyzer
analyzer = SemanticAnalyzer(device="cuda")

# Analyze code
results = analyzer.analyze(
    code,
    file_path,
    analysis_types=["complexity", "dependency", "type", "variable", "function"]
)

# Extract features
features = analyzer.extract_semantic_features(code)
```

### Integration with SMA

```python
from gpu_analysis.config_manager import register_gpu_module

# Register GPU module with SMA
gpu_module = register_gpu_module(sma_config)

# Use GPU module through SMA's plugin system
sma.use_plugin("gpu_analysis")
```

## Configuration

The module can be configured through a configuration dictionary:

```python
config = {
    "gpu.device": "cuda",  # Device to use ("cuda" or "cpu")
    "gpu.batch.initial_size": 10,  # Initial batch size
    "gpu.batch.dynamic_sizing": True,  # Whether to dynamically adjust batch size
    "gpu.performance.dynamic_dispatch": True,  # Whether to dynamically dispatch between CPU and GPU
    "gpu.performance.cpu_threshold": 1000  # Input size threshold for using CPU instead of GPU
}

plugin = GPUAnalysisPlugin(config=config)
```

## Implementation Details

### GPU-Friendly AST Representation

The module uses a GPU-friendly AST representation with parent pointers instead of child pointers, which is more suitable for parallel processing on GPUs. The `GPUASTTensorizer` class converts Python ASTs to this representation.

### Two-Phase Semantic Analysis

The semantic analyzer implements the two-phase approach described in the paper "Parallel Lexing, Parsing and Semantic Analysis on the GPU" by R. F. Voetter:

1. **Phase 1**: Assign initial types to all nodes in parallel
2. **Phase 2**: Verify type compatibility in parallel

### Parallel Pattern Matching

The pattern matcher implements parallel algorithms for string, regex, and AST pattern matching. The `GPUPatternMatcherRegistry` class provides a registry for different pattern matchers.

### Batch Processing

The batch processor optimizes GPU utilization by processing multiple files simultaneously. The `DynamicBatchSizeManager` class dynamically adjusts the batch size based on GPU memory usage.

## Requirements

- Python 3.7+
- PyTorch 1.8.0+
- NumPy

## Original Code

This module is adapted from the Brain project for use with the Semantic Matrix Analyzer. The original code has been modified to integrate with the SMA project and to provide GPU-accelerated alternatives to the core analysis components.

## License

This module is part of the Semantic Matrix Analyzer and is subject to the same license terms.

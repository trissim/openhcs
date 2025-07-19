# Function Registry System

## Overview

OpenHCS implements a revolutionary function registry system that automatically discovers and unifies 574+ functions from multiple GPU libraries with type-safe contracts. This creates the most comprehensive GPU imaging function ecosystem available in scientific computing.

## The Innovation

**What Makes It Unique**: No other scientific computing platform automatically discovers and unifies this many GPU imaging libraries with unified contracts and type safety.

### Automatic Function Discovery

OpenHCS automatically registers functions from:

```python
✅ 230 pyclesperanto functions (GPU-accelerated OpenCL)
✅ 110 scikit-image functions (with GPU variants via CuCIM)  
✅ 124 CuCIM functions (RAPIDS GPU imaging)
✅ CuPy scipy.ndimage functions
✅ Native OpenHCS functions

Total: 574+ functions with unified contracts
```

### Intelligent Contract Classification

The registry analyzes each function to determine its 3D processing behavior:

```python
# Automatic contract detection:
@numpy  # SLICE_SAFE - processes each Z-slice independently
def gaussian_filter(image_stack, sigma=1.0):
    return scipy.ndimage.gaussian_filter(image_stack, sigma)

@cupy   # CROSS_Z - processes entire 3D volume
def watershed_3d(image_stack, markers):
    return cucim.skimage.segmentation.watershed(image_stack, markers)
```

## Architecture

### Registry Discovery Process

```python
# Automatic discovery workflow:
1. Library Detection
   ├── Scan installed packages (pyclesperanto, scikit-image, etc.)
   ├── Identify imaging functions via introspection
   └── Filter for 3D-compatible functions

2. Contract Analysis
   ├── Analyze function signatures
   ├── Determine 3D processing behavior (SLICE_SAFE vs CROSS_Z)
   └── Classify memory type requirements

3. Decoration Application
   ├── Apply appropriate memory type decorators
   ├── Add contract metadata
   └── Register in unified namespace

4. Validation
   ├── Verify all functions have memory type attributes
   ├── Test basic functionality
   └── Generate registry statistics
```

### Unified Contract System

```python
# All functions get unified contracts:
@numpy
@contract_3d(behavior="SLICE_SAFE")
def registered_function(image_stack, **kwargs):
    """Automatically decorated function with unified interface."""
    pass

# Contract metadata includes:
- input_memory_type: numpy, cupy, torch, etc.
- output_memory_type: numpy, cupy, torch, etc.
- contract_3d: SLICE_SAFE, CROSS_Z, UNKNOWN, DIM_CHANGE
- gpu_compatible: True/False
- library_source: pyclesperanto, scikit-image, etc.
```

## Zero-Configuration GPU Library Access

### Traditional Approach (Manual Integration)

```python
# Traditional scientific computing - manual setup:
import scipy.ndimage
import cucim.skimage.filters  
import pyclesperanto as cle
import cupy as cp

# Manual GPU memory management
gpu_image = cp.asarray(image)
result1 = cucim.skimage.filters.gaussian(gpu_image, sigma=2.0)
result2 = cle.binary_opening(result1, footprint=cle.create_disk(3))
result3 = cucim.skimage.measure.label(result2, connectivity=2)
final = cp.asnumpy(result3)  # Manual CPU transfer
```

### OpenHCS Approach (Unified Registry)

```python
# OpenHCS - unified access, no imports needed:
pipeline = [
    FunctionStep(func="gaussian_filter", sigma=2.0),      # Auto-routes to CuCIM GPU
    FunctionStep(func="binary_opening", footprint=disk(3)), # Auto-routes to pyclesperanto
    FunctionStep(func="label", connectivity=2)            # Auto-routes to CuCIM GPU
]

# Benefits:
✅ No manual imports or setup
✅ Automatic GPU memory management
✅ Unified parameter interface
✅ Type-safe conversions between libraries
✅ Consistent error handling
```

## Registry Statistics

### Current Function Counts

```python
Registry Statistics (as of current version):
├── pyclesperanto: 230 functions
│   ├── Morphological operations: 45
│   ├── Filtering: 38
│   ├── Segmentation: 32
│   ├── Measurements: 28
│   └── Transformations: 87
├── scikit-image (via CuCIM): 110 functions
│   ├── Filters: 35
│   ├── Morphology: 25
│   ├── Segmentation: 20
│   ├── Measure: 18
│   └── Transform: 12
├── CuCIM native: 124 functions
│   ├── Core operations: 45
│   ├── Advanced filters: 35
│   ├── Registration: 25
│   └── Utilities: 19
└── Native OpenHCS: 110+ functions
    ├── Pattern processing: 35
    ├── Batch operations: 30
    ├── Memory management: 25
    └── Validation: 20
```

## Performance Benefits

### Unified Memory Management

```python
# Automatic memory type conversion:
Step 1: disk(tiff) → numpy → gaussian_filter → numpy → memory
Step 2: memory → cupy → binary_opening → cupy → memory  
Step 3: memory → numpy → label → numpy → disk(tiff)

# Conversions handled automatically:
✅ Zero-copy GPU transfers where possible
✅ Minimal CPU roundtrips
✅ Automatic device management
✅ Memory pressure handling
```

### Library Optimization

```python
# Intelligent function routing:
Function Request: "gaussian_filter"
├── Available implementations:
│   ├── scipy.ndimage.gaussian_filter (CPU)
│   ├── cucim.skimage.filters.gaussian (GPU)
│   └── pyclesperanto.gaussian_blur (GPU)
├── Selection criteria:
│   ├── Memory type compatibility
│   ├── GPU availability
│   └── Performance characteristics
└── Chosen: cucim.skimage.filters.gaussian (best GPU performance)
```

## Comparison with Other Platforms

### ImageJ/FIJI
- **Functions**: ~1000+ plugins, mostly CPU
- **Integration**: Manual plugin installation
- **GPU Support**: Limited, plugin-dependent
- **Contracts**: None, runtime discovery of capabilities

### CellProfiler
- **Functions**: ~80 modules, mostly CPU
- **Integration**: Built-in modules only
- **GPU Support**: Very limited
- **Contracts**: Module-specific interfaces

### napari
- **Functions**: Plugin ecosystem, variable quality
- **Integration**: Manual plugin management
- **GPU Support**: Plugin-dependent
- **Contracts**: Plugin-specific

### OpenHCS
- **Functions**: 574+ unified functions, GPU-first
- **Integration**: Automatic discovery and registration
- **GPU Support**: Native GPU support across all libraries
- **Contracts**: Unified type-safe contracts for all functions

## Future Expansion

### Planned Library Integrations

```python
# Roadmap for additional libraries:
├── ITK (Insight Toolkit) - Medical imaging
├── SimpleITK - Simplified ITK interface  
├── OpenCV - Computer vision functions
├── Mahotas - Computer vision for biology
├── nd2reader - Nikon microscopy formats
└── AICSImageIO - Allen Institute formats
```

### Registry Evolution

- **Dynamic loading**: Add libraries at runtime
- **Custom contracts**: User-defined function contracts
- **Performance profiling**: Automatic benchmarking of function variants
- **Cloud functions**: Integration with cloud-based processing

## Technical Implementation

### Registry Architecture

```python
class FunctionRegistry:
    """Central registry for all discovered functions."""
    
    def __init__(self):
        self.functions = {}  # name -> function mapping
        self.metadata = {}   # name -> contract metadata
        self.sources = {}    # name -> library source
    
    def discover_functions(self):
        """Discover functions from all available libraries."""
        for library in self.supported_libraries:
            functions = library.discover_functions()
            for func in functions:
                self.register_function(func)
    
    def register_function(self, func):
        """Register function with unified contract."""
        contract = self.analyze_contract(func)
        decorated_func = self.apply_decorators(func, contract)
        self.functions[func.name] = decorated_func
        self.metadata[func.name] = contract
```

This function registry system represents a fundamental innovation in scientific computing - providing unified, type-safe access to the entire GPU imaging ecosystem through a single, consistent interface.

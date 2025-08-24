Function Registry System
========================

OpenHCS implements a unified function registry that automatically discovers and integrates 574+ GPU-accelerated functions from multiple libraries (pyclesperanto, scikit-image, CuCIM) with type-safe contracts and memory management.

## Why Unified Function Discovery

Scientific image processing involves diverse libraries, each with different:

- **Memory types**: NumPy arrays, CuPy arrays, PyTorch tensors
- **Function signatures**: Inconsistent parameter naming and ordering
- **Processing contracts**: 2D-only, 3D-capable, or flexible dimensionality
- **GPU support**: Native GPU, CPU-only, or hybrid implementations

Without unification, pipelines would need library-specific logic throughout. The registry system provides a single interface to all functions while preserving their native performance characteristics.

## Architecture: Contract-Based Classification

The registry system classifies functions by their processing contracts rather than their library origins:

.. code:: python

   class ProcessingContract(Enum):
       PURE_3D = "_execute_pure_3d"        # Processes 3D volumes directly
       PURE_2D = "_execute_pure_2d"        # Processes 2D slices only
       FLEXIBLE = "_execute_flexible"       # Handles both 2D and 3D
       VOLUMETRIC_TO_SLICE = "_execute_volumetric_to_slice"  # 3D→2D reduction

This classification enables OpenHCS to automatically handle dimensionality conversions and choose optimal execution strategies.

## Unified Registry Architecture

All library registries inherit from a common base that provides consistent discovery and registration patterns:

.. code:: python

   class LibraryRegistryBase(ABC):
       # Each registry declares its characteristics
       MODULES_TO_SCAN: List[str]     # Which modules to scan for functions
       MEMORY_TYPE: str               # Native memory type (numpy, cupy, etc.)
       FLOAT_DTYPE: Any              # Preferred floating-point type

       # Common exclusions across all libraries
       COMMON_EXCLUSIONS = {
           'imread', 'imsave', 'load', 'save', 'read', 'write',
           'show', 'imshow', 'plot', 'display', 'view', 'visualize'
       }

This design enables consistent function discovery across all supported libraries while maintaining their native performance characteristics.

## Automatic Function Discovery

The registry automatically scans and registers functions from multiple GPU libraries:

- **230 pyclesperanto functions**: GPU-accelerated OpenCL implementations
- **110 scikit-image functions**: CPU implementations with GPU variants via CuCIM
- **124 CuCIM functions**: RAPIDS GPU imaging library
- **CuPy scipy.ndimage functions**: GPU-accelerated NumPy equivalents
- **Native OpenHCS functions**: Custom implementations for specific workflows

**Total**: 574+ functions with unified contracts and automatic memory type conversion.

## Contract Classification and Type Safety

The registry analyzes each function to determine its processing behavior and memory requirements:

.. code:: python

   # Functions are classified by their 3D processing capabilities
   @numpy  # PURE_2D - processes each Z-slice independently
   def gaussian_filter(image_stack, sigma=1.0):
       return scipy.ndimage.gaussian_filter(image_stack, sigma)

   @cupy   # PURE_3D - processes entire 3D volume
   def watershed_3d(image_stack, markers):
       return cucim.skimage.segmentation.watershed(image_stack, markers)

   # Usage in pipelines remains simple and consistent
   step = FunctionStep(func=[(gaussian_filter, {'sigma': 2.0})])

This classification enables OpenHCS to automatically handle dimensionality conversions and choose optimal execution strategies without user intervention.

## Memory Type Abstraction

The registry provides automatic memory type conversion between different GPU libraries:

### **Automatic Conversion**
- **NumPy ↔ CuPy**: Zero-copy GPU transfers where possible
- **PyTorch ↔ CuPy**: Shared memory GPU tensors
- **Memory type detection**: Automatic input type recognition
- **Optimal routing**: Functions execute on their native memory types

### **Type Safety**
- **Contract validation**: Ensures functions receive compatible data types
- **Dimension checking**: Validates 2D vs 3D requirements before execution
- **Error prevention**: Catches type mismatches at registration time

## Integration with Pipeline System

### **Function Discovery**
Pipelines access registered functions through a simple, consistent interface:

.. code:: python

   # All functions work the same way regardless of underlying library
   from openhcs.functions import gaussian_filter, watershed_3d, cell_count

   # Functions are used as objects, not strings
   step = FunctionStep(func=[(gaussian_filter, {'sigma': 2.0})])

### **Automatic Optimization**
- **GPU acceleration**: Automatically uses GPU variants when available
- **Memory efficiency**: Minimizes CPU↔GPU transfers
- **Contract-based execution**: Chooses optimal processing strategy
- **Caching**: Fast startup through metadata caching

## Design Benefits

### **Developer Experience**
- **Single interface**: All 574+ functions work identically
- **Type safety**: Compile-time validation of function contracts
- **GPU transparency**: Automatic GPU acceleration without code changes
- **Library agnostic**: Switch between implementations without pipeline changes

### **Performance**
- **Native speed**: Functions execute at library-native performance
- **Memory optimization**: Minimal type conversion overhead
- **GPU utilization**: Automatic GPU routing for supported functions
- **Startup speed**: Cached metadata for fast initialization

### **Extensibility**
- **New libraries**: Adding support requires minimal code (~60-120 lines)
- **Custom functions**: Easy integration of laboratory-specific algorithms
- **Contract system**: Automatic classification of new function behaviors
- **Version management**: Automatic cache invalidation on library updates

This unified registry architecture enables OpenHCS to provide a single, consistent interface to hundreds of GPU-accelerated functions while maintaining their native performance characteristics and handling the complexity of memory type conversions transparently.

Function Registry System
========================

OpenHCS implements a unified library registry system that automatically discovers and integrates GPU-accelerated functions from multiple libraries (pyclesperanto, scikit-image, CuCIM) with type-safe contracts, JSON-based caching, and consistent memory management.

## Why Unified Function Discovery

Scientific image processing involves diverse libraries, each with different:

- **Memory types**: NumPy arrays, CuPy arrays, PyTorch tensors
- **Function signatures**: Inconsistent parameter naming and ordering
- **Processing contracts**: 2D-only, 3D-capable, or flexible dimensionality
- **GPU support**: Native GPU, CPU-only, or hybrid implementations

Without unification, pipelines would need library-specific logic throughout. The registry system provides a single interface to all functions while preserving their native performance characteristics.

## Unified Registry Architecture

The new unified registry system is built around the ``LibraryRegistryBase`` abstract class that eliminates ~70% of code duplication across library registries while enforcing consistent behavior:

.. code:: python

   class LibraryRegistryBase(ABC):
       """Minimal ABC for all library registries."""

       # Common exclusions across all libraries
       COMMON_EXCLUSIONS = {
           'imread', 'imsave', 'load', 'save', 'read', 'write',
           'show', 'imshow', 'plot', 'display', 'view', 'visualize'
       }

       # Abstract class attributes - each implementation must define these
       MODULES_TO_SCAN: List[str]
       MEMORY_TYPE: str  # Memory type string value
       FLOAT_DTYPE: Any  # Library-specific float32 type

This design enables consistent function discovery across all supported libraries while maintaining their native performance characteristics.

## Processing Contract System

The registry system classifies functions by their processing contracts using a unified enum:

.. code:: python

   class ProcessingContract(Enum):
       PURE_3D = "_execute_pure_3d"        # Processes 3D volumes directly
       PURE_2D = "_execute_pure_2d"        # Processes 2D slices only
       FLEXIBLE = "_execute_flexible"       # Handles both 2D and 3D
       VOLUMETRIC_TO_SLICE = "_execute_volumetric_to_slice"  # 3D→2D reduction

       def execute(self, registry, func, image, *args, **kwargs):
           """Execute the contract method on the registry."""
           method = getattr(registry, self.value)
           return method(func, image, *args, **kwargs)

This classification enables OpenHCS to automatically handle dimensionality conversions and choose optimal execution strategies.

## JSON-Based Cache Architecture

The unified registry system features a clean, fail-loud JSON-based cache architecture with version validation:

.. code:: python

   def _load_from_cache(self) -> Optional[Dict[str, FunctionMetadata]]:
       """Load function metadata from cache with validation."""
       # Version validation
       cached_version = cache_data.get('library_version', 'unknown')
       current_version = self.get_library_version()
       if cached_version != current_version:
           logger.info(f"Version changed ({cached_version} → {current_version}) - cache invalid")
           return None

       # Age validation (7 day expiry)
       cache_age_days = (time.time() - cache_timestamp) / (24 * 3600)
       if cache_age_days > 7:
           return None

**Cache Benefits**:
- **Fast startup**: Instant loading of all libraries from cache
- **Version safety**: Automatic cache invalidation on library updates
- **Function reconstruction**: Preserves original function names and metadata
- **Fail-loud behavior**: No silent cache corruption or stale data

## Automatic Function Discovery

The registry automatically scans and registers functions from multiple GPU libraries:

- **230+ pyclesperanto functions**: GPU-accelerated OpenCL implementations
- **110+ scikit-image functions**: CPU implementations with GPU variants via CuCIM
- **124+ CuCIM functions**: RAPIDS GPU imaging library
- **CuPy scipy.ndimage functions**: GPU-accelerated NumPy equivalents
- **Native OpenHCS functions**: Custom implementations for specific workflows

**Total**: Comprehensive function library with unified contracts and automatic memory type conversion.

## Registry Service and Automatic Discovery

The ``RegistryService`` provides unified access to all registry implementations with automatic discovery:

.. code:: python

   class RegistryService:
       """Clean service for registry discovery and function metadata access."""

       @classmethod
       def get_all_functions_with_metadata(cls) -> Dict[str, FunctionMetadata]:
           """Get unified metadata for all functions from all registries."""
           # Discover all registry classes automatically
           registry_classes = cls._discover_registries()

           # Load functions from each registry (with caching)
           for registry_class in registry_classes:
               registry_instance = registry_class()
               functions = registry_instance._load_or_discover_functions()
               all_functions.update(functions)

**Automatic Discovery**: Uses ``pkgutil.walk_packages`` to automatically discover all registry implementations in ``openhcs.processing.backends.lib_registry``, ensuring the system automatically adapts to new registries without code changes.

## Directory Structure

The unified registry system moved from the old structure to a clean, organized layout:

**Old Structure** (deprecated):
::

   openhcs/processing/backends/analysis/
   ├── cupy_registry.py
   ├── pyclesperanto_registry.py
   └── scikit_image_registry.py

**New Structure**:
::

   openhcs/processing/backends/lib_registry/
   ├── unified_registry.py          # Base classes and common functionality
   ├── registry_service.py          # Automatic discovery service
   ├── openhcs_registry.py          # OpenHCS native functions
   ├── pyclesperanto_registry.py    # Pyclesperanto GPU functions
   ├── scikit_image_registry.py     # Scikit-image CPU functions
   └── cupy_registry.py             # CuPy GPU functions

## Function Metadata System

Each registered function is wrapped in a ``FunctionMetadata`` dataclass that provides clean metadata without library-specific leakage:

.. code:: python

   @dataclass(frozen=True)
   class FunctionMetadata:
       """Clean metadata with no library-specific leakage."""
       name: str                    # Function name in registry
       func: Callable              # Wrapped function ready for execution
       contract: ProcessingContract # Processing behavior classification
       registry: LibraryRegistryBase # Reference to source registry
       module: str = ""            # Original module path
       doc: str = ""               # First line of docstring
       tags: List[str] = []        # Generated tags for categorization
       original_name: str = ""     # Original function name for cache reconstruction

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
The updated ``func_registry.py`` integrates with the unified registry system:

.. code:: python

   # Phase 1: Register all functions from RegistryService
   from openhcs.processing.backends.lib_registry.registry_service import RegistryService
   all_functions = RegistryService.get_all_functions_with_metadata()

   # Initialize registry structure based on discovered registries
   for func_name, metadata in all_functions.items():
       registry_name = metadata.registry.library_name
       if registry_name not in FUNC_REGISTRY:
           FUNC_REGISTRY[registry_name] = []

   # Register all functions
   for func_name, metadata in all_functions.items():
       registry_name = metadata.registry.library_name
       FUNC_REGISTRY[registry_name].append(metadata.func)

### **Automatic Optimization**
- **GPU acceleration**: Automatically uses GPU variants when available
- **Memory efficiency**: Minimizes CPU↔GPU transfers
- **Contract-based execution**: Chooses optimal processing strategy
- **JSON caching**: Fast startup through metadata caching with version validation

## Design Benefits

### **Code Reduction**
- **Eliminates ~1000+ lines**: Removes duplicated code across library registries
- **Consistent patterns**: Enforces uniform testing and registration behavior
- **Centralized fixes**: Bug fixes and improvements apply to all libraries
- **Type-safe interface**: Abstract base prevents shortcuts and ensures consistency

### **Developer Experience**
- **Single interface**: All functions work identically regardless of library
- **Automatic discovery**: New registries are automatically detected
- **GPU transparency**: Automatic GPU acceleration without code changes
- **Library agnostic**: Switch between implementations without pipeline changes

### **Performance**
- **Native speed**: Functions execute at library-native performance
- **Memory optimization**: Minimal type conversion overhead
- **GPU utilization**: Automatic GPU routing for supported functions
- **Fast startup**: JSON cache enables instant loading of all libraries

### **Extensibility**
- **Minimal code**: Adding new libraries requires only 60-120 lines vs 350-400
- **Automatic integration**: New registries are discovered without configuration
- **Contract system**: Automatic classification of new function behaviors
- **Version safety**: Automatic cache invalidation prevents stale function metadata

### **Architecture Improvements**
- **Clean separation**: Library-specific logic isolated in individual registries
- **Fail-loud behavior**: No defensive programming or silent failures
- **Generic solution**: Automatically adapts to new components without hardcoding
- **Cache architecture**: JSON-based with version validation and age expiry

This unified registry architecture enables OpenHCS to provide a single, consistent interface to hundreds of GPU-accelerated functions while maintaining their native performance characteristics and handling the complexity of memory type conversions transparently. The system eliminates massive code duplication while making it trivial to add support for new libraries.

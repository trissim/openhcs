Virtual File System (VFS) Architecture
======================================

Overview
--------

The OpenHCS Virtual File System (VFS) provides a unified abstraction
layer for data storage and retrieval across different backends. It
enables location-transparent data access, allowing the same code to work
with data stored in memory, on disk, or in other storage systems.

**Note**: This document describes the actual VFS implementation
including the evolved special I/O handling and materialization systems.

Core Concepts
-------------

Backend Abstraction
~~~~~~~~~~~~~~~~~~~

The VFS abstracts away the underlying storage mechanism through a common
interface:

.. code:: python

   # Same API regardless of where data is stored
   filemanager.save(data, "path/to/data", Backend.MEMORY)
   filemanager.save(data, "path/to/data", Backend.DISK)
   filemanager.save(data, "path/to/data", Backend.ZARR)

   # Load from any backend
   data = filemanager.load("path/to/data", Backend.MEMORY)
   data = filemanager.load("path/to/data", Backend.DISK)

Path Virtualization
~~~~~~~~~~~~~~~~~~~

VFS provides a unified path interface where the same logical path works
across all backends:

-  **Unified Path**: ``/pipeline/step1/output/processed_images``
-  **Memory Backend**: Stores in-memory using the same path as key
-  **Disk Backend**: Maps to physical file using the same path structure
-  **Zarr Backend**: Creates zarr store using the same path structure

The key principle is that **paths are identical across all backends** -
the VFS handles the backend-specific storage implementation transparently.

Backend Types
~~~~~~~~~~~~~

Memory Backend
^^^^^^^^^^^^^^

-  **Purpose**: Fast intermediate data storage
-  **Use Cases**: Temporary arrays, tensors between pipeline steps
-  **Characteristics**:

   -  Fastest access
   -  Limited by RAM
   -  Volatile (lost on process exit)
   -  Supports any Python object

Disk Backend
^^^^^^^^^^^^

-  **Purpose**: Persistent data storage
-  **Use Cases**: Input images, final outputs, checkpoints
-  **Characteristics**:

   -  Persistent across runs
   -  Slower than memory
   -  Unlimited capacity (disk space)
   -  Supports standard file formats

Zarr Backend
^^^^^^^^^^^^

-  **Purpose**: Chunked array storage with OME-ZARR support
-  **Use Cases**: Large multidimensional arrays, final outputs
-  **Characteristics**:

   -  Efficient for large arrays
   -  Supports compression (ZSTD, LZ4)
   -  Cloud storage compatible
   -  OME-ZARR metadata support
   -  Parallel access

FileManager Architecture
------------------------

Core Interface
~~~~~~~~~~~~~~

.. code:: python

   class FileManager:
       def save(self, data: Any, path: str, backend: Backend) -> None:
           """Save data to specified path and backend."""

       def load(self, path: str, backend: Backend) -> Any:
           """Load data from specified path and backend."""
           
       def exists(self, path: str, backend: str) -> bool:
           """Check if data exists at path in backend."""
           
       def delete(self, path: str, backend: str) -> None:
           """Delete data at path in backend."""
           
       def list_files(self, path: str, backend: str, **kwargs) -> List[str]:
           """List files in directory for backend."""

Backend Registry
~~~~~~~~~~~~~~~~

The FileManager uses a registry pattern to manage different storage
backends:

.. code:: python

   registry = StorageRegistry()
   registry.register_backend("memory", MemoryStorageBackend)
   registry.register_backend("disk", DiskStorageBackend)
   registry.register_backend("zarr", ZarrStorageBackend)

   filemanager = FileManager(registry)

Type-Aware Serialization
~~~~~~~~~~~~~~~~~~~~~~~~

The VFS automatically handles serialization based on data type and
backend:

.. code:: python

   # Numpy arrays
   filemanager.save(numpy_array, "data.npy", "disk")  # Saves as .npy file
   filemanager.save(numpy_array, "data", "memory")    # Stores object directly

   # PyTorch tensors
   filemanager.save(torch_tensor, "model.pt", "disk") # Saves as .pt file
   filemanager.save(torch_tensor, "tensor", "memory") # Stores object directly

   # Images
   filemanager.save_image(image_array, "image.tif", "disk") # Saves as TIFF

Integration with Pipeline System
--------------------------------

Materialization Strategy
~~~~~~~~~~~~~~~~~~~~~~~~

The pipeline compiler determines optimal storage locations based on:

1. **Step Position**:

   -  First step: Always reads from disk (input images)
   -  Last step: Always writes to disk (final outputs)
   -  Middle steps: Can use memory for intermediate data

2. **Step Type**:

   -  FunctionStep: Can use any backend
   -  Other steps: Must use persistent backends

3. **Resource Constraints**:

   -  Memory availability
   -  Disk space
   -  Performance requirements

4. **Explicit Flags**:

   -  ``force_disk_output``: Override to force disk storage

Step Plan Integration
~~~~~~~~~~~~~~~~~~~~~

Each step’s execution plan specifies VFS usage:

.. code:: python

   step_plan = {
       "input_dir": "/workspace/A01/input",
       "output_dir": "/workspace/A01/step1_out", 
       "read_backend": "disk",
       "write_backend": "memory",
       
       "special_inputs": {
           "positions": {
               "path": "/vfs/positions.pkl",
               "backend": "memory"
           }
       },
       
       "special_outputs": {
           "metadata": {
               "path": "/vfs/metadata.pkl", 
               "backend": "memory"
           }
       }
   }

Cross-Step Communication
~~~~~~~~~~~~~~~~~~~~~~~~

Special I/O uses VFS for data exchange between steps:

.. code:: python

   # Step 1: Generate positions with materialization
   from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs

   @special_outputs(("positions", materialize_positions_to_csv))
   def generate_positions(image_stack):
       positions = calculate_positions(image_stack)
       # Compiler automatically saves to VFS memory backend
       # Materialization function saves to disk as CSV
       return processed_image, positions

   # Step 2: Use positions
   @special_inputs("positions")
   def stitch_images(image_stack, positions):
       # Compiler automatically loads from VFS memory backend
       return stitch(image_stack, positions)

Integration with Stack Utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The VFS works seamlessly with the memory type system:

.. code:: python

   # FunctionStep execution flow
   def _process_single_pattern_group():
       # 1. Load 2D images from VFS
       raw_slices = []
       for file_path in matching_files:
           image = context.filemanager.load_image(file_path, read_backend)
           raw_slices.append(image)  # Usually numpy arrays from disk

       # 2. Stack to 3D with target memory type
       image_stack = stack_slices(
           slices=raw_slices,
           memory_type=input_memory_type,  # From function decorator
           gpu_id=device_id
       )

       # 3. Process with function (operates in native memory type)
       result_stack = func(image_stack, **kwargs)

       # 4. Unstack to 2D slices
       output_slices = unstack_slices(
           array=result_stack,
           memory_type=output_memory_type,  # From function decorator
           gpu_id=device_id
       )

       # 5. Save 2D slices back to VFS
       for i, slice_2d in enumerate(output_slices):
           context.filemanager.save_image(slice_2d, output_path, write_backend)

**Key Integration Points**: - VFS handles serialization/deserialization
(bytes ↔ arrays) - Stack utils handle memory type conversion (numpy ↔
torch/cupy/etc.) - Function decorators specify memory type requirements
- Compiler coordinates the entire flow

Performance Considerations
--------------------------

Memory Management
~~~~~~~~~~~~~~~~~

-  **Memory Backend**: Limited by available RAM
-  **Automatic Cleanup**: Objects removed when no longer referenced
-  **Memory Pressure**: Can trigger materialization to disk

Data Movement Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

The compiler optimizes data movement:

1. **Minimize Transfers**: Keep data in same backend when possible
2. **Batch Operations**: Group related data in same backend
3. **Lazy Loading**: Load data only when needed
4. **Compression**: Use compressed formats for disk storage

Backend Selection Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Backend selection is predetermined during compilation, not dynamic:

   # First step: Always reads from disk/zarr (input images)
   # Intermediate steps: Always use memory backend between steps
   # Last step: Always writes to materialization backend (disk/zarr)
   # Per-step materialization: Uses materialization backend when StepMaterializationConfig provided

   # No runtime switching - backends determined at compile time

Materialization Configuration
-----------------------------

StepMaterializationConfig
~~~~~~~~~~~~~~~~~~~~~~~~~

Per-step materialization is controlled by ``StepMaterializationConfig``:

.. code:: python

   from openhcs.core.pipeline_config import LazyStepMaterializationConfig

   # Step with materialization - writes to materialization backend
   step = FunctionStep(
       func=my_function,
       materialization_config=LazyStepMaterializationConfig(
           sub_dir="analysis_results",
           well_filter=["A01", "A02"]  # Only materialize specific wells
       )
   )

Path Resolution
~~~~~~~~~~~~~~~

.. code:: python

   # VFS provides unified path interface - same path works for all backends
   path = "/pipeline/step1/output/processed_images"

   # Same path used across all backends
   filemanager.save(data, path, Backend.MEMORY)
   filemanager.save(data, path, Backend.DISK)
   filemanager.save(data, path, Backend.ZARR)

   # Load using same path regardless of backend
   data = filemanager.load(path, Backend.MEMORY)
   data = filemanager.load(path, Backend.DISK)

Data Validation
~~~~~~~~~~~~~~~

.. code:: python

   def validate_data_integrity(path, backend, expected_type):
       """Validate loaded data matches expectations."""
       if not filemanager.exists(path, backend):
           raise FileNotFoundError(f"Data not found: {path} in {backend}")

       data = filemanager.load(path, backend)
       if not isinstance(data, expected_type):
           raise TypeError(f"Expected {expected_type}, got {type(data)}")
           
       return data

Configuration
-------------

VFS Configuration
~~~~~~~~~~~~~~~~~

.. code:: python

   from openhcs.core.config import VFSConfig
   from openhcs.constants.constants import Backend, MaterializationBackend

   vfs_config = VFSConfig(
       intermediate_backend=Backend.MEMORY,
       materialization_backend=MaterializationBackend.ZARR
   )

Backend-Specific Settings
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Memory backend settings
   memory_config = {
       "max_objects": 1000,
       "cleanup_threshold": 0.8,
       "enable_compression": False
   }

   # Disk backend settings  
   disk_config = {
       "base_path": "/workspace",
       "create_directories": True,
       "file_permissions": 0o644,
       "enable_compression": True
   }

Best Practices
--------------

Path Naming
~~~~~~~~~~~

-  Use descriptive, hierarchical paths:
   ``/pipeline/step1/output/processed_images``
-  Include step information: ``/step_{step_id}/output/{data_type}``
-  Avoid absolute paths in application code

Backend Selection
~~~~~~~~~~~~~~~~~

-  Use memory for small, temporary data
-  Use disk for large data or persistent storage
-  Consider data lifetime and access patterns
-  Monitor memory usage and adjust accordingly

Backend Usage Patterns
~~~~~~~~~~~~~~~~~~~~~~

-  **Memory**: Always used between pipeline steps for fast intermediate storage
-  **Disk/Zarr**: Used for first step input, last step output, and per-step materialization
-  **No fallbacks**: Backend selection is predetermined, no runtime switching
-  **Fail-loud**: VFS operations fail immediately on errors, no silent fallbacks

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

-  Batch related operations
-  Minimize backend switches
-  Use appropriate data formats
-  Monitor and profile VFS usage

Future Enhancements
-------------------

Cloud Storage Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

-  S3-compatible backends
-  Azure Blob Storage
-  Google Cloud Storage
-  Automatic tiering based on access patterns

Advanced Features
~~~~~~~~~~~~~~~~~

-  Data versioning and lineage tracking
-  Automatic compression and deduplication
-  Distributed storage across multiple nodes
-  Real-time data synchronization

Monitoring and Analytics
~~~~~~~~~~~~~~~~~~~~~~~~

-  VFS usage metrics
-  Performance profiling
-  Storage optimization recommendations
-  Automated cleanup policies

See Also
--------

**Core Integration**:

- :doc:`memory_backend_system` - Backend implementation details
- :doc:`special_io_system` - Cross-step communication using VFS
- :doc:`pipeline_compilation_system` - VFS integration with compilation

**Practical Usage**:

- :doc:`../api/io_storage` - FileManager and storage backend API
- :doc:`../guides/memory_type_integration` - VFS with memory type system
- :doc:`../api/config` - VFS configuration options

**Advanced Topics**:

- :doc:`system_integration` - VFS integration with other OpenHCS systems
- :doc:`compilation_system_detailed` - Backend selection during compilation
- :doc:`function_pattern_system` - Function patterns with VFS storage

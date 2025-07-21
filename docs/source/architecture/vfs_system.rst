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

   # FileManager accessed through ProcessingContext
   context.filemanager.save_array(data, "path/to/data")
   context.filemanager.exists("path/to/data")
   context.filemanager.delete("path/to/data")

   # Backend selection is automatic based on VFS configuration
   # - Intermediate steps use memory backend
   # - Final steps use materialization backend (ZARR/disk)

Path Virtualization
~~~~~~~~~~~~~~~~~~~

VFS paths are logical paths that can be mapped to different physical
storage locations:

-  **Logical Path**: ``/pipeline/step1/output/processed_images``
-  **Physical Path (Memory)**: In-memory object store
-  **Physical Path (Disk)**:
   ``/workspace/A01/step1_out/processed_images.tif``

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
       def save(self, data: Any, path: str, backend: str) -> None:
           """Save data to specified path and backend."""
           
       def load(self, path: str, backend: str) -> Any:
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

   # Arrays (automatic backend selection)
   context.filemanager.save_array(numpy_array, "data/output")
   context.filemanager.save_array(torch_tensor, "tensors/model")

   # Images with VFS integration
   context.filemanager.save_image(image_array, "images/processed")

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
   -  ``requires_disk_input/output``: Step-level requirements

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

   def select_backend(step_position, step_type, data_size, memory_available):
       """Intelligent backend selection."""
       if step_position == 0:  # First step
           return "disk"  # Must read input images
       
       if step_position == last_position:  # Last step
           return "disk"  # Must write final outputs
           
       if data_size > memory_available * 0.8:
           return "disk"  # Too large for memory
           
       if step_type == "FunctionStep":
           return "memory"  # Fast intermediate storage
           
       return "disk"  # Conservative default

Error Handling
--------------

Backend Failures
~~~~~~~~~~~~~~~~

.. code:: python

   # VFS handles backend selection automatically
   if context.filemanager.exists("path/to/data"):
       data = context.filemanager.load_array("path/to/data")
   else:
       raise FileNotFoundError("Data not found in VFS")

Path Resolution
~~~~~~~~~~~~~~~

.. code:: python

   def resolve_path(logical_path, backend):
       """Resolve logical path to physical path."""
       if backend == "memory":
           return logical_path  # Use as-is for memory
       elif backend == "disk":
           return workspace_path / logical_path
       else:
           raise ValueError(f"Unknown backend: {backend}")

Data Validation
~~~~~~~~~~~~~~~

.. code:: python

   def validate_data_integrity(context, path, expected_type):
       """Validate loaded data matches expectations."""
       if not context.filemanager.exists(path):
           raise FileNotFoundError(f"Data not found: {path}")

       data = context.filemanager.load_array(path)
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
       materialization_backend=MaterializationBackend.ZARR,
       persistent_storage_root_path="/workspace/outputs"
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

Error Recovery
~~~~~~~~~~~~~~

-  Implement fallback strategies for backend failures
-  Validate data integrity after loading
-  Use checksums for critical data
-  Log all VFS operations for debugging

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

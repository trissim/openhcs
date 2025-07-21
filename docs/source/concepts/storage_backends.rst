===================
Storage Backends
===================

Overview
--------

OpenHCS uses a Virtual File System (VFS) approach to storage that provides unified file operations across multiple backend types. The VFS system abstracts storage operations through the ``FileManager`` interface, enabling seamless switching between memory, disk, and ZARR backends.

**Key Features**:

* **Backend Abstraction**: Unified API for memory, disk, and ZARR storage
* **Automatic Optimization**: Different backends for intermediate vs. final results
* **Large Dataset Support**: ZARR backend handles datasets from MB to 100GB+
* **Memory Efficiency**: Memory backend for fast intermediate processing

VFS Backend Configuration
-------------------------

OpenHCS supports three primary storage backends configured through the VFS system:

**Real-World Configuration** (from TUI-generated scripts):

.. code-block:: python

    from openhcs.core.config import VFSConfig, ZarrConfig
    from openhcs.constants.constants import Backend, MaterializationBackend

    # VFS backend configuration
    vfs_config = VFSConfig(
        intermediate_backend=Backend.MEMORY,              # Fast memory for intermediate steps
        materialization_backend=MaterializationBackend.ZARR  # ZARR for final results
    )

    # ZARR-specific configuration
    zarr_config = ZarrConfig(
        store_name="images.zarr",                         # ZARR store filename
        compressor=ZarrCompressor.ZSTD,                   # Compression algorithm
        compression_level=1,                              # Compression level
        shuffle=True,                                     # Enable shuffle filter
        chunk_strategy=ZarrChunkStrategy.SINGLE,          # Chunking strategy
        ome_zarr_metadata=True,                           # OME-ZARR metadata
        write_plate_metadata=True                         # Plate-level metadata
    )

    # Integration with global configuration
    global_config = GlobalPipelineConfig(
        vfs=vfs_config,
        zarr=zarr_config,
        # ... other configuration
    )

Backend Types and Use Cases
---------------------------

**Memory Backend** (``Backend.MEMORY``):
- **Use Case**: Intermediate processing steps within pipelines
- **Performance**: Fastest access, no disk I/O overhead
- **Limitations**: Limited by system RAM, temporary storage
- **Best For**: Function chains, preprocessing steps

**Disk Backend** (``Backend.DISK``):
- **Use Case**: Traditional file system storage
- **Performance**: Standard disk I/O performance
- **Limitations**: No compression, individual file management
- **Best For**: Small to medium datasets, debugging

**ZARR Backend** (``MaterializationBackend.ZARR``):
- **Use Case**: Large dataset storage with compression
- **Performance**: Optimized for large arrays, chunked access
- **Features**: Compression, metadata, OME-ZARR compatibility
- **Best For**: Final results, datasets >1GB, long-term storage

FileManager Interface
---------------------

All storage operations go through the ``FileManager`` interface, providing unified access:

.. code-block:: python

    # FileManager operations with explicit backend selection
    filemanager.exists("path/to/file", "memory")         # Check file existence
    filemanager.save(array, "output/path", "disk")       # Save array to disk
    filemanager.load("input/path", "zarr")               # Load array from ZARR
    filemanager.delete("path/to/file", "memory")         # Delete file from memory

    # Backend selection based on use case
    # - Intermediate results → Memory backend (fast, temporary)
    # - Final results → ZARR backend (compressed, persistent)

VFS Integration with FunctionSteps
-----------------------------------

OpenHCS FunctionSteps automatically use the VFS system through the ProcessingContext:

.. code-block:: python

    # FunctionSteps automatically handle VFS operations
    step = FunctionStep(
        func=[(processing_function, {})],
        name="processing_step",
        variable_components=[VariableComponents.SITE],
        force_disk_output=False  # Uses memory backend for intermediate results
    )

    # When force_disk_output=True, results are materialized to final backend
    final_step = FunctionStep(
        func=[(final_function, {})],
        name="final_step",
        variable_components=[VariableComponents.SITE],
        force_disk_output=True   # Forces materialization to ZARR/disk backend
    )

**Automatic Backend Selection**:

- **Intermediate Steps** (``force_disk_output=False``): Use memory backend for speed
- **Final Steps** (``force_disk_output=True``): Use materialization backend for persistence
- **Cross-Step Data Flow**: Automatic conversion between backends as needed

Performance Considerations
--------------------------

**Memory Backend**:
- **Pros**: Fastest access, no I/O overhead, ideal for function chains
- **Cons**: Limited by RAM, temporary storage only
- **Use When**: Intermediate processing, function chains, small datasets

**ZARR Backend**:
- **Pros**: Compression, chunking, OME-ZARR compatibility, handles large datasets
- **Cons**: Slower than memory, compression overhead
- **Use When**: Final results, large datasets (>1GB), long-term storage

**Optimization Tips**:
- Use memory backend for intermediate steps in function chains
- Use ZARR backend only for final materialization
- Configure appropriate ZARR chunk sizes for your data access patterns

See Also
--------

**Technical Details**:

- :doc:`../architecture/vfs_system` - Complete VFS system architecture
- :doc:`../architecture/memory_backend_system` - Memory backend implementation

**Configuration**:

- :doc:`../api/config` - VFSConfig and ZarrConfig documentation
- :doc:`directory_structure` - Directory organization with VFS

**Related Concepts**:

- :doc:`processing_context` - How context integrates with VFS
- :doc:`step` - How FunctionSteps use VFS backends

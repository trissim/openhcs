I/O and Storage System
======================

.. module:: openhcs.io

OpenHCS provides a unified I/O system with multiple storage backends for handling datasets from MB to 100GB+. The system automatically switches between disk, memory, and ZARR backends based on configuration and data size.

FileManager Class
-----------------

The FileManager is the primary interface for all file operations in OpenHCS. It provides backend-agnostic file operations with automatic backend selection.

.. code-block:: python

    from openhcs.io.filemanager import FileManager
    from openhcs.io.base import storage_registry

    # Create FileManager with global registry
    filemanager = FileManager(storage_registry)

The FileManager is the primary interface for all file operations in OpenHCS. It provides backend-agnostic file operations with automatic backend selection.

Storage Backends
----------------

OpenHCS supports three storage backends:

**Disk Backend** (``openhcs.io.disk.DiskStorageBackend``)
    Traditional file system storage for persistent data

**Memory Backend** (``openhcs.io.memory.MemoryStorageBackend``)  
    In-memory storage for intermediate processing steps

**ZARR Backend** (``openhcs.io.zarr.ZarrStorageBackend``)
    Compressed array storage for large datasets (100GB+)

Backend Selection
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.io.filemanager import FileManager
    from openhcs.io.base import storage_registry
    from openhcs.constants.constants import Backend

    # Create FileManager with global registry
    filemanager = FileManager(storage_registry)

    # Backend-specific operations
    files = filemanager.list_files('/path/to/data', Backend.DISK.value)
    filemanager.save_array(array, '/memory/path', Backend.MEMORY.value)
    filemanager.save_array(large_array, '/zarr/path', Backend.ZARR.value)

VFS Configuration
-----------------

The Virtual File System (VFS) allows seamless backend switching:

.. code-block:: python

    from openhcs.core.config import VFSConfig, GlobalPipelineConfig
    from openhcs.constants.constants import Backend, MaterializationBackend

    # Configure VFS for large dataset processing
    vfs_config = VFSConfig(
        intermediate_backend=Backend.MEMORY,      # Fast intermediate storage
        materialization_backend=MaterializationBackend.ZARR,  # Compressed final storage
        memory_limit_gb=32                        # Auto-switch to ZARR when exceeded
    )

    global_config = GlobalPipelineConfig(
        vfs=vfs_config,
        num_workers=8
    )

ZARR Integration
----------------

For large datasets, OpenHCS automatically uses ZARR compression:

.. code-block:: python

    from openhcs.core.config import ZarrConfig, ZarrCompressor, ZarrChunkStrategy

    # ZARR configuration for 100GB+ datasets
    zarr_config = ZarrConfig(
        compressor=ZarrCompressor.LZ4,           # Fast compression
        chunk_strategy=ZarrChunkStrategy.ADAPTIVE, # Automatic chunking
        compression_level=1                       # Balance speed vs size
    )

    # Automatic ZARR usage in pipelines
    global_config = GlobalPipelineConfig(
        zarr=zarr_config,
        vfs=VFSConfig(
            materialization_backend=MaterializationBackend.ZARR
        )
    )

Common Operations
-----------------

File Listing
^^^^^^^^^^^^

.. code-block:: python

    # List image files with filtering
    image_files = filemanager.list_image_files(
        directory='/path/to/images',
        backend=Backend.DISK.value,
        extensions={'.tif', '.tiff'},
        recursive=True
    )

    # List all files with pattern
    files = filemanager.list_files(
        directory='/path/to/data', 
        backend=Backend.DISK.value,
        pattern='*.tif'
    )

Array Operations
^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np

    # Save arrays to different backends
    array = np.random.rand(1000, 1000)
    
    # Memory backend (fast, temporary)
    filemanager.save_array(array, '/memory/temp_array', Backend.MEMORY.value)
    
    # Disk backend (persistent)
    filemanager.save_array(array, '/disk/persistent_array.tif', Backend.DISK.value)
    
    # ZARR backend (compressed, large datasets)
    large_array = np.random.rand(10000, 10000)
    filemanager.save_array(large_array, '/zarr/large_dataset', Backend.ZARR.value)

    # Load arrays (backend auto-detected)
    loaded_array = filemanager.load_array('/memory/temp_array', Backend.MEMORY.value)

Directory Operations
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create directories
    filemanager.ensure_directory('/path/to/new/dir', Backend.DISK.value)
    
    # Check existence
    exists = filemanager.exists('/path/to/file', Backend.DISK.value)
    is_directory = filemanager.is_dir('/path/to/dir', Backend.DISK.value)
    
    # Mirror directory structure
    filemanager.mirror_directory_structure(
        source_dir='/source/path',
        target_dir='/target/path', 
        backend=Backend.DISK.value,
        recursive=True
    )

Backend Registry
----------------

The storage registry provides centralized backend management:

The storage registry provides centralized backend management:

.. code-block:: python

    from openhcs.io.base import storage_registry

    # Get available backends
    backends = storage_registry.get_available_backends()
    
    # Get specific backend
    disk_backend = storage_registry.get_backend('disk')
    memory_backend = storage_registry.get_backend('memory')
    zarr_backend = storage_registry.get_backend('zarr')

Performance Considerations
--------------------------

**Memory Backend**: Fastest for intermediate processing, limited by RAM
**Disk Backend**: Good for persistent storage, I/O bound for large files  
**ZARR Backend**: Best for large datasets, excellent compression ratios

**Automatic Selection**: OpenHCS automatically chooses optimal backends based on:
- Data size vs available memory
- Pipeline configuration
- Step requirements (intermediate vs final output)

See Also
--------

- :doc:`../architecture/vfs_system` - Virtual file system architecture
- :doc:`../architecture/memory_backend_system` - Backend implementation details
- :doc:`../user_guide/production_examples` - Real-world usage examples
